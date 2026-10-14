import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

try:
    from transformers import GPT2Model, GPT2Tokenizer
except Exception:
    GPT2Model, GPT2Tokenizer = None, None


class LearnableTemperatureInfoNCELoss(nn.Module):
    """轻量跨模态对齐损失：同一样本 text/ts 拉近，不同样本拉远。"""
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp, dtype=torch.float32)))

    def forward(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        temp = self.log_temp.exp().clamp_min(1e-6)
        logits = torch.matmul(a, b.transpose(0, 1)) / temp
        labels = torch.arange(a.size(0), device=a.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_ab + loss_ba)


class PrototypeAwareInfoNCELoss(nn.Module):
    """
    Prototype-aware InfoNCE.

    与普通 batch InfoNCE 的区别：
    - 同一样本仍然是正样本；
    - 其他样本仍然是负样本；
    - 但如果两个样本的 temporal prototype context 很相似，
      则降低它们作为负样本时的惩罚，缓解时序 batch 中常见的“伪负样本”问题。
    """
    def __init__(self, init_temp=0.07, neg_rho=0.5):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp, dtype=torch.float32)))
        self.neg_rho = float(neg_rho)

    def forward(self, a, b, proto_ctx=None):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        temp = self.log_temp.exp().clamp_min(1e-6)
        logits = torch.matmul(a, b.transpose(0, 1)) / temp

        if proto_ctx is None or self.neg_rho <= 0 or a.size(0) <= 1:
            labels = torch.arange(a.size(0), device=a.device)
            loss_ab = F.cross_entropy(logits, labels)
            loss_ba = F.cross_entropy(logits.transpose(0, 1), labels)
            return 0.5 * (loss_ab + loss_ba)

        proto = F.normalize(proto_ctx, dim=-1)
        proto_sim = torch.matmul(proto, proto.transpose(0, 1)).clamp(min=0.0, max=1.0)

        # 相似 prototype 的样本更可能不是“强负样本”，因此降低其负样本权重。
        rho = min(max(self.neg_rho, 0.0), 0.95)
        neg_weight = (1.0 - rho * proto_sim).clamp_min(0.05)
        diag = torch.eye(neg_weight.size(0), device=neg_weight.device, dtype=neg_weight.dtype)
        weight = neg_weight * (1.0 - diag) + diag

        loss_ab = self._weighted_directional_loss(logits, weight)
        loss_ba = self._weighted_directional_loss(logits.transpose(0, 1), weight.transpose(0, 1))
        return 0.5 * (loss_ab + loss_ba)

    @staticmethod
    def _weighted_directional_loss(logits, weight):
        # 数值稳定的 weighted log-softmax，正样本是对角线。
        max_logits = logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits - max_logits)
        denom = (weight * exp_logits).sum(dim=1).clamp_min(1e-12)
        pos = exp_logits.diagonal().clamp_min(1e-12)
        return (-torch.log(pos / denom)).mean()


class TemporalPrototypeEncoder(nn.Module):
    """
    APT-inspired temporal prototype encoder.

    输入 x_mark_enc / x_mark_dec，输出样本级时间分布上下文 proto_ctx。
    这里不直接复现 APT 的 affine，而是把 timestamp prototype 作为
    对比学习与残差融合门控的上下文。
    """
    def __init__(self, proto_dim=128, num_prototypes=16, topk=4, dropout=0.1):
        super().__init__()
        self.proto_dim = int(proto_dim)
        self.num_prototypes = int(num_prototypes)
        self.topk = max(1, int(topk))
        self.dropout = float(dropout)

        self.mark_proj = nn.Sequential(
            nn.LazyLinear(self.proto_dim),
            nn.LayerNorm(self.proto_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.proto_dim, self.proto_dim),
            nn.LayerNorm(self.proto_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim) * 0.02)
        self.out_ln = nn.LayerNorm(self.proto_dim)

    def forward(self, x_mark_enc, x_mark_dec, batch_size, device):
        mark_feat = self._build_marker_feature(x_mark_enc, x_mark_dec, batch_size, device)
        mark_emb = self.mark_proj(mark_feat)

        proto = self.prototypes
        logits = torch.matmul(F.normalize(mark_emb, dim=-1), F.normalize(proto, dim=-1).transpose(0, 1))

        k = min(self.topk, self.num_prototypes)
        topv, topi = torch.topk(logits, k=k, dim=-1)
        topw = torch.softmax(topv, dim=-1)

        selected_proto = proto[topi]  # [B, K, D]
        ctx = torch.sum(topw.unsqueeze(-1) * selected_proto, dim=1)
        ctx = self.out_ln(ctx + mark_emb)

        usage_dense = torch.zeros(batch_size, self.num_prototypes, device=device, dtype=mark_emb.dtype)
        usage_dense.scatter_(1, topi, topw)

        aux_loss = self.prototype_aux_loss(usage_dense)
        return ctx, usage_dense, aux_loss

    def prototype_aux_loss(self, usage_dense):
        proto = F.normalize(self.prototypes, dim=-1)
        gram = torch.matmul(proto, proto.transpose(0, 1))
        eye = torch.eye(self.num_prototypes, device=gram.device, dtype=gram.dtype)
        off_diag = gram * (1.0 - eye)
        orth_loss = off_diag.pow(2).mean()

        usage = usage_dense.mean(dim=0)
        uniform = torch.full_like(usage, 1.0 / max(1, self.num_prototypes))
        balance_loss = (usage - uniform).pow(2).sum()

        # 轻量正则：保持 prototype 范数稳定，避免过大。
        norm_loss = (self.prototypes.norm(dim=-1) - 1.0).pow(2).mean()
        return orth_loss + balance_loss + 0.01 * norm_loss

    @staticmethod
    def _safe_mark_stats(x_mark, batch_size, device):
        if x_mark is None:
            return torch.zeros(batch_size, 4, device=device)

        x = x_mark.float().to(device)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.size(1) == 0:
            return torch.zeros(batch_size, 4, device=device)

        mean = x.mean(dim=1)
        last = x[:, -1, :]
        first = x[:, 0, :]
        std = x.std(dim=1, unbiased=False)
        return torch.cat([mean, last, first, std], dim=-1)

    def _build_marker_feature(self, x_mark_enc, x_mark_dec, batch_size, device):
        enc_feat = self._safe_mark_stats(x_mark_enc, batch_size, device)
        dec_feat = self._safe_mark_stats(x_mark_dec, batch_size, device)
        return torch.cat([enc_feat, dec_feat], dim=-1)


class Model(nn.Module):
    """
    Frequency-enhanced numerical backbone + auxiliary text branch.

    设计原则：
    1. 数值主干负责主要预测能力；
    2. 文本分支仅作为辅助残差，不强控主干内部路径；
    3. 对齐损失只返回原始 loss，由训练脚本统一乘以 align_lambda；
    4. 仅保留当前实验真正使用的参数接口，避免旧参数干扰。
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model

        # ========= iTransformer base =========
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # ========= Frequency knobs =========
        self.use_freq_feat = bool(getattr(configs, "use_freq_feat", True))
        self.use_freq_token = bool(getattr(configs, "use_freq_token", False))

        legacy_use_freq = getattr(configs, "use_freq", False)
        if legacy_use_freq and not hasattr(configs, "use_freq_feat"):
            self.use_freq_feat = True

        self.freq_k = int(getattr(configs, "freq_k", 16))
        self.freq_mode = getattr(configs, "freq_mode", "low")
        self.freq_use_log_amp = bool(getattr(configs, "freq_use_log_amp", False))
        self.freq_token_to_encoder = bool(getattr(configs, "freq_token_to_encoder", False))
        self.freq_token_to_residual = bool(getattr(configs, "freq_token_to_residual", True))

        self.freq_feat_proj = nn.Linear(self.freq_k, configs.d_model, bias=True)
        self.freq_feat_ln = nn.LayerNorm(configs.d_model)
        self.freq_alpha_feat = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.cfg_enc_in = int(getattr(configs, "enc_in", 0))
        in_dim_for_token_proj = self.cfg_enc_in if self.cfg_enc_in > 0 else 1
        self.freq_token_proj = nn.Linear(in_dim_for_token_proj, configs.d_model, bias=True)
        self.freq_token_ln = nn.LayerNorm(configs.d_model)
        self.freq_alpha_token = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.use_freq_gate = bool(getattr(configs, "use_freq_gate", False))
        self.freq_gate_type = getattr(configs, "freq_gate_type", "scalar")
        self.freq_gate_init_bias = float(getattr(configs, "freq_gate_init_bias", -2.0))
        self.gate_on_feat = bool(getattr(configs, "gate_on_feat", True))
        self.gate_on_token = bool(getattr(configs, "gate_on_token", True))

        self.use_freq_residual = bool(getattr(configs, "use_freq_residual", False))
        self.freq_res_gate_bias = nn.Parameter(torch.tensor(self.freq_gate_init_bias, dtype=torch.float32))
        self.freq_res_gate_type = getattr(configs, "freq_res_gate_type", "scalar")
        self.freq_alpha_tok2res = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        if self.freq_res_gate_type == "var":
            hidden = max(8, configs.d_model // 2)
            self.freq_gate_var_mlp = nn.Sequential(
                nn.Linear(configs.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1)
            )

        self.projector_res = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        if self.use_freq_gate:
            self.freq_gate_mlp = nn.Sequential(
                nn.Linear(configs.d_model, max(8, configs.d_model // 2)),
                nn.GELU(),
                nn.Linear(max(8, configs.d_model // 2), 1),
                nn.Sigmoid()
            )
            self.freq_gate_bias = nn.Parameter(torch.tensor(self.freq_gate_init_bias, dtype=torch.float32))

        # ========= EMAformer-style embedding armor =========
        self.use_channel_emb = bool(getattr(configs, "use_channel_emb", False))
        self.use_phase_emb = bool(getattr(configs, "use_phase_emb", False))
        self.use_joint_channel_phase_emb = bool(getattr(configs, "use_joint_channel_phase_emb", False))
        self.ema_period = int(getattr(configs, "ema_period", 24))
        self.ema_embed_drop = float(getattr(configs, "ema_embed_drop", 0.0))
        self.ema_alpha = nn.Parameter(torch.tensor(float(getattr(configs, "ema_alpha_init", 0.1)), dtype=torch.float32))

        if self.use_channel_emb or self.use_phase_emb or self.use_joint_channel_phase_emb:
            max_channels = max(1, int(getattr(configs, "enc_in", 1)))
            self.channel_emb = nn.Embedding(max_channels, configs.d_model)
            self.phase_emb = nn.Embedding(max(1, self.ema_period), configs.d_model)
            self.channel_phase_emb = nn.Embedding(max_channels * max(1, self.ema_period), configs.d_model)
            self.ema_embed_ln = nn.LayerNorm(configs.d_model)
            self.ema_embed_dropout = nn.Dropout(self.ema_embed_drop)

        # ========= LLM text branch =========
        self.use_llm = bool(getattr(configs, "use_llm", False))
        self.return_align_loss = bool(getattr(configs, "return_align_loss", True))
        self.llm_pool = str(getattr(configs, "llm_pool", "mean"))
        self.llm_dim = int(getattr(configs, "llm_dim", 768))
        self.learn_prompt_len = int(getattr(configs, "learn_prompt_len", 16))
        self.prompt_max_len = int(getattr(configs, "prompt_max_len", 128))
        self.evidence_dim = 6
        self.return_aux = bool(getattr(configs, "return_aux", False))

        # ========= APT-inspired temporal prototype guided contrastive fusion =========
        # This module does not replace RevIN or the backbone. It uses timestamp markers
        # to learn a sample-level temporal distribution context, then uses it to:
        # 1) reduce false negatives in InfoNCE;
        # 2) lightly FiLM-modulate the TS alignment vector;
        # 3) provide context to text/frequency residual gates.
        self.use_temporal_proto = bool(getattr(configs, "use_temporal_proto", False))
        self.temporal_proto_dim = int(getattr(configs, "temporal_proto_dim", 128))
        self.temporal_proto_num = int(getattr(configs, "temporal_proto_num", 16))
        self.temporal_proto_topk = int(getattr(configs, "temporal_proto_topk", 4))
        self.temporal_proto_dropout = float(getattr(configs, "temporal_proto_dropout", getattr(configs, "dropout", 0.1)))
        self.proto_aux_lambda = float(getattr(configs, "proto_aux_lambda", 0.1))
        self.proto_neg_rho = float(getattr(configs, "proto_neg_rho", 0.5))
        self.use_proto_infonce = bool(getattr(configs, "use_proto_infonce", True))
        self.use_proto_align_film = bool(getattr(configs, "use_proto_align_film", True))
        self.use_proto_text_gate = bool(getattr(configs, "use_proto_text_gate", True))

        if self.use_temporal_proto:
            self.temporal_proto = TemporalPrototypeEncoder(
                proto_dim=self.temporal_proto_dim,
                num_prototypes=self.temporal_proto_num,
                topk=self.temporal_proto_topk,
                dropout=self.temporal_proto_dropout,
            )
            self.proto_align_film = nn.Sequential(
                nn.Linear(self.temporal_proto_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model * 2),
            )
        self.proto_gate_dim = self.temporal_proto_dim if (self.use_temporal_proto and self.use_proto_text_gate) else 0

        # ========= Conservative reliability residual control =========
        # These parameters make the new mechanism conservative rather than letting auxiliary residuals
        # dominate the numerical backbone.
        self.use_reliability_gate = bool(getattr(configs, "use_reliability_gate", False))
        self.use_three_view_align = bool(getattr(configs, "use_three_view_align", True))
        self.use_evidence_gate = bool(getattr(configs, "use_evidence_gate", True))
        self.reliability_hidden = int(getattr(configs, "reliability_hidden", 128))
        self.reliability_dropout = float(getattr(configs, "reliability_dropout", 0.1))
        self.reliability_min_gate = float(getattr(configs, "reliability_min_gate", 0.0))
        self.reliability_detach_score = bool(getattr(configs, "reliability_detach_score", True))
        self.reliability_init_bias_text = float(getattr(configs, "reliability_init_bias_text", -6.0))
        self.reliability_init_bias_freq = float(getattr(configs, "reliability_init_bias_freq", -5.0))
        self.use_var_reliability_gate = bool(getattr(configs, "use_var_reliability_gate", True))
        self.use_ts_freq_align = bool(getattr(configs, "use_ts_freq_align", True))
        self.use_freq_text_align = bool(getattr(configs, "use_freq_text_align", False))

        # Residual budget: auxiliary branches are only allowed to make bounded corrections.
        self.residual_budget_freq = float(getattr(configs, "residual_budget_freq", 0.10))
        self.residual_budget_text = float(getattr(configs, "residual_budget_text", 0.05))
        self.residual_clip = float(getattr(configs, "residual_clip", 1.0))
        self.evidence_gate_dim = max(8, self.d_model // 8)

        # 模态平衡 + 文本残差融合参数
        self.text_residual_init = float(getattr(configs, "text_residual_init", -3.0))
        self.text_residual_mode = str(getattr(configs, "text_residual_mode", "token"))  # token or pred
        self.text_use_cross_attn = bool(getattr(configs, "text_use_cross_attn", False))
        self.text_scale_init = float(getattr(configs, "text_scale_init", 1.0))
        self.ts_scale_init = float(getattr(configs, "ts_scale_init", 1.0))
        self.text_dropout = float(getattr(configs, "text_dropout", getattr(configs, "dropout", 0.1)))

        if self.use_llm:
            if GPT2Model is None or GPT2Tokenizer is None:
                raise ImportError("transformers is required for use_llm=True, but import failed.")

            llm_model_name = getattr(configs, "llm_model", "GPT2")
            llm_repo_or_path = "gpt2" if llm_model_name == "GPT2" else llm_model_name

            self.tokenizer = GPT2Tokenizer.from_pretrained(llm_repo_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.llm = GPT2Model.from_pretrained(llm_repo_or_path)
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()

            self.learn_prompt = nn.Parameter(torch.randn(self.learn_prompt_len, self.llm_dim) * 0.02)

            # 文本与数值模态各自投影到共享对齐空间，先 balance 再对齐（借 BALM 思路）
            self.text_proj = nn.Sequential(
                nn.Linear(self.llm_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(self.text_dropout)
            )
            self.ts_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(self.text_dropout)
            )
            self.text_align_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU()
            )
            self.ts_align_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU()
            )
            self.text_scale = nn.Parameter(torch.tensor(self.text_scale_init, dtype=torch.float32))
            self.ts_scale = nn.Parameter(torch.tensor(self.ts_scale_init, dtype=torch.float32))

            # 文本分支输出残差，不再强控主干内部路径
            self.text_token_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(self.text_dropout)
            )
            self.text_pred_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(self.text_dropout),
                nn.Linear(self.d_model, self.pred_len)
            )
            # 独立的 token-to-prediction 残差头，避免文本分支与基础预测共享 projector。
            # 这样文本残差的梯度不会直接改写 numerical backbone 的输出映射。
            self.text_token_pred_head = nn.Linear(self.d_model, self.pred_len, bias=True)
            self.text_residual_bias = nn.Parameter(torch.tensor(self.text_residual_init, dtype=torch.float32))
            text_gate_in_dim = self.d_model * 2 + self.proto_gate_dim
            self.text_gate_mlp = nn.Sequential(
                nn.Linear(text_gate_in_dim, max(32, self.d_model // 2)),
                nn.GELU(),
                nn.Linear(max(32, self.d_model // 2), 1)
            )

            if self.text_use_cross_attn:
                self.text_cross_attn = nn.MultiheadAttention(
                    embed_dim=self.d_model,
                    num_heads=max(1, getattr(configs, "n_heads", 8) // 2),
                    dropout=getattr(configs, "dropout", 0.1),
                    batch_first=True,
                )

            if self.use_temporal_proto and self.use_proto_infonce:
                self.align_loss_fn = PrototypeAwareInfoNCELoss(
                    init_temp=max(1e-4, getattr(configs, "align_temp", 0.07)),
                    neg_rho=self.proto_neg_rho,
                )
            else:
                self.align_loss_fn = LearnableTemperatureInfoNCELoss(
                    init_temp=max(1e-4, getattr(configs, "align_temp", 0.07))
                )

            # Reliability estimator: project each view into a shared space and convert
            # view agreement into residual gates.
            self.ctrl_ts_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU()
            )
            self.ctrl_freq_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU()
            )
            self.ctrl_text_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU()
            )
            self.evidence_gate_proj = nn.Sequential(
                nn.Linear(self.evidence_dim, self.evidence_gate_dim),
                nn.LayerNorm(self.evidence_gate_dim),
                nn.GELU(),
                nn.Dropout(self.reliability_dropout)
            )

            gate_in_dim = self.d_model * 2 + 1 + self.evidence_gate_dim + self.proto_gate_dim
            self.text_reliability_gate = nn.Sequential(
                nn.Linear(gate_in_dim, self.reliability_hidden),
                nn.GELU(),
                nn.Dropout(self.reliability_dropout),
                nn.Linear(self.reliability_hidden, 1)
            )
            self.freq_reliability_gate = nn.Sequential(
                nn.Linear(gate_in_dim, self.reliability_hidden),
                nn.GELU(),
                nn.Dropout(self.reliability_dropout),
                nn.Linear(self.reliability_hidden, 1)
            )
            self.text_reliability_bias = nn.Parameter(torch.tensor(self.reliability_init_bias_text, dtype=torch.float32))
            self.freq_reliability_bias = nn.Parameter(torch.tensor(self.reliability_init_bias_freq, dtype=torch.float32))

    # ------------------------- core forward -------------------------

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Conservative reliability-guided residual correction.

        Forward structure:
          1) iTransformer produces base forecast Y_base and temporal tokens H_ts.
          2) Frequency and text branches produce residual candidates R_freq / R_text.
          3) Cross-view agreement scores s_freq / s_text and structural evidence e generate gates.
          4) Only bounded, gated residuals are added to Y_base.
        """
        aligned = None
        freq_tokens = None
        freq_res_out = None
        freq_vec = None
        r_text = None
        g_text = None
        g_freq = None
        s_text = None
        s_freq = None
        align_loss = x_enc.new_tensor(0.0)
        proto_ctx = None
        proto_usage = None
        proto_aux_loss = x_enc.new_tensor(0.0)
        self.last_aux = {}

        x_raw = x_enc
        evidence = self._compute_sample_evidence(x_raw)  # [B, 6]

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        B, L, N_vars = x_enc.shape

        # ===== APT-inspired temporal prototype context =====
        # The context is used later in contrastive learning and residual gates.
        if self.use_temporal_proto:
            proto_ctx, proto_usage, proto_aux_loss = self.temporal_proto(
                x_mark_enc=x_mark_enc,
                x_mark_dec=x_mark_dec,
                batch_size=B,
                device=x_enc.device,
            )

        # ===== Numerical backbone =====
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, D]
        T_base = enc_out.size(1)

        if self.use_channel_emb or self.use_phase_emb or self.use_joint_channel_phase_emb:
            armor_emb = self._build_embedding_armor(x_enc, T_base)
            enc_out = enc_out + self.ema_alpha * armor_emb

        # Optional old frequency gate for pre-encoder injection / tokens.
        gate_strength_scalar = None
        gate_strength_token = None
        if self.use_freq_gate:
            if self.freq_gate_type == "token":
                g = self.freq_gate_mlp(enc_out)
                g = self._apply_gate_bias(g, self.freq_gate_bias)
                gate_strength_token = g
            else:
                pooled = enc_out.mean(dim=1)
                g = self.freq_gate_mlp(pooled)
                g = self._apply_gate_bias(g, self.freq_gate_bias)
                gate_strength_scalar = g.view(B, 1, 1)

        # ===== Frequency feature injection before encoder =====
        if self.use_freq_feat:
            freq_feat = self._freq_features(x_enc)  # [B, N, K]
            freq_emb = self.freq_feat_ln(self.freq_feat_proj(freq_feat))  # [B, N, D]
            aligned = self._align_to_token_dim(freq_emb, T_base)          # [B, T, D]

            strength = 1.0
            if self.use_freq_gate and self.gate_on_feat:
                strength = gate_strength_token if (self.freq_gate_type == "token") else gate_strength_scalar

            enc_out = enc_out + (self.freq_alpha_feat * strength) * aligned

        # ===== Optional frequency tokens =====
        if self.use_freq_token:
            A = self._freq_spectrum(x_enc)
            freq_bins = self._select_k_bins(A)
            freq_tokens_raw = freq_bins.permute(0, 2, 1).contiguous()
            freq_tokens_raw = self._match_token_proj_in_dim(freq_tokens_raw, N_vars)
            freq_tokens = self.freq_token_ln(self.freq_token_proj(freq_tokens_raw))

            token_strength = 1.0
            if self.use_freq_gate and self.gate_on_token:
                if self.freq_gate_type == "token":
                    token_strength = gate_strength_token.mean(dim=1, keepdim=True)
                else:
                    token_strength = gate_strength_scalar
            freq_tokens = (self.freq_alpha_token * token_strength) * freq_tokens
            if self.freq_token_to_encoder:
                enc_out = torch.cat([enc_out, freq_tokens], dim=1)

        # ===== iTransformer encoder =====
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        var_tokens = enc_out[:, :N_vars, :]  # H_ts: [B, N, D]

        # Base forecast. Auxiliary views can only make bounded residual corrections.
        y_base = self.projector(var_tokens).permute(0, 2, 1)  # [B, S, N]
        y_hat = y_base
        ts_raw_vec = var_tokens.mean(dim=1)  # [B, D]

        # ===== Frequency residual candidate =====
        if self.use_freq_residual and self.use_freq_feat and (aligned is not None):
            freq_aligned = aligned[:, :N_vars, :]  # [B, N, D]
            freq_vec = freq_aligned.mean(dim=1)    # [B, D]
            freq_res_out = self.projector_res(freq_aligned).permute(0, 2, 1)  # [B, S, N]
            freq_res_out = self._limit_residual(freq_res_out, budget=self.residual_budget_freq)

            # If reliability gate is not active, preserve the old frequency residual behavior.
            if not (self.use_llm and self.use_reliability_gate):
                if self.freq_res_gate_type == "var":
                    var_repr = freq_aligned
                    if (freq_tokens is not None) and self.freq_token_to_residual:
                        tok_pool = freq_tokens.mean(dim=1)
                        var_repr = var_repr + self.freq_alpha_tok2res * tok_pool.unsqueeze(1)
                    g_logits = self.freq_gate_var_mlp(var_repr).squeeze(-1)
                    g_freq = torch.sigmoid(g_logits + self.freq_res_gate_bias)  # [B, N]
                    y_hat = y_hat + g_freq.unsqueeze(1) * freq_res_out
                else:
                    g_freq = torch.sigmoid(self.freq_res_gate_bias).view(1, 1)
                    y_hat = y_hat + g_freq * freq_res_out

        # ===== LLM semantic residual + reliability-aware gates =====
        if self.use_llm:
            text_seq, text_vec = self._run_text_branch(x_raw, evidence=evidence)  # [B, P, D], [B, D]

            ts_vec = self.ts_proj(ts_raw_vec)        # [B, D]

            # Prototype-conditioned FiLM on the TS alignment vector.
            # Keep the scale small so the branch starts close to the original model.
            if self.use_temporal_proto and self.use_proto_align_film and (proto_ctx is not None):
                film = self.proto_align_film(proto_ctx)
                film_gamma, film_beta = film.chunk(2, dim=-1)
                ts_vec = ts_vec * (1.0 + 0.1 * torch.tanh(film_gamma)) + 0.1 * torch.tanh(film_beta)

            z_ts = self.ctrl_ts_proj(ts_vec)         # [B, D]
            z_text = self.ctrl_text_proj(text_vec)   # [B, D]

            # Alignment losses. Keep TS-Text; add TS-Freq by default; keep Freq-Text optional.
            align_losses = []
            ts_align = self.ts_align_proj(self.ts_scale * ts_vec)
            txt_align = self.text_align_proj(self.text_scale * text_vec)
            if self.use_temporal_proto and self.use_proto_infonce:
                align_losses.append(self.align_loss_fn(ts_align, txt_align, proto_ctx))
            else:
                align_losses.append(self.align_loss_fn(ts_align, txt_align))

            # 文本一致性分数必须来自已经被 TS-Text InfoNCE 直接监督的共享对齐空间。
            # 原实现使用 ctrl_ts_proj / ctrl_text_proj 计算 s_text；当
            # reliability_detach_score=True 且 freq-text alignment 关闭时，ctrl_text_proj
            # 缺少有效梯度，可能使 s_text 退化为随机投影下的相似度。
            s_text = F.cosine_similarity(
                F.normalize(ts_align, dim=-1),
                F.normalize(txt_align, dim=-1),
                dim=-1,
                eps=1e-6
            ).unsqueeze(-1)  # [B, 1]

            z_freq = None
            if freq_vec is not None:
                z_freq = self.ctrl_freq_proj(freq_vec)
                s_freq = F.cosine_similarity(
                    F.normalize(z_ts, dim=-1),
                    F.normalize(z_freq, dim=-1),
                    dim=-1,
                    eps=1e-6
                ).unsqueeze(-1)  # [B, 1]

                if self.use_three_view_align and self.use_ts_freq_align:
                    if self.use_temporal_proto and self.use_proto_infonce:
                        align_losses.append(self.align_loss_fn(z_ts, z_freq, proto_ctx))
                    else:
                        align_losses.append(self.align_loss_fn(z_ts, z_freq))
                if self.use_three_view_align and self.use_freq_text_align:
                    if self.use_temporal_proto and self.use_proto_infonce:
                        align_losses.append(self.align_loss_fn(z_freq, z_text, proto_ctx))
                    else:
                        align_losses.append(self.align_loss_fn(z_freq, z_text))

            align_loss = sum(align_losses) / max(1, len(align_losses))
            if self.use_temporal_proto:
                align_loss = align_loss + self.proto_aux_lambda * proto_aux_loss

            # Evidence embedding: e is a gate input, not only prompt text.
            evidence_emb = self.evidence_gate_proj(evidence)
            if not self.use_evidence_gate:
                evidence_emb = torch.zeros_like(evidence_emb)

            proto_gate_vec = proto_ctx if (self.proto_gate_dim > 0 and proto_ctx is not None) else None

            # ===== Text residual candidate =====
            if self.text_residual_mode == "pred":
                r_text = self.text_pred_head(text_vec).unsqueeze(-1).expand(-1, -1, N_vars)  # [B, S, N]
            else:
                # Token residual is converted to a prediction-space residual rather than replacing backbone tokens.
                if self.text_use_cross_attn:
                    txt_ctx, _ = self.text_cross_attn(query=var_tokens, key=text_seq, value=text_seq)
                    text_token_res = self.text_token_proj(txt_ctx)  # [B, N, D]
                else:
                    text_token_res = self.text_token_proj(text_vec).unsqueeze(1).expand(-1, N_vars, -1)
                r_text = self.text_token_pred_head(text_token_res).permute(0, 2, 1)  # [B, S, N]
            r_text = self._limit_residual(r_text, budget=self.residual_budget_text)

            if self.use_reliability_gate:
                score_text = s_text.detach() if self.reliability_detach_score else s_text
                if self.use_var_reliability_gate:
                    ts_gate_tokens = self.ts_proj(var_tokens)  # [B, N, D]
                    text_gate_tokens = text_vec.unsqueeze(1).expand(-1, N_vars, -1)
                    score_tokens = score_text.unsqueeze(1).expand(-1, N_vars, -1)
                    evidence_tokens = evidence_emb.unsqueeze(1).expand(-1, N_vars, -1)
                    text_gate_parts = [ts_gate_tokens, text_gate_tokens, score_tokens, evidence_tokens]
                    if proto_gate_vec is not None:
                        text_gate_parts.append(proto_gate_vec.unsqueeze(1).expand(-1, N_vars, -1))
                    text_gate_in = torch.cat(text_gate_parts, dim=-1)
                    text_logits = self.text_reliability_gate(text_gate_in).squeeze(-1) + self.text_reliability_bias
                    g_text = self._bounded_gate(text_logits, self.reliability_min_gate)  # [B, N]
                else:
                    text_gate_parts = [ts_vec, text_vec, score_text, evidence_emb]
                    if proto_gate_vec is not None:
                        text_gate_parts.append(proto_gate_vec)
                    text_gate_in = torch.cat(text_gate_parts, dim=-1)
                    text_logits = self.text_reliability_gate(text_gate_in) + self.text_reliability_bias
                    g_text = self._bounded_gate(text_logits, self.reliability_min_gate)  # [B, 1]
            else:
                gate_parts = [ts_vec, text_vec]
                if proto_gate_vec is not None:
                    gate_parts.append(proto_gate_vec)
                fusion_logit = self.text_gate_mlp(torch.cat(gate_parts, dim=-1))
                g_text = torch.sigmoid(fusion_logit + self.text_residual_bias)  # [B, 1]

            y_hat = y_hat + g_text.unsqueeze(1) * r_text

            # ===== Frequency residual reliability gate =====
            if (self.use_reliability_gate
                    and self.use_freq_residual
                    and self.use_freq_feat
                    and (freq_res_out is not None)
                    and (freq_vec is not None)):
                if s_freq is None:
                    z_freq = self.ctrl_freq_proj(freq_vec)
                    s_freq = F.cosine_similarity(
                        F.normalize(z_ts, dim=-1),
                        F.normalize(z_freq, dim=-1),
                        dim=-1,
                        eps=1e-6
                    ).unsqueeze(-1)
                score_freq = s_freq.detach() if self.reliability_detach_score else s_freq

                if self.use_var_reliability_gate:
                    ts_gate_tokens = self.ts_proj(var_tokens)  # [B, N, D]
                    freq_gate_tokens = aligned[:, :N_vars, :]  # [B, N, D]
                    score_tokens = score_freq.unsqueeze(1).expand(-1, N_vars, -1)
                    evidence_tokens = evidence_emb.unsqueeze(1).expand(-1, N_vars, -1)
                    freq_gate_parts = [ts_gate_tokens, freq_gate_tokens, score_tokens, evidence_tokens]
                    if proto_gate_vec is not None:
                        freq_gate_parts.append(proto_gate_vec.unsqueeze(1).expand(-1, N_vars, -1))
                    freq_gate_in = torch.cat(freq_gate_parts, dim=-1)
                    freq_logits = self.freq_reliability_gate(freq_gate_in).squeeze(-1) + self.freq_reliability_bias
                    g_freq = self._bounded_gate(freq_logits, self.reliability_min_gate)  # [B, N]
                else:
                    freq_gate_parts = [ts_vec, freq_vec, score_freq, evidence_emb]
                    if proto_gate_vec is not None:
                        freq_gate_parts.append(proto_gate_vec)
                    freq_gate_in = torch.cat(freq_gate_parts, dim=-1)
                    freq_logits = self.freq_reliability_gate(freq_gate_in) + self.freq_reliability_bias
                    g_freq = self._bounded_gate(freq_logits, self.reliability_min_gate)  # [B, 1]

                y_hat = y_hat + g_freq.unsqueeze(1) * freq_res_out

        # Keep auxiliary tensors in the same output scale as y_hat.
        y_base_aux = y_base
        r_freq_aux = freq_res_out
        r_text_aux = r_text

        if self.use_norm:
            scale = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            shift = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y_hat = y_hat * scale + shift
            y_base_aux = y_base * scale + shift
            if r_freq_aux is not None:
                r_freq_aux = r_freq_aux * scale
            if r_text_aux is not None:
                r_text_aux = r_text_aux * scale

        aux = {
            "align_loss": align_loss,
            "y_base": y_base_aux,
            "r_freq": r_freq_aux,
            "r_text": r_text_aux,
            "g_text": g_text,
            "g_freq": g_freq,
            "s_text": s_text,
            "s_freq": s_freq,
            "evidence": evidence,
            "proto_ctx": proto_ctx,
            "proto_usage": proto_usage,
            "proto_aux_loss": proto_aux_loss,
        }
        self.last_aux = {
            k: (None if v is None else v.detach()) if torch.is_tensor(v) else v
            for k, v in aux.items()
        }
        return y_hat, attns, align_loss, aux

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        y_hat, attns, align_loss, aux = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred = y_hat[:, -self.pred_len:, :]

        if self.return_aux:
            if self.output_attention:
                return (pred, attns, align_loss, aux)
            if self.use_llm and self.return_align_loss:
                return (pred, align_loss, aux)
            return (pred, aux)

        if self.use_llm and self.return_align_loss:
            if self.output_attention:
                return (pred, attns, align_loss)
            return (pred, align_loss)

        if self.output_attention:
            return pred, attns
        return pred

    @staticmethod
    def _bounded_gate(logits, min_gate=0.0):
        gate = torch.sigmoid(logits)
        if min_gate > 0:
            gate = min_gate + (1.0 - min_gate) * gate
        return gate

    def _limit_residual(self, residual, budget=1.0):
        if residual is None:
            return None
        if self.residual_clip > 0:
            residual = torch.tanh(residual / self.residual_clip) * self.residual_clip
        return budget * residual

    # ------------------------- text branch helpers -------------------------

    def _run_text_branch(self, x_raw, evidence=None):
        B = x_raw.size(0)
        prompts = self._build_sample_prompts(x_raw, evidence=evidence)

        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.prompt_max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(x_raw.device)
        attn_mask = enc["attention_mask"].to(x_raw.device)

        wte = self.llm.get_input_embeddings()
        stat_emb = wte(input_ids)  # [B, T, llm_dim]
        learn = self.learn_prompt.unsqueeze(0).expand(B, -1, -1)
        prompt_emb = torch.cat([learn, stat_emb], dim=1)
        learn_mask = torch.ones(B, self.learn_prompt_len, device=x_raw.device, dtype=attn_mask.dtype)
        attn = torch.cat([learn_mask, attn_mask], dim=1)

        out = self.llm(inputs_embeds=prompt_emb, attention_mask=attn).last_hidden_state  # [B, P, llm_dim]
        out = self.text_proj(out)  # [B, P, d_model]

        if self.llm_pool == "last":
            text_vec = out[:, -1, :]
        else:
            m = attn.unsqueeze(-1).float()
            text_vec = (out * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        return out, text_vec

    def _build_sample_prompts(self, x_raw, evidence=None):
        if evidence is None:
            evidence = self._compute_sample_evidence(x_raw)
        prompts = []
        for b in range(x_raw.size(0)):
            spec_entropy, peak_ratio, trend_strength, vol, diff_vol, corr_strength = evidence[b].tolist()

            if corr_strength > 0.45:
                rel_desc = "strong and likely useful"
            elif corr_strength > 0.25:
                rel_desc = "moderate and possibly unstable"
            else:
                rel_desc = "weak and likely unreliable"

            if peak_ratio > 0.25:
                freq_desc = "clear periodic structure"
            elif peak_ratio > 0.12:
                freq_desc = "mixed periodic structure"
            else:
                freq_desc = "unclear periodic structure"

            if vol > 1.0:
                noise_desc = "high noise or strong fluctuation"
            elif vol > 0.2:
                noise_desc = "moderate fluctuation"
            else:
                noise_desc = "relatively stable fluctuation"

            prompts.append(
                "Forecasting structure assessment. "
                f"The series shows {freq_desc}. "
                f"Spectral entropy is {spec_entropy:.3f} and peak ratio is {peak_ratio:.3f}. "
                f"Trend strength is {trend_strength:.3f}. "
                f"Overall volatility is {vol:.3f} and first-difference volatility is {diff_vol:.3f}, indicating {noise_desc}. "
                f"Cross-channel coupling is {rel_desc} with score {corr_strength:.3f}. "
                "Generate a compact semantic prior for long-horizon forecasting, emphasizing whether global periodic cues or local temporal interaction should be treated cautiously."
            )
        return prompts

    def _compute_sample_evidence(self, x_raw):
        """
        x_raw: [B, L, N]
        return: [B, 6]
        """
        B, L, N = x_raw.shape
        eps = 1e-6

        x = x_raw
        x_center = x - x.mean(dim=1, keepdim=True)
        x_std = x_center.std(dim=1, keepdim=True).clamp_min(eps)
        x_norm = x_center / x_std

        X = torch.fft.rfft(x_center.transpose(1, 2), dim=-1)   # [B, N, F]
        A = X.abs()[..., 1:]                                   # remove DC

        if A.size(-1) == 0:
            spec_entropy = x.new_zeros(B, 1)
            peak_ratio = x.new_zeros(B, 1)
        else:
            P = A / A.sum(dim=-1, keepdim=True).clamp_min(eps)
            spec_entropy = -(P * P.clamp_min(eps).log()).sum(dim=-1)
            spec_entropy = spec_entropy / math.log(A.size(-1) + 1.0)
            spec_entropy = spec_entropy.mean(dim=-1, keepdim=True)

            peak_ratio = A.max(dim=-1).values / A.sum(dim=-1).clamp_min(eps)
            peak_ratio = peak_ratio.mean(dim=-1, keepdim=True)

        trend_strength = (x[:, -1, :] - x[:, 0, :]).abs().mean(dim=-1, keepdim=True)
        volatility = x_center.std(dim=1).mean(dim=-1, keepdim=True)

        if L > 1:
            dx = x[:, 1:, :] - x[:, :-1, :]
            diff_volatility = dx.std(dim=1).mean(dim=-1, keepdim=True)
        else:
            diff_volatility = x.new_zeros(B, 1)

        corr = torch.matmul(x_norm.transpose(1, 2), x_norm) / float(L)
        eye = torch.eye(N, device=x.device).unsqueeze(0)
        offdiag = 1.0 - eye
        denom = offdiag.sum().clamp_min(1.0)
        corr_strength = (corr.abs() * offdiag).sum(dim=(1, 2), keepdim=True) / denom
        corr_strength = corr_strength.view(B, 1)

        evidence = torch.cat(
            [spec_entropy, peak_ratio, trend_strength, volatility, diff_volatility, corr_strength],
            dim=-1,
        )
        evidence = torch.nan_to_num(evidence, nan=0.0, posinf=0.0, neginf=0.0)
        return evidence

    # ------------------------- EMAformer-style helpers -------------------------

    def _build_embedding_armor(self, x_enc, T_base):
        B, _, N_vars = x_enc.shape
        device = x_enc.device
        aux = 0.0

        num_channel_emb = self.channel_emb.num_embeddings
        ch_ids = torch.arange(N_vars, device=device)
        if N_vars > num_channel_emb:
            ch_ids = ch_ids % num_channel_emb
        ch_ids = ch_ids.unsqueeze(0).expand(B, -1)

        if self.use_channel_emb:
            aux = aux + self.channel_emb(ch_ids)

        if self.use_phase_emb or self.use_joint_channel_phase_emb:
            phase_ids = self._estimate_phase_ids(x_enc)
        else:
            phase_ids = None

        if self.use_phase_emb:
            aux = aux + self.phase_emb(phase_ids)

        if self.use_joint_channel_phase_emb:
            joint_ids = ch_ids * self.ema_period + phase_ids
            aux = aux + self.channel_phase_emb(joint_ids)

        aux = self.ema_embed_dropout(self.ema_embed_ln(aux))
        return self._align_to_token_dim(aux, T_base)

    def _estimate_phase_ids(self, x_enc):
        eps = 1e-6
        x = x_enc.permute(0, 2, 1)
        x = x - x.mean(dim=-1, keepdim=True)
        X = torch.fft.rfft(x, dim=-1)
        mag = torch.abs(X)[..., 1:]

        if mag.size(-1) == 0:
            return torch.zeros(x.size(0), x.size(1), dtype=torch.long, device=x.device)

        dom_idx = mag.argmax(dim=-1) + 1
        dom_coeff = torch.gather(X, -1, dom_idx.unsqueeze(-1)).squeeze(-1)
        angle = torch.atan2(dom_coeff.imag + eps, dom_coeff.real + eps)
        phase = ((angle + math.pi) / (2 * math.pi) * self.ema_period).long()
        phase = phase.clamp_(0, self.ema_period - 1)
        return phase

    # ------------------------- freq helpers -------------------------

    @staticmethod
    def _apply_gate_bias(g, bias_param):
        g = g.clamp(1e-4, 1 - 1e-4)
        logits = torch.log(g / (1 - g))
        return torch.sigmoid(logits + bias_param)

    @staticmethod
    def _align_to_token_dim(freq_emb, T):
        B, N, E = freq_emb.shape
        if T == N:
            return freq_emb
        if T > N:
            pad = T - N
            return F.pad(freq_emb, (0, 0, 0, pad), value=0.0)
        return freq_emb[:, :T, :]

    def _match_token_proj_in_dim(self, freq_tokens_raw, N_vars):
        B, K, Nv = freq_tokens_raw.shape
        in_dim = self.freq_token_proj.in_features
        if in_dim == Nv:
            return freq_tokens_raw
        if in_dim > Nv:
            pad = in_dim - Nv
            return F.pad(freq_tokens_raw, (0, pad), value=0.0)
        return freq_tokens_raw[:, :, :in_dim]

    def _freq_spectrum(self, x_enc):
        x = x_enc.permute(0, 2, 1)
        x = x - x.mean(dim=-1, keepdim=True)
        X = torch.fft.rfft(x, dim=-1)
        A = torch.abs(X)[..., 1:]
        return A

    def _select_k_bins(self, A):
        F_bins = A.size(-1)
        K = min(self.freq_k, F_bins)
        if self.freq_mode == "topk":
            topv, _ = torch.topk(A, k=K, dim=-1)
            freq_bins = topv
        else:
            freq_bins = A[..., :K]
        if K < self.freq_k:
            pad = self.freq_k - K
            freq_bins = F.pad(freq_bins, (0, pad), value=0.0)
        return freq_bins

    def _freq_features(self, x_enc):
        A = self._freq_spectrum(x_enc)
        if self.freq_use_log_amp:
            A = torch.log1p(A)
        return self._select_k_bins(A)
