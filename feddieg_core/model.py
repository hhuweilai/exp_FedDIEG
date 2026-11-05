"""FedDIEG 模型（逐行对齐 `exp_01/run_exp01_movielens_aligned.py` L251-L586）。

对齐关系
--------
``SASRecEncoder``        ← aligned L251-L310（注意 item_emb 乘了 sqrt(dim)，A 版没有）
``FedDIEGModel``         ← aligned L433-L502
``init_weights``         ← aligned L239-L248
``bce_seq_loss``         ← aligned L194-L199

与 aligned 版的唯一差异（都是为了"批处理等价 + 显式化"，不改变数值）
------------------------------------------------------------------
1. ``extract_long_term_interest`` 的邻居聚合从 ``dim=0`` 改为 ``dim=1``。
   aligned 版一次只处理**一个客户端**的邻居，所以 ``sum(..., dim=0)`` 是对邻居维求和；
   这里允许 (B, K, D) 的批量形式，B=1 时与 aligned 版逐元素等价
   （`tests` 里有数值等价断言）。
2. 个性化层判定改为**结构化前缀匹配**。aligned 版用 ``"encoder.forward" in key``，
   能同时命中 ``forward_layernorms`` 与 ``forward_layers``，是对的；
   但 A 版（`FedDIEG.py` L88-L90）用 ``"forward_layers" in key``，
   匹配不到 ``forward_layernorms``，导致 "attention 前的 LayerNorm 个性化、
   FFN 前的 LayerNorm 被聚合" 的不对称。这里把 B 版的语义显式写成前缀列表，
   既对齐 B 版，又不会退化成 A 版的问题。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FedDIEGConfig

# ======================================================================================
# 参数分组：哪些层留在客户端（个性化），哪些层上传服务器（共享）
# ======================================================================================
# 对齐 B 版语义（aligned L589-L596）：
#   personal: encoder 的 attention / FFN（含各自的 LayerNorm）、encoder.last_layernorm、
#             W_proj、q_att
#   shared  : encoder.item_emb、encoder.pos_emb、W_gnn、W_g
PERSONAL_PREFIXES: Tuple[str, ...] = (
    "encoder.attention_layernorms.",
    "encoder.attention_layers.",
    "encoder.forward_layernorms.",
    "encoder.forward_layers.",
    "encoder.last_layernorm.",
    "W_proj.",
    "q_att",
)
SHARED_PREFIXES: Tuple[str, ...] = (
    "encoder.item_emb.",
    "encoder.pos_emb.",
    "W_gnn.",
    "W_g.",
)


def is_personal_param(name: str) -> bool:
    """该参数名是否属于客户端个性化层（不上传）。"""
    if name.startswith(PERSONAL_PREFIXES):
        return True
    if name.startswith(SHARED_PREFIXES):
        return False
    # 兜底：不认识的参数一律按"共享"处理会影响 FedAvg 语义，这里宁可报错
    raise KeyError(f"未知参数名（无法判定个性化/共享）：{name}")


def split_state_keys(model: nn.Module) -> Tuple[List[str], List[str]]:
    """返回 (personal_keys, shared_keys)。"""
    personal: List[str] = []
    shared: List[str] = []
    for name in model.state_dict().keys():
        (personal if is_personal_param(name) else shared).append(name)
    return personal, shared


# ======================================================================================
# 基础模块
# ======================================================================================
def bce_seq_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对齐 aligned L194-L199。"""
    labels_pos = torch.ones_like(pos_logits)
    labels_neg = torch.zeros_like(neg_logits)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loss = criterion(pos_logits, labels_pos) + criterion(neg_logits, labels_neg)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def init_weights(module: nn.Module) -> None:
    """对齐 aligned L239-L248。"""
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(module.weight, 1.0)


class SASRecEncoder(nn.Module):
    """对齐 aligned L251-L310。"""

    def __init__(self, num_items: int, cfg: FedDIEGConfig):
        super().__init__()
        self.num_items = num_items
        self.max_len = cfg.max_len
        self.hidden_units = cfg.latent_dim
        self.item_emb = nn.Embedding(num_items + 1, cfg.latent_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.latent_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(cfg.latent_dim, eps=1e-8)

        for _ in range(cfg.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(cfg.latent_dim, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    cfg.latent_dim,
                    cfg.num_heads,
                    dropout=cfg.dropout_rate,
                    batch_first=True,
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(cfg.latent_dim, eps=1e-8))
            self.forward_layers.append(
                nn.Sequential(
                    nn.Linear(cfg.latent_dim, cfg.latent_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout_rate),
                    nn.Linear(cfg.latent_dim, cfg.latent_dim),
                    nn.Dropout(cfg.dropout_rate),
                )
            )
        self.apply(init_weights)

    def forward(self, log_seqs: torch.Tensor) -> torch.Tensor:
        seq_len = log_seqs.shape[1]
        positions = torch.arange(seq_len, device=log_seqs.device).unsqueeze(0).expand_as(log_seqs)
        timeline_mask = log_seqs == 0
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=log_seqs.device), diagonal=1
        )

        seqs = self.item_emb(log_seqs) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        seqs *= (~timeline_mask.unsqueeze(-1)).float()

        for i in range(len(self.attention_layers)):
            q = self.attention_layernorms[i](seqs)
            attn_out, _ = self.attention_layers[i](q, seqs, seqs, attn_mask=causal_mask)
            seqs = seqs + attn_out
            ffn_in = self.forward_layernorms[i](seqs)
            seqs = seqs + self.forward_layers[i](ffn_in)
            seqs *= (~timeline_mask.unsqueeze(-1)).float()
        return self.last_layernorm(seqs)


class FedDIEGModel(nn.Module):
    """对齐 aligned L433-L502（长短期兴趣 + 图聚合 + 联合损失）。"""

    def __init__(self, num_items: int, cfg: FedDIEGConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SASRecEncoder(num_items, cfg)
        self.W_proj = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.q_att = nn.Parameter(torch.randn(cfg.latent_dim))
        self.W_gnn = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.W_g = nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False)
        # 对齐 aligned L442：整体 apply 一次 init_weights（encoder 内部也 init 过一次，
        # 数值分布一致；这里保持与 aligned 版同样的调用顺序）
        self.apply(init_weights)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    # ---------------------------------------------------------------- 短期兴趣
    def extract_short_term_interest(self, seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对齐 aligned L448-L455：注意力池化得到 s_i。"""
        seq_emb = self.encoder(seqs)
        timeline_mask = seqs == 0
        proj_h = torch.tanh(self.W_proj(seq_emb))
        scores = torch.matmul(proj_h, self.q_att) / (self.cfg.latent_dim ** 0.5)
        scores = scores.masked_fill(timeline_mask, -1e4)
        alphas = F.softmax(scores, dim=-1).unsqueeze(-1)
        return seq_emb, torch.sum(alphas * seq_emb, dim=1)

    # ---------------------------------------------------------------- 长期兴趣
    def extract_long_term_interest(
        self,
        neighbor_summaries: Optional[torch.Tensor],
        neighbor_weights: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """图消息聚合。

        支持两种形状：
          * (K, D) / (K,)     —— 单客户端（与 aligned 版完全一致）
          * (B, K, D) / (B, K) —— 批量形式（B=1 时与 aligned 版等价）
        """
        dim = self.cfg.latent_dim
        if neighbor_summaries is None or neighbor_weights is None:
            return torch.zeros(batch_size, dim, device=self.q_att.device)
        if neighbor_summaries.numel() == 0:
            return torch.zeros(batch_size, dim, device=self.q_att.device)

        if neighbor_summaries.dim() == 2:  # (K, D) -> (1, K, D)
            neighbor_summaries = neighbor_summaries.unsqueeze(0)
            neighbor_weights = neighbor_weights.unsqueeze(0)
        if neighbor_summaries.size(0) != batch_size:
            # 单客户端的邻居被复用到整个 batch（aligned 版的 expand 语义）
            neighbor_summaries = neighbor_summaries.expand(batch_size, -1, -1)
            neighbor_weights = neighbor_weights.expand(batch_size, -1)

        messages = self.W_gnn(neighbor_summaries)                       # (B, K, D)
        agg_i = torch.sum(neighbor_weights.unsqueeze(-1) * messages, dim=1)  # (B, D)
        return torch.relu(agg_i)                                        # (B, D)

    # ---------------------------------------------------------------- 前向
    def forward(
        self,
        seqs: torch.Tensor,
        neighbor_summaries: Optional[torch.Tensor] = None,
        neighbor_weights: Optional[torch.Tensor] = None,
        graph_mix: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (u_i, s_i)。u_i = seq_emb + graph_mix * W_g(relu(Σ w_ij W_gnn(h_j)))。

        对齐 aligned L469-L477；`graph_mix` 默认 1.0（A/B 版无缩放），只为敏感性分析保留。
        """
        seq_emb, s_i = self.extract_short_term_interest(seqs)
        g_i = self.extract_long_term_interest(neighbor_summaries, neighbor_weights, seqs.size(0))
        delta = self.W_g(g_i.unsqueeze(1).expand_as(seq_emb))
        return seq_emb + graph_mix * delta, s_i

    # ---------------------------------------------------------------- 联合损失
    def compute_joint_loss(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        mask: torch.Tensor,
        current_s_i: torch.Tensor,
        prev_s_i: Optional[torch.Tensor],
        current_delta_e: Optional[torch.Tensor],
        prev_struct_target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """对齐 aligned L479-L502。

        `prev_struct_target` 传什么由 `cfg.struct_target` 决定：
          * ``delta_e``（默认，修复版）：上一轮的 Δe 向量，与 current_delta_e 同量纲；
          * ``edge_weight``（A/B 字面行为）：上一轮的 softmax 边权，量纲不一致。
        """
        rec_loss = bce_seq_loss(pos_logits, neg_logits, mask)

        temp_loss = torch.tensor(0.0, device=pos_logits.device)
        if prev_s_i is not None and self.cfg.lambda_temp > 0:
            temp_loss = F.mse_loss(current_s_i, prev_s_i.detach())

        struct_loss = torch.tensor(0.0, device=pos_logits.device)
        if (
            prev_struct_target is not None
            and current_delta_e is not None
            and prev_struct_target.shape == current_delta_e.shape
            and self.cfg.lambda_struct > 0
        ):
            struct_loss = F.mse_loss(current_delta_e, prev_struct_target.detach())

        total = (
            rec_loss
            + self.cfg.lambda_temp * temp_loss
            + self.cfg.lambda_struct * struct_loss
        )
        return total, rec_loss, temp_loss, struct_loss

    # ---------------------------------------------------------------- 便捷方法
    def score_user(
        self,
        seqs: torch.Tensor,
        candidate_items: torch.Tensor,
        neighbor_summaries: Optional[torch.Tensor] = None,
        neighbor_weights: Optional[torch.Tensor] = None,
        graph_mix: float = 1.0,
    ) -> torch.Tensor:
        """对一个 batch 的用户序列与候选集打分（对齐 A/B 的 evaluate：取最后位置）。"""
        feats, _ = self.forward(seqs, neighbor_summaries, neighbor_weights, graph_mix)
        last_step = feats[:, -1, :]
        item_embs = self.item_emb(candidate_items)
        return torch.matmul(last_step.unsqueeze(1), item_embs.transpose(-1, -2)).squeeze(1)


def build_model(num_items: int, cfg: FedDIEGConfig) -> FedDIEGModel:
    return FedDIEGModel(num_items, cfg)


def state_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}
