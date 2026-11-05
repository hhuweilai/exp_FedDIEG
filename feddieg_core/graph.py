"""动态兴趣图：论文公式 ``e_ij^(t+1) = ρ·e_ij^(t) + (1-ρ)·Δe_ij^t``。

论文定义（`F:\\FedDIEG-main\\tools\\edit_mid_report.py` L242-L287）
----------------------------------------------------------------
    a_i^t   = 1[ ||s̄_i^t|| > ε ]                                  （活跃节点掩码）
    Δe_ij^t = a_i^t · a_j^t · max(0, cos(s̄_i^t, s̄_j^t)),  i ≠ j
    e_ij^(t+1) = ρ·e_ij^(t) + (1-ρ)·Δe_ij^t                        （所有点对）
    Top-K → softmax → 作为 GNN 边权

本模块相对审计版 `exp_val_1026/.../val02_run_ema.py::update_graph` 修了 5 处
----------------------------------------------------------------------------
【F1】EMA 的量纲固定为 ``max(0,cos)``。
      旧 val01 把"归一化后、行和为 1 的相对权重"当成 e_ij^(t) 与原始余弦混合，
      而 min_sim=0.20 又作用在这个混合量上 —— 审计复现：ρ=0.8 时整行会被阈值打空、
      在"满 K"和"0 条边"之间振荡。这里 e 与 Δe 同量纲，与论文一致。

【F2】无效槽位写 ``-1``（旧 val02/val03 写的是 ``active[topk_positions]`` 得到的伪 id）。
      审计实测：`n_neighbors` 在 7860 行里恒为 10.0，下游 probe 把伪 id 当成真邻居。

【F3】不再用 ``scatter_`` 把 ``-1`` 钳到列 0。
      旧代码 ``prev.scatter_(1, old_col.clamp_min(0), ...)`` 会让 padding 槽位把 0
      写到某个真实候选的列上，覆盖其历史边权。这里用 ``gather``/高级索引，索引永远合法。

【F4】EMA 状态保存在**候选池**（``graph_state="pool"``）上，而不是只保存当前 Top-K。
      这是"图冻结"的真正原因：旧实现里掉出 Top-K 的边状态被丢弃，新边每轮都从
      (1-ρ)·Δe 重新起步，要挤进 Top-K 需要 Δe 有 1/(1-ρ) 倍优势（ρ=0.8 需 5 倍、
      ρ=0.95 需 20 倍），余弦量级给不出这种差距 ⇒ Top-K 成员永久不变。
      审计复现（audit_freeze_origin.py）：同一公式、同一场景，只把状态从"只存 Top-K"
      换成"保留候选池"，ρ=0.8 时漂移用户的邻居就会在 O(1/(1-ρ)) 轮内完成迁移。

      注意："候选池"必须配合**正确的保留规则**才有用。如果按 EMA 值选池，新出现的
      高相似边每轮都会被低 EMA 挤掉，效果与只存 Top-K 相同（第一版修复就踩了这个坑）。
      这里按 ``max(EMA, Δe)`` 保留——"历史强或当前强都留下"——容量 ``graph_pool_size``。
      新边的 EMA 初值仍按论文取 (1-ρ)·Δe（``graph_new_edge="zero"``）；
      ``"warm"`` 则首次观测直接取 Δe（A/B 版对"无历史行"就是这么做的），仅作消融。

      ``graph_state="topk"`` 刻意保留旧行为（容量 K + 只按 EMA 保留），用于对照复现。

【F5】默认 ``graph_update_scope="all"``：每轮用**缓存摘要**给全部活跃节点异步更新边
      （论文 L285 的语义），而不是只重算本轮被抽样的用户。

其它对齐点
----------
* 边权用 ``softmax(Top-K 的 e 值)``，与 `FedDIEG.py` L438 / aligned L1485 一致。
* ``min_sim`` 默认 0（A/B 主程序没有相似度阈值，只有 ||s̄||>ε 的活跃掩码）。
* ECR 用 Jaccard 距离（对称差/并集），与旧 val01/val02/val03 一致，便于和已有结果对比；
  另外单独返回 ``n_added``（本轮新进入 Top-K 的边数）——审计建议用它区分"慢"与"冻结"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import FedDIEGConfig


def batched_indices(n: int, batch_size: int, device: torch.device) -> List[torch.Tensor]:
    idx = torch.arange(n, device=device)
    return [idx[s : s + batch_size] for s in range(0, n, max(1, batch_size))]


@dataclass
class GraphUpdateStats:
    """一次图更新的统计量。

    ``updated``   本轮实际被重算的用户 id（CPU）
    ``ecr``       每个用户的边变化率（Jaccard 距离，CPU）
    ``n_added``   每个用户本轮**新进入** Top-K 的边数（CPU）
    ``n_kept``    每个用户本轮保留的边数（CPU）
    ``pool_fill`` 候选池的平均占用率（0~1）
    """

    updated: torch.Tensor
    ecr: torch.Tensor
    n_added: torch.Tensor
    n_kept: torch.Tensor
    pool_fill: float

    def __len__(self) -> int:
        return int(self.updated.numel())


class DynamicInterestGraph:
    """服务器端维护的动态兴趣图（一行一个客户端）。"""

    def __init__(self, cfg: FedDIEGConfig, num_users: int, device: torch.device):
        self.cfg = cfg
        self.num_users = int(num_users)
        self.device = device
        self.k = int(max(1, min(cfg.neighbors, max(int(num_users) - 1, 1))))
        self.pool_size = int(min(max(cfg.resolved_pool_size(), self.k), max(int(num_users), 1)))
        # "历史强或当前强都留下"：只在 pool 模式启用（topk 模式复现旧行为）
        self.retention_by_delta = cfg.graph_state == "pool"
        if cfg.graph_state == "topk":
            # 复现旧行为：池子大小 = K，且只按 EMA 值保留，掉出 Top-K 的状态被丢弃
            self.pool_size = self.k

        self.ids = torch.full((self.num_users, self.k), -1, dtype=torch.long, device=device)
        self.weights = torch.zeros((self.num_users, self.k), dtype=torch.float32, device=device)
        self.mask = torch.zeros(self.num_users, dtype=torch.bool, device=device)

        self.pool_ids = torch.full(
            (self.num_users, self.pool_size), -1, dtype=torch.long, device=device
        )
        self.pool_val = torch.zeros((self.num_users, self.pool_size), dtype=torch.float32, device=device)
        self.pool_mask = torch.zeros((self.num_users, self.pool_size), dtype=torch.bool, device=device)

    # ==================================================================================
    # 给模型用：取某批用户的邻居摘要 / 边权
    # ==================================================================================
    def gather(
        self, summary_bank: torch.Tensor, users: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (neighbor_summaries (B,K,D), neighbor_weights (B,K), valid (B,))。

        ``-1`` 槽位对应 summary_bank[0]（任意行），但边权被显式置 0，因此对聚合无贡献。
        ``users`` 允许是 CPU 张量（内部会挪到图的 device），这样调用方不必关心设备。
        """
        users = users.to(self.device, non_blocking=True)
        nbr = self.ids[users]                                   # (B, K)
        valid_slot = nbr >= 0
        user_valid = self.mask[users] & valid_slot.any(dim=1)
        w = self.weights[users] * valid_slot.to(self.weights.dtype)
        w = w * user_valid.to(w.dtype).unsqueeze(1)
        summaries = summary_bank[nbr.clamp_min(0)]
        return summaries, w, user_valid

    def neighbors(self, users: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (ids (B,K), weights (B,K), valid (B,))，ids 中无效槽位是 -1。"""
        users = users.to(self.device, non_blocking=True)
        return self.ids[users], self.weights[users], self.mask[users] & (self.ids[users] >= 0).any(dim=1)

    def evaluate_ready(self, round_idx: int) -> bool:
        return round_idx > self.cfg.graph_warmup_rounds

    def describe(self) -> dict:
        return {
            "neighbors": self.k,
            "pool_size": self.pool_size,
            "state": self.cfg.graph_state,
            "rho": self.cfg.rho,
            "min_sim": self.cfg.min_sim,
            "update_scope": self.cfg.graph_update_scope,
            "warmup_rounds": self.cfg.graph_warmup_rounds,
            "ema_scale": "max(0, cos)",
        }

    # ==================================================================================
    # 服务器端更新
    # ==================================================================================
    @torch.no_grad()
    def update(
        self,
        summary_bank: torch.Tensor,
        active_clients: torch.Tensor,
        update_users: torch.Tensor,
        round_idx: int,
    ) -> Optional[GraphUpdateStats]:
        """对 ``update_users`` 的邻居做 Top-K + EMA 平滑。

        参数
        ----
        summary_bank     (N, D) 服务器缓存的兴趣摘要 s̄
        active_clients   (N,)   是否曾经上传过（用于文档/统计；真正的 a_i 用 ||s̄||>ε 判定）
        update_users     本轮要重算邻居行的用户 id
        round_idx        1 基轮次；``<= warmup`` 时不做图更新（对齐 A/B 的 WARMUP_ROUNDS）
        """
        device = self.device
        cfg = self.cfg
        empty_long = torch.zeros(0, dtype=torch.long, device="cpu")
        empty_float = torch.zeros(0, dtype=torch.float32, device="cpu")
        if round_idx <= cfg.graph_warmup_rounds:
            return None

        # a_i = 1[||s̄_i|| > ε]：从未上传 / 摘要为零的客户端不参与（论文的活跃节点掩码）
        norms = summary_bank.norm(dim=1)
        act = norms > cfg.active_norm_eps
        n_active = int(act.sum().item())
        if n_active <= 1:
            return None

        update_users = update_users.to(device, non_blocking=True)
        rows = update_users[act[update_users]]
        if rows.numel() == 0:
            return None

        k = min(self.k, n_active - 1)
        pool = int(self.pool_size)  # 池宽固定，保证与 pool_ids 的列数一致
        if pool < k:
            k = pool
        norm_bank = F.normalize(summary_bank, p=2, dim=1, eps=1e-8)
        rho = float(min(max(cfg.rho, 0.0), 0.999))
        min_sim = float(cfg.min_sim)

        n_rows = int(rows.numel())
        # 分块大小：让每个块的 sim 矩阵不超过约 2^24 个元素（~64MB/float32）
        block_size = int(max(32, min(cfg.graph_block_size, (2 ** 24) // max(self.num_users, 1))))

        updated: List[torch.Tensor] = []
        ecr_out: List[torch.Tensor] = []
        added_out: List[torch.Tensor] = []
        kept_out: List[torch.Tensor] = []
        fill_num = 0.0
        fill_den = 0.0

        for chunk in batched_indices(n_rows, block_size, device):
            block_users = rows[chunk]
            b = int(block_users.numel())

            # ---------------- ① 当前候选的相似度 Δe = max(0, cos)
            delta = (norm_bank[block_users] @ norm_bank.T).clamp_min(0.0)      # (b, N)
            col_mask = act.unsqueeze(0).expand(b, -1).clone()
            col_mask[torch.arange(b, device=device), block_users] = False      # 排除自身
            if min_sim > 0:
                col_mask &= delta >= min_sim

            # ---------------- ② 取当前 top-pool 作为"新进入"的候选
            score = torch.where(
                col_mask, delta, torch.full_like(delta, -1e9)
            )                                                                   # 不可选 = -1e9
            cur_val, cur_idx = torch.topk(score, pool, dim=1)                   # (b, pool)

            # ---------------- ③ 与保留的候选池合并（去重，不用 scatter_）
            prev_ids = self.pool_ids[block_users]                               # (b, pool)
            prev_val = self.pool_val[block_users]                               # (b, pool)
            prev_mask = self.pool_mask[block_users]                             # (b, pool)

            dup = (cur_idx.unsqueeze(2) == prev_ids.unsqueeze(1)).any(dim=2)     # (b, pool)
            cat_ids = torch.cat([prev_ids, cur_idx], dim=1)                      # (b, 2*pool)
            cat_prev_val = torch.cat([prev_val * prev_mask.to(prev_val.dtype),
                                      torch.zeros_like(cur_val)], dim=1)
            cat_is_prev = torch.cat([prev_mask, torch.zeros_like(prev_mask)], dim=1)
            cat_is_new = torch.cat([torch.zeros_like(dup), ~dup], dim=1)
            entry_valid = cat_is_prev | cat_is_new

            # 当前 Δe：不可选候选按 0 处理（让"证据消失"表现为自然衰减，而不是发散的负数）
            gathered = score.gather(1, cat_ids.clamp_min(0))                     # (b, 2*pool)
            delta_eff = gathered.clamp_min(0.0)
            selectable = col_mask.gather(1, cat_ids.clamp_min(0))

            blended = rho * cat_prev_val + (1.0 - rho) * delta_eff               # 已有边
            if cfg.graph_new_edge == "warm":
                zero_init = delta_eff                                            # 新边直接用当前余弦
            else:
                zero_init = (1.0 - rho) * delta_eff                              # 新边：论文 0 初值
            neg_inf = torch.full_like(blended, -1e9)
            val = torch.where(cat_is_prev, blended, torch.where(cat_is_new, zero_init, neg_inf))
            v_ema = torch.where(entry_valid, val, neg_inf)

            # ---------------- ④ 保留策略：历史强或当前强都留下（这才是"不冻结"的关键）
            #   只按 EMA 值保留会把"刚出现的高相似边"每轮挤掉，等价于仍然冻结；
            #   因此 pool 模式按 max(EMA, Δe) 保留，容量 graph_pool_size。
            #   topk 模式刻意保留"只按 EMA 值 + 容量 K"的旧行为，用于对照复现。
            if self.retention_by_delta:
                retention = torch.where(
                    selectable & entry_valid, torch.maximum(v_ema, delta_eff), neg_inf
                )
            else:
                retention = torch.where(selectable & entry_valid, v_ema, neg_inf)
            pool_rank, pool_pos = torch.topk(retention, pool, dim=1)
            pool_ids_new = cat_ids.gather(1, pool_pos)
            pool_val_new = v_ema.gather(1, pool_pos)
            pool_valid = (pool_rank > -1e8) & (pool_ids_new >= 0)
            self.pool_ids[block_users] = torch.where(
                pool_valid, pool_ids_new, torch.full_like(pool_ids_new, -1)
            )
            self.pool_val[block_users] = torch.where(
                pool_valid, pool_val_new, torch.zeros_like(pool_val_new)
            )
            self.pool_mask[block_users] = pool_valid

            fill_num += float(pool_valid.sum().item())
            fill_den += float(pool_valid.numel())

            # ---------------- ⑤ 选出给 GNN 用的 Top-K：按论文，用 EMA 值排序
            masked_pool = torch.where(
                pool_valid, pool_val_new, torch.full_like(pool_val_new, -1e9)
            )
            top_val, top_pos = torch.topk(masked_pool, min(k, pool), dim=1)
            keep = top_val > -1e8
            # 【F2】无效槽位写 -1，而不是 active[top_pos] 的伪 id
            new_ids = torch.where(
                keep,
                pool_ids_new.gather(1, top_pos),
                torch.full_like(top_pos, -1),
            )
            safe_val = torch.where(keep, top_val, torch.full_like(top_val, -1e9))
            new_weights = F.softmax(safe_val, dim=1)
            new_weights = torch.where(keep, new_weights, torch.zeros_like(new_weights))
            new_mask = keep.any(dim=1)

            # ---------------- ⑥ 边变化率（Jaccard）与新进入边数
            old_ids = self.ids[block_users]
            old_mask = self.mask[block_users] & (old_ids >= 0).any(dim=1)
            old_slot = (old_ids >= 0) & old_mask.unsqueeze(1)
            eq = (
                (old_ids.unsqueeze(2) == new_ids.unsqueeze(1))
                & old_slot.unsqueeze(2)
                & keep.unsqueeze(1)
            )
            inter = eq.any(dim=2).sum(dim=1).to(torch.float32)
            n_old = old_slot.sum(dim=1).to(torch.float32)
            n_new = keep.sum(dim=1).to(torch.float32)
            union = torch.clamp(n_old + n_new - inter, min=1e-8)
            ecr = torch.where(n_old > 0, 1.0 - inter / union, torch.zeros_like(inter))
            n_added = torch.clamp(n_new - inter, min=0.0)

            self.ids[block_users] = new_ids
            self.weights[block_users] = new_weights.to(self.weights.dtype)
            self.mask[block_users] = new_mask

            updated.append(block_users.to("cpu"))
            ecr_out.append(ecr.to("cpu"))
            added_out.append(n_added.to("cpu"))
            kept_out.append(n_new.to("cpu"))

        if not updated:
            return GraphUpdateStats(empty_long, empty_float, empty_float, empty_float, 0.0)
        return GraphUpdateStats(
            updated=torch.cat(updated),
            ecr=torch.cat(ecr_out),
            n_added=torch.cat(added_out),
            n_kept=torch.cat(kept_out),
            pool_fill=(fill_num / fill_den) if fill_den > 0 else 0.0,
        )
