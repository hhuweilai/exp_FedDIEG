"""评测：逐用户 NDCG@10 / HR@10（对齐 A/B 版的"最后位置打分 + 候选集排序"）。

A/B 版的 `evaluate`（aligned L640-L700 / FedDIEG.py L214-L260）对每个用户：
  1. 组装该用户的模型状态（共享层 + 它自己的个性化层）；
  2. 取该用户图谱里的邻居摘要与边权做长短期融合；
  3. 用**最后一个位置的表示**与候选集（1 正 + 99 负）做内积；
  4. Top-10 里找到正样本即 HR=1，NDCG = log2 / log2(rank+2)。

这里保持完全一致的打分方式，只做了一处**等价加速**：把"个性化状态完全相同"的用户
合批（未训练过的客户端共享同一份初值，见 `FedTrainer.personal_state_key`）。
同一份状态下的批处理与逐个前向在数值上完全一致（无跨样本交互）。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import FedDIEGConfig
from .data import ClientWindow
from .model import build_model


@torch.no_grad()
def evaluate_split(
    trainer,
    eval_model: torch.nn.Module,
    window: ClientWindow,
    graph,
    split: str,
    cfg: FedDIEGConfig,
) -> Dict[str, np.ndarray]:
    """返回 {"users", "hr", "ndcg"}，顺序与 ``window.split(split)`` 的行顺序一致。"""
    users_all, seqs_all, items_all = window.split(split)
    total = int(users_all.numel())
    if total == 0:
        return {
            "users": np.zeros(0, dtype=np.int64),
            "hr": np.zeros(0, dtype=np.float64),
            "ndcg": np.zeros(0, dtype=np.float64),
        }

    limit = total
    if cfg.max_eval_users:
        limit = min(total, int(cfg.max_eval_users))
    if limit < total:
        # 固定种子抽子集（而不是取前 N 行）：同一 seed 下各 ρ/各 mode 抽到同一批用户，
        # 保证臂间是配对比较；同时避免"取前 N 行"带来的顺序偏差。
        gen = torch.Generator().manual_seed(int(cfg.seed) + 7)
        picked = torch.randperm(total, generator=gen)[:limit]
        users_view = users_all[picked]
        seqs_view = seqs_all[picked]
        items_view = items_all[picked]
    else:
        users_view = users_all
        seqs_view = seqs_all
        items_view = items_all
    hr = np.zeros(limit, dtype=np.float64)
    ndcg = np.zeros(limit, dtype=np.float64)

    # user id -> 在本 split 里的行号（同一用户在一行里只出现一次）
    row_of: Dict[int, int] = {int(u): i for i, u in enumerate(users_view.tolist())}

    eval_model.eval()
    device = trainer.device
    # rows / chunk 一律留在 CPU（用来索引 CPU 上的 split 张量），只把取出的数据搬到 device
    for _, group in trainer.group_users_by_state(users_view):
        rows = torch.tensor([row_of[int(u)] for u in group.tolist()], dtype=torch.long)
        trainer.load_into(eval_model, int(group[0].item()))
        for chunk in torch.split(rows, max(1, int(cfg.eval_batch_size))):
            uids = users_view[chunk].to(device)
            seqs = seqs_view[chunk].to(device)
            items = items_view[chunk].to(device)
            nbr_sum, nbr_w, nbr_valid = graph.gather(trainer.summary_bank, uids)
            if bool(nbr_valid.any().item()):
                scores = eval_model.score_user(seqs, items, nbr_sum, nbr_w, cfg.graph_mix)
            else:
                scores = eval_model.score_user(seqs, items)
            top_indices = torch.topk(scores, cfg.top_k, dim=1).indices
            hits = top_indices == 0
            found = hits.any(dim=1)
            ranks = hits.float().argmax(dim=1)
            ndcg_vals = torch.where(
                found,
                math.log(2.0) / torch.log(ranks.float() + 2.0),
                torch.zeros_like(ranks.float()),
            )
            idx = chunk.cpu().numpy()
            hr[idx] = found.float().cpu().numpy()
            ndcg[idx] = ndcg_vals.cpu().numpy()
    return {"users": users_view.cpu().numpy(), "hr": hr, "ndcg": ndcg}


def build_eval_model(trainer, cfg: FedDIEGConfig) -> torch.nn.Module:
    """评测用的模型容器；每次前向前 ``load_state_dict``，所以一份就够。"""
    return build_model(trainer.num_items, cfg).to(trainer.device)


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or values.size == 0 or mask.sum() == 0:
        return float("nan")
    return float(values[mask].mean())


def hr_ndcg_from_rank(rank: Optional[int], k: int) -> Tuple[float, float]:
    if rank is None or rank >= k:
        return 0.0, 0.0
    return 1.0, math.log(2.0) / math.log(rank + 2.0)
