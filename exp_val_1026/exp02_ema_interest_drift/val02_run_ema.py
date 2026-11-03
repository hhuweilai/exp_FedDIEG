#!/usr/bin/env python
"""验证实验 2（val02）第 2 步：兴趣真实变化时，固定 EMA 是否适应太慢？

对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验二」。

一次运行 = 一个 (数据集, EMA 系数 ρ, 随机种子)。
四个 ρ 共用**同一份** cohort（`val02_detect_drift.py` 产出）与同一份窗口数据，
因此 4 条曲线是同一批用户的配对比较。

这个脚本改了什么（和验证实验一 / 现状 FedDIEG 的差别）
----------------------------------------------------
1. 唯一的自变量是边权 EMA 系数 ρ（任务书的 ρ=0 / 0.5 / 0.8 / 0.95）。
2. **阈值作用在原始余弦相似度上**（`--graph-gate raw`，默认）。
   原实现先把边权归一化到和为 1、再和 ρ·历史混合、再与 `min_sim` 比较，
   于是 ρ≥0.9 时混合值恒低于阈值、图会整体坍塌 —— 那不是"适应慢"，是"图没了"，
   四个 ρ 之间不可比。这一点必须在报告里写清楚。
3. 记录四类东西：
   (a) 全体 / drift 组 / stable 组的逐轮 NDCG@10（→ 恢复曲线）；
   (b) drift 组邻居里"旧兴趣同伴"与"新兴趣同伴"的权重占比（→ 边是否还挂在旧兴趣上）；
   (c) drift 组邻居集合相对 onset 前的 turnover（→ 多少轮完成明显调整）；
   (d) 每轮边变化率 ECR（全体 / drift / stable）。

输出（`results/raw/<dataset>/rho<ρ>/seed_<seed>/`）
-------------------------------------------------
  config.json                    本次运行的全部参数
  val02_round_metrics.csv        逐评测轮：全体/drift/stable 的 NDCG、ECR、old/new peer 占比、turnover
  val02_user_round_metrics.csv   ★ 逐用户逐评测轮：NDCG + 边指标（恢复曲线的全部原始数据）
  val02_summary.csv              本次运行汇总（一行）
  val02_run.log                  训练日志

用法
----
  python exp_val_1026/exp02_ema_interest_drift/val02_run_ema.py --rho 0.8 --seed 42
  python .../val02_run_ema.py --rho 0.0 --seed 42 --num-windows 5 --rounds-per-window 120
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
VAL02_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_01_fast_gpu.run_exp01_fast_gpu import (  # noqa: E402
    FastData,
    FedDIEGModel,
    Tee,
    batched_indices,
    bce_seq_loss,
    masked_sequence_mean,
    sample_negative_like,
    set_seed,
)
from exp_02_fast_gpu.run_exp02_fast_gpu import Exp2Config, load_rolling_windows, mean  # noqa: E402

DEFAULT_COHORT_ROOT = VAL02_DIR / "results" / "drift"
DEFAULT_OUT_ROOT = VAL02_DIR / "results" / "raw"


# ======================================================================================
# 配置
# ======================================================================================
@dataclass
class Val02Config(Exp2Config):
    """在 Exp2 的时间窗口协议之上，只加"EMA 系数"这一个自变量。"""

    rho: float = 0.80
    num_windows: int = 5
    rounds_per_window: int = 120
    candidate_count: int = 99
    window_mode: str = "slice"
    lambda_temp: float = 0.01
    lambda_struct: float = 0.20
    graph_gate: str = "raw"            # raw=阈值看原始余弦（推荐）；blended=阈值看混合后的值（原实现）
    new_edge_init: str = "ema"         # ema=忠实复刻原实现 | warm=新边不被惩罚 | zero=从 0 涨（极端消融）
    graph_block_size: int = 2048
    graph_strength: str = "paper"      # paper=已发表配置；strong=放大图影响（用于敏感性对照）
    cohort_file: Optional[str] = None
    counts_file: Optional[str] = None
    primary_drift_only: bool = False   # 只保留 cat_old != cat_new 的 drift 用户


# 两组"图影响强度"预设。
# paper  : 与验证实验一 / 论文里的 FedDIEG 配置一致（graph_mix=0.2，并保留 confidence/agreement 门控
#          和 tail_power=4 的尾部加权），图对打分的影响本来就被压得很小；
#          graph_mix=0.05 是论文早期默认值，验证实验一已经发现它会把图效应淹没。
# strong : 打开图消息 —— 提高 mix、去掉两个门控、tail_power 降到 1、不再回拉局部特征。
#          用途：如果 paper 配置下 4 个 ρ 的 NDCG 曲线重合，用它判断"是不是图权重太小才看不出差异"。
GRAPH_PRESETS: Dict[str, Dict[str, float]] = {
    "paper": {
        "feddieg_graph_mix": 0.20,
        "feddieg_tail_power": 4.0,
        "feddieg_confidence_power": 0.5,
        "feddieg_agreement_power": 0.5,
        "feddieg_score_local_mix": 0.2,
    },
    "strong": {
        "feddieg_graph_mix": 0.50,
        "feddieg_tail_power": 1.0,
        "feddieg_confidence_power": 0.0,
        "feddieg_agreement_power": 0.0,
        "feddieg_score_local_mix": 0.0,
    },
    # mix：只提高图权重，其余（tail_power、两个门控、局部回拉）全部保持已发表配置。
    # 用途：strong 组一次改了 5 个超参，会把"图权重"和"去掉门控"混在一起；
    # 用 mix 组可以判断 strong 组的差异到底来自"图注入更强"还是"可靠性门控被拿掉"。
    "mix": {
        "feddieg_graph_mix": 0.50,
        "feddieg_tail_power": 4.0,
        "feddieg_confidence_power": 0.5,
        "feddieg_agreement_power": 0.5,
        "feddieg_score_local_mix": 0.2,
    },
}


ROUND_FIELDS = [
    "dataset",
    "rho",
    "seed",
    "window",
    "window_round",
    "global_round",
    "train_loss",
    "all_ndcg",
    "all_hr",
    "n_eval",
    "drift_ndcg",
    "drift_hr",
    "n_drift_eval",
    "stable_ndcg",
    "stable_hr",
    "n_stable_eval",
    "ecr_all",
    "ecr_drift",
    "ecr_stable",
    "drift_old_share",
    "drift_new_share",
    "drift_turnover",
    "drift_cur_share",
    "drift_cur_base",
    "drift_n_neighbors",
]

USER_FIELDS = [
    "dataset",
    "rho",
    "seed",
    "user",
    "cohort",
    "onset_window",
    "primary_drift",
    "cat_old",
    "cat_new",
    "window",
    "window_round",
    "global_round",
    "rel_round",
    "ndcg",
    "hr",
    "n_neighbors",
    "old_peer_share",
    "new_peer_share",
    "turnover_from_pre",
    "cur_peer_share",
    "cur_peer_base",
]

SUMMARY_FIELDS = [
    "dataset",
    "rho",
    "seed",
    "graph_gate",
    "new_edge_init",
    "graph_strength",
    "graph_mix",
    "edge_ema",
    "num_windows",
    "rounds_per_window",
    "total_rounds",
    "clients_per_round",
    "n_drift",
    "n_stable",
    "all_avg_ndcg",
    "all_final_ndcg",
    "avg_ecr",
    "elapsed_seconds",
]


# ======================================================================================
# 小工具
# ======================================================================================
def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


def make_out_dir(out_root: Path, dataset: str, rho: float, seed: int) -> Path:
    return out_root / safe_slug(dataset) / f"rho{rho:.2f}" / f"seed_{seed}"


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(values[mask].mean())


# ======================================================================================
# cohort：drift / stable 用户 + 各边界的"同伴主类别"表
# ======================================================================================
@dataclass
class Cohort:
    cohort: np.ndarray        # (U,) 0=非 cohort, 1=drift, 2=stable
    onset: np.ndarray         # (U,) onset 窗口（1 基）；非 cohort 为 0
    primary: np.ndarray       # (U,) drift 用户的主类别是否真的换了
    cat_old: np.ndarray       # (U,) 变化前主类别（-1 表示未知）
    cat_new: np.ndarray       # (U,) 变化后主类别
    users: np.ndarray         # (N,) cohort 用户 id（升序）
    pre_dom: Dict[int, np.ndarray]   # boundary -> (U,) 该边界前的主类别
    post_dom: Dict[int, np.ndarray]  # boundary -> (U,) 该边界后的主类别
    win_dom: Dict[int, np.ndarray]   # window(1 基) -> (U,) 该窗口内的主类别（"当前兴趣"）
    win_freq: Dict[int, np.ndarray]  # window(1 基) -> (K,) 该窗口各类别在用户池中的频率（随机基线）
    num_categories: int
    peer_boundaries: List[int]

    @property
    def drift_users(self) -> np.ndarray:
        return self.users[self.cohort[self.users] == 1]

    @property
    def stable_users(self) -> np.ndarray:
        return self.users[self.cohort[self.users] == 2]


def _dominant(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=1)
    return np.where(total > 0, counts.argmax(axis=1), -1)


def load_cohort(cohort_file: Path, counts_file: Path, primary_drift_only: bool) -> Cohort:
    if not cohort_file.exists():
        raise SystemExit(
            f"找不到 cohort 文件：{cohort_file}\n请先运行 val02_detect_drift.py 生成 cohort。"
        )
    if not counts_file.exists():
        raise SystemExit(
            f"找不到类别计数文件：{counts_file}\n请先运行 val02_detect_drift.py。"
        )
    frame = pd.read_csv(cohort_file)
    data = np.load(counts_file)
    counts = data["counts"]  # (U, W, K)
    num_users, num_windows, num_categories = counts.shape

    cohort = np.zeros(num_users, dtype=np.int64)
    onset = np.zeros(num_users, dtype=np.int64)
    primary = np.zeros(num_users, dtype=np.int64)
    cat_old = np.full(num_users, -1, dtype=np.int64)
    cat_new = np.full(num_users, -1, dtype=np.int64)

    peer_boundaries = sorted(int(b) for b in frame["onset_window"].unique().tolist())
    pre_dom: Dict[int, np.ndarray] = {}
    post_dom: Dict[int, np.ndarray] = {}
    for boundary in peer_boundaries:
        boundary = int(np.clip(boundary, 2, num_windows))
        pre_dom[boundary] = _dominant(counts[:, : boundary - 1, :].sum(axis=1))
        post_dom[boundary] = _dominant(counts[:, boundary - 1 :, :].sum(axis=1))

    # 逐窗口的"当前兴趣"及其随机基线：用于"当前窗口兴趣一致度"指标（避免 onset 晚的用户快照错位）
    win_dom: Dict[int, np.ndarray] = {}
    win_freq: Dict[int, np.ndarray] = {}
    for window_idx in range(1, num_windows + 1):
        dom = _dominant(counts[:, window_idx - 1, :])
        win_dom[window_idx] = dom
        valid = dom >= 0
        win_freq[window_idx] = (
            np.bincount(dom[valid], minlength=num_categories) / max(int(valid.sum()), 1)
        )

    for _, row in frame.iterrows():
        user = int(row["user"])
        if user < 0 or user >= num_users:
            continue
        is_drift = str(row["cohort"]) == "drift"
        if primary_drift_only and is_drift and int(row.get("primary_drift", 0)) != 1:
            continue
        cohort[user] = 1 if is_drift else 2
        onset[user] = int(np.clip(int(row["onset_window"]), 2, num_windows))
        primary[user] = int(row.get("primary_drift", 0))
        cat_old[user] = int(row.get("cat_old", -1))
        cat_new[user] = int(row.get("cat_new", -1))

    users = np.flatnonzero(cohort > 0)
    print(
        f"[cohort] drift={int((cohort == 1).sum())} stable={int((cohort == 2).sum())} "
        f"(共 {users.size} 人, 类别数 {num_categories}, 边界 {peer_boundaries})"
    )
    if (cohort == 1).sum() == 0:
        raise SystemExit("drift cohort 为 0：请检查 cohort 文件或放宽 val02_detect_drift.py 的阈值")
    return Cohort(
        cohort=cohort,
        onset=onset,
        primary=primary,
        cat_old=cat_old,
        cat_new=cat_new,
        users=users,
        pre_dom=pre_dom,
        post_dom=post_dom,
        win_dom=win_dom,
        win_freq=win_freq,
        num_categories=num_categories,
        peer_boundaries=peer_boundaries,
    )


# ======================================================================================
# 图：训练时取邻居 / 服务器端更新边权
# ======================================================================================
def graph_batch(
    users: torch.Tensor,
    summary_bank: torch.Tensor,
    graph_neighbors: torch.Tensor,
    graph_weights: torch.Tensor,
    graph_mask: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if graph_neighbors.numel() == 0:
        return None, None, None
    neighbors = graph_neighbors[users]
    weights = graph_weights[users]
    mask = graph_mask[users]
    if not mask.any():
        return None, None, None
    return summary_bank[neighbors.clamp_min(0)], weights * mask.float().unsqueeze(1), mask


@torch.no_grad()
def update_graph(
    update_users: torch.Tensor,
    summary_bank: torch.Tensor,
    active_clients: torch.Tensor,
    graph_neighbors: torch.Tensor,
    graph_weights: torch.Tensor,
    graph_mask: torch.Tensor,
    smooth_state: torch.Tensor,
    topk: int,
    min_sim: float,
    rho: float,
    block_size: int,
    gate_on: str = "raw",
    new_edge_init: str = "warm",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 update_users 的邻居做 Top-K + EMA 平滑。

    e_{ij}^{(t+1)} = ρ · e_{ij}^{(t)} + (1-ρ) · ê_{ij}^{(t)}，其中 ê 是当前摘要的余弦相似度。

    关键实现选择（写报告要交代）：
    * `min_sim` 默认作用在**原始余弦**上（gate_on="raw"）：ρ 只改变"历史边权的记忆长度"，
      不改变"哪些边有证据存在"。否则归一化后的边权（和恒为 1，典型值 ~0.1）
      与阈值混合后必然低于阈值，ρ→1 时图会整体坍塌，四个 ρ 不可比。
    * 新边的初值（`new_edge_init`）：
      - `ema`（默认，忠实复刻原实现）：新边相当于从 (1-ρ)·sim 起步，
        因此 ρ=0.95 时新邻居要涨 ~20 轮才够格进入 Top-K —— 这就是任务书说的"适应太慢"的机制；
      - `warm`：新边直接以当前余弦起步，会把这个滞后机制消掉（只保留旧边衰减变慢）；
      - `zero`：新边从 0 涨，最极端的 EMA 语义。
      首次建立邻居的用户（还没有任何历史状态）一律以当前余弦起步，否则 Top-K 全是 0、选择退化为下标序。
    * EMA 作用在**平滑后的相似度**上（smooth_state），而不是归一化后的权重；
      权重最后再在保留的 Top-K 内归一化，供 GNN 使用。

    返回 (本轮被更新的用户 id, 对应的边变化率 ECR)，都在 CPU 上。
    """
    empty = torch.zeros(0, dtype=torch.long, device="cpu")
    active = torch.nonzero(active_clients, as_tuple=False).flatten()
    users = update_users[active_clients[update_users]]
    k = min(int(topk), int(active.numel()) - 1)
    if users.numel() == 0 or k <= 0:
        return empty, torch.zeros(0, dtype=torch.float32)

    device = summary_bank.device
    rho = float(min(max(rho, 0.0), 0.999))
    norm_bank = F.normalize(summary_bank, p=2, dim=1, eps=1e-8)
    norm_active = norm_bank[active]
    num_active = int(active.numel())

    # 邻居 id → 活跃列表中的列号（-1 表示不是活跃用户 / padding）
    active_slot = torch.full((summary_bank.size(0),), -1, dtype=torch.long, device=device)
    active_slot[active] = torch.arange(num_active, device=device)

    ecr_chunks: List[torch.Tensor] = []
    user_chunks: List[torch.Tensor] = []

    for block in batched_indices(torch.arange(users.numel(), device=device), block_size):
        block_users = users[block]
        rows = block_users.numel()
        sim = torch.matmul(norm_bank[block_users], norm_active.T).clamp_min(0.0)
        sim = sim.masked_fill(block_users.unsqueeze(1) == active.unsqueeze(0), 0.0)

        old_ids = graph_neighbors[block_users]                      # (B, k)
        old_weights = smooth_state[block_users]                     # (B, k) 上次的平滑相似度
        old_col = active_slot[old_ids.clamp_min(0)]                 # (B, k)
        has_prev = (old_ids >= 0) & (old_col >= 0) & graph_mask[block_users].unsqueeze(1)

        if rho > 0.0:
            prev = torch.zeros((rows, num_active), dtype=sim.dtype, device=device)
            prev = prev.scatter_(
                1, old_col.clamp_min(0), old_weights.to(sim.dtype) * has_prev.to(sim.dtype)
            )
            prev_mask = torch.zeros((rows, num_active), dtype=sim.dtype, device=device)
            prev_mask = prev_mask.scatter_(1, old_col.clamp_min(0), has_prev.to(sim.dtype))
            blended = (1.0 - rho) * sim + rho * prev
            if new_edge_init == "warm":
                init_value = sim
            elif new_edge_init == "zero":
                init_value = torch.zeros_like(sim)
            else:  # "ema"：忠实复刻原实现 —— 新边等价于从 (1-ρ)·sim 起步
                init_value = (1.0 - rho) * sim
            cand = torch.where(prev_mask > 0, blended, init_value)
            # 首次建立邻居的用户没有历史状态，直接以当前余弦起步（否则 Top-K 全是 0、选择退化为下标序）
            row_has_prev = (prev_mask > 0).any(dim=1, keepdim=True)
            cand = torch.where(row_has_prev, cand, sim)
        else:
            cand = sim

        # 边的"存在性"判定
        if gate_on == "raw":
            exists = sim >= min_sim
        else:
            exists = cand >= min_sim
        cand_gated = torch.where(exists, cand, torch.full_like(cand, -1e9))

        top_values, top_positions = torch.topk(cand_gated, k, dim=1)
        new_ids = active[top_positions]                              # (B, k)
        keep = top_values > -1e8
        kept_values = torch.where(keep, top_values, torch.zeros_like(top_values))
        denom = kept_values.sum(dim=1, keepdim=True).clamp_min(1e-8)
        new_weights = kept_values / denom
        new_mask = keep.any(dim=1)

        # ---- 边变化率 ECR（集合对称差 / 并集）----
        old_sets = (old_ids >= 0) & graph_mask[block_users].unsqueeze(1)
        eq = (old_ids.unsqueeze(2) == new_ids.unsqueeze(1)) & old_sets.unsqueeze(2) & keep.unsqueeze(1)
        inter = eq.any(dim=2).sum(dim=1).to(torch.float32)
        n_old = old_sets.sum(dim=1).to(torch.float32)
        n_new = keep.sum(dim=1).to(torch.float32)
        union = torch.clamp(n_old + n_new - inter, min=1e-8)
        ecr = torch.where(n_old > 0, 1.0 - inter / union, torch.zeros_like(inter))

        graph_neighbors[block_users] = new_ids
        graph_weights[block_users] = new_weights.to(graph_weights.dtype)
        graph_mask[block_users] = new_mask
        smooth_state[block_users] = kept_values.to(smooth_state.dtype)

        user_chunks.append(block_users.to("cpu"))
        ecr_chunks.append(ecr.to("cpu"))

    return torch.cat(user_chunks), torch.cat(ecr_chunks)


# ======================================================================================
# 评测
# ======================================================================================
@torch.no_grad()
def evaluate_per_user(
    model: FedDIEGModel,
    data: FastData,
    cfg: Val02Config,
    summary_bank: torch.Tensor,
    graph_neighbors: torch.Tensor,
    graph_weights: torch.Tensor,
    graph_mask: torch.Tensor,
    split: str = "test",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    if split == "test":
        users_all, seqs_all, items_all = data.test_users, data.test_seq, data.test_items
    else:
        users_all, seqs_all, items_all = data.val_users, data.val_seq, data.val_items

    total = users_all.numel()
    if cfg.max_eval_users:
        total = min(total, cfg.max_eval_users)
    hr = np.zeros(total, dtype=np.float64)
    ndcg = np.zeros(total, dtype=np.float64)
    if total == 0:
        return np.zeros(0, dtype=np.int64), hr, ndcg

    for batch in batched_indices(torch.arange(total, device=users_all.device), cfg.eval_batch_size):
        b_users = users_all[batch]
        b_items = items_all[batch]
        neighbors, weights, gmask = graph_batch(
            b_users, summary_bank, graph_neighbors, graph_weights, graph_mask
        )
        if neighbors is not None:
            base_feats, _ = model(seqs_all[batch])
            feats, short_summary = model(seqs_all[batch], neighbors, weights, gmask)
            score_feats = model.score_representation(feats[:, -1, :], short_summary, base_feats[:, -1, :])
        else:
            feats, short_summary = model(seqs_all[batch])
            score_feats = model.score_representation(feats[:, -1, :], short_summary)
        scores = torch.matmul(score_feats.unsqueeze(1), model.item_emb(b_items).transpose(-1, -2)).squeeze(1)
        _, top_indices = torch.topk(scores, cfg.top_k, dim=1)
        hits = top_indices == 0
        found = hits.any(dim=1)
        ranks = hits.float().argmax(dim=1)
        idx = batch.cpu().numpy()
        hr[idx] = found.float().cpu().numpy()
        ndcg[idx] = torch.where(
            found, math.log(2.0) / torch.log(ranks.float() + 2.0), torch.zeros_like(ranks.float())
        ).cpu().numpy()
    return users_all[:total].cpu().numpy(), hr, ndcg


# ======================================================================================
# cohort 的边级指标：old/new 同伴权重占比 + 相对 onset 前的 turnover
# ======================================================================================
class CohortEdgeProbe:
    """按 onset 分组，算 drift/stable 用户的邻居"旧兴趣/新兴趣同伴"权重占比与 turnover。

    old_peer_share(u, t)：u 的邻居里，"在自己 onset 边界之前主类别 == u 的旧主类别"的用户
                          所占的边权比例。ρ 越大，这个比例掉得越慢 → 边还挂在旧兴趣上。
    new_peer_share(u, t)：邻居里"在 onset 边界之后主类别 == u 的新主类别"的用户所占比例。
    turnover_from_pre(u, t)：当前邻居集合与 onset 前邻居集合的 Jaccard 距离（1 - Jaccard），
                          衡量"多少轮完成了明显的调整"。
    cur_peer_share(u, t)：邻居里"**当前窗口**主类别 == u 的当前窗口主类别"的边权占比；
    cur_peer_base(u, t) ：其随机基线（该类别在当前窗口用户池中的频率）。
                          这条指标用户与同伴用同一个窗口快照，不存在 onset 晚导致的错位，
                          是"图是否跟踪当前兴趣"的最干净度量（lift = share / base）。
    """

    def __init__(self, cohort: Cohort, device: torch.device):
        self.cohort = cohort
        self.device = device
        self.users = torch.as_tensor(cohort.users, dtype=torch.long, device=device)
        self.user_index = {int(u): idx for idx, u in enumerate(cohort.users.tolist())}
        self.cat_old = torch.as_tensor(cohort.cat_old, dtype=torch.long, device=device)
        self.cat_new = torch.as_tensor(cohort.cat_new, dtype=torch.long, device=device)
        self.pre_dom = {
            int(b): torch.as_tensor(d, dtype=torch.long, device=device) for b, d in cohort.pre_dom.items()
        }
        self.post_dom = {
            int(b): torch.as_tensor(d, dtype=torch.long, device=device) for b, d in cohort.post_dom.items()
        }
        self.win_dom = {
            int(w): torch.as_tensor(d, dtype=torch.long, device=device) for w, d in cohort.win_dom.items()
        }
        self.win_freq = {
            int(w): torch.as_tensor(f, dtype=torch.float32, device=device)
            for w, f in cohort.win_freq.items()
        }
        self.ref_neighbors: Optional[torch.Tensor] = None

    def init_ref(self, num_users: int, topk: int) -> None:
        self.ref_neighbors = torch.full(
            (num_users, topk), -1, dtype=torch.long, device=self.device
        )

    @torch.no_grad()
    def snapshot(self, graph_neighbors: torch.Tensor, window: int) -> None:
        """在窗口 window 开始时，给 onset == window 的 cohort 用户存一份"变化前"的邻居快照。"""
        if self.ref_neighbors is None:
            return
        sel = np.flatnonzero((self.cohort.cohort > 0) & (self.cohort.onset == window))
        if sel.size == 0:
            return
        users = torch.as_tensor(sel, dtype=torch.long, device=self.device)
        self.ref_neighbors[users] = graph_neighbors[users]

    @torch.no_grad()
    def probe(
        self,
        users: np.ndarray,
        graph_neighbors: torch.Tensor,
        graph_weights: torch.Tensor,
        graph_mask: torch.Tensor,
        window: int,
        need_turnover: bool,
    ) -> Dict[str, np.ndarray]:
        """返回这些用户（必须是 cohort 成员）的边级指标，数组顺序与 users 一致。"""
        result = {
            "n_neighbors": np.zeros(users.size, dtype=np.float64),
            "old_peer_share": np.full(users.size, np.nan),
            "new_peer_share": np.full(users.size, np.nan),
            "turnover_from_pre": np.full(users.size, np.nan),
            "cur_peer_share": np.full(users.size, np.nan),
            "cur_peer_base": np.full(users.size, np.nan),
        }
        if users.size == 0:
            return result

        member = np.isin(users, self.cohort.users)
        if not member.any():
            return result

        user_ids = torch.as_tensor(users[member], dtype=torch.long, device=self.device)
        nbr = graph_neighbors[user_ids]                                  # (G, k)
        w = graph_weights[user_ids] * graph_mask[user_ids].float().unsqueeze(1)
        denom = w.sum(dim=1).clamp_min(1e-8)
        n_valid = (nbr >= 0).float().sum(dim=1)

        # ---- 当前窗口兴趣一致度：邻居与用户"本窗口主类别"相同的边权占比 + 随机基线 ----
        # 优点：用户与同伴用的是同一个窗口的兴趣快照，不存在 onset 晚导致的错位。
        window_idx = int(np.clip(window, 1, len(self.win_dom)))
        if window_idx in self.win_dom:
            dom = self.win_dom[window_idx].unsqueeze(0)                  # (1, U)
            user_dom = self.win_dom[window_idx][user_ids]                # (G,)
            match_cur = (
                (dom == user_dom.unsqueeze(1)) & (user_dom.unsqueeze(1) >= 0) & (dom >= 0)
            ).float()
            gathered_cur = match_cur.gather(1, nbr.clamp_min(0))
            cur_share = (gathered_cur * w).sum(dim=1) / denom
            freq = self.win_freq[window_idx]
            cur_base = freq[user_dom.clamp_min(0)]
            result["cur_peer_share"][member] = cur_share.cpu().numpy()
            result["cur_peer_base"][member] = cur_base.cpu().numpy()

        onsets = self.cohort.onset[users[member]]
        old_share = torch.full((user_ids.numel(),), float("nan"), device=self.device)
        new_share = torch.full((user_ids.numel(),), float("nan"), device=self.device)
        for boundary in np.unique(onsets):
            boundary = int(boundary)
            if boundary not in self.pre_dom:
                continue
            sel = torch.as_tensor(np.flatnonzero(onsets == boundary), dtype=torch.long, device=self.device)
            sel_users = user_ids[sel]
            old_cat = self.cat_old[sel_users].unsqueeze(1)               # (g, 1)
            new_cat = self.cat_new[sel_users].unsqueeze(1)
            pre_dom = self.pre_dom[boundary].unsqueeze(0)                # (1, U)
            post_dom = self.post_dom[boundary].unsqueeze(0)
            # 同伴 v 是否"u 的旧兴趣同类"：只用离线类别统计，不参与训练
            match_old = ((pre_dom == old_cat) & (old_cat >= 0) & (pre_dom >= 0)).float()
            match_new = ((post_dom == new_cat) & (new_cat >= 0) & (post_dom >= 0)).float()
            gathered_old = match_old.gather(1, nbr[sel].clamp_min(0))
            gathered_new = match_new.gather(1, nbr[sel].clamp_min(0))
            old_share[sel] = (gathered_old * w[sel]).sum(dim=1) / denom[sel]
            new_share[sel] = (gathered_new * w[sel]).sum(dim=1) / denom[sel]

        result["n_neighbors"][member] = n_valid.cpu().numpy()
        result["old_peer_share"][member] = old_share.cpu().numpy()
        result["new_peer_share"][member] = new_share.cpu().numpy()

        if need_turnover and self.ref_neighbors is not None:
            ref = self.ref_neighbors[user_ids]                           # (G, k)
            eq = (ref.unsqueeze(2) == nbr.unsqueeze(1)) & (ref.unsqueeze(2) >= 0) & (nbr.unsqueeze(1) >= 0)
            inter = eq.any(dim=2).sum(dim=1).float()
            n_ref = (ref >= 0).sum(dim=1).float()
            n_cur = (nbr >= 0).sum(dim=1).float()
            union = (n_ref + n_cur - inter).clamp_min(1e-8)
            jaccard = inter / union
            turnover = torch.where(n_ref > 0, 1.0 - jaccard, torch.full_like(jaccard, float("nan")))
            result["turnover_from_pre"][member] = turnover.cpu().numpy()
        return result


# ======================================================================================
# 参与序列（同 val01：所有实验共用同一份）
# ======================================================================================
def build_participation_schedule(
    num_users: int, total_rounds: int, clients_per_round: int, seed: int, device: torch.device
) -> List[torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 7919)
    clients_per_round = max(1, min(clients_per_round, num_users))
    schedule: List[torch.Tensor] = []
    for _ in range(total_rounds):
        order = torch.randperm(num_users, generator=generator, device=device)
        schedule.append(order if clients_per_round >= num_users else order[:clients_per_round])
    return schedule


# ======================================================================================
# 单次运行
# ======================================================================================
def run_single(
    cfg: Val02Config, windows: List[FastData], cohort: Cohort, device: torch.device, out_dir: Path
) -> None:
    set_seed(cfg.seed)
    model = FedDIEGModel(windows[0].num_items, cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
    num_users = windows[0].num_users
    clients_per_round = min(
        cfg.clients_per_round or max(int(cfg.participation_rate * num_users), 1), num_users
    )
    total_rounds = cfg.num_windows * cfg.rounds_per_window
    schedule = build_participation_schedule(num_users, total_rounds, clients_per_round, cfg.seed, device)

    topk = max(int(cfg.feddieg_neighbors), 1)
    summary_bank = torch.zeros((num_users, cfg.latent_dim), device=device)
    active_clients = torch.zeros(num_users, dtype=torch.bool, device=device)
    graph_neighbors = torch.full((num_users, topk), -1, dtype=torch.long, device=device)
    graph_weights = torch.zeros((num_users, topk), dtype=torch.float32, device=device)
    graph_mask = torch.zeros(num_users, dtype=torch.bool, device=device)
    smooth_state = torch.zeros((num_users, topk), dtype=torch.float32, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    probe = CohortEdgeProbe(cohort, device)
    probe.init_ref(num_users, topk)

    # cohort 的快速查表（逐用户行过滤用）
    is_drift = (cohort.cohort == 1)
    is_stable = (cohort.cohort == 2)
    cohort_users = cohort.users
    onset_of = cohort.onset

    round_rows: List[Dict[str, object]] = []
    user_rows: List[Dict[str, object]] = []
    window_ndcgs: List[float] = []
    window_ecrs: List[float] = []
    start_time = time.time()
    global_round = 0

    for window_idx, data in enumerate(windows, start=1):
        print(
            f"\n[rho={cfg.rho:.2f}] Window {window_idx}/{cfg.num_windows} | "
            f"eval_users={data.test_users.numel()}"
        )
        # 变化前邻居快照（onset == 本窗口的 cohort 用户）
        probe.snapshot(graph_neighbors, window_idx)
        last_ndcg = 0.0
        last_hr = 0.0

        for window_round in range(1, cfg.rounds_per_window + 1):
            global_round += 1
            round_start = time.time()
            sampled = schedule[global_round - 1]

            losses: List[float] = []
            model.train()
            for _ in range(cfg.local_epochs):
                shuffled = sampled[torch.randperm(sampled.numel(), generator=generator, device=device)]
                for users in batched_indices(shuffled, cfg.microbatch_size):
                    seq, pos, mask = data.train_seq[users], data.train_pos[users], data.train_mask[users]
                    neg = sample_negative_like(pos, data.num_items)
                    optimizer.zero_grad(set_to_none=True)

                    neighbors, weights, gmask = graph_batch(
                        users, summary_bank, graph_neighbors, graph_weights, graph_mask
                    )
                    feats, short_summary = model(seq, neighbors, weights, gmask)
                    rec_loss = bce_seq_loss(
                        (feats * model.item_emb(pos)).sum(dim=-1),
                        (feats * model.item_emb(neg)).sum(dim=-1),
                        mask,
                    )

                    has_prev = active_clients[users]
                    if cfg.lambda_temp > 0 and has_prev.any():
                        temp_loss = (
                            1.0
                            - F.cosine_similarity(
                                short_summary[has_prev],
                                summary_bank[users][has_prev].detach(),
                                dim=-1,
                                eps=1e-8,
                            )
                        ).mean()
                    else:
                        temp_loss = torch.zeros((), device=device)

                    if cfg.lambda_struct > 0 and neighbors is not None and gmask is not None and gmask.any():
                        target = weights[gmask].detach()
                        logits = F.cosine_similarity(
                            short_summary[gmask].unsqueeze(1), neighbors[gmask], dim=-1, eps=1e-8
                        ).clamp_min(0.0)
                        logits = logits.masked_fill(target <= 0, -1e9)
                        pred = F.softmax(logits / max(cfg.feddieg_neighbor_temp, 1e-6), dim=1)
                        pred = torch.where(target > 0, pred, torch.zeros_like(pred))
                        pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-8)
                        struct_loss = F.mse_loss(pred, target)
                    else:
                        struct_loss = torch.zeros((), device=device)

                    loss = rec_loss + cfg.lambda_temp * temp_loss + cfg.lambda_struct * struct_loss
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))

                    # 客户端上传兴趣摘要（只有参与的客户端刷新缓存）
                    new_memory = short_summary.detach()
                    momentum = min(max(cfg.feddieg_memory_momentum, 0.0), 0.99)
                    with torch.no_grad():
                        seen = active_clients[users].float().unsqueeze(-1)
                        summary_bank[users] = seen * (
                            momentum * summary_bank[users] + (1.0 - momentum) * new_memory
                        ) + (1.0 - seen) * new_memory
                        active_clients[users] = True

            updated_users, ecr_values = update_graph(
                sampled,
                summary_bank,
                active_clients,
                graph_neighbors,
                graph_weights,
                graph_mask,
                smooth_state,
                cfg.feddieg_neighbors,
                cfg.feddieg_min_sim,
                cfg.rho,
                cfg.graph_block_size,
                gate_on=cfg.graph_gate,
                new_edge_init=cfg.new_edge_init,
            )

            should_eval = (window_round % cfg.eval_every == 0) or (window_round == cfg.rounds_per_window)
            if not should_eval:
                continue

            test_users, test_hr, test_ndcg = evaluate_per_user(
                model, data, cfg, summary_bank, graph_neighbors, graph_weights, graph_mask, "test"
            )
            user_ids = test_users
            drift_sel = is_drift[user_ids] if user_ids.size else np.zeros(0, dtype=bool)
            stable_sel = is_stable[user_ids] if user_ids.size else np.zeros(0, dtype=bool)
            all_ndcg = float(test_ndcg.mean()) if test_ndcg.size else 0.0
            all_hr = float(test_hr.mean()) if test_hr.size else 0.0
            drift_ndcg = _masked_mean(test_ndcg, drift_sel)
            drift_hr = _masked_mean(test_hr, drift_sel)
            stable_ndcg = _masked_mean(test_ndcg, stable_sel)
            stable_hr = _masked_mean(test_hr, stable_sel)
            last_ndcg, last_hr = all_ndcg, all_hr

            # ---- 边变化率分组 ----
            upd = updated_users.numpy()
            ecr = ecr_values.numpy()
            ecr_all = float(ecr.mean()) if ecr.size else 0.0
            ecr_drift = _masked_mean(ecr, is_drift[upd]) if upd.size else float("nan")
            ecr_stable = _masked_mean(ecr, is_stable[upd]) if upd.size else float("nan")

            # ---- cohort 用户的边级指标 + 逐用户 NDCG ----
            present = user_ids[np.isin(user_ids, cohort_users)] if user_ids.size else np.zeros(0, dtype=np.int64)
            edge_stats = probe.probe(
                present,
                graph_neighbors,
                graph_weights,
                graph_mask,
                window_idx,
                need_turnover=True,
            )
            ndcg_map = dict(zip(user_ids.tolist(), test_ndcg.tolist()))
            hr_map = dict(zip(user_ids.tolist(), test_hr.tolist()))
            drift_edges_old: List[float] = []
            drift_edges_new: List[float] = []
            drift_edges_turn: List[float] = []
            drift_cur_share: List[float] = []
            drift_cur_base: List[float] = []
            for pos, user in enumerate(present.tolist()):
                cohort_id = int(cohort.cohort[user])
                onset = int(onset_of[user])
                user_rows.append(
                    {
                        "dataset": cfg.dataset,
                        "rho": cfg.rho,
                        "seed": cfg.seed,
                        "user": int(user),
                        "cohort": "drift" if cohort_id == 1 else "stable",
                        "onset_window": onset,
                        "primary_drift": int(cohort.primary[user]),
                        "cat_old": int(cohort.cat_old[user]),
                        "cat_new": int(cohort.cat_new[user]),
                        "window": window_idx,
                        "window_round": window_round,
                        "global_round": global_round,
                        "rel_round": global_round - (onset - 1) * cfg.rounds_per_window,
                        "ndcg": float(ndcg_map.get(int(user), np.nan)),
                        "hr": float(hr_map.get(int(user), np.nan)),
                        "n_neighbors": float(edge_stats["n_neighbors"][pos]),
                        "old_peer_share": float(edge_stats["old_peer_share"][pos]),
                        "new_peer_share": float(edge_stats["new_peer_share"][pos]),
                        "turnover_from_pre": float(edge_stats["turnover_from_pre"][pos]),
                        "cur_peer_share": float(edge_stats["cur_peer_share"][pos]),
                        "cur_peer_base": float(edge_stats["cur_peer_base"][pos]),
                    }
                )
                if cohort_id == 1 and window_idx >= onset:
                    old_v = edge_stats["old_peer_share"][pos]
                    new_v = edge_stats["new_peer_share"][pos]
                    turn_v = edge_stats["turnover_from_pre"][pos]
                    if np.isfinite(old_v):
                        drift_edges_old.append(float(old_v))
                    if np.isfinite(new_v):
                        drift_edges_new.append(float(new_v))
                    if np.isfinite(turn_v):
                        drift_edges_turn.append(float(turn_v))
                cur_v = edge_stats["cur_peer_share"][pos]
                cur_b = edge_stats["cur_peer_base"][pos]
                if cohort_id == 1 and np.isfinite(cur_v) and np.isfinite(cur_b):
                    drift_cur_share.append(float(cur_v))
                    drift_cur_base.append(float(cur_b))

            round_rows.append(
                {
                    "dataset": cfg.dataset,
                    "rho": cfg.rho,
                    "seed": cfg.seed,
                    "window": window_idx,
                    "window_round": window_round,
                    "global_round": global_round,
                    "train_loss": mean(losses),
                    "all_ndcg": all_ndcg,
                    "all_hr": all_hr,
                    "n_eval": int(test_ndcg.size),
                    "drift_ndcg": drift_ndcg,
                    "drift_hr": drift_hr,
                    "n_drift_eval": int(drift_sel.sum()),
                    "stable_ndcg": stable_ndcg,
                    "stable_hr": stable_hr,
                    "n_stable_eval": int(stable_sel.sum()),
                    "ecr_all": ecr_all,
                    "ecr_drift": ecr_drift,
                    "ecr_stable": ecr_stable,
                    "drift_old_share": mean(drift_edges_old),
                    "drift_new_share": mean(drift_edges_new),
                    "drift_turnover": mean(drift_edges_turn),
                    "drift_cur_share": mean(drift_cur_share),
                    "drift_cur_base": mean(drift_cur_base),
                    "drift_n_neighbors": float(np.mean(edge_stats["n_neighbors"]))
                    if edge_stats["n_neighbors"].size
                    else 0.0,
                }
            )
            window_ecrs.append(ecr_all)
            print(
                f"[rho={cfg.rho:.2f}] W{window_idx} R{window_round:03d}/{cfg.rounds_per_window} | "
                f"loss {mean(losses):.4f} | ECR {ecr_all:.4f} | "
                f"NDCG all {all_ndcg:.4f} drift {drift_ndcg:.4f} (n={int(drift_sel.sum())}) "
                f"stable {stable_ndcg:.4f} | oldpeer {mean(drift_edges_old):.3f} "
                f"turn {mean(drift_edges_turn):.3f} | {time.time() - round_start:.2f}s"
            )

        window_ndcgs.append(last_ndcg)
        write_csv(out_dir / "val02_round_metrics.csv", round_rows, ROUND_FIELDS)
        write_csv(out_dir / "val02_user_round_metrics.csv", user_rows, USER_FIELDS)
        print(
            f"[rho={cfg.rho:.2f}] W{window_idx} end | NDCG@10 {last_ndcg:.4f} | rows={len(user_rows)}"
        )

    summary = {
        "dataset": cfg.dataset,
        "rho": cfg.rho,
        "seed": cfg.seed,
        "graph_gate": cfg.graph_gate,
        "new_edge_init": cfg.new_edge_init,
        "graph_strength": cfg.graph_strength,
        "graph_mix": cfg.feddieg_graph_mix,
        "edge_ema": cfg.rho,
        "num_windows": cfg.num_windows,
        "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": total_rounds,
        "clients_per_round": clients_per_round,
        "n_drift": int(is_drift.sum()),
        "n_stable": int(is_stable.sum()),
        "all_avg_ndcg": mean(window_ndcgs),
        "all_final_ndcg": window_ndcgs[-1] if window_ndcgs else 0.0,
        "avg_ecr": mean(window_ecrs),
        "elapsed_seconds": time.time() - start_time,
    }
    write_csv(out_dir / "val02_summary.csv", [summary], SUMMARY_FIELDS)
    print(
        f"\n[rho={cfg.rho:.2f}] seed{cfg.seed} done in {summary['elapsed_seconds']:.1f}s | "
        f"avg_ndcg={summary['all_avg_ndcg']:.4f} | avg_ecr={summary['avg_ecr']:.4f}"
    )


# ======================================================================================
# CLI
# ======================================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="val02: 兴趣真实变化时固定 EMA 是否适应太慢")
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--rho", type=float, default=0.80, help="边权 EMA 系数（任务书：0 / 0.5 / 0.8 / 0.95）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--rounds-per-window", type=int, default=120)
    parser.add_argument("--window-mode", choices=["slice", "cumulative"], default="slice")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--microbatch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=99)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--feddieg-neighbors", type=int, default=10)
    parser.add_argument(
        "--graph-strength",
        choices=["paper", "strong", "mix"],
        default="paper",
        help="图影响强度预设：paper=已发表配置；strong=放大图影响并去掉两个门控；"
        "mix=只提高 graph_mix、保留门控（区分"
        "「图权重」与「门控」）",
    )
    parser.add_argument("--feddieg-graph-mix", type=float, default=None, help="默认跟随 --graph-strength")
    parser.add_argument("--feddieg-neighbor-temp", type=float, default=0.07)
    parser.add_argument("--feddieg-min-sim", type=float, default=0.20)
    parser.add_argument("--feddieg-tail-power", type=float, default=None, help="默认跟随 --graph-strength")
    parser.add_argument("--feddieg-memory-momentum", type=float, default=0.8)
    parser.add_argument("--feddieg-short-score-mix", type=float, default=0.05)
    parser.add_argument("--feddieg-confidence-power", type=float, default=None, help="默认跟随 --graph-strength")
    parser.add_argument("--feddieg-agreement-power", type=float, default=None, help="默认跟随 --graph-strength")
    parser.add_argument("--feddieg-score-local-mix", type=float, default=None, help="默认跟随 --graph-strength")
    parser.add_argument("--lambda-temp", type=float, default=0.01)
    parser.add_argument("--lambda-struct", type=float, default=0.20)
    parser.add_argument(
        "--graph-gate",
        choices=["raw", "blended"],
        default="raw",
        help="min_sim 阈值作用在原始余弦（raw，推荐）还是混合后的值（blended，原实现）",
    )
    parser.add_argument(
        "--new-edge-init",
        choices=["ema", "warm", "zero"],
        default="ema",
        help="新边的初值：ema=忠实复刻原实现（新边等价于从 (1-ρ)·sim 起步，默认）；"
        "warm=新边直接以当前余弦起步（会消掉滞后机制，仅作对照）；zero=从 0 慢慢涨（极端消融）",
    )
    parser.add_argument("--graph-block-size", type=int, default=2048)
    parser.add_argument("--primary-drift-only", action="store_true", help="drift 组只保留主类别确实改变的用户")
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--cohort-file", type=Path, default=None)
    parser.add_argument("--counts-file", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 图影响强度预设：先套预设，再让显式给出的命令行参数覆盖它
    preset = GRAPH_PRESETS[args.graph_strength]
    for key, preset_value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, preset_value)

    cfg = Val02Config(
        dataset=args.dataset,
        methods=("FedDIEG",),
        seed=args.seed,
        rounds=args.num_windows * args.rounds_per_window,
        participation_rate=args.participation_rate,
        clients_per_round=args.clients_per_round,
        microbatch_size=args.microbatch_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_users=args.max_eval_users,
        local_epochs=args.local_epochs,
        lr=args.lr,
        latent_dim=args.latent_dim,
        max_len=args.max_len,
        top_k=args.top_k,
        eval_every=args.eval_every,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
        feddieg_neighbors=args.feddieg_neighbors,
        feddieg_graph_mix=args.feddieg_graph_mix,
        feddieg_neighbor_temp=args.feddieg_neighbor_temp,
        feddieg_min_sim=args.feddieg_min_sim,
        feddieg_tail_power=args.feddieg_tail_power,
        feddieg_memory_momentum=args.feddieg_memory_momentum,
        feddieg_short_score_mix=args.feddieg_short_score_mix,
        feddieg_confidence_power=args.feddieg_confidence_power,
        feddieg_agreement_power=args.feddieg_agreement_power,
        feddieg_score_local_mix=args.feddieg_score_local_mix,
        feddieg_final_loss_weight=0.0,
        feddieg_boundary_refresh_rate=0.0,
        feddieg_boundary_refresh_momentum=0.2,
        num_threads=args.num_threads,
        num_windows=args.num_windows,
        rounds_per_window=args.rounds_per_window,
        candidate_count=args.candidate_count,
        window_mode=args.window_mode,
        lambda_temp=args.lambda_temp,
        lambda_struct=args.lambda_struct,
        rho=args.rho,
        graph_gate=args.graph_gate,
        new_edge_init=args.new_edge_init,
        graph_block_size=args.graph_block_size,
        graph_strength=args.graph_strength,
        primary_drift_only=args.primary_drift_only,
    )

    if not (0.0 <= cfg.rho < 1.0):
        raise SystemExit("--rho 必须在 [0, 1) 区间内")
    if cfg.feddieg_min_sim <= 0:
        raise SystemExit("--feddieg-min-sim 必须 > 0")

    torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else make_out_dir(args.out_root, cfg.dataset, cfg.rho, cfg.seed)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "val02_run.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, log_handle)  # type: ignore[assignment]

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print("Config:", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    slug = safe_slug(cfg.dataset)
    cohort_file = (args.cohort_file or (args.cohort_root / slug / "val02_drift_users.csv")).resolve()
    counts_file = (args.counts_file or (args.cohort_root / slug / "val02_counts.npz")).resolve()
    print(f"Cohort file : {cohort_file}")
    print(f"Counts file : {counts_file}")
    cohort = load_cohort(cohort_file, counts_file, cfg.primary_drift_only)

    exp2_cfg = Exp2Config(**{field: getattr(cfg, field) for field in Exp2Config.__dataclass_fields__})
    exp2_cfg.num_windows = cfg.num_windows
    exp2_cfg.rounds_per_window = cfg.rounds_per_window
    exp2_cfg.candidate_count = cfg.candidate_count
    exp2_cfg.window_mode = cfg.window_mode
    windows = [window.to(device) for window in load_rolling_windows(exp2_cfg)]

    run_single(cfg, windows, cohort, device, out_dir)
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
