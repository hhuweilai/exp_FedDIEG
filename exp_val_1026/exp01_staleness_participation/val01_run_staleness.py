#!/usr/bin/env python
"""验证实验 1（val01）：客户端长期不参与，会不会影响推荐质量？

对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验一」。
只做诊断，不设计算法。交付物是一张图：
「摘要陈旧程度 × 客户端参与率 × 推荐性能」。

两个条件（共用同一份预生成的客户端参与序列，同一 seed 下逐轮一致）
----------------------------------------------------------------
  upload_only        现状 FedDIEG：只有参与训练的客户端刷新摘要；陈旧摘要照样参与构图。
  upload_fresh_edges 对照：每轮用当前全局模型给全部客户端重算摘要（诊断用 oracle 上界，
                     不可部署），因此图中永无陈旧信息。两者之差 = 陈旧信息造成的损失。

这样设计是因为任务书要求"固定相同的客户端训练参与序列，仅改变服务器是否使用陈旧摘要构建邻居"，
否则无法区分"摘要陈旧"与"训练机会变少"。

输出（每次运行 4 个文件，约 10 KB）
----------------------------------
  exp_val_1026/exp01_staleness_participation/results/raw/
      <dataset>/<mode>/p<rate>/seed_<seed>/
          config.json                  超参数（可复现）
          val01_summary.csv            本次运行汇总（一行）
          val01_staleness_buckets.csv  陈旧度分桶 → 那张图的全部数据
          val01_run.log                训练日志

用法
----
  python exp_val_1026/exp01_staleness_participation/val01_run_staleness.py \
      --dataset movielens-1m --mode upload_only --participation-rate 0.1 --seed 42
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
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
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

# --------------------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------------------
SERVER_MODES = ("upload_only", "upload_fresh_edges")

STALENESS_BUCKETS: List[Tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    ("20-39", 20, 39),
    ("40+", 40, 10**9),
]
UPLOAD_BUCKETS: List[Tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
    ("32+", 32, 10**9),
]

# 从未上传过的客户端用这个哨兵值标陈旧度（不进入任何分桶）
NEVER_UPLOADED = 10**6


@dataclass
class Val01Config(Exp2Config):
    """在 Exp2 的时间窗口数据协议之上，只加"陈旧度感知"的实验旋钮。"""

    server_mode: str = "upload_only"
    num_windows: int = 5
    rounds_per_window: int = 120
    candidate_count: int = 99
    window_mode: str = "slice"
    lambda_temp: float = 0.01
    lambda_struct: float = 0.20
    edge_ema: float = 0.80
    graph_block_size: int = 2048


BUCKET_FIELDS = [
    "dataset",
    "mode",
    "participation_rate",
    "seed",
    "window",
    "bucket_type",
    "bucket",
    "n_users",
    "mean_ndcg",
    "mean_hr",
    "mean_staleness",
    "mean_uploads",
]

SUMMARY_FIELDS = [
    "dataset",
    "mode",
    "participation_rate",
    "seed",
    "num_windows",
    "rounds_per_window",
    "total_rounds",
    "clients_per_round",
    "avg_window_ndcg",
    "final_window_ndcg",
    "avg_window_hr",
    "mean_staleness",
    "p90_staleness",
    "avg_ecr",
    "avg_boundary_drop",
    "elapsed_seconds",
]


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def write_csv(path: Path, rows: List[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


def make_out_dir(out_root: Path, dataset: str, mode: str, rate: float, seed: int) -> Path:
    """四级分类：<out_root>/<dataset>/<mode>/p<rate>/seed_<seed>/"""
    return out_root / safe_slug(dataset) / mode / f"p{rate:.2f}" / f"seed_{seed}"


def bucket_name(value: int, buckets: List[Tuple[str, int, int]]) -> Optional[str]:
    for name, low, high in buckets:
        if low <= value <= high:
            return name
    return None


def quantile(sorted_values: np.ndarray, q: float) -> float:
    if sorted_values.size == 0:
        return 0.0
    return float(sorted_values[int(round(q * (sorted_values.size - 1)))])


# --------------------------------------------------------------------------------------
# 图：训练时取邻居 / 服务器端更新边权
# --------------------------------------------------------------------------------------
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
    topk: int,
    min_sim: float,
    edge_ema: float,
    block_size: int,
) -> float:
    """对 update_users 的邻居做 Top-K + EMA 平滑，返回本轮边变化率 ECR。

    用缓存摘要的余弦相似度做候选，再按 ρ 与历史边权做指数平滑。
    首次出现的边直接以第一次观测作为初始权值（warm start）：若像原实现那样从 0 开始平滑，
    由于 min_sim > (1-ρ)·cos_max，图会永远是空的。
    """
    active = torch.nonzero(active_clients, as_tuple=False).flatten()
    users = update_users[active_clients[update_users]]
    k = min(int(topk), int(active.numel()) - 1)
    if users.numel() == 0 or k <= 0:
        return 0.0

    device = summary_bank.device
    norm_bank = F.normalize(summary_bank, p=2, dim=1, eps=1e-8)
    norm_active = norm_bank[active]
    neighbors_cpu = graph_neighbors.tolist()
    weights_cpu = graph_weights.tolist()
    mask_cpu = graph_mask.tolist()
    active_pos = {int(user.item()): idx for idx, user in enumerate(active)}
    edge_ema = min(max(float(edge_ema), 0.0), 0.99)

    changes: List[float] = []
    for block in batched_indices(torch.arange(users.numel(), device=device), block_size):
        block_users = users[block]
        block_list = block_users.tolist()
        sim = torch.matmul(norm_bank[block_users], norm_active.T).clamp_min(0.0)
        sim = sim.masked_fill(block_users.unsqueeze(1) == active.unsqueeze(0), 0.0)

        if edge_ema > 0:
            prev = torch.zeros_like(sim)
            flat_rows: List[int] = []
            flat_cols: List[int] = []
            flat_vals: List[float] = []
            for row, user in enumerate(block_list):
                if not mask_cpu[user]:
                    continue
                for neighbor, weight in zip(neighbors_cpu[user], weights_cpu[user]):
                    col = active_pos.get(int(neighbor))
                    if col is None:
                        continue
                    flat_rows.append(row)
                    flat_cols.append(col)
                    flat_vals.append(float(weight))
            if flat_rows:
                row_t = torch.tensor(flat_rows, dtype=torch.long, device=device)
                col_t = torch.tensor(flat_cols, dtype=torch.long, device=device)
                prev[row_t, col_t] = torch.tensor(flat_vals, dtype=torch.float32, device=device)
                blended = (1.0 - edge_ema) * sim + edge_ema * prev
                has_prev = torch.zeros(len(block_list), dtype=torch.bool, device=device)
                has_prev[row_t] = True
                sim = torch.where(has_prev.unsqueeze(1), blended, sim)

        top_values, top_positions = torch.topk(sim, k, dim=1)
        ids_cpu = active[top_positions].tolist()
        vals_cpu = top_values.tolist()

        ids_nested: List[List[int]] = []
        weights_nested: List[List[float]] = []
        mask_nested: List[bool] = []
        for row, user in enumerate(block_list):
            old = {int(v) for v in neighbors_cpu[user] if int(v) >= 0} if mask_cpu[user] else set()
            kept = [(int(i), float(v)) for i, v in zip(ids_cpu[row], vals_cpu[row]) if v >= min_sim]
            if kept:
                new_ids = [i for i, _ in kept]
                total = sum(v for _, v in kept)
                new_weights = [v / max(total, 1e-8) for _, v in kept]
            else:
                new_ids, new_weights = [], []
            if old:
                union = old | set(new_ids)
                changes.append(len(old ^ set(new_ids)) / max(len(union), 1))
            pad = k - len(new_ids)
            ids_nested.append(new_ids + [-1] * pad)
            weights_nested.append(new_weights + [0.0] * pad)
            mask_nested.append(len(new_ids) > 0)

        graph_neighbors[block_users] = torch.tensor(ids_nested, dtype=torch.long, device=graph_neighbors.device)
        graph_weights[block_users] = torch.tensor(
            weights_nested, dtype=graph_weights.dtype, device=graph_weights.device
        )
        graph_mask[block_users] = torch.tensor(mask_nested, dtype=torch.bool, device=graph_mask.device)

    return mean(changes)


# --------------------------------------------------------------------------------------
# 评测：逐用户返回，便于把"陈旧度"和"推荐效果"对齐到同一个客户端
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_per_user(
    model: FedDIEGModel,
    data: FastData,
    cfg: Val01Config,
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


def summarize_buckets(
    cfg: Val01Config,
    window_idx: int,
    test_users: np.ndarray,
    test_ndcg: np.ndarray,
    test_hr: np.ndarray,
    staleness: torch.Tensor,
    n_uploads: torch.Tensor,
) -> List[Dict[str, object]]:
    """把"逐用户 NDCG × 自身陈旧度"汇总成桶表 —— 那张图的全部数据来源。"""
    if test_users.size == 0:
        return []
    users = torch.as_tensor(test_users, dtype=torch.long, device=staleness.device)
    stale_user = staleness[users].tolist()
    uploads_user = n_uploads[users].tolist()

    rows: List[Dict[str, object]] = []
    for bucket_type, buckets, values in (
        ("staleness", STALENESS_BUCKETS, stale_user),
        ("uploads", UPLOAD_BUCKETS, uploads_user),
    ):
        groups: Dict[str, List[int]] = {}
        for i, value in enumerate(values):
            name = bucket_name(int(value), buckets) if value < NEVER_UPLOADED else None
            if name is not None:
                groups.setdefault(name, []).append(i)
        for name, _, _ in buckets:
            indices = groups.get(name)
            if not indices:
                continue
            sel = np.asarray(indices, dtype=np.int64)
            rows.append(
                {
                    "dataset": cfg.dataset,
                    "mode": cfg.server_mode,
                    "participation_rate": cfg.participation_rate,
                    "seed": cfg.seed,
                    "window": window_idx,
                    "bucket_type": bucket_type,
                    "bucket": name,
                    "n_users": int(sel.size),
                    "mean_ndcg": float(test_ndcg[sel].mean()),
                    "mean_hr": float(test_hr[sel].mean()),
                    "mean_staleness": float(np.mean([stale_user[i] for i in sel])),
                    "mean_uploads": float(np.mean([uploads_user[i] for i in sel])),
                }
            )
    return rows


# --------------------------------------------------------------------------------------
# 参与序列：所有 mode 共用，保证"只改变是否使用陈旧摘要"
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# 单次运行
# --------------------------------------------------------------------------------------
def run_single(cfg: Val01Config, windows: List[FastData], device: torch.device, out_dir: Path) -> None:
    mode = cfg.server_mode
    set_seed(cfg.seed)

    model = FedDIEGModel(windows[0].num_items, cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
    num_users = windows[0].num_users
    clients_per_round = min(cfg.clients_per_round or max(int(cfg.participation_rate * num_users), 1), num_users)
    total_rounds = cfg.num_windows * cfg.rounds_per_window
    schedule = build_participation_schedule(num_users, total_rounds, clients_per_round, cfg.seed, device)

    topk = max(int(cfg.feddieg_neighbors), 1)
    summary_bank = torch.zeros((num_users, cfg.latent_dim), device=device)
    active_clients = torch.zeros(num_users, dtype=torch.bool, device=device)
    graph_neighbors = torch.full((num_users, topk), -1, dtype=torch.long, device=device)
    graph_weights = torch.zeros((num_users, topk), dtype=torch.float32, device=device)
    graph_mask = torch.zeros(num_users, dtype=torch.bool, device=device)
    last_upload_round = torch.full((num_users,), -1, dtype=torch.long, device=device)
    n_uploads = torch.zeros(num_users, dtype=torch.int32, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    bucket_rows: List[Dict[str, object]] = []
    window_ndcgs: List[float] = []
    window_hrs: List[float] = []
    window_staleness: List[float] = []
    window_p90: List[float] = []
    window_drops: List[float] = []
    window_ecrs: List[float] = []
    start_time = time.time()
    global_round = 0
    prev_window_last_ndcg: Optional[float] = None

    def staleness_now() -> torch.Tensor:
        return torch.where(
            last_upload_round >= 0,
            global_round - last_upload_round,
            torch.full_like(last_upload_round, NEVER_UPLOADED),
        )

    for window_idx, data in enumerate(windows, start=1):
        print(f"\n[{mode}] Window {window_idx}/{cfg.num_windows} | eval_users={data.test_users.numel()}")
        staleness = staleness_now()
        _, init_hr, init_ndcg = evaluate_per_user(
            model, data, cfg, summary_bank, graph_neighbors, graph_weights, graph_mask, "test"
        )
        initial_ndcg = float(init_ndcg.mean()) if init_ndcg.size else 0.0
        drop = 0.0 if prev_window_last_ndcg is None else max(prev_window_last_ndcg - initial_ndcg, 0.0)
        last_ndcg = initial_ndcg
        last_hr = float(init_hr.mean()) if init_hr.size else 0.0
        window_ecs: List[float] = []

        for window_round in range(1, cfg.rounds_per_window + 1):
            global_round += 1
            round_start = time.time()
            sampled = schedule[global_round - 1]

            if mode == "upload_fresh_edges":
                # 诊断上界：服务器每轮给全部客户端重算摘要（前向不消耗随机数）
                model.eval()
                eligible = torch.nonzero(data.train_mask.sum(dim=1) > 0, as_tuple=False).flatten()
                momentum = min(max(cfg.feddieg_memory_momentum, 0.0), 0.99)
                with torch.no_grad():
                    for users in batched_indices(eligible, cfg.microbatch_size):
                        seq = data.train_seq[users]
                        output = model(seq)
                        local_feats, short = (output if isinstance(output, tuple) else (output, None))
                        new_memory = short if short is not None else masked_sequence_mean(local_feats, seq)
                        seen = active_clients[users].float().unsqueeze(-1)
                        summary_bank[users] = seen * (
                            momentum * summary_bank[users] + (1.0 - momentum) * new_memory
                        ) + (1.0 - seen) * new_memory
                        active_clients[users] = True
                model.train()

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

                    # 只有参与训练的客户端才刷新缓存摘要与"上次上传轮次"
                    new_memory = short_summary.detach()
                    momentum = min(max(cfg.feddieg_memory_momentum, 0.0), 0.99)
                    with torch.no_grad():
                        seen = active_clients[users].float().unsqueeze(-1)
                        summary_bank[users] = seen * (
                            momentum * summary_bank[users] + (1.0 - momentum) * new_memory
                        ) + (1.0 - seen) * new_memory
                        active_clients[users] = True
                        last_upload_round[users] = global_round
                        n_uploads[users] += 1

            staleness = staleness_now()
            window_ecs.append(
                update_graph(
                    sampled,
                    summary_bank,
                    active_clients,
                    graph_neighbors,
                    graph_weights,
                    graph_mask,
                    cfg.feddieg_neighbors,
                    cfg.feddieg_min_sim,
                    cfg.edge_ema,
                    cfg.graph_block_size,
                )
            )

            should_eval = (window_round % cfg.eval_every == 0) or (window_round == cfg.rounds_per_window)
            if should_eval:
                test_users, test_hr, test_ndcg = evaluate_per_user(
                    model, data, cfg, summary_bank, graph_neighbors, graph_weights, graph_mask, "test"
                )
                last_ndcg = float(test_ndcg.mean()) if test_ndcg.size else 0.0
                last_hr = float(test_hr.mean()) if test_hr.size else 0.0
                active_staleness = staleness[active_clients & (last_upload_round >= 0)].float()
                s_mean = float(active_staleness.mean().item()) if active_staleness.numel() else 0.0
                if window_round == cfg.rounds_per_window:
                    bucket_rows.extend(
                        summarize_buckets(cfg, window_idx, test_users, test_ndcg, test_hr, staleness, n_uploads)
                    )
                print(
                    f"[{mode}] p{cfg.participation_rate:.2f} W{window_idx} R{window_round:03d}/"
                    f"{cfg.rounds_per_window} | loss {mean(losses):.4f} | ECR {window_ecs[-1]:.4f} | "
                    f"staleness {s_mean:.1f} | NDCG@10 {last_ndcg:.4f} | {time.time() - round_start:.2f}s"
                )

        active_staleness = staleness[active_clients & (last_upload_round >= 0)].float()
        if active_staleness.numel():
            sorted_staleness = torch.sort(active_staleness).values.cpu().numpy()
            s_mean = float(active_staleness.mean().item())
            s_p90 = quantile(sorted_staleness, 0.9)
        else:
            s_mean = s_p90 = 0.0

        window_ndcgs.append(last_ndcg)
        window_hrs.append(last_hr)
        window_staleness.append(s_mean)
        window_p90.append(s_p90)
        window_drops.append(drop)
        window_ecrs.extend(window_ecs)
        prev_window_last_ndcg = last_ndcg
        print(f"[{mode}] W{window_idx} end | NDCG@10 {last_ndcg:.4f} | staleness {s_mean:.2f} | p90 {s_p90:.0f}")

        # 每窗口落盘一次，长跑中断也能保住结果
        write_csv(out_dir / "val01_staleness_buckets.csv", bucket_rows, BUCKET_FIELDS)

    summary = {
        "dataset": cfg.dataset,
        "mode": mode,
        "participation_rate": cfg.participation_rate,
        "seed": cfg.seed,
        "num_windows": cfg.num_windows,
        "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": total_rounds,
        "clients_per_round": clients_per_round,
        "avg_window_ndcg": mean(window_ndcgs),
        "final_window_ndcg": window_ndcgs[-1] if window_ndcgs else 0.0,
        "avg_window_hr": mean(window_hrs),
        "mean_staleness": mean(window_staleness),
        "p90_staleness": mean(window_p90),
        "avg_ecr": mean(window_ecrs),
        "avg_boundary_drop": mean(window_drops[1:]) if len(window_drops) > 1 else 0.0,
        "elapsed_seconds": time.time() - start_time,
    }
    write_csv(out_dir / "val01_summary.csv", [summary], SUMMARY_FIELDS)
    print(
        f"\n[{mode}] p{cfg.participation_rate:.2f} seed{cfg.seed} done in {summary['elapsed_seconds']:.1f}s | "
        f"avg_window_ndcg={summary['avg_window_ndcg']:.4f} | mean_staleness={summary['mean_staleness']:.2f} | "
        f"buckets={len(bucket_rows)}"
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="val01: 客户端长期不参与是否影响推荐质量")
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--mode", default="upload_only", choices=SERVER_MODES)
    parser.add_argument("--participation-rate", type=float, default=0.1)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--feddieg-graph-mix", type=float, default=0.05)
    parser.add_argument("--feddieg-neighbor-temp", type=float, default=0.07)
    parser.add_argument("--feddieg-min-sim", type=float, default=0.20)
    parser.add_argument("--feddieg-tail-power", type=float, default=4.0)
    parser.add_argument("--feddieg-memory-momentum", type=float, default=0.8)
    parser.add_argument("--feddieg-short-score-mix", type=float, default=0.05)
    parser.add_argument("--feddieg-confidence-power", type=float, default=0.5)
    parser.add_argument("--feddieg-agreement-power", type=float, default=0.5)
    parser.add_argument("--feddieg-score-local-mix", type=float, default=0.2)
    parser.add_argument("--lambda-temp", type=float, default=0.01)
    parser.add_argument("--lambda-struct", type=float, default=0.20)
    parser.add_argument("--edge-ema", type=float, default=0.80)
    parser.add_argument("--graph-block-size", type=int, default=2048)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "exp_val_1026" / "exp01_staleness_participation" / "results" / "raw",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Val01Config(
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
        edge_ema=args.edge_ema,
        graph_block_size=args.graph_block_size,
        server_mode=args.mode,
    )

    if not (0.0 < cfg.participation_rate <= 1.0):
        raise SystemExit("--participation-rate 必须在 (0, 1] 区间内")
    if cfg.feddieg_min_sim <= 0:
        raise SystemExit("--feddieg-min-sim 必须 > 0")

    torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else make_out_dir(args.out_root, cfg.dataset, cfg.server_mode, cfg.participation_rate, cfg.seed)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "val01_run.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print("Config:", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    exp2_cfg = Exp2Config(**{field: getattr(cfg, field) for field in Exp2Config.__dataclass_fields__})
    exp2_cfg.num_windows = cfg.num_windows
    exp2_cfg.rounds_per_window = cfg.rounds_per_window
    exp2_cfg.candidate_count = cfg.candidate_count
    exp2_cfg.window_mode = cfg.window_mode
    windows = [window.to(device) for window in load_rolling_windows(exp2_cfg)]

    run_single(cfg, windows, device, out_dir)
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
