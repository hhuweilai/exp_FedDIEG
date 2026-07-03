"""Fast GPU Experiment 3: FedDIEG mechanism ablations.

This runner reuses the aligned fast Experiment 2 data protocol and evaluates
FedDIEG internal ablations under chronological drift. It is intentionally
separate from the legacy per-client Exp03 scripts so it can run efficiently on
the 4090 server.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_01_fast_gpu.run_exp01_fast_gpu import (  # noqa: E402
    ALL_METHODS,
    FastConfig,
    FastData,
    FedDIEGModel,
    Tee,
    batched_indices,
    bce_seq_loss,
    masked_sequence_mean,
    sample_negative_like,
    set_seed,
)
from exp_02_fast_gpu.run_exp02_fast_gpu import (  # noqa: E402
    Exp2Config,
    load_rolling_windows,
    mean,
    run_method_over_windows,
    variance,
)


@dataclass
class VariantConfig:
    name: str
    lambda_temp: float
    lambda_struct: float
    edge_ema: float
    memory_momentum: float
    graph_mix_scale: float = 1.0


@dataclass
class Exp3Config(FastConfig):
    variants: Tuple[str, ...] = ALL_METHODS
    num_windows: int = 5
    rounds_per_window: int = 80
    candidate_count: int = 99
    window_mode: str = "slice"
    lambda_temp: float = 0.05
    lambda_struct: float = 0.20
    edge_ema: float = 0.80
    ecr_update_scope: str = "sampled"
    stability_ecr_weight: float = 0.05
    stability_drop_weight: float = 1.0
    graph_block_size: int = 2048
    oscillation_window: int = 10


ABLATION_VARIANTS = ["Ours-Plain", "Ours-NoTemp", "Ours-NoStruct", "Ours-Full"]
VARIANT_ORDER = list(ALL_METHODS)
VARIANT_CHOICES = list(ALL_METHODS) + ABLATION_VARIANTS


ROUND_FIELDS = [
    "variant",
    "dataset",
    "window",
    "global_round",
    "window_round",
    "phase",
    "train_loss",
    "rec_loss",
    "temp_loss",
    "struct_loss",
    "ecr",
    "edge_density",
    "val_hr",
    "val_ndcg",
    "val_recall",
    "val_auc",
    "test_hr",
    "test_ndcg",
    "test_recall",
    "test_auc",
    "seconds",
]

WINDOW_FIELDS = [
    "variant",
    "dataset",
    "window",
    "eval_users",
    "initial_test_ndcg",
    "last_test_ndcg",
    "best_test_ndcg",
    "adaptation_gain",
    "boundary_drop_from_prev_window",
    "window_max_drop",
    "loss_oscillation",
    "avg_train_loss",
    "avg_ecr",
    "avg_edge_density",
]

SUMMARY_FIELDS = [
    "variant",
    "dataset",
    "seed",
    "num_windows",
    "rounds_per_window",
    "total_rounds",
    "clients_per_round",
    "avg_window_ndcg",
    "best_test_ndcg",
    "final_test_ndcg",
    "max_drop",
    "boundary_max_drop",
    "avg_boundary_drop",
    "window_variance",
    "loss_oscillation",
    "avg_ecr",
    "avg_edge_density",
    "stability_score",
    "elapsed_seconds",
]


def compute_max_drop(values: List[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_drop = 0.0
    for value in values:
        peak = max(peak, value)
        max_drop = max(max_drop, peak - value)
    return max_drop


def compute_loss_oscillation(losses: List[float], window_size: int = 10) -> float:
    clean = [value for value in losses if math.isfinite(value)]
    if len(clean) <= 1:
        return 0.0
    window_size = max(min(window_size, len(clean)), 2)
    ranges = [max(clean[idx : idx + window_size]) - min(clean[idx : idx + window_size]) for idx in range(len(clean) - window_size + 1)]
    return mean(ranges)


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_variant_configs(cfg: Exp3Config) -> Dict[str, VariantConfig]:
    return {
        "Ours-Plain": VariantConfig(
            "Ours-Plain",
            lambda_temp=0.0,
            lambda_struct=0.0,
            edge_ema=0.0,
            memory_momentum=0.0,
            graph_mix_scale=1.0,
        ),
        "Ours-NoTemp": VariantConfig(
            "Ours-NoTemp",
            lambda_temp=0.0,
            lambda_struct=cfg.lambda_struct,
            edge_ema=cfg.edge_ema,
            memory_momentum=0.0,
            graph_mix_scale=1.0,
        ),
        "Ours-NoStruct": VariantConfig(
            "Ours-NoStruct",
            lambda_temp=cfg.lambda_temp,
            lambda_struct=0.0,
            edge_ema=0.0,
            memory_momentum=cfg.feddieg_memory_momentum,
            graph_mix_scale=1.0,
        ),
        "Ours-Full": VariantConfig(
            "Ours-Full",
            lambda_temp=cfg.lambda_temp,
            lambda_struct=cfg.lambda_struct,
            edge_ema=cfg.edge_ema,
            memory_momentum=cfg.feddieg_memory_momentum,
            graph_mix_scale=1.0,
        ),
    }


def run_method_as_variant(
    method: str,
    windows: List[FastData],
    cfg: Exp3Config,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    exp2_rounds, exp2_windows, exp2_summary = run_method_over_windows(method, windows, cfg, device)
    round_rows: List[Dict[str, object]] = []
    losses: List[float] = []
    test_ndcgs: List[float] = []
    losses_by_window: Dict[int, List[float]] = {}
    ndcgs_by_window: Dict[int, List[float]] = {}

    for row in exp2_rounds:
        window_idx = int(row.get("window", 0))
        train_loss = float(row.get("train_loss", 0.0))
        test_ndcg = float(row.get("test_ndcg", 0.0))
        losses.append(train_loss)
        test_ndcgs.append(test_ndcg)
        losses_by_window.setdefault(window_idx, []).append(train_loss)
        ndcgs_by_window.setdefault(window_idx, []).append(test_ndcg)
        round_rows.append(
            {
                "variant": method,
                "dataset": row.get("dataset", cfg.dataset),
                "window": window_idx,
                "global_round": row.get("global_round", ""),
                "window_round": row.get("window_round", ""),
                "phase": "baseline",
                "train_loss": train_loss,
                "rec_loss": train_loss,
                "temp_loss": 0.0,
                "struct_loss": 0.0,
                "ecr": 0.0,
                "edge_density": 0.0,
                "val_hr": row.get("val_hr", ""),
                "val_ndcg": row.get("val_ndcg", ""),
                "val_recall": row.get("val_recall", ""),
                "val_auc": row.get("val_auc", ""),
                "test_hr": row.get("test_hr", ""),
                "test_ndcg": test_ndcg,
                "test_recall": row.get("test_recall", ""),
                "test_auc": row.get("test_auc", ""),
                "seconds": row.get("seconds", ""),
            }
        )

    window_rows: List[Dict[str, object]] = []
    window_last_ndcgs: List[float] = []
    boundary_drops: List[float] = []
    for row in exp2_windows:
        window_idx = int(row.get("window", 0))
        initial_ndcg = float(row.get("initial_test_ndcg", 0.0))
        last_ndcg = float(row.get("last_test_ndcg", 0.0))
        best_ndcg = float(row.get("best_test_ndcg_at_best_val", last_ndcg))
        boundary_drop = float(row.get("boundary_drop_from_prev_window", 0.0))
        series = [initial_ndcg] + ndcgs_by_window.get(window_idx, [])
        window_last_ndcgs.append(last_ndcg)
        boundary_drops.append(boundary_drop)
        window_rows.append(
            {
                "variant": method,
                "dataset": row.get("dataset", cfg.dataset),
                "window": window_idx,
                "eval_users": row.get("eval_users", ""),
                "initial_test_ndcg": initial_ndcg,
                "last_test_ndcg": last_ndcg,
                "best_test_ndcg": best_ndcg,
                "adaptation_gain": row.get("adaptation_gain", ""),
                "boundary_drop_from_prev_window": boundary_drop,
                "window_max_drop": compute_max_drop(series),
                "loss_oscillation": compute_loss_oscillation(losses_by_window.get(window_idx, []), cfg.oscillation_window),
                "avg_train_loss": mean(losses_by_window.get(window_idx, [])),
                "avg_ecr": 0.0,
                "avg_edge_density": 0.0,
            }
        )

    max_drop = compute_max_drop(test_ndcgs or window_last_ndcgs)
    summary = {
        "variant": method,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "num_windows": cfg.num_windows,
        "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": cfg.num_windows * cfg.rounds_per_window,
        "clients_per_round": exp2_summary.get("clients_per_round", ""),
        "avg_window_ndcg": exp2_summary.get("avg_window_ndcg", mean(window_last_ndcgs)),
        "best_test_ndcg": max([float(row.get("best_test_ndcg", 0.0)) for row in window_rows], default=0.0),
        "final_test_ndcg": exp2_summary.get("final_window_ndcg", window_last_ndcgs[-1] if window_last_ndcgs else 0.0),
        "max_drop": max_drop,
        "boundary_max_drop": max(boundary_drops) if boundary_drops else 0.0,
        "avg_boundary_drop": mean(boundary_drops),
        "window_variance": variance(window_last_ndcgs),
        "loss_oscillation": compute_loss_oscillation(losses, cfg.oscillation_window),
        "avg_ecr": 0.0,
        "avg_edge_density": 0.0,
        "stability_score": mean(window_last_ndcgs) - cfg.stability_drop_weight * max_drop,
        "elapsed_seconds": exp2_summary.get("total_time_seconds", 0.0),
    }
    return round_rows, window_rows, summary


def make_model(num_items: int, cfg: Exp3Config, variant: VariantConfig, device: torch.device) -> FedDIEGModel:
    model_cfg = Exp3Config(**asdict(cfg))
    model_cfg.feddieg_graph_mix = cfg.feddieg_graph_mix * variant.graph_mix_scale
    return FedDIEGModel(num_items, model_cfg).to(device)


def graph_batch(
    users: torch.Tensor,
    memory_bank: torch.Tensor,
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
    safe_neighbors = neighbors.clamp_min(0)
    neighbor_summaries = memory_bank[safe_neighbors]
    weights = weights * mask.float().unsqueeze(1)
    return neighbor_summaries, weights, mask


@torch.no_grad()
def update_graph_for_users(
    users: torch.Tensor,
    memory_bank: torch.Tensor,
    active_clients: torch.Tensor,
    graph_neighbors: torch.Tensor,
    graph_weights: torch.Tensor,
    graph_mask: torch.Tensor,
    topk: int,
    min_sim: float,
    edge_ema: float,
) -> Tuple[float, float]:
    active_indices = torch.nonzero(active_clients, as_tuple=False).flatten()
    if active_indices.numel() <= 1 or users.numel() == 0:
        return 0.0, 0.0
    topk = min(int(topk), int(active_indices.numel()) - 1)
    if topk <= 0:
        return 0.0, 0.0

    norm_bank = F.normalize(memory_bank, p=2, dim=1, eps=1e-8)
    users = users[active_clients[users]]
    if users.numel() == 0:
        return 0.0, 0.0

    changes: List[float] = []
    densities: List[float] = []
    edge_ema = min(max(float(edge_ema), 0.0), 0.99)
    sim = torch.matmul(norm_bank[users], norm_bank[active_indices].T).clamp_min(0.0)
    self_mask = users.unsqueeze(1) == active_indices.unsqueeze(0)
    sim = sim.masked_fill(self_mask, 0.0)

    if edge_ema > 0:
        prev_dense = torch.zeros_like(sim)
        active_pos = {int(user.item()): idx for idx, user in enumerate(active_indices)}
        for row_idx, user in enumerate(users.tolist()):
            if not bool(graph_mask[user].item()):
                continue
            for neighbor, weight in zip(graph_neighbors[user].tolist(), graph_weights[user].tolist()):
                pos = active_pos.get(int(neighbor))
                if pos is not None and neighbor >= 0:
                    prev_dense[row_idx, pos] = float(weight)
        sim = (1.0 - edge_ema) * sim + edge_ema * prev_dense

    top_values, top_positions = torch.topk(sim, topk, dim=1)
    top_ids = active_indices[top_positions]
    score_mask = top_values >= min_sim

    for row_idx, user in enumerate(users.tolist()):
        old_set = set(int(v) for v in graph_neighbors[user].tolist() if int(v) >= 0 and bool(graph_mask[user].item()))
        valid = score_mask[row_idx]
        if valid.any():
            ids = top_ids[row_idx][valid]
            values = top_values[row_idx][valid]
            weights = values / values.sum().clamp_min(1e-8)
            new_ids = ids.tolist()
            new_weights = weights.tolist()
        else:
            new_ids = []
            new_weights = []
        new_set = set(int(v) for v in new_ids)
        if old_set:
            union = old_set | new_set
            changes.append(len(old_set ^ new_set) / max(len(union), 1))
        densities.append(len(new_set) / max(topk, 1))

        graph_neighbors[user].fill_(-1)
        graph_weights[user].zero_()
        graph_mask[user] = len(new_ids) > 0
        if new_ids:
            count = min(len(new_ids), topk)
            graph_neighbors[user, :count] = torch.tensor(new_ids[:count], dtype=torch.long, device=graph_neighbors.device)
            graph_weights[user, :count] = torch.tensor(new_weights[:count], dtype=graph_weights.dtype, device=graph_weights.device)

    return mean(changes), mean(densities)


@torch.no_grad()
def evaluate_variant(
    model: FedDIEGModel,
    data_users: torch.Tensor,
    data_seqs: torch.Tensor,
    data_items: torch.Tensor,
    data: FastData,
    cfg: Exp3Config,
    memory_bank: torch.Tensor,
    graph_neighbors: torch.Tensor,
    graph_weights: torch.Tensor,
    graph_mask: torch.Tensor,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total_recall = 0.0
    total_auc = 0.0
    total = 0
    users = data_users
    seqs = data_seqs
    items = data_items
    if cfg.max_eval_users is not None and cfg.max_eval_users > 0:
        users = users[: cfg.max_eval_users]
        seqs = seqs[: cfg.max_eval_users]
        items = items[: cfg.max_eval_users]
    if users.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    for batch_idx in batched_indices(torch.arange(users.numel(), device=users.device), cfg.eval_batch_size):
        batch_users = users[batch_idx]
        batch_seqs = seqs[batch_idx]
        batch_items = items[batch_idx]
        neighbors, weights, mask = graph_batch(batch_users, memory_bank, graph_neighbors, graph_weights, graph_mask)
        if neighbors is not None:
            base_feats, _ = model(batch_seqs)
            feats, short_summary = model(batch_seqs, neighbors, weights, mask)
            score_feats = model.score_representation(feats[:, -1, :], short_summary, base_feats[:, -1, :])
        else:
            feats, short_summary = model(batch_seqs)
            score_feats = model.score_representation(feats[:, -1, :], short_summary)

        scores = torch.matmul(score_feats.unsqueeze(1), model.item_emb(batch_items).transpose(-1, -2)).squeeze(1)
        _, top_indices = torch.topk(scores, cfg.top_k, dim=1)
        hits = top_indices == 0
        found = hits.any(dim=1)
        ranks = hits.float().argmax(dim=1)
        ndcg = torch.where(found, math.log(2.0) / torch.log(ranks.float() + 2.0), torch.zeros_like(ranks.float()))
        positive_scores = scores[:, 0:1]
        negative_scores = scores[:, 1:]
        auc = (positive_scores > negative_scores).float().mean(dim=1)
        total_hr += found.float().sum().item()
        total_ndcg += ndcg.sum().item()
        total_recall += found.float().sum().item()
        total_auc += auc.sum().item()
        total += batch_users.size(0)
    model.train()
    return total_hr / total, total_ndcg / total, total_recall / total, total_auc / total


def record_eval(
    variant: str,
    cfg: Exp3Config,
    window_idx: int,
    global_round: int,
    window_round: int,
    phase: str,
    ecr: float,
    density: float,
    val_metrics: Tuple[float, float, float, float],
    test_metrics: Tuple[float, float, float, float],
    seconds: float,
) -> Dict[str, object]:
    return {
        "variant": variant,
        "dataset": cfg.dataset,
        "window": window_idx,
        "global_round": global_round,
        "window_round": window_round,
        "phase": phase,
        "train_loss": "",
        "rec_loss": "",
        "temp_loss": "",
        "struct_loss": "",
        "ecr": ecr,
        "edge_density": density,
        "val_hr": val_metrics[0],
        "val_ndcg": val_metrics[1],
        "val_recall": val_metrics[2],
        "val_auc": val_metrics[3],
        "test_hr": test_metrics[0],
        "test_ndcg": test_metrics[1],
        "test_recall": test_metrics[2],
        "test_auc": test_metrics[3],
        "seconds": seconds,
    }


@torch.no_grad()
def refresh_boundary_memory(
    model: FedDIEGModel,
    data: FastData,
    cfg: Exp3Config,
    variant: VariantConfig,
    memory_bank: torch.Tensor,
    active_clients: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    rate = min(max(float(cfg.feddieg_boundary_refresh_rate), 0.0), 1.0)
    if rate <= 0 or variant.memory_momentum <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    eligible = torch.nonzero(data.train_mask.sum(dim=1) > 0, as_tuple=False).flatten()
    if eligible.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if rate >= 1.0:
        selected = eligible
    else:
        count = max(int(rate * eligible.numel()), 1)
        perm = torch.randperm(eligible.numel(), generator=generator, device=device)[:count]
        selected = eligible[perm]

    model.eval()
    momentum = min(max(float(cfg.feddieg_boundary_refresh_momentum), 0.0), 0.99)
    for users in batched_indices(selected, cfg.microbatch_size):
        seq = data.train_seq[users]
        output = model(seq)
        if isinstance(output, tuple):
            local_feats, short_summary = output
        else:
            local_feats = output
            short_summary = masked_sequence_mean(local_feats, seq)
        new_memory = short_summary if short_summary.dim() == 2 else masked_sequence_mean(local_feats, seq)
        seen = active_clients[users].float().unsqueeze(-1)
        memory_bank[users] = seen * (momentum * memory_bank[users] + (1.0 - momentum) * new_memory) + (
            1.0 - seen
        ) * new_memory
        active_clients[users] = True
    model.train()
    return selected


def train_variant(
    variant: VariantConfig,
    windows: List[FastData],
    cfg: Exp3Config,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    print(f"\n[{variant.name}] Exp3 fast GPU start")
    print(
        f"[{variant.name}] lambda_temp={variant.lambda_temp} "
        f"lambda_struct={variant.lambda_struct} edge_ema={variant.edge_ema} "
        f"memory_momentum={variant.memory_momentum}"
    )
    set_seed(cfg.seed)
    model = make_model(windows[0].num_items, cfg, variant, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
    clients_per_round = cfg.clients_per_round or max(int(cfg.participation_rate * windows[0].num_users), 1)
    clients_per_round = min(clients_per_round, windows[0].num_users)
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    topk = max(int(cfg.feddieg_neighbors), 1)
    memory_bank = torch.zeros((windows[0].num_users, cfg.latent_dim), device=device)
    active_clients = torch.zeros(windows[0].num_users, dtype=torch.bool, device=device)
    graph_neighbors = torch.full((windows[0].num_users, topk), -1, dtype=torch.long, device=device)
    graph_weights = torch.zeros((windows[0].num_users, topk), dtype=torch.float32, device=device)
    graph_mask = torch.zeros(windows[0].num_users, dtype=torch.bool, device=device)

    round_records: List[Dict[str, object]] = []
    window_records: List[Dict[str, object]] = []
    eval_ndcgs: List[float] = []
    all_losses: List[float] = []
    all_ecrs: List[float] = []
    all_densities: List[float] = []
    start_time = time.time()
    global_round = 0
    prev_window_last_ndcg: Optional[float] = None
    best_test_ndcg = 0.0

    for window_idx, data in enumerate(windows, start=1):
        print(f"\n[{variant.name}] Window {window_idx}/{cfg.num_windows} | eval_users={data.test_users.numel()}")
        if window_idx > 1:
            refreshed_users = refresh_boundary_memory(model, data, cfg, variant, memory_bank, active_clients, generator, device)
            if refreshed_users.numel() > 0:
                update_graph_for_users(
                    refreshed_users,
                    memory_bank,
                    active_clients,
                    graph_neighbors,
                    graph_weights,
                    graph_mask,
                    cfg.feddieg_neighbors,
                    cfg.feddieg_min_sim,
                    variant.edge_ema,
                )
                print(f"[{variant.name}] W{window_idx} boundary memory refresh | clients={refreshed_users.numel()}")
        window_losses: List[float] = []
        window_ecrs: List[float] = []
        window_densities: List[float] = []
        window_eval_ndcgs: List[float] = []
        eval_start = time.time()
        initial_val = evaluate_variant(model, data.val_users, data.val_seq, data.val_items, data, cfg, memory_bank, graph_neighbors, graph_weights, graph_mask)
        initial_test = evaluate_variant(model, data.test_users, data.test_seq, data.test_items, data, cfg, memory_bank, graph_neighbors, graph_weights, graph_mask)
        round_records.append(
            record_eval(
                variant.name,
                cfg,
                window_idx,
                global_round,
                0,
                "initial",
                0.0,
                float(graph_mask.float().mean().item()),
                initial_val,
                initial_test,
                time.time() - eval_start,
            )
        )
        window_eval_ndcgs.append(initial_test[1])
        eval_ndcgs.append(initial_test[1])
        best_test_ndcg = max(best_test_ndcg, initial_test[1])
        boundary_drop = 0.0 if prev_window_last_ndcg is None else max(prev_window_last_ndcg - initial_test[1], 0.0)
        print(f"[{variant.name}] W{window_idx} initial | Test NDCG {initial_test[1]:.4f} | BoundaryDrop {boundary_drop:.4f}")

        for window_round in range(1, cfg.rounds_per_window + 1):
            global_round += 1
            round_start = time.time()
            sampled = torch.randperm(data.num_users, generator=generator, device=device)[:clients_per_round]
            losses: List[float] = []
            rec_losses: List[float] = []
            temp_losses: List[float] = []
            struct_losses: List[float] = []
            model.train()
            for _ in range(cfg.local_epochs):
                shuffled = sampled[torch.randperm(sampled.numel(), generator=generator, device=device)]
                for users in batched_indices(shuffled, cfg.microbatch_size):
                    seq = data.train_seq[users]
                    pos = data.train_pos[users]
                    mask = data.train_mask[users]
                    neg = sample_negative_like(pos, data.num_items)
                    optimizer.zero_grad(set_to_none=True)
                    neighbors, weights, gmask = graph_batch(users, memory_bank, graph_neighbors, graph_weights, graph_mask)
                    feats, short_summary = model(seq, neighbors, weights, gmask)
                    pos_logits = (feats * model.item_emb(pos)).sum(dim=-1)
                    neg_logits = (feats * model.item_emb(neg)).sum(dim=-1)
                    rec_loss = bce_seq_loss(pos_logits, neg_logits, mask)

                    has_prev = active_clients[users]
                    if variant.lambda_temp > 0 and has_prev.any():
                        prev_summary = memory_bank[users][has_prev].detach()
                        current_summary = short_summary[has_prev]
                        temp_loss = (1.0 - F.cosine_similarity(current_summary, prev_summary, dim=-1, eps=1e-8)).mean()
                    else:
                        temp_loss = torch.zeros((), device=device)

                    if variant.lambda_struct > 0 and neighbors is not None and weights is not None and gmask is not None and gmask.any():
                        valid_rows = gmask
                        current = short_summary[valid_rows]
                        neigh = neighbors[valid_rows]
                        target = weights[valid_rows].detach()
                        logits = F.cosine_similarity(current.unsqueeze(1), neigh, dim=-1, eps=1e-8).clamp_min(0.0)
                        logits = logits.masked_fill(target <= 0, -1e9)
                        pred = F.softmax(logits / max(cfg.feddieg_neighbor_temp, 1e-6), dim=1)
                        pred = torch.where(target > 0, pred, torch.zeros_like(pred))
                        pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-8)
                        struct_loss = F.mse_loss(pred, target)
                    else:
                        struct_loss = torch.zeros((), device=device)

                    loss = rec_loss + variant.lambda_temp * temp_loss + variant.lambda_struct * struct_loss
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
                    rec_losses.append(float(rec_loss.item()))
                    temp_losses.append(float(temp_loss.item()))
                    struct_losses.append(float(struct_loss.item()))

                    new_memory = short_summary.detach()
                    momentum = min(max(variant.memory_momentum, 0.0), 0.99)
                    seen = active_clients[users].float().unsqueeze(-1)
                    memory_bank[users] = seen * (momentum * memory_bank[users] + (1.0 - momentum) * new_memory) + (
                        1.0 - seen
                    ) * new_memory
                    active_clients[users] = True

            update_users = sampled
            if cfg.ecr_update_scope == "active":
                update_users = torch.nonzero(active_clients, as_tuple=False).flatten()
            ecr, density = update_graph_for_users(
                update_users,
                memory_bank,
                active_clients,
                graph_neighbors,
                graph_weights,
                graph_mask,
                cfg.feddieg_neighbors,
                cfg.feddieg_min_sim,
                variant.edge_ema,
            )
            all_ecrs.append(ecr)
            all_densities.append(density)
            window_ecrs.append(ecr)
            window_densities.append(density)
            avg_loss = mean(losses)
            avg_rec = mean(rec_losses)
            avg_temp = mean(temp_losses)
            avg_struct = mean(struct_losses)
            all_losses.append(avg_loss)
            window_losses.append(avg_loss)

            if window_round % cfg.eval_every == 0 or window_round == cfg.rounds_per_window:
                val_metrics = evaluate_variant(model, data.val_users, data.val_seq, data.val_items, data, cfg, memory_bank, graph_neighbors, graph_weights, graph_mask)
                test_metrics = evaluate_variant(model, data.test_users, data.test_seq, data.test_items, data, cfg, memory_bank, graph_neighbors, graph_weights, graph_mask)
                window_eval_ndcgs.append(test_metrics[1])
                eval_ndcgs.append(test_metrics[1])
                best_test_ndcg = max(best_test_ndcg, test_metrics[1])
                round_records.append(
                    {
                        "variant": variant.name,
                        "dataset": cfg.dataset,
                        "window": window_idx,
                        "global_round": global_round,
                        "window_round": window_round,
                        "phase": "eval",
                        "train_loss": avg_loss,
                        "rec_loss": avg_rec,
                        "temp_loss": avg_temp,
                        "struct_loss": avg_struct,
                        "ecr": ecr,
                        "edge_density": density,
                        "val_hr": val_metrics[0],
                        "val_ndcg": val_metrics[1],
                        "val_recall": val_metrics[2],
                        "val_auc": val_metrics[3],
                        "test_hr": test_metrics[0],
                        "test_ndcg": test_metrics[1],
                        "test_recall": test_metrics[2],
                        "test_auc": test_metrics[3],
                        "seconds": time.time() - round_start,
                    }
                )
                print(
                    f"[{variant.name}] W{window_idx} R{window_round:03d}/{cfg.rounds_per_window} "
                    f"G{global_round:03d} | Loss {avg_loss:.4f} | ECR {ecr:.4f} | "
                    f"Test NDCG {test_metrics[1]:.4f} | {time.time() - round_start:.2f}s"
                )

        last_ndcg = window_eval_ndcgs[-1]
        prev_window_last_ndcg = last_ndcg
        window_records.append(
            {
                "variant": variant.name,
                "dataset": cfg.dataset,
                "window": window_idx,
                "eval_users": data.test_users.numel(),
                "initial_test_ndcg": initial_test[1],
                "last_test_ndcg": last_ndcg,
                "best_test_ndcg": max(window_eval_ndcgs),
                "adaptation_gain": last_ndcg - initial_test[1],
                "boundary_drop_from_prev_window": boundary_drop,
                "window_max_drop": compute_max_drop(window_eval_ndcgs),
                "loss_oscillation": compute_loss_oscillation(window_losses, cfg.oscillation_window),
                "avg_train_loss": mean(window_losses),
                "avg_ecr": mean(window_ecrs),
                "avg_edge_density": mean(window_densities),
            }
        )

    window_last_ndcgs = [float(row["last_test_ndcg"]) for row in window_records]
    boundary_drops = [float(row["boundary_drop_from_prev_window"]) for row in window_records[1:]]
    max_drop = compute_max_drop(eval_ndcgs)
    boundary_max_drop = max(boundary_drops) if boundary_drops else 0.0
    avg_ecr = mean(all_ecrs)
    summary = {
        "variant": variant.name,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "num_windows": cfg.num_windows,
        "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": cfg.num_windows * cfg.rounds_per_window,
        "clients_per_round": clients_per_round,
        "avg_window_ndcg": mean(window_last_ndcgs),
        "best_test_ndcg": best_test_ndcg,
        "final_test_ndcg": window_last_ndcgs[-1] if window_last_ndcgs else 0.0,
        "max_drop": max_drop,
        "boundary_max_drop": boundary_max_drop,
        "avg_boundary_drop": mean(boundary_drops),
        "window_variance": variance(window_last_ndcgs),
        "loss_oscillation": compute_loss_oscillation(all_losses, cfg.oscillation_window),
        "avg_ecr": avg_ecr,
        "avg_edge_density": mean(all_densities),
        "stability_score": mean(window_last_ndcgs) - cfg.stability_drop_weight * max_drop - cfg.stability_ecr_weight * avg_ecr,
        "elapsed_seconds": time.time() - start_time,
    }
    return round_records, window_records, summary


def make_plots(summary_rows: List[Dict[str, object]], round_rows: List[Dict[str, object]], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    order = [variant for variant in VARIANT_ORDER if any(row["variant"] == variant for row in summary_rows)]
    colors = {
        "FedMF": "#8c564b",
        "FedSASRec": "#7f7f7f",
        "GPFedRec": "#9467bd",
        "FCUCR": "#bcbd22",
        "TarDGR": "#17becf",
        "F3CRec": "#f58518",
        "DGFedRS": "#4c78a8",
        "FedDIEG": "#54a24b",
        "Ours-Plain": "#7f7f7f",
        "Ours-NoTemp": "#f58518",
        "Ours-NoStruct": "#4c78a8",
        "Ours-Full": "#54a24b",
    }

    def bar(metric: str, title: str, ylabel: str, filename: str) -> None:
        values = {str(row["variant"]): float(row[metric]) for row in summary_rows}
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        ax.bar(order, [values[v] for v in order], color=[colors.get(v) for v in order])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=260)
        plt.close(fig)

    bar("max_drop", "Experiment 3: Max Drop", "Max Drop of NDCG@10", "exp03_max_drop.png")
    bar("boundary_max_drop", "Experiment 3: Boundary Max Drop", "Boundary Max Drop", "exp03_boundary_max_drop.png")
    bar("avg_ecr", "Experiment 3: Edge Change Rate", "Average ECR", "exp03_avg_ecr.png")
    bar("loss_oscillation", "Experiment 3: Loss Oscillation", "Loss Oscillation", "exp03_loss_oscillation.png")
    bar("stability_score", "Experiment 3: Stability Score", "Higher is better", "exp03_stability_score.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for variant in order:
        rows = [row for row in round_rows if row["variant"] == variant and row.get("test_ndcg") not in ("", None)]
        rows.sort(key=lambda row: int(row["global_round"]))
        ax.plot(
            [int(row["global_round"]) for row in rows],
            [float(row["test_ndcg"]) for row in rows],
            marker="o" if variant in {"FedDIEG", "Ours-Full"} else ".",
            linewidth=2.3 if variant in {"FedDIEG", "Ours-Full"} else 1.6,
            label=variant,
            color=colors.get(variant),
        )
    ax.set_title("Experiment 3: NDCG@10 Under Temporal Drift")
    ax.set_xlabel("Global Round")
    ax.set_ylabel("Test NDCG@10")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "exp03_ndcg_curves.png", dpi=260)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast GPU Experiment 3 ablation runner.")
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--variants", nargs="+", default=VARIANT_ORDER, choices=VARIANT_CHOICES)
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--rounds-per-window", type=int, default=80)
    parser.add_argument("--window-mode", choices=["slice", "cumulative"], default="slice")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--participation-rate", type=float, default=0.1)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=99)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--gpfedrec-neighbors", type=int, default=5)
    parser.add_argument("--gpfedrec-mix", type=float, default=0.1)
    parser.add_argument("--gpfedrec-warmup-rounds", type=int, default=20)
    parser.add_argument("--fcucR-kb-len", type=int, default=5)
    parser.add_argument("--fcucR-noise-std", type=float, default=0.01)
    parser.add_argument("--fcucR-distill-weight", type=float, default=0.1)
    parser.add_argument("--tardgr-retrieval-mix", type=float, default=0.2)
    parser.add_argument("--tardgr-warmup-rounds", type=int, default=20)
    parser.add_argument("--f3crec-replay-topk", type=int, default=8)
    parser.add_argument("--f3crec-replay-mix", type=float, default=0.15)
    parser.add_argument("--f3crec-temporal-beta", type=float, default=0.15)
    parser.add_argument("--f3crec-memory-momentum", type=float, default=0.60)
    parser.add_argument("--f3crec-warmup-rounds", type=int, default=20)
    parser.add_argument("--dgfedrs-aug-ratio", type=float, default=0.15)
    parser.add_argument("--dgfedrs-aug-mix", type=float, default=0.30)
    parser.add_argument("--dgfedrs-noise-std", type=float, default=0.02)
    parser.add_argument("--dgfedrs-schedule-power", type=float, default=1.0)
    parser.add_argument("--feddieg-neighbors", type=int, default=10)
    parser.add_argument("--feddieg-graph-mix", type=float, default=0.05)
    parser.add_argument("--feddieg-neighbor-temp", type=float, default=0.07)
    parser.add_argument("--feddieg-min-sim", type=float, default=0.20)
    parser.add_argument("--feddieg-tail-power", type=float, default=4.0)
    parser.add_argument("--feddieg-memory-momentum", type=float, default=0.80)
    parser.add_argument("--feddieg-short-score-mix", type=float, default=0.05)
    parser.add_argument("--feddieg-confidence-power", type=float, default=0.5)
    parser.add_argument("--feddieg-agreement-power", type=float, default=0.5)
    parser.add_argument("--feddieg-score-local-mix", type=float, default=0.2)
    parser.add_argument("--feddieg-boundary-refresh-rate", type=float, default=0.0)
    parser.add_argument("--feddieg-boundary-refresh-momentum", type=float, default=0.5)
    parser.add_argument("--lambda-temp", type=float, default=0.05)
    parser.add_argument("--lambda-struct", type=float, default=0.20)
    parser.add_argument("--edge-ema", type=float, default=0.80)
    parser.add_argument("--ecr-update-scope", choices=["sampled", "active"], default="sampled")
    parser.add_argument("--oscillation-window", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp_03_fast_gpu" / "results_movielens_1m")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Exp3Config(
        dataset=args.dataset,
        variants=tuple(args.variants),
        seed=args.seed,
        rounds=args.num_windows * args.rounds_per_window,
        eval_every=args.eval_every,
        participation_rate=args.participation_rate,
        clients_per_round=args.clients_per_round,
        microbatch_size=args.microbatch_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_users=args.max_eval_users,
        local_epochs=args.local_epochs,
        lr=args.lr,
        num_threads=args.num_threads,
        latent_dim=args.latent_dim,
        max_len=args.max_len,
        top_k=args.top_k,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
        gpfedrec_neighbors=args.gpfedrec_neighbors,
        gpfedrec_mix=args.gpfedrec_mix,
        gpfedrec_warmup_rounds=args.gpfedrec_warmup_rounds,
        fcucR_kb_len=args.fcucR_kb_len,
        fcucR_noise_std=args.fcucR_noise_std,
        fcucR_distill_weight=args.fcucR_distill_weight,
        tardgr_retrieval_mix=args.tardgr_retrieval_mix,
        tardgr_warmup_rounds=args.tardgr_warmup_rounds,
        f3crec_replay_topk=args.f3crec_replay_topk,
        f3crec_replay_mix=args.f3crec_replay_mix,
        f3crec_temporal_beta=args.f3crec_temporal_beta,
        f3crec_memory_momentum=args.f3crec_memory_momentum,
        f3crec_warmup_rounds=args.f3crec_warmup_rounds,
        dgfedrs_aug_ratio=args.dgfedrs_aug_ratio,
        dgfedrs_aug_mix=args.dgfedrs_aug_mix,
        dgfedrs_noise_std=args.dgfedrs_noise_std,
        dgfedrs_schedule_power=args.dgfedrs_schedule_power,
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
        feddieg_boundary_refresh_rate=args.feddieg_boundary_refresh_rate,
        feddieg_boundary_refresh_momentum=args.feddieg_boundary_refresh_momentum,
        num_windows=args.num_windows,
        rounds_per_window=args.rounds_per_window,
        candidate_count=args.candidate_count,
        window_mode=args.window_mode,
        lambda_temp=args.lambda_temp,
        lambda_struct=args.lambda_struct,
        edge_ema=args.edge_ema,
        ecr_update_scope=args.ecr_update_scope,
        oscillation_window=args.oscillation_window,
    )
    torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "exp03_fast_gpu.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    print(f"Output directory: {out_dir}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Config:", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    exp2_cfg = Exp2Config(**{field: getattr(cfg, field) for field in FastConfig.__dataclass_fields__})
    exp2_cfg.num_windows = cfg.num_windows
    exp2_cfg.rounds_per_window = cfg.rounds_per_window
    exp2_cfg.candidate_count = cfg.candidate_count
    exp2_cfg.window_mode = cfg.window_mode
    windows = [window.to(device) for window in load_rolling_windows(exp2_cfg)]

    variants = build_variant_configs(cfg)
    all_rounds: List[Dict[str, object]] = []
    all_windows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for variant_name in cfg.variants:
        if variant_name in variants:
            round_rows, window_rows, summary = train_variant(variants[variant_name], windows, cfg, device)
        else:
            round_rows, window_rows, summary = run_method_as_variant(variant_name, windows, cfg, device)
        all_rounds.extend(round_rows)
        all_windows.extend(window_rows)
        summaries.append(summary)
        write_csv(out_dir / "exp03_round_metrics.csv", all_rounds, ROUND_FIELDS)
        write_csv(out_dir / "exp03_window_metrics.csv", all_windows, WINDOW_FIELDS)
        write_csv(out_dir / "exp03_summary.csv", summaries, SUMMARY_FIELDS)
        print(f"[{variant_name}] outputs updated.")

    if not args.no_plot:
        make_plots(summaries, all_rounds, out_dir)
    print(f"\nDone. Experiment 3 outputs are in: {out_dir}")


if __name__ == "__main__":
    main()
