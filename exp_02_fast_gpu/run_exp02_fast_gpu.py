"""
GPU-batched Experiment 2: robustness over chronological windows.

This runner replaces the legacy per-client Exp2 scripts. It uses one aligned
protocol for every method: the same chronological windows, client sampling,
negative sampling, candidate evaluation, SASRec backbone, and GPU microbatches.
Window metrics are produced directly from the run; no report-facing value
alignment or post-hoc offset is applied.
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_01_fast_gpu.run_exp01_fast_gpu import (  # noqa: E402
    ALL_METHODS,
    FCUCRModel,
    DGFedRSModel,
    F3CRecModel,
    FastConfig,
    FastData,
    FedDIEGModel,
    FedMFModel,
    FedSASRecModel,
    MEMORY_METHODS,
    TarDGRModel,
    Tee,
    apply_f3crec_replay,
    augment_sequence_for_dgfedrs,
    batched_indices,
    bce_seq_loss,
    build_retrieved_memory,
    build_weighted_neighbors,
    init_weights,
    masked_sequence_mean,
    sample_negative_like,
    set_seed,
    update_f3crec_item_temporal_mean,
    update_memory_bank,
)


@dataclass
class Exp2Config(FastConfig):
    methods: Tuple[str, ...] = ALL_METHODS
    num_windows: int = 5
    rounds_per_window: int = 120
    candidate_count: int = 99
    window_mode: str = "slice"


ROUND_FIELDS = [
    "method",
    "dataset",
    "window",
    "global_round",
    "window_round",
    "train_loss",
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
    "method",
    "dataset",
    "window",
    "eval_users",
    "last_global_round",
    "last_window_round",
    "last_val_hr",
    "last_val_ndcg",
    "last_val_recall",
    "last_val_auc",
    "last_test_hr",
    "last_test_ndcg",
    "last_test_recall",
    "last_test_auc",
    "initial_val_hr",
    "initial_val_ndcg",
    "initial_val_recall",
    "initial_val_auc",
    "initial_test_hr",
    "initial_test_ndcg",
    "initial_test_recall",
    "initial_test_auc",
    "best_val_ndcg",
    "best_test_hr_at_best_val",
    "best_test_ndcg_at_best_val",
    "best_test_recall_at_best_val",
    "best_test_auc_at_best_val",
    "ecr_from_prev_window",
    "boundary_ecr_from_prev_window",
    "boundary_drop_from_prev_window",
    "adaptation_gain",
]

SUMMARY_FIELDS = [
    "method",
    "dataset",
    "seed",
    "num_windows",
    "rounds_per_window",
    "total_rounds",
    "clients_per_round",
    "microbatch_size",
    "lr",
    "avg_window_ndcg",
    "final_window_ndcg",
    "best_window_ndcg",
    "max_drop",
    "max_boundary_drop",
    "average_boundary_drop",
    "window_variance",
    "average_ecr",
    "average_boundary_ecr",
    "average_adaptation_gain",
    "avg_window_hr",
    "final_window_hr",
    "total_time_seconds",
]


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: List[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def compute_max_drop(values: List[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_drop = 0.0
    for value in values:
        peak = max(peak, value)
        max_drop = max(max_drop, peak - value)
    return max_drop


def build_train_arrays_from_sequences(
    user_sequences: Dict[int, List[int]],
    num_users: int,
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_seq = np.zeros((num_users, max_len), dtype=np.int64)
    train_pos = np.zeros((num_users, max_len), dtype=np.int64)
    for user, raw_items in user_sequences.items():
        items = np.asarray(raw_items, dtype=np.int64) + 1
        if len(items) < 2:
            continue
        nxt = int(items[-1])
        idx_pos = max_len - 1
        for item in reversed(items[:-1]):
            train_seq[int(user), idx_pos] = int(item)
            train_pos[int(user), idx_pos] = nxt
            nxt = int(item)
            idx_pos -= 1
            if idx_pos < 0:
                break
    return train_seq, train_pos


def sample_candidates(
    seen: set[int],
    num_items: int,
    count: int,
    rng: random.Random,
) -> List[int]:
    candidates: set[int] = set()
    while len(candidates) < count:
        item = rng.randrange(num_items)
        if item not in seen:
            candidates.add(item)
    return list(candidates)


def build_eval_arrays_from_records(
    records: List[Tuple[int, int]],
    histories: Dict[int, List[int]],
    candidates: Dict[int, List[int]],
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    users: List[int] = []
    seqs: List[np.ndarray] = []
    item_rows: List[List[int]] = []
    for user, target_item in records:
        seq = np.zeros(max_len, dtype=np.int64)
        idx_pos = max_len - 1
        for item in reversed(histories[user]):
            seq[idx_pos] = int(item) + 1
            idx_pos -= 1
            if idx_pos < 0:
                break
        users.append(user)
        seqs.append(seq)
        item_rows.append([int(target_item) + 1] + [int(item) + 1 for item in candidates[user]])
    if not users:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, max_len), dtype=np.int64),
            np.zeros((0, 1), dtype=np.int64),
        )
    return (
        np.asarray(users, dtype=np.int64),
        np.stack(seqs).astype(np.int64),
        np.asarray(item_rows, dtype=np.int64),
    )


def load_rolling_windows(cfg: Exp2Config) -> List[FastData]:
    processed_dir = ROOT / "datasets" / "processed" / cfg.dataset
    splits_dir = ROOT / "datasets" / "splits" / cfg.dataset
    with (processed_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])

    frames = [
        pd.read_csv(splits_dir / "loo_train.csv"),
        pd.read_csv(splits_dir / "loo_val.csv"),
        pd.read_csv(splits_dir / "loo_test.csv"),
    ]
    full_df = pd.concat(frames, ignore_index=True)
    if "timestamp" in full_df.columns:
        full_df = full_df.sort_values(["timestamp", "userId"]).reset_index(drop=True)
    else:
        full_df = full_df.reset_index(drop=True)
        full_df["timestamp"] = full_df.index

    windows: List[FastData] = []
    total_interactions = len(full_df)
    rng = random.Random(cfg.seed + 202)
    if cfg.window_mode not in {"slice", "cumulative"}:
        raise ValueError(f"Unsupported window_mode={cfg.window_mode}. Use 'slice' or 'cumulative'.")
    print(
        f"Building {cfg.num_windows} chronological {cfg.window_mode} windows for {cfg.dataset}: "
        f"users={num_users}, items={num_items}, interactions={total_interactions}"
    )

    for window_idx in range(1, cfg.num_windows + 1):
        start_idx = int(total_interactions * (window_idx - 1) / cfg.num_windows)
        end_idx = int(total_interactions * window_idx / cfg.num_windows)
        history_df = full_df.iloc[:end_idx]
        eval_df = history_df if cfg.window_mode == "cumulative" else full_df.iloc[start_idx:end_idx]
        history_grouped = history_df.groupby("userId", sort=False)["itemId"].apply(
            lambda x: list(x.to_numpy(dtype=np.int64))
        ).to_dict()
        eval_grouped = eval_df.groupby("userId", sort=False)["itemId"].apply(
            lambda x: list(x.to_numpy(dtype=np.int64))
        ).to_dict()

        train_sequences: Dict[int, List[int]] = {}
        val_records: List[Tuple[int, int]] = []
        test_records: List[Tuple[int, int]] = []
        val_histories: Dict[int, List[int]] = {}
        test_histories: Dict[int, List[int]] = {}
        val_candidates: Dict[int, List[int]] = {}
        test_candidates: Dict[int, List[int]] = {}

        for user in range(num_users):
            history_seq = history_grouped.get(user, [])
            eval_seq = eval_grouped.get(user, [])
            if len(history_seq) < 3:
                train_sequences[user] = history_seq
                continue
            if cfg.window_mode == "slice":
                if len(eval_seq) < 2:
                    train_sequences[user] = history_seq[:-1]
                    continue
                val_item = int(eval_seq[-2])
                test_item = int(eval_seq[-1])
                val_pos = len(history_seq) - 2
                test_pos = len(history_seq) - 1
                if val_pos < 1 or test_pos < 2:
                    train_sequences[user] = history_seq[:-1]
                    continue
                train_sequences[user] = history_seq[:val_pos]
                val_histories[user] = history_seq[:val_pos]
                test_histories[user] = history_seq[:test_pos]
            else:
                train_sequences[user] = history_seq[:-2]
                val_histories[user] = history_seq[:-2]
                test_histories[user] = history_seq[:-1]
                val_item = int(history_seq[-2])
                test_item = int(history_seq[-1])
            val_records.append((user, val_item))
            test_records.append((user, test_item))
            val_candidates[user] = sample_candidates(set(val_histories[user]) | {val_item}, num_items, cfg.candidate_count, rng)
            test_candidates[user] = sample_candidates(set(test_histories[user]) | {test_item}, num_items, cfg.candidate_count, rng)

        train_seq, train_pos = build_train_arrays_from_sequences(train_sequences, num_users, cfg.max_len)
        val_users, val_seq, val_items = build_eval_arrays_from_records(
            val_records, val_histories, val_candidates, cfg.max_len
        )
        test_users, test_seq, test_items = build_eval_arrays_from_records(
            test_records, test_histories, test_candidates, cfg.max_len
        )
        data = FastData(
            num_users=num_users,
            num_items=num_items,
            train_seq=torch.from_numpy(train_seq),
            train_pos=torch.from_numpy(train_pos),
            val_users=torch.from_numpy(val_users),
            val_seq=torch.from_numpy(val_seq),
            val_items=torch.from_numpy(val_items),
            test_users=torch.from_numpy(test_users),
            test_seq=torch.from_numpy(test_seq),
            test_items=torch.from_numpy(test_items),
        )
        data.window_idx = window_idx
        data.window_start_interactions = start_idx
        data.window_end_interactions = end_idx
        windows.append(data)
        print(
            f"  W{window_idx}: interactions={start_idx}..{end_idx}, "
            f"eval_users={data.test_users.numel()}, train_shape={tuple(data.train_seq.shape)}"
        )

    return windows


@torch.no_grad()
def evaluate_model_with_auc(
    method: str,
    model: nn.Module,
    users: torch.Tensor,
    seqs: torch.Tensor,
    items: torch.Tensor,
    data: FastData,
    cfg: Exp2Config,
    memory_bank: Optional[torch.Tensor] = None,
    active_clients: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total_recall = 0.0
    total_auc = 0.0
    total = 0
    if users.numel() == 0:
        model.train()
        return 0.0, 0.0, 0.0, 0.0
    if cfg.max_eval_users is not None:
        users = users[: cfg.max_eval_users]
        seqs = seqs[: cfg.max_eval_users]
        items = items[: cfg.max_eval_users]

    norm_bank = None
    uses_memory = method in MEMORY_METHODS
    if uses_memory and memory_bank is not None and active_clients is not None and active_clients.sum().item() > 1:
        norm_bank = F.normalize(memory_bank, p=2, dim=1)

    for batch_idx in batched_indices(torch.arange(users.numel(), device=users.device), cfg.eval_batch_size):
        batch_users = users[batch_idx]
        batch_seqs = seqs[batch_idx]
        batch_items = items[batch_idx]
        short_summary = None
        base_score_feats = None
        if method == "FedMF":
            feats = model(batch_users, batch_seqs)
        elif method == "GPFedRec" and norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(
                batch_users, memory_bank, active_clients, norm_bank, cfg.gpfedrec_neighbors
            )
            local_feats = model(batch_seqs)
            if retrieved is not None and retrieval_mask is not None:
                neighbor_feat = retrieved.mean(dim=1).unsqueeze(1).expand_as(local_feats)
                feats = local_feats + cfg.gpfedrec_mix * retrieval_mask.float().view(-1, 1, 1) * neighbor_feat
            else:
                feats = local_feats
        elif method == "FCUCR" and norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(
                batch_users, memory_bank, active_clients, norm_bank, cfg.fcucR_kb_len
            )
            kb_feats = None
            if retrieved is not None and retrieval_mask is not None:
                kb_feats = retrieved.mean(dim=1) * retrieval_mask.float().unsqueeze(-1)
            feats, _ = model(batch_seqs, kb_feats)
        elif method == "TarDGR" and norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(
                batch_users, memory_bank, active_clients, norm_bank, cfg.tardgr_retrieval_topk
            )
            feats, _ = model(batch_seqs, retrieved, retrieval_mask)
        elif method == "F3CRec" and norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(
                batch_users, memory_bank, active_clients, norm_bank, cfg.f3crec_replay_topk
            )
            local_feats = model(batch_seqs)
            feats = apply_f3crec_replay(
                batch_users,
                batch_seqs,
                local_feats,
                retrieved,
                retrieval_mask,
                memory_bank,
                active_clients,
                cfg.f3crec_replay_mix,
            )
        elif method == "FedDIEG" and norm_bank is not None:
            base_feats, _ = model(batch_seqs)
            base_score_feats = base_feats[:, -1, :]
            neighbors, weights, graph_mask = build_weighted_neighbors(
                batch_users,
                memory_bank,
                active_clients,
                norm_bank,
                cfg.feddieg_neighbors,
                cfg.feddieg_neighbor_temp,
                cfg.feddieg_min_sim,
            )
            feats, short_summary = model(batch_seqs, neighbors, weights, graph_mask)
        else:
            output = model(batch_seqs)
            if method == "FedDIEG" and isinstance(output, tuple):
                feats, short_summary = output
            else:
                feats = output[0] if isinstance(output, tuple) else output

        score_feats = feats[:, -1, :]
        if method == "FedDIEG" and short_summary is not None and hasattr(model, "score_representation"):
            score_feats = model.score_representation(score_feats, short_summary, base_score_feats)
        scores = torch.matmul(score_feats.unsqueeze(1), model.item_emb(batch_items).transpose(-1, -2)).squeeze(1)

        _, top_indices = torch.topk(scores, cfg.top_k, dim=1)
        hits = top_indices == 0
        found = hits.any(dim=1)
        if found.any():
            ranks = hits.float().argmax(dim=1)
            ndcg = torch.where(found, math.log(2.0) / torch.log(ranks.float() + 2.0), torch.zeros_like(ranks.float()))
        else:
            ndcg = torch.zeros(batch_users.size(0), device=users.device)
        pos_scores = scores[:, 0].unsqueeze(1)
        neg_scores = scores[:, 1:]
        auc = ((pos_scores > neg_scores).float() + 0.5 * (pos_scores == neg_scores).float()).mean(dim=1)

        total_hr += found.float().sum().item()
        total_ndcg += ndcg.sum().item()
        total_recall += found.float().sum().item()
        total_auc += auc.sum().item()
        total += batch_users.size(0)

    model.train()
    return total_hr / total, total_ndcg / total, total_recall / total, total_auc / total


def make_model(method: str, num_users: int, num_items: int, cfg: Exp2Config, device: torch.device) -> nn.Module:
    if method == "FedMF":
        return FedMFModel(num_users, num_items, cfg).to(device)
    if method == "FedSASRec":
        return FedSASRecModel(num_items, cfg).to(device)
    if method == "GPFedRec":
        return FedSASRecModel(num_items, cfg).to(device)
    if method == "FCUCR":
        return FCUCRModel(num_items, cfg).to(device)
    if method == "TarDGR":
        return TarDGRModel(num_items, cfg).to(device)
    if method == "F3CRec":
        return F3CRecModel(num_items, cfg).to(device)
    if method == "DGFedRS":
        return DGFedRSModel(num_items, cfg).to(device)
    if method == "FedDIEG":
        return FedDIEGModel(num_items, cfg).to(device)
    raise ValueError(f"Unsupported method: {method}")


@torch.no_grad()
def refresh_feddieg_boundary_memory(
    model: nn.Module,
    data: FastData,
    cfg: Exp2Config,
    memory_bank: torch.Tensor,
    active_clients: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    rate = min(max(float(cfg.feddieg_boundary_refresh_rate), 0.0), 1.0)
    if rate <= 0:
        return 0
    eligible = torch.nonzero(data.train_mask.sum(dim=1) > 0, as_tuple=False).flatten()
    if eligible.numel() == 0:
        return 0
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
    return int(selected.numel())


def run_method_over_windows(
    method: str,
    windows: List[FastData],
    cfg: Exp2Config,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    print(f"\n[{method}] Exp2 fast GPU start")
    set_seed(cfg.seed)
    model = make_model(method, windows[0].num_users, windows[0].num_items, cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
    clients_per_round = cfg.clients_per_round or max(int(cfg.participation_rate * windows[0].num_users), 1)
    clients_per_round = min(clients_per_round, windows[0].num_users)
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    memory_bank = torch.zeros((windows[0].num_users, cfg.latent_dim), device=device)
    active_clients = torch.zeros(windows[0].num_users, dtype=torch.bool, device=device)

    round_records: List[Dict[str, object]] = []
    window_records: List[Dict[str, object]] = []
    start_time = time.time()
    global_round = 0
    prev_window_ndcg: Optional[float] = None
    prev_last_ndcg: Optional[float] = None

    for window_idx, data in enumerate(windows, start=1):
        print(f"\n[{method}] Window {window_idx}/{cfg.num_windows} | eval_users={data.test_users.numel()}")
        if method == "FedDIEG" and window_idx > 1:
            refreshed = refresh_feddieg_boundary_memory(model, data, cfg, memory_bank, active_clients, generator, device)
            if refreshed > 0:
                print(f"[{method}] W{window_idx} boundary memory refresh | clients={refreshed}")
        best_val_ndcg = -1.0
        best_test = (0.0, 0.0, 0.0, 0.0)
        last_eval_record: Optional[Dict[str, object]] = None
        last_eval_time = time.time()
        initial_val_metrics = evaluate_model_with_auc(
            method, model, data.val_users, data.val_seq, data.val_items, data, cfg, memory_bank, active_clients
        )
        initial_test_metrics = evaluate_model_with_auc(
            method, model, data.test_users, data.test_seq, data.test_items, data, cfg, memory_bank, active_clients
        )
        boundary_ecr = 0.0 if prev_last_ndcg is None else abs(initial_test_metrics[1] - prev_last_ndcg)
        boundary_drop = 0.0 if prev_last_ndcg is None else max(prev_last_ndcg - initial_test_metrics[1], 0.0)
        print(
            f"[{method}] W{window_idx} initial | "
            f"Val NDCG {initial_val_metrics[1]:.4f} | Test NDCG {initial_test_metrics[1]:.4f} | "
            f"BoundaryDrop {boundary_drop:.4f}"
        )

        for window_round in range(1, cfg.rounds_per_window + 1):
            global_round += 1
            round_start = time.time()
            sampled = torch.randperm(data.num_users, generator=generator, device=device)[:clients_per_round]
            use_memory = (
                method == "GPFedRec" and global_round > cfg.gpfedrec_warmup_rounds and active_clients.sum().item() > 1
            ) or (
                method == "TarDGR" and global_round > cfg.tardgr_warmup_rounds and active_clients.sum().item() > 1
            ) or (
                method == "F3CRec" and global_round > cfg.f3crec_warmup_rounds and active_clients.sum().item() > 1
            ) or (
                method == "FedDIEG" and global_round > cfg.feddieg_warmup_rounds and active_clients.sum().item() > 1
            ) or (
                method == "FCUCR" and active_clients.sum().item() > 1
            )
            norm_bank = F.normalize(memory_bank, p=2, dim=1) if use_memory else None
            losses: List[float] = []
            model.train()
            for _ in range(cfg.local_epochs):
                shuffled = sampled[torch.randperm(sampled.numel(), generator=generator, device=device)]
                for users in batched_indices(shuffled, cfg.microbatch_size):
                    seq = data.train_seq[users]
                    pos = data.train_pos[users]
                    mask = data.train_mask[users]
                    neg = sample_negative_like(pos, data.num_items)
                    optimizer.zero_grad(set_to_none=True)
                    if method == "FedMF":
                        feats = model(users, seq)
                        local_feats = feats
                    elif method == "GPFedRec":
                        local_feats = model(seq)
                        feats = local_feats
                        if use_memory and norm_bank is not None:
                            retrieved, retrieval_mask = build_retrieved_memory(
                                users, memory_bank, active_clients, norm_bank, cfg.gpfedrec_neighbors
                            )
                            if retrieved is not None and retrieval_mask is not None:
                                neighbor_feat = retrieved.mean(dim=1).unsqueeze(1).expand_as(local_feats)
                                feats = local_feats + cfg.gpfedrec_mix * retrieval_mask.float().view(-1, 1, 1) * neighbor_feat
                    elif method == "FCUCR":
                        kb_feats = None
                        if use_memory and norm_bank is not None:
                            retrieved, retrieval_mask = build_retrieved_memory(
                                users, memory_bank, active_clients, norm_bank, cfg.fcucR_kb_len
                            )
                            if retrieved is not None and retrieval_mask is not None:
                                kb_feats = retrieved.mean(dim=1) * retrieval_mask.float().unsqueeze(-1)
                                if cfg.fcucR_noise_std > 0:
                                    kb_feats = kb_feats + torch.randn_like(kb_feats) * cfg.fcucR_noise_std
                        feats, local_feats = model(seq, kb_feats)
                    elif method == "TarDGR":
                        retrieved = retrieval_mask = None
                        if use_memory and norm_bank is not None:
                            retrieved, retrieval_mask = build_retrieved_memory(
                                users, memory_bank, active_clients, norm_bank, cfg.tardgr_retrieval_topk
                            )
                        feats, local_feats = model(seq, retrieved, retrieval_mask)
                    elif method == "F3CRec":
                        local_feats = model(seq)
                        feats = local_feats
                        if use_memory and norm_bank is not None:
                            retrieved, retrieval_mask = build_retrieved_memory(
                                users, memory_bank, active_clients, norm_bank, cfg.f3crec_replay_topk
                            )
                            feats = apply_f3crec_replay(
                                users,
                                seq,
                                local_feats,
                                retrieved,
                                retrieval_mask,
                                memory_bank,
                                active_clients,
                                cfg.f3crec_replay_mix,
                            )
                    elif method == "DGFedRS":
                        augmented_seq = augment_sequence_for_dgfedrs(
                            seq,
                            data.num_items,
                            cfg.dgfedrs_aug_ratio,
                            generator,
                            cfg.dgfedrs_schedule_power,
                        )
                        feats, local_feats = model.forward_augmented(
                            seq,
                            augmented_seq,
                            cfg.dgfedrs_aug_mix,
                            cfg.dgfedrs_noise_std,
                        )
                    elif method == "FedDIEG":
                        neighbors = weights = graph_mask = None
                        if use_memory and norm_bank is not None:
                            neighbors, weights, graph_mask = build_weighted_neighbors(
                                users,
                                memory_bank,
                                active_clients,
                                norm_bank,
                                cfg.feddieg_neighbors,
                                cfg.feddieg_neighbor_temp,
                                cfg.feddieg_min_sim,
                            )
                        feats, local_feats = model(seq, neighbors, weights, graph_mask)
                    else:
                        feats = model(seq)
                        local_feats = feats

                    pos_logits = (feats * model.item_emb(pos)).sum(dim=-1)
                    neg_logits = (feats * model.item_emb(neg)).sum(dim=-1)
                    loss = bce_seq_loss(pos_logits, neg_logits, mask)
                    if method == "FedDIEG" and cfg.feddieg_final_loss_weight > 0:
                        lengths = mask.sum(dim=1).long()
                        has_target = lengths > 0
                        if has_target.any():
                            row_ids = torch.arange(users.size(0), device=device)[has_target]
                            last_positions = (lengths[has_target] - 1).clamp_min(0)
                            final_feats = feats[row_ids, last_positions]
                            if local_feats.dim() == 2 and hasattr(model, "score_representation"):
                                final_feats = model.score_representation(final_feats, local_feats[row_ids])
                            final_pos = pos[row_ids, last_positions]
                            final_neg = neg[row_ids, last_positions]
                            final_pos_logits = (final_feats * model.item_emb(final_pos)).sum(dim=-1)
                            final_neg_logits = (final_feats * model.item_emb(final_neg)).sum(dim=-1)
                            final_loss = F.binary_cross_entropy_with_logits(
                                final_pos_logits, torch.ones_like(final_pos_logits)
                            ) + F.binary_cross_entropy_with_logits(
                                final_neg_logits, torch.zeros_like(final_neg_logits)
                            )
                            loss = loss + cfg.feddieg_final_loss_weight * final_loss
                    if method == "FCUCR" and cfg.fcucR_distill_weight > 0:
                        loss = loss + cfg.fcucR_distill_weight * F.mse_loss(feats, local_feats.detach())

                    loss.backward()
                    optimizer.step()
                    if method == "F3CRec":
                        update_f3crec_item_temporal_mean(model, seq, pos, neg, cfg.f3crec_temporal_beta)
                    losses.append(loss.item())
                    if method in MEMORY_METHODS:
                        if local_feats.dim() == 2:
                            new_memory = local_feats.detach()
                        else:
                            new_memory = masked_sequence_mean(local_feats.detach(), seq)
                        update_memory_bank(method, users, new_memory, memory_bank, active_clients, cfg)

            avg_loss = float(np.mean(losses)) if losses else 0.0
            if window_round % cfg.eval_every == 0:
                val_metrics = evaluate_model_with_auc(
                    method, model, data.val_users, data.val_seq, data.val_items, data, cfg, memory_bank, active_clients
                )
                test_metrics = evaluate_model_with_auc(
                    method,
                    model,
                    data.test_users,
                    data.test_seq,
                    data.test_items,
                    data,
                    cfg,
                    memory_bank,
                    active_clients,
                )
                seconds = time.time() - last_eval_time
                last_eval_time = time.time()
                record = {
                    "method": method,
                    "dataset": cfg.dataset,
                    "window": window_idx,
                    "global_round": global_round,
                    "window_round": window_round,
                    "train_loss": avg_loss,
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
                round_records.append(record)
                last_eval_record = record
                if val_metrics[1] > best_val_ndcg:
                    best_val_ndcg = val_metrics[1]
                    best_test = test_metrics
                print(
                    f"[{method}] W{window_idx} R{window_round:03d}/{cfg.rounds_per_window} "
                    f"G{global_round:03d} | Loss {avg_loss:.4f} | "
                    f"Val NDCG {val_metrics[1]:.4f} | Test NDCG {test_metrics[1]:.4f} | "
                    f"{time.time() - round_start:.2f}s"
                )

        if last_eval_record is None:
            raise RuntimeError(f"No evaluation record was produced for {method} window {window_idx}.")
        current_ndcg = float(last_eval_record["test_ndcg"])
        ecr = 0.0 if prev_window_ndcg is None else abs(current_ndcg - prev_window_ndcg)
        prev_window_ndcg = current_ndcg
        adaptation_gain = current_ndcg - float(initial_test_metrics[1])
        window_records.append(
            {
                "method": method,
                "dataset": cfg.dataset,
                "window": window_idx,
                "eval_users": data.test_users.numel(),
                "last_global_round": last_eval_record["global_round"],
                "last_window_round": last_eval_record["window_round"],
                "last_val_hr": last_eval_record["val_hr"],
                "last_val_ndcg": last_eval_record["val_ndcg"],
                "last_val_recall": last_eval_record["val_recall"],
                "last_val_auc": last_eval_record["val_auc"],
                "last_test_hr": last_eval_record["test_hr"],
                "last_test_ndcg": last_eval_record["test_ndcg"],
                "last_test_recall": last_eval_record["test_recall"],
                "last_test_auc": last_eval_record["test_auc"],
                "initial_val_hr": initial_val_metrics[0],
                "initial_val_ndcg": initial_val_metrics[1],
                "initial_val_recall": initial_val_metrics[2],
                "initial_val_auc": initial_val_metrics[3],
                "initial_test_hr": initial_test_metrics[0],
                "initial_test_ndcg": initial_test_metrics[1],
                "initial_test_recall": initial_test_metrics[2],
                "initial_test_auc": initial_test_metrics[3],
                "best_val_ndcg": best_val_ndcg,
                "best_test_hr_at_best_val": best_test[0],
                "best_test_ndcg_at_best_val": best_test[1],
                "best_test_recall_at_best_val": best_test[2],
                "best_test_auc_at_best_val": best_test[3],
                "ecr_from_prev_window": ecr,
                "boundary_ecr_from_prev_window": boundary_ecr,
                "boundary_drop_from_prev_window": boundary_drop,
                "adaptation_gain": adaptation_gain,
            }
        )
        prev_last_ndcg = current_ndcg

    ndcgs = [float(row["last_test_ndcg"]) for row in window_records]
    hrs = [float(row["last_test_hr"]) for row in window_records]
    ecrs = [float(row["ecr_from_prev_window"]) for row in window_records[1:]]
    boundary_drops = [float(row["boundary_drop_from_prev_window"]) for row in window_records[1:]]
    boundary_ecrs = [float(row["boundary_ecr_from_prev_window"]) for row in window_records[1:]]
    adaptation_gains = [float(row["adaptation_gain"]) for row in window_records]
    summary = {
        "method": method,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "num_windows": cfg.num_windows,
        "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": cfg.num_windows * cfg.rounds_per_window,
        "clients_per_round": clients_per_round,
        "microbatch_size": cfg.microbatch_size,
        "lr": cfg.lr,
        "avg_window_ndcg": mean(ndcgs),
        "final_window_ndcg": ndcgs[-1],
        "best_window_ndcg": max(ndcgs),
        "max_drop": compute_max_drop(ndcgs),
        "max_boundary_drop": max(boundary_drops) if boundary_drops else 0.0,
        "average_boundary_drop": mean(boundary_drops),
        "window_variance": variance(ndcgs),
        "average_ecr": mean(ecrs),
        "average_boundary_ecr": mean(boundary_ecrs),
        "average_adaptation_gain": mean(adaptation_gains),
        "avg_window_hr": mean(hrs),
        "final_window_hr": hrs[-1],
        "total_time_seconds": time.time() - start_time,
    }
    return round_records, window_records, summary


def plot_window_metrics(window_rows: List[Dict[str, object]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
        print(f"Plot skipped: {exc}")
        return

    order = list(ALL_METHODS)
    labels = {
        "FedMF": "FedMF",
        "FedSASRec": "FedSASRec",
        "GPFedRec": "GPFedRec",
        "FCUCR": "FCUCR",
        "TarDGR": "TarDGR",
        "F3CRec": "F3CRec",
        "DGFedRS": "DGFedRS",
        "FedDIEG": "Ours",
    }
    by_method: Dict[str, List[Tuple[int, float]]] = {}
    for row in window_rows:
        by_method.setdefault(str(row["method"]), []).append((int(row["window"]), float(row["last_test_ndcg"])))

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in order:
        if method not in by_method:
            continue
        points = sorted(by_method[method])
        ax.plot(
            [f"W{idx}" for idx, _ in points],
            [score for _, score in points],
            marker="o" if method != "FedDIEG" else "*",
            linewidth=2.0 if method != "FedDIEG" else 2.8,
            markersize=6 if method != "FedDIEG" else 11,
            label=labels.get(method, method),
        )
    ax.set_title("Experiment 2: Robustness over Chronological Windows")
    ax.set_xlabel("Chronological Window")
    ax.set_ylabel("NDCG@10")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast GPU-batched Experiment 2 runner.")
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS),
        choices=list(ALL_METHODS),
    )
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--rounds-per-window", type=int, default=120)
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
    parser.add_argument("--feddieg-neighbors", type=int, default=20)
    parser.add_argument("--feddieg-graph-mix", type=float, default=0.2)
    parser.add_argument("--feddieg-warmup-rounds", type=int, default=20)
    parser.add_argument("--feddieg-neighbor-temp", type=float, default=0.05)
    parser.add_argument("--feddieg-min-sim", type=float, default=0.10)
    parser.add_argument("--feddieg-tail-power", type=float, default=2.0)
    parser.add_argument("--feddieg-final-loss-weight", type=float, default=0.2)
    parser.add_argument("--feddieg-memory-momentum", type=float, default=0.5)
    parser.add_argument("--feddieg-short-score-mix", type=float, default=0.3)
    parser.add_argument("--feddieg-confidence-power", type=float, default=0.0)
    parser.add_argument("--feddieg-agreement-power", type=float, default=0.0)
    parser.add_argument("--feddieg-score-residual-mix", type=float, default=0.0)
    parser.add_argument("--feddieg-score-local-mix", type=float, default=0.0)
    parser.add_argument("--feddieg-boundary-refresh-rate", type=float, default=0.0)
    parser.add_argument("--feddieg-boundary-refresh-momentum", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp_02_fast_gpu" / "results_movielens_1m")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Exp2Config(
        dataset=args.dataset,
        methods=tuple(args.methods),
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
        feddieg_warmup_rounds=args.feddieg_warmup_rounds,
        feddieg_neighbor_temp=args.feddieg_neighbor_temp,
        feddieg_min_sim=args.feddieg_min_sim,
        feddieg_tail_power=args.feddieg_tail_power,
        feddieg_final_loss_weight=args.feddieg_final_loss_weight,
        feddieg_memory_momentum=args.feddieg_memory_momentum,
        feddieg_short_score_mix=args.feddieg_short_score_mix,
        feddieg_confidence_power=args.feddieg_confidence_power,
        feddieg_agreement_power=args.feddieg_agreement_power,
        feddieg_score_residual_mix=args.feddieg_score_residual_mix,
        feddieg_score_local_mix=args.feddieg_score_local_mix,
        feddieg_boundary_refresh_rate=args.feddieg_boundary_refresh_rate,
        feddieg_boundary_refresh_momentum=args.feddieg_boundary_refresh_momentum,
        num_windows=args.num_windows,
        rounds_per_window=args.rounds_per_window,
        candidate_count=args.candidate_count,
        window_mode=args.window_mode,
    )
    torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "exp02_fast_gpu.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {json.dumps(asdict(cfg), ensure_ascii=False, indent=2)}")
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2)

    windows = [window.to(device) for window in load_rolling_windows(cfg)]

    all_round_records: List[Dict[str, object]] = []
    all_window_records: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for method in cfg.methods:
        round_records, window_records, summary = run_method_over_windows(method, windows, cfg, device)
        all_round_records.extend(round_records)
        all_window_records.extend(window_records)
        summaries.append(summary)
        write_csv(out_dir / "exp02_round_metrics.csv", all_round_records, ROUND_FIELDS)
        write_csv(out_dir / "exp02_window_metrics.csv", all_window_records, WINDOW_FIELDS)
        write_csv(out_dir / "exp02_summary.csv", summaries, SUMMARY_FIELDS)
        print(f"[{method}] outputs updated.")

    if not args.no_plot:
        plot_window_metrics(all_window_records, out_dir / "exp02_window_ndcg.png")
    print(f"\nDone. Final outputs are in: {out_dir}")


if __name__ == "__main__":
    main()
