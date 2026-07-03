"""
Fast GPU Experiment 4: prototype semantic anchoring for new clients.

The legacy exp_04 scripts mixed real FedDIEG training with report-shaped
baseline curves.  This runner keeps the report protocol but trains every method
under the same cold-start split on GPU:

1. Hold out a fixed ratio of users as unseen new clients.
2. Pretrain on old clients only.
3. Inject new clients with only d_init visible interactions.
4. Track NDCG@10/HR@10 over local adaptation rounds 0..K.

Outputs are written as CSVs that can directly support the new-client
pre-validation of the prototype semantic anchoring mechanism:
trajectory.csv, summary.csv, ablation_trajectory.csv, ablation_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[1]

BASELINE_METHODS = ("FedMF", "FCUCR", "F3CRec", "DGFedRS")
DEFAULT_METHODS = BASELINE_METHODS + ("FedDIEG",)
EXTRA_SUPPORTED_METHODS = ("FedSASRec", "GPFedRec", "TarDGR")
SUPPORTED_METHODS = DEFAULT_METHODS + EXTRA_SUPPORTED_METHODS
MEMORY_METHODS = {"GPFedRec", "FCUCR", "TarDGR", "F3CRec", "FedDIEG"}


@dataclass
class Exp4Config:
    dataset: str = "movielens-1m"
    methods: Tuple[str, ...] = DEFAULT_METHODS
    seed: int = 42
    pretrain_rounds: int = 50
    adapt_rounds: int = 10
    new_client_ratio: float = 0.25
    new_user_strategy: str = "low_history"
    min_new_user_history: int = 5
    d_init: int = 3
    new_interactions_per_round: int = 2
    participation_rate: float = 0.1
    clients_per_round: Optional[int] = None
    adapt_clients_per_round: Optional[int] = None
    local_epochs: int = 1
    adapt_epochs: int = 1
    lr: float = 0.001
    adapt_lr: float = 0.0002
    microbatch_size: int = 512
    eval_batch_size: int = 1024
    max_eval_users: Optional[int] = None
    latent_dim: int = 64
    max_len: int = 50
    top_k: int = 10
    num_heads: int = 4
    num_blocks: int = 2
    dropout_rate: float = 0.2
    gpfedrec_neighbors: int = 8
    gpfedrec_mix: float = 0.12
    gpfedrec_warmup_rounds: int = 10
    fcucR_kb_len: int = 5
    fcucR_noise_std: float = 0.01
    fcucR_distill_weight: float = 0.1
    tardgr_retrieval_topk: int = 10
    tardgr_retrieval_temp: float = 0.1
    tardgr_retrieval_mix: float = 0.2
    tardgr_warmup_rounds: int = 10
    f3crec_replay_topk: int = 8
    f3crec_replay_mix: float = 0.15
    f3crec_temporal_beta: float = 0.15
    f3crec_memory_momentum: float = 0.60
    f3crec_warmup_rounds: int = 10
    dgfedrs_aug_ratio: float = 0.15
    dgfedrs_aug_mix: float = 0.30
    dgfedrs_noise_std: float = 0.02
    dgfedrs_schedule_power: float = 1.0
    feddieg_neighbors: int = 20
    feddieg_graph_mix: float = 0.30
    feddieg_neighbor_temp: float = 0.05
    feddieg_min_sim: float = 0.05
    feddieg_tail_power: float = 1.8
    feddieg_final_loss_weight: float = 0.15
    feddieg_memory_momentum: float = 0.45
    feddieg_short_score_mix: float = 0.20
    feddieg_disable_short_attention: bool = False
    feddieg_proto_topk: int = 8
    feddieg_proto_state_mix: float = 0.35
    feddieg_proto_memory_mix: float = 0.75
    feddieg_proto_edge_mix: float = 0.45
    feddieg_proto_score_mix: float = 0.55
    feddieg_ablation_proto_score_mix: float = 0.0
    num_threads: int = 4


TRAJECTORY_FIELDS = [
    "method",
    "dataset",
    "seed",
    "local_round",
    "ndcg_at_10",
    "hr_at_10",
    "visible_interactions",
    "eval_users",
]

SUMMARY_FIELDS = [
    "method",
    "dataset",
    "seed",
    "initial_ndcg",
    "initial_hr",
    "final_ndcg",
    "final_hr",
    "best_ndcg",
    "best_round",
    "growth_rate",
    "early_growth_rate",
    "relative_gain",
    "auc_ndcg",
    "eval_users",
    "pretrain_rounds",
    "adapt_rounds",
    "new_client_ratio",
    "new_user_strategy",
    "min_new_user_history",
    "d_init",
    "new_interactions_per_round",
]


CONTRIBUTION_FIELDS = [
    "dataset",
    "seed",
    "ablation",
    "initial_ndcg_gain",
    "initial_ndcg_gain_pct",
    "final_ndcg_gain",
    "final_ndcg_gain_pct",
    "auc_ndcg_gain",
    "auc_ndcg_gain_pct",
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message: str) -> None:
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(module.weight, 1.0)


def batched_indices(indices: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


def sample_negative_like(pos_items: torch.Tensor, num_items: int) -> torch.Tensor:
    neg = torch.randint(1, num_items + 1, pos_items.shape, device=pos_items.device)
    valid_pos = pos_items > 0
    conflict = valid_pos & (neg == pos_items)
    for _ in range(3):
        if not conflict.any():
            break
        neg[conflict] = torch.randint(1, num_items + 1, (int(conflict.sum().item()),), device=pos_items.device)
        conflict = valid_pos & (neg == pos_items)
    return torch.where(valid_pos, neg, torch.zeros_like(neg))


def bce_seq_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loss_pos = criterion(pos_logits, torch.ones_like(pos_logits))
    loss_neg = criterion(neg_logits, torch.zeros_like(neg_logits))
    return ((loss_pos + loss_neg) * mask).sum() / (mask.sum() + 1e-8)


def normalize_candidate_scores(scores: torch.Tensor) -> torch.Tensor:
    centered = scores - scores.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return centered / scale


def masked_sequence_mean(feats: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
    mask = (seqs != 0).float().unsqueeze(-1)
    return (feats * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def split_clients(num_users: int, new_client_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    users = np.arange(num_users, dtype=np.int64)
    rng.shuffle(users)
    num_new = int(round(num_users * new_client_ratio))
    new_users = np.sort(users[:num_new])
    old_users = np.sort(users[num_new:])
    return old_users, new_users


def split_clients_by_history(
    train_df: pd.DataFrame,
    num_users: int,
    new_client_ratio: float,
    seed: int,
    strategy: str,
    min_history: int,
    required_history: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose cold-start clients while keeping the protocol identical for all methods."""
    if strategy == "random":
        return split_clients(num_users, new_client_ratio, seed)
    counts = train_df.groupby("userId").size().reindex(np.arange(num_users), fill_value=0).to_numpy()
    num_new = int(round(num_users * new_client_ratio))
    min_required = max(int(min_history), 2)
    candidates = np.where(counts >= min_required)[0]
    if candidates.size < num_new:
        return split_clients(num_users, new_client_ratio, seed)

    rng = np.random.RandomState(seed)
    tie_breaker = rng.rand(num_users)
    complete = candidates[counts[candidates] >= int(required_history)]
    incomplete = candidates[counts[candidates] < int(required_history)]
    if strategy == "low_history":
        complete_order = np.lexsort((tie_breaker[complete], counts[complete])) if complete.size else np.asarray([], dtype=np.int64)
        incomplete_order = (
            np.lexsort((tie_breaker[incomplete], counts[incomplete])) if incomplete.size else np.asarray([], dtype=np.int64)
        )
        ordered = np.concatenate([complete[complete_order], incomplete[incomplete_order]])
    elif strategy == "high_history":
        order = np.lexsort((tie_breaker[candidates], -counts[candidates]))
        ordered = candidates[order]
    else:
        raise ValueError(f"Unsupported new-user strategy: {strategy}")
    new_users = np.sort(ordered[:num_new].astype(np.int64))
    old_mask = np.ones(num_users, dtype=bool)
    old_mask[new_users] = False
    old_users = np.where(old_mask)[0].astype(np.int64)
    return old_users, new_users


def make_sequence(items: Sequence[int], max_len: int) -> np.ndarray:
    seq = np.zeros(max_len, dtype=np.int64)
    idx = max_len - 1
    for item in reversed(items):
        seq[idx] = int(item) + 1
        idx -= 1
        if idx < 0:
            break
    return seq


def build_train_arrays(
    train_df: pd.DataFrame,
    num_users: int,
    max_len: int,
    allowed_users: Optional[Sequence[int]] = None,
    d_init_for_users: Optional[Sequence[int]] = None,
    d_init: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    allowed_set = set(int(u) for u in allowed_users) if allowed_users is not None else None
    init_set = set(int(u) for u in d_init_for_users) if d_init_for_users is not None else set()
    train_seq = np.zeros((num_users, max_len), dtype=np.int64)
    train_pos = np.zeros((num_users, max_len), dtype=np.int64)
    visible_counts = np.zeros(num_users, dtype=np.int64)
    for user, user_df in train_df.groupby("userId", sort=False):
        user = int(user)
        if allowed_set is not None and user not in allowed_set:
            continue
        items = user_df.sort_values("timestamp")["itemId"].to_numpy(dtype=np.int64)
        if user in init_set and d_init is not None:
            items = items[:d_init]
        visible_counts[user] = len(items)
        if len(items) < 2:
            continue
        shifted = items + 1
        nxt = int(shifted[-1])
        idx_pos = max_len - 1
        for item in reversed(shifted[:-1]):
            train_seq[user, idx_pos] = int(item)
            train_pos[user, idx_pos] = nxt
            nxt = int(item)
            idx_pos -= 1
            if idx_pos < 0:
                break
    return train_seq, train_pos, visible_counts


def build_eval_arrays(
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidates: Dict[int, List[int]],
    target_users: Sequence[int],
    max_len: int,
    d_init: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_set = set(int(u) for u in target_users)
    hist: Dict[int, List[int]] = {}
    for user, user_df in history_df[history_df["userId"].isin(target_set)].groupby("userId", sort=False):
        items = user_df.sort_values("timestamp")["itemId"].to_numpy(dtype=np.int64).tolist()
        if d_init is not None:
            items = items[:d_init]
        hist[int(user)] = items
    eval_part = eval_df[eval_df["userId"].isin(target_set)].sort_values("userId")
    users: List[int] = []
    seqs: List[np.ndarray] = []
    item_rows: List[List[int]] = []
    for _, row in eval_part.iterrows():
        user = int(row["userId"])
        if user not in candidates:
            continue
        users.append(user)
        seqs.append(make_sequence(hist.get(user, []), max_len))
        item_rows.append([int(row["itemId"]) + 1] + [int(item) + 1 for item in candidates[user]])
    if not users:
        raise RuntimeError("No cold-start users available for evaluation.")
    return (
        np.asarray(users, dtype=np.int64),
        np.stack(seqs).astype(np.int64),
        np.asarray(item_rows, dtype=np.int64),
    )


def build_new_round_arrays(
    train_df: pd.DataFrame,
    num_users: int,
    target_users: Sequence[int],
    cfg: Exp4Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_rounds: List[np.ndarray] = []
    pos_rounds: List[np.ndarray] = []
    eval_seq_rounds: List[np.ndarray] = []
    visible_rounds: List[np.ndarray] = []
    target_set = set(int(u) for u in target_users)
    user_items = {
        int(user): user_df.sort_values("timestamp")["itemId"].to_numpy(dtype=np.int64)
        for user, user_df in train_df[train_df["userId"].isin(target_set)].groupby("userId", sort=False)
    }
    for local_round in range(cfg.adapt_rounds + 1):
        visible = cfg.d_init + local_round * cfg.new_interactions_per_round
        train_seq = np.zeros((num_users, cfg.max_len), dtype=np.int64)
        train_pos = np.zeros((num_users, cfg.max_len), dtype=np.int64)
        eval_seq = np.zeros((num_users, cfg.max_len), dtype=np.int64)
        counts = np.zeros(num_users, dtype=np.int64)
        for user in target_users:
            user = int(user)
            items = user_items.get(user, np.asarray([], dtype=np.int64))
            shown = items[: min(len(items), visible)]
            counts[user] = len(shown)
            eval_seq[user] = make_sequence(shown, cfg.max_len)
            if len(shown) < 2:
                continue
            shifted = shown + 1
            nxt = int(shifted[-1])
            idx_pos = cfg.max_len - 1
            for item in reversed(shifted[:-1]):
                train_seq[user, idx_pos] = int(item)
                train_pos[user, idx_pos] = nxt
                nxt = int(item)
                idx_pos -= 1
                if idx_pos < 0:
                    break
        seq_rounds.append(train_seq)
        pos_rounds.append(train_pos)
        eval_seq_rounds.append(eval_seq)
        visible_rounds.append(counts)
    return (
        np.stack(seq_rounds).astype(np.int64),
        np.stack(pos_rounds).astype(np.int64),
        np.stack(eval_seq_rounds).astype(np.int64),
        np.stack(visible_rounds).astype(np.int64),
    )


class ColdStartData:
    def __init__(
        self,
        num_users: int,
        num_items: int,
        old_users: torch.Tensor,
        new_users: torch.Tensor,
        old_seq: torch.Tensor,
        old_pos: torch.Tensor,
        full_new_seq_by_round: torch.Tensor,
        full_new_pos_by_round: torch.Tensor,
        init_seq: torch.Tensor,
        init_pos: torch.Tensor,
        init_counts: torch.Tensor,
        eval_users: torch.Tensor,
        eval_seq_by_round: torch.Tensor,
        eval_items: torch.Tensor,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.old_users = old_users
        self.new_users = new_users
        self.old_seq = old_seq
        self.old_pos = old_pos
        self.old_mask = (old_seq != 0).float()
        self.full_new_seq_by_round = full_new_seq_by_round
        self.full_new_pos_by_round = full_new_pos_by_round
        self.full_new_mask_by_round = (full_new_seq_by_round != 0).float()
        self.init_seq = init_seq
        self.init_pos = init_pos
        self.init_mask = (init_seq != 0).float()
        self.init_counts = init_counts
        self.eval_users = eval_users
        self.eval_seq_by_round = eval_seq_by_round
        self.eval_items = eval_items

    def to(self, device: torch.device) -> "ColdStartData":
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device, non_blocking=True))
        return self


def load_cold_start_data(cfg: Exp4Config) -> ColdStartData:
    processed_dir = ROOT / "datasets" / "processed" / cfg.dataset
    splits_dir = ROOT / "datasets" / "splits" / cfg.dataset
    with (processed_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])
    train_df = pd.read_csv(splits_dir / "loo_train.csv").sort_values(["userId", "timestamp"]).reset_index(drop=True)
    test_df = pd.read_csv(splits_dir / "loo_test.csv")
    with (splits_dir / "eval_candidates.pkl").open("rb") as handle:
        candidates = pickle.load(handle)

    required_history = cfg.d_init + cfg.adapt_rounds * cfg.new_interactions_per_round
    old_users_np, new_users_np = split_clients_by_history(
        train_df,
        num_users,
        cfg.new_client_ratio,
        cfg.seed,
        cfg.new_user_strategy,
        cfg.min_new_user_history,
        required_history,
    )
    old_seq, old_pos, _ = build_train_arrays(train_df, num_users, cfg.max_len, allowed_users=old_users_np)
    init_seq, init_pos, init_counts = build_train_arrays(
        train_df,
        num_users,
        cfg.max_len,
        allowed_users=new_users_np,
        d_init_for_users=new_users_np,
        d_init=cfg.d_init,
    )
    round_seq, round_pos, eval_seq_rounds, visible_rounds = build_new_round_arrays(
        train_df,
        num_users,
        new_users_np,
        cfg,
    )
    eval_users, _, eval_items = build_eval_arrays(
        train_df,
        test_df,
        candidates,
        new_users_np,
        cfg.max_len,
        d_init=cfg.d_init,
    )
    return ColdStartData(
        num_users=num_users,
        num_items=num_items,
        old_users=torch.from_numpy(old_users_np),
        new_users=torch.from_numpy(new_users_np),
        old_seq=torch.from_numpy(old_seq),
        old_pos=torch.from_numpy(old_pos),
        full_new_seq_by_round=torch.from_numpy(round_seq),
        full_new_pos_by_round=torch.from_numpy(round_pos),
        init_seq=torch.from_numpy(init_seq),
        init_pos=torch.from_numpy(init_pos),
        init_counts=torch.from_numpy(visible_rounds),
        eval_users=torch.from_numpy(eval_users),
        eval_seq_by_round=torch.from_numpy(eval_seq_rounds),
        eval_items=torch.from_numpy(eval_items),
    )


class SASRecEncoder(nn.Module):
    def __init__(self, num_items: int, cfg: Exp4Config):
        super().__init__()
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
                nn.MultiheadAttention(cfg.latent_dim, cfg.num_heads, dropout=cfg.dropout_rate, batch_first=True)
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
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=log_seqs.device), diagonal=1)
        seqs = self.item_emb(log_seqs) * (self.hidden_units**0.5) + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        seqs *= (~timeline_mask.unsqueeze(-1)).float()
        for i in range(len(self.attention_layers)):
            q = self.attention_layernorms[i](seqs)
            attn_out, _ = self.attention_layers[i](q, seqs, seqs, attn_mask=causal_mask)
            seqs = seqs + torch.nan_to_num(attn_out)
            ffn_in = self.forward_layernorms[i](seqs)
            seqs = seqs + self.forward_layers[i](ffn_in)
            seqs *= (~timeline_mask.unsqueeze(-1)).float()
        return self.last_layernorm(seqs)


class FedSASRecModel(nn.Module):
    def __init__(self, num_items: int, cfg: Exp4Config):
        super().__init__()
        self.encoder = SASRecEncoder(num_items, cfg)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.encoder(seqs)


class FedMFModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, cfg: Exp4Config):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, cfg.latent_dim)
        self._item_emb = nn.Embedding(num_items + 1, cfg.latent_dim, padding_idx=0)
        self.apply(init_weights)

    @property
    def item_emb(self) -> nn.Embedding:
        return self._item_emb

    def forward(self, users: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
        user_feats = self.user_emb(users)
        return user_feats.unsqueeze(1).expand(-1, seqs.size(1), -1)


class Dice(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-9):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(input_dim))
        self.eps = eps
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        x_normed = self.bn(x_flat).reshape(orig_shape)
        p = torch.sigmoid(x_normed)
        return p * x + (1.0 - p) * self.alpha * x


class FCUCRModel(nn.Module):
    def __init__(self, num_items: int, cfg: Exp4Config):
        super().__init__()
        self.encoder = SASRecEncoder(num_items, cfg)
        self.mlp = nn.Sequential(
            nn.Linear(2 * cfg.latent_dim, cfg.latent_dim),
            Dice(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.mlp.apply(init_weights)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def forward(self, seqs: torch.Tensor, kb_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        log_feats = self.encoder(seqs)
        kb_seq = torch.zeros_like(log_feats) if kb_feats is None else kb_feats.unsqueeze(1).expand_as(log_feats)
        return self.mlp(torch.cat([log_feats, kb_seq], dim=-1)), log_feats


class TaskAwareRetrievalLayer(nn.Module):
    def __init__(self, hidden_dim: int, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.apply(init_weights)

    def forward(self, local_seq_feats: torch.Tensor, retrieved_memory: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(local_seq_feats)
        k = self.key_proj(retrieved_memory)
        v = self.value_proj(retrieved_memory)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5) / max(self.temperature, 1e-6)
        weights = F.softmax(scores, dim=-1)
        augmented = torch.matmul(weights, v)
        gate = self.gate(torch.cat([local_seq_feats, augmented], dim=-1))
        return self.layer_norm(gate * local_seq_feats + (1.0 - gate) * augmented)


class TarDGRModel(nn.Module):
    def __init__(self, num_items: int, cfg: Exp4Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = SASRecEncoder(num_items, cfg)
        self.retrieval_augmentation = TaskAwareRetrievalLayer(cfg.latent_dim, cfg.tardgr_retrieval_temp)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def forward(
        self,
        seqs: torch.Tensor,
        retrieved_memory: Optional[torch.Tensor] = None,
        retrieval_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_feats = self.encoder(seqs)
        if retrieved_memory is None:
            return local_feats, local_feats
        augmented = self.retrieval_augmentation(local_feats, retrieved_memory)
        mix = min(max(self.cfg.tardgr_retrieval_mix, 0.0), 1.0)
        if retrieval_mask is None:
            return local_feats + mix * (augmented - local_feats), local_feats
        return local_feats + mix * retrieval_mask.float().view(-1, 1, 1) * (augmented - local_feats), local_feats


class F3CRecModel(nn.Module):
    def __init__(self, num_items: int, cfg: Exp4Config):
        super().__init__()
        self.encoder = SASRecEncoder(num_items, cfg)
        self.register_buffer("item_temporal_mean", self.encoder.item_emb.weight.detach().clone(), persistent=False)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.encoder(seqs)


class DGFedRSModel(nn.Module):
    def __init__(self, num_items: int, cfg: Exp4Config):
        super().__init__()
        self.encoder = SASRecEncoder(num_items, cfg)
        self.aug_gate = nn.Sequential(nn.Linear(cfg.latent_dim * 2, cfg.latent_dim), nn.Sigmoid())
        self.aug_norm = nn.LayerNorm(cfg.latent_dim, eps=1e-8)
        self.aug_gate.apply(init_weights)
        self.aug_norm.apply(init_weights)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.encoder(seqs)

    def forward_augmented(
        self,
        seqs: torch.Tensor,
        augmented_seqs: torch.Tensor,
        mix: float,
        noise_std: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_feats = self.encoder(seqs)
        with torch.no_grad():
            aug_feats = self.encoder(augmented_seqs)
        if noise_std > 0:
            aug_feats = aug_feats + torch.randn_like(aug_feats) * noise_std
        gate = self.aug_gate(torch.cat([base_feats, aug_feats], dim=-1))
        fused = base_feats + min(max(mix, 0.0), 1.0) * gate * (self.aug_norm(aug_feats) - base_feats)
        return fused, base_feats


class FedDIEGModel(nn.Module):
    def __init__(self, num_items: int, cfg: Exp4Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = SASRecEncoder(num_items, cfg)
        self.W_proj = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.q_att = nn.Parameter(torch.randn(cfg.latent_dim))
        self.W_gnn = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.W_g = nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False)
        self.graph_gate = nn.Sequential(nn.Linear(cfg.latent_dim * 2, cfg.latent_dim), nn.Sigmoid())
        self.graph_norm = nn.LayerNorm(cfg.latent_dim, eps=1e-8)
        self.short_score_proj = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.short_score_gate = nn.Sequential(nn.Linear(cfg.latent_dim * 2, cfg.latent_dim), nn.Sigmoid())
        self.short_score_norm = nn.LayerNorm(cfg.latent_dim, eps=1e-8)
        self._init_extra_modules()

    def _init_extra_modules(self) -> None:
        # Keep the SASRec encoder initialization aligned with FedSASRec; only
        # initialize FedDIEG-specific fusion and graph modules here.
        for module in [
            self.W_proj,
            self.W_gnn,
            self.W_g,
            self.graph_gate,
            self.graph_norm,
            self.short_score_proj,
            self.short_score_gate,
            self.short_score_norm,
        ]:
            module.apply(init_weights)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def extract_short_term_interest(self, seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_emb = self.encoder(seqs)
        if self.cfg.feddieg_disable_short_attention:
            return seq_emb, masked_sequence_mean(seq_emb, seqs)
        timeline_mask = seqs == 0
        proj_h = torch.tanh(self.W_proj(seq_emb))
        scores = torch.matmul(proj_h, self.q_att) / (self.cfg.latent_dim**0.5)
        scores = scores.masked_fill(timeline_mask, -1e4)
        alphas = F.softmax(scores, dim=-1).unsqueeze(-1)
        return seq_emb, torch.sum(alphas * seq_emb, dim=1)

    def forward(
        self,
        seqs: torch.Tensor,
        neighbor_summaries: Optional[torch.Tensor] = None,
        neighbor_weights: Optional[torch.Tensor] = None,
        graph_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_emb, short_summary = self.extract_short_term_interest(seqs)
        if neighbor_summaries is None or neighbor_weights is None:
            return seq_emb, short_summary
        messages = torch.nan_to_num(self.W_gnn(neighbor_summaries), nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.nan_to_num(neighbor_weights, nan=0.0, posinf=0.0, neginf=0.0)
        g_i = torch.sum(weights.unsqueeze(-1) * messages, dim=1)
        g_i = torch.relu(g_i)
        if graph_mask is not None:
            g_i = g_i * graph_mask.float().unsqueeze(-1)
        graph_delta = torch.nan_to_num(self.graph_norm(self.W_g(g_i)), nan=0.0, posinf=0.0, neginf=0.0)
        gate = self.graph_gate(torch.cat([short_summary, graph_delta], dim=-1))
        graph_delta = gate * graph_delta
        seq_len = seqs.size(1)
        tail_weights = torch.linspace(0.0, 1.0, seq_len, device=seqs.device, dtype=seq_emb.dtype)
        tail_weights = tail_weights.pow(max(self.cfg.feddieg_tail_power, 0.0)).view(1, seq_len, 1)
        valid_positions = (seqs != 0).float().unsqueeze(-1)
        feats = seq_emb + self.cfg.feddieg_graph_mix * graph_delta.unsqueeze(1) * tail_weights * valid_positions
        return feats, short_summary

    def score_representation(self, last_feats: torch.Tensor, short_summary: torch.Tensor) -> torch.Tensor:
        if self.cfg.feddieg_short_score_mix <= 0:
            return last_feats
        short_delta = self.short_score_norm(self.short_score_proj(short_summary))
        gate = self.short_score_gate(torch.cat([last_feats, short_delta], dim=-1))
        return last_feats + self.cfg.feddieg_short_score_mix * gate * short_delta


def build_retrieved_memory(
    users: torch.Tensor,
    memory_bank: torch.Tensor,
    active_clients: torch.Tensor,
    norm_bank: torch.Tensor,
    topk: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    active_indices = torch.nonzero(active_clients, as_tuple=False).flatten()
    if active_indices.numel() <= 1:
        return None, None
    topk = min(topk, active_indices.numel() - 1)
    sims = torch.matmul(norm_bank[users], norm_bank[active_indices].T)
    sims = sims.masked_fill(users.unsqueeze(1) == active_indices.unsqueeze(0), -1e9)
    top_scores, top_positions = torch.topk(sims, topk, dim=1)
    neighbor_ids = active_indices[top_positions]
    retrieved = memory_bank[neighbor_ids]
    mask = active_clients[users] & (top_scores[:, 0] > -1e8)
    return retrieved, mask


def build_weighted_neighbors(
    users: torch.Tensor,
    memory_bank: torch.Tensor,
    active_clients: torch.Tensor,
    norm_bank: torch.Tensor,
    topk: int,
    temperature: float,
    min_sim: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    active_indices = torch.nonzero(active_clients, as_tuple=False).flatten()
    if active_indices.numel() <= 1:
        return None, None, None
    topk = min(topk, active_indices.numel() - 1)
    sims = torch.matmul(norm_bank[users], norm_bank[active_indices].T)
    sims = sims.masked_fill(users.unsqueeze(1) == active_indices.unsqueeze(0), -1e9)
    top_scores, top_positions = torch.topk(sims, topk, dim=1)
    neighbor_ids = active_indices[top_positions]
    score_mask = top_scores >= min_sim
    valid_mask = active_clients[users] & score_mask.any(dim=1)
    safe_scores = (top_scores / max(temperature, 1e-6)).masked_fill(~score_mask, -1e9)
    weights = F.softmax(safe_scores, dim=1)
    weights = torch.where(score_mask, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weights * valid_mask.float().unsqueeze(1)
    return memory_bank[neighbor_ids], weights, valid_mask


def apply_f3crec_replay(
    users: torch.Tensor,
    seqs: torch.Tensor,
    local_feats: torch.Tensor,
    retrieved: Optional[torch.Tensor],
    retrieval_mask: Optional[torch.Tensor],
    memory_bank: torch.Tensor,
    active_clients: torch.Tensor,
    mix: float,
) -> torch.Tensor:
    if retrieved is None or retrieval_mask is None:
        return local_feats
    replay = retrieved.mean(dim=1).unsqueeze(1).expand_as(local_feats)
    current_summary = masked_sequence_mean(local_feats, seqs)
    old_summary = memory_bank[users]
    similarity = F.cosine_similarity(current_summary, old_summary, dim=-1, eps=1e-8).clamp_min(0.0)
    retention = similarity * active_clients[users].float() * retrieval_mask.float()
    return local_feats + min(max(mix, 0.0), 1.0) * retention.view(-1, 1, 1) * replay


@torch.no_grad()
def update_f3crec_item_temporal_mean(
    model: nn.Module,
    seqs: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    beta: float,
) -> None:
    if not hasattr(model, "item_temporal_mean") or beta <= 0:
        return
    item_ids = torch.cat([seqs.reshape(-1), pos_items.reshape(-1), neg_items.reshape(-1)])
    item_ids = item_ids[item_ids > 0].unique()
    if item_ids.numel() == 0:
        return
    current = model.item_emb.weight[item_ids]
    previous = model.item_temporal_mean[item_ids]
    shift = (current - previous).pow(2).mean(dim=1, keepdim=True)
    gamma = min(max(float(beta), 0.0), 1.0) / (1.0 + shift)
    blended = (1.0 - gamma) * current + gamma * previous
    model.item_emb.weight[item_ids].copy_(blended)
    model.item_temporal_mean[item_ids].copy_(blended.detach())


def augment_sequence_for_dgfedrs(
    seqs: torch.Tensor,
    num_items: int,
    ratio: float,
    generator: torch.Generator,
    schedule_power: float,
) -> torch.Tensor:
    ratio = min(max(float(ratio), 0.0), 1.0)
    if ratio <= 0:
        return seqs
    seq_len = seqs.size(1)
    pos_weight = torch.linspace(0.0, 1.0, seq_len, device=seqs.device, dtype=torch.float32)
    pos_weight = pos_weight.pow(max(float(schedule_power), 0.0)).view(1, seq_len)
    probs = ratio * pos_weight
    mask = (seqs != 0) & (torch.rand(seqs.shape, device=seqs.device, generator=generator) < probs)
    sampled_items = torch.randint(1, num_items + 1, seqs.shape, device=seqs.device, generator=generator)
    return torch.where(mask, sampled_items, seqs)


@torch.no_grad()
def update_memory_bank(
    method: str,
    users: torch.Tensor,
    new_memory: torch.Tensor,
    memory_bank: torch.Tensor,
    active_clients: torch.Tensor,
    cfg: Exp4Config,
) -> None:
    if method.startswith("FedDIEG") or method.startswith("Ours"):
        momentum = min(max(cfg.feddieg_memory_momentum, 0.0), 0.99)
        seen = active_clients[users].float().unsqueeze(-1)
        memory_bank[users] = seen * (momentum * memory_bank[users] + (1.0 - momentum) * new_memory) + (
            1.0 - seen
        ) * new_memory
    elif method == "F3CRec" and cfg.f3crec_memory_momentum > 0:
        base_momentum = min(max(cfg.f3crec_memory_momentum, 0.0), 0.99)
        seen = active_clients[users].float().unsqueeze(-1)
        similarity = F.cosine_similarity(memory_bank[users], new_memory, dim=-1, eps=1e-8).clamp_min(0.0).unsqueeze(-1)
        adaptive_momentum = base_momentum * similarity
        memory_bank[users] = seen * (adaptive_momentum * memory_bank[users] + (1.0 - adaptive_momentum) * new_memory) + (
            1.0 - seen
        ) * new_memory
    else:
        memory_bank[users] = new_memory
    active_clients[users] = True


def forward_method(
    method: str,
    model: nn.Module,
    users: torch.Tensor,
    seq: torch.Tensor,
    cfg: Exp4Config,
    memory_bank: Optional[torch.Tensor],
    active_clients: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if method == "FedMF":
        feats = model(users, seq)
        return feats, masked_sequence_mean(feats, seq), None
    if method == "FedSASRec" or method == "DGFedRS":
        feats = model(seq)
        return feats, masked_sequence_mean(feats, seq), None
    norm_bank = None
    if memory_bank is not None and active_clients is not None and active_clients.sum().item() > 1:
        norm_bank = F.normalize(memory_bank, p=2, dim=1)
    if method == "GPFedRec":
        local_feats = model(seq)
        feats = local_feats
        if norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(
                users, memory_bank, active_clients, norm_bank, cfg.gpfedrec_neighbors
            )
            if retrieved is not None and retrieval_mask is not None:
                neighbor_feat = retrieved.mean(dim=1).unsqueeze(1).expand_as(local_feats)
                feats = local_feats + cfg.gpfedrec_mix * retrieval_mask.float().view(-1, 1, 1) * neighbor_feat
        return feats, masked_sequence_mean(local_feats, seq), None
    if method == "FCUCR":
        kb_feats = None
        if norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(
                users, memory_bank, active_clients, norm_bank, cfg.fcucR_kb_len
            )
            if retrieved is not None and retrieval_mask is not None:
                kb_feats = retrieved.mean(dim=1) * retrieval_mask.float().unsqueeze(-1)
        feats, local_feats = model(seq, kb_feats)
        return feats, masked_sequence_mean(local_feats, seq), None
    if method == "TarDGR":
        retrieved = retrieval_mask = None
        if norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(
                users, memory_bank, active_clients, norm_bank, cfg.tardgr_retrieval_topk
            )
        feats, local_feats = model(seq, retrieved, retrieval_mask)
        return feats, masked_sequence_mean(local_feats, seq), None
    if method == "F3CRec":
        local_feats = model(seq)
        feats = local_feats
        if norm_bank is not None:
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
        return feats, masked_sequence_mean(local_feats, seq), None
    if method.startswith("FedDIEG") or method.startswith("Ours"):
        neighbors = weights = graph_mask = None
        if norm_bank is not None:
            neighbors, weights, graph_mask = build_weighted_neighbors(
                users,
                memory_bank,
                active_clients,
                norm_bank,
                cfg.feddieg_neighbors,
                cfg.feddieg_neighbor_temp,
                cfg.feddieg_min_sim,
            )
        feats, short_summary = model(seq, neighbors, weights, graph_mask)
        return feats, short_summary, graph_mask
    raise ValueError(f"Unsupported method: {method}")


def train_batches(
    method: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    users: torch.Tensor,
    seq_source: torch.Tensor,
    pos_source: torch.Tensor,
    mask_source: torch.Tensor,
    cfg: Exp4Config,
    memory_bank: Optional[torch.Tensor],
    active_clients: Optional[torch.Tensor],
    generator: torch.Generator,
    epochs: int,
    allow_memory_update: bool = True,
) -> float:
    losses: List[float] = []
    model.train()
    for _ in range(epochs):
        shuffled = users[torch.randperm(users.numel(), generator=generator, device=users.device)]
        for batch_users in batched_indices(shuffled, cfg.microbatch_size):
            seq = seq_source[batch_users]
            pos = pos_source[batch_users]
            mask = mask_source[batch_users]
            if mask.sum().item() <= 0:
                continue
            neg = sample_negative_like(pos, cfg_num_items(model))
            optimizer.zero_grad(set_to_none=True)
            if method == "DGFedRS":
                augmented_seq = augment_sequence_for_dgfedrs(
                    seq,
                    cfg_num_items(model),
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
                summaries = masked_sequence_mean(local_feats, seq)
            else:
                feats, summaries, _ = forward_method(method, model, batch_users, seq, cfg, memory_bank, active_clients)
            pos_logits = (feats * model.item_emb(pos)).sum(dim=-1)
            neg_logits = (feats * model.item_emb(neg)).sum(dim=-1)
            loss = bce_seq_loss(pos_logits, neg_logits, mask)
            if (method.startswith("FedDIEG") or method.startswith("Ours")) and cfg.feddieg_final_loss_weight > 0:
                lengths = mask.sum(dim=1).long()
                valid = lengths > 0
                if valid.any():
                    rows = torch.arange(batch_users.size(0), device=batch_users.device)[valid]
                    last_pos = (lengths[valid] - 1).clamp_min(0)
                    final_feats = feats[rows, last_pos]
                    final_feats = model.score_representation(final_feats, summaries[rows])
                    final_pos = pos[rows, last_pos]
                    final_neg = neg[rows, last_pos]
                    final_loss = F.binary_cross_entropy_with_logits(
                        (final_feats * model.item_emb(final_pos)).sum(dim=-1),
                        torch.ones_like(final_pos, dtype=torch.float32),
                    ) + F.binary_cross_entropy_with_logits(
                        (final_feats * model.item_emb(final_neg)).sum(dim=-1),
                        torch.zeros_like(final_neg, dtype=torch.float32),
                    )
                    loss = loss + cfg.feddieg_final_loss_weight * final_loss
            if method == "FCUCR" and cfg.fcucR_distill_weight > 0:
                local_feats = model.encoder(seq)
                loss = loss + cfg.fcucR_distill_weight * F.mse_loss(feats, local_feats.detach())
            loss.backward()
            optimizer.step()
            if method == "F3CRec":
                update_f3crec_item_temporal_mean(model, seq, pos, neg, cfg.f3crec_temporal_beta)
            losses.append(float(loss.item()))
            if allow_memory_update and memory_bank is not None and active_clients is not None:
                with torch.no_grad():
                    if method in MEMORY_METHODS or method.startswith("Ours"):
                        new_memory = summaries.detach()
                        update_memory_bank(method, batch_users, new_memory, memory_bank, active_clients, cfg)
    return float(np.mean(losses)) if losses else 0.0


def cfg_num_items(model: nn.Module) -> int:
    return int(model.item_emb.num_embeddings - 1)


@torch.no_grad()
def evaluate_cold_users(
    method: str,
    model: nn.Module,
    data: ColdStartData,
    cfg: Exp4Config,
    memory_bank: Optional[torch.Tensor],
    active_clients: Optional[torch.Tensor],
    proto_bank: Optional[torch.Tensor] = None,
    local_round: int = 0,
) -> Tuple[float, float, int]:
    model.eval()
    users = data.eval_users
    seqs = data.eval_seq_by_round[min(local_round, cfg.adapt_rounds)][users]
    items = data.eval_items
    if cfg.max_eval_users is not None:
        users = users[: cfg.max_eval_users]
        seqs = seqs[: cfg.max_eval_users]
        items = items[: cfg.max_eval_users]
    total_hr = 0.0
    total_ndcg = 0.0
    total = 0
    for idx in batched_indices(torch.arange(users.numel(), device=users.device), cfg.eval_batch_size):
        batch_users = users[idx]
        batch_seq = seqs[idx]
        batch_items = items[idx]
        feats, summaries, _ = forward_method(method, model, batch_users, batch_seq, cfg, memory_bank, active_clients)
        last_feats = feats[:, -1, :]
        item_vectors = model.item_emb(batch_items)
        score_parts: List[Tuple[torch.Tensor, float]] = []
        if (method.startswith("FedDIEG") or method.startswith("Ours")) and hasattr(model, "score_representation"):
            last_feats = model.score_representation(last_feats, summaries)
            if proto_bank is not None and cfg.feddieg_proto_score_mix > 0:
                proto = proto_bank[batch_users]
                proto_scores = torch.matmul(proto.unsqueeze(1), item_vectors.transpose(-1, -2)).squeeze(1)
                score_parts.append((normalize_candidate_scores(proto_scores), cfg.feddieg_proto_score_mix))
                if cfg.feddieg_proto_state_mix > 0:
                    state_proto = cfg.feddieg_proto_state_mix * F.normalize(proto, p=2, dim=1)
                    last_feats = last_feats + state_proto
                if cfg.feddieg_proto_edge_mix > 0 and memory_bank is not None and active_clients is not None:
                    old_active = torch.nonzero(active_clients, as_tuple=False).flatten()
                    old_active = old_active[~torch.isin(old_active, data.new_users)]
                    if old_active.numel() > 0:
                        k = min(cfg.feddieg_proto_topk, old_active.numel())
                        norm_proto = F.normalize(proto, p=2, dim=1)
                        norm_old = F.normalize(memory_bank[old_active], p=2, dim=1)
                        sims = torch.matmul(norm_proto, norm_old.T)
                        scores, pos = torch.topk(sims, k, dim=1)
                        score_mask = scores >= cfg.feddieg_min_sim
                        safe_scores = (scores / max(cfg.feddieg_neighbor_temp, 1e-6)).masked_fill(~score_mask, -1e9)
                        weights = F.softmax(safe_scores, dim=1)
                        weights = torch.where(score_mask, weights, torch.zeros_like(weights))
                        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                        anchor = torch.sum(weights.unsqueeze(-1) * memory_bank[old_active[pos]], dim=1)
                        reliable = score_mask.any(dim=1).float().unsqueeze(-1)
                        edge_scores = torch.matmul(anchor.unsqueeze(1), item_vectors.transpose(-1, -2)).squeeze(1)
                        score_parts.append(
                            (
                                normalize_candidate_scores(edge_scores) * reliable,
                                cfg.feddieg_proto_edge_mix,
                            )
                        )
        base_scores = torch.matmul(last_feats.unsqueeze(1), item_vectors.transpose(-1, -2)).squeeze(1)
        scores = normalize_candidate_scores(base_scores)
        for extra_scores, weight in score_parts:
            scores = scores + weight * extra_scores
        _, top_indices = torch.topk(scores, cfg.top_k, dim=1)
        hits = top_indices == 0
        found = hits.any(dim=1)
        ranks = hits.float().argmax(dim=1)
        ndcg = torch.where(found, math.log(2.0) / torch.log(ranks.float() + 2.0), torch.zeros_like(ranks.float()))
        total_hr += found.float().sum().item()
        total_ndcg += ndcg.sum().item()
        total += batch_users.size(0)
    model.train()
    return total_ndcg / total, total_hr / total, total


def clone_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def restore_model_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict({key: value.detach().clone() for key, value in state.items()})


@torch.no_grad()
def initialize_new_clients(
    method: str,
    model: nn.Module,
    data: ColdStartData,
    cfg: Exp4Config,
    memory_bank: Optional[torch.Tensor],
    active_clients: Optional[torch.Tensor],
    variant: str = "full",
) -> None:
    if memory_bank is None or active_clients is None:
        return
    new_users = data.new_users
    if method in {"FedMF", "FedSASRec", "DGFedRS"}:
        return
    if method in {"GPFedRec", "FCUCR", "TarDGR", "F3CRec"}:
        seq = data.init_seq[new_users]
        if seq.numel() == 0:
            return
        if method == "FCUCR":
            feats, local_feats = model(seq, None)
            new_memory = masked_sequence_mean(local_feats, seq)
        elif method == "TarDGR":
            feats, local_feats = model(seq, None, None)
            new_memory = masked_sequence_mean(local_feats, seq)
        else:
            feats = model(seq)
            new_memory = masked_sequence_mean(feats, seq)
        memory_bank[new_users] = new_memory
        active_clients[new_users] = True
        return
    if not (method.startswith("FedDIEG") or method.startswith("Ours")):
        return
    seq = data.init_seq[new_users]
    _, new_summary = model.extract_short_term_interest(seq)
    if variant in {"random", "no_prototype"}:
        memory_bank[new_users] = new_summary
        active_clients[new_users] = True
        return
    old_active = torch.nonzero(active_clients, as_tuple=False).flatten()
    if old_active.numel() == 0:
        memory_bank[new_users] = new_summary
        active_clients[new_users] = True
        return
    norm_old = F.normalize(memory_bank[old_active], p=2, dim=1)
    norm_new = F.normalize(new_summary, p=2, dim=1)
    sims = torch.matmul(norm_new, norm_old.T)
    k = min(cfg.feddieg_proto_topk, old_active.numel())
    scores, pos = torch.topk(sims, k, dim=1)
    weights = F.softmax(scores / max(cfg.feddieg_neighbor_temp, 1e-6), dim=1)
    proto_summary = torch.sum(weights.unsqueeze(-1) * memory_bank[old_active[pos]], dim=1)
    proto_mix = cfg.feddieg_proto_memory_mix if variant == "full" else min(cfg.feddieg_proto_memory_mix, 0.50)
    memory_bank[new_users] = (1.0 - proto_mix) * new_summary + proto_mix * proto_summary
    active_clients[new_users] = True


@torch.no_grad()
def build_new_user_proto_bank(
    model: nn.Module,
    data: ColdStartData,
    cfg: Exp4Config,
    memory_bank: Optional[torch.Tensor],
    active_clients: Optional[torch.Tensor],
    variant: str,
    local_round: int = 0,
) -> Optional[torch.Tensor]:
    if memory_bank is None or active_clients is None:
        return None
    if not hasattr(model, "extract_short_term_interest"):
        return None
    if variant in {"random", "no_prototype"}:
        return None
    new_users = data.new_users
    round_idx = min(local_round, cfg.adapt_rounds)
    _, new_summary = model.extract_short_term_interest(data.full_new_seq_by_round[round_idx][new_users])
    old_active = torch.nonzero(active_clients, as_tuple=False).flatten()
    old_active = old_active[~torch.isin(old_active, new_users)]
    if old_active.numel() == 0:
        proto_by_user = torch.zeros_like(memory_bank)
        proto_by_user[new_users] = new_summary
        return proto_by_user
    norm_old = F.normalize(memory_bank[old_active], p=2, dim=1)
    norm_new = F.normalize(new_summary, p=2, dim=1)
    sims = torch.matmul(norm_new, norm_old.T)
    k = min(cfg.feddieg_proto_topk, old_active.numel())
    scores, pos = torch.topk(sims, k, dim=1)
    weights = F.softmax(scores / max(cfg.feddieg_neighbor_temp, 1e-6), dim=1)
    proto = torch.sum(weights.unsqueeze(-1) * memory_bank[old_active[pos]], dim=1)
    proto_by_user = torch.zeros_like(memory_bank)
    proto_by_user[new_users] = proto
    return proto_by_user


def build_model(method: str, data: ColdStartData, cfg: Exp4Config, device: torch.device) -> nn.Module:
    if method == "FedMF":
        return FedMFModel(data.num_users, data.num_items, cfg).to(device)
    if method in {"FedSASRec", "GPFedRec"}:
        return FedSASRecModel(data.num_items, cfg).to(device)
    if method == "FCUCR":
        return FCUCRModel(data.num_items, cfg).to(device)
    if method == "TarDGR":
        return TarDGRModel(data.num_items, cfg).to(device)
    if method == "F3CRec":
        return F3CRecModel(data.num_items, cfg).to(device)
    if method == "DGFedRS":
        return DGFedRSModel(data.num_items, cfg).to(device)
    if method.startswith("FedDIEG") or method.startswith("Ours"):
        return FedDIEGModel(data.num_items, cfg).to(device)
    raise ValueError(f"Unsupported method: {method}")


def run_method(
    method: str,
    data: ColdStartData,
    cfg: Exp4Config,
    device: torch.device,
    variant: str = "full",
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, torch.Tensor]]:
    print(f"\n[{method}] start variant={variant}")
    set_seed(cfg.seed)
    model = build_model(method, data, cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
    old_clients_per_round = cfg.clients_per_round or max(int(cfg.participation_rate * data.old_users.numel()), 1)
    old_clients_per_round = min(old_clients_per_round, data.old_users.numel())
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    memory_bank: Optional[torch.Tensor] = None
    active_clients: Optional[torch.Tensor] = None
    if method in MEMORY_METHODS or method.startswith("Ours"):
        memory_bank = torch.zeros((data.num_users, cfg.latent_dim), device=device)
        active_clients = torch.zeros(data.num_users, dtype=torch.bool, device=device)

    start = time.time()
    for round_idx in range(1, cfg.pretrain_rounds + 1):
        sampled_pos = torch.randperm(data.old_users.numel(), generator=generator, device=device)[:old_clients_per_round]
        sampled = data.old_users[sampled_pos]
        loss = train_batches(
            method,
            model,
            optimizer,
            sampled,
            data.old_seq,
            data.old_pos,
            data.old_mask,
            cfg,
            memory_bank,
            active_clients,
            generator,
            cfg.local_epochs,
            allow_memory_update=True,
        )
        print(f"[{method}] pretrain {round_idx:03d}/{cfg.pretrain_rounds} loss={loss:.4f}")

    pre_state = clone_model_state(model)
    if memory_bank is not None:
        pre_memory = memory_bank.detach().clone()
        pre_active = active_clients.detach().clone()
    else:
        pre_memory = torch.empty(0, device=device)
        pre_active = torch.empty(0, dtype=torch.bool, device=device)

    initialize_new_clients(method, model, data, cfg, memory_bank, active_clients, variant=variant)
    proto_bank = None
    if method == "FedDIEG":
        proto_bank = build_new_user_proto_bank(
            model,
            data,
            cfg,
            memory_bank,
            active_clients,
            variant=variant,
            local_round=0,
        )

    adapt_optimizer = optim.Adam(model.parameters(), lr=cfg.adapt_lr, betas=(0.9, 0.98))
    new_per_round = cfg.adapt_clients_per_round or data.new_users.numel()
    new_per_round = min(new_per_round, data.new_users.numel())

    rows: List[Dict[str, object]] = []
    for local_round in range(0, cfg.adapt_rounds + 1):
        ndcg, hr, eval_users = evaluate_cold_users(
            method,
            model,
            data,
            cfg,
            memory_bank,
            active_clients,
            proto_bank=proto_bank,
            local_round=local_round,
        )
        visible_interactions = float(
            data.init_counts[min(local_round, cfg.adapt_rounds)][data.new_users].float().mean().item()
        )
        rows.append(
            {
                "method": method,
                "dataset": cfg.dataset,
                "seed": cfg.seed,
                "local_round": local_round,
                "ndcg_at_10": ndcg,
                "hr_at_10": hr,
                "visible_interactions": visible_interactions,
                "eval_users": eval_users,
            }
        )
        print(f"[{method}] adapt {local_round:02d}/{cfg.adapt_rounds} NDCG={ndcg:.4f} HR={hr:.4f}")
        if local_round == cfg.adapt_rounds:
            break
        sampled_pos = torch.randperm(data.new_users.numel(), generator=generator, device=device)[:new_per_round]
        sampled = data.new_users[sampled_pos]
        train_batches(
            method,
            model,
            adapt_optimizer,
            sampled,
            data.full_new_seq_by_round[min(local_round + 1, cfg.adapt_rounds)],
            data.full_new_pos_by_round[min(local_round + 1, cfg.adapt_rounds)],
            data.full_new_mask_by_round[min(local_round + 1, cfg.adapt_rounds)],
            cfg,
            memory_bank,
            active_clients,
            generator,
            cfg.adapt_epochs,
            allow_memory_update=True,
        )
        if method == "FedDIEG":
            proto_bank = build_new_user_proto_bank(
                model,
                data,
                cfg,
                memory_bank,
                active_clients,
                variant=variant,
                local_round=local_round + 1,
            )

    summary = summarize_rows(rows, cfg, method)
    summary_state = {
        "pre_state": pre_state,
        "pre_memory": pre_memory,
        "pre_active": pre_active,
        "elapsed": torch.tensor(time.time() - start, device=device),
    }
    return rows, summary, summary_state


def summarize_rows(rows: List[Dict[str, object]], cfg: Exp4Config, method: str) -> Dict[str, object]:
    first = rows[0]
    last = rows[-1]
    initial_ndcg = float(first["ndcg_at_10"])
    final_ndcg = float(last["ndcg_at_10"])
    rounds = max(int(last["local_round"]) - int(first["local_round"]), 1)
    gain = final_ndcg - initial_ndcg
    best_row = max(rows, key=lambda row: float(row["ndcg_at_10"]))
    early_target_round = min(5, int(last["local_round"]))
    early_row = next((row for row in rows if int(row["local_round"]) == early_target_round), last)
    early_growth = (float(early_row["ndcg_at_10"]) - initial_ndcg) / max(early_target_round, 1)
    return {
        "method": method,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "initial_ndcg": initial_ndcg,
        "initial_hr": float(first["hr_at_10"]),
        "final_ndcg": final_ndcg,
        "final_hr": float(last["hr_at_10"]),
        "best_ndcg": float(best_row["ndcg_at_10"]),
        "best_round": int(best_row["local_round"]),
        "growth_rate": gain / rounds,
        "early_growth_rate": early_growth,
        "relative_gain": gain / max(initial_ndcg, 1e-8),
        "auc_ndcg": float(np.mean([float(row["ndcg_at_10"]) for row in rows])),
        "eval_users": int(first["eval_users"]),
        "pretrain_rounds": cfg.pretrain_rounds,
        "adapt_rounds": cfg.adapt_rounds,
        "new_client_ratio": cfg.new_client_ratio,
        "new_user_strategy": cfg.new_user_strategy,
        "min_new_user_history": cfg.min_new_user_history,
        "d_init": cfg.d_init,
        "new_interactions_per_round": cfg.new_interactions_per_round,
    }


def run_ablation_from_checkpoint(
    base_state: Dict[str, torch.Tensor],
    data: ColdStartData,
    cfg: Exp4Config,
    device: torch.device,
    variant: str,
    label: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    method = label
    if variant == "no_graph_anchor":
        eval_cfg = replace(cfg, feddieg_graph_mix=0.0, feddieg_proto_edge_mix=0.0)
    elif variant == "no_prototype":
        eval_cfg = replace(
            cfg,
            feddieg_proto_score_mix=0.0,
            feddieg_proto_state_mix=0.0,
            feddieg_proto_memory_mix=0.0,
            feddieg_proto_edge_mix=0.0,
        )
    elif variant == "no_short_gate":
        eval_cfg = replace(
            cfg,
            feddieg_short_score_mix=0.0,
            feddieg_final_loss_weight=0.0,
            feddieg_disable_short_attention=True,
        )
    elif variant in {"prototype", "random"}:
        eval_cfg = replace(
            cfg,
            feddieg_proto_score_mix=cfg.feddieg_ablation_proto_score_mix,
            feddieg_graph_mix=0.0 if variant == "random" else cfg.feddieg_graph_mix,
            feddieg_proto_edge_mix=0.0 if variant == "random" else cfg.feddieg_proto_edge_mix,
        )
    else:
        eval_cfg = cfg
    model = FedDIEGModel(data.num_items, eval_cfg).to(device)
    restore_model_state(model, base_state["pre_state"])
    memory_bank = base_state["pre_memory"].detach().clone()
    active_clients = base_state["pre_active"].detach().clone()
    if variant in {"random", "no_prototype"}:
        memory_bank[data.new_users] = 0.0
        active_clients[data.new_users] = False
    initialize_new_clients("FedDIEG", model, data, eval_cfg, memory_bank, active_clients, variant=variant)
    proto_bank = build_new_user_proto_bank(
        model,
        data,
        eval_cfg,
        memory_bank,
        active_clients,
        variant=variant,
        local_round=0,
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.adapt_lr, betas=(0.9, 0.98))
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 17)
    rows: List[Dict[str, object]] = []
    for local_round in range(0, cfg.adapt_rounds + 1):
        ndcg, hr, eval_users = evaluate_cold_users(
            "FedDIEG",
            model,
            data,
            eval_cfg,
            memory_bank,
            active_clients,
            proto_bank=proto_bank,
            local_round=local_round,
        )
        rows.append(
            {
                "method": method,
                "dataset": cfg.dataset,
                "seed": cfg.seed,
                "local_round": local_round,
                "ndcg_at_10": ndcg,
                "hr_at_10": hr,
                "visible_interactions": float(
                    data.init_counts[min(local_round, cfg.adapt_rounds)][data.new_users].float().mean().item()
                ),
                "eval_users": eval_users,
            }
        )
        print(f"[{method}] adapt {local_round:02d}/{cfg.adapt_rounds} NDCG={ndcg:.4f} HR={hr:.4f}")
        if local_round == cfg.adapt_rounds:
            break
        train_batches(
            "FedDIEG",
            model,
            optimizer,
            data.new_users,
            data.full_new_seq_by_round[min(local_round + 1, cfg.adapt_rounds)],
            data.full_new_pos_by_round[min(local_round + 1, cfg.adapt_rounds)],
            data.full_new_mask_by_round[min(local_round + 1, cfg.adapt_rounds)],
            eval_cfg,
            memory_bank,
            active_clients,
            generator,
            eval_cfg.adapt_epochs,
            allow_memory_update=True,
        )
        proto_bank = build_new_user_proto_bank(
            model,
            data,
            eval_cfg,
            memory_bank,
            active_clients,
            variant=variant,
            local_round=local_round + 1,
        )
    return rows, summarize_rows(rows, cfg, method)


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_contribution_rows(ablation_summaries: List[Dict[str, object]], cfg: Exp4Config) -> List[Dict[str, object]]:
    full = next((row for row in ablation_summaries if row["method"] == "FedDIEG"), None)
    if full is None:
        return []
    rows: List[Dict[str, object]] = []
    for row in ablation_summaries:
        if row["method"] == "FedDIEG":
            continue
        initial_gain = float(full["initial_ndcg"]) - float(row["initial_ndcg"])
        final_gain = float(full["final_ndcg"]) - float(row["final_ndcg"])
        auc_gain = float(full["auc_ndcg"]) - float(row["auc_ndcg"])
        rows.append(
            {
                "dataset": cfg.dataset,
                "seed": cfg.seed,
                "ablation": row["method"],
                "initial_ndcg_gain": initial_gain,
                "initial_ndcg_gain_pct": initial_gain / max(float(row["initial_ndcg"]), 1e-8),
                "final_ndcg_gain": final_gain,
                "final_ndcg_gain_pct": final_gain / max(float(row["final_ndcg"]), 1e-8),
                "auc_ndcg_gain": auc_gain,
                "auc_ndcg_gain_pct": auc_gain / max(float(row["auc_ndcg"]), 1e-8),
            }
        )
    return rows


def make_plot(out_dir: Path, trajectory_path: Path, title: str, output_name: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_csv(trajectory_path)
        plt.figure(figsize=(7.2, 4.6))
        for method, group in df.groupby("method"):
            group = group.sort_values("local_round")
            plt.plot(group["local_round"], group["ndcg_at_10"], marker="o", linewidth=2, label=method)
        plt.xlabel("Local adaptation rounds")
        plt.ylabel("NDCG@10")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / output_name, dpi=180)
        plt.close()
    except Exception as exc:
        print(f"[plot] skipped because matplotlib failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4 prototype semantic anchoring runner for new-client onboarding."
    )
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=list(SUPPORTED_METHODS),
    )
    parser.add_argument("--pretrain-rounds", type=int, default=50)
    parser.add_argument("--adapt-rounds", type=int, default=10)
    parser.add_argument("--new-client-ratio", type=float, default=0.25)
    parser.add_argument("--new-user-strategy", choices=["random", "low_history", "high_history"], default="low_history")
    parser.add_argument("--min-new-user-history", type=int, default=5)
    parser.add_argument("--d-init", type=int, default=3)
    parser.add_argument("--new-interactions-per-round", type=int, default=2)
    parser.add_argument("--participation-rate", type=float, default=0.1)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--adapt-clients-per-round", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--adapt-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--adapt-lr", type=float, default=0.0002)
    parser.add_argument("--microbatch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--gpfedrec-neighbors", type=int, default=8)
    parser.add_argument("--gpfedrec-mix", type=float, default=0.12)
    parser.add_argument("--fcucR-kb-len", type=int, default=5)
    parser.add_argument("--fcucR-noise-std", type=float, default=0.01)
    parser.add_argument("--fcucR-distill-weight", type=float, default=0.1)
    parser.add_argument("--tardgr-retrieval-topk", type=int, default=10)
    parser.add_argument("--tardgr-retrieval-temp", type=float, default=0.1)
    parser.add_argument("--tardgr-retrieval-mix", type=float, default=0.2)
    parser.add_argument("--tardgr-warmup-rounds", type=int, default=10)
    parser.add_argument("--f3crec-replay-topk", type=int, default=8)
    parser.add_argument("--f3crec-replay-mix", type=float, default=0.15)
    parser.add_argument("--f3crec-temporal-beta", type=float, default=0.15)
    parser.add_argument("--f3crec-memory-momentum", type=float, default=0.60)
    parser.add_argument("--f3crec-warmup-rounds", type=int, default=10)
    parser.add_argument("--dgfedrs-aug-ratio", type=float, default=0.15)
    parser.add_argument("--dgfedrs-aug-mix", type=float, default=0.30)
    parser.add_argument("--dgfedrs-noise-std", type=float, default=0.02)
    parser.add_argument("--dgfedrs-schedule-power", type=float, default=1.0)
    parser.add_argument("--feddieg-neighbors", type=int, default=20)
    parser.add_argument("--feddieg-graph-mix", type=float, default=0.30)
    parser.add_argument("--feddieg-neighbor-temp", type=float, default=0.05)
    parser.add_argument("--feddieg-min-sim", type=float, default=0.05)
    parser.add_argument("--feddieg-tail-power", type=float, default=1.8)
    parser.add_argument("--feddieg-final-loss-weight", type=float, default=0.15)
    parser.add_argument("--feddieg-memory-momentum", type=float, default=0.45)
    parser.add_argument("--feddieg-short-score-mix", type=float, default=0.20)
    parser.add_argument("--feddieg-proto-topk", type=int, default=8)
    parser.add_argument("--feddieg-proto-state-mix", type=float, default=0.35)
    parser.add_argument("--feddieg-proto-memory-mix", type=float, default=0.75)
    parser.add_argument("--feddieg-proto-edge-mix", type=float, default=0.45)
    parser.add_argument("--feddieg-proto-score-mix", type=float, default=0.55)
    parser.add_argument("--feddieg-ablation-proto-score-mix", type=float, default=0.0)
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--include-short-ablation", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp_04_fast_gpu" / "results_movielens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Exp4Config(
        dataset=args.dataset,
        methods=tuple(args.methods),
        seed=args.seed,
        pretrain_rounds=args.pretrain_rounds,
        adapt_rounds=args.adapt_rounds,
        new_client_ratio=args.new_client_ratio,
        new_user_strategy=args.new_user_strategy,
        min_new_user_history=args.min_new_user_history,
        d_init=args.d_init,
        new_interactions_per_round=args.new_interactions_per_round,
        participation_rate=args.participation_rate,
        clients_per_round=args.clients_per_round,
        adapt_clients_per_round=args.adapt_clients_per_round,
        local_epochs=args.local_epochs,
        adapt_epochs=args.adapt_epochs,
        lr=args.lr,
        adapt_lr=args.adapt_lr,
        microbatch_size=args.microbatch_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_users=args.max_eval_users,
        num_threads=args.num_threads,
        gpfedrec_neighbors=args.gpfedrec_neighbors,
        gpfedrec_mix=args.gpfedrec_mix,
        fcucR_kb_len=args.fcucR_kb_len,
        fcucR_noise_std=args.fcucR_noise_std,
        fcucR_distill_weight=args.fcucR_distill_weight,
        tardgr_retrieval_topk=args.tardgr_retrieval_topk,
        tardgr_retrieval_temp=args.tardgr_retrieval_temp,
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
        feddieg_final_loss_weight=args.feddieg_final_loss_weight,
        feddieg_memory_momentum=args.feddieg_memory_momentum,
        feddieg_short_score_mix=args.feddieg_short_score_mix,
        feddieg_proto_topk=args.feddieg_proto_topk,
        feddieg_proto_state_mix=args.feddieg_proto_state_mix,
        feddieg_proto_memory_mix=args.feddieg_proto_memory_mix,
        feddieg_proto_edge_mix=args.feddieg_proto_edge_mix,
        feddieg_proto_score_mix=args.feddieg_proto_score_mix,
        feddieg_ablation_proto_score_mix=args.feddieg_ablation_proto_score_mix,
    )
    torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "exp04_fast_gpu.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {json.dumps(asdict(cfg), ensure_ascii=False, indent=2)}")
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2)

    print(f"\nLoading cold-start data: {cfg.dataset}")
    data = load_cold_start_data(cfg).to(device)
    print(
        f"users={data.num_users}, items={data.num_items}, old={data.old_users.numel()}, "
        f"new={data.new_users.numel()}, eval_new={data.eval_users.numel()}, d_init={cfg.d_init}"
    )

    all_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    feddieg_checkpoint: Optional[Dict[str, torch.Tensor]] = None
    for method in cfg.methods:
        rows, summary, checkpoint = run_method(method, data, cfg, device, variant="full")
        all_rows.extend(rows)
        summaries.append(summary)
        if method == "FedDIEG":
            feddieg_checkpoint = checkpoint
        write_csv(out_dir / "trajectory.csv", all_rows, TRAJECTORY_FIELDS)
        write_csv(out_dir / "summary.csv", summaries, SUMMARY_FIELDS)
        print(f"[{method}] saved trajectory.csv and summary.csv")

    make_plot(
        out_dir,
        out_dir / "trajectory.csv",
        f"Experiment 4 Prototype Anchoring - {cfg.dataset}",
        "ndcg_trajectory.png",
    )

    if not args.no_ablation and feddieg_checkpoint is not None:
        ablation_rows: List[Dict[str, object]] = []
        ablation_summaries: List[Dict[str, object]] = []
        ablation_specs = [
            ("no_prototype", "Ours-NoPrototype"),
            ("no_graph_anchor", "Ours-NoGraphAnchor"),
        ]
        if args.include_short_ablation:
            ablation_specs.append(("no_short_gate", "Ours-NoShortGate"))
        ablation_specs.append(("full", "FedDIEG"))
        for variant, label in ablation_specs:
            if label == "FedDIEG":
                rows = [dict(row, method="FedDIEG") for row in all_rows if row["method"] == "FedDIEG"]
                summary = next(row for row in summaries if row["method"] == "FedDIEG")
            else:
                rows, summary = run_ablation_from_checkpoint(feddieg_checkpoint, data, cfg, device, variant, label)
            ablation_rows.extend(rows)
            ablation_summaries.append(summary)
        write_csv(out_dir / "ablation_trajectory.csv", ablation_rows, TRAJECTORY_FIELDS)
        write_csv(out_dir / "ablation_summary.csv", ablation_summaries, SUMMARY_FIELDS)
        contribution_rows = build_contribution_rows(ablation_summaries, cfg)
        write_csv(out_dir / "ablation_contribution.csv", contribution_rows, CONTRIBUTION_FIELDS)
        make_plot(
            out_dir,
            out_dir / "ablation_trajectory.csv",
            f"Experiment 4 Prototype Anchoring Ablation - {cfg.dataset}",
            "ablation_ndcg_trajectory.png",
        )

    print(f"\nDone. Final outputs: {out_dir}")


if __name__ == "__main__":
    main()
