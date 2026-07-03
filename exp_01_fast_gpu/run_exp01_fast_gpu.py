"""
GPU-batched Experiment 1 prototype.

This runner is intentionally separate from the original per-client FedAvg code.
It keeps the same LOO split and candidate evaluation, but batches sampled clients
on GPU so the 4090 can do useful work. All methods in this file share the same
batched protocol: client sampling, negative sampling, candidate evaluation, and
microbatch size are aligned.
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
from dataclasses import asdict, dataclass
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

BASELINE_METHODS = ("FedMF", "FedSASRec", "GPFedRec", "FCUCR", "TarDGR", "F3CRec", "DGFedRS")
ALL_METHODS = BASELINE_METHODS + ("FedDIEG",)
MEMORY_METHODS = {"GPFedRec", "FCUCR", "TarDGR", "F3CRec", "FedDIEG"}


@dataclass
class FastConfig:
    dataset: str = "amazon-digital music"
    methods: Tuple[str, ...] = ALL_METHODS
    seed: int = 42
    rounds: int = 600
    participation_rate: float = 0.1
    clients_per_round: Optional[int] = None
    microbatch_size: int = 512
    eval_batch_size: int = 1024
    max_eval_users: Optional[int] = None
    local_epochs: int = 1
    lr: float = 0.001
    feddieg_lr: Optional[float] = None
    latent_dim: int = 64
    max_len: int = 50
    top_k: int = 10
    eval_every: int = 10
    num_heads: int = 4
    num_blocks: int = 2
    dropout_rate: float = 0.2
    gpfedrec_neighbors: int = 5
    gpfedrec_mix: float = 0.1
    gpfedrec_warmup_rounds: int = 20
    fcucR_kb_len: int = 5
    fcucR_noise_std: float = 0.01
    fcucR_distill_weight: float = 0.1
    tardgr_retrieval_topk: int = 10
    tardgr_retrieval_temp: float = 0.1
    tardgr_retrieval_mix: float = 0.2
    tardgr_warmup_rounds: int = 20
    f3crec_replay_topk: int = 8
    f3crec_replay_mix: float = 0.15
    f3crec_temporal_beta: float = 0.15
    f3crec_memory_momentum: float = 0.60
    f3crec_warmup_rounds: int = 20
    dgfedrs_aug_ratio: float = 0.15
    dgfedrs_aug_mix: float = 0.30
    dgfedrs_noise_std: float = 0.02
    dgfedrs_schedule_power: float = 1.0
    feddieg_neighbors: int = 20
    feddieg_graph_mix: float = 0.2
    feddieg_warmup_rounds: int = 20
    feddieg_neighbor_temp: float = 0.05
    feddieg_min_sim: float = 0.10
    feddieg_tail_power: float = 2.0
    feddieg_final_loss_weight: float = 0.2
    feddieg_memory_momentum: float = 0.5
    feddieg_short_score_mix: float = 0.3
    feddieg_confidence_power: float = 0.0
    feddieg_agreement_power: float = 0.0
    feddieg_score_residual_mix: float = 0.0
    feddieg_score_local_mix: float = 0.0
    feddieg_boundary_refresh_rate: float = 0.0
    feddieg_boundary_refresh_momentum: float = 0.5
    num_threads: int = 4


ROUND_FIELDS = [
    "method",
    "round",
    "train_loss",
    "val_hr",
    "val_ndcg",
    "val_recall",
    "test_hr",
    "test_ndcg",
    "test_recall",
    "seconds",
]
SUMMARY_FIELDS = [
    "method",
    "dataset",
    "seed",
    "rounds",
    "clients_per_round",
    "microbatch_size",
    "lr",
    "best_round",
    "best_val_ndcg",
    "best_test_hr",
    "best_test_ndcg",
    "best_test_recall",
    "total_time_seconds",
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


def bce_seq_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    labels_pos = torch.ones_like(pos_logits)
    labels_neg = torch.zeros_like(neg_logits)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loss = criterion(pos_logits, labels_pos) + criterion(neg_logits, labels_neg)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def masked_sequence_mean(feats: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
    mask = (seqs != 0).float().unsqueeze(-1)
    return (feats * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class FastData:
    def __init__(
        self,
        num_users: int,
        num_items: int,
        train_seq: torch.Tensor,
        train_pos: torch.Tensor,
        val_users: torch.Tensor,
        val_seq: torch.Tensor,
        val_items: torch.Tensor,
        test_users: torch.Tensor,
        test_seq: torch.Tensor,
        test_items: torch.Tensor,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.train_seq = train_seq
        self.train_pos = train_pos
        self.train_mask = (train_seq != 0).float()
        self.val_users = val_users
        self.val_seq = val_seq
        self.val_items = val_items
        self.test_users = test_users
        self.test_seq = test_seq
        self.test_items = test_items

    def to(self, device: torch.device) -> "FastData":
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device, non_blocking=True))
        return self


def build_train_arrays(train_df: pd.DataFrame, num_users: int, num_items: int, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    train_seq = np.zeros((num_users, max_len), dtype=np.int64)
    train_pos = np.zeros((num_users, max_len), dtype=np.int64)
    for user, user_df in train_df.groupby("userId", sort=False):
        items = user_df["itemId"].to_numpy(dtype=np.int64) + 1
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


def build_eval_arrays(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidates: Dict[int, List[int]],
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_hist = train_df.groupby("userId", sort=False)["itemId"].apply(lambda x: list(x.to_numpy(dtype=np.int64) + 1)).to_dict()
    users: List[int] = []
    seqs: List[np.ndarray] = []
    item_rows: List[List[int]] = []
    for _, row in eval_df.iterrows():
        user = int(row["userId"])
        seq = np.zeros(max_len, dtype=np.int64)
        idx_pos = max_len - 1
        for item in reversed(user_hist.get(user, [])):
            seq[idx_pos] = int(item)
            idx_pos -= 1
            if idx_pos < 0:
                break
        users.append(user)
        seqs.append(seq)
        item_rows.append([int(row["itemId"]) + 1] + [int(item) + 1 for item in candidates[user]])
    return (
        np.asarray(users, dtype=np.int64),
        np.stack(seqs).astype(np.int64),
        np.asarray(item_rows, dtype=np.int64),
    )


def load_fast_data(dataset: str, max_len: int) -> FastData:
    processed_dir = ROOT / "datasets" / "processed" / dataset
    splits_dir = ROOT / "datasets" / "splits" / dataset
    with (processed_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])
    train_df = pd.read_csv(splits_dir / "loo_train.csv")
    val_df = pd.read_csv(splits_dir / "loo_val.csv")
    test_df = pd.read_csv(splits_dir / "loo_test.csv")
    with (splits_dir / "eval_candidates.pkl").open("rb") as handle:
        candidates = pickle.load(handle)

    train_seq, train_pos = build_train_arrays(train_df, num_users, num_items, max_len)
    val_users, val_seq, val_items = build_eval_arrays(train_df, val_df, candidates, max_len)
    test_users, test_seq, test_items = build_eval_arrays(train_df, test_df, candidates, max_len)
    return FastData(
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


class SASRecEncoder(nn.Module):
    def __init__(self, num_items: int, cfg: FastConfig):
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


class FedSASRecModel(nn.Module):
    def __init__(self, num_items: int, cfg: FastConfig):
        super().__init__()
        self.encoder = SASRecEncoder(num_items, cfg)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.encoder(seqs)


class FedMFModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, cfg: FastConfig):
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


class F3CRecModel(nn.Module):
    def __init__(self, num_items: int, cfg: FastConfig):
        super().__init__()
        self.encoder = SASRecEncoder(num_items, cfg)
        self.register_buffer("item_temporal_mean", self.encoder.item_emb.weight.detach().clone(), persistent=False)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.encoder(seqs)


class DGFedRSModel(nn.Module):
    def __init__(self, num_items: int, cfg: FastConfig):
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
    def __init__(self, num_items: int, cfg: FastConfig):
        super().__init__()
        self.encoder = SASRecEncoder(num_items, cfg)
        self.mlp = nn.Sequential(
            nn.Linear(2 * cfg.latent_dim, cfg.latent_dim),
            Dice(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.apply(init_weights)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def log2feats(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.encoder(seqs)

    def forward(self, seqs: torch.Tensor, kb_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        log_feats = self.log2feats(seqs)
        if kb_feats is None:
            kb_seq = torch.zeros_like(log_feats)
        else:
            kb_seq = kb_feats.unsqueeze(1).expand_as(log_feats)
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

    def forward(self, local_seq_feats: torch.Tensor, retrieved_memory: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(local_seq_feats)
        k = self.key_proj(retrieved_memory)
        v = self.value_proj(retrieved_memory)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5) / self.temperature
        weights = F.softmax(scores, dim=-1)
        augmented = torch.matmul(weights, v)
        gate = self.gate(torch.cat([local_seq_feats, augmented], dim=-1))
        return self.layer_norm(gate * local_seq_feats + (1.0 - gate) * augmented)


class TarDGRModel(nn.Module):
    def __init__(self, num_items: int, cfg: FastConfig):
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
        mask = retrieval_mask.float().view(-1, 1, 1)
        return local_feats + mix * mask * (augmented - local_feats), local_feats


class FedDIEGModel(nn.Module):
    def __init__(self, num_items: int, cfg: FastConfig):
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
        self.apply(init_weights)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.encoder.item_emb

    def extract_short_term_interest(self, seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_emb = self.encoder(seqs)
        timeline_mask = seqs == 0
        proj_h = torch.tanh(self.W_proj(seq_emb))
        scores = torch.matmul(proj_h, self.q_att) / (self.cfg.latent_dim ** 0.5)
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
        neighbor_weights = torch.nan_to_num(neighbor_weights, nan=0.0, posinf=0.0, neginf=0.0)
        g_i = torch.sum(neighbor_weights.unsqueeze(-1) * messages, dim=1)
        g_i = torch.relu(g_i)
        if graph_mask is not None:
            g_i = g_i * graph_mask.float().unsqueeze(-1)
        mix = max(self.cfg.feddieg_graph_mix, 0.0)
        g_i = torch.nan_to_num(g_i, nan=0.0, posinf=0.0, neginf=0.0)
        graph_delta = torch.nan_to_num(self.graph_norm(self.W_g(g_i)), nan=0.0, posinf=0.0, neginf=0.0)
        gate = self.graph_gate(torch.cat([short_summary, graph_delta], dim=-1))
        graph_delta = torch.nan_to_num(gate * graph_delta, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.feddieg_confidence_power > 0:
            confidence = neighbor_weights.max(dim=1).values.clamp(0.0, 1.0)
            confidence_mask = (confidence > 0).float()
            confidence = confidence.clamp_min(1e-6)
            confidence = confidence.pow(self.cfg.feddieg_confidence_power).unsqueeze(-1)
            graph_delta = graph_delta * confidence * confidence_mask.unsqueeze(-1)
        if self.cfg.feddieg_agreement_power > 0:
            agreement = F.cosine_similarity(short_summary, g_i, dim=-1, eps=1e-8)
            agreement = torch.nan_to_num(agreement, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            agreement_mask = (agreement > 0).float()
            agreement = agreement.clamp_min(1e-6)
            agreement = agreement.pow(self.cfg.feddieg_agreement_power).unsqueeze(-1)
            graph_delta = graph_delta * agreement * agreement_mask.unsqueeze(-1)
        graph_delta = torch.nan_to_num(graph_delta, nan=0.0, posinf=0.0, neginf=0.0)
        seq_len = seqs.size(1)
        tail_weights = torch.linspace(0.0, 1.0, seq_len, device=seqs.device, dtype=seq_emb.dtype)
        tail_weights = tail_weights.pow(max(self.cfg.feddieg_tail_power, 0.0)).view(1, seq_len, 1)
        valid_positions = (seqs != 0).float().unsqueeze(-1)
        feats = seq_emb + mix * graph_delta.unsqueeze(1) * tail_weights * valid_positions
        return feats, short_summary

    def score_representation(
        self,
        last_feats: torch.Tensor,
        short_summary: torch.Tensor,
        base_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fused = last_feats
        if self.cfg.feddieg_short_score_mix > 0:
            short_delta = self.short_score_norm(self.short_score_proj(short_summary))
            gate = self.short_score_gate(torch.cat([last_feats, short_delta], dim=-1))
            fused = last_feats + self.cfg.feddieg_short_score_mix * gate * short_delta
        residual_mix = min(max(self.cfg.feddieg_score_residual_mix, 0.0), 1.0)
        if residual_mix > 0:
            fused = (1.0 - residual_mix) * fused + residual_mix * last_feats
        local_mix = min(max(self.cfg.feddieg_score_local_mix, 0.0), 1.0)
        if local_mix > 0 and base_feats is not None:
            fused = (1.0 - local_mix) * fused + local_mix * base_feats
        return fused


def sample_negative_like(pos_items: torch.Tensor, num_items: int) -> torch.Tensor:
    neg = torch.randint(1, num_items + 1, pos_items.shape, device=pos_items.device)
    valid_pos = pos_items > 0
    conflict = valid_pos & (neg == pos_items)
    for _ in range(3):
        if not conflict.any():
            break
        neg[conflict] = torch.randint(1, num_items + 1, (int(conflict.sum().item()),), device=pos_items.device)
        conflict = valid_pos & (neg == pos_items)
    neg = torch.where(valid_pos, neg, torch.zeros_like(neg))
    return neg


def batched_indices(indices: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


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
    self_mask = users.unsqueeze(1) == active_indices.unsqueeze(0)
    sims = sims.masked_fill(self_mask, -1e9)
    top_scores, top_positions = torch.topk(sims, topk, dim=1)
    neighbor_ids = active_indices[top_positions]
    retrieved = memory_bank[neighbor_ids]
    retrieval_mask = active_clients[users] & (top_scores[:, 0] > -1e8)
    return retrieved, retrieval_mask


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
    self_mask = users.unsqueeze(1) == active_indices.unsqueeze(0)
    sims = sims.masked_fill(self_mask, -1e9)
    top_scores, top_positions = torch.topk(sims, topk, dim=1)
    neighbor_ids = active_indices[top_positions]
    score_mask = top_scores >= min_sim
    valid_mask = active_clients[users] & score_mask.any(dim=1)
    temp = max(float(temperature), 1e-6)
    safe_scores = (top_scores / temp).masked_fill(~score_mask, -1e9)
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
    cfg: FastConfig,
) -> None:
    if method == "FedDIEG" and cfg.feddieg_memory_momentum > 0:
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


@torch.no_grad()
def evaluate_model(
    method: str,
    model: nn.Module,
    users: torch.Tensor,
    seqs: torch.Tensor,
    items: torch.Tensor,
    data: FastData,
    cfg: FastConfig,
    memory_bank: Optional[torch.Tensor] = None,
    active_clients: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    model.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total_recall = 0.0
    total = 0
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
        if method == "FedMF":
            feats = model(batch_users, batch_seqs)
        elif method == "GPFedRec" and norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(batch_users, memory_bank, active_clients, norm_bank, cfg.gpfedrec_neighbors)
            local_feats = model(batch_seqs)
            if retrieved is not None and retrieval_mask is not None:
                neighbor_feat = retrieved.mean(dim=1).unsqueeze(1).expand_as(local_feats)
                feats = local_feats + cfg.gpfedrec_mix * retrieval_mask.float().view(-1, 1, 1) * neighbor_feat
            else:
                feats = local_feats
        elif method == "FCUCR" and norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(batch_users, memory_bank, active_clients, norm_bank, cfg.fcucR_kb_len)
            kb_feats = None
            if retrieved is not None and retrieval_mask is not None:
                kb_feats = retrieved.mean(dim=1) * retrieval_mask.float().unsqueeze(-1)
            feats, _ = model(batch_seqs, kb_feats)
        elif method == "TarDGR" and norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(batch_users, memory_bank, active_clients, norm_bank, cfg.tardgr_retrieval_topk)
            feats, _ = model(batch_seqs, retrieved, retrieval_mask)
        elif method == "F3CRec" and norm_bank is not None:
            retrieved, retrieval_mask = build_retrieved_memory(batch_users, memory_bank, active_clients, norm_bank, cfg.f3crec_replay_topk)
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
                short_summary = None
        score_feats = feats[:, -1, :]
        if method == "FedDIEG" and short_summary is not None and hasattr(model, "score_representation"):
            score_feats = model.score_representation(score_feats, short_summary)
        scores = torch.matmul(score_feats.unsqueeze(1), model.item_emb(batch_items).transpose(-1, -2)).squeeze(1)
        _, top_indices = torch.topk(scores, cfg.top_k, dim=1)
        hits = top_indices == 0
        found = hits.any(dim=1)
        if found.any():
            ranks = hits.float().argmax(dim=1)
            ndcg = torch.where(found, math.log(2.0) / torch.log(ranks.float() + 2.0), torch.zeros_like(ranks.float()))
        else:
            ndcg = torch.zeros(batch_users.size(0), device=users.device)
        total_hr += found.float().sum().item()
        total_ndcg += ndcg.sum().item()
        total_recall += found.float().sum().item()
        total += batch_users.size(0)
    model.train()
    return total_hr / total, total_ndcg / total, total_recall / total


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def make_round_record(
    method: str,
    round_idx: int,
    train_loss: float,
    val_metrics: Tuple[float, float, float],
    test_metrics: Tuple[float, float, float],
    seconds: float,
) -> Dict[str, object]:
    return {
        "method": method,
        "round": round_idx,
        "train_loss": train_loss,
        "val_hr": val_metrics[0],
        "val_ndcg": val_metrics[1],
        "val_recall": val_metrics[2],
        "test_hr": test_metrics[0],
        "test_ndcg": test_metrics[1],
        "test_recall": test_metrics[2],
        "seconds": seconds,
    }


def effective_lr(method: str, cfg: FastConfig) -> float:
    if method == "FedDIEG" and cfg.feddieg_lr is not None:
        return cfg.feddieg_lr
    return cfg.lr


def make_summary(
    method: str,
    cfg: FastConfig,
    clients_per_round: int,
    best_round: int,
    best_val_ndcg: float,
    best_test: Tuple[float, float, float],
    elapsed: float,
) -> Dict[str, object]:
    return {
        "method": method,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "rounds": cfg.rounds,
        "clients_per_round": clients_per_round,
        "microbatch_size": cfg.microbatch_size,
        "lr": effective_lr(method, cfg),
        "best_round": best_round,
        "best_val_ndcg": best_val_ndcg,
        "best_test_hr": best_test[0],
        "best_test_ndcg": best_test[1],
        "best_test_recall": best_test[2],
        "total_time_seconds": elapsed,
    }


def run_method(method: str, data: FastData, cfg: FastConfig, device: torch.device) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    print(f"\n[{method}] fast GPU start")
    set_seed(cfg.seed)
    if method == "FedMF":
        model: nn.Module = FedMFModel(data.num_users, data.num_items, cfg).to(device)
    elif method == "FedSASRec":
        model: nn.Module = FedSASRecModel(data.num_items, cfg).to(device)
    elif method == "GPFedRec":
        model = FedSASRecModel(data.num_items, cfg).to(device)
    elif method == "FCUCR":
        model = FCUCRModel(data.num_items, cfg).to(device)
    elif method == "TarDGR":
        model = TarDGRModel(data.num_items, cfg).to(device)
    elif method == "F3CRec":
        model = F3CRecModel(data.num_items, cfg).to(device)
    elif method == "DGFedRS":
        model = DGFedRSModel(data.num_items, cfg).to(device)
    elif method == "FedDIEG":
        model = FedDIEGModel(data.num_items, cfg).to(device)
    else:
        raise ValueError(f"Unsupported method: {method}")
    method_lr = effective_lr(method, cfg)
    if method == "FedDIEG" and cfg.feddieg_lr is not None:
        print(f"[{method}] using method-specific lr={method_lr}")
    optimizer = optim.Adam(model.parameters(), lr=method_lr, betas=(0.9, 0.98))
    clients_per_round = cfg.clients_per_round or max(int(cfg.participation_rate * data.num_users), 1)
    clients_per_round = min(clients_per_round, data.num_users)
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    memory_bank = torch.zeros((data.num_users, cfg.latent_dim), device=device)
    active_clients = torch.zeros(data.num_users, dtype=torch.bool, device=device)

    records: List[Dict[str, object]] = []
    best_val_ndcg = -1.0
    best_round = 0
    best_test = (0.0, 0.0, 0.0)
    start_time = time.time()
    last_eval_time = start_time

    for round_idx in range(1, cfg.rounds + 1):
        round_start = time.time()
        sampled = torch.randperm(data.num_users, generator=generator, device=device)[:clients_per_round]
        norm_bank = None
        use_memory = (
            method == "GPFedRec" and round_idx > cfg.gpfedrec_warmup_rounds and active_clients.sum().item() > 1
        ) or (
            method == "TarDGR" and round_idx > cfg.tardgr_warmup_rounds and active_clients.sum().item() > 1
        ) or (
            method == "F3CRec" and round_idx > cfg.f3crec_warmup_rounds and active_clients.sum().item() > 1
        ) or (
            method == "FedDIEG" and round_idx > cfg.feddieg_warmup_rounds and active_clients.sum().item() > 1
        ) or (
            method == "FCUCR" and active_clients.sum().item() > 1
        )
        if use_memory:
            norm_bank = F.normalize(memory_bank, p=2, dim=1)
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
                            users,
                            memory_bank,
                            active_clients,
                            norm_bank,
                            cfg.gpfedrec_neighbors,
                        )
                        if retrieved is not None and retrieval_mask is not None:
                            neighbor_feat = retrieved.mean(dim=1).unsqueeze(1).expand_as(local_feats)
                            feats = local_feats + cfg.gpfedrec_mix * retrieval_mask.float().view(-1, 1, 1) * neighbor_feat
                elif method == "FCUCR":
                    kb_feats = None
                    if use_memory and norm_bank is not None:
                        retrieved, retrieval_mask = build_retrieved_memory(
                            users,
                            memory_bank,
                            active_clients,
                            norm_bank,
                            cfg.fcucR_kb_len,
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
                            users,
                            memory_bank,
                            active_clients,
                            norm_bank,
                            cfg.tardgr_retrieval_topk,
                        )
                    feats, local_feats = model(seq, retrieved, retrieval_mask)
                elif method == "F3CRec":
                    local_feats = model(seq)
                    feats = local_feats
                    if use_memory and norm_bank is not None:
                        retrieved, retrieval_mask = build_retrieved_memory(
                            users,
                            memory_bank,
                            active_clients,
                            norm_bank,
                            cfg.f3crec_replay_topk,
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
                            final_pos_logits,
                            torch.ones_like(final_pos_logits),
                        ) + F.binary_cross_entropy_with_logits(
                            final_neg_logits,
                            torch.zeros_like(final_neg_logits),
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
        print(f"[{method}] Round {round_idx:03d}/{cfg.rounds} | Loss {avg_loss:.4f} | {time.time() - round_start:.2f}s", end="")
        if round_idx % cfg.eval_every == 0:
            val_metrics = evaluate_model(
                method,
                model,
                data.val_users,
                data.val_seq,
                data.val_items,
                data,
                cfg,
                memory_bank,
                active_clients,
            )
            test_metrics = evaluate_model(
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
            records.append(make_round_record(method, round_idx, avg_loss, val_metrics, test_metrics, seconds))
            print(f" | Val NDCG {val_metrics[1]:.4f} | Test NDCG {test_metrics[1]:.4f} | eval window {seconds:.1f}s")
            if val_metrics[1] > best_val_ndcg:
                best_val_ndcg = val_metrics[1]
                best_round = round_idx
                best_test = test_metrics
        else:
            print()
    summary = make_summary(method, cfg, clients_per_round, best_round, best_val_ndcg, best_test, time.time() - start_time)
    del optimizer, model, memory_bank, active_clients
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast GPU-batched Experiment 1 prototype.")
    parser.add_argument("--dataset", default="amazon-digital music")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS),
        choices=list(ALL_METHODS),
    )
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--participation-rate", type=float, default=0.1)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--feddieg-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp_01_fast_gpu" / "results_digital_music_fast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FastConfig(
        dataset=args.dataset,
        methods=tuple(args.methods),
        seed=args.seed,
        rounds=args.rounds,
        eval_every=args.eval_every,
        participation_rate=args.participation_rate,
        clients_per_round=args.clients_per_round,
        microbatch_size=args.microbatch_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_users=args.max_eval_users,
        local_epochs=args.local_epochs,
        lr=args.lr,
        feddieg_lr=args.feddieg_lr,
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
    )
    torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "fast_gpu.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {json.dumps(asdict(cfg), ensure_ascii=False, indent=2)}")
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2)

    print(f"\nLoading {cfg.dataset} into GPU tensors...")
    data = load_fast_data(cfg.dataset, cfg.max_len).to(device)
    print(
        f"Loaded dataset={cfg.dataset}, users={data.num_users}, items={data.num_items}, "
        f"train_shape={tuple(data.train_seq.shape)}, val={data.val_users.numel()}, test={data.test_users.numel()}"
    )

    all_records: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for method in cfg.methods:
        records, summary = run_method(method, data, cfg, device)
        all_records.extend(records)
        summaries.append(summary)
        write_csv(out_dir / "round_metrics.csv", all_records, ROUND_FIELDS)
        write_csv(out_dir / "summary.csv", summaries, SUMMARY_FIELDS)
        print(f"[{method}] outputs updated: round_metrics.csv, summary.csv")
    print(f"\nDone. Final outputs are in: {out_dir}")


if __name__ == "__main__":
    main()
