#!/usr/bin/env python
"""验证实验 2（val02）：兴趣真实变化时，固定 EMA 是否适应太慢？

对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验二」。

⚠️ 2026-11-05 重写说明（必读）
------------------------------
1. **训练协议换成真正的 FedAvg**（客户端本地训练 + 服务器参数聚合），
   对齐 `FedDIEG.py` / `exp_01/run_exp01_movielens_aligned.py`。
   旧的"单模型 + 单 Adam 逐 microbatch 更新"已废弃（见审计报告 §4）。
2. **动态图换成论文公式的正确落地**（`feddieg_core.graph`）：
   EMA 量纲 = max(0,cos)；无效槽位写 -1；不再有 scatter 别名；
   EMA 状态保存在候选池上，掉出 Top-K 的边不会被清零（否则图会冻结）。
   旧的 `--graph-gate` / `--new-edge-init` / `--graph-strength` 已由
   `--graph-state` / `--graph-new-edge` / `--graph-mix` 取代。
3. **ρ 的默认值改成 0.2**（论文 `FedDIEG.py` L56 `RHO=0.2`）。任务书要扫的
   0 / 0.5 / 0.8 / 0.95 用 `--rho` 指定；建议把 0.2 也跑一遍作为论文默认点。
4. **参与率默认 0.1**（论文 C=0.1），不再默认全员参与。

一次运行 = 一个 (数据集, EMA 系数 ρ, 随机种子)。
四个 ρ 共用**同一份** cohort（`val02_detect_drift.py` 产出）与同一份窗口数据，
因此 4 条曲线是同一批用户的配对比较。

记录什么
--------
  (a) 全体 / drift 组 / stable 组的逐轮 NDCG@10（→ 恢复曲线）；
  (b) drift 组邻居里"旧兴趣同伴"与"新兴趣同伴"的边权占比；
  (c) drift 组邻居集合相对 onset 前的 turnover；
  (d) 每轮边变化率 ECR（全体 / drift / stable）、平均新进入边数。

输出（`results/raw/<dataset>/rho<ρ>/seed_<seed>/`）
-------------------------------------------------
  config.json / val02_round_metrics.csv / val02_user_round_metrics.csv
  val02_summary.csv / val02_run.log

用法
----
  python -X utf8 exp_val_1026/exp02_ema_interest_drift/val02_run_ema.py --smoke
  python .../val02_run_ema.py --rho 0.8 --seed 42
  python .../val02_run_ema.py --rho 0.2 --seed 42            # 论文默认
  python .../val02_run_ema.py --rho 0.8 --graph-state topk   # 复现旧的"冻结"行为
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
VAL02_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feddieg_core import __version__ as CORE_VERSION
from feddieg_core.cli import add_feddieg_args, config_from_args, prepare_runtime
from feddieg_core.config import FedDIEGConfig
from feddieg_core.data import ClientWindow, load_client_windows
from feddieg_core.evaluate import build_eval_model, evaluate_split
from feddieg_core.federated import FedTrainer, build_participation_schedule
from feddieg_core.graph import DynamicInterestGraph
from feddieg_core.utils import (
    dump_json,
    make_device,
    masked_mean,
    mean,
    open_log,
    safe_slug,
    set_seed,
    warn_if_legacy_results,
    write_csv,
)

DEFAULT_COHORT_ROOT = VAL02_DIR / "results" / "drift"
DEFAULT_OUT_ROOT = VAL02_DIR / "results" / "raw"


ROUND_FIELDS = [
    "dataset", "rho", "seed", "window", "window_round", "global_round",
    "train_loss", "all_ndcg", "all_hr", "n_eval",
    "drift_ndcg", "drift_hr", "n_drift_eval",
    "stable_ndcg", "stable_hr", "n_stable_eval",
    "ecr_all", "ecr_drift", "ecr_stable",
    "drift_old_share", "drift_new_share", "drift_turnover", "drift_cur_share",
    "drift_cur_base", "drift_n_neighbors",
    # ---- 新增：用"新进入边数"区分"适应慢"与"图冻结" ----
    "added_edges_mean", "n_trained", "n_uploaded",
]

USER_FIELDS = [
    "dataset", "rho", "seed", "user", "cohort", "onset_window", "primary_drift",
    "cat_old", "cat_new", "window", "window_round", "global_round", "rel_round",
    "ndcg", "hr", "n_neighbors", "old_peer_share", "new_peer_share",
    "turnover_from_pre", "cur_peer_share", "cur_peer_base",
]

SUMMARY_FIELDS = [
    "dataset", "rho", "seed", "graph_gate", "new_edge_init", "graph_strength",
    "graph_mix", "edge_ema", "num_windows", "rounds_per_window", "total_rounds",
    "clients_per_round", "n_drift", "n_stable", "all_avg_ndcg", "all_final_ndcg",
    "avg_ecr", "elapsed_seconds",
    # ---- 新增 ----
    "core_version", "local_epochs", "fedavg_weighting", "personalization",
    "graph_state", "graph_update_scope", "neighbors", "summary_momentum",
    "avg_added_edges", "eval_seconds", "train_seconds",
]


# ======================================================================================
# cohort：drift / stable 用户 + 各边界的"同伴主类别"表
# ======================================================================================
class Cohort:
    def __init__(
        self,
        cohort: np.ndarray,
        onset: np.ndarray,
        primary: np.ndarray,
        cat_old: np.ndarray,
        cat_new: np.ndarray,
        users: np.ndarray,
        pre_dom: Dict[int, np.ndarray],
        post_dom: Dict[int, np.ndarray],
        win_dom: Dict[int, np.ndarray],
        win_freq: Dict[int, np.ndarray],
        num_categories: int,
        peer_boundaries: List[int],
    ):
        self.cohort = cohort
        self.onset = onset
        self.primary = primary
        self.cat_old = cat_old
        self.cat_new = cat_new
        self.users = users
        self.pre_dom = pre_dom
        self.post_dom = post_dom
        self.win_dom = win_dom
        self.win_freq = win_freq
        self.num_categories = num_categories
        self.peer_boundaries = peer_boundaries

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
        raise SystemExit(f"找不到类别计数文件：{counts_file}\n请先运行 val02_detect_drift.py。")
    frame = pd.read_csv(cohort_file)
    counts = np.load(counts_file)["counts"]  # (U, W, K)
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
        b = int(np.clip(boundary, 2, num_windows))
        pre_dom[b] = _dominant(counts[:, : b - 1, :].sum(axis=1))
        post_dom[b] = _dominant(counts[:, b - 1 :, :].sum(axis=1))

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
        cohort=cohort, onset=onset, primary=primary, cat_old=cat_old, cat_new=cat_new,
        users=users, pre_dom=pre_dom, post_dom=post_dom, win_dom=win_dom, win_freq=win_freq,
        num_categories=num_categories, peer_boundaries=peer_boundaries,
    )


# ======================================================================================
# cohort 的边级指标
# ======================================================================================
class CohortEdgeProbe:
    """按 onset 分组，算 drift/stable 用户的"旧/新兴趣同伴权重占比"与 turnover。"""

    def __init__(self, cohort: Cohort, device: torch.device):
        self.cohort = cohort
        self.device = device
        self.users = torch.as_tensor(cohort.users, dtype=torch.long, device=device)
        self.cat_old = torch.as_tensor(cohort.cat_old, dtype=torch.long, device=device)
        self.cat_new = torch.as_tensor(cohort.cat_new, dtype=torch.long, device=device)
        self.pre_dom = {
            int(b): torch.as_tensor(d, dtype=torch.long, device=device)
            for b, d in cohort.pre_dom.items()
        }
        self.post_dom = {
            int(b): torch.as_tensor(d, dtype=torch.long, device=device)
            for b, d in cohort.post_dom.items()
        }
        self.win_dom = {
            int(w): torch.as_tensor(d, dtype=torch.long, device=device)
            for w, d in cohort.win_dom.items()
        }
        self.win_freq = {
            int(w): torch.as_tensor(f, dtype=torch.float32, device=device)
            for w, f in cohort.win_freq.items()
        }
        self.ref_neighbors: Optional[torch.Tensor] = None

    def init_ref(self, num_users: int, topk: int) -> None:
        self.ref_neighbors = torch.full((num_users, topk), -1, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def snapshot(self, graph: DynamicInterestGraph, window: int) -> None:
        """窗口开始时，给 onset == window 的 cohort 用户存一份"变化前"的邻居快照。"""
        if self.ref_neighbors is None:
            return
        sel = np.flatnonzero((self.cohort.cohort > 0) & (self.cohort.onset == window))
        if sel.size == 0:
            return
        users = torch.as_tensor(sel, dtype=torch.long, device=self.device)
        self.ref_neighbors[users] = graph.ids[users]

    @torch.no_grad()
    def probe(
        self,
        users: np.ndarray,
        graph: DynamicInterestGraph,
        window: int,
        need_turnover: bool,
    ) -> Dict[str, np.ndarray]:
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
        nbr = graph.ids[user_ids]                                        # (G, K)，无效位是 -1
        slot_valid = nbr >= 0
        user_valid = graph.mask[user_ids]
        w = graph.weights[user_ids] * slot_valid.float() * user_valid.float().unsqueeze(1)
        denom = w.sum(dim=1).clamp_min(1e-8)
        n_valid = slot_valid.float().sum(dim=1)

        window_idx = int(np.clip(window, 1, len(self.win_dom)))
        if window_idx in self.win_dom:
            dom = self.win_dom[window_idx].unsqueeze(0)
            user_dom = self.win_dom[window_idx][user_ids]
            match_cur = (
                (dom == user_dom.unsqueeze(1)) & (user_dom.unsqueeze(1) >= 0) & (dom >= 0)
            ).float()
            gathered_cur = match_cur.gather(1, nbr.clamp_min(0))
            result["cur_peer_share"][member] = ((gathered_cur * w).sum(dim=1) / denom).cpu().numpy()
            freq = self.win_freq[window_idx]
            result["cur_peer_base"][member] = freq[user_dom.clamp_min(0)].cpu().numpy()

        onsets = self.cohort.onset[users[member]]
        old_share = torch.full((user_ids.numel(),), float("nan"), device=self.device)
        new_share = torch.full((user_ids.numel(),), float("nan"), device=self.device)
        for boundary in np.unique(onsets):
            boundary = int(boundary)
            if boundary not in self.pre_dom:
                continue
            sel = torch.as_tensor(np.flatnonzero(onsets == boundary), dtype=torch.long, device=self.device)
            sel_users = user_ids[sel]
            old_cat = self.cat_old[sel_users].unsqueeze(1)
            new_cat = self.cat_new[sel_users].unsqueeze(1)
            pre_dom = self.pre_dom[boundary].unsqueeze(0)
            post_dom = self.post_dom[boundary].unsqueeze(0)
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
            ref = self.ref_neighbors[user_ids]
            eq = (ref.unsqueeze(2) == nbr.unsqueeze(1)) & (ref.unsqueeze(2) >= 0) & (nbr.unsqueeze(1) >= 0)
            inter = eq.any(dim=2).sum(dim=1).float()
            n_ref = (ref >= 0).sum(dim=1).float()
            n_cur = n_valid
            union = (n_ref + n_cur - inter).clamp_min(1e-8)
            jaccard = inter / union
            turnover = torch.where(n_ref > 0, 1.0 - jaccard, torch.full_like(jaccard, float("nan")))
            result["turnover_from_pre"][member] = turnover.cpu().numpy()
        return result


# ======================================================================================
# 单次运行
# ======================================================================================
def run_single(
    cfg: FedDIEGConfig,
    windows: List[ClientWindow],
    cohort: Cohort,
    device: torch.device,
    out_dir: Path,
) -> None:
    set_seed(cfg.seed)
    num_users = windows[0].num_users
    num_items = windows[0].num_items
    clients_per_round = cfg.resolved_clients_per_round(num_users)
    schedule = build_participation_schedule(
        num_users, cfg.total_rounds, clients_per_round, cfg.seed
    )

    trainer = FedTrainer(cfg, num_users, num_items, device)
    graph = DynamicInterestGraph(cfg, num_users, device)
    eval_model = build_eval_model(trainer, cfg)
    rng = np.random.default_rng(cfg.seed)

    print(
        f"[rho={cfg.rho:.2f}] 联邦设置：N={num_users}, m={clients_per_round} "
        f"(C={cfg.participation_rate}), E={cfg.local_epochs}, "
        f"personalization={cfg.personalization}, FedAvg={cfg.fedavg_weighting}"
    )
    print(f"[rho={cfg.rho:.2f}] 图设置：{graph.describe()}")

    probe = CohortEdgeProbe(cohort, device)
    probe.init_ref(num_users, graph.k)

    is_drift = cohort.cohort == 1
    is_stable = cohort.cohort == 2
    cohort_users = cohort.users
    onset_of = cohort.onset

    round_rows: List[Dict[str, object]] = []
    user_rows: List[Dict[str, object]] = []
    window_ndcgs: List[float] = []
    window_ecrs: List[float] = []
    window_added: List[float] = []
    eval_time = 0.0
    train_time = 0.0
    start_time = time.time()
    global_round = 0

    for window_idx, window in enumerate(windows, start=1):
        print(
            f"\n[rho={cfg.rho:.2f}] Window {window_idx}/{cfg.num_windows} | "
            f"eval_users={window.test_users.numel()}"
        )
        probe.snapshot(graph, window_idx)
        last_ndcg = 0.0

        for window_round in range(1, cfg.rounds_per_window + 1):
            global_round += 1
            round_start = time.time()
            sampled = schedule[global_round - 1]

            stats = trainer.train_round(sampled, window, graph, global_round, rng)

            if cfg.graph_update_scope == "all":
                update_users = torch.arange(num_users, device=device)
            else:
                update_users = torch.tensor(sampled, dtype=torch.long, device=device)
            gstats = graph.update(
                trainer.summary_bank, trainer.summary_active, update_users, global_round
            )
            train_time += time.time() - round_start

            should_eval = (window_round % cfg.eval_every == 0) or (window_round == cfg.rounds_per_window)
            if not should_eval:
                continue

            eval_start = time.time()
            result = evaluate_split(trainer, eval_model, window, graph, "test", cfg)
            user_ids = result["users"]
            test_hr, test_ndcg = result["hr"], result["ndcg"]
            eval_time += time.time() - eval_start

            drift_sel = is_drift[user_ids] if user_ids.size else np.zeros(0, dtype=bool)
            stable_sel = is_stable[user_ids] if user_ids.size else np.zeros(0, dtype=bool)
            all_ndcg = float(test_ndcg.mean()) if test_ndcg.size else 0.0
            all_hr = float(test_hr.mean()) if test_hr.size else 0.0
            drift_ndcg = masked_mean(test_ndcg, drift_sel)
            drift_hr = masked_mean(test_hr, drift_sel)
            stable_ndcg = masked_mean(test_ndcg, stable_sel)
            stable_hr = masked_mean(test_hr, stable_sel)
            last_ndcg = all_ndcg

            if gstats is not None and len(gstats) > 0:
                upd = gstats.updated.numpy()
                ecr = gstats.ecr.numpy()
                ecr_all = float(ecr.mean())
                ecr_drift = masked_mean(ecr, is_drift[upd])
                ecr_stable = masked_mean(ecr, is_stable[upd])
                added_mean = float(gstats.n_added.mean())
            else:
                ecr_all = ecr_drift = ecr_stable = 0.0
                added_mean = 0.0

            present = (
                user_ids[np.isin(user_ids, cohort_users)]
                if user_ids.size
                else np.zeros(0, dtype=np.int64)
            )
            edge_stats = probe.probe(present, graph, window_idx, need_turnover=True)
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
                        "dataset": cfg.dataset, "rho": cfg.rho, "seed": cfg.seed,
                        "user": int(user),
                        "cohort": "drift" if cohort_id == 1 else "stable",
                        "onset_window": onset,
                        "primary_drift": int(cohort.primary[user]),
                        "cat_old": int(cohort.cat_old[user]),
                        "cat_new": int(cohort.cat_new[user]),
                        "window": window_idx, "window_round": window_round,
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
                    for key, bucket in (
                        ("old_peer_share", drift_edges_old),
                        ("new_peer_share", drift_edges_new),
                        ("turnover_from_pre", drift_edges_turn),
                    ):
                        value = edge_stats[key][pos]
                        if np.isfinite(value):
                            bucket.append(float(value))
                cur_v = edge_stats["cur_peer_share"][pos]
                cur_b = edge_stats["cur_peer_base"][pos]
                if cohort_id == 1 and np.isfinite(cur_v) and np.isfinite(cur_b):
                    drift_cur_share.append(float(cur_v))
                    drift_cur_base.append(float(cur_b))

            round_rows.append(
                {
                    "dataset": cfg.dataset, "rho": cfg.rho, "seed": cfg.seed,
                    "window": window_idx, "window_round": window_round,
                    "global_round": global_round,
                    "train_loss": stats.loss,
                    "all_ndcg": all_ndcg, "all_hr": all_hr, "n_eval": int(test_ndcg.size),
                    "drift_ndcg": drift_ndcg, "drift_hr": drift_hr,
                    "n_drift_eval": int(drift_sel.sum()),
                    "stable_ndcg": stable_ndcg, "stable_hr": stable_hr,
                    "n_stable_eval": int(stable_sel.sum()),
                    "ecr_all": ecr_all, "ecr_drift": ecr_drift, "ecr_stable": ecr_stable,
                    "drift_old_share": mean(drift_edges_old),
                    "drift_new_share": mean(drift_edges_new),
                    "drift_turnover": mean(drift_edges_turn),
                    "drift_cur_share": mean(drift_cur_share),
                    "drift_cur_base": mean(drift_cur_base),
                    "drift_n_neighbors": float(np.mean(edge_stats["n_neighbors"]))
                    if edge_stats["n_neighbors"].size
                    else 0.0,
                    "added_edges_mean": added_mean,
                    "n_trained": stats.n_trained,
                    "n_uploaded": stats.n_uploads,
                }
            )
            window_ecrs.append(ecr_all)
            window_added.append(added_mean)
            print(
                f"[rho={cfg.rho:.2f}] W{window_idx} R{window_round:03d}/{cfg.rounds_per_window} | "
                f"loss {stats.loss:.4f} | trained {stats.n_trained}/{stats.n_clients} | "
                f"ECR {ecr_all:.4f} (+{added_mean:.2f}) | "
                f"NDCG@{cfg.top_k} all {all_ndcg:.4f} drift {drift_ndcg:.4f} "
                f"(n={int(drift_sel.sum())}) stable {stable_ndcg:.4f} | "
                f"oldpeer {mean(drift_edges_old):.3f} turn {mean(drift_edges_turn):.3f} | "
                f"{time.time() - round_start:.2f}s"
            )

        window_ndcgs.append(last_ndcg)
        write_csv(out_dir / "val02_round_metrics.csv", round_rows, ROUND_FIELDS)
        write_csv(out_dir / "val02_user_round_metrics.csv", user_rows, USER_FIELDS)
        print(f"[rho={cfg.rho:.2f}] W{window_idx} end | NDCG@{cfg.top_k} {last_ndcg:.4f} | rows={len(user_rows)}")

    summary = {
        "dataset": cfg.dataset, "rho": cfg.rho, "seed": cfg.seed,
        "graph_gate": "raw (余弦量纲)", "new_edge_init": cfg.graph_new_edge,
        "graph_strength": f"graph_mix={cfg.graph_mix}",
        "graph_mix": cfg.graph_mix, "edge_ema": cfg.rho,
        "num_windows": cfg.num_windows, "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": cfg.total_rounds, "clients_per_round": clients_per_round,
        "n_drift": int(is_drift.sum()), "n_stable": int(is_stable.sum()),
        "all_avg_ndcg": mean(window_ndcgs),
        "all_final_ndcg": window_ndcgs[-1] if window_ndcgs else 0.0,
        "avg_ecr": mean(window_ecrs),
        "elapsed_seconds": time.time() - start_time,
        "core_version": CORE_VERSION,
        "local_epochs": cfg.local_epochs,
        "fedavg_weighting": cfg.fedavg_weighting,
        "personalization": cfg.personalization,
        "graph_state": cfg.graph_state,
        "graph_update_scope": cfg.graph_update_scope,
        "neighbors": cfg.neighbors,
        "summary_momentum": cfg.summary_momentum,
        "avg_added_edges": mean(window_added),
        "eval_seconds": eval_time,
        "train_seconds": train_time,
    }
    write_csv(out_dir / "val02_summary.csv", [summary], SUMMARY_FIELDS)
    print(
        f"\n[rho={cfg.rho:.2f}] seed{cfg.seed} done in {summary['elapsed_seconds']:.1f}s | "
        f"avg_ndcg={summary['all_avg_ndcg']:.4f} | avg_ecr={summary['avg_ecr']:.4f}"
    )


# ======================================================================================
# CLI
# ======================================================================================
def make_out_dir(out_root: Path, dataset: str, rho: float, seed: int) -> Path:
    return out_root / safe_slug(dataset) / f"rho{rho:.2f}" / f"seed_{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="val02: 兴趣真实变化时固定 EMA 是否适应太慢（FedAvg 版）")
    parser.add_argument("--primary-drift-only", action="store_true", help="drift 组只保留主类别确实改变的用户")
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--cohort-file", type=Path, default=None)
    parser.add_argument("--counts-file", type=Path, default=None)
    add_feddieg_args(parser)
    parser.set_defaults(out_root=DEFAULT_OUT_ROOT, rho=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.validate()
    prepare_runtime(cfg)
    set_seed(cfg.seed)
    device = make_device(cfg.device)

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else make_out_dir(args.out_root, cfg.dataset, cfg.rho, cfg.seed)
    )
    if args.smoke and args.out_dir is None:
        # smoke 输出单独放到 <out_root>_smoke，避免污染正式结果的聚合脚本
        out_dir = Path(str(args.out_root) + "_smoke") / out_dir.relative_to(args.out_root)
    warn_if_legacy_results(out_dir)
    open_log(out_dir, "val02_run.log")

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print("Config:", json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str))
    dump_json(out_dir / "config.json", {**asdict(cfg), "core_version": CORE_VERSION})

    slug = safe_slug(cfg.dataset)
    cohort_file = (args.cohort_file or (args.cohort_root / slug / "val02_drift_users.csv")).resolve()
    counts_file = (args.counts_file or (args.cohort_root / slug / "val02_counts.npz")).resolve()
    print(f"Cohort file : {cohort_file}")
    print(f"Counts file : {counts_file}")
    cohort = load_cohort(cohort_file, counts_file, args.primary_drift_only)

    windows = load_client_windows(cfg)
    run_single(cfg, windows, cohort, device, out_dir)
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
