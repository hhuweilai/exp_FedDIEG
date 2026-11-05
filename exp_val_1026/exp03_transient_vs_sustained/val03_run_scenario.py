#!/usr/bin/env python
"""验证实验 3（val03）第 2 步：跑一个 (数据集, 场景, EMA 系数 ρ, 随机种子) 训练。

对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验三」
（"短期兴趣波动会不会导致不必要的图更新？"）。

⚠️ 2026-11-05 重写说明（必读）
------------------------------
1. **训练协议换成真正的 FedAvg**（客户端本地训练 + 服务器参数聚合），对齐
   `FedDIEG.py` / `exp_01/run_exp01_movielens_aligned.py`。旧的"单模型 + 单 Adam
   逐 microbatch 更新"已废弃（审计报告 §4）。
2. **动态图换成论文公式的正确落地**（`feddieg_core.graph`）：EMA 量纲 = max(0,cos)，
   无效槽位写 -1，无 scatter 别名，EMA 状态存在候选池上（修掉"图冻结"）。
   旧的 `--graph-gate` / `--new-edge-init` / `--graph-strength` 由
   `--graph-state` / `--graph-new-edge` / `--graph-mix` 取代。
3. **ρ 默认 0.2**（论文 `RHO=0.2`）；**参与率默认 0.1**（论文 C=0.1）。
4. 场景构造（`val03_build_scenarios.py`）、注入流程、`GroupProbe` 的指标定义**完全没变**，
   因此新旧结果在"测什么"上是对齐的，只是"怎么训"变了。

一次运行同时测量 5 个用户组
---------------------------
  注入 arm（人工构造，见 val03_build_scenarios.py）
    inj               该运行对应场景下的注入组（sustained: 注入持续到最后的窗口；transient: 只注入 onset 窗口）
    pair              配对对照：同样的窗口、同样的替换流程，但实际替换成自己旧主类别的物品
  真实 arm（只打标签，不修改任何交互）
    real_sustained    真实数据里"跳变后保持新类别"的用户
    real_transient    真实数据里"跳变后回到旧类别"的用户（任务书场景 B）
    real_stable       同一 onset、活动量匹配、真实数据里从未跳变的对照

记录什么
--------
  (a) 逐用户逐评测轮 NDCG@10 / HR@10                → 推荐性能损失
  (b) 逐用户逐评测轮 ECR（本轮边变化率）             → 边关系变化程度
  (c) 邻居里"旧兴趣同伴 / 新兴趣同伴"的边权占比
  (d) 当前邻居集合与 onset 前快照的 Jaccard 距离（turnover_from_pre）
  (e) summary_drift = 1 − cos(当前上传摘要, onset 前快照)

输出（`results/raw/<dataset>/<scenario>/rho<ρ>/seed_<seed>/`）
-------------------------------------------------------------
  config.json / val03_user_round_metrics.csv / val03_round_metrics.csv
  val03_summary.csv / val03_run.log

用法
----
  # 自检（几十秒）
  python -X utf8 exp_val_1026/exp03_transient_vs_sustained/val03_run_scenario.py --smoke --scenario transient

  # 正式跑
  python .../val03_run_scenario.py --dataset movielens-1m --scenario sustained --rho 0.2 --seed 42
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
VAL03_DIR = Path(__file__).resolve().parent
for _path in (str(ROOT), str(VAL03_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from feddieg_core import __version__ as CORE_VERSION
from feddieg_core.cli import add_feddieg_args, config_from_args, prepare_runtime
from feddieg_core.config import FedDIEGConfig
from feddieg_core.data import (
    ClientWindow,
    from_fast_data,
    history_sets_from_frame,
    verify_consistency,
)
from feddieg_core.evaluate import build_eval_model, evaluate_split
from feddieg_core.federated import FedTrainer, build_participation_schedule
from feddieg_core.graph import DynamicInterestGraph
from feddieg_core.utils import (
    warn_if_legacy_results,
    dump_json,
    make_device,
    mean,
    open_log,
    safe_slug,
    set_seed,
)

# 数据窗口协议仍然复用 exp_01/exp_02 的官方实现（数据层面不变，才和旧结果可比）
from exp_01_fast_gpu.run_exp01_fast_gpu import FastData  # noqa: E402
from exp_02_fast_gpu.run_exp02_fast_gpu import (  # noqa: E402
    Exp2Config,
    build_eval_arrays_from_records,
    build_train_arrays_from_sequences,
    load_rolling_windows,
    sample_candidates,
)

DEFAULT_PLAN_ROOT = VAL03_DIR / "results" / "scenarios"
DEFAULT_OUT_ROOT = VAL03_DIR / "results" / "raw"

SCENARIOS = ("sustained", "transient")
GROUPS = ("inj", "pair", "real_sustained", "real_transient", "real_stable")
INJECTED_GROUPS = ("inj", "pair")


# 场景与分组
SCENARIOS = ("sustained", "transient")
GROUPS = ("inj", "pair", "real_sustained", "real_transient", "real_stable")
INJECTED_GROUPS = ("inj", "pair")


USER_FIELDS = [
    "dataset",
    "scenario",
    "rho",
    "seed",
    "group",
    "user",
    "pair_id",
    "onset_window",
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
    "summary_drift",
    "ecr",
]

ROUND_FIELDS = [
    "dataset",
    "scenario",
    "rho",
    "seed",
    "window",
    "window_round",
    "global_round",
    "group",
    "rel_round",
    "n_eval",
    "n_users",
    "ndcg",
    "hr",
    "old_peer_share",
    "new_peer_share",
    "turnover",
    "ecr_mean",
    "summary_drift",
    "train_loss",
]

SUMMARY_FIELDS = [
    "dataset",
    "scenario",
    "rho",
    "seed",
    "graph_gate",
    "new_edge_init",
    "graph_strength",
    "graph_mix",
    "num_windows",
    "rounds_per_window",
    "total_rounds",
    "clients_per_round",
    "inject_onset",
    "inject_windows",
    "n_inj",
    "n_pair",
    "n_real_sustained",
    "n_real_transient",
    "n_real_stable",
    "inject_cos_mean",
    "inject_rows_replaced",
    "all_avg_ndcg",
    "all_final_ndcg",
    "avg_ecr",
    "elapsed_seconds",
    # ---- 2026-11-05 新增：把联邦协议与图配置写进汇总 ----
    "core_version",
    "local_epochs",
    "fedavg_weighting",
    "personalization",
    "graph_state",
    "graph_update_scope",
    "neighbors",
    "summary_momentum",
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


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if values.size == 0 or mask.sum() == 0:
        return float("nan")
    return float(values[mask].mean())


def make_out_dir(
    out_root: Path, dataset: str, scenario: str, rho: float, seed: int
) -> Path:
    return out_root / safe_slug(dataset) / scenario / f"rho{rho:.2f}" / f"seed_{seed}"


# ======================================================================================
# 场景计划：谁被注入 / 谁是配对对照 / 真实队列标签
# ======================================================================================
@dataclass
class Plan:
    users: np.ndarray           # (P,) 计划里所有用户（升序）
    group: np.ndarray           # (P,) 0=inj 1=pair 2=real_sustained 3=real_transient 4=real_stable
    onset: np.ndarray           # (P,) onset 窗口（1 基）
    cat_old: np.ndarray         # (P,)
    cat_new: np.ndarray         # (P,) 指标参照类别（pair 用配对对象的注入目标）
    replace_category: np.ndarray  # (P,) 实际替换成哪个类别（pair = 自己的旧主类别）
    pair_id: np.ndarray         # (P,) inj/pair 用；真实 arm 为 -1
    is_placebo: np.ndarray      # (P,) pair / real_stable 为 1
    inject_onset: int
    num_categories: int
    inject_windows: List[int]
    counts: np.ndarray          # (U, W, K) 供"同伴主类别"查表
    item_category: np.ndarray   # (I,) -1 = 未覆盖
    num_users: int
    num_items: int

    @property
    def group_names(self) -> np.ndarray:
        names = np.array(GROUPS, dtype=object)
        return names[self.group]


GROUP_TO_ID = {name: idx for idx, name in enumerate(GROUPS)}


def load_plan(plan_dir: Path, scenario: str) -> Plan:
    plan_path = plan_dir / "val03_injection_plan.csv"
    real_path = plan_dir / "val03_real_cohorts.csv"
    counts_path = plan_dir / "val03_counts.npz"
    meta_path = plan_dir / "val03_scenario_meta.json"
    for path in (plan_path, counts_path, meta_path):
        if not path.exists():
            raise SystemExit(
                f"缺少 {path}\n请先运行 val03_build_scenarios.py --dataset <数据集>。"
            )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data = np.load(counts_path)
    counts = data["counts"]
    item_category = data["item_category"].astype(np.int64)
    num_users, num_windows, num_cat = counts.shape

    frames = [pd.read_csv(plan_path)]
    if real_path.exists():
        frames.append(pd.read_csv(real_path))
    frame = pd.concat(frames, ignore_index=True)

    users = frame["user"].to_numpy(dtype=np.int64)
    group = np.array([GROUP_TO_ID[str(g)] for g in frame["group"].tolist()], dtype=np.int64)
    onset = frame["onset_window"].to_numpy(dtype=np.int64)
    cat_old = frame["cat_old"].to_numpy(dtype=np.int64)
    cat_new = frame["cat_new"].to_numpy(dtype=np.int64)
    replace_category = (
        frame["replace_category"].fillna(frame["cat_old"]).to_numpy(dtype=np.int64)
        if "replace_category" in frame.columns
        else cat_old.copy()
    )
    is_placebo = (
        frame["is_placebo"].fillna(0).to_numpy(dtype=np.int64)
        if "is_placebo" in frame.columns
        else np.zeros(len(frame), dtype=np.int64)
    )
    if "pair_id" in frame.columns:
        pair_id = frame["pair_id"].fillna(-1).to_numpy(dtype=np.int64)
    else:
        pair_id = np.full(len(frame), -1, dtype=np.int64)

    # 去重：同一个用户只能属于一个组。真实队列优先保留（注入脚本本来就会排除真实漂移用户，
    # 这里只是兜底，避免 plan 被手工改动后出现"既是注入对象又是真实对照"）。
    priority = {name: idx for idx, name in enumerate(("real_sustained", "real_transient", "real_stable", "inj", "pair"))}
    keep: Dict[int, int] = {}
    for row_idx in range(len(users)):
        user = int(users[row_idx])
        name = GROUPS[int(group[row_idx])]
        current = keep.get(user)
        if current is None or priority[name] < priority[GROUPS[int(group[current])]]:
            keep[user] = row_idx
    if len(keep) != len(users):
        print(f"[plan] 去重：{len(users)} → {len(keep)} 行（同一用户只保留一个组）")
        picked = np.array(sorted(keep.values()), dtype=np.int64)
        users, group, onset = users[picked], group[picked], onset[picked]
        cat_old, cat_new = cat_old[picked], cat_new[picked]
        replace_category = replace_category[picked]
        pair_id, is_placebo = pair_id[picked], is_placebo[picked]

    # 按 user 升序（后面用 searchsorted 做 user → 行号 的查表）
    order = np.argsort(users, kind="stable")
    users, group, onset = users[order], group[order], onset[order]
    cat_old, cat_new = cat_old[order], cat_new[order]
    replace_category = replace_category[order]
    pair_id, is_placebo = pair_id[order], is_placebo[order]

    inject_onset = int(meta["inject"]["inject_onset"])
    inject_windows = (
        list(range(inject_onset, num_windows + 1)) if scenario == "sustained" else [inject_onset]
    )
    plan = Plan(
        users=users,
        group=group,
        onset=onset,
        cat_old=cat_old,
        cat_new=cat_new,
        replace_category=replace_category,
        pair_id=pair_id,
        is_placebo=is_placebo,
        inject_onset=inject_onset,
        num_categories=num_cat,
        inject_windows=inject_windows,
        counts=counts,
        item_category=item_category,
        num_users=num_users,
        num_items=int(item_category.shape[0]),
    )
    counts_by_group = ", ".join(
        f"{name}={int((group == GROUP_TO_ID[name]).sum())}" for name in GROUPS
    )
    print(f"[plan] {plan_dir}")
    print(f"[plan] 组规模: {counts_by_group}")
    print(f"[plan] 场景={scenario} → 注入窗口 {inject_windows}")
    return plan


# ======================================================================================
# 数据级注入：把指定窗口里该用户的 itemId 换成目标类别的物品
# ======================================================================================
def _item_pool_by_category(item_category: np.ndarray) -> Dict[int, np.ndarray]:
    pool: Dict[int, np.ndarray] = {}
    for category in np.unique(item_category):
        if category < 0:
            continue
        pool[int(category)] = np.flatnonzero(item_category == int(category))
    return pool


def apply_injections(
    full_df: pd.DataFrame,
    windows: np.ndarray,
    plan: Plan,
    scenario: str,
    inject_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """替换 inj / pair 两组在注入窗口里的 itemId。

    为什么是"替换 itemId"而不是"删/加交互"：
      窗口按行位置等分（`int(N*w/W)`），只改 itemId 不会改变任何一行的窗口归属，
      所以两组的窗口边界、评测用户数、训练轮数完全一致，组间唯一差异就是"兴趣内容"。
    为什么 pair 也要替换：
      pair 的目标类别 = 自己的旧主类别，等价于"物品被重采样、兴趣没变"，
      这样"替换这个动作本身"（物品流行度变了、目标物品变了）被对照掉。
    """
    frame = full_df.copy()
    items = frame["itemId"].to_numpy(dtype=np.int64).copy()
    users_arr = frame["userId"].to_numpy(dtype=np.int64)
    pool_by_cat = _item_pool_by_category(plan.item_category)

    stats = {"rows_replaced": 0, "users_injected": 0, "cos_shift": [], "per_group": {}}
    rng = random.Random(seed * 1000003 + 17)

    for idx in range(plan.users.size):
        group = int(plan.group[idx])
        if group not in (GROUP_TO_ID["inj"], GROUP_TO_ID["pair"]):
            continue
        user = int(plan.users[idx])
        target_cat = int(plan.replace_category[idx])
        pool = pool_by_cat.get(target_cat)
        if pool is None or pool.size == 0:
            continue
        user_rows = np.flatnonzero(users_arr == user)
        if user_rows.size == 0:
            continue
        onset = int(plan.onset[idx])
        # 排除"该用户 onset 之前已经交互过"的物品，避免注入变成"又看了一遍老片"
        pre_rows = user_rows[windows[user_rows] < onset - 1]
        seen = set(items[pre_rows].tolist())
        candidates = np.array([it for it in pool.tolist() if it not in seen], dtype=np.int64)
        if candidates.size == 0:
            candidates = pool

        replaced = 0
        for window_idx in plan.inject_windows:
            window_rows = user_rows[windows[user_rows] == window_idx - 1]
            if window_rows.size == 0:
                continue
            n_replace = max(1, int(round(float(inject_fraction) * window_rows.size)))
            n_replace = min(n_replace, window_rows.size)
            chosen_rows = sorted(rng.sample(range(window_rows.size), n_replace))
            picks = rng.sample(range(candidates.size), min(n_replace, candidates.size))
            for slot, row_offset in enumerate(chosen_rows):
                items[window_rows[row_offset]] = int(candidates[picks[slot % len(picks)]])
            replaced += n_replace
        if replaced > 0:
            stats["rows_replaced"] += replaced
            stats["users_injected"] += 1
            key = "inj" if group == GROUP_TO_ID["inj"] else "pair"
            stats["per_group"][key] = int(stats["per_group"].get(key, 0)) + 1

    frame["itemId"] = items
    print(
        f"[inject] 替换了 {stats['rows_replaced']} 条交互，涉及 {stats['users_injected']} 位用户"
        f"（{stats['per_group']}）；注入窗口 {plan.inject_windows}，比例 {inject_fraction}"
    )
    return frame, stats


# ======================================================================================
# 窗口构造：与 exp02.load_rolling_windows 完全同口径，但允许传入被改过的 full_df
# ======================================================================================
def load_rolling_windows_from_frame(cfg: Exp2Config, full_df: pd.DataFrame) -> List[FastData]:
    """复刻 `load_rolling_windows`，数据源换成已经排好序的 full_df。

    逐行对照 exp02 的实现（含 random.Random(cfg.seed + 202) 的取候选顺序），
    因此 full_df 未被修改时，结果与 `load_rolling_windows(cfg)` 逐元素相同（见 --self-check）。
    """
    processed_dir = ROOT / "datasets" / "processed" / cfg.dataset
    with (processed_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])

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
            val_candidates[user] = sample_candidates(
                set(val_histories[user]) | {val_item}, num_items, cfg.candidate_count, rng
            )
            test_candidates[user] = sample_candidates(
                set(test_histories[user]) | {test_item}, num_items, cfg.candidate_count, rng
            )

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


def build_full_frame(dataset: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """与 exp02.load_rolling_windows / val02.load_interactions 完全一致的排序与窗口切分。"""
    splits_dir = ROOT / "datasets" / "splits" / dataset
    frames = [pd.read_csv(splits_dir / f"loo_{name}.csv") for name in ("train", "val", "test")]
    full_df = pd.concat(frames, ignore_index=True)
    if "timestamp" in full_df.columns:
        full_df = full_df.sort_values(["timestamp", "userId"]).reset_index(drop=True)
    else:
        full_df = full_df.reset_index(drop=True)
        full_df["timestamp"] = full_df.index
    return full_df, full_df["timestamp"].to_numpy()


def frame_windows(num_rows: int, num_windows: int) -> np.ndarray:
    """每行属于哪个窗口（0 基）；与 load_rolling_windows 的 `int(N*w/W)` 切分一致。"""
    bounds = np.array([int(num_rows * w / num_windows) for w in range(num_windows + 1)], dtype=np.int64)
    positions = np.arange(num_rows)
    return np.searchsorted(bounds[1:], positions, side="right")


# ======================================================================================
# 图：训练时取邻居 / 服务器端更新边权（与验证实验二同一实现）
# ======================================================================================
# 动态图与评测都来自 `feddieg_core`（论文公式的正确落地 + FedAvg 评测），
# 这里不再保留旧的 `graph_batch` / `update_graph` / `evaluate_per_user`：
#   * 旧版 `update_graph` 混用量纲（归一化边权 × 原始余弦）、无效槽位写伪 id、
#     `scatter_` 把 -1 钳到列 0、只存 Top-K 导致图冻结 —— 见审计报告 §3；
#   * 旧版评测用快速仿真模型的打分头（tail_power / short_score_mix 等），
#     A/B 主程序里没有这些模块，已统一到 aligned 版的打分方式。

# ======================================================================================
# 评测
# ======================================================================================
# ======================================================================================
# 边级探针：旧/新兴趣同伴占比 + 相对 onset 前的 turnover
# ======================================================================================
class GroupProbe:
    """算计划内用户的邻居构成与"相对 onset 前快照"的调整程度。

    old_peer_share(u,t)：u 的邻居里"在自己 onset 之前主类别 == u 的旧主类别"的边权占比。
                         持续变化场景下它应该下降；短期扰动撤销后它应该回升。
    new_peer_share(u,t)：邻居里"onset 之后主类别 == u 的新类别"的边权占比
                         （真实 arm 的 stable 组 cat_new == cat_old，此时它等于"当前兴趣一致度"）。
    turnover_from_pre(u,t)：当前邻居集合 vs onset 前快照的 1 - Jaccard。
                          transient 组撤销后不回落 = 过度更新；sustained 组不升高 = 更新滞后。
    """

    def __init__(self, plan: Plan, device: torch.device):
        self.plan = plan
        self.device = device
        self.users = torch.as_tensor(plan.users, dtype=torch.long, device=device)
        self.cat_old = torch.as_tensor(plan.cat_old, dtype=torch.long, device=device)
        self.cat_new = torch.as_tensor(plan.cat_new, dtype=torch.long, device=device)
        # user → onset 的全量查表（0 = 不在计划内），避免用行号去索引用户 id
        self.onset_of = np.zeros(int(plan.num_users), dtype=np.int64)
        self.onset_of[plan.users] = plan.onset
        # user → 计划行号
        self.row_of = np.full(int(plan.num_users), -1, dtype=np.int64)
        self.row_of[plan.users] = np.arange(plan.users.size, dtype=np.int64)
        self.row_t = torch.as_tensor(self.row_of, dtype=torch.long, device=device)
        self.onset_np = plan.onset
        self.ref_neighbors: Optional[torch.Tensor] = None
        # 逐边界的"同伴主类别"查表：pre_dom[b][v] = v 在窗口 < b 的主类别
        counts = plan.counts
        num_windows = counts.shape[1]
        self.pre_dom: Dict[int, torch.Tensor] = {}
        self.post_dom: Dict[int, torch.Tensor] = {}
        for boundary in sorted(set(int(b) for b in plan.onset.tolist())):
            boundary = int(np.clip(boundary, 2, num_windows))
            pre = counts[:, : boundary - 1, :].sum(axis=1)
            post = counts[:, boundary - 1 :, :].sum(axis=1)
            self.pre_dom[boundary] = torch.as_tensor(
                np.where(pre.sum(axis=1) > 0, pre.argmax(axis=1), -1), dtype=torch.long, device=device
            )
            self.post_dom[boundary] = torch.as_tensor(
                np.where(post.sum(axis=1) > 0, post.argmax(axis=1), -1), dtype=torch.long, device=device
            )

    def init_ref(self, num_users: int, topk: int) -> None:
        self.ref_neighbors = torch.full((num_users, topk), -1, dtype=torch.long, device=self.device)
        self.ref_summary: Optional[torch.Tensor] = None
        self.num_users = num_users

    @torch.no_grad()
    def snapshot(self, graph, summary_bank: Optional[torch.Tensor], window: int) -> None:
        """窗口开始时，给 onset == 本窗口的计划内用户存一份"变化前"的邻居与摘要快照。"""
        if self.ref_neighbors is None:
            return
        sel = np.flatnonzero(self.onset_of[self.plan.users] == window)
        if sel.size == 0:
            return
        users = torch.as_tensor(self.plan.users[sel], dtype=torch.long, device=self.device)
        self.ref_neighbors[users] = graph.ids[users]
        if summary_bank is not None:
            if self.ref_summary is None:
                self.ref_summary = torch.zeros_like(summary_bank)
            self.ref_summary[users] = summary_bank[users]

    @torch.no_grad()
    def probe(
        self,
        users: np.ndarray,
        graph,
        summary_bank: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """返回这些用户（必须是计划内成员）的边级指标，数组顺序与 users 一致。"""
        result = {
            "n_neighbors": np.zeros(users.size, dtype=np.float64),
            "old_peer_share": np.full(users.size, np.nan),
            "new_peer_share": np.full(users.size, np.nan),
            "turnover_from_pre": np.full(users.size, np.nan),
            "summary_drift": np.full(users.size, np.nan),
        }
        if users.size == 0:
            return result
        member = np.isin(users, self.plan.users)
        if not member.any():
            return result

        user_ids = torch.as_tensor(users[member], dtype=torch.long, device=self.device)
        nbr = graph.ids[user_ids]                                        # (G, K)，无效位是 -1
        slot_valid = nbr >= 0
        user_valid = graph.mask[user_ids]
        w = graph.weights[user_ids] * slot_valid.float() * user_valid.float().unsqueeze(1)
        denom = w.sum(dim=1).clamp_min(1e-8)
        n_valid = slot_valid.float().sum(dim=1)

        # 上传摘要相对 onset 前快照的位移：把"兴趣摘要本身有没有恢复"与"图有没有恢复"分开看
        if summary_bank is not None and self.ref_summary is not None:
            ref_sum = self.ref_summary[user_ids]
            ref_norm = ref_sum.norm(dim=1)
            cos = F.cosine_similarity(summary_bank[user_ids], ref_sum, dim=-1, eps=1e-8)
            drift = torch.where(
                ref_norm > 1e-6, 1.0 - cos, torch.full_like(cos, float("nan"))
            )
            result["summary_drift"][member] = drift.cpu().numpy()

        onsets = self.onset_of[users[member]]
        old_share = torch.full((user_ids.numel(),), float("nan"), device=self.device)
        new_share = torch.full((user_ids.numel(),), float("nan"), device=self.device)
        for boundary in np.unique(onsets):
            boundary = int(boundary)
            if boundary not in self.pre_dom:
                continue
            sel = torch.as_tensor(
                np.flatnonzero(onsets == boundary), dtype=torch.long, device=self.device
            )
            sel_users = user_ids[sel]
            sel_rows = self.row_t[sel_users]                            # 用户 → 计划行号
            old_cat = self.cat_old[sel_rows].unsqueeze(1)
            new_cat = self.cat_new[sel_rows].unsqueeze(1)
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

        if self.ref_neighbors is not None:
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


# 参与序列统一由 feddieg_core.federated.build_participation_schedule 提供（对齐 A/B 版）。


# ======================================================================================
# 单次运行
# ======================================================================================
# ======================================================================================
# 单次运行（真正的 FedAvg + 论文公式的动态图）
# ======================================================================================
def run_single(
    cfg: FedDIEGConfig,
    scenario: str,
    windows: List[ClientWindow],
    plan: Plan,
    device: torch.device,
    out_dir: Path,
    inject_stats: Dict[str, object],
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
        f"[{scenario}|rho={cfg.rho:.2f}] 联邦设置：N={num_users}, m={clients_per_round} "
        f"(C={cfg.participation_rate}), E={cfg.local_epochs}, "
        f"personalization={cfg.personalization}, FedAvg={cfg.fedavg_weighting}"
    )
    print(f"[{scenario}|rho={cfg.rho:.2f}] 图设置：{graph.describe()}")

    probe = GroupProbe(plan, device)
    probe.init_ref(num_users, graph.k)

    group_of = np.full(num_users, -1, dtype=np.int64)
    group_of[plan.users] = plan.group
    plan_users = plan.users

    round_rows: List[Dict[str, object]] = []
    user_rows: List[Dict[str, object]] = []
    window_ndcgs: List[float] = []
    window_ecrs: List[float] = []
    start_time = time.time()
    global_round = 0

    for window_idx, window in enumerate(windows, start=1):
        print(
            f"\n[{scenario}|rho={cfg.rho:.2f}] Window {window_idx}/{cfg.num_windows} | "
            f"eval_users={window.test_users.numel()} | train_clients={window.active_users.numel()}"
        )
        probe.snapshot(graph, trainer.summary_bank, window_idx)
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
            if gstats is not None and len(gstats) > 0:
                upd = gstats.updated.numpy()
                ecr_arr = gstats.ecr.numpy()
                ecr_map = dict(zip(upd.tolist(), ecr_arr.tolist()))
                ecr_overall = float(ecr_arr.mean())
                added_overall = float(gstats.n_added.mean())
            else:
                ecr_map = {}
                ecr_overall = 0.0
                added_overall = 0.0

            should_eval = (window_round % cfg.eval_every == 0) or (window_round == cfg.rounds_per_window)
            if not should_eval:
                continue

            result = evaluate_split(trainer, eval_model, window, graph, "test", cfg)
            test_users, test_hr, test_ndcg = result["users"], result["hr"], result["ndcg"]
            all_ndcg = float(test_ndcg.mean()) if test_ndcg.size else 0.0
            last_ndcg = all_ndcg

            present = (
                test_users[np.isin(test_users, plan_users)]
                if test_users.size
                else np.zeros(0, dtype=np.int64)
            )
            edge_stats = probe.probe(present, graph, trainer.summary_bank)
            ndcg_map = dict(zip(test_users.tolist(), test_ndcg.tolist()))
            hr_map = dict(zip(test_users.tolist(), test_hr.tolist()))

            # ---- 逐用户 ----
            per_group_values: Dict[int, Dict[str, List[float]]] = {
                gid: {
                    "ndcg": [], "hr": [], "old": [], "new": [],
                    "turn": [], "ecr": [], "nbr": [], "drift": [],
                }
                for gid in range(len(GROUPS))
            }
            for pos, user in enumerate(present.tolist()):
                gid = int(group_of[user])
                if gid < 0:
                    continue
                row = int(probe.row_of[user])
                onset = int(plan.onset[row])
                rel_round = global_round - (onset - 1) * cfg.rounds_per_window
                ecr_user = float(ecr_map.get(int(user), float("nan")))
                user_rows.append(
                    {
                        "dataset": cfg.dataset,
                        "scenario": scenario,
                        "rho": cfg.rho,
                        "seed": cfg.seed,
                        "group": GROUPS[gid],
                        "user": int(user),
                        "pair_id": int(plan.pair_id[row]),
                        "onset_window": onset,
                        "cat_old": int(plan.cat_old[row]),
                        "cat_new": int(plan.cat_new[row]),
                        "window": window_idx,
                        "window_round": window_round,
                        "global_round": global_round,
                        "rel_round": rel_round,
                        "ndcg": float(ndcg_map.get(int(user), np.nan)),
                        "hr": float(hr_map.get(int(user), np.nan)),
                        "n_neighbors": float(edge_stats["n_neighbors"][pos]),
                        "old_peer_share": float(edge_stats["old_peer_share"][pos]),
                        "new_peer_share": float(edge_stats["new_peer_share"][pos]),
                        "turnover_from_pre": float(edge_stats["turnover_from_pre"][pos]),
                        "ecr": ecr_user,
                        "summary_drift": float(edge_stats["summary_drift"][pos]),
                    }
                )
                bucket = per_group_values[gid]
                bucket["ndcg"].append(float(ndcg_map.get(int(user), np.nan)))
                bucket["hr"].append(float(hr_map.get(int(user), np.nan)))
                bucket["old"].append(float(edge_stats["old_peer_share"][pos]))
                bucket["new"].append(float(edge_stats["new_peer_share"][pos]))
                bucket["turn"].append(float(edge_stats["turnover_from_pre"][pos]))
                bucket["nbr"].append(float(edge_stats["n_neighbors"][pos]))
                bucket["drift"].append(float(edge_stats["summary_drift"][pos]))
                if np.isfinite(ecr_user):
                    bucket["ecr"].append(ecr_user)

            # ---- 逐轮 × 分组聚合 ----
            def _nanmean(values: Sequence[float]) -> float:
                arr = np.asarray(values, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                return float(arr.mean()) if arr.size else float("nan")

            group_round_rows: List[Dict[str, object]] = []
            for gid in range(len(GROUPS)):
                bucket = per_group_values[gid]
                if not bucket["ndcg"]:
                    continue
                onset_values = plan.onset[plan.group == gid]
                shared_onset = int(onset_values[0]) if np.unique(onset_values).size == 1 else 0
                rel_round = (
                    global_round - (shared_onset - 1) * cfg.rounds_per_window
                    if shared_onset
                    else float("nan")
                )
                group_round_rows.append(
                    {
                        "dataset": cfg.dataset,
                        "scenario": scenario,
                        "rho": cfg.rho,
                        "seed": cfg.seed,
                        "window": window_idx,
                        "window_round": window_round,
                        "global_round": global_round,
                        "group": GROUPS[gid],
                        "rel_round": rel_round,
                        "n_eval": len(bucket["ndcg"]),
                        "n_users": len(bucket["ndcg"]),
                        "ndcg": _nanmean(bucket["ndcg"]),
                        "hr": _nanmean(bucket["hr"]),
                        "old_peer_share": _nanmean(bucket["old"]),
                        "new_peer_share": _nanmean(bucket["new"]),
                        "turnover": _nanmean(bucket["turn"]),
                        "ecr_mean": _nanmean(bucket["ecr"]),
                        "summary_drift": _nanmean(bucket["drift"]),
                        "train_loss": stats.loss,
                    }
                )
            # 全体（不限计划内用户）
            group_round_rows.append(
                {
                    "dataset": cfg.dataset,
                    "scenario": scenario,
                    "rho": cfg.rho,
                    "seed": cfg.seed,
                    "window": window_idx,
                    "window_round": window_round,
                    "global_round": global_round,
                    "group": "all",
                    "rel_round": float("nan"),
                    "n_eval": int(test_ndcg.size),
                    "n_users": int(test_ndcg.size),
                    "ndcg": all_ndcg,
                    "hr": float(test_hr.mean()) if test_hr.size else float("nan"),
                    "old_peer_share": float("nan"),
                    "new_peer_share": float("nan"),
                    "turnover": float("nan"),
                    "ecr_mean": ecr_overall,
                    "summary_drift": float("nan"),
                    "train_loss": stats.loss,
                }
            )
            round_rows.extend(group_round_rows)
            window_ecrs.append(ecr_overall)

            inj_summary = next((row for row in group_round_rows if row["group"] == "inj"), None)
            print(
                f"[{scenario}|rho={cfg.rho:.2f}] W{window_idx} R{window_round:03d}/"
                f"{cfg.rounds_per_window} | loss {stats.loss:.4f} | "
                f"trained {stats.n_trained}/{stats.n_clients} | "
                f"ECR {ecr_overall:.4f} (+{added_overall:.2f}) | NDCG all {all_ndcg:.4f}"
                + (
                    f" | inj {inj_summary['ndcg']:.4f} turn {inj_summary['turnover']:.3f}"
                    if inj_summary is not None
                    else ""
                )
                + f" | {time.time() - round_start:.2f}s"
            )

        window_ndcgs.append(last_ndcg)
        write_csv(out_dir / "val03_round_metrics.csv", round_rows, ROUND_FIELDS)
        write_csv(out_dir / "val03_user_round_metrics.csv", user_rows, USER_FIELDS)
        print(
            f"[{scenario}|rho={cfg.rho:.2f}] W{window_idx} end | NDCG@{cfg.top_k} {last_ndcg:.4f} "
            f"| rows={len(user_rows)}"
        )

    summary = {
        "dataset": cfg.dataset,
        "scenario": scenario,
        "rho": cfg.rho,
        "seed": cfg.seed,
        "graph_gate": "raw (余弦量纲)",
        "new_edge_init": cfg.graph_new_edge,
        "graph_strength": f"graph_mix={cfg.graph_mix}",
        "graph_mix": cfg.graph_mix,
        "num_windows": cfg.num_windows,
        "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": cfg.total_rounds,
        "clients_per_round": clients_per_round,
        "inject_onset": plan.inject_onset,
        "inject_windows": " ".join(str(w) for w in plan.inject_windows),
        "n_inj": int((plan.group == GROUP_TO_ID["inj"]).sum()),
        "n_pair": int((plan.group == GROUP_TO_ID["pair"]).sum()),
        "n_real_sustained": int((plan.group == GROUP_TO_ID["real_sustained"]).sum()),
        "n_real_transient": int((plan.group == GROUP_TO_ID["real_transient"]).sum()),
        "n_real_stable": int((plan.group == GROUP_TO_ID["real_stable"]).sum()),
        "inject_cos_mean": float("nan"),
        "inject_rows_replaced": int(inject_stats.get("rows_replaced", 0)),
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
    }
    write_csv(out_dir / "val03_summary.csv", [summary], SUMMARY_FIELDS)
    print(
        f"\n[{scenario}|rho={cfg.rho:.2f}] seed{cfg.seed} done in "
        f"{summary['elapsed_seconds']:.1f}s | avg_ndcg={summary['all_avg_ndcg']:.4f} | "
        f"avg_ecr={summary['avg_ecr']:.4f}"
    )


# ======================================================================================
# CLI
# ======================================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="val03: 短期兴趣扰动 vs 持续兴趣变化，固定 EMA 是否区分得开（FedAvg 版）"
    )
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="sustained")
    parser.add_argument("--inject-fraction", type=float, default=None,
                        help="注入窗口内被替换的交互比例；默认读 plan 里的值")
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--plan-dir", type=Path, default=None)
    parser.add_argument("--self-check", action="store_true",
                        help="校验未被修改的数据帧与原 load_rolling_windows 逐元素一致（开发用）")
    add_feddieg_args(parser)
    parser.set_defaults(out_root=DEFAULT_OUT_ROOT, rho=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    if args.smoke and args.out_dir is None:
        # smoke 至少要有 3 个窗口，否则 plan 的注入窗口（onset=3）根本不存在
        cfg = cfg.clone(num_windows=max(3, cfg.num_windows))
        cfg.rounds = cfg.total_rounds
    cfg.validate()
    prepare_runtime(cfg)

    inject_fraction = args.inject_fraction if args.inject_fraction is not None else 1.0
    if not (0.0 < inject_fraction <= 1.0):
        raise SystemExit("--inject-fraction 必须在 (0, 1] 区间内")

    set_seed(cfg.seed)
    device = make_device(cfg.device)

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else make_out_dir(args.out_root, cfg.dataset, args.scenario, cfg.rho, cfg.seed)
    )
    if args.smoke:
        # smoke 输出单独放到 <out_root>_smoke，避免污染正式结果的聚合脚本
        out_dir = Path(str(args.out_root) + "_smoke") / out_dir.relative_to(args.out_root)
    warn_if_legacy_results(out_dir)
    open_log(out_dir, "val03_run.log")

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print("Config:", json.dumps({**asdict(cfg), "scenario": args.scenario}, indent=2,
                                ensure_ascii=False, default=str))
    dump_json(out_dir / "config.json", {**asdict(cfg), "scenario": args.scenario,
                                        "core_version": CORE_VERSION})

    plan_dir = (
        Path(args.plan_dir).resolve()
        if args.plan_dir
        else (Path(args.plan_root) / safe_slug(cfg.dataset)).resolve()
    )
    plan = load_plan(plan_dir, args.scenario)
    plan_meta = json.loads((plan_dir / "val03_scenario_meta.json").read_text(encoding="utf-8"))
    if inject_fraction == 1.0 and "config" in plan_meta:
        inject_fraction = float(plan_meta["config"].get("inject_fraction", 1.0))
        print(f"[plan] 采用 plan 里的 inject_fraction={inject_fraction}")

    full_df, _timestamps = build_full_frame(cfg.dataset)
    windows_idx = frame_windows(len(full_df), cfg.num_windows)
    exp2_cfg = _exp2_cfg(cfg)

    if args.self_check:
        reference = load_rolling_windows(exp2_cfg)
        mine = load_rolling_windows_from_frame(exp2_cfg, full_df)
        worst = 0.0
        for ref, cur in zip(reference, mine):
            for name in ("train_seq", "train_pos", "val_seq", "test_seq", "test_items"):
                a = getattr(ref, name)
                b = getattr(cur, name)
                worst = max(worst, float((a - b).abs().max().item()) if a.numel() else 0.0)
        status = "OK" if worst == 0.0 else f"不一致（最大差 {worst}）"
        print(f"[self-check] 未注入数据帧 vs exp02.load_rolling_windows：{status}")
        if worst != 0.0:
            raise SystemExit("[self-check] 窗口构造与 exp02 不一致，停止（避免两个实验口径不同）")

    injected_df, inject_stats = apply_injections(
        full_df, windows_idx, plan, args.scenario, inject_fraction, cfg.seed
    )
    fast_windows = load_rolling_windows_from_frame(exp2_cfg, injected_df)
    num_users = int(fast_windows[0].num_users)
    history_sets = history_sets_from_frame(injected_df, cfg.num_windows, num_users)
    problems = verify_consistency(fast_windows, history_sets)
    if problems:
        raise SystemExit("历史集合与训练序列不一致：" + "; ".join(problems[:3]))
    windows = [
        from_fast_data(fast, hist, idx)
        for idx, (fast, hist) in enumerate(zip(fast_windows, history_sets), start=1)
    ]
    run_single(cfg, args.scenario, windows, plan, device, out_dir, inject_stats)
    print(f"\nDone: {out_dir}")


def _exp2_cfg(cfg: FedDIEGConfig) -> Exp2Config:
    """构造 exp_02 的 Exp2Config（数据窗口协议仍走官方实现）。"""
    return Exp2Config(
        dataset=cfg.dataset,
        seed=cfg.seed,
        latent_dim=cfg.latent_dim,
        max_len=cfg.max_len,
        top_k=cfg.top_k,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        dropout_rate=cfg.dropout_rate,
        num_windows=cfg.num_windows,
        rounds_per_window=cfg.rounds_per_window,
        candidate_count=cfg.candidate_count,
        window_mode=cfg.window_mode,
        eval_every=cfg.eval_every,
        eval_batch_size=cfg.eval_batch_size,
        max_eval_users=cfg.max_eval_users,
        num_threads=cfg.num_threads,
    )


if __name__ == "__main__":
    main()
