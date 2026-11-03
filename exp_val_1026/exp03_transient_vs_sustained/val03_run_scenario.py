#!/usr/bin/env python
"""验证实验 3（val03）第 2 步：跑一个 (数据集, 场景, EMA 系数 ρ, 随机种子) 训练。

对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验三」
（"短期兴趣波动会不会导致不必要的图更新？"）。

一次运行同时测量 5 个用户组
---------------------------
  注入 arm（人工构造，见 val03_build_scenarios.py）
    inj               该运行对应场景下的注入组（sustained: 注入持续到最后的窗口；transient: 只注入 onset 窗口）
    pair              配对对照：同样的窗口、同样的替换流程，但**实际替换成自己旧主类别**的物品
                      （同类别重采样）；它的 `cat_new` 用配对对象的注入目标类别，好让两组的
                      old/new 同伴占比在同一参照下相减。
  真实 arm（只打标签，不修改任何交互）
    real_sustained    真实数据里"跳变后保持新类别"的用户
    real_transient    真实数据里"跳变后回到旧类别"的用户（任务书场景 B）
    real_stable       同一 onset、活动量匹配、真实数据里从未跳变的对照

由于 5 个组在**同一次运行**内共存，组间比较是同一份训练轨迹、同一份参与序列下的配对比较。

两个场景各跑一次
---------------
  同一个 plan 文件，只是 `--scenario` 不同：
    sustained → 注入窗口 = onset..W（持续证据）
    transient → 注入窗口 = onset 仅此一个（之后数据恢复原样）
  两组的 onset、目标类别、配对关系完全相同，因此场景之间唯一的差别是"证据持续多久"。

记录什么（任务书观察 1~4 全部对应）
-----------------------------------
  (a) 逐用户逐评测轮 NDCG@10 / HR@10                → 推荐性能损失
  (b) 逐用户逐评测轮 ECR（本轮边变化率）             → 边关系变化程度
  (c) 逐用户 neighbors 里"旧兴趣同伴 / 新兴趣同伴"的边权占比
  (d) 逐用户当前邻居集合与 onset 前快照的 Jaccard 距离（turnover_from_pre）
        onset 后升高 = 图被这次证据改动了；
        transient 组在撤销后回落 = 图能撤销；不回落 = 过度更新（把一次性噪声当持久变化）；
        sustained 组一直不升高 = 更新滞后。
  (e) 逐用户 summary_drift = 1 − cos(当前上传摘要, onset 前快照)
        用来把"摘要本身没恢复"与"图没恢复"分开：摘要已恢复而图没恢复 → 责任在 EMA 边权规则。

输出（`results/raw/<dataset>/<scenario>/rho<ρ>/seed_<seed>/`）
--------------------------------------------------------------
  config.json                    本次运行的全部参数
  val03_user_round_metrics.csv   ★ 逐用户逐评测轮（曲线的全部原始数据）
  val03_round_metrics.csv        逐评测轮 × 分组的聚合（含 train_loss、ECR）
  val03_summary.csv              本次运行汇总（一行）
  val03_run.log                  日志

用法
----
  python exp_val_1026/exp03_transient_vs_sustained/val03_run_scenario.py \
      --dataset "movielens-1m" --scenario sustained --rho 0.8 --seed 42
  python .../val03_run_scenario.py --dataset "movielens-1m" --scenario transient --rho 0.8 --seed 42

  图影响强度（paper=已发表配置；strong=放大图影响，敏感性对照）：
  python .../val03_run_scenario.py ... --graph-strength strong
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

from exp_01_fast_gpu.run_exp01_fast_gpu import (  # noqa: E402
    FastData,
    FedDIEGModel,
    Tee,
    batched_indices,
    bce_seq_loss,
    sample_negative_like,
    set_seed,
)
from exp_02_fast_gpu.run_exp02_fast_gpu import (  # noqa: E402
    Exp2Config,
    build_eval_arrays_from_records,
    build_train_arrays_from_sequences,
    load_rolling_windows,
    mean,
    sample_candidates,
)
from val03_build_scenarios import safe_slug  # noqa: E402

DEFAULT_PLAN_ROOT = VAL03_DIR / "results" / "scenarios"
DEFAULT_OUT_ROOT = VAL03_DIR / "results" / "raw"

SCENARIOS = ("sustained", "transient")
GROUPS = ("inj", "pair", "real_sustained", "real_transient", "real_stable")
INJECTED_GROUPS = ("inj", "pair")


# ======================================================================================
# 配置
# ======================================================================================
@dataclass
class Val03Config(Exp2Config):
    """在 Exp2 的时间窗口协议之上，只加"场景 / EMA 系数"两个自变量。"""

    rho: float = 0.80
    scenario: str = "sustained"
    num_windows: int = 5
    rounds_per_window: int = 120
    candidate_count: int = 99
    window_mode: str = "slice"
    lambda_temp: float = 0.01
    lambda_struct: float = 0.20
    graph_gate: str = "raw"            # raw=阈值看原始余弦（推荐）；blended=阈值看混合后的值（原实现）
    new_edge_init: str = "ema"         # ema=忠实复刻原实现 | warm=新边不被惩罚 | zero=从 0 涨
    graph_block_size: int = 2048
    graph_strength: str = "paper"      # paper=已发表配置；strong=放大图影响（敏感性对照）
    plan_root: Path = DEFAULT_PLAN_ROOT
    plan_dir: Optional[str] = None
    inject_fraction: float = 1.0       # 与 plan 一致：注入窗口内被替换的交互比例
    self_check: bool = False           # 与 exp02.load_rolling_windows 对比，确认未改窗口口径


# 两组"图影响强度"预设（与验证实验二完全一致，方便把两个实验的数字放在一起读）
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
}


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
    new_edge_init: str = "ema",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 update_users 的邻居做 Top-K + EMA 平滑：e ← ρ·e + (1-ρ)·ê。

    与验证实验二完全同一实现（写报告时可直接引用实验二的说明）：
      * `min_sim` 默认作用在原始余弦上（gate_on="raw"），否则 ρ 大时图会整体坍塌；
      * `new_edge_init="ema"`（默认）等价于新边从 (1-ρ)·sim 起步 —— 忠实复刻原实现；
      * EMA 作用在平滑后的相似度上，权重最后在保留的 Top-K 内归一化给 GNN 用。
    返回 (本轮被更新的用户, 对应边变化率 ECR)。
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

    active_slot = torch.full((summary_bank.size(0),), -1, dtype=torch.long, device=device)
    active_slot[active] = torch.arange(num_active, device=device)

    ecr_chunks: List[torch.Tensor] = []
    user_chunks: List[torch.Tensor] = []

    for block in batched_indices(torch.arange(users.numel(), device=device), block_size):
        block_users = users[block]
        rows = block_users.numel()
        sim = torch.matmul(norm_bank[block_users], norm_active.T).clamp_min(0.0)
        sim = sim.masked_fill(block_users.unsqueeze(1) == active.unsqueeze(0), 0.0)

        old_ids = graph_neighbors[block_users]
        old_weights = smooth_state[block_users]
        old_col = active_slot[old_ids.clamp_min(0)]
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
            else:  # "ema"
                init_value = (1.0 - rho) * sim
            cand = torch.where(prev_mask > 0, blended, init_value)
            row_has_prev = (prev_mask > 0).any(dim=1, keepdim=True)
            cand = torch.where(row_has_prev, cand, sim)
        else:
            cand = sim

        exists = sim >= min_sim if gate_on == "raw" else cand >= min_sim
        cand_gated = torch.where(exists, cand, torch.full_like(cand, -1e9))

        top_values, top_positions = torch.topk(cand_gated, k, dim=1)
        new_ids = active[top_positions]
        keep = top_values > -1e8
        kept_values = torch.where(keep, top_values, torch.zeros_like(top_values))
        denom = kept_values.sum(dim=1, keepdim=True).clamp_min(1e-8)
        new_weights = kept_values / denom
        new_mask = keep.any(dim=1)

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
    cfg: Val03Config,
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
    def snapshot(
        self, graph_neighbors: torch.Tensor, summary_bank: Optional[torch.Tensor], window: int
    ) -> None:
        """窗口开始时，给 onset == 本窗口的计划内用户存一份"变化前"的邻居与摘要快照。"""
        if self.ref_neighbors is None:
            return
        sel = np.flatnonzero(self.onset_of[self.plan.users] == window)
        if sel.size == 0:
            return
        users = torch.as_tensor(self.plan.users[sel], dtype=torch.long, device=self.device)
        self.ref_neighbors[users] = graph_neighbors[users]
        if summary_bank is not None:
            if self.ref_summary is None:
                self.ref_summary = torch.zeros_like(summary_bank)
            self.ref_summary[users] = summary_bank[users]

    @torch.no_grad()
    def probe(
        self,
        users: np.ndarray,
        graph_neighbors: torch.Tensor,
        graph_weights: torch.Tensor,
        graph_mask: torch.Tensor,
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
        nbr = graph_neighbors[user_ids]
        w = graph_weights[user_ids] * graph_mask[user_ids].float().unsqueeze(1)
        denom = w.sum(dim=1).clamp_min(1e-8)
        n_valid = (nbr >= 0).float().sum(dim=1)

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
            n_cur = (nbr >= 0).sum(dim=1).float()
            union = (n_ref + n_cur - inter).clamp_min(1e-8)
            turnover = torch.where(n_ref > 0, 1.0 - inter / union, torch.full_like(inter, float("nan")))
            result["turnover_from_pre"][member] = turnover.cpu().numpy()
        return result


# ======================================================================================
# 参与序列（与实验一/实验二同一实现）
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
    cfg: Val03Config, windows: List[FastData], plan: Plan, device: torch.device, out_dir: Path,
    inject_stats: Dict[str, object],
) -> None:
    set_seed(cfg.seed)
    num_users = windows[0].num_users
    model = FedDIEGModel(windows[0].num_items, cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
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

    probe = GroupProbe(plan, device)
    probe.init_ref(num_users, topk)

    # 查表：每个用户属于哪个组（-1 = 不在计划内）
    group_of = np.full(num_users, -1, dtype=np.int64)
    group_of[plan.users] = plan.group
    plan_users = plan.users

    round_rows: List[Dict[str, object]] = []
    user_rows: List[Dict[str, object]] = []
    window_ndcgs: List[float] = []
    window_ecrs: List[float] = []
    start_time = time.time()
    global_round = 0

    for window_idx, data in enumerate(windows, start=1):
        print(
            f"\n[{cfg.scenario}|rho={cfg.rho:.2f}] Window {window_idx}/{cfg.num_windows} | "
            f"eval_users={data.test_users.numel()}"
        )
        probe.snapshot(graph_neighbors, summary_bank, window_idx)
        last_ndcg = 0.0

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
            all_ndcg = float(test_ndcg.mean()) if test_ndcg.size else 0.0
            last_ndcg = all_ndcg

            upd = updated_users.numpy()
            ecr = ecr_values.numpy()
            ecr_map = dict(zip(upd.tolist(), ecr.tolist()))

            present = (
                test_users[np.isin(test_users, plan_users)] if test_users.size else np.zeros(0, dtype=np.int64)
            )
            edge_stats = probe.probe(present, graph_neighbors, graph_weights, graph_mask, summary_bank)
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
                        "scenario": cfg.scenario,
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
                    global_round - (shared_onset - 1) * cfg.rounds_per_window if shared_onset else float("nan")
                )
                group_round_rows.append(
                    {
                        "dataset": cfg.dataset,
                        "scenario": cfg.scenario,
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
                        "train_loss": mean(losses),
                    }
                )
            # 全体（不限计划内用户）
            group_round_rows.append(
                {
                    "dataset": cfg.dataset,
                    "scenario": cfg.scenario,
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
                    "ecr_mean": float(ecr.mean()) if ecr.size else float("nan"),
                    "summary_drift": float("nan"),
                    "train_loss": mean(losses),
                }
            )
            round_rows.extend(group_round_rows)
            window_ecrs.append(float(ecr.mean()) if ecr.size else 0.0)

            inj_summary = next(
                (row for row in group_round_rows if row["group"] == "inj"), None
            )
            print(
                f"[{cfg.scenario}|rho={cfg.rho:.2f}] W{window_idx} R{window_round:03d}/"
                f"{cfg.rounds_per_window} | loss {mean(losses):.4f} | "
                f"NDCG all {all_ndcg:.4f}"
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
        print(f"[{cfg.scenario}|rho={cfg.rho:.2f}] W{window_idx} end | NDCG@10 {last_ndcg:.4f} | rows={len(user_rows)}")

    summary = {
        "dataset": cfg.dataset,
        "scenario": cfg.scenario,
        "rho": cfg.rho,
        "seed": cfg.seed,
        "graph_gate": cfg.graph_gate,
        "new_edge_init": cfg.new_edge_init,
        "graph_strength": cfg.graph_strength,
        "graph_mix": cfg.feddieg_graph_mix,
        "num_windows": cfg.num_windows,
        "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": total_rounds,
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
    }
    write_csv(out_dir / "val03_summary.csv", [summary], SUMMARY_FIELDS)
    print(
        f"\n[{cfg.scenario}|rho={cfg.rho:.2f}] seed{cfg.seed} done in "
        f"{summary['elapsed_seconds']:.1f}s | avg_ndcg={summary['all_avg_ndcg']:.4f} | "
        f"avg_ecr={summary['avg_ecr']:.4f}"
    )


# ======================================================================================
# CLI
# ======================================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="val03: 短期兴趣扰动 vs 持续兴趣变化，固定 EMA 是否区分得开"
    )
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="sustained")
    parser.add_argument("--rho", type=float, default=0.80, help="边权 EMA 系数（论文默认 0.8）")
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
        choices=["paper", "strong"],
        default="paper",
        help="图影响强度预设：paper=已发表配置；strong=放大图影响（敏感性对照，见 GRAPH_PRESETS）",
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
        help="新边初值：ema=忠实复刻原实现（默认）；warm=新边不被惩罚（消掉滞后机制的对照）；zero=从 0 涨",
    )
    parser.add_argument("--graph-block-size", type=int, default=2048)
    parser.add_argument("--inject-fraction", type=float, default=None,
                        help="注入窗口内被替换的交互比例；默认读 plan 里的值（1.0）")
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--plan-dir", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--self-check", action="store_true",
                        help="校验未被修改的数据帧与原 load_rolling_windows 逐元素一致（开发用）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = GRAPH_PRESETS[args.graph_strength]
    for key, preset_value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, preset_value)

    cfg = Val03Config(
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
        scenario=args.scenario,
        graph_gate=args.graph_gate,
        new_edge_init=args.new_edge_init,
        graph_block_size=args.graph_block_size,
        graph_strength=args.graph_strength,
        plan_root=args.plan_root,
        plan_dir=str(args.plan_dir) if args.plan_dir else None,
        inject_fraction=args.inject_fraction if args.inject_fraction is not None else 1.0,
        self_check=args.self_check,
    )

    if not (0.0 <= cfg.rho < 1.0):
        raise SystemExit("--rho 必须在 [0, 1) 区间内")
    if cfg.feddieg_min_sim <= 0:
        raise SystemExit("--feddieg-min-sim 必须 > 0")
    if not (0.0 < cfg.inject_fraction <= 1.0):
        raise SystemExit("--inject-fraction 必须在 (0, 1] 区间内")

    torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else make_out_dir(args.out_root, cfg.dataset, cfg.scenario, cfg.rho, cfg.seed)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "val03_run.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, log_handle)  # type: ignore[assignment]

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print("Config:", json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str))
    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    plan_dir = (
        Path(cfg.plan_dir).resolve()
        if cfg.plan_dir
        else (Path(cfg.plan_root) / safe_slug(cfg.dataset)).resolve()
    )
    plan = load_plan(plan_dir, cfg.scenario)
    plan_meta = json.loads((plan_dir / "val03_scenario_meta.json").read_text(encoding="utf-8"))
    if cfg.inject_fraction == 1.0 and "config" in plan_meta:
        cfg.inject_fraction = float(plan_meta["config"].get("inject_fraction", 1.0))
        print(f"[plan] 采用 plan 里的 inject_fraction={cfg.inject_fraction}")

    full_df, timestamps = build_full_frame(cfg.dataset)
    windows_idx = frame_windows(len(full_df), cfg.num_windows)

    exp2_cfg = Exp2Config(**{field: getattr(cfg, field) for field in Exp2Config.__dataclass_fields__})
    exp2_cfg.num_windows = cfg.num_windows
    exp2_cfg.rounds_per_window = cfg.rounds_per_window
    exp2_cfg.candidate_count = cfg.candidate_count
    exp2_cfg.window_mode = cfg.window_mode

    if cfg.self_check:
        reference = load_rolling_windows(exp2_cfg)
        mine = load_rolling_windows_from_frame(exp2_cfg, full_df)
        worst = 0.0
        for w_idx, (ref, cur) in enumerate(zip(reference, mine), start=1):
            for name in ("train_seq", "train_pos", "val_seq", "test_seq", "test_items"):
                a = getattr(ref, name)
                b = getattr(cur, name)
                worst = max(worst, float((a - b).abs().max().item()) if a.numel() else 0.0)
        status = "OK" if worst == 0.0 else f"不一致（最大差 {worst}）"
        print(f"[self-check] 未注入数据帧 vs exp02.load_rolling_windows：{status}")
        if worst != 0.0:
            raise SystemExit("[self-check] 窗口构造与 exp02 不一致，停止（避免两个实验口径不同）")

    injected_df, inject_stats = apply_injections(
        full_df, windows_idx, plan, cfg.scenario, cfg.inject_fraction, cfg.seed
    )
    windows = [w.to(device) for w in load_rolling_windows_from_frame(exp2_cfg, injected_df)]
    run_single(cfg, windows, plan, device, out_dir, inject_stats)
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
