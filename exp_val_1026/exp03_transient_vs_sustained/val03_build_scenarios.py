#!/usr/bin/env python
"""验证实验 3（val03）第 1 步：离线构造"持续兴趣变化"与"短期兴趣扰动"两类场景（不训练模型）。

对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验三」
（"短期兴趣波动会不会导致不必要的图更新？"）。

这一步只做**离线诊断 + 场景构造**，产出后面所有 (scenario, ρ, 种子) 共用的固定清单，
因此同一份清单下的所有运行是**同一批用户**的配对比较。

两类场景（任务书要求"必须清楚区分真实数据结论和人工构造的结果"）
-----------------------------------------------------------------
**Arm R：真实数据观察组**（只打标签，不修改任何交互）
  用类别计数把真实用户分成两类，都不进入人工干预：
    real_sustained 窗口 t 的主类别跳变到新类别，且**之后一直保持**新类别；
    real_transient 窗口 t 的主类别跳变到新类别，但**之后又回到旧类别**（任务书的"场景 B"）；
    real_stable    在可判定的边界上主类别**从未跳变**，按活动量 + 同 onset 与上面两组一对一匹配
                   （作为窗口边界换测试样本的对照）。
  只使用事后类别统计做分组，不进入模型、不影响任何在线决策。

**Arm S：可控注入组**（人工构造，显式声明）
  在**用户自己的交互序列**上做替换：把 onset 窗口（以及持续场景下之后所有窗口）里该用户的
  interaction itemId 换成"目标类别"的物品，其余行、时间戳、窗口边界完全不动
  （因为窗口按行位置切分，替换 itemId 不改变任何窗口归属 → 组间唯一差异就是兴趣内容）。
    inj   目标类别 = 用户最不熟悉的类别（"突然改看另一类"），按场景决定替换窗口：
            sustained → 窗口 t0..W 全部替换（持续证据）
            transient → 只有窗口 t0（一个窗口后恢复原样）
    pair  配对对照：**同样的窗口、同样的替换流程**，但目标类别 = 用户自己 onset 前的主类别
          （等价于"物品被重采样、兴趣没变"），用来把"替换动作本身"从"兴趣改变"里分离出来。
  注入幅度由 `--inject-fraction` 控制（默认 1.0 = 该窗口的交互全部换成目标类别），
  两组完全一致，因此两组在 onset 窗口受到的"数据扰动强度"相同。

为什么要两套 arm
---------------
  - 真实 arm 回答"真实数据里能不能观察到这两种模式，图上表现有何不同"；
  - 注入 arm 回答"同样强度的证据，只改持续时间，图的更新行为是否不同"（因果、可比）；
  - 两者结论分开写，不混在一起。

窗口与类别口径
-------------
  窗口切分、类别构造、类别计数**直接复用验证实验二已验证的实现**
  （`exp02_ema_interest_drift/val02_detect_drift.py`），保证实验二与实验三的窗口/类别完全同口径。
  有 MovieLens 原始 `movies.dat` 时用真实 genre，否则用第 1 个窗口的共现行为构造 18 个伪类别。

输出（默认 `exp_val_1026/exp03_transient_vs_sustained/results/scenarios/<dataset>/`）
------------------------------------------------------------------------------------
  val03_injection_plan.csv      ★ 注入组清单：user / group(inj|pair) / pair_id / onset / cat_old / cat_new ...
  val03_real_cohorts.csv        ★ 真实 arm 清单：user / cohort(real_sustained|real_transient|real_stable) / onset / cat_old / cat_new
  val03_counts.npz              counts[user, window, category] + n_inter[user, window] + item_category[item]
  val03_category_names.json     类别名（伪类别或 genre）
  val03_category_meta.json      类别来源、覆盖率
  val03_scenario_meta.json      ★ 本次构造的全部阈值、人数、放宽阶梯（写报告用）
  val03_partition.csv           每个物品的类别（便于人工检查）
  val03_build.log               运行日志

用法
----
  python exp_val_1026/exp03_transient_vs_sustained/val03_build_scenarios.py --dataset "movielens-1m"

  有原始 genre 文件时（推荐，类别可解释）：
  python .../val03_build_scenarios.py --dataset "movielens-1m" --genres-file "D:/data/ml-1m/movies.dat"

  注入窗口改到第 2 个窗口（前置历史更短、后置观察更长）：
  python .../val03_build_scenarios.py --dataset "movielens-1m" --inject-onset 2

  组太小 / 想换阈值：
  python .../val03_build_scenarios.py --dataset "amazon-digital music" --min-inter 1 --min-pool 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL03_DIR = Path(__file__).resolve().parent
VAL02_DIR = ROOT / "exp_val_1026" / "exp02_ema_interest_drift"
DEFAULT_OUT_ROOT = VAL03_DIR / "results" / "scenarios"

for _path in (str(ROOT), str(VAL02_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# 复用验证实验二的类别/窗口实现：保证两个验证实验的窗口边界、类别视图、计数口径完全一致。
try:
    from val02_detect_drift import (  # noqa: E402
        DetectConfig,
        assign_windows,
        build_category_matrix,
        compute_counts,
        load_interactions,
        load_meta,
        row_normalize,
        total_variation,
    )
except Exception as exc:  # pragma: no cover - 只在实验二目录被删/改名时触发
    raise SystemExit(
        "无法导入验证实验二的 val02_detect_drift.py（类别构造与窗口切分口径）。\n"
        f"请确认 {VAL02_DIR / 'val02_detect_drift.py'} 存在。原始错误：{exc}"
    )


# ======================================================================================
# 配置
# ======================================================================================
@dataclass
class ScenarioConfig:
    dataset: str = "movielens-1m"
    num_windows: int = 5
    # ---- 注入 arm ----
    inject_onset: int = 3            # 注入发生在哪个窗口（1 基）。pre = 1..t0-1
    min_inter: int = 2               # 参与注入的用户，在 onset 前/onset/onset 后各窗口至少多少条交互
    require_all_post: bool = False    # True = 之后每个窗口都要够活跃（更严，组更小）
    inject_fraction: float = 1.0      # 注入窗口内被替换的交互比例（1.0 = 全部）
    max_inject_users: int = 200       # inj + pair 总人数上限（会被截成偶数）
    min_pool: int = 30                # 注入候选（配对前）少于这个数就自动放宽 min_inter
    target_category_mode: str = "rare"  # rare=用户最不熟悉的类别（推荐） | hash=按用户 hash 随机类别
    min_items_per_category: int = 30  # 目标类别至少要有的物品数（避免抽不出物品）
    # ---- 真实 arm ----
    min_inter_pooled: int = 3        # pre / post 各自至少多少条交互
    min_spike_inter: int = 2         # 跳变窗口至少多少条交互
    min_drift_excess: float = 0.05   # 跳变强度下限（TVD 减统计噪声零点）
    max_real_users: int = 250        # 每个真实队列的人数上限
    min_real_cohort: int = 25        # 真实队列人数低于这个值就自动放宽阈值
    # ---- 公共 ----
    num_categories: int = 18
    category_mode: str = "auto"      # auto | genres | cooccur
    genres_file: Optional[str] = None
    partition_scope: str = "window1"
    proj_dim: int = 64
    random_seed: int = 2026
    cohort_seed: int = 2026
    out_root: Path = DEFAULT_OUT_ROOT
    out_dir: Optional[str] = None
    force: bool = False

    @property
    def inject_windows(self) -> List[int]:
        return list(range(int(self.inject_onset), int(self.num_windows) + 1))


PLAN_FIELDS = [
    "dataset",
    "user",
    "group",              # inj | pair
    "pair_id",
    "onset_window",
    "cat_old",
    "cat_new",            # 指标参照类别（pair 用配对对象的注入目标，便于两组相减）
    "replace_category",   # 实际替换成哪个类别（inj=目标类别；pair=自己的旧主类别）
    "is_placebo",         # pair 组为 1（同类别重采样）
    "target_rarity",      # cat_old 占该用户 onset 前交互的比例（越小 = 越"陌生"）
    "n_inter_pooled_pre",
    "n_inter_onset",
    "n_inter_post",
    "n_inter_total",
    "activity_rank",
]

REAL_FIELDS = [
    "dataset",
    "user",
    "group",              # real_sustained | real_transient | real_stable
    "onsets_matched",     # real_stable 用：配对的 drift 用户的 onset
    "onset_window",
    "cat_old",
    "cat_new",
    "spike_window",
    "n_inter_pooled_pre",
    "n_inter_spike",
    "n_inter_post",
    "n_inter_total",
    "spike_purity",       # 跳变窗口里属于 cat_new 的交互占比
    "n_post_windows",     # onset 之后还有几个窗口够活跃（决定能观察多久）
    "drift_excess",       # 噪声零点校正后的 TVD（跳变强度）
    "tvd_spike",          # 跳变窗口 vs onset 前 的原始 TVD
    "null_tvd",           # 同规模两样本的 TVD 零点
    "matched_user",
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


def _dominant(counts_2d: np.ndarray) -> np.ndarray:
    total = counts_2d.sum(axis=1)
    return np.where(total > 0, counts_2d.argmax(axis=1), -1)


def _purity(counts_2d: np.ndarray, category: np.ndarray) -> np.ndarray:
    total = counts_2d.sum(axis=1)
    out = np.zeros(counts_2d.shape[0], dtype=np.float64)
    valid = (total > 0) & (category >= 0)
    if valid.any():
        idx = np.flatnonzero(valid)
        out[idx] = counts_2d[idx, category[idx]] / np.maximum(total[idx], 1e-12)
    return out


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


# ======================================================================================
# 输入：交互、窗口、类别、计数
# ======================================================================================
@dataclass
class Inputs:
    full_df: pd.DataFrame
    windows: np.ndarray          # (N,) 0 基窗口号
    counts: np.ndarray           # (U, W, K) 类别加权交互数
    n_inter: np.ndarray          # (U, W) 原始交互条数
    item_category: np.ndarray    # (I,) 单标签类别；-1 = 未覆盖
    cat_names: List[str]
    cat_meta: Dict[str, object]
    num_users: int
    num_items: int


def load_inputs(cfg: ScenarioConfig) -> Inputs:
    meta = load_meta(cfg.dataset)
    num_users, num_items = int(meta["num_users"]), int(meta["num_items"])
    full_df = load_interactions(cfg.dataset)
    windows = assign_windows(len(full_df), cfg.num_windows)
    detect_cfg = DetectConfig(
        dataset=cfg.dataset,
        num_windows=cfg.num_windows,
        category_mode=cfg.category_mode,
        genres_file=cfg.genres_file,
        num_categories=cfg.num_categories,
        partition_scope=cfg.partition_scope,
        proj_dim=cfg.proj_dim,
        random_seed=cfg.random_seed,
    )
    cat_matrix, cat_names, cat_meta = build_category_matrix(
        detect_cfg, num_users, num_items, full_df, windows
    )
    counts = compute_counts(full_df, windows, num_users, cfg.num_windows, cat_matrix)

    users_arr = full_df["userId"].to_numpy(dtype=np.int64)
    n_inter = np.zeros((num_users, cfg.num_windows), dtype=np.int64)
    for window_idx in range(cfg.num_windows):
        mask = windows == window_idx
        if mask.any():
            n_inter[:, window_idx] = np.bincount(users_arr[mask], minlength=num_users)

    # 单标签类别视图（注入取样用）：多标签 genre 取权重最大的那一个
    item_category = np.where(cat_matrix.sum(axis=1) > 0, cat_matrix.argmax(axis=1), -1).astype(np.int64)

    print(
        f"[input] {cfg.dataset}: users={num_users} items={num_items} interactions={len(full_df)} "
        f"| windows={cfg.num_windows} | categories={len(cat_names)} "
        f"| 物品覆盖率={cat_meta.get('item_coverage', float('nan')):.2%}"
    )
    for window_idx in range(cfg.num_windows):
        print(
            f"[input]   W{window_idx + 1}: 有交互用户 {int((n_inter[:, window_idx] > 0).sum())}, "
            f">=2 条 {int((n_inter[:, window_idx] >= 2).sum())}"
        )
    return Inputs(
        full_df=full_df,
        windows=windows,
        counts=counts,
        n_inter=n_inter,
        item_category=item_category,
        cat_names=cat_names,
        cat_meta=cat_meta,
        num_users=num_users,
        num_items=num_items,
    )


# ======================================================================================
# 真实 arm：只打标签
# ======================================================================================
@dataclass
class RealDetect:
    """真实 arm 的逐边界中间结果（与具体阈值绑定，因为阈值会改变 can_check / excess）。"""

    tvd_spike: np.ndarray
    null_spike: np.ndarray
    can_check: np.ndarray
    cat_old: np.ndarray
    cat_spike: np.ndarray
    cat_post: np.ndarray
    jump: np.ndarray
    trans_tab: np.ndarray
    sust_tab: np.ndarray
    excess: np.ndarray
    best: np.ndarray
    best_ok: np.ndarray
    n_post_windows: np.ndarray
    activity: np.ndarray


def detect_real_jumps(inputs: Inputs, cfg: ScenarioConfig, verbose: bool = True) -> RealDetect:
    """逐边界判定"主类别跳变"以及跳变后是"保持"还是"回归"。

    判定只用类别计数（离线诊断），不进入模型。
    边界 b（1 基，=2..W-1）：pre = 窗口 1..b-1，spike = 窗口 b，post = 窗口 b+1..W。
      real_transient: 主类别在 b 窗口跳走、之后回到旧类别            → 任务书"场景 B"
      real_sustained: 主类别在 b 窗口跳走、之后保持在新类别          → 任务书"场景 A"
      （b = W 时没有 b+1 窗口，无法判定持续性，直接排除。）
    每个用户取"噪声校正后跳变强度最大"的那个边界作为 onset，不同用户 onset 可以不同。
    """
    num_users, num_windows, num_cat = inputs.counts.shape
    counts = inputs.counts.astype(np.float64)
    n_inter = inputs.n_inter

    # 逐边界特征
    tvd_spike = np.zeros((num_users, num_windows + 1), dtype=np.float64)
    null_spike = np.zeros((num_users, num_windows + 1), dtype=np.float64)
    can_check = np.zeros((num_users, num_windows + 1), dtype=bool)
    cat_old_tab = np.full((num_users, num_windows + 1), -1, dtype=np.int64)
    cat_spike_tab = np.full((num_users, num_windows + 1), -1, dtype=np.int64)
    cat_post_tab = np.full((num_users, num_windows + 1), -1, dtype=np.int64)
    post_total_tab = np.zeros((num_users, num_windows + 1), dtype=np.float64)
    n_post_windows = np.zeros((num_users, num_windows + 1), dtype=np.int64)

    for boundary in range(2, num_windows + 1):
        pre_c = counts[:, : boundary - 1, :].sum(axis=1)
        spike_c = counts[:, boundary - 1, :]
        post_c = counts[:, boundary:, :].sum(axis=1)
        pre_total = pre_c.sum(axis=1)
        spike_total = spike_c.sum(axis=1)
        post_total = post_c.sum(axis=1)
        cat_old_tab[:, boundary] = _dominant(pre_c)
        cat_spike_tab[:, boundary] = _dominant(spike_c)
        cat_post_tab[:, boundary] = _dominant(post_c)
        post_total_tab[:, boundary] = post_total
        if boundary + 1 <= num_windows:
            n_post_windows[:, boundary] = (n_inter[:, boundary:] >= 2).sum(axis=1)

        tvd = total_variation(row_normalize(pre_c), row_normalize(spike_c))
        # 噪声零点：把 pre + spike 的交互混在一起按同规模随机拆分得到的 TVD 期望
        # （与实验二一致：否则交互少的用户会被误判成"跳变"）
        rng = np.random.default_rng(cfg.random_seed + boundary)
        trials = 8
        acc = np.zeros(num_users, dtype=np.float64)
        pooled = pre_c + spike_c
        for _ in range(trials):
            p = np.divide(pre_total, np.maximum(pre_total + spike_total, 1e-9))
            s1 = rng.binomial(np.maximum(pooled, 0).astype(np.int64), p[:, None]).astype(np.float64)
            s2 = np.maximum(pooled, 0) - s1
            acc += total_variation(row_normalize(s1), row_normalize(s2))
        null = acc / trials

        tvd_spike[:, boundary] = tvd
        null_spike[:, boundary] = null
        can_check[:, boundary] = (
            (boundary + 1 <= num_windows)
            & (pre_total >= cfg.min_inter_pooled)
            & (spike_total >= cfg.min_spike_inter)
            & (post_total >= cfg.min_inter_pooled)
            & (n_inter[:, boundary - 2] >= 2)
        )
        if verbose:
            n_ok = int(can_check[:, boundary].sum())
            print(
                f"[real] 边界 b={boundary}: 可判定 {n_ok} 人, "
                f"TVD 中位 {np.median(tvd[can_check[:, boundary]]) if n_ok else float('nan'):.3f}, "
                f"零点中位 {np.median(null[can_check[:, boundary]]) if n_ok else float('nan'):.3f}"
            )

    # 跳变判定 + 持续/回归判定
    cat_old = cat_old_tab
    cat_spike = cat_spike_tab
    cat_post = cat_post_tab
    # 跳变 = spike 窗口的主类别 != onset 前主类别（之后是否保持由下面两张表区分）
    jump = (
        can_check
        & (cat_old >= 0)
        & (cat_spike >= 0)
        & (cat_post >= 0)
        & (cat_spike != cat_old)
    )
    excess = np.where(jump, np.maximum(tvd_spike - null_spike, 0.0), -np.inf)
    excess = np.where(excess >= cfg.min_drift_excess, excess, -np.inf)
    trans_tab = jump & (cat_post == cat_old)
    sust_tab = jump & (cat_post == cat_spike)

    # 每个用户取 excess 最大的边界
    best = np.argmax(np.where(np.isfinite(excess), excess, -np.inf), axis=1)
    rows_idx = np.arange(num_users)
    best_ok = np.isfinite(excess[rows_idx, best])
    best[~best_ok] = 0

    def _cohort_mask(kind: str) -> np.ndarray:
        out = np.zeros(num_users, dtype=bool)
        for user in range(num_users):
            boundary = int(best[user])
            if boundary == 0:
                continue
            out[user] = bool(trans_tab[user, boundary]) if kind == "transient" else bool(
                sust_tab[user, boundary]
            )
        return out

    trans_mask = _cohort_mask("transient")
    sust_mask = _cohort_mask("sustained")
    if verbose:
        print(f"[real] 跳变且可判定持续性的用户: {int(best_ok.sum())}")
        print(f"[real] real_transient={int(trans_mask.sum())}  real_sustained={int(sust_mask.sum())}")

    activity = n_inter.sum(axis=1).astype(np.float64) + counts.sum(axis=(1, 2))
    return RealDetect(
        tvd_spike=tvd_spike,
        null_spike=null_spike,
        can_check=can_check,
        cat_old=cat_old,
        cat_spike=cat_spike,
        cat_post=cat_post,
        jump=jump,
        trans_tab=trans_tab,
        sust_tab=sust_tab,
        excess=excess,
        best=best,
        best_ok=best_ok,
        n_post_windows=n_post_windows,
        activity=activity,
    )


def assemble_real_cohorts(
    det: RealDetect,
    inputs: Inputs,
    cfg: ScenarioConfig,
    reserved_users: Sequence[int] = (),
    verbose: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """把检测结果组装成 real_sustained / real_transient / real_stable 三个队列。

    `reserved_users`（例如注入 arm 用到的人）不会被选进 real_stable，
    避免同一个人既被人工注入、又充当"稳定对照"。
    """
    num_users, num_windows, _ = inputs.counts.shape
    counts = inputs.counts.astype(np.float64)
    n_inter = inputs.n_inter
    cat_old, cat_spike, cat_post = det.cat_old, det.cat_spike, det.cat_post
    jump, trans_tab, sust_tab = det.jump, det.trans_tab, det.sust_tab
    excess, best, best_ok = det.excess, det.best, det.best_ok
    can_check, n_post_windows, activity = det.can_check, det.n_post_windows, det.activity

    trans_mask = np.zeros(num_users, dtype=bool)
    sust_mask = np.zeros(num_users, dtype=bool)
    for user in range(num_users):
        boundary = int(best[user])
        if boundary == 0:
            continue
        trans_mask[user] = bool(trans_tab[user, boundary])
        sust_mask[user] = bool(sust_tab[user, boundary])

    # ---- real_stable：可判定边界上从未跳变，且与上面两组同 onset、活动量最近匹配 ----
    reserved = set(int(u) for u in reserved_users)
    checkable_any = can_check.any(axis=1)
    jump_any = jump.any(axis=1)
    stable_pool = np.array(
        [u for u in np.flatnonzero(checkable_any & ~jump_any).tolist() if u not in reserved],
        dtype=np.int64,
    )
    taken = set(np.flatnonzero(trans_mask | sust_mask).tolist())
    stable_rows: List[Dict[str, object]] = []
    for user in sorted(taken):
        boundary = int(best[user])
        # 只匹配"在该边界上 onset 前有类别覆盖（cat_old >= 0）"的用户，
        # 否则 old/new 同伴占比指标没有参照，会出现 cat_old=-1 的空指标。
        pool = np.array(
            [v for v in stable_pool.tolist() if v not in taken and cat_old[v, boundary] >= 0],
            dtype=np.int64,
        )
        if pool.size == 0:
            continue
        dist = np.abs(np.log1p(activity[pool]) - np.log1p(activity[user]))
        pick = int(pool[int(np.argmin(dist))])
        taken.add(pick)
        stable_rows.append({"user": pick, "onset": boundary, "matched_user": int(user)})
        if len(stable_rows) >= cfg.max_real_users:
            break
    if verbose:
        print(f"[real] real_stable（匹配后）={len(stable_rows)}")

    # ---- 组装 ----
    plan_rows: List[Dict[str, object]] = []
    for kind, mask in (("real_sustained", sust_mask), ("real_transient", trans_mask)):
        users = np.flatnonzero(mask)[: cfg.max_real_users]
        for user in users.tolist():
            boundary = int(best[user])
            spike_window = int(cat_spike[user, boundary])
            pre_c = counts[user, : boundary - 1, :].sum(axis=0)
            spike_c = counts[user, boundary - 1, :]
            post_c = counts[user, boundary:, :].sum(axis=0)
            plan_rows.append(
                {
                    "dataset": cfg.dataset,
                    "user": int(user),
                    "group": kind,
                    "onsets_matched": "",
                    "onset_window": boundary,
                    "cat_old": int(cat_old[user, boundary]),
                    "cat_new": int(spike_window),
                    "spike_window": boundary,
                    "n_inter_pooled_pre": float(pre_c.sum()),
                    "n_inter_spike": float(spike_c.sum()),
                    "n_inter_post": float(post_c.sum()),
                    "n_inter_total": float(activity[user]),
                    "spike_purity": float(_purity(spike_c[None, :], np.array([spike_window]))[0]),
                    "n_post_windows": int(n_post_windows[user, boundary]),
                    "drift_excess": float(excess[user, boundary]),
                    "tvd_spike": float(det.tvd_spike[user, boundary]),
                    "null_tvd": float(det.null_spike[user, boundary]),
                    "matched_user": "",
                }
            )
    for item in stable_rows:
        user = int(item["user"])
        boundary = int(item["onset"])
        stable_cat = int(cat_old[user, boundary])
        plan_rows.append(
            {
                "dataset": cfg.dataset,
                "user": user,
                "group": "real_stable",
                "onsets_matched": boundary,
                "onset_window": boundary,
                "cat_old": stable_cat,
                "cat_new": stable_cat,
                "spike_window": boundary,
                "n_inter_pooled_pre": float(counts[user, : boundary - 1, :].sum()),
                "n_inter_spike": float(counts[user, boundary - 1, :].sum()),
                "n_inter_post": float(counts[user, boundary:, :].sum()),
                "n_inter_total": float(activity[user]),
                "spike_purity": float(
                    _purity(counts[user, boundary - 1, :][None, :], np.array([stable_cat]))[0]
                ),
                "n_post_windows": int(n_post_windows[user, boundary]),
                "drift_excess": 0.0,
                "tvd_spike": 0.0,
                "null_tvd": 0.0,
                "matched_user": int(item["matched_user"]),
            }
        )

    meta = {
        "min_inter_pooled": cfg.min_inter_pooled,
        "min_spike_inter": cfg.min_spike_inter,
        "min_drift_excess": cfg.min_drift_excess,
        "num_real_sustained": int(sum(1 for r in plan_rows if r["group"] == "real_sustained")),
        "num_real_transient": int(sum(1 for r in plan_rows if r["group"] == "real_transient")),
        "num_real_stable": int(sum(1 for r in plan_rows if r["group"] == "real_stable")),
        "num_jump_checkable_users": int(best_ok.sum()),
        "any_jump_users": int(jump.any(axis=1).sum()),
        "median_drift_excess": float(
            np.median([r["drift_excess"] for r in plan_rows if r["group"] != "real_stable"] or [0.0])
        ),
    }
    return plan_rows, meta


# ======================================================================================
# 注入 arm：配对 + 目标类别 + 替换窗口
# ======================================================================================
def select_inject_pool(inputs: Inputs, cfg: ScenarioConfig, min_inter: int) -> np.ndarray:
    """注入 arm 的候选池：onset 前、onset、onset 之后都够活跃。

    之所以要求 onset 之后也够活跃：transient 场景必须在 onset 之后至少一个窗口仍有交互，
    否则"恢复原样"这件事在数据里根本看不到；sustained 场景也需要之后有窗口才有观察点。
    """
    t0 = int(cfg.inject_onset)
    if t0 < 2 or t0 > cfg.num_windows - 1:
        raise SystemExit(f"--inject-onset 必须落在 [2, {cfg.num_windows - 1}]（要留出 pre 与 post 窗口）")
    required = [t0 - 1, t0, t0 + 1]
    if cfg.require_all_post:
        required = list(range(t0 - 1, cfg.num_windows + 1))
    mask = np.ones(inputs.num_users, dtype=bool)
    for window_idx in required:
        mask &= inputs.n_inter[:, window_idx - 1] >= int(min_inter)
    # onset 之前的交互必须**有类别覆盖**：伪类别只由第 1 个窗口的共现构造，
    # 稀疏数据集里后来的物品可能没被覆盖，这类用户算不出"旧主类别"，无法配对，直接排除。
    pre_covered = inputs.counts[:, : t0 - 1, :].sum(axis=(1, 2)) > 0
    mask &= pre_covered
    pool = np.flatnonzero(mask)
    print(
        f"[inject] onset={t0}, 要求窗口 {required} 各 >= {min_inter} 条 → 候选 {pool.size} 人"
    )
    return pool


def build_injection_plan(
    inputs: Inputs,
    cfg: ScenarioConfig,
    exclude_users: Sequence[int],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """构造 inj / pair 两组（活动量分层配对），并给每个用户选一个"目标类别"。"""
    t0 = int(cfg.inject_onset)
    min_inter = int(cfg.min_inter)
    ladder: List[int] = []
    for value in (min_inter, max(min_inter - 1, 1), 1):
        if value not in ladder:
            ladder.append(int(value))
    attempted: List[Dict[str, object]] = []
    chosen_pool = np.zeros(0, dtype=np.int64)
    chosen_min = min_inter
    for rung, value in enumerate(ladder):
        pool = select_inject_pool(inputs, cfg, value)
        exclude = np.zeros(inputs.num_users, dtype=bool)
        exclude[np.asarray(list(exclude_users), dtype=np.int64)] = True
        pool = np.array([u for u in pool.tolist() if not exclude[u]], dtype=np.int64)
        attempted.append(
            {"rung": rung, "min_inter": int(value), "pool_before_exclude": int(pool.size)}
        )
        print(f"[inject]   阶梯 {rung}: min_inter={value} → 排除真实漂移用户后候选 {pool.size} 人")
        chosen_pool, chosen_min = pool, int(value)
        if pool.size >= cfg.min_pool:
            break
        if rung + 1 < len(ladder):
            print(f"[inject]   候选 < {cfg.min_pool}，自动放宽到 min_inter={ladder[rung + 1]}")
    if chosen_pool.size < 4:
        raise SystemExit(
            f"注入候选只有 {chosen_pool.size} 人，太少。请降低 --min-inter 或换 --inject-onset。"
        )

    # ---- 活动量分层配对：按活动量排序后相邻两人配成一对，再随机决定谁是 inj ----
    span = slice(t0 - 1, cfg.num_windows)
    activity = inputs.n_inter[:, span].sum(axis=1).astype(np.float64)
    order = np.argsort(activity[chosen_pool], kind="stable")
    ordered = chosen_pool[order]
    limit = min(ordered.size - (ordered.size % 2), int(cfg.max_inject_users))
    limit -= limit % 2          # 必须偶数：保证人人都有配对对照
    ordered = ordered[:limit]
    rng = random.Random(cfg.cohort_seed + 17)

    # ---- 目标类别池：物品数足够、且（伪类别时）确实有交互 ----
    item_cat = inputs.item_category
    cat_item_counts = np.bincount(item_cat[item_cat >= 0], minlength=inputs.counts.shape[2])
    valid_cats = [k for k in range(inputs.counts.shape[2]) if cat_item_counts[k] >= cfg.min_items_per_category]
    if not valid_cats:
        valid_cats = [int(np.argmax(cat_item_counts))]
    print(f"[inject] 可用目标类别 {len(valid_cats)} 个（每类物品数 >= {cfg.min_items_per_category}）")

    pre_counts = inputs.counts[:, : t0 - 1, :].sum(axis=1).astype(np.float64)
    pre_total = pre_counts.sum(axis=1)
    share = np.divide(pre_counts, np.maximum(pre_total[:, None], 1e-12))

    rows: List[Dict[str, object]] = []
    pair_id = 0
    for idx in range(0, ordered.size, 2):
        pair_id += 1
        pair = [int(ordered[idx]), int(ordered[idx + 1])]
        rng.shuffle(pair)
        inj_user, pair_user = pair[0], pair[1]
        old_cat_inj = int(np.argmax(pre_counts[inj_user])) if pre_total[inj_user] > 0 else -1
        old_cat_pair = int(np.argmax(pre_counts[pair_user])) if pre_total[pair_user] > 0 else -1
        if old_cat_inj < 0 or old_cat_pair < 0:
            continue

        # ---- 注入组的目标类别（"突然改看另一类"）----
        if cfg.target_category_mode == "hash":
            local_rng = random.Random(cfg.cohort_seed + 101 + inj_user)
            candidates = [k for k in valid_cats if k != old_cat_inj]
            target = int(local_rng.choice(candidates)) if candidates else old_cat_inj
        else:  # rare：用户最不熟悉、但物品足够的类别
            candidates = [k for k in valid_cats if k != old_cat_inj]
            if not candidates:
                candidates = [old_cat_inj]
            shares = [share[inj_user, k] for k in candidates]
            min_share = min(shares)
            tied = [k for k, s in zip(candidates, shares) if s <= min_share + 1e-12]
            local_rng = random.Random(cfg.cohort_seed + 211 + inj_user)
            target = int(local_rng.choice(tied))

        for group, user, old_cat in (
            ("inj", inj_user, old_cat_inj),
            ("pair", pair_user, old_cat_pair),
        ):
            # `cat_new` 是**指标参照类别**：pair 也用配对对象的注入目标类别，
            # 这样两组的 old/new 同伴占比是在同一个参照下算的，可以直接相减。
            cat_new = target
            # `replace_category` 是**实际替换用的类别**：pair 用自己 onset 前的主类别（同类别重采样）。
            replace_category = target if group == "inj" else old_cat
            rows.append(
                {
                    "dataset": cfg.dataset,
                    "user": int(user),
                    "group": group,
                    "pair_id": pair_id,
                    "onset_window": t0,
                    "cat_old": old_cat,
                    "cat_new": cat_new,
                    "replace_category": replace_category,
                    "is_placebo": 1 if group == "pair" else 0,
                    "target_rarity": float(share[user, replace_category]),
                    "n_inter_pooled_pre": float(inputs.n_inter[user, : t0 - 1].sum()),
                    "n_inter_onset": float(inputs.n_inter[user, t0 - 1]),
                    "n_inter_post": float(inputs.n_inter[user, t0:].sum()),
                    "n_inter_total": float(inputs.n_inter[user].sum()),
                    "activity_rank": int(np.flatnonzero(ordered == user)[0]) + 1,
                }
            )

    n_inj = sum(1 for r in rows if r["group"] == "inj")
    n_pair = sum(1 for r in rows if r["group"] == "pair")
    # 保底：只保留成员齐全的配对（万一某个成员被跳过，另一个也不留下）
    by_pair: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_pair.setdefault(int(row["pair_id"]), []).append(row)
    balanced = [r for r in rows if len(by_pair[int(r["pair_id"])]) == 2]
    if len(balanced) != len(rows):
        print(
            f"[inject] 丢弃 {len(rows) - len(balanced)} 个不完整配对的成员，保证 inj / pair 严格一对一"
        )
        rows = balanced
        n_inj = sum(1 for r in rows if r["group"] == "inj")
        n_pair = sum(1 for r in rows if r["group"] == "pair")
    print(f"[inject] 最终 inj={n_inj}  pair={n_pair}（共 {n_inj + n_pair} 人，配对 {n_inj} 对）")
    if n_inj > 0:
        rarity = [r["target_rarity"] for r in rows if r["group"] == "inj"]
        print(
            f"[inject] 目标类别在用户历史中的占比：中位数 {np.median(rarity):.3f} "
            f"（0 = 该用户从未看过这一类）"
        )
    meta = {
        "inject_onset": t0,
        "inject_windows_default": list(range(t0, cfg.num_windows + 1)),
        "min_inter_used": chosen_min,
        "min_inter_ladder": attempted,
        "inject_fraction": cfg.inject_fraction,
        "target_category_mode": cfg.target_category_mode,
        "min_items_per_category": cfg.min_items_per_category,
        "num_inj": int(n_inj),
        "num_pair": int(n_pair),
        "num_pairs": int(pair_id),
        "median_target_rarity": float(
            np.median([r["target_rarity"] for r in rows if r["group"] == "inj"] or [float("nan")])
        ),
    }
    return rows, meta


# ======================================================================================
# 主流程
# ======================================================================================
def parse_args() -> ScenarioConfig:
    parser = argparse.ArgumentParser(
        description="val03 第 1 步：构造持续变化 / 短期扰动两类场景（不训练）"
    )
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--inject-onset", type=int, default=3,
                        help="注入发生在第几个窗口（1 基，默认 3；要留出 pre 与 post）")
    parser.add_argument("--min-inter", type=int, default=2,
                        help="注入候选在 onset 前/onset/onset 后各窗口的最少交互条数")
    parser.add_argument("--require-all-post", action="store_true",
                        help="要求 onset 之后每个窗口都够活跃（更严，注入组更小但观察更完整）")
    parser.add_argument("--inject-fraction", type=float, default=1.0,
                        help="注入窗口内被替换的交互比例（1.0=全部替换）")
    parser.add_argument("--max-inject-users", type=int, default=200,
                        help="inj + pair 总人数上限（会被截成偶数）")
    parser.add_argument("--min-pool", type=int, default=30, help="注入候选少于这个数就自动放宽 min-inter")
    parser.add_argument("--target-category-mode", choices=["rare", "hash"], default="rare")
    parser.add_argument("--min-items-per-category", type=int, default=30)
    parser.add_argument("--min-inter-pooled", type=int, default=3)
    parser.add_argument("--min-spike-inter", type=int, default=2)
    parser.add_argument("--min-drift-excess", type=float, default=0.05)
    parser.add_argument("--max-real-users", type=int, default=250)
    parser.add_argument("--min-real-cohort", type=int, default=25)
    parser.add_argument("--num-categories", type=int, default=18)
    parser.add_argument("--category-mode", choices=["auto", "genres", "cooccur"], default="auto")
    parser.add_argument("--genres-file", type=str, default=None)
    parser.add_argument("--partition-scope", choices=["window1", "full"], default="window1")
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--cohort-seed", type=int, default=2026)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="已有结果也重新构造")
    args = parser.parse_args()
    fields = {f for f in ScenarioConfig.__dataclass_fields__}
    return ScenarioConfig(**{k: v for k, v in vars(args).items() if k in fields})


def main() -> None:
    cfg = parse_args()
    if str(cfg.target_category_mode) not in {"rare", "hash"}:
        raise SystemExit("--target-category-mode 只能是 rare / hash")
    if not (0.0 < cfg.inject_fraction <= 1.0):
        raise SystemExit("--inject-fraction 必须在 (0, 1] 区间内")

    out_dir = (
        Path(cfg.out_dir).resolve()
        if cfg.out_dir is not None
        else (Path(cfg.out_root) / safe_slug(cfg.dataset)).resolve()
    )
    plan_path = out_dir / "val03_injection_plan.csv"
    if plan_path.exists() and not cfg.force:
        print(f"[skip] 已存在：{plan_path}（--force 可重新构造）")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "val03_build.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, log_handle)  # type: ignore[assignment]

    print("=" * 78)
    print(f"验证实验 3 · 场景构造 · {cfg.dataset}")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str))
    print("=" * 78)

    inputs = load_inputs(cfg)

    # ---- 真实 arm（先做：它的用户要从注入池里排除，避免把真实漂移混进人工干预）----
    # 稀疏数据集（Amazon 类每窗口每用户只有约 2 条交互）用实验二的阈值会筛出个位数用户，
    # 所以带一个自动放宽阶梯；实际用了哪一级会写进 val03_scenario_meta.json。
    real_ladder = [
        (cfg.min_inter_pooled, cfg.min_drift_excess),
        (max(cfg.min_inter_pooled - 1, 2), min(cfg.min_drift_excess, 0.02)),
        (2, 0.0),
    ]
    detect: Optional[RealDetect] = None
    real_attempted: List[Dict[str, object]] = []
    final_cfg = cfg
    for rung, (pooled, excess_floor) in enumerate(real_ladder):
        trial_cfg = replace(cfg, min_inter_pooled=int(pooled), min_drift_excess=float(excess_floor))
        detect = detect_real_jumps(inputs, trial_cfg, verbose=(rung == 0))
        _, trial_meta = assemble_real_cohorts(detect, inputs, trial_cfg, verbose=False)
        real_attempted.append(
            {
                "rung": rung,
                "min_inter_pooled": int(pooled),
                "min_drift_excess": float(excess_floor),
                "num_real_sustained": int(trial_meta["num_real_sustained"]),
                "num_real_transient": int(trial_meta["num_real_transient"]),
            }
        )
        enough = min(trial_meta["num_real_sustained"], trial_meta["num_real_transient"]) >= cfg.min_real_cohort
        print(
            f"[real] 阶梯 {rung}: min_inter_pooled={pooled}, min_drift_excess={excess_floor} → "
            f"sustained={trial_meta['num_real_sustained']} / transient={trial_meta['num_real_transient']}"
            + ("（已达标）" if enough else "（不足，继续放宽）")
        )
        final_cfg = trial_cfg
        if enough:
            break

    # 先用"不含 real_stable"的版本拿到真实漂移用户，把他们从注入池里排除
    real_rows, real_meta = assemble_real_cohorts(detect, inputs, final_cfg, verbose=False)  # type: ignore[arg-type]
    real_drift_users = [
        int(r["user"]) for r in real_rows if r["group"] in {"real_sustained", "real_transient"}
    ]
    print(f"[real] 真实漂移用户 {len(real_drift_users)} 人（注入池将排除他们）")

    # ---- 注入 arm ----
    inject_rows, inject_meta = build_injection_plan(inputs, cfg, real_drift_users)

    # ---- 真实 arm 的 real_stable：匹配时排除注入 arm 用到的人（同一人不能既是注入对象又是稳定对照）----
    inject_users = [int(r["user"]) for r in inject_rows]
    real_rows, real_meta = assemble_real_cohorts(
        detect, inputs, final_cfg, reserved_users=inject_users, verbose=True  # type: ignore[arg-type]
    )
    real_meta["auto_relax_ladder"] = real_attempted
    real_meta["auto_relax_rung"] = int(real_attempted[-1]["rung"])
    print(f"[real] 真实队列合计 {len(real_rows)} 人")

    # ---- 写盘 ----
    write_csv(plan_path, inject_rows, PLAN_FIELDS)
    write_csv(out_dir / "val03_real_cohorts.csv", real_rows, REAL_FIELDS)
    np.savez_compressed(
        out_dir / "val03_counts.npz",
        counts=inputs.counts.astype(np.float32),
        n_inter=inputs.n_inter.astype(np.int32),
        item_category=inputs.item_category.astype(np.int32),
    )
    (out_dir / "val03_category_names.json").write_text(
        json.dumps(inputs.cat_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "val03_category_meta.json").write_text(
        json.dumps(inputs.cat_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    partition_rows = [
        {
            "item": int(item),
            "category_id": int(inputs.item_category[item]),
            "category_name": (
                inputs.cat_names[int(inputs.item_category[item])]
                if inputs.item_category[item] >= 0
                else ""
            ),
        }
        for item in range(inputs.num_items)
    ]
    write_csv(
        out_dir / "val03_partition.csv", partition_rows, ["item", "category_id", "category_name"]
    )

    warnings: List[str] = []
    if int(real_meta["num_real_transient"]) < cfg.min_real_cohort:
        warnings.append(
            f"real_transient 只有 {real_meta['num_real_transient']} 人（< {cfg.min_real_cohort}）："
            "该数据集上的真实 arm 只能作方向性参考，主证据请用注入 arm。"
        )
    if int(real_meta["num_real_sustained"]) < cfg.min_real_cohort:
        warnings.append(
            f"real_sustained 只有 {real_meta['num_real_sustained']} 人（< {cfg.min_real_cohort}）：同上。"
        )
    if int(inject_meta["num_inj"]) < 20:
        warnings.append(
            f"注入组只有 {inject_meta['num_inj']} 人：组均值噪声会比较大，建议 --inject-onset 换个窗口"
            " 或降低 --min-inter。"
        )
    for message in warnings:
        print(f"[warn] {message}")

    scenario_meta = {
        "dataset": cfg.dataset,
        "num_windows": cfg.num_windows,
        "num_users": inputs.num_users,
        "num_items": inputs.num_items,
        "num_categories": len(inputs.cat_names),
        "category_names": inputs.cat_names,
        "category_source": inputs.cat_meta.get("category_mode"),
        "item_coverage": inputs.cat_meta.get("item_coverage"),
        "config": asdict(cfg) | {"out_dir": str(out_dir)},
        "real": real_meta,
        "inject": inject_meta,
        "warnings": warnings,
        "real_arm_definition": (
            "pre=窗口1..b-1, spike=窗口b, post=窗口b+1..W；real_sustained=主类别跳变且之后保持新类别，"
            "real_transient=主类别跳变但之后回到旧类别，real_stable=可判定边界上从未跳变（按 activity+同 onset 配对）"
        ),
        "inject_arm_definition": (
            "把 onset 窗口（sustained: 及其后所有窗口 / transient: 仅 onset 窗口）内该用户的 itemId "
            "替换为目标类别的物品；pair 组做同样的替换，但目标类别 = 自己的旧主类别（同类别重采样）"
        ),
        "leakage_note": (
            "真实 arm 的分组用到了 onset 之后的窗口（事后分组），只决定统计哪些用户，不进入模型、"
            "不影响任何在线决策；在线算法每个窗口只能看到该窗口已到达的交互。"
        ),
    }
    (out_dir / "val03_scenario_meta.json").write_text(
        json.dumps(scenario_meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("\n---------------- 构造结果 ----------------")
    print(f"  注入 arm: inj {inject_meta['num_inj']} 人 / pair {inject_meta['num_pair']} 人 "
          f"（{inject_meta['num_pairs']} 对，min_inter={inject_meta['min_inter_used']}）")
    print(f"  真实 arm: sustained {real_meta['num_real_sustained']} / "
          f"transient {real_meta['num_real_transient']} / stable {real_meta['num_real_stable']}")
    print(f"  输出目录: {out_dir}")
    print("  下一步: val03_run_scenario.py --scenario sustained|transient --rho ... --seed ...")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
