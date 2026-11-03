#!/usr/bin/env python
"""验证实验 2（val02）第 1 步：离线识别"兴趣真实发生变化"的用户 + 构造类别视图。

对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验二」
（"首先采用现有的时间窗口实验，寻找真实存在明显偏好变化的用户"）。

这一步完全不训练模型，只做**离线诊断**，产出后面 4 个 ρ 共用的固定 cohort。

1. 窗口切分与 `exp_02_fast_gpu.run_exp02_fast_gpu.load_rolling_windows` 完全一致
   （按 timestamp+userId 排序后按交互数等分；窗口内再做 LOO 划分）。
2. 给每个物品一个"类别"：
   - 有 MovieLens 原始 `movies.dat`（genre）→ 用真实类型（多标签，按 1/|genres| 加权）；
   - 否则 → 用**第 1 个窗口**的共现行为构造"伪类别"（随机符号投影 ≈ 共现内积 + KMeans），
     只用最早窗口的数据，避免把未来窗口的信息带入诊断。
3. 逐边界 t（=2..W）识别漂移：把窗口 1..t-1 的类别分布与窗口 t..W 的类别分布对比，
   得到 TVD；再减掉**统计噪声零点**（H0 下同样本量两样本的 TVD 期望，多项分布模拟），
   得到 drift_excess ∈ [0, 1)。
4. 持续性检验：变化之后用户应更接近"新自己"而非"旧自己"
   （tvd(p_t, p_{t+1}) < tvd(p_{t-1}, p_{t+1})）→ 把一次性的短期跳变留给实验三。
5. 输出固定的 drift cohort（兴趣变化组）与 stable cohort（稳定对照组，
   逐边界 + 活动量分层匹配）。cohort 由 `--cohort-seed` 固定，
   与后面的 ρ 和训练种子无关 → 4 个 ρ × 3 个种子是**同一批用户**的配对比较。

为什么必须做噪声零点与"逐边界"识别（这个数据集的两个坑）
--------------------------------------------------------
坑 1：每个窗口每用户平均只有 30 条交互，18 个类别上两次独立抽样的 TVD 期望就有 0.2；
      不减零点会把"交互少"误判成"兴趣变了"。
坑 2：MovieLens-1M 用户行为是**爆发式**的——999,611 条交互里，只有 23 个用户 5 个窗口
      都活跃，1416 个用户跨 ≥2 个窗口。所以"每个窗口都要够活跃"这种筛选会直接筛选出 0 人；
      正确做法是只要求该用户在**当前边界两侧**各自够活跃（默认各 ≥5 条），
      onset 取用户自己 drift_excess 最大的那个边界，不同用户的 onset 可以不同。

注意（写报告要带上）
--------------------
* cohort 的筛选用到整段时间跨度的数据（含 onset 之后的窗口），这是**事后分组**，
  只决定"统计哪些用户"，不进入模型、不影响任何在线决策；在线算法每个窗口只能看到
  该窗口已到达的交互。
* 类别统计只用于离线诊断，训练与构图完全不使用。

输出（默认 `exp_val_1026/exp02_ema_interest_drift/results/drift/<dataset>/`）
--------------------------------------------------------------------------
  config.json                    本次诊断的全部参数
  val02_partition.csv            每个物品的类别（itemId, category_id, category_name, weight）
  val02_category_meta.json       类别名、类别来源（genres / cooccur）、覆盖率
  val02_user_window_profile.csv  用户 × 窗口 → 交互数、主类别、熵、类别分布
  val02_counts.npz               类别计数张量 counts[user, window, category]
  val02_user_boundary_drift.csv  用户 × 边界 → tvd / 噪声零点 / drift_excess / 是否可用
  val02_drift_users.csv          ★ cohort 清单（drift / stable、onset_window、漂移分数、主类别变化）
  val02_drift_summary.csv        不同阈值下的 cohort 规模（写报告用）
  val02_cohort_meta.json         cohort 元信息
  val02_detect.log               运行日志

用法
----
  python exp_val_1026/exp02_ema_interest_drift/val02_detect_drift.py --dataset movielens-1m

  有原始 genre 文件时（推荐，类别可解释）：
  python .../val02_detect_drift.py --dataset movielens-1m --genres-file "D:/data/ml-1m/movies.dat"

  cohort 太小/太大时调阈值：
  python .../val02_detect_drift.py --tau-drift 0.10 --min-inter-per-window 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL02_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = VAL02_DIR / "results" / "drift"

GENRE_FILE_CANDIDATES = (
    "datasets/raw/{dataset}/movies.dat",
    "datasets/{dataset}/movies.dat",
    "datasets/raw/ml-1m/movies.dat",
    "datasets/ml-1m/movies.dat",
)


# ======================================================================================
# 配置
# ======================================================================================
@dataclass
class DetectConfig:
    dataset: str = "movielens-1m"
    num_windows: int = 5
    # ---- 类别视图 ----
    category_mode: str = "auto"          # auto | genres | cooccur
    genres_file: Optional[str] = None    # MovieLens movies.dat 路径
    num_categories: int = 18             # cooccur 伪类别的簇数（= ML-1M 的 genre 数量）
    partition_scope: str = "window1"     # window1 | full（伪类别用哪些交互）
    proj_dim: int = 64                   # 共现随机投影维度
    # ---- 漂移识别 ----
    min_inter_per_window: int = 3        # 边界两侧窗口各自至少多少条交互
    min_inter_pooled: int = 20           # 漂移前 / 后各自至少多少条交互（统计功效）
    null_trials: int = 10                # 噪声零点的随机拆分次数
    drift_quantile: float = 0.25         # drift cohort = 漂移分数最高的 25%
    stable_quantile: float = 0.25        # stable cohort = 漂移分数最低的 25%
    min_drift_excess: float = 0.15       # drift cohort 的绝对下限（防止整体都低时误标）
    require_sustained: bool = False      # 是否强制"持续性检验"（ML-1M 上会砍到 ~13%，默认关）
    include_last_boundary: bool = True   # 是否纳入最后一个边界（只有单窗口 post，无法验证持续性）
    auto_relax: bool = True              # cohort 太小（< min_cohort）时按阶梯自动放宽阈值
    min_cohort: int = 60                 # 期望的最小 drift cohort 人数
    max_drift_users: int = 400
    max_stable_users: int = 400
    cohort_seed: int = 2026              # cohort 抽样种子（必须与 ρ / 训练种子无关）
    random_seed: int = 2026              # 伪类别 / kmeans 的种子
    out_dir: Optional[str] = None


# 自动放宽阶梯：第 0 级是默认值，之后逐级放宽。稀疏数据集（Amazon 类）每窗口每用户只有几条交互，
# 用默认阈值会只剩个位数用户（实测 amazon-digital music 只剩 8 人），必须放宽才有可用 cohort。
# 注意 min_inter_per_window 不会低于 2：低于 2 的用户在该窗口进不了测试集，
# 也就没有"变化前"的评测点，曲线无法对齐。
AUTO_RELAX_LADDER: Tuple[Dict[str, object], ...] = (
    {"min_inter_per_window": 3, "min_inter_pooled": 20, "drift_quantile": 0.25, "stable_quantile": 0.25, "min_drift_excess": 0.15},
    {"min_inter_per_window": 2, "min_inter_pooled": 14, "drift_quantile": 0.25, "stable_quantile": 0.25, "min_drift_excess": 0.12},
    {"min_inter_per_window": 2, "min_inter_pooled": 10, "drift_quantile": 0.30, "stable_quantile": 0.30, "min_drift_excess": 0.10},
    {"min_inter_per_window": 2, "min_inter_pooled": 6, "drift_quantile": 0.33, "stable_quantile": 0.33, "min_drift_excess": 0.05},
)


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


def entropy(probs: np.ndarray) -> float:
    safe = np.where(probs > 0, probs, 1.0)
    return float(-(probs * np.log2(safe)).sum())


def total_variation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """逐行 TVD = 0.5 * L1，取值 [0, 1]。"""
    return 0.5 * np.abs(p - q).sum(axis=-1)


def row_normalize(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True)
    return np.where(total > 0, counts / np.maximum(total, 1e-12), 0.0)


# ======================================================================================
# 数据与窗口（口径必须与 load_rolling_windows 一致）
# ======================================================================================
def load_interactions(dataset: str) -> pd.DataFrame:
    splits_dir = ROOT / "datasets" / "splits" / dataset
    frames = [pd.read_csv(splits_dir / f"loo_{name}.csv") for name in ("train", "val", "test")]
    full_df = pd.concat(frames, ignore_index=True)
    if "timestamp" in full_df.columns:
        full_df = full_df.sort_values(["timestamp", "userId"]).reset_index(drop=True)
    else:
        full_df = full_df.reset_index(drop=True)
        full_df["timestamp"] = full_df.index
    return full_df


def assign_windows(num_rows: int, num_windows: int) -> np.ndarray:
    """复刻 load_rolling_windows：窗口 w 覆盖 [int(N*(w-1)/W), int(N*w/W))，返回 0 基窗口号。"""
    bounds = np.array([int(num_rows * w / num_windows) for w in range(num_windows + 1)], dtype=np.int64)
    positions = np.arange(num_rows)
    return np.searchsorted(bounds[1:], positions, side="right")


def load_meta(dataset: str) -> Dict[str, int]:
    with (ROOT / "datasets" / "processed" / dataset / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return {"num_users": int(meta["num_users"]), "num_items": int(meta["num_items"])}


def load_item_mapping(dataset: str) -> Dict[str, int]:
    path = ROOT / "datasets" / "processed" / dataset / "mappings.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        mappings = json.load(handle)
    return {str(key): int(value) for key, value in mappings.get("item2id", {}).items()}


# ======================================================================================
# 类别视图 1：MovieLens 原始 genre（movies.dat，多标签）
# ======================================================================================
def find_genre_file(cfg: DetectConfig) -> Optional[Path]:
    if cfg.genres_file:
        path = Path(cfg.genres_file)
        return path if path.exists() else None
    for template in GENRE_FILE_CANDIDATES:
        candidate = ROOT / template.format(dataset=cfg.dataset)
        if candidate.exists():
            return candidate
    return None


def build_genre_matrix(cfg: DetectConfig, num_items: int) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    """解析 MovieLens movies.dat，返回 (num_items, K) 权重矩阵（每行和为 1）。"""
    genre_path = find_genre_file(cfg)
    if genre_path is None:
        raise FileNotFoundError("没有找到 movies.dat：请用 --genres-file 指定，或改用 --category-mode cooccur")
    item2id = load_item_mapping(cfg.dataset)

    raw_rows: List[Tuple[int, List[str]]] = []
    with genre_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 3:
                continue
            internal = item2id.get(str(parts[0]))
            tags = [tag for tag in parts[2].split("|") if tag]
            if internal is not None and tags:
                raw_rows.append((internal, tags))

    names = sorted({tag for _, tags in raw_rows for tag in tags})
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    matrix = np.zeros((num_items, len(names)), dtype=np.float32)
    for internal, tags in raw_rows:
        weight = 1.0 / len(tags)
        for tag in tags:
            matrix[internal, name_to_idx[tag]] += weight

    covered = int((matrix.sum(axis=1) > 0).sum())
    meta = {
        "category_mode": "genres",
        "source": str(genre_path),
        "num_categories": len(names),
        "category_names": names,
        "num_items": int(num_items),
        "num_items_covered": covered,
        "item_coverage": covered / max(num_items, 1),
        "multi_label": True,
    }
    return matrix, names, meta


# ======================================================================================
# 类别视图 2：共现伪类别（无 genre 标签时的回退方案）
# ======================================================================================
def _kmeans_numpy(x: np.ndarray, k: int, seed: int, iters: int = 40) -> np.ndarray:
    """极简 k-means++（纯 numpy，避免额外依赖）。x 已行归一化。"""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = min(k, n)
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(n)]
    closest = ((x - centers[0]) ** 2).sum(axis=1)
    for idx in range(1, k):
        total = closest.sum()
        if not np.isfinite(total) or total <= 0:
            centers[idx] = x[rng.integers(n)]
        else:
            centers[idx] = x[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, ((x - centers[idx]) ** 2).sum(axis=1))

    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = labels == idx
            if members.any():
                centers[idx] = x[members].mean(axis=0)
    return labels


def build_cooccur_matrix(
    cfg: DetectConfig, num_users: int, num_items: int, full_df: pd.DataFrame, windows: np.ndarray
) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    """用交互共现行为给物品聚类，生成单标签"伪类别"。

    1. 取 `partition_scope` 范围内的交互，构造 user-item 0/1 关系；
    2. 随机符号投影：item_emb[item] = Σ_{user 交互过 item} R[user]（R 为 ±1 随机矩阵），
       于是 <item_emb[i], item_emb[j]> ≈ 共现次数（Johnson-Lindenstrauss）；
    3. 行归一化后 KMeans 聚成 K 类。
    复杂度只与物品数有关，不需要 user×item 稠密矩阵。
    """
    use_full = cfg.partition_scope == "full"
    mask = np.ones(len(full_df), dtype=bool) if use_full else (windows == 0)
    users = full_df["userId"].to_numpy(dtype=np.int64)[mask]
    items = full_df["itemId"].to_numpy(dtype=np.int64)[mask]
    print(
        f"[partition] 伪类别来源：{'全部窗口' if use_full else '第 1 个窗口'}，"
        f"交互 {len(users)} 条，投影维度 {cfg.proj_dim}，簇数 {cfg.num_categories}"
    )

    rng = np.random.default_rng(cfg.random_seed)
    proj = rng.integers(0, 2, size=(num_users, cfg.proj_dim), dtype=np.int8).astype(np.float32) * 2.0 - 1.0

    order = np.argsort(items, kind="stable")
    items_sorted, users_sorted = items[order], users[order]
    starts = np.flatnonzero(np.r_[True, items_sorted[1:] != items_sorted[:-1]])
    item_emb = np.zeros((num_items, cfg.proj_dim), dtype=np.float32)
    item_emb[items_sorted[starts]] = np.add.reduceat(proj[users_sorted], starts, axis=0)

    norms = np.linalg.norm(item_emb, axis=1, keepdims=True)
    covered_mask = norms.ravel() > 0
    item_emb = item_emb / np.maximum(norms, 1e-12)
    labels = np.full(num_items, -1, dtype=np.int64)
    labels[covered_mask] = _kmeans_numpy(item_emb[covered_mask], cfg.num_categories, cfg.random_seed)

    matrix = np.zeros((num_items, cfg.num_categories), dtype=np.float32)
    labeled = np.flatnonzero(labels >= 0)
    matrix[labeled, labels[labeled]] = 1.0
    names = [f"pseudo-{idx:02d}" for idx in range(cfg.num_categories)]
    covered = int((matrix.sum(axis=1) > 0).sum())
    meta = {
        "category_mode": "cooccur",
        "source": f"{'full' if use_full else 'window1'} interactions + random-sign projection + kmeans",
        "num_categories": cfg.num_categories,
        "category_names": names,
        "num_items": int(num_items),
        "num_items_covered": covered,
        "item_coverage": covered / max(num_items, 1),
        "multi_label": False,
        "cluster_sizes": [int((labels == idx).sum()) for idx in range(cfg.num_categories)],
    }
    return matrix, names, meta


def build_category_matrix(
    cfg: DetectConfig, num_users: int, num_items: int, full_df: pd.DataFrame, windows: np.ndarray
) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    mode = cfg.category_mode
    if mode == "auto":
        mode = "genres" if find_genre_file(cfg) is not None else "cooccur"
    print(f"[partition] category_mode = {mode}")
    if mode == "genres":
        return build_genre_matrix(cfg, num_items)
    return build_cooccur_matrix(cfg, num_users, num_items, full_df, windows)


# ======================================================================================
# 每用户每窗口的类别计数
# ======================================================================================
def compute_counts(
    full_df: pd.DataFrame, windows: np.ndarray, num_users: int, num_windows: int, cat_matrix: np.ndarray
) -> np.ndarray:
    """counts[u, w, k]：用户 u 在窗口 w 中属于类别 k 的加权交互数。"""
    num_cat = cat_matrix.shape[1]
    users = full_df["userId"].to_numpy(dtype=np.int64)
    items = full_df["itemId"].to_numpy(dtype=np.int64)
    counts = np.zeros((num_users, num_windows, num_cat), dtype=np.float32)
    offsets = np.arange(num_cat, dtype=np.int64)
    for window_idx in range(num_windows):
        mask = windows == window_idx
        if not mask.any():
            continue
        window_users = users[mask]
        rows = cat_matrix[items[mask]]
        flat = (window_users[:, None] * num_cat + offsets[None, :]).ravel()
        counts[:, window_idx, :] = np.bincount(
            flat, weights=rows.ravel(), minlength=num_users * num_cat
        ).reshape(num_users, num_cat)
    return counts


def compute_n_inter(full_df: pd.DataFrame, windows: np.ndarray, num_users: int, num_windows: int) -> np.ndarray:
    users = full_df["userId"].to_numpy(dtype=np.int64)
    out = np.zeros((num_users, num_windows), dtype=np.int64)
    for window_idx in range(num_windows):
        mask = windows == window_idx
        if mask.any():
            out[:, window_idx] = np.bincount(users[mask], minlength=num_users)
    return out


# ======================================================================================
# 噪声零点：H0 下两样本 TVD 的期望
# ======================================================================================
def null_tvd_split(
    pooled_counts: np.ndarray, n1: np.ndarray, n2: np.ndarray, trials: int, seed: int
) -> np.ndarray:
    """H0 下的 TVD 零点：把 pooled 的交互随机拆成规模 n1 / n2 的两份得到的 TVD 期望。

    这是标准的置换/随机化零分布：若"变化前后同分布"，则把两段交互混在一起随机对半切，
    得到的 TVD 分布就是观测统计量在 H0 下的参照分布。
    多个类别按二项独立拆分（单标签时与随机拆分完全等价；多标签 genre 时是很好的近似）。

    每个窗口每用户只有几十条交互，18 个类别上"两次独立抽样"的 TVD 期望就有 0.19 左右；
    不减掉这个零点，会把"交互少"误判成"兴趣变了"。
    """
    rng = np.random.default_rng(seed)
    pooled = np.asarray(pooled_counts, dtype=np.float64)
    base = np.floor(pooled)
    frac = pooled - base
    ratio = np.divide(
        np.asarray(n1, dtype=np.float64),
        np.maximum(np.asarray(n1, dtype=np.float64) + np.asarray(n2, dtype=np.float64), 1e-9),
    )
    num_users = pooled.shape[0]
    acc = np.zeros(num_users, dtype=np.float64)
    for _ in range(max(trials, 1)):
        # 分数计数（多标签 genre 的 1/|genres| 权重）用随机取整，保证无系统性偏差
        integer_counts = base + (rng.random(pooled.shape) < frac).astype(np.float64)
        s1 = rng.binomial(integer_counts.astype(np.int64), ratio[:, None]).astype(np.float64)
        s2 = integer_counts - s1
        acc += total_variation(row_normalize(s1), row_normalize(s2))
    return acc / max(trials, 1)


# ======================================================================================
# 漂移识别
# ======================================================================================
def detect_drift(
    cfg: DetectConfig, counts: np.ndarray, n_inter: np.ndarray, verbose: bool = True
) -> Tuple[pd.DataFrame, List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]:
    """逐边界识别漂移 → 固定 drift / stable cohort。

    边界 t（1 基，= 2..W）表示"兴趣变化发生在窗口 t-1 与 t 之间"，于是
    窗口 t 是第一个属于新兴趣的窗口。pre = 窗口 1..t-1，post = 窗口 t..W。
    """
    num_users, num_windows, _ = counts.shape
    profiles = row_normalize(counts)
    window_totals = counts.sum(axis=2)

    boundaries = list(range(2, num_windows + 1))
    if not cfg.include_last_boundary:
        boundaries = [t for t in boundaries if t + 1 <= num_windows]
    if verbose:
        print(f"[drift] 参与识别的边界（onset 窗口）：{boundaries}")

    tvd_table = np.zeros((num_users, num_windows + 1), dtype=np.float64)
    null_table = np.zeros((num_users, num_windows + 1), dtype=np.float64)
    excess_table = np.full((num_users, num_windows + 1), -np.inf, dtype=np.float64)
    activity_ok = np.zeros((num_users, num_windows + 1), dtype=bool)
    sustained_ok = np.zeros((num_users, num_windows + 1), dtype=bool)
    sustained_known = np.zeros((num_users, num_windows + 1), dtype=bool)

    for onset in boundaries:
        pre = counts[:, : onset - 1, :].sum(axis=1)
        post = counts[:, onset - 1 :, :].sum(axis=1)
        pre_total = pre.sum(axis=1)
        post_total = post.sum(axis=1)
        pooled = pre + post
        tvd = total_variation(row_normalize(pre), row_normalize(post))
        null = null_tvd_split(
            pooled,
            np.maximum(pre_total, 0.0),
            np.maximum(post_total, 0.0),
            cfg.null_trials,
            cfg.random_seed + onset,
        )

        # 边界两侧都要有足够交互
        act = (
            (n_inter[:, onset - 2] >= cfg.min_inter_per_window)
            & (n_inter[:, onset - 1] >= cfg.min_inter_per_window)
            & (pre_total >= cfg.min_inter_pooled)
            & (post_total >= cfg.min_inter_pooled)
            & (window_totals[:, onset - 2] > 0)
            & (window_totals[:, onset - 1] > 0)
        )
        if onset + 1 <= num_windows:
            # 持续性（描述性指标，也是 --require-sustained 的判据）：
            # 变化之后应更接近"新自己"而不是"旧自己"
            p_prev = profiles[:, onset - 2, :]
            p_cur = profiles[:, onset - 1, :]
            p_next = profiles[:, onset, :]
            sustain = total_variation(p_cur, p_next) < total_variation(p_prev, p_next)
            known = act & (n_inter[:, onset] >= 2)
            sustained_known[:, onset] = known
            sustained_ok[:, onset] = known & sustain
        # onset = W 时没有后续窗口，持续性无法验证（sustained_known 保持 False）

        tvd_table[:, onset] = tvd
        null_table[:, onset] = null
        excess_table[:, onset] = np.where(act, np.maximum(tvd - null, 0.0), -np.inf)
        activity_ok[:, onset] = act
        if not verbose:
            continue
        if bool(act.any()):
            print(
                f"[drift] 边界 t={onset}: 边界两侧可用用户 {int(act.sum())}, "
                f"持续性可验证 {int(sustained_known[:, onset].sum())} / 通过 {int(sustained_ok[:, onset].sum())}, "
                f"TVD 中位数 {np.median(tvd[act]):.3f}, 零点中位数 {np.median(null[act]):.3f}, "
                f"excess 中位数 {np.median(excess_table[act, onset]):.3f}"
            )
        else:
            print(f"[drift] 边界 t={onset}: 没有边界两侧都够活跃的用户")

    # ---- 每个用户的最佳 onset ----
    allowed = np.zeros(num_windows + 1, dtype=bool)
    allowed[boundaries] = True
    usable = activity_ok.copy()
    usable[:, ~allowed] = False
    if cfg.require_sustained:
        usable &= sustained_ok
    any_usable = usable.any(axis=1)
    masked = np.where(usable, excess_table, -np.inf)
    idx = masked.argmax(axis=1)
    rows = np.arange(num_users)
    best_onset = np.where(any_usable, idx, 0)
    best_excess = np.where(any_usable, masked[rows, idx], 0.0)

    if verbose:
        print(f"[drift] 可用用户（至少一个边界两侧都够活跃）: {int(any_usable.sum())} / {num_users}")

    # ---- drift / stable cohort：按漂移分数的高低分位（相对比较，避免绝对值不可标定）----
    rng = np.random.default_rng(cfg.cohort_seed)
    valid_excess = best_excess[any_usable]
    if valid_excess.size == 0:
        drift_threshold = stable_threshold = float("inf")
    else:
        drift_threshold = max(
            float(cfg.min_drift_excess), float(np.quantile(valid_excess, 1.0 - cfg.drift_quantile))
        )
        stable_threshold = float(np.quantile(valid_excess, cfg.stable_quantile))
    if verbose:
        print(
            f"[drift] drift 阈值 = max({cfg.min_drift_excess}, "
            f"{cfg.drift_quantile:.0%} 分位) = {drift_threshold:.3f}；"
            f"stable 阈值 = {cfg.stable_quantile:.0%} 分位 = {stable_threshold:.3f}"
        )

    drift_pool = np.flatnonzero(any_usable & (best_excess >= drift_threshold))
    if drift_pool.size > cfg.max_drift_users:
        order = np.argsort(-best_excess[drift_pool], kind="stable")
        drift_pool = drift_pool[order[: cfg.max_drift_users]]
    drift_set = set(int(u) for u in drift_pool.tolist())

    stable_candidates = np.flatnonzero(any_usable & (best_excess <= stable_threshold))
    stable_candidates = np.array([u for u in stable_candidates.tolist() if u not in drift_set], dtype=np.int64)

    activity = n_inter.astype(np.float64).sum(axis=1) + counts.sum(axis=(1, 2))
    stable_onset_map = _match_stable_users(
        cfg, drift_pool, best_onset, stable_candidates, activity, activity_ok, rng
    )
    stable_users = np.sort(np.asarray(sorted(stable_onset_map), dtype=np.int64))
    stable_set = set(int(u) for u in stable_users.tolist())
    if verbose:
        print(f"[drift] 对照组候选: {stable_candidates.size}")
        print(f"[drift] 最终 cohort: drift={len(drift_pool)}  stable={len(stable_users)}")

    # ---- 组装清单 ----
    cohort_rows: List[Dict[str, object]] = []
    for user in range(num_users):
        if user in drift_set:
            cohort, onset = "drift", int(best_onset[user])
            excess_value = float(best_excess[user])
        elif user in stable_set:
            cohort, onset = "stable", int(stable_onset_map[user])
            excess_value = float(best_excess[user])
        else:
            continue
        pre_profile = row_normalize(counts[user, : onset - 1, :].sum(axis=0, keepdims=True))[0]
        post_profile = row_normalize(counts[user, onset - 1 :, :].sum(axis=0, keepdims=True))[0]
        cat_old = int(pre_profile.argmax()) if pre_profile.sum() > 0 else -1
        cat_new = int(post_profile.argmax()) if post_profile.sum() > 0 else -1
        cohort_rows.append(
            {
                "dataset": cfg.dataset,
                "user": user,
                "cohort": cohort,
                "onset_window": onset,
                "drift_excess": excess_value,
                "tvd_pooled": float(tvd_table[user, onset]),
                "null_tvd": float(null_table[user, onset]),
                "cat_old": cat_old,
                "cat_new": cat_new,
                "primary_drift": int(cat_old != cat_new and cat_old >= 0 and cat_new >= 0),
                "n_inter_before": int(n_inter[user, onset - 2]),
                "n_inter_after": int(n_inter[user, onset - 1]),
                "n_inter_total": int(n_inter[user].sum()),
                "entropy_pre": entropy(pre_profile),
                "entropy_post": entropy(post_profile),
            }
        )
    drift_df = pd.DataFrame(cohort_rows)

    # ---- 用户 × 边界明细（写报告用）----
    cohort_of: Dict[int, str] = {user: "drift" for user in drift_set}
    cohort_of.update({user: "stable" for user in stable_set})
    boundary_rows: List[Dict[str, object]] = []
    for user in range(num_users):
        for onset in boundaries:
            if not activity_ok[user, onset]:
                continue
            boundary_rows.append(
                {
                    "dataset": cfg.dataset,
                    "user": user,
                    "boundary": onset,
                    "tvd": float(tvd_table[user, onset]),
                    "null_tvd": float(null_table[user, onset]),
                    "drift_excess": float(max(excess_table[user, onset], 0.0)),
                    "sustained": int(sustained_ok[user, onset]),
                    "sustained_known": int(sustained_known[user, onset]),
                    "is_best_onset": int(best_onset[user] == onset),
                    "cohort": cohort_of.get(user, ""),
                }
            )

    # ---- 阈值敏感性表 ----
    summary: List[Dict[str, object]] = []
    for quantile in (0.10, 0.20, 0.25, 0.33, 0.50):
        if valid_excess.size:
            n_qual = int((any_usable & (best_excess >= np.quantile(valid_excess, 1.0 - quantile))).sum())
        else:
            n_qual = 0
        summary.append(
            {
                "dataset": cfg.dataset,
                "row_type": "drift_quantile",
                "threshold": quantile,
                "n_users": n_qual,
                "share_of_users": n_qual / max(num_users, 1),
                "note": "按漂移分数取最高的 quantile 比例 → 用户数",
            }
        )
    for tau in (0.10, 0.15, 0.20, 0.25, 0.30):
        n_qual = int((any_usable & (best_excess >= tau)).sum())
        summary.append(
            {
                "dataset": cfg.dataset,
                "row_type": "drift_absolute",
                "threshold": tau,
                "n_users": n_qual,
                "share_of_users": n_qual / max(num_users, 1),
                "note": "drift_excess ≥ 绝对阈值 的用户数（按最佳 onset 计）",
            }
        )
    summary.append(
        {
            "dataset": cfg.dataset,
            "row_type": "cohort",
            "threshold": "",
            "n_users": int(len(drift_pool)),
            "share_of_users": len(drift_pool) / max(num_users, 1),
            "note": "最终 drift cohort 人数",
        }
    )
    summary.append(
        {
            "dataset": cfg.dataset,
            "row_type": "cohort",
            "threshold": "",
            "n_users": int(len(stable_users)),
            "share_of_users": len(stable_users) / max(num_users, 1),
            "note": "最终 stable cohort 人数（逐边界 + 活动量分层匹配）",
        }
    )
    if len(drift_df):
        drift_part = drift_df[drift_df["cohort"] == "drift"]
        summary.append(
            {
                "dataset": cfg.dataset,
                "row_type": "primary_share",
                "threshold": "",
                "n_users": int(drift_part["primary_drift"].sum()),
                "share_of_users": float(drift_part["primary_drift"].mean()),
                "note": "主类别确实换了（cat_old != cat_new）的 drift 用户数 / 占比",
            }
        )
        for onset, group in drift_part.groupby("onset_window"):
            summary.append(
                {
                    "dataset": cfg.dataset,
                    "row_type": "onset_distribution",
                    "threshold": int(onset),
                    "n_users": int(len(group)),
                    "share_of_users": len(group) / max(len(drift_part), 1),
                    "note": f"onset = 窗口 {onset} 的 drift 用户数",
                }
            )

    meta = {
        "dataset": cfg.dataset,
        "num_windows": num_windows,
        "boundaries": boundaries,
        "n_usable": int(any_usable.sum()),
        "n_drift": int(len(drift_pool)),
        "n_stable": int(len(stable_users)),
        "drift_quantile": cfg.drift_quantile,
        "stable_quantile": cfg.stable_quantile,
        "drift_threshold": float(drift_threshold),
        "stable_threshold": float(stable_threshold),
        "min_inter_per_window": cfg.min_inter_per_window,
        "min_inter_pooled": cfg.min_inter_pooled,
        "require_sustained": cfg.require_sustained,
        "category_mode": None,  # 由 main 补齐
        "cohort_seed": cfg.cohort_seed,
        "drift_onset_distribution": {
            int(k): int(v)
            for k, v in drift_df[drift_df["cohort"] == "drift"]["onset_window"].value_counts().sort_index().items()
        }
        if len(drift_df)
        else {},
    }
    return drift_df, summary, meta, boundary_rows


def _match_stable_users(
    cfg: DetectConfig,
    drift_users: np.ndarray,
    best_onset: np.ndarray,
    stable_candidates: np.ndarray,
    activity: np.ndarray,
    activity_ok: np.ndarray,
    rng: np.random.Generator,
) -> Dict[int, int]:
    """为每组的 drift 用户（按 onset 分组）找活动量分布相近的对照用户。

    两条约束：
    * 该对照用户必须在**同一个 onset 边界两侧都够活跃**（activity_ok[:, onset]），
      否则它在"变化前"窗口没有评测记录、也没有变化前主类别，无法与 drift 用户配对；
    * 活动量分层匹配：NDCG 与交互数强相关，不匹配活动量会把活动量差异误读成漂移效应。

    返回 {user: 该对照用户在曲线上对齐用的 onset}（与其配对的 drift 用户相同）。
    """
    result: Dict[int, int] = {}
    if drift_users.size == 0 or stable_candidates.size == 0:
        return result

    remaining = set(int(u) for u in stable_candidates.tolist())
    for onset in np.unique(best_onset[drift_users]):
        onset = int(onset)
        group = drift_users[best_onset[drift_users] == onset]
        pool = np.array(
            [u for u in sorted(remaining) if 0 <= onset < activity_ok.shape[1] and activity_ok[u, onset]],
            dtype=np.int64,
        )
        if pool.size == 0 or group.size == 0:
            continue
        target = min(len(group), cfg.max_stable_users - len(result))
        if target <= 0:
            break

        combined = np.concatenate([group, pool])
        edges = np.unique(np.quantile(activity[combined], np.linspace(0, 1, 11)))
        if edges.size < 2:
            edges = np.array([activity[combined].min(), activity[combined].max() + 1.0])

        def bin_of(values: np.ndarray) -> np.ndarray:
            return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)

        group_bins, pool_bins = bin_of(activity[group]), bin_of(activity[pool])
        for bin_idx in range(edges.size - 1):
            n_in_group = int((group_bins == bin_idx).sum())
            if n_in_group == 0:
                continue
            want = max(1, int(round(target * n_in_group / len(group))))
            candidates = pool[pool_bins == bin_idx]
            take = min(want, candidates.size)
            if take <= 0:
                continue
            picked = rng.choice(candidates, size=take, replace=False).tolist()
            for user in picked:
                result[int(user)] = onset
                remaining.discard(int(user))

    return result


def run_detection_with_relax(
    cfg: DetectConfig, counts: np.ndarray, n_inter: np.ndarray
) -> Tuple[DetectConfig, pd.DataFrame, List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]:
    """按 AUTORELAX 阶梯识别漂移：先试调用者给的阈值，不够再用更宽的阈值。

    为什么需要：MovieLens-1M 默认阈值能给出 146 人，但稀疏数据集（amazon-digital music，
    平均每用户每窗口只有约 2 条交互）用默认阈值只剩 8 人、对照组 0 人，实验二直接跑不动。
    这里逐级放宽，取第一个达到 `--min-cohort` 的级别；若全都不够，取人数最多的那一级。

    返回 (实际采用的配置, cohort 表, 阈值敏感性表, cohort 元信息, 用户×边界明细)。
    """
    rungs: List[DetectConfig] = [cfg]
    if cfg.auto_relax:
        for rung in AUTO_RELAX_LADDER[1:]:
            rungs.append(replace(cfg, **rung))

    best: Optional[Tuple[DetectConfig, object]] = None
    best_n_drift = -1
    best_index = 0
    for index, rung_cfg in enumerate(rungs):
        result = detect_drift(rung_cfg, counts, n_inter, verbose=False)
        n_drift = int(result[2]["n_drift"])
        n_stable = int(result[2]["n_stable"])
        tag = "默认" if index == 0 else f"放宽{index}"
        print(
            f"[drift] 阶梯 {index}({tag}): 边界两侧 ≥{rung_cfg.min_inter_per_window} 条 / 前后各 "
            f"≥{rung_cfg.min_inter_pooled} 条 / 取最高 {rung_cfg.drift_quantile:.0%} 分位 "
            f"(绝对下限 {rung_cfg.min_drift_excess:.2f}) → drift={n_drift} stable={n_stable}"
        )
        if n_drift > best_n_drift:
            best, best_n_drift, best_index = (rung_cfg, result), n_drift, index
        if n_drift >= cfg.min_cohort:
            break

    assert best is not None
    chosen_cfg = best[0]
    if best_index > 0:
        print(
            f"[drift] 采用阶梯 {best_index}（默认阈值不足 {cfg.min_cohort} 人，已自动放宽；"
            f"稀疏数据集上的漂移结论会明显弱于 MovieLens-1M，报告里要写明）"
        )

    # 用选中的参数再跑一次，把逐边界明细打进日志（结果确定，两次一致）
    drift_df, summary_rows, cohort_meta, boundary_rows = detect_drift(
        chosen_cfg, counts, n_inter, verbose=True
    )
    cohort_meta["auto_relax"] = bool(cfg.auto_relax)
    cohort_meta["auto_relax_rung"] = int(best_index)
    cohort_meta["auto_relax_used"] = bool(best_index > 0)
    cohort_meta["min_cohort"] = int(cfg.min_cohort)
    cohort_meta["chosen_params"] = {
        "min_inter_per_window": int(chosen_cfg.min_inter_per_window),
        "min_inter_pooled": int(chosen_cfg.min_inter_pooled),
        "drift_quantile": float(chosen_cfg.drift_quantile),
        "stable_quantile": float(chosen_cfg.stable_quantile),
        "min_drift_excess": float(chosen_cfg.min_drift_excess),
        "require_sustained": bool(chosen_cfg.require_sustained),
        "include_last_boundary": bool(chosen_cfg.include_last_boundary),
    }
    return chosen_cfg, drift_df, summary_rows, cohort_meta, boundary_rows


# ======================================================================================
# 主流程
# ======================================================================================
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


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.out_dir).resolve() if cfg.out_dir else (DEFAULT_OUT_ROOT / safe_slug(cfg.dataset))
    out_dir.mkdir(parents=True, exist_ok=True)

    log_handle = (out_dir / "val02_detect.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)  # type: ignore[assignment]

    print(f"Output directory: {out_dir}")
    print("Config:", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    meta = load_meta(cfg.dataset)
    num_users, num_items = meta["num_users"], meta["num_items"]
    full_df = load_interactions(cfg.dataset)
    windows = assign_windows(len(full_df), cfg.num_windows)
    print(
        f"[data] {cfg.dataset}: users={num_users}, items={num_items}, interactions={len(full_df)}, "
        f"windows={cfg.num_windows}"
    )
    print(f"[data] 各窗口交互数: {np.bincount(windows, minlength=cfg.num_windows).tolist()}")

    cat_matrix, cat_names, cat_meta = build_category_matrix(cfg, num_users, num_items, full_df, windows)
    print(f"[partition] 物品覆盖率 {cat_meta['item_coverage']:.2%}，类别数 {cat_meta['num_categories']}")
    if cat_meta["item_coverage"] < 0.9:
        print("[partition] 警告：物品覆盖率 < 90%，未覆盖物品的交互不计入类别分布")

    counts = compute_counts(full_df, windows, num_users, cfg.num_windows, cat_matrix)
    n_inter = compute_n_inter(full_df, windows, num_users, cfg.num_windows)
    empty_cells = int((n_inter == 0).sum())
    print(
        f"[data] (用户, 窗口) 无交互的格子：{empty_cells} / {counts.shape[0] * counts.shape[1]}"
    )
    if empty_cells > 0.5 * n_inter.size:
        print("[data] 注意：该数据集用户行为很集中，跨窗口漂移可观测的用户有限")

    # ---- 落盘：类别视图 ----
    part_rows: List[Dict[str, object]] = []
    for item in range(num_items):
        weights = cat_matrix[item]
        if weights.sum() <= 0:
            part_rows.append(
                {
                    "dataset": cfg.dataset,
                    "itemId": item,
                    "category_id": -1,
                    "category_name": "unlabeled",
                    "weight": 0.0,
                }
            )
            continue
        for cat_idx in np.flatnonzero(weights > 0):
            part_rows.append(
                {
                    "dataset": cfg.dataset,
                    "itemId": item,
                    "category_id": int(cat_idx),
                    "category_name": cat_names[int(cat_idx)],
                    "weight": float(weights[cat_idx]),
                }
            )
    write_csv(
        out_dir / "val02_partition.csv",
        part_rows,
        ["dataset", "itemId", "category_id", "category_name", "weight"],
    )
    (out_dir / "val02_category_meta.json").write_text(
        json.dumps(cat_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / "val02_counts.npz",
        counts=counts,
        n_inter=n_inter,
        category_names_json=np.asarray(json.dumps(cat_names, ensure_ascii=False)),
    )

    # ---- 落盘：用户 × 窗口画像 ----
    profiles = row_normalize(counts)
    profile_rows: List[Dict[str, object]] = []
    for user in range(num_users):
        for window_idx in range(cfg.num_windows):
            row = profiles[user, window_idx]
            if row.sum() <= 0:
                continue
            profile_rows.append(
                {
                    "dataset": cfg.dataset,
                    "user": user,
                    "window": window_idx + 1,
                    "n_inter": int(n_inter[user, window_idx]),
                    "n_inter_categorized": float(counts[user, window_idx].sum()),
                    "dominant_category": cat_names[int(row.argmax())],
                    "dominant_category_id": int(row.argmax()),
                    "entropy": entropy(row),
                    "profile": " ".join(f"{cat_names[k]}:{row[k]:.3f}" for k in np.flatnonzero(row > 0)),
                }
            )
    write_csv(
        out_dir / "val02_user_window_profile.csv",
        profile_rows,
        [
            "dataset",
            "user",
            "window",
            "n_inter",
            "n_inter_categorized",
            "dominant_category",
            "dominant_category_id",
            "entropy",
            "profile",
        ],
    )

    # ---- 漂移识别（含自动放宽阶梯：稀疏数据集上默认阈值会只剩个位数用户）----
    chosen_cfg, drift_df, summary_rows, cohort_meta, boundary_rows = run_detection_with_relax(
        cfg, counts, n_inter
    )
    write_csv(
        out_dir / "val02_user_boundary_drift.csv",
        boundary_rows,
        ["dataset", "user", "boundary", "tvd", "null_tvd", "drift_excess", "sustained", "is_best_onset", "cohort"],
    )
    drift_df.to_csv(out_dir / "val02_drift_users.csv", index=False, encoding="utf-8-sig")
    write_csv(
        out_dir / "val02_drift_summary.csv",
        summary_rows,
        ["dataset", "row_type", "threshold", "n_users", "share_of_users", "note"],
    )
    cohort_meta["category_meta"] = {
        "category_mode": cat_meta["category_mode"],
        "num_categories": cat_meta["num_categories"],
        "item_coverage": cat_meta["item_coverage"],
    }
    (out_dir / "val02_cohort_meta.json").write_text(
        json.dumps(cohort_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- 结论提示 ----
    print("\n================ 漂移识别结论 ================")
    print(
        f"  采用的阈值：边界两侧 ≥{chosen_cfg.min_inter_per_window} 条 / 前后各 ≥{chosen_cfg.min_inter_pooled} 条 / "
        f"取最高 {chosen_cfg.drift_quantile:.0%} 分位（绝对下限 {chosen_cfg.min_drift_excess:.2f}）"
        + ("  ← 已自动放宽" if cohort_meta.get("auto_relax_used") else "")
    )
    if cohort_meta["n_drift"] == 0:
        print("  ❌ 没有任何用户成为 drift cohort。可尝试：")
        print("     --min-inter-per-window 2 --min-inter-pooled 6 --min-drift-excess 0.05 --min-cohort 20")
        print("     若仍为 0：该数据集的用户行为跨窗口太集中，不适合做实验二（换 MovieLens-1M）")
    else:
        drift_part = drift_df[drift_df["cohort"] == "drift"]
        print(
            f"  兴趣变化组 {cohort_meta['n_drift']} 人（漂移分数最高的 {cohort_meta['drift_quantile']:.0%}，"
            f"阈值 {cohort_meta['drift_threshold']:.3f}），其中主类别确实改变 "
            f"{int(drift_part['primary_drift'].sum())} 人"
        )
        print(
            f"  稳定对照组 {cohort_meta['n_stable']} 人（漂移分数最低的 {cohort_meta['stable_quantile']:.0%}，"
            f"阈值 {cohort_meta['stable_threshold']:.3f}，已按 onset + 活动量分层匹配）"
        )
        print(f"  onset 分布：{cohort_meta['drift_onset_distribution']}")
        print(
            "  drift_excess 分位："
            f"p10={drift_part['drift_excess'].quantile(0.1):.3f}, "
            f"p50={drift_part['drift_excess'].quantile(0.5):.3f}, "
            f"p90={drift_part['drift_excess'].quantile(0.9):.3f}"
        )
        stable_part = drift_df[drift_df["cohort"] == "stable"]
        if len(stable_part):
            print(
                "  对照组 drift_excess 分位："
                f"p50={stable_part['drift_excess'].quantile(0.5):.3f}, "
                f"p90={stable_part['drift_excess'].quantile(0.9):.3f}"
            )
    print("==============================================")
    print(f"\n输出：{out_dir}")
    print("  val02_drift_users.csv   ← cohort 清单（训练脚本默认读它）")
    print("  val02_counts.npz        ← 类别计数张量（训练脚本算 old/new peer 占比用）")
    if cohort_meta["n_drift"] < 10:
        print(
            f"\n⚠️ drift cohort 只有 {cohort_meta['n_drift']} 人：实验二在该数据集上不可靠，"
            "批次脚本会自动跳过它、只跑其它数据集。"
        )
    print("\n下一步：")
    print("  python exp_val_1026/exp02_ema_interest_drift/val02_run_ema.py --rho 0.8 --seed 42")


def parse_args() -> DetectConfig:
    parser = argparse.ArgumentParser(description="val02 第 1 步：识别兴趣真实漂移的用户并构造类别视图")
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--category-mode", choices=["auto", "genres", "cooccur"], default="auto")
    parser.add_argument("--genres-file", default=None, help="MovieLens movies.dat 路径（有则用真实 genre）")
    parser.add_argument("--num-categories", type=int, default=18, help="cooccur 伪类别的簇数")
    parser.add_argument("--partition-scope", choices=["window1", "full"], default="window1")
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--min-inter-per-window", type=int, default=3)
    parser.add_argument("--min-inter-pooled", type=int, default=20)
    parser.add_argument("--null-trials", type=int, default=10)
    parser.add_argument("--drift-quantile", type=float, default=0.25, help="drift cohort = 漂移分数最高的这个比例")
    parser.add_argument("--stable-quantile", type=float, default=0.25, help="stable cohort = 漂移分数最低的这个比例")
    parser.add_argument("--min-drift-excess", type=float, default=0.15, help="drift cohort 的绝对下限")
    parser.add_argument("--require-sustained", action="store_true", help="强制持续性检验（ML-1M 上会把 cohort 砍到约 13%%）")
    parser.add_argument("--no-last-boundary", action="store_true", help="排除最后一个边界（无法验证持续性）")
    parser.add_argument(
        "--no-auto-relax",
        action="store_true",
        help="关闭自动放宽：只在阈值不足时才会放宽（默认开启，稀疏数据集需要它）",
    )
    parser.add_argument("--min-cohort", type=int, default=60, help="期望的最小 drift cohort 人数（不足则自动放宽）")
    parser.add_argument("--max-drift-users", type=int, default=400)
    parser.add_argument("--max-stable-users", type=int, default=400)
    parser.add_argument("--cohort-seed", type=int, default=2026)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    return DetectConfig(
        dataset=args.dataset,
        num_windows=args.num_windows,
        category_mode=args.category_mode,
        genres_file=args.genres_file,
        num_categories=args.num_categories,
        partition_scope=args.partition_scope,
        proj_dim=args.proj_dim,
        min_inter_per_window=args.min_inter_per_window,
        min_inter_pooled=args.min_inter_pooled,
        null_trials=args.null_trials,
        drift_quantile=args.drift_quantile,
        stable_quantile=args.stable_quantile,
        min_drift_excess=args.min_drift_excess,
        require_sustained=args.require_sustained,
        include_last_boundary=not args.no_last_boundary,
        auto_relax=not args.no_auto_relax,
        min_cohort=args.min_cohort,
        max_drift_users=args.max_drift_users,
        max_stable_users=args.max_stable_users,
        cohort_seed=args.cohort_seed,
        random_seed=args.random_seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
