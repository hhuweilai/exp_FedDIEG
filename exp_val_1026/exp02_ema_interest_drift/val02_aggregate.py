#!/usr/bin/env python
"""验证实验 2（val02）结果聚合：出「兴趣变化前后不同 ρ 的 NDCG@10 恢复曲线」+ 滞后表 + 自动结论。

输入：`results/raw/<dataset>/rho<ρ>/seed_<seed>/` 下的
      val02_summary.csv / val02_round_metrics.csv / val02_user_round_metrics.csv

输出（`results/aggregated/`）
--------------------------
  figures/val02_fig_ema_recovery.png     ★ 任务书要求的交付图：
                                         横轴 = 兴趣变化后经过的轮数（rel_round，0 = 变化前最后一轮），
                                         纵轴 = 该组用户的平均 NDCG@10，每条线 = 一个 ρ；
                                         实线 = 兴趣变化组（drift），虚线 = 稳定对照组（stable）
  figures/val02_fig_edge_adjust.png      边是否还挂在旧兴趣上：
                                         左 = 邻居中「旧兴趣同伴」的边权占比随轮数下降；
                                         右 = 邻居集合相对变化前的 turnover（1 - Jaccard）
  tables/val02_recovery_curve.csv        图 A 的全部数据（含每点用户数、用户级 SEM）
  tables/val02_edge_curve_*.csv          图 B 的全部数据
  tables/val02_lag_table.csv             每个 ρ 的滞后指标（下降幅度 / 恢复轮数 / 旧同伴半衰期…）
  tables/val02_lag_by_seed.csv           同上但保留到种子级（配对比较用）
  tables/val02_verdict.csv               自动结论判定
  val02_report.md                        自动填好数字的报告骨架（解释需要人工补写）

用法
----
  python exp_val_1026/exp02_ema_interest_drift/val02_aggregate.py
  python .../val02_aggregate.py --max-rel-round 240 --datasets movielens-1m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL02_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = VAL02_DIR / "results" / "raw"
DEFAULT_OUT = VAL02_DIR / "results" / "aggregated"
DEFAULT_DRIFT_ROOT = VAL02_DIR / "results" / "drift"

COHORT_LABELS = {"drift": "interest-changed users", "stable": "stable control users"}
RHO_COLORS = {0.0: "#1b9e77", 0.50: "#66a61e", 0.80: "#d95f02", 0.95: "#7570b3"}
NOISE_FLOOR = 0.002          # 与验证实验一一致的 NDCG 噪声地板
RECOVERY_FRACS = (0.25, 0.50, 0.90)
EPS_SPAN = 0.002             # 下降幅度低于这个值就认为"没有可见变化"


# ======================================================================================
# 读数据
# ======================================================================================
def safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


def _match_dataset(path: Path, datasets: Sequence[str]) -> bool:
    if not datasets:
        return True
    normalized = str(path).replace("\\", "/")
    return any(f"/{safe(d)}/" in normalized for d in datasets)


def load_user_rows(root: Path, datasets: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(root.rglob("val02_user_round_metrics.csv")):
        if not _match_dataset(path, datasets):
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:  # pragma: no cover
            print(f"[skip] {path}: {exc}")
    if not frames:
        raise SystemExit(f"在 {root} 下没有找到 val02_user_round_metrics.csv，请先跑实验。")
    data = pd.concat(frames, ignore_index=True)
    data["rho"] = data["rho"].astype(float).round(3)
    data["rel_round"] = data["rel_round"].astype(int)
    return data


def load_round_rows(root: Path, datasets: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(root.rglob("val02_round_metrics.csv")):
        if not _match_dataset(path, datasets):
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception:  # pragma: no cover
            continue
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["rho"] = data["rho"].astype(float).round(3)
    return data


# ======================================================================================
# 对齐曲线：rel_round → 组均值
# ======================================================================================
def build_curve(data: pd.DataFrame, metric: str, max_rel_round: int) -> pd.DataFrame:
    sub = data.dropna(subset=[metric, "rel_round"]).copy()
    sub = sub[(sub["rel_round"] >= -max_rel_round) & (sub["rel_round"] <= max_rel_round)]
    if sub.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for (dataset, rho, cohort, rel_round), group in sub.groupby(
        ["dataset", "rho", "cohort", "rel_round"], dropna=False
    ):
        values = group[metric].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "rho": float(rho),
                "cohort": cohort,
                "rel_round": int(rel_round),
                "mean": float(values.mean()),
                "sem_user": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
                "n_users": float(values.size),
                "n_seeds": float(group["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "rho", "cohort", "rel_round"])


def _smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    if values.size < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.concatenate([np.full(pad, values[0]), values, np.full(pad, values[-1])])
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def _first_round_reaching(
    rel_rounds: np.ndarray, values: np.ndarray, target: float, rising: bool
) -> float:
    """从 rel_round>0 的第一点开始，找第一次越过 target 的轮数（rising=True 求上穿）。"""
    order = np.argsort(rel_rounds)
    rel_rounds, values = rel_rounds[order], values[order]
    mask = rel_rounds > 0
    if mask.sum() == 0:
        return float("nan")
    rel_post, val_post = rel_rounds[mask], _smooth(values[mask])
    for rel, val in zip(rel_post, val_post):
        if rising and val >= target:
            return float(rel)
        if (not rising) and val <= target:
            return float(rel)
    return float("nan")


def analyse_curve(
    curve: pd.DataFrame, max_rel_round: int, rising: bool, early_window: int = 30
) -> Dict[str, float]:
    """给定一条对齐曲线（rel_round → mean），提取「下降幅度 / 恢复轮数 / 半衰期」。

    先按样本量裁掉右端：onset 晚的用户只有很短的 post 区间，
    若不裁掉，曲线末端会变成"只剩 onset 早的一小批用户"的均值，
    恢复轮数会被这批人的水平带偏。
    """
    if curve.empty:
        return {}
    curve = curve.sort_values("rel_round")
    rel = curve["rel_round"].to_numpy(dtype=float)
    val = curve["mean"].to_numpy(dtype=float)

    # 支撑区间：该 rel_round 的用户数不低于峰值 30%
    if "n_users" in curve.columns:
        counts = curve["n_users"].to_numpy(dtype=float)
        peak = float(np.nanmax(counts)) if counts.size else 0.0
        supported = rel[(counts >= 0.3 * peak) & (rel > 0)]
        support_cap = float(supported.max()) if supported.size else float(rel.max())
    else:
        support_cap = float(rel.max())
    cap = min(float(max_rel_round), support_cap)
    keep = rel <= cap
    rel, val = rel[keep], val[keep]

    pre_mask = (rel <= 0) & (rel >= -60)
    post_mask = (rel > 0) & (rel <= cap)
    if pre_mask.sum() == 0 or post_mask.sum() == 0:
        return {}

    pre_level = float(np.nanmean(val[pre_mask]))
    post_vals = val[post_mask]
    early_mask = (rel > 0) & (rel <= early_window)
    late_mask = (rel > 0.75 * cap) & (rel <= cap)
    post_early = float(np.nanmean(val[early_mask])) if early_mask.any() else float("nan")
    post_late = float(np.nanmean(val[late_mask])) if late_mask.any() else float(np.nanmean(post_vals))
    extreme = float(np.nanmin(post_vals)) if rising else float(np.nanmax(post_vals))

    out: Dict[str, float] = {
        "pre_level": pre_level,
        "post_early": post_early,
        "post_late": post_late,
        "post_extreme": extreme,
        "support_cap": cap,
        "gap_pre_to_early": (pre_level - post_early) if rising else (post_early - pre_level),
    }

    span = (post_late - extreme) if rising else (extreme - post_late)
    out["recoverable_span"] = float(span)
    for frac in RECOVERY_FRACS:
        if abs(span) >= EPS_SPAN:
            target = extreme + frac * span if rising else extreme - frac * span
            out[f"lag_recover_{int(frac * 100)}"] = _first_round_reaching(rel, val, target, rising)
        else:
            out[f"lag_recover_{int(frac * 100)}"] = float("nan")

    base = float(np.interp(0.0, rel, val))
    out["value_at_onset"] = base
    drift_span = (post_late - base) if rising else (base - post_late)
    if abs(drift_span) >= EPS_SPAN:
        half_target = base + 0.5 * drift_span if rising else base - 0.5 * drift_span
        out["half_lag"] = _first_round_reaching(rel, val, half_target, rising)
    else:
        out["half_lag"] = float("nan")
    return out


# ======================================================================================
# 滞后表（种子级 + 聚合）
# ======================================================================================
METRIC_SPECS = (
    ("ndcg", "ndcg", True),
    ("old_peer_share", "old_peer_share", False),
    ("new_peer_share", "new_peer_share", True),
    ("turnover_from_pre", "turnover", True),
)


def build_lag_by_seed(user_data: pd.DataFrame, max_rel_round: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (dataset, rho, seed, cohort), group in user_data.groupby(["dataset", "rho", "seed", "cohort"]):
        record: Dict[str, object] = {
            "dataset": dataset,
            "rho": float(rho),
            "seed": int(seed),
            "cohort": cohort,
            "n_users": int(group["user"].nunique()),
        }
        any_metric = False
        for metric, prefix, rising in METRIC_SPECS:
            if metric not in group.columns or group[metric].dropna().empty:
                continue
            curve = build_curve(group, metric, max_rel_round)
            stats = analyse_curve(curve, max_rel_round, rising)
            if not stats:
                continue
            any_metric = True
            for key, value in stats.items():
                record[f"{prefix}_{key}"] = value
        if not any_metric:
            continue
        rows.append(record)
    return pd.DataFrame(rows)


def aggregate_lag(lag_by_seed: pd.DataFrame, round_data: pd.DataFrame) -> pd.DataFrame:
    if lag_by_seed.empty:
        return lag_by_seed
    numeric = [c for c in lag_by_seed.columns if c not in ("dataset", "rho", "seed", "cohort")]
    lag = (
        lag_by_seed.groupby(["dataset", "rho", "cohort"], dropna=False)[numeric]
        .mean()
        .reset_index()
    )
    if not round_data.empty and "ecr_all" in round_data.columns:
        ecr = (
            round_data.groupby(["dataset", "rho"], dropna=False)["ecr_all"].mean().reset_index()
        )
        lag = lag.merge(ecr, on=["dataset", "rho"], how="left")
    return lag


# ======================================================================================
# 结论判定
# ======================================================================================
def _paired_delta(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """同一 (dataset, rho, seed) 下 drift - stable。"""
    if value not in frame.columns:
        return pd.DataFrame()
    left = frame[frame["cohort"] == "drift"][["dataset", "rho", "seed", value]].rename(columns={value: "l"})
    right = frame[frame["cohort"] == "stable"][["dataset", "rho", "seed", value]].rename(columns={value: "r"})
    merged = left.merge(right, on=["dataset", "rho", "seed"])
    if merged.empty:
        return merged
    merged["delta"] = merged["l"] - merged["r"]
    return merged


def summarise(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "same_sign": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "n": int(values.size),
        "same_sign": float(bool(np.all(values > 0)) or bool(np.all(values < 0))),
    }


def is_real_effect(stats: Dict[str, float]) -> bool:
    if stats.get("n", 0) < 2 or not np.isfinite(stats.get("mean", np.nan)):
        return False
    std = stats["std"] if np.isfinite(stats["std"]) else 0.0
    return abs(stats["mean"]) > max(NOISE_FLOOR, std) and bool(stats["same_sign"])


def build_verdict(lag: pd.DataFrame, lag_by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if lag.empty:
        return pd.DataFrame()

    for dataset, group in lag.groupby("dataset"):
        drift = group[group["cohort"] == "drift"].set_index("rho").sort_index()
        stable = group[group["cohort"] == "stable"].set_index("rho").sort_index()
        seed_group = lag_by_seed[lag_by_seed["dataset"] == dataset]
        if drift.empty:
            continue
        rhos = [float(r) for r in drift.index]

        rows.append(
            {
                "dataset": dataset,
                "evidence": "覆盖的 ρ",
                "detail": ", ".join(f"{r:.2f}" for r in rhos),
                "value": float("nan"),
                "significant": float("nan"),
            }
        )

        # ① 兴趣变化是否真的让性能变差：每个 ρ 下 drift - stable 的"变化前后落差"配对差
        if "ndcg_gap_pre_to_early" in seed_group.columns and not stable.empty:
            deltas = _paired_delta(seed_group, "ndcg_gap_pre_to_early")
            per_rho_detail: List[str] = []
            any_real = False
            for rho, sub in deltas.groupby("rho"):
                stats = summarise(sub["delta"].to_numpy(dtype=float))
                real = is_real_effect(stats)
                any_real = any_real or real
                per_rho_detail.append(
                    f"ρ{rho:.2f}:{stats['mean']:+.4f}±{stats['std']:.4f}"
                    f"({'显著' if real else '不显著'},n={stats['n']})"
                )
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": "① 兴趣变化造成的 NDCG 落差（drift - stable，正 = 变化组更差）",
                    "detail": "; ".join(per_rho_detail) if per_rho_detail else "无数据",
                    "value": float("nan"),
                    "significant": float(any_real),
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": "① 兴趣变化造成的 NDCG 落差（drift - stable）",
                    "detail": "缺少 stable 对照或该指标",
                    "value": float("nan"),
                    "significant": 0.0,
                }
            )

        # ② ρ 越大恢复越慢？
        lag_key = None
        for candidate in ("ndcg_lag_recover_90", "ndcg_lag_recover_50", "turnover_lag_recover_50"):
            if candidate in drift.columns and drift[candidate].notna().any():
                lag_key = candidate
                break
        if lag_key is not None:
            values = drift[lag_key].to_numpy(dtype=float)
            valid = np.isfinite(values)
            monotone = bool(np.all(np.diff(values[valid]) >= -1e-9)) if valid.sum() > 1 else False
            spread = float(np.nanmax(values) - np.nanmin(values)) if valid.any() else float("nan")
            slow_rho = rhos[int(np.nanargmax(values))]
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": f"② 恢复速度是否随 ρ 单调变慢（{lag_key}）",
                    "detail": "/".join(f"ρ{r:.2f}:{v:.0f}" for r, v in zip(rhos, values))
                    + f"；最慢 ρ={slow_rho:.2f}，跨度 {spread:.0f} 轮",
                    "value": spread,
                    "significant": float(monotone and np.isfinite(spread) and spread >= 20),
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": "② 恢复速度是否随 ρ 单调变慢",
                    "detail": "没有可用的恢复轮数（下降幅度低于噪声地板）",
                    "value": float("nan"),
                    "significant": 0.0,
                }
            )

        # ③ 变化前最优 ρ vs 变化后最优 ρ
        if "ndcg_pre_level" in drift.columns and "ndcg_post_late" in drift.columns:
            pre_best = float(drift["ndcg_pre_level"].idxmax())
            post_best = float(drift["ndcg_post_late"].idxmax())
            same = abs(pre_best - post_best) < 1e-9
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": "③ 变化前最优 ρ vs 变化后最优 ρ",
                    "detail": f"pre 最优 ρ={pre_best:.2f}，post 最优 ρ={post_best:.2f}"
                    + ("（同一个 ρ → 不存在权衡，自适应必要性弱）" if same else "（不同 ρ → 存在权衡，自适应有意义）"),
                    "value": float(drift.loc[post_best, "ndcg_post_late"] - drift.loc[pre_best, "ndcg_post_late"]),
                    "significant": float(not same),
                }
            )

        # ④ ρ=0（完全用新估计）是否已经足够好
        rho_zero = 0.0 if 0.0 in drift.index else float(min(drift.index))
        if "ndcg_post_late" in drift.columns:
            column = drift["ndcg_post_late"]
            better = [
                float(r) for r in drift.index
                if column.loc[r] > column.loc[rho_zero] + NOISE_FLOOR
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": f"④ 固定 ρ={rho_zero:.2f} 是否已经足够好（不需要自适应）",
                    "detail": (
                        "没有 ρ 明显优于它 → 直接把 ρ 调小即可，自适应机制的收益空间很小"
                        if not better
                        else "这些 ρ 在变化后更优：" + ", ".join(f"{r:.2f}" for r in sorted(better))
                    ),
                    "value": float(column.loc[rho_zero] - column.max()),
                    "significant": float(len(better) == 0),
                }
            )

        # ⑤ 旧兴趣同伴边权占比的半衰期
        if "old_peer_share_half_lag" in drift.columns and drift["old_peer_share_half_lag"].notna().any():
            values = drift["old_peer_share_half_lag"].to_numpy(dtype=float)
            valid = np.isfinite(values)
            monotone = bool(np.all(np.diff(values[valid]) >= -1e-9)) if valid.sum() > 1 else False
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": "⑤ 旧兴趣同伴边权占比的半衰期是否随 ρ 变长",
                    "detail": "/".join(
                        f"ρ{r:.2f}:{v:.0f}" if np.isfinite(v) else f"ρ{r:.2f}:NA"
                        for r, v in zip(rhos, values)
                    ),
                    "value": float(np.nanmax(values) - np.nanmin(values)) if valid.any() else float("nan"),
                    "significant": float(monotone),
                }
            )

        # ⑥ 总结论
        dip_row = next((r for r in rows if r["dataset"] == dataset and str(r["evidence"]).startswith("①")), None)
        lag_row = next((r for r in rows if r["dataset"] == dataset and str(r["evidence"]).startswith("②")), None)
        dip_real = bool(dip_row and dip_row["significant"] == 1.0)
        lag_real = bool(lag_row and lag_row["significant"] == 1.0)
        if dip_real and lag_real:
            conclusion = (
                "成立：兴趣真实变化确实造成性能下降，且固定 EMA 的 ρ 越大恢复/换邻居越慢 "
                "→ 自适应更新有依据"
            )
        elif dip_real and not lag_real:
            conclusion = (
                "部分成立：兴趣变化确实造成下降，但没有证据表明固定 EMA 让它恢复得更慢"
                "（任务书停止信号 2：固定 EMA 已能适应）"
            )
        elif (not dip_real) and lag_real:
            conclusion = (
                "部分成立：边关系确实按 ρ 变慢地调整，但 NDCG 层面看不到稳定代价"
                "（先检查 graph_mix 是否太小，再考虑任务书停止信号 2）"
            )
        else:
            conclusion = "不成立：既没有稳定的性能下降，也没有 ρ 造成的恢复差异（任务书停止信号 2）"
        rows.append(
            {
                "dataset": dataset,
                "evidence": "★ 总结论",
                "detail": conclusion,
                "value": float("nan"),
                "significant": float(dip_real and lag_real),
            }
        )
    return pd.DataFrame(rows)


# ======================================================================================
# 图
# ======================================================================================
def _rho_style(rho: float, index: int) -> Tuple[str, str]:
    color = RHO_COLORS.get(round(float(rho), 2))
    if color is None:
        palette = ["#1b9e77", "#66a61e", "#d95f02", "#7570b3", "#e7298a", "#666666"]
        color = palette[index % len(palette)]
    marker = "o" if rho < 0.4 else ("s" if rho < 0.7 else ("^" if rho < 0.9 else "D"))
    return color, marker


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def make_recovery_figure(curve: pd.DataFrame, out_path: Path) -> None:
    if curve.empty:
        print("[figure] 没有 NDCG 曲线数据，跳过")
        return
    try:
        plt = _plt()
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return

    datasets = list(dict.fromkeys(curve["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.8 * len(datasets), 4.9), squeeze=False)
    for panel, dataset in enumerate(datasets):
        ax = axes[0][panel]
        sub = curve[curve["dataset"] == dataset]
        rhos = sorted(sub["rho"].unique())
        for idx, rho in enumerate(rhos):
            color, marker = _rho_style(float(rho), idx)
            for cohort, style, width, alpha in (
                ("drift", "-", 2.0, 1.0),
                ("stable", "--", 1.1, 0.7),
            ):
                part = sub[(sub["rho"] == rho) & (sub["cohort"] == cohort)].sort_values("rel_round")
                if part.empty:
                    continue
                ax.plot(
                    part["rel_round"],
                    part["mean"],
                    linestyle=style,
                    color=color,
                    marker=marker,
                    markersize=3.2,
                    linewidth=width,
                    alpha=alpha,
                    label=f"ρ={rho:.2f} | {COHORT_LABELS[cohort]}",
                )
                if cohort == "drift":
                    sem = part["sem_user"].fillna(0.0).to_numpy()
                    ax.fill_between(
                        part["rel_round"].to_numpy(),
                        part["mean"].to_numpy() - sem,
                        part["mean"].to_numpy() + sem,
                        color=color,
                        alpha=0.13,
                        linewidth=0,
                    )
        ax.axvline(0, color="black", linestyle=":", linewidth=1.4)
        ax.set_title(f"{dataset}: NDCG@10 recovery around a true interest change", fontsize=10)
        ax.set_xlabel("Rounds since interest change (0 = last round before the change)")
        ax.set_ylabel("Mean NDCG@10 of the group")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


def make_delta_figure(curve: pd.DataFrame, out_path: Path, ref_rho: float = 0.0) -> None:
    """Δ NDCG 相对 ρ=ref 的差值曲线：绝对曲线几乎重合时，差值才能看清 ρ 的代价。

    每个 rel_round 上，同批用户、同一窗口 → 差值是配对量；误差带取两者用户级 SEM 的合成。
    """
    if curve.empty:
        print("[figure] 没有数据，跳过 Δ 图")
        return
    try:
        plt = _plt()
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return

    datasets = list(dict.fromkeys(curve["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.6 * len(datasets), 4.6), squeeze=False)
    for panel, dataset in enumerate(datasets):
        ax = axes[0][panel]
        sub = curve[curve["dataset"] == dataset]
        ref = sub[np.isclose(sub["rho"], ref_rho)]
        if ref.empty:
            ax.set_visible(False)
            continue
        rhos = [r for r in sorted(sub["rho"].unique()) if not np.isclose(r, ref_rho)]
        for idx, rho in enumerate(rhos):
            color, marker = _rho_style(float(rho), idx)
            for cohort, style, alpha in (("drift", "-", 1.0), ("stable", "--", 0.7)):
                part = sub[(sub["rho"] == rho) & (sub["cohort"] == cohort)].sort_values("rel_round")
                base = ref[ref["cohort"] == cohort].sort_values("rel_round")
                merged = part.merge(base, on="rel_round", suffixes=("", "_ref"))
                if merged.empty:
                    continue
                delta = merged["mean"] - merged["mean_ref"]
                sem = np.sqrt(merged["sem_user"] ** 2 + merged["sem_user_ref"] ** 2)
                ax.plot(
                    merged["rel_round"],
                    delta,
                    linestyle=style,
                    color=color,
                    marker=marker,
                    markersize=3.2,
                    linewidth=2.0 if cohort == "drift" else 1.2,
                    alpha=alpha,
                    label=f"ρ={rho:.2f} − ρ={ref_rho:.2f} | {COHORT_LABELS[cohort]}",
                )
                if cohort == "drift":
                    ax.fill_between(
                        merged["rel_round"].to_numpy(),
                        (delta - sem).to_numpy(),
                        (delta + sem).to_numpy(),
                        color=color,
                        alpha=0.13,
                        linewidth=0,
                    )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.4)
        ax.axvline(0, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(f"{dataset}: ΔNDCG@10 relative to ρ={ref_rho:.2f}", fontsize=10)
        ax.set_xlabel("Rounds since interest change (0 = last round before the change)")
        ax.set_ylabel("Δ mean NDCG@10 (vs ρ={:.2f})".format(ref_rho))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


def detect_window_rounds(root: Path, datasets: Sequence[str], default: int = 120) -> int:
    """从任意一次运行的 summary 里读每窗口轮数（用于把曲线裁到"同批用户都在"的窗口内）。"""
    counts: Dict[int, int] = {}
    for path in sorted(root.rglob("val02_summary.csv")):
        if not _match_dataset(path, datasets):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:  # pragma: no cover
            continue
        if "rounds_per_window" in frame.columns and len(frame):
            value = int(frame["rounds_per_window"].iloc[0])
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return default
    return max(counts, key=counts.get)


def make_edge_figure(old_curve: pd.DataFrame, turn_curve: pd.DataFrame, out_path: Path) -> None:
    if old_curve.empty and turn_curve.empty:
        print("[figure] 没有边指标曲线数据，跳过")
        return
    try:
        plt = _plt()
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return

    source = old_curve if not old_curve.empty else turn_curve
    datasets = list(dict.fromkeys(source["dataset"]))
    fig, axes = plt.subplots(1, 2 * len(datasets), figsize=(6.0 * 2 * len(datasets), 4.4), squeeze=False)
    for panel, dataset in enumerate(datasets):
        for col_idx, (curve, title, ylabel) in enumerate(
            (
                (old_curve, "edge weight on OLD-interest peers", "Old-peer weight share"),
                (turn_curve, "neighbour turnover vs pre-change snapshot", "1 - Jaccard (turnover)"),
            )
        ):
            ax = axes[0][panel * 2 + col_idx]
            sub = curve[curve["dataset"] == dataset] if not curve.empty else curve
            if sub is None or sub.empty:
                ax.set_visible(False)
                continue
            rhos = sorted(sub["rho"].unique())
            for idx, rho in enumerate(rhos):
                color, marker = _rho_style(float(rho), idx)
                part = sub[(sub["rho"] == rho) & (sub["cohort"] == "drift")].sort_values("rel_round")
                part = part[part["rel_round"] >= 0]
                if part.empty:
                    continue
                ax.plot(
                    part["rel_round"],
                    part["mean"],
                    color=color,
                    marker=marker,
                    markersize=3.2,
                    linewidth=2.0,
                    label=f"ρ={rho:.2f}",
                )
            ax.set_title(f"{dataset}: {title}", fontsize=9)
            ax.set_xlabel("Rounds since interest change")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


# ======================================================================================
# Markdown 报告骨架
# ======================================================================================
def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "（空表）"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, sep] + body)


def write_report(
    path: Path, lag: pd.DataFrame, verdict: pd.DataFrame, cohort_meta: Dict[str, Dict[str, object]]
) -> None:
    lines: List[str] = ["# 验证实验 2（val02）：兴趣真实变化时，固定 EMA 是否适应太慢？\n"]
    lines.append("> 本文件的数字由 `val02_aggregate.py` 自动填充，解释性文字需要人工补写。\n")

    lines.append("\n## 0. cohort 概况（按数据集）\n")
    if cohort_meta:
        for dataset, meta in cohort_meta.items():
            lines.append(f"\n### {dataset}\n")
            for key in (
                "category_mode",
                "n_usable",
                "n_drift",
                "n_stable",
                "drift_threshold",
                "stable_threshold",
                "auto_relax_used",
            ):
                if key in meta and meta[key] is not None:
                    lines.append(f"- {key}: `{meta[key]}`")
            chosen = meta.get("chosen_params")
            if isinstance(chosen, dict):
                lines.append(
                    "- 实际阈值: "
                    f"边界两侧 ≥{chosen.get('min_inter_per_window')} 条 / 前后各 "
                    f"≥{chosen.get('min_inter_pooled')} 条 / 取最高 "
                    f"{float(chosen.get('drift_quantile', 0)):.0%} 分位"
                )
            dist = meta.get("drift_onset_distribution")
            if dist:
                lines.append(f"- onset 分布: `{dist}`")
    lines.append(
        "\n> cohort 由离线类别统计选出，只用于事后分组，不进入模型；"
        "在线算法每个窗口只能看到该窗口已到达的交互。\n"
        "> 不同数据集的阈值可能不同（稀疏数据集会自动放宽），跨数据集比较时只看趋势，不要比绝对值。\n"
    )

    lines.append("\n## 1. 各 ρ 的关键指标（兴趣变化组）\n")
    if not lag.empty:
        drift = lag[lag["cohort"] == "drift"].copy()
        keep = [
            c
            for c in (
                "dataset",
                "rho",
                "n_users",
                "ndcg_pre_level",
                "ndcg_post_early",
                "ndcg_post_late",
                "ndcg_gap_pre_to_early",
                "ndcg_lag_recover_50",
                "ndcg_lag_recover_90",
                "old_peer_share_value_at_onset",
                "old_peer_share_post_late",
                "old_peer_share_half_lag",
                "turnover_value_at_onset",
                "turnover_post_late",
                "ecr_all",
            )
            if c in drift.columns
        ]
        lines.append(_to_markdown(drift[keep].round(4).reset_index(drop=True)))
    else:
        lines.append("（没有数据）")

    lines.append("\n## 2. 自动结论判定\n")
    if not verdict.empty:
        for _, row in verdict.iterrows():
            if str(row["evidence"]).startswith("★"):
                lines.append(f"\n**{row['detail']}**\n")
            else:
                lines.append(f"- {row['evidence']}：{row['detail']}")
    else:
        lines.append("（没有数据）")

    lines.append("\n## 3. 读图提示\n")
    lines.append(
        "- 横轴 0 是「兴趣变化前的最后一轮」；0 的右侧才是变化之后。\n"
        "- 实线 = 兴趣变化组，虚线 = 活动量匹配的稳定对照组。"
        "真正的「漂移代价」应看 0 右侧两条线的**间距**，而不是实线自身的高低。\n"
        "- 若 4 条实线在 0 右侧几乎重合，说明固定 EMA 的取值对推荐性能没有可见影响"
        "（任务书停止信号 2）；此时结合 `val02_fig_edge_adjust.png` 判断是不是图权重太小。\n"
        "- 每个 rel_round 的实际样本量在 `val02_recovery_curve.csv` 的 `n_users` 列；"
        "样本量掉到 cohort 30% 以下的点不要当作结论。\n"
    )

    lines.append("\n## 4. 待人工补写\n")
    lines.append(
        "- 实验设置与控制变量：ρ 是唯一自变量；cohort 固定；参与序列与随机种子在四个 ρ 间完全一致。\n"
        "- 现象描述：下降幅度、恢复轮数、旧同伴占比半衰期是否随 ρ 单调变化。\n"
        "- 其他解释：窗口边界同时换掉了测试样本，NDCG 变化里有多少来自「新兴趣本身更难预测」。\n"
        "- 结论与下一步。\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] 已保存：{path}")


# ======================================================================================
# 主流程
# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="val02 结果聚合：恢复曲线 + 滞后表 + 自动结论")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--drift-root", type=Path, default=DEFAULT_DRIFT_ROOT)
    parser.add_argument("--datasets", default="", help="逗号分隔；留空 = 全部")
    parser.add_argument("--max-rel-round", type=int, default=240)
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="恢复曲线用完整 rel_round 跨度（默认裁到 ±1 个窗口，避免窗口边界台阶掩盖 ρ 之间的差异）",
    )
    parser.add_argument(
        "--narrow-rounds",
        type=int,
        default=None,
        help="自定义裁切半径（默认 = 每窗口轮数，从 summary 自动读取）",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out.resolve()
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    user_data = load_user_rows(root, datasets)
    round_data = load_round_rows(root, datasets)
    print(f"Loaded {len(user_data)} user-round rows from {root}")
    print(
        user_data.groupby(["dataset", "rho", "cohort"])["user"].nunique().unstack(fill_value=0).to_string()
    )

    ndcg_curve = build_curve(user_data, "ndcg", args.max_rel_round)
    old_curve = build_curve(user_data, "old_peer_share", args.max_rel_round)
    new_curve = build_curve(user_data, "new_peer_share", args.max_rel_round)
    turn_curve = build_curve(user_data, "turnover_from_pre", args.max_rel_round)

    # 恢复曲线默认裁到 ±1 个窗口：那里"同一批用户、相邻两个窗口"，没有窗口组成变化造成的台阶。
    # 想看全量跨度就用 --full-range。
    window_rounds = detect_window_rounds(root, datasets)
    if not args.full_range:
        limit = int(args.narrow_rounds or window_rounds)
        print(f"[curve] 恢复曲线裁到 rel_round ∈ [-{limit}, +{limit}]（每窗口 {window_rounds} 轮；--full-range 可关闭）")
        if not ndcg_curve.empty:
            ndcg_curve = ndcg_curve[
                (ndcg_curve["rel_round"] >= -limit) & (ndcg_curve["rel_round"] <= limit)
            ]
        if not old_curve.empty:
            old_curve = old_curve[
                (old_curve["rel_round"] >= -limit) & (old_curve["rel_round"] <= limit)
            ]
        if not new_curve.empty:
            new_curve = new_curve[
                (new_curve["rel_round"] >= -limit) & (new_curve["rel_round"] <= limit)
            ]
        if not turn_curve.empty:
            turn_curve = turn_curve[
                (turn_curve["rel_round"] >= -limit) & (turn_curve["rel_round"] <= limit)
            ]

    if not ndcg_curve.empty:
        ndcg_curve.to_csv(tables_dir / "val02_recovery_curve.csv", index=False, encoding="utf-8-sig")
    for name, curve in (
        ("old_peer_share", old_curve),
        ("new_peer_share", new_curve),
        ("turnover", turn_curve),
    ):
        if not curve.empty:
            curve.to_csv(tables_dir / f"val02_edge_curve_{name}.csv", index=False, encoding="utf-8-sig")

    lag_by_seed = build_lag_by_seed(user_data, args.max_rel_round)
    if not lag_by_seed.empty:
        lag_by_seed.to_csv(tables_dir / "val02_lag_by_seed.csv", index=False, encoding="utf-8-sig")
    lag = aggregate_lag(lag_by_seed, round_data)
    if not lag.empty:
        lag.to_csv(tables_dir / "val02_lag_table.csv", index=False, encoding="utf-8-sig")
    verdict = build_verdict(lag, lag_by_seed)
    if not verdict.empty:
        verdict.to_csv(tables_dir / "val02_verdict.csv", index=False, encoding="utf-8-sig")

    make_recovery_figure(ndcg_curve, figures_dir / "val02_fig_ema_recovery.png")
    make_edge_figure(old_curve, turn_curve, figures_dir / "val02_fig_edge_adjust.png")
    make_delta_figure(ndcg_curve, figures_dir / "val02_fig_ema_delta.png")

    cohort_meta: Dict[str, Dict[str, object]] = {}
    for meta_path in sorted(args.drift_root.resolve().rglob("val02_cohort_meta.json")):
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            cohort_meta[str(meta.get("dataset", meta_path.parent.name))] = meta
        except Exception:  # pragma: no cover
            continue
    write_report(out_dir / "val02_report.md", lag, verdict, cohort_meta)

    if not verdict.empty:
        print("\n================ 结论判定 ================")
        for _, row in verdict.iterrows():
            if str(row["evidence"]).startswith("★"):
                print(f"  [{row['dataset']}] {row['detail']}")
            else:
                print(f"  [{row['dataset']}] {row['evidence']}：{row['detail']}")
        print("==========================================")
    if not round_data.empty:
        # 与其他实验口径一致：取每个窗口最后一次评测，再跨窗口/种子平均
        last = (
            round_data.sort_values("window_round")
            .groupby(["dataset", "rho", "seed", "window"])["all_ndcg"]
            .last()
            .reset_index()
        )
        print("\n全体 NDCG@10（每个窗口最后一次评测的均值）:")
        print(last.groupby(["dataset", "rho"])["all_ndcg"].mean().round(4).to_string())
    print(f"\n输出：{out_dir}")


if __name__ == "__main__":
    main()
