"""阶段 1：零训练重算（只读 raw，不训练、不改任何实验代码）。

产出（val_DynamicGraph/results/recheck/）：
  recheck_seed_level.csv          主对照的「种子级」统计（n=3）与「窗口级」（n=15）对照
  recheck_freeze_cells.csv        冻结 2x2 网格的冻结指标（均值/末值/分母）
  recheck_freeze_trajectory.csv   冻结指标的逐评测点轨迹（用于画/查「越冻越深」）
  recheck_freeze_contrasts.csv    冻结网格的种子级配对（ndcg/frozen/turnover/ecr）
  recheck_coverage_round.csv      各臂逐评测轮的有边用户数（覆盖轨迹）
  recheck_coverage_windows.csv    各臂逐窗口的覆盖（窗口内首/末/峰值）
  recheck_blindspot.csv           periodic 的窗口开头盲区与刷新后追平比例
  recheck_consistency.csv         与已提交的 dg_arm_window.csv 交叉核对
  recheck_console.txt             人读的控制台摘要

用法：
  F:\\Anaconda\\python.exe -X utf8 val_DynamicGraph/recheck_phase1.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DG_DIR = Path(__file__).resolve().parent
RAW = DG_DIR / "results" / "raw"
AGG = DG_DIR / "results" / "aggregated" / "tables"

DATASETS: Dict[str, int] = {"movielens-1m": 6040, "amazon-digital_music": 12381}

CORE_ARMS = ["nograph", "static", "periodic", "dynamic", "static_at120"]

#: 主对照（a − b），与 dg_aggregate.py 的命名保持一致
MAIN_CONTRASTS: List[Tuple[str, str, str]] = [
    ("static_minus_nograph", "static", "nograph"),
    ("dynamic_minus_static", "dynamic", "static"),
    ("dynamic_minus_nograph", "dynamic", "nograph"),
    ("periodic_minus_nograph", "periodic", "nograph"),
    ("dynamic_minus_periodic", "dynamic", "periodic"),
    ("static_at120_minus_static", "static_at120", "static"),
]

#: 冻结 2x2 网格：cell -> 目录名（参照格是 paper 默认 dynamic）
FREEZE_CELLS: List[Tuple[str, str]] = [
    ("paper_default_pool_zero_rho0.2", "dynamic"),
    ("pool_zero_rho0.8", "dynamic_rho0.8"),
    ("pool_warm_rho0.8", "dynamic_rho0.8_warm"),
    ("topk_zero_rho0.8", "dynamic_rho0.8_topk"),
    ("topk_warm_rho0.8", "dynamic_rho0.8_topk_warm"),
]

#: 冻结网格的种子级配对（标签, a_dir, b_dir）—— 注意用目录名，不是 cell 名
FREEZE_CONTRASTS: List[Tuple[str, str, str]] = [
    ("topk_zero - pool_zero", "dynamic_rho0.8_topk", "dynamic_rho0.8"),
    ("topk_warm - topk_zero", "dynamic_rho0.8_topk_warm", "dynamic_rho0.8_topk"),
    ("pool_warm - pool_zero", "dynamic_rho0.8_warm", "dynamic_rho0.8"),
    ("pool_zero(rho0.8) - pool_zero(rho0.2)", "dynamic_rho0.8", "dynamic"),
]

FREEZE_METRICS = ["ndcg_win_mean", "frozen_share", "turnover_from_ref", "ecr_all"]

#: 已提交的聚合表用于交叉核对
AGG_WINDOW = AGG / "dg_arm_window.csv"


# ======================================================================================
# 统计工具
# ======================================================================================
def t_crit(df: int) -> float:
    """双侧 95% t 临界值（df 很小，硬编码兜底，保证不依赖 scipy）。"""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145}
    if df in table:
        return table[df]
    try:  # pragma: no cover
        from scipy import stats

        return float(stats.t.ppf(0.975, df))
    except Exception:
        return float("nan")


def t_ci(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))
    se = float(arr.std(ddof=1) / math.sqrt(n))
    h = t_crit(n - 1) * se
    return (float(arr.mean() - h), float(arr.mean() + h))


def paired_p(values: Sequence[float]) -> float:
    """单样本 t 检验（H0: 均值=0），返回双侧 p。"""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size < 2:
        return float("nan")
    try:
        from scipy import stats

        return float(stats.ttest_1samp(arr, 0.0).pvalue)
    except Exception:  # pragma: no cover
        return float("nan")


# ======================================================================================
# 读取 raw
# ======================================================================================
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] 读不了 {path}: {exc}")
        return pd.DataFrame()


def load_rounds(dataset_dir: str) -> pd.DataFrame:
    frames = []
    base = RAW / dataset_dir
    for arm_dir in sorted(p.name for p in base.iterdir() if p.is_dir()):
        for seed_dir in sorted((base / arm_dir).glob("seed_*")):
            f = seed_dir / "dg_round_metrics.csv"
            df = read_csv_safe(f)
            if df.empty:
                continue
            df["dataset_dir"] = dataset_dir
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["seed"] = out["seed"].astype(int)
    out["global_round"] = out["global_round"].astype(int)
    out["window"] = out["window"].astype(int)
    return out


def window_table(rounds: pd.DataFrame) -> pd.DataFrame:
    """与 dg_aggregate.window_table 等价：每个 (arm, seed, window) 的指标均值/峰值。"""
    g = rounds.groupby(["dataset_dir", "arm_dir", "seed", "window"], dropna=False)
    out = g.agg(
        ndcg_win_mean=("all_ndcg", "mean"),
        hr_win_mean=("all_hr", "mean"),
        ndcg_win_last=("all_ndcg", "last"),
        n_eval_points=("all_ndcg", "size"),
        n_eval=("n_eval", "max"),
        n_edges_max=("graph_n_edges", "max"),
        n_edges_first=("graph_n_edges", "first"),
        n_edges_last=("graph_n_edges", "last"),
        graph_updates=("graph_updates", "max"),
        ecr=("ecr_all", "mean"),
        added=("added_mean", "mean"),
    ).reset_index()
    return out


def contrasts_from_window(win: pd.DataFrame, contrasts: Sequence[Tuple[str, str, str]],
                          value_col: str = "ndcg_win_mean") -> pd.DataFrame:
    """返回每个 (contrast, seed, window) 的 a − b。"""
    rows = []
    for label, a, b in contrasts:
        pa = win[win["arm_dir"] == a].set_index(["seed", "window"])[value_col]
        pb = win[win["arm_dir"] == b].set_index(["seed", "window"])[value_col]
        joined = pd.concat({"a": pa, "b": pb}, axis=1, join="inner").dropna()
        for (seed, window), row in joined.iterrows():
            rows.append({"contrast": label, "seed": int(seed), "window": int(window),
                         "delta": float(row["a"] - row["b"])})
    return pd.DataFrame(rows)


def summarise(part: pd.DataFrame) -> Dict[str, float]:
    """种子级（每种子先对窗口求均值）+ 窗口级（所有窗口观测）两套统计。"""
    per_seed = part.groupby("seed")["delta"].mean()
    d_seed = per_seed.to_numpy(dtype=float)
    d_win = part["delta"].to_numpy(dtype=float)
    ci_s = t_ci(d_seed)
    ci_w = t_ci(d_win)
    return {
        "n_seeds": int(d_seed.size),
        "n_obs_windows": int(d_win.size),
        "delta_seed_mean": float(np.mean(d_seed)),
        "delta_seed_sd": float(np.std(d_seed, ddof=1)) if d_seed.size > 1 else float("nan"),
        "delta_seed_min": float(np.min(d_seed)),
        "delta_seed_max": float(np.max(d_seed)),
        "ci95_seed_lo": ci_s[0],
        "ci95_seed_hi": ci_s[1],
        "seeds_positive": int((d_seed > 0).sum()),
        "p_seed_t": paired_p(d_seed),
        "delta_window_mean": float(np.mean(d_win)),
        "ci95_window_lo": ci_w[0],
        "ci95_window_hi": ci_w[1],
        "window_positive_rate": float((d_win > 0).mean()),
        "p_window_t": paired_p(d_win),
    }


# ======================================================================================
# 冻结指标
# ======================================================================================
def build_freeze_cells(rounds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """每个 (dataset, cell, seed) 的冻结指标 + 逐评测点轨迹。"""
    cell_of = {dir_name: cell for cell, dir_name in FREEZE_CELLS}
    sub = rounds[rounds["arm_dir"].isin(cell_of)].copy()
    sub["cell"] = sub["arm_dir"].map(cell_of)

    rows, traj = [], []
    for (ds, cell, seed), g in sub.groupby(["dataset_dir", "cell", "seed"], dropna=False):
        g = g.sort_values("global_round")
        fz = g["frozen_share"].to_numpy(dtype=float) if "frozen_share" in g else np.array([])
        to = g["turnover_from_ref"].to_numpy(dtype=float) if "turnover_from_ref" in g else np.array([])
        ref = g["freeze_ref_n"].to_numpy(dtype=float) if "freeze_ref_n" in g else np.array([])
        cov = g["coverage_gain"].to_numpy(dtype=float) if "coverage_gain" in g else np.array([])
        finite = np.isfinite(fz)
        n_users = DATASETS.get(ds, 0)
        ref_n = float(np.nanmin(ref[ref > 0])) if np.isfinite(ref).any() and np.nanmax(ref) > 0 else float("nan")
        cov_last = float(np.nanmax(cov)) if np.isfinite(cov).any() else float("nan")
        rounds_sorted = g["global_round"].to_numpy(dtype=int)
        finite_rounds = rounds_sorted[finite] if finite.size == rounds_sorted.size else np.array([], dtype=int)
        # 三种口径：逐点池化 / 剔除参考点 / 窗口等权（复刻 dg_freeze.cell_summary）
        fz_finite = fz[finite]
        frozen_pooled = float(np.nanmean(fz_finite)) if finite.any() else float("nan")
        frozen_excl_ref = float(np.nanmean(fz_finite[1:])) if fz_finite.size > 1 else float("nan")
        win_means = []
        for _, wg in g.groupby("window"):
            col = wg["frozen_share"].to_numpy(dtype=float)
            if np.isfinite(col).any():
                win_means.append(float(np.nanmean(col)))
        frozen_window_eq = float(np.mean(win_means)) if win_means else float("nan")
        to_finite = to[np.isfinite(to)]
        turnover_pooled = float(np.nanmean(to_finite)) if to_finite.size else float("nan")
        turnover_excl_ref = float(np.nanmean(to_finite[1:])) if to_finite.size > 1 else float("nan")
        rows.append({
            "dataset_dir": ds, "cell": cell, "seed": int(seed),
            "arm_dir": g["arm_dir"].iloc[0],
            "n_eval_rows": int(len(g)),
            "n_finite_frozen": int(finite.sum()),
            "ref_round": int(finite_rounds[0]) if finite_rounds.size else -1,
            "ref_n": ref_n,
            "pct_ref_vs_users": ref_n / n_users if n_users and np.isfinite(ref_n) else float("nan"),
            "coverage_gain_last": cov_last,
            "frozen_pooled": frozen_pooled,
            "frozen_excl_ref": frozen_excl_ref,
            "frozen_window_equal": frozen_window_eq,
            "frozen_last": float(fz_finite[-1]) if finite.any() else float("nan"),
            "frozen_first": float(fz_finite[0]) if finite.any() else float("nan"),
            "turnover_pooled": turnover_pooled,
            "turnover_excl_ref": turnover_excl_ref,
            "turnover_last": float(to_finite[-1]) if to_finite.size else float("nan"),
            "ecr_mean": float(np.nanmean(g["ecr_all"].to_numpy(dtype=float))),
        })
        for _, r in g.iterrows():
            traj.append({
                "dataset_dir": ds, "cell": cell, "seed": int(seed),
                "global_round": int(r["global_round"]),
                "n_edges": float(r.get("graph_n_edges", np.nan)),
                "ref_n": float(r.get("freeze_ref_n", np.nan)),
                "turnover_from_ref": float(r.get("turnover_from_ref", np.nan)),
                "frozen_share": float(r.get("frozen_share", np.nan)),
                "coverage_gain": float(r.get("coverage_gain", np.nan)),
                "ecr_all": float(r.get("ecr_all", np.nan)),
            })
    return pd.DataFrame(rows), pd.DataFrame(traj)


def build_freeze_contrasts(rounds: pd.DataFrame) -> pd.DataFrame:
    """冻结网格的种子级配对小表（ndcg / frozen / turnover / ecr）。"""
    win = window_table(rounds)
    # window_table 只有 ndcg；冻结指标要按 (seed, window) 重新聚合
    g = rounds.groupby(["dataset_dir", "arm_dir", "seed", "window"], dropna=False)
    fz = g.agg(
        ndcg_win_mean=("all_ndcg", "mean"),
        frozen_share=("frozen_share", "mean"),
        turnover_from_ref=("turnover_from_ref", "mean"),
        ecr_all=("ecr_all", "mean"),
    ).reset_index()

    cell_of = {dir_name: cell for cell, dir_name in FREEZE_CELLS}
    rows = []
    for ds, part in fz[fz["arm_dir"].isin(cell_of)].groupby("dataset_dir"):
        for label, a, b in FREEZE_CONTRASTS:
            for metric in FREEZE_METRICS:
                pa = part[part["arm_dir"] == a].set_index(["seed", "window"])[metric]
                pb = part[part["arm_dir"] == b].set_index(["seed", "window"])[metric]
                joined = pd.concat({"a": pa, "b": pb}, axis=1, join="inner").dropna()
                if joined.empty:
                    continue
                joined["delta"] = joined["a"] - joined["b"]
                statsrow = summarise(joined.reset_index()[["seed", "delta"]])
                rows.append({"dataset_dir": ds, "contrast": label, "metric": metric, **statsrow})
    return pd.DataFrame(rows)


# ======================================================================================
# 覆盖轨迹 / 盲区
# ======================================================================================
def build_coverage(rounds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = rounds[rounds["arm_dir"].isin(CORE_ARMS)].copy()
    long = sub[["dataset_dir", "arm_dir", "seed", "global_round", "window",
                "graph_n_edges", "graph_updates"]].sort_values(
        ["dataset_dir", "arm_dir", "seed", "global_round"])

    win = window_table(sub)
    wide = win.pivot_table(index=["dataset_dir", "arm_dir", "seed", "window"],
                           values=["n_edges_max", "n_edges_last"], aggfunc="max").reset_index()
    wide = wide.sort_values(["dataset_dir", "arm_dir", "seed", "window"])
    wide["new_users_in_window"] = wide.groupby(["dataset_dir", "arm_dir", "seed"])["n_edges_max"].diff()
    return long, wide


def build_blindspot(rounds: pd.DataFrame) -> pd.DataFrame:
    """periodic 在窗口开头 10 轮的覆盖盲区，以及刷新后相对 dynamic 的追平比例。"""
    rows = []
    sub = rounds[rounds["arm_dir"].isin(["periodic", "dynamic", "static"])]
    piv = sub.pivot_table(index=["dataset_dir", "seed", "global_round"],
                          columns="arm_dir", values="graph_n_edges", aggfunc="max").reset_index()
    for ds, part in piv.groupby("dataset_dir"):
        for w in range(2, 6):
            start = 120 * (w - 1) + 1
            r_pre = start + 9          # 窗口内第 10 轮（刷新还没发生）
            r_post = start + 29        # 刷新（第 21 轮）之后的第一个评测点
            for tag, r in (("pre_refresh", r_pre), ("post_refresh", r_post)):
                hit = part[part["global_round"] == r]
                for _, rr in hit.iterrows():
                    dyn = rr.get("dynamic", np.nan)
                    per = rr.get("periodic", np.nan)
                    sta = rr.get("static", np.nan)
                    rows.append({
                        "dataset_dir": ds, "window": w, "round": r, "stage": tag,
                        "seed": int(rr["seed"]),
                        "dynamic_edges": float(dyn) if np.isfinite(dyn) else float("nan"),
                        "periodic_edges": float(per) if np.isfinite(per) else float("nan"),
                        "static_edges": float(sta) if np.isfinite(sta) else float("nan"),
                        "periodic_over_dynamic": float(per / dyn) if dyn and np.isfinite(dyn) and np.isfinite(per) else float("nan"),
                        "periodic_minus_static": float(per - sta) if np.isfinite(per) and np.isfinite(sta) else float("nan"),
                    })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    agg = out.groupby(["dataset_dir", "window", "round", "stage"], as_index=False).agg(
        dynamic_edges=("dynamic_edges", "mean"),
        periodic_edges=("periodic_edges", "mean"),
        static_edges=("static_edges", "mean"),
        periodic_over_dynamic=("periodic_over_dynamic", "mean"),
    )
    return agg


# ======================================================================================
# 交叉核对
# ======================================================================================
def build_consistency(win: pd.DataFrame) -> pd.DataFrame:
    if not AGG_WINDOW.exists():
        return pd.DataFrame()
    ref = pd.read_csv(AGG_WINDOW)
    ref = ref.rename(columns={"dataset": "dataset_dir"})
    ref["dataset_dir"] = ref["dataset_dir"].str.replace(" ", "_", regex=False)
    keys = ["dataset_dir", "arm_dir", "seed", "window"]
    mine = win[keys + ["ndcg_win_mean", "n_edges_max"]].copy()
    merged = mine.merge(ref[keys + ["ndcg_win_mean", "n_edges"]], on=keys, how="inner",
                        suffixes=("_mine", "_agg"))
    if merged.empty:
        return pd.DataFrame()
    merged["abs_diff_ndcg"] = (merged["ndcg_win_mean_mine"] - merged["ndcg_win_mean_agg"]).abs()
    merged["abs_diff_n_edges"] = (merged["n_edges_max"] - merged["n_edges"]).abs()
    out = merged.groupby(["dataset_dir", "arm_dir"], as_index=False).agg(
        n_matched=("abs_diff_ndcg", "size"),
        max_abs_diff_ndcg=("abs_diff_ndcg", "max"),
        max_abs_diff_n_edges=("abs_diff_n_edges", "max"),
    )
    return out


# ======================================================================================
# 主流程
# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 1：零训练重算（只读 raw）")
    ap.add_argument("--out", default=str(DG_DIR / "results" / "recheck"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    def say(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    all_rounds = []
    for ds in DATASETS:
        r = load_rounds(ds)
        if r.empty:
            say(f"[warn] {ds} 没有 raw 数据")
            continue
        all_rounds.append(r)
        say(f"[data] {ds}: {len(r)} 行 round metrics，"
            f"{r['arm_dir'].nunique()} 个臂/变体，{r['seed'].nunique()} 个种子")
    rounds = pd.concat(all_rounds, ignore_index=True, sort=False)
    win = window_table(rounds)

    # ---------------- 1) 主对照的种子级统计
    main_rows = []
    for ds, part in win.groupby("dataset_dir"):
        c = contrasts_from_window(part, MAIN_CONTRASTS)
        for label, g in c.groupby("contrast"):
            main_rows.append({"dataset_dir": ds, "contrast": label, **summarise(g)})
    seed_level = pd.DataFrame(main_rows)
    seed_level.to_csv(out_dir / "recheck_seed_level.csv", index=False, encoding="utf-8-sig")

    say("\n================ 1) 主对照：种子级 vs 窗口级 ================")
    say("(Δ 单位 NDCG@10；种子级 n=3，窗口级 n=15)")
    for _, r in seed_level.iterrows():
        say(f"  {r['dataset_dir']:<22} {r['contrast']:<28} "
            f"种子级 {r['delta_seed_mean']:+.4f} [{r['ci95_seed_lo']:+.4f},{r['ci95_seed_hi']:+.4f}]"
            f" 种子 {r['seeds_positive']}/{r['n_seeds']} 为正 p={r['p_seed_t']:.3g} | "
            f"窗口级 {r['delta_window_mean']:+.4f} [{r['ci95_window_lo']:+.4f},{r['ci95_window_hi']:+.4f}]"
            f" p={r['p_window_t']:.3g}")

    # ---------------- 2) 冻结指标
    cells, traj = build_freeze_cells(rounds)
    cells.to_csv(out_dir / "recheck_freeze_cells.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(out_dir / "recheck_freeze_trajectory.csv", index=False, encoding="utf-8-sig")

    fail = build_freeze_contrasts(rounds)
    fail.to_csv(out_dir / "recheck_freeze_contrasts.csv", index=False, encoding="utf-8-sig")

    say("\n================ 2) 冻结指标（三种平均口径 + 分母）================")
    agg_cells = cells.groupby(["dataset_dir", "cell"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        ref_n=("ref_n", "mean"),
        pct_ref_vs_users=("pct_ref_vs_users", "mean"),
        coverage_gain_last=("coverage_gain_last", "mean"),
        frozen_pooled=("frozen_pooled", "mean"),
        frozen_excl_ref=("frozen_excl_ref", "mean"),
        frozen_window_equal=("frozen_window_equal", "mean"),
        frozen_last=("frozen_last", "mean"),
        turnover_pooled=("turnover_pooled", "mean"),
        turnover_excl_ref=("turnover_excl_ref", "mean"),
        turnover_last=("turnover_last", "mean"),
        ecr_mean=("ecr_mean", "mean"),
    )
    for _, r in agg_cells.iterrows():
        say(f"  {r['dataset_dir']:<22} {r['cell']:<32} ref_n={r['ref_n']:.0f}"
            f" ({r['pct_ref_vs_users']:.1%}) cov_gain={r['coverage_gain_last']:.0f} | "
            f"frozen 池化 {r['frozen_pooled']:.3f} 剔参考点 {r['frozen_excl_ref']:.3f} "
            f"窗口等权 {r['frozen_window_equal']:.3f} 末值 {r['frozen_last']:.3f} | "
            f"turnover 池化 {r['turnover_pooled']:.3f} 末值 {r['turnover_last']:.3f} | "
            f"ECR {r['ecr_mean']:.4f}")

    say("\n  -- 冻结网格的种子级配对（n=3）--")
    for _, r in fail.iterrows():
        if r["metric"] not in ("ndcg_win_mean", "frozen_share", "turnover_from_ref"):
            continue
        say(f"  {r['dataset_dir']:<22} {r['contrast']:<42} {r['metric']:<18} "
            f"{r['delta_seed_mean']:+.4f} [{r['ci95_seed_lo']:+.4f},{r['ci95_seed_hi']:+.4f}]"
            f" 种子 {r['seeds_positive']}/{r['n_seeds']} 为正")

    # ---------------- 3) 覆盖轨迹与盲区
    cov_long, cov_win = build_coverage(rounds)
    cov_long.to_csv(out_dir / "recheck_coverage_round.csv", index=False, encoding="utf-8-sig")
    cov_win.to_csv(out_dir / "recheck_coverage_windows.csv", index=False, encoding="utf-8-sig")
    blind = build_blindspot(rounds)
    blind.to_csv(out_dir / "recheck_blindspot.csv", index=False, encoding="utf-8-sig")

    say("\n================ 3) 覆盖轨迹（跨种子均值，有边用户数）================")
    cov_mean = cov_long.groupby(["dataset_dir", "arm_dir", "global_round"], as_index=False).agg(
        n_edges=("graph_n_edges", "mean"))
    picks = [30, 60, 120, 130, 150, 250, 270, 390, 510, 600]
    for ds, part in cov_mean.groupby("dataset_dir"):
        say(f"  {ds}:")
        for arm in ("static", "static_at120", "periodic", "dynamic"):
            q = part[part["arm_dir"] == arm].set_index("global_round")["n_edges"]
            vals = " ".join(f"r{r}={q.get(r, float('nan')):.0f}" for r in picks if r in q.index)
            say(f"    {arm:<14} {vals}")

    say("\n  -- periodic 的窗口开头盲区 / 刷新后追平 --")
    for _, r in blind.iterrows():
        say(f"  {r['dataset_dir']:<22} W{int(r['window'])} r{int(r['round'])} {r['stage']:<12} "
            f"dynamic={r['dynamic_edges']:.0f} periodic={r['periodic_edges']:.0f} "
            f"ratio={r['periodic_over_dynamic']:.2f} (static={r['static_edges']:.0f})")

    # ---------------- 4) 与已提交聚合表交叉核对
    cons = build_consistency(win)
    if not cons.empty:
        cons.to_csv(out_dir / "recheck_consistency.csv", index=False, encoding="utf-8-sig")
        say("\n================ 4) 与 dg_arm_window.csv 交叉核对 ================")
        say(f"  最大 NDCG 偏差 {cons['max_abs_diff_ndcg'].max():.3e}；"
            f"最大 n_edges 偏差 {cons['max_abs_diff_n_edges'].max():.0f}；"
            f"匹配 {int(cons['n_matched'].sum())} 个 (臂,种子,窗口)")

    (out_dir / "recheck_console.txt").write_text("\n".join(lines), encoding="utf-8")
    say(f"\n产物已写入 {out_dir}")


if __name__ == "__main__":
    main()
