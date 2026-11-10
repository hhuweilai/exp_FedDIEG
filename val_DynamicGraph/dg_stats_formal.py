#!/usr/bin/env python
"""阶段 5：把正式复跑（5 种子）的聚合表整理成「可直接写进报告」的统计三件套。

三个文件（都在 `--out` 指定的聚合目录的 tables/ 下）
----------------------------------------------------
1. `dg_stats_seed.csv`     每个（数据集 × 对照 × 指标）：**种子级**主证据
                           （均值/标准差/t 分布 95%CI/配对 t 的 p/Cohen's dz/
                           正号种子数）+ 窗口级敏感性分析（同一差值的窗口级 p）。
2. `dg_stats_decomp.csv`   三项分解：dynamic−static =（onboarding−static）+（dynamic−onboarding），
                           即「覆盖（新上传者首次获得邻居）」+「演化（已覆盖用户关系重估）」，
                           并给出恒等残差（必须是 0，否则说明对照的口径不一致）。
3. `dg_stats_coverage.csv` 逐臂的覆盖/结构指标：`active_coverage`、`graph_turnover`、
                           ECR（v2 定义，仅在更新轮上取均值）、更新轮数、刷新轮数、末轮边数。
4. `dg_stats_legacy_vs_formal.csv`  修前（legacy，3 种子）vs 修后（formal，5 种子）的头号数字对照。
                           **注意**：两者差同时含「F7/F8 修复」与「种子集变化」，只作方向性检查，
                           不能把差值全部归因于修复。

为什么单独一个脚本
------------------
`dg_aggregate.py` 负责「算」，这里负责「排版与判定」——报告里每个数都能追到这些 CSV，
不手抄。判定阈值沿用 `--practical-threshold`（默认 0.002 NDCG@10）。

用法
----
  python val_DynamicGraph/dg_stats_formal.py
  python val_DynamicGraph/dg_stats_formal.py --agg results/agg_formal --legacy results/aggregated
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DG_DIR = Path(__file__).resolve().parent
RESULTS = DG_DIR / "results"
DEFAULT_AGG = RESULTS / "agg_formal"
DEFAULT_LEGACY = RESULTS / "aggregated"

PRIMARY = ["static_minus_nograph", "dynamic_minus_static", "dynamic_minus_nograph"]
EXTRA = [
    "periodic_minus_nograph",
    "dynamic_minus_periodic",
    "onboarding_minus_static",
    "dynamic_minus_onboarding",
    "fresh_only_minus_static",
    "dynamic_minus_fresh_only",
]
ARM_LABEL = {
    "nograph": "NoGraph",
    "static": "StaticGraph",
    "static_at120": "StaticGraph（更新到 120 轮再冻结）",
    "static_build120": "StaticGraph（第 120 轮一次性建图后冻结）",
    "periodic": "PeriodicGraph（每 120 轮重建）",
    "periodic_boundary": "PeriodicGraph（窗口边界重建）",
    "periodic_p20": "PeriodicGraph（每 20 轮重建）",
    "periodic_p240": "PeriodicGraph（每 240 轮重建）",
    "dynamic": "DynamicGraph",
    "onboarding": "OnboardingGraph（首次有摘要即建边）",
    "fresh_only": "FreshOnlyGraph（只给新上传者建边）",
}
CONTRAST_LABEL = {
    "static_minus_nograph": "① StaticGraph − NoGraph",
    "dynamic_minus_static": "② DynamicGraph − StaticGraph",
    "dynamic_minus_nograph": "③ DynamicGraph − NoGraph",
    "periodic_minus_nograph": "PeriodicGraph − NoGraph",
    "dynamic_minus_periodic": "DynamicGraph − PeriodicGraph",
    "onboarding_minus_static": "覆盖项：OnboardingGraph − StaticGraph",
    "dynamic_minus_onboarding": "演化项：DynamicGraph − OnboardingGraph",
    "fresh_only_minus_static": "针对性项：FreshOnlyGraph − StaticGraph",
    "dynamic_minus_fresh_only": "DynamicGraph − FreshOnlyGraph",
}


# ----------------------------------------------------------------- 工具
def read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if nd and abs(x) < 1 else f"{x:.{nd}f}"


def fp(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.4f}" if x >= 0.001 else f"{x:.2e}"


def _ci(lo, hi) -> str:
    """95% CI 排版：两端都缺就写「—」。"""
    if (lo is None or not math.isfinite(float(lo))) and (hi is None or not math.isfinite(float(hi))):
        return "—"
    return f"[{fmt(lo)}, {fmt(hi)}]"


def md_table(rows: List[List[str]], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def judge(seed_mean: float, p_seed: float, thr: float) -> str:
    """判定：量级够不够 + 方向一致性 + 显著性（种子级）。"""
    if seed_mean is None or not math.isfinite(seed_mean):
        return "无数据"
    if abs(seed_mean) < thr:
        return "量级可忽略"
    if p_seed is not None and math.isfinite(p_seed):
        if p_seed < 0.05:
            return "显著为正" if seed_mean > 0 else "显著为负"
        if p_seed < 0.10:
            return "边缘（趋势为正）" if seed_mean > 0 else "边缘（趋势为负）"
    return "方向为正但不显著" if seed_mean > 0 else "方向为负但不显著"


# ----------------------------------------------------------------- 1) 种子级主表
def build_seed_table(contrasts: pd.DataFrame, thr: float) -> pd.DataFrame:
    need = {"dataset", "metric", "contrast"}
    if contrasts is None or not need.issubset(contrasts.columns):
        return pd.DataFrame()
    rows = []
    for _, r in contrasts.iterrows():
        if r["metric"] != "ndcg_win_mean":
            continue
        seed_mean = r.get("delta_seed_mean", np.nan)
        seed_sd = r.get("delta_seed_sd", np.nan)
        dz = (
            float(seed_mean) / float(seed_sd)
            if seed_sd and math.isfinite(float(seed_sd)) and float(seed_sd) > 0
            else np.nan
        )
        p_seed = r.get("p_seed_t", np.nan)
        rows.append(
            {
                "dataset": r["dataset"],
                "contrast": r["contrast"],
                "metric": r["metric"],
                "n_seeds": int(r.get("n_seeds", 0)),
                "delta_seed_mean": seed_mean,
                "delta_seed_sd": seed_sd,
                "ci95_seed_lo": r.get("ci95_seed_lo", np.nan),
                "ci95_seed_hi": r.get("ci95_seed_hi", np.nan),
                "p_seed_t": p_seed,
                "cohen_dz": dz,
                "seeds_positive": int(r.get("seeds_positive", 0)),
                "seeds_total": int(r.get("seeds_total", 0)),
                # 窗口级只作敏感性分析（窗口不是独立观测）
                "n_windows_obs": int(r.get("n_obs", 0)),
                "delta_window_mean": r.get("delta_mean", np.nan),
                "p_window_paired": r.get("p_paired", np.nan),
                "p_window_sign": r.get("p_sign", np.nan),
                "win_rate_window": r.get("win_rate", np.nan),
                "practical": judge(float(seed_mean) if seed_mean == seed_mean else np.nan, p_seed, thr),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 2) 三项分解
def build_decomp(seed_tbl: pd.DataFrame) -> pd.DataFrame:
    if seed_tbl.empty:
        return pd.DataFrame()
    rows = []
    for ds, part in seed_tbl.groupby("dataset"):
        get = lambda name: part[part["contrast"] == name]  # noqa: E731
        j2, cov, ev = get("dynamic_minus_static"), get("onboarding_minus_static"), get("dynamic_minus_onboarding")
        if j2.empty:
            continue
        total = float(j2["delta_seed_mean"].iloc[0])
        c = float(cov["delta_seed_mean"].iloc[0]) if not cov.empty else np.nan
        e = float(ev["delta_seed_mean"].iloc[0]) if not ev.empty else np.nan
        rows.append(
            {
                "dataset": ds,
                "total_dynamic_minus_static": total,
                "cover_onboarding_minus_static": c,
                "evolve_dynamic_minus_onboarding": e,
                "identity_residual": total - (c + e) if math.isfinite(c) and math.isfinite(e) else np.nan,
                "cover_share": (c / total) if total not in (0, np.nan) and math.isfinite(c) and total != 0 else np.nan,
                "n_seeds": int(j2["n_seeds"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 3) 覆盖/结构
def build_coverage(arm_summary: pd.DataFrame) -> pd.DataFrame:
    if arm_summary is None or arm_summary.empty:
        return pd.DataFrame()
    keep = [
        "dataset", "arm_dir", "n_seeds", "ndcg_win_mean",
        "n_edges", "graph_updates", "update_rounds", "refresh_rounds",
        "active_coverage_mean", "graph_turnover_mean", "ecr_update_mean",
        "ecr", "neighbor_lift", "delta_rel",
    ]
    cols = [c for c in keep if c in arm_summary.columns]
    d = arm_summary[cols].copy()
    d["arm_label"] = d["arm_dir"].map(lambda a: ARM_LABEL.get(a, a))
    return d.sort_values(["dataset", "arm_dir"]).reset_index(drop=True)


# ----------------------------------------------------------------- 4) 修前 vs 修后
def build_legacy(seed_tbl: pd.DataFrame, legacy: Path) -> pd.DataFrame:
    old = read_csv(legacy / "tables" / "dg_contrasts.csv")
    if old is None or seed_tbl.empty:
        return pd.DataFrame()
    rows = []
    for _, r in seed_tbl.iterrows():
        if r["contrast"] not in PRIMARY:
            continue
        sel = old[
            (old["dataset"] == r["dataset"])
            & (old["contrast"] == r["contrast"])
            & (old["metric"] == "ndcg_win_mean")
        ]
        if sel.empty:
            continue
        o = sel.iloc[0]
        rows.append(
            {
                "dataset": r["dataset"],
                "contrast": r["contrast"],
                "legacy_delta": float(o["delta_mean"]),
                "legacy_seeds": int(o.get("n_seeds", 0)),
                "formal_delta_seed": float(r["delta_seed_mean"]),
                "formal_seeds": int(r["n_seeds"]),
                "formal_p_seed": r["p_seed_t"],
                "diff": float(r["delta_seed_mean"]) - float(o["delta_mean"]),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 排版
def render(seed_tbl, decomp, coverage, legacy, thr: float) -> str:
    L: List[str] = []
    L.append("# 阶段 5 正式统计表（由 `dg_stats_formal.py` 生成，未手抄）\n")
    L.append(f"- 判定阈值：|ΔNDCG@10| ≥ {thr}（`--practical-threshold`）")
    L.append("- **主证据是种子级**（每种子一个观测，n=种子数）；窗口级只作敏感性分析"
             "（同一种子跨窗口不独立）\n")

    if not seed_tbl.empty:
        L.append("## 1. 种子级主表（NDCG@10，窗口内均值）\n")
        for ds, part in seed_tbl.groupby("dataset"):
            rows = []
            for _, r in part.iterrows():
                if r["contrast"] not in PRIMARY + EXTRA:
                    continue
                rows.append([
                    CONTRAST_LABEL.get(r["contrast"], r["contrast"]),
                    fmt(r["delta_seed_mean"]),
                    _ci(r["ci95_seed_lo"], r["ci95_seed_hi"]),
                    fmt(r["delta_seed_sd"]),
                    fp(r["p_seed_t"]),
                    fp(r["cohen_dz"]),
                    f"{int(r['seeds_positive'])}/{int(r['seeds_total'])}",
                    fmt(r["delta_window_mean"]),
                    fp(r["p_window_paired"]),
                    r["practical"],
                ])
            L.append(f"**{ds}**\n")
            L.append(md_table(rows, [
                "对照", "Δ种子均值", "95%CI（t, df=n−1）", "Δ种子SD", "p（种子 t）",
                "Cohen dz", "正号种子", "窗口级 Δ", "p（窗口）", "判定",
            ]))
            L.append("")

    if not decomp.empty:
        L.append("## 2. 三项分解：收益来自「覆盖」还是「演化」\n")
        rows = []
        for _, r in decomp.iterrows():
            rows.append([
                r["dataset"],
                fmt(r["total_dynamic_minus_static"]),
                fmt(r["cover_onboarding_minus_static"]),
                fmt(r["evolve_dynamic_minus_onboarding"]),
                f"{r['cover_share']*100:.0f}%" if math.isfinite(r["cover_share"]) else "—",
                f"{r['identity_residual']:+.2e}",
                str(int(r["n_seeds"])),
            ])
        L.append(md_table(rows, [
            "数据集", "② Dynamic−Static（总）", "覆盖项（Onboarding−Static）",
            "演化项（Dynamic−Onboarding）", "覆盖占比", "恒等残差", "种子数",
        ]))
        L.append("\n> 恒等残差必须为 0（浮点误差内）。覆盖项只由「第 21 轮之后才第一次获得邻居的用户」"
                 "贡献，因此它衡量的是**覆盖（含时点）**，不是「有无边」的二元效应。\n")

    if not coverage.empty:
        L.append("## 3. 逐臂覆盖 / 结构指标\n")
        for ds, part in coverage.groupby("dataset"):
            rows = []
            for _, r in part.iterrows():
                rows.append([
                    r["arm_label"],
                    str(int(r["n_seeds"])),
                    fmt(r["ndcg_win_mean"]),
                    f"{r['active_coverage_mean']:.4f}" if pd.notna(r.get("active_coverage_mean")) else "—",
                    f"{r['graph_turnover_mean']:.4f}" if pd.notna(r.get("graph_turnover_mean")) else "—",
                    f"{r['ecr_update_mean']:.4f}" if pd.notna(r.get("ecr_update_mean")) else "—",
                    f"{r['refresh_rounds']:.0f}" if pd.notna(r.get("refresh_rounds")) else "—",
                    f"{r['n_edges']:.0f}" if pd.notna(r.get("n_edges")) else "—",
                ])
            L.append(f"**{ds}**\n")
            L.append(md_table(rows, [
                "臂", "种子", "NDCG@10", "active_coverage", "graph_turnover",
                "ECR（更新轮，v2）", "刷新轮数", "末轮边数",
            ]))
            L.append("")

    if not legacy.empty:
        L.append("## 4. 修前（legacy）vs 修后（formal）\n")
        rows = []
        for _, r in legacy.iterrows():
            rows.append([
                r["dataset"],
                CONTRAST_LABEL.get(r["contrast"], r["contrast"]),
                fmt(r["legacy_delta"]),
                str(int(r["legacy_seeds"])),
                fmt(r["formal_delta_seed"]),
                str(int(r["formal_seeds"])),
                fmt(r["diff"]),
                fp(r["formal_p_seed"]),
            ])
        L.append(md_table(rows, [
            "数据集", "对照", "修前 Δ（窗口级）", "种子数", "修后 Δ（种子级）", "种子数", "差值", "p（种子）",
        ]))
        L.append("\n> 这个差值同时含「F7 摘要上传时机 + F8 时间切分泄漏」的修复与「3 种子 → 5 种子」的"
                 "样本变化，**不能**把差值整体归因于修复；修复带来的单种子对照见"
                 "`问题核实与修复清单_2026-11-08.md`（raw_phase2b → raw_fix1）。\n")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 5：正式统计表（种子级主证据 + 分解 + 覆盖 + 修前/修后）")
    ap.add_argument("--agg", type=Path, default=DEFAULT_AGG, help="正式聚合目录（含 tables/）")
    ap.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY, help="legacy 聚合目录（对照用）")
    ap.add_argument("--practical-threshold", type=float, default=0.002)
    ap.add_argument("--md", type=Path, default=None, help="Markdown 输出路径（默认 <agg>/阶段5_统计表.md）")
    args = ap.parse_args()

    tables = args.agg / "tables"
    if not (tables / "dg_contrasts.csv").exists():
        raise SystemExit(f"{tables} 下没有 dg_contrasts.csv，先跑 dg_aggregate.py --out {args.agg}")

    seed_tbl = build_seed_table(read_csv(tables / "dg_contrasts.csv"), args.practical_threshold)
    decomp = build_decomp(seed_tbl)
    coverage = build_coverage(read_csv(tables / "dg_arm_summary.csv"))
    legacy = build_legacy(seed_tbl, args.legacy)

    for name, df in (
        ("dg_stats_seed", seed_tbl),
        ("dg_stats_decomp", decomp),
        ("dg_stats_coverage", coverage),
        ("dg_stats_legacy_vs_formal", legacy),
    ):
        if df is None or df.empty:
            print(f"[warn] {name} 为空（缺输入或列不全）")
            continue
        df.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"[ok] {tables / (name + '.csv')}（{len(df)} 行）")

    md = args.md or (args.agg / "阶段5_统计表.md")
    md.write_text(render(seed_tbl, decomp, coverage, legacy, args.practical_threshold), encoding="utf-8")
    print(f"[ok] {md}")

    if not decomp.empty:
        for _, r in decomp.iterrows():
            print(f"  分解 {r['dataset']}：总 {r['total_dynamic_minus_static']:+.4f} = "
                  f"覆盖 {r['cover_onboarding_minus_static']:+.4f} + 演化 {r['evolve_dynamic_minus_onboarding']:+.4f}"
                  f"（残差 {r['identity_residual']:+.2e}）")


if __name__ == "__main__":
    main()
