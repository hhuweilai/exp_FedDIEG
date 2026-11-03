#!/usr/bin/env python
"""验证实验 2 的**关系质量**分析：邻居的兴趣一致度相对"随机邻居"的提升度（lift）。

为什么需要它
------------
`old_peer_share`（邻居中"旧兴趣同类"的边权占比）只是个绝对占比，说明不了好坏：
一个用户的邻居里同类占 0.10，可能是"图很差"，也可能是因为他那个类别本来只占 10% 的用户。
所以必须除以**随机基线**：若邻居是从用户池里随机挑的，同类占比的期望 = 该类别在池中的频率。

    lift = 观测同类边权占比 / 随机期望
    lift ≈ 1  → 邻居与随机无异（图没有表达兴趣）
    lift > 1  → 图确实把同兴趣的人放在一起（越大越好）

一个必须注意的口径问题（决定了结论能不能写）
------------------------------------------
对 onset 较晚的用户，"旧兴趣"（窗口 1..onset-1 的主类别）这个快照离**图的建立时点**（窗口 1）很远。
此时 lift 会跌到 1 以下，但那不是因为"图比随机还差"，而是因为**比较的对象错位了**
（图还停在更早的兴趣状态）。所以本脚本同时报告：

  * `all`   ：所有边界（会被上述错位污染，只用来看趋势，不要单独引用）；
  * `clean` ：onset == `--clean-boundary`（默认 2）的子集，此时"图建立时点 == 兴趣快照时点"，
              是唯一可以干净解读的对比。

输出（默认 `results/aggregated/`）
-------------------------------
  tables/val02_relation_quality.csv          逐用户逐种子的明细（obs / base / lift）
  tables/val02_relation_quality_summary.csv  按 数据集 × ρ × cohort × phase × 子集 的汇总（含跨种子 sd）
  tables/val02_relation_quality_by_boundary.csv  按 onset 边界拆开（展示"跟踪 vs 冻结"）
  figures/val02_fig_relation_quality.png     三栏：变化前 lift / 变化后 lift / lift 随 onset 边界

用法
----
  python exp_val_1026/exp02_ema_interest_drift/val02_relation_quality.py
  python .../val02_relation_quality.py --rel-post 60 --clean-boundary 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL02_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = VAL02_DIR / "results" / "raw"
DEFAULT_DRIFT_ROOT = VAL02_DIR / "results" / "drift"
DEFAULT_OUT = VAL02_DIR / "results" / "aggregated"

RHO_COLORS = {0.0: "#1b9e77", 0.50: "#66a61e", 0.80: "#d95f02", 0.95: "#7570b3"}
# 随机基线太小时，比值不稳（例如某类别只占 0.2% 的用户，lift 会爆炸）。低于该阈值的用户不计入。
MIN_BASE = 0.01


# ======================================================================================
# 读数据
# ======================================================================================
def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def dominant(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=1)
    return np.where(total > 0, counts.argmax(axis=1), -1)


def frequency(values: np.ndarray, num_cat: int) -> np.ndarray:
    valid = values >= 0
    return np.bincount(values[valid], minlength=num_cat) / max(int(valid.sum()), 1)


def discover_datasets(drift_root: Path, only: Sequence[str]) -> List[Tuple[str, str, Path]]:
    """返回 [(dataset 显示名, 目录 slug, drift 目录)]。"""
    found: List[Tuple[str, str, Path]] = []
    for meta_path in sorted(drift_root.rglob("val02_cohort_meta.json")):
        drift_dir = meta_path.parent
        slug = drift_dir.name
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        name = str(meta.get("dataset", slug))
        if only and not any(name == item or slug == item for item in only):
            continue
        found.append((name, slug, drift_dir))
    return found


def load_counts(drift_dir: Path) -> Tuple[np.ndarray, List[str]]:
    data = np.load(drift_dir / "val02_counts.npz", allow_pickle=False)
    counts = data["counts"]
    names = json.loads(str(data["category_names_json"]))
    return counts, names


# ======================================================================================
# 逐用户明细
# ======================================================================================
def collect_rows(
    root: Path, name: str, slug: str, drift_dir: Path, rel_post: int
) -> pd.DataFrame:
    counts, _ = load_counts(drift_dir)
    num_users, num_windows, num_cat = counts.shape
    cohort = pd.read_csv(drift_dir / "val02_drift_users.csv")

    # 每个边界一套随机基线（pre / post 两侧的类别频率）
    baselines: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for boundary in sorted(int(b) for b in cohort["onset_window"].unique()):
        boundary = int(np.clip(boundary, 2, num_windows))
        pre = dominant(counts[:, : boundary - 1, :].sum(axis=1))
        post = dominant(counts[:, boundary - 1 :, :].sum(axis=1))
        baselines[boundary] = (frequency(pre, num_cat), frequency(post, num_cat))

    cat_old = dict(zip(cohort["user"].astype(int), cohort["cat_old"].astype(int)))
    cat_new = dict(zip(cohort["user"].astype(int), cohort["cat_new"].astype(int)))
    onset = dict(zip(cohort["user"].astype(int), cohort["onset_window"].astype(int)))
    tag = dict(zip(cohort["user"].astype(int), cohort["cohort"].astype(str)))

    rows: List[Dict[str, object]] = []
    pattern = os.path.join(root, slug, "rho*", "seed_*", "val02_user_round_metrics.csv")
    for path in sorted(glob.glob(pattern)):
        rho = float(path.split(os.sep)[-3].replace("rho", ""))
        seed = int(path.split(os.sep)[-2].replace("seed_", ""))
        frame = pd.read_csv(path)

        for phase, rel_round, column, cats in (
            ("pre", 0, "old_peer_share", cat_old),
            ("post", int(rel_post), "new_peer_share", cat_new),
        ):
            sel = frame[frame["rel_round"] == rel_round]
            for _, record in sel.iterrows():
                user = int(record["user"])
                category = cats.get(user, -1)
                value = record[column]
                if category is None or category < 0 or not np.isfinite(value):
                    continue
                boundary = int(onset.get(user, 0))
                if boundary not in baselines:
                    continue
                pre_freq, post_freq = baselines[boundary]
                base = float((post_freq if phase == "post" else pre_freq)[int(category)])
                if base < MIN_BASE:
                    continue
                rows.append(
                    {
                        "dataset": name,
                        "slug": slug,
                        "rho": rho,
                        "seed": seed,
                        "phase": phase,
                        "rel_round": int(rel_round),
                        "cohort": str(tag.get(user, "")),
                        "boundary": boundary,
                        "user": user,
                        "observed": float(value),
                        "baseline": base,
                        "lift": float(value) / base,
                    }
                )
    return pd.DataFrame(rows)


# ======================================================================================
# 汇总
# ======================================================================================
def _per_seed(detail: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    """按 (数据集, ρ, 种子, phase, cohort[, boundary]) 先求用户平均，得到种子级数值。"""
    group_cols = ["dataset", "slug", "rho", "seed", "phase", "cohort"]
    if subset_name == "by_boundary":
        group_cols = group_cols + ["boundary"]
    out = (
        detail.groupby(group_cols)
        .agg(observed=("observed", "mean"), baseline=("baseline", "mean"),
             lift=("lift", "mean"), n_users=("user", "nunique"))
        .reset_index()
    )
    out["subset"] = subset_name
    return out


def summarise(
    detail: pd.DataFrame, clean_boundary: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回 (汇总表, 按边界表, 种子级表)。"""
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_seed = _per_seed(detail, "all")
    clean_seed = _per_seed(detail[detail["boundary"] == int(clean_boundary)], "clean")
    boundary_seed = _per_seed(detail, "by_boundary")
    seed_level = pd.concat([all_seed, clean_seed], ignore_index=True)

    summary = (
        seed_level.groupby(["dataset", "rho", "phase", "cohort", "subset"])
        .agg(observed=("observed", "mean"), baseline=("baseline", "mean"),
             lift_mean=("lift", "mean"), lift_sd=("lift", "std"),
             n_seeds=("seed", "nunique"), n_users=("n_users", "mean"))
        .reset_index()
    )
    by_boundary = (
        boundary_seed.groupby(["dataset", "rho", "phase", "cohort", "boundary"])
        .agg(lift_mean=("lift", "mean"), lift_sd=("lift", "std"), n_seeds=("seed", "nunique"))
        .reset_index()
    )
    return summary, by_boundary, seed_level


# ======================================================================================
# 图
# ======================================================================================
def make_figure(
    summary: pd.DataFrame,
    by_boundary: pd.DataFrame,
    out_path: Path,
    clean_boundary: int,
    rel_post: int,
) -> None:
    if summary.empty:
        print("[figure] 没有数据，跳过")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return

    datasets = list(dict.fromkeys(summary["dataset"]))
    fig, axes = plt.subplots(len(datasets), 3, figsize=(16.5, 4.3 * len(datasets)), squeeze=False)
    cohorts = [("drift", "interest-changed users"), ("stable", "stable control users")]

    for row_idx, dataset in enumerate(datasets):
        for col_idx, (phase, title) in enumerate(
            (
                ("pre", "Before the change (anchor = own pre-window)"),
                ("post", f"After the change (+{rel_post} rounds, new interest)"),
            )
        ):
            ax = axes[row_idx][col_idx]
            sub = summary[(summary["dataset"] == dataset) & (summary["phase"] == phase)
                          & (summary["subset"] == "clean")]
            if sub.empty:
                ax.set_title(f"{dataset}: {title}\n(no clean subset)", fontsize=9)
                ax.set_visible(False)
                continue
            rhos = sorted(sub["rho"].unique())
            width = 0.36
            for k, (cohort, label) in enumerate(cohorts):
                part = sub[sub["cohort"] == cohort].set_index("rho").reindex(rhos)
                xs = np.arange(len(rhos)) + (k - 0.5) * width
                ax.bar(
                    xs,
                    part["lift_mean"].to_numpy(dtype=float),
                    width=width,
                    yerr=part["lift_sd"].to_numpy(dtype=float),
                    capsize=3,
                    color=[RHO_COLORS.get(round(float(r), 2), "#666666") for r in rhos],
                    alpha=0.9 if cohort == "drift" else 0.42,
                    edgecolor="black",
                    linewidth=0.6,
                    label=label,
                )
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1.4)
            ax.set_xticks(np.arange(len(rhos)))
            ax.set_xticklabels([f"ρ={r:.2f}" for r in rhos], fontsize=8)
            ax.set_ylabel("Interest agreement / random")
            ax.set_title(f"{dataset}: {title}\n(clean subset, onset={clean_boundary})", fontsize=9)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax.legend(fontsize=7)

        # 第三栏：lift 随 onset 边界（ρ=0 应跨边界都高；高 ρ 会崩）
        ax = axes[row_idx][2]
        sub = by_boundary[(by_boundary["dataset"] == dataset) & (by_boundary["phase"] == "pre")
                          & (by_boundary["cohort"] == "drift")]
        if sub.empty:
            ax.set_visible(False)
            continue
        for rho in sorted(sub["rho"].unique()):
            part = sub[sub["rho"] == rho].sort_values("boundary")
            color = RHO_COLORS.get(round(float(rho), 2), "#666666")
            ax.plot(
                part["boundary"],
                part["lift_mean"],
                marker="o",
                markersize=4,
                linewidth=2.0,
                color=color,
                label=f"ρ={rho:.2f}",
            )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.4)
        ax.set_xlabel("onset boundary (window index)")
        ax.set_ylabel("Interest agreement / random")
        ax.set_title(
            f"{dataset}: does the graph track interest?\n(before the change, split by onset)", fontsize=9
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


# ======================================================================================
# 当前窗口兴趣一致度（新指标：用户与同伴用同一个窗口快照，无 onset 错位）
# ======================================================================================
def collect_current_rows(root: Path, name: str, slug: str) -> pd.DataFrame:
    """读取 runner 直接写入的 cur_peer_share / cur_peer_base（老结果没有这两列就返回空）。"""
    rows: List[Dict[str, object]] = []
    pattern = os.path.join(root, slug, "rho*", "seed_*", "val02_user_round_metrics.csv")
    for path in sorted(glob.glob(pattern)):
        rho = float(path.split(os.sep)[-3].replace("rho", ""))
        seed = int(path.split(os.sep)[-2].replace("seed_", ""))
        frame = pd.read_csv(path)
        if "cur_peer_share" not in frame.columns:
            continue
        sub = frame.dropna(subset=["cur_peer_share", "cur_peer_base"])
        if sub.empty:
            continue
        for (window, cohort), group in sub.groupby(["window", "cohort"]):
            share = float(group["cur_peer_share"].mean())
            base = float(group["cur_peer_base"].mean())
            if base < MIN_BASE:
                continue
            rows.append(
                {
                    "dataset": name,
                    "rho": rho,
                    "seed": seed,
                    "cohort": cohort,
                    "window": int(window),
                    "observed": share,
                    "baseline": base,
                    "lift": share / base if base > 1e-9 else float("nan"),
                    "n_users": int(group["user"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def summarise_current(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    per_seed = (
        detail.groupby(["dataset", "rho", "seed", "cohort", "window"])
        .agg(observed=("observed", "mean"), baseline=("baseline", "mean"),
             lift=("lift", "mean"), n_users=("n_users", "mean"))
        .reset_index()
    )
    return (
        per_seed.groupby(["dataset", "rho", "cohort", "window"])
        .agg(observed=("observed", "mean"), baseline=("baseline", "mean"),
             lift_mean=("lift", "mean"), lift_sd=("lift", "std"),
             n_seeds=("seed", "nunique"), n_users=("n_users", "mean"))
        .reset_index()
    )


def make_current_figure(current: pd.DataFrame, out_path: Path) -> None:
    """每个窗口的"当前兴趣一致度 / 随机基线"：ρ=0 应保持高位，高 ρ 应逐窗口下滑。"""
    if current.empty:
        print("[figure] 没有『当前窗口兴趣一致度』数据（老结果不含这两列），跳过")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return

    datasets = list(dict.fromkeys(current["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.6 * len(datasets), 4.4), squeeze=False)
    for panel, dataset in enumerate(datasets):
        ax = axes[0][panel]
        sub = current[(current["dataset"] == dataset) & (current["cohort"] == "drift")]
        for rho in sorted(sub["rho"].unique()):
            part = sub[sub["rho"] == rho].sort_values("window")
            color = RHO_COLORS.get(round(float(rho), 2), "#666666")
            ax.errorbar(
                part["window"],
                part["lift_mean"],
                yerr=part["lift_sd"].fillna(0.0),
                marker="o",
                markersize=4,
                linewidth=2.0,
                capsize=3,
                color=color,
                label=f"ρ={rho:.2f}",
            )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.4)
        ax.set_xticks(sorted(sub["window"].unique()))
        ax.set_xlabel("window index (1 = earliest)")
        ax.set_ylabel("Current-interest agreement / random")
        ax.set_title(
            f"{dataset}: does the graph track CURRENT interest?\n(drift cohort, per window)", fontsize=9
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


# ======================================================================================
# 主流程
# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="val02 关系质量分析：邻居兴趣一致度 vs 随机基线")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--drift-root", type=Path, default=DEFAULT_DRIFT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", default="", help="逗号分隔；留空 = 全部")
    parser.add_argument("--rel-post", type=int, default=60, help="变化后取第几轮做对齐（默认 60）")
    parser.add_argument("--clean-boundary", type=int, default=2,
                        help="干净子集用的 onset 边界（默认 2：图建立时点 == 兴趣快照时点）")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out.resolve()
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    only = [d.strip() for d in args.datasets.split(",") if d.strip()]
    datasets = discover_datasets(args.drift_root.resolve(), only)
    if not datasets:
        raise SystemExit(f"在 {args.drift_root} 下没有找到 val02_cohort_meta.json")

    frames: List[pd.DataFrame] = []
    for name, slug, drift_dir in datasets:
        detail = collect_rows(root, name, slug, drift_dir, args.rel_post)
        print(f"[data] {name}: {len(detail)} 条用户级记录")
        frames.append(detail)
    detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if detail.empty:
        raise SystemExit("没有可用数据：请确认 results/raw 下已有 val02_user_round_metrics.csv")

    write_csv(
        tables_dir / "val02_relation_quality.csv",
        detail.to_dict("records"),
        ["dataset", "slug", "rho", "seed", "phase", "rel_round", "cohort", "boundary",
         "user", "observed", "baseline", "lift"],
    )

    summary, by_boundary, seed_level = summarise(detail, args.clean_boundary)
    if not summary.empty:
        summary.round(4).to_csv(
            tables_dir / "val02_relation_quality_summary.csv", index=False, encoding="utf-8-sig"
        )
    if not by_boundary.empty:
        by_boundary.round(4).to_csv(
            tables_dir / "val02_relation_quality_by_boundary.csv", index=False, encoding="utf-8-sig"
        )
    if not seed_level.empty:
        seed_level.round(4).to_csv(
            tables_dir / "val02_relation_quality_by_seed.csv", index=False, encoding="utf-8-sig"
        )

    make_figure(
        summary, by_boundary, figures_dir / "val02_fig_relation_quality.png",
        args.clean_boundary, args.rel_post,
    )

    # ---- 当前窗口兴趣一致度（新指标；老结果没有这两列时自动跳过）----
    cur_frames: List[pd.DataFrame] = []
    for name, slug, _drift_dir in datasets:
        cur_frames.append(collect_current_rows(root, name, slug))
    cur_detail = pd.concat(cur_frames, ignore_index=True) if cur_frames else pd.DataFrame()
    if not cur_detail.empty:
        cur_detail.round(4).to_csv(
            tables_dir / "val02_relation_quality_current.csv", index=False, encoding="utf-8-sig"
        )
        cur_summary = summarise_current(cur_detail)
        cur_summary.round(4).to_csv(
            tables_dir / "val02_relation_current_summary.csv", index=False, encoding="utf-8-sig"
        )
        make_current_figure(cur_summary, figures_dir / "val02_fig_relation_current.png")
        print("\n---- 当前窗口兴趣一致度 / 随机基线（drift 组，按窗口）----")
        print(
            cur_summary[cur_summary["cohort"] == "drift"]
            .pivot_table(index="dataset", columns=["rho", "window"], values="lift_mean")
            .round(2)
            .to_string()
        )
    else:
        print("[data] 没有 cur_peer_share 列（这是老结果），跳过『当前窗口兴趣一致度』")

    # ---- 终端摘要 ----
    print("\n================ 关系质量：邻居兴趣一致度 / 随机基线 ================")
    print("（lift≈1 = 邻居与随机无异；>1 = 图确实把同兴趣的用户放在一起）")
    for dataset in sorted(summary["dataset"].unique()):
        for phase, phase_label in (("pre", "变化前"), ("post", f"变化后 +{args.rel_post} 轮（对新兴趣）")):
            for subset, subset_label in (("clean", "干净子集"), ("all", "全部边界(仅看趋势)")):
                sub = summary[(summary["dataset"] == dataset) & (summary["subset"] == subset)
                              & (summary["phase"] == phase)]
                if sub.empty:
                    continue
                print(f"\n[{dataset}] {phase_label} / {subset_label}")
                print(
                    sub.pivot_table(index="cohort", columns="rho", values="lift_mean")
                    .round(2)
                    .to_string()
                )
    print(f"\n输出：{out_dir}")


if __name__ == "__main__":
    main()
