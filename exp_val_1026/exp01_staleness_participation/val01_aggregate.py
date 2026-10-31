#!/usr/bin/env python
"""验证实验 1 结果聚合：把 runs 汇总成「一张图 + 两个小表」。

输入：`results/raw/` 下递归查找所有 `val01_summary.csv`。
输出（`results/aggregated/`，共 3 个文件）：
    figures/val01_fig_staleness_rate.png   陈旧度 × 参与率 → 平均 NDCG@10（任务书的交付物）
    tables/val01_figure_data.csv           图背后的数据（mode/rate/bucket/n_users/mean_ndcg）
    tables/val01_verdict.csv               自动结论判定（参与率效应 / 陈旧损失 / 占比 / 总结论）

用法：
    python exp_val_1026/exp01_staleness_participation/val01_aggregate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL01_DIR = ROOT / "exp_val_1026" / "exp01_staleness_participation"
DEFAULT_ROOT = VAL01_DIR / "results" / "raw"
DEFAULT_OUT = VAL01_DIR / "results" / "aggregated"

MODE_LABELS = {
    "upload_only": "FedDIEG (use stale)",
    "upload_fresh_edges": "fresh summaries (oracle)",
}
BUCKET_POS = {"0": 0, "1": 1, "2": 2, "3-4": 3, "5-9": 4, "10-19": 5, "20-39": 6, "40+": 7}
METRIC = "avg_window_ndcg"
NOISE_FLOOR = 0.002


# --------------------------------------------------------------------------------------
# 读数据
# --------------------------------------------------------------------------------------
def load_runs(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(root.rglob("val01_summary.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:  # pragma: no cover
            print(f"[skip] {path}: {exc}")
    if not frames:
        raise SystemExit(f"在 {root} 下没有找到 val01_summary.csv，请先跑实验。")
    data = pd.concat(frames, ignore_index=True)
    data["participation_rate"] = data["participation_rate"].astype(float).round(3)
    return data


def load_buckets(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(root.rglob("val01_staleness_buckets.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:  # pragma: no cover
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# 图的数据：按用户数加权聚合（跨 seed、跨窗口）
# --------------------------------------------------------------------------------------
def build_curve(buckets: pd.DataFrame) -> pd.DataFrame:
    if buckets.empty:
        return buckets
    stale = buckets[buckets["bucket_type"] == "staleness"].copy()
    stale = stale[stale["bucket"].isin(BUCKET_POS)]
    if stale.empty:
        return stale
    stale["bucket_pos"] = stale["bucket"].map(BUCKET_POS)

    def weighted(group: pd.DataFrame) -> pd.Series:
        weight = np.maximum(group["n_users"].to_numpy(dtype=float), 1.0)
        return pd.Series(
            {
                "mean_ndcg": float(np.average(group["mean_ndcg"], weights=weight)),
                "mean_hr": float(np.average(group["mean_hr"], weights=weight)),
                "n_users": float(weight.sum()),
            }
        )

    curve = (
        stale.groupby(["dataset", "mode", "participation_rate", "bucket", "bucket_pos"], dropna=False)
        .apply(weighted)
        .reset_index()
    )
    return curve.sort_values(["dataset", "mode", "participation_rate", "bucket_pos"])


# --------------------------------------------------------------------------------------
# 结论判定
# --------------------------------------------------------------------------------------
def per_seed_delta(
    data: pd.DataFrame, mode: str, rate: float, base_mode: str, base_rate: Optional[float] = None
) -> pd.DataFrame:
    """同一 (dataset, seed) 下，mode@rate 减去 base_mode@base_rate。"""
    base_rate = rate if base_rate is None else base_rate
    left = data[(data["mode"] == mode) & np.isclose(data["participation_rate"], rate)]
    right = data[(data["mode"] == base_mode) & np.isclose(data["participation_rate"], base_rate)]
    if left.empty or right.empty:
        return pd.DataFrame()
    merged = left.merge(right, on=["dataset", "seed"], suffixes=("", "_base"))
    merged["delta"] = pd.to_numeric(merged[METRIC], errors="coerce") - pd.to_numeric(
        merged[f"{METRIC}_base"], errors="coerce"
    )
    return merged[["dataset", "seed", "delta"]].dropna()


def summarize(deltas: pd.DataFrame) -> Dict[str, float]:
    if deltas.empty:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "same_sign": 0.0}
    values = deltas["delta"].to_numpy(dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "n": int(len(values)),
        "same_sign": float(np.all(values > 0) or np.all(values < 0)),
    }


def is_effect(summary: Dict[str, float]) -> bool:
    """判定为真实效应：至少 2 个种子、|均值| 超过噪声地板、且各种子方向一致。"""
    if summary.get("n", 0) < 2:
        return False
    std = summary["std"] if np.isfinite(summary["std"]) else 0.0
    return abs(summary["mean"]) > max(NOISE_FLOOR, std) and bool(summary["same_sign"])


def build_verdict(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset, group in data.groupby("dataset"):
        base = group[group["mode"] == "upload_only"]
        if base.empty:
            continue
        rates = sorted(base["participation_rate"].unique())
        if len(rates) < 2:
            continue
        high_rate, low_rate = rates[-1], rates[0]

        # 1) 参与率效应：最低参与率 vs 最高参与率（同一策略）
        participation = summarize(
            per_seed_delta(group, "upload_only", low_rate, "upload_only", high_rate)
        )
        rows.append(
            {
                "dataset": dataset,
                "evidence": "参与率效应 ΔNDCG@10",
                "detail": f"upload_only p{low_rate:.2f} - p{high_rate:.2f}",
                "mean": participation["mean"],
                "std": participation["std"],
                "n_seeds": participation["n"],
                "significant": float(is_effect(participation)),
            }
        )

        # 2) 各低参与率下的陈旧损失：现状 - 新鲜摘要
        stale_effects: List[Dict[str, float]] = []
        for rate in rates:
            if np.isclose(rate, high_rate):
                continue
            stale = summarize(per_seed_delta(group, "upload_only", rate, "upload_fresh_edges"))
            if not stale["n"]:
                continue
            stale_effects.append(stale)
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": "陈旧信息造成的损失 NDCG(upload_only) - NDCG(fresh)",
                    "detail": f"p={rate:.2f}（负值=陈旧确实拖累）",
                    "mean": stale["mean"],
                    "std": stale["std"],
                    "n_seeds": stale["n"],
                    "significant": float(is_effect(stale)),
                }
            )

        stale_mean = (
            float(np.mean([item["mean"] for item in stale_effects])) if stale_effects else float("nan")
        )
        stale_real = any(is_effect(item) for item in stale_effects)
        participation_real = is_effect(participation)
        has_fresh_ctrl = bool(stale_effects)

        if participation["n"] < 2:
            conclusion = "无法判断：种子数不足（当前 <2），请先补齐 3 个随机种子"
        elif not participation_real:
            conclusion = "不成立：参与率下降对 NDCG 没有稳定影响（任务书停止信号 1）"
        elif not has_fresh_ctrl:
            conclusion = "现象成立（参与率越低 NDCG 越低），但缺少 fresh 对照，无法判断损失是否来自陈旧摘要"
        elif stale_real and np.isfinite(stale_mean) and stale_mean < 0:
            conclusion = "成立：陈旧信息确实造成损失（换成新鲜摘要能挽回），值得继续做动态图更新"
        else:
            conclusion = "部分成立：参与率确实有影响，但损失主要来自训练机会，陈旧信息不是主因"

        ratio = None
        if np.isfinite(stale_mean) and abs(participation["mean"]) > 1e-9:
            ratio = abs(stale_mean) / abs(participation["mean"])
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": "陈旧损失占参与率损失的比例",
                    "detail": "越大说明陈旧越是主因",
                    "mean": float(ratio),
                    "std": float("nan"),
                    "n_seeds": participation["n"],
                    "significant": float(np.isfinite(ratio)),
                }
            )
        if ratio is not None:
            conclusion = f"{conclusion}（陈旧约解释 {ratio:.0%} 的参与率损失）"

        rows.append(
            {
                "dataset": dataset,
                "evidence": "★ 总结论",
                "detail": conclusion,
                "mean": stale_mean,
                "std": float("nan"),
                "n_seeds": participation["n"],
                "significant": float(stale_real),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 那张图
# --------------------------------------------------------------------------------------
def make_figure(curve: pd.DataFrame, out_path: Path) -> None:
    """陈旧度（横轴）× 参与率（每条线）× 平均 NDCG@10（纵轴）。

    实线 = 现状服务器（用陈旧摘要）；虚线 = 对照（信息永远新鲜）。
    """
    if curve.empty:
        print("[figure] 没有分桶数据，跳过作图")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return

    bucket_order = [b for b, _ in sorted(BUCKET_POS.items(), key=lambda kv: kv[1])]
    datasets = list(dict.fromkeys(curve["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 4.6), squeeze=False)
    cmap = plt.get_cmap("viridis")

    for panel, dataset in enumerate(datasets):
        ax = axes[0][panel]
        sub = curve[curve["dataset"] == dataset]
        rates = sorted(sub["participation_rate"].unique())
        rate_color = {rate: cmap(i / max(len(rates) - 1, 1)) for i, rate in enumerate(rates)}
        for mode in [m for m in MODE_LABELS if m in set(sub["mode"])]:
            style = "-" if mode == "upload_only" else "--"
            marker = "o" if mode == "upload_only" else "s"
            for rate in rates:
                part = sub[(sub["mode"] == mode) & np.isclose(sub["participation_rate"], rate)]
                if part.empty:
                    continue
                part = part.sort_values("bucket_pos")
                ax.plot(
                    part["bucket_pos"],
                    part["mean_ndcg"],
                    linestyle=style,
                    marker=marker,
                    markersize=4.5,
                    linewidth=2.0 if mode == "upload_only" else 1.4,
                    color=rate_color[rate],
                    label=f"{MODE_LABELS[mode]} | p={rate:.2f}",
                )
        ax.set_xticks(range(len(bucket_order)))
        ax.set_xticklabels(bucket_order, fontsize=8)
        ax.set_title(f"{dataset}: staleness x participation -> NDCG@10")
        ax.set_xlabel("Staleness of own interest summary (rounds)")
        ax.set_ylabel("Mean NDCG@10 of users in bucket")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="val01 结果聚合：出一张图 + 自动结论")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="原始结果根目录（默认 results/raw）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录（默认 results/aggregated）")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out.resolve()
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_runs(root)
    print(f"Loaded {len(data)} runs from {root}")
    print(data.groupby(["dataset", "mode", "participation_rate"]).size().to_string())

    curve = build_curve(load_buckets(root))
    verdict = build_verdict(data)

    columns = [
        c
        for c in ("dataset", "mode", "participation_rate", "bucket", "bucket_pos", "n_users", "mean_ndcg", "mean_hr")
        if c in curve.columns
    ]
    curve[columns].to_csv(tables_dir / "val01_figure_data.csv", index=False, encoding="utf-8-sig")
    if not verdict.empty:
        verdict.to_csv(tables_dir / "val01_verdict.csv", index=False, encoding="utf-8-sig")
    make_figure(curve, figures_dir / "val01_fig_staleness_rate.png")

    if not verdict.empty:
        print("\n================ 结论判定 ================")
        for _, row in verdict.iterrows():
            if row["evidence"].startswith("★"):
                print(f"  {row['dataset']}: {row['detail']}")
                continue
            print(
                f"  [{row['dataset']}] {row['evidence']} ({row['detail']}): "
                f"{row['mean']:+.4f} ± {row['std']:.4f} (n={int(row['n_seeds'])}) -> "
                f"{'显著' if row['significant'] == 1.0 else '不显著'}"
            )
        print("==========================================")

    print(f"\n输出：{out_dir}")
    print("  figures/val01_fig_staleness_rate.png    ← 任务书要求的那张图")
    print("  tables/val01_figure_data.csv            ← 图背后的数据")
    print("  tables/val01_verdict.csv                ← 自动结论判定")


if __name__ == "__main__":
    main()
