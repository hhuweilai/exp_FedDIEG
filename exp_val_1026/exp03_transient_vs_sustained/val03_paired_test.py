#!/usr/bin/env python
"""验证实验 3（val03）第 4 步（可选）：逐对配对检验 —— 图的变化有没有变成推荐损失。

为什么要单独做这一步
-------------------
`val03_aggregate.py` 给的是"组均值 ± 用户级 SEM"。但注入 arm 是**一对一配对**设计
（每个 `inj` 用户都有一个活动量匹配的 `pair` 对照），配对设计下更该看的是
「逐对差值」：每个 pair 一个观测，再做符号检验 / Wilcoxon。

本脚本回答三个问题（只针对 MovieLens 这类有分辨力的数据集）：
  1. 短期扰动撤销之后，注入用户的 NDCG 是否**逐对**变差？（配对 Wilcoxon）
  2. 同期他们的"旧兴趣同伴占比"变了多少？
  3. **两者相关吗？** 如果不相关，说明推荐损失不是图更新造成的 ——
     这正是本实验关键的排除性结论。

输出：`results/aggregated/tables/val03_ndcg_paired_test.csv`

用法：
  python exp_val_1026/exp03_transient_vs_sustained/val03_paired_test.py
  python .../val03_paired_test.py --root results/raw --out results/aggregated/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:  # Wilcoxon 只在装了 scipy 时给（符号检验不依赖 scipy，永远有）
    from scipy import stats as _sp_stats
except Exception:  # pragma: no cover
    _sp_stats = None

VAL03_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = VAL03_DIR / "results" / "raw"
DEFAULT_OUT = VAL03_DIR / "results" / "aggregated" / "tables"

# rel_round 分块（0 = 变化前最后一轮）
BLOCKS = {
    "pre": ("pre", None),          # 特殊处理：[-R, 0]
    "during": ("during", None),    # [1, R]
    "revert": ("revert", None),    # [R+1, 2R]
    "last": ("last", None),        # [2R+1, 3R]
    "post": ("post", None),        # [R+1, 3R]（两个撤销后窗口合起来，样本最多、检验功效最好）
}


def _tests(diff: np.ndarray):
    """符号检验（永远可用）+ Wilcoxon（有 scipy 时）。"""
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n == 0:
        return float("nan"), float("nan"), int(0), int(0), n
    n_neg = int((diff < 0).sum())
    n_pos = int((diff > 0).sum())
    from math import comb

    k = min(n_neg, n_pos)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_sign = min(1.0, 2.0 * tail)
    p_wil = float("nan")
    if _sp_stats is not None and n >= 6 and (n_pos + n_neg) > 0:
        try:
            p_wil = float(_sp_stats.wilcoxon(diff).pvalue)
        except Exception:
            p_wil = float("nan")
    return p_sign, p_wil, n_neg, n_pos, n


def main() -> None:
    parser = argparse.ArgumentParser(description="val03 逐对配对检验：图变化 vs 推荐损失")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", default="", help="逗号分隔；留空 = 全部")
    parser.add_argument("--rounds-per-window", type=int, default=120)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    frames: List[pd.DataFrame] = []
    for path in sorted(root.rglob("val03_user_round_metrics.csv")):
        if datasets and not any(ds in str(path) for ds in datasets):
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"{root} 下没有找到 val03_user_round_metrics.csv")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["group"].isin(["inj", "pair"])]

    R = args.rounds_per_window
    bounds = {
        "pre": (-R, 0),
        "during": (1, R),
        "revert": (R + 1, 2 * R),
        "last": (2 * R + 1, 3 * R),
        "post": (R + 1, 3 * R),
    }

    rows = []
    for (dataset, scenario, rho), d in raw.groupby(["dataset", "scenario", "rho"]):
        if d["pair_id"].isna().all():
            continue
        for seed, g in d.groupby("seed"):
            per_pair = {}
            for name, (lo, hi) in bounds.items():
                q = g[(g["rel_round"] >= lo) & (g["rel_round"] <= hi)]
                if q.empty:
                    continue
                tab = q.groupby(["pair_id", "group"])[["ndcg", "old_peer_share"]].mean()
                tab = tab.unstack("group")
                per_pair[name] = tab
            if "pre" not in per_pair or "during" not in per_pair:
                continue
            for block in ("during", "revert", "last", "post"):
                if block not in per_pair:
                    continue
                cur, pre = per_pair[block], per_pair["pre"]
                common = cur.index.intersection(pre.index)
                if len(common) < 5:
                    continue
                d_ndcg = (cur.loc[common, ("ndcg", "inj")] - cur.loc[common, ("ndcg", "pair")]) - (
                    pre.loc[common, ("ndcg", "inj")] - pre.loc[common, ("ndcg", "pair")]
                )
                d_peer = (cur.loc[common, ("old_peer_share", "inj")] - cur.loc[common, ("old_peer_share", "pair")]) - (
                    pre.loc[common, ("old_peer_share", "inj")] - pre.loc[common, ("old_peer_share", "pair")]
                )
                ok = d_ndcg.notna() & d_peer.notna()
                if ok.sum() < 5:
                    continue
                p_sign, p_wil, n_neg, n_pos, n = _tests(d_ndcg[ok].to_numpy())
                corr = float(np.corrcoef(d_ndcg[ok], d_peer[ok])[0, 1]) if n > 2 else float("nan")
                rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "rho": float(rho),
                        "seed": seed,
                        "block": block,
                        "n_pairs": int(n),
                        # 下面两列都用「正 = 对注入组不利」
                        "d_ndcg_worse": float(-d_ndcg[ok].mean()),
                        "d_ndcg_median_worse": float(-d_ndcg[ok].median()),
                        "d_old_peer_drop": float(-d_peer[ok].mean()),
                        "pairs_ndcg_worse": int(n_neg),
                        "sign_test_p": float(p_sign),
                        "wilcoxon_p": float(p_wil),
                        "corr_dndcg_dold": corr,
                    }
                )
    if not rows:
        raise SystemExit("没有足够的配对数据，先跑训练。")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "val03_ndcg_paired_test.csv", index=False, encoding="utf-8-sig")

    print("逐对配对检验（正 = 对注入组不利；corr 接近 0 = 图变化与 NDCG 损失无关）")
    summary = (
        table.groupby(["dataset", "scenario", "rho", "block"])
        .agg(
            n_pairs=("n_pairs", "max"),
            d_ndcg_worse=("d_ndcg_worse", "mean"),
            d_ndcg_median=("d_ndcg_median_worse", "mean"),
            wilcoxon_p=("wilcoxon_p", "mean"),
            pairs_worse=("pairs_ndcg_worse", "mean"),
            d_old_peer_drop=("d_old_peer_drop", "mean"),
            corr=("corr_dndcg_dold", "mean"),
        )
        .round(4)
    )
    print(summary.to_string())
    print(f"\n已写出：{out_dir / 'val03_ndcg_paired_test.csv'}")


if __name__ == "__main__":
    main()
