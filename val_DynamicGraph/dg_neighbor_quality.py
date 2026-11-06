#!/usr/bin/env python
"""问题三的无标签版本：邻居质量越高，图的收益越大吗？（两个数据集都能做）

为什么需要它
------------
问题三原本用 MovieLens-1M 的真实 Genre 标签做（`dg_genre.py`）。但 Amazon 数据集
只有 4 个统计字段的 `meta.json`，**没有任何 item 类别/Genre 元数据**，做不了真实标签分析。

于是这里换一个不需要标签、且两个数据集都能跑的口径，直接检验「协同有效」的机制：
  对每个用户，图给他带来的**逐用户收益** ΔNDCG = NDCG(dynamic) − NDCG(nograph)，
  与「他邻居的质量」做相关与五分位比较。邻居质量用两个量度量（都在探针里已有）：
    `neighbor_lift_user` = 邻居兴趣摘要余弦 / 随机用户余弦   （图里的邻居有多像自己）
    `delta_rel`          = 图分支对用户表示的位移 / 序列表示   （图注入的力度）
若 ΔNDCG 随 `neighbor_lift_user` 单调上升，说明「邻居确实像自己」是协同有效的来源；
若两者无关，则「兴趣相似 ⇒ 协同有效」在两处都不成立。

用法
----
  python val_DynamicGraph/dg_neighbor_quality.py --dataset "movielens-1m"
  python val_DynamicGraph/dg_neighbor_quality.py --dataset "amazon-digital music"
  python val_DynamicGraph/dg_neighbor_quality.py --datasets "movielens-1m,amazon-digital music"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DG_DIR = Path(__file__).resolve().parent
ROOT = DG_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = DG_DIR / "results" / "raw"
DEFAULT_OUT = DG_DIR / "results" / "quality"
ARM_GRAPH = "dynamic"
ARM_BASE = "nograph"

try:
    from scipy import stats as _sp
except Exception:  # pragma: no cover
    _sp = None


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def fmt(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return str(x)
    return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 20261026):
    d = _finite(x)
    if d.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    chunk = int(max(1, min(n_boot, 2_000_000 // max(d.size, 1))))
    means, filled = [], 0
    while filled < n_boot:
        k = min(chunk, n_boot - filled)
        means.append(d[rng.integers(0, d.size, size=(k, d.size))].mean(axis=1))
        filled += k
    m = np.concatenate(means)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def corr_pair(x, y) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    out = {"n": int(ok.sum()), "pearson": float("nan"), "pearson_p": float("nan"),
           "spearman": float("nan"), "spearman_p": float("nan")}
    if ok.sum() < 8:
        return out
    if _sp is not None:
        r, p = _sp.pearsonr(x[ok], y[ok])
        out["pearson"], out["pearson_p"] = float(r), float(p)
        r, p = _sp.spearmanr(x[ok], y[ok])
        out["spearman"], out["spearman_p"] = float(r), float(p)
    return out


def _level(span: float, base: float, what: str) -> str:
    """按「五分位均值差 / 平均收益」的倍数分三档判定。"""
    if not np.isfinite(span) or not np.isfinite(base) or abs(base) < 1e-9:
        return "无法判定"
    ratio = span / abs(base)
    if ratio > 0.5 and span > 0:
        return f"是（明显）：{what}最低的 1/5 与最高的 1/5 之间，收益相差 {span:+.4f}，" \
               f"相当于平均收益的 {ratio:.0%}"
    if ratio > 0.2 and span > 0:
        return f"弱正相关：{what}与收益同向，但跨度只有平均收益的 {ratio:.0%}"
    return f"无关：{what}与个体收益没有可测关系（跨度 {span:+.4f}）"


def quintiles(df: pd.DataFrame, key: str, target: str = "dndcg", q: int = 5) -> pd.DataFrame:
    d = df[[key, target]].dropna()
    if len(d) < q * 20:
        return pd.DataFrame()
    try:
        d = d.assign(bucket=pd.qcut(d[key], q, labels=False, duplicates="drop"))
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    rows = []
    for b, g in d.groupby("bucket"):
        lo, hi = bootstrap_ci(g[target])
        rows.append({
            "bucket": int(b) + 1, "n": int(len(g)),
            f"{key}_lo": float(g[key].min()), f"{key}_hi": float(g[key].max()),
            f"{key}_mean": float(g[key].mean()),
            f"{target}_mean": float(g[target].mean()),
            "ci95_lo": lo, "ci95_hi": hi,
            f"{target}_pos_rate": float((g[target] > 0).mean()),
        })
    return pd.DataFrame(rows)


def load_joined(root: Path, dataset: str) -> pd.DataFrame:
    base = root / dataset
    probe = sorted((base / ARM_GRAPH).glob("seed_*/dg_probe_metrics.csv"))
    dyn = sorted((base / ARM_GRAPH).glob("seed_*/dg_user_metrics.csv"))
    nog = sorted((base / ARM_BASE).glob("seed_*/dg_user_metrics.csv"))
    if not probe:
        raise SystemExit(f"{base}/{ARM_GRAPH} 下没有 dg_probe_metrics.csv")
    if not dyn or not nog:
        raise SystemExit(f"{base} 下缺 {ARM_GRAPH} 或 {ARM_BASE} 的 dg_user_metrics.csv")

    p = pd.concat([pd.read_csv(f) for f in probe], ignore_index=True)
    p = p.rename(columns={"ndcg": "ndcg_probe"}) if "ndcg" in p.columns else p
    d = pd.concat([pd.read_csv(f) for f in dyn], ignore_index=True).rename(
        columns={"ndcg": "ndcg_dyn"})
    n = pd.concat([pd.read_csv(f) for f in nog], ignore_index=True).rename(
        columns={"ndcg": "ndcg_base"})

    keep = ["seed", "window", "user", "delta_rel", "delta_cos",
            "neighbor_cos", "neighbor_cos_base", "has_edge"]
    j = p[keep].merge(d[["seed", "window", "user", "ndcg_dyn"]],
                      on=["seed", "window", "user"], how="inner")
    j = j.merge(n[["seed", "window", "user", "ndcg_base"]],
                on=["seed", "window", "user"], how="inner")
    j["dndcg"] = j["ndcg_dyn"] - j["ndcg_base"]
    j["neighbor_lift_user"] = j["neighbor_cos"] / j["neighbor_cos_base"].where(
        j["neighbor_cos_base"] > 1e-9)
    return j


def analyze(root: Path, dataset: str, out_root: Path) -> Dict[str, object]:
    out = out_root / _slug(dataset)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    j = load_joined(root, dataset)
    if j.empty:
        raise SystemExit(f"{dataset}: 探针与逐用户结果无法对齐")
    j.to_csv(tables / "dg_quality_merged.csv", index=False, encoding="utf-8-sig")

    # ---- 描述统计
    desc_rows = []
    for m in ("neighbor_cos", "neighbor_cos_base", "neighbor_lift_user",
              "delta_rel", "delta_cos", "dndcg"):
        d = _finite(j[m])
        if d.size == 0:
            continue
        lo, hi = bootstrap_ci(d)
        desc_rows.append({"metric": m, "n": int(d.size), "mean": float(d.mean()),
                          "median": float(np.median(d)), "ci95_lo": lo, "ci95_hi": hi,
                          "pos_rate": float((d > 0).mean())})
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(tables / "dg_quality_descriptives.csv", index=False, encoding="utf-8-sig")

    # ---- 相关
    corr_rows = []
    for m in ("neighbor_lift_user", "neighbor_cos", "delta_rel", "delta_cos"):
        corr_rows.append({"metric": m, "target": "dndcg", **corr_pair(j[m], j["dndcg"])})
    corrs = pd.DataFrame(corr_rows)
    corrs.to_csv(tables / "dg_quality_correlations.csv", index=False, encoding="utf-8-sig")

    # ---- 五分位
    buckets = {}
    frames = []
    for key in ("neighbor_lift_user", "delta_rel"):
        b = quintiles(j, key)
        if b.empty:
            continue
        b.insert(0, "key", key)
        buckets[key] = b
        frames.append(b)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            tables / "dg_quality_quintiles.csv", index=False, encoding="utf-8-sig")

    # ---- 结论
    def q_span(key):
        b = buckets.get(key)
        if b is None or len(b) < 2:
            return float("nan")
        return float(b.iloc[-1]["dndcg_mean"]) - float(b.iloc[0]["dndcg_mean"])

    rows = []
    c_lift = corrs[corrs["metric"] == "neighbor_lift_user"].iloc[0]
    c_dr = corrs[corrs["metric"] == "delta_rel"].iloc[0]
    span_lift = q_span("neighbor_lift_user")
    span_dr = q_span("delta_rel")
    base = float(j["dndcg"].mean())
    rows.append({
        "dataset": dataset, "verdict_id": "①",
        "question": "图的收益有多大（dynamic − nograph，逐用户）",
        "value": round(base, 6),
        "evidence": f"平均 ΔNDCG={base:+.4f}（n={int(j['dndcg'].notna().sum())} 个 用户×窗口；"
                    f"ΔNDCG>0 比例 {float((j['dndcg'] > 0).mean()):.3f}）",
        "judgement": "收益为" + ("正" if base > 0 else "负") + "，见问题一的配对检验",
    })
    rows.append({
        "dataset": dataset, "verdict_id": "②",
        "question": "邻居越像自己（neighbor_lift 越高），图的收益是否越大？",
        "value": round(span_lift, 6),
        "evidence": f"逐用户 corr(ΔNDCG, neighbor_lift)：Spearman={fmt(c_lift['spearman'], 4)}"
                    f"（p={fmt(c_lift['spearman_p'], 3)}，n={int(c_lift['n'])}）；"
                    f"五分位 Q1→Q5 均值差={span_lift:+.4f}"
                    + (f"；分位：" + "，".join(
                        f"Q{int(r['bucket'])}={fmt(r['dndcg_mean'], 4)}"
                        for _, r in buckets['neighbor_lift_user'].iterrows())
                       if 'neighbor_lift_user' in buckets else ""),
        "judgement": _level(span_lift, base, "邻居质量"),
    })
    rows.append({
        "dataset": dataset, "verdict_id": "③",
        "question": "图注入力度（delta_rel）越大，图的收益是否越大？",
        "value": round(span_dr, 6),
        "evidence": f"逐用户 corr(ΔNDCG, delta_rel)：Spearman={fmt(c_dr['spearman'], 4)}"
                    f"（p={fmt(c_dr['spearman_p'], 3)}）；五分位 Q1→Q5 均值差={span_dr:+.4f}",
        "judgement": _level(span_dr, base, "图注入力度"),
    })
    verdict = pd.DataFrame(rows)
    verdict.to_csv(tables / "dg_quality_verdict.csv", index=False, encoding="utf-8-sig")

    # ---- 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = [k for k in ("neighbor_lift_user", "delta_rel") if k in buckets]
        if keys:
            fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.4), squeeze=False)
            for ax, key in zip(axes[0], keys):
                b = buckets[key]
                x = b["bucket"].to_numpy()
                y = b["dndcg_mean"].to_numpy()
                lo = y - b["ci95_lo"].to_numpy()
                hi = b["ci95_hi"].to_numpy() - y
                ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=5, lw=1.6, capsize=3, color="#d62728")
                ax.axhline(0.0, color="k", lw=0.8, ls="--")
                ax.set_xticks(x)
                ax.set_xlabel(f"{key} quintile (low → high)")
                ax.set_ylabel("mean ΔNDCG@10 (dynamic − nograph)")
                ax.set_title(f"{dataset}: graph gain vs {key}", fontsize=10)
                ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(figures / "dg_fig_quality_gain.png", dpi=160)
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 出图跳过：{exc}")

    return {"dataset": dataset, "n": int(len(j)), "desc": desc, "corrs": corrs,
            "verdict": verdict, "buckets": buckets, "out": out}


def main() -> None:
    parser = argparse.ArgumentParser(description="邻居质量 vs 图收益（无标签版问题三，跨数据集）")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default="", help="单个数据集")
    parser.add_argument("--datasets", default="", help="逗号分隔多个数据集")
    args = parser.parse_args()

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = []
        for p in sorted((args.root / "" ).glob("*")):
            if p.is_dir() and (p / ARM_GRAPH).is_dir():
                datasets.append(p.name)
    if not datasets:
        raise SystemExit("没有可分析的数据集")

    pd.set_option("display.width", 220)
    out_root = args.out.resolve()
    for ds in datasets:
        print("\n" + "=" * 78)
        print(f"数据集：{ds}")
        print("=" * 78)
        try:
            res = analyze(args.root.resolve(), ds, out_root)
        except SystemExit as exc:
            print(f"  跳过：{exc}")
            continue
        print(f"  对齐观测数 {res['n']}")
        print(res["desc"][["metric", "n", "mean", "median", "ci95_lo", "ci95_hi", "pos_rate"]]
              .round(4).to_string(index=False))
        print("\n  相关（target=ΔNDCG）：")
        print(res["corrs"][["metric", "n", "pearson", "pearson_p", "spearman", "spearman_p"]]
              .round(4).to_string(index=False))
        print()
        for _, r in res["verdict"].iterrows():
            print(f"  [{r['verdict_id']}] {r['question']}")
            print(f"       {r['evidence']}")
            print(f"       -> {r['judgement']}")
        print(f"\n  输出：{res['out']}")


if __name__ == "__main__":
    main()
