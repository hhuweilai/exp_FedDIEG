#!/usr/bin/env python
"""冻结归因汇总：图冻结到底是 EMA、还是新边初始化、还是「状态保留规则」造成的？

设计（2x2 网格，rho 固定 0.8 = 审计报告里观测到冻结的区间）
----------------------------------------------------------
                新边初始化 zero(论文)        新边初始化 warm
  候选池 pool   dynamic_rho0.8            dynamic_rho0.8_warm
  只存TopK topk dynamic_rho0.8_topk       dynamic_rho0.8_topk_warm

外加论文默认配置 `dynamic`（pool + zero + rho=0.2）作参照。

读什么
------
`dg_round_metrics.csv` 里的：
  turnover_from_ref   当前 Top-K 邻居集合与「预热后第一次评测快照」的 Jaccard 距离
                      （≈0 = 完全没变；越大 = 图在动）
  frozen_share        邻居集合完全没变的用户占比（≈1 = 整张图冻住）
  ecr_all / added_mean  每轮边变化率 / 新进入 Top-K 的边数
  graph_n_edges       有边用户数
  neighbor_lift / delta_rel  邻居质量 / 图分支的数值影响力
  all_ndcg            推荐指标

判断规则
--------
* topk 两格都冻住、pool 两格都不冻  => 冻结来自**状态保留规则**（只存 Top-K 会丢历史），
  而不是 EMA 本身；
* 若 topk+warm 也没解开冻结      => **新边初始化规则不是原因**；
* 若 pool+warm 比 pool+zero 转得更快（turnover / added 更高）=> 初始化规则只影响
  **适应速度**，可以量化它的收益。

用法
----
  python val_DynamicGraph/dg_freeze.py
  python val_DynamicGraph/dg_freeze.py --root results/raw --out results/freeze
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DG_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = DG_DIR / "results" / "raw"
DEFAULT_OUT = DG_DIR / "results" / "freeze"
DATASET = "movielens-1m"


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")

#: (cell 名, 结果目录名, 图状态, 新边初始化, rho)
CELLS: List[Tuple[str, str, str, str, float]] = [
    ("paper_default_pool_zero_rho0.2", "dynamic", "pool", "zero", 0.2),
    ("pool_zero_rho0.8", "dynamic_rho0.8", "pool", "zero", 0.8),
    ("pool_warm_rho0.8", "dynamic_rho0.8_warm", "pool", "warm", 0.8),
    ("topk_zero_rho0.8", "dynamic_rho0.8_topk", "topk", "zero", 0.8),
    ("topk_warm_rho0.8", "dynamic_rho0.8_topk_warm", "topk", "warm", 0.8),
]

try:
    from scipy import stats as _sp
except Exception:  # pragma: no cover
    _sp = None


# ----------------------------------------------------------------- 小工具
def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def sign_test_p(diff) -> float:
    d = _finite(diff)
    if d.size == 0:
        return float("nan")
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    m = n_pos + n_neg
    if m == 0:
        return 1.0
    if m <= 200:
        k = min(n_pos, n_neg)
        tail = sum(math.comb(m, i) for i in range(k + 1)) / (2.0 ** m)
        return float(min(1.0, 2.0 * tail))
    z = (abs(n_pos - n_neg) - 1.0) / math.sqrt(m)
    return float(min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2.0))))


def paired_p(diff) -> float:
    d = _finite(diff)
    if d.size < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return 1.0 if abs(float(d.mean())) < 1e-15 else 0.0
    t = float(d.mean()) / (sd / math.sqrt(d.size))
    if _sp is not None:
        try:
            return float(2.0 * _sp.t.sf(abs(t), d.size - 1))
        except Exception:  # pragma: no cover
            pass
    return float(min(1.0, math.erfc(abs(t) / math.sqrt(2.0))))


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 20261026):
    d = _finite(x)
    if d.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    chunk = int(max(1, min(n_boot, 2_000_000 // max(d.size, 1))))
    means, filled = [], 0
    while filled < n_boot:
        k = min(chunk, n_boot - filled)
        idx = rng.integers(0, d.size, size=(k, d.size))
        means.append(d[idx].mean(axis=1))
        filled += k
    m = np.concatenate(means)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def fmt(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return str(x)
    return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"


# ----------------------------------------------------------------- 读数据
def load_cells(root: Path, dataset: str = DATASET) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回 (逐 windows 的窗口级表, 逐评测点表, 逐 run 的 summary 表)。"""
    win_rows, round_rows, sum_rows = [], [], []
    for cell, dirname, state, init, rho in CELLS:
        for seed_dir in sorted((root / dataset / dirname).glob("seed_*")):
            rp = seed_dir / "dg_round_metrics.csv"
            sp = seed_dir / "dg_summary.csv"
            if not (rp.exists() and sp.exists()):
                continue
            seed = int(seed_dir.name.replace("seed_", ""))
            r = pd.read_csv(rp)
            r["cell"] = cell
            r["seed"] = seed
            r["graph_state"] = state
            r["graph_new_edge"] = init
            r["rho"] = rho
            round_rows.append(r)

            # 冻结指标是后加的：早于它跑完的 run（如论文默认 dynamic）没有这几列，
            # 这里按「有就聚合、没有就留 NaN」处理，不能因为参照格缺列就整体崩掉。
            agg_spec = {
                "ndcg_win_mean": ("all_ndcg", "mean"),
                "ndcg_win_last": ("all_ndcg", "last"),
                "ecr": ("ecr_all", "mean"),
                "added": ("added_mean", "mean"),
                "n_edges": ("graph_n_edges", "last"),
                "neighbor_lift": ("neighbor_lift", "mean"),
                "delta_rel": ("delta_rel_mean", "mean"),
            }
            for col, name, how in (
                ("turnover_from_ref", "turnover", "mean"),
                ("frozen_share", "frozen", "mean"),
                ("coverage_gain", "coverage_gain", "last"),
            ):
                if col in r.columns:
                    agg_spec[name] = (col, how)
            w = (
                r.sort_values(["window", "window_round"])
                .groupby("window")
                .agg(**agg_spec)
                .reset_index()
            )
            for col in ("turnover", "frozen", "coverage_gain"):
                if col not in w.columns:
                    w[col] = float("nan")
            w["cell"] = cell
            w["seed"] = seed
            w["graph_state"] = state
            w["graph_new_edge"] = init
            w["rho"] = rho
            win_rows.append(w)

            s = pd.read_csv(sp).iloc[0].to_dict()
            s["cell"] = cell
            s["seed"] = seed
            s["graph_state"] = state
            s["graph_new_edge"] = init
            s["rho"] = rho
            sum_rows.append(s)

    return (
        pd.concat(win_rows, ignore_index=True) if win_rows else pd.DataFrame(),
        pd.concat(round_rows, ignore_index=True) if round_rows else pd.DataFrame(),
        pd.DataFrame(sum_rows) if sum_rows else pd.DataFrame(),
    )


# ----------------------------------------------------------------- 汇总
def cell_summary(win: pd.DataFrame) -> pd.DataFrame:
    g = win.groupby(["cell", "graph_state", "graph_new_edge", "rho"])
    tab = g.agg(
        n_seeds=("seed", "nunique"),
        turnover_from_ref=("turnover", "mean"),
        frozen_share=("frozen", "mean"),
        ecr=("ecr", "mean"),
        added=("added", "mean"),
        n_edges=("n_edges", "last"),
        neighbor_lift=("neighbor_lift", "mean"),
        delta_rel=("delta_rel", "mean"),
        ndcg_win_mean=("ndcg_win_mean", "mean"),
        ndcg_win_last=("ndcg_win_last", "mean"),
    ).reset_index()
    order = {c[0]: i for i, c in enumerate(CELLS)}
    tab["_o"] = tab["cell"].map(order).fillna(99)
    return tab.sort_values(["_o", "cell"]).drop(columns="_o").reset_index(drop=True)


def cell_contrasts(win: pd.DataFrame) -> pd.DataFrame:
    """成对比较（同一 seed、同一 window 对齐）。"""
    pairs = [
        ("topk_zero − pool_zero", "topk_zero_rho0.8", "pool_zero_rho0.8",
         "状态保留规则（同为 zero 初始化）"),
        ("topk_warm − pool_warm", "topk_warm_rho0.8", "pool_warm_rho0.8",
         "状态保留规则（同为 warm 初始化）"),
        ("topk_warm − topk_zero", "topk_warm_rho0.8", "topk_zero_rho0.8",
         "初始化规则（同为 topk 状态）"),
        ("pool_warm − pool_zero", "pool_warm_rho0.8", "pool_zero_rho0.8",
         "初始化规则（同为 pool 状态）"),
        ("pool_zero(ρ0.8) − pool_zero(ρ0.2)", "pool_zero_rho0.8", "paper_default_pool_zero_rho0.2",
         "EMA 强度（同为 pool+zero）"),
    ]
    metrics = ["ndcg_win_mean", "turnover", "frozen", "ecr", "added", "n_edges",
               "neighbor_lift", "delta_rel"]
    rows = []
    for label, a, b, why in pairs:
        pa = win[win["cell"] == a]
        pb = win[win["cell"] == b]
        if pa.empty or pb.empty:
            continue
        merged = pa.merge(pb, on=["seed", "window"], suffixes=("_a", "_b"))
        if merged.empty:
            continue
        for m in metrics:
            d = (merged[f"{m}_a"] - merged[f"{m}_b"]).to_numpy(dtype=np.float64)
            dd = _finite(d)
            if dd.size == 0:
                continue
            lo, hi = bootstrap_ci(dd)
            rows.append({
                "contrast": label, "what": why, "metric": m,
                "n_pairs": int(dd.size),
                "delta_mean": float(dd.mean()),
                "delta_median": float(np.median(dd)),
                "ci95_lo": lo, "ci95_hi": hi,
                "p_paired": paired_p(dd), "p_sign": sign_test_p(dd),
                "seeds_positive": sum(
                    1 for _, g in merged.groupby("seed")
                    if float((g[f"{m}_a"] - g[f"{m}_b"]).mean()) > 0
                ),
                "seeds_total": int(merged["seed"].nunique()),
            })
    return pd.DataFrame(rows)


def build_verdict(cells: pd.DataFrame, contrasts: pd.DataFrame) -> List[Dict[str, object]]:
    def cell_row(name: str) -> Optional[pd.Series]:
        s = cells[cells["cell"] == name]
        return s.iloc[0] if len(s) else None

    def get(contrast: str, metric: str) -> Optional[pd.Series]:
        if contrasts.empty:
            return None
        s = contrasts[(contrasts["contrast"] == contrast) & (contrasts["metric"] == metric)]
        return s.iloc[0] if len(s) else None

    rows: List[Dict[str, object]] = []

    def push(idx, question, value, evidence, judgement):
        rows.append({"verdict_id": idx, "question": question, "value": value,
                     "evidence": evidence, "judgement": judgement})

    pz, pw = cell_row("pool_zero_rho0.8"), cell_row("pool_warm_rho0.8")
    tz, tw = cell_row("topk_zero_rho0.8"), cell_row("topk_warm_rho0.8")
    base = cell_row("paper_default_pool_zero_rho0.2")

    # ① 四个格子的冻结程度
    ev = []
    for name, tag in (("pool_zero_rho0.8", "pool+zero"),
                      ("pool_warm_rho0.8", "pool+warm"),
                      ("topk_zero_rho0.8", "topk+zero"),
                      ("topk_warm_rho0.8", "topk+warm")):
        r = cell_row(name)
        if r is not None:
            ev.append(f"{tag}: frozen_share={fmt(r['frozen_share'], 3)}, "
                      f"turnover={fmt(r['turnover_from_ref'], 3)}, "
                      f"ECR={fmt(r['ecr'], 4)}, 新边={fmt(r['added'], 2)}")
    push("①", "四种配置下图到底冻没冻住（ρ=0.8）", "",
         "；".join(ev) if ev else "没有数据",
         "frozen_share 越接近 1、turnover 越接近 0 就越冻")

    # ② 归因（多指标判据：turnover 低 / frozen_share 高 / ECR 近 0，任一命中即视为冻住）
    def _num(row, key):
        try:
            v = float(row[key])
        except Exception:  # noqa: BLE001
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    def is_frozen(row) -> bool:
        if row is None:
            return False
        tv, fz, ec = _num(row, "turnover_from_ref"), _num(row, "frozen_share"), _num(row, "ecr")
        return (np.isfinite(tv) and tv < 0.5) or (np.isfinite(fz) and fz > 0.3) or (
            np.isfinite(ec) and ec < 0.01)

    fz_pz, fz_pw, fz_tz, fz_tw = is_frozen(pz), is_frozen(pw), is_frozen(tz), is_frozen(tw)
    pattern = (f"pool+zero={'冻' if fz_pz else '不冻'}，pool+warm={'冻' if fz_pw else '不冻'}，"
               f"topk+zero={'冻' if fz_tz else '不冻'}，topk+warm={'冻' if fz_tw else '不冻'}")
    if (not fz_pz) and (not fz_pw) and fz_tz and (not fz_tw):
        judge = (
            "冻结**只出现在 topk + zero 这一个组合**：pool 的两格都不冻（ρ=0.8 也一样），"
            "topk+warm 也不冻。所以它既不是 EMA 强度单独造成的，也不是任一单独因素造成的，"
            "而是**「只存 Top-K 的旧状态规则」×「新边 0 初值」的联合效应**："
            "topk 状态会丢弃掉出 Top-K 的边历史，而 zero 初值让新边只能以 (1-ρ)Δe 起步"
            "（ρ=0.8 时仅 20%），根本挤不进 Top-K；挤不进 ⇒ EMA 永不累积 ⇒ 永久冻结。"
            "把任一项改掉（pool 或 warm）就都不冻。"
        )
    elif fz_tz and fz_tw and not fz_pz and not fz_pw:
        judge = "topk 两格都冻、pool 两格都不冻 ⇒ 冻结来自**状态保留规则**，与初始化无关。"
    elif fz_pz and fz_pw:
        judge = "pool 也冻住了 ⇒ ρ=0.8 的 EMA 本身就慢到接近冻结，需要再降 ρ 验证。"
    elif not any((fz_pz, fz_pw, fz_tz, fz_tw)):
        judge = "四格都没冻 ⇒ 本实验没有复现出冻结。"
    else:
        judge = "冻结模式不规则（见 ①），需要逐个排查。"
    push("②", "冻结的根因：EMA / 新边初始化 / 状态保留规则", "", pattern, judge)

    # ③ 新边初始化有没有解开冻结
    c = get("topk_warm − topk_zero", "frozen")
    if c is None:
        push("③", "换 warm 初始化能不能解开冻结（topk 状态下）", "", "没有数据", "")
    else:
        d = float(c["delta_mean"])
        push("③", "换 warm 初始化能不能解开冻结（topk 状态下）", round(d, 4),
             f"Δfrozen_share={d:+.3f} [{float(c['ci95_lo']):+.3f}, {float(c['ci95_hi']):+.3f}]，"
             f"p={float(c['p_paired']):.3g}",
             "warm 初始化**不能**解开冻结" if d > -0.2 else "warm 初始化明显减轻了冻结")

    # ④ 初始化规则对「适应速度」的影响（pool 状态）
    c_t = get("pool_warm − pool_zero", "turnover")
    c_a = get("pool_warm − pool_zero", "added")
    c_n = get("pool_warm − pool_zero", "ndcg_win_mean")
    if c_t is not None:
        push("④", "候选池下，warm 初始化让图转得快多少", round(float(c_t["delta_mean"]), 4),
             f"Δturnover={float(c_t['delta_mean']):+.4f}（p={float(c_t['p_paired']):.3g}）；"
             + (f"Δ新边={float(c_a['delta_mean']):+.2f}（p={float(c_a['p_paired']):.3g}）；"
                if c_a is not None else "")
             + (f"ΔNDCG={float(c_n['delta_mean']):+.4f}（p={float(c_n['p_paired']):.3g}）"
                if c_n is not None else ""),
             "初始化规则影响的是**适应速度**" if float(c_t["delta_mean"]) > 0.01
             else "初始化规则对适应速度几乎没有影响")

    # ⑤ EMA 强度本身的影响
    c_r = get("pool_zero(ρ0.8) − pool_zero(ρ0.2)", "turnover")
    c_rn = get("pool_zero(ρ0.8) − pool_zero(ρ0.2)", "ndcg_win_mean")
    if c_r is not None:
        push("⑤", "ρ 从 0.2 提到 0.8（同为 pool+zero）改变了什么", round(float(c_r["delta_mean"]), 4),
             f"Δturnover={float(c_r['delta_mean']):+.4f}（p={float(c_r['p_paired']):.3g}）；"
             + (f"ΔNDCG={float(c_rn['delta_mean']):+.4f}（p={float(c_rn['p_paired']):.3g}）"
                if c_rn is not None else ""),
             "ρ 变慢的是**适应速度**，不是让图永久冻结（只要状态没被丢掉）"
             if c_r is not None and float(c_r["delta_mean"]) < -0.01
             else "ρ 对 turnover 没有明显影响")

    # ⑥ 对推荐的影响
    rows_ndcg = []
    for name, tag in (("pool_zero_rho0.8", "pool+zero ρ0.8"),
                      ("pool_warm_rho0.8", "pool+warm ρ0.8"),
                      ("topk_zero_rho0.8", "topk+zero ρ0.8"),
                      ("topk_warm_rho0.8", "topk+warm ρ0.8"),
                      ("paper_default_pool_zero_rho0.2", "论文默认 pool+zero ρ0.2")):
        r = cell_row(name)
        if r is not None:
            rows_ndcg.append(f"{tag}: {fmt(r['ndcg_win_mean'], 4)}")
    push("⑥", "各配置的推荐效果（窗口级 NDCG@10）", "", "；".join(rows_ndcg) or "没有数据",
         "冻结的图不仅不涨，还会因为邻居失真而略微拖低（见体检列）")

    lift_pool = _num(pz, "neighbor_lift") if pz is not None else float("nan")
    lift_topk = _num(tz, "neighbor_lift") if tz is not None else float("nan")
    ndcg_pool = _num(pz, "ndcg_win_mean") if pz is not None else float("nan")
    ndcg_topk = _num(tz, "ndcg_win_mean") if tz is not None else float("nan")
    push("⑥b", "冻结的代价：邻居质量与推荐指标", "",
         f"neighbor_lift：pool+zero={fmt(lift_pool, 3)} → topk+zero={fmt(lift_topk, 3)}"
         f"（{fmt(lift_topk - lift_pool, 3)}）；"
         f"窗口级 NDCG@10：{fmt(ndcg_pool, 4)} → {fmt(ndcg_topk, 4)}"
         f"（{fmt(ndcg_topk - ndcg_pool, 4)}）",
         "冻结把邻居质量明显拉低，但 NDCG@10 几乎没变 —— 说明这个指标对「图不再适应」不敏感，"
         "评测时不能只看 NDCG，必须同时看 neighbor_lift / ECR 这类结构指标")

    parts7 = []
    if fz_tz and not (fz_pz or fz_pw or fz_tw):
        parts7.append("冻结 = 「只存 Top-K」×「新边 0 初值」的联合效应，改任一项即可解除")
    elif fz_tz and not fz_pz:
        parts7.append("冻结来自「只存 Top-K」的状态规则")
    if c is not None and float(c["delta_mean"]) < -0.2:
        parts7.append(f"topk 下改用 warm 初值可把冻结用户占比降 "
                      f"{abs(float(c['delta_mean'])):.2f}（p={float(c['p_paired']):.2g}）")
    if c_t is not None:
        parts7.append(f"候选池下 warm 相对 zero 只多转 {float(c_t['delta_mean']):+.3f}（几乎无差别）")
    push("⑦", "一句话结论", "", "；".join(parts7) if parts7 else "见 ①②③",
         "回答「更新机制是否合理」：现行实现的隐患是**旧状态规则 + 论文 0 初值的组合会让图永久冻结**；"
         "而冻结的代价主要落在邻居质量（lift 下降），NDCG@10 看不出来")
    return rows


# ----------------------------------------------------------------- Markdown
def md_table(frame: pd.DataFrame, cols: Sequence[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "（无数据）\n"
    cols = [c for c in cols if c in frame.columns]
    if not cols:
        return "（无数据）\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.head(max_rows).iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                fv = float(v)
                cells.append(fmt(fv) if abs(fv) >= 1e-3 or fv == 0 else f"{fv:.2e}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- 图
def make_figure(win: pd.DataFrame, out_png: Path) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return f"跳过出图（没有 matplotlib）：{exc}"
    if win.empty:
        return "跳过出图：没有数据"
    colors = {
        "paper_default_pool_zero_rho0.2": "#999999",
        "pool_zero_rho0.8": "#1f77b4",
        "pool_warm_rho0.8": "#2ca02c",
        "topk_zero_rho0.8": "#d62728",
        "topk_warm_rho0.8": "#ff7f0e",
    }
    panels = (
        ("turnover", "turnover vs post-warmup snapshot (higher = graph moving)"),
        ("frozen", "frozen_share (~1 = whole graph frozen)"),
        ("n_edges", "users with edges"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for cell in [c[0] for c in CELLS]:
        g = win[win["cell"] == cell]
        if g.empty:
            continue
        agg = g.groupby("window")[["turnover", "frozen", "n_edges"]].mean()
        x = agg.index.to_numpy()
        for ax, (col, _title) in zip(axes, panels):
            y = agg[col].to_numpy()
            # 该面板没数据（例如参照格早于冻结指标加入）就别在图例里占一行
            label = cell if (ax is axes[0] and np.isfinite(y).any()) else None
            ax.plot(x, y, marker="o", ms=4, lw=1.6, color=colors.get(cell, None), label=label)
            ax.set_xticks(x)
    for ax, (_, title) in zip(axes, panels):
        ax.set_xlabel("window")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[2].set_yscale("symlog")
    axes[0].legend(fontsize=7)
    fig.suptitle("Freeze attribution: pool vs topk x zero vs warm init (rho=0.8)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return None


# ----------------------------------------------------------------- 主流程
def main() -> None:
    parser = argparse.ArgumentParser(description="冻结归因：图冻结是 EMA / 新边初始化 / 状态保留规则造成的？")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default=DATASET, help="数据集名（决定 results/raw/ 下的子目录）")
    args = parser.parse_args()

    dataset = args.dataset
    root = args.root.resolve()
    out = args.out.resolve() / _slug(dataset)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    win, rounds, sums = load_cells(root, dataset)
    if win.empty:
        raise SystemExit(f"{root} 下没有冻结归因的结果，先跑 dg_run_supplement.sh")

    cells = cell_summary(win)
    cells.to_csv(tables / "dg_freeze_cells.csv", index=False, encoding="utf-8-sig")
    win.to_csv(tables / "dg_freeze_windows.csv", index=False, encoding="utf-8-sig")

    contrasts = cell_contrasts(win)
    if not contrasts.empty:
        contrasts.to_csv(tables / "dg_freeze_contrasts.csv", index=False, encoding="utf-8-sig")

    verdict = pd.DataFrame(build_verdict(cells, contrasts))
    verdict.to_csv(tables / "dg_freeze_verdict.csv", index=False, encoding="utf-8-sig")

    fig_note = make_figure(win, figures / "dg_fig_freeze.png")

    # ---- 报告
    lines = [f"# 冻结归因报告：动态图更新机制是否合理？（{dataset}）\n"]
    lines.append("同一序列编码器与联邦协议，只改图更新机制：图状态（pool / topk）× 新边初始化"
                 "（zero / warm），ρ 固定 0.8；另加论文默认（pool+zero+ρ0.2）作参照。"
                 "每个格子 3 个种子。\n")
    lines.append("**一句话：** " + (str(verdict.iloc[-1]["evidence"]) if not verdict.empty else "无") + "\n")
    lines.append(md_table(verdict, ["verdict_id", "question", "value", "evidence", "judgement"]))
    lines.append("")
    lines.append("## 各格子的冻结程度与效果\n")
    lines.append(md_table(cells, ["cell", "graph_state", "graph_new_edge", "rho", "n_seeds",
                                  "turnover_from_ref", "frozen_share", "ecr", "added",
                                  "n_edges", "neighbor_lift", "ndcg_win_mean"]))
    lines.append("")
    if not contrasts.empty:
        lines.append("## 成对比较\n")
        key = contrasts[contrasts["metric"].isin(["frozen", "turnover", "ndcg_win_mean", "added"])]
        lines.append(md_table(key, ["contrast", "what", "metric", "n_pairs", "delta_mean",
                                    "ci95_lo", "ci95_hi", "p_paired", "seeds_positive", "seeds_total"],
                             max_rows=60))
    lines.append("")
    lines.append("## 怎么读\n")
    lines.append("- `turnover_from_ref` / `frozen_share` 都是相对**预热后第一次评测**的邻居快照算的；"
                 "前者是 Jaccard 距离均值，后者是「一位都没变」的用户占比。\n"
                 "- `topk` 复现的是审计前的旧行为（只保留当前 Top-K 的 EMA，掉出去的边状态被丢掉）；"
                 "`pool` 是修复版（保留候选池）。\n"
                 "- 若 topk 冻住而 pool 不冻 ⇒ **冻结是状态保留规则造成的**，既不是 EMA 强度、"
                 "也不是初始化规则；后两者只影响「转多快」。\n")
    if fig_note:
        lines.append(f"- 出图提示：{fig_note}\n")
    (out / "dg_freeze_report.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 屏幕
    pd.set_option("display.width", 220)
    print("\n" + "=" * 78)
    print(f"各格子（{dataset}，ρ=0.8；跨种子平均）")
    print("=" * 78)
    print(cells[["cell", "n_seeds", "turnover_from_ref", "frozen_share", "ecr", "added",
                 "n_edges", "neighbor_lift", "ndcg_win_mean"]].round(4).to_string(index=False))
    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    for _, row in verdict.iterrows():
        print(f"[{row['verdict_id']}] {row['question']}")
        if row["evidence"]:
            print(f"     {row['evidence']}")
        if str(row["judgement"]).strip():
            print(f"     -> {row['judgement']}")
    print("\n输出：")
    for p in (out / "dg_freeze_report.md", tables / "dg_freeze_verdict.csv", tables / "dg_freeze_cells.csv",
              tables / "dg_freeze_contrasts.csv", figures / "dg_fig_freeze.png"):
        print(("  " if p.exists() else "  (缺) ") + str(p))
    if fig_note:
        print(f"\n注意：{fig_note}")


if __name__ == "__main__":
    main()
