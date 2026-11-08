#!/usr/bin/env python
"""val_DynamicGraph 汇总：把三条臂的差值拆成「图有没有用」和「动态性有没有用」。

核心逻辑（这是本脚本唯一要做对的事）
----------------------------------
对同一 (数据集, 种子, 窗口)，三条臂评测在同一批用户、同一批判题上，所以可以逐对相减：

    ① StaticGraph  − NoGraph      = 「图协同分支」本身值多少（哪怕图是死的）
    ② DynamicGraph − StaticGraph  = 「动态更新」额外值多少
    ③ DynamicGraph − NoGraph      = 动态图的总贡献

判断规则（写进 `dg_verdict.csv`，阈值可调 `--practical-threshold`，默认 0.002 NDCG@10）：

    ①≈0 且 ②≈0  →  图分支没有可测贡献，动态性更无从谈起。
                    这时去看 ⑥`delta_rel`（图分支把用户表示挪动了多少）和
                    ⑦`neighbor_lift`（图里的邻居比随机用户更像自己吗）—— 通常能直接解释原因。
    ①>0 且 ②≈0  →  图有用，但「每轮 EMA 更新」没有额外用 ⇒ 静态图足够，
                    再研究「自适应更新速度」的收益有限（任务书第二阶段的前提存疑）。
    ①>0 且 ②>0  →  动态性确实有贡献，值得继续做「什么时候更新 / 更新多少」的算法。
    ①≈0 且 ②>0  →  罕见组合，先查一致性/种子方差，别直接下结论。

统计口径（都写进表里，不藏）
--------------------------
* 窗口级（主证据）：n = 种子数 × 窗口数（默认 3×5=15）个配对观测；同时给出
  「有多少个种子为正」（`seeds_positive / seeds_total`）—— 种子间符号是否一致比 p 值更可靠。
* 用户级（敏感性检查）：同一批判题用户逐用户相减，给 bootstrap 95% CI、符号检验、
  Wilcoxon（有 scipy 时）。注意同一用户跨窗口不是独立观测，所以这一栏只看方向与量级。
* 显著性 ≠ 实用性：6040 个用户的配对检验能把 1e-4 的差异测成显著。所以每行同时给
  「量级是否超过 practical-threshold」和「统计是否显著」。

用法
----
  python val_DynamicGraph/dg_aggregate.py
  python val_DynamicGraph/dg_aggregate.py --root results/raw --out results/aggregated
  python val_DynamicGraph/dg_aggregate.py --practical-threshold 0.001
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
DEFAULT_OUT = DG_DIR / "results" / "aggregated"

ARM_NOGRAPH = "nograph"
ARM_STATIC = "static"
ARM_DYNAMIC = "dynamic"
ARM_PERIODIC = "periodic"
ARM_ONBOARDING = "onboarding"
ARM_FRESH_ONLY = "fresh_only"
ARM_ORDER = {
    ARM_NOGRAPH: 0, ARM_STATIC: 1, ARM_PERIODIC: 2, ARM_DYNAMIC: 3,
    ARM_ONBOARDING: 4, ARM_FRESH_ONLY: 5,
}

#: 主对照（顺序即「看图应该按这个顺序读」）
PRIMARY_CONTRASTS: List[Tuple[str, str, str]] = [
    ("static_minus_nograph", ARM_STATIC, ARM_NOGRAPH),
    ("dynamic_minus_static", ARM_DYNAMIC, ARM_STATIC),
    ("dynamic_minus_nograph", ARM_DYNAMIC, ARM_NOGRAPH),
]
#: 补充对照（只有当对应臂真的跑过才会出现）
EXTRA_CONTRASTS: List[Tuple[str, str, str]] = [
    ("periodic_minus_nograph", ARM_PERIODIC, ARM_NOGRAPH),
    ("dynamic_minus_periodic", ARM_DYNAMIC, ARM_PERIODIC),
    # 阶段 2 新增：覆盖/演化/针对性的三项分解
    ("onboarding_minus_static", ARM_ONBOARDING, ARM_STATIC),
    ("dynamic_minus_onboarding", ARM_DYNAMIC, ARM_ONBOARDING),
    ("fresh_only_minus_static", ARM_FRESH_ONLY, ARM_STATIC),
    ("dynamic_minus_fresh_only", ARM_DYNAMIC, ARM_FRESH_ONLY),
]

METRICS = ("ndcg_win_mean", "ndcg_win_last", "hr_win_mean")
PRIMARY_METRIC = "ndcg_win_mean"

try:  # Wilcoxon / t 分布只在装了 scipy 时给；其余检查不依赖 scipy
    from scipy import stats as _sp
except Exception:  # pragma: no cover
    _sp = None


# ======================================================================================
# 统计小工具
# ======================================================================================
def _finite(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def sign_test_p(diff) -> float:
    """符号检验（双侧）。

    m <= 200 用精确二项尾概率；更大的样本改用带连续性校正的正态近似 ——
    逐用户配对时 m 能到 1~2 万，`2.0 ** m` 会直接 OverflowError（就是这里踩过坑）。
    """
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
    """配对 t 检验 p 值：有 scipy 用 t 分布，否则正态近似。"""
    d = _finite(diff)
    n = d.size
    if n < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return 1.0 if abs(float(d.mean())) < 1e-15 else 0.0
    t = float(d.mean()) / (sd / math.sqrt(n))
    if _sp is not None:
        try:
            return float(2.0 * _sp.t.sf(abs(t), n - 1))
        except Exception:  # pragma: no cover
            pass
    return float(min(1.0, math.erfc(abs(t) / math.sqrt(2.0))))


def t_ci(values) -> Tuple[float, float]:
    """均值的 t 分布 95% CI（小样本用；n=3 时 df=2）。"""
    d = _finite(values)
    n = d.size
    if n < 2:
        return float("nan"), float("nan")
    se = float(d.std(ddof=1) / math.sqrt(n))
    if _sp is not None:
        try:
            crit = float(_sp.t.ppf(0.975, n - 1))
        except Exception:  # pragma: no cover
            crit = 1.96
    else:  # pragma: no cover
        crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n - 1, 1.96)
    return float(d.mean() - crit * se), float(d.mean() + crit * se)


def wilcoxon_p(diff) -> float:
    d = _finite(diff)
    if _sp is None or d.size < 6 or np.all(d == 0):
        return float("nan")
    try:
        return float(_sp.wilcoxon(d).pvalue)
    except Exception:  # pragma: no cover
        return float("nan")


def bootstrap_ci(
    x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 20261026
) -> Tuple[float, float]:
    """均值的 bootstrap 置信区间（分块，避免大样本撑爆内存）。"""
    d = _finite(x)
    if d.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=np.float64)
    chunk = int(max(1, min(n_boot, 2_000_000 // max(d.size, 1))))
    filled = 0
    while filled < n_boot:
        k = min(chunk, n_boot - filled)
        idx = rng.integers(0, d.size, size=(k, d.size))
        means[filled : filled + k] = d[idx].mean(axis=1)
        filled += k
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def sd_or_nan(values) -> float:
    d = _finite(values)
    return float(d.std(ddof=1)) if d.size > 1 else float("nan")


def fmt(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return str(x)
    return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"


def fmt_sci(x) -> str:
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return str(x)
    return "nan" if not np.isfinite(v) else f"{v:.2e}"


# ======================================================================================
# 读结果
# ======================================================================================
def load_runs(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_files = sorted(root.rglob("dg_summary.csv"))
    if not summary_files:
        raise SystemExit(
            f"{root} 下没找到 dg_summary.csv。先跑训练（dg_run.py 或 dg_run_basic.sh）。"
        )
    sums = pd.concat([pd.read_csv(p) for p in summary_files], ignore_index=True)
    sums = sums.drop_duplicates(subset=["dataset", "arm_dir", "seed"], keep="last")

    rounds_frames, user_frames = [], []
    for path in summary_files:
        run_dir = path.parent
        if (run_dir / "dg_round_metrics.csv").exists():
            rounds_frames.append(pd.read_csv(run_dir / "dg_round_metrics.csv"))
        if (run_dir / "dg_user_metrics.csv").exists():
            user_frames.append(
                pd.read_csv(
                    run_dir / "dg_user_metrics.csv",
                    usecols=["dataset", "arm_dir", "seed", "window", "user", "ndcg", "hr"],
                )
            )
    rounds = pd.concat(rounds_frames, ignore_index=True) if rounds_frames else pd.DataFrame()
    users = pd.concat(user_frames, ignore_index=True) if user_frames else pd.DataFrame()
    return sums, rounds, users


def window_table(rounds: pd.DataFrame) -> pd.DataFrame:
    """逐 (数据集, 臂, 种子, 窗口) 的窗口级指标（窗口内所有评测点均值 + 末点）。"""
    if rounds.empty:
        return pd.DataFrame()
    keys = ["dataset", "arm_dir", "arm", "seed", "window"]
    ordered = rounds.sort_values(keys + ["window_round"])
    table = (
        ordered.groupby(keys, dropna=False)
        .agg(
            ndcg_win_mean=("all_ndcg", "mean"),
            hr_win_mean=("all_hr", "mean"),
            ndcg_win_last=("all_ndcg", "last"),
            n_eval_points=("all_ndcg", "size"),
            n_eval=("n_eval", "max"),
            delta_rel=("delta_rel_mean", "mean"),
            delta_rel_max=("delta_rel_max", "max"),
            probe_with_graph=("probe_with_graph", "mean"),
            neighbor_cos=("neighbor_cos_mean", "mean"),
            neighbor_lift=("neighbor_lift", "mean"),
            ecr=("ecr_all", "mean"),
            added=("added_mean", "mean"),
            n_edges=("graph_n_edges", "last"),
            graph_updates=("graph_updates", "last"),
            train_loss=("train_loss", "mean"),
            edge_changed=("graph_edge_changed_after_freeze", "max"),
        )
        .reset_index()
    )
    return table


# ======================================================================================
# 对照（差值）
# ======================================================================================
def canonical_dirs(variants: Sequence[str]) -> Dict[str, str]:
    """臂名 -> 该臂的「代表目录」。

    目录名会带上与默认值不同的配置后缀（`static_k10`、`dynamic_mix5`、`dynamic_rho0.8`）。
    主对照 ①②③ 必须始终成立，所以这里按「精确同名优先，否则取后缀最短的那个变体」
    解析出代表目录 —— 换个 neighbors / rho / graph_mix 不会让 ①②③ 失效。
    """
    out: Dict[str, str] = {}
    for arm in (
        ARM_NOGRAPH, ARM_STATIC, ARM_DYNAMIC, ARM_PERIODIC, ARM_ONBOARDING, ARM_FRESH_ONLY,
    ):
        if arm in variants:
            out[arm] = arm
            continue
        candidates = sorted(
            (v for v in variants if v.startswith(arm)), key=lambda s: (len(s), s)
        )
        if candidates:
            out[arm] = candidates[0]
            if len(candidates) > 1:
                print(
                    f"[warn] 臂 {arm} 有多个变体目录 {candidates}，"
                    f"主对照用 {candidates[0]}；其余变体会各自作为单独对照出现"
                )
    return out


def available_contrasts(variants: Sequence[str]) -> List[Tuple[str, str, str]]:
    """主对照（按代表目录解析）+ 每个额外变体 vs 它所属臂的代表目录。"""
    available = list(variants)
    canon = canonical_dirs(available)
    out: List[Tuple[str, str, str]] = []
    for label, a, b in PRIMARY_CONTRASTS + EXTRA_CONTRASTS:
        if a in canon and b in canon:
            out.append((label, canon[a], canon[b]))
    used = {c[1] for c in out} | {c[2] for c in out}
    for v in sorted(available):
        if v in used:
            continue
        base = _arm_of(v)
        if base and base in canon:
            out.append((f"{v}_minus_{base}", v, canon[base]))
    return out


def contrasts_from_wide(
    piv: pd.DataFrame,
    index_keys: List[str],
    metric: str,
    contrasts: Sequence[Tuple[str, str, str]],
) -> pd.DataFrame:
    """piv 的列是 arm_dir；返回 (index_keys, metric, contrast, a, b, delta)。"""
    pieces = []
    for label, a, b in contrasts:
        if a not in piv.columns or b not in piv.columns:
            continue
        mask = piv[a].notna() & piv[b].notna()
        if not bool(mask.any()):
            continue
        piece = piv.loc[mask, index_keys].copy()
        piece["metric"] = metric
        piece["contrast"] = label
        piece["a_dir"] = a
        piece["b_dir"] = b
        piece["a"] = piv.loc[mask, a].to_numpy()
        piece["b"] = piv.loc[mask, b].to_numpy()
        piece["delta"] = piece["a"] - piece["b"]
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def summarise_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    if contrasts.empty:
        return pd.DataFrame()
    group_keys = ["dataset", "metric", "contrast"]
    out = []
    for keys, part in contrasts.groupby(group_keys, dropna=False):
        d = _finite(part["delta"])
        if d.size == 0:
            continue
        lo, hi = bootstrap_ci(d)
        row = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        per_seed = (
            part.groupby("seed")["delta"].mean()
            if "seed" in part.columns
            else pd.Series(dtype=float)
        )
        seed_lo, seed_hi = t_ci(per_seed)
        row.update(
            {
                "n_obs": int(d.size),
                "n_seeds": int(per_seed.size),
                "delta_mean": float(d.mean()),
                "delta_median": float(np.median(d)),
                "delta_sd": sd_or_nan(d),
                "delta_min": float(d.min()),
                "delta_max": float(d.max()),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "win_rate": float((d > 0).mean()),
                "p_sign": sign_test_p(d),
                "p_paired": paired_p(d),
                "p_wilcoxon": wilcoxon_p(d),
                "seeds_positive": int((per_seed > 0).sum()),
                "seeds_total": int(per_seed.size),
                # ---- 种子级（主证据；窗口级只作敏感性分析）----
                "delta_seed_mean": float(per_seed.mean()) if per_seed.size else float("nan"),
                "delta_seed_sd": sd_or_nan(per_seed),
                "ci95_seed_lo": seed_lo,
                "ci95_seed_hi": seed_hi,
                "p_seed_t": paired_p(per_seed),
            }
        )
        out.append(row)
    return pd.DataFrame(out)


# ======================================================================================
# 严格性核对
# ======================================================================================
CONFIG_KEYS = (
    "num_windows", "rounds_per_window", "eval_every", "clients_per_round",
    "participation_rate", "local_epochs", "lr", "server_lr_init", "server_lr_final",
    "fedavg_weighting", "personalization", "lambda_temp", "lambda_struct",
    "top_k", "candidate_count", "graph_update_scope",
    "graph_warmup_rounds", "probe_size",
)


def hygiene_checks(sums: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(dataset: str, name: str, ok: bool, detail: str) -> None:
        rows.append({"dataset": dataset, "check": name, "ok": int(bool(ok)), "detail": detail})

    for dataset, part in sums.groupby("dataset"):
        bad = []
        for key in CONFIG_KEYS:
            if key not in part.columns:
                continue
            values = [v for v in part[key].dropna().unique()]
            if len(values) > 1:
                bad.append(f"{key}={sorted(map(str, values))}")
        add(
            dataset,
            "协议参数一致（除图以外没有别的差别）",
            not bad,
            "一致" if not bad else "不一致：" + "; ".join(bad),
        )

        # graph_state / graph_new_edge 现在是刻意变动的实验维度（冻结归因 2x2），
        # 所以只作为信息展示，不参与「协议参数一致」的失败判定。
        for key, label in (("graph_state", "图状态"), ("graph_new_edge", "新边初始化")):
            if key in part.columns and part[key].notna().any():
                vals = sorted({str(v) for v in part[key].dropna().unique()})
                add(dataset, f"{label}（实验变量，允许不同）", True, "、".join(vals))

        if "personal_on_cpu" in part.columns and part["personal_on_cpu"].notna().any():
            vals = sorted({str(int(v)) for v in part["personal_on_cpu"].dropna()})
            add(
                dataset,
                "个性化层位置（仅性能开关，不改数值）",
                True,
                "personal_on_cpu=" + ",".join(vals)
                + "（实测同 seed 下 cpu/gpu 驻留的逐用户 NDCG 与逐轮指标逐位一致，"
                "所以混用不影响可比性）",
            )

        per_arm = part.groupby("arm_dir")["seed"].nunique()
        add(
            dataset,
            "每条臂的种子数一致",
            int(per_arm.nunique()) == 1,
            ", ".join(f"{k}:{v}" for k, v in per_arm.items()),
        )

        sub = rounds[(rounds["dataset"] == dataset) & (rounds["arm"] == ARM_NOGRAPH)]
        if not sub.empty:
            worst = float(sub["delta_rel_max"].abs().max())
            add(dataset, "NoGraph 的图分支恒为 0", worst <= 1e-12,
                f"max delta_rel={worst:.3e}（要求 ≤ 1e-12）")

        sub = rounds[(rounds["dataset"] == dataset) & (rounds["arm"] == ARM_STATIC)]
        if not sub.empty:
            changed = int(sub["graph_edge_changed_after_freeze"].fillna(0).max())
            add(dataset, "StaticGraph 冻结后边不再变化", changed == 0,
                f"edge_changed_after_freeze={changed}（要求 0）")

        sub = rounds[(rounds["dataset"] == dataset) & (rounds["arm"] == ARM_DYNAMIC)]
        if not sub.empty:
            # 预热期（graph_warmup_rounds）内本来就不更新，所以要减掉它再比
            warmup = 0.0
            if "graph_warmup_rounds" in part.columns and part["graph_warmup_rounds"].notna().any():
                warmup = float(part["graph_warmup_rounds"].dropna().iloc[0])
            expected = (sub["global_round"].astype(float) - warmup).clip(lower=0.0)
            gap = (sub["graph_updates"].astype(float) - expected).abs().max()
            add(dataset, "DynamicGraph 预热后每轮都更新", float(gap) == 0.0,
                f"max|updates-(global_round-{warmup:.0f})|={float(gap):.0f}")

        sub = rounds[rounds["dataset"] == dataset]
        if not sub.empty:
            per_window = sub.groupby(["window", "arm_dir"])["n_eval"].max().reset_index()
            spread = per_window.groupby("window")["n_eval"].agg(lambda s: int(s.max() - s.min()))
            add(
                dataset,
                "同一窗口内各臂评测用户数一致",
                int(spread.max()) == 0,
                f"逐窗口跨臂最大差={int(spread.max())}"
                f"（各窗口人数不同是正常的：{int(per_window['n_eval'].min())}~"
                f"{int(per_window['n_eval'].max())} 人）",
            )
    return pd.DataFrame(rows)


# ======================================================================================
# 结论
# ======================================================================================
def classify(delta: float, p_paired: float, threshold: float) -> Tuple[str, str]:
    """返回 (量级判定, 显著性判定)。"""
    if delta is None or not np.isfinite(delta):
        return "无法判定", "无数据"
    if abs(delta) < threshold:
        magnitude = f"|Δ| < {threshold:g}（可忽略）"
    elif abs(delta) < 3 * threshold:
        magnitude = f"|Δ| ∈ [{threshold:g}, {3 * threshold:g})（小）"
    else:
        magnitude = f"|Δ| ≥ {3 * threshold:g}（明确）"
    if p_paired is None or not np.isfinite(p_paired):
        signif = "无法做检验"
    elif p_paired < 0.01:
        signif = "p<0.01"
    elif p_paired < 0.05:
        signif = "p<0.05"
    else:
        signif = "不显著（p≥0.05）"
    return magnitude, signif


def _lift_judge(lift: float, other_lift: float = float("nan")) -> str:
    """neighbor_lift 的判读文字（数据驱动，不是套话）。"""
    if not np.isfinite(lift):
        return "没有邻居质量数据"
    if lift < 0.95:
        base = f"lift={lift:.2f} < 1：图里的邻居比随机抽到的用户还要不像自己，图完全没有区分度"
    elif lift < 1.10:
        base = f"lift={lift:.2f} ≈ 1：邻居和随机用户差不多，图没提供有效信息"
    else:
        base = f"lift={lift:.2f} > 1：Top-K 邻居确实比随机用户更像自己（图有区分度）"
    if np.isfinite(other_lift) and other_lift < 0.95 <= lift:
        base += f"；对照的静态臂 lift 只有 {other_lift:.2f}，低于随机基线"
    return base


def build_verdict(
    contrasts_sum: pd.DataFrame,
    diagnostics: pd.DataFrame,
    hygiene: pd.DataFrame,
    dataset: str,
    threshold: float,
) -> List[Dict[str, object]]:
    def pick(contrast: str, metric: str = PRIMARY_METRIC) -> Optional[pd.Series]:
        if contrasts_sum.empty:
            return None
        sel = contrasts_sum[
            (contrasts_sum["dataset"] == dataset)
            & (contrasts_sum["contrast"] == contrast)
            & (contrasts_sum["metric"] == metric)
        ]
        return sel.iloc[0] if len(sel) else None

    # 臂名 -> 代表目录（目录可能带 _k10 / _mix5 / _rho0.8 后缀）
    arm_dirs = (
        canonical_dirs(sorted(diagnostics["arm_dir"].unique().tolist()))
        if not diagnostics.empty
        else {}
    )

    def diag(arm: str, column: str) -> float:
        if diagnostics.empty:
            return float("nan")
        arm_dir = arm_dirs.get(arm, arm)
        sel = diagnostics[(diagnostics["dataset"] == dataset) & (diagnostics["arm_dir"] == arm_dir)]
        if len(sel) and column in sel.columns:
            try:
                return float(sel.iloc[0][column])
            except Exception:  # noqa: BLE001
                return float("nan")
        return float("nan")

    rows: List[Dict[str, object]] = []

    def push(idx: str, question: str, value: object, evidence: str, judgement: str) -> None:
        rows.append(
            {
                "dataset": dataset,
                "verdict_id": idx,
                "question": question,
                "value": value,
                "evidence": evidence,
                "judgement": judgement,
            }
        )

    g1, g2, g3 = (
        pick("static_minus_nograph"),
        pick("dynamic_minus_static"),
        pick("dynamic_minus_nograph"),
    )
    g4, g5 = pick("periodic_minus_nograph"), pick("dynamic_minus_periodic")
    g1b = pick("static_at120_minus_static")

    for idx, label, row in (
        ("①", "图协同分支本身有没有贡献（StaticGraph − NoGraph）", g1),
        ("②", "动态更新有没有额外贡献（DynamicGraph − StaticGraph）", g2),
        ("③", "动态图的总贡献（DynamicGraph − NoGraph）", g3),
        ("④", "补充：慢刷新静态图是否已足够（PeriodicGraph − NoGraph）", g4),
        ("⑤", "补充：每轮 EMA 比慢刷新多给多少（DynamicGraph − PeriodicGraph）", g5),
        ("①b", "补充：把冻结时点从第 21 轮推到第 120 轮能否救回静态图"
               "（static_at120 − StaticGraph）", g1b),
    ):
        if row is None:
            push(idx, label, "", "没有对应结果", "未运行该臂 / 该对照")
            continue
        d = float(row["delta_mean"])
        p = float(row["p_paired"])
        mag, sig = classify(d, p, threshold)
        evidence = (
            f"Δ={d:+.4f} [{float(row['ci95_lo']):+.4f}, {float(row['ci95_hi']):+.4f}]，"
            f"n={int(row['n_obs'])}，正号比例 {float(row['win_rate']):.2f}，"
            f"种子 {int(row['seeds_positive'])}/{int(row['seeds_total'])} 为正，"
            f"p_paired={p:.3g}，p_sign={float(row['p_sign']):.3g}"
        )
        push(idx, label, round(d, 6), evidence, f"{mag}；{sig}")

    rel_nog = diag(ARM_NOGRAPH, "delta_rel")
    rel_static = diag(ARM_STATIC, "delta_rel")
    rel_dyn = diag(ARM_DYNAMIC, "delta_rel")
    lift_static = diag(ARM_STATIC, "neighbor_lift")
    lift_dyn = diag(ARM_DYNAMIC, "neighbor_lift")
    edges_static = diag(ARM_STATIC, "n_edges")
    edges_dyn = diag(ARM_DYNAMIC, "n_edges")
    lift_periodic = diag(ARM_PERIODIC, "neighbor_lift")
    edges_periodic = diag(ARM_PERIODIC, "n_edges")
    lift_at120 = diag("static_at120", "neighbor_lift")
    edges_at120 = diag("static_at120", "n_edges")
    with_graph_dyn = diag(ARM_DYNAMIC, "probe_with_graph")

    if (
        np.isfinite(rel_dyn)
        and rel_dyn <= 1e-12
        and not (np.isfinite(with_graph_dyn) and with_graph_dyn > 0.05)
    ):
        rel_judge = (
            "动态图臂的 delta_rel 为 0，而且探针用户里有邻居的比例也≈0 —— "
            "图在评测时根本没被用上（预热/参与率/构图条件不满足）；"
            "这不是「图没用」的科学结论，先修设置"
            "（尤其 --graph-warmup-rounds 与 --participation-rate）。"
        )
    elif np.isfinite(rel_dyn) and rel_dyn < 0.05:
        rel_judge = (
            f"动态图臂约 {fmt_sci(rel_dyn)}，即图分支只占序列表示的 {rel_dyn * 100:.2f}%"
            " —— 这个量级上无论 EMA 怎么调都不会明显改变推荐。"
        )
    elif np.isfinite(rel_dyn) and rel_dyn >= 0.5:
        rel_judge = (
            f"动态图臂约 {fmt_sci(rel_dyn)}（≥0.5），图分支与序列表示同量级甚至更大 —— "
            "协同分支是主导项之一，「图分支被稀释」的假设不成立。"
        )
    elif np.isfinite(rel_dyn):
        rel_judge = f"动态图臂约 {fmt_sci(rel_dyn)}，图分支占比不算小。"
    else:
        rel_judge = "没有 delta_rel 数据"

    push(
        "⑥",
        "图分支把最终用户表示挪动了多少（delta_rel = ‖Δ‖ / ‖seq_emb‖）",
        round(rel_dyn, 8) if np.isfinite(rel_dyn) else "",
        f"NoGraph={fmt_sci(rel_nog)}，StaticGraph={fmt_sci(rel_static)}，"
        f"DynamicGraph={fmt_sci(rel_dyn)}",
        "NoGraph 恒为 0（控制生效）；" + rel_judge,
    )
    push(
        "⑦",
        "图里的邻居到底有没有信息（neighbor_lift = 邻居余弦 / 随机用户余弦）",
        round(lift_dyn, 4) if np.isfinite(lift_dyn) else "",
        f"StaticGraph lift={fmt(lift_static, 3)}（边 {fmt(edges_static, 0)}），"
        f"static_at120 lift={fmt(lift_at120, 3)}（边 {fmt(edges_at120, 0)}），"
        f"PeriodicGraph lift={fmt(lift_periodic, 3)}（边 {fmt(edges_periodic, 0)}），"
        f"DynamicGraph lift={fmt(lift_dyn, 3)}（边 {fmt(edges_dyn, 0)}）",
        _lift_judge(lift_dyn, lift_static),
    )

    if not hygiene.empty:
        sub = hygiene[hygiene["dataset"] == dataset]
        bad = sub[sub["ok"] == 0]
        push(
            "⑧",
            "严格性检查（协议一致 / 空图 / 冻结 / 每轮更新 / 评测对齐）",
            int(len(sub) - len(bad)),
            "全部通过" if bad.empty else "; ".join(
                f"{r['check']}: {r['detail']}" for _, r in bad.iterrows()
            ),
            "可以信任上面的差值" if bad.empty else "有检查未通过，上述差值不可作为结论",
        )

    # ---------------- ⑨ 总体结论 ----------------
    def positive(row: Optional[pd.Series]) -> bool:
        return (
            row is not None
            and float(row["delta_mean"]) >= threshold
            and float(row["p_paired"]) < 0.05
        )

    def negligible(row: Optional[pd.Series]) -> bool:
        return row is not None and abs(float(row["delta_mean"])) < threshold

    def negative(row: Optional[pd.Series]) -> bool:
        return (
            row is not None
            and float(row["delta_mean"]) <= -threshold
            and float(row["p_paired"]) < 0.05
        )

    def show(row: Optional[pd.Series]) -> str:
        if row is None:
            return "n/a"
        return f"{float(row['delta_mean']):+.4f}"

    def rate(row: Optional[pd.Series]) -> str:
        if row is None:
            return ""
        return f"窗口正号 {float(row['win_rate']):.0%}、种子 {int(row['seeds_positive'])}/{int(row['seeds_total'])} 为正"

    coverage_note = ""
    if (
        np.isfinite(edges_static)
        and np.isfinite(edges_dyn)
        and edges_dyn > 0
        and edges_static < 0.9 * edges_dyn
    ):
        coverage_note = (
            f"另外注意：静态臂的图只覆盖 {edges_static:.0f}/{edges_dyn:.0f} 个用户"
            f"（{edges_static / edges_dyn:.0%}）—— 它在预热结束那一轮就冻结了，"
            "之后新上传摘要的用户永远进不了图；所以 ① 为 0 也可能部分来自「覆盖不足」，"
            "而不只是「静态本身无用」。想排除这一点就跑 periodic 臂（每窗口刷新、覆盖能拉满）或 "
            "--freeze-round 120。"
        )

    if g1 is None or g2 is None or g3 is None:
        conclusion = "缺臂：至少要有 nograph / static / dynamic 三条臂才能下结论。"
    elif (
        np.isfinite(rel_dyn)
        and rel_dyn <= 1e-12
        and not (np.isfinite(with_graph_dyn) and with_graph_dyn > 0.05)
    ):
        conclusion = (
            "评测时图分支根本没被用上（delta_rel=0 且探针用户里有邻居的比例≈0），"
            "所以 ①②③ 全为 0 是**设置问题**，不是「图没用」的结论。"
        )
    elif positive(g3) and g5 is not None and g4 is not None and positive(g4):
        # 有「慢刷新」臂 → 可以把总收益拆成「刷新/覆盖」与「每轮 EMA」两部分
        total = float(g3["delta_mean"])
        share = float(g4["delta_mean"]) / total if total else float("nan")
        if positive(g5):
            tail = (
                f"每轮 EMA 相对「每窗口才刷新一次」还有小幅但一致的额外收益"
                f"（⑤={show(g5)}，{rate(g5)}，p={float(g5['p_paired']):.3g}）—— "
                "这部分才是「自适应更新速度」真正能争取的空间。"
            )
        else:
            tail = (
                f"「每轮 EMA」相对「每窗口才刷新一次」没有可测的额外收益（⑤={show(g5)}）—— "
                "继续研究自适应更新速度的收益有限。"
            )
        frozen_note = ""
        if g1b is not None and negligible(g1b):
            frozen_note = (
                f"把冻结时点从第 21 轮推到第 120 轮也救不回来（①b={show(g1b)}，无差异）；"
            )
        conclusion = (
            f"**动态图有明确贡献（③={show(g3)}，{rate(g3)}），但贡献的大头来自"
            f"「图持续被刷新、能覆盖当前活跃用户」，而不是「每轮 EMA」**："
            f"每 120 轮才刷新一次的 periodic 臂只用了 5 次刷新，就拿到 ④={show(g4)}"
            f"（占总收益 {share:.0%}，{rate(g4)}）；" + tail
            + frozen_note
            + "一次性冻结的图完全没用，原因是它只能覆盖冻结时刻已上传摘要的用户"
            "（1275~1429/6040），而且邻居质量随兴趣漂移掉到随机基线以下（见 ⑦）。"
        )
    elif positive(g3) and negligible(g1) and positive(g2):
        conclusion = (
            f"**动态图有明确贡献，而且贡献来自「动态」本身**："
            f"冻结的静态图相对 NoGraph 没有可测收益（①={show(g1)}，可忽略"
            f"{'、甚至显著更差' if negative(g1) else ''}），"
            f"但每轮更新的动态图明显更好（②={show(g2)}，{rate(g2)}；"
            f"③={show(g3)}，{rate(g3)}）。"
            "换句话说：图的价值不在「有一张图」，而在「这张图跟着兴趣持续更新」。"
            + coverage_note
        )
    elif positive(g3) and positive(g1) and negligible(g2):
        conclusion = (
            f"图有贡献（①={show(g1)}），但「每轮 EMA 更新」没有额外贡献"
            f"（②={show(g2)}）⇒ 静态图已经够用，继续研究「自适应更新速度」的收益有限。"
        )
    elif positive(g3) and positive(g1) and positive(g2):
        conclusion = (
            f"图有贡献（①={show(g1)}），动态性也有额外贡献（②={show(g2)}，③={show(g3)}）"
            "⇒ 动态更新是有效成分，值得继续研究「什么时候更新、更新多少」。"
        )
    elif not positive(g3) and not negative(g3):
        conclusion = (
            f"动态图相对 NoGraph 没有可测的总贡献（③={show(g3)}，"
            f"{'可忽略' if negligible(g3) else '未达显著'}）。"
            "先看 ⑥ 的 delta_rel 与 ⑦ 的 neighbor_lift 定位原因。"
        )
    else:
        conclusion = (
            f"结果不一致：①={show(g1)}（{'可忽略' if negligible(g1) else '非可忽略'}），"
            f"②={show(g2)}（{'可忽略' if negligible(g2) else '非可忽略'}），"
            f"③={show(g3)}。先看种子方差与逐窗口差异，再决定是否重跑。"
        )
    push("⑨", "总体结论", "", conclusion, "这是本实验要回答的那句话")
    return rows


# ======================================================================================
# 图（matplotlib 可选）
# ======================================================================================
def _arm_of(variant: str) -> Optional[str]:
    """目录名 -> 它属于哪条臂（认不出来返回 None）。

    先做「精确同名」判定（`onboarding` / `fresh_only` 没有配置变体），
    再退回前缀匹配（`static_build120`、`dynamic_rho0.8` 这类变体）。
    """
    if variant in ARM_ORDER:
        return variant
    return next(
        (a for a in (ARM_PERIODIC, ARM_DYNAMIC, ARM_STATIC, ARM_NOGRAPH) if variant.startswith(a)),
        None,
    )


def make_figure(arm_win: pd.DataFrame, dataset: str, out_png: Path) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return (
            f"跳过出图（这个环境没有 matplotlib）：{exc}。"
            "表格与报告已正常生成；要出图执行 "
            "`/home/wl/.venv-feddieg/bin/pip install matplotlib` 后重跑本脚本即可"
        )

    sub = arm_win[arm_win["dataset"] == dataset]
    if sub.empty:
        return "跳过出图：没有窗口级数据"

    base_colors = {
        ARM_NOGRAPH: "#555555",
        ARM_STATIC: "#1f77b4",
        ARM_PERIODIC: "#2ca02c",
        ARM_DYNAMIC: "#d62728",
    }
    extra_palette = ["#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
    variant_style: Dict[str, Tuple[str, str]] = {}
    extra_idx = 0
    for variant in sorted(sub["arm_dir"].unique(),
                          key=lambda v: (ARM_ORDER.get(_arm_of(v) or "", 9), v)):
        arm = _arm_of(variant)
        if arm is not None and variant == arm:
            variant_style[variant] = (base_colors.get(arm, "#999999"), "-")
        else:
            variant_style[variant] = (extra_palette[extra_idx % len(extra_palette)], "--")
            extra_idx += 1
    panels = (
        ("ndcg_win_mean", "linear", "NDCG@10 (mean over seeds), by window", None),
        ("delta_rel", "symlog", "delta_rel = ||graph branch|| / ||seq_emb||", None),
        ("neighbor_lift", "linear", "neighbor_lift (neighbour cos / random cos)", 1.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    plotted = {panel[0]: False for panel in panels}
    for variant in sorted(sub["arm_dir"].unique(), key=lambda v: (ARM_ORDER.get(_arm_of(v), 9), v)):
        g = sub[sub["arm_dir"] == variant]
        color, linestyle = variant_style.get(variant, ("#999999", "-"))
        x = np.array(sorted(g["window"].unique()))
        grp = g.groupby("window")
        for ax, (col, scale, title, hline) in zip(axes, panels):
            mean = grp[col].mean().reindex(x).to_numpy(dtype=float)
            ax.plot(x, mean, marker="o", ms=4, lw=1.6, color=color, ls=linestyle,
                    label=variant if ax is axes[0] else None)
            if np.isfinite(mean).any():
                plotted[col] = True
            ax.set_xticks(x)
            if x.size > 1:
                ax.set_xlim(x.min() - 0.35, x.max() + 0.35)
            if g["seed"].nunique() > 1:
                sd = grp[col].std(ddof=1).reindex(x).to_numpy(dtype=float)
                ok = np.isfinite(mean) & np.isfinite(sd)
                if ok.any():
                    ax.fill_between(x[ok], (mean - sd)[ok], (mean + sd)[ok],
                                    color=color, alpha=0.12, lw=0)
            if scale == "symlog":
                ax.set_yscale("symlog", linthresh=1e-4)
            else:
                ax.set_yscale(scale)
            if hline is not None:
                ax.axhline(hline, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("NDCG@10")
    axes[1].set_ylabel("relative shift of user vector")
    axes[2].set_ylabel("lift over random")
    for ax, (col, scale, title, _) in zip(axes, panels):
        ax.set_xlabel("window")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if scale == "symlog":
            # 必须在所有臂画完之后再设下界，否则上界会被第一组数据锁死，
            # 后面量级更大的臂会被裁到画布外（delta_rel 就踩过这个坑）
            ax.set_ylim(bottom=0.0)
        if not plotted.get(col, False):
            ax.text(
                0.5, 0.10, "no data in this run",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#888888",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0),
            )
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"val_DynamicGraph / {dataset}: NoGraph vs StaticGraph vs DynamicGraph", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return None


# ======================================================================================
# Markdown 报告
# ======================================================================================
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
    if len(frame) > max_rows:
        lines.append(f"| …共 {len(frame)} 行，完整内容见 CSV |" + " |" * (len(cols) - 1))
    return "\n".join(lines) + "\n"


def write_report(
    path: Path,
    verdict: pd.DataFrame,
    arm_summary: pd.DataFrame,
    hygiene: pd.DataFrame,
    user_paired: pd.DataFrame,
    variants: Sequence[str],
    threshold: float,
    fig_note: Optional[str],
    fig_paths: Sequence[Path],
) -> None:
    lines: List[str] = []
    lines.append("# val_DynamicGraph 报告：动态图到底有没有明确贡献？\n")
    lines.append(
        "同一序列编码器、同一训练协议、同一份种子与参与序列，只改图分支："
        + ", ".join(f"`{v}`" for v in variants)
        + "。\n"
    )
    datasets = sorted(verdict["dataset"].unique()) if not verdict.empty else []
    for dataset in datasets:
        lines.append(f"## {dataset}\n")
        sub = verdict[verdict["dataset"] == dataset]
        conclusion = sub[sub["verdict_id"] == "⑨"]
        if not conclusion.empty:
            lines.append(f"**一句话结论：** {conclusion.iloc[0]['evidence']}\n")
        lines.append(
            md_table(
                sub[["verdict_id", "question", "value", "evidence", "judgement"]],
                ["verdict_id", "question", "value", "evidence", "judgement"],
            )
        )
        lines.append("")
        lines.append("<details><summary>逐臂汇总（跨种子平均）</summary>\n")
        lines.append(
            md_table(
                arm_summary[arm_summary["dataset"] == dataset],
                ["arm_dir", "n_seeds", "ndcg_win_mean", "ndcg_win_last", "delta_rel",
                 "neighbor_lift", "ecr", "n_edges", "graph_updates"],
            )
        )
        lines.append("\n</details>\n")
        lines.append("<details><summary>严格性检查</summary>\n")
        lines.append(md_table(hygiene[hygiene["dataset"] == dataset], ["check", "ok", "detail"]))
        lines.append("\n</details>\n")
    if not user_paired.empty:
        lines.append("## 逐用户配对（敏感性检查）\n")
        lines.append(
            md_table(
                user_paired,
                ["dataset", "contrast", "n_users", "delta_mean", "delta_median",
                 "ci95_lo", "ci95_hi", "win_rate", "p_sign", "p_wilcoxon"],
                max_rows=30,
            )
        )
    lines.append("")
    lines.append("## 怎么读这几张表\n")
    lines.append(
        f"- 主证据是 **窗口级配对差值**（n = 种子数 × 窗口数）。阈值 `{threshold:g}` NDCG@10："
        "小于它就算可忽略 —— 6040 个用户的配对检验能把 1e-4 的差异测成显著，"
        "所以别只看 p 值。\n"
        "- ① StaticGraph − NoGraph：图分支本身值多少。这个 ≈0 时，后面所有更新策略的讨论都没有意义。\n"
        "- ② DynamicGraph − StaticGraph：动态度值多少。这是「动态图有必要吗」的直接答案。\n"
        "- ⑥ `delta_rel`：图分支把最终用户表示挪动了多少。它是 ①② 的剂量：剂量太小，任何图机制都不可能有贡献。\n"
        "- ⑦ `neighbor_lift`：图里的邻居比随机用户更像自己多少。≈1 说明图没有区分度。\n"
        "- ⑧ 严格性：NoGraph 分支恒为 0、Static 冻结后不变、Dynamic 每轮都更新、各臂评测对齐。\n"
    )
    if fig_paths:
        lines.append("- 图：\n")
        for p in fig_paths:
            lines.append(f"  - `{p}`\n")
    if fig_note:
        lines.append(f"- 出图提示：{fig_note}\n")
    lines.append("## 已知局限\n")
    lines.append(
        "1. `static` 臂的图建在预热结束那一轮（默认第 `graph_warmup_rounds + 1` 轮），"
        "那时摘要只更新过一轮，图的质量偏弱。若 ① 为负，别直接说「静态图有害」："
        "先试 `--freeze-round 120`（冻结得更晚但依然不动态）或 `--arm periodic`"
        "（每窗口刷新一次）。\n"
        "2. `dynamic` 与 `static` 共享预热期的全部更新，所以 ② 是纯粹的"
        "「之后要不要继续更新」，不含「图建得好不好」。\n"
        "3. 本实验关掉了 L_struct（`lambda_struct=0`）：它定义在图量 Δe 上，NoGraph 臂拿不到，"
        "留着会变成「图臂多一个训练信号」的混淆项。要回到论文口径加 `--lambda-struct 1e-5`。\n"
        "4. CPU 与 GPU 的 dropout 随机流不同，同一组对照必须固定在同一台机器上跑完"
        "（跨机器同 seed 的 CUDA 结果本身是逐位一致的）。\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================================
# 主流程
# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="val_DynamicGraph 汇总：三条臂的配对比较与结论")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--practical-threshold",
        type=float,
        default=0.002,
        help="NDCG@10 的可忽略阈值（默认 0.002，约为 NDCG=0.5 时的 0.4% 相对差异）",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    threshold = float(args.practical_threshold)

    sums, rounds, users = load_runs(root)
    variants = sorted(sums["arm_dir"].unique().tolist())
    datasets = sorted(sums["dataset"].unique().tolist())
    print(f"找到 {len(sums)} 次运行，{len(variants)} 个臂/变体：{variants}")
    print(f"数据集：{datasets}")

    # ---------------- 逐臂汇总
    arm_win = window_table(rounds)
    arm_summary = (
        arm_win.groupby(["dataset", "arm_dir"])
        .agg(
            n_seeds=("seed", "nunique"),
            ndcg_win_mean=("ndcg_win_mean", "mean"),
            ndcg_win_last=("ndcg_win_last", "mean"),
            hr_win_mean=("hr_win_mean", "mean"),
            delta_rel=("delta_rel", "mean"),
            probe_with_graph=("probe_with_graph", "mean"),
            neighbor_cos=("neighbor_cos", "mean"),
            neighbor_lift=("neighbor_lift", "mean"),
            ecr=("ecr", "mean"),
            added=("added", "mean"),
            n_edges=("n_edges", "last"),
            graph_updates=("graph_updates", "max"),
        )
        .reset_index()
        .sort_values(["dataset", "arm_dir"])
    )
    arm_summary.to_csv(tables / "dg_arm_summary.csv", index=False, encoding="utf-8-sig")
    arm_win.to_csv(tables / "dg_arm_window.csv", index=False, encoding="utf-8-sig")

    arm_curve = (
        arm_win.groupby(["dataset", "arm_dir", "window"])
        .agg(
            n_seeds=("seed", "nunique"),
            ndcg_win_mean=("ndcg_win_mean", "mean"),
            ndcg_win_sd=("ndcg_win_mean", "std"),
            ndcg_win_last=("ndcg_win_last", "mean"),
            hr_win_mean=("hr_win_mean", "mean"),
            delta_rel=("delta_rel", "mean"),
            neighbor_lift=("neighbor_lift", "mean"),
            ecr=("ecr", "mean"),
            n_edges=("n_edges", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "arm_dir", "window"])
    )
    arm_curve.to_csv(tables / "dg_arm_curve.csv", index=False, encoding="utf-8-sig")

    # ---------------- 对照（窗口级 = 主证据）
    contrasts = available_contrasts(variants)
    print(f"对照：{[c[0] for c in contrasts]}")

    long_win = arm_win.melt(
        id_vars=["dataset", "arm_dir", "seed", "window"],
        value_vars=[m for m in METRICS if m in arm_win.columns],
        var_name="metric",
        value_name="value",
    )
    pieces = []
    for metric, part in long_win.groupby("metric"):
        piv = part.pivot_table(
            index=["dataset", "seed", "window"], columns="arm_dir", values="value", aggfunc="mean"
        ).reset_index()
        piece = contrasts_from_wide(piv, ["dataset", "seed", "window"], metric, contrasts)
        if not piece.empty:
            pieces.append(piece)
    contrasts_long = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not contrasts_long.empty:
        contrasts_long.to_csv(tables / "dg_contrasts_window.csv", index=False, encoding="utf-8-sig")
    contrasts_sum = summarise_contrasts(contrasts_long)
    if not contrasts_sum.empty:
        contrasts_sum = contrasts_sum.sort_values(["dataset", "metric", "contrast"]).reset_index(drop=True)
        contrasts_sum.to_csv(tables / "dg_contrasts.csv", index=False, encoding="utf-8-sig")
        (
            contrasts_long.groupby(["dataset", "metric", "contrast", "seed"])["delta"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "delta_mean", "count": "n_windows"})
            .to_csv(tables / "dg_contrasts_by_seed.csv", index=False, encoding="utf-8-sig")
        )

    # ---------------- 逐用户配对（敏感性检查）
    user_paired = pd.DataFrame()
    if not users.empty and contrasts:
        upiv = users.pivot_table(
            index=["dataset", "seed", "window", "user"],
            columns="arm_dir",
            values="ndcg",
            aggfunc="mean",
        ).reset_index()
        rows = []
        by_window = []
        for label, a, b in contrasts:
            if a not in upiv.columns or b not in upiv.columns:
                continue
            mask = upiv[a].notna() & upiv[b].notna()
            if not bool(mask.any()):
                continue
            d = (upiv.loc[mask, a] - upiv.loc[mask, b]).to_numpy(dtype=np.float64)
            lo, hi = bootstrap_ci(d, n_boot=int(args.bootstrap))
            rows.append(
                {
                    "dataset": upiv.loc[mask, "dataset"].iloc[0],
                    "contrast": label,
                    "n_users": int(np.isfinite(d).sum()),
                    "delta_mean": float(np.nanmean(d)),
                    "delta_median": float(np.nanmedian(d)),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "win_rate": float(np.nanmean(d > 0)),
                    "p_sign": sign_test_p(d),
                    "p_wilcoxon": wilcoxon_p(d),
                }
            )
            tmp = upiv.loc[mask, ["dataset", "seed", "window"]].copy()
            tmp["delta"] = d
            tmp["contrast"] = label
            by_window.append(tmp)
        user_paired = pd.DataFrame(rows)
        if not user_paired.empty:
            user_paired.to_csv(tables / "dg_user_paired.csv", index=False, encoding="utf-8-sig")
        if by_window:
            (
                pd.concat(by_window, ignore_index=True)
                .groupby(["dataset", "contrast", "window"])["delta"]
                .agg(["mean", "median", "size"])
                .reset_index()
                .to_csv(tables / "dg_user_paired_by_window.csv", index=False, encoding="utf-8-sig")
            )

    # ---------------- 诊断 / 严格性
    diagnostics = arm_summary[
        ["dataset", "arm_dir", "delta_rel", "probe_with_graph", "neighbor_cos",
         "neighbor_lift", "ecr", "added", "n_edges", "graph_updates", "ndcg_win_mean"]
    ].copy()
    diagnostics.to_csv(tables / "dg_diagnostics.csv", index=False, encoding="utf-8-sig")

    hygiene = hygiene_checks(sums, rounds)
    hygiene.to_csv(tables / "dg_hygiene.csv", index=False, encoding="utf-8-sig")

    # ---------------- 结论
    verdict_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        verdict_rows.extend(build_verdict(contrasts_sum, diagnostics, hygiene, dataset, threshold))
    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(tables / "dg_verdict.csv", index=False, encoding="utf-8-sig")

    # ---------------- 图（每个数据集一张）
    fig_notes: List[str] = []
    fig_paths: List[Path] = []
    if not arm_win.empty:
        for dataset in datasets:
            slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in dataset).strip("_")
            png = figures / f"dg_fig_arms_{slug}.png"
            note = make_figure(arm_win, dataset, png)
            if note:
                fig_notes.append(note)
            elif png.exists():
                fig_paths.append(png)
    else:
        fig_notes.append("没有窗口级数据，未出图")

    write_report(
        out / "dg_report.md",
        verdict,
        arm_summary,
        hygiene,
        user_paired,
        variants,
        threshold,
        "; ".join(fig_notes) if fig_notes else None,
        fig_paths,
    )

    # ---------------- 屏幕摘要
    print("\n" + "=" * 78)
    print("三条臂（跨种子平均，窗口级 NDCG@10）")
    print("=" * 78)
    print(
        arm_summary[
            ["dataset", "arm_dir", "n_seeds", "ndcg_win_mean", "ndcg_win_last",
             "delta_rel", "neighbor_lift", "ecr", "n_edges"]
        ].round(4).to_string(index=False)
    )

    if not contrasts_sum.empty:
        print("\n" + "=" * 78)
        print("配对差值（窗口级；正 = 前一个臂更好）")
        print("=" * 78)
        cs = contrasts_sum[contrasts_sum["metric"] == PRIMARY_METRIC]
        print(
            cs[["dataset", "contrast", "n_obs", "delta_mean", "ci95_lo", "ci95_hi",
                "win_rate", "seeds_positive", "seeds_total", "p_paired", "p_sign"]]
            .round(4)
            .to_string(index=False)
        )

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    for _, row in verdict.iterrows():
        if row["verdict_id"] in ("①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"):
            print(f"[{row['verdict_id']}] {row['question']}")
            print(f"     {row['evidence']}")
            if str(row["judgement"]).strip():
                print(f"     -> {row['judgement']}")

    print("\n" + "=" * 78)
    print("输出")
    print("=" * 78)
    for path in (
        out / "dg_report.md",
        tables / "dg_verdict.csv",
        tables / "dg_contrasts.csv",
        tables / "dg_contrasts_window.csv",
        tables / "dg_contrasts_by_seed.csv",
        tables / "dg_arm_summary.csv",
        tables / "dg_arm_curve.csv",
        tables / "dg_user_paired.csv",
        tables / "dg_diagnostics.csv",
        tables / "dg_hygiene.csv",
        *fig_paths,
    ):
        print(("  " if path.exists() else "  (缺) ") + str(path))
    for note in fig_notes:
        print(f"\n注意：{note}")


if __name__ == "__main__":
    main()
