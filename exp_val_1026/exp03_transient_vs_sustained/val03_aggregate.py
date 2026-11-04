#!/usr/bin/env python
"""验证实验 3（val03）第 3 步：聚合出图 + 出表 + 自动结论。

对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验三」。

主指标是什么，为什么
-------------------
任务书要的「图变化率」有两种读法，本脚本两个都给：
  * `turnover_from_pre`：当前邻居集合与 onset 前快照的 1 − Jaccard。**无方向的「变动量」。**
    注意它在 ρ 小时会饱和（MovieLens 有 6000 个可连接用户、只留 Top-10 邻居，
    一批证据回来就能把 10 个邻居全换掉）。实测 ρ=0 时它已经≈1.0，
    所以它只能区分「图在不在动」（ρ 大小），不能区分「动得对不对」。
  * `old_peer_share` / `new_peer_share`：邻居里「仍是旧兴趣同类 / 已经是新兴趣同类」的边权占比。
    **有方向的关系质量**，这才是「图是否该变时变、该撤销时撤销」的判据，因此作为主指标。

过度更新 / 更新滞后的定义（成对差值口径，减去配对对照）
------------------------------------------------------
  excess(m, r) = inj(m) − pair(m)，同一 pair_id、同一 rel_round（严格配对）
  attack_during    = −mean{ excess(old_peer_share), rel_round ∈ [1, R] }     >0 = 扰动把图从旧兴趣同伴拉开
  residual_post    = −mean{ excess(old_peer_share), rel_round ∈ [R+1, 3R] } >0 = 撤销之后仍然偏离
  over_update      = residual_post / attack_during
                     ≈1 = 图改了就回不来（把一次性噪声当成持久变化）；≈0 = 完全撤销
  tracked_sustained= −mean{ excess(old_peer_share), rel_round ∈ [R+1, 3R] }（sustained 场景）
  under_update     = 1 − tracked_sustained(ρ) / tracked_sustained(ρ=0)
                     0 = 完全跟上持续变化；1 = 完全不更新（更新滞后）
另外用 `summary_drift`（上传摘要相对 onset 前的 1−cos）区分「摘要本身没恢复」与「图没恢复」。

★ 任务书特别提醒：**图变化率越低不一定越好**。所以结论不是「谁 turnover 小」，
  而是同时看 over_update（该撤销没撤销）与 under_update（该变没变）两个方向。

输出（默认 `results/aggregated/`）
---------------------------------
  figures/val03_fig_arm_synthetic.png  ★ 主图：注入 arm 的关系质量 + NDCG（sustained vs transient）
  figures/val03_fig_arm_real.png       ★ 真实数据 arm：real_sustained vs real_transient vs real_stable
  figures/val03_fig_pair_excess.png    ★ 净效应图（注入组 − 配对对照），结论就看这张最干净
  figures/val03_fig_rho_tradeoff.png   ★ 关键图：不存在同时「不过度更新」与「不滞后」的 ρ
  figures/val03_fig_rho_curves.png     按 ρ 展开的关系质量曲线
  tables/val03_curve_<metric>.csv      各指标曲线（含 n_users 与用户级 SEM）
  tables/val03_pair_excess.csv         成对差值（净效应）
  tables/val03_window_summary.csv      每个 (数据集, 场景, 组, ρ) 的分窗口汇总
  tables/val03_tradeoff.csv            过度更新 vs 更新滞后（每个 ρ 一个点）
  tables/val03_lag_turnover.csv        邻居 turnover 达到 50% 变化所需轮数
  tables/val03_verdict.csv             自动结论判定（逐数据集）
  val03_report.md                      已填好数字的报告骨架

用法
----
  python exp_val_1026/exp03_transient_vs_sustained/val03_aggregate.py
  python .../val03_aggregate.py --root results/raw_warm --out results/aggregated_warm
  python .../val03_aggregate.py --datasets "movielens-1m"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VAL03_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = VAL03_DIR / "results" / "raw"
DEFAULT_OUT = VAL03_DIR / "results" / "aggregated"
DEFAULT_PLAN_ROOT = VAL03_DIR / "results" / "scenarios"

GROUP_LABELS = {
    "inj": "injected users (this scenario)",
    "pair": "paired control (same-category resample)",
    "real_sustained": "real data: sustained change",
    "real_transient": "real data: transient fluctuation",
    "real_stable": "real data: matched stable control",
}
GROUP_COLORS = {
    "inj": "#d95f02",
    "pair": "#999999",
    "real_sustained": "#1b9e77",
    "real_transient": "#7570b3",
    "real_stable": "#bbbbbb",
}
SCENARIO_LABELS = {
    "sustained": "sustained interest change",
    "transient": "transient fluctuation (reverted)",
}
SCENARIO_COLORS = {"sustained": "#1b9e77", "transient": "#d95f02"}
RHO_COLORS = {0.0: "#1b9e77", 0.30: "#66a61e", 0.50: "#d95f02", 0.80: "#7570b3", 0.95: "#e7298a"}

USER_METRICS = ("old_peer_share", "new_peer_share", "turnover_from_pre", "ndcg", "hr", "summary_drift", "ecr")
PRIMARY_METRIC = "old_peer_share"     # 主指标：有方向的关系质量
NDCG_NOISE_FLOOR = 0.002              # 与实验一/二一致的 NDCG 噪声地板
PEER_NOISE_FLOOR = 0.02               # old_peer_share 的绝对噪声地板
TURNOVER_NOISE_FLOOR = 0.02
EPS = 1e-6


# ======================================================================================
# 读取
# ======================================================================================
def load_all(root: Path, datasets: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_frames: List[pd.DataFrame] = []
    round_frames: List[pd.DataFrame] = []
    for path in sorted(root.rglob("val03_user_round_metrics.csv")):
        if datasets and not any(ds in str(path) for ds in datasets):
            continue
        try:
            user_frames.append(pd.read_csv(path))
        except Exception:
            continue
    for path in sorted(root.rglob("val03_round_metrics.csv")):
        if datasets and not any(ds in str(path) for ds in datasets):
            continue
        try:
            round_frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not user_frames:
        raise SystemExit(f"{root} 下没有找到 val03_user_round_metrics.csv，先跑训练。")
    user = pd.concat(user_frames, ignore_index=True)
    rnd = pd.concat(round_frames, ignore_index=True) if round_frames else pd.DataFrame()
    return user, rnd


def detect_window_rounds(root: Path, default: int = 120) -> int:
    counts: Dict[int, int] = {}
    for path in sorted(root.rglob("val03_summary.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "rounds_per_window" not in frame.columns or frame.empty:
            continue
        value = int(frame["rounds_per_window"].iloc[0])
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return default
    return max(counts.items(), key=lambda item: item[1])[0]


# ======================================================================================
# 曲线
# ======================================================================================
def build_curve(user: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in user.columns:
        return pd.DataFrame()
    frame = user[np.isfinite(user[metric].astype(float))].copy()
    if frame.empty:
        return pd.DataFrame()
    per_user = frame.groupby(
        ["dataset", "scenario", "group", "rho", "seed", "rel_round", "user"], as_index=False
    )[metric].mean()
    grouped = per_user.groupby(["dataset", "scenario", "group", "rho", "rel_round"])[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["sem_user"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out = out.rename(columns={"count": "n_users"})
    out["metric"] = metric
    return out


def build_pair_excess(user: pd.DataFrame, metric: str) -> pd.DataFrame:
    """注入组 − 配对对照 的成对差值（同一 pair_id、同一 rel_round）。"""
    if metric not in user.columns:
        return pd.DataFrame()
    frame = user[user["group"].isin(["inj", "pair"])].copy()
    frame = frame[np.isfinite(frame[metric].astype(float))]
    if frame.empty:
        return pd.DataFrame()
    wide = frame.pivot_table(
        index=["dataset", "scenario", "rho", "seed", "pair_id", "rel_round"],
        columns="group",
        values=metric,
        aggfunc="mean",
    ).reset_index()
    if "inj" not in wide.columns or "pair" not in wide.columns:
        return pd.DataFrame()
    wide["excess"] = wide["inj"] - wide["pair"]
    grouped = wide.groupby(["dataset", "scenario", "rho", "rel_round"])["excess"]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["sem_pair"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["metric"] = metric
    # group 列统一记为 "inj"：含义是"注入组的净效应（inj − pair）"，
    # 这样它可以直接套用 summarise_curve / _pick 那套 (dataset, scenario, group, rho) 分组逻辑。
    out["group"] = "inj"
    return out


# ======================================================================================
# 分窗口汇总
# ======================================================================================
def window_ranges(rounds: int) -> Dict[str, Tuple[int, int]]:
    return {
        "pre": (-rounds, 0),
        "during": (1, rounds),
        "post1": (rounds + 1, 2 * rounds),
        "post2": (2 * rounds + 1, 3 * rounds),
        "post": (rounds + 1, 3 * rounds),
    }


def _slice(curve: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return curve[(curve["rel_round"] >= lo) & (curve["rel_round"] <= hi)]


def _weighted_level(part: pd.DataFrame, column: str = "mean") -> Tuple[float, float, int]:
    if part.empty:
        return float("nan"), float("nan"), 0
    weights = part["n_users"].to_numpy(dtype=np.float64) if "n_users" in part else np.ones(len(part))
    values = part[column].to_numpy(dtype=np.float64)
    err_column = "sem_user" if "sem_user" in part else "sem_pair"
    sem = part[err_column].fillna(0.0).to_numpy(dtype=np.float64)
    if weights.sum() <= 0:
        return float("nan"), float("nan"), 0
    level = float((values * weights).sum() / weights.sum())
    err = float(np.sqrt(((sem ** 2) * weights).sum()) / weights.sum())
    return level, err, int(np.max(weights))


def summarise_curve(curve: pd.DataFrame, metric: str, rounds: int) -> pd.DataFrame:
    """对一条曲线（可能是原始曲线，也可能是成对差值曲线）做分窗口汇总。"""
    if curve.empty:
        return pd.DataFrame()
    sub = curve[curve["metric"] == metric] if "metric" in curve.columns else curve
    if sub.empty:
        return pd.DataFrame()
    ranges = window_ranges(rounds)
    rows: List[Dict[str, object]] = []
    for (dataset, scenario, group, rho), part in sub.groupby(["dataset", "scenario", "group", "rho"]):
        record: Dict[str, object] = {
            "dataset": dataset,
            "scenario": scenario,
            "group": group,
            "rho": float(rho),
            "metric": metric,
        }
        for name, (lo, hi) in ranges.items():
            level, err, n_users = _weighted_level(_slice(part, lo, hi))
            record[f"{name}_level"] = level
            record[f"{name}_sem"] = err
            record[f"{name}_n_users"] = n_users
        max_rel = int(part["rel_round"].max())
        last_lo = max(1, max_rel - rounds + 1)
        level, err, n_users = _weighted_level(_slice(part, last_lo, max_rel))
        record["plateau_level"] = level
        record["plateau_sem"] = err
        record["plateau_n_users"] = n_users
        record["max_rel_round"] = max_rel
        if np.isfinite(record["pre_level"]):
            for name in ("during", "post1", "post2", "post", "plateau"):
                if np.isfinite(record.get(f"{name}_level", float("nan"))):
                    record[f"delta_{name}"] = record[f"{name}_level"] - record["pre_level"]
                else:
                    record[f"delta_{name}"] = float("nan")
        rows.append(record)
    return pd.DataFrame(rows)


def compute_lag_50(curve: pd.DataFrame, metric: str, rounds: int) -> pd.DataFrame:
    """从 rel_round=0 走多少轮才达到「最大变动量的 50%」。"""
    if curve.empty:
        return pd.DataFrame()
    sub = curve[curve["metric"] == metric]
    if sub.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for (dataset, scenario, group, rho), part in sub.groupby(["dataset", "scenario", "group", "rho"]):
        base, _, _ = _weighted_level(_slice(part, -rounds, 0))
        after = part[part["rel_round"] > 0].sort_values("rel_round")
        if after.empty or not np.isfinite(base):
            continue
        delta = after["mean"].to_numpy(dtype=np.float64) - base
        if delta.size == 0 or not np.isfinite(delta).any():
            continue
        peak = float(np.nanmax(np.abs(delta)))
        if peak < EPS:
            lag = float("nan")
        else:
            reached = after["rel_round"].to_numpy()[np.abs(delta) >= 0.5 * peak]
            lag = float(reached.min()) if reached.size else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "group": group,
                "rho": float(rho),
                "metric": metric,
                "pre_level": base,
                "peak_abs_delta": peak,
                "lag_reach_50": lag,
            }
        )
    return pd.DataFrame(rows)


# ======================================================================================
# 过度更新 / 更新滞后
# ======================================================================================
def _pick(
    summary: pd.DataFrame, dataset: str, scenario: str, group: str, rho: float, column: str
) -> float:
    if summary.empty:
        return float("nan")
    hit = summary[
        (summary["dataset"] == dataset)
        & (summary["scenario"] == scenario)
        & (summary["group"] == group)
        & (np.isclose(summary["rho"].astype(float), float(rho)))
    ]
    if hit.empty:
        return float("nan")
    return float(hit[column].iloc[0])


def build_tradeoff(
    excess_peer: pd.DataFrame,
    excess_turn: pd.DataFrame,
    excess_drift: pd.DataFrame,
    peer_summary: pd.DataFrame,
    datasets: Sequence[str],
    rhos: Sequence[float],
) -> pd.DataFrame:
    """每个 (数据集, ρ) 一个点：过度更新（横轴）vs 更新滞后（纵轴）。

    全部用成对差值（inj − pair）：
      old_peer_share 下降 = 邻居从「旧兴趣同伴」被换走，所以取负号的增量记为「改动量」。
    """
    rows: List[Dict[str, object]] = []
    for dataset in datasets:

        def pick_delta(frame: pd.DataFrame, scenario: str, rho: float, window: str) -> float:
            """优先用"相对变化前基线"的增量；没有基线时（例如 summary_drift 的 pre 为 nan）退回原始水平。"""
            for column in (f"delta_{window}", f"{window}_level"):
                if column in frame.columns:
                    value = _pick(frame, dataset, scenario, "inj", rho, column)
                    if np.isfinite(value):
                        return value
            return float("nan")

        # 用"相对变化前基线"的增量（delta = level − pre_level），否则 onset 之前的基线偏差
        # （19 对用户、每个窗口只有几个评测点时会有一点点噪声）会被误读成"撤销后的残留"。
        base_track = -pick_delta(excess_peer, "sustained", 0.0, "post")
        for rho in rhos:
            attack_trans = -pick_delta(excess_peer, "transient", float(rho), "during")
            # ★ 撤销后要分两段看：
            #   post1 = 撤销发生的那个窗口（"还赖着不走"= 迟滞）
            #   post2 = 再下一个窗口（"永久残留"= 真正的过度更新）
            residual_post1 = -pick_delta(excess_peer, "transient", float(rho), "post1")
            residual_post2 = -pick_delta(excess_peer, "transient", float(rho), "post2")
            residual_trans = -pick_delta(excess_peer, "transient", float(rho), "post")
            tracked_sust = -pick_delta(excess_peer, "sustained", float(rho), "post")
            tracked_sust_post2 = -pick_delta(excess_peer, "sustained", float(rho), "post2")

            def _ratio(num: float, den: float) -> float:
                return num / den if np.isfinite(den) and den > PEER_NOISE_FLOOR and np.isfinite(num) else float("nan")

            over_early = _ratio(residual_post1, attack_trans)
            over = _ratio(residual_post2, attack_trans)      # 主判定用它（是否永久残留）
            under = (
                1.0 - max(tracked_sust, 0.0) / base_track
                if np.isfinite(base_track) and base_track > PEER_NOISE_FLOOR and np.isfinite(tracked_sust)
                else float("nan")
            )
            # 摘要层面的对照：撤销后摘要是否恢复了（用 1−cos 的成对差值；pre 无快照时退回原始水平）
            drift_during = pick_delta(excess_drift, "transient", float(rho), "during")
            drift_post = pick_delta(excess_drift, "transient", float(rho), "post")
            drift_ratio = (
                drift_post / drift_during
                if np.isfinite(drift_during) and abs(drift_during) > EPS
                else float("nan")
            )
            rows.append(
                {
                    "dataset": dataset,
                    "rho": float(rho),
                    "over_update": over,
                    "over_update_early": over_early,
                    "under_update": under,
                    "transient_attack_during": attack_trans,
                    "transient_residual_post1": residual_post1,
                    "transient_residual_post2": residual_post2,
                    "transient_residual_post": residual_trans,
                    "sustained_tracked_post": tracked_sust,
                    "sustained_tracked_post2": tracked_sust_post2,
                    "sustained_tracked_at_rho0": base_track,
                    "summary_drift_excess_during": drift_during,
                    "summary_drift_excess_post": drift_post,
                    "summary_residual_ratio": drift_ratio,
                    "turnover_excess_during_transient": _pick(
                        excess_turn, dataset, "transient", "inj", float(rho), "during_level"
                    ),
                    "turnover_excess_post_transient": _pick(
                        excess_turn, dataset, "transient", "inj", float(rho), "post_level"
                    ),
                    "raw_old_peer_share_inj_sustained_post": _pick(
                        peer_summary, dataset, "sustained", "inj", float(rho), "post_level"
                    ),
                    "raw_old_peer_share_pair_sustained_post": _pick(
                        peer_summary, dataset, "sustained", "pair", float(rho), "post_level"
                    ),
                    "raw_old_peer_share_inj_transient_during": _pick(
                        peer_summary, dataset, "transient", "inj", float(rho), "during_level"
                    ),
                    "raw_old_peer_share_inj_transient_post": _pick(
                        peer_summary, dataset, "transient", "inj", float(rho), "post_level"
                    ),
                }
            )
    return pd.DataFrame(rows)


# ======================================================================================
# 判定
# ======================================================================================
def build_verdict(
    tradeoff: pd.DataFrame,
    ndcg_summary: pd.DataFrame,
    peer_summary: pd.DataFrame,
    datasets: Sequence[str],
    reference_rho: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(dataset: str, check: str, holds: bool, detail: str) -> None:
        rows.append(
            {"dataset": dataset, "check": check, "holds": "yes" if holds else "no", "detail": detail}
        )

    for dataset in datasets:
        part = tradeoff[(tradeoff["dataset"] == dataset) & np.isclose(tradeoff["rho"], reference_rho)]
        attack = float(part["transient_attack_during"].iloc[0]) if not part.empty else float("nan")
        over = float(part["over_update"].iloc[0]) if not part.empty else float("nan")
        under = float(part["under_update"].iloc[0]) if not part.empty else float("nan")
        tracked = float(part["sustained_tracked_post"].iloc[0]) if not part.empty else float("nan")
        drift_ratio = float(part["summary_residual_ratio"].iloc[0]) if not part.empty else float("nan")

        # ① 扰动是否真的进了图
        add(
            dataset,
            "① 短期扰动确实改动了图（净 old-peer 占比下降 > 噪声地板）",
            bool(np.isfinite(attack) and attack > PEER_NOISE_FLOOR),
            f"ρ={reference_rho:.2f} 扰动窗口净下降 attack={attack:.3f}"
            f"（噪声地板 {PEER_NOISE_FLOOR}）；同一口径下持续变化的跟进量={tracked:.3f}",
        )

        # ② 撤销后图是否回退（过度更新）。分两段：
        #    撤销窗口（迟滞，一个窗口内还赖着不走）与再下一个窗口（永久残留）。
        over_early = float(part["over_update_early"].iloc[0]) if not part.empty else float("nan")
        if not np.isfinite(over):
            add(
                dataset,
                "② 短期扰动撤销后图能回退（过度更新不成立）",
                True,
                f"ρ={reference_rho:.2f} 扰动期净下降低于噪声地板，图这次几乎没被改动"
                "（这是「图没动」，不能算「能撤销」，需与 ③ 一起读）",
            )
        else:
            add(
                dataset,
                "② 短期扰动撤销后图能回退（过度更新不成立）",
                bool(over < 0.5),
                f"ρ={reference_rho:.2f} 撤销窗口残留 over_update_early={over_early:.2f}、"
                f"再下一个窗口残留 over_update={over:.2f}"
                f"（后者 = 永久残留比例，<0.5 视为最终能回退；前者 >0.5 说明至少要晚一个窗口）",
            )

        # ③ 持续变化是否被跟上（更新滞后）
        add(
            dataset,
            "③ 持续变化被跟上（更新滞后不成立）",
            bool(np.isfinite(under) and under <= 0.5),
            f"ρ={reference_rho:.2f} tracked={tracked:.3f}，"
            f"相当于 ρ=0 的 {(1 - under) * 100:.0f}%"
            f"（<50% 视为明显滞后）"
            if np.isfinite(under)
            else f"ρ={reference_rho:.2f} 无可比参照（ρ=0 的跟进量太低）",
        )

        # ④ 是否存在两全的固定 ρ
        candidates = tradeoff[tradeoff["dataset"] == dataset]
        ok = candidates[(candidates["over_update"] <= 0.5) & (candidates["under_update"] <= 0.5)]
        measurable = candidates[np.isfinite(candidates["under_update"]) & np.isfinite(candidates["over_update"])]
        if ok.empty:
            if measurable.empty:
                detail = (
                    "该数据集上无法判定：over_update / under_update 全部为 nan"
                    "（关系质量指标没有分辨力，通常发生在极稀疏数据集上）"
                )
            else:
                detail = (
                    "没有任何 ρ 同时满足两个条件 → 固定 EMA 存在结构性权衡；"
                    "可判定的 ρ 有 " + ", ".join(f"{r:.2f}" for r in measurable["rho"].tolist())
                )
        else:
            names = ", ".join(f"{r:.2f}" for r in ok["rho"].tolist())
            detail = f"满足 over_update≤0.5 且 under_update≤0.5 的 ρ: {names}"
            if bool((np.isclose(ok["rho"].astype(float), 0.0)).any()):
                detail += (
                    "。★ 但里面包含 ρ=0：这等价于「直接关掉 EMA」，正是任务书第 8 节停止信号 2 的情形，"
                    "报告里必须同时讨论它的代价（每轮高频改边带来的抖动），不能只写「ρ=0 够用」"
                )
            else:
                detail += "。★ 存在非零 ρ 两全 → 支持任务书停止信号 2，自适应更新的必要性减弱"
        add(dataset, "④ 存在同时不过度更新、也不明显滞后的固定 ρ", bool(not ok.empty), detail)

        # ⑤ 责任在哪一层：摘要还是图
        add(
            dataset,
            "⑤ 撤销后「摘要恢复但图没恢复」（说明责任在图更新规则）",
            bool(
                np.isfinite(drift_ratio)
                and np.isfinite(over)
                and drift_ratio < 0.5
                and over >= 0.5
            ),
            f"ρ={reference_rho:.2f} 摘要残余比={drift_ratio:.2f}"
            f"（<0.5 = 摘要基本恢复）、图残余比 over_update={over:.2f}"
            f"（>=0.5 = 图没恢复）。两者一起看才能判断是摘要记忆还是 EMA 边权的问题。",
        )

        # ⑥ NDCG 代价（撤销之后的窗口，测试目标同为群众真实物品，可比）
        gap_post_inj = _pick(ndcg_summary, dataset, "transient", "inj", reference_rho, "delta_post")
        gap_post_pair = _pick(ndcg_summary, dataset, "transient", "pair", reference_rho, "delta_post")
        did_post = (
            gap_post_inj - gap_post_pair
            if np.isfinite(gap_post_inj) and np.isfinite(gap_post_pair)
            else float("nan")
        )
        sust_inj = _pick(ndcg_summary, dataset, "sustained", "inj", reference_rho, "delta_post")
        sust_pair = _pick(ndcg_summary, dataset, "sustained", "pair", reference_rho, "delta_post")
        did_sust = (
            sust_inj - sust_pair if np.isfinite(sust_inj) and np.isfinite(sust_pair) else float("nan")
        )
        add(
            dataset,
            "⑥ NDCG 上出现可观测代价（DID > 噪声地板）",
            bool(
                (np.isfinite(did_post) and did_post < -NDCG_NOISE_FLOOR)
                or (np.isfinite(did_sust) and did_sust < -NDCG_NOISE_FLOOR)
            ),
            f"ρ={reference_rho:.2f} transient 的 ΔNDCG DID={did_post:+.4f}、"
            f"sustained 的 ΔNDCG DID={did_sust:+.4f}（|DID| > {NDCG_NOISE_FLOOR} 才有代价）",
        )

        # ⑦ 真实 arm（描述性）
        for group, scenario in (("real_sustained", "sustained"), ("real_transient", "transient")):
            during = _pick(peer_summary, dataset, scenario, group, reference_rho, "during_level")
            post = _pick(peer_summary, dataset, scenario, group, reference_rho, "post_level")
            pre = _pick(peer_summary, dataset, scenario, group, reference_rho, "pre_level")
            add(
                dataset,
                f"⑦ 真实 {group} 的关系质量变化（描述性）",
                bool(np.isfinite(post) and np.isfinite(pre) and abs(post - pre) > PEER_NOISE_FLOOR),
                f"ρ={reference_rho:.2f} old-peer 占比 {pre:.3f}（变化前）→ {during:.3f}（变化窗口）"
                f"→ {post:.3f}（之后）",
            )
    return pd.DataFrame(rows)


# ======================================================================================
# 图
# ======================================================================================
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _rho_of(frame: pd.DataFrame) -> List[float]:
    return sorted({round(float(r), 2) for r in frame["rho"].unique()})


def _plot_series(ax, part: pd.DataFrame, color: str, style: str, width: float, alpha: float, label: str) -> None:
    ax.plot(
        part["rel_round"],
        part["mean"],
        color=color,
        linestyle=style,
        linewidth=width,
        alpha=alpha,
        label=label,
    )
    sem = part["sem_user"].fillna(0.0).to_numpy()
    ax.fill_between(
        part["rel_round"].to_numpy(),
        part["mean"].to_numpy() - sem,
        part["mean"].to_numpy() + sem,
        color=color,
        alpha=0.12 if alpha >= 1.0 else 0.06,
        linewidth=0,
    )


def make_arm_figure(
    curves: Dict[str, pd.DataFrame],
    metric_specs: Sequence[Tuple[str, str]],
    out_path: Path,
    groups: Sequence[str],
    title: str,
    reference_rho: float,
) -> None:
    """行 = (指标 × 场景)，列 = 数据集。"""
    usable: Dict[str, pd.DataFrame] = {}
    for metric, _ in metric_specs:
        frame = curves.get(metric, pd.DataFrame())
        if frame.empty:
            continue
        sub = frame[np.isclose(frame["rho"].astype(float), reference_rho)]
        if not sub.empty:
            usable[metric] = sub
    if not usable:
        print(f"[figure] ρ={reference_rho} 没有可用曲线，跳过 {out_path.name}")
        return
    try:
        plt = _plt()
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return

    any_curve = next(iter(usable.values()))
    datasets = list(dict.fromkeys(any_curve["dataset"]))
    scenarios = [s for s in ("sustained", "transient") if s in set(any_curve["scenario"])]
    if not scenarios:
        scenarios = list(dict.fromkeys(any_curve["scenario"]))
    rows: List[Tuple[str, str, str]] = []
    for metric, ylabel in metric_specs:
        if metric not in usable:
            continue
        for scenario in scenarios:
            if scenario in set(usable[metric]["scenario"]):
                rows.append((metric, ylabel, scenario))
    if not rows:
        print(f"[figure] 没有匹配的行，跳过 {out_path.name}")
        return

    fig, axes = plt.subplots(
        len(rows), len(datasets), figsize=(6.2 * len(datasets), 3.4 * len(rows)), squeeze=False
    )
    for row_idx, (metric, ylabel, scenario) in enumerate(rows):
        sub = usable[metric]
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            part_ds = sub[(sub["dataset"] == dataset) & (sub["scenario"] == scenario)]
            for group in groups:
                part = part_ds[part_ds["group"] == group].sort_values("rel_round")
                if part.empty:
                    continue
                strong = group == "inj" or group.startswith("real_")
                _plot_series(
                    ax,
                    part,
                    GROUP_COLORS.get(group, "#333333"),
                    "-" if strong else "--",
                    2.1 if strong else 1.3,
                    1.0 if strong else 0.8,
                    GROUP_LABELS.get(group, group),
                )
            ax.axvline(0, color="black", linestyle=":", linewidth=1.3)
            ax.set_title(
                f"{dataset} | {SCENARIO_LABELS.get(scenario, scenario)} | ρ={reference_rho:.2f}", fontsize=9.5
            )
            ax.set_xlabel("Rounds since onset (0 = last round before onset)")
            ax.set_ylabel(ylabel, fontsize=8.5)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=6.2, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


def make_excess_figure(excess: pd.DataFrame, out_path: Path, reference_rho: float) -> None:
    """净效应图：每个面板里把**所有 ρ** 都画出来（颜色 = ρ）。

    为什么主图不用"单一 ρ"：ρ 大时图根本不动（实验二已证明 ρ≥0.5 接近冻结），
    ρ=0 时才有明显的扰动与残留。把 ρ 放在同一条横轴上，一眼就能看出
    「低 ρ 会跟、但撤不回来；高 ρ 干脆不动」这个结构性权衡。
    """
    metrics = (
        ("old_peer_share", "Δ old-interest peer share (inj − pair)"),
        ("ndcg", "Δ NDCG@10 (inj − pair)"),
    )
    usable = {metric: excess[excess["metric"] == metric] for metric, _ in metrics}
    usable = {k: v for k, v in usable.items() if not v.empty}
    if not usable:
        print(f"[figure] 没有成对差值数据，跳过 {out_path.name}")
        return
    try:
        plt = _plt()
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return
    any_frame = next(iter(usable.values()))
    datasets = list(dict.fromkeys(any_frame["dataset"]))
    scenarios = [s for s in ("sustained", "transient") if s in set(any_frame["scenario"])]
    if not scenarios:
        scenarios = list(dict.fromkeys(any_frame["scenario"]))
    rows = [(m, y, s) for m, y in metrics if m in usable for s in scenarios if s in set(usable[m]["scenario"])]
    if not rows:
        return
    fig, axes = plt.subplots(
        len(rows), len(datasets), figsize=(6.0 * len(datasets), 3.2 * len(rows)), squeeze=False
    )
    for row_idx, (metric, ylabel, scenario) in enumerate(rows):
        sub = usable[metric]
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            part_ds = sub[(sub["dataset"] == dataset) & (sub["scenario"] == scenario)]
            if part_ds.empty:
                ax.set_visible(False)
                continue
            for rho in _rho_of(part_ds):
                part = part_ds[np.isclose(part_ds["rho"].astype(float), rho)].sort_values("rel_round")
                if part.empty:
                    continue
                color = RHO_COLORS.get(round(rho, 2), "#333333")
                width = 2.4 if np.isclose(rho, reference_rho) else 1.6
                ax.plot(
                    part["rel_round"],
                    part["mean"],
                    color=color,
                    linewidth=width,
                    label=f"ρ={rho:.2f}",
                )
                sem = part["sem_pair"].fillna(0.0).to_numpy()
                ax.fill_between(
                    part["rel_round"].to_numpy(),
                    part["mean"].to_numpy() - sem,
                    part["mean"].to_numpy() + sem,
                    color=color,
                    alpha=0.13 if np.isclose(rho, reference_rho) else 0.07,
                    linewidth=0,
                )
            ax.axvline(0, color="black", linestyle=":", linewidth=1.3)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_title(
                f"{dataset} | {SCENARIO_LABELS.get(scenario, scenario)}", fontsize=9.5
            )
            ax.set_xlabel("Rounds since onset (0 = last round before onset)")
            ax.set_ylabel(ylabel, fontsize=8.5)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("Net effect of the intervention (injected − paired control), by EMA coefficient", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


def make_tradeoff_figure(tradeoff: pd.DataFrame, out_path: Path) -> None:
    """两个面板：上行"最终是否残留"（over_update）、下行"撤销窗口内是否还赖着"（over_update_early）。

    为什么必须画两行：只看"最终残留"会得到「ρ=0 落在左下角 = 两全」的结论，
    但 ρ=0 在撤销窗口内的残留是扰动幅度的 1.04 倍 —— **撤销晚了一整个窗口**。
    把 early 单独画出来，"没有任何 ρ 同时做到及时撤销 + 完全跟随"才看得见。
    """
    if tradeoff.empty:
        print(f"[figure] 没有权衡数据，跳过 {out_path.name}")
        return
    try:
        plt = _plt()
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return
    datasets = list(dict.fromkeys(tradeoff["dataset"]))
    plottable = [
        ds for ds in datasets
        if np.isfinite(tradeoff[tradeoff["dataset"] == ds]["under_update"]).any()
    ]
    if not plottable:
        print(f"[figure] 没有可判定的 ρ（over/under_update 全为 nan），跳过 {out_path.name}")
        return
    datasets = plottable
    panels = (
        ("over_update_early", "over-update INSIDE the revert window\n(residual right after the evidence stopped)"),
        ("over_update", "over-update AFTER the revert window\n(permanent residual)"),
    )
    fig, axes = plt.subplots(
        len(panels), len(datasets), figsize=(5.8 * len(datasets), 4.6 * len(panels)), squeeze=False
    )
    for row_idx, (column, xlabel) in enumerate(panels):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            part = tradeoff[tradeoff["dataset"] == dataset].sort_values("rho")
            part = part[np.isfinite(part["under_update"])]
            if part.empty:
                ax.set_visible(False)
                continue
            xs = part[column].to_numpy(dtype=np.float64)
            no_point = ~np.isfinite(xs)
            xs_plot = np.where(no_point, 0.0, xs)
            ys = part["under_update"].to_numpy(dtype=np.float64)
            ax.plot(xs_plot, ys, color="#888888", linewidth=1.1, linestyle="--", zorder=1)
            for idx, row in enumerate(part.itertuples()):
                color = RHO_COLORS.get(round(float(row.rho), 2), "#333333")
                if no_point[idx]:
                    ax.scatter(xs_plot[idx], ys[idx], s=110, facecolors="white",
                               edgecolors=color, linewidths=2.0, zorder=2)
                    label = f"ρ={row.rho:.2f} (graph did not update)"
                else:
                    ax.scatter(xs_plot[idx], ys[idx], s=110, color=color, zorder=2)
                    label = f"ρ={row.rho:.2f}"
                ax.annotate(label, (xs_plot[idx], ys[idx]), textcoords="offset points",
                            xytext=(9, 7), fontsize=8)
            ax.axvline(0.5, color="#bbbbbb", linestyle=":", linewidth=1.0)
            ax.axhline(0.5, color="#bbbbbb", linestyle=":", linewidth=1.0)
            x_lo = min(-0.05, float(np.nanmin(xs_plot)) - 0.06)
            y_lo = min(-0.05, float(np.nanmin(ys)) - 0.06)
            ax.set_xlim(x_lo, max(1.12, float(np.nanmax(xs_plot)) + 0.12))
            ax.set_ylim(y_lo, max(1.12, float(np.nanmax(ys)) + 0.12))
            ax.set_xlabel(xlabel, fontsize=8.5)
            ax.set_ylabel("under-update: 1 − tracked change / change at ρ=0", fontsize=8.5)
            ax.set_title(f"{dataset} — ideal = bottom-left", fontsize=9.5)
            ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle(
        "No fixed EMA coefficient is both timely on undo\nand faithful on sustained change",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


def make_rho_curves_figure(curve: pd.DataFrame, out_path: Path, metric: str, ylabel: str) -> None:
    if curve.empty:
        print(f"[figure] 没有 {metric} 曲线，跳过 {out_path.name}")
        return
    sub = curve[(curve["metric"] == metric) & (curve["group"] == "inj")]
    if sub.empty:
        print(f"[figure] 没有 inj 组的 {metric} 曲线，跳过 {out_path.name}")
        return
    try:
        plt = _plt()
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib 不可用：{exc}")
        return
    datasets = list(dict.fromkeys(sub["dataset"]))
    scenarios = [s for s in ("sustained", "transient") if s in set(sub["scenario"])] or list(
        dict.fromkeys(sub["scenario"])
    )
    fig, axes = plt.subplots(
        len(datasets), len(scenarios), figsize=(5.8 * len(scenarios), 3.6 * len(datasets)), squeeze=False
    )
    for row_idx, dataset in enumerate(datasets):
        for col_idx, scenario in enumerate(scenarios):
            ax = axes[row_idx][col_idx]
            part_ds = sub[(sub["dataset"] == dataset) & (sub["scenario"] == scenario)]
            for rho in _rho_of(part_ds):
                part = part_ds[np.isclose(part_ds["rho"].astype(float), rho)].sort_values("rel_round")
                if part.empty:
                    continue
                ax.plot(
                    part["rel_round"],
                    part["mean"],
                    color=RHO_COLORS.get(round(rho, 2), "#333333"),
                    linewidth=1.9,
                    label=f"ρ={rho:.2f}",
                )
            ax.axvline(0, color="black", linestyle=":", linewidth=1.3)
            ax.set_title(f"{dataset} | {SCENARIO_LABELS.get(scenario, scenario)}", fontsize=9.5)
            ax.set_xlabel("Rounds since onset (0 = last round before onset)")
            ax.set_ylabel(ylabel, fontsize=8.5)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("Injected users only: how far the neighbourhood moves, by EMA coefficient", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[figure] 已保存：{out_path}")


# ======================================================================================
# 报告
# ======================================================================================
def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_（无数据）_"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append(_fmt(value, 4))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    out_path: Path,
    tradeoff: pd.DataFrame,
    verdict: pd.DataFrame,
    window_summary: pd.DataFrame,
    scenario_meta: Dict[str, Dict[str, object]],
    reference_rho: float,
    rhos: Sequence[float],
) -> None:
    lines: List[str] = []
    lines.append("# 验证实验 3（val03）：短期兴趣波动会不会导致不必要的图更新？\n")
    lines.append(
        "> 本文件由 `val03_aggregate.py` 自动生成，数字全部可从 `results/raw/` 复算，不需要手工填。"
        "**结论性文字请写在 `val03_findings.md`**（人工整理），这里只保证数字与图表可追溯。\n"
    )

    lines.append("## 1. 设计\n")
    lines.append(
        "- **注入 arm（人工构造）**：把 onset 窗口（sustained: 及其后所有窗口；transient: 仅该窗口）里"
        "注入组用户的 itemId 换成目标类别（该用户最不熟悉的类别）的物品，其余行、时间戳、窗口边界都不动。"
        "配对对照 `pair` 用**完全相同的流程与窗口**，但替换成自己旧主类别的物品（同类别重采样）。\n"
        "- **真实 arm（只打标签）**：把真实用户分成 `real_sustained`（跳变后保持新类别）、"
        "`real_transient`（跳变后回到旧类别）、`real_stable`（从未跳变、活动量与 onset 匹配）。\n"
        "- 5 个组在**同一次运行**里共存 → 组间比较共享同一份训练轨迹、同一份参与序列。\n"
        f"- 主结论用参考 ρ = {reference_rho:.2f}（默认取论文配置 0.8；若该 ρ 没跑则自动取最接近的）；"
        f"ρ 扫描 = "
        + ", ".join(f"{r:.2f}" for r in rhos)
        + "。\n"
    )

    if scenario_meta:
        lines.append("### 1.1 各数据集的场景规模\n")
        for dataset, meta in scenario_meta.items():
            real = meta.get("real", {}) or {}
            inject = meta.get("inject", {}) or {}
            lines.append(
                f"- **{dataset}**：注入 inj={inject.get('num_inj')} / pair={inject.get('num_pair')}"
                f"（min_inter={inject.get('min_inter_used')}，onset={inject.get('inject_onset')}）；"
                f"真实 sustained={real.get('num_real_sustained')} / transient={real.get('num_real_transient')}"
                f" / stable={real.get('num_real_stable')}；类别来源={meta.get('category_source')}。"
            )
            for warning in meta.get("warnings", []) or []:
                lines.append(f"  - ⚠️ {warning}")
        lines.append("")

    lines.append("## 2. 关键指标：过度更新 vs 更新滞后\n")
    lines.append(
        "两个指标都是**成对差值口径**（注入组 − 配对对照），所以「物品被替换」这个动作本身的影响被对照掉了；"
        "主指标是 `old_peer_share`（邻居里「仍属于用户旧兴趣」的边权占比），因为它有方向、且在 ρ 小时不会像 "
        "Jaccard turnover 那样饱和到 1。\n\n"
        "- `over_update_early` = **撤销发生那个窗口**的残留 ÷ 扰动期改动。>0.5 = 撤销至少要晚一个窗口（迟滞）。\n"
        "- `over_update` = **再下一个窗口**的残留 ÷ 扰动期改动。>0.5 = 永久残留（真正意义上的过度更新）。\n"
        "- `under_update` = 1 − 持续变化场景的净跟进量 ÷ ρ=0 的净跟进量。"
        "0 = 完全跟上；1 = 完全不更新（更新滞后）。\n"
        "- `summary_residual_ratio` = 撤销后摘要的残余偏离 ÷ 扰动期偏离。"
        "它 <1 说明「兴趣摘要本身恢复了」，此时图若没恢复，责任就在边权更新规则；"
        "它 ≥1 说明摘要根本没恢复，残留主要来自客户端/摘要层而不是 EMA。\n"
        "- 理想固定 ρ 应同时给出 `over_update≈0` 与 `under_update≈0`。\n"
    )
    columns = [
        "dataset",
        "rho",
        "over_update_early",
        "over_update",
        "under_update",
        "transient_attack_during",
        "transient_residual_post1",
        "transient_residual_post2",
        "sustained_tracked_post",
        "sustained_tracked_at_rho0",
        "summary_residual_ratio",
    ]
    lines.append(_to_markdown(tradeoff[[c for c in columns if c in tradeoff.columns]]))
    lines.append("")

    lines.append("## 3. 自动判定\n")
    lines.append(
        f"判定规则：关系质量的噪声地板 {PEER_NOISE_FLOOR}，NDCG 的噪声地板 {NDCG_NOISE_FLOOR}；"
        "`over_update < 0.5`（再下一个窗口的残留）视为最终能回退，`under_update > 0.5` 视为明显滞后。\n"
    )
    if not verdict.empty:
        lines.append(_to_markdown(verdict[["dataset", "check", "holds", "detail"]]))
    lines.append("")

    lines.append("## 4. 分窗口汇总（可直接抄进表格）\n")
    if not window_summary.empty:
        keep = window_summary[
            (window_summary["metric"].isin(["old_peer_share", "ndcg", "turnover_from_pre"]))
            & (np.isclose(window_summary["rho"].astype(float), reference_rho))
        ].copy()
        columns = [
            "dataset",
            "scenario",
            "group",
            "metric",
            "pre_level",
            "during_level",
            "post1_level",
            "post2_level",
            "plateau_level",
            "post_n_users",
        ]
        lines.append(_to_markdown(keep[[c for c in columns if c in keep.columns]]))
    lines.append("")

    lines.append("## 5. 怎么读这几张图\n")
    lines.append(
        "- `val03_fig_arm_synthetic.png`：注入 arm。看 (a) 扰动窗口（rel_round 1..R）transient 的 "
        "old-peer 占比是否也掉下来（扰动确实进了图）；(b) 撤销窗口（R+1 之后）它是否回升到配对对照的水平"
        "（能撤销）还是停在 sustained 的水平（过度更新）。\n"
        "- `val03_fig_pair_excess.png`：★★ 主结论图 —— 已经是「注入组 − 配对对照」的净效应，"
        "而且把所有 ρ 画在同一张里（颜色=ρ）。低 ρ 会是「先掉下去、撤不回来」，高 ρ 基本贴着 0。\n"
        "- `val03_fig_rho_tradeoff.png`：★★ 关键图，**两行**。上行横轴是「撤销窗口内的残留」"
        "（衡量撤销是否及时），下行是「撤销后第二窗的残留」（衡量是否永久损伤）。"
        "空心点 = 该 ρ 下图根本没动。"
        "★ 读法：ρ=0 在下行已经靠近左下角（最终能撤销），但在上行远离左下角（**晚一整窗才撤销**）；"
        "ρ≥0.5 在上行下行都远离。因此**没有任何 ρ 同时做到「及时撤销 + 完全跟随 + 图稳定」**，"
        "但这也不能简单写成「ρ=0 就够用」——它的代价是撤销延迟与最大抖动，见 `val03_findings.md` §4、§9。\n"
        "- `val03_fig_arm_synthetic.png`：注入 arm 的原始水平（含配对对照）。注意在 ρ 大时两条线会几乎重合，"
        "那正是「图冻结」的直接证据。\n"
        "- `val03_fig_arm_real.png`：真实数据答案（无人工注入），与注入 arm 的结论对照着写。\n"
        "- `val03_fig_rho_curves.png`：ρ 对「图动多少」的影响；ρ≥0.5 时往往已经接近冻结（与实验二一致）。\n"
    )

    lines.append("## 6. 写报告时必须交代的局限\n")
    lines.append(
        "1. **注入是人工构造**：只替换 itemId，不改窗口边界/交互数/时间戳；结论必须与真实 arm 分开写。\n"
        "2. **短期扰动的粒度是 1 个窗口**（受窗口划分限制），不是 1 轮或 1 天；"
        "真实数据里若「几天」的波动远短于一个窗口，本实验会**低估**过度更新的程度。\n"
        "3. **注入的痕迹会留在历史上**：窗口 t0 的注入物品在之后的窗口里仍在用户历史中"
        "（真实系统同样无法抹掉一次「连刷几部恐怖片」）。所以本实验测的是"
        "「摘要记忆 + 边权记忆」的合计残留，`summary_residual_ratio` 用来把这两层分开。\n"
        "4. **注入组规模有限**：MovieLens-1M 严格口径约 19 对；"
        "稀疏数据集（Amazon 类）每窗口每用户只有约 2 条交互，真实 arm 的 transient 组往往不足 25 人，"
        "只能作方向性参考（与实验二的结论一致）。\n"
        "5. **NDCG 的成对差值不完美**：sustained 组在扰动窗口之后的测试目标是「新兴趣物品」、"
        "transient 组是「旧兴趣物品」，两组绝对水平不可直接比较，只能各自对自己 pre 期做差再相减（DID）。\n"
        "6. **真实 arm 的分组用到了 onset 之后的窗口**（事后分组），只决定统计哪些人，不进入模型。\n"
        "7. 3 个种子只够看趋势；正式论文建议 5 个以上。\n"
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] 已写出：{out_path}")


# ======================================================================================
# 主流程
# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="val03 结果聚合：过度更新 vs 更新滞后 + 自动结论")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--datasets", default="", help="逗号分隔；留空 = 全部")
    parser.add_argument("--reference-rho", type=float, default=0.80, help="主图用的 ρ（论文默认 0.8）")
    parser.add_argument("--rounds-per-window", type=int, default=0, help="0 = 自动从 summary 读")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out.resolve()
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    user, rnd = load_all(root, datasets)
    rounds = args.rounds_per_window or detect_window_rounds(root)
    print(f"Loaded {len(user)} user-round rows from {root}（每窗口 {rounds} 轮）")
    print(user.groupby(["dataset", "scenario", "group", "rho"])["user"].nunique().to_string())

    rhos = _rho_of(user)
    dataset_list = list(dict.fromkeys(user["dataset"]))
    print(f"数据集: {dataset_list} | ρ: {rhos}")
    if rhos and not any(np.isclose(r, args.reference_rho) for r in rhos):
        nearest = min(rhos, key=lambda r: abs(r - args.reference_rho))
        print(
            f"[warn] 参考 ρ={args.reference_rho:.2f} 在这些结果里不存在（可用 {rhos}），"
            f"自动改用最接近的 ρ={nearest:.2f}"
        )
        args.reference_rho = float(nearest)

    curves: Dict[str, pd.DataFrame] = {}
    for metric in USER_METRICS:
        frame = build_curve(user, metric)
        curves[metric] = frame
        if not frame.empty:
            frame.to_csv(tables_dir / f"val03_curve_{metric}.csv", index=False, encoding="utf-8-sig")

    summaries: Dict[str, pd.DataFrame] = {
        metric: summarise_curve(curves.get(metric, pd.DataFrame()), metric, rounds) for metric in USER_METRICS
    }
    window_summary = pd.concat([s for s in summaries.values() if not s.empty], ignore_index=True)
    if not window_summary.empty:
        window_summary.to_csv(tables_dir / "val03_window_summary.csv", index=False, encoding="utf-8-sig")

    excess_frames: List[pd.DataFrame] = []
    for metric in ("old_peer_share", "ndcg", "turnover_from_pre", "summary_drift"):
        frame = build_pair_excess(user, metric)
        if not frame.empty:
            excess_frames.append(frame)
    excess = pd.concat(excess_frames, ignore_index=True) if excess_frames else pd.DataFrame()
    if not excess.empty:
        excess.to_csv(tables_dir / "val03_pair_excess.csv", index=False, encoding="utf-8-sig")

    excess_peer = summarise_curve(excess, "old_peer_share", rounds) if not excess.empty else pd.DataFrame()
    excess_turn = summarise_curve(excess, "turnover_from_pre", rounds) if not excess.empty else pd.DataFrame()
    excess_drift = summarise_curve(excess, "summary_drift", rounds) if not excess.empty else pd.DataFrame()

    lag = compute_lag_50(
        curves.get(PRIMARY_METRIC, pd.DataFrame()), PRIMARY_METRIC, rounds
    )
    if not lag.empty:
        lag.to_csv(tables_dir / f"val03_lag_{PRIMARY_METRIC}.csv", index=False, encoding="utf-8-sig")

    tradeoff = build_tradeoff(
        excess_peer, excess_turn, excess_drift, summaries[PRIMARY_METRIC], dataset_list, rhos
    )
    if not tradeoff.empty:
        tradeoff.to_csv(tables_dir / "val03_tradeoff.csv", index=False, encoding="utf-8-sig")

    verdict = build_verdict(tradeoff, summaries["ndcg"], summaries[PRIMARY_METRIC], dataset_list, args.reference_rho)
    if not verdict.empty:
        verdict.to_csv(tables_dir / "val03_verdict.csv", index=False, encoding="utf-8-sig")

    make_arm_figure(
        curves,
        (("old_peer_share", "Old-interest peer share"), ("ndcg", "Mean NDCG@10")),
        figures_dir / "val03_fig_arm_synthetic.png",
        groups=("inj", "pair"),
        title="Synthetic arm: injected users vs paired control",
        reference_rho=args.reference_rho,
    )
    make_arm_figure(
        curves,
        (("old_peer_share", "Old-interest peer share"), ("ndcg", "Mean NDCG@10")),
        figures_dir / "val03_fig_arm_real.png",
        groups=("real_sustained", "real_transient", "real_stable"),
        title="Real-data arm (observational, no injection)",
        reference_rho=args.reference_rho,
    )
    make_excess_figure(excess, figures_dir / "val03_fig_pair_excess.png", args.reference_rho)
    make_tradeoff_figure(tradeoff, figures_dir / "val03_fig_rho_tradeoff.png")
    make_rho_curves_figure(
        curves.get("old_peer_share", pd.DataFrame()),
        figures_dir / "val03_fig_rho_curves.png",
        "old_peer_share",
        "Old-interest peer share",
    )

    scenario_meta: Dict[str, Dict[str, object]] = {}
    for meta_path in sorted(args.plan_root.resolve().rglob("val03_scenario_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            continue
        name = str(meta.get("dataset", meta_path.parent.name))
        if datasets and name not in datasets:
            continue
        scenario_meta[name] = meta
    write_report(
        out_dir / "val03_report.md",
        tradeoff,
        verdict,
        window_summary,
        scenario_meta,
        args.reference_rho,
        rhos,
    )

    if not verdict.empty:
        print("\n================ 自动判定 ================")
        for _, row in verdict.iterrows():
            mark = "[yes]" if row["holds"] == "yes" else "[no ]"
            print(f"  {mark} [{row['dataset']}] {row['check']}")
            print(f"        {row['detail']}")
        print("=========================================")
    print(f"\n输出：{out_dir}")


if __name__ == "__main__":
    main()
