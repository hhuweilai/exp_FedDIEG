#!/usr/bin/env python
"""验证实验 1 · 三项前置检验的汇总与判定。

读取：
    正式结果            results/raw/                     （graph_mix=0.05，边只对抽样用户重算）
    检验 1 图权重        results/checks/graphmix0.2/      （graph_mix=0.2）
    检验 2 边更新范围    results/checks/edgescope_all/    （每轮重算全部活跃用户的边）
    检验 3 摘要漂移      results/checks/drift/            （每窗口末测 缓存 vs 新鲜 摘要）

输出：
    results/checks/checks_report.md    带判定的 markdown
    results/checks/checks_summary.csv  一张扁平表（便于贴进报告或画图）

用法：
    python val01_checks_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "exp_val_1026" / "exp01_staleness_participation"
DEFAULT_BASE = BASE / "results"
METRIC = "avg_window_ndcg"

BIG_EFFECT = 0.005      # 达到它就是"实质性效应"
NOISE_FLOOR = 0.002     # 低于它就是"可忽略"（与主实验噪声地板一致）


def load_summaries(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(root.rglob("val01_summary.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:  # pragma: no cover
            continue
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["participation_rate"] = data["participation_rate"].astype(float).round(3)
    return data


def load_drift(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(root.rglob("val01_drift_buckets.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:  # pragma: no cover
            continue
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["participation_rate"] = data["participation_rate"].astype(float).round(3)
    return data


def paired_delta(
    df: pd.DataFrame,
    mode: str,
    rate: float,
    base_mode: str,
    base_rate: Optional[float] = None,
) -> pd.DataFrame:
    """同一 (dataset, seed) 下 mode@rate 减去 base_mode@base_rate。"""
    base_rate = rate if base_rate is None else base_rate
    left = df[(df["mode"] == mode) & np.isclose(df["participation_rate"], rate)]
    right = df[(df["mode"] == base_mode) & np.isclose(df["participation_rate"], base_rate)]
    if left.empty or right.empty:
        return pd.DataFrame()
    merged = left.merge(right, on=["dataset", "seed"], suffixes=("", "_base"))
    if merged.empty:
        return pd.DataFrame()
    merged["delta"] = pd.to_numeric(merged[METRIC], errors="coerce") - pd.to_numeric(
        merged[f"{METRIC}_base"], errors="coerce"
    )
    return merged[["dataset", "seed", "delta"]].dropna()


def stat(deltas: pd.DataFrame) -> Dict[str, float]:
    if deltas.empty:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0, "neg": 0}
    values = deltas["delta"].to_numpy(dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "n": len(values),
        "neg": int((values < 0).sum()),
    }


def fmt(s: Dict[str, float], digits: int = 5) -> str:
    if not s["n"]:
        return "-"
    return f"{s['mean']:+.{digits}f} ± {s['sd']:.{digits}f} (n={s['n']}, 负 {s['neg']}/{s['n']})"


def main() -> None:
    parser = argparse.ArgumentParser(description="val01 三项前置检验汇总")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="results 根目录")
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--rates", default="0.1,0.3")
    args = parser.parse_args()

    results = args.base.resolve()
    checks = results / "checks"
    rates = [float(x) for x in args.rates.split(",") if x.strip()]

    baseline = load_summaries(results / "raw")
    if baseline.empty:
        raise SystemExit(f"没有找到正式结果：{results / 'raw'}")
    baseline = baseline[baseline["dataset"] == args.dataset]

    gmix = load_summaries(checks / "graphmix0.2") if (checks / "graphmix0.2").exists() else pd.DataFrame()
    scope = load_summaries(checks / "edgescope_all") if (checks / "edgescope_all").exists() else pd.DataFrame()
    drift = load_drift(checks / "drift") if (checks / "drift").exists() else pd.DataFrame()

    lines: List[str] = []
    flat: List[Dict[str, object]] = []
    verdicts: List[str] = []

    lines.append(f"# 验证实验 1 · 三项前置检验（{args.dataset}）")
    lines.append("")
    lines.append(
        "判定阈值：效应绝对值 ≥ "
        f"{BIG_EFFECT} 视为实质性；< {NOISE_FLOOR} 视为可忽略（与主实验噪声地板一致）。"
    )
    lines.append("")

    # 预算一致性检查：检验 run 与基线的窗口数/每窗口轮数必须相同，否则差值没有意义
    budgets = {}
    for name, frame in (("基线", baseline), ("检验1", gmix), ("检验2", scope)):
        if isinstance(frame, pd.DataFrame) and not frame.empty and "num_windows" in frame.columns:
            budgets[name] = (
                int(frame["num_windows"].mode().iloc[0]),
                int(frame["rounds_per_window"].mode().iloc[0]),
            )
    if len(set(budgets.values())) > 1:
        lines.append("> [!] 预算不一致，下表差值不可直接解释：")
        for name, value in budgets.items():
            lines.append(f"> - {name}: {value[0]} 窗口 × {value[1]} 轮")
        lines.append("> 请用与主实验相同的预算（默认 5 × 120）重跑检验。")
        lines.append("")
    elif budgets:
        first = next(iter(budgets.values()))
        lines.append(f"预算一致：{first[0]} 窗口 × {first[1]} 轮（基线、检验 1、检验 2 相同）。")
        lines.append("")

    base_part = pd.DataFrame()
    base_stale = pd.DataFrame()
    for rate in rates:
        base_part = pd.concat([base_part, paired_delta(baseline, "upload_only", rate, "upload_only", 1.0)])
        base_stale = pd.concat([base_stale, paired_delta(baseline, "upload_only", rate, "upload_fresh_edges")])

    lines.append("## 0. 基线（graph_mix=0.05，边只对抽样用户重算）")
    lines.append("")
    lines.append("| 参与率 | 参与率效应 ΔNDCG@10 | 陈旧损失（upload_only 减 fresh） |")
    lines.append("|---|---|---|")
    for rate in rates:
        p_stat = stat(paired_delta(baseline, "upload_only", rate, "upload_only", 1.0))
        s_stat = stat(paired_delta(baseline, "upload_only", rate, "upload_fresh_edges"))
        lines.append(f"| p={rate:.2f} | {fmt(p_stat)} | {fmt(s_stat)} |")
    lines.append("")
    lines.append(f"陈旧损失合并（{len(rates)} 个参与率）：{fmt(stat(base_stale))}")
    lines.append("")

    # ---------------- 检验 1：图权重 ----------------
    lines.append("## 1. 图权重敏感性：graph_mix 0.05 → 0.2")
    lines.append("")
    if gmix.empty:
        lines.append("_尚未运行（results/checks/graphmix0.2 为空）。_")
        lines.append("")
    else:
        lines.append("| 参与率 | NDCG@10 (graph_mix=0.2) | 陈旧损失 (0.2) | 陈旧损失 (基线 0.05) |")
        lines.append("|---|---|---|---|")
        stale_02: List[Dict[str, float]] = []
        for rate in rates:
            sub = gmix[(gmix["mode"] == "upload_only") & np.isclose(gmix["participation_rate"], rate)]
            level = float(sub[METRIC].mean()) if not sub.empty else float("nan")
            s = stat(paired_delta(gmix, "upload_only", rate, "upload_fresh_edges"))
            b = stat(paired_delta(baseline, "upload_only", rate, "upload_fresh_edges"))
            if s["n"]:
                stale_02.append(s)
            level_str = f"{level:.4f}" if np.isfinite(level) else "-"
            lines.append(f"| p={rate:.2f} | {level_str} | {fmt(s)} | {fmt(b)} |")
            flat.append({"check": "graphmix", "metric": f"level_p{rate:.2f}", "value": level})
            flat.append({"check": "graphmix", "metric": f"staleness_loss_p{rate:.2f}", "value": s["mean"]})
            flat.append({"check": "graphmix", "metric": f"baseline_staleness_loss_p{rate:.2f}", "value": b["mean"]})
        lines.append("")
        if stale_02:
            pooled = float(np.mean([s["mean"] for s in stale_02]))
            flat.append({"check": "graphmix", "metric": "staleness_loss_pooled", "value": pooled})
            if abs(pooled) >= BIG_EFFECT:
                msg = (
                    f"**判定：效应被小图权重淹没。** graph_mix=0.2 时陈旧损失放大到 {pooled:+.4f}"
                    f"（≥{BIG_EFFECT}），说明方向可救，应继续做自适应图更新。"
                )
            elif abs(pooled) <= NOISE_FLOOR:
                msg = (
                    f"**判定：结论闭合。** graph_mix=0.2 时陈旧损失仍只有 {pooled:+.4f}"
                    f"（≤{NOISE_FLOOR}），小权重淹没这一解释被排除。"
                )
            else:
                direction = "放大" if pooled < 0 else "反向缩小"
                msg = (
                    f"**判定：介于两者之间。** 陈旧损失 {pooled:+.4f}，比基线 -0.0013 有所{direction}"
                    f"但未达 {BIG_EFFECT}；建议把 graph_mix 提到 0.4 复核，或直接看检验 2、3 的结果。"
                )
            verdicts.append("检验 1：" + msg)
            lines.append(msg)
            lines.append("")

    # ---------------- 检验 2：边更新范围 ----------------
    lines.append("## 2. 边更新频率：每轮重算全部活跃用户的边（--update-scope all）")
    lines.append("")
    if scope.empty:
        lines.append("_尚未运行（results/checks/edgescope_all 为空）。_")
        lines.append("")
    else:
        lines.append("| 参与率 | NDCG@10 (scope=all) | 相对基线 | ECR 基线 → all |")
        lines.append("|---|---|---|---|")
        deltas_all: List[Dict[str, float]] = []
        for rate in rates:
            cur = scope[(scope["mode"] == "upload_only") & np.isclose(scope["participation_rate"], rate)]
            base = baseline[(baseline["mode"] == "upload_only") & np.isclose(baseline["participation_rate"], rate)]
            if cur.empty or base.empty:
                continue
            merged = cur.merge(base, on=["dataset", "seed"], suffixes=("", "_base"))
            diff = (
                pd.to_numeric(merged[METRIC], errors="coerce")
                - pd.to_numeric(merged[f"{METRIC}_base"], errors="coerce")
            ).dropna()
            s = stat(pd.DataFrame({"delta": diff}))
            deltas_all.append(s)
            ecr_cur = float(cur["avg_ecr"].mean())
            ecr_base = float(base["avg_ecr"].mean())
            lines.append(
                f"| p={rate:.2f} | {cur[METRIC].mean():.4f} | {fmt(s)} | {ecr_base:.4f} → {ecr_cur:.4f} |"
            )
            flat.append({"check": "edgescope", "metric": f"gain_p{rate:.2f}", "value": s["mean"]})
            flat.append({"check": "edgescope", "metric": f"ecr_p{rate:.2f}", "value": ecr_cur})
        lines.append("")
        lines.append(
            "注：ECR 是在本轮实际重算的行上取平均；scope=all 会把大量本来就没变化的行也算进去，"
            "因此 ECR 反而下降属正常现象，不代表图更稳定。"
        )
        lines.append("")
        if deltas_all:
            pooled = float(np.mean([s["mean"] for s in deltas_all]))
            flat.append({"check": "edgescope", "metric": "gain_pooled", "value": pooled})
            if pooled >= BIG_EFFECT:
                msg = (
                    f"**判定：真正的问题是边更新滞后。** 每轮重算全部用户的边带来 {pooled:+.4f} 收益"
                    f"（≥{BIG_EFFECT}），说明自适应地决定哪些边该被重算，比调整 ρ 更有价值。"
                )
            elif abs(pooled) <= NOISE_FLOOR:
                msg = (
                    f"**判定：边更新滞后也不构成问题。** 收益仅 {pooled:+.4f}（≤{NOISE_FLOOR}），"
                    "与摘要陈旧无害的结论一致，图通路整体影响很小。"
                )
            elif pooled > 0:
                msg = (
                    f"**判定：小幅正收益 {pooled:+.4f}。** 未达实质性阈值但方向为正，"
                    "可作为更频繁重算边的弱支持证据。"
                )
            else:
                msg = (
                    f"**判定：小幅负收益 {pooled:+.4f}（方向为负）。** 更频繁地重算全部边反而略差，"
                    "说明现有“只更新被抽样用户”的做法并没有拖后腿，边更新滞后不是瓶颈。"
                )
            verdicts.append("检验 2：" + msg)
            lines.append(msg)
            lines.append("")

    # ---------------- 检验 3：摘要漂移 ----------------
    lines.append("## 3. 摘要漂移：缓存摘要 vs 此刻重算的新鲜摘要")
    lines.append("")
    if drift.empty:
        lines.append("_尚未运行（results/checks/drift 为空）。_")
        lines.append("")
    else:
        bucket_pos = {"0": 0, "1": 1, "2": 2, "3-4": 3, "5-9": 4, "10-19": 5, "20-39": 6, "40+": 7}
        lines.append("| 陈旧度桶 | 平均陈旧度 | 平均 cos(缓存, 新鲜) | 平均 L2 | 用户数 |")
        lines.append("|---|---|---|---|---|")
        rows = []
        for name, group in drift.groupby("bucket"):
            weight = np.maximum(group["n_users"].to_numpy(dtype=float), 1.0)
            cos = float(np.average(group["mean_cos"], weights=weight))
            l2 = float(np.average(group["mean_l2"], weights=weight))
            stale = float(np.average(group["mean_staleness"], weights=weight))
            rows.append((name, stale, cos, l2, int(weight.sum())))
            flat.append({"check": "drift", "metric": f"cos_{name}", "value": cos})
            flat.append({"check": "drift", "metric": f"l2_{name}", "value": l2})
        rows.sort(key=lambda r: bucket_pos.get(r[0], 99))
        for name, stale, cos, l2, n in rows:
            lines.append(f"| {name} | {stale:.1f} | {cos:.4f} | {l2:.3f} | {n} |")
        lines.append("")
        if rows:
            worst = min(rows, key=lambda r: r[2])
            if worst[2] > 0.9:
                detail = "cos 仍然很高（>0.9），说明摘要本来就漂移很小，所谓陈旧其实是「几乎没变旧」。"
            else:
                detail = (
                    "cos 已明显下降，说明摘要确实变旧了；但主实验显示换掉它只值约 0.0013 NDCG，"
                    "因此瓶颈不在信息新鲜度，而在图对预测的贡献太小。"
                )
            msg = f"**判定：最陈旧桶（{worst[0]}，约 {worst[1]:.0f} 轮）cos={worst[2]:.4f}。** {detail}"
            verdicts.append("检验 3：" + msg)
            lines.append(msg)
            lines.append("")

    # ---------------- 总结 ----------------
    lines.append("## 4. 总结论")
    lines.append("")
    if verdicts:
        for item in verdicts:
            lines.append(f"- {item}")
    else:
        lines.append("- 三项检验都还没跑，先执行 val01_run_checks.ps1。")
    lines.append("")
    lines.append("汇报建议：把三项判定与主实验结论（两数据集均命中停止信号 1）放在一起。")
    lines.append("若三项都指向无实质效应，则可带着完整证据讨论换场景或转向其它叙事；")
    lines.append("只要有任意一项显示实质效应（≥0.005），就说明还有可做空间，应优先沿那条线深入。")
    lines.append("")

    out_md = checks / "checks_report.md"
    out_csv = checks / "checks_summary.csv"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    if flat:
        pd.DataFrame(flat).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n".join(lines))
    print(f"\n已写入：{out_md}")
    if flat:
        print(f"已写入：{out_csv}")


if __name__ == "__main__":
    main()
