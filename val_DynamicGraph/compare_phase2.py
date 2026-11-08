"""阶段 2 回归对比：新代码（记录层改动）是否**逐位**复现旧结果。

对比对象：
  旧：`val_DynamicGraph/results/raw/<dataset>/<arm>/seed_<s>/`
  新：`--new` 指定的目录（如 `results/raw_phase2`），同一 arm / seed

检查项：
  1. `dg_summary.csv` 的 avg/final NDCG@10、HR@10 —— 逐位一致（≤1e-12）
  2. `dg_round_metrics.csv` 的逐评测点 all_ndcg / all_hr —— 逐位一致
  3. `dg_user_metrics.csv` 的逐用户 ndcg / hr —— 逐位一致
  4. 新列是否落盘：`active_coverage`（summary）、`dg_graph_round_metrics.csv`、
     `dg_user_coverage.csv`；并打印 ECR 非零行数 / 刷新轮 / 首次覆盖统计

用法（服务器）：
  /home/wl/.venv-feddieg/bin/python -X utf8 val_DynamicGraph/compare_phase2.py \
      --old val_DynamicGraph/results/raw \
      --new val_DynamicGraph/results/raw_phase2 \
      --dataset movielens-1m --seed 42 \
      --arms nograph static periodic dynamic
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DG_DIR = Path(__file__).resolve().parent
TOL = 1e-12


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def diff_max(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    if a.size != b.size:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(a - b)))


def compare_arm(old_dir: Path, new_dir: Path) -> Tuple[Dict[str, object], List[str]]:
    row: Dict[str, object] = {"arm": old_dir.parent.name}
    problems: List[str] = []

    s_old, s_new = read_csv(old_dir / "dg_summary.csv"), read_csv(new_dir / "dg_summary.csv")
    if s_old.empty or s_new.empty:
        return {"arm": old_dir.parent.name, "error": "missing summary"}, ["缺少 dg_summary.csv"]
    for col in ("avg_window_ndcg", "final_window_ndcg", "avg_window_hr", "final_window_hr"):
        d = abs(float(s_old[col].iloc[0]) - float(s_new[col].iloc[0]))
        row[f"d_{col}"] = d
        if not np.isfinite(d) or d > TOL:
            problems.append(f"{col} 差 {d:.3e}")

    r_old, r_new = read_csv(old_dir / "dg_round_metrics.csv"), read_csv(new_dir / "dg_round_metrics.csv")
    if len(r_old) != len(r_new):
        problems.append(f"评测点数 {len(r_old)} -> {len(r_new)}")
    else:
        for col in ("all_ndcg", "all_hr"):
            d = diff_max(r_old[col], r_new[col])
            row[f"d_round_{col}"] = d
            if not np.isfinite(d) or d > TOL:
                problems.append(f"逐评测点 {col} 差 {d:.3e}")
        row["n_eval_points"] = len(r_new)

    u_old, u_new = read_csv(old_dir / "dg_user_metrics.csv"), read_csv(new_dir / "dg_user_metrics.csv")
    if len(u_old) == len(u_new) and len(u_old):
        key = ["window", "user"]
        m = u_old.merge(u_new, on=key, suffixes=("_o", "_n"))
        for col in ("ndcg", "hr"):
            d = diff_max(m[f"{col}_o"], m[f"{col}_n"])
            row[f"d_user_{col}"] = d
            if not np.isfinite(d) or d > TOL:
                problems.append(f"逐用户 {col} 差 {d:.3e}")
        row["n_user_rows"] = len(m)
    elif len(u_old) or len(u_new):
        problems.append(f"逐用户行数 {len(u_old)} -> {len(u_new)}")

    # ---- 新列/新文件
    row["has_active_coverage"] = "mean_active_coverage" in s_new.columns
    g = read_csv(new_dir / "dg_graph_round_metrics.csv")
    c = read_csv(new_dir / "dg_user_coverage.csv")
    row["graph_round_rows"] = len(g)
    row["coverage_rows"] = len(c)
    if len(g):
        row["rounds_is_update"] = int(g["is_update"].sum())
        row["rounds_is_refresh"] = int(g["is_refresh"].sum())
        row["ecr_nonzero_rounds"] = int((g["ecr_round_mean"].fillna(0) > 0).sum())
        row["n_edges_last"] = float(g["n_edges"].iloc[-1])
        row["active_coverage_mean"] = float(g["active_coverage"].mean())
        refresh_rounds = g.loc[g["is_refresh"] == 1, "global_round"].tolist()
        row["refresh_rounds"] = ",".join(str(int(x)) for x in refresh_rounds) if refresh_rounds else ""
    if len(c):
        row["n_covered_users"] = int((c["first_covered_round"] >= 0).sum())
        row["median_first_covered"] = float(
            c.loc[c["first_covered_round"] >= 0, "first_covered_round"].median()
        )
    return row, problems


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 2 回归：逐位对比新旧结果")
    ap.add_argument("--old", type=Path, default=DG_DIR / "results" / "raw")
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--dataset", default="movielens-1m")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", nargs="+", default=["nograph", "static", "periodic", "dynamic"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows, failed = [], []
    for arm in args.arms:
        old_dir = args.old / args.dataset / arm / f"seed_{args.seed}"
        new_dir = args.new / args.dataset / arm / f"seed_{args.seed}"
        if not new_dir.exists():
            print(f"[skip] {arm}: 新目录不存在 {new_dir}")
            continue
        if not old_dir.exists():
            print(f"[skip] {arm}: 旧目录不存在 {old_dir}")
            continue
        row, problems = compare_arm(old_dir, new_dir)
        rows.append(row)
        if problems:
            failed.append((arm, problems))
        print(f"[{'FAIL' if problems else ' OK '}] {arm}" + ("" if not problems else " | " + "; ".join(problems)))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print("\n=== 逐位对比 ===")
    print(df.to_string(index=False))

    out = args.out or (args.new / "phase2_regress_compare.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n表已写出：{out}")

    bitwise_cols = [c for c in df.columns if c.startswith("d_")]
    worst = float(np.nanmax(df[bitwise_cols].to_numpy(dtype=float))) if bitwise_cols and len(df) else float("nan")
    print(f"最大逐位偏差：{worst:.3e}（要求 ≤ {TOL:g}）")
    if failed:
        print("\n有不一致：")
        for arm, problems in failed:
            print(f"  - {arm}: {'; '.join(problems)}")
        return 1
    print("\nALL BITWISE-IDENTICAL —— 记录层改动没有改变任何数值")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
