"""阶段 2 记录层验收：把每轮图指标汇总成一张人读表。

用途：确认 C1/C4/A6/C6 的落盘真的有用 —— 不只看「列存在」，而是看数值是否合理：

  * 每条臂的**更新轮 / 重建轮**（哪个轮次真的动了图）
  * 重建幅度（`added_round_sum`、`rebuild_turnover`；ECR 在 reset 型臂上恒为 0）
  * 本窗口**被评测用户覆盖率** `active_coverage`（A6）
  * 逐用户首次覆盖（C6）：覆盖人数 / 中位首次覆盖轮

用法：
  /home/wl/.venv-feddieg/bin/python -X utf8 val_DynamicGraph/inspect_phase2_records.py \
      --root val_DynamicGraph/results/raw_phase2 --dataset movielens-1m --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DG_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 2 记录层验收")
    ap.add_argument("--root", type=Path, default=DG_DIR / "results" / "raw_phase2")
    ap.add_argument("--dataset", default="movielens-1m")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", nargs="+",
                    default=["nograph", "static", "periodic", "dynamic",
                             "onboarding", "fresh_only"])
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        run = args.root / args.dataset / arm / f"seed_{args.seed}"
        g_path = run / "dg_graph_round_metrics.csv"
        c_path = run / "dg_user_coverage.csv"
        if not g_path.exists():
            print(f"[skip] {arm}: 没有 {g_path}")
            continue
        g = pd.read_csv(g_path)
        # 阶段 2 早期叫 rebuild_turnover，2026-11-08 起统一为 graph_turnover（语义相同）
        turnover_col = (
            "graph_turnover" if "graph_turnover" in g.columns
            else ("rebuild_turnover" if "rebuild_turnover" in g.columns else None)
        )
        updates = g[g["is_update"] == 1]
        refreshes = g[g["is_refresh"] == 1]
        cover = pd.read_csv(c_path) if c_path.exists() else pd.DataFrame()
        covered = cover[cover["first_covered_round"] >= 0] if len(cover) else cover
        rows.append({
            "arm": arm,
            "rounds": len(g),
            "update_rounds": len(updates),
            "update_rounds_list": ",".join(str(int(x)) for x in updates["global_round"].tolist()[:8])
            + ("..." if len(updates) > 8 else ""),
            "refresh_rounds": ",".join(str(int(x)) for x in refreshes["global_round"].tolist()),
            "updated_users_sum": int(updates["n_updated"].sum()) if len(updates) else 0,
            "added_sum": int(updates["added_round_sum"].sum()) if len(updates) else 0,
            "rebuild_turnover_mean": (
                round(float(updates[turnover_col].dropna().mean()), 4)
                if turnover_col and len(updates) else float("nan")
            ),
            "ecr_nonzero_rounds": int((g["ecr_round_mean"].fillna(0) > 0).sum()),
            "n_edges_last": int(g["n_edges"].iloc[-1]),
            "active_coverage_mean": round(float(g["active_coverage"].mean()), 4),
            "n_covered_users": int(len(covered)),
            "median_first_covered": (
                float(covered["first_covered_round"].median()) if len(covered) else float("nan")
            ),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False))
    out = args.root / f"phase2_records_{args.dataset.replace(' ', '_')}_seed{args.seed}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n表已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
