#!/usr/bin/env python
"""一键自检：用极小配置把 val01 / val02 / val03 各跑一遍，验证"可跑通"。

它做三件事：
  1. 先跑 `feddieg_core/verify_fixes.py` 的 27 项单元断言；
  2. 依次以 `--smoke` 跑三个验证实验（每个几十秒内）；
  3. 检查每个实验的 4 个输出文件是否存在、表头是否包含关键列，并打印一行汇总。

输出统一写到 `<results>/raw_smoke/...`，不会污染正式结果目录。

    F:\\Anaconda\\python.exe -X utf8 exp_val_1026/run_smoke_test.py
    F:\\Anaconda\\python.exe -X utf8 exp_val_1026/run_smoke_test.py --device cuda

注意：默认用 `F:\\Anaconda\\python.exe`（装了 pandas 的解释器）。脚本会用
`sys.executable` 去跑子进程，所以先用哪个解释器启动它，子进程就用哪个。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL_ROOT = ROOT / "exp_val_1026"

JOBS = [
    {
        "name": "val01 陈旧度 × 参与率",
        "script": VAL_ROOT / "exp01_staleness_participation" / "val01_run_staleness.py",
        "args": ["--smoke", "--mode", "upload_only"],
        "out": VAL_ROOT / "exp01_staleness_participation" / "results" / "raw_smoke",
        "expect": ["val01_summary.csv", "val01_staleness_buckets.csv", "config.json", "val01_run.log"],
        "required_columns": {
            "val01_summary.csv": ["avg_window_ndcg", "avg_ecr", "mean_staleness", "graph_state", "core_version"],
        },
    },
    {
        "name": "val02 固定 EMA × 兴趣漂移",
        "script": VAL_ROOT / "exp02_ema_interest_drift" / "val02_run_ema.py",
        "args": ["--smoke", "--rho", "0.2"],
        "out": VAL_ROOT / "exp02_ema_interest_drift" / "results" / "raw_smoke",
        "expect": ["val02_summary.csv", "val02_round_metrics.csv",
                   "val02_user_round_metrics.csv", "config.json", "val02_run.log"],
        "required_columns": {
            "val02_summary.csv": ["all_avg_ndcg", "avg_ecr", "edge_ema", "graph_state", "avg_added_edges"],
            "val02_round_metrics.csv": ["drift_ndcg", "ecr_all", "added_edges_mean", "n_trained"],
        },
    },
    {
        "name": "val03 持续漂移 vs 短期扰动",
        "script": VAL_ROOT / "exp03_transient_vs_sustained" / "val03_run_scenario.py",
        "args": ["--smoke", "--scenario", "transient"],
        "out": VAL_ROOT / "exp03_transient_vs_sustained" / "results" / "raw_smoke",
        "expect": ["val03_summary.csv", "val03_round_metrics.csv",
                   "val03_user_round_metrics.csv", "config.json", "val03_run.log"],
        "required_columns": {
            "val03_summary.csv": ["all_avg_ndcg", "avg_ecr", "n_inj", "graph_state"],
            "val03_round_metrics.csv": ["group", "ndcg", "turnover", "ecr_mean"],
        },
    },
]


def run_step(name: str, cmd: list[str]) -> tuple[bool, str, float]:
    print("\n" + "=" * 78)
    print(f"▶ {name}")
    print("  " + " ".join(str(c) for c in cmd))
    print("=" * 78)
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    elapsed = time.time() - start
    ok = proc.returncode == 0
    print(f"  → {'OK' if ok else f'失败（exit {proc.returncode}）'}  {elapsed:.1f}s")
    return ok, name, elapsed


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle), [])


def read_first_row(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def check_outputs(job: dict) -> tuple[bool, list[str]]:
    out_root: Path = job["out"]
    problems: list[str] = []

    # 找到本次 smoke 的输出目录（只有一层：<split>/<sub>/.../seed_x）
    found = list(out_root.rglob("val01_summary.csv")) if "val01" in job["script"].name else []
    if "val02" in job["script"].name:
        found = list(out_root.rglob("val02_summary.csv"))
    if "val03" in job["script"].name:
        found = list(out_root.rglob("val03_summary.csv"))
    if not found:
        return False, [f"没有在 {out_root} 下找到 summary 文件"]

    run_dir = found[0].parent
    for name in job["expect"]:
        if not (run_dir / name).exists():
            problems.append(f"缺少输出文件 {name}")
    for name, columns in job.get("required_columns", {}).items():
        path = run_dir / name
        if not path.exists():
            continue
        header = read_header(path)
        missing = [c for c in columns if c not in header]
        if missing:
            problems.append(f"{name} 缺列 {missing}")
    return (len(problems) == 0), problems


def main() -> int:
    parser = argparse.ArgumentParser(description="FedDIEG 修复后的一键自检")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu",
                        help="smoke 默认 cpu，避免占用 GPU")
    parser.add_argument("--skip-unit", action="store_true", help="跳过 feddieg_core 单元断言")
    args = parser.parse_args()

    results: list[tuple[bool, str, float]] = []

    if not args.skip_unit:
        results.append(
            run_step(
                "feddieg_core 单元断言（27 项）",
                [sys.executable, "-X", "utf8", str(ROOT / "feddieg_core" / "verify_fixes.py")],
            )
        )

    for job in JOBS:
        cmd = [sys.executable, "-X", "utf8", str(job["script"]), *job["args"], "--device", args.device]
        ok, name, elapsed = run_step(job["name"], cmd)
        if ok:
            ok, problems = check_outputs(job)
            if problems:
                ok = False
                print("  → 输出检查失败：")
                for p in problems:
                    print(f"     - {p}")
            else:
                summary_file = next(iter(job["out"].rglob("val0*_summary.csv")), None)
                if summary_file is not None:
                    row = read_first_row(summary_file)
                    keys = [k for k in ("avg_window_ndcg", "all_avg_ndcg", "avg_ecr") if k in row]
                    print("  → 输出检查通过：" + ", ".join(f"{k}={row[k][:8]}" for k in keys))
        results.append((ok, name, elapsed))

    print("\n" + "=" * 78)
    print("自检汇总")
    print("=" * 78)
    for ok, name, elapsed in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {elapsed:.1f}s")
    failed = [name for ok, name, _ in results if not ok]
    if failed:
        print(f"\n失败 {len(failed)} 项：{', '.join(failed)}")
        return 1
    print("\n全部通过。可以开始正式运行（见各脚本 docstring 的用法示例）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
