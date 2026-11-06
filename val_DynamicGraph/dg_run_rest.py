#!/usr/bin/env python
"""按「每卡实际进程数」均衡分派，把 Amazon 还没跑的任务跑完。

为什么需要它：`dg_run_amazon.sh` 用 `IDX % 2` 分配 GPU，一旦某些 run 崩溃，
后续任务会继续按索引奇偶分派，实测出现过 **4 个进程全在一张卡、另一张卡 0%** 的偏斜
（单进程独占一卡 40 分钟 vs 3 个挤一卡 58 分钟）。这里改成每次启动前重新统计
每卡实际进程数，选少的那个。

用法：python dg_run_rest.py [--jobs 4]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

DG_DIR = Path(__file__).resolve().parent
RAW = DG_DIR / "results" / "raw"
LOGS = DG_DIR / "results" / "amazon_logs"
DATASET = "amazon-digital music"
SLUG = "amazon-digital_music"
SEEDS = (42, 777, 2024)

#: (arm_dir, 额外参数)
TASKS: List[Tuple[str, List[str]]] = [
    ("dynamic", ["--arm", "dynamic"]),
    ("dynamic_rho0.8", ["--arm", "dynamic", "--rho", "0.8", "--graph-state", "pool",
                        "--graph-new-edge", "zero"]),
    ("dynamic_rho0.8_warm", ["--arm", "dynamic", "--rho", "0.8", "--graph-state", "pool",
                             "--graph-new-edge", "warm"]),
    ("dynamic_rho0.8_topk_warm", ["--arm", "dynamic", "--rho", "0.8", "--graph-state", "topk",
                                  "--graph-new-edge", "warm"]),
    ("dynamic_rho0.8_topk", ["--arm", "dynamic", "--rho", "0.8", "--graph-state", "topk",
                             "--graph-new-edge", "zero"]),
    ("periodic", ["--arm", "periodic"]),
    ("static", ["--arm", "static"]),
    ("nograph", ["--arm", "nograph"]),
]


def arm_dir_of(cmd: str) -> str:
    def grab(flag: str, default: str) -> str:
        m = re.search(rf"--{flag} (\S+)", cmd)
        return m.group(1) if m else default

    name = grab("arm", "dynamic")
    rho = float(grab("rho", "0.2"))
    if abs(rho - 0.2) > 1e-9:
        name += f"_rho{rho:g}"
    if grab("graph-state", "pool") != "pool":
        name += f"_{grab('graph-state', 'pool')}"
    if grab("graph-new-edge", "zero") != "zero":
        name += f"_{grab('graph-new-edge', 'zero')}"
    return name


def running_tasks() -> Tuple[set, Dict[str, int]]:
    """返回 (正在跑的 (arm_dir, seed) 集合, 每卡进程数)。"""
    running = set()
    per_gpu: Dict[str, int] = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().decode(errors="ignore").replace("\0", " ")
        except Exception:
            continue
        if "dg_run.py" not in cmd or "amazon-digital" not in cmd:
            continue
        seed = re.search(r"--seed (\d+)", cmd)
        if not seed:
            continue
        arm = arm_dir_of(cmd)
        running.add((arm, int(seed.group(1))))
        try:
            env = (p / "environ").read_bytes().decode(errors="ignore")
            m = re.search(r"CUDA_VISIBLE_DEVICES=(\S+?)\0", env)
            gpu = m.group(1) if m else "?"
        except Exception:
            gpu = "?"
        per_gpu[gpu] = per_gpu.get(gpu, 0) + 1
    return running, per_gpu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    running, per_gpu = running_tasks()
    done = {(f.parent.parent.name, int(f.parent.name.replace("seed_", "")))
            for f in (RAW / SLUG).glob("*/seed_*/dg_summary.csv")}
    todo = [(a, s, extra) for a, extra in TASKS for s in SEEDS
            if (a, s) not in done and (a, s) not in running]

    print(f"已完成 {len(done)}/24，正在跑 {len(running)} 个（每卡 {per_gpu}），待启动 {len(todo)} 个")
    if not todo:
        print("没有需要启动的任务")
        return 0

    procs = []
    for arm, seed, extra in todo:
        while True:
            # 并发上限必须按「全局 dg_run.py 进程数」算，不能只算自己启动的
            running_now, per_gpu_now = running_tasks()
            mine_alive = sum(1 for p in procs if p.poll() is None)
            total = len(running_now) if running_now else mine_alive
            if total < args.jobs:
                break
            time.sleep(10)
        _, per_gpu = running_tasks()
        gpu = min(("0", "1"), key=lambda g: per_gpu.get(g, 0))
        logf = LOGS / f"{arm}_seed{seed}.rest.log"
        cmd = [sys.executable, "-X", "utf8", str(DG_DIR / "dg_run.py"),
               "--dataset", DATASET, "--seed", str(seed),
               "--num-windows", "5", "--rounds-per-window", "120",
               "--participation-rate", "0.1", "--probe-size", "256",
               "--personal-on-gpu", "--device", "cuda", "--out-root", str(RAW)] + extra
        env = {"CUDA_VISIBLE_DEVICES": gpu, "PATH": "/usr/bin:/bin"}
        import os
        full_env = dict(os.environ)
        full_env.update(env)
        fh = logf.open("w")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=full_env,
                             cwd=str(DG_DIR))
        procs.append(p)
        print(f"  [{time.strftime('%H:%M:%S')}] 启动 GPU{gpu} | {arm} | seed {seed} "
              f"(该卡现有 {per_gpu.get(gpu, 0) + 1} 个)", flush=True)
        time.sleep(4)

    # 等所有（含先前就在跑的）训练进程结束
    while True:
        running_now, _ = running_tasks()
        if not running_now and all(p.poll() is not None for p in procs):
            break
        time.sleep(30)
    fails = sum(1 for p in procs if p.returncode != 0)
    print(f"全部结束；失败 {fails} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
