#!/usr/bin/env python
"""给动态图更新计时，用来估算一次正式运行的耗时。

    F:\\Anaconda\\python.exe -X utf8 feddieg_core/bench_graph.py
    F:\\Anaconda\\python.exe -X utf8 feddieg_core/bench_graph.py --dataset "amazon-video games" --device cuda

它只跑「服务器端图更新」这一步（每轮都要做一次），不训练模型。
客户端本地训练那部分的开销与 `--participation-rate` × N 成正比，可用
「每客户端前向耗时 × m × 总轮数」估。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feddieg_core.config import FedDIEGConfig
from feddieg_core.graph import DynamicInterestGraph
from feddieg_core.utils import make_device


def main() -> int:
    parser = argparse.ArgumentParser(description="动态图更新的耗时基准")
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--state", choices=["pool", "topk"], default="pool")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    meta = json.loads(
        (ROOT / "datasets" / "processed" / args.dataset / "meta.json").read_text(encoding="utf-8")
    )
    num_users = int(meta["num_users"])
    device = make_device(args.device)
    cfg = FedDIEGConfig(
        dataset=args.dataset,
        latent_dim=args.latent_dim,
        neighbors=args.neighbors,
        rho=args.rho,
        graph_state=args.state,
        graph_warmup_rounds=0,
    )
    graph = DynamicInterestGraph(cfg, num_users, device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    # 注意：CPU 的 Generator 只能配合 CPU 张量，所以先造在 CPU 再搬到 device
    centers = F.normalize(torch.randn(8, args.latent_dim, generator=gen), p=2, dim=1).to(device)
    noise = 0.05 * torch.randn(num_users, args.latent_dim, generator=gen)
    labels = torch.arange(num_users) % 8
    bank = F.normalize(centers.cpu()[labels] + noise, p=2, dim=1).to(device)
    active = torch.ones(num_users, dtype=torch.bool, device=device)
    users = torch.arange(num_users, device=device)

    print(f"数据集={args.dataset}  N={num_users}  d={args.latent_dim}  K={args.neighbors}")
    print(f"图状态={args.state}  候选池={graph.pool_size}  device={device}")
    graph.update(bank, active, users, round_idx=1)  # 预热
    times = []
    for r in range(2, 2 + args.repeats):
        start = time.time()
        stats = graph.update(bank, active, users, round_idx=r)
        times.append(time.time() - start)
    mean_t = sum(times) / len(times)
    print(f"单轮图更新耗时：{mean_t * 1000:.1f} ms（{args.repeats} 次均值，min {min(times)*1000:.1f} ms）")
    print(f"  本轮 ECR 均值={float(stats.ecr.mean()):.4f}，候选池填充率={stats.pool_fill:.2f}")
    print()
    print("按 5 窗口 × 120 轮 = 600 轮估算，图更新总耗时：")
    print(f"  ≈ {mean_t * 600 / 60:.1f} 分钟")
    print("再加上客户端本地训练：m = C·N 个客户端 × 每客户端一次前向/反向 × 600 轮。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
