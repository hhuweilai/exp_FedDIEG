#!/usr/bin/env python
"""复现 F：`val01` 的 EMA 尺度混用会把"图"变成"开关"（阈值作用在混合量纲上）。

`val01_run_staleness.py` L279 的混合公式是

    blended = (1-rho)*sim + rho*w        # sim 是余弦，w 是"和为 1"的相对权重

而 L293 的阈值 `min_sim=0.20` 作用在 `blended` 上。于是"一条边能不能留下"取决于

    (1-rho)*sim + rho*w >= min_sim    =>    sim >= (min_sim - rho*w) / (1 - rho)

w 不是常数（刚 warm start 重建时约 1/K，边越少 w 越大）。本脚本直接调用
`val01.update_graph`，在受控数据上观察 ρ 改变时图的实际形态。

  python exp_val_1026/code_audit_2026-11-05/audit_val01_gate.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "pandas" not in sys.modules:
    sys.modules["pandas"] = types.ModuleType("pandas")

from exp_val_1026.exp01_staleness_participation.val01_run_staleness import (  # noqa: E402
    update_graph,
)

N, K, MIN_SIM = 200, 10, 0.20
D = 64


def build(seed: int = 7) -> torch.Tensor:
    """4 个兴趣簇，簇内余弦约 0.33–0.50（真实摘要量级）。"""
    gen = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(4, D, generator=gen), p=2, dim=1)
    labels = torch.arange(N) % 4
    return F.normalize(centers[labels] + 0.25 * torch.randn(N, D, generator=gen), p=2, dim=1)


def main() -> None:
    bank = build()
    norm = F.normalize(bank, p=2, dim=1)
    print("受控数据：4 簇 × 50 用户，簇内余弦量级（用户 0 的前 12 名）：")
    print("  ", [round(float(v), 3) for v in torch.topk(norm[0] @ norm.T, 12).values.tolist()])
    print()
    print("阈值门槛 sim >= (min_sim - rho*w)/(1-rho)，取 w = 1/K = 0.10：")
    for rho in (0.0, 0.5, 0.8, 0.95):
        if rho == 0:
            print(f"  rho={rho:<5} 无阈值混合（warm start 直接用余弦）")
            continue
        bound = (MIN_SIM - rho * (1.0 / K)) / (1.0 - rho)
        print(f"  rho={rho:<5} 需要 sim >= {bound:6.3f}"
              f"{'   （余弦上确界 1.0，永远存不下边）' if bound > 1 else ''}")
    print()
    print("实测（直接调用 val01.update_graph，每轮更新全部用户，K=10，min_sim=0.20）：")
    print("  rho    第6轮边数  第20轮边数  第6~20轮 min~max   末轮ECR   形态")
    for rho in (0.0, 0.5, 0.8, 0.95):
        graph_neighbors = torch.full((N, K), -1, dtype=torch.long)
        graph_weights = torch.zeros(N, K)
        graph_mask = torch.zeros(N, dtype=torch.bool)
        active = torch.ones(N, dtype=torch.bool)
        ecr = 0.0
        k6 = k20 = 0.0
        trace = []
        for r in range(1, 21):
            ecr = update_graph(
                torch.arange(N), bank, active,
                graph_neighbors, graph_weights, graph_mask,
                K, MIN_SIM, rho, 512,
            )
            kept = float((graph_neighbors >= 0).sum(dim=1).float().mean())
            trace.append(kept)
            if r == 6:
                k6 = kept
            if r == 20:
                k20 = kept
        tail = trace[5:]
        lo, hi = min(tail), max(tail)
        if rho == 0.0:
            shape = "稳定"
        elif hi - lo > 0.5:
            shape = "在 0 附近与满 K 之间振荡（每两轮清零一次）"
        elif k20 == 0.0:
            shape = "整行被清空"
        else:
            shape = "边数被压窄后冻结"
        print(f"  {rho:<6} {k6:>9.2f} {k20:>10.2f} {lo:>10.2f}~{hi:<5.2f} {ecr:>10.4f}   {shape}")

    print()
    print("单用户轨迹（rho=0.8）：")
    graph_neighbors = torch.full((N, K), -1, dtype=torch.long)
    graph_weights = torch.zeros(N, K)
    graph_mask = torch.zeros(N, dtype=torch.bool)
    active = torch.ones(N, dtype=torch.bool)
    for r in range(1, 7):
        ecr = update_graph(torch.arange(N), bank, active, graph_neighbors,
                           graph_weights, graph_mask, K, MIN_SIM, 0.80, 512)
        kept = int((graph_neighbors[0] >= 0).sum())
        sims = [round(float(norm[0] @ norm[int(j)]), 3) for j in graph_neighbors[0] if j >= 0][:3]
        print(f"  r{r}: 用户0 保留 {kept:2d} 条边  ECR={ecr:.4f}  前3条边余弦={sims}")

    print()
    print("结论：")
    print("  * rho=0.8 时'一条边能否留下'由 (min_sim - rho*w)/(1-rho) 与余弦量级的相对关系决定，")
    print("    而不是由'历史保留强度'决定；实测表现为'整行清零 -> warm start 重建 -> 再清零'的振荡。")
    print("  * 因此 val01 里 rho 扫描测到的不是'平滑强度'，而是'图有没有被阈值打空'。")
    print("  * 修法见审计报告 P0-3 / P0-4：EMA 状态用余弦量纲单独保存，阈值只判存在性。")


if __name__ == "__main__":
    main()
