#!/usr/bin/env python
"""复现 E：图冻结来自 (1-rho) 初始化，还是来自"EMA 状态只存当前 Top-K"？

论文公式 e^(t+1)_ij = rho*e^(t)_ij + (1-rho)*Δe^(t)_ij 是对**所有点对**定义的。
两种落地方式：

  truncated（= val02 / val03 的 update_graph 实际做法）
      只保存每个用户当前 Top-K 的 smooth_state。掉出 Top-K 的边，其 EMA 状态被丢弃，
      下一轮只能重新从 (1-rho)*Δe 起步。
  full（= 论文公式字面实现）
      保存 N×N 的 e 矩阵，掉出 Top-K 的边继续按公式累积；Δe 恒定时 e_t -> Δe。

本脚本不训练模型、不用真实数据，直接构造受控的相似度矩阵，
让"候选换不换人"这件事可以被精确解读（这是一个机制实验，不是数据实验）。

  python exp_val_1026/code_audit_2026-11-05/audit_freeze_origin.py
"""

from __future__ import annotations

import torch

N_A = 60                 # 旧兴趣组 A：用户 0..59
N_B = 60                 # 新兴趣组 B：用户 60..119
N = N_A + N_B
K = 10
MIN_SIM = 0.20
ROUNDS = 120
SWITCH_ROUND = 31
DRIFTERS = list(range(10))          # A 组里发生兴趣漂移的 10 个用户

SIM_AA = 0.62              # A 组内部相似度
SIM_AB = 0.40              # A 组与 B 组之间的相似度
SIM_DRIFT_OLD = 0.50       # 漂移用户与"旧同伴"的相似度（下降）
SIM_DRIFT_NEW = 0.62       # 漂移用户与"新兴趣组 B"的相似度（只高 24%）


def build_sim(drifted: bool) -> torch.Tensor:
    """构造一个受控的余弦相似度矩阵（对称、对角线为 0）。"""
    sim = torch.zeros(N, N)
    a = torch.arange(N_A)
    b = torch.arange(N_A, N)
    # A 组内部
    sim[a[:, None], a[None, :]] = SIM_AA
    # A-B 之间
    sim[a[:, None], b[None, :]] = SIM_AB
    sim[b[:, None], a[None, :]] = SIM_AB
    # B 组内部
    sim[b[:, None], b[None, :]] = SIM_AA
    if drifted:
        d = torch.tensor(DRIFTERS)
        # 漂移用户 -> 旧同伴：降到 0.50；-> 新兴趣组 B：升到 0.62
        sim[d[:, None], a[None, :]] = SIM_DRIFT_OLD
        sim[a[:, None], d[None, :]] = SIM_DRIFT_OLD
        sim[d[:, None], b[None, :]] = SIM_DRIFT_NEW
        sim[b[:, None], d[None, :]] = SIM_DRIFT_NEW
        sim[d[:, None], d[None, :]] = SIM_DRIFT_OLD
    sim.fill_diagonal_(0.0)
    # 打破并列，让 Top-K 排序确定（1e-4 量级，不改变结论）
    gen = torch.Generator().manual_seed(0)
    sim = sim + 1e-4 * torch.rand(N, N, generator=gen)
    return sim.clamp_min(0.0)


def run(rho: float, variant: str):
    """返回 (每轮全体 ECR, 每轮漂移用户邻居里"属于新兴趣组 B"的比例)。"""
    sim = build_sim(drifted=False)
    exists = sim >= MIN_SIM

    state_ids = torch.full((N, K), -1, dtype=torch.long)
    state_vals = torch.zeros((N, K))
    state_full = torch.zeros((N, N))
    prev_sets = [set() for _ in range(N)]
    ecr_hist, new_share = [], []

    for r in range(1, ROUNDS + 1):
        if r == SWITCH_ROUND:
            sim = build_sim(drifted=True)
            exists = sim >= MIN_SIM

        if variant == "truncated":
            if rho > 0:
                cand = (1.0 - rho) * sim
                for u in range(N):
                    if not bool((state_ids[u] >= 0).any()):
                        cand[u] = sim[u]                 # 首次建立邻居：warm start
                        continue
                    for slot in range(K):
                        nid = int(state_ids[u, slot])
                        if nid < 0:
                            continue
                        cand[u, nid] = rho * float(state_vals[u, slot]) + (1.0 - rho) * float(sim[u, nid])
            else:
                cand = sim.clone()
            cand_gated = torch.where(exists, cand, torch.full_like(cand, -1e9))
            topv, topp = torch.topk(cand_gated, K, dim=1)
            keep = topv > -1e8
            state_ids = topp.clone()
            state_vals = torch.where(keep, topv, torch.zeros_like(topv))
        else:
            if rho > 0:
                state_full = rho * state_full + (1.0 - rho) * sim
            else:
                state_full = sim.clone()
            cand_gated = torch.where(exists, state_full, torch.full_like(state_full, -1e9))
            topv, topp = torch.topk(cand_gated, K, dim=1)
            keep = topv > -1e8

        ecr = 0.0
        for u in range(N):
            cur = set(int(v) for v, k in zip(topp[u].tolist(), keep[u].tolist()) if k)
            union = cur | prev_sets[u]
            if union:
                ecr += len(cur ^ prev_sets[u]) / len(union)
            prev_sets[u] = cur
        ecr_hist.append(ecr / N)

        hits = cnt = 0
        for u in DRIFTERS:
            for v, k in zip(topp[u].tolist(), keep[u].tolist()):
                if k:
                    hits += int(v >= N_A)          # 邻居是不是"新兴趣组 B"的成员
                    cnt += 1
        new_share.append(hits / max(cnt, 1))

    return ecr_hist, new_share


def main() -> None:
    print("受控机制实验（无数据、无模型，只比较两种 EMA 状态的存放方式）")
    print(f"  {N} 个节点：A 组 0..{N_A-1}，B 组 {N_A}..{N-1}；K={K}，min_sim={MIN_SIM}")
    print(f"  第 {SWITCH_ROUND} 轮：{len(DRIFTERS)} 个 A 组用户漂移到 B 的兴趣")
    print(f"    与旧同伴相似度 {SIM_AA} -> {SIM_DRIFT_OLD}，与新兴趣组相似度 {SIM_AB} -> {SIM_DRIFT_NEW}")
    print(f"    （新邻居只比旧邻居高 {(SIM_DRIFT_NEW/SIM_DRIFT_OLD-1)*100:.0f}% —— 这是真实数据里的典型量级）")
    print()
    head = "  rho    变体        切换前ECR  切换轮ECR  切换后10轮ECR  R+10新邻居占比  R+40    R+100"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for rho in (0.0, 0.5, 0.8, 0.95):
        for variant in ("truncated", "full"):
            ecr, share = run(rho, variant)
            pre = sum(ecr[: SWITCH_ROUND - 1]) / (SWITCH_ROUND - 1)
            at_switch = ecr[SWITCH_ROUND - 1]
            after = sum(ecr[SWITCH_ROUND: SWITCH_ROUND + 10]) / 10
            print(f"  {rho:<6} {variant:<11} {pre:.5f}    {at_switch:.5f}    {after:.5f}        "
                  f"{share[min(SWITCH_ROUND + 9, ROUNDS - 1)]:.3f}            "
                  f"{share[min(SWITCH_ROUND + 39, ROUNDS - 1)]:.3f}   "
                  f"{share[min(SWITCH_ROUND + 99, ROUNDS - 1)]:.3f}")
    print()
    print("读法：")
    print("  * truncated：rho>=0.5 时切换后 ECR 立刻回到 ~0，漂移用户的邻居几乎不换人")
    print("    物理原因：新边每轮都从 (1-rho)*0.62 起步，而旧边早已收敛在 ~0.62；")
    print("    要换人必须满足 0.62*(1-rho) > 旧边值，即相似度差距需 > 1/(1-rho) 倍 ——")
    print("    rho=0.5 要 2 倍、rho=0.8 要 5 倍、rho=0.95 要 20 倍，余弦相似度根本给不出这个差距。")
    print("  * full：同样的 rho，ECR 保持非零，新邻居在 O(1/(1-rho)) 轮内替换掉旧邻居。")
    print("  => '图冻结'不是 EMA 公式的固有性质，而是'只保存当前 Top-K 的 EMA 状态'引入的。")


if __name__ == "__main__":
    main()
