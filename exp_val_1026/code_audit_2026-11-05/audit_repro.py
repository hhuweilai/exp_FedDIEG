#!/usr/bin/env python
"""代码审计复现脚本（2026-11-05）。

用途：用最小合成输入，直接调用 `val01_run_staleness.py` / `val02_run_ema.py` 里
**未被修改的原始函数**，复现 4 个实现缺陷。不训练模型、不读数据集。

  python exp_val_1026/code_audit_2026-11-05/audit_repro.py

四组复现：
  A. scatter_ 把 -1 padding 钳到列 0 → 覆盖有效历史边权（val02 / val03）
  B. 无效候选位置没有写回 -1 → graph_neighbors 里全是合法 user id（val02 / val03）
  C. EMA 混用"归一化边权"与"原始余弦" → 同一阈值下两套数值尺度（val01）
  D. 新边初值 = (1-rho)*sim 与 min_sim 的相互作用 → 新邻居进不了 Top-K

注意：本脚本只依赖 numpy + torch（pandas 用空壳顶掉，因为被测函数不需要它）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 被测模块在文件头 import pandas，但被测函数本身不用；用空壳占位以免额外依赖
if "pandas" not in sys.modules:
    sys.modules["pandas"] = types.ModuleType("pandas")

from exp_val_1026.exp01_staleness_participation.val01_run_staleness import (  # noqa: E402
    update_graph as val01_update_graph,
)
from exp_val_1026.exp02_ema_interest_drift.val02_run_ema import (  # noqa: E402
    update_graph as val02_update_graph,
)

D = 8  # 摘要维度（只为让余弦可控，与真实 64 维无关）


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def make_bank(n_users: int, groups: int, seed: int, noise: float = 0.02) -> torch.Tensor:
    """构造 n_users 个摘要：分成 groups 个兴趣簇，簇内高相似、簇间低相似。"""
    gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(groups, D, generator=gen)
    centers = F.normalize(centers, p=2, dim=1)
    labels = torch.arange(n_users) % groups
    bank = centers[labels] + noise * torch.randn(n_users, D, generator=gen)
    return F.normalize(bank, p=2, dim=1) * 1.0


# --------------------------------------------------------------------------------------
# A. scatter_ 的 -1 → 0 别名污染
# --------------------------------------------------------------------------------------
def _run_a(has_pad: bool):
    """构造最小场景：用户 0 是自己的第 0 个邻居；用户 1 的历史里含用户 0。

    has_pad=True 时用户 1 的最后一个槽位是 -1 padding（即"有效邻居不足 k 条"）。
    """
    users, k = 4, 3
    bank = make_bank(users, 4, seed=0, noise=0.02)
    active = torch.ones(users, dtype=torch.bool)
    pad = -1 if has_pad else 3
    graph_neighbors = torch.tensor(
        [
            [1, 2, 3],
            [0, 2, pad],     # <-- 用户 1 的旧邻居：用户 0、用户 2、(padding)
            [0, 1, 3],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )
    graph_weights = torch.full((users, k), 1.0 / k, dtype=torch.float32)
    graph_mask = torch.ones(users, dtype=torch.bool)
    smooth_state = torch.zeros((users, k))
    # 用户 1 对两个旧邻居的历史平滑相似度
    smooth_state[1, 0] = 0.50   # -> 用户 0
    smooth_state[1, 1] = 0.40   # -> 用户 2
    val02_update_graph(
        update_users=torch.arange(users),
        summary_bank=bank,
        active_clients=active,
        graph_neighbors=graph_neighbors,
        graph_weights=graph_weights,
        graph_mask=graph_mask,
        smooth_state=smooth_state,
        topk=k,
        min_sim=0.0,     # 关掉阈值，只观察 EMA
        rho=0.8,
        block_size=64,
        gate_on="raw",
        new_edge_init="ema",
    )
    norm = F.normalize(bank, p=2, dim=1)
    sim_to0 = float(torch.matmul(norm[1], norm[0]))
    return graph_neighbors[1].tolist(), smooth_state[1].tolist(), sim_to0


def repro_scatter_alias() -> None:
    banner("A. val02/val03.update_graph：scatter_ 的 -1 -> 0 钳位覆盖有效历史边权")

    nb_pad, ss_pad, sim_to0 = _run_a(has_pad=True)
    nb_ok, ss_ok, _ = _run_a(has_pad=False)

    print("三行分别是：用户 1 更新后的邻居 / smooth_state（即 EMA 状态）")
    print(f"  含 -1 padding 槽 : neighbors={nb_pad}  smooth={[round(v, 4) for v in ss_pad]}")
    print(f"  无 padding（对照）: neighbors={nb_ok}  smooth={[round(v, 4) for v in ss_ok]}")
    print(f"  用户 1 与用户 0 的当前余弦 = {sim_to0:.4f}")
    print()
    print("预期（正确 EMA）：0.8*0.50 + 0.2*sim =", round(0.8 * 0.5 + 0.2 * sim_to0, 4))
    print("对照组的实测值          =", round(max(ss_ok), 4))
    print("含 padding 组的实测值   =", round(max(ss_pad), 4), "  <- 历史分量被 0 顶掉了")
    print()
    print("原因：old_col = active_slot[old_ids.clamp_min(0)]，-1 被钳到用户 0 的列，")
    print("      再用 scatter_ 写入 src = old_weights * has_prev = 0，")
    print("      与'用户 0 本来就是有效旧邻居'的那一列重复 -> 0 覆盖真实历史，")
    print("      同时 prev_mask 该列被写成 0 -> 该候选被当成'全新边'处理。")

    idx = torch.tensor([[1, 3, 0, 0]])   # 列 0 出现两次（真实邻居 + padding）
    src = torch.tensor([[0.80, 0.60, 0.0, 0.0]])
    base = torch.zeros(1, 5)
    base[0, 0] = 0.50                    # 列 0 上一轮的真实历史
    out = base.clone().scatter_(1, idx, src)
    print(f"\ntorch 语义检查：base[0,0]=0.5, scatter_(index=[[.., 0, 0]], src=[[.., 0, 0]])")
    print(f"  -> {out.tolist()}   列 0 从 0.5 变成 {out[0, 0].item():.2f}（后者获胜）")


# --------------------------------------------------------------------------------------
# B. 无效候选没有写回 -1
# --------------------------------------------------------------------------------------
def repro_missing_minus_one() -> None:
    banner("B. val02.update_graph：无效候选未标记 -1，graph_neighbors 恒为合法 user id")

    users, topk = 8, 3
    bank = make_bank(users, users, seed=1, noise=0.05)  # 每簇 1 人 → 余弦接近 0
    active = torch.ones(users, dtype=torch.bool)
    graph_neighbors = torch.full((users, topk), -1, dtype=torch.long)
    graph_weights = torch.zeros((users, topk))
    graph_mask = torch.zeros(users, dtype=torch.bool)
    smooth_state = torch.zeros((users, topk))

    # min_sim 故意设成 2.0（不可能达到）→ 所有候选都不满足"存在"条件
    upd, ecr = val02_update_graph(
        update_users=torch.arange(users),
        summary_bank=bank,
        active_clients=active,
        graph_neighbors=graph_neighbors,
        graph_weights=graph_weights,
        graph_mask=graph_mask,
        smooth_state=smooth_state,
        topk=topk,
        min_sim=2.0,
        rho=0.8,
        block_size=64,
        gate_on="raw",
        new_edge_init="ema",
    )
    print("min_sim=2.0 ⇒ 没有任何候选应该成为邻居")
    print("graph_mask (全部应为 False):", graph_mask.tolist())
    print("graph_weights (全部应为 0):", graph_weights.tolist())
    print("graph_neighbors (应为 全 -1):")
    for u in range(users):
        print(f"  u{u}: {graph_neighbors[u].tolist()}")
    bad = int((graph_neighbors >= 0).sum())
    print(f"\n仍然 >=0（=被下游计为'有效邻居'）的槽位数：{bad} / {users * topk}")
    print("下游用法：probe/CohortEdgeProbe 里 n_valid = (nbr >= 0).sum(dim=1)")
    print("  ⇒ 这些伪 id 会被统计成真实邻居，n_neighbors / turnover 的 Jaccard 都被污染")
    print(f"\n对照：val01 的实现会把无效槽位补 -1（val01_run_staleness.py L304）")


# --------------------------------------------------------------------------------------
# C. 归一化边权 × 原始余弦 的尺度混用（val01）
# --------------------------------------------------------------------------------------
def repro_scale_mix_val01() -> None:
    banner("C. val01.update_graph：EMA 混用归一化边权（Σ=1）与原始余弦（0~1 同尺度于 cos）")

    users, k = 6, 4
    bank = make_bank(users, 2, seed=2, noise=0.03)
    active = torch.ones(users, dtype=torch.bool)

    graph_neighbors = torch.tensor(
        [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 2, 3]],
        dtype=torch.long,
    )
    # 历史边权 = 上一轮 renormalize 之后的和为 1 的相对权重
    graph_weights = torch.full((users, k), 1.0 / k)
    graph_mask = torch.ones(users, dtype=torch.bool)

    norm = F.normalize(bank, p=2, dim=1)
    sim_row = torch.matmul(norm[0], norm.T).clamp_min(0.0)
    print("用户 0 的原始余弦 sim（小数点后 3 位）：",
          [round(float(v), 3) for v in sim_row.tolist()])

    ecr = val01_update_graph(
        update_users=torch.arange(users),
        summary_bank=bank,
        active_clients=active,
        graph_neighbors=graph_neighbors,
        graph_weights=graph_weights,
        graph_mask=graph_mask,
        topk=k,
        min_sim=0.20,      # 与 val01 默认一致
        edge_ema=0.80,     # 与 val01 默认一致
        block_size=64,
    )
    print(f"\nECR = {ecr}")
    print("更新后 graph_weights（每行 sum 应 = 1，注意这是'相对权重'）：")
    for u in range(users):
        print(f"  u{u}: {[round(float(v), 4) for v in graph_weights[u].tolist()]}"
              f"  sum={float(graph_weights[u].sum()):.4f}  "
              f"n_keep={int((graph_neighbors[u] >= 0).sum())}")

    print("\n混合公式（val01 L279）：blended = (1-rho)*sim + rho*prev")
    print("  prev 的量纲是'和为 1 的相对权重'，K=10 时为 1/K = 0.10；")
    print("  sim  的量纲是'余弦相似度'（论文里 Δe = max(0,cos)），典型值 0.25~0.35。")
    print("  阈值 min_sim = 0.20 却同时作用在 mixed 值上：\n")
    print("   rho   blended(prev=1/10, sim=0.30)   是否过 min_sim=0.20")
    print("   " + "-" * 52)
    for rho in (0.0, 0.5, 0.8, 0.95):
        prev, sim = 1.0 / 10, 0.30
        blended = (1.0 - rho) * sim + rho * prev
        keep = "保留" if blended >= 0.20 else "被阈值丢弃 <<<"
        print(f"   {rho:<5} {blended:>10.4f}                    {keep}")
    print("\n  论文定义：e^(t+1) = rho*e^(t) + (1-rho)*Δe，其中 Δe 与 e 同为 cos 量纲，")
    print("  阈值只用于'是否存在边'，平滑与阈值在同一量纲上。代码把 e^(t) 换成了")
    print("  归一化后的相对权重，于是 rho 越大、blended 越被拉到 1/K 量级，")
    print("  阈值 0.20 逐步变成'只保留少数几条最强势的边'——图不是'变平滑'，是'被削掉'。")


# --------------------------------------------------------------------------------------
# D. 新边初值 (1-rho)*sim 与 Top-K / 阈值的相互作用
# --------------------------------------------------------------------------------------
def repro_new_edge_penalty() -> None:
    banner("D. 新边初值 (1-rho)*sim：高 rho 下新邻居天然进不了 Top-K")

    print("EMA 从 0 起步时，新边第 t 轮的值 = (1-rho) * sim * Σ_{i<t} rho^i")
    print("要达到与收敛旧边同量级（≈ sim），需要的轮数：")
    for rho in (0.0, 0.5, 0.8, 0.95):
        if rho == 0:
            print(f"  rho=0.00 -> 0 轮（无记忆，直接等于 sim）")
            continue
        # 求最小 t 使 1 - rho^t >= 0.9
        t = 1
        while 1 - rho ** t < 0.9:
            t += 1
        print(f"  rho={rho:.2f} -> 约 {t} 轮才能达到收敛值的 90%")

    print("\n在当前代码里新边还要同时过两道门：(1) 是否进 Top-K；(2) >= min_sim")
    print("val02 默认 min_sim=0.20、gate_on='raw'（阈值看 cos），因此在 Top-K 排序里：")
    sim_old = 0.35   # 旧边收敛值（cos 尺度）
    for rho in (0.0, 0.5, 0.8, 0.95):
        new_val = (1.0 - rho) * sim_old
        print(f"  rho={rho:<4} 新边首轮值={new_val:.4f}  旧边≈{sim_old:.2f} -> "
              f"{'可竞争' if new_val > 0.9 * sim_old else '排序上必输，必须等旧边自己衰减'}")

    # 用真实函数验证：新邻居余弦更高，但在 rho>0 时是否仍被旧邻居压住
    print("\n真实函数验证（val02.update_graph）：用户 0 有 2 条旧边 cos=0.70，")
    print("候选里另有 2 条新边 cos=0.90（更高），K=2：")
    D_ = 8
    e = torch.eye(D_)
    def unit(v):
        return F.normalize(v, p=2, dim=0)
    bank = torch.stack([
        unit(e[0]),
        unit(0.70 * e[0] + (1 - 0.49) ** 0.5 * e[1]),      # 旧邻居，cos=0.70
        unit(0.70 * e[0] - (1 - 0.49) ** 0.5 * e[1]),      # 旧邻居，cos=0.70
        unit(0.90 * e[0] + (1 - 0.81) ** 0.5 * e[2]),      # 新邻居，cos=0.90
        unit(0.90 * e[0] - (1 - 0.81) ** 0.5 * e[2]),      # 新邻居，cos=0.90
    ])
    norm = F.normalize(bank, p=2, dim=1)
    print("  用户 0 对各候选的余弦：",
          [round(float(v), 4) for v in torch.matmul(norm[0], norm.T).tolist()])

    for init_mode in ("ema", "warm"):
        for rho in (0.0, 0.5, 0.8, 0.95):
            users, k = 5, 2
            gn = torch.tensor([[[1, 2], [0, 2], [0, 1], [0, 1], [0, 1]][u] for u in range(5)],
                              dtype=torch.long)
            gw = torch.tensor([[0.5, 0.5], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
            gm = torch.ones(users, dtype=torch.bool)
            ss = torch.tensor([[0.70, 0.70], [0.70, 0.0], [0.70, 0.0], [0.70, 0.0], [0.70, 0.0]])
            ac = torch.ones(users, dtype=torch.bool)
            val02_update_graph(
                update_users=torch.arange(users),
                summary_bank=bank,
                active_clients=ac,
                graph_neighbors=gn,
                graph_weights=gw,
                graph_mask=gm,
                smooth_state=ss,
                topk=k,
                min_sim=0.20,
                rho=rho,
                block_size=8,
                gate_on="raw",
                new_edge_init=init_mode,
            )
            picked = gn[0].tolist()
            is_new = all(p in (3, 4) for p in picked)
            print(f"  new_edge_init={init_mode:<5} rho={rho:<5} -> 用户0 的 Top-2 = {picked}"
                  f"  {'（换成了余弦更高的新邻居）' if is_new else '（仍被旧邻居占住 -> 图冻结）'}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    repro_scatter_alias()
    repro_missing_minus_one()
    repro_scale_mix_val01()
    repro_new_edge_penalty()
    print("\n全部复现完成。")
