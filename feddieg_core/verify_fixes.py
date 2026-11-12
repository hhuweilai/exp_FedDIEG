#!/usr/bin/env python
"""修复验证：逐条断言 `feddieg_core` 是否真的修掉了审计报告里的缺陷，
以及联邦协议 / 动态图 / 数据切分是否按论文语义工作。

全部是离线单元测试（不读数据集、不训练大模型），CPU 上几秒跑完：

    python -X utf8 feddieg_core/verify_fixes.py

分五组（具体条数见运行输出末尾的汇总）：
  A. 模型与个性化层划分（对齐 A/B 主程序）
  B. 动态图：量纲 / 无效位 / 无别名 / 候选池 / 全局更新 / **ECR 定义** / **Top-K 边界**
  C. 联邦协议：FedAvg 聚合 = 客户端参数均值；客户端之间互不干扰
  D. 摘要缓存：直接替换语义（论文）+ **上传时点（训练后重算）**
  E. 端到端 + **负采样的因果历史/回退**
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feddieg_core.config import FedDIEGConfig
from feddieg_core.data import (
    ClientWindow,
    causal_history_sets,
    history_sets_from_frame,
    training_history_from_fast,
)
from feddieg_core.evaluate import build_eval_model, evaluate_split
from feddieg_core.federated import FedTrainer, build_participation_schedule
from feddieg_core.graph import DynamicInterestGraph
from feddieg_core.model import (
    FedDIEGModel,
    bce_seq_loss,
    build_model,
    is_personal_param,
    split_state_keys,
)
from feddieg_core.summary_upload import (  # 摘要上传时点的统一实现（F7）
    assert_upload_summary,
    upload_summary_after_update,
)

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}" + (f"   {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}   {detail}")


def tiny_cfg(**overrides) -> FedDIEGConfig:
    base = dict(
        latent_dim=16,
        max_len=8,
        num_heads=2,
        num_blocks=1,
        dropout_rate=0.0,
        neighbors=5,
        graph_pool_size=0,
        graph_warmup_rounds=0,
        rho=0.5,
        min_sim=0.0,
        top_k=5,
        eval_batch_size=64,
        rounds=4,
        local_epochs=1,
        server_lr_init=1.0,
        server_lr_final=1.0,
        lambda_temp=0.0,
        lambda_struct=0.0,
    )
    base.update(overrides)
    cfg = FedDIEGConfig(**base)
    return cfg


def make_bank(n: int, dim: int, clusters: int, seed: int, noise: float = 0.05) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(clusters, dim, generator=gen), p=2, dim=1)
    labels = torch.arange(n) % clusters
    bank = centers[labels] + noise * torch.randn(n, dim, generator=gen)
    return F.normalize(bank, p=2, dim=1)


def make_window(num_users: int, num_items: int, max_len: int, seed: int = 0) -> ClientWindow:
    gen = torch.Generator().manual_seed(seed)
    seq = torch.randint(1, num_items + 1, (num_users, max_len), generator=gen)
    pos = torch.randint(1, num_items + 1, (num_users, max_len), generator=gen)
    mask = torch.ones(num_users, max_len)
    history = [np.arange(1, num_items + 1, dtype=np.int64) for _ in range(num_users)]
    eval_items = torch.randint(1, num_items + 1, (num_users, 6), generator=gen)
    eval_items[:, 0] = pos[:, -1]
    return ClientWindow(
        window_idx=1,
        num_users=num_users,
        num_items=num_items,
        seq=seq,
        pos=pos,
        mask=mask,
        history=history,
        val_users=torch.arange(num_users),
        val_seq=seq,
        val_items=eval_items,
        test_users=torch.arange(num_users),
        test_seq=seq,
        test_items=eval_items,
    )


# ======================================================================================
# A. 模型与个性化层划分
# ======================================================================================
def group_a() -> None:
    print("\n[A] 模型与个性化层划分（对齐 A/B 主程序）")
    cfg = tiny_cfg()
    model = build_model(50, cfg)
    personal, shared = split_state_keys(model)

    # A1：个性化层必须同时包含 attention 与 forward 的 LayerNorm（A 版的不对称已修）
    has_attn_ln = any("attention_layernorms" in k for k in personal)
    has_ffn_ln = any("forward_layernorms" in k for k in personal)
    check(
        "A1 attention/forward 的 LayerNorm 都在个性化侧",
        has_attn_ln and has_ffn_ln,
        f"attention_layernorms={has_attn_ln}, forward_layernorms={has_ffn_ln}",
    )

    # A2：共享层恰好是 item_emb / pos_emb / W_gnn / W_g
    shared_modules = sorted({k.split(".")[0] if not k.startswith("encoder") else "encoder." + k.split(".")[1]
                             for k in shared})
    check(
        "A2 共享层 = item_emb/pos_emb/W_gnn/W_g",
        set(shared_modules) == {"encoder.item_emb", "encoder.pos_emb", "W_gnn", "W_g"},
        f"实际 = {shared_modules}",
    )

    # A3：每个参数都能被判定（不认识的参数会 KeyError，说明判定是穷尽的）
    try:
        for k in list(model.state_dict().keys()):
            is_personal_param(k)
        a3 = True
        detail = f"personal={len(personal)}, shared={len(shared)}"
    except KeyError as exc:
        a3 = False
        detail = str(exc)
    check("A3 个性化/共享判定覆盖全部参数", a3, detail)

    # A4：邻居聚合的批量形式与 A/B 版 dim=0 形式数值等价
    torch.manual_seed(0)
    k, dim = 5, cfg.latent_dim
    nbr = torch.randn(k, dim)
    w = torch.softmax(torch.randn(k), dim=0)
    model.eval()
    with torch.no_grad():
        got = model.extract_long_term_interest(nbr, w, 1)  # (1, D)
        # A/B 版写法：messages = W_gnn(nbr); agg = sum(w.unsqueeze(-1)*messages, dim=0)
        messages = model.W_gnn(nbr)
        expected = torch.relu(torch.sum(w.unsqueeze(-1) * messages, dim=0)).unsqueeze(0)
    check(
        "A4 图消息聚合与 A/B 版逐元素等价",
        torch.allclose(got, expected, atol=1e-6),
        f"max|diff|={float((got - expected).abs().max()):.2e}",
    )

    # A5：投影后不是 NaN/Inf
    feats = model.forward(torch.randint(1, 50, (3, cfg.max_len)))
    check("A5 前向输出有限", bool(torch.isfinite(feats[0]).all()), "u_i 无 NaN/Inf")


# ======================================================================================
# B. 动态图
# ======================================================================================
def group_b() -> None:
    print("\n[B] 动态图：量纲 / -1 / scatter / 候选池 / 全局更新")

    # ---------------- F2：无效槽位必须是 -1
    cfg = tiny_cfg(neighbors=4, rho=0.5)
    num_users, dim = 40, cfg.latent_dim
    bank = make_bank(num_users, dim, clusters=8, seed=1, noise=0.4)
    graph = DynamicInterestGraph(cfg, num_users, torch.device("cpu"))
    active = torch.ones(num_users, dtype=torch.bool)
    # 只让 6 个客户端"活跃"：其他人摘要为零 → a_i=0
    bank_inactive = bank.clone()
    bank_inactive[6:] = 0.0
    stats = graph.update(bank_inactive, active, torch.arange(num_users), round_idx=1)
    ids = graph.ids
    has_minus_one = bool((ids < 0).any())
    all_valid_or_minus_one = bool(((ids >= 0) | (ids == -1)).all())
    # 有效槽位必须指向真正的活跃用户（<6）
    valid_ids = ids[ids >= 0]
    only_active = bool((valid_ids < 6).all())
    check(
        "B1 有效邻居不足时槽位写 -1（不是伪 id）",
        has_minus_one and all_valid_or_minus_one,
        f"含 -1 槽位={has_minus_one}",
    )
    check("B2 有效槽位只指向活跃客户端", only_active, f"max id={int(valid_ids.max()) if valid_ids.numel() else -1}")
    nbr, w, valid = graph.neighbors(torch.arange(num_users))
    n_valid = (nbr >= 0).sum(dim=1)
    check(
        "B3 n_neighbors 不再恒等于 K",
        bool((n_valid < cfg.neighbors).any()),
        f"有效邻居数分布={sorted({int(v) for v in n_valid.tolist()})} / K={cfg.neighbors}",
    )

    # ---------------- F1：EMA 量纲必须是 cos（0~1）
    cfg2 = tiny_cfg(neighbors=3, rho=0.5, graph_state="topk")
    bank2 = make_bank(num_users, dim, clusters=4, seed=2, noise=0.1)
    g2 = DynamicInterestGraph(cfg2, num_users, torch.device("cpu"))
    act2 = torch.ones(num_users, dtype=torch.bool)
    for r in range(1, 12):
        g2.update(bank2, act2, torch.arange(num_users), round_idx=r)
    vals = g2.pool_val[g2.pool_mask]
    check(
        "B4 EMA 值域在 [0,1]（与 max(0,cos) 同量纲）",
        bool((vals >= 0).all() and (vals <= 1.0 + 1e-5).all()),
        f"范围=[{float(vals.min()):.3f}, {float(vals.max()):.3f}]",
    )
    # 收敛后 e ≈ Δe（cos 量纲）
    norm = F.normalize(bank2, p=2, dim=1)
    u0 = 0
    nbr0 = g2.ids[u0][g2.ids[u0] >= 0]
    if nbr0.numel():
        cur = (norm[u0] @ norm[nbr0].T).clamp_min(0.0)
        stored = g2.pool_val[u0][g2.pool_mask[u0]]
        check(
            "B5 EMA 收敛到当前余弦量级（量纲一致）",
            abs(float(stored.max()) - float(cur.max())) < 0.15,
            f"EMA max={float(stored.max()):.3f} vs cos max={float(cur.max()):.3f}",
        )
    else:
        check("B5 EMA 收敛到当前余弦量级（量纲一致）", False, "没有邻居")

    # ---------------- F3：每条边的 EMA 只由"它自己的历史和当前 Δe"决定
    #   逐槽位核对：pool_val[row, slot] 必须等于
    #       ρ·(上一轮该 id 的值) + (1-ρ)·Δe[row, id]     （之前跟踪过）
    #       (1-ρ)·Δe[row, id]                            （本轮新出现）
    #   只要实现里存在"_1 被钳到别的列"这类别名，这条断言就会立刻失败。
    def ema_invariant() -> tuple[bool, str]:
        cfg3 = tiny_cfg(neighbors=4, rho=0.5, graph_state="pool", graph_warmup_rounds=0)
        n, dim_ = 30, 12
        bank_ = make_bank(n, dim_, clusters=6, seed=5, noise=0.05)
        g = DynamicInterestGraph(cfg3, n, torch.device("cpu"))
        act = torch.ones(n, dtype=torch.bool)
        g.update(bank_, act, torch.arange(n), round_idx=1)

        prev_ids = g.pool_ids.clone()
        prev_val = g.pool_val.clone()
        prev_mask = g.pool_mask.clone()

        bank2_ = make_bank(n, dim_, clusters=5, seed=6, noise=0.08)  # 摘要整体变化
        g.update(bank2_, act, torch.arange(n), round_idx=2)

        norm = F.normalize(bank2_, p=2, dim=1, eps=1e-8)
        delta = (norm @ norm.T).clamp_min(0.0)
        rho_ = 0.5
        checked = 0
        worst = 0.0
        for row in range(0, n, 3):
            for slot in range(g.pool_size):
                if not bool(g.pool_mask[row, slot]):
                    continue
                cid = int(g.pool_ids[row, slot])
                if cid < 0 or cid == row:
                    continue
                d = float(delta[row, cid]) if cid != row else 0.0
                hits = torch.nonzero(prev_mask[row] & (prev_ids[row] == cid), as_tuple=False)
                if hits.numel():
                    expected = rho_ * float(prev_val[row, int(hits[0])]) + (1 - rho_) * d
                else:
                    expected = (1 - rho_) * d
                worst = max(worst, abs(float(g.pool_val[row, slot]) - expected))
                checked += 1
        return checked > 0 and worst < 1e-5, f"核对 {checked} 个槽位, max|误差|={worst:.2e}"

    ok6, detail6 = ema_invariant()
    check("B6 每条边的 EMA 只由自身历史与当前 Δe 决定（无跨槽位别名）", ok6, detail6)

    # ---------------- F4：候选池让新邻居能进来；只存 Top-K 则冻结
    #   场景（与审计 audit_freeze_origin.py 同一构造）：漂移后旧同伴相似度 0.50、
    #   新兴趣组 0.62 —— 新邻居只比旧邻居高 24%，是真实数据里的典型量级。
    def drift_run(state: str, rho: float, rounds: int = 60):
        dim_ = 16
        n_a, n_b, k = 60, 60, 5
        n = n_a + n_b
        drifters = list(range(6))
        cfg4 = tiny_cfg(
            neighbors=k, rho=rho, graph_state=state, graph_warmup_rounds=0, min_sim=0.0,
            latent_dim=dim_, top_k=k,
        )
        gen = torch.Generator().manual_seed(11)
        e = torch.zeros(dim_, dim_)
        for i in range(dim_):
            e[i, i] = 1.0

        def vec(base: torch.Tensor, noise: float, seed: int) -> torch.Tensor:
            g2 = torch.Generator().manual_seed(seed)
            return F.normalize(base + noise * torch.randn(dim_, generator=g2), p=2, dim=0)

        bank_a = torch.stack([vec(e[0], 0.02, 100 + i) for i in range(n_a)])
        bank_b = torch.stack([vec(e[1], 0.02, 200 + i) for i in range(n_b)])

        def drifter_vec(i: int) -> torch.Tensor:
            v = torch.zeros(dim_)
            v[0] = 0.50   # 与旧同伴的余弦
            v[1] = 0.62   # 与新兴趣组的余弦
            v[2 + i] = math.sqrt(max(0.0, 1.0 - 0.25 - 0.3844))  # 正交补，保证单位长
            return F.normalize(v, p=2, dim=0)

        bank = torch.cat([bank_a, bank_b])
        for i in drifters:
            bank[i] = vec(e[0], 0.02, 300 + i)          # 漂移前：属于 A 组

        g = DynamicInterestGraph(cfg4, n, torch.device("cpu"))
        act = torch.ones(n, dtype=torch.bool)
        for r in range(1, rounds + 1):
            if r == 15:
                bank = bank.clone()
                for i in drifters:
                    bank[i] = drifter_vec(i)
            g.update(bank, act, torch.arange(n), round_idx=r)

        hits = 0
        for u in drifters:
            nbr = g.ids[u]
            hits += int((nbr >= n_a).sum().item())
        return hits / (len(drifters) * k)

    frozen = drift_run("topk", 0.8)
    pooled = drift_run("pool", 0.8)
    check(
        "B7 graph_state=topk 复现旧行为（部分漂移下新邻居进不来）",
        frozen <= 0.2,
        f"新兴趣组邻居占比={frozen:.2f}",
    )
    check(
        "B8 graph_state=pool 修掉冻结（同一场景下邻居换到新兴趣组）",
        pooled >= 0.8,
        f"新兴趣组邻居占比={pooled:.2f}",
    )

    # ---------------- F5：更新范围 all vs sampled
    cfg5 = tiny_cfg(graph_update_scope="all")
    g5 = DynamicInterestGraph(cfg5, num_users, torch.device("cpu"))
    act5 = torch.ones(num_users, dtype=torch.bool)
    st = g5.update(bank2, act5, torch.arange(num_users), round_idx=1)
    check(
        "B9 默认对全部活跃节点异步更新边（论文语义）",
        st is not None and int(st.updated.numel()) == num_users,
        f"更新行数={int(st.updated.numel()) if st else 0}/{num_users}",
    )
    check(
        "B10 图有 ECR 与 n_added 两个指标（可区分'慢'与'冻结'）",
        st is not None and st.ecr.numel() == st.n_added.numel() == st.updated.numel(),
        "ecr / n_added 形状一致",
    )

    # ---------------- F6：ECR 的 Jaccard 定义（2026-11-08 修正）
    #   * 旧集合为空、新集合非空 ⇒ 1.0（"从无到有"是 100% 变化）
    #   * 旧、新都为空 ⇒ 0.0
    #   * 邻居集合不变 ⇒ 0.0
    # 旧实现 `where(n_old > 0, ..., 0)` 把首次建边/reset 后重建都记成 0。
    st2 = g5.update(bank2, act5, torch.arange(num_users), round_idx=2)
    with_edges = g5.mask & (g5.ids >= 0).any(dim=1)
    sel = st.updated.to(torch.long)
    sel = sel[with_edges[sel]]
    ecr_first = float(st.ecr[with_edges[st.updated.to(torch.long)]].mean()) if sel.numel() else float("nan")
    sel2 = st2.updated.to(torch.long)
    sel2 = sel2[with_edges[sel2]]
    ecr_same = float(st2.ecr[with_edges[st2.updated.to(torch.long)]].mean()) if sel2.numel() else float("nan")
    check(
        "B11 ECR 定义正确（空→非空 = 1.0；邻居集合不变 = 0.0）",
        np.isfinite(ecr_first) and abs(ecr_first - 1.0) < 1e-6
        and np.isfinite(ecr_same) and abs(ecr_same) < 1e-6,
        f"首次建边 ECR={ecr_first:.3f}，重复更新 ECR={ecr_same:.3f}",
    )

    # ---------------- F9：候选数 < K 时的 Top-K 边界（不得重复邻居 / 权重和 >1）
    for n_small, k_small in ((2, 4), (3, 4)):
        cfgk = tiny_cfg(neighbors=k_small, graph_warmup_rounds=0, rho=0.2)
        gk = DynamicInterestGraph(cfgk, n_small, torch.device("cpu"))
        gen_k = torch.Generator().manual_seed(7)
        bankk = F.normalize(
            torch.randn(n_small, cfgk.latent_dim, generator=gen_k), p=2, dim=1
        )
        gk.update(
            bankk, torch.ones(n_small, dtype=torch.bool),
            torch.arange(n_small), round_idx=1,
        )
        dups, over = 0, 0
        for u in range(n_small):
            valid = [int(v) for v in gk.ids[u].tolist() if v >= 0]
            if len(valid) != len(set(valid)):
                dups += 1
            if float(gk.weights[u].sum()) > 1.0 + 1e-5:
                over += 1
        check(
            f"B12 候选数<K（N={n_small}, K={k_small}）：无重复邻居、权重和≤1",
            dups == 0 and over == 0,
            f"dup={dups}, sum>1={over}",
        )


# ======================================================================================
# C. 联邦协议
# ======================================================================================
def group_c() -> None:
    print("\n[C] 联邦协议：FedAvg 聚合与客户端隔离")
    cfg = tiny_cfg(rounds=4, neighbors=2, graph_warmup_rounds=10, server_lr_init=1.0, server_lr_final=1.0)
    num_users, num_items = 6, 30
    window = make_window(num_users, num_items, cfg.max_len, seed=3)
    trainer = FedTrainer(cfg, num_users, num_items, torch.device("cpu"))
    graph = DynamicInterestGraph(cfg, num_users, torch.device("cpu"))

    recorded: dict[int, dict[str, torch.Tensor]] = {}

    def hook(user, state):
        recorded[user] = state

    trainer.on_client_trained = hook
    before = {k: v.clone() for k, v in trainer.global_state.items()}
    sampled = [0, 1, 2]
    rng = np.random.default_rng(0)
    stats = trainer.train_round(sampled, window, graph, round_idx=1, rng=rng)

    # C1：FedAvg = 参与客户端的共享层参数均值（item_emb 除外，它走 server_lr 外推）
    ok = True
    detail = []
    for k in trainer.shared_keys:
        trained = torch.stack([recorded[u][k].float() for u in sampled]).mean(dim=0)
        got = trainer.global_state[k].float()
        if "item_emb" in k:
            expected = before[k].float() + cfg.server_lr(1) * (trained - before[k].float())
        else:
            expected = trained
        if not torch.allclose(got, expected, atol=1e-6):
            ok = False
            detail.append(f"{k}: max|diff|={float((got - expected).abs().max()):.2e}")
    check("C1 服务器聚合 = 客户端共享层参数的平均", ok, "; ".join(detail[:2]) or "全部一致")
    check("C2 本轮有 3 个客户端完成本地训练", stats.n_trained == 3, f"n_trained={stats.n_trained}")

    # C3：个性化层留在客户端（不同客户端互不相同、且未参与全局平均）
    p0 = trainer.personal_overrides[0]
    p1 = trainer.personal_overrides[1]
    diff = max(float((p0[k] - p1[k]).abs().max()) for k in p0)
    check("C3 个性化层留本地且各客户端不同", diff > 0, f"max|p0-p1|={diff:.2e}")

    # C4：未采样客户端仍用初值（与 A/B 版"每个客户端从同一份 global 复制"等价）
    check(
        "C4 未采样客户端读共享初值",
        trainer.personal_state_key(4) == "__init__",
        "user 4 未被采样",
    )

    # C5：客户端之间不共享本地训练过程（客户端隔离）
    #    再跑一轮，只有 client 3 参与；client 0 的个性化层不应改变
    p0_before = {k: v.clone() for k, v in trainer.personal_overrides[0].items()}
    stats2 = trainer.train_round([3], window, graph, round_idx=2, rng=rng)
    unchanged = all(torch.allclose(p0_before[k], trainer.personal_overrides[0][k]) for k in p0_before)
    check("C5 未被抽样的客户端个性化层保持不变", unchanged, f"n_trained={stats2.n_trained}")

    # C6：参与序列与 A/B 版一致（random.sample，轮内不重复）
    sched = build_participation_schedule(6040, 5, 604, seed=42)
    uniq = all(len(set(round_clients)) == len(round_clients) for round_clients in sched)
    import random as _random

    rng2 = _random.Random(42)
    expected0 = rng2.sample(list(range(6040)), 604)
    check(
        "C6 参与序列对齐 A/B（random.sample, 轮内不重复）",
        uniq and sched[0] == expected0 and all(len(c) == 604 for c in sched),
        f"第一轮前 3 个客户端={sched[0][:3]}",
    )

    # C7：local_epochs 真的控制本地步数
    cfg_e3 = tiny_cfg(local_epochs=3, neighbors=2, graph_warmup_rounds=10, rounds=4)
    tr3 = FedTrainer(cfg_e3, num_users, num_items, torch.device("cpu"))
    g3 = DynamicInterestGraph(cfg_e3, num_users, torch.device("cpu"))
    steps = {"n": 0}
    orig_forward = tr3.worker.forward

    def counting_forward(*a, **kw):
        steps["n"] += 1
        return orig_forward(*a, **kw)

    tr3.worker.forward = counting_forward  # type: ignore[assignment]
    tr3.train_round([0], window, g3, round_idx=1, rng=np.random.default_rng(1))
    check("C7 local_epochs=3 → 每客户端 3 次**训练**前向 + 1 次上传摘要前向",
          steps["n"] == 4, f"前向次数={steps['n']}（3 训练 + 1 摘要）")


# ======================================================================================
# D. 摘要缓存
# ======================================================================================
def group_d() -> None:
    print("\n[D] 摘要缓存语义")
    cfg = tiny_cfg()
    trainer = FedTrainer(cfg, 4, 20, torch.device("cpu"))
    summaries = torch.randn(2, cfg.latent_dim)
    users = torch.tensor([0, 1])
    trainer.upload_summary(users, summaries, round_idx=1)
    check(
        "D1 momentum=0 → 直接替换（论文 s̄_i = s_i）",
        torch.allclose(trainer.summary_bank[users], summaries),
        "缓存 == 上传摘要",
    )
    check(
        "D2 上传后 active / last_upload / n_uploads 同步更新",
        bool(trainer.summary_active[users].all())
        and bool((trainer.last_upload_round[users] == 1).all())
        and bool((trainer.n_uploads[users] == 1).all()),
        "ok",
    )
    stale = trainer.staleness(round_idx=5)
    check("D3 staleness 计算正确", int(stale[0].item()) == 4 and int(stale[2].item()) == -1,
          f"stale[0]={int(stale[0])}, stale[2]={int(stale[2])}")

    cfg_m = tiny_cfg(summary_momentum=0.8)
    tr_m = FedTrainer(cfg_m, 4, 20, torch.device("cpu"))
    old = tr_m.summary_bank[users].clone()
    tr_m.summary_active[users] = True
    tr_m.upload_summary(users, summaries, round_idx=1)
    expected = 0.8 * old + 0.2 * summaries
    check(
        "D4 momentum>0 → 动量 EMA（旧行为可选）",
        torch.allclose(tr_m.summary_bank[users], expected, atol=1e-6),
        "缓存 == 0.8*old + 0.2*new",
    )

    # ---------------- F7：上传的是「本地训练完成后」重算的摘要（论文口径）
    cfg_d5 = tiny_cfg(
        local_epochs=1, lr=0.5, dropout_rate=0.0, neighbors=2, graph_warmup_rounds=10
    )
    n_d5 = 3
    tr_d5 = FedTrainer(cfg_d5, n_d5, 20, torch.device("cpu"))
    g_d5 = DynamicInterestGraph(cfg_d5, n_d5, torch.device("cpu"))
    win_d5 = make_window(n_d5, 20, cfg_d5.max_len, seed=3)
    seq_d5 = win_d5.seq[0].unsqueeze(0)
    tr_d5.worker.load_state_dict(tr_d5.state_for(0))
    with torch.no_grad():
        _, s_pre = tr_d5.worker(seq_d5, None, None, cfg_d5.graph_mix)
    s_pre = s_pre[0].clone()
    captured = {}
    orig_up = tr_d5.upload_summary

    def _spy_up(users, summaries, round_idx):
        # 此刻 worker 仍是该客户端**训练后**的本地参数（服务器聚合只改 global_state）
        with torch.no_grad():
            _, s_now = tr_d5.worker(seq_d5, None, None, cfg_d5.graph_mix)
        captured["uploaded"] = summaries[0].clone()
        captured["expected"] = s_now[0].clone()
        return orig_up(users, summaries, round_idx)

    tr_d5.upload_summary = _spy_up  # type: ignore[assignment]
    tr_d5.train_round([0], win_d5, g_d5, round_idx=1, rng=np.random.default_rng(3))
    up = captured["uploaded"]
    gap_post = float((up - captured["expected"]).abs().max())
    gap_pre = float((up - s_pre).abs().max())
    check(
        "D5 上传摘要 = 本地训练**完成后**重算（不再是训练前的 s_i）",
        gap_post < 1e-6 and gap_pre > 1e-4,
        f"与训练后 max|Δ|={gap_post:.2e}；与训练前 max|Δ|={gap_pre:.2e}",
    )

    # ---------------- F7b：统一实现 upload_summary_after_update 的语义（快速实验也走它）
    class _ToySummaryModel(torch.nn.Module):
        """只有一层线性 + dropout 的玩具模型：用来验证「训练后 / eval 下」取摘要。"""

        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=False)
            self.drop = torch.nn.Dropout(0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.drop(self.lin(x))

    torch.manual_seed(0)
    toy = _ToySummaryModel()
    x_toy = torch.randn(3, 4)
    toy.eval()
    with torch.no_grad():
        pre_toy = toy(x_toy).clone()          # 训练前（eval：关 dropout，可比）
    with torch.no_grad():
        for p in toy.parameters():
            p.add_(0.5)                        # 模拟 optimizer.step() 改了参数
    toy.train()                                # 让 helper 自己切 eval 再切回来
    uploaded = upload_summary_after_update(toy, lambda: toy(x_toy))
    toy.eval()
    with torch.no_grad():
        post_toy = toy(x_toy).clone()
    toy.train()
    gap_post = float((uploaded - post_toy).abs().max())
    gap_pre = float((uploaded - pre_toy).abs().max())
    check(
        "D6 统一实现：上传摘要 = 更新后参数在 eval 下的前向（且不改模型的 training 状态）",
        gap_post < 1e-6 and gap_pre > 1e-4 and toy.training and not uploaded.requires_grad,
        f"与训练后 max|Δ|={gap_post:.2e}；与训练前 max|Δ|={gap_pre:.2e}；training={toy.training}",
    )
    ok_assert = True
    try:
        assert_upload_summary(uploaded, post_toy, pre_toy)
    except AssertionError:
        ok_assert = False
    raised_on_wrong = False
    try:
        assert_upload_summary(pre_toy, post_toy)   # 把「训练前」当上传摘要 → 必须报错
    except AssertionError:
        raised_on_wrong = True
    check(
        "D7 assert_upload_summary 能挡住「旧口径」（训练前摘要当上传值会报错）",
        ok_assert and raised_on_wrong,
        f"正例通过={ok_assert}；反例报错={raised_on_wrong}",
    )

    # ---------------- F7c：仓库级审计——四个快速实验脚本都改成训练后上传
    fast_sites = {
        "exp_01_fast_gpu/run_exp01_fast_gpu.py": "new_memory = local_feats.detach()",
        "exp_02_fast_gpu/run_exp02_fast_gpu.py": "new_memory = local_feats.detach()",
        "exp_03_fast_gpu/run_exp03_fast_gpu.py": "new_memory = short_summary.detach()",
        "exp_04_fast_gpu/run_exp04_fast_gpu.py": "new_memory = summaries.detach()",
    }
    audited, bad = [], []
    for rel, old_pattern in fast_sites.items():
        path = ROOT / rel
        if not path.exists():
            bad.append(f"{rel}: 文件不存在")
            continue
        src = path.read_text(encoding="utf-8")
        uses_helper = "upload_summary_after_update" in src
        leftover = old_pattern in src
        if uses_helper and not leftover:
            audited.append(rel)
        else:
            bad.append(f"{rel}: 用统一实现={uses_helper} 旧口径残留={leftover}")
    check(
        "D8 四个快速实验（exp_01~exp_04）统一走 upload_summary_after_update，无旧口径残留",
        not bad,
        f"通过 {len(audited)}/4" + (f"；问题：{bad}" if bad else ""),
    )


# ======================================================================================
# E. 端到端
# ======================================================================================
def group_e() -> None:
    print("\n[E] 端到端：一轮 train_round + evaluate_split")
    cfg = tiny_cfg(neighbors=2, rounds=2, graph_warmup_rounds=0, local_epochs=1)
    num_users, num_items = 8, 25
    window = make_window(num_users, num_items, cfg.max_len, seed=11)
    trainer = FedTrainer(cfg, num_users, num_items, torch.device("cpu"))
    graph = DynamicInterestGraph(cfg, num_users, torch.device("cpu"))
    rng = np.random.default_rng(0)

    ok = True
    detail = ""
    try:
        stats = trainer.train_round(list(range(num_users)), window, graph, round_idx=1, rng=rng)
        st = graph.update(
            trainer.summary_bank,
            trainer.summary_active,
            torch.arange(num_users),
            round_idx=1,
        )
        eval_model = build_eval_model(trainer, cfg)
        result = evaluate_split(trainer, eval_model, window, graph, "test", cfg)
        detail = (
            f"loss={stats.loss:.4f}, 有效邻居均值={float(st.n_kept.mean()):.2f}, "
            f"NDCG@{cfg.top_k}={float(result['ndcg'].mean()):.3f}"
        )
        ok = (
            math.isfinite(stats.loss)
            and result["ndcg"].shape[0] == num_users
            and np.isfinite(result["ndcg"]).all()
        )
    except Exception as exc:  # noqa: BLE001
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    check("E1 联邦训练 + 图更新 + 评测全流程可跑通", ok, detail)

    # ---------------- F10：负采样只用因果训练历史（不含 val/test），且回退不取历史物品
    cfg_e2 = tiny_cfg(neighbors=2, graph_warmup_rounds=10)
    win_e2 = make_window(2, 20, cfg_e2.max_len, seed=5)
    # 只允许 16..20 作为负样本（1..15 全部视为"已交互"）
    win_e2.history = [np.arange(1, 16, dtype=np.int64) for _ in range(2)]
    rng_e2 = np.random.default_rng(0)
    hist_hits, picked_items = 0, set()
    for _ in range(50):
        for u in range(2):
            neg = win_e2.sample_negatives(u, rng_e2)
            vals = neg[win_e2.mask[u].numpy() > 0]
            hist_hits += int(np.isin(vals, win_e2.history[u]).sum())
            picked_items |= {int(v) for v in vals.tolist()}
    check(
        "E2 负采样：绝不取已交互物品（含回退路径），且不排除 val/test 物品",
        hist_hits == 0 and picked_items.issubset({16, 17, 18, 19, 20}) and picked_items,
        f"历史命中={hist_hits}，取到={sorted(picked_items)}",
    )

    # ---------------- F10b：因果训练历史 = seq ∪ pos（不含 val/test，去 padding）
    class _FakeFast:
        pass

    fake = _FakeFast()
    fake.train_seq = torch.tensor([[1, 2, 3], [3, 0, 0]])
    fake.train_pos = torch.tensor([[2, 3, 4], [3, 0, 0]])
    hist = training_history_from_fast(fake)
    check(
        "E3 因果训练历史 = seq ∪ pos（不含 val/test，去掉 padding 0）",
        set(hist[0].tolist()) == {1, 2, 3, 4} and set(hist[1].tolist()) == {3},
        f"user0={hist[0].tolist()}，user1={hist[1].tolist()}",
    )

    # ---------------- F10c：精确因果历史 = 截止点前的全部交互（有评测去 2，其余去 1）
    import pandas as _pd

    df_e4 = _pd.DataFrame(
        {
            "userId": [0, 0, 0, 0, 1, 1],
            "itemId": [1, 2, 3, 4, 5, 6],
            "timestamp": list(range(6)),
        }
    )

    class _FakeFast2:
        pass

    ff2 = _FakeFast2()
    ff2.test_users = torch.tensor([0])   # user0 有评测记录 → 去掉 val/test
    causal = causal_history_sets(df_e4, 1, 2, [ff2])
    end_hist = history_sets_from_frame(df_e4, 1, 2)[0]
    check(
        "E4 精确因果历史：有评测用户去掉 val/test，其余去掉最后 1 个",
        causal[0][0].tolist() == end_hist[0][:-2].tolist()
        and causal[0][1].tolist() == end_hist[1][:-1].tolist(),
        f"user0={causal[0][0].tolist()}（原 {end_hist[0].tolist()}），"
        f"user1={causal[0][1].tolist()}（原 {end_hist[1].tolist()}）",
    )


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print("FedDIEG 修复验证（feddieg_core）")
    print("=" * 78)
    group_a()
    group_b()
    group_c()
    group_d()
    group_e()
    print("\n" + "=" * 78)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
