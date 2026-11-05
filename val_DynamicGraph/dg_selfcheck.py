#!/usr/bin/env python
"""val_DynamicGraph 自检：证明「三条臂只差图这一件事」。

`dg_run.py` 会打印一堆数字，但「两条曲线不一样」本身并不能说明差异来自图 ——
也可能是随机种子、负样本、参与顺序、初始化不一样。本脚本用四条**同一性断言**把这种
可能性排除掉，全部在极小配置下几十秒内完成：

  ① **训练数据流逐位相同**：三条臂喂给模型的负样本流（顺序 + 字节）完全一致
     ⇒ 后面的 NDCG 差异不是「抽到了不同的负样本」造成的，而是配对比较。
  ② **NoGraph 真的关干净了**：该臂每个评测点的 `delta_rel_max` 恒为 0（不是「很小」）。
     同时确认开图臂在有边用户上的 `delta_rel` 确实 > 0（否则就是「图臂其实也没接上」）。
  ③ **Static 真的冻结了 / Dynamic 真的在更新**：静态臂冻结后边的签名一位不变；
     动态臂的更新次数 == 总轮数（每轮都更新）。
  ④ **`--graph-mix 0` 与 NoGraph 逐用户 NDCG 逐位相同**：这是「空图控制」的黄金标准 ——
     说明 NoGraph 不是「把图调小」，而是把图分支乘零到完全等价。

额外还有三条便宜的检查：
  ⑤ `branch_forward`（诊断用的前向）与 `model.score_user`（真正打分用的前向）分数一致
     —— 否则我们测的 `delta_rel` 不是实际影响推荐的那个量；
  ⑥ 三条臂的**初始参数**逐位相同（同一个 seed 下构造出来的 FedTrainer 一致）；
  ⑦ 客户端参与序列可复现（同 seed 两次构造相同）。

用法
----
  /home/wl/.venv-feddieg/bin/python val_DynamicGraph/dg_selfcheck.py
  ... dg_selfcheck.py --device cpu
  ... dg_selfcheck.py --dataset movielens-1m --rounds 2

全部通过时打印 `ALL CHECKS PASSED` 并返回 0；任一失败返回 1。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dg_common import (
    ARM_DYNAMIC,
    ARM_NOGRAPH,
    ARM_STATIC,
    add_dg_args,
    branch_forward,
    build_graph,
    edge_signature,
    n_users_with_edges,
    pick_probe_rows,
    pick_random_base,
    probe_diagnostics,
)

from feddieg_core.cli import add_feddieg_args, config_from_args, prepare_runtime
from feddieg_core.config import FedDIEGConfig
from feddieg_core.data import ClientWindow, load_client_windows
from feddieg_core.evaluate import build_eval_model, evaluate_split
from feddieg_core.federated import FedTrainer, build_participation_schedule
from feddieg_core.utils import make_device, mean, set_seed


# ======================================================================================
# 结果收集
# ======================================================================================
@dataclass
class Report:
    rows: List[Tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.rows)


class NegRecorder:
    """记录每个客户端每个 epoch 实际抽到的负样本（用来断言三条臂的数据流一致）。"""

    def __init__(self) -> None:
        self.rows: List[Tuple[int, int, bytes]] = []
        self._orig = None

    def __enter__(self) -> "NegRecorder":
        self._orig = ClientWindow.sample_negatives
        recorder = self

        def patched(window_self, user, rng, max_tries: int = 32):
            out = recorder._orig(window_self, user, rng, max_tries)
            recorder.rows.append((int(window_self.window_idx), int(user), out.tobytes()))
            return out

        ClientWindow.sample_negatives = patched  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        ClientWindow.sample_negatives = self._orig  # type: ignore[assignment]


def params_digest(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.md5()
    for key in sorted(state.keys()):
        arr = np.ascontiguousarray(state[key].detach().cpu().numpy())
        h.update(key.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


# ======================================================================================
# 一次极小规模的臂运行
# ======================================================================================
def run_arm(
    cfg: FedDIEGConfig,
    windows: List[ClientWindow],
    arm: str,
    device: torch.device,
    rounds: int,
    graph_mix: Optional[float] = None,
) -> dict:
    cfg2 = cfg if graph_mix is None else cfg.clone(graph_mix=graph_mix)
    set_seed(cfg2.seed)
    num_users = int(windows[0].num_users)
    num_items = int(windows[0].num_items)
    clients_per_round = cfg2.resolved_clients_per_round(num_users)
    schedule = build_participation_schedule(
        num_users, cfg2.total_rounds, clients_per_round, cfg2.seed
    )

    trainer = FedTrainer(cfg2, num_users, num_items, device)
    init_digest = params_digest(trainer.global_state)
    graph = build_graph(cfg2, arm, num_users, device)
    eval_model = build_eval_model(trainer, cfg2)
    rng = np.random.default_rng(cfg2.seed)

    sigs: List[str] = []
    diags: List[Dict[str, float]] = []
    per_user_ndcg: Optional[np.ndarray] = None
    last_window = windows[-1]
    global_round = 0

    with NegRecorder() as recorder:
        for window in windows:
            for _ in range(int(rounds)):
                global_round += 1
                trainer.train_round(schedule[global_round - 1], window, graph, global_round, rng)
                graph.update(
                    trainer.summary_bank,
                    trainer.summary_active,
                    torch.arange(num_users, device=device),
                    global_round,
                )
            result = evaluate_split(trainer, eval_model, window, graph, "test", cfg2)
            per_user_ndcg = result["ndcg"]
            sigs.append(edge_signature(graph) if arm != ARM_NOGRAPH else "-")
            probe = pick_probe_rows(window, 32, cfg2.seed)
            if probe is not None:
                d, _, _ = probe_diagnostics(
                    trainer, eval_model, graph, probe[0], probe[1], None, cfg2,
                    random_base=pick_random_base(trainer.summary_active, 32, cfg2.seed),
                )
                diags.append(d)
            last_window = window

    return {
        "arm": arm,
        "trainer": trainer,
        "graph": graph,
        "eval_model": eval_model,
        "cfg": cfg2,
        "window": last_window,
        "negatives": recorder.rows,
        "sigs": sigs,
        "diags": diags,
        "ndcg": per_user_ndcg,
        "n_updates": int(getattr(graph, "n_updates", 0)),
        "n_edges": n_users_with_edges(graph),
        "init_digest": init_digest,
        "total_rounds": global_round,
    }


def edge_probe(run: dict) -> Optional[Dict[str, float]]:
    """在「真的有邻居」的用户上做一次图分支诊断。

    小规模（smoke）下绝大多数用户还没有边，随机抽的探针用户往往一个边都没有，
    所以「图分支是否接通」必须在有边的用户上验证。
    """
    trainer, eval_model, graph, cfg = run["trainer"], run["eval_model"], run["graph"], run["cfg"]
    users_all, seqs_all, _ = run["window"].split("test")
    row_of = {int(u): i for i, u in enumerate(users_all.tolist())}
    edge_users = torch.nonzero(graph.mask).flatten().tolist()
    rows = [row_of[u] for u in edge_users if u in row_of][:16]
    if not rows:
        return None
    diag, _, _ = probe_diagnostics(
        trainer, eval_model, graph, users_all[rows], seqs_all[rows], None, cfg
    )
    return diag


def check_score_equivalence(run: dict, device: torch.device) -> Tuple[bool, str]:
    """⑤ 诊断前向 == 真正的打分前向。"""
    trainer, eval_model, graph, cfg = run["trainer"], run["eval_model"], run["graph"], run["cfg"]
    users, seqs, items = run["window"].split("test")
    if int(users.numel()) == 0:
        return False, "没有可评测用户"
    user0 = int(users[0].item())
    trainer.load_into(eval_model, user0)
    users_t = users[:1].to(device)
    seqs_t = seqs[:1].to(device)
    items_t = items[:1].to(device)
    nbr_sum, nbr_w, nbr_valid = graph.gather(trainer.summary_bank, users_t)
    if bool(nbr_valid.any().item()):
        ref = eval_model.score_user(seqs_t, items_t, nbr_sum, nbr_w, cfg.graph_mix)
    else:
        ref = eval_model.score_user(seqs_t, items_t)
    _, _, got = branch_forward(
        eval_model, trainer.summary_bank, graph, users_t, seqs_t, cfg, items=items_t
    )
    if got is None:
        return False, "branch_forward 没有返回分数"
    diff = float((ref - got).abs().max().item())
    return diff <= 1e-5, f"max|Δscore|={diff:.2e}（要求 ≤ 1e-5）"


# ======================================================================================
# 主流程
# ======================================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="val_DynamicGraph 自检：三条臂只差图这一件事")
    add_feddieg_args(parser)   # 已含 --device / --smoke / --dataset 等
    add_dg_args(parser)        # 把 lambda_struct 默认对齐正式运行（= 0）
    parser.add_argument("--rounds", type=int, default=2, help="每个窗口跑几轮（默认 2）")
    parser.set_defaults(device="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 用 --smoke 的极小配置（2 窗口、neighbors=4、32 个评测用户…），轮数由 --rounds 控制
    original_argv = sys.argv
    sys.argv = [original_argv[0], "--smoke"]
    try:
        smoke_args = parse_args()
    finally:
        sys.argv = original_argv
    cfg = config_from_args(smoke_args)
    cfg = cfg.clone(eval_every=1, device=args.device)
    cfg.rounds = cfg.total_rounds
    cfg.validate()
    prepare_runtime(cfg)
    set_seed(cfg.seed)
    device = make_device(args.device)

    print("=" * 78)
    print("val_DynamicGraph 自检")
    print("=" * 78)
    print(
        f"设备: {device} | 数据集: {cfg.dataset} | 窗口: {cfg.num_windows} | "
        f"每窗口轮数: {args.rounds} | m={cfg.clients_per_round} | "
        f"neighbors={cfg.neighbors} | warmup={cfg.graph_warmup_rounds} | "
        f"lambda_struct={cfg.lambda_struct}"
    )
    t_start = time.time()
    windows = load_client_windows(cfg)
    print(f"数据准备完成：{time.time() - t_start:.1f}s\n")

    report = Report()
    runs: Dict[str, dict] = {}
    for arm in (ARM_NOGRAPH, ARM_STATIC, ARM_DYNAMIC):
        t0 = time.time()
        runs[arm] = run_arm(cfg, windows, arm, device, args.rounds)
        print(
            f"  · {arm:8s} 跑完 {time.time() - t0:.1f}s | "
            f"updates={runs[arm]['n_updates']} | edges(用户数)={runs[arm]['n_edges']} | "
            f"delta_rel={mean([d['delta_rel_mean'] for d in runs[arm]['diags']]):.3e} | "
            f"ndcg={float(np.mean(runs[arm]['ndcg'])):.4f}"
        )
    mix0 = run_arm(cfg, windows, ARM_DYNAMIC, device, args.rounds, graph_mix=0.0)
    print("  · dynamic(mix=0) 跑完（用于 ④ 等价性检查）")
    print()

    # ---------------- ① 训练数据流逐位相同
    base = runs[ARM_NOGRAPH]["negatives"]
    same = True
    detail = ""
    for arm in (ARM_STATIC, ARM_DYNAMIC):
        other = runs[arm]["negatives"]
        if len(other) != len(base):
            same = False
            detail = f"{arm} 负样本条数 {len(other)} != nograph {len(base)}"
            break
        for i, (a, b) in enumerate(zip(base, other)):
            if a != b:
                same = False
                detail = f"{arm} 第 {i} 条负样本不同（window={a[0]}, user={a[1]}）"
                break
        if not same:
            break
    if same and len(base) == 0:
        same = False
        detail = "没有记录到任何负样本（检查配置）"
    if same:
        detail = f"{len(base)} 条负样本（顺序 + 内容）三条臂完全一致"
    report.add("① 三条臂的训练数据流逐位相同", same, detail)

    # ---------------- ② NoGraph 分支恒为 0；图臂确实接通
    nog_max = max(d["delta_rel_max"] for d in runs[ARM_NOGRAPH]["diags"])
    report.add(
        "②a NoGraph 的图分支恒为 0",
        nog_max <= 1e-12,
        f"delta_rel_max={nog_max:.3e}（要求 ≤ 1e-12）",
    )
    dyn_edge_diag = edge_probe(runs[ARM_DYNAMIC])
    if dyn_edge_diag is None:
        report.add(
            "②b DynamicGraph 的图分支确实接通",
            False,
            f"图里没有任何有边的用户（n_edges={runs[ARM_DYNAMIC]['n_edges']}，"
            "评测用户里也找不到有边的用户）—— 图根本没建起来",
        )
    else:
        rel = dyn_edge_diag["delta_rel_mean"]
        report.add(
            "②b DynamicGraph 的图分支确实接通",
            rel > 0 and dyn_edge_diag["probe_with_graph"] > 0,
            f"在有边的 {int(dyn_edge_diag['probe_n'])} 个用户上 "
            f"delta_rel_mean={rel:.3e}，有邻居占比={dyn_edge_diag['probe_with_graph']:.2f}"
            f"（全图有边的用户数={runs[ARM_DYNAMIC]['n_edges']}）",
        )

    # ---------------- ③ Static 冻结 / Dynamic 每轮更新
    static_sigs = runs[ARM_STATIC]["sigs"]
    report.add(
        "③a StaticGraph 冻结后边的签名一位不变",
        len(set(static_sigs)) == 1,
        f"signatures={static_sigs}",
    )
    static_updates = runs[ARM_STATIC]["n_updates"]
    dyn_updates = runs[ARM_DYNAMIC]["n_updates"]
    total = runs[ARM_DYNAMIC]["total_rounds"]
    report.add(
        "③b DynamicGraph 每轮都更新 / StaticGraph 只更新一次",
        dyn_updates == total and static_updates == 1,
        f"dynamic {dyn_updates}/{total} 轮，static {static_updates} 次",
    )

    # ---------------- ④ mix=0 与 NoGraph 逐位等价
    mix0_max = max(d["delta_rel_max"] for d in mix0["diags"])
    ref = runs[ARM_NOGRAPH]["ndcg"]
    got = mix0["ndcg"]
    if ref is None or got is None or ref.shape != got.shape:
        report.add("④ graph-mix=0 与 NoGraph 逐用户 NDCG 逐位相同", False, "形状不一致")
    else:
        diff = float(np.max(np.abs(ref - got)))
        report.add(
            "④ graph-mix=0 与 NoGraph 逐用户 NDCG 逐位相同",
            diff == 0.0 and mix0_max <= 1e-12,
            f"n={ref.size}, max|ΔNDCG|={diff:.3e}, delta_rel_max={mix0_max:.3e}",
        )

    # ---------------- ⑤ 诊断前向 == 打分前向
    ok, detail = check_score_equivalence(runs[ARM_DYNAMIC], device)
    report.add("⑤ 诊断前向与 score_user 打分一致", ok, detail)

    # ---------------- ⑥ 初始参数逐位相同
    digests = {arm: runs[arm]["init_digest"] for arm in runs}
    report.add("⑥ 三条臂的初始参数逐位相同", len(set(digests.values())) == 1, f"{digests}")

    # ---------------- ⑦ 参与序列可复现
    m = cfg.resolved_clients_per_round(int(windows[0].num_users))
    sched_a = build_participation_schedule(int(windows[0].num_users), cfg.total_rounds, m, cfg.seed)
    sched_b = build_participation_schedule(int(windows[0].num_users), cfg.total_rounds, m, cfg.seed)
    report.add("⑦ 客户端参与序列可复现", sched_a == sched_b, f"{len(sched_a)} 轮")

    print()
    print("=" * 78)
    n_pass = sum(1 for _, ok, _ in report.rows if ok)
    print(f"自检汇总：{n_pass}/{len(report.rows)} PASS | 总用时 {time.time() - t_start:.1f}s")
    if report.ok:
        print("ALL CHECKS PASSED —— 三条臂确实只差「图」这一件事，可以跑正式实验。")
        return 0
    print("有检查未通过：先修好再跑正式实验，否则对比结论不成立。")
    for name, ok, detail in report.rows:
        if not ok:
            print(f"  - FAIL: {name} — {detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
