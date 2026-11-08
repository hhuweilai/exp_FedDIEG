"""新臂的离线断言（阶段 2）：static_build / onboarding / fresh_only / periodic boundary。

与 `dg_selfcheck.py`（原有 9 条，证明「只差图」）互补：本脚本只检查**新更新策略**的语义：

  ① static --build-only：建图轮之前没有边；建图后签名逐位不变；只建一次
  ② onboarding：覆盖单调增长；用户**建过边之后其邻居行不再变化**
  ③ fresh_only：只更新「有新上传」的用户 ⇒ 总更新人数 ≤ dynamic
  ④ periodic --refresh-phase boundary：重建轮 = 预热后第一轮 + 每个窗口边界

用法：
  F:\\Anaconda\\python.exe -X utf8 val_DynamicGraph/dg_selfcheck_arms.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

DG_DIR = Path(__file__).resolve().parent
ROOT = DG_DIR.parent
for p in (str(ROOT), str(DG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dg_common import (  # noqa: E402
    ARM_DYNAMIC,
    ARM_FRESH_ONLY,
    ARM_NOGRAPH,
    ARM_ONBOARDING,
    ARM_PERIODIC,
    ARM_STATIC,
    build_graph,
    edge_signature,
)
from feddieg_core.cli import prepare_runtime  # noqa: E402
from feddieg_core.config import FedDIEGConfig  # noqa: E402
from feddieg_core.data import load_client_windows  # noqa: E402
from feddieg_core.federated import FedTrainer, build_participation_schedule  # noqa: E402
from feddieg_core.utils import make_device, set_seed  # noqa: E402


class Checks:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.rows)


def make_cfg() -> FedDIEGConfig:
    cfg = FedDIEGConfig(
        dataset="movielens-1m",
        latent_dim=16, max_len=8, num_heads=2, num_blocks=1,
        num_windows=2, rounds_per_window=4,
        clients_per_round=8, participation_rate=0.5,
        neighbors=4, graph_pool_size=32, graph_warmup_rounds=2, graph_block_size=256,
        eval_every=4, max_eval_users=32, graph_update_scope="all",
    )
    cfg.rounds = cfg.total_rounds
    return cfg


def run_arm(
    cfg: FedDIEGConfig,
    windows,
    arm: str,
    device: torch.device,
    **graph_kw,
) -> Tuple[object, List[Dict[str, object]], torch.Tensor, torch.Tensor]:
    """跑一次极小规模实验；返回 (graph, 每轮记录, 第 4 轮后的 ids 快照, 末轮 ids)。"""
    set_seed(cfg.seed)
    num_users = int(windows[0].num_users)
    trainer = FedTrainer(cfg, num_users, int(windows[0].num_items), device)
    graph = build_graph(cfg, arm, num_users, device, **graph_kw)
    schedule = build_participation_schedule(
        num_users, cfg.total_rounds, cfg.resolved_clients_per_round(num_users), cfg.seed
    )
    rng = np.random.default_rng(cfg.seed)
    rec: List[Dict[str, object]] = []
    snap_ids = None
    snap_round = 4
    g = 0
    for window in windows:
        for _ in range(int(cfg.rounds_per_window)):
            g += 1
            trainer.train_round(schedule[g - 1], window, graph, g, rng)
            users = torch.arange(num_users, device=device)
            if arm == ARM_FRESH_ONLY:
                pending = (
                    trainer.last_upload_round[users] > graph.last_user_update[users]
                )
                users = users[pending]
            stats = graph.update(
                trainer.summary_bank, trainer.summary_active, users, g
            )
            if g == snap_round and arm != ARM_NOGRAPH:
                snap_ids = graph.ids.detach().clone()
            rec.append(
                {
                    "round": g,
                    "n_edges": int(graph.mask.sum().item()),
                    "refresh": bool(getattr(graph, "last_update_was_refresh", False)),
                    "n_updated": int(len(stats)) if stats is not None else 0,
                    "sig": edge_signature(graph) if arm != ARM_NOGRAPH else "-",
                }
            )
    return graph, rec, snap_ids if snap_ids is not None else torch.empty(0), graph.ids.detach()


def main() -> int:
    ap = argparse.ArgumentParser(description="新臂的离线断言（阶段 2）")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    device = make_device(args.device)
    cfg = make_cfg()
    cfg.validate()
    prepare_runtime(cfg)
    windows = load_client_windows(cfg)
    n_users = int(windows[0].num_users)
    checks = Checks()
    warmup = int(cfg.graph_warmup_rounds)

    # ---------------- ① static --build-only
    graph, rec, _, _ = run_arm(
        cfg, windows, ARM_STATIC, device, freeze_round=6, build_only=True
    )
    before = [r["n_edges"] for r in rec if int(r["round"]) < 6]
    after = [r["n_edges"] for r in rec if int(r["round"]) >= 6]
    sigs = [r["sig"] for r in rec if int(r["round"]) >= 6]
    checks.add(
        "①a static_build：建图轮（r6）之前没有任何边",
        all(v == 0 for v in before), f"edges(r1..5)={before}",
    )
    checks.add(
        "①b static_build：建图轮之后有边且签名逐位不变",
        bool(after) and after[0] > 0 and len(set(sigs)) == 1,
        f"edges(r6..8)={after} sigs={len(set(sigs))}",
    )
    checks.add(
        "①c static_build：只建一次图",
        int(getattr(graph, "n_updates", 0)) == 1,
        f"n_updates={getattr(graph, 'n_updates', 0)}",
    )

    # ---------------- ② onboarding
    graph, rec, snap_ids, end_ids = run_arm(cfg, windows, ARM_ONBOARDING, device)
    edges = [int(r["n_edges"]) for r in rec]
    checks.add(
        "②a onboarding：覆单调不下降且确实增长",
        all(b >= a for a, b in zip(edges, edges[1:])) and edges[-1] > edges[0],
        f"edges={edges}",
    )
    covered_at_snap = (snap_ids >= 0).any(dim=1)
    same_rows = bool(
        (snap_ids[covered_at_snap] == end_ids[covered_at_snap]).all().item()
    )
    checks.add(
        "②b onboarding：建过边的用户，其邻居行之后逐位不变",
        same_rows,
        f"snap 轮已覆盖 {int(covered_at_snap.sum())} 人",
    )

    # ---------------- ③ fresh_only vs dynamic
    graph_d, rec_d, _, _ = run_arm(cfg, windows, ARM_DYNAMIC, device)
    graph_f, rec_f, _, _ = run_arm(cfg, windows, ARM_FRESH_ONLY, device)
    n_dyn = int(getattr(graph_d, "n_user_updates", 0))
    n_fresh = int(getattr(graph_f, "n_user_updates", 0))
    checks.add(
        "③ fresh_only：更新人数 > 0 且 ≤ dynamic（只更新有新证据的用户）",
        0 < n_fresh <= n_dyn,
        f"fresh={n_fresh} vs dynamic={n_dyn}",
    )

    # ---------------- ④ periodic boundary
    _, rec_p, _, _ = run_arm(
        cfg, windows, ARM_PERIODIC, device, refresh_phase="boundary"
    )
    rounds = sorted(int(r["round"]) for r in rec_p if r["refresh"])
    expected = [warmup + 1] + [
        r for r in range(1, int(cfg.total_rounds) + 1)
        if r > warmup and (r - 1) % int(cfg.rounds_per_window) == 0
    ]
    checks.add(
        "④ periodic boundary：重建轮 = 预热后首轮 + 窗口边界",
        rounds == sorted(set(expected)),
        f"实际 {rounds} / 期望 {sorted(set(expected))}",
    )

    total = len(checks.rows)
    passed = sum(1 for _, ok, _ in checks.rows if ok)
    print("\n" + "=" * 70)
    print(f"新臂断言：{passed}/{total} PASS")
    print("ALL CHECKS PASSED —— 新更新策略的语义与设计一致" if checks.ok() else "有断言失败")
    return 0 if checks.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
