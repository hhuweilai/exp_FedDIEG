#!/usr/bin/env python
"""验证实验 1（val01）：客户端长期不参与，会不会影响推荐质量？

对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验一」。
只做诊断，不设计算法。交付物是一张图：「摘要陈旧程度 × 客户端参与率 × 推荐性能」。

⚠️ 2026-11-05 重写说明（必读）
------------------------------
旧版是"单模型快速仿真"：1 个共享模型 + 1 个 Adam，逐 microbatch 直接改共享参数，
没有客户端本地训练、没有服务器参数聚合。审计报告（`code_audit_2026-11-05/`）指出
这会让"客户端长期不参与"失去它在 FedAvg 里的主要后果（本地模型版本落后）。
本版改用 `feddieg_core` 的**真实联邦协议**（对齐 `FedDIEG.py` / aligned 主程序）：

  * 每轮按 C 的比例采样客户端（A/B 版 `random.sample`）；
  * 每个被采样客户端从**同一个全局 θ_t** 组装本地模型（共享层 + 自己的个性化层），
    在**自己的本地数据**上跑 `--local-epochs` 个 epoch，用本地 Adam 更新；
  * 个性化层留本地，共享层上传，服务器取平均（`item_emb` 另有 server_lr 外推）；
  * 参与客户端上传兴趣摘要；服务器用缓存摘要给**全部活跃节点**异步更新边。

两个条件（共用同一份预生成的客户端参与序列，同一 seed 下逐轮一致）
----------------------------------------------------------------
  upload_only        现状 FedDIEG：只有参与训练的客户端刷新摘要；陈旧摘要照样参与构图。
  upload_fresh_edges 对照：每轮用当前全局模型给全部客户端重算摘要（诊断用 oracle 上界，
                     不可部署），因此图中永无陈旧信息。两者之差 = 陈旧信息造成的损失。

输出（每次运行 4 个文件）
------------------------
  results/raw/<dataset>/<mode>/p<rate>/seed_<seed>/
      config.json                  超参数（可复现，含 feddieg_core 版本）
      val01_summary.csv            本次运行汇总（一行）
      val01_staleness_buckets.csv  陈旧度分桶 → 那张图的全部数据
      val01_run.log                训练日志

用法
----
  # 先自检能不能跑（几十秒，CPU 亦可）
  python -X utf8 exp_val_1026/exp01_staleness_participation/val01_run_staleness.py --smoke

  # 正式跑
  python -X utf8 exp_val_1026/exp01_staleness_participation/val01_run_staleness.py \
      --dataset movielens-1m --mode upload_only --participation-rate 0.1 --seed 42

  # 与旧实现做对照（复现"图冻结"）
  python ... --graph-state topk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
VAL01_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feddieg_core import __version__ as CORE_VERSION
from feddieg_core.cli import add_feddieg_args, config_from_args, prepare_runtime
from feddieg_core.config import FedDIEGConfig
from feddieg_core.data import ClientWindow, load_client_windows
from feddieg_core.evaluate import build_eval_model, evaluate_split
from feddieg_core.federated import FedTrainer, build_participation_schedule
from feddieg_core.graph import DynamicInterestGraph
from feddieg_core.utils import (
    dump_json,
    mean,
    open_log,
    safe_slug,
    set_seed,
    warn_if_legacy_results,
    write_csv,
    make_device,
)

SERVER_MODES = ("upload_only", "upload_fresh_edges")

STALENESS_BUCKETS: List[Tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    ("20-39", 20, 39),
    ("40+", 40, 10**9),
]
UPLOAD_BUCKETS: List[Tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
    ("32+", 32, 10**9),
]

# 从未上传过的客户端用这个哨兵值标陈旧度（不进入任何分桶）
NEVER_UPLOADED = 10**6

DRIFT_FIELDS = [
    "dataset", "mode", "participation_rate", "seed", "window",
    "bucket", "n_users", "mean_cos", "mean_l2", "mean_staleness",
]

BUCKET_FIELDS = [
    "dataset", "mode", "participation_rate", "seed", "window",
    "bucket_type", "bucket", "n_users", "mean_ndcg", "mean_hr",
    "mean_staleness", "mean_uploads",
]

SUMMARY_FIELDS = [
    "dataset", "mode", "participation_rate", "seed",
    "num_windows", "rounds_per_window", "total_rounds", "clients_per_round",
    "avg_window_ndcg", "final_window_ndcg", "avg_window_hr",
    "mean_staleness", "p90_staleness", "avg_ecr", "avg_boundary_drop", "elapsed_seconds",
    # ---- 2026-11-05 新增：把联邦协议与图配置写进汇总，便于复现与对照 ----
    "core_version", "local_epochs", "fedavg_weighting", "personalization",
    "graph_state", "graph_new_edge", "graph_update_scope", "rho", "neighbors",
    "summary_momentum", "avg_ecr_added", "eval_seconds", "train_seconds",
]


# --------------------------------------------------------------------------------------
# 分桶工具
# --------------------------------------------------------------------------------------
def bucket_name(value: int, buckets: List[Tuple[str, int, int]]) -> Optional[str]:
    for name, low, high in buckets:
        if low <= value <= high:
            return name
    return None


def quantile(sorted_values: np.ndarray, q: float) -> float:
    if sorted_values.size == 0:
        return 0.0
    return float(sorted_values[int(round(q * (sorted_values.size - 1)))])


def summarize_buckets(
    cfg: FedDIEGConfig,
    mode: str,
    window_idx: int,
    test_users: np.ndarray,
    test_ndcg: np.ndarray,
    test_hr: np.ndarray,
    staleness: np.ndarray,
    n_uploads: np.ndarray,
) -> List[Dict[str, object]]:
    """把"逐用户 NDCG × 自身陈旧度"汇总成桶表。"""
    if test_users.size == 0:
        return []
    rows: List[Dict[str, object]] = []
    for bucket_type, buckets in (
        ("staleness", STALENESS_BUCKETS),
        ("uploads", UPLOAD_BUCKETS),
    ):
        values = staleness if bucket_type == "staleness" else n_uploads
        groups: Dict[str, List[int]] = {}
        for i, user in enumerate(test_users.tolist()):
            raw = int(values[user])
            if bucket_type == "staleness" and raw >= NEVER_UPLOADED:
                continue  # 从未上传的不进分桶
            name = bucket_name(raw, buckets)
            if name is not None:
                groups.setdefault(name, []).append(i)
        for name, _, _ in buckets:
            indices = groups.get(name)
            if not indices:
                continue
            sel = np.asarray(indices, dtype=np.int64)
            rows.append(
                {
                    "dataset": cfg.dataset,
                    "mode": mode,
                    "participation_rate": cfg.participation_rate,
                    "seed": cfg.seed,
                    "window": window_idx,
                    "bucket_type": bucket_type,
                    "bucket": name,
                    "n_users": int(sel.size),
                    "mean_ndcg": float(test_ndcg[sel].mean()),
                    "mean_hr": float(test_hr[sel].mean()),
                    "mean_staleness": float(
                        np.mean([int(staleness[u]) for u in test_users[sel].tolist()])
                    ),
                    "mean_uploads": float(
                        np.mean([int(n_uploads[u]) for u in test_users[sel].tolist()])
                    ),
                }
            )
    return rows


def measure_drift(
    cfg: FedDIEGConfig,
    mode: str,
    window_idx: int,
    trainer: FedTrainer,
    window: ClientWindow,
    graph: DynamicInterestGraph,
    sample_users: torch.Tensor,
    staleness: np.ndarray,
) -> List[Dict[str, object]]:
    """前置检验 3：量"缓存摘要"与"此刻重算的新鲜摘要"差多少（按陈旧度分桶）。"""
    if sample_users.numel() == 0:
        return []
    users, fresh = trainer.compute_summaries(window, sample_users, graph)
    if users.numel() == 0:
        return []
    cached = trainer.summary_bank[users]
    cos = torch.nn.functional.cosine_similarity(cached, fresh, dim=-1, eps=1e-8)
    l2 = (cached - fresh).norm(dim=-1)
    cos_cpu = cos.detach().cpu().numpy()
    l2_cpu = l2.detach().cpu().numpy()
    users_cpu = users.detach().cpu().numpy()

    groups: Dict[str, List[int]] = {}
    for i, user in enumerate(users_cpu.tolist()):
        raw = int(staleness[user])
        if raw >= NEVER_UPLOADED:
            continue
        name = bucket_name(raw, STALENESS_BUCKETS)
        if name is not None:
            groups.setdefault(name, []).append(i)

    rows: List[Dict[str, object]] = []
    for name, _, _ in STALENESS_BUCKETS:
        indices = groups.get(name)
        if not indices:
            continue
        sel = np.asarray(indices, dtype=np.int64)
        rows.append(
            {
                "dataset": cfg.dataset,
                "mode": mode,
                "participation_rate": cfg.participation_rate,
                "seed": cfg.seed,
                "window": window_idx,
                "bucket": name,
                "n_users": int(sel.size),
                "mean_cos": float(cos_cpu[sel].mean()),
                "mean_l2": float(l2_cpu[sel].mean()),
                "mean_staleness": float(np.mean([int(staleness[u]) for u in users_cpu[sel].tolist()])),
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# 单次运行
# --------------------------------------------------------------------------------------
def run_single(
    cfg: FedDIEGConfig,
    mode: str,
    windows: List[ClientWindow],
    drift_sample_users: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    set_seed(cfg.seed)
    num_users = windows[0].num_users
    num_items = windows[0].num_items
    clients_per_round = cfg.resolved_clients_per_round(num_users)
    schedule = build_participation_schedule(
        num_users, cfg.total_rounds, clients_per_round, cfg.seed
    )

    trainer = FedTrainer(cfg, num_users, num_items, device)
    graph = DynamicInterestGraph(cfg, num_users, device)
    eval_model = build_eval_model(trainer, cfg)
    rng = np.random.default_rng(cfg.seed)

    print(
        f"[{mode}] 联邦设置：N={num_users}, m={clients_per_round} (C={cfg.participation_rate}), "
        f"E={cfg.local_epochs}, personalization={cfg.personalization}, "
        f"FedAvg={cfg.fedavg_weighting}, server_lr={cfg.server_lr_init}->{cfg.server_lr_final}"
    )
    print(f"[{mode}] 图设置：{graph.describe()}")

    bucket_rows: List[Dict[str, object]] = []
    drift_rows: List[Dict[str, object]] = []
    window_ndcgs: List[float] = []
    window_hrs: List[float] = []
    window_staleness: List[float] = []
    window_p90: List[float] = []
    window_drops: List[float] = []
    window_ecrs: List[float] = []
    window_added: List[float] = []
    eval_time = 0.0
    train_time = 0.0
    start_time = time.time()
    global_round = 0
    prev_window_last_ndcg: Optional[float] = None

    drift_users = torch.zeros(0, dtype=torch.long, device=device)
    if drift_sample_users and drift_sample_users > 0:
        eligible = torch.nonzero(windows[0].mask.sum(dim=1) > 0, as_tuple=False).flatten()
        drift_gen = torch.Generator().manual_seed(cfg.seed + 13)
        perm = torch.randperm(eligible.numel(), generator=drift_gen)
        drift_users = eligible[perm[: min(int(drift_sample_users), int(eligible.numel()))]]

    for window_idx, window in enumerate(windows, start=1):
        print(
            f"\n[{mode}] Window {window_idx}/{cfg.num_windows} | "
            f"eval_users={window.test_users.numel()} | train_clients={window.active_users.numel()}"
        )
        _, init_hr, init_ndcg = _evaluate(cfg, trainer, eval_model, window, graph)
        initial_ndcg = float(init_ndcg.mean()) if init_ndcg.size else 0.0
        drop = 0.0 if prev_window_last_ndcg is None else max(prev_window_last_ndcg - initial_ndcg, 0.0)
        last_ndcg = initial_ndcg
        last_hr = float(init_hr.mean()) if init_hr.size else 0.0
        window_ecs: List[float] = []
        window_adds: List[float] = []

        for window_round in range(1, cfg.rounds_per_window + 1):
            global_round += 1
            round_start = time.time()
            sampled = schedule[global_round - 1]

            # 诊断上界：每轮用当前全局模型给全部客户端重算摘要（不可部署）
            if mode == "upload_fresh_edges":
                trainer.refresh_summaries_from_global(
                    window, window.active_users, graph, global_round
                )

            stats = trainer.train_round(sampled, window, graph, global_round, rng)

            # 图更新：默认对全部活跃节点异步更新（论文语义）
            if cfg.graph_update_scope == "all":
                update_users = torch.arange(num_users, device=device)
            else:
                update_users = torch.tensor(sampled, dtype=torch.long, device=device)
            gstats = graph.update(
                trainer.summary_bank, trainer.summary_active, update_users, global_round
            )
            if gstats is not None and len(gstats) > 0:
                window_ecs.append(float(gstats.ecr.mean()))
                window_adds.append(float(gstats.n_added.mean()))
            else:
                # warmup 期间不做图更新（对齐 A/B 的 WARMUP_ROUNDS），记 0 便于后续聚合
                window_ecs.append(0.0)
                window_adds.append(0.0)

            train_time += time.time() - round_start

            should_eval = (window_round % cfg.eval_every == 0) or (window_round == cfg.rounds_per_window)
            if not should_eval:
                continue

            eval_start = time.time()
            test_users, test_hr, test_ndcg = _evaluate(
                cfg, trainer, eval_model, window, graph
            )
            eval_time += time.time() - eval_start

            staleness_t = trainer.staleness(global_round)
            staleness_np = _staleness_np(staleness_t)
            n_uploads_np = trainer.n_uploads.detach().cpu().numpy()

            last_ndcg = float(test_ndcg.mean()) if test_ndcg.size else 0.0
            last_hr = float(test_hr.mean()) if test_hr.size else 0.0
            seen = (trainer.last_upload_round >= 0) & trainer.summary_active
            active_staleness = staleness_t[seen].float()
            s_mean = float(active_staleness.mean().item()) if active_staleness.numel() else 0.0

            if window_round == cfg.rounds_per_window:
                bucket_rows.extend(
                    summarize_buckets(
                        cfg, mode, window_idx, test_users, test_ndcg, test_hr,
                        staleness_np, n_uploads_np,
                    )
                )
            print(
                f"[{mode}] p{cfg.participation_rate:.2f} W{window_idx} R{window_round:03d}/"
                f"{cfg.rounds_per_window} | loss {stats.loss:.4f} | trained {stats.n_trained}/{stats.n_clients}"
                f" | ECR {window_ecs[-1] if window_ecs else 0.0:.4f}"
                f" | +edges {window_adds[-1] if window_adds else 0.0:.2f}"
                f" | staleness {s_mean:.1f} | NDCG@{cfg.top_k} {last_ndcg:.4f}"
                f" | {time.time() - round_start:.2f}s"
            )

        staleness_np = _staleness_np(trainer.staleness(global_round))
        seen = (trainer.last_upload_round >= 0) & trainer.summary_active
        active_staleness = trainer.staleness(global_round)[seen].float()
        if active_staleness.numel():
            sorted_staleness = torch.sort(active_staleness).values.cpu().numpy()
            s_mean = float(active_staleness.mean().item())
            s_p90 = quantile(sorted_staleness, 0.9)
        else:
            s_mean = s_p90 = 0.0

        window_ndcgs.append(last_ndcg)
        window_hrs.append(last_hr)
        window_staleness.append(s_mean)
        window_p90.append(s_p90)
        window_drops.append(drop)
        window_ecrs.extend(window_ecs)
        window_added.extend(window_adds)
        prev_window_last_ndcg = last_ndcg
        print(
            f"[{mode}] W{window_idx} end | NDCG@{cfg.top_k} {last_ndcg:.4f} | staleness {s_mean:.2f}"
            f" | p90 {s_p90:.0f} | ECR {mean(window_ecs):.4f}"
        )

        write_csv(out_dir / "val01_staleness_buckets.csv", bucket_rows, BUCKET_FIELDS)

        if drift_users.numel() > 0:
            drift_rows.extend(
                measure_drift(
                    cfg, mode, window_idx, trainer, window, graph, drift_users, staleness_np
                )
            )
            write_csv(out_dir / "val01_drift_buckets.csv", drift_rows, DRIFT_FIELDS)

    summary = {
        "dataset": cfg.dataset,
        "mode": mode,
        "participation_rate": cfg.participation_rate,
        "seed": cfg.seed,
        "num_windows": cfg.num_windows,
        "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": cfg.total_rounds,
        "clients_per_round": clients_per_round,
        "avg_window_ndcg": mean(window_ndcgs),
        "final_window_ndcg": window_ndcgs[-1] if window_ndcgs else 0.0,
        "avg_window_hr": mean(window_hrs),
        "mean_staleness": mean(window_staleness),
        "p90_staleness": mean(window_p90),
        "avg_ecr": mean(window_ecrs),
        "avg_boundary_drop": mean(window_drops[1:]) if len(window_drops) > 1 else 0.0,
        "elapsed_seconds": time.time() - start_time,
        "core_version": CORE_VERSION,
        "local_epochs": cfg.local_epochs,
        "fedavg_weighting": cfg.fedavg_weighting,
        "personalization": cfg.personalization,
        "graph_state": cfg.graph_state,
        "graph_new_edge": cfg.graph_new_edge,
        "graph_update_scope": cfg.graph_update_scope,
        "rho": cfg.rho,
        "neighbors": cfg.neighbors,
        "summary_momentum": cfg.summary_momentum,
        "avg_ecr_added": mean(window_added),
        "eval_seconds": eval_time,
        "train_seconds": train_time,
    }
    write_csv(out_dir / "val01_summary.csv", [summary], SUMMARY_FIELDS)
    print(
        f"\n[{mode}] p{cfg.participation_rate:.2f} seed{cfg.seed} done in "
        f"{summary['elapsed_seconds']:.1f}s | avg_window_ndcg={summary['avg_window_ndcg']:.4f} | "
        f"mean_staleness={summary['mean_staleness']:.2f} | buckets={len(bucket_rows)}"
    )


def _evaluate(cfg: FedDIEGConfig, trainer: FedTrainer, eval_model, window, graph):
    result = evaluate_split(trainer, eval_model, window, graph, "test", cfg)
    return result["users"], result["hr"], result["ndcg"]


def _staleness_np(staleness: torch.Tensor) -> np.ndarray:
    arr = staleness.detach().cpu().numpy().astype(np.int64)
    arr[arr < 0] = NEVER_UPLOADED
    return arr


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="val01: 客户端长期不参与是否影响推荐质量（FedAvg 版）")
    parser.add_argument("--mode", default="upload_only", choices=SERVER_MODES)
    parser.add_argument("--drift-sample-users", type=int, default=0,
                        help=">0 时在每个窗口末抽样这么多用户，测量缓存摘要与新鲜摘要的差异")
    add_feddieg_args(parser)
    parser.set_defaults(out_root=VAL01_DIR / "results" / "raw")
    return parser.parse_args()


def make_out_dir(out_root: Path, dataset: str, mode: str, rate: float, seed: int) -> Path:
    """四级分类：<out_root>/<dataset>/<mode>/p<rate>/seed_<seed>/"""
    return out_root / safe_slug(dataset) / mode / f"p{rate:.2f}" / f"seed_{seed}"


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.validate()
    prepare_runtime(cfg)

    if not (0.0 < cfg.participation_rate <= 1.0):
        raise SystemExit("--participation-rate 必须在 (0, 1] 区间内")

    set_seed(cfg.seed)
    device = make_device(cfg.device)

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else make_out_dir(args.out_root, cfg.dataset, args.mode, cfg.participation_rate, cfg.seed)
    )
    if args.smoke and args.out_dir is None:
        # smoke 输出单独放到 <out_root>_smoke，避免污染正式结果的聚合脚本
        out_dir = Path(str(args.out_root) + "_smoke") / out_dir.relative_to(args.out_root)
    warn_if_legacy_results(out_dir)
    open_log(out_dir, "val01_run.log")

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print("Config:", json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str))
    dump_json(out_dir / "config.json", {**asdict(cfg), "core_version": CORE_VERSION, "mode": args.mode})

    windows = load_client_windows(cfg)
    run_single(cfg, args.mode, windows, args.drift_sample_users, out_dir, device)
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
