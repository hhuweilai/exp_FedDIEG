#!/usr/bin/env python
"""val_DynamicGraph 主脚本：一次运行 = 一个 (数据集, 臂, 随机种子)。

回答的问题
----------
「在**相同的序列编码器和训练协议**下，动态图到底有没有明确贡献？」

三条臂（见 `dg_common.py` 的模块文档）：
  `nograph`  图分支关掉（u_i = seq_emb_i）
  `static`   预热结束建一次图，之后完全冻结
  `dynamic`  论文公式，每轮更新（`--rho` 控制 EMA）
  `periodic` （补充对照）每 `--refresh-rounds` 轮清零重建

三条臂共用：同一个 seed → 同一份窗口数据、同一个候选集、同一条客户端参与序列、
同一份负样本流（`dg_selfcheck.py` 已断言逐位相同）、同样的本地 epoch / lr /
服务器 lr / FedAvg 加权 / 个性化层划分 / 评测轮次 / 评测用户。

记录什么
--------
逐评测点（`dg_round_metrics.csv`）：
  * 推荐指标：`all_ndcg` / `all_hr` / `n_eval`（test 视图，@top_k）
  * 图状态：`graph_n_edges` / `graph_edge_sig` / `graph_edge_changed_after_freeze`
    / `graph_updates` / `ecr_all` / `added_mean` / `pool_fill`
  * 分支诊断：`delta_rel_mean|median|max`、`delta_cos_mean`、
    `neighbor_cos_mean` / `neighbor_cos_base` / `neighbor_lift`、
    `probe_with_graph`、`seq_norm_mean` / `delta_norm_mean` / `u_norm_mean`
  逐用户（`dg_user_metrics.csv`，每个窗口最后一次评测）：配对检验用
  逐用户探针（`dg_probe_metrics.csv`）：`delta_rel` 与 NDCG 的相关性用
  汇总（`dg_summary.csv`）：一次运行一行

严格性（本脚本自己会硬检查，不通过就抛错退出）
----------------------------------------------
1. `nograph` 臂每个评测点的 `delta_rel_max` 必须 **恰好 0**（不是「很小」）。
2. `static` 臂冻结之后边的签名必须**一位都不变**（`graph_edge_changed_after_freeze==0`）。
3. 预热结束后图臂必须真的建出了边（否则打印醒目警告，因为那样等于偷偷退化成 NoGraph）。

用法
----
  # 冒烟（几十秒，输出写到 results/raw_smoke/）
  python val_DynamicGraph/dg_run.py --smoke --arm dynamic

  # 单个正式运行
  python val_DynamicGraph/dg_run.py --dataset movielens-1m --arm nograph --seed 42
  python val_DynamicGraph/dg_run.py --dataset movielens-1m --arm static  --seed 42
  python val_DynamicGraph/dg_run.py --dataset movielens-1m --arm dynamic --seed 42 --rho 0.2

  # 补充对照：慢刷新静态图（每 120 轮重建一次 ≈ 每个窗口一次）
  python val_DynamicGraph/dg_run.py --arm periodic --refresh-rounds 120 --seed 42

  # 剂量反应：把图分支放大 / 缩小（回答「是不是被稀释了」）
  python val_DynamicGraph/dg_run.py --arm dynamic --graph-mix 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from dg_common import (  # noqa: E402  (dg_common 会先把 ROOT 塞进 sys.path)
    ARM_FRESH_ONLY,
    ARM_LABELS,
    ARM_NOGRAPH,
    ARM_STATIC,
    add_dg_args,
    arm_dir_name,
    build_graph,
    build_user_genre_profiles,
    center_profiles,
    edge_signature,
    genre_metrics_for_users,
    graph_is_empty,
    load_genre_matrix,
    n_users_with_edges,
    pick_probe_rows,
    pick_random_base,
    probe_diagnostics,
    turnover_vs_ref,
)

from feddieg_core import __version__ as CORE_VERSION
from feddieg_core.cli import add_feddieg_args, config_from_args, prepare_runtime
from feddieg_core.config import FedDIEGConfig
from feddieg_core.data import ClientWindow, load_client_windows
from feddieg_core.evaluate import build_eval_model, evaluate_split
from feddieg_core.federated import FedTrainer, build_participation_schedule
from feddieg_core.utils import dump_json, make_device, mean, open_log, safe_slug, set_seed, write_csv

DG_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = DG_DIR / "results" / "raw"

# ======================================================================================
# CSV 表头（aggregate 依赖这些列名，改列名要同步改 dg_aggregate.py）
# ======================================================================================
ROUND_FIELDS = [
    "dataset", "arm", "arm_dir", "seed", "window", "window_round", "global_round",
    "top_k", "n_eval", "all_ndcg", "all_hr", "train_loss",
    # ---- 图状态 ----
    "graph_updates", "graph_n_edges", "graph_edge_sig", "graph_edge_changed_after_freeze",
    "ecr_all", "added_mean", "pool_fill",
    # ---- 冻结诊断（相对预热后第一次评测的邻居快照）----
    "freeze_ref_n", "turnover_from_ref", "frozen_share", "coverage_gain",
    # ---- 图协同分支的数值诊断 ----
    "probe_n", "probe_with_graph",
    "delta_rel_mean", "delta_rel_median", "delta_rel_max", "delta_cos_mean",
    "seq_norm_mean", "delta_norm_mean", "u_norm_mean",
    "neighbor_cos_mean", "neighbor_cos_base", "neighbor_lift",
    # ---- 联邦协议 ----
    "n_trained", "n_uploaded", "round_seconds",
]

USER_FIELDS = ["dataset", "arm", "arm_dir", "seed", "window", "user", "ndcg", "hr"]

#: 阶段 2 新增：**每轮**图指标（ECR 不再只写在评测轮 → C1）
GRAPH_ROUND_FIELDS = [
    "dataset", "arm", "arm_dir", "seed", "window", "window_round", "global_round",
    "n_edges", "is_update", "is_refresh", "n_updated",
    "ecr_round_mean", "added_round_sum", "rebuild_turnover", "pool_fill",
    "ecr_since_eval", "added_since_eval", "rounds_since_eval",
    "has_eval", "active_coverage", "n_eval",
]

#: 阶段 2 新增：逐用户「首次有邻居」的轮次（C6）
USER_COVERAGE_FIELDS = [
    "dataset", "arm", "arm_dir", "seed", "user", "first_covered_round",
]

#: Genre 分析用：逐用户「邻居 vs 随机」的 genre / 语义一致度（只在 --genre-npz 时产出）
GENRE_FIELDS = [
    "dataset", "arm", "arm_dir", "seed", "window", "user", "ndcg",
    "sem_nbr", "sem_base", "sem_lift", "sem_diff",
    "genre_nbr", "genre_base", "genre_lift", "genre_diff",
    "genrec_nbr", "genrec_base", "genrec_lift", "genrec_diff",
    "genre_entropy", "genre_drift",
]

PROBE_FIELDS = [
    "dataset", "arm", "arm_dir", "seed", "window", "user",
    "delta_rel", "delta_cos", "neighbor_cos", "neighbor_cos_base", "has_edge",
]

SUMMARY_FIELDS = [
    "dataset", "arm", "arm_dir", "seed",
    "final_window_ndcg", "avg_window_ndcg", "final_window_hr", "avg_window_hr",
    "mean_delta_rel", "mean_delta_rel_median", "mean_neighbor_lift", "mean_probe_with_graph",
    "mean_frozen_share", "mean_turnover_from_ref", "final_frozen_share", "run_tag",
    "genre_rows",
    "avg_ecr", "avg_added", "final_n_edges", "graph_updates_total",
    "edge_changed_after_freeze", "n_eval_users",
    # ---- 会直接影响可比性的配置（聚合时用来核对「确实只差图」）----
    "graph_mix", "rho", "neighbors", "graph_state", "graph_new_edge",
    "graph_update_scope", "graph_warmup_rounds", "freeze_round", "period",
    "build_only", "refresh_phase",
    "num_windows", "rounds_per_window", "total_rounds", "eval_every",
    "clients_per_round", "participation_rate", "local_epochs", "lr",
    "server_lr_init", "server_lr_final", "fedavg_weighting", "personalization", "personal_on_cpu",
    "lambda_temp", "lambda_struct", "top_k", "candidate_count",
    "probe_size", "n_users", "n_items",
    "mean_active_coverage", "n_first_covered", "n_user_updates", "mean_rebuild_turnover",
    "elapsed_seconds", "train_seconds", "eval_seconds", "core_version",
]


# ======================================================================================
# 单次运行
# ======================================================================================
def run_single(
    cfg: FedDIEGConfig,
    args: argparse.Namespace,
    windows: List[ClientWindow],
    device: torch.device,
    out_dir: Path,
) -> Dict[str, object]:
    set_seed(cfg.seed)
    num_users = int(windows[0].num_users)
    num_items = int(windows[0].num_items)
    clients_per_round = cfg.resolved_clients_per_round(num_users)
    schedule = build_participation_schedule(
        num_users, cfg.total_rounds, clients_per_round, cfg.seed
    )

    trainer = FedTrainer(cfg, num_users, num_items, device)
    build_only = bool(getattr(args, "build_only", False))
    refresh_phase = str(getattr(args, "refresh_phase", "warmup"))
    graph = build_graph(
        cfg, args.arm, num_users, device, args.freeze_round, args.refresh_rounds,
        build_only=build_only, refresh_phase=refresh_phase,
    )
    eval_model = build_eval_model(trainer, cfg)
    rng = np.random.default_rng(cfg.seed)

    arm_dir = arm_dir_name(
        args.arm, cfg, args.freeze_round, args.refresh_rounds, getattr(args, "run_tag", ""),
        build_only=build_only, refresh_phase=refresh_phase,
    )
    tag = f"[{args.arm}]"

    genre_mat = None
    if getattr(args, "genre_npz", None) is not None:
        genre_mat, _genre_names = load_genre_matrix(args.genre_npz)
        print(
            f"{tag} genre 诊断已启用：{genre_mat.shape[0]} 个物品 x {genre_mat.shape[1]} 个真实类型"
        )
    print(f"{tag} {ARM_LABELS[args.arm]}")
    print(f"{tag} 图设置：{json.dumps(graph.describe(), ensure_ascii=False)}")
    print(
        f"{tag} 联邦设置：N={num_users}, items={num_items}, m={clients_per_round} "
        f"(C={cfg.participation_rate}), E={cfg.local_epochs}, lr={cfg.lr}, "
        f"FedAvg={cfg.fedavg_weighting}, personalization={cfg.personalization}"
    )
    print(
        f"{tag} 只差图这一件事：lambda_temp={cfg.lambda_temp}, lambda_struct={cfg.lambda_struct}, "
        f"graph_mix={cfg.graph_mix}, neighbors={cfg.neighbors}, rho={cfg.rho}, "
        f"warmup={cfg.graph_warmup_rounds}, update_scope={cfg.graph_update_scope}"
    )
    if cfg.lambda_struct > 0:
        print(
            f"{tag} [注意] lambda_struct={cfg.lambda_struct} > 0：L_struct 定义在图量 Δe 上，"
            "NoGraph 臂没有图就拿不到它 ⇒ 图臂会多一个训练信号（不再是「只差图」的对照）。"
        )

    round_rows: List[Dict[str, object]] = []
    user_rows: List[Dict[str, object]] = []
    probe_rows: List[Dict[str, object]] = []
    graph_round_rows: List[Dict[str, object]] = []
    cover_rows: List[Dict[str, object]] = []
    window_ndcg: List[float] = []
    window_hr: List[float] = []
    delta_rel_points: List[float] = []
    delta_rel_medians: List[float] = []
    lifts: List[float] = []
    with_graph_points: List[float] = []
    ecrs: List[float] = []
    added: List[float] = []
    active_cov_points: List[float] = []
    rebuild_turnovers: List[float] = []
    edge_changed_total = 0
    eval_time = 0.0
    train_time = 0.0
    empty_warned = False
    start_time = time.time()
    global_round = 0
    final_n_edges = 0
    genre_rows: List[Dict[str, object]] = []
    profile_prev: Optional[np.ndarray] = None
    frozen_points: List[float] = []
    turnover_points: List[float] = []
    ref_ids: Optional[torch.Tensor] = None
    ref_valid: Optional[torch.Tensor] = None
    # ---- 阶段 2 新增：逐用户首次覆盖 + 评测间隔累计 ----
    first_covered: Optional[torch.Tensor] = (
        None if args.arm == ARM_NOGRAPH
        else torch.full((num_users,), -1, dtype=torch.long, device=device)
    )
    ecr_since_eval = 0.0
    added_since_eval = 0.0
    rounds_since_eval = 0

    for window_idx, window in enumerate(windows, start=1):
        print(
            f"\n{tag} Window {window_idx}/{cfg.num_windows} | eval_users={window.test_users.numel()}"
        )
        probe = pick_probe_rows(window, int(args.probe_size), int(cfg.seed) + window_idx)
        if probe is None:
            print(f"{tag} [warn] 本窗口没有可评测用户，跳过")
            continue
        probe_users, probe_seqs, _ = probe
        profile = None
        centered = None
        if genre_mat is not None:
            profile = build_user_genre_profiles(window, genre_mat)
            centered = center_profiles(profile)

        last_ndcg = 0.0
        last_hr = 0.0
        frozen_sig: Optional[str] = None

        for window_round in range(1, cfg.rounds_per_window + 1):
            global_round += 1
            round_start = time.time()
            sampled = schedule[global_round - 1]

            # ---------------- ① 一轮联邦训练（本地训练 + 聚合 + 摘要上传）
            stats = trainer.train_round(sampled, window, graph, global_round, rng)

            # ---------------- ② 服务器端图更新（臂的差别只在这里）
            if cfg.graph_update_scope == "all":
                update_users = torch.arange(num_users, device=device)
            else:
                update_users = torch.tensor(sampled, dtype=torch.long, device=device)
            if args.arm == ARM_FRESH_ONLY and hasattr(graph, "last_user_update"):
                # 只更新「自上次更新后有新上传」的用户（针对性更新策略）
                pending = (
                    trainer.last_upload_round[update_users]
                    > graph.last_user_update[update_users]
                )
                update_users = update_users[pending]
            gstats = graph.update(
                trainer.summary_bank, trainer.summary_active, update_users, global_round
            )
            train_time += time.time() - round_start

            # ---------------- ②' 每轮图指标（C1：ECR 不再只写在评测轮）
            n_edges_now = n_users_with_edges(graph)
            if first_covered is not None:
                covered_now = graph.mask & (graph.ids >= 0).any(dim=1)
                newly_covered = covered_now & (first_covered < 0)
                if bool(newly_covered.any()):
                    first_covered[newly_covered] = global_round
            n_updated = int(len(gstats)) if gstats is not None else 0
            ecr_round = (
                float(gstats.ecr.mean()) if (gstats is not None and len(gstats) > 0)
                else float("nan")
            )
            added_round = (
                float(gstats.n_added.sum()) if (gstats is not None and len(gstats) > 0) else 0.0
            )
            pool_fill_round = float(gstats.pool_fill) if gstats is not None else float("nan")
            rebuild_turnover = float(
                getattr(graph, "last_rebuild_turnover", float("nan"))
            )
            if np.isfinite(rebuild_turnover):
                rebuild_turnovers.append(rebuild_turnover)
            if np.isfinite(ecr_round):
                ecr_since_eval += ecr_round
            added_since_eval += added_round
            rounds_since_eval += 1

            should_eval = (window_round % max(1, cfg.eval_every) == 0) or (
                window_round == cfg.rounds_per_window
            )
            gr_row = {
                "dataset": cfg.dataset, "arm": args.arm, "arm_dir": arm_dir, "seed": cfg.seed,
                "window": window_idx, "window_round": window_round,
                "global_round": global_round,
                "n_edges": n_edges_now,
                "is_update": int(gstats is not None and len(gstats) > 0),
                "is_refresh": int(bool(getattr(graph, "last_update_was_refresh", False))),
                "n_updated": n_updated,
                "ecr_round_mean": ecr_round, "added_round_sum": added_round,
                "rebuild_turnover": float(getattr(graph, "last_rebuild_turnover", float("nan"))),
                "pool_fill": pool_fill_round,
                "ecr_since_eval": float("nan"), "added_since_eval": float("nan"),
                "rounds_since_eval": 0,
                "has_eval": 0, "active_coverage": float("nan"), "n_eval": 0,
            }
            if not should_eval:
                graph_round_rows.append(gr_row)
                continue

            # ---------------- ③ 评测（与 val01/02/03 完全同一套评测函数）
            eval_start = time.time()
            result = evaluate_split(trainer, eval_model, window, graph, "test", cfg)
            eval_time += time.time() - eval_start
            uids, hrs, ndcgs = result["users"], result["hr"], result["ndcg"]
            all_ndcg = float(ndcgs.mean()) if ndcgs.size else 0.0
            all_hr = float(hrs.mean()) if hrs.size else 0.0
            last_ndcg, last_hr = all_ndcg, all_hr

            # ---------------- ③' 本窗口有效覆盖（A6：分母 = 本窗口被评测用户）
            if first_covered is not None and uids.size:
                uids_t = torch.as_tensor(uids, dtype=torch.long, device=device)
                active_cov = float(
                    (graph.ids[uids_t] >= 0).any(dim=1).to(torch.float32).mean().item()
                )
            else:
                active_cov = float("nan")
            gr_row["has_eval"] = 1
            gr_row["n_eval"] = int(ndcgs.size)
            gr_row["active_coverage"] = active_cov
            gr_row["ecr_since_eval"] = ecr_since_eval
            gr_row["added_since_eval"] = added_since_eval
            gr_row["rounds_since_eval"] = rounds_since_eval
            graph_round_rows.append(gr_row)
            if np.isfinite(active_cov):
                active_cov_points.append(active_cov)
            ecr_since_eval = 0.0
            added_since_eval = 0.0
            rounds_since_eval = 0

            # ---------------- ④ 图协同分支诊断（三条臂同一批探针用户）
            # 邻居质量的随机基线每次评测重抽：窗口 1 开头摘要库还是空的，
            # 若整窗只抽一次，那一整窗的 neighbor_lift 都会是 nan。
            random_base = pick_random_base(
                trainer.summary_active, int(args.probe_size), cfg.seed + global_round
            )
            diag, per_user, _ = probe_diagnostics(
                trainer, eval_model, graph,
                probe_users, probe_seqs, None, cfg, random_base=random_base,
            )

            # ---------------- ⑤ 严格性硬检查
            if args.arm == ARM_NOGRAPH and not (diag["delta_rel_max"] <= 1e-12):
                raise RuntimeError(
                    f"NoGraph 臂的图分支没有被真正关掉：delta_rel_max={diag['delta_rel_max']!r}"
                    f"（第 {global_round} 轮）。这条臂必须满足 u_i = seq_emb_i。"
                )

            sig = edge_signature(graph) if args.arm != ARM_NOGRAPH else "-"
            changed = 0
            # static 臂：冻结轮次之后边的签名必须一位都不变（这是这条臂的定义）
            if args.arm == ARM_STATIC:
                if global_round > int(graph.freeze_round):
                    if frozen_sig is None:
                        frozen_sig = sig
                    elif sig != frozen_sig:
                        raise RuntimeError(
                            f"StaticGraph 臂在冻结后仍被改动了（第 {global_round} 轮，"
                            f"sig {frozen_sig} -> {sig}）。这条臂必须完全冻结。"
                        )
                changed = 0
                edge_changed_total += changed

            n_edges = n_users_with_edges(graph)
            final_n_edges = n_edges
            if (
                not empty_warned
                and args.arm != ARM_NOGRAPH
                and global_round > int(cfg.graph_warmup_rounds) + 1
                and graph_is_empty(graph)
            ):
                empty_warned = True
                print(
                    f"{tag} [warn] 第 {global_round} 轮图仍然是空的（没有任何用户拿到邻居）："
                    "这条图臂实际等价于 NoGraph，对比将没有意义。请检查 "
                    "graph_warmup_rounds / graph_state / min_sim / 参与率。"
                )

            # ---------------- 冻结诊断（相对预热后第一次评测的邻居快照）
            if (
                args.arm != ARM_NOGRAPH
                and ref_ids is None
                and global_round > int(cfg.graph_warmup_rounds)
            ):
                ref_ids = graph.ids.detach().clone()
                ref_valid = graph.mask.detach().clone() & (ref_ids >= 0).any(dim=1)
            if ref_ids is not None and ref_valid is not None:
                fz = turnover_vs_ref(graph, ref_ids, ref_valid)
            else:
                fz = {
                    "freeze_ref_n": 0.0, "turnover_from_ref": float("nan"),
                    "frozen_share": float("nan"), "coverage_gain": 0.0,
                }
            if np.isfinite(fz["frozen_share"]):
                frozen_points.append(float(fz["frozen_share"]))
                turnover_points.append(float(fz["turnover_from_ref"]))

            ecr_all = float(gstats.ecr.mean()) if (gstats is not None and len(gstats) > 0) else 0.0
            added_mean = (
                float(gstats.n_added.mean()) if (gstats is not None and len(gstats) > 0) else 0.0
            )
            pool_fill = float(getattr(gstats, "pool_fill", 0.0)) if gstats is not None else 0.0

            round_rows.append(
                {
                    "dataset": cfg.dataset, "arm": args.arm, "arm_dir": arm_dir, "seed": cfg.seed,
                    "window": window_idx, "window_round": window_round, "global_round": global_round,
                    "top_k": cfg.top_k, "n_eval": int(ndcgs.size),
                    "all_ndcg": all_ndcg, "all_hr": all_hr, "train_loss": stats.loss,
                    "graph_updates": int(getattr(graph, "n_updates", 0)),
                    "graph_n_edges": n_edges, "graph_edge_sig": sig,
                    "graph_edge_changed_after_freeze": changed,
                    "ecr_all": ecr_all, "added_mean": added_mean, "pool_fill": pool_fill,
                    "freeze_ref_n": fz["freeze_ref_n"],
                    "turnover_from_ref": fz["turnover_from_ref"],
                    "frozen_share": fz["frozen_share"],
                    "coverage_gain": fz["coverage_gain"],
                    "probe_n": diag["probe_n"], "probe_with_graph": diag["probe_with_graph"],
                    "delta_rel_mean": diag["delta_rel_mean"],
                    "delta_rel_median": diag["delta_rel_median"],
                    "delta_rel_max": diag["delta_rel_max"],
                    "delta_cos_mean": diag["delta_cos_mean"],
                    "seq_norm_mean": diag["seq_norm_mean"],
                    "delta_norm_mean": diag["delta_norm_mean"],
                    "u_norm_mean": diag["u_norm_mean"],
                    "neighbor_cos_mean": diag["neighbor_cos_mean"],
                    "neighbor_cos_base": diag["neighbor_cos_base"],
                    "neighbor_lift": diag["neighbor_lift"],
                    "n_trained": stats.n_trained, "n_uploaded": stats.n_uploads,
                    "round_seconds": time.time() - round_start,
                }
            )
            delta_rel_points.append(diag["delta_rel_mean"])
            delta_rel_medians.append(diag["delta_rel_median"])
            lifts.append(diag["neighbor_lift"])
            with_graph_points.append(diag["probe_with_graph"])
            ecrs.append(ecr_all)
            added.append(added_mean)

            print(
                f"{tag} W{window_idx} R{window_round:03d}/{cfg.rounds_per_window} | "
                f"loss {stats.loss:.4f} | NDCG@{cfg.top_k} {all_ndcg:.4f} HR {all_hr:.4f} "
                f"(n={int(ndcgs.size)}) | "
                f"edges {n_edges} upd {int(getattr(graph, 'n_updates', 0))} ECR {ecr_all:.4f} "
                f"(+{added_mean:.2f}) | delta_rel {diag['delta_rel_mean']:.2e} "
                f"nbr_lift {diag['neighbor_lift']:.2f} | {time.time() - round_start:.2f}s"
            )

            # ---------------- ⑥ 逐用户 / 逐用户探针（只在窗口最后一次评测写）
            if window_round == cfg.rounds_per_window:
                if args.user_rows:
                    for user, ndcg_v, hr_v in zip(uids.tolist(), ndcgs.tolist(), hrs.tolist()):
                        user_rows.append(
                            {
                                "dataset": cfg.dataset, "arm": args.arm, "arm_dir": arm_dir,
                                "seed": cfg.seed, "window": window_idx,
                                "user": int(user), "ndcg": float(ndcg_v), "hr": float(hr_v),
                            }
                        )
                if genre_mat is not None and profile is not None:
                    uids_t = torch.as_tensor(uids, dtype=torch.long, device=device)
                    gm = genre_metrics_for_users(
                        profile, centered, profile_prev, trainer.summary_bank,
                        graph, uids_t, random_base,
                    )
                    for i, user in enumerate(uids.tolist()):
                        genre_rows.append(
                            {
                                "dataset": cfg.dataset, "arm": args.arm, "arm_dir": arm_dir,
                                "seed": cfg.seed, "window": window_idx, "user": int(user),
                                "ndcg": float(ndcgs[i]),
                                "sem_nbr": float(gm["sem_nbr"][i]),
                                "sem_base": float(gm["sem_base"][i]),
                                "sem_lift": float(gm["sem_lift"][i]),
                                "sem_diff": float(gm["sem_diff"][i]),
                                "genre_nbr": float(gm["genre_nbr"][i]),
                                "genre_base": float(gm["genre_base"][i]),
                                "genre_lift": float(gm["genre_lift"][i]),
                                "genre_diff": float(gm["genre_diff"][i]),
                                "genrec_nbr": float(gm["genrec_nbr"][i]),
                                "genrec_base": float(gm["genrec_base"][i]),
                                "genrec_lift": float(gm["genrec_lift"][i]),
                                "genrec_diff": float(gm["genrec_diff"][i]),
                                "genre_entropy": float(gm["genre_entropy"][i]),
                                "genre_drift": float(gm["genre_drift"][i]),
                            }
                        )

                rel_arr = per_user["delta_rel"]
                for i, user in enumerate(probe_users.tolist()):
                    probe_rows.append(
                        {
                            "dataset": cfg.dataset, "arm": args.arm, "arm_dir": arm_dir,
                            "seed": cfg.seed, "window": window_idx, "user": int(user),
                            "delta_rel": float(rel_arr[i]),
                            "delta_cos": float(per_user["delta_cos"][i]),
                            "neighbor_cos": float(per_user["neighbor_cos"][i]),
                            "neighbor_cos_base": float(per_user["neighbor_cos_base"][i]),
                            "has_edge": float(per_user["has_edge"][i]),
                        }
                    )

        window_ndcg.append(last_ndcg)
        window_hr.append(last_hr)
        if profile is not None:
            profile_prev = profile
        write_csv(out_dir / "dg_round_metrics.csv", round_rows, ROUND_FIELDS)
        write_csv(out_dir / "dg_graph_round_metrics.csv", graph_round_rows, GRAPH_ROUND_FIELDS)
        if genre_rows:
            write_csv(out_dir / "dg_genre_metrics.csv", genre_rows, GENRE_FIELDS)
        if user_rows:
            write_csv(out_dir / "dg_user_metrics.csv", user_rows, USER_FIELDS)
        if probe_rows:
            write_csv(out_dir / "dg_probe_metrics.csv", probe_rows, PROBE_FIELDS)
        print(
            f"{tag} W{window_idx} end | NDCG@{cfg.top_k} {last_ndcg:.4f} HR {last_hr:.4f} | "
            f"rows={len(user_rows)} probe={len(probe_rows)} | {time.time() - start_time:.0f}s"
        )

    # ---- 逐用户「首次有邻居」的轮次（C6）：一次运行写一次 ----
    if first_covered is not None:
        for _u in range(num_users):
            cover_rows.append(
                {
                    "dataset": cfg.dataset, "arm": args.arm, "arm_dir": arm_dir,
                    "seed": cfg.seed, "user": int(_u),
                    "first_covered_round": int(first_covered[_u]),
                }
            )
        write_csv(out_dir / "dg_user_coverage.csv", cover_rows, USER_COVERAGE_FIELDS)

    summary = {
        "dataset": cfg.dataset, "arm": args.arm, "arm_dir": arm_dir, "seed": cfg.seed,
        "final_window_ndcg": window_ndcg[-1] if window_ndcg else float("nan"),
        "avg_window_ndcg": mean(window_ndcg),
        "final_window_hr": window_hr[-1] if window_hr else float("nan"),
        "avg_window_hr": mean(window_hr),
        "mean_delta_rel": mean(delta_rel_points),
        "mean_delta_rel_median": mean(delta_rel_medians),
        "mean_neighbor_lift": mean(lifts),
        "mean_probe_with_graph": mean(with_graph_points),
        "mean_frozen_share": mean(frozen_points),
        "mean_turnover_from_ref": mean(turnover_points),
        "final_frozen_share": frozen_points[-1] if frozen_points else float("nan"),
        "run_tag": getattr(args, "run_tag", ""),
        "genre_rows": len(genre_rows),
        "avg_ecr": mean(ecrs), "avg_added": mean(added),
        "final_n_edges": final_n_edges,
        "graph_updates_total": int(getattr(graph, "n_updates", 0)),
        "edge_changed_after_freeze": edge_changed_total,
        "n_eval_users": int(windows[0].test_users.numel()) if windows else 0,
        "graph_mix": cfg.graph_mix, "rho": cfg.rho, "neighbors": cfg.neighbors,
        "graph_state": cfg.graph_state, "graph_new_edge": cfg.graph_new_edge,
        "graph_update_scope": cfg.graph_update_scope,
        "graph_warmup_rounds": cfg.graph_warmup_rounds,
        "freeze_round": (
            int(getattr(graph, "freeze_round", 0) or 0) if args.arm == ARM_STATIC else ""
        ),
        "period": (
            int(getattr(graph, "period", 0) or 0)
            if args.arm == "periodic"
            else ""
        ),
        "build_only": int(bool(getattr(args, "build_only", False))),
        "refresh_phase": str(getattr(args, "refresh_phase", "warmup")),
        "mean_active_coverage": mean(active_cov_points),
        "mean_rebuild_turnover": mean(rebuild_turnovers),
        "n_first_covered": (
            int((first_covered >= 0).sum().item()) if first_covered is not None else 0
        ),
        "n_user_updates": int(getattr(graph, "n_user_updates", 0)),
        "num_windows": cfg.num_windows, "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": cfg.total_rounds, "eval_every": cfg.eval_every,
        "clients_per_round": clients_per_round, "participation_rate": cfg.participation_rate,
        "local_epochs": cfg.local_epochs, "lr": cfg.lr,
        "server_lr_init": cfg.server_lr_init, "server_lr_final": cfg.server_lr_final,
        "fedavg_weighting": cfg.fedavg_weighting, "personalization": cfg.personalization,
        "personal_on_cpu": int(bool(cfg.personal_on_cpu)),
        "lambda_temp": cfg.lambda_temp, "lambda_struct": cfg.lambda_struct,
        "top_k": cfg.top_k, "candidate_count": cfg.candidate_count,
        "probe_size": int(args.probe_size), "n_users": num_users, "n_items": num_items,
        "elapsed_seconds": time.time() - start_time,
        "train_seconds": train_time, "eval_seconds": eval_time,
        "core_version": CORE_VERSION,
    }
    write_csv(out_dir / "dg_summary.csv", [summary], SUMMARY_FIELDS)
    print(
        f"\n{tag} seed{cfg.seed} done in {summary['elapsed_seconds']:.1f}s | "
        f"avg_window_ndcg={summary['avg_window_ndcg']:.4f} "
        f"final={summary['final_window_ndcg']:.4f} | "
        f"delta_rel={summary['mean_delta_rel']:.3e} "
        f"nbr_lift={summary['mean_neighbor_lift']:.2f} | "
        f"edges={final_n_edges} updates={summary['graph_updates_total']}"
    )
    return summary


# ======================================================================================
# CLI
# ======================================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="val_DynamicGraph: 同一编码器/训练协议下，NoGraph vs StaticGraph vs DynamicGraph"
    )
    add_feddieg_args(parser)
    add_dg_args(parser)   # 必须在 add_feddieg_args 之后（会把 lambda_struct 默认改成 0）
    parser.add_argument(
        "--run-tag", default="",
        help="追加到结果目录名的标记，用于区分「同臂不同配置」（如 genre）",
    )
    parser.add_argument(
        "--genre-npz", type=Path, default=None,
        help="真实 genre 矩阵（val_DynamicGraph/data/item_genres_ml1m.npz）；给了就写 dg_genre_metrics.csv",
    )
    parser.set_defaults(out_root=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.validate()
    prepare_runtime(cfg)
    set_seed(cfg.seed)
    device = make_device(cfg.device)

    arm_dir = arm_dir_name(
        args.arm, cfg, args.freeze_round, args.refresh_rounds, getattr(args, "run_tag", ""),
        build_only=bool(getattr(args, "build_only", False)),
        refresh_phase=str(getattr(args, "refresh_phase", "warmup")),
    )
    out_root = Path(args.out_root)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = out_root / safe_slug(cfg.dataset) / arm_dir / f"seed_{cfg.seed}"
    if args.smoke and args.out_dir is None:
        out_dir = Path(str(out_root) + "_smoke") / out_dir.relative_to(out_root)

    open_log(out_dir, "dg_run.log")
    print(f"Output directory: {out_dir}")
    print(
        f"Device: {device}"
        + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
    )
    print("Config:", json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str))
    dump_json(
        out_dir / "config.json",
        {
            **asdict(cfg),
            "core_version": CORE_VERSION,
            "arm": args.arm,
            "arm_dir": arm_dir,
            "freeze_round": args.freeze_round,
            "refresh_rounds": args.refresh_rounds,
            "build_only": bool(getattr(args, "build_only", False)),
            "refresh_phase": str(getattr(args, "refresh_phase", "warmup")),
            "probe_size": int(args.probe_size),
            "run_tag": getattr(args, "run_tag", ""),
            "genre_npz": str(args.genre_npz) if getattr(args, "genre_npz", None) else None,
        },
    )

    windows = load_client_windows(cfg)
    run_single(cfg, args, windows, device, out_dir)
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
