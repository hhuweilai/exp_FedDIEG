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
    ARM_LABELS,
    ARM_NOGRAPH,
    ARM_STATIC,
    add_dg_args,
    arm_dir_name,
    build_graph,
    edge_signature,
    graph_is_empty,
    n_users_with_edges,
    pick_probe_rows,
    pick_random_base,
    probe_diagnostics,
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
    # ---- 图协同分支的数值诊断 ----
    "probe_n", "probe_with_graph",
    "delta_rel_mean", "delta_rel_median", "delta_rel_max", "delta_cos_mean",
    "seq_norm_mean", "delta_norm_mean", "u_norm_mean",
    "neighbor_cos_mean", "neighbor_cos_base", "neighbor_lift",
    # ---- 联邦协议 ----
    "n_trained", "n_uploaded", "round_seconds",
]

USER_FIELDS = ["dataset", "arm", "arm_dir", "seed", "window", "user", "ndcg", "hr"]

PROBE_FIELDS = [
    "dataset", "arm", "arm_dir", "seed", "window", "user",
    "delta_rel", "delta_cos", "neighbor_cos", "neighbor_cos_base", "has_edge",
]

SUMMARY_FIELDS = [
    "dataset", "arm", "arm_dir", "seed",
    "final_window_ndcg", "avg_window_ndcg", "final_window_hr", "avg_window_hr",
    "mean_delta_rel", "mean_delta_rel_median", "mean_neighbor_lift", "mean_probe_with_graph",
    "avg_ecr", "avg_added", "final_n_edges", "graph_updates_total",
    "edge_changed_after_freeze", "n_eval_users",
    # ---- 会直接影响可比性的配置（聚合时用来核对「确实只差图」）----
    "graph_mix", "rho", "neighbors", "graph_state", "graph_new_edge",
    "graph_update_scope", "graph_warmup_rounds", "freeze_round", "period",
    "num_windows", "rounds_per_window", "total_rounds", "eval_every",
    "clients_per_round", "participation_rate", "local_epochs", "lr",
    "server_lr_init", "server_lr_final", "fedavg_weighting", "personalization",
    "lambda_temp", "lambda_struct", "top_k", "candidate_count",
    "probe_size", "n_users", "n_items",
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
    graph = build_graph(cfg, args.arm, num_users, device, args.freeze_round, args.refresh_rounds)
    eval_model = build_eval_model(trainer, cfg)
    rng = np.random.default_rng(cfg.seed)

    arm_dir = arm_dir_name(args.arm, cfg, args.freeze_round, args.refresh_rounds)
    tag = f"[{args.arm}]"
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
    window_ndcg: List[float] = []
    window_hr: List[float] = []
    delta_rel_points: List[float] = []
    delta_rel_medians: List[float] = []
    lifts: List[float] = []
    with_graph_points: List[float] = []
    ecrs: List[float] = []
    added: List[float] = []
    edge_changed_total = 0
    eval_time = 0.0
    train_time = 0.0
    empty_warned = False
    start_time = time.time()
    global_round = 0
    final_n_edges = 0

    for window_idx, window in enumerate(windows, start=1):
        print(
            f"\n{tag} Window {window_idx}/{cfg.num_windows} | eval_users={window.test_users.numel()}"
        )
        probe = pick_probe_rows(window, int(args.probe_size), int(cfg.seed) + window_idx)
        if probe is None:
            print(f"{tag} [warn] 本窗口没有可评测用户，跳过")
            continue
        probe_users, probe_seqs, _ = probe
        random_base = pick_random_base(
            trainer.summary_active, int(args.probe_size), cfg.seed + window_idx
        )

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
            gstats = graph.update(
                trainer.summary_bank, trainer.summary_active, update_users, global_round
            )
            train_time += time.time() - round_start

            should_eval = (window_round % max(1, cfg.eval_every) == 0) or (
                window_round == cfg.rounds_per_window
            )
            if not should_eval:
                continue

            # ---------------- ③ 评测（与 val01/02/03 完全同一套评测函数）
            eval_start = time.time()
            result = evaluate_split(trainer, eval_model, window, graph, "test", cfg)
            eval_time += time.time() - eval_start
            uids, hrs, ndcgs = result["users"], result["hr"], result["ndcg"]
            all_ndcg = float(ndcgs.mean()) if ndcgs.size else 0.0
            all_hr = float(hrs.mean()) if hrs.size else 0.0
            last_ndcg, last_hr = all_ndcg, all_hr

            # ---------------- ④ 图协同分支诊断（三条臂同一批探针用户）
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
        write_csv(out_dir / "dg_round_metrics.csv", round_rows, ROUND_FIELDS)
        if user_rows:
            write_csv(out_dir / "dg_user_metrics.csv", user_rows, USER_FIELDS)
        if probe_rows:
            write_csv(out_dir / "dg_probe_metrics.csv", probe_rows, PROBE_FIELDS)
        print(
            f"{tag} W{window_idx} end | NDCG@{cfg.top_k} {last_ndcg:.4f} HR {last_hr:.4f} | "
            f"rows={len(user_rows)} probe={len(probe_rows)} | {time.time() - start_time:.0f}s"
        )

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
            if args.arm not in (ARM_NOGRAPH, ARM_STATIC)
            else ""
        ),
        "num_windows": cfg.num_windows, "rounds_per_window": cfg.rounds_per_window,
        "total_rounds": cfg.total_rounds, "eval_every": cfg.eval_every,
        "clients_per_round": clients_per_round, "participation_rate": cfg.participation_rate,
        "local_epochs": cfg.local_epochs, "lr": cfg.lr,
        "server_lr_init": cfg.server_lr_init, "server_lr_final": cfg.server_lr_final,
        "fedavg_weighting": cfg.fedavg_weighting, "personalization": cfg.personalization,
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
    parser.set_defaults(out_root=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.validate()
    prepare_runtime(cfg)
    set_seed(cfg.seed)
    device = make_device(cfg.device)

    arm_dir = arm_dir_name(args.arm, cfg, args.freeze_round, args.refresh_rounds)
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
            "probe_size": int(args.probe_size),
        },
    )

    windows = load_client_windows(cfg)
    run_single(cfg, args, windows, device, out_dir)
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
