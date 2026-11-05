"""val01 / val02 / val03 共用的命令行参数（全部直接映射到 `FedDIEGConfig`）。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from .config import FedDIEGConfig


def add_feddieg_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("数据与模型（对齐 aligned 主程序）")
    g.add_argument("--dataset", default="movielens-1m")
    g.add_argument("--latent-dim", type=int, default=64)
    g.add_argument("--max-len", type=int, default=50)
    g.add_argument("--num-heads", type=int, default=4)
    g.add_argument("--num-blocks", type=int, default=2)
    g.add_argument("--dropout-rate", type=float, default=0.2)
    g.add_argument("--top-k", type=int, default=10, help="评测截断 @K")
    g.add_argument("--candidate-count", type=int, default=99)
    g.add_argument("--window-mode", choices=["slice", "cumulative"], default="slice")

    g = parser.add_argument_group("联邦协议（对齐 FedDIEG.py / aligned）")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--num-windows", type=int, default=5)
    g.add_argument("--rounds-per-window", type=int, default=120)
    g.add_argument("--participation-rate", type=float, default=0.1, help="C：每轮采样比例")
    g.add_argument("--clients-per-round", type=int, default=None)
    g.add_argument("--local-epochs", type=int, default=1, help="E：客户端本地 epoch 数")
    g.add_argument("--lr", type=float, default=0.001)
    g.add_argument("--server-lr-init", type=float, default=15.0, help="item_emb 聚合外推初值")
    g.add_argument("--server-lr-final", type=float, default=1.0)
    g.add_argument("--fedavg-weighting", choices=["uniform", "sample"], default="uniform",
                   help="uniform=A/B 版（按客户端数平均）；sample=按本地样本数加权")
    g.add_argument("--personalization", choices=["encoder", "none"], default="encoder",
                   help="encoder=A/B 版（编码器留本地）；none=全部共享")
    g.add_argument("--personal-on-cpu", dest="personal_on_cpu", action="store_true", default=True,
                   help="个性化层存 CPU（默认，省显存）")
    g.add_argument("--personal-on-gpu", dest="personal_on_cpu", action="store_false",
                   help="个性化层驻留 GPU（与 A/B 版字面一致；N 大时吃显存）")
    g.add_argument("--lambda-temp", type=float, default=1e-5)
    g.add_argument("--lambda-struct", type=float, default=1e-5)
    g.add_argument("--struct-target", choices=["delta_e", "edge_weight"], default="delta_e",
                   help="delta_e=与 Δe 同量纲（修复版）；edge_weight=A/B 字面行为")

    g = parser.add_argument_group("动态兴趣图（论文公式 + 审计修复）")
    g.add_argument("--neighbors", type=int, default=20, help="Top-K 邻居数")
    g.add_argument("--rho", type=float, default=0.2, help="EMA 系数 ρ")
    g.add_argument("--min-sim", type=float, default=0.0, help="相似度阈值（A/B 版没有，默认 0）")
    g.add_argument("--graph-warmup-rounds", type=int, default=20)
    g.add_argument("--graph-state", choices=["pool", "topk"], default="pool",
                   help="pool=保留候选池（修复冻结）；topk=旧行为（复现冻结）")
    g.add_argument("--graph-pool-size", type=int, default=0, help="0 => max(8*neighbors,128)")
    g.add_argument("--graph-new-edge", choices=["zero", "warm"], default="zero",
                   help="zero=论文 0 初值；warm=首次观测直接取 Δe")
    g.add_argument("--graph-update-scope", choices=["all", "sampled"], default="all",
                   help="all=论文异步更新（全部活跃节点）；sampled=旧行为")
    g.add_argument("--graph-mix", type=float, default=1.0, help="图注入强度（A/B 版=1.0）")
    g.add_argument("--graph-block-size", type=int, default=1024)
    g.add_argument("--summary-momentum", type=float, default=0.0,
                   help="0=直接替换（论文）；>0=动量 EMA（旧行为）")

    g = parser.add_argument_group("运行")
    g.add_argument("--eval-every", type=int, default=10)
    g.add_argument("--eval-batch-size", type=int, default=2048)
    g.add_argument("--max-eval-users", type=int, default=None)
    g.add_argument("--num-threads", type=int, default=4)
    g.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    g.add_argument("--smoke", action="store_true", help="极小配置快速自检（几十秒）")
    g.add_argument("--out-root", type=Path, default=None)
    g.add_argument("--out-dir", type=Path, default=None)


SMOKE_OVERRIDES: Dict[str, Any] = dict(
    latent_dim=16,
    max_len=8,
    num_heads=2,
    num_blocks=1,
    num_windows=2,
    rounds_per_window=2,
    clients_per_round=8,
    eval_every=1,
    eval_batch_size=32,
    max_eval_users=32,
    neighbors=4,
    graph_pool_size=32,
    graph_warmup_rounds=0,
    graph_block_size=256,
    participation_rate=0.5,
)


def config_from_args(args: argparse.Namespace, **extra: Any) -> FedDIEGConfig:
    """把命令行参数转成 `FedDIEGConfig`；``--smoke`` 时套用极小配置。"""
    values: Dict[str, Any] = dict(
        dataset=args.dataset,
        latent_dim=args.latent_dim,
        max_len=args.max_len,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
        top_k=args.top_k,
        candidate_count=args.candidate_count,
        window_mode=args.window_mode,
        seed=args.seed,
        num_windows=args.num_windows,
        rounds_per_window=args.rounds_per_window,
        participation_rate=args.participation_rate,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        lr=args.lr,
        server_lr_init=args.server_lr_init,
        server_lr_final=args.server_lr_final,
        fedavg_weighting=args.fedavg_weighting,
        personalization=args.personalization,
        personal_on_cpu=getattr(args, "personal_on_cpu", True),
        lambda_temp=args.lambda_temp,
        lambda_struct=args.lambda_struct,
        struct_target=args.struct_target,
        neighbors=args.neighbors,
        rho=args.rho,
        min_sim=args.min_sim,
        graph_warmup_rounds=args.graph_warmup_rounds,
        graph_state=args.graph_state,
        graph_pool_size=args.graph_pool_size,
        graph_new_edge=args.graph_new_edge,
        graph_update_scope=args.graph_update_scope,
        graph_mix=args.graph_mix,
        graph_block_size=args.graph_block_size,
        summary_momentum=args.summary_momentum,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        max_eval_users=args.max_eval_users,
        num_threads=args.num_threads,
        device=args.device,
    )
    if getattr(args, "smoke", False):
        values.update(SMOKE_OVERRIDES)
    values.update(extra)
    cfg = FedDIEGConfig(**values)
    cfg.rounds = cfg.total_rounds
    return cfg


def prepare_runtime(cfg: FedDIEGConfig) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cfg.num_threads))
    import torch

    torch.set_num_threads(max(1, int(cfg.num_threads)))
