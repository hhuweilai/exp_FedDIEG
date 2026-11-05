"""FedDIEG 超参定义。

分三组：
  A) 模型 / 数据      —— 对齐 `exp_01/run_exp01_movielens_aligned.py` 的 `ExpConfig`
  B) 联邦协议         —— 对齐 `FedDIEG.py` L44-L64 与 aligned L1401-L1500
  C) 动态兴趣图       —— 对齐论文公式（`tools/edit_mid_report.py` L215-L287）+ 审计修复

每个字段的注释都写明"对齐到哪一行"或"为什么与旧实现不同"。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass
class FedDIEGConfig:
    # ==================================================================================
    # A) 数据与模型（对齐 aligned 主程序）
    # ==================================================================================
    dataset: str = "movielens-1m"
    latent_dim: int = 64                    # aligned L49 LATENT_DIM=64
    max_len: int = 50                       # aligned L52 MAX_LEN=50
    num_heads: int = 4
    num_blocks: int = 2
    dropout_rate: float = 0.2
    top_k: int = 10                         # 评测截断（@10）
    candidate_count: int = 99               # 候选集 = 1 正 + 99 负
    window_mode: str = "slice"              # slice / cumulative（沿用 exp_02 协议）

    # ==================================================================================
    # B) 联邦协议（对齐 FedDIEG.py L44-L64 / aligned L1401-L1500）
    # ==================================================================================
    seed: int = 42
    rounds: int = 600                       # NUM_ROUNDS
    participation_rate: float = 0.1         # C=0.1；m = max(int(C*N), 1)
    clients_per_round: Optional[int] = None  # 显式覆盖 m（None 时按 C 计算）
    local_epochs: int = 1                   # E=1：客户端本地 epoch 数
    lr: float = 0.001                       # 客户端本地 Adam lr
    server_lr_init: float = 15.0            # INITIAL_SERVER_LR（只作用于 item_emb）
    server_lr_final: float = 1.0            # FINAL_SERVER_LR
    server_lr_decay_frac: float = 0.8       # decay_steps = int(rounds*0.8)
    fedavg_weighting: str = "uniform"       # uniform=A/B 版（/m）；sample=按本地样本数加权
    personalization: str = "encoder"        # encoder=A/B 版；none=全部共享（可批量训练）
    # 个性化层存在哪里：True=存 CPU（每客户端 214 KB，N=5万时省 10 GB 显存；默认）
    #                   False=存 GPU（与 A/B 版字面一致，更快，但显存随 N 线性增长）
    personal_on_cpu: bool = True

    # 联合损失（对齐 A/B：lambda_1 = lambda_2 = 1e-5）
    lambda_temp: float = 1e-5
    lambda_struct: float = 1e-5
    # L_struct 的目标量：delta_e=上一轮的 Δe（量纲一致，修复版）；
    # edge_weight=上一轮的 softmax 边权（A/B 版字面行为，量纲不一致，仅用于复现）
    struct_target: str = "delta_e"

    # ==================================================================================
    # C) 动态兴趣图（论文公式 + 审计修复）
    # ==================================================================================
    neighbors: int = 20                     # NEIGHBORHOOD_SIZE=20
    rho: float = 0.2                        # RHO=0.2（论文默认；扫描时命令行覆盖）
    min_sim: float = 0.0                    # A/B 版没有阈值；>0 时只作为"存在性"门槛
    graph_warmup_rounds: int = 20           # WARMUP_ROUNDS=20
    graph_update_scope: str = "all"         # all=论文"异步更新"（每轮全部活跃节点）；
                                            # sampled=旧行为（只重算被抽样用户）
    # 状态存放方式：
    #   pool  —— 保留候选池：每个节点跟踪 graph_pool_size 条候选边的 EMA，
    #            保留规则 max(EMA, Δe)（"历史强或当前强都留下"），
    #            掉出 Top-K 的边不会被清零 → 修掉"图冻结"（推荐，默认）
    #   topk  —— 只保存当前 Top-K 的 EMA（旧实现）→ 会冻结，仅用于复现/对照
    graph_state: str = "pool"
    graph_pool_size: int = 0                # 0 => max(8*neighbors, 128)
    # 新边的 EMA 初值：zero = 论文公式的 0 初值（默认）；
    # warm = 首次观测直接取当前 Δe（A/B 版对"无历史行"就是这么做的，会让图更灵敏）
    graph_new_edge: str = "zero"
    graph_mix: float = 1.0                  # 图注入强度；A/B 版无额外缩放 = 1.0
    active_norm_eps: float = 1e-6           # a_i = 1[||s̄_i|| > ε]
    graph_block_size: int = 1024            # 分块大小（显存/内存上限）
    # 摘要缓存：0 = 直接替换（论文 s̄_i = s_i，A/B 版行为）；>0 = 动量 EMA（旧 val01/02/03）
    summary_momentum: float = 0.0

    # ==================================================================================
    # D) 运行控制
    # ==================================================================================
    num_windows: int = 5
    rounds_per_window: int = 120
    eval_every: int = 10
    eval_batch_size: int = 2048             # 按"共享同一份个性化状态"的用户分组批处理
    max_eval_users: Optional[int] = None
    num_threads: int = 4
    device: str = "auto"
    out_dir: Optional[Path] = None

    # ---------------------------------------------------------------------------------
    def clone(self, **overrides: object) -> "FedDIEGConfig":
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.update(overrides)
        return FedDIEGConfig(**data)

    @property
    def total_rounds(self) -> int:
        return int(self.num_windows) * int(self.rounds_per_window)

    def resolved_clients_per_round(self, num_users: int) -> int:
        m = self.clients_per_round or max(int(self.participation_rate * num_users), 1)
        return int(max(1, min(m, num_users)))

    def resolved_pool_size(self) -> int:
        if self.graph_pool_size and self.graph_pool_size > 0:
            return int(max(self.graph_pool_size, self.neighbors))
        return int(max(8 * self.neighbors, 128))

    def server_lr(self, round_idx: int) -> float:
        """对齐 A/B：线性衰减到 final_lr（1 基轮次）。"""
        decay_steps = max(int(self.rounds * self.server_lr_decay_frac), 1)
        frac = max(0.0, 1.0 - (round_idx - 1) / decay_steps)
        return float(self.server_lr_final + (self.server_lr_init - self.server_lr_final) * frac)

    def validate(self) -> None:
        if not (0.0 < self.participation_rate <= 1.0):
            raise SystemExit("participation_rate 必须在 (0, 1] 区间内")
        if not (0.0 <= self.rho < 1.0):
            raise SystemExit("rho 必须在 [0, 1) 区间内")
        if self.local_epochs < 1:
            raise SystemExit("local_epochs 必须 >= 1")
        if self.min_sim < 0:
            raise SystemExit("min_sim 必须 >= 0")
        if self.graph_state not in {"pool", "topk"}:
            raise SystemExit("graph_state 只能是 pool 或 topk")
        if self.graph_new_edge not in {"zero", "warm"}:
            raise SystemExit("graph_new_edge 只能是 zero 或 warm")
        if self.graph_update_scope not in {"all", "sampled"}:
            raise SystemExit("graph_update_scope 只能是 all 或 sampled")
        if self.personalization not in {"encoder", "none"}:
            raise SystemExit("personalization 只能是 encoder 或 none")
        if self.struct_target not in {"delta_e", "edge_weight"}:
            raise SystemExit("struct_target 只能是 delta_e 或 edge_weight")
        if self.fedavg_weighting not in {"uniform", "sample"}:
            raise SystemExit("fedavg_weighting 只能是 uniform 或 sample")
