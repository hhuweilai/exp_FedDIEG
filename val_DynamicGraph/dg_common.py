#!/usr/bin/env python
"""val_DynamicGraph 公共件：三条臂（NoGraph / StaticGraph / DynamicGraph）的严格定义。

问题来源
--------
`exp_val_1026` 的三个验证实验跑完之后出现了两个信号：
  1. 实验二里改变 EMA 系数 ρ，推荐性能几乎不动；
  2. 因此无法判断「动态图本身」到底有没有贡献 —— 有可能图协同分支的作用
     被别的机制（个性化序列编码器、图分支的数值尺度）稀释掉了。

所以在动「自适应 EMA」之前，必须先把最基础的问题钉死：**动态图有用吗？**
这就是本目录的唯一目的。

三条臂（同一份序列编码器、同一套训练协议，只改图分支）
----------------------------------------------------
| 臂 | 图分支前向 | 服务器上的边状态 |
|---|---|---|
| `nograph`  | **关**：永远拿不到邻居 ⇒ `u_i = seq_emb_i` | 不建图，也从不更新 |
| `static`   | 开 | 只在 `freeze_round` 建**一次**，之后完全冻结 |
| `dynamic`  | 开 | 每轮按论文 `e ← ρ·e + (1-ρ)·Δe` 更新 |
| `periodic` | 开 | 每 `period` 轮清零重建一次（慢刷新静态图，补充对照） |

为什么这样切才能回答「动态图有没有贡献」
------------------------------------
`static` 与 `dynamic` 在 `freeze_round`（默认 = 预热结束那一轮）之前是**逐位相同**的：
同一个种子、同一批参与客户端、同一份摘要、同一次构图、同一个 Top-K。之后两条臂唯一的
差别就是「要不要继续更新边」。于是：

    StaticGraph  − NoGraph       = 「图分支」本身有没有用（哪怕图是死的）
    DynamicGraph − StaticGraph   = 「动态性」有没有额外用
    DynamicGraph − NoGraph       = 动态图的总贡献

这三个差值（`dg_aggregate.py` 里的 ①②③）合起来才是对「动态图到底有没有明确贡献」的回答。
只有 ② 显著为正，才谈得上「动态更新是有效成分」；若 ①≈0，那连图分支都没用上，
② 的正负没有意义（先去看 `delta_rel` 诊断）。

为了让三条臂只差「图」这一件事，本实验做了两处**刻意偏离论文默认值**的选择（都会打印）：
  * `lambda_struct = 0`（默认）：L_struct 定义在图量 Δe 上，NoGraph 臂没有图就必然没有它。
    留着它会变成「图臂多一个训练信号」的混淆项。要回到论文口径传 `--lambda-struct 1e-5`。
  * `nograph` 臂不建图、不更新边，因此也不做图相关的日志；训练/评测路径完全复用
    `feddieg_core`，一行都没改（`NullGraph` 用鸭子类型顶替 `DynamicInterestGraph`）。

严格性由 `dg_selfcheck.py` 断言（四条同一性检查），核心是两条：
  * 三条臂喂给模型的**负样本流逐位相同**（所以是配对比较，不是三次独立实验）；
  * `dynamic --graph-mix 0` 与 `nograph` 的逐用户 NDCG **逐位相同**
    （所以 NoGraph 确实把图分支关干净了，而不是「关小了」）。

图分支到底占多大比重
------------------
`branch_forward()` 顺便把 `delta_rel = ‖W_g(relu(Σ w_ij W_gnn h_j))‖ / ‖seq_emb‖`
（最后位置）记下来。这个数就是「图协同分支的位移相对序列表示有多大」：它若只有 1e-3，
那么无论 EMA 怎么调、边怎么变，都不可能影响推荐 —— 直接解释了实验二的现象。
它同时回答「是不是被稀释了」：`nograph` 臂必须恒为 0；`dynamic` 臂若在 1e-2 量级，
说明图分支在数值上几乎没有话语权，需要单独提升（例如 `--graph-mix`）才谈得上对比。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feddieg_core.config import FedDIEGConfig  # noqa: E402
from feddieg_core.graph import DynamicInterestGraph  # noqa: E402

# ======================================================================================
# 臂名
# ======================================================================================
ARM_NOGRAPH = "nograph"
ARM_STATIC = "static"
ARM_DYNAMIC = "dynamic"
ARM_PERIODIC = "periodic"
ARM_ONBOARDING = "onboarding"
ARM_FRESH_ONLY = "fresh_only"

ARM_CHOICES = (
    ARM_NOGRAPH, ARM_STATIC, ARM_DYNAMIC, ARM_PERIODIC, ARM_ONBOARDING, ARM_FRESH_ONLY,
)
#: 主实验默认三条臂（回答「动态图有没有贡献」的最小完整设计）
DEFAULT_ARMS = (ARM_NOGRAPH, ARM_STATIC, ARM_DYNAMIC)
#: 补充臂：不是必须，但能挡住「静态臂是稻草人」「更新是否必要」的反驳
EXTRA_ARMS = (ARM_PERIODIC, ARM_ONBOARDING, ARM_FRESH_ONLY)

ARM_SHORT = {
    ARM_NOGRAPH: "NoGraph",
    ARM_STATIC: "StaticGraph",
    ARM_DYNAMIC: "DynamicGraph",
    ARM_PERIODIC: "PeriodicGraph",
    ARM_ONBOARDING: "OnboardingGraph",
    ARM_FRESH_ONLY: "FreshOnlyGraph",
}
ARM_LABELS = {
    ARM_NOGRAPH: "NoGraph：无图分支（u_i = seq_emb_i）",
    ARM_STATIC: "StaticGraph：预热结束建一次图，之后完全冻结",
    ARM_DYNAMIC: "DynamicGraph：论文 EMA，每轮更新全部活跃节点的边",
    ARM_PERIODIC: "PeriodicGraph：每 period 轮清零重建（慢刷新静态图，补充对照）",
    ARM_ONBOARDING: "OnboardingGraph：用户首次有摘要时建边，之后不再更新（覆盖对照）",
    ARM_FRESH_ONLY: "FreshOnlyGraph：只更新「自上次更新后有新上传」的用户（针对性更新对照）",
}
ARM_ORDER = {
    arm: i
    for i, arm in enumerate(
        (ARM_NOGRAPH, ARM_STATIC, ARM_PERIODIC, ARM_DYNAMIC, ARM_ONBOARDING, ARM_FRESH_ONLY)
    )
}


# ======================================================================================
# NoGraph 臂：空图（与 DynamicInterestGraph 接口完全一致的鸭子类型）
# ======================================================================================
class NullGraph:
    """永远没有邻居的「图」。

    对外接口与 `DynamicInterestGraph` 一致（``gather`` / ``neighbors`` / ``update`` /
    ``ids`` / ``weights`` / ``mask`` / ``describe``），因此 `feddieg_core` 的训练与评测
    路径一行都不用改：
      * `FedTrainer.train_round` 拿不到邻居（``nbr_valid=False``）⇒ ``nbr_sum=None``
        ⇒ 模型的长短期融合项恒为 0 ⇒ ``u_i = seq_emb_i``；
      * `evaluate_split` 的 ``nbr_valid.any()`` 为 False ⇒ 走「无邻居」打分分支。
    """

    def __init__(self, cfg: FedDIEGConfig, num_users: int, device: torch.device):
        self.cfg = cfg
        self.num_users = int(num_users)
        self.device = device
        self.k = 0                          # 聚合脚本会读它
        self.pool_size = 0
        self.n_updates = 0
        self.last_update_round = -1
        self.skipped_updates = 0
        self._ids = torch.full((self.num_users, 1), -1, dtype=torch.long, device=device)
        self._w = torch.zeros((self.num_users, 1), dtype=torch.float32, device=device)
        self._mask = torch.zeros(self.num_users, dtype=torch.bool, device=device)

    # ---- 与 DynamicInterestGraph 同名（探针脚本直接读属性，这里用 property 对齐）
    @property
    def ids(self) -> torch.Tensor:
        return self._ids

    @property
    def weights(self) -> torch.Tensor:
        return self._w

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    # ---- 查询
    def gather(
        self, summary_bank: torch.Tensor, users: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = int(users.numel())
        return (
            torch.zeros((0, self.cfg.latent_dim), device=self.device),
            torch.zeros((0,), device=self.device),
            torch.zeros((b,), dtype=torch.bool, device=self.device),
        )

    def neighbors(self, users: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        users = users.to(self.device)
        b = int(users.numel())
        return (
            self._ids[users],
            self._w[users],
            torch.zeros((b,), dtype=torch.bool, device=self.device),
        )

    def update(self, *args, **kwargs):
        return None

    def describe(self) -> dict:
        return {
            "arm": ARM_NOGRAPH,
            "neighbors": 0,
            "pool_size": 0,
            "state": "none",
            "rho": float(self.cfg.rho),
            "min_sim": float(self.cfg.min_sim),
            "update_scope": "none",
            "warmup_rounds": int(self.cfg.graph_warmup_rounds),
            "ema_scale": "none",
            "freeze_round": None,
            "period": None,
        }


# ======================================================================================
# Static / Dynamic / Periodic：完全复用论文公式，只改「什么时候允许更新」
# ======================================================================================
class PolicyGraph(DynamicInterestGraph):
    """在 `DynamicInterestGraph`（论文公式 + 审计修复）之上加一层「更新策略」。

    边怎么算、Top-K 怎么选、softmax 边权怎么归一 —— 一个字都没改，全部沿用基类；
    本类只决定 ``update()`` 在哪些轮次、对哪些用户真正执行：

      * ``dynamic``    ：预热之后每轮都更新（论文语义，`graph_update_scope` 决定更新谁）
      * ``static``     ：只更新到 ``freeze_round`` 为止（默认预热结束那一轮 ⇒ 只建一次）
                        ``build_only=True`` 时改为「第 freeze_round 轮清零重建一次，之后冻结」
                        用来回答「第 120 轮才建图是否有救」（与「晚冻结」是两个问题）。
      * ``periodic``   ：每 ``period`` 轮把状态清零后重建一次（快照刷新）
                        ``refresh_phase="boundary"`` 时改为「预热后建一次 + 每个窗口边界重建」
      * ``onboarding`` ：只对「当前还没有任何邻居」的用户建边，建过就不再更新
      * ``fresh_only`` ：只更新「自上次更新后有新上传」的用户（过滤在 dg_run 里做）

    ``last_user_update``（逐用户上次被更新的轮次）与 ``last_update_was_refresh``
    （本轮是否发生过清零重建）供 ``dg_run.py`` 落盘，用于 ECR/覆盖类指标的分母注记。
    """

    def __init__(
        self,
        cfg: FedDIEGConfig,
        num_users: int,
        device: torch.device,
        arm: str,
        freeze_round: Optional[int] = None,
        period: Optional[int] = None,
        build_only: bool = False,
        refresh_phase: str = "warmup",
    ):
        if arm not in (ARM_STATIC, ARM_DYNAMIC, ARM_PERIODIC, ARM_ONBOARDING, ARM_FRESH_ONLY):
            raise ValueError(f"PolicyGraph 不支持 arm={arm!r}")
        super().__init__(cfg, num_users, device)
        self.arm = arm
        # 默认：预热结束的那一轮就建图，此后 static 不再更新
        self.freeze_round = (
            int(cfg.graph_warmup_rounds) + 1 if freeze_round is None else int(freeze_round)
        )
        self.period = int(cfg.rounds_per_window) if period is None else int(max(1, period))
        self.build_only = bool(build_only)
        self.refresh_phase = str(refresh_phase or "warmup")
        if self.refresh_phase not in ("warmup", "boundary"):
            raise ValueError(f"refresh_phase 只能是 warmup / boundary，收到 {refresh_phase!r}")
        if self.build_only and arm != ARM_STATIC:
            raise ValueError("--build-only 只用于 static 臂")
        self.n_updates = 0
        self.n_user_updates = 0
        self.last_update_round = -1
        self.skipped_updates = 0
        self.last_update_was_refresh = False
        self.last_graph_turnover = float("nan")
        self.last_user_update = torch.full(
            (int(num_users),), -1, dtype=torch.long, device=device
        )
        if arm == ARM_STATIC and self.freeze_round < int(cfg.graph_warmup_rounds) + 1:
            # 冻结轮次早于预热结束 ⇒ 一次都不会更新 ⇒ 等价于 NoGraph（但分支还开着）
            print(
                f"[warn] freeze_round={self.freeze_round} 早于预热结束轮 "
                f"{int(cfg.graph_warmup_rounds) + 1}，static 臂将永远建不出图，"
                "退化成 dy=0 的 NoGraph。"
            )

    # ------------------------------------------------------------------ 更新策略
    def should_update(self, round_idx: int) -> bool:
        warmup = int(self.cfg.graph_warmup_rounds)
        r = int(round_idx)
        if r <= warmup:          # 与基类一致的预热
            return False
        if self.arm == ARM_DYNAMIC:
            return True
        if self.arm in (ARM_ONBOARDING, ARM_FRESH_ONLY):
            # 每轮都「扫描」；真正要更新谁由 update()/dg_run 过滤
            return True
        if self.arm == ARM_STATIC:
            if self.build_only:
                return r == self.freeze_round
            return r <= self.freeze_round
        if self.arm == ARM_PERIODIC:
            if self.refresh_phase == "boundary":
                # 预热后建一次（r=warmup+1），之后每个窗口边界（r=1,121,241,...）重建
                if r == warmup + 1:
                    return True
                return (r - 1) % max(self.period, 1) == 0
            return (r - warmup - 1) % max(self.period, 1) == 0
        return False

    def reset_state(self) -> None:
        """把边状态清零（periodic 的快照刷新 / static 的一次性重建用）。"""
        self.ids.fill_(-1)
        self.weights.zero_()
        self.mask.fill_(False)
        self.pool_ids.fill_(-1)
        self.pool_val.zero_()
        self.pool_mask.fill_(False)

    @torch.no_grad()
    def update(self, summary_bank, active_clients, update_users, round_idx):
        self.last_update_was_refresh = False
        self.last_graph_turnover = float("nan")
        if not self.should_update(round_idx):
            self.skipped_updates += 1
            return None
        refresh = (self.arm == ARM_PERIODIC) or (self.arm == ARM_STATIC and self.build_only)
        # 统一的「图变化」度量：更新**前**的邻居集合 vs 更新后的邻居集合。
        #   * 普通臂：更新前 == 上一轮状态 ⇒ 与 ECR 完全等价
        #   * reset 型臂（periodic / build_only）：更新前是 reset **之前**的状态
        #     ⇒ 得到"重建到底换掉多少邻居"，而不是被清空后的 1.0
        pre_ids = self.ids.detach().clone()
        if refresh:
            self.reset_state()
            self.last_update_was_refresh = True
        update_users = update_users.to(self.device, non_blocking=True)
        if self.arm == ARM_ONBOARDING:
            # 只给「还没有任何邻居行」的用户建边
            update_users = update_users[~self.mask[update_users]]
        stats = super().update(summary_bank, active_clients, update_users, round_idx)
        if stats is not None and len(stats) > 0:
            self.n_updates += 1
            self.n_user_updates += int(len(stats))
            self.last_update_round = int(round_idx)
            self.last_user_update[stats.updated.to(self.device)] = int(round_idx)
            idx = stats.updated.to(self.device)
            self.last_graph_turnover = float(
                _jaccard_turnover(pre_ids[idx], self.ids[idx]).mean().item()
            )
        return stats

    def describe(self) -> dict:
        base = super().describe()
        base.update(
            {
                "arm": self.arm,
                "freeze_round": int(self.freeze_round) if self.arm == ARM_STATIC else None,
                "build_only": bool(self.build_only),
                "period": int(self.period) if self.arm == ARM_PERIODIC else None,
                "refresh_phase": str(self.refresh_phase) if self.arm == ARM_PERIODIC else None,
                "n_updates": int(self.n_updates),
                "n_user_updates": int(self.n_user_updates),
            }
        )
        return base


def build_graph(
    cfg: FedDIEGConfig,
    arm: str,
    num_users: int,
    device: torch.device,
    freeze_round: Optional[int] = None,
    period: Optional[int] = None,
    build_only: bool = False,
    refresh_phase: str = "warmup",
):
    """按臂名构造图对象。各臂的构造顺序一致、都不消耗随机数。"""
    if arm == ARM_NOGRAPH:
        return NullGraph(cfg, num_users, device)
    return PolicyGraph(
        cfg, num_users, device, arm=arm, freeze_round=freeze_round, period=period,
        build_only=build_only, refresh_phase=refresh_phase,
    )


# ======================================================================================
# 输出目录命名（不同配置各自独立目录，避免互相覆盖 / 被「已完成就跳过」误判）
# ======================================================================================
def arm_dir_name(
    arm: str,
    cfg: FedDIEGConfig,
    freeze_round: Optional[int] = None,
    period: Optional[int] = None,
    run_tag: str = "",
    build_only: bool = False,
    refresh_phase: str = "warmup",
) -> str:
    """运行目录名 = 臂名 + 与默认值不同的配置后缀。

    这样「同一个臂的不同配置」不会互相覆盖，而且聚合脚本会自动把它们当成独立变体
    （例如 `dynamic_rho0.8` 会自动与 `dynamic` 配成一条「ρ 的影响」对照）。
    """
    name = arm
    if arm == ARM_STATIC and build_only:
        # 「第 N 轮一次性建图」与「更新到第 N 轮再冻结」是两个问题，目录名必须能区分
        fr = (int(cfg.graph_warmup_rounds) + 1) if freeze_round is None else int(freeze_round)
        name = f"static_build{fr}"
    if abs(float(cfg.graph_mix) - 1.0) > 1e-12:
        name += f"_mix{float(cfg.graph_mix):g}"
    if abs(float(cfg.rho) - 0.2) > 1e-12:
        name += f"_rho{float(cfg.rho):g}"
    if int(cfg.neighbors) != 20:
        name += f"_k{int(cfg.neighbors)}"
    if arm == ARM_STATIC and not build_only:
        default_freeze = int(cfg.graph_warmup_rounds) + 1
        fr = default_freeze if freeze_round is None else int(freeze_round)
        if fr != default_freeze:
            name += f"_at{fr}"
    if arm == ARM_PERIODIC:
        default_period = int(cfg.rounds_per_window)
        pd = default_period if period is None else int(period)
        if pd != default_period:
            name += f"_p{pd}"
        if str(refresh_phase) != "warmup":
            name += f"_{refresh_phase}"
    # 图状态 / 新边初始化规则（冻结归因的 2x2 网格靠这两个后缀区分，否则会互相覆盖）
    if str(cfg.graph_state) != "pool":
        name += f"_{cfg.graph_state}"
    if str(cfg.graph_new_edge) != "zero":
        name += f"_{cfg.graph_new_edge}"
    if run_tag:
        name += f"_{run_tag}"
    return name


# ======================================================================================
# 图状态签名：用来证明「静态臂真的冻结了」、「动态臂真的在变」
# ======================================================================================
def edge_signature(graph) -> str:
    """(ids, weights, mask) 的短哈希。冻结后每轮都必须完全一样。"""
    h = hashlib.md5()
    for name in ("ids", "weights", "mask"):
        t = getattr(graph, name)
        arr = np.ascontiguousarray(t.detach().cpu().numpy())
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def n_users_with_edges(graph) -> int:
    try:
        return int(graph.mask.sum().item())
    except Exception:  # noqa: BLE001
        return 0


def _jaccard_turnover(a_ids: torch.Tensor, b_ids: torch.Tensor) -> torch.Tensor:
    """两批邻居集合（n,K）的逐用户 Jaccard 距离；<0 的槽位视为空。"""
    eq = (
        (a_ids.unsqueeze(2) == b_ids.unsqueeze(1))
        & (a_ids.unsqueeze(2) >= 0)
        & (b_ids.unsqueeze(1) >= 0)
    )
    inter = eq.any(dim=2).sum(dim=1).to(torch.float32)
    n_a = (a_ids >= 0).sum(dim=1).to(torch.float32)
    n_b = (b_ids >= 0).sum(dim=1).to(torch.float32)
    union = (n_a + n_b - inter).clamp_min(1e-8)
    return 1.0 - inter / union


def graph_is_empty(graph) -> bool:
    return n_users_with_edges(graph) == 0


# ======================================================================================
# 诊断：图协同分支的实际数值贡献
# ======================================================================================
def _tmean(t: torch.Tensor) -> float:
    return float(t.mean().item()) if t.numel() else float("nan")


def _tnanmean(t: torch.Tensor) -> float:
    if not t.numel():
        return float("nan")
    mask = torch.isfinite(t)
    return float(t[mask].mean().item()) if bool(mask.any()) else float("nan")


PROBE_KEYS = ("delta_rel", "delta_cos", "neighbor_cos", "neighbor_cos_base", "has_edge")


@torch.no_grad()
def branch_forward(
    model,
    summary_bank: torch.Tensor,
    graph,
    users: torch.Tensor,
    seqs: torch.Tensor,
    cfg: FedDIEGConfig,
    random_base: Optional[torch.Tensor] = None,
    items: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Optional[torch.Tensor]]:
    """一次前向，既算分数也给图分支的数值诊断（三条臂通用）。

    前向与 `FedDIEGModel.score_user` **逐算子一致**：
        seq_emb, s_i = extract_short_term_interest(seqs)
        agg           = relu(Σ_j w_ij · W_gnn(h_j))
        delta         = graph_mix · W_g(agg)
        u             = seq_emb + delta            # 打分只取最后一个位置
    `dg_selfcheck.py` 会断言这里算出的分数与 `score_user` 相等。

    关键诊断量：
      ``delta_rel``         = ‖delta[:, -1]‖ / ‖seq_emb[:, -1]‖
                              —— 图分支把最终用户表示挪动了多少（相对值）。
                              NoGraph 必须恒为 0；图臂若在 1e-3 附近，
                              那么图怎么变都不会影响推荐，实验二「ρ 无感」就有物理解释。
      ``delta_cos``         = cos(seq_emb[:, -1], delta[:, -1])，位移方向是否与序列表示同向。
      ``neighbor_cos``      = 用户摘要与它 Top-K 邻居摘要的加权余弦（只用边权做权重）。
      ``neighbor_cos_base`` = 与随机活跃用户摘要的平均余弦（同尺度的随机基线）。
      ``neighbor_lift``     = 前者 / 后者：>1 说明图里的邻居确实比随机更像自己。
    """
    device = seqs.device
    model.eval()
    b = int(seqs.size(0))
    seq_emb, s_i = model.extract_short_term_interest(seqs)

    nbr_sum, nbr_w, nbr_valid = graph.gather(summary_bank, users)
    if bool(nbr_valid.any().item()):
        agg = model.extract_long_term_interest(nbr_sum, nbr_w, b)
    else:
        agg = torch.zeros(b, cfg.latent_dim, device=device)
    delta = model.W_g(agg.unsqueeze(1).expand(-1, seq_emb.size(1), -1)) * float(cfg.graph_mix)

    last_seq = seq_emb[:, -1, :]
    last_delta = delta[:, -1, :]
    seq_norm = last_seq.norm(dim=-1)
    delta_norm = last_delta.norm(dim=-1)
    rel = delta_norm / seq_norm.clamp_min(1e-8)
    cos = F.cosine_similarity(last_seq, last_delta, dim=-1, eps=1e-8)

    # ---------------- 邻居质量（只在有边的用户上统计）
    users_g = users.to(graph.ids.device)
    ids = graph.ids[users_g]
    slot = ids >= 0
    w = graph.weights[users_g] * slot.to(torch.float32)
    wsum = w.sum(dim=1)
    unit_self = F.normalize(s_i, dim=-1, eps=1e-8)
    unit_nbr = F.normalize(summary_bank[ids.clamp_min(0)], dim=-1, eps=1e-8)
    cos_nbr = (unit_nbr * unit_self.unsqueeze(1)).sum(dim=-1)          # (B, K)
    neighbor_cos = (cos_nbr.clamp_min(0.0) * w).sum(dim=1) / wsum.clamp_min(1e-8)
    has_edge = (wsum > 0) & nbr_valid.to(wsum.device)

    if random_base is not None and int(random_base.numel()) > 0:
        base = random_base.to(device)
        unit_base = F.normalize(summary_bank[base], dim=-1, eps=1e-8)  # (R, D)
        base_cos = (unit_self @ unit_base.transpose(0, 1)).clamp_min(0.0).mean(dim=1)
    else:
        base_cos = torch.full((b,), float("nan"), device=device)

    neighbor_mean = _tnanmean(neighbor_cos[has_edge])
    base_mean = _tnanmean(base_cos)
    lift = (
        neighbor_mean / base_mean
        if np.isfinite(neighbor_mean) and np.isfinite(base_mean) and base_mean > 1e-9
        else float("nan")
    )

    diag = {
        "probe_n": float(b),
        "probe_with_graph": float(has_edge.float().mean().item()),
        "delta_rel_mean": _tmean(rel),
        "delta_rel_median": float(rel.median().item()),
        "delta_rel_max": float(rel.max().item()),
        "delta_cos_mean": _tmean(cos),
        "seq_norm_mean": _tmean(seq_norm),
        "delta_norm_mean": _tmean(delta_norm),
        "u_norm_mean": _tmean((last_seq + last_delta).norm(dim=-1)),
        "neighbor_cos_mean": neighbor_mean,
        "neighbor_cos_base": base_mean,
        "neighbor_lift": lift,
    }
    per_user = {
        "delta_rel": rel.detach().cpu().numpy(),
        "delta_cos": cos.detach().cpu().numpy(),
        "neighbor_cos": neighbor_cos.detach().cpu().numpy(),
        "neighbor_cos_base": base_cos.detach().cpu().numpy(),
        "has_edge": has_edge.float().detach().cpu().numpy(),
    }

    scores: Optional[torch.Tensor] = None
    if items is not None:
        last_step = (last_seq + last_delta).unsqueeze(1)               # (B, 1, D)
        item_embs = model.item_emb(items)
        scores = torch.matmul(last_step, item_embs.transpose(-1, -2)).squeeze(1)
    return diag, per_user, scores


@torch.no_grad()
def probe_diagnostics(
    trainer,
    eval_model,
    graph,
    users: torch.Tensor,
    seqs: torch.Tensor,
    items: Optional[torch.Tensor],
    cfg: FedDIEGConfig,
    random_base: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Optional[torch.Tensor]]:
    """按「个性化状态」分组跑 `branch_forward`（与 `evaluate_split` 的分组方式一致）。

    必须分组：评测用的模型状态是逐客户端的（个性化层留本地），
    用同一个状态给所有用户算诊断会把「个性化差异」混进 delta_rel。
    """
    n = int(users.numel())
    diag: Dict[str, float] = {"probe_n": float(n)}
    per_user: Dict[str, np.ndarray] = {
        k: np.full(n, np.nan, dtype=np.float64) for k in PROBE_KEYS
    }
    scores_out: Optional[torch.Tensor] = None
    if n == 0:
        for key in (
            "probe_with_graph", "delta_rel_mean", "delta_rel_median", "delta_rel_max",
            "delta_cos_mean", "seq_norm_mean", "delta_norm_mean", "u_norm_mean",
            "neighbor_cos_mean", "neighbor_cos_base", "neighbor_lift",
        ):
            diag[key] = float("nan")
        return diag, per_user, None

    device = trainer.device
    pos_of = {int(u): i for i, u in enumerate(users.tolist())}
    if items is not None:
        scores_out = torch.zeros((n, items.size(1)), dtype=torch.float32)

    # 分组前向时顺带把「逐组标量」（seq/delta/u 的范数）按组大小加权平均
    weighted = {"seq_norm_mean": 0.0, "delta_norm_mean": 0.0, "u_norm_mean": 0.0}
    weight_sum = 0.0
    for _, group in trainer.group_users_by_state(users):
        rows = torch.tensor([pos_of[int(u)] for u in group.tolist()], dtype=torch.long)
        trainer.load_into(eval_model, int(group[0].item()))
        part, one, sc = branch_forward(
            eval_model,
            trainer.summary_bank,
            graph,
            users[rows].to(device),
            seqs[rows].to(device),
            cfg,
            random_base=random_base,
            items=None if items is None else items[rows].to(device),
        )
        for key in PROBE_KEYS:
            per_user[key][rows.numpy()] = one[key]
        w = float(rows.numel())
        weight_sum += w
        for key in weighted:
            value = float(part.get(key, float("nan")))
            if np.isfinite(value):
                weighted[key] += value * w
        if scores_out is not None and sc is not None:
            scores_out[rows] = sc.detach().cpu()

    rel_t = torch.as_tensor(per_user["delta_rel"], dtype=torch.float32)
    cos_t = torch.as_tensor(per_user["delta_cos"], dtype=torch.float32)
    has_edge = torch.as_tensor(per_user["has_edge"], dtype=torch.float32)
    nbr_cos_t = torch.as_tensor(per_user["neighbor_cos"], dtype=torch.float32)
    base_t = torch.as_tensor(per_user["neighbor_cos_base"], dtype=torch.float32)
    valid_edge = has_edge > 0

    neighbor_mean = _tnanmean(nbr_cos_t[valid_edge])
    base_mean = _tnanmean(base_t)
    diag.update(
        {
            "probe_with_graph": float(has_edge.mean().item()),
            "delta_rel_mean": _tmean(rel_t),
            "delta_rel_median": float(rel_t.median().item()) if rel_t.numel() else float("nan"),
            "delta_rel_max": float(rel_t.max().item()) if rel_t.numel() else float("nan"),
            "delta_cos_mean": _tmean(cos_t),
            "seq_norm_mean": (weighted["seq_norm_mean"] / weight_sum) if weight_sum > 0 else float("nan"),
            "delta_norm_mean": (weighted["delta_norm_mean"] / weight_sum) if weight_sum > 0 else float("nan"),
            "u_norm_mean": (weighted["u_norm_mean"] / weight_sum) if weight_sum > 0 else float("nan"),
            "neighbor_cos_mean": neighbor_mean,
            "neighbor_cos_base": base_mean,
            "neighbor_lift": (
                neighbor_mean / base_mean
                if np.isfinite(neighbor_mean) and np.isfinite(base_mean) and base_mean > 1e-9
                else float("nan")
            ),
        }
    )
    return diag, per_user, scores_out


# ======================================================================================
# 探针用户 / 随机基线（同一 seed 下三条臂抽到同一批人 ⇒ 配对诊断）
# ======================================================================================
def pick_probe_rows(
    window, size: int, seed: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """从 test 视图里固定种子抽一个用户子集（各臂相同）。"""
    users, seqs, items = window.split("test")
    n = int(users.numel())
    if n == 0:
        return None
    k = min(int(max(1, size)), n)
    if k == n:
        idx = torch.arange(n)
    else:
        gen = torch.Generator().manual_seed(int(seed) + 17)
        idx = torch.randperm(n, generator=gen)[:k]
    return users[idx], seqs[idx], items[idx]


def pick_random_base(summary_active: torch.Tensor, size: int, seed: int) -> Optional[torch.Tensor]:
    """随机活跃用户（邻居质量的随机基线）。"""
    active = torch.nonzero(summary_active).flatten().cpu()
    if active.numel() == 0:
        return None
    k = min(int(max(1, size)), int(active.numel()))
    gen = torch.Generator().manual_seed(int(seed) + 29)
    idx = torch.randperm(active.numel(), generator=gen)[:k]
    return active[idx]




# ======================================================================================
# 真实 Genre 画像与「邻居的 genre 相似度」（Genre 协同有效性）
# ======================================================================================
def load_genre_matrix(path) -> Tuple[np.ndarray, list]:
    """读取 item_genres_ml1m.npz -> (num_items, num_genres) float32 矩阵 + genre 名称。"""
    data = np.load(Path(path))
    mat = np.asarray(data["genres"], dtype=np.float64)
    names = [str(x) for x in data["genre_names"]] if "genre_names" in data.files else []
    return mat, names


def build_user_genre_profiles(window, genre_mat: np.ndarray) -> np.ndarray:
    """按「该窗口可见的历史」构建每个用户的 genre 画像（L2 归一化，无未来信息）。

    `ClientWindow.history[u]` 存的是 ``itemId + 1``（见 `feddieg_core/data.py`），
    这里减 1 换回 0 基的内部 itemId 再去查 genre 矩阵。
    画像 = 历史物品的 genre 多热向量之和（多标签已按 1/|genres| 加权），再 L2 归一化。
    """
    num_users = int(window.num_users)
    num_items, num_genres = genre_mat.shape
    prof = np.zeros((num_users, num_genres), dtype=np.float64)
    lengths = np.array([int(h.size) for h in window.history], dtype=np.int64)
    if lengths.sum() == 0:
        return prof
    items = np.concatenate([np.asarray(h, dtype=np.int64) - 1 for h in window.history])
    rows = np.repeat(np.arange(num_users, dtype=np.int64), lengths)
    valid = (items >= 0) & (items < num_items) & (lengths[rows] > 0)
    if valid.any():
        np.add.at(prof, rows[valid], genre_mat[items[valid]])
    norm = np.linalg.norm(prof, axis=1, keepdims=True)
    return prof / np.maximum(norm, 1e-12)


def center_profiles(prof: np.ndarray) -> np.ndarray:
    """减掉全体均值后再归一化：去掉「Drama/Comedy 这种人人都看」的公共成分，
    让相似度真正反映「口味差异」而不是「都看了热门类型」。"""
    if prof.size == 0:
        return prof
    mean = prof.mean(axis=0, keepdims=True)
    out = prof - mean
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def _weighted_neighbor_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    wsum = weights.sum(axis=1)
    return (values * weights).sum(axis=1) / np.maximum(wsum, 1e-12)


@torch.no_grad()
def genre_metrics_for_users(
    profile: np.ndarray,
    centered: np.ndarray,
    profile_prev: Optional[np.ndarray],
    summary_bank: torch.Tensor,
    graph,
    users: torch.Tensor,
    random_base: Optional[torch.Tensor],
) -> Dict[str, np.ndarray]:
    """对一批用户算「邻居 vs 随机用户」的 genre / 语义相似度。

    返回的每个数组长度都等于 ``users.numel()``：
      sem_nbr / sem_base / sem_lift          语义（服务器缓存的兴趣摘要）相似度
      genre_nbr / genre_base / genre_lift    原始 genre 余弦相似度
      genrec_nbr / genrec_base / genrec_lift 去掉公共成分后的 genre 相似度
      genre_entropy                          用户 genre 画像的归一化熵（兴趣分散程度）
      genre_drift                            与上一窗口画像的 1−cos（兴趣变化幅度）
    """
    n = int(users.numel())
    out = {k: np.full(n, np.nan) for k in (
        "sem_nbr", "sem_base", "sem_lift", "sem_diff",
        "genre_nbr", "genre_base", "genre_lift", "genre_diff",
        "genrec_nbr", "genrec_base", "genrec_lift", "genrec_diff",
        "genre_entropy", "genre_drift",
    )}
    if n == 0:
        return out

    dev = graph.ids.device
    u = users.to(dev)
    ids = graph.ids[u]
    slot = ids >= 0
    w = graph.weights[u] * slot.to(torch.float32)
    valid_user = graph.mask[u] & slot.any(dim=1)
    nbr_idx = ids.clamp_min(0).detach().cpu().numpy()
    w_np = w.detach().cpu().numpy()
    u_np = users.detach().cpu().numpy()

    # ---- 语义相似度（兴趣摘要）
    sb = summary_bank.detach().cpu().numpy()
    unit_sb = sb / np.maximum(np.linalg.norm(sb, axis=1, keepdims=True), 1e-12)
    sem_nbr = _weighted_neighbor_mean(
        (unit_sb[nbr_idx] * unit_sb[u_np][:, None, :]).sum(-1), w_np
    )
    if random_base is not None and int(random_base.numel()) > 0:
        base_idx = random_base.detach().cpu().numpy()
        sem_base = (unit_sb[u_np] @ unit_sb[base_idx].T).mean(axis=1)
    else:
        sem_base = np.full(n, np.nan)

    # ---- genre 相似度（原始 / 去公共成分）
    def _genre_pair(mat: np.ndarray):
        nbr = _weighted_neighbor_mean(
            (mat[nbr_idx] * mat[u_np][:, None, :]).sum(-1), w_np
        )
        if random_base is not None and int(random_base.numel()) > 0:
            base = (mat[u_np] @ mat[base_idx].T).mean(axis=1)
        else:
            base = np.full(n, np.nan)
        return nbr, base

    g_nbr, g_base = _genre_pair(profile)
    c_nbr, c_base = _genre_pair(centered)

    # ---- 兴趣分散程度 / 漂移
    raw = np.clip(profile[u_np], 0.0, None)
    total = raw.sum(axis=1, keepdims=True)
    p_norm = raw / np.maximum(total, 1e-12)
    ent = -(np.where(p_norm > 0, p_norm * np.log(np.maximum(p_norm, 1e-12)), 0.0)).sum(axis=1)
    ent = ent / np.log(max(profile.shape[1], 2))
    if profile_prev is not None:
        # 上一窗口画像为空（用户当时还没交互）时不能算成 drift=1，
        # 那会把「新用户」误当成「兴趣完全变了」；这种情况记 NaN。
        prev_norm = np.linalg.norm(profile_prev[u_np], axis=1)
        cos_prev = (profile[u_np] * profile_prev[u_np]).sum(axis=1)
        drift = np.where(prev_norm > 1e-9, 1.0 - cos_prev, np.nan)
    else:
        drift = np.full(n, np.nan)

    ok = valid_user.detach().cpu().numpy()
    out.update({
        "sem_nbr": sem_nbr, "sem_base": sem_base,
        "sem_lift": sem_nbr / np.where(sem_base > 1e-9, sem_base, np.nan),
        "genre_nbr": g_nbr, "genre_base": g_base,
        "genre_lift": g_nbr / np.where(g_base > 1e-9, g_base, np.nan),
        "genrec_nbr": c_nbr, "genrec_base": c_base,
        "genrec_lift": c_nbr / np.where(np.abs(c_base) > 1e-9, c_base, np.nan),
        "sem_diff": sem_nbr - sem_base,
        "genre_diff": g_nbr - g_base,
        "genrec_diff": c_nbr - c_base,
        "genre_entropy": ent, "genre_drift": drift,
    })
    for key in ("sem_nbr", "sem_base", "sem_lift", "sem_diff",
                "genre_nbr", "genre_base", "genre_lift", "genre_diff",
                "genrec_nbr", "genrec_base", "genrec_lift", "genrec_diff"):
        out[key] = np.where(ok, out[key], np.nan)
    return out


# ======================================================================================
# 冻结诊断（相对「预热后第一次评测」的邻居快照）
# ======================================================================================
@torch.no_grad()
def turnover_vs_ref(graph, ref_ids: torch.Tensor, ref_valid: torch.Tensor) -> Dict[str, float]:
    """当前 Top-K 邻居集合相对快照的 Jaccard 距离，以及「完全没变」的用户占比。

    这是判断「图冻结」最直接的量：turnover ≈ 0 且 frozen_share ≈ 1 就是冻住了。
    """
    cur = graph.ids
    cur_valid = graph.mask & (cur >= 0).any(dim=1)
    both = ref_valid & cur_valid
    n_both = int(both.sum().item())
    if n_both == 0:
        return {
            "freeze_ref_n": 0.0, "turnover_from_ref": float("nan"),
            "frozen_share": float("nan"), "coverage_gain": 0.0,
        }
    r = ref_ids[both]
    c = cur[both]
    eq = (r.unsqueeze(2) == c.unsqueeze(1)) & (r.unsqueeze(2) >= 0) & (c.unsqueeze(1) >= 0)
    inter = eq.any(dim=2).sum(dim=1).to(torch.float32)
    n_r = (r >= 0).sum(dim=1).to(torch.float32)
    n_c = (c >= 0).sum(dim=1).to(torch.float32)
    union = (n_r + n_c - inter).clamp_min(1e-8)
    turnover = 1.0 - inter / union
    return {
        "freeze_ref_n": float(n_both),
        "turnover_from_ref": float(turnover.mean().item()),
        "frozen_share": float((turnover < 0.01).to(torch.float32).mean().item()),
        "coverage_gain": float(int((cur_valid & ~ref_valid).sum().item())),
    }


# ======================================================================================
# CLI
# ======================================================================================
def add_dg_args(parser: argparse.ArgumentParser) -> None:
    """val_DynamicGraph 专用参数。**必须在 `add_feddieg_args` 之后调用**（会改默认值）。"""
    g = parser.add_argument_group("val_DynamicGraph：三条臂（只改图分支）")
    g.add_argument("--arm", choices=list(ARM_CHOICES), default=ARM_DYNAMIC,
                   help="nograph / static / dynamic / periodic / onboarding / fresh_only")
    g.add_argument("--freeze-round", type=int, default=None,
                   help="static 臂冻结在第几轮（1 基）；默认 = graph_warmup_rounds+1（只建一次图）")
    g.add_argument("--build-only", action="store_true",
                   help="static 臂：第 freeze_round 轮清零重建一次后冻结（配 --freeze-round N）")
    g.add_argument("--refresh-rounds", type=int, default=None,
                   help="periodic 臂每多少轮清零重建；默认 = rounds_per_window")
    g.add_argument("--refresh-phase", choices=["warmup", "boundary"], default="warmup",
                   help="periodic 的相位：warmup=预热后每 period 轮；boundary=预热后建一次 + 窗口边界重建")
    g.add_argument("--probe-size", type=int, default=256,
                   help="每个评测点抽多少用户做图分支/邻居质量诊断（默认 256）")
    g.add_argument("--user-rows", dest="user_rows", action="store_true", default=True,
                   help="每个窗口最后一次评测写逐用户 NDCG（配对检验要用）")
    g.add_argument("--no-user-rows", dest="user_rows", action="store_false")
    # 严格隔离：L_struct 定义在图量 Δe 上，NoGraph 臂没有图 → 必须关掉才是「只差图」的对照。
    # 要回到论文口径：--lambda-struct 1e-5
    parser.set_defaults(lambda_struct=0.0)
