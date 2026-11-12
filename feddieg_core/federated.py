"""联邦训练协议：客户端本地训练 + 服务器参数聚合（FedAvg）。

对齐来源
--------
* `F:\\FedDIEG-main\\FedDIEG.py` L301-L464
* `F:\\FedDIEG-main\\exp_01\\run_exp01_movielens_aligned.py` L1401-L1502

一轮的流程（与 A/B 版逐句对应）
------------------------------
1. 服务器采样 m = max(int(C·N), 1) 个客户端；用 ``random.Random(seed).sample``，
   与 A/B 的 ``sample_schedule``（aligned L233-L236）一致。
2. 对每个客户端 u：
   a. 组装本地模型状态：共享层 ← 当前全局参数，个性化层 ← u 自己的副本；
   b. ``optim.Adam(local_params, lr)``，在 u 自己的本地数据上跑 ``local_epochs`` 个 epoch；
   c. 训练结束后，个性化层留在本地，共享层交给服务器累加；
   d. 用训练后的本地模型算出兴趣摘要 s_u 并"上传"（服务器写进 ``summary_bank``）。
3. 服务器聚合：共享层取平均；
   ``item_emb`` 走 ``global += server_lr·(avg - global)`` 的外推，其余共享层直接取 avg。
   ``server_lr`` 从 15 线性衰减到 1（A/B 版超参）。

与旧 `exp_val_1026` 实现的关键差别
---------------------------------
* 旧实现是「1 个共享模型 + 1 个 Adam」，每个 microbatch 直接改共享参数 ——
  那不是联邦学习，只是"按用户分片的小批量 SGD"。现在同一轮内所有客户端都从
  **同一个 θ_t** 出发，本地训练互不干扰，最后由服务器聚合。
* ``local_epochs`` 现在真的是"客户端本地 epoch 数"（旧实现只是把同一批数据再扫一遍）。
* 服务器缓存的兴趣摘要默认**直接替换**（论文 s̄_i = s_i；旧实现是 0.8 动量 EMA）。
* 负样本从"该客户端没交互过的物品"里抽（对齐 `dataset_01.Seq_TrainDataset`）。

内存说明
--------
A/B 版为**所有**客户端各存一份个性化参数（arts_crafts 上约 11 GB）。这里利用
"A/B 版所有客户端都从同一份全局初值复制而来"这一事实，只保存
``personal_init`` + 被采样过客户端的覆盖项，未采样客户端读 ``personal_init``。
这与 A/B 版**数值等价**，但内存占用从 O(N·|θ_p|) 降到 O(#sampled·|θ_p|)。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import FedDIEGConfig
from .data import ClientWindow
from .model import build_model, split_state_keys


@dataclass
class RoundStats:
    loss: float
    n_clients: int
    n_trained: int
    n_uploads: int
    server_lr: float


def build_participation_schedule(
    num_users: int,
    total_rounds: int,
    clients_per_round: int,
    seed: int,
) -> List[List[int]]:
    """对齐 aligned L233-L236：每轮 ``random.sample``（轮内不重复），跨轮独立。"""
    rng = random.Random(seed)
    m = int(max(1, min(clients_per_round, num_users)))
    pool = list(range(num_users))
    return [rng.sample(pool, m) for _ in range(total_rounds)]


class FedTrainer:
    """FedDIEG 联邦训练器（服务器 + 客户端模拟）。"""

    def __init__(self, cfg: FedDIEGConfig, num_users: int, num_items: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.num_users = int(num_users)
        self.num_items = int(num_items)

        bootstrap = build_model(num_items, cfg).to(device)
        self.personal_keys, self.shared_keys = split_state_keys(bootstrap)
        full_state = bootstrap.state_dict()
        self.global_state: Dict[str, torch.Tensor] = {
            k: full_state[k].detach().clone() for k in self.shared_keys
        }
        self.personal_init: Dict[str, torch.Tensor] = {
            k: full_state[k].detach().clone() for k in self.personal_keys
        }
        self.personal_overrides: Dict[int, Dict[str, torch.Tensor]] = {}
        self.worker = build_model(num_items, cfg).to(device)

        # ---------------- 服务器状态：兴趣摘要缓存 ----------------
        self.summary_bank = torch.zeros((self.num_users, cfg.latent_dim), device=device)
        self.summary_active = torch.zeros(self.num_users, dtype=torch.bool, device=device)
        self.last_upload_round = torch.full((self.num_users,), -1, dtype=torch.long, device=device)
        self.n_uploads = torch.zeros(self.num_users, dtype=torch.long, device=device)

        # ---------------- 客户端本地状态（用于 L_temp / L_struct） ----------------
        self.client_prev_summary: Dict[int, torch.Tensor] = {}
        # 键 = 轮次；值 = {"neighbors": (id...), "delta": (K,), "weights": (K,)}
        self.client_prev_state: Dict[int, Dict[str, object]] = {}

        # 观测钩子：本地训练结束时被调用 (user, trained_state)，供验证脚本/调试使用
        self.on_client_trained = None  # type: ignore[assignment]

    # ==================================================================================
    # 状态组装
    # ==================================================================================
    def personal_state(self, user: int) -> Dict[str, torch.Tensor]:
        return self.personal_overrides.get(user, self.personal_init)

    def state_for(self, user: int) -> Dict[str, torch.Tensor]:
        """组装该用户的完整模型状态（结果张量都在 ``self.device`` 上）。

        ``personal_on_cpu`` 打开时个性化覆盖项存在 CPU，这里按需搬上 device。
        """
        state = dict(self.global_state)
        stored = self.personal_overrides.get(user)
        if stored is None:
            state.update(self.personal_init)
            return state
        device = self.device
        state.update(
            {
                k: (v if v.device == device else v.to(device, non_blocking=True))
                for k, v in stored.items()
            }
        )
        return state

    def load_into(self, model: nn.Module, user: int) -> None:
        model.load_state_dict(self.state_for(user))

    def personal_state_key(self, user: int) -> str:
        """评测/摘要刷新时用于把"个性化状态完全相同"的用户合批。"""
        return "__init__" if user not in self.personal_overrides else f"u{int(user)}"

    def group_users_by_state(self, users: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        if users.numel() == 0:
            return []
        buckets: Dict[str, List[int]] = {}
        for u in users.detach().cpu().tolist():
            buckets.setdefault(self.personal_state_key(int(u)), []).append(int(u))
        return [
            (key, torch.tensor(group, dtype=torch.long, device=users.device))
            for key, group in buckets.items()
        ]

    # ==================================================================================
    # 摘要缓存（服务器侧）
    # ==================================================================================
    @torch.no_grad()
    def upload_summary(self, users: torch.Tensor, summaries: torch.Tensor, round_idx: int) -> int:
        """把客户端上传的摘要写进缓存。``summary_momentum=0`` 时等于论文的 s̄_i = s_i。"""
        if users.numel() == 0:
            return 0
        momentum = float(min(max(self.cfg.summary_momentum, 0.0), 0.99))
        # 防设备不一致：调用方可能传 CPU 张量（例如 window.active_users）
        users = users.to(self.summary_bank.device, non_blocking=True)
        new = summaries.detach().to(device=self.summary_bank.device, dtype=self.summary_bank.dtype)
        if momentum > 0:
            seen = self.summary_active[users].to(new.dtype).unsqueeze(-1)
            self.summary_bank[users] = seen * (
                momentum * self.summary_bank[users] + (1.0 - momentum) * new
            ) + (1.0 - seen) * new
        else:
            self.summary_bank[users] = new
        self.summary_active[users] = True
        self.last_upload_round[users] = int(round_idx)
        self.n_uploads[users] += 1
        return int(users.numel())

    def staleness(self, round_idx: int) -> torch.Tensor:
        """staleness[u] = 当前轮 − 上次上传轮次；从未上传返回 -1。"""
        last = self.last_upload_round
        return torch.where(
            last >= 0,
            torch.full_like(last, int(round_idx)) - last,
            torch.full_like(last, -1),
        )

    # ==================================================================================
    # 一轮联邦训练
    # ==================================================================================
    def train_round(
        self,
        sampled: Sequence[int],
        window: ClientWindow,
        graph,
        round_idx: int,
        rng: np.random.Generator,
    ) -> RoundStats:
        """执行一轮：客户端本地训练 → 上传摘要 → 服务器聚合。"""
        cfg = self.cfg
        device = self.device
        weight_by_samples = cfg.fedavg_weighting == "sample"

        sums: Dict[str, Optional[torch.Tensor]] = {k: None for k in self.shared_keys}
        total_weight = 0.0
        total_loss = 0.0
        n_trained = 0
        upload_users: List[int] = []
        upload_summaries: List[torch.Tensor] = []

        for raw_user in sampled:
            user = int(raw_user)
            if not window.has_data(user):
                continue  # 没有可用训练数据的客户端不参与（也不刷新摘要）

            self.worker.load_state_dict(self.state_for(user))
            self.worker.train()
            optimizer = optim.Adam(self.worker.parameters(), lr=cfg.lr)

            users_t = torch.tensor([user], dtype=torch.long, device=device)
            seq = window.seq[user].unsqueeze(0).to(device)
            pos = window.pos[user].unsqueeze(0).to(device)
            mask = window.mask[user].unsqueeze(0).to(device)

            # 邻居来自服务器缓存的图，本轮训练期间固定（A/B 版同样在本轮开始时取一次）
            nbr_ids, nbr_w_all, nbr_valid = graph.neighbors(users_t)
            nbr_ids_tuple = tuple(int(v) for v in nbr_ids[0].tolist() if int(v) >= 0)
            if bool(nbr_valid[0].item()):
                nbr_sum, nbr_w, _ = graph.gather(self.summary_bank, users_t)
                nbr_w_vec = nbr_w[0].detach()
            else:
                nbr_sum, nbr_w, nbr_w_vec = None, None, None

            prev_s = self.client_prev_summary.get(user)
            struct_target = self._struct_target(user, nbr_ids_tuple)

            last_summary: Optional[torch.Tensor] = None
            last_delta: Optional[torch.Tensor] = None
            user_loss = 0.0

            for _ in range(cfg.local_epochs):
                # 对齐 dataset_01.Seq_TrainDataset：每个 epoch 重新采负样本
                neg = torch.from_numpy(window.sample_negatives(user, rng)).to(device).unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)

                feats, s_i = self.worker(seq, nbr_sum, nbr_w, cfg.graph_mix)
                pos_logits = (feats * self.worker.item_emb(pos)).sum(dim=-1)
                neg_logits = (feats * self.worker.item_emb(neg)).sum(dim=-1)

                current_s_i = s_i.squeeze(0)  # batch = 1
                delta = None
                if nbr_sum is not None:
                    delta = F.cosine_similarity(
                        current_s_i.unsqueeze(0), nbr_sum[0], dim=-1, eps=1e-8
                    ).clamp_min(0.0)

                loss, _, _, _ = self.worker.compute_joint_loss(
                    pos_logits, neg_logits, mask, current_s_i, prev_s, delta, struct_target
                )
                loss.backward()
                optimizer.step()

                user_loss += float(loss.item())
                # 这里只是一个「本轮确实训练过」的哨兵值：训练前的 s_i **不会**被上传，
                # 下面 F7 那段会用更新后的参数重算并覆盖它（见 285-302 行）。
                last_summary = current_s_i.detach()
                if delta is not None:
                    last_delta = delta.detach()

            # ---------------- 个性化层留在本地，共享层交给服务器
            trained = self.worker.state_dict()
            if self.on_client_trained is not None:
                self.on_client_trained(user, {k: v.detach().clone() for k, v in trained.items()})
            store_device = "cpu" if cfg.personal_on_cpu else self.device
            self.personal_overrides[user] = {
                k: trained[k].detach().to(store_device).clone() for k in self.personal_keys
            }
            client_weight = float(mask.sum().item()) if weight_by_samples else 1.0
            for k in self.shared_keys:
                value = trained[k].detach().to(torch.float32)
                if sums[k] is None:
                    sums[k] = value.clone() * client_weight
                else:
                    sums[k] += value * client_weight
            total_weight += client_weight

            # ---- 【F7，2026-11-08 修正】论文口径：本地训练**完成后**再上传摘要
            # 旧实现（与 A/B 主程序一致）取的是最后一个 epoch 前向得到的 s_i，
            # 也就是参数更新**之前**的摘要；论文写的是"客户端完成本地训练后上传当前短期兴趣摘要"。
            # 这里用更新后的参数重算一次（no_grad + eval 关闭 dropout，保证可复现）。
            if last_summary is not None:
                was_training = self.worker.training
                self.worker.eval()
                with torch.no_grad():
                    _, s_after = self.worker(seq, nbr_sum, nbr_w, cfg.graph_mix)
                self.worker.train(was_training)
                last_summary = s_after.squeeze(0).detach()
                if nbr_sum is not None and last_delta is not None:
                    last_delta = F.cosine_similarity(
                        last_summary.unsqueeze(0), nbr_sum[0], dim=-1, eps=1e-8
                    ).clamp_min(0.0)
            if last_summary is not None:
                upload_users.append(user)
                upload_summaries.append(last_summary)
                self.client_prev_summary[user] = last_summary.clone()
            if last_delta is not None:
                self.client_prev_state[user] = {
                    "neighbors": nbr_ids_tuple,
                    "delta": last_delta.clone(),
                    "weights": nbr_w_vec.clone() if nbr_w_vec is not None else None,
                }
            else:
                self.client_prev_state.pop(user, None)

            total_loss += user_loss / max(cfg.local_epochs, 1)
            n_trained += 1

        # ---------------- 服务器聚合（FedAvg）
        server_lr = cfg.server_lr(round_idx)
        if n_trained > 0 and total_weight > 0:
            for k in self.shared_keys:
                summed = sums[k]
                if summed is None:
                    continue
                avg = summed / total_weight
                if "item_emb" in k:
                    self.global_state[k] = (
                        self.global_state[k] + server_lr * (avg - self.global_state[k].to(avg.dtype))
                    ).to(self.global_state[k].dtype)
                else:
                    self.global_state[k] = avg.to(self.global_state[k].dtype)

        # ---------------- 摘要上传（论文：参与客户端本轮训练后上传）
        n_uploads = 0
        if upload_users:
            users_t = torch.tensor(upload_users, dtype=torch.long, device=device)
            n_uploads = self.upload_summary(users_t, torch.stack(upload_summaries), round_idx)

        return RoundStats(
            loss=(total_loss / max(n_trained, 1)),
            n_clients=len(sampled),
            n_trained=n_trained,
            n_uploads=n_uploads,
            server_lr=server_lr,
        )

    # ------------------------------------------------------------------ L_struct 目标
    def _struct_target(self, user: int, nbr_ids_tuple: Tuple[int, ...]) -> Optional[torch.Tensor]:
        cfg = self.cfg
        if cfg.lambda_struct <= 0:
            return None
        prev = self.client_prev_state.get(user)
        if prev is None or tuple(prev.get("neighbors", ())) != nbr_ids_tuple:
            return None
        if cfg.struct_target == "edge_weight":
            # A/B 版字面行为：把上一轮的 softmax 边权当目标（量纲与 Δe 不一致，仅为复现）
            weights = prev.get("weights")
            return weights if isinstance(weights, torch.Tensor) else None
        delta = prev.get("delta")
        return delta if isinstance(delta, torch.Tensor) else None

    # ==================================================================================
    # 摘要计算 / 刷新（诊断用：不训练，只用当前模型重算摘要）
    # ==================================================================================
    @torch.no_grad()
    def compute_summaries(
        self, window: ClientWindow, users: torch.Tensor, graph=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """用当前（全局 + 各用户个性化）模型算出这些客户端的兴趣摘要。

        按个性化状态分组批处理（未训练过的用户共享同一份初值）。
        返回 (user_ids, summaries)，顺序一致。
        """
        if users.numel() == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, self.cfg.latent_dim)
        self.worker.eval()
        out_users: List[int] = []
        out_summaries: List[torch.Tensor] = []
        for _, group in self.group_users_by_state(users):
            group = group[window.mask[group].sum(dim=1) > 0]
            if group.numel() == 0:
                continue
            self.worker.load_state_dict(self.state_for(int(group[0].item())))
            for chunk in torch.split(group, max(1, int(self.cfg.eval_batch_size))):
                seq = window.seq[chunk].to(self.device)
                if graph is not None:
                    nbr_sum, nbr_w, nbr_valid = graph.gather(self.summary_bank, chunk)
                    if bool(nbr_valid.any().item()):
                        _, s_i = self.worker(seq, nbr_sum, nbr_w, self.cfg.graph_mix)
                    else:
                        _, s_i = self.worker(seq)
                else:
                    _, s_i = self.worker(seq)
                out_users.extend(int(v) for v in chunk.tolist())
                out_summaries.append(s_i.detach())
        if not out_users:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, self.cfg.latent_dim)
        return (
            torch.tensor(out_users, dtype=torch.long, device=self.device),
            torch.cat(out_summaries, dim=0),
        )

    @torch.no_grad()
    def refresh_summaries_from_global(
        self, window: ClientWindow, users: torch.Tensor, graph, round_idx: int
    ) -> int:
        """val01 的 ``upload_fresh_edges`` 对照上界（不可部署）：全部客户端重算摘要。"""
        user_ids, summaries = self.compute_summaries(window, users, graph)
        if user_ids.numel() == 0:
            return 0
        return self.upload_summary(user_ids, summaries, round_idx)
