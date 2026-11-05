"""滚动时间窗口 + 逐客户端样本。

数据协议本身**没有改动**：窗口划分、候选集采样、训练序列构造仍然复用
`exp_02_fast_gpu.run_exp02_fast_gpu.load_rolling_windows`（val01/val02 直接调它，
val03 用自己等价实现 `load_rolling_windows_from_frame`），因此修复后的数字与旧结果
在数据层面是可比的。

本模块补的是旧代码缺的两件事：
1. **逐客户端样本视图**：联邦训练需要一个客户端一次本地训练用的 (seq, pos, mask)。
   旧实现是"按 microbatch 采样用户"，没有"客户端"这个对象。
2. **客户端历史物品集合**：对齐 `dataset_01.Seq_TrainDataset`（A/B 主程序的负采样：
   负样本从"用户没交互过的物品"里抽）。旧实现用 `sample_negative_like` 全局均匀采样，
   可能把用户历史里真实存在的物品当负样本。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .config import FedDIEGConfig

ROOT = Path(__file__).resolve().parents[1]


# ======================================================================================
# 时间窗口（与 exp_02 / val03 完全同一套切分）
# ======================================================================================
def build_full_frame(dataset: str) -> pd.DataFrame:
    """读取 loo_train/val/test 并按时间排序（与 exp_02.load_rolling_windows 一致）。"""
    splits_dir = ROOT / "datasets" / "splits" / dataset
    frames = [pd.read_csv(splits_dir / f"loo_{name}.csv") for name in ("train", "val", "test")]
    full_df = pd.concat(frames, ignore_index=True)
    if "timestamp" in full_df.columns:
        full_df = full_df.sort_values(["timestamp", "userId"]).reset_index(drop=True)
    else:
        full_df = full_df.reset_index(drop=True)
        full_df["timestamp"] = full_df.index
    return full_df


def window_bounds(total_interactions: int, num_windows: int) -> List[Tuple[int, int]]:
    return [
        (
            int(total_interactions * (w - 1) / num_windows),
            int(total_interactions * w / num_windows),
        )
        for w in range(1, num_windows + 1)
    ]


def history_sets_from_frame(
    full_df: pd.DataFrame, num_windows: int, num_users: int
) -> List[List[np.ndarray]]:
    """每个窗口开始时，每个用户的历史物品集合（itemId + 1，与 embedding 对齐）。

    窗口 w 的历史 = ``full_df.iloc[: end_w]``（与 ``load_rolling_windows`` / 
    ``load_rolling_windows_from_frame`` 的 ``history_df`` 完全一致）。
    用它做负采样排除，是 A/B 版 ``Seq_TrainDataset`` 的语义（只抽未交互过的物品）。
    注意这里排除的是**截至窗口末的完整历史**，比 A/B 版（只用 loo_train）更严格一点，
    只会让负样本更"干净"，不会引入未来信息。
    """
    out: List[List[np.ndarray]] = []
    for _, end in window_bounds(len(full_df), num_windows):
        grouped = (
            full_df.iloc[:end]
            .groupby("userId", sort=False)["itemId"]
            .apply(lambda x: (x.to_numpy(dtype=np.int64) + 1))
            .to_dict()
        )
        per_user: List[np.ndarray] = []
        for u in range(num_users):
            arr = grouped.get(u)
            per_user.append(np.zeros(0, dtype=np.int64) if arr is None else np.asarray(arr))
        out.append(per_user)
    return out


# ======================================================================================
# 客户端视角的一个窗口
# ======================================================================================
@dataclass
class ClientWindow:
    """一个时间窗口的逐客户端视图。

    ``seq/pos/mask`` 直接来自 `FastData`（每行一个客户端的一条序列）。
    ``history`` 是该窗口下每个客户端的历史物品集合（+1 偏移），用于负采样。
    """

    window_idx: int
    num_users: int
    num_items: int
    seq: torch.Tensor            # (N, L) long
    pos: torch.Tensor            # (N, L) long
    mask: torch.Tensor           # (N, L) float
    history: List[np.ndarray]
    val_users: torch.Tensor
    val_seq: torch.Tensor
    val_items: torch.Tensor
    test_users: torch.Tensor
    test_seq: torch.Tensor
    test_items: torch.Tensor
    active_users: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.active_users = torch.nonzero(self.mask.sum(dim=1) > 0, as_tuple=False).flatten()

    # ------------------------------------------------------------------ 客户端批次
    def client_batch(self, user: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """一个客户端的本地训练样本，(1, L) 的三元组（对齐 dataset_01 的 batch=1 语义）。"""
        return (
            self.seq[user].unsqueeze(0).to(device),
            self.pos[user].unsqueeze(0).to(device),
            self.mask[user].unsqueeze(0).to(device),
        )

    def has_data(self, user: int) -> bool:
        return bool(self.mask[user].sum().item() > 0)

    # ------------------------------------------------------------------ 负采样
    def sample_negatives(self, user: int, rng: np.random.Generator, max_tries: int = 32) -> np.ndarray:
        """对齐 ``dataset_01.Seq_TrainDataset``：负样本从"用户没交互过的物品"里抽。"""
        pos = self.pos[user].numpy()
        mask = self.mask[user].numpy()
        sel = np.nonzero(mask > 0)[0]
        neg = np.zeros(pos.shape[0], dtype=np.int64)
        if sel.size == 0:
            return neg
        hist = self.history[user]
        targets = pos[sel]
        trials = 8
        cand = rng.integers(1, self.num_items + 1, size=(sel.size, trials))
        for attempt in range(max_tries // trials):
            bad = np.isin(cand, hist) | (cand == targets[:, None])
            ok = ~bad
            done = ok.any(axis=1)
            if done.all():
                break
            pending = ~done
            cand[pending] = rng.integers(
                1, self.num_items + 1, size=(int(pending.sum()), trials)
            )
        bad = np.isin(cand, hist) | (cand == targets[:, None])
        ok = ~bad
        first = ok.argmax(axis=1)
        valid = ok.any(axis=1)
        picked = cand[np.arange(sel.size), first]
        picked = np.where(valid, picked, cand[:, 0])
        neg[sel] = picked
        return neg

    # ------------------------------------------------------------------ 评测视图
    def split(self, name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if name == "val":
            return self.val_users, self.val_seq, self.val_items
        if name == "test":
            return self.test_users, self.test_seq, self.test_items
        raise ValueError(f"unknown split: {name}")


def from_fast_data(
    fast,
    history: Sequence[np.ndarray],
    window_idx: int,
) -> ClientWindow:
    """把 `FastData` + 历史物品集合包装成 `ClientWindow`。"""
    return ClientWindow(
        window_idx=int(window_idx),
        num_users=int(fast.num_users),
        num_items=int(fast.num_items),
        seq=fast.train_seq,
        pos=fast.train_pos,
        mask=fast.train_mask,
        history=list(history),
        val_users=fast.val_users,
        val_seq=fast.val_seq,
        val_items=fast.val_items,
        test_users=fast.test_users,
        test_seq=fast.test_seq,
        test_items=fast.test_items,
    )


def to_exp2_config(dataset: str, cfg: FedDIEGConfig):
    """构造 exp_02 的 Exp2Config，用于复用官方的窗口/候选集协议。"""
    from exp_02_fast_gpu.run_exp02_fast_gpu import Exp2Config

    return Exp2Config(
        dataset=dataset,
        seed=cfg.seed,
        latent_dim=cfg.latent_dim,
        max_len=cfg.max_len,
        top_k=cfg.top_k,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        dropout_rate=cfg.dropout_rate,
        num_windows=cfg.num_windows,
        rounds_per_window=cfg.rounds_per_window,
        candidate_count=cfg.candidate_count,
        window_mode=cfg.window_mode,
        eval_every=cfg.eval_every,
        eval_batch_size=cfg.eval_batch_size,
        max_eval_users=cfg.max_eval_users,
        num_threads=cfg.num_threads,
    )


def verify_consistency(fast_windows, history_sets) -> List[str]:
    """自检：训练序列里的每个物品都必须出现在该窗口的"历史集合"里。

    这条检查保证 ``history_sets`` 与本窗口的 ``train_seq`` 用的是同一个 ``end_idx``
    （若窗口切分不一致，会立刻报出来）。返回问题列表，空列表表示通过。
    """
    problems: List[str] = []
    for w, (fast, hist) in enumerate(zip(fast_windows, history_sets), start=1):
        seq = fast.train_seq.numpy()
        for u in range(seq.shape[0]):
            items = seq[u][seq[u] > 0]
            if items.size == 0:
                continue
            missing = np.setdiff1d(items, hist[u])
            if missing.size:
                problems.append(
                    f"window {w} user {u}: {missing.size} 个训练物品不在历史集合里"
                )
                break
    return problems


# ======================================================================================
# 便捷入口：一次加载全部窗口的客户端视图
# ======================================================================================
def load_client_windows(
    cfg: FedDIEGConfig,
    fast_windows=None,
    full_df: Optional[pd.DataFrame] = None,
    verify: bool = True,
) -> List[ClientWindow]:
    """给 val01/val02 用：复用 exp_02 的 `load_rolling_windows`，补上客户端历史。"""
    from exp_02_fast_gpu.run_exp02_fast_gpu import load_rolling_windows

    if fast_windows is None:
        fast_windows = load_rolling_windows(to_exp2_config(cfg.dataset, cfg))
    if full_df is None:
        full_df = build_full_frame(cfg.dataset)
    num_users = int(fast_windows[0].num_users)
    history_sets = history_sets_from_frame(full_df, cfg.num_windows, num_users)
    if verify:
        problems = verify_consistency(fast_windows, history_sets)
        if problems:
            raise RuntimeError("历史集合与训练序列不一致：" + "; ".join(problems[:3]))
    return [
        from_fast_data(fast, hist, idx)
        for idx, (fast, hist) in enumerate(zip(fast_windows, history_sets), start=1)
    ]
