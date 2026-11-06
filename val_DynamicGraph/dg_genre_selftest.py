#!/usr/bin/env python
"""Genre 诊断的构造用例自检（不训练、秒级）。

构造两个兴趣截然不同的用户群：
  · 群 A（user 0-49）只看 Action 类物品（item 0-9）
  · 群 B（user 50-99）只看 Comedy 类物品（item 10-19）
然后手工指定图：
  · 正确图：A 的邻居都在 A 里 → genre 相似度应显著高于随机基线（lift > 1.5）
  · 错误图：A 的邻居都在 B 里 → gen 相似度应低于随机基线（lift < 1）
`genre_metrics_for_users` 若能把这两种情况区分开，就说明 genre 诊断链路（画像构建 +
邻居加权 + 随机基线）是对的。
"""

from __future__ import annotations

import numpy as np
import torch

from dg_common import (
    build_user_genre_profiles,
    center_profiles,
    genre_metrics_for_users,
)


class FakeWindow:
    def __init__(self, num_users: int, history):
        self.num_users = num_users
        self.history = history


class FakeGraph:
    def __init__(self, ids: np.ndarray, weights: np.ndarray, mask: np.ndarray, dim: int = 8):
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self._dim = dim

    def gather(self, summary_bank, users):
        nbr = self.ids[users]
        valid_slot = nbr >= 0
        w = self.weights[users] * valid_slot.to(self.weights.dtype)
        uvalid = self.mask[users] & valid_slot.any(dim=1)
        return summary_bank[nbr.clamp_min(0)], w, uvalid


K = 5
N = 100


def build_case(neighbors_same_group: bool):
    ids = np.full((N, K), -1, dtype=np.int64)
    w = np.zeros((N, K), dtype=np.float32)
    for u in range(N):
        lo, hi = (0, 50) if u < 50 else (50, 100)
        pick_from = (lo, hi) if neighbors_same_group else ((50, 100) if u < 50 else (0, 50))
        cands = [v for v in range(*pick_from) if v != u][:K]
        ids[u, : len(cands)] = cands
        w[u, : len(cands)] = 1.0 / max(len(cands), 1)
    mask = np.ones(N, dtype=bool)
    return FakeGraph(ids, w, mask)


def main() -> int:
    # 物品 genre 矩阵：item 0-9 只有 Action(下标 0)，item 10-19 只有 Comedy(下标 1)
    num_genres = 18
    genre_mat = np.zeros((20, num_genres), dtype=np.float32)
    genre_mat[0:10, 0] = 1.0     # Action
    genre_mat[10:20, 4] = 1.0    # Comedy（ML-1M 的字母序第 5 个）
    # history 存 itemId+1
    history = [np.arange(1, 11, dtype=np.int64) if u < 50 else np.arange(11, 21, dtype=np.int64)
               for u in range(N)]
    window = FakeWindow(N, history)

    profile = build_user_genre_profiles(window, genre_mat)
    centered = center_profiles(profile)
    # 摘要向量：也按群分开（模拟服务器缓存的兴趣摘要）
    sb = np.zeros((N, 8), dtype=np.float32)
    sb[:50, 0] = 1.0
    sb[50:, 1] = 1.0
    summary_bank = torch.as_tensor(sb)
    random_base = torch.arange(0, N, dtype=torch.long)

    print("=" * 70)
    print("Genre 诊断构造用例")
    print("=" * 70)
    print(f"  profile 行和为 1 的用户占比: {np.isclose(profile.sum(1), 1.0).mean():.0%}")
    print(f"  A 群用户画像的 Action 分量: {profile[0, 0]:.3f}  Comedy: {profile[0, 4]:.3f}")
    print(f"  B 群用户画像的 Action 分量: {profile[60, 0]:.3f}  Comedy: {profile[60, 4]:.3f}")

    ok = True
    for label, same in (("邻居同群（正确图）", True), ("邻居异群（错误图）", False)):
        graph = build_case(same)
        users = torch.arange(0, N, dtype=torch.long)
        m = genre_metrics_for_users(profile, centered, None, summary_bank, graph, users, random_base)
        g = float(np.nanmean(m["genre_lift"]))
        gd = float(np.nanmean(m["genre_diff"]))
        cd = float(np.nanmean(m["genrec_diff"]))
        s = float(np.nanmean(m["sem_lift"]))
        print(f"\n  [{label}]")
        print(f"    genre_nbr={np.nanmean(m['genre_nbr']):.3f}  genre_base={np.nanmean(m['genre_base']):.3f}  "
              f"genre_lift={g:.3f}")
        print(f"    genre_diff={gd:+.3f}  genrec_diff（去公共成分）={cd:+.3f}  "
              f"sem_lift={s:.3f}  entropy={np.nanmean(m['genre_entropy']):.3f}")
        if not same:
            print("    （注：本用例里两群各占一半，中心化基线恰好为 0，genrec_lift 因此是 NaN，")
            print("      所以判据用差值 genrec_diff 而不是比值）")
        if same:
            if not g > 1.5:
                print("    [FAIL] 同群邻居的 genre_lift 应 > 1.5")
                ok = False
            if not gd > 0.3:
                print("    [FAIL] 同群邻居的 genre_diff 应显著为正")
                ok = False
            if not s > 1.5:
                print("    [FAIL] 同群邻居的 sem_lift 应 > 1.5")
                ok = False
        else:
            if not g < 1.0:
                print("    [FAIL] 异群邻居的 genre_lift 应 < 1")
                ok = False
            if not gd < -0.3:
                print("    [FAIL] 异群邻居的 genre_diff 应显著为负")
                ok = False
            if not s < 1.0:
                print("    [FAIL] 异群邻居的 sem_lift 应 < 1")
                ok = False

    # 漂移：第二窗口只加了另一个群的物品 -> drift 应接近 1
    history2 = [np.concatenate([h, np.arange(11, 21, dtype=np.int64)]) if u < 50
                else np.concatenate([h, np.arange(1, 11, dtype=np.int64)]) for u, h in enumerate(history)]
    profile2 = build_user_genre_profiles(FakeWindow(N, history2), genre_mat)
    graph = build_case(True)
    m2 = genre_metrics_for_users(profile2, center_profiles(profile2), profile, summary_bank, graph,
                                 torch.arange(0, N, dtype=torch.long), random_base)
    drift = float(np.nanmean(m2["genre_drift"]))
    # 第二窗口在原有 10 个动作片之外又加了 10 个喜剧片 => 画像变成 50/50，
    # 与原来 (1,0) 的余弦是 0.707，所以 drift = 1-0.707 ≈ 0.29，这是正确值。
    print(f"\n  兴趣被掺入一半异类后的 genre_drift = {drift:.3f}（预期 ≈ 0.293）")
    if not 0.25 < drift < 0.35:
        print("    [FAIL] drift 应在 0.29 附近")
        ok = False

    print("\n" + ("ALL CHECKS PASSED" if ok else "有检查未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
