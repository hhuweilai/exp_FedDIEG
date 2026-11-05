"""诊断 ③：`drift_cur_share` 为什么低于随机基线 `drift_cur_base`。

只用 cohort 数据（不训练、不需要 GPU），把 val02 里 `CohortEdgeProbe` 的基线
口径逐步复算一遍，判定"低于随机"是**指标口径问题**还是**真实信号**。

    F:\\Anaconda\\python.exe -X utf8 exp_val_1026/code_audit_2026-11-05/diagnose_cur_share.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DRIFT = Path(r"F:\exp_FedDIEG\exp_val_1026\exp02_ema_interest_drift\results\drift\movielens-1m")


def dominant(counts: np.ndarray) -> np.ndarray:
    """与 val02_run_ema.py 的 _dominant 完全一致：总数>0 才取 argmax，否则 -1。"""
    total = counts.sum(axis=1)
    return np.where(total > 0, counts.argmax(axis=1), -1)


def main() -> None:
    data = np.load(DRIFT / "val02_counts.npz")
    counts = data["counts"]                     # (U, W, K)
    n_inter = data["n_inter"] if "n_inter" in data else None
    num_users, num_windows, num_cat = counts.shape
    frame = pd.read_csv(DRIFT / "val02_drift_users.csv")
    cohort_users = frame["user"].to_numpy(dtype=np.int64)
    is_drift = (frame["cohort"].to_numpy() == "drift")

    print(f"counts {counts.shape}  类别数 {num_cat}  cohort {cohort_users.size} 人"
          f"（drift {is_drift.sum()} / stable {(~is_drift).sum()}）")
    if n_inter is not None:
        ni = n_inter[cohort_users]
        print(f"cohort 用户总交互数：中位数 {np.median(ni):.0f}，最小 {ni.min():.0f}")

    # ==================================================================================
    # 1) 复算 val02 里的 win_dom / win_freq
    # ==================================================================================
    win_dom, win_freq = {}, {}
    for w in range(1, num_windows + 1):
        dom = dominant(counts[:, w - 1, :])
        win_dom[w] = dom
        valid = dom >= 0
        win_freq[w] = np.bincount(dom[valid], minlength=num_cat) / max(int(valid.sum()), 1)

    print("\n=== 1) 基线的数学期望 ===")
    print("  窗口  有效用户   Σfreq²（真正随机基线）  freq[0]（-1 被 clamp 后误用的值）")
    for w in range(1, num_windows + 1):
        dom, freq = win_dom[w], win_freq[w]
        valid = (dom >= 0).sum()
        print(f"   W{w}   {valid:>6}          {float((freq ** 2).sum()):.4f}"
              f"                    {freq[0]:.4f}")

    # ==================================================================================
    # 2) cohort 用户里有多少人的主类别是"未知(-1)"
    # ==================================================================================
    print("\n=== 2) cohort 用户中主类别未知（dom=-1）的比例 ===")
    print("  窗口  cohort 用户 dom=-1 人数  占比     其中 drift 组占比")
    for w in range(1, num_windows + 1):
        dom = win_dom[w][cohort_users]
        unknown = dom < 0
        drift_unknown = unknown & is_drift
        print(f"   W{w}      {int(unknown.sum()):>4} / {cohort_users.size}"
              f"            {unknown.mean() * 100:5.1f}%     "
              f"{drift_unknown.sum()}/{is_drift.sum()} "
              f"({drift_unknown.sum() / max(is_drift.sum(), 1) * 100:.1f}%)")

    # ==================================================================================
    # 3) 关键对比：把 dom=-1 的用户排除后，base 与 share 的可比期望
    # ==================================================================================
    print("\n=== 3) base 的两种算法（证明 -1 被 clamp 到类别 0 会污染基线）===")
    print("  窗口   mean(freq[clamp(dom,-1→0)])  mean(freq[dom] | dom>=0)   两者差")
    for w in range(1, num_windows + 1):
        dom = win_dom[w][cohort_users]
        freq = win_freq[w]
        all_clamped = freq[np.clip(dom, 0, num_cat - 1)].mean()
        valid_only = freq[dom[dom >= 0]].mean()
        print(f"   W{w}            {all_clamped:.4f}                    {valid_only:.4f}"
              f"            {all_clamped - valid_only:+.4f}")

    # ==================================================================================
    # 4) 决定性对照：随机邻居的 share 应该等于 base
    #    —— 用纯数据模拟"随机抽 K 个邻居"，看 share 与 base 是否一致
    # ==================================================================================
    print("\n=== 4) 随机邻居对照（同一批 cohort 用户，随机抽 K=20 个邻居）===")
    rng = np.random.default_rng(0)
    k = 20
    print("  窗口   含 dom=-1 用户的 share   仅 dom>=0 用户的 share   base(dom>=0)   判定")
    for w in range(1, num_windows + 1):
        dom = win_dom[w]
        freq = win_freq[w]
        users = cohort_users
        user_dom = dom[users]
        # 随机抽邻居
        nbrs = rng.integers(0, num_users, size=(users.size, k))
        same = (dom[nbrs] == user_dom[:, None]) & (user_dom[:, None] >= 0) & (dom[nbrs] >= 0)
        share_all = same.mean(axis=1)                 # 含 dom=-1 用户（它们恒为 0）
        valid = user_dom >= 0
        share_valid = same[valid].mean(axis=1)
        base_valid = freq[user_dom[valid]].mean()
        verdict = "OK（随机邻居≈base）" if abs(share_valid.mean() - base_valid) < 0.02 else "不一致"
        print(f"   W{w}        {share_all.mean():.4f}                {share_valid.mean():.4f}"
              f"             {base_valid:.4f}      {verdict}")

    # ==================================================================================
    # 5) 主类别本身的稳定性（决定这个指标有没有意义）
    # ==================================================================================
    print("\n=== 5) cohort 用户的主类别跨窗口稳定性 ===")
    doms = np.stack([win_dom[w][cohort_users] for w in range(1, num_windows + 1)], axis=1)  # (G, W)
    valid_any = (doms >= 0).any(axis=1)
    for label, sel in (("全部 cohort", np.ones_like(valid_any)), ("drift 组", is_drift), ("stable 组", ~is_drift)):
        sub = doms[sel & valid_any]
        # 相邻窗口主类别相同的比例（只看两边都有效的）
        same_pairs = []
        for w in range(num_windows - 1):
            a, b = sub[:, w], sub[:, w + 1]
            m = (a >= 0) & (b >= 0)
            if m.sum():
                same_pairs.append(float((a[m] == b[m]).mean()))
        uniq = [len(set(row[row >= 0].tolist())) if (row >= 0).any() else 0 for row in sub]
        print(f"  {label:<12} 相邻窗口同类别率 "
              + " ".join(f"{v:.2f}" for v in same_pairs)
              + f"  | 每人出现过的类别数 中位数 {np.median(uniq):.0f}")

    # ==================================================================================
    # 6) 每个窗口内交互数：主类别估计的可靠性
    # ==================================================================================
    print("\n=== 6) cohort 用户在每个窗口内的交互数（主类别估计的可靠性）===")
    print("  窗口   中位数   25%分位   最小值   交互数<3 的用户占比")
    for w in range(1, num_windows + 1):
        per_user = counts[cohort_users, w - 1, :].sum(axis=1)
        print(f"   W{w}     {np.median(per_user):>5.0f}   {np.percentile(per_user, 25):>6.0f}"
              f"   {per_user.min():>6.0f}        {(per_user < 3).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
