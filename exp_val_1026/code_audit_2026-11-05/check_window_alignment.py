"""检查 val02_detect_drift.py 产出 counts 的窗口切分是否与实验加载器一致。

如果两者不一致，那么 cat_old / cat_new / onset / win_dom 全都和实际训练窗口错位，
val02 + val03 的所有 drift 结论都会受影响。

    F:\\Anaconda\\python.exe -X utf8 exp_val_1026/code_audit_2026-11-05/check_window_alignment.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\exp_FedDIEG")
DRIFT = ROOT / "exp_val_1026/exp02_ema_interest_drift/results/drift/movielens-1m"
SPLITS = ROOT / "datasets/splits/movielens-1m"


def main() -> None:
    npz = np.load(DRIFT / "val02_counts.npz")
    counts = npz["counts"]      # (U, W, K)
    n_inter = npz["n_inter"]    # (U, W)
    print(f"counts {counts.shape}   n_inter {n_inter.shape}")

    print("\n=== npz 里各窗口的交互数 ===")
    print("  窗口   n_inter 总和   被类别覆盖（counts 总和）")
    for w in range(n_inter.shape[1]):
        print(f"   W{w + 1}     {int(n_inter[:, w].sum()):>9}      {int(counts[:, w, :].sum()):>9}")
    print(f"  合计      {int(n_inter.sum()):>9}      {int(counts.sum()):>9}")

    print("\n=== npz 里各窗口的「有交互用户数」===")
    for w in range(n_inter.shape[1]):
        print(f"   W{w + 1}: {int((n_inter[:, w] > 0).sum()):>5} 人")

    # ------------------------------------------------------------------------------
    # 实验加载器的窗口切分
    # ------------------------------------------------------------------------------
    full = pd.concat(
        [pd.read_csv(SPLITS / f"loo_{name}.csv") for name in ("train", "val", "test")],
        ignore_index=True,
    )
    full = full.sort_values(["timestamp", "userId"]).reset_index(drop=True)
    total = len(full)
    print(f"\n=== 加载器切分（时间序，共 {total} 交互）===")
    print("  窗口   交互数   覆盖用户数   用户 0-4 的出现次数")
    loader = {}
    for w in range(1, 6):
        s, e = int(total * (w - 1) / 5), int(total * w / 5)
        seg = full.iloc[s:e]
        g = seg.groupby("userId").size()
        loader[w] = g
        head = [int(g.get(u, 0)) for u in range(5)]
        print(f"   W{w}   {e - s:>7}   {seg['userId'].nunique():>8}      {head}")

    print("\n=== 判定 ===")
    ok = True
    for w in range(1, 6):
        npz_total = int(n_inter[:, w - 1].sum())
        load_total = int(loader[w].sum())
        # npz 的窗口总数应为 1/5 左右
        if abs(npz_total - load_total) > 0.05 * load_total:
            ok = False
            print(f"  W{w}: npz {npz_total} vs 加载器 {load_total}  —— 不一致")
        else:
            print(f"  W{w}: npz {npz_total} vs 加载器 {load_total}  —— 一致")
    print("\n结论：" + ("窗口切分一致" if ok else "窗口切分不一致（需要修 detect_drift）"))

    # ------------------------------------------------------------------------------
    # 用户-窗口活动矩阵的形状（是否某个窗口集中了所有人的交互）
    # ------------------------------------------------------------------------------
    print("\n=== 用户活动在窗口上的分布（前 5 个用户）===")
    print("  user |  W1   W2   W3   W4   W5  | 加载器 W1..W5")
    for u in range(5):
        npz_row = [int(n_inter[u, w]) for w in range(5)]
        load_row = [int(loader[w].get(u, 0)) for w in range(1, 6)]
        print(f"  {u:>4} | " + " ".join(f"{v:>4}" for v in npz_row) + " | " + " ".join(f"{v:>4}" for v in load_row))


if __name__ == "__main__":
    main()
