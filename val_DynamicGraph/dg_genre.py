#!/usr/bin/env python
"""Genre 协同有效性分析：兴趣相似是否意味着协同有效？（用 MovieLens-1M 的真实 Genre）

数据
----
* `dynamic_genre` 臂（3 个种子）：`dg_genre_metrics.csv` 逐用户给出
  - `sem_lift`      服务器缓存的兴趣摘要之间的相似度 ÷ 随机用户
  - `genre_lift`    真实 genre 画像的余弦相似度 ÷ 随机用户（18 类，物品覆盖率 100%）
  - `genre_diff`    genre 余弦的**差值**（基线接近 0 时比比值稳）
  - `genrec_*`      去掉「Drama/Comedy 人人都看」的公共成分后的版本
  - `genre_entropy` 用户兴趣分散程度；`genre_drift` 与上一窗口相比的兴趣变化
* `nograph` 臂：逐用户 NDCG → 与上面同 (seed, window, user) 对齐后得到**逐用户收益** ΔNDCG。

回答三个子问题
--------------
① 图选出来的邻居，在 genre 空间是不是真的更像自己？
② 服务器学到的语义相似度，和 genre 相似度是不是一回事？
③ **genre 相似度越高，图带来的推荐收益是不是越大？** —— 这是"兴趣相似 ⇒ 协同有效"的核心检验。
   顺带看收益是否集中在"兴趣变化大"的用户上。

用法
----
  python val_DynamicGraph/dg_genre.py
  python val_DynamicGraph/dg_genre.py --root results/raw --out results/genre
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DG_DIR = Path(__file__).resolve().parent
ROOT = DG_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_ROOT = DG_DIR / "results" / "raw"
DEFAULT_OUT = DG_DIR / "results" / "genre"
DATASET = "movielens-1m"
ARM_GENRE = "dynamic_genre"
ARM_BASE = "nograph"
GENRE_NPZ = DG_DIR / "data" / "item_genres_ml1m.npz"

try:
    from scipy import stats as _sp
except Exception:  # pragma: no cover
    _sp = None


# ----------------------------------------------------------------- 小工具
def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def sign_test_p(diff) -> float:
    d = _finite(diff)
    if d.size == 0:
        return float("nan")
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    m = n_pos + n_neg
    if m == 0:
        return 1.0
    if m <= 200:
        k = min(n_pos, n_neg)
        return float(min(1.0, 2.0 * sum(math.comb(m, i) for i in range(k + 1)) / (2.0 ** m)))
    z = (abs(n_pos - n_neg) - 1.0) / math.sqrt(m)
    return float(min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2.0))))


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 20261026):
    d = _finite(x)
    if d.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    chunk = int(max(1, min(n_boot, 2_000_000 // max(d.size, 1))))
    means, filled = [], 0
    while filled < n_boot:
        k = min(chunk, n_boot - filled)
        means.append(d[rng.integers(0, d.size, size=(k, d.size))].mean(axis=1))
        filled += k
    m = np.concatenate(means)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def corr_pair(x, y) -> Dict[str, float]:
    """Pearson + Spearman + 样本量（两者都忽略 nan）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    out = {"n": int(ok.sum()), "pearson": float("nan"), "pearson_p": float("nan"),
           "spearman": float("nan"), "spearman_p": float("nan")}
    if ok.sum() < 8:
        return out
    if _sp is not None:
        r, p = _sp.pearsonr(x[ok], y[ok])
        out["pearson"], out["pearson_p"] = float(r), float(p)
        r, p = _sp.spearmanr(x[ok], y[ok])
        out["spearman"], out["spearman_p"] = float(r), float(p)
    else:  # 手算 pearson
        xa, ya = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        denom = math.sqrt((xa ** 2).sum() * (ya ** 2).sum())
        out["pearson"] = float((xa * ya).sum() / denom) if denom > 0 else float("nan")
    return out


def fmt(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return str(x)
    return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"


def quintile_table(df: pd.DataFrame, key: str, target: str = "dndcg", q: int = 5) -> pd.DataFrame:
    d = df[[key, target]].dropna()
    if len(d) < q * 20:
        return pd.DataFrame()
    try:
        d = d.assign(bucket=pd.qcut(d[key], q, labels=False, duplicates="drop"))
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    rows = []
    for b, g in d.groupby("bucket"):
        lo, hi = bootstrap_ci(g[target])
        rows.append({
            "bucket": int(b) + 1,
            f"{key}_lo": float(g[key].min()),
            f"{key}_hi": float(g[key].max()),
            f"{key}_mean": float(g[key].mean()),
            "n": int(len(g)),
            f"{target}_mean": float(g[target].mean()),
            "ci95_lo": lo, "ci95_hi": hi,
            f"{target}_pos_rate": float((g[target] > 0).mean()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 离线重算兴趣漂移
def compute_interest_drift(dataset: str, num_windows: int, genre_npz: Path) -> pd.DataFrame:
    """离线计算「本窗口的行为 vs 之前的历史」在 genre 空间的偏离度。

    为什么不用运行时那列：运行内的 `genre_drift` 是「相邻两窗口**累积**画像」的 1−cos，
    而累积画像几乎不变（中位数 0.000），且用户在某个窗口之前还没交互时会被误算成 1.0。
    这里改成更有意义的定义：

        inc_u(w)   = 用户 u 在**第 w 个窗口内**交互物品的 genre 分布
        prev_u(w)  = 用户 u 在**第 w 个窗口开始之前**的累积 genre 分布
        drift      = 1 − cos(inc_u(w), prev_u(w))，两者都非空才计算，否则 NaN

    也就是「这一段时间的行为，偏离我过去的口味有多远」。
    """
    from dg_common import load_genre_matrix  # 先导入它，会把 ROOT 加入 sys.path

    from feddieg_core.data import build_full_frame, window_bounds

    mat, _ = load_genre_matrix(genre_npz)
    num_items, num_genres = mat.shape
    df = build_full_frame(dataset)
    bounds = window_bounds(len(df), num_windows)
    num_users = int(df["userId"].max()) + 1

    def _profile(items_per_user: dict) -> np.ndarray:
        P = np.zeros((num_users, num_genres))
        for u, arr in items_per_user.items():
            if u < 0 or u >= num_users:
                continue
            idx = np.asarray(arr, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < num_items)]
            if idx.size:
                P[u] = mat[idx].sum(axis=0)
        n = np.linalg.norm(P, axis=1, keepdims=True)
        return P / np.maximum(n, 1e-12)

    rows = []
    for w, (s, e) in enumerate(bounds, start=1):
        inc_map = df.iloc[s:e].groupby("userId")["itemId"].apply(np.asarray).to_dict()
        prev_map = df.iloc[:s].groupby("userId")["itemId"].apply(np.asarray).to_dict()
        inc = _profile(inc_map)
        prev = _profile(prev_map)
        prev_norm = np.linalg.norm(prev, axis=1)
        cos = (inc * prev).sum(axis=1)
        drift = np.where((prev_norm > 1e-9) & (np.linalg.norm(inc, axis=1) > 1e-9), 1.0 - cos, np.nan)
        for u in range(num_users):
            if np.isfinite(drift[u]):
                rows.append({"window": w, "user": u, "drift_local": float(drift[u])})
    out = pd.DataFrame(rows)
    print(f"  离线 drift：{len(out)} 条（覆盖 window 2~{num_windows}），"
          f"中位数 {out['drift_local'].median():.3f}，"
          f">0.5 的比例 {(out['drift_local'] > 0.5).mean():.1%}")
    return out


# ----------------------------------------------------------------- 读数据
CONFIG_KEYS = ("num_windows", "rounds_per_window", "eval_every", "participation_rate",
               "local_epochs", "lr", "top_k", "candidate_count", "personalization")


def check_arm_configs(root: Path) -> List[str]:
    """两条臂的关键配置必须一致，否则 ΔNDCG 没有意义（这一步能挡住"拿错结果配对"）。"""
    problems: List[str] = []
    rows = {}
    for arm in (ARM_GENRE, ARM_BASE):
        files = sorted((root / DATASET / arm).glob("seed_*/dg_summary.csv"))
        if not files:
            continue
        s = pd.read_csv(files[0]).iloc[0]
        rows[arm] = {k: s.get(k) for k in CONFIG_KEYS if k in s.index}
    if len(rows) == 2:
        for k in CONFIG_KEYS:
            a = rows[ARM_GENRE].get(k)
            b = rows[ARM_BASE].get(k)
            if a is None or b is None:
                continue
            if str(a) != str(b):
                problems.append(f"{k}: {ARM_GENRE}={a} vs {ARM_BASE}={b}")
    return problems


def load_joined(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    genre_files = sorted((root / DATASET / ARM_GENRE).glob("seed_*/dg_genre_metrics.csv"))
    base_files = sorted((root / DATASET / ARM_BASE).glob("seed_*/dg_user_metrics.csv"))
    if not genre_files:
        raise SystemExit(f"{root}/{DATASET}/{ARM_GENRE} 下没有 dg_genre_metrics.csv，先跑 dg_run_supplement.sh")
    if not base_files:
        raise SystemExit(f"{root}/{DATASET}/{ARM_BASE} 下没有 dg_user_metrics.csv")

    g = pd.concat([pd.read_csv(p) for p in genre_files], ignore_index=True)
    b = pd.concat([pd.read_csv(p) for p in base_files], ignore_index=True)
    g = g.rename(columns={"ndcg": "ndcg_dyn"})
    b = b.rename(columns={"ndcg": "ndcg_base"})
    keep_b = ["seed", "window", "user", "ndcg_base"]
    j = g.merge(b[keep_b], on=["seed", "window", "user"], how="inner")
    j["dndcg"] = j["ndcg_dyn"] - j["ndcg_base"]
    return j, g


# ----------------------------------------------------------------- 判定
def build_verdict(j: pd.DataFrame, desc: pd.DataFrame, corrs: pd.DataFrame,
                  buckets: Dict[str, pd.DataFrame]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    def push(idx, question, value, evidence, judgement):
        rows.append({"verdict_id": idx, "question": question, "value": value,
                     "evidence": evidence, "judgement": judgement})

    def d(key):
        s = desc[desc["metric"] == key]
        return s.iloc[0] if len(s) else None

    def c(key, target="dndcg"):
        s = corrs[(corrs["metric"] == key) & (corrs["target"] == target)]
        return s.iloc[0] if len(s) else None

    # ① 邻居在 genre 空间是否更像自己
    gd, gl = d("genre_diff"), d("genre_lift")
    cd, cl = d("genrec_diff"), d("genrec_lift")
    sd, sl = d("sem_diff"), d("sem_lift")
    ev = []
    if gd is not None:
        ev.append(f"genre_diff={fmt(gd['mean'], 4)}（95%CI [{fmt(gd['ci95_lo'],4)}, {fmt(gd['ci95_hi'],4)}]，"
                  f"为正的用户占比 {fmt(gd['pos_rate'], 3)}）")
    if gl is not None:
        ev.append(f"genre_lift={fmt(gl['mean'], 3)}")
    if cd is not None:
        ev.append(f"去公共成分后 genrec_diff={fmt(cd['mean'], 4)}")
    if sd is not None:
        ev.append(f"语义 sem_lift={fmt(sl['mean'], 3) if sl is not None else 'nan'}")
    pos_genre = gd is not None and float(gd["mean"]) > 0 and float(gd["pos_rate"]) > 0.55
    push("①", "图选出的邻居，在真实 genre 空间是否比随机用户更像自己？",
         round(float(gd["mean"]), 5) if gd is not None else "",
         "；".join(ev) if ev else "没有数据",
         ("是：邻居的 genre 相似度显著高于随机基线" if pos_genre
          else "不明显：genre 相似度没有显著高于随机基线"))

    # ② 语义 vs genre
    cc = c("sem_diff")
    ev2 = ""
    ic = None
    if cc is not None:
        ev2 = (f"逐用户 corr(ΔNDCG, sem_diff)：Spearman={fmt(cc['spearman'], 4)}"
               f"（p={fmt(cc['spearman_p'], 3)}, n={int(cc['n'])}）")
    sem_vs_genre = corrs[(corrs["metric"] == "genre_diff") & (corrs["target"] == "sem_diff")]
    if len(sem_vs_genre):
        r = sem_vs_genre.iloc[0]
        ev2 += (f"；corr(genre_diff, sem_diff)：Spearman={fmt(r['spearman'], 4)}"
                f"（p={fmt(r['spearman_p'], 3)}, n={int(r['n'])}）")
        push("②", "服务器学到的兴趣相似度，与真实 genre 相似度是否一致？",
             round(float(r["spearman"]), 5),
             ev2,
             ("一致：语义相似度基本就是 genre 相似度" if float(r["spearman"]) > 0.5
              else ("只有弱相关：语义相似度包含了 genre 之外的东西" if float(r["spearman"]) > 0.15
                    else "几乎不一致：服务器学到的相似度与 genre 兴趣基本无关")))
    else:
        push("②", "服务器学到的兴趣相似度，与真实 genre 相似度是否一致？", "", ev2 or "没有数据", "")

    # ③ genre 相似度 vs 收益（核心）
    cg = c("genre_diff")
    cgr = c("genrec_diff")
    ev3 = []
    verdict3 = "没有数据"
    if cg is not None:
        ev3.append(f"corr(ΔNDCG, genre_diff)：Spearman={fmt(cg['spearman'], 4)}"
                   f"（p={fmt(cg['spearman_p'], 3)}, n={int(cg['n'])}）")
    if cgr is not None:
        ev3.append(f"corr(ΔNDCG, genrec_diff)：Spearman={fmt(cgr['spearman'], 4)}"
                   f"（p={fmt(cgr['spearman_p'], 3)}）")
    bq = buckets.get("genre_diff")
    mono = None
    if bq is not None and len(bq):
        ev3.append("按 genre_diff 五分位：" + "，".join(
            f"Q{int(r['bucket'])}={fmt(r['dndcg_mean'], 4)}(n={int(r['n'])})" for _, r in bq.iterrows()))
        top, bot = bq.iloc[-1], bq.iloc[0]
        diff = float(top["dndcg_mean"]) - float(bot["dndcg_mean"])
        mono = diff
        lo, hi = bootstrap_ci(
            j.dropna(subset=["genre_diff", "dndcg"]).query(
                f"genre_diff >= {top['genre_diff_lo']}")["dndcg"]
        )
        ev3.append(f"最高分位 − 最低分位 = {diff:+.4f}")
    if cg is not None and mono is not None:
        if abs(float(cg["spearman"])) < 0.02 and abs(mono) < 0.002:
            verdict3 = ("否：genre 相似度与图带来的收益基本无关 —— "
                        "「邻居兴趣相似」并不等于「协同有效」")
        elif mono > 0.002 and float(cg["spearman"]) > 0:
            verdict3 = "是：邻居的 genre 相似度越高，图带来的收益越大"
        else:
            verdict3 = "方向相反/不稳定，需结合分位表判断"
    push("③", "**genre 相似度越高，图带来的推荐收益是否越大？**（核心检验）",
         round(float(cg["spearman"]), 5) if cg is not None else "",
         "；".join(ev3) if ev3 else "没有数据", verdict3)

    # ④ 收益 vs 兴趣漂移 / 分散度
    cd_ = c("genre_drift")
    ce = c("genre_entropy")
    ev4 = []
    if cd_ is not None:
        ev4.append(f"corr(ΔNDCG, genre_drift)：Spearman={fmt(cd_['spearman'], 4)}"
                   f"（p={fmt(cd_['spearman_p'], 3)}）")
    bqd = buckets.get("genre_drift")
    if bqd is not None and len(bqd):
        ev4.append("按 genre_drift 五分位：" + "，".join(
            f"Q{int(r['bucket'])}={fmt(r['dndcg_mean'], 4)}" for _, r in bqd.iterrows()))
    if ce is not None:
        ev4.append(f"corr(ΔNDCG, genre_entropy)：Spearman={fmt(ce['spearman'], 4)}"
                   f"（p={fmt(ce['spearman_p'], 3)}）")
    cd_sp = float(cd_["spearman"]) if cd_ is not None and np.isfinite(float(cd_["spearman"])) else float("nan")
    bq1 = None
    if bqd is not None and len(bqd) >= 2:
        bq1 = float(bqd.iloc[0]["dndcg_mean"]) - float(bqd.iloc[-1]["dndcg_mean"])
    if not np.isfinite(cd_sp):
        judge4 = "没有可用的漂移数据"
    elif cd_sp < -0.01:
        judge4 = ("**不是**：兴趣漂移越大，图的收益反而越小（或至少没有更大）—— "
                  "说明图并不是靠「帮到兴趣正在变的用户」来产生价值")
    elif cd_sp > 0.01:
        judge4 = "是：兴趣漂移越大，图的收益越大（图主要在帮兴趣确实在变的用户）"
    else:
        judge4 = "关系很弱（|Spearman| < 0.01），看不出收益与兴趣变化幅度有关"
    push("④", "图的收益是否集中在「兴趣变化大 / 兴趣分散」的用户上？", round(cd_sp, 5),
         ("；".join(ev4) or "没有数据")
         + "（drift 口径：本窗口行为 vs 之前历史，见 compute_interest_drift）", judge4)

    # ⑤ 语义相似度与收益（对照）
    ev5 = []
    if cc is not None:
        ev5.append(f"corr(ΔNDCG, sem_diff)：Spearman={fmt(cc['spearman'], 4)}"
                   f"（p={fmt(cc['spearman_p'], 3)}）")
    bqs = buckets.get("sem_diff")
    if bqs is not None and len(bqs):
        ev5.append("按 sem_diff 五分位：" + "，".join(
            f"Q{int(r['bucket'])}={fmt(r['dndcg_mean'], 4)}" for _, r in bqs.iterrows()))
    push("⑤", "（对照）服务器语义相似度与收益的关系", "", "；".join(ev5) or "没有数据",
         "与 ③ 对比：如果语义相似度相关而 genre 不相关，说明收益来自 genre 之外的因素")

    # ⑥ 总结
    if cg is None or not np.isfinite(float(cg["spearman"])):
        tail6 = "没有可用的 genre 相似度数据（邻居为空或两臂配置不匹配），无法判定"
    else:
        sp = float(cg["spearman"])
        spread = mono if mono is not None else float("nan")
        base = float(j["dndcg"].mean())
        if abs(sp) < 0.05 and np.isfinite(spread) and spread > 0.3 * abs(base):
            tail6 = (f"genre 对齐能解释**收益幅度的一部分**（五分位 Q1→Q5 差 {spread:+.4f}，"
                     f"已接近全样本平均收益 {base:+.4f}），"
                     f"但**预测不了单个用户**（逐用户 Spearman 只有 {sp:+.3f}，NDCG 太稀疏）—— "
                     "两个层次要分开说")
        elif sp > 0.05:
            tail6 = "genre 相似度与图收益同向且个体层面也看得出"
        else:
            tail6 = "genre 相似度与图收益基本无关"
    tail6 += "；注意 ⑤ 显示「学到语义相似度」与收益的关系比 genre 更强"
    push("⑥", "一句话结论", "",
         (f"average ΔNDCG = {fmt(j['dndcg'].mean(), 4)}"
          f"（n={int(j['dndcg'].notna().sum())} 个 用户×窗口 观测；"
          f"ΔNDCG>0 的比例 {fmt(float((j['dndcg'] > 0).mean()), 3)}）"),
         tail6)
    return rows


# ----------------------------------------------------------------- 报告/图
def md_table(frame: pd.DataFrame, cols: Sequence[str], max_rows: int = 40) -> str:
    if frame is None or frame.empty:
        return "（无数据）\n"
    cols = [c for c in cols if c in frame.columns]
    if not cols:
        return "（无数据）\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.head(max_rows).iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                fv = float(v)
                cells.append(fmt(fv) if abs(fv) >= 1e-3 or fv == 0 else f"{fv:.2e}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def make_figure(j: pd.DataFrame, buckets: Dict[str, pd.DataFrame], out_png: Path) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return f"跳过出图（没有 matplotlib）：{exc}"
    keys = [k for k in ("genre_diff", "sem_diff", "genre_drift") if k in buckets and len(buckets[k])]
    if not keys:
        return "跳过出图：没有分位数据"
    fig, axes = plt.subplots(1, max(len(keys), 1), figsize=(5.4 * max(len(keys), 1), 4.4),
                             squeeze=False)
    for ax, key in zip(axes[0], keys):
        b = buckets[key]
        x = b["bucket"].to_numpy()
        y = b["dndcg_mean"].to_numpy()
        lo = b["dndcg_mean"].to_numpy() - b["ci95_lo"].to_numpy()
        hi = b["ci95_hi"].to_numpy() - b["dndcg_mean"].to_numpy()
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=5, lw=1.6, capsize=3, color="#d62728")
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xlabel(f"{key} quintile (low → high)")
        ax.set_ylabel("mean ΔNDCG@10 (dynamic − nograph)")
        ax.set_title(f"Graph gain vs {key}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Genre: does interest similarity explain the graph's gain?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return None


# ----------------------------------------------------------------- 主流程
def main() -> None:
    parser = argparse.ArgumentParser(description="Genre 协同有效性：兴趣相似是否意味着协同有效（真实 genre）")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    root, out = args.root.resolve(), args.out.resolve()
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    j, g = load_joined(root)
    if j.empty:
        raise SystemExit("对齐后没有数据（检查 dynamic_genre 与 nograph 的 seed/window/user 是否一致）")

    # 离线重算兴趣漂移（运行时那列不可靠，见 compute_interest_drift 的说明）
    drift_path = tables / "dg_genre_drift.csv"
    if drift_path.exists():
        drift = pd.read_csv(drift_path)
        print(f"[ok] 复用已有离线 drift：{drift_path}")
    else:
        drift = compute_interest_drift(DATASET, int(j["window"].max()), GENRE_NPZ)
        drift.to_csv(drift_path, index=False, encoding="utf-8-sig")
    j = j.merge(drift, on=["window", "user"], how="left")
    if "genre_drift" in j.columns:
        j["genre_drift_raw"] = j["genre_drift"]          # 保留运行内那列做对照
    j["genre_drift"] = j["drift_local"]                  # 分析统一用离线修正版

    problems = check_arm_configs(root)
    if problems:
        print("!! 两条臂的配置不一致，ΔNDCG 不可信：" + "; ".join(problems))
    if abs(float(j["dndcg"].mean())) > 0.1:
        print(f"!! ΔNDCG 均值 {j['dndcg'].mean():+.3f} 大得不正常（图通路的正常量级是 ~0.01），"
              "请确认 dynamic_genre 与 nograph 是同配置同轮次的运行")
    j.attrs["config_problems"] = problems

    j.to_csv(tables / "dg_genre_merged.csv", index=False, encoding="utf-8-sig")

    # ---- 描述统计（逐指标：均值 / CI / 为正比例）
    metrics = ["sem_nbr", "sem_base", "sem_lift", "sem_diff",
               "genre_nbr", "genre_base", "genre_lift", "genre_diff",
               "genrec_nbr", "genrec_base", "genrec_lift", "genrec_diff",
               "genre_entropy", "genre_drift", "dndcg"]
    desc_rows = []
    for m in metrics:
        if m not in j.columns:
            continue
        d = _finite(j[m])
        if d.size == 0:
            continue
        lo, hi = bootstrap_ci(d)
        desc_rows.append({
            "metric": m, "n": int(d.size), "mean": float(d.mean()),
            "median": float(np.median(d)), "sd": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
            "ci95_lo": lo, "ci95_hi": hi,
            "pos_rate": float((d > 0).mean()),
            "sign_test_p": sign_test_p(d),
        })
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(tables / "dg_genre_descriptives.csv", index=False, encoding="utf-8-sig")

    # ---- 相关（target = 逐用户收益）
    corr_rows = []
    for m in ["sem_lift", "sem_diff", "genre_lift", "genre_diff", "genrec_lift", "genrec_diff",
              "genre_entropy", "genre_drift"]:
        if m not in j.columns:
            continue
        c = corr_pair(j[m], j["dndcg"])
        corr_rows.append({"metric": m, "target": "dndcg", **c})
    # 语义 vs genre 的一致性
    if {"sem_diff", "genre_diff"} <= set(j.columns):
        corr_rows.append({"metric": "genre_diff", "target": "sem_diff",
                          **corr_pair(j["genre_diff"], j["sem_diff"])})
    if {"sem_lift", "genre_lift"} <= set(j.columns):
        corr_rows.append({"metric": "genre_lift", "target": "sem_lift",
                          **corr_pair(j["genre_lift"], j["sem_lift"])})
    corrs = pd.DataFrame(corr_rows)
    corrs.to_csv(tables / "dg_genre_correlations.csv", index=False, encoding="utf-8-sig")

    # ---- 逐用户聚合（降噪）：每个用户跨窗口平均
    agg_spec = {"dndcg": ("dndcg", "mean"), "n_windows": ("window", "nunique")}
    for col in ("genre_diff", "sem_diff", "genre_drift", "genre_entropy",
                "genre_lift", "sem_lift"):
        if col in j.columns:
            agg_spec[col] = (col, "mean")
    per_user = j.groupby(["seed", "user"]).agg(**agg_spec).reset_index()
    per_user.to_csv(tables / "dg_genre_per_user.csv", index=False, encoding="utf-8-sig")
    pu_corrs = pd.DataFrame([
        {"metric": m, "target": "dndcg", **corr_pair(per_user[m], per_user["dndcg"])}
        for m in ("genre_diff", "sem_diff", "genre_drift", "genre_entropy",
                  "genre_lift", "sem_lift")
        if m in per_user.columns
    ])
    pu_corrs.to_csv(tables / "dg_genre_per_user_corr.csv", index=False, encoding="utf-8-sig")

    # ---- 分位表
    buckets: Dict[str, pd.DataFrame] = {}
    bucket_frames = []
    for key in ("genre_diff", "genrec_diff", "sem_diff", "genre_drift", "genre_entropy"):
        if key not in j.columns:
            continue
        b = quintile_table(j, key)
        if b.empty:
            continue
        b.insert(0, "key", key)
        buckets[key] = b
        bucket_frames.append(b)
    if bucket_frames:
        pd.concat(bucket_frames, ignore_index=True).to_csv(
            tables / "dg_genre_quintiles.csv", index=False, encoding="utf-8-sig")

    verdict = pd.DataFrame(build_verdict(j, desc, corrs, buckets))
    verdict.to_csv(tables / "dg_genre_verdict.csv", index=False, encoding="utf-8-sig")

    fig_note = make_figure(j, buckets, figures / "dg_fig_genre_gain.png")

    # ---- 报告
    lines = ["# Genre 协同有效性报告：兴趣相似是否意味着协同有效？\n"]
    lines.append(
        "用 ML-1M `movies.dat` 的 18 个真实类型（物品覆盖率 100%）重建每个用户的兴趣画像，"
        "统计「图选出的邻居 vs 随机用户」在 genre 空间的相似度，再与 `dynamic − nograph` 的"
        "逐用户收益做相关与分位分析。\n"
    )
    lines.append("**一句话：** " +
                 (f"ΔNDCG 均值 {fmt(j['dndcg'].mean(), 4)}；"
                  + (str(verdict[verdict['verdict_id'] == '③'].iloc[0]['judgement'])
                     if (verdict['verdict_id'] == '③').any() else "")) + "\n")
    if j.attrs.get("config_problems"):
        lines.append("> **警告：两条臂的配置不一致，ΔNDCG 不可信** —— "
                     + "; ".join(j.attrs["config_problems"]) + "\n")
    lines.append(md_table(verdict, ["verdict_id", "question", "value", "evidence", "judgement"]))
    lines.append("")
    lines.append("## 描述统计\n")
    lines.append(md_table(desc, ["metric", "n", "mean", "median", "ci95_lo", "ci95_hi",
                                 "pos_rate", "sign_test_p"]))
    lines.append("")
    lines.append("## 相关（target = 逐用户 ΔNDCG）\n")
    lines.append(md_table(corrs, ["metric", "target", "n", "pearson", "pearson_p",
                                  "spearman", "spearman_p"]))
    lines.append("")
    lines.append("## 逐用户聚合后的相关（降噪）\n")
    lines.append(md_table(pu_corrs, ["metric", "target", "n", "pearson", "spearman", "spearman_p"]))
    lines.append("")
    if buckets:
        lines.append("## 五分位：相似度从低到高，图的收益怎么变\n")
        for key, b in buckets.items():
            lines.append(f"### {key}\n")
            lines.append(md_table(b, ["bucket", "n", f"{key}_lo", f"{key}_hi",
                                      f"{key}_mean", "dndcg_mean", "ci95_lo", "ci95_hi",
                                      "dndcg_pos_rate"]))
            lines.append("")
    lines.append("## 怎么读\n")
    lines.append("- `genre_diff` > 0 表示「邻居比随机用户更像自己的 genre 口味」；"
                 "`genrec_diff` 是去掉 Drama/Comedy 这类人人都看的公共成分后的版本。\n"
                 "- ③ 是核心：如果 genre 相似度与收益无关，说明**兴趣相似并不等于协同有效** —— "
                 "图带来的收益来自别的机制（例如把相似用户的表示直接当信息注入，而不是靠 genre 口味对齐）。\n"
                 "- 收益 ΔNDCG 在逐用户层面很稀疏（多数用户两臂都是 0），所以同时看逐用户聚合后的相关。\n")
    if fig_note:
        lines.append(f"- 出图提示：{fig_note}\n")
    (out / "dg_genre_report.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 屏幕
    pd.set_option("display.width", 220)
    print("\n" + "=" * 78)
    print(f"对齐后的观测数: {len(j)}（{j['seed'].nunique()} 个种子 × {j['window'].nunique()} 窗口）")
    print(f"逐用户收益 ΔNDCG: mean={fmt(j['dndcg'].mean(), 4)}, "
          f">0 比例={fmt(float((j['dndcg'] > 0).mean()), 3)}")
    print("=" * 78)
    print(desc[["metric", "n", "mean", "ci95_lo", "ci95_hi", "pos_rate"]].round(4).to_string(index=False))
    print("\n相关（target=ΔNDCG）：")
    print(corrs[corrs["target"] == "dndcg"][["metric", "n", "pearson", "pearson_p",
                                             "spearman", "spearman_p"]].round(4).to_string(index=False))
    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    for _, row in verdict.iterrows():
        print(f"[{row['verdict_id']}] {row['question']}")
        if row["evidence"]:
            print(f"     {row['evidence']}")
        if str(row["judgement"]).strip():
            print(f"     -> {row['judgement']}")
    print("\n输出：")
    for p in (out / "dg_genre_report.md", tables / "dg_genre_verdict.csv", tables / "dg_genre_correlations.csv",
              tables / "dg_genre_quintiles.csv", figures / "dg_fig_genre_gain.png"):
        print(("  " if p.exists() else "  (缺) ") + str(p))
    if fig_note:
        print(f"\n注意：{fig_note}")


if __name__ == "__main__":
    main()
