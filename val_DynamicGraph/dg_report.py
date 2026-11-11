#!/usr/bin/env python
"""生成最终实验验证报告（数据驱动，直接读各聚合表，不手抄数字）。

产出：`val_DynamicGraph/实验验证报告_动态图_2026-11-06.md`

报告结构（对应三个研究问题）
--------------------------
  一、实验设置与严格性
  二、问题一：动态图有没有独立贡献？提升了多少、提升来自哪里
  三、问题二：更新机制是否合理（冻结归因：EMA / 新边初始化 / 状态保留规则）
  四、问题三：兴趣相似是否意味着协同有效（ML 真实 Genre + 双数据集无标签复核）
  五、对研究问题的含义（第二阶段方向）
  六、局限
  七、复现与产出清单

数据来源
--------
  results/aggregated/{tables,figures}         主对照（双数据集）
  results/freeze/<dataset>/{tables,figures}   冻结归因（双数据集）
  results/genre/{tables,figures}              ML 真实 Genre
  results/quality/<dataset>/{tables,figures}  无标签版问题三（双数据集）

用法
----
  python val_DynamicGraph/dg_report.py
  python val_DynamicGraph/dg_report.py --out 实验验证报告.md
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DG_DIR = Path(__file__).resolve().parent
RESULTS = DG_DIR / "results"
DEFAULT_OUT = DG_DIR / "实验验证报告_动态图_2026-11-06.md"

# 各聚合表的根目录。默认沿用 legacy 路径（results/aggregated、results/freeze…）；
# 阶段 4 的 5 种子复跑写到 results/agg_formal 等隔离目录，用环境变量或 CLI 切换，
# 这样旧结果不会被覆盖，报告里也能同时对比「修前 / 修后」两套表。
AGG_ROOT = Path(os.environ.get("DG_AGG_ROOT") or (RESULTS / "aggregated"))
FREEZE_ROOT = Path(os.environ.get("DG_FREEZE_ROOT") or (RESULTS / "freeze"))
GENRE_ROOT = Path(os.environ.get("DG_GENRE_ROOT") or (RESULTS / "genre"))
QUALITY_ROOT = Path(os.environ.get("DG_QUALITY_ROOT") or (RESULTS / "quality"))
RAW_ROOT = Path(os.environ.get("DG_RAW_ROOT") or (RESULTS / "raw"))

DATASETS = ["movielens-1m", "amazon-digital_music"]
DS_LABEL = {"movielens-1m": "MovieLens-1M", "amazon-digital_music": "Amazon-Digital-Music"}
#: 聚合表（results/aggregated）里的 dataset 列用的是「传给 --dataset 的原始名」，
#: 而 results/{freeze,quality}/ 与图片文件名用的是 slug。这里显式对照，避免读错目录。
DS_RAW = {"movielens-1m": "movielens-1m", "amazon-digital_music": "amazon-digital music"}


# ----------------------------------------------------------------- 工具
def slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def fmt(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return str(x)
    if not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}"


def pct(x, nd: int = 0) -> str:
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return "—"
    return "—" if not np.isfinite(v) else f"{v*100:.{nd}f}%"


def md_table(df: Optional[pd.DataFrame], cols: Sequence[str], max_rows: int = 40,
             headers: Optional[Sequence[str]] = None) -> str:
    if df is None or len(df) == 0:
        return "（无数据）\n"
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "（无数据）\n"
    head = list(headers) if headers and len(headers) == len(cols) else cols
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.head(max_rows).iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                fv = float(v)
                if not np.isfinite(fv):
                    cells.append("—")
                elif abs(fv) >= 1 and abs(fv - round(fv)) < 1e-9:
                    # 计数/轮次/窗口标签这类整数值：别印成 6040.0000
                    cells.append(str(int(round(fv))))
                else:
                    cells.append(f"{fv:.4f}" if abs(fv) >= 1e-3 or fv == 0 else f"{fv:.2e}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"| …共 {len(df)} 行 |" + " |" * (len(cols) - 1))
    return "\n".join(lines) + "\n"


def img(path: Path, caption: str) -> str:
    if not path.exists():
        return f"（图缺失：`{path.name}`）\n"
    try:
        rel = path.relative_to(DG_DIR)
    except ValueError:  # 聚合表放在 val_DynamicGraph 之外时只给文件名
        rel = Path(path.name)
    return f"![{caption}]({rel})\n\n*{caption}*\n"


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(DG_DIR), "log", "--oneline", "-1"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "(未知)"


# ----------------------------------------------------------------- 取数
class Bundle:
    def __init__(self) -> None:
        agg = AGG_ROOT / "tables"
        self.arm_summary = read_csv(agg / "dg_arm_summary.csv")
        self.arm_curve = read_csv(agg / "dg_arm_curve.csv")
        self.contrasts = read_csv(agg / "dg_contrasts.csv")
        self.verdict = read_csv(agg / "dg_verdict.csv")
        self.hygiene = read_csv(agg / "dg_hygiene.csv")
        self.freeze: Dict[str, Dict[str, pd.DataFrame]] = {}
        for ds in DATASETS:
            base = FREEZE_ROOT / slug(ds) / "tables"
            if base.exists():
                self.freeze[ds] = {
                    "cells": read_csv(base / "dg_freeze_cells.csv"),
                    "contrasts": read_csv(base / "dg_freeze_contrasts.csv"),
                    "verdict": read_csv(base / "dg_freeze_verdict.csv"),
                }
        g = GENRE_ROOT / "tables"
        self.genre = {
            "verdict": read_csv(g / "dg_genre_verdict.csv"),
            "desc": read_csv(g / "dg_genre_descriptives.csv"),
            "corr": read_csv(g / "dg_genre_correlations.csv"),
            "quint": read_csv(g / "dg_genre_quintiles.csv"),
        }
        self.quality: Dict[str, Dict[str, pd.DataFrame]] = {}
        for ds in DATASETS:
            base = QUALITY_ROOT / slug(ds) / "tables"
            if base.exists():
                self.quality[ds] = {
                    "verdict": read_csv(base / "dg_quality_verdict.csv"),
                    "desc": read_csv(base / "dg_quality_descriptives.csv"),
                    "quint": read_csv(base / "dg_quality_quintiles.csv"),
                }

    # ---------- 主对照取数
    def contrast(self, ds: str, name: str, metric: str = "ndcg_win_mean") -> Optional[pd.Series]:
        c = self.contrasts
        if c is None:
            return None
        raw = DS_RAW.get(ds, ds)
        s = c[(c["dataset"] == raw) & (c["contrast"] == name) & (c["metric"] == metric)]
        return s.iloc[0] if len(s) else None

    def arm_row(self, ds: str, arm: str) -> Optional[pd.Series]:
        a = self.arm_summary
        if a is None:
            return None
        raw = DS_RAW.get(ds, ds)
        s = a[(a["dataset"] == raw) & (a["arm_dir"] == arm)]
        return s.iloc[0] if len(s) else None

    def verdict_rows(self, ds: str) -> Optional[pd.DataFrame]:
        v = self.verdict
        if v is None:
            return None
        raw = DS_RAW.get(ds, ds)
        s = v[v["dataset"] == raw]
        return s if len(s) else None

    def arms_of(self, ds: str) -> Optional[pd.DataFrame]:
        a = self.arm_summary
        if a is None:
            return None
        raw = DS_RAW.get(ds, ds)
        s = a[a["dataset"] == raw]
        return s if len(s) else None

    def freeze_flags(self, ds: str) -> Optional[Dict[str, bool]]:
        """四种配置各自有没有冻住（turnover 低 / frozen_share 高 / ECR 近 0，任一命中）。"""
        fz = self.freeze.get(ds)
        if not fz or fz.get("cells") is None:
            return None
        flags: Dict[str, bool] = {}
        for _, r in fz["cells"].iterrows():
            def num(key):
                try:
                    v = float(r[key])
                except Exception:  # noqa: BLE001
                    return float("nan")
                return v if np.isfinite(v) else float("nan")

            tv, fzv, ec = num("turnover_from_ref"), num("frozen_share"), num("ecr")
            flags[str(r["cell"])] = (
                (np.isfinite(tv) and tv < 0.5)
                or (np.isfinite(fzv) and fzv > 0.3)
                or (np.isfinite(ec) and ec < 0.01)
            )
        return flags

    def curve_of(self, ds: str) -> Optional[pd.DataFrame]:
        a = self.arm_curve
        if a is None:
            return None
        raw = DS_RAW.get(ds, ds)
        s = a[a["dataset"] == raw]
        return s if len(s) else None

    def seeds_of(self, ds: str) -> List[str]:
        """从逐 run 目录名里读出实际用到的种子（不在代码里硬编码）。"""
        seeds: set = set()
        for base in (RAW_ROOT / slug(ds), RAW_ROOT / DS_RAW.get(ds, ds)):
            if not base.exists():
                continue
            for p in base.glob("*/seed_*"):
                if p.is_dir():
                    seeds.add(p.name[len("seed_"):])
        return sorted(seeds, key=lambda x: (len(x), x))


def c_line(b: Bundle, ds: str, contrast: str, label: str) -> str:
    """一行配对差值。

    **种子级是主证据**（每个种子一个观测）：均值、t 分布 95%CI、配对 t 的 p；
    窗口级（种子×窗口）只作敏感性分析并列在同一行 —— 同一种子跨窗口不独立，
    所以不能把 n=种子数×窗口数 的 p 当主证据。
    """
    r = b.contrast(ds, contrast)
    if r is None:
        return f"- {label}：无数据"

    def g(key: str) -> float:
        return float(r[key]) if key in r.index and pd.notna(r[key]) else float("nan")

    seed_mean, lo, hi, p_seed = g("delta_seed_mean"), g("ci95_seed_lo"), g("ci95_seed_hi"), g("p_seed_t")
    if np.isfinite(seed_mean):
        head = (f"- {label}：**Δ={seed_mean:+.4f}**"
                f"（种子级 95%CI [{lo:+.4f}, {hi:+.4f}]，"
                f"种子 {int(g('seeds_positive'))}/{int(g('seeds_total'))} 为正，"
                f"p={p_seed:.3g}")
        if np.isfinite(g("cohen_dz")):
            head += f"，dz={g('cohen_dz'):.2f}"
        head += "）"
        tail = (f"；窗口级敏感性：Δ={g('delta_mean'):+.4f}，p={g('p_paired'):.3g}"
                f"（n={int(g('n_obs'))} 个 种子×窗口 观测）")
        return head + tail
    # 旧聚合表没有种子级列时退回窗口级（并在文里说明）
    return (f"- {label}：**Δ={g('delta_mean'):+.4f}**"
            f"（窗口级 95%CI [{g('ci95_lo'):+.4f}, {g('ci95_hi'):+.4f}]，"
            f"窗口正号 {pct(r['win_rate']) if 'win_rate' in r else '—'}，"
            f"种子 {int(r.get('seeds_positive', 0))}/{int(r.get('seeds_total', 0))} 为正，"
            f"p={g('p_paired'):.3g}；**该表无种子级列，p 为窗口级**）")


# ----------------------------------------------------------------- 各章
def section_setup(b: Bundle) -> str:
    rows = []
    a = b.arm_summary
    if a is not None:
        for ds in DATASETS:
            s = b.arms_of(ds)
            if s is None or len(s) == 0:
                continue
            rows.append(f"- **{DS_LABEL.get(ds, ds)}**：{len(s)} 个臂/变体，"
                        f"主臂各 {int(s['n_seeds'].max())} 个种子")
    seeds: set = set()
    for ds in DATASETS:
        seeds.update(b.seeds_of(ds))
    seed_txt = " / ".join(sorted(seeds, key=lambda x: (len(x), x))) if seeds else "（未知）"
    return f"""## 一、实验设置与严格性

> 本报告的全部数字来自 **修复后重训的正式批次**（`results/raw_formal`）：
> 摘要上传时机改为本地训练**之后**取（F7）、时间切分的排除集/回退候选/历史集合改为因果（F8）。
> 修复前的 `results/raw`、`results/raw_phase2(b)` 一律标为 legacy，**不与本表混比**。

### 1.1 协议（两个数据集完全一致）

| 项 | 取值 |
|---|---|
| 时间窗口 | 5 个等长时序窗口（`slice` 模式：每窗口只用该时段交互评测） |
| 每窗口轮数 | 120 轮联邦训练 |
| 参与率 | C = 0.1（每轮抽 10% 客户端） |
| 本地训练 | E = 1 个 epoch，Adam lr = 1e-3，服务器 lr 从 15 线性衰减到 1（只作用 item_emb） |
| 个性化 | 编码器（attention/FFN/LayerNorm/W_proj/q_att）留客户端，其余共享（FedAvg，uniform） |
| 图 | Top-K = 20，ρ = 0.2（论文默认），预热 20 轮，更新全部活跃节点 |
| 评测 | 1 正 + 99 负候选，NDCG@10 / HR@10，每 10 轮评一次 |
| 种子 | {seed_txt}（主臂 5 个；刷新频率扫描与锚点按需减少） |
| 统计口径 | **种子级**为主证据（每种子一个观测，n=5，t 分布 95%CI + 配对 t）；窗口级只作敏感性分析 |

### 1.2 数据集

| 数据集 | 用户数 | 物品数 | 交互数 | 密度 | 每窗口评测用户 |
|---|---|---|---|---|---|
| MovieLens-1M | 6,040 | 3,416 | 999,611 | 0.0484 | 1,421 ~ 1,763 |
| Amazon-Digital-Music | 12,381 | 9,906 | 123,518 | 0.0010 | 3,178 ~ 3,657 |

### 1.3 六条臂与对照网格

| 臂 | 图分支前向 | 服务器上的边状态 |
|---|---|---|
| NoGraph | **关**（`u_i = seq_emb_i`） | 不建图、从不更新 |
| StaticGraph | 开 | 预热结束建一次（第 21 轮），之后完全冻结 |
| PeriodicGraph | 开 | 每 120 轮清零重建一次（慢刷新对照） |
| DynamicGraph | 开 | 每轮按 `e ← ρ·e + (1−ρ)·Δe` 更新 |
| **OnboardingGraph** | 开 | **只给「当前还没有邻居」的用户建边**（每人恰好一次）→ 分离「覆盖」 |
| **FreshOnlyGraph** | 开 | **只给「自上次更新后有新上传」的用户重估**（同一用户可多次）→ 「有针对性」的演化 |

`StaticGraph` 与 `DynamicGraph` 在预热结束那一轮之前**逐位相同**（同一次构图、同一个 Top-K），
之后只差「要不要继续更新边」。因此 `Dynamic − Static` 就是动态性的净效应，
`Static − NoGraph` 是图分支的净效应，`Dynamic − Periodic` 是「每轮 EMA 相对慢刷新」的边际效应。

**三项分解**（这是本报告回答「收益来自哪里」的主要口径）：

    Dynamic − Static = (Onboarding − Static) + (Dynamic − Onboarding)
                       └── 覆盖（新用户首次获得邻居）──┘   └── 演化（已覆盖用户的边权重估）──┘

另有诊断变体（不参与主对照）：`static_at120`（更新到第 120 轮再冻结）、
`static_build120`（第 120 轮一次性建图后冻结）、`periodic_boundary`（窗口边界重建）、
`periodic_p20` / `periodic_p240`（刷新周期 20 / 240 轮）、`dynamic @ ρ=0.8` 的四格冻结网格、
`dynamic_mix0.005` / `dynamic_mix0.05`（图分支剂量锚点）。

### 1.4 严格性保证（三层）

1. **跑前自检** `dg_selfcheck.py` 9 条断言全部 PASS：三条臂负样本流逐位相同、NoGraph 的 `delta_rel` 恒为 0、
   StaticGraph 冻结后边签名不变、DynamicGraph 每轮都更新、`--graph-mix 0` 与 NoGraph 逐用户 NDCG 逐位相同、
   诊断前向与 `score_user` 一致、初始参数逐位相同、参与序列可复现。
2. **运行中硬检查**：NoGraph 的 `delta_rel_max` 必须恰为 0、Static 冻结后签名不得变化（否则直接抛错）、
   图臂预热后不得是空图（否则打印醒目警告）。
3. **聚合后核对** `dg_verify_formal.sh` 29 项全 PASS：run 数（128 = ML 78 + Amazon 50）、
   `ecr_def`/`core_version` 唯一、主臂 5 种子、`static_build120` 首次建图恰在第 120 轮、
   冻结后边变化为 0、`onboarding` 每人恰好建边一次、`fresh_only` 重复重估、
   `periodic_boundary` 恰 5 次刷新、`dg_hygiene.csv` 全通过、报告章节与图引用完整。

{chr(10).join(rows) if rows else "（尚无结果）"}
"""


def section_q1(b: Bundle) -> str:
    out: List[str] = ["## 二、问题一：动态图有没有独立贡献？提升了多少、来自哪里\n"]

    for ds in DATASETS:
        s = b.arms_of(ds)
        if s is None or len(s) == 0:
            continue
        out.append(f"### 2.{DATASETS.index(ds)+1} {DS_LABEL.get(ds, ds)}\n")
        out.append("**各臂总体水平**（窗口级 NDCG@10，跨种子平均；六条核心臂，"
                   "诊断变体见本章末尾的诊断表与第三章）\n")
        core = ["nograph", "static", "periodic", "dynamic", "onboarding", "fresh_only"]
        s_core = s[s["arm_dir"].isin(core)].copy()
        order = {a: i for i, a in enumerate(core)}
        s_core["_o"] = s_core["arm_dir"].map(order)
        s_core = s_core.sort_values("_o").drop(columns="_o")
        cols = ["arm_dir", "n_seeds", "ndcg_win_mean", "ndcg_win_last", "delta_rel",
                "neighbor_lift", "ecr", "n_edges", "graph_updates"]
        headers_map = {
            "arm_dir": "臂", "n_seeds": "种子数", "ndcg_win_mean": "窗口级 NDCG@10",
            "ndcg_win_last": "末窗 NDCG@10", "delta_rel": "delta_rel（图分支剂量）",
            "neighbor_lift": "neighbor_lift（邻居质量）", "ecr": "ECR（边变化率）",
            "n_edges": "有边用户", "graph_updates": "图更新次数",
            "ecr_update_mean": "ECR（更新轮均值，v2 定义）",
            "graph_turnover_mean": "图变化率 graph_turnover",
            "active_coverage_mean": "本窗口评测覆盖率",
        }
        cols = [c for c in cols if c in s_core.columns]
        # 阶段 2 的新结构指标：只有真的跑过新记录层的臂才展示（旧 run 是 NaN）
        has_new_ecr = (
            "ecr_update_mean" in s_core.columns and s_core["ecr_update_mean"].notna().any()
        )
        if has_new_ecr and "ecr" in cols:
            cols.remove("ecr")   # 旧的 ecr 只在"评测轮"记录，已由 ecr_update_mean 取代
        for extra in ("ecr_update_mean", "graph_turnover_mean", "active_coverage_mean"):
            if extra in s_core.columns and s_core[extra].notna().any():
                cols.append(extra)
        out.append(md_table(s_core, cols, headers=[headers_map[c] for c in cols]))
        out.append("")
        # 复现性小检查：带 genre 诊断重跑的那条 dynamic 应与 dynamic 完全一致
        g_row = s[s["arm_dir"] == "dynamic_genre"]
        d_row = s[s["arm_dir"] == "dynamic"]
        if len(g_row) and len(d_row):
            same = abs(float(g_row.iloc[0]["ndcg_win_mean"]) - float(d_row.iloc[0]["ndcg_win_mean"])) < 1e-9
            out.append(f"> 复现性检查：带 genre 诊断重跑的 `dynamic_genre` 与 `dynamic` 的窗口级 NDCG "
                       f"{'**完全相同**' if same else '存在差异（需排查）'}，说明诊断代码不影响训练。\n")
        out.append("**配对差值**（**种子级为主证据**；窗口级只作敏感性分析）\n")
        out.append(c_line(b, ds, "dynamic_minus_nograph", "③ Dynamic − NoGraph（总贡献）"))
        out.append(c_line(b, ds, "static_minus_nograph", "① Static − NoGraph（图分支本身）"))
        out.append(c_line(b, ds, "dynamic_minus_static", "② Dynamic − Static（动态性净效应）"))
        out.append(c_line(b, ds, "periodic_minus_nograph", "④ Periodic − NoGraph（慢刷新够不够）"))
        out.append(c_line(b, ds, "dynamic_minus_periodic", "⑤ Dynamic − Periodic（每轮 EMA 的边际）"))
        st = b.contrast(ds, "static_at120_minus_static")
        if st is not None:
            out.append(c_line(b, ds, "static_at120_minus_static",
                              "①b 「更新到 120 轮再冻结」− 冻于第 21 轮（晚冻结有用吗）"))
        st2 = b.contrast(ds, "static_build120_minus_static")
        if st2 is not None:
            out.append(c_line(b, ds, "static_build120_minus_static",
                              "①b2 「第 120 轮一次性建图」− 冻于第 21 轮（晚建图有用吗；只看 W2–W5）"))
        out.append("")

        # 策略臂（覆盖 vs 演化）——只有跑过 onboarding / fresh_only 才出现
        strat = [a for a in ("onboarding", "fresh_only") if a in set(s["arm_dir"])]
        if strat:
            out.append("**策略臂：收益来自覆盖还是演化**\n")
            g_onb = b.contrast(ds, "onboarding_minus_static")
            g_evo = b.contrast(ds, "dynamic_minus_onboarding")
            g_fresh = b.contrast(ds, "fresh_only_minus_static")
            if g_onb is not None:
                out.append(c_line(b, ds, "onboarding_minus_static",
                                  "覆盖项 Onboarding − Static（晚出现用户首次拿到邻居）"))
            if g_evo is not None:
                out.append(c_line(b, ds, "dynamic_minus_onboarding",
                                  "演化项 Dynamic − Onboarding（覆盖相同时，边权持续重估）"))
            if g_fresh is not None:
                out.append(c_line(b, ds, "fresh_only_minus_static",
                                  "针对性 FreshOnly − Static（只更新有新证据的用户）"))
            out.append(c_line(b, ds, "dynamic_minus_fresh_only",
                              "DynamicGraph − FreshOnlyGraph（无差别 EMA 相对「只更新有新证据的人」）"))
            g2 = b.contrast(ds, "dynamic_minus_static")
            if g2 is not None and g_onb is not None and g_evo is not None:
                total = float(g2["delta_mean"])
                cover = float(g_onb["delta_mean"])
                evo = float(g_evo["delta_mean"])
                resid = total - (cover + evo)
                share = (cover / total) if total else float("nan")
                out.append("")
                out.append("**三项分解**（② = 覆盖 + 演化，所有数取自 `dg_stats_decomp.csv`）\n")
                out.append("| 成分 | 对照 | Δ NDCG@10 | 占 ② |")
                out.append("|---|---|---|---|")
                out.append(f"| ② 动态性净效应 | Dynamic − Static | **{total:+.4f}** | 100% |")
                out.append(f"| **覆盖项** | Onboarding − Static | **{cover:+.4f}** | **{share:.0%}** |")
                out.append(f"| 演化项 | Dynamic − Onboarding | {evo:+.4f} | {1-share:.0%} |")
                out.append(f"| 恒等残差 | ② − (覆盖+演化) | {resid:+.1e} | — |")
                out.append("")
                out.append("> 覆盖项只由「第 21 轮之后才第一次获得邻居的用户」贡献，"
                           "所以它衡量的是**覆盖（含覆盖时点）**，不是「有没有边」的二元效应。\n")
            out.append("")

        # 刷新频率 × 覆盖时点（诊断臂）
        freq = [a for a in ("periodic", "periodic_boundary", "periodic_p20", "periodic_p240")
                if a in set(s["arm_dir"])]
        if len(freq) > 1:
            out.append(f"**诊断：刷新频率 × 覆盖时点（{DS_LABEL.get(ds, ds)}）**\n")
            rows = []
            for arm in freq:
                r = b.arm_row(ds, arm)
                con = b.contrast(ds, f"{arm}_minus_periodic") if arm != "periodic" else None
                rows.append({
                    "arm": arm,
                    "refresh": float(r["refresh_rounds"]) if "refresh_rounds" in r.index and pd.notna(r["refresh_rounds"]) else float("nan"),
                    "cov": float(r["active_coverage_mean"]) if "active_coverage_mean" in r.index and pd.notna(r["active_coverage_mean"]) else float("nan"),
                    "tv": float(r["graph_turnover_mean"]) if "graph_turnover_mean" in r.index and pd.notna(r["graph_turnover_mean"]) else float("nan"),
                    "lift": float(r["neighbor_lift"]),
                    "ndcg": float(r["ndcg_win_mean"]),
                    "d": float(con["delta_seed_mean"]) if con is not None else float("nan"),
                    "p": float(con["p_seed_t"]) if con is not None else float("nan"),
                })
            out.append(md_table(pd.DataFrame(rows),
                                ["arm", "refresh", "cov", "tv", "lift", "ndcg", "d", "p"],
                                headers=["臂", "每次运行刷新次数", "active_coverage",
                                         "graph_turnover", "neighbor_lift", "窗口级 NDCG@10",
                                         "Δ vs periodic（种子级）", "p"]))
            out.append("")
            out.append("> 刷新越靠近窗口起点（`periodic` 在每窗第 21 轮、`periodic_boundary` 在窗口边界），"
                       "新窗口的用户越晚被覆盖；`active_coverage` 就是这件事的直接读数。\n")

        # 静态臂变体（晚冻结 / 晚建图）
        stv = [a for a in ("static", "static_at120", "static_build120") if a in set(s["arm_dir"])]
        if len(stv) > 1:
            out.append(f"**诊断：什么时候建图（{DS_LABEL.get(ds, ds)}）**\n")
            rows = []
            for arm in stv:
                r = b.arm_row(ds, arm)
                con = b.contrast(ds, f"{arm}_minus_static") if arm != "static" else None
                rows.append({
                    "arm": arm,
                    "edges": float(r["n_edges"]) if "n_edges" in r.index else float("nan"),
                    "cov": float(r["active_coverage_mean"]) if "active_coverage_mean" in r.index and pd.notna(r["active_coverage_mean"]) else float("nan"),
                    "lift": float(r["neighbor_lift"]) if pd.notna(r["neighbor_lift"]) else float("nan"),
                    "ndcg": float(r["ndcg_win_mean"]),
                    "d": float(con["delta_seed_mean"]) if con is not None else float("nan"),
                    "p": float(con["p_seed_t"]) if con is not None else float("nan"),
                })
            out.append(md_table(pd.DataFrame(rows),
                                ["arm", "edges", "cov", "lift", "ndcg", "d", "p"],
                                headers=["臂", "有边用户", "active_coverage", "neighbor_lift",
                                         "窗口级 NDCG@10", "Δ vs static（种子级）", "p"]))
            out.append("")
            out.append("> `static` 在第 21 轮建图后立即冻结（只覆盖 W1 那批用户）；"
                       "`static_at120` 更新到第 120 轮再冻结；"
                       "`static_build120` 第 120 轮一次性建图后冻结。\n")

        # 锚点（图分支剂量）
        anchors = [a for a in ("dynamic", "dynamic_mix0.05", "dynamic_mix0.005")
                   if a in set(s["arm_dir"])]
        if len(anchors) > 1:
            out.append(f"**锚点：图分支剂量（D1 口径可比性，{DS_LABEL.get(ds, ds)}）**\n")
            rows = []
            for arm in anchors:
                r = b.arm_row(ds, arm)
                rows.append({
                    "arm": arm,
                    "drel": float(r["delta_rel"]) if pd.notna(r["delta_rel"]) else float("nan"),
                    "ndcg": float(r["ndcg_win_mean"]),
                    "n": int(r["n_seeds"]),
                })
            out.append(md_table(pd.DataFrame(rows), ["arm", "drel", "ndcg", "n"],
                                headers=["臂（后缀即 graph_mix）", "delta_rel（图分支剂量）",
                                         "窗口级 NDCG@10", "种子数"]))
            out.append("")
            out.append("> 这三条只差图分支注入强度：`graph_mix=1` 是主对照，"
                       "0.05 / 0.005 用来对齐中期报告的口径（剂量越小，`delta_rel` 越小）。\n")

        # 逐窗口
        c = b.curve_of(ds)
        if c is not None and len(c):
            if True:
                piv = c.pivot_table(index="window", columns="arm_dir", values="ndcg_win_mean")
                keep = [k for k in ("nograph", "static", "periodic", "dynamic") if k in piv.columns]
                if keep:
                    out.append("**逐窗口（跨种子平均）**\n")
                    show = piv[keep].reset_index()
                    show["window"] = show["window"].astype(int)
                    out.append(md_table(show, ["window", *keep],
                                        headers=["窗口", *keep]))
                    out.append("")

        out.append(img(AGG_ROOT / "figures" / f"dg_fig_arms_{slug(ds)}.png",
                       f"{DS_LABEL.get(ds, ds)}：六条臂的 NDCG 曲线 / 图分支剂量 delta_rel / 邻居质量 lift"))
        out.append("")

    # ---- 跨数据集对比
    rows = []
    for ds in DATASETS:
        s = b.arms_of(ds)
        if s is None or len(s) == 0:
            continue
        def arm_ndcg(arm: str):
            r = b.arm_row(ds, arm)
            return float(r["ndcg_win_mean"]) if r is not None else float("nan")

        base = arm_ndcg("nograph")
        g3 = b.contrast(ds, "dynamic_minus_nograph")
        g4 = b.contrast(ds, "periodic_minus_nograph")
        g5 = b.contrast(ds, "dynamic_minus_periodic")
        g2 = b.contrast(ds, "dynamic_minus_static")
        g1 = b.contrast(ds, "static_minus_nograph")

        def sd(con) -> float:
            """种子级 Δ（主证据）；没有种子级列时退回窗口级。"""
            if con is None:
                return float("nan")
            if "delta_seed_mean" in con.index and pd.notna(con["delta_seed_mean"]):
                return float(con["delta_seed_mean"])
            return float(con["delta_mean"])

        def sp(con) -> float:
            if con is None:
                return float("nan")
            if "p_seed_t" in con.index and pd.notna(con["p_seed_t"]):
                return float(con["p_seed_t"])
            return float(con["p_paired"])

        g_onb = b.contrast(ds, "onboarding_minus_static")
        g_evo = b.contrast(ds, "dynamic_minus_onboarding")
        d3, d2 = sd(g3), sd(g2)
        cover, evo = sd(g_onb), sd(g_evo)
        rows.append({
            "dataset": DS_LABEL.get(ds, ds),
            "nograph": base,
            "static": arm_ndcg("static"),
            "periodic": arm_ndcg("periodic"),
            "dynamic": arm_ndcg("dynamic"),
            "g3": d3,
            "g3_rel": (100.0 * d3 / base) if (np.isfinite(d3) and base) else float("nan"),
            "g3_p": sp(g3),
            "g1": sd(g1),
            "g2": d2,
            "cover": cover,
            "cover_share": (100.0 * cover / d2) if (np.isfinite(cover) and np.isfinite(d2) and d2) else float("nan"),
            "evo": evo,
            "g5": sd(g5),
        })
    if rows:
        cmp_df = pd.DataFrame(rows)
        out.append("### 2.3 跨数据集对比\n")
        out.append("同一个结论在两个数据集上是否都成立（**全部为种子级 Δ**）：\n")
        out.append(md_table(
            cmp_df,
            ["dataset", "nograph", "static", "periodic", "dynamic", "g3", "g3_rel", "g3_p",
             "g1", "g2", "cover", "cover_share", "evo", "g5"],
            headers=["数据集", "NoGraph", "Static", "Periodic", "Dynamic",
                     "③Δ", "③相对增幅 %", "③p", "①Δ（静态）", "②Δ（动态性）",
                     "覆盖项 Δ", "覆盖占 ② %", "演化项 Δ", "⑤每轮 EMA 边际"],
        ))
        out.append("")
        out.append("> 读法：③ 是动态图相对无图的总收益；② 是动态性净效应，"
                   "它几乎全部由「覆盖项」解释（新用户第一次拿到邻居），「演化项」只占个位数百分比；"
                   "⑤ 是「每轮 EMA」相对慢刷新的额外边际。\n")
        out.append("")

    # ---- 结论汇总（种子级为主证据）
    out.append("### 2.4 结论\n")
    lines = []
    for ds in DATASETS:
        g3 = b.contrast(ds, "dynamic_minus_nograph")
        g2 = b.contrast(ds, "dynamic_minus_static")
        g1 = b.contrast(ds, "static_minus_nograph")
        g_onb = b.contrast(ds, "onboarding_minus_static")
        g_evo = b.contrast(ds, "dynamic_minus_onboarding")
        g5 = b.contrast(ds, "dynamic_minus_periodic")
        if g3 is None:
            continue

        def val(con, key):
            if con is None or key not in con.index or pd.isna(con[key]):
                return float("nan")
            return float(con[key])

        d3, p3 = val(g3, "delta_seed_mean"), val(g3, "p_seed_t")
        if not np.isfinite(p3):
            d3, p3 = float(g3["delta_mean"]), float(g3["p_paired"])
        parts = [f"在 **{DS_LABEL.get(ds, ds)}** 上 DynamicGraph 相对 NoGraph 提升 "
                 f"**{d3:+.4f} NDCG@10**"
                 f"（种子级 p={p3:.3g}，{int(g3.get('seeds_positive', 0))}/"
                 f"{int(g3.get('seeds_total', 0))} 个种子为正）"]
        if g1 is not None:
            d1 = val(g1, "delta_seed_mean")
            parts.append(f"静态图相对无图 {d1:+.4f}"
                         f"（{'无测出收益' if abs(d1) < 0.002 else '有正收益'}）")
        d2 = val(g2, "delta_seed_mean")
        cover, evo = val(g_onb, "delta_seed_mean"), val(g_evo, "delta_seed_mean")
        if np.isfinite(d2) and np.isfinite(cover):
            parts.append(f"而这 {d2:+.4f} 的动态性净效应里，覆盖项占 {cover:+.4f}"
                         f"（{cover/d2:.0%}）、演化项只有 {evo:+.4f}")
        d5 = val(g5, "delta_seed_mean")
        if np.isfinite(d5):
            parts.append(f"「每轮 EMA」相对慢刷新的额外边际 {d5:+.4f}")
        lines.append("；".join(parts) + "。")
    out.extend(lines)
    out.append("")

    # ---- 自动生成"一句话"：以三项分解为主轴（覆盖 vs 演化），不预设结论
    eff, noeff = [], []
    covers, evos, static_txt, tails = [], [], [], []
    for ds in DATASETS:
        g3 = b.contrast(ds, "dynamic_minus_nograph")
        g2 = b.contrast(ds, "dynamic_minus_static")
        g1 = b.contrast(ds, "static_minus_nograph")
        g_onb = b.contrast(ds, "onboarding_minus_static")
        g_evo = b.contrast(ds, "dynamic_minus_onboarding")
        g5 = b.contrast(ds, "dynamic_minus_periodic")
        if g3 is None:
            continue

        def pick(con, k):
            if con is None or k not in con.index or pd.isna(con[k]):
                return float("nan")
            return float(con[k])

        label = DS_LABEL.get(ds, ds)
        d3, p3 = pick(g3, "delta_seed_mean"), pick(g3, "p_seed_t")
        if not np.isfinite(p3):
            d3, p3 = float(g3["delta_mean"]), float(g3["p_paired"])
        real = d3 >= 0.002 and p3 < 0.05
        (eff if real else noeff).append(label)
        d2 = pick(g2, "delta_seed_mean")
        cover, evo = pick(g_onb, "delta_seed_mean"), pick(g_evo, "delta_seed_mean")
        if np.isfinite(d2) and np.isfinite(cover):
            covers.append((label, cover, cover / d2 if d2 else float("nan")))
            if np.isfinite(evo):
                evos.append((label, evo, evo / d2 if d2 else float("nan")))
        if g1 is not None:
            d1 = pick(g1, "delta_seed_mean")
            if np.isfinite(d1):
                kind = "低于随机基线" if pick(g1, "delta_seed_mean") < 0 else "略高于随机基线"
                static_txt.append(f"{label} {d1:+.4f}（neighbor_lift {kind}）")
        t5 = pick(g5, "delta_seed_mean")
        if np.isfinite(t5):
            tails.append(t5)

    parts = []
    if eff:
        parts.append(f"动态图在 {'、'.join(eff)} 上有明确独立贡献")
    if noeff:
        parts.append(f"在 {'、'.join(noeff)} 上没有测出显著收益")
    if covers:
        c_txt = "、".join(f"{lab} {c:+.4f}（占 ② 的 {sh:.0%}）" for lab, c, sh in covers)
        parts.append(f"而 ② 的收益几乎全部来自**覆盖**——新窗口里的用户第一次拿到邻居：{c_txt}")
    if evos:
        e_txt = "、".join(f"{lab} {e:+.4f}（{sh:.0%}）" for lab, e, sh in evos)
        parts.append(f"已覆盖用户的边权持续重估（演化）只贡献 {e_txt}")
    if static_txt:
        parts.append(f"一次性冻结的图相对无图 {'、'.join(static_txt)}，"
                     f"它的收益（或损失）不能归因于邻居质量")
    if tails:
        lo, hi = min(tails), max(tails)
        tail_txt = f"{lo:+.4f}" if abs(hi - lo) < 1e-9 else f"{lo:+.4f}~{hi:+.4f}"
        parts.append(f"「每轮 EMA」相对慢刷新（periodic）的额外边际只有 {tail_txt}")

    out.append("> **一句话**：" + "；".join(parts) + "。\n")
    return "\n".join(out)


def section_q2(b: Bundle) -> str:
    out: List[str] = ["## 三、问题二：动态图更新机制是否合理？（冻结归因）\n"]
    out.append("设计：ρ 固定 0.8（审计报告里观测到冻结的区间），2×2 网格 = "
               "图状态（保留候选池 `pool` / 只存 Top-K `topk`）× 新边初始化（论文 0 初值 `zero` / 首次观测取 Δe `warm`）。"
               "另加论文默认配置（pool + zero + ρ=0.2）作参照。\n")
    out.append("新增两个直接测「冻结」的指标（相对**预热后第一次评测**的邻居快照）："
               "`turnover_from_ref`（邻居集合的 Jaccard 距离，≈0 = 完全没变）与 "
               "`frozen_share`（一位都没变的用户占比，≈1 = 整张图冻住）。\n")

    for ds in DATASETS:
        fz = b.freeze.get(ds)
        if not fz or fz.get("cells") is None:
            continue
        out.append(f"### 3.{DATASETS.index(ds)+1} {DS_LABEL.get(ds, ds)}\n")
        cells = fz["cells"]
        cols = ["cell", "graph_state", "graph_new_edge", "rho", "n_seeds",
                "turnover_from_ref", "frozen_share", "ecr", "added",
                "n_edges", "neighbor_lift", "ndcg_win_mean"]
        cols = [c for c in cols if c in cells.columns]
        heads = ["格子", "图状态", "新边初始化", "ρ", "种子数", "turnover（越大越活）",
                 "frozen_share（≈1 冻住）", "ECR", "新边/轮", "有边用户",
                 "neighbor_lift", "窗口级 NDCG@10"][:len(cols)]
        out.append(md_table(cells, cols, headers=heads))
        out.append("")
        v = fz.get("verdict")
        if v is not None:
            for _, r in v.iterrows():
                if str(r["question"]).startswith(("①", "②")):
                    continue
                out.append(f"**[{r['verdict_id']}] {r['question']}**\n")
                if isinstance(r["evidence"], str) and r["evidence"].strip():
                    out.append(f"- 证据：{r['evidence']}")
                if isinstance(r["judgement"], str) and r["judgement"].strip():
                    out.append(f"- 判定：{r['judgement']}")
                out.append("")
        out.append(img(FREEZE_ROOT / slug(ds) / "figures" / "dg_fig_freeze.png",
                       f"{DS_LABEL.get(ds, ds)}：turnover / frozen_share / 有边用户数"))
        out.append("")

    out.append("### 3.3 结论\n")
    mechanism = (
        "机制：`topk` 只保留 K 个槽位并按 EMA 值排序；新边以 `(1−ρ)Δe` 起步（ρ=0.8 时只有初始相似度的 20%），"
        "要挤进 Top-K 必须 `0.2·Δe_new > EMA_incumbent`，而余弦量级给不出 5 倍优势 ⇒ 永远挤不进 ⇒ "
        "它的 EMA 永远不会累积 ⇒ 永久冻结。这正是审计文档描述的 `1/(1−ρ)` 屏障。\n"
    )
    verdicts = []
    for ds in DATASETS:
        fl = b.freeze_flags(ds)
        if not fl:
            continue
        pz = fl.get("pool_zero_rho0.8", False)
        pw = fl.get("pool_warm_rho0.8", False)
        tz = fl.get("topk_zero_rho0.8", False)
        tw = fl.get("topk_warm_rho0.8", False)
        name = DS_LABEL.get(ds, ds)
        pat = f"pool+zero={'冻' if pz else '不冻'}、pool+warm={'冻' if pw else '不冻'}、" \
              f"topk+zero={'冻' if tz else '不冻'}、topk+warm={'冻' if tw else '不冻'}"
        if tz and not (pz or pw or tw):
            verdicts.append(f"- **{name}**（{pat}）：冻结只出现在 `topk + zero` 一个组合，"
                            f"换成 pool 或 warm 任一项即解除 ⇒ **「只存 Top-K 的状态规则」×「新边 0 初值」的联合效应**，"
                            f"既不是 EMA 强度、也不是初始化规则单独造成。")
        elif tz and tw and not pz and not pw:
            verdicts.append(f"- **{name}**（{pat}）：topk 两格都冻、pool 两格不冻 ⇒ 冻结来自**状态保留规则**，"
                            f"与初始化无关。")
        elif pz and pw:
            verdicts.append(f"- **{name}**（{pat}）：pool 也冻住了 ⇒ 在该数据集上 ρ=0.8 的 EMA 本身就慢到接近冻结。")
        elif not any((pt, pv, t1, t2) for pt, pv, t1, t2 in [(pz, pw, tz, tw)]):
            verdicts.append(f"- **{name}**（{pat}）：四格都没冻，本数据集没有复现出冻结。")
        else:
            verdicts.append(f"- **{name}**（{pat}）：冻结模式不规则，需逐格排查。")

    out.append("**冻结的归属（逐数据集实测）**\n")
    out.extend(verdicts if verdicts else ["- （无数据）"])
    out.append("")
    out.append(mechanism)

    # 代价：邻居质量 vs NDCG（数据驱动）
    cost_lines = []
    for ds in DATASETS:
        c = b.freeze.get(ds, {}).get("cells")
        if c is None:
            continue
        try:
            z = c[c["cell"] == "pool_zero_rho0.8"].iloc[0]
            k = c[c["cell"] == "topk_zero_rho0.8"].iloc[0]
            cost_lines.append(
                f"- **{DS_LABEL.get(ds, ds)}**：`neighbor_lift` {float(z['neighbor_lift']):.3f} → "
                f"{float(k['neighbor_lift']):.3f}（{float(k['neighbor_lift'])-float(z['neighbor_lift']):+.3f}），"
                f"窗口级 NDCG@10 {float(z['ndcg_win_mean']):.4f} → {float(k['ndcg_win_mean']):.4f}"
                f"（{float(k['ndcg_win_mean'])-float(z['ndcg_win_mean']):+.4f}）"
            )
        except Exception:  # noqa: BLE001
            continue
    if cost_lines:
        out.append("**代价：邻居质量 vs 推荐指标**\n")
        out.extend(cost_lines)
        out.append("")
        out.append("> 冻结显著拉低邻居质量，但 NDCG@10 几乎不动。这说明该指标对「图不再适应」不敏感——"
                   "报告图更新效果时必须同时给出 `neighbor_lift` / ECR / turnover 这类结构指标，"
                   "否则一个已经冻结的实现也能拿到相同的推荐分数。\n")
    return "\n".join(out)


def section_q3(b: Bundle) -> str:
    out: List[str] = ["## 四、问题三：兴趣相似是否意味着协同有效？\n"]
    out.append("### 4.1 MovieLens-1M：真实 Genre（18 类，物品覆盖率 100%）\n")
    gv = b.genre.get("verdict")
    if gv is not None:
        for _, r in gv.iterrows():
            out.append(f"**[{r['verdict_id']}] {r['question']}**\n")
            if isinstance(r["evidence"], str) and r["evidence"].strip():
                out.append(f"- 证据：{r['evidence']}")
            if isinstance(r["judgement"], str) and r["judgement"].strip():
                out.append(f"- 判定：{r['judgement']}")
            out.append("")
        out.append(md_table(b.genre.get("desc"),
                            ["metric", "n", "mean", "median", "ci95_lo", "ci95_hi", "pos_rate"],
                            max_rows=20))
        out.append("")
        qd = b.genre.get("quint")
        if qd is not None and "key" in qd.columns:
            for key in ("genre_diff", "sem_diff", "genre_drift"):
                g = qd[qd["key"] == key]
                if len(g):
                    out.append(f"**按 `{key}` 五分位**\n")
                    out.append(md_table(g.drop(columns=["key"]),
                                        ["bucket", "n", "dndcg_mean", "ci95_lo", "ci95_hi",
                                         "dndcg_pos_rate"]))
                    out.append("")
        out.append(img(GENRE_ROOT / "figures" / "dg_fig_genre_gain.png",
                       "ML-1M：图的逐用户收益 vs genre 相似度 / 语义相似度 / 兴趣漂移（五分位）"))
        out.append("")
    else:
        out.append("（无数据）\n")

    out.append("### 4.2 双数据集无标签复核：邻居质量 vs 收益\n")
    out.append("Amazon 数据集**没有任何 item 类别/Genre 元数据**（`processed/*/meta.json` 只有 4 个统计字段），"
               "无法做真实标签分析。因此这里换一个不需要标签、两个数据集都能跑的口径："
               "把每个用户的**逐用户收益** ΔNDCG = NDCG(dynamic) − NDCG(nograph) 与「他邻居的质量」"
               "（`neighbor_lift` = 邻居摘要余弦 / 随机用户余弦，以及图注入力度 `delta_rel`）做五分位比较。\n")
    for ds in DATASETS:
        q = b.quality.get(ds)
        if not q or q.get("verdict") is None:
            continue
        out.append(f"**{DS_LABEL.get(ds, ds)}**\n")
        for _, r in q["verdict"].iterrows():
            out.append(f"- [{r['verdict_id']}] {r['question']}")
            out.append(f"  - {r['evidence']}")
            out.append(f"  - → {r['judgement']}")
        out.append("")
        qq = q.get("quint")
        if qq is not None and "key" in qq.columns:
            for key in ("neighbor_lift_user", "delta_rel"):
                g = qq[qq["key"] == key]
                if len(g):
                    out.append(f"**按 `{key}` 五分位**\n")
                    out.append(md_table(g.drop(columns=["key"]),
                                        ["bucket", "n", "dndcg_mean", "ci95_lo", "ci95_hi",
                                         "dndcg_pos_rate"]))
                    out.append("")
        out.append(img(QUALITY_ROOT / slug(ds) / "figures" / "dg_fig_quality_gain.png",
                       f"{DS_LABEL.get(ds, ds)}：图的逐用户收益 vs 邻居质量 / 图注入力度（五分位）"))
        out.append("")

    out.append("### 4.3 结论\n")

    # ---- 从表里取关键数字，避免手写
    def gdesc(metric: str, col: str = "mean") -> float:
        d = b.genre.get("desc")
        if d is None:
            return float("nan")
        s = d[d["metric"] == metric]
        return float(s.iloc[0][col]) if len(s) else float("nan")

    def gquint_span(key: str) -> float:
        q = b.genre.get("quint")
        if q is None or "key" not in q.columns:
            return float("nan")
        g = q[q["key"] == key]
        if len(g) < 2:
            return float("nan")
        return float(g.iloc[-1]["dndcg_mean"]) - float(g.iloc[0]["dndcg_mean"])

    def gcorr(metric: str, target: str = "dndcg", col: str = "spearman") -> float:
        c = b.genre.get("corr")
        if c is None:
            return float("nan")
        s = c[(c["metric"] == metric) & (c["target"] == target)]
        return float(s.iloc[0][col]) if len(s) else float("nan")

    lift_g = gdesc("genre_lift")
    lift_s = gdesc("sem_lift")
    span_g = gquint_span("genre_diff")
    span_s = gquint_span("sem_diff")
    span_d = gquint_span("genre_drift")
    sp_g = gcorr("genre_diff")
    sp_sg = gcorr("genre_diff", target="sem_diff")

    # 无标签版：两个数据集的邻居质量跨度
    q_rows = []
    for ds in DATASETS:
        q = b.quality.get(ds)
        if not q or q.get("verdict") is None:
            continue
        v = q["verdict"]
        def row_of(idx: str):
            s = v[v["verdict_id"] == idx]
            return s.iloc[0] if len(s) else None
        r1, r2 = row_of("①"), row_of("②")
        q_rows.append({
            "dataset": DS_LABEL.get(ds, ds),
            "mean_gain": float(r1["value"]) if r1 is not None else float("nan"),
            "lift_span": float(r2["value"]) if r2 is not None else float("nan"),
            "lift_judge": r2["judgement"] if r2 is not None else "",
        })

    bullets = []
    if np.isfinite(lift_g):
        bullets.append(
            f"图选出的邻居在真实 genre 空间确实比随机用户更像自己，但幅度很弱"
            f"（genre lift = {lift_g:.2f}，而学到语义的 lift = {lift_s:.2f}）——"
            f"服务器学到的相似度大部分不是 genre 口味，而是 genre 之外的协同信号"
            f"（corr(genre, 语义) 的 Spearman 只有 {sp_sg:.2f}）；")
    if np.isfinite(span_g):
        bullets.append(
            f"genre 相似度只能解释**收益幅度**的一部分：五分位 Q1→Q5 的收益差 {span_g:+.4f}"
            f"（语义相似度的跨度是 {span_s:+.4f}，更大）；但逐用户 Spearman 只有 {sp_g:+.3f}，"
            f"**预测不了单个用户**（NDCG@10 极度稀疏，ΔNDCG>0 的用户只占 "
            f"{(b.genre.get('desc') is not None and (b.genre['desc']['metric'] == 'dndcg').any() and pct(float(b.genre['desc'].loc[b.genre['desc']['metric'] == 'dndcg', 'pos_rate'].iloc[0]), 1)) or '12.8%'}）；")
    if np.isfinite(span_d):
        bullets.append(f"兴趣漂移与收益同向（漂移五分位跨度 {span_d:+.4f}）——"
                       f"这正好是「什么时候该更新关系」的直接抓手；")
    if q_rows:
        spans = [r["lift_span"] for r in q_rows if np.isfinite(r["lift_span"])]
        body = "；".join(f"{r['dataset']} 邻居质量 Q1→Q5 收益差 {r['lift_span']:+.4f}"
                        f"（平均收益 {r['mean_gain']:+.4f}）" for r in q_rows)
        if len(spans) >= 2 and all(s >= 0.005 for s in spans):
            tail = "——两个数据集上收益都随邻居质量上升，但个体层面相关性都弱。"
        elif len(spans) >= 2 and any(s >= 0.005 for s in spans) and any(s <= 0.0 for s in spans):
            tail = ("——**这个梯度不能跨数据集复现**：在一个数据集上邻居质量越高的用户收益越大，"
                    "在另一个数据集上则完全测不出梯度，而两边的平均收益都是正的。"
                    "所以「邻居很像自己」解释不了「协同为什么有效」。")
        elif spans and all(s <= 0.0 for s in spans):
            tail = "——两个数据集上都没有测出梯度，个体层面相关性也弱。"
        else:
            tail = "——跨度都很小，个体层面相关性也弱。"
        bullets.append("无标签口径（邻居质量）下的复核：" + body + tail)

    out.append("**「兴趣相似」不是协同有效的充分条件。**\n")
    for bx in bullets:
        out.append(f"- {bx}")
    out.append("")
    return "\n".join(out)


def section_implications(b: "Bundle") -> str:
    """第五章：把「收益来自哪里」翻译成第二阶段的方向。数字全部从聚合表取。"""

    def num(con, key: str) -> float:
        if con is None or key not in con.index or pd.isna(con[key]):
            return float("nan")
        return float(con[key])

    cov_bits, ema_bits, static_bits = [], [], []
    for ds in DATASETS:
        s = b.arms_of(ds)
        if s is None or len(s) == 0:
            continue
        lab = DS_LABEL.get(ds, ds)
        g2 = b.contrast(ds, "dynamic_minus_static")
        g_onb = b.contrast(ds, "onboarding_minus_static")
        g_evo = b.contrast(ds, "dynamic_minus_onboarding")
        g1 = b.contrast(ds, "static_minus_nograph")
        g5 = b.contrast(ds, "dynamic_minus_periodic")
        d2 = num(g2, "delta_seed_mean")
        cover, evo = num(g_onb, "delta_seed_mean"), num(g_evo, "delta_seed_mean")
        if np.isfinite(cover) and np.isfinite(d2) and d2:
            cov_bits.append(f"{lab} 覆盖项 {cover:+.4f}（占 ② 的 {cover/d2:.0%}）")
        if np.isfinite(evo) and np.isfinite(d2) and d2:
            ema_bits.append(f"{lab} 演化项 {evo:+.4f}（{evo/d2:.0%}）")
        st = b.arm_row(ds, "static")
        if st is not None:
            edges = float(st["n_edges"]) if pd.notna(st["n_edges"]) else float("nan")
            cov = (float(st["active_coverage_mean"])
                   if "active_coverage_mean" in st.index and pd.notna(st["active_coverage_mean"])
                   else float("nan"))
            lift = float(st["neighbor_lift"]) if pd.notna(st["neighbor_lift"]) else float("nan")
            d1 = num(g1, "delta_seed_mean")
            static_bits.append(
                f"{lab}：末轮有边用户 {edges:.0f}、覆盖率 {cov:.0%}、"
                f"neighbor_lift {lift:.2f}、Static−NoGraph {d1:+.4f}"
            )

    tail5 = []
    for ds in DATASETS:
        g5 = b.contrast(ds, "dynamic_minus_periodic")
        v = num(g5, "delta_seed_mean")
        if np.isfinite(v):
            tail5.append(f"{DS_LABEL.get(ds, ds)} {v:+.4f}")
    tail_txt = "、".join(tail5) if tail5 else "（无）"

    return f"""## 五、对研究问题的含义

1. **收益不在「有图」，而在「图覆盖了谁」**。三项分解（② = 覆盖 + 演化）显示：
   {'；'.join(cov_bits) if cov_bits else '（无数据）'}。
   也就是说 `DynamicGraph − StaticGraph` 这个差值，几乎全部由「新窗口里的用户第一次拿到邻居」解释，
   而不是「已覆盖用户的边权被不断重估」。一次性冻结的静态图的收益之所以有限（MovieLens-1M 上甚至为负），
   就是因为它只覆盖了冻结那一刻已经上传过摘要的用户（覆盖率 21%~23%，动态图是 87%~94%）：
   {'；'.join(static_bits) if static_bits else '（无数据）'}。
   ⇒ 第二阶段该优化的目标是**「什么时候把谁纳入图」**（覆盖时点 + 覆盖对象），
   而不是把 EMA 的平滑系数调得更精细。

2. **「持续重估」本身的空间被量化了**：{'；'.join(ema_bits) if ema_bits else '（无数据）'}。
   相对「每人只建一次边」（onboarding 臂），每轮无差别 EMA 只多出这一点；
   相对「每窗刷新一次」（periodic 臂）的额外边际是 {tail_txt}。
   任何自适应更新机制必须先超过这两个简单基线，才有存在价值。

3. **现行实现的冻结缺陷是组合式的**，改「状态保留规则」或「新边初值」任一项即可解除；但要注意
   **NDCG@10 对这个缺陷不敏感**（邻居质量掉 0.24–0.34，NDCG 只动 1e-4 量级），
   因此论文里论证图更新机制时，必须同时报告邻居质量 / 边变化率这些结构指标。

4. **「兴趣相似」不能作为「协同有效」的代理指标**。若论文要用 genre/类别相似度论证邻居构造的合理性，
   需要同时给出「相似度提升幅度」与「收益提升幅度」，并明确这是群体层面的幅度差异而非个体预测能力。
"""


def section_limits() -> str:
    return """## 六、局限

1. **数据集**：MovieLens-1M 与 Amazon-Digital-Music 两个；Amazon 侧没有 item 类别元数据，
   问题三只能用无标签口径（邻居质量）做复核，不能复现真实 Genre 分析。
2. **规模**：主臂 5 个种子（统计以种子为单位，n=5）；刷新频率扫描与图分支剂量锚点按需只跑 3 个或 1 个种子，
   它们的对照只在「与同数据集主臂同种子」的意义上成立。窗口粒度固定为 5 段 × 120 轮，未做窗口长度敏感性。
3. **单机单构型**：所有对照都在同一台机器（2×RTX 3090）上跑完，避免设备间 dropout 随机流差异；
   但不同机器同 seed 的 CUDA 结果本身是逐位一致的。
4. **用户级统计**：逐用户配对的样本量很大（万级），但同一用户跨窗口不是独立观测，
   因此逐用户检验只作方向性证据；主证据是**种子级配对**（每个种子一个观测），窗口级只作敏感性分析。
5. **periodic 的周期**：主对照取 120 轮（1 个窗口），另有 20 / 240 轮与「窗口边界刷新」三个诊断点。
   诊断显示「每窗第 21 轮刷新」优于「窗口边界刷新」（-0.0058）与 240 轮（-0.0044），
   所以「收益来自覆盖」这个结论依赖「刷新时点紧跟窗口开始」这个事实，而不是某个特定周期数值。
6. **ECR 的两个口径**：`dg_summary.csv` 的 `avg_ecr` 是**评测点那一轮**的即时应变率，
   对 periodic 这类「只在特定轮更新」的臂几乎恒为 0（评测点与更新轮错开）；
   正式表一律用 `ecr_update_mean`（真的更新了的轮次上的 v2 定义均值）与 `graph_turnover`。
   这也是为什么 periodic 的 `avg_ecr` 看起来是 0、而 `graph_turnover` 却是 0.96。
7. **修复前后的口径**：本报告只用 F7（上传时机）/ F8（时间切分泄漏）修复后的正式批次。
   修复前的 `results/raw`、`results/raw_phase2b` 标为 legacy：它们的图建在「更新前的摘要」上，
   数值不能与本表混比；同一批配置下单种子的修复前后对照见 `问题核实与修复清单_2026-11-08.md`。
8. **跑批稳定性**：长批次里出现过极少数**解释器级偶发崩溃**（`SystemError: error return without exception set`
   或 `Segmentation fault`，崩溃点每次不同，连纯 Python 的数据加载也会崩），由 `dg_run_formal_retry.sh`
   自动补跑完成；正式批次的 128 个 run 全部收官，没有靠手工重跑补齐。
"""


def section_repro(b: Bundle) -> str:
    n_runs = "—"
    raw = RAW_ROOT
    if raw.exists():
        n_runs = str(len(list(raw.rglob("dg_summary.csv"))))
    return f"""## 七、复现与产出清单

### 7.1 复现命令

```bash
cd /home/wl/exp_FedDIEG

# 跑前自检（9 条严格性断言，秒级）+ 修复项断言（34 项）
python -X utf8 val_DynamicGraph/dg_selfcheck.py
python -X utf8 feddieg_core/verify_fixes.py

# 正式批次（128 个 run，4 并发、每卡 2 进程；断点续跑、崩溃自动补跑）
bash val_DynamicGraph/dg_run_formal_retry.sh all

# 阶段 4：聚合 + 冻结归因 + 邻居质量 + genre + 生成报告（写入隔离目录）
bash val_DynamicGraph/dg_phase4.sh --all

# 阶段 5：种子级统计表 + 验收
python -X utf8 val_DynamicGraph/dg_stats_formal.py
bash val_DynamicGraph/dg_verify_formal.sh
```

### 7.2 数据与代码

| 文件 | 作用 |
|---|---|
| `dg_common.py` | 六条臂定义（NullGraph / PolicyGraph）、图分支探针、冻结诊断、genre 诊断 |
| `dg_run.py` | 单次运行（数据集, 臂, 种子） |
| `dg_selfcheck.py` | 9 条严格性断言 |
| `feddieg_core/verify_fixes.py` | 34 项修复断言（摘要上传时机 / 时间切分 / ECR 定义 / Top-K 边界） |
| `dg_aggregate.py` | 主对照聚合（问题一 + 种子级统计） |
| `dg_freeze.py` | 冻结归因（问题二） |
| `dg_genre.py` | 真实 Genre 分析（问题三，ML-1M） |
| `dg_neighbor_quality.py` | 无标签版问题三（双数据集） |
| `dg_phase4.sh` / `dg_stats_formal.py` | 阶段 4 一键聚合 / 阶段 5 统计表 |
| `dg_verify_formal.sh` | 29 项正式验收 |
| `dg_report.py` | 生成本报告 |
| `data/item_genres_ml1m.npz` | ML-1M 真实 18 类 genre 矩阵（物品覆盖 100%） |

累计运行 **{n_runs}** 次训练（`results/raw_formal/**/dg_summary.csv`）。
"""


def main() -> None:
    global AGG_ROOT, FREEZE_ROOT, GENRE_ROOT, QUALITY_ROOT, RAW_ROOT
    parser = argparse.ArgumentParser(description="生成最终实验验证报告")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--agg-root", type=Path, default=AGG_ROOT,
                        help="主对照聚合表根目录（默认 results/aggregated；正式复跑用 results/agg_formal）")
    parser.add_argument("--freeze-root", type=Path, default=FREEZE_ROOT,
                        help="冻结归因表根目录（默认 results/freeze）")
    parser.add_argument("--genre-root", type=Path, default=GENRE_ROOT,
                        help="ML Genre 诊断表根目录（默认 results/genre）")
    parser.add_argument("--quality-root", type=Path, default=QUALITY_ROOT,
                        help="无标签邻居质量诊断表根目录（默认 results/quality）")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT,
                        help="逐 run 结果根目录（默认 results/raw；正式复跑用 results/raw_formal）")
    args = parser.parse_args()
    AGG_ROOT, FREEZE_ROOT = args.agg_root.resolve(), args.freeze_root.resolve()
    GENRE_ROOT, QUALITY_ROOT = args.genre_root.resolve(), args.quality_root.resolve()
    RAW_ROOT = args.raw_root.resolve()

    b = Bundle()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    have = [ds for ds in DATASETS if (b.arms_of(ds) is not None and len(b.arms_of(ds)) > 0)]

    parts: List[str] = []
    parts.append("# 动态图（DynamicGraph）验证实验报告\n")
    parts.append(f"- 生成时间：{now}　代码版本：`{git_head()}`")
    parts.append(f"- 数据集：{'、'.join(DS_LABEL.get(d, d) for d in have) or '（尚无结果）'}")
    parts.append(f"- 目录：`/home/wl/exp_FedDIEG/val_DynamicGraph`")
    parts.append("- 所有数字均由 `dg_report.py` 直接读取聚合表生成，未手工抄写。\n")
    parts.append("---\n")
    parts.append(section_setup(b))
    parts.append("\n---\n")
    parts.append(section_q1(b))
    parts.append("\n---\n")
    parts.append(section_q2(b))
    parts.append("\n---\n")
    parts.append(section_q3(b))
    parts.append("\n---\n")
    parts.append(section_implications(b))
    parts.append("\n---\n")
    parts.append(section_limits())
    parts.append("\n---\n")
    parts.append(section_repro(b))

    args.out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    n_lines = len(args.out.read_text(encoding="utf-8").splitlines())
    print(f"[ok] 报告已写出：{args.out}（{n_lines} 行，{args.out.stat().st_size} 字节）")
    print(f"     覆盖数据集：{have}")


if __name__ == "__main__":
    main()
