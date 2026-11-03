# 验证实验 2（val02）：兴趣真实变化时，固定 EMA 是否适应太慢？

- 目录：`exp_val_1026/exp02_ema_interest_drift/`
- 对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验二」
- 只做诊断，不设计算法。**交付物是一张图**：
  `results/aggregated/figures/val02_fig_ema_recovery.png`
  （横轴 = 兴趣变化后经过的轮数，每条线 = 一个 EMA 系数 ρ，纵轴 = 平均 NDCG@10；
  实线 = 兴趣变化组，虚线 = 活动量匹配的稳定对照组）

## 1. 这一版做了什么

| | 内容 |
|---|---|
| 自变量 | 边权 EMA 系数 ρ（任务书要求）：`0` / `0.5` / `0.8` / `0.95` |
| 控制变量 | 同一份窗口数据、同一份客户端参与序列、同一批用户（cohort）、同一个随机种子 → 4 条曲线是**同一批用户的配对比较** |
| 参与率 | 100%（边更新不再受"谁参与"影响，EMA 是唯一的滞后来源；陈旧性属于实验一） |
| 数据集 | **默认两个**：MovieLens-1M + `amazon-digital music`（`-Datasets` 可改） |
| 种子 | 42 / 777 / 2024（3 个独立随机种子） |
| 规模 | 4 个 ρ × 3 种子 × **2 个数据集 = 24 次运行**，每次 5 窗口 × 120 轮 |
| 产出 | 每次运行 4 个文件；聚合出 **2 张图 + 5 张表 + 报告骨架**，一次命令全自动 |
| 用时 | MovieLens-1M 约 3 分钟/次、Amazon 约 7 分钟/次 → 全量约 **2 小时**（paper 图强度） |

## 2. 三步流程（已封装成一条命令）

```
① val02_detect_drift.py   离线识别"兴趣真实变化"的用户 + 构造类别视图  → results/drift/
② val02_run_ema.py        对每个 ρ 训练一遍，逐轮记录 NDCG 与边级指标 → results/raw/
③ val02_aggregate.py      对齐 rel_round、出恢复曲线 + 滞后表 + 结论   → results/aggregated/
```

`val02_run_basic.ps1 / .sh` 会把 ①②③ 串起来：**跑一条命令就得到全部结果**。
第①步不训练模型（两个数据集共约 1 分钟）；第②步是主要成本；第③步几秒。
单步也可以手动跑（见 §5.2）。

## 3. 与任务书「实验二」的符合性核对

| 任务书要求 | 本实现 | 状态 |
|---|---|---|
| 采用现有时间窗口实验，寻找真实存在明显偏好变化的用户 | 5 个等长交互窗口（与 `load_rolling_windows` 完全同口径），按窗口边界做离线漂移识别 | ✅ |
| 用电影类别 + 不同时间窗口的交互分布识别兴趣变化 | 有 `movies.dat` 时用真实 genre（`--genres-file`）；没有时用第 1 个窗口的共现行为构造 18 个伪类别 | ✅（见 §9 局限 1） |
| 类别统计只用于离线诊断，不能提前看到未来窗口 | 训练/构图完全不使用类别信息；类别只用于 ①选 cohort ②事后统计 old/new 同伴占比 | ✅ |
| 比较 ρ = 0 / 0.5 / 0.8 / 0.95 | 同（`--rho`） | ✅ |
| 观察① 兴趣变化后不同设置的性能如何下降和恢复 | `val02_fig_ema_recovery.png` + `val02_recovery_curve.csv` | ✅ |
| 观察② 边关系需要经过多少轮才能完成明显调整 | `turnover_from_pre`（邻居集合 vs 变化前快照的 1−Jaccard）→ `val02_fig_edge_adjust.png` 右图 + `lag_table` 的 `turnover_*` | ✅ |
| 观察③ EMA 越强是否一定恢复越慢 | `val02_verdict.csv` 第②行（对 ρ 的单调性检验） | ✅ |
| 观察④ 哪个系数能取得较好的推荐准确率 | `lag_table` 的 `ndcg_pre_level` / `ndcg_post_late` + verdict 第③④行 | ✅ |
| 预期交付：兴趣变化前后不同 EMA 系数的 NDCG@10 恢复曲线 | `val02_fig_ema_recovery.png` | ✅ |
| 3 个独立随机种子 | 同 | ✅ |
| 先在一个数据集上验证，暂不要求四个 | 默认跑 MovieLens-1M + 一个 Amazon 数据集两个；稀疏数据集的 cohort 会自动放宽阈值 | ✅ |

## 4. 文件清单

```
exp_val_1026/exp02_ema_interest_drift/
├─ val02_detect_drift.py       ① 漂移识别 + 类别视图（不训练，含自动放宽阶梯）
├─ val02_run_ema.py            ② 固定 ρ 的一次完整训练
├─ val02_aggregate.py          ③ 恢复曲线 + 滞后表 + 自动结论（支持多数据集）
├─ val02_relation_quality.py   ④ 关系质量：邻居兴趣一致度 vs 随机基线（只用已有结果，不重训）
├─ val02_run_basic.ps1 / .sh   一键跑完：①②③④ 全自动（默认两个数据集，约 2 小时）
├─ val02_run_pilot.ps1 / .sh   冒烟：两个数据集 × 2 个 ρ × 40 轮/窗口（约 8 分钟）
├─ README_val02_ema_drift.md   本文件
└─ results/
   ├─ drift/<dataset>/         cohort 清单、类别计数张量、逐用户画像（已预生成，可直接用）
   ├─ raw/<dataset>/rho<ρ>/seed_<s>/     paper 图强度（默认）
   ├─ raw_strong/...           strong 图强度（敏感性对照）
   ├─ raw_warm/...             新边 warm 初始化（机制对照）
   ├─ aggregated/              主结果：figures/ + tables/ + val02_report.md + val02_findings.md
   └─ archive/                 pilot 试跑
```

## 5. 怎么跑

### 5.1 一条命令跑完（推荐）

```powershell
# 先设好 Python（用你跑实验那个环境）
$env:PYTHON = "D:\Myenvs\FedDIEG\python.exe"

# 冒烟（可选，约 8 分钟）：两个数据集 × 2 个 ρ × 40 轮，结果写 results/archive/
powershell -NoProfile -ExecutionPolicy Bypass -File .\exp_val_1026\exp02_ema_interest_drift\val02_run_pilot.ps1

# 正式：两个数据集 × 4 个 ρ × 3 个种子 = 24 次运行，末尾自动出图（约 2 小时）
powershell -NoProfile -ExecutionPolicy Bypass -File .\exp_val_1026\exp02_ema_interest_drift\val02_run_basic.ps1
```

Linux / WSL：

```bash
bash exp_val_1026/exp02_ema_interest_drift/val02_run_pilot.sh
bash exp_val_1026/exp02_ema_interest_drift/val02_run_basic.sh
```

跑完直接看 `results/aggregated/`：图 + 表 + `val02_report.md` 都有了，不需要再手动执行聚合。

脚本行为（都是为了"一条命令跑完"）：

- **已完成的任务自动跳过**：中断后重跑同一条命令即可续跑。
- **单个任务失败不中断整轮**：失败的组合会在最后汇总打印出来。
- **不可用的数据集自动跳过**：若某个数据集的 drift cohort 少于 10 人，只警告并跳过它，其它数据集照常跑完。
- **末尾自动聚合**：paper 与 strong 两组分别聚合到 `results/aggregated/` 与 `results/aggregated_strong/`。
- 想改网格就加参数，例如 `-Seeds "42" -Rounds 40`、`-Datasets "movielens-1m"`、`-NoAggregate`。

### 5.2 拆开手动跑（调试用）

```powershell
# ① 漂移识别（两个数据集各一次；cohort 已预生成时自动跳过）
python exp_val_1026\exp02_ema_interest_drift\val02_detect_drift.py --dataset "movielens-1m"
python exp_val_1026\exp02_ema_interest_drift\val02_detect_drift.py --dataset "amazon-digital music"

# ② 单个 (数据集, ρ, 种子)
python exp_val_1026\exp02_ema_interest_drift\val02_run_ema.py --dataset "movielens-1m" --rho 0.0  --seed 42
python exp_val_1026\exp02_ema_interest_drift\val02_run_ema.py --dataset "movielens-1m" --rho 0.95 --seed 42

# ③ 聚合出图
python exp_val_1026\exp02_ema_interest_drift\val02_aggregate.py
```

## 6. 三个必须知道的实现选择（会直接决定你看到什么）

### 6.1 `min_sim` 阈值作用在原始余弦上（`--graph-gate raw`，默认）

原实现是「先把边权归一化到和为 1 → 与 ρ·历史混合 → 再和 `min_sim` 比」。
归一化后的边权典型值只有 0.1 量级，与 ρ 混合后必然低于阈值 0.2，
于是 **ρ≥0.9 时图会整体坍塌**（不是"适应慢"，是"没有边"），四个 ρ 之间根本不可比。
改成"阈值只看有没有证据（原始余弦），EMA 只决定历史权重的记忆长度"之后，ρ 才是干净的自变量。
`--graph-gate blended` 保留原实现，供核对。

### 6.2 新边的初值（`--new-edge-init`，默认 `ema`）

| 取值 | 新边的初值 | 效果 |
|---|---|---|
| `ema`（默认） | `(1−ρ)·sim` | **忠实复刻原实现**：等价于"这条边以前权重为 0，现在开始做 EMA"。ρ 越大，新邻居越难挤进 Top-K |
| `warm` | `sim` | 新边不被惩罚，EMA 只让旧边衰减变慢 → 滞后机制被消掉大半（对照组） |
| `zero` | `0` | 最极端的 EMA 语义 |

第一次建立邻居的用户（还没有任何历史状态）在三种设置下都以当前余弦起步，
否则 Top-K 全为 0、选择会退化成"按下标取前 k 个"。

### 6.3 图影响强度（`--graph-strength`，很重要）

已发表配置下图消息对打分的影响被压得很小（`graph_mix=0.2` + `tail_power=4` +
`confidence/agreement` 两道门控 + `score_local_mix=0.2` 回拉局部特征），
所以两个预设：

| 预设 | graph_mix | tail_power | confidence/agreement | score_local_mix | 用途 |
|---|---|---|---|---|---|
| `paper`（默认） | 0.20 | 4.0 | 0.5 / 0.5 | 0.2 | 与论文/验证实验一一致，主表 |
| `strong` | 0.50 | 1.0 | 0 / 0 | 0.0 | 让图真正影响打分，检验"是不是图权重太小才看不出差异" |

```powershell
# strong 组写到 results/raw_strong（不覆盖 paper），末尾会自动聚合到 results/aggregated_strong
powershell ... -File .\val02_run_basic.ps1 -GraphStrength strong

# 两组一次跑完（约 4 小时；已完成的会跳过）
powershell ... -File .\val02_run_basic.ps1 -GraphStrength all
```

### 6.4 pilot 实测参考值（40 轮/窗口、单种子、paper 强度，**只作量级参考**）

| ρ | 平均 ECR | 变化后第一个窗口末的 turnover | drift 组 NDCG@10 |
|---|---|---|---|
| 0.00 | 0.465 | 0.987 | 0.4759 |
| 0.30 | 0.006 | 0.842 | 0.4640 |
| 0.50 | 0.003 | 0.639 | 0.4707 |
| 0.80 | 0.0003 | 0.279 | 0.4638 |
| 0.95 | 0.0004 | 0.263 | 0.4695 |

能读出三件事，也决定了报告怎么写：

1. **turnover 随 ρ 单调下降，信号很干净**（0.99 → 0.84 → 0.64 → 0.28 → 0.26），
   这正是任务书观察②想要的量：ρ 越大，兴趣变化后图调整得越少。
2. **ECR 在 ρ≥0.3 时就掉到 ~0**：一旦 Top-K 建立，EMA 的"粘性"让旧邻居极难被替换，
   图在窗口内几乎冻结。这不是 bug，而是「固定 EMA + 硬 Top-K」这套机制本身的性质
   （原实现更极端：归一化边权 + 阈值让新边几乎永远进不来）。
3. **NDCG 在 paper 强度下几乎不动**（0.4638–0.4759，非单调，量级与噪声接近）。
   所以主图可能"四条线重合"——那就必须用 `strong` 那组再回答一次，并在报告里说明
   "现状配置把动态图的作用压得太低，NDCG 层面看不出 ρ 的代价"。

### 6.5 稀疏数据集：cohort 自动放宽阶梯

Amazon 类数据集每个窗口每用户只有约 2 条交互，用 MovieLens-1M 的阈值会直接筛出个位数用户
（实测 `amazon-digital music` 默认阈值只剩 8 人、对照组 0 人）。所以第①步带了一个
**自动放宽阶梯**（`--min-cohort` 默认 60，达不到就自动降一级）：

| 阶梯 | 边界两侧 | 前后各 | 分位 | 绝对下限 |
|---|---|---|---|---|
| 0（默认） | ≥3 条 | ≥20 条 | 25% | 0.15 |
| 1 | ≥2 条 | ≥14 条 | 25% | 0.12 |
| 2 | ≥2 条 | ≥10 条 | 30% | 0.10 |
| 3 | ≥2 条 | ≥6 条 | 33% | 0.05 |

实际用了哪一级会写进 `val02_cohort_meta.json` 的 `auto_relax_rung` / `chosen_params`，
也会在日志里打印「← 已自动放宽」。两个数据集的实际结果（已预生成在 `results/drift/`）：

| 数据集 | 采用阶梯 | drift 组 | stable 组 | 主类别确实改变 | onset 分布 |
|---|---|---|---|---|---|
| movielens-1m | 0（默认） | 146 | 102 | 133 | {2:21, 3:42, 4:30, 5:53} |
| amazon-digital music | 3（放宽） | 93 | 46 | 91 | {2:48, 3:15, 4:16, 5:14} |

> 写报告时必须说明：Amazon 上的漂移识别是在更宽松的阈值下得到的，
> `drift_excess` 的绝对量级也更小（MovieLens 中位 0.287 vs Amazon 0.221），
> 因此**跨数据集只比趋势，不比较绝对值**。如果某个数据集连阶梯 3 都凑不出 10 个人，
> 批次脚本会自动跳过它并在最后汇总里列出。

## 7. 输出与字段说明

### 7.1 第①步：`results/drift/<dataset>/`

| 文件 | 内容 |
|---|---|
| `val02_drift_users.csv` | ★ cohort 清单：`user, cohort(drift/stable), onset_window, drift_excess, tvd_pooled, null_tvd, cat_old, cat_new, primary_drift, n_inter_before/after/total` |
| `val02_counts.npz` | `counts[user, window, category]`：类别计数张量（训练脚本算 old/new 同伴占比用） |
| `val02_user_boundary_drift.csv` | 用户 × 边界：`tvd / null_tvd / drift_excess / sustained / is_best_onset` |
| `val02_drift_summary.csv` | 阈值敏感性：不同分位 / 绝对阈值下的 cohort 规模 |
| `val02_user_window_profile.csv` | 用户 × 窗口：交互数、主类别、熵、完整类别分布 |
| `val02_partition.csv` / `val02_category_meta.json` | 每个物品的类别、类别来源与覆盖率 |
| `val02_cohort_meta.json` | cohort 元信息（阈值、人数、onset 分布） |

### 7.2 第②步：`results/raw/<dataset>/rho<ρ>/seed_<s>/`

| 文件 | 内容 |
|---|---|
| `val02_user_round_metrics.csv` | ★ 逐用户逐评测轮：`rel_round, ndcg, hr, old_peer_share, new_peer_share, turnover_from_pre` —— 恢复曲线的全部原始数据 |
| `val02_round_metrics.csv` | 逐评测轮聚合：全体/drift/stable 的 NDCG、ECR、old/new 同伴占比、turnover |
| `val02_summary.csv` | 本次运行汇总（一行） |
| `config.json` / `val02_run.log` | 超参数与日志 |

`rel_round = 当前轮 − (onset_window − 1) × 每窗口轮数`，**0 = 兴趣变化前最后一轮**，
正数 = 变化之后。onset 不同的用户被对齐到同一条横轴上。

### 7.3 第③步：`results/aggregated/`

| 文件 | 内容 |
|---|---|
| `figures/val02_fig_ema_recovery.png` | ★ 任务书要的那张图（默认裁到 ±1 窗口，见下） |
| `figures/val02_fig_ema_delta.png` | ★ 相对 ρ=0 的 ΔNDCG@10：绝对曲线重合时，用它看 ρ 的代价 |
| `figures/val02_fig_relation_quality.png` | ★ 关系质量：邻居兴趣一致度 / 随机基线（变化前、变化后、按 onset 边界） |
| `figures/val02_fig_edge_adjust.png` | 左：邻居中旧兴趣同伴的边权占比；右：邻居 turnover |
| `tables/val02_recovery_curve.csv` | 图 A 的数据（含每点 `n_users`、用户级 SEM） |
| `tables/val02_edge_curve_*.csv` | 图 B 的数据 |
| `tables/val02_lag_table.csv` | 每个 ρ 的滞后指标 |
| `tables/val02_lag_by_seed.csv` | 同上但保留到种子级（配对比较用） |
| `tables/val02_relation_quality*.csv` | 关系质量明细 / 汇总 / 按边界（含 `clean` 与 `all` 两种口径） |
| `tables/val02_verdict.csv` | 自动结论判定 |
| `val02_report.md` | 已填好数字的报告骨架 |
| `val02_findings.md` | ★ 人工整理后的结论文档（推荐以它为准） |

**恢复曲线默认裁剪**：`rel_round` 混了不同窗口的测试集，全跨度上会在 ±120 处出现台阶，
把 ρ 之间的差异压住。所以默认裁到 `[-1 窗口, +1 窗口]`（即"同一批用户、相邻两个窗口"），
想看全跨度用 `--full-range`。若绝对曲线仍几乎重合，看 `val02_fig_ema_delta.png`。

**关系质量（lift）怎么读**：`lift = 邻居中同兴趣的边权占比 ÷ 随机邻居的期望占比`。
lift≈1 表示邻居与随机无异，>1 表示图确实把同兴趣的人放在一起。
注意两点：① 只能**同队列内跨 ρ** 比较，不能跨队列（类别基准率不同）；
② onset≥3 的用户存在"旧兴趣快照 vs 图建立时点"错位，可干净引用的只有 onset=2 子集与按边界拆分表。

`val02_lag_table.csv` 每个指标都有前缀（`ndcg_` / `old_peer_share_` / `new_peer_share_` / `turnover_`）：

| 列 | 含义 |
|---|---|
| `pre_level` | 变化前（rel_round ∈ [−60, 0]）的平均值 |
| `post_early` | 变化后早期（rel_round ∈ [1, 30]）的平均值 |
| `post_late` | 变化后后期（曲线末段）的平均值 → "最终恢复到什么水平" |
| `support_cap` | 右端截断位置：样本量掉到峰值 30% 的点之后不参与计算 |
| `gap_pre_to_early` | `pre_level − post_early`：兴趣变化的即时代价 |
| `lag_recover_25/50/90` | 从最差点恢复到"最终水平"的 25%/50%/90% 要多少轮 |
| `half_lag` | 从 rel_round=0 走到"最终水平一半"要多少轮（旧同伴占比用它最有意义） |

## 8. 关键设计决定（写报告要交代）

1. **EMA 作用在平滑后的相似度上**，权重最后在保留的 Top-K 内归一化给 GNN 用（见 §6.1、§6.2）。
2. **cohort 固定**：由 `--cohort-seed 2026` 固定，与 ρ、训练种子无关。
3. **漂移分数减掉统计噪声零点**：每个窗口每用户平均只有 30 条交互，
   18 个类别上"两次独立抽样"的 TVD 期望就有 0.19；不减零点会把"交互少"误判成"兴趣变了"。
   零点用**随机拆分（置换）零分布**：把两段交互混在一起按同样规模随机对半切，重复 10 次取均值。
4. **逐边界识别，不做"所有窗口都活跃"的筛选**：MovieLens-1M 只有 23 个用户 5 个窗口都活跃、
   1416 个用户跨 ≥2 个窗口（用户行为是爆发式的）。所以要求"当前边界两侧各 ≥3 条交互"，
   onset 取每个用户 drift_excess 最大的边界，不同用户的 onset 可以不同。
5. **drift / stable 用分位而不是绝对阈值**：观测到的 drift_excess 分不清"真实漂移"和"时间上的自然波动"，
   用"最高的 25% vs 最低的 25%"做相对比较更稳，两组样本量也相当；另留 `--min-drift-excess` 作绝对下限。
6. **对照组按 onset 分组 + 活动量分层匹配**：NDCG 与交互数强相关，
   不匹配活动量会把"活动量差异"读成"漂移效应"；同一 onset 保证两组在曲线上有相同的可用区间。
7. **两个数据集各自独立选 cohort**：阈值可能不同（见 §6.5），但同一数据集内 4 个 ρ × 3 个种子
   共用**同一份** cohort，所以每个数据集内部的比较仍是严格配对的。
8. **聚合按数据集分面**：图与 `val02_report.md` 都按数据集分开呈现，不把两个数据集的曲线混在一张图里。

## 9. 怎么判断结论成立（`val02_verdict.csv` 自动执行）

判定规则与验证实验一一致：`|均值| > max(0.002, 种子标准差)` 且 3 个种子同向才算真实效应；
恢复轮数要求对 ρ 单调 **且** 最慢−最快 ≥ 20 轮。

| 行 | 判定 | 含义 |
|---|---|---|
| ① | `drift − stable` 的"变化前后落差" > 噪声地板 | 兴趣真实变化确实造成性能下降（现象成立） |
| ② | 恢复/换邻居轮数随 ρ 单调变长 | 固定 EMA 确实让恢复变慢（任务书问题二的核心） |
| ③ | 变化前最优 ρ ≠ 变化后最优 ρ | 存在权衡 → 自适应更新有意义；同一个 ρ 两边都最优 → 没必要 |
| ④ | ρ=0 在变化后已经最好 | 那"直接把 ρ 调小"就够，自适应机制收益空间很小 |
| ⑤ | 旧兴趣同伴边权占比的半衰期随 ρ 变长 | 边确实还挂在旧兴趣上 |

★ 总结论四种：成立 / 部分成立（有下降但 EMA 无差异）/ 部分成立（EMA 有差异但 NDCG 无代价）/ 不成立。
后两种对应任务书第 8 节的停止信号 2，属于**有价值的否定结果**。

## 10. 已知局限（写报告要带上）

1. **类别是伪类别**：仓库里没有 MovieLens 原始 `movies.dat`，默认用第 1 个窗口的共现行为聚类出 18 个"伪类型"
   （随机符号投影 ≈ 共现内积 + KMeans，物品覆盖率 93%）。有原始 genre 文件时用
   `--genres-file "D:/.../movies.dat"` 换成真实类型，结论更可解释。
2. **cohort 规模受数据限制**：MovieLens-1M drift 146 / stable 102（阶梯 0），
   `amazon-digital music` drift 93 / stable 46（阶梯 3，已自动放宽）。
   想扩大可以加 `--drift-quantile 0.33`，但会混入漂移更弱的用户。
3. **变化后的观察窗口偏短**：MovieLens 用户大多只在 1–2 个窗口活跃，
   多数 drift 用户变化后只有 1 个窗口（120 轮）的评测点。曲线右端样本量会掉，
   必须同时看 `val02_recovery_curve.csv` 的 `n_users` 列；聚合脚本已自动在样本量掉到峰值 30% 处截断。
4. **onset 在最后一个边界（t=5）的用户无法验证持续性**（没有第 6 个窗口），默认仍纳入（约占 1/3）；
   只想用可验证持续性的用户时加 `--require-sustained`（会把可用用户从 582 砍到 78）。
5. **cohort 筛选用了整段时间跨度的数据**（含 onset 之后的窗口）。这是事后分组，只决定"统计哪些用户"，
   不进入模型、不影响任何在线决策；在线算法每个窗口只能看到该窗口已到达的交互。
6. **两组的 NDCG 绝对水平不同**（drift 组更活跃、更容易预测）。结论要基于"变化前后的落差"
   （difference-in-differences），不要直接比较两组的高度。
7. **窗口边界同时换掉了测试样本**：NDCG 在 rel_round=0 之后的变化里，有一部分来自
   "新兴趣的样本本身更难/更容易预测"。这正是必须有 stable 对照组的原因。
8. **`ema` 初始化下图在 ρ≥0.3 时几乎冻结**（ECR≈0），这是该机制的性质而非 bug；
   汇报时要同时给出 `warm` 对照组，避免把"图冻住了"误读成"推荐性能没受影响"。
9. 3 个种子只够看趋势，写论文建议扩到 5 个。
10. 本实验不涉及短期扰动（实验三）；"持续性"只能用一个窗口做粗筛。

## 11. 常见问题

- **只想跑一个数据集**：`-Datasets "movielens-1m"`。
- **cohort 想重算 / 为空 / 太少**：先删掉 `results/drift/<dataset>/`（批次脚本发现 cohort 已存在会跳过第①步），
  再按需放宽 `--min-inter-per-window`、`--min-drift-excess`，或降低 `--min-cohort`。
  第①步的 `val02_drift_summary.csv` 会列出各阈值下可用人数。
- **某个数据集被跳过了**：看日志里 `[skip] ... drift cohort 只有 N 人`。可以把 `-MinDriftCohort` 调小后重跑，
  或直接接受"该数据集不适合做实验二"（任务书也只点名 MovieLens-1M）。
- **核对原实现的图坍塌/冻结**：`python ...\val02_run_ema.py --graph-gate blended --rho 0.95`，
  看 `drift_n_neighbors` 以及 ECR 掉到 0。
- **想分离"旧边留得太久"与"新边进不来"**：再跑一遍 `--new-edge-init warm`（对照），
  两组结果并排看，就能说清滞后到底来自哪一侧。
- **重新聚合**：`python ...\val02_aggregate.py`；不会重训。删掉 `results/aggregated/` 可以清空重出。
- **跑了一半断了**：重跑同一条 `val02_run_basic.ps1` 即可，已完成的会自动跳过。
