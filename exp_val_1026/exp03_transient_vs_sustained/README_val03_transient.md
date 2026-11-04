# 验证实验 3（val03）：短期兴趣波动会不会导致不必要的图更新？

- 目录：`exp_val_1026/exp03_transient_vs_sustained/`
- 对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验三」
- 只做诊断，不设计算法。**交付物是两张图**：
  - `results/aggregated/figures/val03_fig_arm_synthetic.png`（注入 arm：持续变化 vs 短期扰动）
  - `results/aggregated/figures/val03_fig_pair_excess.png`（净效应：注入组 − 配对对照）
  - 另外 `val03_fig_rho_tradeoff.png` 回答"有没有一个 ρ 能两全"。

## 1. 这一版做了什么

| | 内容 |
|---|---|
| 自变量① | **场景**：`sustained`（兴趣持续变化）/ `transient`（短期扰动后恢复） |
| 自变量② | 边权 EMA 系数 ρ：默认 `0` / `0.5` / `0.8`（论文默认 0.8） |
| 控制变量 | 同一份窗口数据、同一份客户端参与序列、**同一份用户配对**、同一个随机种子 → 组间是配对比较 |
| 参与率 | 100%（让"谁参与"不再成为干扰；陈旧性属于实验一） |
| 数据集 | MovieLens-1M + `amazon-digital music`（`-Datasets` 可改） |
| 种子 | 42 / 777 / 2024（3 个独立随机种子） |
| 规模 | 2 数据集 × 2 场景 × 3 ρ × 3 种子 = **36 次运行**，每次 5 窗口 × 120 轮 |
| 产出 | 每次运行 4 个文件；聚合出 **5 张图 + 8 张表 + 报告骨架**，一条命令全自动 |
| 用时 | MovieLens-1M 约 3 分钟/次、Amazon 约 7 分钟/次 → 默认全量约 **3 小时** |

## 2. 实验设计：一个运行里同时装 5 个用户组

任务书要求"可以利用真实数据中能够识别的行为模式，也可以构造可控的扰动实验，**但必须清楚区分真实数据结论和人工构造的结果**"。
所以本实验把两类证据放进**同一次训练运行**，共 5 个组：

| arm | 组名 | 数据怎么来的 | 作用 |
|---|---|---|---|
| 注入（人工构造） | `inj` | 把 onset 窗口里该用户的 `itemId` 换成**目标类别**（该用户最不熟悉的类别）的物品 | 被检验的场景 |
| 注入（人工构造） | `pair` | **同样的窗口、同样的替换流程**，但换成**自己旧主类别**的物品（同类别重采样） | 配对对照：把"替换动作本身"对照掉 |
| 真实（只打标签） | `real_sustained` | 真实数据里"主类别跳变后**保持**新类别"的用户 | 真实数据里的场景 A |
| 真实（只打标签） | `real_transient` | 真实数据里"主类别跳变后**回到**旧类别"的用户 | 真实数据里的场景 B |
| 真实（只打标签） | `real_stable` | 真实数据里从未跳变、且与上面两组同 onset + 活动量匹配 | 窗口边界换测试样本的对照 |

**两个场景各跑一次，只差"注入窗口的个数"**：

```
sustained  注入窗口 = onset .. W      （持续证据：3 个窗口）
transient  注入窗口 = onset           （一次性证据：1 个窗口，之后数据恢复原样）
```

其余全部相同（同一批 inj 用户、同一批 pair 配对、同一个 onset=3、同样的目标类别），
所以两轮之间的差别**只有证据持续多久**。

### 2.1 为什么"替换 itemId"而不是改摘要向量

- 窗口是按行位置等分的（`int(N*w/W)`），只改 `itemId` **不会改变任何一行的窗口归属** →
  两组的窗口边界、评测用户数、训练轮数完全一致；
- 改了 itemId 之后，客户端的序列、上传的兴趣摘要、服务器端的边权全部自然跟着变，
  不需要另外发明一套"扰动摘要"的机制；
- `pair` 组做同样动作（同类别重采样）→ "物品被换过"这个混杂因素被对照掉。

### 2.2 为什么 pair 组的 `cat_new` 用"配对对象的注入目标类别"

`old_peer_share` / `new_peer_share` 是"邻居里多少边权还落在**我的旧兴趣/新兴趣**上"。
如果 pair 组的"新兴趣"取自己的旧类别，两组的参照系不同、不能相减。
所以 plan 里存了两列：

| 列 | 含义 |
|---|---|
| `replace_category` | 实际替换成哪个类别（`inj`=目标类别；`pair`=自己的旧主类别） |
| `cat_new` | **指标参照类别**（`pair` 也用配对对象的目标类别） |

这样 `inj − pair` 才是干净的净效应。

## 3. 三步流程（已封装成一条命令）

```
① val03_build_scenarios.py  离线构造场景（不训练）：类别视图 + 真实队列 + 注入配对 → results/scenarios/
② val03_run_scenario.py     跑一个 (数据集, 场景, ρ, 种子) → results/raw/
③ val03_aggregate.py        聚合出图 + 表 + 自动结论 → results/aggregated/
```

`val03_run_basic.ps1 / .sh` 把 ①②③ 串起来：**跑一条命令就得到全部结果**。
第①步不训练（两个数据集共约 2 分钟）；第②步是主要成本；第③步几秒。

## 4. 与任务书「实验三」的符合性核对

| 任务书要求 | 本实现 | 状态 |
|---|---|---|
| **场景 A：持续兴趣变化**（后续多个窗口持续表现新兴趣） | `sustained`：注入窗口 onset..W，持续 3 个窗口 | ✅ |
| **场景 B：短期兴趣扰动**（少量交互表现短暂变化，后续恢复） | `transient`：只注入 onset 一个窗口，之后数据恢复原样 | ✅（粒度=1 窗口，见 §9 局限 2） |
| 可利用真实数据中可识别的行为模式 | `real_sustained` / `real_transient` / `real_stable` 三个真实队列 | ✅ |
| 也可构造可控扰动实验 | 注入 arm（`inj` / `pair`），扰动幅度由 `--inject-fraction` 控制 | ✅ |
| 必须清楚区分真实数据结论与人工构造结果 | 两套 arm 在**同一个 run 内**共存、但统计时**分开出图**（`fig_arm_real` vs `fig_arm_synthetic`），报告分节写 | ✅ |
| 观察① 两种情况下边关系变化程度是否不同 | `turnover_from_pre` + `old/new_peer_share` 曲线（`val03_curve_*.csv`、两张 arm 图） | ✅ |
| 观察② 对短期扰动是否出现过度更新 | `over_update` = 撤销后残留 ÷ 扰动期改动（成对差值口径），见 `val03_tradeoff.csv` | ✅ |
| 观察③ 对持续变化是否出现更新滞后 | `under_update` = 1 − 跟进量 ÷ ρ=0 跟进量；`lag_reach_50` | ✅ |
| 观察④ 哪种情况下推荐性能损失更大 | `gap/delta` 与 DID（注入组 − 配对对照），见 `val03_window_summary.csv` 与判定⑥ | ✅ |
| 预期交付：比较持续漂移与短期扰动的**推荐性能及图变化率**的图 | `val03_fig_arm_synthetic.png`（NDCG + 关系质量）+ `val03_fig_pair_excess.png` | ✅ |
| **特别注意：图变化率越低不一定越好** | 结论同时看两个方向：`over_update`（该撤销没撤销）与 `under_update`（该变没变），不排序 turnover 大小 | ✅ |
| 3 个独立随机种子 | 同 | ✅ |

## 5. 文件清单

```
exp_val_1026/exp03_transient_vs_sustained/
├─ val03_build_scenarios.py    ① 场景构造（不训练）：真实队列 + 注入配对 + 类别视图
├─ val03_run_scenario.py       ② 一次完整训练（5 个组一起测）
├─ val03_aggregate.py          ③ 曲线 + 过度更新/滞后指标 + 自动结论 + 出图
├─ val03_paired_test.py        ④ 逐对配对检验（ΔNDCG vs Δold-peer / Wilcoxon / 相关性）
├─ val03_run_basic.ps1 / .sh   一键跑完：①②③ 全自动（默认两个数据集，约 3 小时）
├─ val03_run_pilot.ps1 / .sh   冒烟：两个数据集 × 2 场景 × 2 ρ × 40 轮/窗口（约 15 分钟）
├─ README_val03_transient.md   本文件
└─ results/
   ├─ scenarios/<dataset>/     注入计划 + 真实队列清单 + 类别计数（已预生成，可直接用）
   ├─ raw/<dataset>/<scenario>/rho<ρ>/seed_<s>/   正式结果
   ├─ raw_strong/...           放大图影响的敏感性结果（-GraphStrength strong）
   ├─ raw_warm/...             新边 warm 初始化（-NewEdgeInit warm，会消掉"图冻住"）
   ├─ aggregated/              主结果：figures/ + tables/ + val03_report.md + ★ val03_findings.md
   └─ archive/                 pilot 试跑
```

## 6. 怎么跑

### 6.1 一条命令跑完（推荐）

```powershell
# 先设好 Python（用你跑实验那个环境）
$env:PYTHON = "D:\Myenvs\FedDIEG\python.exe"

# 冒烟（可选，约 15 分钟）：两个数据集 × 2 场景 × 2 ρ × 40 轮，结果写 results/archive/
powershell -NoProfile -ExecutionPolicy Bypass -File .\exp_val_1026\exp03_transient_vs_sustained\val03_run_pilot.ps1

# 正式：两个数据集 × 2 场景 × 3 ρ × 3 种子 = 36 次运行，末尾自动出图（约 3 小时）
powershell -NoProfile -ExecutionPolicy Bypass -File .\exp_val_1026\exp03_transient_vs_sustained\val03_run_basic.ps1
```

Linux / WSL：

```bash
bash exp_val_1026/exp03_transient_vs_sustained/val03_run_pilot.sh
bash exp_val_1026/exp03_transient_vs_sustained/val03_run_basic.sh
```

跑完直接看 `results/aggregated/`：图 + 表 + `val03_report.md` 都有了，不需要再手动执行聚合。

脚本行为（都是为了"一条命令跑完"）：

- **已完成的任务自动跳过**：中断后重跑同一条命令即可续跑。
- **单个任务失败不中断整轮**：失败的组合会在最后汇总打印出来。
- **不可用的数据集自动跳过**：注入组少于 `-MinInjectPairs`（默认 8）人时只警告并跳过，其它数据集照常跑完。
- **末尾自动聚合**：不同图强度/新边初值分别聚合到 `results/aggregated*`。
- 想改网格就加参数，例如 `-Rhos "0,0.8" -Seeds "42"`（约 1 小时）、`-Datasets "movielens-1m"`、`-NoAggregate`。

### 6.2 拆开手动跑（调试用）

```powershell
# ① 场景构造（两个数据集各一次；已存在时自动跳过）
python exp_val_1026\exp03_transient_vs_sustained\val03_build_scenarios.py --dataset "movielens-1m"
python exp_val_1026\exp03_transient_vs_sustained\val03_build_scenarios.py --dataset "amazon-digital music"

# ② 单个 (数据集, 场景, ρ, 种子)
python ...\val03_run_scenario.py --dataset "movielens-1m" --scenario sustained --rho 0.8 --seed 42
python ...\val03_run_scenario.py --dataset "movielens-1m" --scenario transient --rho 0.8 --seed 42

# ③ 聚合出图
python ...\val03_aggregate.py

# ④ 逐对配对检验（ΔNDCG vs Δold-peer + Wilcoxon + 相关性；结论图之外的统计支撑）
python ...\val03_paired_test.py --datasets "movielens-1m"
```

`val03_run_scenario.py --self-check` 会把"未被注入的数据帧"与实验二的 `load_rolling_windows`
逐元素对比（`train_seq / train_pos / val_seq / test_seq / test_items`），
用来确认本实验没有偷偷改掉窗口口径。已实测：**完全一致**。

## 7. 输出与字段说明

### 7.1 第①步：`results/scenarios/<dataset>/`

| 文件 | 内容 |
|---|---|
| `val03_injection_plan.csv` | ★ 注入清单：`user, group(inj/pair), pair_id, onset_window, cat_old, cat_new, replace_category, is_placebo, target_rarity, n_inter_*` |
| `val03_real_cohorts.csv` | ★ 真实队列：`user, group(real_sustained/real_transient/real_stable), onset_window, cat_old, cat_new, spike_purity, n_post_windows, drift_excess, tvd_spike, null_tvd, matched_user` |
| `val03_counts.npz` | `counts[user, window, category]` + `n_inter[user, window]` + `item_category[item]` |
| `val03_partition.csv` / `val03_category_names.json` / `val03_category_meta.json` | 物品→类别、类别名、类别来源与覆盖率 |
| `val03_scenario_meta.json` | ★ 全部阈值、人数、自动放宽阶梯、两套 arm 的定义、泄漏说明（写报告直接用） |
| `val03_build.log` | 构造日志 |

### 7.2 第②步：`results/raw/<dataset>/<scenario>/rho<ρ>/seed_<s>/`

| 文件 | 内容 |
|---|---|
| `val03_user_round_metrics.csv` | ★ 逐用户逐评测轮：`rel_round, ndcg, hr, old_peer_share, new_peer_share, turnover_from_pre, summary_drift, ecr` —— 所有曲线的原始数据 |
| `val03_round_metrics.csv` | 逐评测轮 × 分组聚合（含 `train_loss`、`ecr_mean`、`summary_drift`） |
| `val03_summary.csv` | 本次运行汇总（一行，含各组人数、注入窗口） |
| `config.json` / `val03_run.log` | 超参数与日志 |

`rel_round = 当前轮 − (onset − 1) × 每窗口轮数`，**0 = 兴趣变化前最后一轮**，正数 = 变化之后。
注入组/对照组的 onset 固定（默认 3），真实组各人 onset 不同 → 都在同一根横轴上对齐。

### 7.3 第③步：`results/aggregated/`

| 文件 | 内容 |
|---|---|
| `figures/val03_fig_arm_synthetic.png` | ★ 注入 arm 原始曲线：old-peer 占比 + NDCG，`inj` 与 `pair` 各一条 |
| `figures/val03_fig_pair_excess.png` | ★★ **主结论图**：净效应（inj − pair），**所有 ρ 画在同一张里**（颜色=ρ） |
| `figures/val03_fig_arm_real.png` | ★ 真实 arm：`real_sustained` / `real_transient` / `real_stable` |
| `figures/val03_fig_rho_tradeoff.png` | ★★ 关键图：过度更新（横）vs 更新滞后（纵），每个 ρ 一个点；图没动时画成空心点 |
| `figures/val03_fig_rho_curves.png` | 按 ρ 展开的关系质量曲线（只看 inj 组） |
| `tables/val03_curve_<metric>.csv` | 各指标曲线（含 `n_users`、用户级 SEM） |
| `tables/val03_pair_excess.csv` | 成对差值（`inj − pair`）曲线 |
| `tables/val03_window_summary.csv` | 每个 (数据集, 场景, 组, ρ) 的 `pre / during / post1 / post2 / plateau` 水平 |
| `tables/val03_tradeoff.csv` | ★ `over_update_early` / `over_update` / `under_update` / 摘要残余比 |
| `tables/val03_lag_old_peer_share.csv` | 关系质量达到 50% 最大变化所需轮数（窗口粒度=10 轮，容易饱和，看趋势即可） |
| `tables/val03_ndcg_paired_test.csv` | ★★ 逐对配对检验（ΔNDCG / Δold-peer / Wilcoxon / 相关），第④步产出 |
| `tables/val03_verdict.csv` | 自动结论判定（逐数据集，7 行） |
| `val03_report.md` | 已填好数字的报告骨架 |
| `val03_findings.md` | ★★ 人工整理后的结论（**引用数字以它为准**） |

## 8. 怎么判断结论（`val03_verdict.csv` 自动执行）

判定门槛：关系质量 `old_peer_share` 的噪声地板 **0.02**；NDCG 噪声地板 **0.002**
（与实验一/二一致，并要求种子同向）。

| 行 | 判定 | 含义 |
|---|---|---|
| ① | 扰动窗口里注入组的 old-peer 占比**净下降** > 0.02 | 短期扰动**确实进了图**（不是"什么都没发生"） |
| ② | `over_update`（再下一个窗口的残留）< 0.5 | 撤销后图能**回退** → "过度更新"不成立；≥0.5 → 图改了就回不来。另看 `over_update_early`（撤销窗口的残留）：>0.5 = 至少晚一个窗口才回退 |
| ③ | `under_update` ≤ 0.5 | 持续变化被**跟上**；>0.5 → 更新滞后 |
| ④ | 存在某个 ρ 同时满足 ②③ | 有 → 固定 EMA 够用（任务书第 8 节停止信号 2 成立）；没有 → 需要自适应。**注意**：命中时若唯一满足的 ρ 就是 0，必须同时讨论"关掉 EMA"的代价（撤销延迟 + 图抖动） |
| ⑤ | 摘要恢复（残余比<0.5）但图没恢复（over_update≥0.5） | 责任在**边权更新规则**，不在"摘要记忆" |
| ⑥ | NDCG 的 DID < −0.002 | 图的变化**有推荐代价**（否则只能说"关系层面有问题"） |
| ⑦ | 真实 `real_sustained` / `real_transient` 的 old-peer 变化（描述性） | 真实数据上的方向性核对 |

★ 四种可能的总结论，对应四种写法（都要照写，不要挑好看的）：

| 结果 | 结论怎么写 |
|---|---|
| ②③ 都成立（over_update<0.5、under_update≤0.5） | **任务书的假设不成立**：固定 EMA 已经能区分两类证据 → 第 8 节停止信号 2 命中，这条方向要放弃 |
| ②不成立（过度更新）但③成立 | 失效在"该撤销时没撤销" → 研究问题应写成"**如何判断一条协同关系是否仍然可信**" |
| ②成立但③不成立（更新滞后） | 与实验二同向（图要么冻结要么过冲）→ 研究问题写成"**何时该更新、更新多少**" |
| ②③都不成立 | 固定 EMA 两端都错，且没有一个 ρ 能两全 → 最强的"需要自适应"证据 |

另外必须注意：**如果某个 ρ 下 ①都不成立**（扰动期净下降 < 0.02），说明该 ρ 下图根本没动
（ρ≥0.5 时这就是"图冻结"），此时 ② 的"能回退"是假象，**报告里必须写成"图没动"而不是"能撤销"**。

> **本次实测结论**（36 次运行跑完后的实际结果，细节见 `val03_findings.md`）：
> - ρ=0.8：①**不成立**（扰动期净下降 0.014≈0）→ 现状配置下**没有**"过度更新"；③ 也不成立（只跟上 29%）。
> - ρ=0.0：①成立（0.164，3/3 种子），图跟着扰动走；`over_update_early`≈1.0（**整整一个撤销窗口没回来**），
>   但 `over_update`≈0（**再下一个窗口完全恢复**）；③ 完全跟上（0.303），代价是 ECR 0.161/轮（图每轮大改）。
> - ρ=0.5：①成立但很小（0.067）；`over_update`≈0.6–0.7（**撤不回**，3/3 种子），③ 只跟上 22%。
>   （两个口径略有差别：`val03_tradeoff.csv` 是曲线加权、`val03_ndcg_paired_test.csv` 是逐对等权，
>   差 ≤0.04；上面取约数。）
> - **没有任何 ρ 同时做到"跟得上 + 及时撤销 + 图稳定"**：ρ=0 只输在"及时"和"抖动"，ρ≥0.5 输在另外两条。
>   ⚠️ 只按"最终是否回到基线"打分的话 ρ=0 是最接近可接受的 —— 这就是任务书停止信号 2 的**部分命中**，
>   报告里必须正面论证"为什么 ρ=0 不够"，不能回避。
> - 而且**推荐损失与图变化无关**（NDCG 差距在三个 ρ 上几乎相同：0.195 / 0.218 / 0.183），
>   所以"过度更新"这条失效路径**不足以单独支撑一个新算法**；研究问题应转到"证据是否值得写进摘要"。

## 9. 已知局限（写报告要带上）

1. **注入是人工构造**：只替换 `itemId`，不改窗口边界/交互数/时间戳；结论必须与真实 arm 分开写。
2. **短期扰动的粒度是 1 个窗口**（受窗口等分限制），不是 1 轮或 1 天。真实数据里若"几天"的波动
   远短于一个窗口，本实验会**低估**过度更新的程度。想更细只能用轮级注入，但那样就缺乏真实数据支撑。
3. **注入的痕迹会留在历史上**：onset 窗口被替换的物品，在之后的窗口里仍在用户的历史序列中
   （真实系统同样无法抹掉"连刷几部恐怖片"）。所以测得的是「摘要记忆 + 边权记忆」的合计残留，
   用 `summary_drift`（上传摘要相对 onset 前的 1−cos）把这两层分开：
   摘要已恢复而图没恢复 → 责任在 EMA 边权规则。
4. **注入组规模有限**（实测）：MovieLens-1M 严格口径 **19 对**（`--min-inter 2`，要求窗口 2/3/4 各 ≥2 条交互）；
   想要更多人：`--inject-onset 4`（约 55 对，但撤销后只有 1 个观察窗口）或 `--min-inter 1`（约 24 对，部分用户缺一个窗口的评测点）。
5. **Amazon 类稀疏数据集**：每窗口每用户仅约 2 条交互、伪类别只覆盖 64.5% 物品，
   真实 arm 的 transient 组自动放宽后只有 24 人 → **只能作方向性参考**（与实验二的结论一致）。
6. **NDCG 的成对差值不完美**：sustained 组在扰动窗口之后的测试目标是"新兴趣物品"、
   transient 组是"旧兴趣物品"，两组绝对水平不可直接比较；只能各自对自己 pre 期做差再相减（DID）。
   另外注入物品是从目标类别**均匀抽样**的（多半是长尾物品），扰动窗口内的绝对 NDCG 会偏低 ——
   这正是 `pair` 组也要替换的原因（两组同样受这个影响）。
7. **真实 arm 的分组用到了 onset 之后的窗口**（事后分组），只决定统计哪些用户，
   不进入模型、不影响任何在线决策；在线算法每个窗口只能看到该窗口已到达的交互。
8. **`turnover_from_pre` 在 ρ 小时会饱和**：MovieLens 有 6000 个可连接用户、只留 Top-10 邻居，
   一批证据回来就能把 10 个邻居全换掉（实测 ρ=0 时≈1.0，与实验二一致）。
   所以它只用来区分"图在不在动"，**方向性判断一律看 `old/new_peer_share`**。
9. **真实队列的 onset 不统一，且变化后可观察窗口偏短**（多数人 `n_post_windows`≈1），
   曲线右端样本量会掉；聚合表里每个点都带 `n_users`，引用时要一起看。
10. 3 个种子只够看趋势；写论文建议扩到 5 个。

## 10. 参数与常见问题

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | `movielens-1m` | 数据集（构造阶段用，训练阶段自动读同名场景） |
| `--scenario` | `sustained` | `sustained` / `transient` |
| `--rho` | `0.80` | 边权 EMA 系数（论文默认 0.8） |
| `--inject-onset` | `3` | 注入发生在第几个窗口（要留出 pre 与 post） |
| `--min-inter` | `2` | 注入候选在 onset 前/onset/onset 后各窗口的最少交互条数 |
| `--min-pool` | `30` | 注入候选少于这个数就自动放宽 `--min-inter` |
| `--inject-fraction` | `1.0` | 注入窗口内被替换的交互比例（0.5 = 只换一半，"少量交互"的更强版本） |
| `--target-category-mode` | `rare` | 目标类别取"用户最不熟悉的类别"（`hash` = 按用户 hash 随机取） |
| `--graph-strength` | `paper` | `paper` = 已发表配置；`strong` = 放大图影响（敏感性对照） |
| `--new-edge-init` | `ema` | `ema` = 忠实复刻原实现；`warm` = 新边不被惩罚（会消掉"图冻住"） |
| `--graph-gate` | `raw` | `min_sim` 作用在原始余弦（推荐）或混合后的值（原实现） |
| `--reference-rho` | `0.80` | 聚合时主图用的 ρ（该 ρ 缺失时自动选最接近的） |

- **只想跑一个数据集**：`-Datasets "movielens-1m"`。
- **只想快速看趋势**：`-Rhos "0,0.8" -Seeds "42"`（约 1 小时）。
- **注入组太小 / 想重算场景**：删掉 `results/scenarios/<dataset>/` 后重跑；
  或加 `--inject-onset 4`、`--min-inter 1`、`--min-pool 20`。
- **想确认"图冻住"不是我的实现问题**：`--new-edge-init warm` 再跑两组，
  看 ρ 之间的权衡是否显现（实验二 §6.2 有同样开关）。
- **ρ≥0.5 时 over_update 算不出来**：说明该 ρ 下扰动期净改动低于噪声地板（图几乎没动），
  这是"图冻结"的表现，不是程序错，报告里要写成"图没动"。
- **重新聚合**：`python ...\val03_aggregate.py`；不会重训。删掉 `results/aggregated/` 可清空重出。
- **跑了一半断了**：重跑同一条 `val03_run_basic.ps1` 即可，已完成的会自动跳过。
