# val_DynamicGraph：动态图到底有没有明确贡献？

- 目录：`val_DynamicGraph/`（在服务器 `/home/wl/exp_FedDIEG` 下）
- 前置：`exp_val_1026` 三个验证实验已经跑完，但出现了两个信号
  1. 实验二里改变 EMA 系数 ρ，推荐性能几乎不动；
  2. 因此无法判断「动态图」本身有没有贡献 —— 有可能是图协同分支的作用被别的机制稀释了。
- 本目录只做一件事：**在同一序列编码器、同一训练协议下，严格比较 NoGraph / StaticGraph /
  DynamicGraph**，先把「动态图有用吗」钉死，再谈自适应 EMA。

## 1. 三条臂（唯一的差别是图）

| 臂 | 图分支前向 | 服务器上的边状态 |
|---|---|---|
| `nograph` | **关**：永远拿不到邻居 ⇒ `u_i = seq_emb_i` | 不建图，从不更新 |
| `static` | 开 | 预热结束建**一次**，之后完全冻结 |
| `dynamic` | 开 | 每轮按论文 `e ← ρ·e + (1-ρ)·Δe` 更新 |
| `periodic`（补充） | 开 | 每 `period` 轮清零重建一次（「慢刷新静态图」） |

三条臂共用：同一份窗口数据、同一批候选集、同一条客户端参与序列、同一份负样本流、
同样的本地 epoch / lr / 服务器 lr / FedAvg 加权 / 个性化层划分 / 评测轮次 / 评测用户。
`static` 与 `dynamic` 在预热结束那一轮之前是**逐位相同**的（同一次构图、同一个 Top-K），
此后唯一的差别就是「要不要继续更新边」。

## 2. 结论怎么读（这是本实验的核心）

对同一 (数据集, 种子, 窗口) 逐对相减：

```
① StaticGraph  − NoGraph      = 图协同分支本身值多少（哪怕图是死的）
② DynamicGraph − StaticGraph  = 动态更新额外值多少
③ DynamicGraph − NoGraph      = 动态图的总贡献
```

| 结果 | 含义 |
|---|---|
| ①≈0 且 ②≈0 | 图分支没有可测贡献，动态性更无从谈起 → 先看 ⑥⑦ 找原因，**别急着做自适应 EMA** |
| ①>0 且 ②≈0 | 图有用，但每轮 EMA 更新没有额外用 → 静态图足够，自适应研究的收益有限 |
| ①>0 且 ②>0 | 动态性确实是有效成分 → 值得研究「什么时候更新、更新多少」 |
| ①≈0 且 ②>0 | 罕见组合，先查种子方差/一致性，别直接下结论 |

`dg_verdict.csv` 会把这四行连同数字、置信区间、p 值和一句判定文本自动生成出来；
`dg_report.md` 是给人读的版本。

## 3. 三个诊断量（解释「为什么」）

| 列 | 含义 | 怎么用 |
|---|---|---|
| `delta_rel` | `‖W_g(relu(Σ w_ij W_gnn h_j))‖ / ‖seq_emb‖`（最后位置） | **图分支的剂量**。NoGraph 必须恒为 0；若动态臂只有 1e-3，那么 EMA 怎么调都不会影响推荐 —— 直接解释实验二的 ρ 不敏感 |
| `neighbor_lift` | 用户与 Top-K 邻居摘要的加权余弦 ÷ 与随机活跃用户的平均余弦 | >1 说明图里的邻居确实更像自己（图有区分度）；≈1 说明图没提供信息 |
| `ecr` / `added_mean` | 每轮边变化率 / 新进入 Top-K 的边数 | 区分「图在慢慢变」与「图早就冻结了」 |

## 4. 严格性（脚本自己会硬检查）

`dg_selfcheck.py` 的 9 条断言（服务器实测 9/9 PASS，约 3 秒）：

| # | 断言 | 为什么必须 |
|---|---|---|
| ① | 三条臂喂给模型的负样本流**逐位相同** | 否则 NDCG 差异可能来自「抽到了不同负样本」，就不是配对比较 |
| ②a | NoGraph 的 `delta_rel_max` 恒为 **0**（不是「很小」） | 证明空图控制真的关干净了 |
| ②b | DynamicGraph 在有边用户上的 `delta_rel > 0` | 证明图臂真的接上了，而不是「图空着」 |
| ③a | StaticGraph 冻结后边签名一位不变 | 证明静态臂真的冻结 |
| ③b | DynamicGraph 每轮都更新、StaticGraph 只更新一次 | 证明动态臂真的在动 |
| ④ | `--graph-mix 0` 与 NoGraph 的逐用户 NDCG **逐位相同** | 空图控制的黄金标准：不是「把图调小」，而是等价 |
| ⑤ | 诊断前向与 `score_user` 打分一致 | 否则 `delta_rel` 测的不是真正影响推荐的量 |
| ⑥ | 三条臂的初始参数逐位相同 | 同一个 seed 下初始化一致 |
| ⑦ | 客户端参与序列可复现 | 同 seed 两次构造完全相同 |

`dg_run.py` 运行时还会硬检查：NoGraph 的 `delta_rel_max` 必须为 0（否则直接抛错）、
StaticGraph 冻结后签名不得变化（否则抛错）、图臂预热后不得是空图（否则打印醒目警告）。

## 5. 文件清单

```
val_DynamicGraph/
├─ dg_common.py        三条臂的定义（NullGraph / PolicyGraph）+ 图分支探针 + CLI 参数
├─ dg_run.py           单次运行 = (数据集, 臂, 种子)；4 个 CSV + 日志 + config
├─ dg_selfcheck.py     9 条严格性断言（约 3 秒，跑正式实验前先跑这个）
├─ dg_aggregate.py     配对差值 + 结论表 + 报告 + 图
├─ dg_run_pilot.sh     冒烟：3 臂 × 1 种子 × 40 轮/窗口（约 8~12 分钟）
├─ dg_run_basic.sh     正式：3 臂 × 3 种子 × 120 轮/窗口（约 4~4.5 小时）
├─ README_val_DynamicGraph.md
└─ results/
   ├─ raw/<dataset>/<arm>/seed_<s>/   dg_summary.csv / dg_round_metrics.csv /
   │                                  dg_user_metrics.csv / dg_probe_metrics.csv /
   │                                  config.json / dg_run.log
   ├─ raw_pilot/ …                    冒烟结果
   └─ aggregated/                     dg_report.md + tables/ + figures/
```

聚合产出（`results/aggregated/`）：

| 文件 | 内容 |
|---|---|
| `dg_report.md` | **先看这个**：一句话结论 + 全部表 + 已知局限 |
| `tables/dg_verdict.csv` | ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨ 九行结论（含数值、CI、p 值、判定文本） |
| `tables/dg_contrasts.csv` | 每条对照的 Δ 均值/中位数/CI/正号比例/种子方向/p 值 |
| `tables/dg_contrasts_by_seed.csv` | 每个种子各自的 Δ（看符号是否一致，比 p 值可靠） |
| `tables/dg_contrasts_window.csv` | 逐个 (种子, 窗口) 的配对差值（明细） |
| `tables/dg_user_paired.csv` | 逐用户配对 + bootstrap CI + 符号检验 + Wilcoxon |
| `tables/dg_user_paired_by_window.csv` | 逐用户差值按窗口汇总 |
| `tables/dg_arm_summary.csv` / `dg_arm_curve.csv` | 每条臂的汇总 / 逐窗口曲线 |
| `tables/dg_arm_window.csv` | 逐 (臂, 种子, 窗口) 的窗口级指标（合并用） |
| `tables/dg_diagnostics.csv` | `delta_rel` / `neighbor_lift` / `ecr` / 边数 |
| `tables/dg_hygiene.csv` | 聚合侧的一致性检查（协议参数、空图、冻结、每轮更新、评测对齐） |
| `figures/dg_fig_arms_<dataset>.png` | 三联图：NDCG 曲线 / delta_rel / neighbor_lift |

## 6. 怎么跑

解释器：`/home/wl/.venv-feddieg/bin/python`（Python 3.12.3 / torch 2.15.0+cu132 / pandas 3.1.1）。
脚本会自动找到它；也可以 `PYTHON=...` 覆盖。

```bash
cd /home/wl/exp_FedDIEG

# ① 自检（约 3 秒）：证明三条臂只差图这一件事
/home/wl/.venv-feddieg/bin/python -X utf8 val_DynamicGraph/dg_selfcheck.py

# ② 冒烟（约 8~12 分钟，可选但建议）：三条臂小规模跑通 + 自动出表
bash val_DynamicGraph/dg_run_pilot.sh

# ③ 正式：3 臂 × 3 种子 = 9 次运行，末尾自动聚合（约 4~4.5 小时）
bash val_DynamicGraph/dg_run_basic.sh

# ④ 看结论
cat  val_DynamicGraph/results/aggregated/dg_report.md
column -s, -t < val_DynamicGraph/results/aggregated/tables/dg_verdict.csv | cut -c1-160
```

单条运行（调试用）：

```bash
/home/wl/.venv-feddieg/bin/python -X utf8 val_DynamicGraph/dg_run.py \
    --dataset movielens-1m --arm dynamic --seed 42
/home/wl/.venv-feddieg/bin/python -X utf8 val_DynamicGraph/dg_run.py \
    --dataset movielens-1m --arm static --seed 42 --freeze-round 120
/home/wl/.venv-feddieg/bin/python -X utf8 val_DynamicGraph/dg_aggregate.py
```

脚本行为：已完成的运行自动跳过（中断后重跑同一条命令即可续跑）、单次失败不中断整轮、
末尾自动聚合。用时估算按 3090 实测 2.45 s/轮（`5 窗口 × 120 轮 ≈ 25 分钟/次`）。

## 7. 接下来的补充对照（都不是必须，但能挡反驳）

```bash
# (a) 慢刷新静态图：每窗口重建一次，而不是每轮 EMA
#     用来区分「动态性有用」与「静态臂的图建得太早所以太差」
ARMS="nograph static dynamic periodic" bash val_DynamicGraph/dg_run_basic.sh

# (b) 剂量反应：把图分支放大 5 倍，看 delta_rel 太小时是否才显出贡献
#     若 ①≈0 且 ⑥ 的 delta_rel 只有 1e-3，这一条直接回答「是不是被稀释了」
GRAPH_MIX=5 bash val_DynamicGraph/dg_run_basic.sh

# (c) 只换动态臂的 EMA 系数（自动进 dynamic_rho0.8 目录，并与 dynamic 配对）
RHO=0.8 ARMS="dynamic" bash val_DynamicGraph/dg_run_basic.sh
```

三者都写进同一棵 `results/raw/`，聚合脚本会把它们当成独立变体，自动额外生成
`dynamic_mix5 − dynamic`、`dynamic_rho0.8 − dynamic` 这类对照行。

## 8. 几个必须知道的实现选择

1. **`lambda_struct = 0`（本实验默认，偏离论文的 1e-5）**
   L_struct 定义在图量 Δe 上，NoGraph 臂没有图就必然没有它。留着它会变成
   「图臂多一个训练信号」的混淆项。要回到论文口径：加 `--lambda-struct 1e-5`，
   但那样三条臂就**不再只差图**了（`dg_run.py` 会打印提醒）。
2. **`static` 的构图时点**默认是预热结束那一轮（第 `graph_warmup_rounds+1` 轮）。
   那时摘要只更新过一轮，图偏弱。若 ① 为负，别直接说「静态图有害」，先用
   `--freeze-round 120`（冻得更晚但不动态）或 `periodic` 臂复查。
3. **`nograph` 没有改 `feddieg_core` 一行代码**：`NullGraph` 用鸭子类型顶替
   `DynamicInterestGraph`，`FedTrainer` / `evaluate_split` 走的是同一个「无邻居」分支。
4. **目录名会带配置后缀**：`static_k10`（neighbors=10）、`dynamic_mix5`、`dynamic_rho0.8`，
   避免同臂不同配置互相覆盖；聚合脚本按「精确同名优先，否则取后缀最短的变体」解析主对照。
5. **同一组对照必须固定在同一台机器上跑完**：CPU 与 GPU 的 dropout 随机流不同
   （跨机器同 seed 的 CUDA 结果本身逐位一致）。

## 9. 环境提示

- 服务器**没有安装 matplotlib**：`dg_aggregate.py` 会跳过出图并给出提示，表格与报告不受影响。
  要出 PNG：`/home/wl/.venv-feddieg/bin/pip install matplotlib` 后重跑聚合脚本即可。
- scipy 可选：装了就用 t 分布 / Wilcoxon，没装则用正态近似 + 精确符号检验（不依赖 scipy）。
- 磁盘：`/` 只剩约 27 GB。本实验每次运行的全部输出约 2~5 MB（主要是逐用户 CSV），
  9 次运行不到 50 MB，够用。
