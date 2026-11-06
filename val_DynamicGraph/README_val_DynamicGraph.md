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
   └─ aggregated/                     dg_report.md + tables/*.csv + figures/*.png
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

# ③ 正式（推荐并行版）：3 臂 × 3 种子 = 9 次运行，末尾自动聚合
bash val_DynamicGraph/dg_run_parallel.sh     # 4 并发，约 40~50 分钟

# ③' 或者串行版（每卡只跑 1 个进程，约 3.5 小时）
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

## 10. 服务器实测：算力利用率与并行（2026-11-05）

**单个进程完全用不满卡**（`nograph` 跑第一个 run 时实测，2× RTX 3090）：

| 指标 | 实测 | 说明 |
|---|---|---|
| GPU0 利用率 | **31.6%**（28~35%，40 秒采样 20 次） | 空闲 2/3 时间没有 kernel 在跑 |
| GPU0 显存 | **452 MiB / 24576 MiB**（1.8%） | 显存从来不是瓶颈 |
| GPU0 功耗 | **113 W / 350 W**（32%） | SM 时钟 1695/2100 MHz |
| GPU1 | **0%** | `--device auto` 只用 0 号卡 |
| CPU | 进程占 **1 个核**（97.6%），32 核里 3% | CPU 也很闲 |
| 推进速度 | **约 1.2 s/轮**（非评测轮）、评测轮 2~5 s | 首轮评测 2.2 s，末轮涨到 5.3 s |

**瓶颈不在算力**：训练循环是「逐客户端 batch=1 串行」，每轮 604 个客户端，每个客户端都要把
个性化层的几十个小张量 CPU→GPU 拷一次（一轮上万次小拷贝），所以瓶颈是 kernel launch 与
拷贝开销。评测同理：`evaluate_split` 按个性化状态分组，越到后面每组一个用户，就变成上千次
小前向 + 上千次 `load_state_dict`（这也是评测耗时从 2.2 s 涨到 5.3 s 的原因）。

**并发扩容实测**（同一张卡，固定 nograph / 2 窗口 x 20 轮）：

| 配置 | 单进程训练 | 墙钟加速比 |
|---|---|---|
| 个性化层存 CPU，1 进程 | 2000 ms/轮 | 1.00× |
| 个性化层存 CPU，2 进程并发 | 2641 ms/轮 | **1.47×**（互相拖慢） |
| 个性化层存 CPU，3 进程并发 | 3309 ms/轮 | 1.77×（过度订阅） |
| 个性化层存 GPU，1 进程 | 1782 ms/轮 | 1.00×（比 CPU 驻留快 11%） |
| 个性化层存 GPU，**2 进程并发** | 1782 ms/轮 | **2.21×**（几乎不互相拖慢） |
| 个性化层存 GPU，3 进程并发 | 3083 ms/轮 | 1.89×（过度订阅） |

**结论与默认配置**：`--personal-on-gpu` + **每卡 2 个进程**（两张卡共 4 并发）是最优；
`dg_run_parallel.sh` 就按这个默认值跑，9 次运行约 40~50 分钟（串行约 3.5 小时）。

`--personal-on-gpu` 的安全性已验证：同 seed、同配置下跑两遍（只改这个开关），
`dg_round_metrics.csv` 与 `dg_user_metrics.csv`（6643 行逐用户记录）**逐字段字符串完全相同** ——
它只改"个性化参数存在哪块内存"，不改任何计算。为可追溯，`dg_summary.csv` 里仍记录
`personal_on_cpu`，聚合时它会单独显示为信息项（不算不一致）。

## 11. 当前结论（2026-11-06，15 次运行 / 5 条臂）

MovieLens-1M，5 窗口 × 120 轮，C=0.1，3 个种子，严格性检查 7/7 通过。

| 臂 | 窗口级 NDCG@10 | 末窗 NDCG@10 | `delta_rel` | `neighbor_lift` | 有边用户 | 图更新次数 |
|---|---|---|---|---|---|---|
| NoGraph | 0.2303 | 0.2376 | 0 | — | 0 | 0 |
| StaticGraph（冻结于第 21 轮） | 0.2291 | 0.2355 | 0.19 | **0.76** | 1275 | 1 |
| StaticGraph（冻结于第 120 轮） | 0.2291 | 0.2349 | 0.22 | 0.95 | 1429 | 100 |
| PeriodicGraph（每 120 轮刷新） | 0.2354 | 0.2454 | 0.97 | 1.34 | 5973 | 5 |
| DynamicGraph（每轮 EMA） | **0.2379** | **0.2479** | **1.18** | **1.56** | 6040 | 580 |

| 对照 | Δ NDCG@10 | 95% CI | 窗口正号 | 种子 | p |
|---|---|---|---|---|---|
| ① Static − NoGraph | −0.0012 | [−0.0022, −0.0003] | 33% | 0/3 | 0.026（量级可忽略） |
| ①b static_at120 − Static | +0.0001 | [−0.0002, +0.0003] | 47% | 2/3 | 0.64（无差异） |
| ② Dynamic − Static | +0.0088 | [+0.0047, +0.0131] | 80% | 3/3 | 0.0018 |
| ③ Dynamic − NoGraph | +0.0076 | [+0.0041, +0.0113] | 93% | 3/3 | 0.0015 |
| ④ Periodic − NoGraph | +0.0051 | [+0.0027, +0.0076] | **100%** | 3/3 | 0.0019 |
| ⑤ Dynamic − Periodic | +0.0025 | [+0.0012, +0.0039] | 73% | 3/3 | 0.0043 |

**一句话结论：动态图有明确贡献（+0.0076，相对 +3.3%），但贡献的大头来自「图持续被刷新、
能覆盖当前活跃用户」，而不是「每轮 EMA」。** 只刷新 5 次的 periodic 臂就拿到 67% 的总收益
（④=+0.0051，15/15 个窗口全为正）；每轮 EMA 的额外收益只有 +0.0025（⑤）—— 这部分才是
「自适应更新速度」真正能争取的空间。一次性冻结的图完全没用，把冻结时点从第 21 轮推到第
120 轮也救不回来（①b=+0.0001），因为它只能覆盖冻结时刻已上传摘要的用户（1275~1429/6040），
而且邻居质量随兴趣漂移掉到随机基线以下（`neighbor_lift` 0.76~0.95 < 1）。

**对研究问题的含义**：`delta_rel ≈ 1.2` 说明协同分支与序列表示同量级，「图分支被稀释」
的假设不成立；因此实验二里 ρ 不敏感不能归因于分支太弱，更可能是候选池机制下不同 ρ
收敛到相近的 Top-K 邻居集合。更新策略的价值主要在「要不要刷新 / 覆盖」而非「每轮 EMA 的
系数」。

## 12. 冻结归因：图更新机制是否合理？

**问题**：之前观察到的「图冻结」，是 EMA 本身（ρ 太大）造成的，还是新边初始化规则造成的？

**设计**：2×2 网格，ρ 固定 0.8（审计报告里观测到冻结的区间），每个格子 3 个种子：

|  | 新边初始化 `zero`（论文） | 新边初始化 `warm`（首次观测直接取 Δe） |
|---|---|---|
| 图状态 `pool`（保留候选池，修复版） | `dynamic_rho0.8` | `dynamic_rho0.8_warm` |
| 图状态 `topk`（只存 Top-K，审计前旧行为） | `dynamic_rho0.8_topk` | `dynamic_rho0.8_topk_warm` |

外加论文默认 `dynamic`（pool + zero + ρ=0.2）作参照。

**新增的冻结指标**（`dg_round_metrics.csv`）：相对「预热后第一次评测」的邻居快照算

| 列 | 含义 |
|---|---|
| `turnover_from_ref` | 当前 Top-K 邻居集合与快照的 Jaccard 距离均值（≈0 = 完全没变） |
| `frozen_share` | 邻居集合**一位都没变**的用户占比（≈1 = 整张图冻住） |
| `coverage_gain` | 相对快照新增了边覆盖的用户数 |

**判据**：topk 两格冻住、pool 两格不冻 ⇒ 冻结来自**状态保留规则**（只存 Top-K 会丢掉
掉出 Top-K 的边历史，新边每轮从 `(1-ρ)Δe` 重新起步、几乎挤不回来）；若 topk+warm
也解不开冻结 ⇒ **新边初始化不是原因**；若 pool+warm 只是转得更快 ⇒ 初始化规则影响的是
**适应速度**，可以量化它的收益。

跑 + 分析：

```bash
bash val_DynamicGraph/dg_run_supplement.sh   # 冻结归因 + Genre 两组一起跑（15 个 run，4 并发约 80~90 分钟）
python val_DynamicGraph/dg_freeze.py         # -> results/freeze/{dg_freeze_report.md,tables,figures}
```

## 13. Genre 协同有效性：兴趣相似是否意味着协同有效？

**问题**：当前基于「兴趣相似度」构建的邻居，是否真的有助于推荐？换句话说，
「邻居在 genre 上像自己」是否等于「换来了推荐收益」？

**数据**：ML-1M 官方 `movies.dat` 的 **18 个真实类型**，通过 `mappings.json` 的
`item2id` 还原内部 itemId → genre 多热向量（多标签按 1/|genres| 加权），
**物品覆盖率 100%（3416/3416）**，存在 `data/item_genres_ml1m.npz`。
注意：`exp_val_1026` 里 val02 用的是**共现伪类别**（`category_mode=cooccur`），
不是真 genre —— 这一步正好补上。

**逐用户指标**（`dg_genre_metrics.csv`，只在 `--genre-npz` 时产出）：

| 列 | 含义 |
|---|---|
| `genre_lift` / `genre_diff` | 邻居的 genre 余弦相似度 ÷（或 −）随机活跃用户基线 |
| `genrec_*` | 去掉「Drama/Comedy 人人都看」的公共成分后的版本（更干净的口味相似度） |
| `sem_lift` / `sem_diff` | 服务器兴趣摘要之间的语义相似度（对照：学到的相似度是不是 genre） |
| `genre_entropy` | 用户兴趣的分散程度 |
| `genre_drift` | 与上一窗口画像相比的 `1−cos`（兴趣变化幅度） |

再把 `dynamic_genre` 与 `nograph` 的**逐用户 NDCG** 对齐，得到逐用户收益
`ΔNDCG = dynamic − nograph`，然后看相似度与收益的关系（相关 + 五分位）。

**核心检验**：若 `ΔNDCG` 与 `genre_diff` 无关，而图确实有收益（前文已证），
则「兴趣相似 ⇒ 协同有效」不成立 —— 图的收益来自 genre 口味对齐之外的东西。

```bash
python val_DynamicGraph/dg_genre.py          # -> results/genre/{dg_genre_report.md,tables,figures}
python val_DynamicGraph/dg_genre_selftest.py  # 构造用例自检（秒级）：同群邻居 lift>1.5、异群<1
```

## 15. 最终验证报告（双数据集）

- **报告**：`实验验证报告_动态图_2026-11-06.md`（由 `dg_report.py` 从聚合表自动生成，不手抄数字）
- **覆盖**：MovieLens-1M + Amazon-Digital-Music 两个数据集，三个研究问题
- **验收**：`bash dg_verify.sh` —— 核验 run 数、四类聚合产出、严格性检查、报告章节完整性、图引用是否齐全

生成/刷新报告：

```bash
python -X utf8 dg_report.py          # 读 results/{aggregated,freeze,genre,quality} 生成报告
bash dg_verify.sh                    # 验收（应输出 ALL CHECKS PASSED）
```

全流程编排（Amazon 为例，均已封装）：

```bash
bash dg_run_amazon.sh        # 24 个 run（实验一 4 臂 + 实验二 2x2，各 3 种子），4 并发
bash dg_finish_all.sh        # 训练结束后自动跑全部聚合
bash dg_finalize.sh          # 聚合结束后自动生成报告并提交
```

## 14. 补充实验结论（2026-11-06，新增 15 次运行）

### 14.1 冻结归因（ρ=0.8，每格 3 种子）

| 格子 | turnover | frozen_share | ECR | 新边/轮 | `neighbor_lift` | NDCG@10 |
|---|---|---|---|---|---|---|
| pool + zero（修复版） | 0.962 | 0.02 | 0.106 | 1.20 | 1.557 | 0.2378 |
| pool + warm | 0.967 | 0.02 | 0.132 | 1.54 | 1.559 | 0.2378 |
| **topk + zero（旧行为）** | **0.165** | **0.48** | **0.0008** | **0.04** | **1.325** | 0.2377 |
| topk + warm | 0.967 | 0.02 | 0.140 | 1.66 | 1.560 | 0.2378 |
| 参照：pool+zero+ρ0.2 | — | — | 0.168 | 2.04 | 1.562 | 0.2379 |

**结论：冻结 = 「只存 Top-K 的旧状态规则」×「新边 0 初值」的联合效应，改任一项即解除。**

- ρ 从 0.2 提到 0.8 只让 ECR 从 0.168 降到 0.106（−0.062，p<1e-4）—— EMA 强度只影响「转多慢」，不会冻结；
- topk+zero 才会真冻：ECR≈0.0008、新边 0.04/轮、48% 用户的邻居集合一位不变；
- topk 下改用 warm 初值即可解冻（frozen_share 0.48→0.02，Δ=−0.46，p=1.8e-05）；
- 候选池下 warm 相对 zero 只多转 +0.005（p=0.07）—— 初始化规则本身影响很小。

**代价与警告**：冻结把 `neighbor_lift` 从 1.557 拉到 1.325（−0.232，p<1e-4），
但 **NDCG@10 只差 −0.0001（p=0.32，不显著）**。也就是说这个指标对「图不再适应」几乎不敏感，
报告图更新效果时必须同时给 `neighbor_lift` / ECR 这类结构指标。

### 14.2 Genre 协同有效性（真实 18 类，物品覆盖 100%）

| 指标 | 值 |
|---|---|
| 邻居 genre 余弦 vs 随机基线 | 0.789 vs 0.736 → **lift 1.08**（去公共成分后 diff 0.146，中位 lift 1.14） |
| 语义（摘要）lift | **3.25**（学到的相似度远比 genre 尖锐） |
| corr(genre_diff, sem_diff) | Spearman **0.368**（只是弱相关） |
| corr(ΔNDCG, genre_diff) | Spearman **+0.017**（p=0.008，个体层面极弱） |
| ΔNDCG 按 genre_diff 五分位 | Q1 0.0050 → Q5 **0.0167**（+0.0117 ≈ 全样本均值 0.0108 的 1.1 倍） |
| ΔNDCG 按语义 sem_diff 五分位 | Q1 0.0015 → Q5 **0.0220**（+0.0205，比 genre 更强） |
| ΔNDCG vs 兴趣漂移（本窗口行为 vs 之前历史） | Spearman **+0.0315**；五分位 Q1 0.0134 → Q5 0.0264 |

**结论：兴趣相似（genre 口径）不是协同有效的充分条件。** 图像确实挑了 genre 更像的邻居，
但幅度很弱（lift 1.08），且只能解释「收益幅度」的一部分、预测不了单个用户（Spearman 0.017）。
解释力更强的是**学到的语义相似度**（五分位跨度 0.0205）与**兴趣漂移**（drift 越大收益越大）——
后者正好是第二阶段「什么时候该更新关系」的切入点。

**口径提醒**：运行内的 `genre_drift` 是「相邻窗口累积画像」的 1−cos，而累积画像几乎不变
（中位 0.000），且用户在窗口前还没交互时会被误算成 1.0。`dg_genre.py` 已改为**离线重算**：
`drift = 1 − cos(本窗口行为, 之前历史)`，两者都非空才算，结果存 `tables/dg_genre_drift.csv`。

## 9. 环境提示

- 绘图：服务器已装 **matplotlib 3.11.2**（Agg 后端实测可出图），聚合时直接输出
  `figures/dg_fig_arms_<dataset>.png` —— 三联图：NDCG@10 曲线 / `delta_rel`（symlog，能显示 0）/
  `neighbor_lift`（含 1.0 随机基准线，无数据时标注 "no data in this run"）。
  若换到没有 matplotlib 的环境，聚合会跳过出图并提示，表格与报告不受影响。
- scipy 可选：装了就用 t 分布 / Wilcoxon，没装则用正态近似 + 精确符号检验（不依赖 scipy）。
- 磁盘：`/` 只剩约 27 GB。本实验每次运行的全部输出约 2~5 MB（主要是逐用户 CSV），
  9 次运行不到 50 MB，够用。
