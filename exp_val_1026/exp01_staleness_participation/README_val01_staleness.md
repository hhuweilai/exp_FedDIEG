# 验证实验 1（val01）：客户端长期不参与，会不会影响推荐质量？

- 目录：`exp_val_1026/exp01_staleness_participation/`
- 对应任务书：`2026-10-26 动态联邦推荐研究问题验证任务书.md` 第三节「实验一」
- 只做诊断，不设计算法。**交付物是一张图**：
  `results/aggregated/figures/val01_fig_staleness_rate.png`
  （横轴 = 摘要陈旧程度，每条线 = 一个客户端参与率，纵轴 = 平均 NDCG@10）

## 1. 这一版做了什么（只保留核心）

| | 内容 |
|---|---|
| 模式 1 | `upload_only`：现状 FedDIEG，只有参与训练的客户端刷新摘要，陈旧摘要照样参与构图 |
| 模式 2 | `upload_fresh_edges`：对照，每轮给全部客户端重算摘要（诊断用 oracle 上界，不可部署） |
| 参与率 | 100% / 50% / 30% / 10%（模式 2 只在 30% / 10% 跑，因为 100% 时不存在陈旧） |
| 数据集 | MovieLens-1M + 一个 Amazon 数据集（`amazon-digital music`） |
| 种子 | 42 / 777 / 2024（3 个独立随机种子） |
| 规模 | 6 组 × 3 种子 × 2 数据集 = **36 次运行** |
| 产出 | 每次运行 **4 个文件约 6 KB**；聚合出 **1 张图 + 2 个小表** |

为什么必须有模式 2：任务书要求「固定相同的客户端训练参与序列，仅改变服务器是否使用陈旧摘要构建邻居」，
否则无法区分「摘要陈旧」和「训练机会变少」。两个模式共用同一份预生成参与序列（同一 seed 下逐轮一致）。

## 2. 与任务书「实验一」的符合性核对

| 任务书要求 | 本实现 | 状态 |
|---|---|---|
| 选择 MovieLens-1M 和一个 Amazon 数据集 | 默认两个都跑（`DATASETS` 可覆盖） | ✅ |
| 参与率 100% / 50% / 30% / 10% | 同 | ✅ |
| 每轮记录客户端上次上传时间、计算摘要陈旧程度 | `last_upload_round` 逐轮维护，`staleness = 当前轮 − 上次上传`，按桶写进 `val01_staleness_buckets.csv` | ✅ |
| 观察① 参与率降低后准确率是否下降 | 同一张图上四条实线的高度差 | ✅ |
| 观察② 摘要越陈旧用户效果是否越差 | 同一张图上每条线的斜率 + 桶表 | ✅ |
| 观察③ 陈旧摘要参与的边是否更易成为坏邻居 | **未做**（需要边级真值快照，属选做，已删） | ⚠️ 未覆盖 |
| 观察④ 图是否该更新没更新 / 不必要的更新 | **未做**（需要 lag 指标，属选做，已删） | ⚠️ 未覆盖 |
| 控制变量：固定参与序列、只改是否用陈旧摘要 | 两个模式共用同一份 schedule | ✅ |
| 不引入未来数据 | `upload_fresh_edges` 只用当前窗口已到达的交互重算摘要，不用未来窗口 | ✅ |
| 预期交付：一张「陈旧度 × 参与率 × 性能」图 | `val01_fig_staleness_rate.png` | ✅ |
| 3 个独立随机种子 | 同 | ✅ |
| 每组实验说明设置 / 控制变量 / 图表 / 现象 / 其他解释 / 结论 | 设置与控制变量见本文；现象与结论由 `val01_verdict.csv` 给出骨架，人工补写 | ✅ |
| 先在 2 个数据集验证、暂不要求四个 | 同 | ✅ |

> 说明：观察点③④ 属于「协同关系是否准确」的进阶证据，任务书把它们列在观察项里、但**预期交付只要一张图**。
> 按「只保留核心」的要求已删除相关代码（边精度、陈旧边 vs 新鲜边、更新滞后、逐用户快照、逐轮指标）。
> 如果之后要补，需要重新加上边级真值快照与 lag 测量——现在的图只能回答「性能层面有没有损失」。

## 3. 文件清单

```
exp_val_1026/exp01_staleness_participation/
├─ val01_run_staleness.py     单次运行（训练 + 陈旧度统计 + 分桶汇总）
├─ val01_aggregate.py         聚合：出那张图 + 自动结论
├─ val01_run_basic.sh         批次脚本（Linux）：36 次运行
├─ val01_run_basic.ps1        批次脚本（Windows）
├─ val01_run_pilot.sh/.ps1    冒烟（1 窗口 20 轮，验证流程）
├─ README_val01_staleness.md  本文件
└─ results/
   ├─ raw/                    <dataset>/<mode>/p<rate>/seed_<seed>/
   ├─ aggregated/             figures/ + tables/
   └─ archive/                冒烟/试跑
```

## 4. 怎么跑

```bash
# 1) 冒烟（可选，1 分钟）
bash exp_val_1026/exp01_staleness_participation/val01_run_pilot.sh

# 2) 正式跑（36 次运行；已完成的会自动跳过，可中断重跑）
bash exp_val_1026/exp01_staleness_participation/val01_run_basic.sh

# 3) 出图 + 结论（几秒）
python exp_val_1026/exp01_staleness_participation/val01_aggregate.py
```

Windows 把 1)、2) 换成：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\val01_run_pilot.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\val01_run_basic.ps1
```

常用变体：

```bash
DATASETS="movielens-1m" bash .../val01_run_basic.sh   # 只跑一个数据集（时间约减半）
SEEDS="42" ROUNDS=60 bash .../val01_run_basic.sh      # 先探路
```
```powershell
powershell ... -File .\val01_run_basic.ps1 -Datasets "movielens-1m"
powershell ... -File .\val01_run_basic.ps1 -Seeds "42" -Rounds 60
```

成本：MovieLens-1M 单次（5 窗口 × 120 轮）p=1.0 约 6 min、p=0.5 约 4 min、p=0.3 约 2.5 min、p=0.1 约 2 min；
Amazon 数据集用户更多（12381 个），单次略慢。两个数据集全量约 **2 小时**；只跑 MovieLens-1M 约 **50 分钟**。

## 5. 输出

每次运行（`results/raw/<dataset>/<mode>/p<rate>/seed_<seed>/`）：

| 文件 | 内容 | 体积 |
|---|---|---|
| `config.json` | 完整超参数（可复现） | ~2 KB |
| `val01_summary.csv` | 本次运行汇总（一行） | ~0.5 KB |
| `val01_staleness_buckets.csv` | 陈旧度/参与次数分桶 → **那张图的全部数据** | ~2 KB |
| `val01_run.log` | 训练日志 | ~2 KB |

聚合（`results/aggregated/`）：

| 文件 | 内容 |
|---|---|
| `figures/val01_fig_staleness_rate.png` | **任务书要求的那张图** |
| `tables/val01_figure_data.csv` | 图背后的数据（mode / rate / bucket / n_users / mean_ndcg） |
| `tables/val01_verdict.csv` | 自动结论判定 |

## 6. 怎么读这张图、怎么判断成立

图上能读四个量（以 bucket 0 为例）：

| 量 | 含义 |
|---|---|
| 四条实线的高度差 | **参与率效应**：参与率越低，效果越差多少 |
| 单条实线随横轴的斜率 | 自身陈旧度的**总效应**（含"上传少 = 训练机会少"的混杂） |
| 同参与率下实线与虚线的**垂直间距** | **陈旧信息的因果效应**：换成新鲜摘要能挽回多少 |
| 间距 ÷ 参与率效应 | 陈旧能解释的比例（是不是主因） |

判定规则（`val01_aggregate.py` 自动执行）：`|均值| > max(0.002, 种子标准差)` 且 3 个种子同号，
才算真实效应；种子数 < 2 一律判「无法判断」。

| 结论 | 含义 |
|---|---|
| 成立：陈旧信息确实造成损失 | 掉得多，且换成新鲜摘要能挽回 → 值得继续做动态图更新 |
| 部分成立 | 参与率确实有影响，但损失主要来自训练机会 |
| 不成立：参与率下降没有稳定影响 | 现象本身不成立（任务书停止信号 1） |
| 现象成立但缺少对照 | 只跑了 `upload_only` |
| 无法判断：种子数不足 | 少于 2 个种子 |

三个读图陷阱：

1. **别把横轴斜率当因果**——陈旧度与上传次数天然负相关，实线在下降可能是"训练机会少"；
   判断"信息陈旧"要看同桶上实线与虚线的间距。
2. **右端桶人很少**，会抖动；`val01_figure_data.csv` 里 `n_users < 100` 的点直接忽略。
3. **100% 参与率只有一个点**（陈旧度恒为 0），它只用来当"参与率效应"的基准，不参与斜率判断。

## 7. 已知局限（写报告要带上）

1. `upload_fresh_edges` 是不可部署的 oracle，只能作为"消除陈旧能挽回多少"的上界。
2. 陈旧度与上传次数负相关，逐用户层面看到的相关性主要是训练机会的效应；
   结论以聚合层面的配对差值（`val01_verdict.csv`）为准。
3. 本版**不覆盖**任务书的观察点③④（边级质量、更新滞后），只回答"性能层面有没有损失"。
4. 3 个种子只够看趋势，写论文建议扩到 5 个。
5. 本实验不涉及兴趣真实漂移（实验二）与短期扰动（实验三）。
