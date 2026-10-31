# results/ 数据存放规则（验证实验 1）

本目录存放 `val01_*` 脚本跑出来的全部数据，分三类。

```
results/
├─ raw/           正式结果：每次运行一个目录，四级分类
│  └─ <dataset>/<mode>/p<rate>/seed_<seed>/
├─ aggregated/    聚合产物（由 val01_aggregate.py 生成）
│  ├─ figures/val01_fig_staleness_rate.png   ← 交付物：那张图
│  └─ tables/val01_figure_data.csv            ← 图的数据
│     tables/val01_verdict.csv                 ← 自动结论
└─ archive/       冒烟 / 试跑，不参与正式统计
   └─ pilot/      bash / pwsh 冒烟脚本的默认输出
```

## 分类依据

| 层级 | 取值 | 说明 |
|---|---|---|
| `dataset` | `movielens-1m`、`amazon-digital_music` | 数据集（空格替换为下划线） |
| `mode` | `upload_only`、`upload_fresh_edges` | 现状 vs 新鲜摘要对照 |
| `p<rate>` | `p1.00`、`p0.50`、`p0.30`、`p0.10` | 每轮客户端参与率 |
| `seed_<seed>` | `seed_42`、`seed_777`、`seed_2024` | 随机种子 |

示例：

```
results/raw/movielens-1m/upload_only/p0.10/seed_42/val01_staleness_buckets.csv
results/raw/amazon-digital_music/upload_fresh_edges/p0.30/seed_777/val01_summary.csv
```

## 单个 run 里有什么（4 个文件，约 6 KB）

| 文件 | 内容 |
|---|---|
| `config.json` | 完整超参数（可复现） |
| `val01_summary.csv` | 汇总（一行）：`avg_window_ndcg`、`mean_staleness`、`avg_ecr` 等 |
| `val01_staleness_buckets.csv` | 陈旧度 / 参与次数分桶 → **那张图的全部数据** |
| `val01_run.log` | 训练日志 |

## 约定

1. 聚合脚本默认只读 `raw/`，不会把 `archive/` 里的试跑混进统计；
2. 正式跑前建议先跑冒烟（`results/archive/pilot`），确认流程无误；
3. 想临时试跑：`--out-root results/archive/<自定义名字>`；
4. 每个 run 的 `val01_summary.csv` 是索引文件，聚合靠它发现 run；
   目录名与文件内的 `dataset / mode / participation_rate / seed` 一致，便于人工核对；
5. `*.log` 已被 `.gitignore` 忽略；`csv / json / png` 可纳入版本管理；
6. 备份只需 `raw/` 与 `aggregated/`：36 次运行合计约 0.3 MB。
