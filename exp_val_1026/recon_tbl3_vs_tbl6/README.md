# 表 3 与表 6 "默认设置"不一致的两次复现运行

本目录是 `研究问题验证报告_第一阶段_2026-11-04.md` §3.3 的证据，**不参与** exp01/02/03 的任何统计。

| 子目录 | 用途 | runner | 超参 | Avg Window NDCG@10（MovieLens-1M，seed 42） |
|---|---|---|---|---|
| `B2_exp03_tuned/` | 固定"持久 EMA 图"机制，只把可调超参换成表 3 的 tuned 值 | `exp_03_fast_gpu/run_exp03_fast_gpu.py` | lr 0.0012、graph_mix 0.02、confidence/agreement 关、score_local_mix 关、boundary_refresh 关、short_score_mix 0、neighbor_temp 0.05（warmup 与 final_loss 由 runner 固定，无法设成 100 / 0.0） | **0.41214** |
| `B3_exp02_base/` | 固定 BASE 超参，换成"无状态 kNN 图"机制 | `exp_02_fast_gpu/run_exp02_fast_gpu.py` | BASE_FEDDIEG（graph_mix 0.05、门控 0.5/0.5、score_local_mix 0.2、boundary_refresh 0.3、short_score_mix 0.05、neighbor_temp 0.07、lr 0.001、warmup 20、final_loss 0.2） | **0.39055** |

已存在的两个参照点（未复制，见原目录）：

- **B0**：`exp_05_sensitivity/final/movielens_1m/default/exp03_summary.csv` = **0.40915**（exp03 runner + BASE）
- **B1**：`exp_02_fast_gpu/significance_exp02/results/movielens-1m/seed_42/feddieg/exp02_summary.csv` = **0.42497**（exp02 runner + tuned；表 3 的 0.4224 ± 0.0028 是同一配置的 5 种子均值）

## 由此得到的单因素效应量

| 对照 | 只改变了什么 | Δ NDCG@10 |
|---|---|---|
| B2 − B0 | 超参（固定 exp03 机制） | +0.0030 |
| B1 − B3 | 超参（固定 exp02 机制） | +0.0344 |
| B0 − B3 | 机制 / runner（固定 BASE 超参） | +0.0186 |
| B1 − B0 | 机制 + 超参（= 两表实际报的东西） | +0.0158 |

结论：表 3（0.4224）与表 6（0.4091）的差异**不是统计口径**，而是"两套图实现 × 两套默认超参"的组合；
超参效应与机制存在强交互（同一套 tuned 超参在两种机制上分别值 +0.034 与 +0.003）。

复现命令见报告附录 B。
