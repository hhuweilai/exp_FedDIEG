# val_DynamicGraph 报告：动态图到底有没有明确贡献？

同一序列编码器、同一训练协议、同一份种子与参与序列，只改图分支：`dynamic`, `dynamic_genre`, `dynamic_mix0.005`, `dynamic_mix0.05`, `dynamic_rho0.8`, `dynamic_rho0.8_topk`, `dynamic_rho0.8_topk_warm`, `dynamic_rho0.8_warm`, `fresh_only`, `nograph`, `onboarding`, `periodic`, `periodic_boundary`, `periodic_p20`, `periodic_p240`, `static`, `static_at120`, `static_build120`。

## amazon-digital music

**一句话结论：** **动态图有明确贡献（③=+0.0183，窗口正号 100%、种子 5/5 为正），但贡献的大头来自「图持续被刷新、能覆盖当前活跃用户」，而不是「每轮 EMA」**：每 120 轮才刷新一次的 periodic 臂只用了 5 次刷新，就拿到 ④=+0.0147（占总收益 81%，窗口正号 100%、种子 5/5 为正）；每轮 EMA 相对「每窗口才刷新一次」还有小幅但一致的额外收益（⑤=+0.0035，窗口正号 100%、种子 5/5 为正，p=4.87e-09）—— 这部分才是「自适应更新速度」真正能争取的空间。一次性冻结的图完全没用，原因是它只能覆盖冻结时刻已上传摘要的用户（1275~1429/6040），而且邻居质量随兴趣漂移掉到随机基线以下（见 ⑦）。

| verdict_id | question | value | evidence | judgement |
|---|---|---|---|---|
| ① | 图协同分支本身有没有贡献（StaticGraph − NoGraph） | 0.0047 | Δ=+0.0047 [+0.0035, +0.0059]，n=25，正号比例 0.96，种子 5/5 为正，p_paired=2.03e-07，p_sign=1.55e-06 | |Δ| ∈ [0.002, 0.006)（小）；p<0.01 |
| ② | 动态更新有没有额外贡献（DynamicGraph − StaticGraph） | 0.0136 | Δ=+0.0136 [+0.0109, +0.0162]，n=25，正号比例 1.00，种子 5/5 为正，p_paired=8.9e-10，p_sign=5.96e-08 | |Δ| ≥ 0.006（明确）；p<0.01 |
| ③ | 动态图的总贡献（DynamicGraph − NoGraph） | 0.0183 | Δ=+0.0183 [+0.0159, +0.0207]，n=25，正号比例 1.00，种子 5/5 为正，p_paired=1.82e-13，p_sign=5.96e-08 | |Δ| ≥ 0.006（明确）；p<0.01 |
| ④ | 补充：慢刷新静态图是否已足够（PeriodicGraph − NoGraph） | 0.0147 | Δ=+0.0147 [+0.0130, +0.0165]，n=25，正号比例 1.00，种子 5/5 为正，p_paired=2.4e-14，p_sign=5.96e-08 | |Δ| ≥ 0.006（明确）；p<0.01 |
| ⑤ | 补充：每轮 EMA 比慢刷新多给多少（DynamicGraph − PeriodicGraph） | 0.0035 | Δ=+0.0035 [+0.0028, +0.0043]，n=25，正号比例 1.00，种子 5/5 为正，p_paired=4.87e-09，p_sign=5.96e-08 | |Δ| ∈ [0.002, 0.006)（小）；p<0.01 |
| ①b | 补充：把冻结时点从第 21 轮推到第 120 轮能否救回静态图（static_at120 − StaticGraph） |  | 没有对应结果 | 未运行该臂 / 该对照 |
| ①b2 | 补充：第 120 轮一次性建图是否比第 21 轮建图好（只看窗口 2–5）（static_build120 − StaticGraph） |  | 没有对应结果 | 未运行该臂 / 该对照 |
| ⑥ | 图分支把最终用户表示挪动了多少（delta_rel = ‖Δ‖ / ‖seq_emb‖） | 1.5727 | NoGraph=0.00e+00，StaticGraph=1.58e-01，DynamicGraph=1.57e+00 | NoGraph 恒为 0（控制生效）；动态图臂约 1.57e+00（≥0.5），图分支与序列表示同量级甚至更大 —— 协同分支是主导项之一，「图分支被稀释」的假设不成立。 |
| ⑦ | 图里的邻居到底有没有信息（neighbor_lift = 邻居余弦 / 随机用户余弦） | 1.8640 | StaticGraph lift=0.863（边 2914），static_at120 lift=nan（边 nan），PeriodicGraph lift=1.646（边 12212），DynamicGraph lift=1.864（边 12381） | lift=1.86 > 1：Top-K 邻居确实比随机用户更像自己（图有区分度）；对照的静态臂 lift 只有 0.86，低于随机基线 |
| ⑧ | 严格性检查（协议一致 / 空图 / 冻结 / 每轮更新 / 评测对齐） | 10 | 全部通过 | 可以信任上面的差值 |
| ⑨ | 总体结论 |  | **动态图有明确贡献（③=+0.0183，窗口正号 100%、种子 5/5 为正），但贡献的大头来自「图持续被刷新、能覆盖当前活跃用户」，而不是「每轮 EMA」**：每 120 轮才刷新一次的 periodic 臂只用了 5 次刷新，就拿到 ④=+0.0147（占总收益 81%，窗口正号 100%、种子 5/5 为正）；每轮 EMA 相对「每窗口才刷新一次」还有小幅但一致的额外收益（⑤=+0.0035，窗口正号 100%、种子 5/5 为正，p=4.87e-09）—— 这部分才是「自适应更新速度」真正能争取的空间。一次性冻结的图完全没用，原因是它只能覆盖冻结时刻已上传摘要的用户（1275~1429/6040），而且邻居质量随兴趣漂移掉到随机基线以下（见 ⑦）。 | 这是本实验要回答的那句话 |


<details><summary>逐臂汇总（跨种子平均）</summary>

| arm_dir | n_seeds | ndcg_win_mean | ndcg_win_last | delta_rel | neighbor_lift | ecr | n_edges | graph_updates |
|---|---|---|---|---|---|---|---|---|
| dynamic | 5 | 0.0935 | 0.1092 | 1.5727 | 1.8640 | 0.1083 | 12381 | 580 |
| dynamic_rho0.8 | 5 | 0.0935 | 0.1092 | 1.5708 | 1.8579 | 0.0726 | 12381 | 580 |
| dynamic_rho0.8_topk | 5 | 0.0934 | 0.1091 | 1.5533 | 1.5208 | 0.0026 | 12381 | 580 |
| dynamic_rho0.8_topk_warm | 5 | 0.0935 | 0.1092 | 1.5727 | 1.8626 | 0.0831 | 12381 | 580 |
| dynamic_rho0.8_warm | 5 | 0.0935 | 0.1092 | 1.5730 | 1.8620 | 0.0761 | 12381 | 580 |
| fresh_only | 5 | 0.0934 | 0.1091 | 1.5624 | 1.8442 | 0.4122 | 12381 | 580 |
| nograph | 5 | 0.0752 | 0.0836 | 0.0000 | nan | 0.0000 | 0 | 0 |
| onboarding | 5 | 0.0934 | 0.1090 | 1.5466 | 1.4077 | 0.4067 | 12381 | 277 |
| periodic | 5 | 0.0899 | 0.1055 | 1.2859 | 1.6465 | 0.0000 | 12212 | 5 |
| static | 5 | 0.0799 | 0.0917 | 0.1584 | 0.8633 | 0.0000 | 2914 | 1 |


</details>

<details><summary>严格性检查</summary>

| check | ok | detail |
|---|---|---|
| 协议参数一致（除图以外没有别的差别） | 1 | 一致 |
| 图状态（实验变量，允许不同） | 1 | pool、topk |
| 新边初始化（实验变量，允许不同） | 1 | warm、zero |
| 个性化层位置（仅性能开关，不改数值） | 1 | personal_on_cpu=0（实测同 seed 下 cpu/gpu 驻留的逐用户 NDCG 与逐轮指标逐位一致，所以混用不影响可比性） |
| ECR 定义版本（信息项，不参与失败判定） | 1 | v2_empty_to_full |
| 每条臂的种子数一致 | 1 | dynamic:5, dynamic_rho0.8:5, dynamic_rho0.8_topk:5, dynamic_rho0.8_topk_warm:5, dynamic_rho0.8_warm:5, fresh_only:5, nograph:5, onboarding:5, periodic:5, static:5（诊断臂的种子数允许不同） |
| NoGraph 的图分支恒为 0 | 1 | max delta_rel=0.000e+00（要求 ≤ 1e-12） |
| StaticGraph 冻结后边不再变化 | 1 | edge_changed_after_freeze=0（要求 0） |
| DynamicGraph 预热后每轮都更新 | 1 | max|updates-(global_round-20)|=0 |
| 同一窗口内各臂评测用户数一致 | 1 | 逐窗口跨臂最大差=0（各窗口人数不同是正常的：3178~3657 人） |


</details>

## movielens-1m

**一句话结论：** **动态图有明确贡献（③=+0.0099，窗口正号 96%、种子 5/5 为正），但贡献的大头来自「图持续被刷新、能覆盖当前活跃用户」，而不是「每轮 EMA」**：每 120 轮才刷新一次的 periodic 臂只用了 5 次刷新，就拿到 ④=+0.0063（占总收益 63%，窗口正号 88%、种子 5/5 为正）；每轮 EMA 相对「每窗口才刷新一次」还有小幅但一致的额外收益（⑤=+0.0036，窗口正号 80%、种子 5/5 为正，p=6.05e-05）—— 这部分才是「自适应更新速度」真正能争取的空间。把冻结时点从第 21 轮推到第 120 轮也救不回来（①b=+0.0001，无差异）；一次性冻结的图完全没用，原因是它只能覆盖冻结时刻已上传摘要的用户（1275~1429/6040），而且邻居质量随兴趣漂移掉到随机基线以下（见 ⑦）。

| verdict_id | question | value | evidence | judgement |
|---|---|---|---|---|
| ① | 图协同分支本身有没有贡献（StaticGraph − NoGraph） | -0.0013 | Δ=-0.0013 [-0.0022, -0.0005]，n=25，正号比例 0.28，种子 0/5 为正，p_paired=0.00564，p_sign=0.0433 | |Δ| < 0.002（可忽略）；p<0.01 |
| ② | 动态更新有没有额外贡献（DynamicGraph − StaticGraph） | 0.0112 | Δ=+0.0112 [+0.0073, +0.0152]，n=25，正号比例 0.80，种子 5/5 为正，p_paired=1.92e-05，p_sign=0.00408 | |Δ| ≥ 0.006（明确）；p<0.01 |
| ③ | 动态图的总贡献（DynamicGraph − NoGraph） | 0.0099 | Δ=+0.0099 [+0.0064, +0.0135]，n=25，正号比例 0.96，种子 5/5 为正，p_paired=1.84e-05，p_sign=1.55e-06 | |Δ| ≥ 0.006（明确）；p<0.01 |
| ④ | 补充：慢刷新静态图是否已足够（PeriodicGraph − NoGraph） | 0.0063 | Δ=+0.0063 [+0.0039, +0.0087]，n=25，正号比例 0.88，种子 5/5 为正，p_paired=4.86e-05，p_sign=0.000157 | |Δ| ≥ 0.006（明确）；p<0.01 |
| ⑤ | 补充：每轮 EMA 比慢刷新多给多少（DynamicGraph − PeriodicGraph） | 0.0036 | Δ=+0.0036 [+0.0023, +0.0051]，n=25，正号比例 0.80，种子 5/5 为正，p_paired=6.05e-05，p_sign=0.00408 | |Δ| ∈ [0.002, 0.006)（小）；p<0.01 |
| ①b | 补充：把冻结时点从第 21 轮推到第 120 轮能否救回静态图（static_at120 − StaticGraph） | 1.48e-04 | Δ=+0.0001 [-0.0001, +0.0004]，n=25，正号比例 0.56，种子 3/5 为正，p_paired=0.209，p_sign=0.69 | |Δ| < 0.002（可忽略）；不显著（p≥0.05） |
| ①b2 | 补充：第 120 轮一次性建图是否比第 21 轮建图好（只看窗口 2–5）（static_build120 − StaticGraph） | 0.0014 | Δ=+0.0014 [+0.0006, +0.0022]，n=20，正号比例 0.80，种子 4/5 为正，p_paired=0.00285，p_sign=0.0118（窗口 2–5 子集；该臂在 W1 无图） | |Δ| < 0.002（可忽略）；p<0.01 |
| ⑥ | 图分支把最终用户表示挪动了多少（delta_rel = ‖Δ‖ / ‖seq_emb‖） | 1.3261 | NoGraph=0.00e+00，StaticGraph=2.14e-01，DynamicGraph=1.33e+00 | NoGraph 恒为 0（控制生效）；动态图臂约 1.33e+00（≥0.5），图分支与序列表示同量级甚至更大 —— 协同分支是主导项之一，「图分支被稀释」的假设不成立。 |
| ⑦ | 图里的邻居到底有没有信息（neighbor_lift = 邻居余弦 / 随机用户余弦） | 1.6959 | StaticGraph lift=0.807（边 1280），static_at120 lift=1.003（边 1429），PeriodicGraph lift=1.460（边 5977），DynamicGraph lift=1.696（边 6040） | lift=1.70 > 1：Top-K 邻居确实比随机用户更像自己（图有区分度）；对照的静态臂 lift 只有 0.81，低于随机基线 |
| ⑧ | 严格性检查（协议一致 / 空图 / 冻结 / 每轮更新 / 评测对齐） | 10 | 全部通过 | 可以信任上面的差值 |
| ⑨ | 总体结论 |  | **动态图有明确贡献（③=+0.0099，窗口正号 96%、种子 5/5 为正），但贡献的大头来自「图持续被刷新、能覆盖当前活跃用户」，而不是「每轮 EMA」**：每 120 轮才刷新一次的 periodic 臂只用了 5 次刷新，就拿到 ④=+0.0063（占总收益 63%，窗口正号 88%、种子 5/5 为正）；每轮 EMA 相对「每窗口才刷新一次」还有小幅但一致的额外收益（⑤=+0.0036，窗口正号 80%、种子 5/5 为正，p=6.05e-05）—— 这部分才是「自适应更新速度」真正能争取的空间。把冻结时点从第 21 轮推到第 120 轮也救不回来（①b=+0.0001，无差异）；一次性冻结的图完全没用，原因是它只能覆盖冻结时刻已上传摘要的用户（1275~1429/6040），而且邻居质量随兴趣漂移掉到随机基线以下（见 ⑦）。 | 这是本实验要回答的那句话 |


<details><summary>逐臂汇总（跨种子平均）</summary>

| arm_dir | n_seeds | ndcg_win_mean | ndcg_win_last | delta_rel | neighbor_lift | ecr | n_edges | graph_updates |
|---|---|---|---|---|---|---|---|---|
| dynamic | 5 | 0.2396 | 0.2506 | 1.3261 | 1.6959 | 0.1238 | 6040 | 580 |
| dynamic_genre | 5 | 0.2396 | 0.2506 | 1.3261 | 1.6959 | 0.1238 | 6040 | 580 |
| dynamic_mix0.005 | 1 | 0.2309 | 0.2380 | 0.0304 | 1.5222 | 0.1254 | 6040 | 580 |
| dynamic_mix0.05 | 1 | 0.2316 | 0.2389 | 0.1749 | 1.5619 | 0.1234 | 6040 | 580 |
| dynamic_rho0.8 | 5 | 0.2395 | 0.2505 | 1.3255 | 1.6902 | 0.0838 | 6040 | 580 |
| dynamic_rho0.8_topk | 5 | 0.2389 | 0.2494 | 1.3161 | 1.4549 | 0.0022 | 6040 | 580 |
| dynamic_rho0.8_topk_warm | 5 | 0.2396 | 0.2503 | 1.3266 | 1.6944 | 0.1002 | 6040 | 580 |
| dynamic_rho0.8_warm | 5 | 0.2396 | 0.2504 | 1.3256 | 1.6931 | 0.0924 | 6040 | 580 |
| fresh_only | 5 | 0.2396 | 0.2508 | 1.3247 | 1.6730 | 0.4438 | 6040 | 580 |
| nograph | 5 | 0.2297 | 0.2374 | 0.0000 | nan | 0.0000 | 0 | 0 |
| onboarding | 5 | 0.2389 | 0.2492 | 1.3271 | 1.3856 | 0.3300 | 6040 | 240 |
| periodic | 5 | 0.2360 | 0.2466 | 1.0977 | 1.4600 | 0.0000 | 5977 | 5 |
| periodic_boundary | 5 | 0.2302 | 0.2373 | 0.5385 | 1.3171 | 0.0000 | 5461 | 5 |
| periodic_p20 | 3 | 0.2368 | 0.2475 | 1.1689 | 1.5687 | 0.0000 | 6040 | 29 |
| periodic_p240 | 3 | 0.2312 | 0.2388 | 0.6428 | 1.2279 | 0.0000 | 5973 | 3 |
| static | 5 | 0.2284 | 0.2346 | 0.2140 | 0.8066 | 0.0000 | 1280 | 1 |
| static_at120 | 5 | 0.2285 | 0.2347 | 0.2466 | 1.0030 | 0.0261 | 1429 | 100 |
| static_build120 | 5 | 0.2293 | 0.2369 | 0.0712 | 1.1114 | 0.0167 | 1429 | 1 |


</details>

<details><summary>严格性检查</summary>

| check | ok | detail |
|---|---|---|
| 协议参数一致（除图以外没有别的差别） | 1 | 一致 |
| 图状态（实验变量，允许不同） | 1 | pool、topk |
| 新边初始化（实验变量，允许不同） | 1 | warm、zero |
| 个性化层位置（仅性能开关，不改数值） | 1 | personal_on_cpu=0（实测同 seed 下 cpu/gpu 驻留的逐用户 NDCG 与逐轮指标逐位一致，所以混用不影响可比性） |
| ECR 定义版本（信息项，不参与失败判定） | 1 | v2_empty_to_full |
| 每条臂的种子数一致 | 1 | dynamic:5, dynamic_genre:5, dynamic_mix0.005:1, dynamic_mix0.05:1, dynamic_rho0.8:5, dynamic_rho0.8_topk:5, dynamic_rho0.8_topk_warm:5, dynamic_rho0.8_warm:5, fresh_only:5, nograph:5, onboarding:5, periodic:5, periodic_boundary:5, periodic_p20:3, periodic_p240:3, static:5, static_at120:5, static_build120:5（诊断臂的种子数允许不同） |
| NoGraph 的图分支恒为 0 | 1 | max delta_rel=0.000e+00（要求 ≤ 1e-12） |
| StaticGraph 冻结后边不再变化 | 1 | edge_changed_after_freeze=0（要求 0） |
| DynamicGraph 预热后每轮都更新 | 1 | max|updates-(global_round-20)|=0 |
| 同一窗口内各臂评测用户数一致 | 1 | 逐窗口跨臂最大差=0（各窗口人数不同是正常的：1421~1763 人） |


</details>

## 逐用户配对（敏感性检查）

| dataset | contrast | n_users | delta_mean | delta_median | ci95_lo | ci95_hi | win_rate | p_sign | p_wilcoxon |
|---|---|---|---|---|---|---|---|---|---|
| amazon-digital music | static_minus_nograph | 126440 | 0.0044 | 0.0000 | 0.0039 | 0.0049 | 0.0613 | 1.47e-11 | 2.68e-44 |
| amazon-digital music | dynamic_minus_static | 126440 | 0.0174 | 0.0000 | 0.0166 | 0.0182 | 0.1156 | 3.83e-313 | 0.0000 |
| amazon-digital music | dynamic_minus_nograph | 126440 | 0.0218 | 0.0000 | 0.0209 | 0.0228 | 0.1299 | 7.89e-300 | 0.0000 |
| amazon-digital music | periodic_minus_nograph | 126440 | 0.0181 | 0.0000 | 0.0172 | 0.0190 | 0.1174 | 5.92e-219 | 0.0000 |
| amazon-digital music | dynamic_minus_periodic | 126440 | 0.0038 | 0.0000 | 0.0034 | 0.0042 | 0.0592 | 1.32e-119 | 1.48e-111 |
| amazon-digital music | onboarding_minus_static | 126440 | 0.0168 | 0.0000 | 0.0160 | 0.0177 | 0.1081 | 1.06e-252 | 0.0000 |
| amazon-digital music | dynamic_minus_onboarding | 126440 | 5.57e-04 | 0.0000 | 2.52e-04 | 8.96e-04 | 0.0486 | 4.94e-15 | 6.35e-09 |
| amazon-digital music | fresh_only_minus_static | 126440 | 0.0174 | 0.0000 | 0.0166 | 0.0183 | 0.1151 | 8.29e-311 | 0.0000 |
| amazon-digital music | dynamic_minus_fresh_only | 126440 | -3.89e-05 | 0.0000 | -1.99e-04 | 1.06e-04 | 0.0171 | 0.2973 | 0.9744 |
| movielens-1m | dynamic_genre_minus_dynamic | 39490 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| movielens-1m | dynamic_mix0.005_minus_dynamic | 7898 | -0.0122 | 0.0000 | -0.0153 | -0.0091 | 0.1151 | 4.25e-06 | 2.51e-11 |
| movielens-1m | dynamic_mix0.05_minus_dynamic | 7898 | -0.0112 | 0.0000 | -0.0142 | -0.0084 | 0.1104 | 2.04e-06 | 2.78e-11 |
| amazon-digital music | dynamic_rho0.8_minus_dynamic | 126440 | -2.82e-05 | 0.0000 | -1.72e-04 | 1.31e-04 | 0.0154 | 0.0798 | 0.3424 |
| amazon-digital music | dynamic_rho0.8_topk_minus_dynamic | 126440 | -4.44e-04 | 0.0000 | -7.56e-04 | -1.61e-04 | 0.0378 | 4.66e-13 | 3.29e-07 |
| amazon-digital music | dynamic_rho0.8_topk_warm_minus_dynamic | 126440 | -8.28e-05 | 0.0000 | -2.25e-04 | 6.72e-05 | 0.0147 | 0.3530 | 0.2461 |
| amazon-digital music | dynamic_rho0.8_warm_minus_dynamic | 126440 | -7.47e-05 | 0.0000 | -2.19e-04 | 8.14e-05 | 0.0149 | 0.1383 | 0.2192 |
| movielens-1m | periodic_boundary_minus_periodic | 39490 | -0.0096 | 0.0000 | -0.0106 | -0.0086 | 0.0671 | 1.07e-58 | 1.96e-73 |
| movielens-1m | periodic_p20_minus_periodic | 23694 | 0.0019 | 0.0000 | 0.0011 | 0.0027 | 0.0587 | 8.17e-12 | 2.52e-10 |
| movielens-1m | periodic_p240_minus_periodic | 23694 | -0.0069 | 0.0000 | -0.0081 | -0.0058 | 0.0560 | 3.88e-25 | 5.96e-32 |
| movielens-1m | static_at120_minus_static | 39490 | 1.88e-04 | 0.0000 | -2.15e-04 | 5.89e-04 | 0.0283 | 0.5950 | 0.2098 |
| movielens-1m | static_build120_minus_static | 39490 | 0.0024 | 0.0000 | 0.0017 | 0.0031 | 0.0716 | 1.76e-25 | 7.79e-21 |


## 怎么读这几张表

- 主证据是 **窗口级配对差值**（n = 种子数 × 窗口数）。阈值 `0.002` NDCG@10：小于它就算可忽略 —— 6040 个用户的配对检验能把 1e-4 的差异测成显著，所以别只看 p 值。
- ① StaticGraph − NoGraph：图分支本身值多少。这个 ≈0 时，后面所有更新策略的讨论都没有意义。
- ② DynamicGraph − StaticGraph：动态度值多少。这是「动态图有必要吗」的直接答案。
- ⑥ `delta_rel`：图分支把最终用户表示挪动了多少。它是 ①② 的剂量：剂量太小，任何图机制都不可能有贡献。
- ⑦ `neighbor_lift`：图里的邻居比随机用户更像自己多少。≈1 说明图没有区分度。
- ⑧ 严格性：NoGraph 分支恒为 0、Static 冻结后不变、Dynamic 每轮都更新、各臂评测对齐。

- 图：

  - `/home/wl/exp_FedDIEG/val_DynamicGraph/results/agg_formal/figures/dg_fig_arms_amazon-digital_music.png`

  - `/home/wl/exp_FedDIEG/val_DynamicGraph/results/agg_formal/figures/dg_fig_arms_movielens-1m.png`

## 已知局限

1. `static` 臂的图建在预热结束那一轮（默认第 `graph_warmup_rounds + 1` 轮），那时摘要只更新过一轮，图的质量偏弱。若 ① 为负，别直接说「静态图有害」：先试 `--freeze-round 120`（冻结得更晚但依然不动态）或 `--arm periodic`（每窗口刷新一次）。
2. `dynamic` 与 `static` 共享预热期的全部更新，所以 ② 是纯粹的「之后要不要继续更新」，不含「图建得好不好」。
3. 本实验关掉了 L_struct（`lambda_struct=0`）：它定义在图量 Δe 上，NoGraph 臂拿不到，留着会变成「图臂多一个训练信号」的混淆项。要回到论文口径加 `--lambda-struct 1e-5`。
4. CPU 与 GPU 的 dropout 随机流不同，同一组对照必须固定在同一台机器上跑完（跨机器同 seed 的 CUDA 结果本身是逐位一致的）。
