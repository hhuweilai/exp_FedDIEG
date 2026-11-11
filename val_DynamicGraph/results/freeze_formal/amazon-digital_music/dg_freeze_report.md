# 冻结归因报告：动态图更新机制是否合理？（amazon-digital music）

同一序列编码器与联邦协议，只改图更新机制：图状态（pool / topk）× 新边初始化（zero / warm），ρ 固定 0.8；另加论文默认（pool+zero+ρ0.2）作参照。每个格子 3 个种子。

**一句话：** 冻结 = 「只存 Top-K」×「新边 0 初值」的联合效应，改任一项即可解除；topk 下改用 warm 初值可把冻结用户占比降 0.38（p=3.7e-05）；候选池下 warm 相对 zero 只多转 +0.002（几乎无差别）

| verdict_id | question | value | evidence | judgement |
|---|---|---|---|---|
| ① | 四种配置下图到底冻没冻住（ρ=0.8） |  | pool+zero: frozen_share=0.020, turnover=0.934, ECR=0.0726, 新边=0.79；pool+warm: frozen_share=0.020, turnover=0.935, ECR=0.0761, 新边=0.84；topk+zero: frozen_share=0.401, turnover=0.248, ECR=0.0026, 新边=0.04；topk+warm: frozen_share=0.020, turnover=0.937, ECR=0.0831, 新边=0.93 | frozen_share 越接近 1、turnover 越接近 0 就越冻 |
| ② | 冻结的根因：EMA / 新边初始化 / 状态保留规则 |  | pool+zero=不冻，pool+warm=不冻，topk+zero=冻，topk+warm=不冻 | 冻结**只出现在 topk + zero 这一个组合**：pool 的两格都不冻（ρ=0.8 也一样），topk+warm 也不冻。所以它既不是 EMA 强度单独造成的，也不是任一单独因素造成的，而是**「只存 Top-K 的旧状态规则」×「新边 0 初值」的联合效应**：topk 状态会丢弃掉出 Top-K 的边历史，而 zero 初值让新边只能以 (1-ρ)Δe 起步（ρ=0.8 时仅 20%），根本挤不进 Top-K；挤不进 ⇒ EMA 永不累积 ⇒ 永久冻结。把任一项改掉（pool 或 warm）就都不冻。 |
| ③ | 换 warm 初始化能不能解开冻结（topk 状态下） | -0.3811 | Δfrozen_share=-0.381 [-0.530, -0.236]，p=3.66e-05 | warm 初始化**几乎完全解开**了冻结：topk+warm 的 frozen_share 只剩 0.020，已与 pool 两格同量级 |
| ④ | 候选池下，warm 初始化让图转得快多少 | 0.0016 | Δturnover=+0.0016（p=0.000329）；Δ新边=+0.04（p=2.33e-11）；ΔNDCG=-0.0000（p=0.414） | 初始化规则对适应速度几乎没有影响 |
| ⑤ | ρ 从 0.2 提到 0.8（同为 pool+zero）改变了什么 | -0.0054 | Δturnover=-0.0054（p=0.00514）；ΔNDCG=+0.0000（p=0.259） | ρ 对 turnover 没有明显影响 |
| ⑥ | 各配置的推荐效果（窗口级 NDCG@10） |  | pool+zero ρ0.8: 0.0935；pool+warm ρ0.8: 0.0935；topk+zero ρ0.8: 0.0934；topk+warm ρ0.8: 0.0935；论文默认 pool+zero ρ0.2: 0.0935 | 冻结的图不仅不涨，还会因为邻居失真而略微拖低（见体检列） |
| ⑥b | 冻结的代价：邻居质量与推荐指标 |  | neighbor_lift：pool+zero=1.858 → topk+zero=1.521（-0.337）；窗口级 NDCG@10：0.0935 → 0.0934（-0.0001） | 冻结把邻居质量明显拉低，但 NDCG@10 几乎没变 —— 说明这个指标对「图不再适应」不敏感，评测时不能只看 NDCG，必须同时看 neighbor_lift / ECR 这类结构指标 |
| ⑦ | 一句话结论 |  | 冻结 = 「只存 Top-K」×「新边 0 初值」的联合效应，改任一项即可解除；topk 下改用 warm 初值可把冻结用户占比降 0.38（p=3.7e-05）；候选池下 warm 相对 zero 只多转 +0.002（几乎无差别） | 回答「更新机制是否合理」：现行实现的隐患是**旧状态规则 + 论文 0 初值的组合会让图永久冻结**；而冻结的代价主要落在邻居质量（lift 下降），NDCG@10 看不出来 |


## 各格子的冻结程度与效果

| cell | graph_state | graph_new_edge | rho | n_seeds | turnover_from_ref | frozen_share | ecr | added | n_edges | neighbor_lift | ndcg_win_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper_default_pool_zero_rho0.2 | pool | zero | 0.2000 | 5 | 0.9389 | 0.0200 | 0.1083 | 1.2412 | 12381 | 1.8640 | 0.0935 |
| pool_zero_rho0.8 | pool | zero | 0.8000 | 5 | 0.9335 | 0.0200 | 0.0726 | 0.7935 | 12381 | 1.8579 | 0.0935 |
| pool_warm_rho0.8 | pool | warm | 0.8000 | 5 | 0.9352 | 0.0200 | 0.0761 | 0.8371 | 12381 | 1.8620 | 0.0935 |
| topk_zero_rho0.8 | topk | zero | 0.8000 | 5 | 0.2477 | 0.4011 | 0.0026 | 0.0396 | 12381 | 1.5208 | 0.0934 |
| topk_warm_rho0.8 | topk | warm | 0.8000 | 5 | 0.9366 | 0.0200 | 0.0831 | 0.9314 | 12381 | 1.8626 | 0.0935 |


## 成对比较

| contrast | what | metric | n_pairs | delta_mean | ci95_lo | ci95_hi | p_paired | seeds_positive | seeds_total |
|---|---|---|---|---|---|---|---|---|---|
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | ndcg_win_mean | 25 | -1.11e-04 | -3.39e-04 | 1.09e-04 | 0.3407 | 2 | 5 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | turnover | 25 | -0.6859 | -0.7537 | -0.6209 | 4.23e-16 | 0 | 5 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | frozen | 25 | 0.3811 | 0.2359 | 0.5296 | 3.66e-05 | 5 | 5 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | added | 25 | -0.7539 | -0.7888 | -0.7238 | 3.49e-24 | 0 | 5 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | ndcg_win_mean | 25 | -1.65e-06 | -4.46e-05 | 4.34e-05 | 0.9448 | 4 | 5 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | turnover | 25 | 0.0015 | 3.49e-04 | 0.0026 | 0.0195 | 5 | 5 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | frozen | 25 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 5 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | added | 25 | 0.0943 | 0.0887 | 0.0998 | 4.21e-21 | 5 | 5 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | ndcg_win_mean | 25 | 8.73e-05 | -1.28e-04 | 3.11e-04 | 0.4471 | 3 | 5 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | turnover | 25 | 0.6890 | 0.6234 | 0.7574 | 4.60e-16 | 5 | 5 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | frozen | 25 | -0.3811 | -0.5296 | -0.2359 | 3.66e-05 | 0 | 5 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | added | 25 | 0.8918 | 0.8508 | 0.9368 | 4.26e-23 | 5 | 5 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | ndcg_win_mean | 25 | -2.20e-05 | -6.98e-05 | 2.80e-05 | 0.4142 | 2 | 5 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | turnover | 25 | 0.0016 | 8.84e-04 | 0.0024 | 3.29e-04 | 5 | 5 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | frozen | 25 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 5 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | added | 25 | 0.0436 | 0.0371 | 0.0507 | 2.33e-11 | 5 | 5 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | ndcg_win_mean | 25 | 2.64e-05 | -1.71e-05 | 6.92e-05 | 0.2590 | 4 | 5 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | turnover | 25 | -0.0054 | -0.0089 | -0.0022 | 0.0051 | 0 | 5 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | frozen | 25 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 5 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | added | 25 | -0.4477 | -0.4711 | -0.4262 | 1.01e-22 | 0 | 5 |


## 怎么读

- `turnover_from_ref` / `frozen_share` 都是相对**预热后第一次评测**的邻居快照算的；前者是 Jaccard 距离均值，后者是「一位都没变」的用户占比。
- `topk` 复现的是审计前的旧行为（只保留当前 Top-K 的 EMA，掉出去的边状态被丢掉）；`pool` 是修复版（保留候选池）。
- 若 topk 冻住而 pool 不冻 ⇒ **冻结是状态保留规则造成的**，既不是 EMA 强度、也不是初始化规则；后两者只影响「转多快」。
