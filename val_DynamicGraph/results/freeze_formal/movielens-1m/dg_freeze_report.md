# 冻结归因报告：动态图更新机制是否合理？（movielens-1m）

同一序列编码器与联邦协议，只改图更新机制：图状态（pool / topk）× 新边初始化（zero / warm），ρ 固定 0.8；另加论文默认（pool+zero+ρ0.2）作参照。每个格子 3 个种子。

**一句话：** 冻结 = 「只存 Top-K」×「新边 0 初值」的联合效应，改任一项即可解除；topk 下改用 warm 初值可把冻结用户占比降 0.48（p=5e-09）；候选池下 warm 相对 zero 只多转 +0.004（几乎无差别）

| verdict_id | question | value | evidence | judgement |
|---|---|---|---|---|
| ① | 四种配置下图到底冻没冻住（ρ=0.8） |  | pool+zero: frozen_share=0.020, turnover=0.951, ECR=0.0838, 新边=0.92；pool+warm: frozen_share=0.020, turnover=0.955, ECR=0.0924, 新边=1.03；topk+zero: frozen_share=0.496, turnover=0.161, ECR=0.0022, 新边=0.04；topk+warm: frozen_share=0.020, turnover=0.955, ECR=0.1002, 新边=1.14 | frozen_share 越接近 1、turnover 越接近 0 就越冻 |
| ② | 冻结的根因：EMA / 新边初始化 / 状态保留规则 |  | pool+zero=不冻，pool+warm=不冻，topk+zero=冻，topk+warm=不冻 | 冻结**只出现在 topk + zero 这一个组合**：pool 的两格都不冻（ρ=0.8 也一样），topk+warm 也不冻。所以它既不是 EMA 强度单独造成的，也不是任一单独因素造成的，而是**「只存 Top-K 的旧状态规则」×「新边 0 初值」的联合效应**：topk 状态会丢弃掉出 Top-K 的边历史，而 zero 初值让新边只能以 (1-ρ)Δe 起步（ρ=0.8 时仅 20%），根本挤不进 Top-K；挤不进 ⇒ EMA 永不累积 ⇒ 永久冻结。把任一项改掉（pool 或 warm）就都不冻。 |
| ③ | 换 warm 初始化能不能解开冻结（topk 状态下） | -0.4764 | Δfrozen_share=-0.476 [-0.579, -0.373]，p=5.04e-09 | warm 初始化**几乎完全解开**了冻结：topk+warm 的 frozen_share 只剩 0.020，已与 pool 两格同量级 |
| ④ | 候选池下，warm 初始化让图转得快多少 | 0.0042 | Δturnover=+0.0042（p=0.00952）；Δ新边=+0.11（p=7.65e-11）；ΔNDCG=+0.0001（p=0.0716） | 初始化规则对适应速度几乎没有影响 |
| ⑤ | ρ 从 0.2 提到 0.8（同为 pool+zero）改变了什么 | -0.0055 | Δturnover=-0.0055（p=0.013）；ΔNDCG=-0.0001（p=0.0695） | ρ 对 turnover 没有明显影响 |
| ⑥ | 各配置的推荐效果（窗口级 NDCG@10） |  | pool+zero ρ0.8: 0.2395；pool+warm ρ0.8: 0.2396；topk+zero ρ0.8: 0.2389；topk+warm ρ0.8: 0.2396；论文默认 pool+zero ρ0.2: 0.2396 | 冻结的图不仅不涨，还会因为邻居失真而略微拖低（见体检列） |
| ⑥b | 冻结的代价：邻居质量与推荐指标 |  | neighbor_lift：pool+zero=1.690 → topk+zero=1.455（-0.235）；窗口级 NDCG@10：0.2395 → 0.2389（-0.0006） | 冻结把邻居质量明显拉低，但 NDCG@10 几乎没变 —— 说明这个指标对「图不再适应」不敏感，评测时不能只看 NDCG，必须同时看 neighbor_lift / ECR 这类结构指标 |
| ⑦ | 一句话结论 |  | 冻结 = 「只存 Top-K」×「新边 0 初值」的联合效应，改任一项即可解除；topk 下改用 warm 初值可把冻结用户占比降 0.48（p=5e-09）；候选池下 warm 相对 zero 只多转 +0.004（几乎无差别） | 回答「更新机制是否合理」：现行实现的隐患是**旧状态规则 + 论文 0 初值的组合会让图永久冻结**；而冻结的代价主要落在邻居质量（lift 下降），NDCG@10 看不出来 |


## 各格子的冻结程度与效果

| cell | graph_state | graph_new_edge | rho | n_seeds | turnover_from_ref | frozen_share | ecr | added | n_edges | neighbor_lift | ndcg_win_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper_default_pool_zero_rho0.2 | pool | zero | 0.2000 | 5 | 0.9562 | 0.0200 | 0.1238 | 1.4372 | 6040 | 1.6959 | 0.2396 |
| pool_zero_rho0.8 | pool | zero | 0.8000 | 5 | 0.9507 | 0.0200 | 0.0838 | 0.9234 | 6040 | 1.6902 | 0.2395 |
| pool_warm_rho0.8 | pool | warm | 0.8000 | 5 | 0.9550 | 0.0200 | 0.0924 | 1.0327 | 6040 | 1.6931 | 0.2396 |
| topk_zero_rho0.8 | topk | zero | 0.8000 | 5 | 0.1614 | 0.4964 | 0.0022 | 0.0369 | 6040 | 1.4549 | 0.2389 |
| topk_warm_rho0.8 | topk | warm | 0.8000 | 5 | 0.9545 | 0.0200 | 0.1002 | 1.1420 | 6040 | 1.6944 | 0.2396 |


## 成对比较

| contrast | what | metric | n_pairs | delta_mean | ci95_lo | ci95_hi | p_paired | seeds_positive | seeds_total |
|---|---|---|---|---|---|---|---|---|---|
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | ndcg_win_mean | 25 | -6.06e-04 | -0.0011 | -1.87e-04 | 0.0167 | 1 | 5 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | turnover | 25 | -0.7894 | -0.8226 | -0.7573 | 9.43e-25 | 0 | 5 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | frozen | 25 | 0.4764 | 0.3735 | 0.5792 | 5.04e-09 | 5 | 5 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | added | 25 | -0.8866 | -0.9295 | -0.8437 | 1.93e-23 | 0 | 5 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | ndcg_win_mean | 25 | 1.14e-05 | -7.31e-05 | 1.05e-04 | 0.7996 | 3 | 5 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | turnover | 25 | -4.38e-04 | -7.16e-04 | -1.99e-04 | 0.0038 | 0 | 5 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | frozen | 25 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 5 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | added | 25 | 0.1093 | 0.0779 | 0.1394 | 4.62e-07 | 5 | 5 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | ndcg_win_mean | 25 | 7.21e-04 | 3.10e-04 | 0.0012 | 0.0038 | 4 | 5 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | turnover | 25 | 0.7932 | 0.7605 | 0.8262 | 9.94e-25 | 5 | 5 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | frozen | 25 | -0.4764 | -0.5792 | -0.3735 | 5.04e-09 | 0 | 5 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | added | 25 | 1.1051 | 1.0664 | 1.1440 | 1.21e-26 | 5 | 5 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | ndcg_win_mean | 25 | 1.03e-04 | 2.29e-06 | 2.08e-04 | 0.0716 | 4 | 5 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | turnover | 25 | 0.0042 | 0.0015 | 0.0073 | 0.0095 | 5 | 5 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | frozen | 25 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 5 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | added | 25 | 0.1092 | 0.0899 | 0.1283 | 7.65e-11 | 5 | 5 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | ndcg_win_mean | 25 | -1.34e-04 | -2.74e-04 | 7.78e-07 | 0.0695 | 1 | 5 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | turnover | 25 | -0.0055 | -0.0096 | -0.0017 | 0.0130 | 0 | 5 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | frozen | 25 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 5 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | added | 25 | -0.5137 | -0.5347 | -0.4917 | 7.45e-25 | 0 | 5 |


## 怎么读

- `turnover_from_ref` / `frozen_share` 都是相对**预热后第一次评测**的邻居快照算的；前者是 Jaccard 距离均值，后者是「一位都没变」的用户占比。
- `topk` 复现的是审计前的旧行为（只保留当前 Top-K 的 EMA，掉出去的边状态被丢掉）；`pool` 是修复版（保留候选池）。
- 若 topk 冻住而 pool 不冻 ⇒ **冻结是状态保留规则造成的**，既不是 EMA 强度、也不是初始化规则；后两者只影响「转多快」。
