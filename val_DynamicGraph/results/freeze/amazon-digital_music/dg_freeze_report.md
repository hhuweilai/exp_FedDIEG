# 冻结归因报告：动态图更新机制是否合理？（amazon-digital_music）

同一序列编码器与联邦协议，只改图更新机制：图状态（pool / topk）× 新边初始化（zero / warm），ρ 固定 0.8；另加论文默认（pool+zero+ρ0.2）作参照。每个格子 3 个种子。

**一句话：** 冻结 = 「只存 Top-K」×「新边 0 初值」的联合效应，改任一项即可解除；候选池下 warm 相对 zero 只多转 +0.006（几乎无差别）

| verdict_id | question | value | evidence | judgement |
|---|---|---|---|---|
| ① | 四种配置下图到底冻没冻住（ρ=0.8） |  | pool+zero: frozen_share=0.020, turnover=0.967, ECR=0.1767, 新边=2.12；pool+warm: frozen_share=0.020, turnover=0.973, ECR=0.2388, 新边=3.06；topk+zero: frozen_share=0.194, turnover=0.476, ECR=0.0039, 新边=0.07；topk+warm: frozen_share=0.020, turnover=0.972, ECR=0.2342, 新边=3.01 | frozen_share 越接近 1、turnover 越接近 0 就越冻 |
| ② | 冻结的根因：EMA / 新边初始化 / 状态保留规则 |  | pool+zero=不冻，pool+warm=不冻，topk+zero=冻，topk+warm=不冻 | 冻结**只出现在 topk + zero 这一个组合**：pool 的两格都不冻（ρ=0.8 也一样），topk+warm 也不冻。所以它既不是 EMA 强度单独造成的，也不是任一单独因素造成的，而是**「只存 Top-K 的旧状态规则」×「新边 0 初值」的联合效应**：topk 状态会丢弃掉出 Top-K 的边历史，而 zero 初值让新边只能以 (1-ρ)Δe 起步（ρ=0.8 时仅 20%），根本挤不进 Top-K；挤不进 ⇒ EMA 永不累积 ⇒ 永久冻结。把任一项改掉（pool 或 warm）就都不冻。 |
| ③ | 换 warm 初始化能不能解开冻结（topk 状态下） | -0.1742 | Δfrozen_share=-0.174 [-0.289, -0.078]，p=0.0084 | warm 初始化**不能**解开冻结 |
| ④ | 候选池下，warm 初始化让图转得快多少 | 0.0056 | Δturnover=+0.0056（p=0.0324）；Δ新边=+0.94（p=2.63e-17）；ΔNDCG=-0.0002（p=0.00177） | 初始化规则对适应速度几乎没有影响 |
| ⑤ | ρ 从 0.2 提到 0.8（同为 pool+zero）改变了什么 | -0.0054 | Δturnover=-0.0054（p=0.0369）；ΔNDCG=+0.0001（p=0.0248） | ρ 对 turnover 没有明显影响 |
| ⑥ | 各配置的推荐效果（窗口级 NDCG@10） |  | pool+zero ρ0.8: 0.0924；pool+warm ρ0.8: 0.0922；topk+zero ρ0.8: 0.0923；topk+warm ρ0.8: 0.0922；论文默认 pool+zero ρ0.2: 0.0923 | 冻结的图不仅不涨，还会因为邻居失真而略微拖低（见体检列） |
| ⑥b | 冻结的代价：邻居质量与推荐指标 |  | neighbor_lift：pool+zero=1.804 → topk+zero=1.446（-0.357）；窗口级 NDCG@10：0.0924 → 0.0923（-0.0002） | 冻结把邻居质量明显拉低，但 NDCG@10 几乎没变 —— 说明这个指标对「图不再适应」不敏感，评测时不能只看 NDCG，必须同时看 neighbor_lift / ECR 这类结构指标 |
| ⑦ | 一句话结论 |  | 冻结 = 「只存 Top-K」×「新边 0 初值」的联合效应，改任一项即可解除；候选池下 warm 相对 zero 只多转 +0.006（几乎无差别） | 回答「更新机制是否合理」：现行实现的隐患是**旧状态规则 + 论文 0 初值的组合会让图永久冻结**；而冻结的代价主要落在邻居质量（lift 下降），NDCG@10 看不出来 |


## 各格子的冻结程度与效果

| cell | graph_state | graph_new_edge | rho | n_seeds | turnover_from_ref | frozen_share | ecr | added | n_edges | neighbor_lift | ndcg_win_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper_default_pool_zero_rho0.2 | pool | zero | 0.2000 | 3 | 0.9723 | 0.0200 | 0.2754 | 3.7020 | 12381 | 1.7897 | 0.0923 |
| pool_zero_rho0.8 | pool | zero | 0.8000 | 3 | 0.9669 | 0.0200 | 0.1767 | 2.1201 | 12381 | 1.8037 | 0.0924 |
| pool_warm_rho0.8 | pool | warm | 0.8000 | 3 | 0.9725 | 0.0200 | 0.2388 | 3.0556 | 12381 | 1.7782 | 0.0922 |
| topk_zero_rho0.8 | topk | zero | 0.8000 | 3 | 0.4759 | 0.1942 | 0.0039 | 0.0670 | 12381 | 1.4464 | 0.0923 |
| topk_warm_rho0.8 | topk | warm | 0.8000 | 3 | 0.9720 | 0.0200 | 0.2342 | 3.0103 | 12381 | 1.7861 | 0.0922 |


## 成对比较

| contrast | what | metric | n_pairs | delta_mean | ci95_lo | ci95_hi | p_paired | seeds_positive | seeds_total |
|---|---|---|---|---|---|---|---|---|---|
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | ndcg_win_mean | 15 | -1.51e-04 | -3.30e-04 | 3.82e-06 | 0.1104 | 0 | 3 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | turnover | 15 | -0.4910 | -0.5930 | -0.3929 | 3.55e-07 | 0 | 3 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | frozen | 15 | 0.1742 | 0.0777 | 0.2894 | 0.0084 | 3 | 3 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | added | 15 | -2.0532 | -2.1369 | -1.9737 | 9.50e-17 | 0 | 3 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | ndcg_win_mean | 15 | 2.18e-05 | -2.85e-05 | 7.18e-05 | 0.4306 | 2 | 3 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | turnover | 15 | -5.34e-04 | -9.68e-04 | -1.79e-04 | 0.0231 | 0 | 3 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | frozen | 15 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 3 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | added | 15 | -0.0453 | -0.0551 | -0.0352 | 6.18e-07 | 0 | 3 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | ndcg_win_mean | 15 | -3.31e-05 | -1.86e-04 | 1.67e-04 | 0.7325 | 1 | 3 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | turnover | 15 | 0.4961 | 0.3958 | 0.6016 | 4.31e-07 | 3 | 3 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | frozen | 15 | -0.1742 | -0.2894 | -0.0777 | 0.0084 | 0 | 3 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | added | 15 | 2.9433 | 2.8433 | 3.0445 | 8.28e-18 | 3 | 3 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | ndcg_win_mean | 15 | -2.06e-04 | -3.09e-04 | -1.00e-04 | 0.0018 | 0 | 3 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | turnover | 15 | 0.0056 | 0.0016 | 0.0104 | 0.0324 | 3 | 3 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | frozen | 15 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 3 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | added | 15 | 0.9355 | 0.8995 | 0.9676 | 2.63e-17 | 3 | 3 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | ndcg_win_mean | 15 | 1.40e-04 | 2.97e-05 | 2.42e-04 | 0.0248 | 3 | 3 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | turnover | 15 | -0.0054 | -0.0101 | -0.0014 | 0.0369 | 0 | 3 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | frozen | 15 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 3 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | added | 15 | -1.5819 | -1.6286 | -1.5325 | 1.54e-18 | 0 | 3 |


## 怎么读

- `turnover_from_ref` / `frozen_share` 都是相对**预热后第一次评测**的邻居快照算的；前者是 Jaccard 距离均值，后者是「一位都没变」的用户占比。
- `topk` 复现的是审计前的旧行为（只保留当前 Top-K 的 EMA，掉出去的边状态被丢掉）；`pool` 是修复版（保留候选池）。
- 若 topk 冻住而 pool 不冻 ⇒ **冻结是状态保留规则造成的**，既不是 EMA 强度、也不是初始化规则；后两者只影响「转多快」。
