# 冻结归因报告：动态图更新机制是否合理？（movielens-1m）

同一序列编码器与联邦协议，只改图更新机制：图状态（pool / topk）× 新边初始化（zero / warm），ρ 固定 0.8；另加论文默认（pool+zero+ρ0.2）作参照。每个格子 3 个种子。

**一句话：** 冻结 = 「只存 Top-K」×「新边 0 初值」的联合效应，改任一项即可解除；topk 下改用 warm 初值可把冻结用户占比降 0.46（p=1.8e-05）；候选池下 warm 相对 zero 只多转 +0.005（几乎无差别）

| verdict_id | question | value | evidence | judgement |
|---|---|---|---|---|
| ① | 四种配置下图到底冻没冻住（ρ=0.8） |  | pool+zero: frozen_share=0.020, turnover=0.962, ECR=0.1061, 新边=1.20；pool+warm: frozen_share=0.020, turnover=0.967, ECR=0.1324, 新边=1.54；topk+zero: frozen_share=0.480, turnover=0.165, ECR=0.0008, 新边=0.04；topk+warm: frozen_share=0.020, turnover=0.967, ECR=0.1401, 新边=1.66 | frozen_share 越接近 1、turnover 越接近 0 就越冻 |
| ② | 冻结的根因：EMA / 新边初始化 / 状态保留规则 |  | pool+zero=不冻，pool+warm=不冻，topk+zero=冻，topk+warm=不冻 | 冻结**只出现在 topk + zero 这一个组合**：pool 的两格都不冻（ρ=0.8 也一样），topk+warm 也不冻。所以它既不是 EMA 强度单独造成的，也不是任一单独因素造成的，而是**「只存 Top-K 的旧状态规则」×「新边 0 初值」的联合效应**：topk 状态会丢弃掉出 Top-K 的边历史，而 zero 初值让新边只能以 (1-ρ)Δe 起步（ρ=0.8 时仅 20%），根本挤不进 Top-K；挤不进 ⇒ EMA 永不累积 ⇒ 永久冻结。把任一项改掉（pool 或 warm）就都不冻。 |
| ③ | 换 warm 初始化能不能解开冻结（topk 状态下） | -0.4600 | Δfrozen_share=-0.460 [-0.593, -0.328]，p=1.81e-05 | warm 初始化**几乎完全解开**了冻结：topk+warm 的 frozen_share 只剩 0.020，已与 pool 两格同量级 |
| ④ | 候选池下，warm 初始化让图转得快多少 | 0.0047 | Δturnover=+0.0047（p=0.0708）；Δ新边=+0.34（p=1.09e-07）；ΔNDCG=-0.0000（p=0.471） | 初始化规则对适应速度几乎没有影响 |
| ⑤ | ρ 从 0.2 提到 0.8（同为 pool+zero）改变了什么 | -0.0620 | ΔECR=-0.0620（p=1.12e-14）（参照格 turnover 缺列，用 ECR 代替；ECR 越高=边换得越快）；ΔNDCG=-0.0001（p=0.345） | ρ 变大减慢的是**适应速度**，不是让图永久冻结（只要旧状态没被丢掉） |
| ⑥ | 各配置的推荐效果（窗口级 NDCG@10） |  | pool+zero ρ0.8: 0.2378；pool+warm ρ0.8: 0.2378；topk+zero ρ0.8: 0.2377；topk+warm ρ0.8: 0.2378；论文默认 pool+zero ρ0.2: 0.2379 | 冻结的图不仅不涨，还会因为邻居失真而略微拖低（见体检列） |
| ⑥b | 冻结的代价：邻居质量与推荐指标 |  | neighbor_lift：pool+zero=1.557 → topk+zero=1.325（-0.232）；窗口级 NDCG@10：0.2378 → 0.2377（-0.0001） | 冻结把邻居质量明显拉低，但 NDCG@10 几乎没变 —— 说明这个指标对「图不再适应」不敏感，评测时不能只看 NDCG，必须同时看 neighbor_lift / ECR 这类结构指标 |
| ⑦ | 一句话结论 |  | 冻结 = 「只存 Top-K」×「新边 0 初值」的联合效应，改任一项即可解除；topk 下改用 warm 初值可把冻结用户占比降 0.46（p=1.8e-05）；候选池下 warm 相对 zero 只多转 +0.005（几乎无差别） | 回答「更新机制是否合理」：现行实现的隐患是**旧状态规则 + 论文 0 初值的组合会让图永久冻结**；而冻结的代价主要落在邻居质量（lift 下降），NDCG@10 看不出来 |


## 各格子的冻结程度与效果

| cell | graph_state | graph_new_edge | rho | n_seeds | turnover_from_ref | frozen_share | ecr | added | n_edges | neighbor_lift | ndcg_win_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper_default_pool_zero_rho0.2 | pool | zero | 0.2000 | 3 | nan | nan | 0.1681 | 2.0409 | 6040 | 1.5622 | 0.2379 |
| pool_zero_rho0.8 | pool | zero | 0.8000 | 3 | 0.9623 | 0.0200 | 0.1061 | 1.1997 | 6040 | 1.5568 | 0.2378 |
| pool_warm_rho0.8 | pool | warm | 0.8000 | 3 | 0.9670 | 0.0200 | 0.1324 | 1.5439 | 6040 | 1.5590 | 0.2378 |
| topk_zero_rho0.8 | topk | zero | 0.8000 | 3 | 0.1653 | 0.4800 | 7.56e-04 | 0.0366 | 6040 | 1.3250 | 0.2377 |
| topk_warm_rho0.8 | topk | warm | 0.8000 | 3 | 0.9666 | 0.0200 | 0.1401 | 1.6635 | 6040 | 1.5595 | 0.2378 |


## 成对比较

| contrast | what | metric | n_pairs | delta_mean | ci95_lo | ci95_hi | p_paired | seeds_positive | seeds_total |
|---|---|---|---|---|---|---|---|---|---|
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | ndcg_win_mean | 15 | -1.15e-04 | -3.37e-04 | 8.27e-05 | 0.3195 | 0 | 3 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | turnover | 15 | -0.7971 | -0.8406 | -0.7541 | 8.19e-15 | 0 | 3 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | frozen | 15 | 0.4600 | 0.3284 | 0.5925 | 1.81e-05 | 3 | 3 |
| topk_zero − pool_zero | 状态保留规则（同为 zero 初始化） | added | 15 | -1.1631 | -1.2323 | -1.0934 | 3.36e-14 | 0 | 3 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | ndcg_win_mean | 15 | 1.53e-05 | -9.77e-05 | 1.43e-04 | 0.8145 | 2 | 3 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | turnover | 15 | -3.58e-04 | -7.48e-04 | -5.04e-05 | 0.0813 | 0 | 3 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | frozen | 15 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 3 |
| topk_warm − pool_warm | 状态保留规则（同为 warm 初始化） | added | 15 | 0.1196 | 0.0718 | 0.1646 | 3.32e-04 | 3 | 3 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | ndcg_win_mean | 15 | 8.94e-05 | -4.33e-05 | 2.52e-04 | 0.2784 | 2 | 3 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | turnover | 15 | 0.8014 | 0.7568 | 0.8456 | 1.09e-14 | 3 | 3 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | frozen | 15 | -0.4600 | -0.5925 | -0.3284 | 1.81e-05 | 0 | 3 |
| topk_warm − topk_zero | 初始化规则（同为 topk 状态） | added | 15 | 1.6269 | 1.5335 | 1.7197 | 1.47e-14 | 3 | 3 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | ndcg_win_mean | 15 | -4.09e-05 | -1.50e-04 | 5.75e-05 | 0.4712 | 1 | 3 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | turnover | 15 | 0.0047 | 5.25e-04 | 0.0094 | 0.0708 | 3 | 3 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | frozen | 15 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 3 |
| pool_warm − pool_zero | 初始化规则（同为 pool 状态） | added | 15 | 0.3442 | 0.2813 | 0.4084 | 1.09e-07 | 3 | 3 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | ndcg_win_mean | 15 | -7.91e-05 | -2.30e-04 | 6.55e-05 | 0.3448 | 1 | 3 |
| pool_zero(ρ0.8) − pool_zero(ρ0.2) | EMA 强度（同为 pool+zero） | added | 15 | -0.8412 | -0.8997 | -0.7831 | 1.96e-13 | 0 | 3 |


## 怎么读

- `turnover_from_ref` / `frozen_share` 都是相对**预热后第一次评测**的邻居快照算的；前者是 Jaccard 距离均值，后者是「一位都没变」的用户占比。
- `topk` 复现的是审计前的旧行为（只保留当前 Top-K 的 EMA，掉出去的边状态被丢掉）；`pool` 是修复版（保留候选池）。
- 若 topk 冻住而 pool 不冻 ⇒ **冻结是状态保留规则造成的**，既不是 EMA 强度、也不是初始化规则；后两者只影响「转多快」。
