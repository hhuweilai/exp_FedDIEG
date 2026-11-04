# 验证实验 3（val03）：短期兴趣波动会不会导致不必要的图更新？

> 本文件由 `val03_aggregate.py` 自动生成，数字全部可从 `results/raw/` 复算，不需要手工填。**结论性文字请写在 `val03_findings.md`**（人工整理），这里只保证数字与图表可追溯。

## 1. 设计

- **注入 arm（人工构造）**：把 onset 窗口（sustained: 及其后所有窗口；transient: 仅该窗口）里注入组用户的 itemId 换成目标类别（该用户最不熟悉的类别）的物品，其余行、时间戳、窗口边界都不动。配对对照 `pair` 用**完全相同的流程与窗口**，但替换成自己旧主类别的物品（同类别重采样）。
- **真实 arm（只打标签）**：把真实用户分成 `real_sustained`（跳变后保持新类别）、`real_transient`（跳变后回到旧类别）、`real_stable`（从未跳变、活动量与 onset 匹配）。
- 5 个组在**同一次运行**里共存 → 组间比较共享同一份训练轨迹、同一份参与序列。
- 主结论用参考 ρ = 0.80（默认取论文配置 0.8；若该 ρ 没跑则自动取最接近的）；ρ 扫描 = 0.00, 0.50, 0.80。

### 1.1 各数据集的场景规模

- **amazon-digital music**：注入 inj=100 / pair=100（min_inter=2，onset=3）；真实 sustained=40 / transient=24 / stable=22；类别来源=cooccur。
  - ⚠️ real_transient 只有 24 人（< 25）：该数据集上的真实 arm 只能作方向性参考，主证据请用注入 arm。
- **movielens-1m**：注入 inj=19 / pair=19（min_inter=2，onset=3）；真实 sustained=36 / transient=65 / stable=50；类别来源=cooccur。
  - ⚠️ 注入组只有 19 人：组均值噪声会比较大，建议 --inject-onset 换个窗口 或降低 --min-inter。

## 2. 关键指标：过度更新 vs 更新滞后

两个指标都是**成对差值口径**（注入组 − 配对对照），所以「物品被替换」这个动作本身的影响被对照掉了；主指标是 `old_peer_share`（邻居里「仍属于用户旧兴趣」的边权占比），因为它有方向、且在 ρ 小时不会像 Jaccard turnover 那样饱和到 1。

- `over_update_early` = **撤销发生那个窗口**的残留 ÷ 扰动期改动。>0.5 = 撤销至少要晚一个窗口（迟滞）。
- `over_update` = **再下一个窗口**的残留 ÷ 扰动期改动。>0.5 = 永久残留（真正意义上的过度更新）。
- `under_update` = 1 − 持续变化场景的净跟进量 ÷ ρ=0 的净跟进量。0 = 完全跟上；1 = 完全不更新（更新滞后）。
- `summary_residual_ratio` = 撤销后摘要的残余偏离 ÷ 扰动期偏离。它 <1 说明「兴趣摘要本身恢复了」，此时图若没恢复，责任就在边权更新规则；它 ≥1 说明摘要根本没恢复，残留主要来自客户端/摘要层而不是 EMA。
- 理想固定 ρ 应同时给出 `over_update≈0` 与 `under_update≈0`。

| dataset | rho | over_update_early | over_update | under_update | transient_attack_during | transient_residual_post1 | transient_residual_post2 | sustained_tracked_post | sustained_tracked_at_rho0 | summary_residual_ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| amazon-digital music | 0.0000 | nan | nan | nan | -0.0067 | 0.0020 | -0.1037 | -0.0329 | -0.0329 | 1.8320 |
| amazon-digital music | 0.5000 | nan | nan | nan | 0.0009 | 0.0011 | -0.0180 | -0.0139 | -0.0329 | 2.7831 |
| amazon-digital music | 0.8000 | nan | nan | nan | -0.0000 | -0.0004 | 0.0135 | 0.0066 | -0.0329 | 2.5763 |
| movielens-1m | 0.0000 | 1.0432 | -0.0833 | 0.0000 | 0.1762 | 0.1838 | -0.0147 | 0.2653 | 0.2653 | 1.6295 |
| movielens-1m | 0.5000 | 1.4331 | 0.6856 | 0.6014 | 0.0761 | 0.1090 | 0.0522 | 0.1058 | 0.2653 | 1.0094 |
| movielens-1m | 0.8000 | nan | nan | 0.7135 | 0.0049 | 0.0972 | 0.0621 | 0.0760 | 0.2653 | 0.5260 |

## 3. 自动判定

判定规则：关系质量的噪声地板 0.02，NDCG 的噪声地板 0.002；`over_update < 0.5`（再下一个窗口的残留）视为最终能回退，`under_update > 0.5` 视为明显滞后。

| dataset | check | holds | detail |
|---|---|---|---|
| amazon-digital music | ① 短期扰动确实改动了图（净 old-peer 占比下降 > 噪声地板） | no | ρ=0.80 扰动窗口净下降 attack=-0.000（噪声地板 0.02）；同一口径下持续变化的跟进量=0.007 |
| amazon-digital music | ② 短期扰动撤销后图能回退（过度更新不成立） | yes | ρ=0.80 扰动期净下降低于噪声地板，图这次几乎没被改动（这是「图没动」，不能算「能撤销」，需与 ③ 一起读） |
| amazon-digital music | ③ 持续变化被跟上（更新滞后不成立） | no | ρ=0.80 无可比参照（ρ=0 的跟进量太低） |
| amazon-digital music | ④ 存在同时不过度更新、也不明显滞后的固定 ρ | no | 该数据集上无法判定：over_update / under_update 全部为 nan（关系质量指标没有分辨力，通常发生在极稀疏数据集上） |
| amazon-digital music | ⑤ 撤销后「摘要恢复但图没恢复」（说明责任在图更新规则） | no | ρ=0.80 摘要残余比=2.58（<0.5 = 摘要基本恢复）、图残余比 over_update=nan（>=0.5 = 图没恢复）。两者一起看才能判断是摘要记忆还是 EMA 边权的问题。 |
| amazon-digital music | ⑥ NDCG 上出现可观测代价（DID > 噪声地板） | no | ρ=0.80 transient 的 ΔNDCG DID=+0.0849、sustained 的 ΔNDCG DID=+0.0031（|DID| > 0.002 才有代价） |
| amazon-digital music | ⑦ 真实 real_sustained 的关系质量变化（描述性） | no | ρ=0.80 old-peer 占比 0.072（变化前）→ 0.066（变化窗口）→ 0.065（之后） |
| amazon-digital music | ⑦ 真实 real_transient 的关系质量变化（描述性） | no | ρ=0.80 old-peer 占比 0.089（变化前）→ 0.100（变化窗口）→ 0.100（之后） |
| movielens-1m | ① 短期扰动确实改动了图（净 old-peer 占比下降 > 噪声地板） | no | ρ=0.80 扰动窗口净下降 attack=0.005（噪声地板 0.02）；同一口径下持续变化的跟进量=0.076 |
| movielens-1m | ② 短期扰动撤销后图能回退（过度更新不成立） | yes | ρ=0.80 扰动期净下降低于噪声地板，图这次几乎没被改动（这是「图没动」，不能算「能撤销」，需与 ③ 一起读） |
| movielens-1m | ③ 持续变化被跟上（更新滞后不成立） | no | ρ=0.80 tracked=0.076，相当于 ρ=0 的 29%（<50% 视为明显滞后） |
| movielens-1m | ④ 存在同时不过度更新、也不明显滞后的固定 ρ | yes | 满足 over_update≤0.5 且 under_update≤0.5 的 ρ: 0.00。★ 但里面包含 ρ=0：这等价于「直接关掉 EMA」，正是任务书第 8 节停止信号 2 的情形，报告里必须同时讨论它的代价（每轮高频改边带来的抖动），不能只写「ρ=0 够用」 |
| movielens-1m | ⑤ 撤销后「摘要恢复但图没恢复」（说明责任在图更新规则） | no | ρ=0.80 摘要残余比=0.53（<0.5 = 摘要基本恢复）、图残余比 over_update=nan（>=0.5 = 图没恢复）。两者一起看才能判断是摘要记忆还是 EMA 边权的问题。 |
| movielens-1m | ⑥ NDCG 上出现可观测代价（DID > 噪声地板） | yes | ρ=0.80 transient 的 ΔNDCG DID=-0.2083、sustained 的 ΔNDCG DID=-0.0135（|DID| > 0.002 才有代价） |
| movielens-1m | ⑦ 真实 real_sustained 的关系质量变化（描述性） | no | ρ=0.80 old-peer 占比 0.151（变化前）→ 0.162（变化窗口）→ 0.150（之后） |
| movielens-1m | ⑦ 真实 real_transient 的关系质量变化（描述性） | yes | ρ=0.80 old-peer 占比 0.217（变化前）→ 0.243（变化窗口）→ 0.237（之后） |

## 4. 分窗口汇总（可直接抄进表格）

| dataset | scenario | group | metric | pre_level | during_level | post1_level | post2_level | plateau_level | post_n_users |
|---|---|---|---|---|---|---|---|---|---|
| amazon-digital music | sustained | inj | old_peer_share | 0.0535 | 0.0561 | 0.0561 | 0.0485 | 0.0485 | 300 |
| amazon-digital music | sustained | pair | old_peer_share | 0.0570 | 0.0632 | 0.0629 | 0.0576 | 0.0576 | 300 |
| amazon-digital music | sustained | real_stable | old_peer_share | 0.0740 | 0.0848 | 0.0915 | 0.0805 | 0.0660 | 42 |
| amazon-digital music | sustained | real_sustained | old_peer_share | 0.0721 | 0.0659 | 0.0631 | 0.0685 | 0.0774 | 99 |
| amazon-digital music | sustained | real_transient | old_peer_share | 0.0895 | 0.0994 | 0.0915 | 0.1202 | 0.1348 | 57 |
| amazon-digital music | transient | inj | old_peer_share | 0.0535 | 0.0562 | 0.0558 | 0.0491 | 0.0491 | 300 |
| amazon-digital music | transient | pair | old_peer_share | 0.0570 | 0.0637 | 0.0629 | 0.0573 | 0.0573 | 300 |
| amazon-digital music | transient | real_stable | old_peer_share | 0.0717 | 0.0829 | 0.0891 | 0.0803 | 0.0659 | 42 |
| amazon-digital music | transient | real_sustained | old_peer_share | 0.0721 | 0.0659 | 0.0631 | 0.0681 | 0.0773 | 99 |
| amazon-digital music | transient | real_transient | old_peer_share | 0.0895 | 0.0996 | 0.0917 | 0.1208 | 0.1346 | 57 |
| movielens-1m | sustained | inj | old_peer_share | 0.1575 | 0.2277 | 0.2353 | 0.2831 | 0.2831 | 57 |
| movielens-1m | sustained | pair | old_peer_share | 0.0847 | 0.1839 | 0.2455 | 0.2648 | 0.2648 | 57 |
| movielens-1m | sustained | real_stable | old_peer_share | 0.0986 | 0.0806 | 0.0934 | 0.1383 | 0.1383 | 90 |
| movielens-1m | sustained | real_sustained | old_peer_share | 0.1511 | 0.1616 | 0.1605 | 0.1391 | 0.2416 | 54 |
| movielens-1m | sustained | real_transient | old_peer_share | 0.2181 | 0.2424 | 0.2544 | 0.2039 | 0.3616 | 135 |
| movielens-1m | transient | inj | old_peer_share | 0.1575 | 0.2120 | 0.2108 | 0.2567 | 0.2567 | 57 |
| movielens-1m | transient | pair | old_peer_share | 0.0847 | 0.1548 | 0.2459 | 0.2401 | 0.2401 | 57 |
| movielens-1m | transient | real_stable | old_peer_share | 0.0962 | 0.0801 | 0.0907 | 0.1425 | 0.1425 | 90 |
| movielens-1m | transient | real_sustained | old_peer_share | 0.1494 | 0.1563 | 0.1522 | 0.1372 | 0.2402 | 54 |
| movielens-1m | transient | real_transient | old_peer_share | 0.2169 | 0.2431 | 0.2548 | 0.2033 | 0.3619 | 135 |
| amazon-digital music | sustained | inj | turnover_from_pre | nan | 0.0047 | 0.0070 | 0.0000 | 0.0000 | 300 |
| amazon-digital music | sustained | pair | turnover_from_pre | nan | 0.0035 | 0.0070 | 0.0043 | 0.0043 | 300 |
| amazon-digital music | sustained | real_stable | turnover_from_pre | nan | 0.0255 | 0.0402 | 0.0452 | 0.0260 | 42 |
| amazon-digital music | sustained | real_sustained | turnover_from_pre | nan | 0.0457 | 0.0349 | 0.0694 | 0.0883 | 99 |
| amazon-digital music | sustained | real_transient | turnover_from_pre | nan | 0.0446 | 0.0652 | 0.0697 | 0.0121 | 57 |
| amazon-digital music | transient | inj | turnover_from_pre | nan | 0.0053 | 0.0076 | 0.0023 | 0.0023 | 300 |
| amazon-digital music | transient | pair | turnover_from_pre | nan | 0.0016 | 0.0034 | 0.0043 | 0.0043 | 300 |
| amazon-digital music | transient | real_stable | turnover_from_pre | nan | 0.0283 | 0.0445 | 0.0452 | 0.0260 | 42 |
| amazon-digital music | transient | real_sustained | turnover_from_pre | nan | 0.0457 | 0.0349 | 0.0694 | 0.0883 | 99 |
| amazon-digital music | transient | real_transient | turnover_from_pre | nan | 0.0446 | 0.0652 | 0.0697 | 0.0121 | 57 |
| movielens-1m | sustained | inj | turnover_from_pre | nan | 0.3629 | 0.4361 | 0.3693 | 0.3693 | 57 |
| movielens-1m | sustained | pair | turnover_from_pre | nan | 0.1989 | 0.3131 | 0.3280 | 0.3280 | 57 |
| movielens-1m | sustained | real_stable | turnover_from_pre | nan | 0.0594 | 0.0131 | 0.1373 | 0.1373 | 90 |
| movielens-1m | sustained | real_sustained | turnover_from_pre | nan | 0.0964 | 0.1311 | 0.1622 | 0.0795 | 54 |
| movielens-1m | sustained | real_transient | turnover_from_pre | nan | 0.1106 | 0.1082 | 0.2412 | 0.1564 | 135 |
| movielens-1m | transient | inj | turnover_from_pre | nan | 0.3380 | 0.4057 | 0.2954 | 0.2954 | 57 |
| movielens-1m | transient | pair | turnover_from_pre | nan | 0.1785 | 0.3833 | 0.4064 | 0.4064 | 57 |
| movielens-1m | transient | real_stable | turnover_from_pre | nan | 0.0579 | 0.0131 | 0.1408 | 0.1408 | 90 |
| movielens-1m | transient | real_sustained | turnover_from_pre | nan | 0.0899 | 0.1110 | 0.1569 | 0.0795 | 54 |
| movielens-1m | transient | real_transient | turnover_from_pre | nan | 0.1146 | 0.1097 | 0.2267 | 0.1564 | 135 |
| amazon-digital music | sustained | inj | ndcg | 0.2356 | 0.0496 | 0.0605 | 0.0553 | 0.0553 | 300 |
| amazon-digital music | sustained | pair | ndcg | 0.2582 | 0.0643 | 0.0833 | 0.0624 | 0.0624 | 300 |
| amazon-digital music | sustained | real_stable | ndcg | 0.1665 | 0.2740 | 0.1936 | 0.3608 | 0.1992 | 42 |
| amazon-digital music | sustained | real_sustained | ndcg | 0.1852 | 0.2364 | 0.1203 | 0.3182 | 0.1141 | 99 |
| amazon-digital music | sustained | real_transient | ndcg | 0.2367 | 0.2522 | 0.2389 | 0.4436 | 0.5928 | 57 |
| amazon-digital music | transient | inj | ndcg | 0.2356 | 0.0648 | 0.3704 | 0.3579 | 0.3579 | 300 |
| amazon-digital music | transient | pair | ndcg | 0.2582 | 0.0470 | 0.3087 | 0.2934 | 0.2934 | 300 |
| amazon-digital music | transient | real_stable | ndcg | 0.1674 | 0.2756 | 0.1702 | 0.3684 | 0.1757 | 42 |
| amazon-digital music | transient | real_sustained | ndcg | 0.1913 | 0.2413 | 0.1199 | 0.3278 | 0.0962 | 99 |
| amazon-digital music | transient | real_transient | ndcg | 0.2378 | 0.2441 | 0.2253 | 0.4338 | 0.6260 | 57 |
| movielens-1m | sustained | inj | ndcg | 0.3799 | 0.1598 | 0.2229 | 0.4102 | 0.4102 | 57 |
| movielens-1m | sustained | pair | ndcg | 0.3455 | 0.2046 | 0.2771 | 0.2931 | 0.2931 | 57 |
| movielens-1m | sustained | real_stable | ndcg | 0.4281 | 0.4667 | 0.4110 | 0.3979 | 0.3979 | 90 |
| movielens-1m | sustained | real_sustained | ndcg | 0.3716 | 0.4011 | 0.4165 | 0.4458 | 0.3554 | 54 |
| movielens-1m | sustained | real_transient | ndcg | 0.3426 | 0.3345 | 0.3694 | 0.1500 | 0.3701 | 135 |
| movielens-1m | transient | inj | ndcg | 0.3799 | 0.1551 | 0.2875 | 0.1573 | 0.1573 | 57 |
| movielens-1m | transient | pair | ndcg | 0.3455 | 0.2308 | 0.3759 | 0.4353 | 0.4353 | 57 |
| movielens-1m | transient | real_stable | ndcg | 0.4252 | 0.4783 | 0.4206 | 0.3874 | 0.3874 | 90 |
| movielens-1m | transient | real_sustained | ndcg | 0.3728 | 0.4067 | 0.4142 | 0.4335 | 0.3917 | 54 |
| movielens-1m | transient | real_transient | ndcg | 0.3417 | 0.3308 | 0.3634 | 0.1475 | 0.3766 | 135 |

## 5. 怎么读这几张图

- `val03_fig_arm_synthetic.png`：注入 arm。看 (a) 扰动窗口（rel_round 1..R）transient 的 old-peer 占比是否也掉下来（扰动确实进了图）；(b) 撤销窗口（R+1 之后）它是否回升到配对对照的水平（能撤销）还是停在 sustained 的水平（过度更新）。
- `val03_fig_pair_excess.png`：★★ 主结论图 —— 已经是「注入组 − 配对对照」的净效应，而且把所有 ρ 画在同一张里（颜色=ρ）。低 ρ 会是「先掉下去、撤不回来」，高 ρ 基本贴着 0。
- `val03_fig_rho_tradeoff.png`：★★ 关键图，**两行**。上行横轴是「撤销窗口内的残留」（衡量撤销是否及时），下行是「撤销后第二窗的残留」（衡量是否永久损伤）。空心点 = 该 ρ 下图根本没动。★ 读法：ρ=0 在下行已经靠近左下角（最终能撤销），但在上行远离左下角（**晚一整窗才撤销**）；ρ≥0.5 在上行下行都远离。因此**没有任何 ρ 同时做到「及时撤销 + 完全跟随 + 图稳定」**，但这也不能简单写成「ρ=0 就够用」——它的代价是撤销延迟与最大抖动，见 `val03_findings.md` §4、§9。
- `val03_fig_arm_synthetic.png`：注入 arm 的原始水平（含配对对照）。注意在 ρ 大时两条线会几乎重合，那正是「图冻结」的直接证据。
- `val03_fig_arm_real.png`：真实数据答案（无人工注入），与注入 arm 的结论对照着写。
- `val03_fig_rho_curves.png`：ρ 对「图动多少」的影响；ρ≥0.5 时往往已经接近冻结（与实验二一致）。

## 6. 写报告时必须交代的局限

1. **注入是人工构造**：只替换 itemId，不改窗口边界/交互数/时间戳；结论必须与真实 arm 分开写。
2. **短期扰动的粒度是 1 个窗口**（受窗口划分限制），不是 1 轮或 1 天；真实数据里若「几天」的波动远短于一个窗口，本实验会**低估**过度更新的程度。
3. **注入的痕迹会留在历史上**：窗口 t0 的注入物品在之后的窗口里仍在用户历史中（真实系统同样无法抹掉一次「连刷几部恐怖片」）。所以本实验测的是「摘要记忆 + 边权记忆」的合计残留，`summary_residual_ratio` 用来把这两层分开。
4. **注入组规模有限**：MovieLens-1M 严格口径约 19 对；稀疏数据集（Amazon 类）每窗口每用户只有约 2 条交互，真实 arm 的 transient 组往往不足 25 人，只能作方向性参考（与实验二的结论一致）。
5. **NDCG 的成对差值不完美**：sustained 组在扰动窗口之后的测试目标是「新兴趣物品」、transient 组是「旧兴趣物品」，两组绝对水平不可直接比较，只能各自对自己 pre 期做差再相减（DID）。
6. **真实 arm 的分组用到了 onset 之后的窗口**（事后分组），只决定统计哪些人，不进入模型。
7. 3 个种子只够看趋势；正式论文建议 5 个以上。

