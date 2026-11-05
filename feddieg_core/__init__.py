"""FedDIEG 对齐实现（2026-11-05 审计修复版）。

本包把 `F:\\FedDIEG-main` 下两个真正的 FedDIEG 主程序
（`FedDIEG.py` 与 `exp_01/run_exp01_movielens_aligned.py`）的**模型 / 客户端本地训练 /
服务器聚合 / 动态图更新**整理成可复用模块，并把 `exp_val_1026` 三条验证实验
从"单模型快速仿真"切换回真正的联邦协议。

模块划分
--------
``config``     超参（分「模型 / 联邦 / 动态图」三组，逐项标注对齐来源）
``model``      SASRec 编码器 + FedDIEG 模型 + 个性化层判定（逐行对齐 aligned 主程序）
``graph``      动态兴趣图：论文公式 e←ρe+(1-ρ)Δe，候选池保留非 Top-K 边状态
``data``       滚动时间窗口 + 逐客户端样本 + 负采样（排除客户端自身历史）
``federated``  客户端本地训练 + 服务器参数聚合（FedAvg）+ 摘要缓存（论文语义）
``evaluate``   逐用户评测（对齐 A/B 版的"最后位置打分 + 候选集排序"）
``cli``        val01/02/03 共用的命令行参数
``utils``      Tee / set_seed / CSV 写出等
``verify_fixes`` 27 项离线断言，逐条核对本包是否真的修掉了审计报告里的缺陷

与旧实现的差别（对应审计报告 2026-11-05）
------------------------------------------
1. 训练：由「1 个共享模型 + 1 个优化器逐 microbatch 更新」改为
   「客户端副本本地训练 → 上传共享层 → 服务器加权平均」，见 ``federated``。
2. 动态图：EMA 状态量纲固定为 max(0,cos)（旧 val01 混用归一化权重）；
   无效槽位写 -1（旧 val02/val03 写伪 id）；不再用 scatter_ 别名（旧代码 -1 被钳到列 0）；
   EMA 状态保存在候选池上，掉出 Top-K 的边不会被清零（旧代码因此图冻结）。
3. 摘要缓存：默认直接替换（论文 s̄_i = s_i），动量 EMA 改为可选。
4. 未参与客户端：默认每轮用缓存摘要异步更新其边（论文语义），不再只重算被抽样用户。
"""

from __future__ import annotations

__all__ = [
    "FedDIEGConfig",
    "FedDIEGModel",
    "DynamicInterestGraph",
    "FedTrainer",
    "ClientWindow",
    "load_client_windows",
]

__version__ = "2026.11.05"
