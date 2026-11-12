"""上传摘要时点的**唯一实现**：本地训练完成后重算（论文口径，F7）。

为什么单独成文件
----------------
论文写的是「客户端完成本地训练后上传当前短期兴趣摘要」。而早期实现取的是
**最后一个 epoch 前向、``optimizer.step()`` 之前**的表示 —— 也就是「训练前」的摘要：

* ``feddieg_core/federated.py``（2026-11-08 已修，F7）——正式实验 `val_DynamicGraph` 走这条路径；
* ``exp_01_fast_gpu`` / ``exp_02_fast_gpu`` / ``exp_03_fast_gpu`` / ``exp_04_fast_gpu``
  的独立训练循环 —— 原本还是旧口径，2026-11-11 统一改成本模块（见各文件里的 `【F7 口径】` 注释）。

两条路径必须给出同一个语义，所以这里只留一份实现：**先 ``eval()``（关掉 dropout）、
在 ``no_grad()`` 下用更新后的参数前向一次，取到的才是「训练完成后」的摘要**。

用法
----
::

    from feddieg_core.summary_upload import upload_summary_after_update

    # 训练循环结束后（optimizer.step() 已经执行过）
    new_summary = upload_summary_after_update(
        model,
        lambda: model(seq, neighbors, weights, graph_mask)[1],   # 返回「当前短期兴趣摘要」
    )

``forward_fn`` 里要用**本轮训练时用的同一套输入**（同一段序列、同一批邻居），
只换参数 —— 这样「上传的摘要」与「产出它的那次前向」严格对应。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

__all__ = ["upload_summary_after_update", "assert_upload_summary"]


def upload_summary_after_update(
    model: torch.nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    *,
    detach: bool = True,
) -> torch.Tensor:
    """用**更新后**的参数重算摘要（论文口径：训练完成后再上传）。

    Parameters
    ----------
    model
        客户端本地模型。这里只临时切换 ``eval()/train()``，不改参数、不改任何缓冲区。
    forward_fn
        零参回调：用当前参数前向一次并返回摘要张量（形状 ``[B, D]`` 或更大，
        只要是「会被当作摘要上传」的那个张量）。
    detach
        是否 ``.detach()``。上传摘要永远不需要梯度，默认 ``True``。

    Returns
    -------
    torch.Tensor
        训练完成后的摘要；调用前后模型的 ``training`` 状态保持一致。
    """
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            summary = forward_fn()
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
    if not torch.is_tensor(summary):
        raise TypeError(
            "forward_fn 必须返回一个张量（那个会被上传的兴趣摘要），"
            f"实际拿到 {type(summary).__name__}"
        )
    return summary.detach() if detach else summary


def assert_upload_summary(
    uploaded: torch.Tensor,
    summary_after: torch.Tensor,
    summary_before: Optional[torch.Tensor] = None,
    *,
    atol: float = 1e-6,
) -> None:
    """断言「上传的摘要是训练后的那份」（给测试/离线断言用）。

    * 必须与 ``summary_after``（训练后重算）逐位一致；
    * 若给了 ``summary_before``（训练前那次前向），还必须**不等于**它 ——
      否则说明训练根本没改变表示，或者代码又退回了旧口径。
    """
    just = float((uploaded.reshape(uploaded.shape[0], -1) -
                  summary_after.reshape(summary_after.shape[0], -1)).abs().max().item())
    if just > atol:
        raise AssertionError(f"上传摘要与「训练后重算」不一致：max|Δ|={just:.3e}")
    if summary_before is not None:
        changed = float((summary_after.reshape(summary_after.shape[0], -1) -
                         summary_before.reshape(summary_before.shape[0], -1)).abs().max().item())
        if changed <= atol:
            raise AssertionError(
                "训练前后摘要完全相同：这次训练没有改变表示，无法区分新旧口径"
            )
