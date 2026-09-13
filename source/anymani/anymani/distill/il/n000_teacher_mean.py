r"""Accepted single-asset teacher环境动作mean imitation的纯split与诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class TeacherMeanMetrics:
    r"""总体、饱和动作和非饱和切换状态的互补拟合指标。"""

    mse: float
    mae: float
    saturated_mse: float
    transition_mse: float
    saturated_fraction: float
    sign_accuracy: float
    skill_vs_zero: float
    transition_skill_vs_zero: float
    previous_action_mse: float | None
    skill_vs_previous_action: float | None
    previous_action_saturated_mse: float | None
    previous_action_transition_mse: float | None
    transition_skill_vs_previous_action: float | None


TeacherMeanLossReduction = Literal["element_mean", "balanced_action_regimes"]


def teacher_mean_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: TeacherMeanLossReduction = "element_mean",
    saturation_threshold: float = 1.0 - 1.0e-6,
) -> torch.Tensor:
    r"""计算普通元素均值或饱和/切换区等权的teacher-action回归损失。

    旧teacher环境动作约85%被裁到$\pm1$。普通元素均值
    $\mathcal{L}_{elem}=|\mathcal{A}|^{-1}\sum_{i\in\mathcal{A}}e_i^2$会按数据频率给饱和区约
    $0.85$的权重。等权reduction改为：

    $$
    \mathcal{L}_{balanced}=\frac{1}{2}\operatorname{mean}_{i\in\mathcal{A}_{sat}}e_i^2
    +\frac{1}{2}\operatorname{mean}_{i\in\mathcal{A}_{switch}}e_i^2.
    $$

    该式不改变teacher target或动作单位，只检验闭环敏感的非饱和切换元素是否被经验分布频率遮蔽。

    Args:
        prediction (torch.Tensor): Student deterministic mean，形状$[B,16]$，范围$[-1,1]$。
        target (torch.Tensor): Wrapper-clipped teacher action，形状$[B,16]$，范围$[-1,1]$。
        reduction (TeacherMeanLossReduction): `element_mean`或两类等权的`balanced_action_regimes`。
        saturation_threshold (float): 判定teacher动作为裁剪饱和值的绝对值阈值。

    Returns:
        torch.Tensor: 保持autograd graph的标量MSE。
    """

    if prediction.shape != target.shape:
        raise ValueError(f"teacher mean prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    if reduction not in {"element_mean", "balanced_action_regimes"}:
        raise ValueError(f"unknown teacher mean loss reduction: {reduction!r}")
    squared_error = (prediction - target).square()  # $e_i^2$，逐动作轴无量纲平方误差
    if reduction == "element_mean":
        return squared_error.mean()  # 经验分布元素均值，保留原始matched baseline
    saturated = target.abs() >= saturation_threshold  # teacher wrapper输出恰位于$\pm1$的元素集合
    transition = ~saturated  # 未饱和元素承载bang-bang动作的切换时机与过渡幅度
    if not bool(saturated.any().item() and transition.any().item()):
        raise ValueError("balanced teacher mean loss requires both saturated and transition target elements")
    return 0.5 * (squared_error[saturated].mean() + squared_error[transition].mean())


def replica_modulo_split(
    replica_count: int,
    *,
    validation_modulus: int = 4,
    validation_remainder: int = 3,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    r"""按replica ID确定性切分完整trajectory，避免相邻时间sample跨train/validation泄漏。"""

    if replica_count < 2 or validation_modulus < 2 or validation_remainder not in range(validation_modulus):
        raise ValueError("replica split requires at least two replicas and a valid modulo rule")
    validation = tuple(index for index in range(replica_count) if index % validation_modulus == validation_remainder)
    train = tuple(index for index in range(replica_count) if index not in validation)
    if not train or not validation:
        raise ValueError("replica modulo rule must leave non-empty train and validation sets")
    return train, validation


def training_batch_indices(
    *,
    sample_count: int,
    batch_size: int,
    full_batch: bool,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    r"""返回显式完整轴或有放回随机minibatch，并使二者身份不可混淆。"""

    if sample_count < 1 or batch_size < 1:
        raise ValueError("teacher training sample and batch counts must be positive")
    if full_batch:
        if batch_size != sample_count:
            raise ValueError("full-batch size must equal the fixed teacher sample count")
        return torch.arange(sample_count, dtype=torch.long, device=device)
    return torch.randint(sample_count, (batch_size,), generator=generator, device=device)


def teacher_mean_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    previous_action: torch.Tensor | None = None,
    saturation_threshold: float = 1.0 - 1.0e-6,
    sign_threshold: float = 0.5,
) -> TeacherMeanMetrics:
    r"""评估student mean，并防止大量$\pm1$标签遮蔽动作切换误差。"""

    if prediction.shape != target.shape:
        raise ValueError(f"teacher mean prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("teacher mean prediction and target must be floating tensors")
    if not bool(torch.isfinite(prediction).all().item() and torch.isfinite(target).all().item()):
        raise ValueError("teacher mean prediction and target must be finite")
    if previous_action is not None:
        if previous_action.shape != target.shape:
            raise ValueError("previous-action baseline shape must match teacher target")
        if not bool(torch.isfinite(previous_action).all().item()):
            raise ValueError("previous-action baseline must be finite")
    if not 0.0 < saturation_threshold <= 1.0 or not 0.0 <= sign_threshold <= 1.0:
        raise ValueError("teacher mean thresholds must lie in the action interval")

    squared_error = (prediction - target).square()
    saturated = target.abs() >= saturation_threshold
    transition = ~saturated
    sign_mask = target.abs() >= sign_threshold
    if not bool(saturated.any().item() and transition.any().item() and sign_mask.any().item()):
        raise ValueError("teacher mean metrics require saturated, transition, and sign-bearing target elements")
    zero_mse = target.square().mean()
    zero_transition_mse = target[transition].square().mean()
    if float(zero_mse.item()) <= 0.0:
        raise ValueError("teacher mean zero-action baseline must have positive error")
    mse = squared_error.mean()
    previous_mse = (previous_action - target).square().mean() if previous_action is not None else None
    previous_squared_error = (previous_action - target).square() if previous_action is not None else None
    return TeacherMeanMetrics(
        mse=float(mse.item()),
        mae=float((prediction - target).abs().mean().item()),
        saturated_mse=float(squared_error[saturated].mean().item()),
        transition_mse=float(squared_error[transition].mean().item()),
        saturated_fraction=float(saturated.float().mean().item()),
        sign_accuracy=float((prediction[sign_mask].sign() == target[sign_mask].sign()).float().mean().item()),
        skill_vs_zero=float((1.0 - mse / zero_mse).item()),
        transition_skill_vs_zero=float(
            (
                1.0
                - squared_error[transition].mean()
                / zero_transition_mse.clamp_min(torch.finfo(target.dtype).eps)
            ).item()
        ),
        previous_action_mse=float(previous_mse.item()) if previous_mse is not None else None,
        skill_vs_previous_action=(
            float((1.0 - mse / previous_mse).item())
            if previous_mse is not None and float(previous_mse.item()) > 0.0
            else None
        ),
        previous_action_saturated_mse=(
            float(previous_squared_error[saturated].mean().item()) if previous_squared_error is not None else None
        ),
        previous_action_transition_mse=(
            float(previous_squared_error[transition].mean().item()) if previous_squared_error is not None else None
        ),
        transition_skill_vs_previous_action=(
            float(
                (
                    1.0
                    - squared_error[transition].mean()
                    / previous_squared_error[transition].mean().clamp_min(torch.finfo(target.dtype).eps)
                ).item()
            )
            if previous_squared_error is not None
            else None
        ),
    )


__all__ = [
    "TeacherMeanLossReduction",
    "TeacherMeanMetrics",
    "replica_modulo_split",
    "teacher_mean_loss",
    "teacher_mean_metrics",
    "training_batch_indices",
]
