r"""Structured heterogeneous actor的backend-independent masked diagonal Normal数学。

PPO likelihood ratio使用active-joint log-prob总和；entropy、KL、bounds与regularization按active count均值。
Ghost dimension的distribution参数可取$\mu=0,\sigma=1$，但sample/action在送入环境前严格乘mask。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskedNormalParameters:
    r"""广播后的masked Normal参数。"""

    mean: torch.Tensor  # $[B,J]$，ghost 0
    log_std: torch.Tensor  # $[B,J]$，ghost 0、active共享scalar
    sigma: torch.Tensor  # $[B,J]$，ghost 1
    active_mask: torch.Tensor  # bool$[B,J]$


def masked_normal_parameters(
    mean: torch.Tensor,
    global_log_std: torch.Tensor,
    active_mask: torch.Tensor,
) -> MaskedNormalParameters:
    r"""把actor mean与单scalar$\log\sigma$规约为masked parameters。"""

    if mean.ndim != 2 or active_mask.shape != mean.shape or active_mask.dtype != torch.bool:
        raise ValueError("mean and bool active_mask must share [B,J] shape")
    if global_log_std.numel() != 1:
        raise ValueError("global_log_std must contain one scalar")
    torch._assert_async(torch.all(torch.isfinite(global_log_std)), "global_log_std must be finite")
    active_float = active_mask.to(dtype=mean.dtype)
    masked_mean = mean * active_float
    log_std = global_log_std.reshape(1, 1).expand_as(mean) * active_float
    sigma = torch.exp(log_std)  # ghost$\sigma=1$；其概率项随后被mask排除
    return MaskedNormalParameters(masked_mean, log_std, sigma, active_mask)


def masked_sample(parameters: MaskedNormalParameters, noise: torch.Tensor | None = None) -> torch.Tensor:
    r"""重参数化采样并清零ghost actions。"""

    epsilon = torch.randn_like(parameters.mean) if noise is None else noise
    if epsilon.shape != parameters.mean.shape:
        raise ValueError("Normal sampling noise must match mean shape")
    sample = parameters.mean + parameters.sigma * epsilon
    return sample * parameters.active_mask.to(dtype=sample.dtype)


def masked_negative_log_probability(actions: torch.Tensor, parameters: MaskedNormalParameters) -> torch.Tensor:
    r"""对active dimensions求diagonal-Normal negative log-prob总和。"""

    if actions.shape != parameters.mean.shape:
        raise ValueError("actions and Normal parameters must share shape")
    per_joint = (
        0.5 * ((actions - parameters.mean) / parameters.sigma).square()
        + parameters.log_std
        + 0.5 * math.log(2.0 * math.pi)
    )
    return (per_joint * parameters.active_mask.to(dtype=per_joint.dtype)).sum(dim=-1)


def masked_entropy(parameters: MaskedNormalParameters) -> torch.Tensor:
    r"""按active joint数均值计算Normal entropy。"""

    per_joint = 0.5 * math.log(2.0 * math.pi * math.e) + parameters.log_std
    weights = parameters.active_mask.to(dtype=per_joint.dtype)
    return (per_joint * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def masked_policy_kl(
    current: MaskedNormalParameters,
    previous: MaskedNormalParameters,
) -> torch.Tensor:
    r"""按active count均值计算$D_{KL}(\pi_{old}\|\pi_{new})$的rl_games形式。"""

    if current.mean.shape != previous.mean.shape:
        raise ValueError("KL distributions must share shape and active mask")
    torch._assert_async(
        torch.all(current.active_mask == previous.active_mask),
        "KL distributions must share active mask",
    )
    c1 = torch.log(previous.sigma / current.sigma + 1.0e-5)
    c2 = (current.sigma.square() + (previous.mean - current.mean).square()) / (
        2.0 * (previous.sigma.square() + 1.0e-5)
    )
    weights = current.active_mask.to(dtype=current.mean.dtype)
    return ((c1 + c2 - 0.5) * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def masked_bound_loss(mean: torch.Tensor, active_mask: torch.Tensor, *, soft_bound: float = 1.1) -> torch.Tensor:
    r"""按active count均值计算超出$[-1.1,1.1]$的quadratic bound loss。"""

    if mean.shape != active_mask.shape or active_mask.dtype != torch.bool:
        raise ValueError("bound loss requires mean and bool mask with shared shape")
    high = torch.clamp_min(mean - soft_bound, 0.0).square()
    low = torch.clamp_max(mean + soft_bound, 0.0).square()
    weights = active_mask.to(dtype=mean.dtype)
    return ((high + low) * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def clipped_ppo_actor_loss(
    new_negative_log_probability: torch.Tensor,
    old_negative_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    r"""返回per-sample clipped PPO actor loss，ratio由active joint联合log-prob定义。"""

    if not (
        new_negative_log_probability.shape
        == old_negative_log_probability.shape
        == advantage.shape
    ):
        raise ValueError("PPO neglogp and advantage tensors must share shape")
    ratio = torch.exp(old_negative_log_probability - new_negative_log_probability)
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return torch.maximum(unclipped, clipped)


__all__ = [
    "MaskedNormalParameters",
    "clipped_ppo_actor_loss",
    "masked_bound_loss",
    "masked_entropy",
    "masked_negative_log_probability",
    "masked_normal_parameters",
    "masked_policy_kl",
    "masked_sample",
]
