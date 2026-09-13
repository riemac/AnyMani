r"""有界掌旋策略的分布统计；均值坐标和概率分布保持一致。"""

from __future__ import annotations

import torch


def mean_preserving_squashed_kl(
    current_mean: torch.Tensor,
    current_sigma: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_sigma: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    action_epsilon: float = 1.0e-6,
) -> torch.Tensor:
    r"""返回每状态、每有效关节平均的D_KL(current || reference)。

    mean是[-1,1]中的确定性动作中心；潜Normal中心为atanh(mean)，sigma是潜空间标准差。
    tanh为两个分布共同的可逆变换，KL在该变换下不变。令u=log(sigma_current/sigma_reference)，
    单关节KL为0.5*expm1(2u)-u+0.5*((m_current-m_reference)/sigma_reference)^2。
    使用expm1避免相近尺度时相减损失精度；不向方差分母加epsilon，因此同分布KL精确为零。

    Args:
        current_mean: 当前有界动作中心[B,J]，无量纲。
        current_sigma: 当前正的潜标准差[B,J]，无量纲。
        reference_mean: 参考有界动作中心[B,J]。
        reference_sigma: 参考正的潜标准差[B,J]。
        active_mask: bool[B,J]；ghost不参与分子、分母或有限性检查。
        action_epsilon: 与生产likelihood相同的atanh边界裕量。

    Returns:
        [B] KL，单位为每有效关节nats。联合动作KL需乘相应有效关节数，不能混用阈值。
    """
    shape = current_mean.shape
    if current_mean.ndim != 2 or any(
        value.shape != shape for value in (current_sigma, reference_mean, reference_sigma, active_mask)
    ):
        raise ValueError("squashed KL requires aligned [B,J] tensors")
    if active_mask.dtype != torch.bool or not 0 < action_epsilon < 1:
        raise ValueError("squashed KL requires boolean masks and valid action epsilon")

    # 先移除ghost，避免无效sigma/mean在log或除法中生成NaN后再乘零。
    current = torch.where(active_mask, current_mean, torch.zeros_like(current_mean))
    reference = torch.where(active_mask, reference_mean, torch.zeros_like(reference_mean))
    sigma = torch.where(active_mask, current_sigma, torch.ones_like(current_sigma))
    old_sigma = torch.where(active_mask, reference_sigma, torch.ones_like(reference_sigma))
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        (torch.isfinite(sigma) & torch.isfinite(old_sigma) & (sigma > 0) & (old_sigma > 0)).all(),
        "active policy sigma must be finite and positive",
    )
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        (torch.isfinite(current) & torch.isfinite(reference)).all(), "active policy means must be finite"
    )
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        ((current.abs() <= 1 + action_epsilon) & (reference.abs() <= 1 + action_epsilon)).all(),
        "policy means must remain in bounded action coordinates",
    )
    count = active_mask.sum(dim=-1)
    torch._assert_async((count > 0).all(), "policy KL needs at least one active joint")  # pyright: ignore[reportPrivateImportUsage]

    # 与mean-preserving tanh likelihood采用同一个潜中心；不是把有界mean当Normal的location。
    current_latent = torch.atanh(current.clamp(-1 + action_epsilon, 1 - action_epsilon))
    reference_latent = torch.atanh(reference.clamp(-1 + action_epsilon, 1 - action_epsilon))
    log_ratio = sigma.log() - old_sigma.log()
    per_joint = 0.5 * torch.expm1(2 * log_ratio) - log_ratio
    per_joint = per_joint + 0.5 * ((current_latent - reference_latent) / old_sigma).square()
    per_joint = per_joint.clamp_min(0.0)  # 精确表达式非负，仅约束浮点舍入下界。
    return (per_joint * active_mask).sum(dim=-1) / count
