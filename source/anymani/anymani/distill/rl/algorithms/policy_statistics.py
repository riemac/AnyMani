'Compute masked policy-distribution statistics in the latent Gaussian coordinates.'

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
    'Compute dimensionless masked KL between current and reference action distributions.'
    shape = current_mean.shape
    if current_mean.ndim != 2 or any(
        value.shape != shape for value in (current_sigma, reference_mean, reference_sigma, active_mask)
    ):
        raise ValueError("squashed KL requires aligned [B,J] tensors")
    if active_mask.dtype != torch.bool or not 0 < action_epsilon < 1:
        raise ValueError("squashed KL requires boolean masks and valid action epsilon")


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


    current_latent = torch.atanh(current.clamp(-1 + action_epsilon, 1 - action_epsilon))
    reference_latent = torch.atanh(reference.clamp(-1 + action_epsilon, 1 - action_epsilon))
    log_ratio = sigma.log() - old_sigma.log()
    per_joint = 0.5 * torch.expm1(2 * log_ratio) - log_ratio
    per_joint = per_joint + 0.5 * ((current_latent - reference_latent) / old_sigma).square()
    per_joint = per_joint.clamp_min(0.0)
    return (per_joint * active_mask).sum(dim=-1) / count
