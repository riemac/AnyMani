'Select deterministic mean or seeded sample actions from a frozen policy.'

from __future__ import annotations

from typing import Literal

import torch

from anymani.distill.models.palm_rotation_policy import expanded_policy_log_std

from .palm_rotation_network import PalmRotationMaskedContinuousModel

LATENT_ACTION_EPSILON = PalmRotationMaskedContinuousModel.Network._ACTION_EPS


def validate_frozen_action_mode(mode: str, seed: int | None) -> None:
    'Validate frozen action mode.'
    if mode not in {"mean", "sample"}:
        raise ValueError("action mode must be mean or sample")
    if mode == "mean" and seed is not None:
        raise ValueError("mean action mode does not consume an action seed")
    if mode == "sample" and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("sample action mode requires an explicit non-negative 63-bit action seed")


def select_frozen_actor_actions(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    joint_valid: torch.Tensor,
    *,
    mode: Literal["mean", "sample"] = "mean",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    'Select frozen Actor actions; shapes [B,16].'
    if mode == "mean":
        if generator is not None:
            raise ValueError("mean action mode must not receive an action generator")
        return mean, {}
    if mode != "sample" or generator is None:
        raise ValueError("sample action mode requires its own generator")
    if mean.ndim != 2 or mean.shape[-1] != 16 or joint_valid.shape != mean.shape or joint_valid.dtype != torch.bool:
        raise ValueError("frozen sampling expects aligned [B,16] mean and boolean joint mask")  # canonical ABI
    if mean.dtype != torch.float32 or log_std.dtype != torch.float32:
        raise ValueError("frozen sampling requires FP32 center and log-standard-deviation")


    sigma = torch.exp(expanded_policy_log_std(log_std, mean, joint_valid))
    latent_mean = PalmRotationMaskedContinuousModel.Network._action_to_latent(mean)
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        torch.all(torch.isfinite(sigma) & (sigma > 0)), "invalid frozen sigma"
    )
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
        torch.all(mean.abs() <= 1 + 1e-6), "frozen action center escaped bounds"
    )
    latent_sample = torch.normal(latent_mean, sigma, generator=generator)
    action = torch.tanh(latent_sample) * joint_valid.to(dtype=mean.dtype)  # shapes [B,16]
    return action, {
        "policy_action_mean": mean,
        "policy_latent_sigma": sigma,
        "policy_latent_sample": latent_sample,  # units M
        "policy_joint_valid": joint_valid,
    }
