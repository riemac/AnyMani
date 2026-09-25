'Measure active-joint action rejected by joint limits when all valid fingertips are silent.'

from __future__ import annotations

import math

import torch

_POLICY_STEP_AUTHORITY_RAD = 1.0 / 24.0  # units rad
_REJECTION_DIAGNOSTIC_THRESHOLD = 1.0e-6


def _validate_action_regularization_shapes(
    mean: torch.Tensor,
    target_normalized: torch.Tensor,
    limits_normalized: torch.Tensor,
    joint_valid: torch.Tensor,
    tip_contact: torch.Tensor,
    tip_valid: torch.Tensor,
) -> None:
    'Validate action regularization shapes; shapes [B,J], [B,J,2], [B,F].'

    rank = mean.ndim
    if rank not in (1, 2):
        raise ValueError(f"mean must have shape [B,J] (or rank-1 vmap sample), got {tuple(mean.shape)}")

    expected_joint_shape = tuple(mean.shape)  # shapes [B,J], [J]
    if target_normalized.shape != mean.shape:
        raise ValueError(
            f"target_normalized must have shape {expected_joint_shape} matching mean, got {tuple(target_normalized.shape)}"
        )
    if tuple(limits_normalized.shape[:-1]) != expected_joint_shape or limits_normalized.shape[-1] != 2:
        raise ValueError(
            "limits_normalized must have shape [B,J,2] (or [J,2] for vmap sample), "
            f"got {tuple(limits_normalized.shape)}"
        )
    if joint_valid.shape != mean.shape:
        raise ValueError(f"joint_valid must have shape {expected_joint_shape}, got {tuple(joint_valid.shape)}")

    expected_tip_prefix = tuple(mean.shape[:-1])  # shapes [B]
    for name, value in (("tip_contact", tip_contact), ("tip_valid", tip_valid)):
        if value.ndim != rank or tuple(value.shape[:-1]) != expected_tip_prefix:
            raise ValueError(
                f"{name} must have shape [B,F] (or [F] for vmap sample) aligned with mean, got {tuple(value.shape)}"
            )
    if tip_contact.shape != tip_valid.shape:
        raise ValueError(
            f"tip_contact and tip_valid must share shape, got {tuple(tip_contact.shape)} and {tuple(tip_valid.shape)}"
        )

    floating_inputs = (mean, target_normalized, limits_normalized)
    if any(not torch.is_floating_point(value) for value in floating_inputs):
        raise TypeError("mean, target_normalized and limits_normalized must be floating-point tensors")
    if any(value.dtype != mean.dtype for value in floating_inputs[1:]):
        raise TypeError(
            "mean, target_normalized and limits_normalized must share dtype, "
            f"got {mean.dtype}, {target_normalized.dtype}, {limits_normalized.dtype}"
        )

    all_inputs = (target_normalized, limits_normalized, joint_valid, tip_contact, tip_valid)
    if any(value.device != mean.device for value in all_inputs):
        raise ValueError("mean, target_normalized, limits_normalized and masks must share one device")


def tip_silent_rejected_action_cost(
    *,
    mean: torch.Tensor,
    target_normalized: torch.Tensor,
    limits_normalized: torch.Tensor,
    joint_valid: torch.Tensor,
    tip_contact: torch.Tensor,
    tip_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    'Handle TIP silent rejected action cost; shapes [B,J], [J], [l,h]; units rad, m.'

    _validate_action_regularization_shapes(
        mean,
        target_normalized,
        limits_normalized,
        joint_valid,
        tip_contact,
        tip_valid,
    )

    joint_mask = joint_valid.to(dtype=torch.bool)  # shapes [B,J]
    tip_mask = tip_valid.to(dtype=torch.bool)  # shapes [B,F]
    contact_mask = tip_contact.to(dtype=torch.bool)  # shapes [B,F]


    clean_mean = torch.where(joint_mask, mean, torch.zeros_like(mean))
    clean_target = torch.where(joint_mask, target_normalized, torch.zeros_like(target_normalized))
    clean_limits = torch.where(
        joint_mask.unsqueeze(-1), limits_normalized, torch.zeros_like(limits_normalized)
    )  # shapes [l_j,h_j], [0,0]


    effective_contact = torch.where(tip_mask, contact_mask, torch.zeros_like(contact_mask))
    silent_gate = tip_mask.any(dim=-1) & ~effective_contact.any(dim=-1)


    authority_scale = clean_mean.new_tensor(1.0 / _POLICY_STEP_AUTHORITY_RAD)
    lower_action = authority_scale * math.pi * (clean_limits[..., 0] - clean_target)
    upper_action = authority_scale * math.pi * (clean_limits[..., 1] - clean_target)
    accepted_action = torch.clamp(clean_mean, min=lower_action, max=upper_action)  # action-space accepted portion
    rejected = clean_mean - accepted_action


    joint_weight = joint_mask.to(dtype=mean.dtype)  # Dimensionless active-joint mask.
    valid_joint_count = joint_weight.sum(dim=-1).clamp_min(1.0)  # shapes [B]
    gate_weight = silent_gate.to(dtype=mean.dtype)


    cost = gate_weight * (joint_weight * rejected.square()).sum(dim=-1) / valid_joint_count  # shapes [B]
    rejection_event = rejected.abs() > _REJECTION_DIAGNOSTIC_THRESHOLD
    rejected_fraction = (
        gate_weight * (joint_weight * rejection_event.to(dtype=mean.dtype)).sum(dim=-1) / valid_joint_count
    )  # shapes [B]
    return cost, rejected_fraction


__all__ = ["tip_silent_rejected_action_cost"]
