(
    'Pure-Torch builder for structured actor/critic joint observations. Preserve '
    'role and history axes; do not define flattened policy slices. Canonical '
    'storage uses 16 JOINT and 4 TIP slots. Apply validity masks to every dynamic '
    'channel so poisoned ghost position/target/action/contact values cannot leak.'
)

from __future__ import annotations

import math

import torch

from .runtime_state import CANONICAL_JOINT_COUNT, CANONICAL_TIP_COUNT, derive_tip_and_owner_masks


def _validate_joint_inputs(active_joint_mask: torch.Tensor, *values: torch.Tensor) -> None:
    'Validate joint tensors [B,16] and a bool mask.'

    expected = active_joint_mask.shape
    if active_joint_mask.ndim != 2 or expected[1] != CANONICAL_JOINT_COUNT:
        raise ValueError("joint observation mask must have shape [B,16]")
    if active_joint_mask.dtype != torch.bool:
        raise TypeError("joint observation mask must be bool")
    if any(value.shape != expected for value in values):
        raise ValueError("joint observation values must share [B,16] shape")
    if any(not bool(torch.isfinite(value).all().item()) for value in values):
        raise ValueError("joint observation values must be finite")


def broadcast_tip_contact_to_joints(tip_contact_bits: torch.Tensor, active_joint_mask: torch.Tensor) -> torch.Tensor:
    (
        'Broadcast index/middle/ring/thumb TIP bits over the depth-major JOINT axis. '
        'Joint index is j=4d+f, so c_j=c_f; then apply the joint mask to zero ghost '
        'slots.'
    )

    if tip_contact_bits.ndim != 2 or tip_contact_bits.shape[1] != CANONICAL_TIP_COUNT:
        raise ValueError("tip_contact_bits must have shape [B,4]")
    if tip_contact_bits.shape[0] != active_joint_mask.shape[0] or tip_contact_bits.dtype != torch.bool:
        raise TypeError("TIP contact batch must align with joint mask and use bool dtype")
    active_tip_mask, _ = derive_tip_and_owner_masks(active_joint_mask)
    valid_tip_bits = tip_contact_bits & active_tip_mask
    joint_bits = valid_tip_bits.unsqueeze(1).expand(-1, 4, -1).reshape(-1, CANONICAL_JOINT_COUNT)
    return joint_bits & active_joint_mask


def actor_joint_current(
    joint_pos_rad: torch.Tensor,
    joint_target_rad: torch.Tensor,
    previous_policy_action: torch.Tensor,
    active_joint_mask: torch.Tensor,
) -> torch.Tensor:
    'Build the current actor JOINT frame [q/pi, u/pi, a_(t-1)], shape [B,16,3].'

    _validate_joint_inputs(active_joint_mask, joint_pos_rad, joint_target_rad, previous_policy_action)
    frame = torch.stack((joint_pos_rad / math.pi, joint_target_rad / math.pi, previous_policy_action), dim=-1)
    return frame * active_joint_mask.unsqueeze(-1).to(dtype=frame.dtype)


def actor_joint_history_frame(
    joint_pos_rad: torch.Tensor,
    joint_target_rad: torch.Tensor,
    previous_policy_action: torch.Tensor,
    tip_contact_bits: torch.Tensor,
    active_joint_mask: torch.Tensor,
) -> torch.Tensor:
    (
        'Build one raw History30 frame [q/pi, u/pi, a_(t-1), c_tip(f(j))], shape '
        '[B,16,4]. ObservationManager/CircularBuffer owns the 30-step '
        'oldest-to-latest axis.'
    )

    current = actor_joint_current(joint_pos_rad, joint_target_rad, previous_policy_action, active_joint_mask)
    contact = broadcast_tip_contact_to_joints(tip_contact_bits, active_joint_mask).to(dtype=current.dtype)
    return torch.cat((current, contact.unsqueeze(-1)), dim=-1)


def actor_joint_contact_frame(
    joint_pos_rad: torch.Tensor,
    joint_target_rad: torch.Tensor,
    previous_policy_action: torch.Tensor,
    joint_owner_contact_bits: torch.Tensor,
    tip_contact_bits: torch.Tensor,
    active_joint_mask: torch.Tensor,
    *,
    tip_only: bool = False,
) -> torch.Tensor:
    (
        'Build the MVP actor JOINT frame [q/pi, u/pi, a_(t-1), c_j, c_tip(f(j))]. c_j '
        "is this JOINT owner's EMA binary contact; c_tip(f(j)) broadcasts the finger "
        'TIP bit across depth-major slots. Keep both channels for local link support '
        'and the validated TIP release/recontact event. Shape is [B,16,5], with all '
        'ghost channels zero. tip_only zeros c_j without changing storage for '
        'existing TCN weights. Use the same producer for current and History30 '
        'frames.'
    )

    _validate_joint_inputs(
        active_joint_mask,
        joint_pos_rad,
        joint_target_rad,
        previous_policy_action,
    )
    if joint_owner_contact_bits.shape != active_joint_mask.shape or joint_owner_contact_bits.dtype != torch.bool:
        raise ValueError("joint owner contact bits must be bool [B,16]")
    current = actor_joint_current(joint_pos_rad, joint_target_rad, previous_policy_action, active_joint_mask)
    own_contact = joint_owner_contact_bits & active_joint_mask  # This JOINT link's object contact.
    if tip_only:
        own_contact = torch.zeros_like(own_contact)  # Do not mutate contact truth shared with reward/critic terms.
    finger_tip_contact = broadcast_tip_contact_to_joints(tip_contact_bits, active_joint_mask)
    contacts = torch.stack((own_contact, finger_tip_contact), dim=-1).to(dtype=current.dtype)
    return torch.cat((current, contacts), dim=-1)  # `[B,16,5]`


def actor_owner_contact(
    owner_contact_bits: torch.Tensor, active_joint_mask: torch.Tensor, *, tip_only: bool = False
) -> torch.Tensor:
    'Return current binary contact tokens for PALM+JOINT16+TIP4, shape [B,21,1].'

    _, owner_mask = derive_tip_and_owner_masks(active_joint_mask)
    if owner_contact_bits.shape != owner_mask.shape or owner_contact_bits.dtype != torch.bool:
        raise ValueError("owner contact bits must be bool [B,21]")
    observed = owner_contact_bits & owner_mask
    if tip_only:
        observed[:, :17] = False  # Remove PALM/JOINT tactile inputs only; physical and geometric validity is unchanged.
    return observed.to(dtype=torch.float32).unsqueeze(-1)


def actor_joint_limits(soft_joint_limits_rad: torch.Tensor, active_joint_mask: torch.Tensor) -> torch.Tensor:
    'Return static normalized limits [q_min/pi, q_max/pi], shape [B,16,2]; ghost slots are zero.'

    if soft_joint_limits_rad.shape != (*active_joint_mask.shape, 2):
        raise ValueError("soft joint limits must have shape [B,16,2]")
    if not bool(torch.isfinite(soft_joint_limits_rad).all().item()):
        raise ValueError("soft joint limits must be finite")
    return (soft_joint_limits_rad / math.pi) * active_joint_mask.unsqueeze(-1).to(dtype=soft_joint_limits_rad.dtype)


def actor_tip_contact(tip_contact_bits: torch.Tensor, active_joint_mask: torch.Tensor) -> torch.Tensor:
    'Return TIP-only actor contact, shape [B,4,1]; inactive fingertips are zero.'

    active_tip_mask, _ = derive_tip_and_owner_masks(active_joint_mask)
    if tip_contact_bits.shape != active_tip_mask.shape or tip_contact_bits.dtype != torch.bool:
        raise ValueError("tip contact bits must be bool [B,4]")
    return (tip_contact_bits & active_tip_mask).to(dtype=torch.float32).unsqueeze(-1)


def critic_joint_state(
    joint_pos_rad: torch.Tensor,
    joint_vel_rad_s: torch.Tensor,
    joint_target_rad: torch.Tensor,
    previous_policy_action: torch.Tensor,
    active_joint_mask: torch.Tensor,
) -> torch.Tensor:
    'Build privileged JOINT frame [q/pi, qdot, u/pi, a_(t-1)], shape [B,16,4].'

    _validate_joint_inputs(
        active_joint_mask,
        joint_pos_rad,
        joint_vel_rad_s,
        joint_target_rad,
        previous_policy_action,
    )
    frame = torch.stack(
        (joint_pos_rad / math.pi, joint_vel_rad_s, joint_target_rad / math.pi, previous_policy_action), dim=-1
    )
    return frame * active_joint_mask.unsqueeze(-1).to(dtype=frame.dtype)


__all__ = [
    "actor_joint_current",
    "actor_joint_history_frame",
    "actor_joint_contact_frame",
    "actor_joint_limits",
    "actor_tip_contact",
    "actor_owner_contact",
    "broadcast_tip_contact_to_joints",
    "critic_joint_state",
]
