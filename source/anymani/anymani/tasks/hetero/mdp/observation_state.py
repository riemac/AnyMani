r"""Structured actor/critic joint observation的纯Torch构造函数。

Task科学接口保留role与history axes，不定义1969D flat slices。Canonical storage为16 JOINT与4 TIP；所有
dynamic channels在返回前统一乘有效mask，ghost q/target/action/contact即使被上游poison也不能泄漏。
"""

from __future__ import annotations

import math

import torch

from .runtime_state import CANONICAL_JOINT_COUNT, CANONICAL_TIP_COUNT, derive_tip_and_owner_masks


def _validate_joint_inputs(active_joint_mask: torch.Tensor, *values: torch.Tensor) -> None:
    r"""验证一组$[B,16]$ joint tensors与bool mask。"""

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
    r"""把index/middle/ring/thumb TIP bits广播到depth-major JOINT axis。

    Joint index为$j=4d+f$，因此$c_j=c_f$；最终再乘joint mask使ghost为零。
    """

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
    r"""构造$O^a_{t,\mathrm{jnt}}$当前帧$[q/\pi,u/\pi,a_{t-1}]$，形状$[B,16,3]$。"""

    _validate_joint_inputs(active_joint_mask, joint_pos_rad, joint_target_rad, previous_policy_action)
    frame = torch.stack(
        (joint_pos_rad / math.pi, joint_target_rad / math.pi, previous_policy_action), dim=-1
    )
    return frame * active_joint_mask.unsqueeze(-1).to(dtype=frame.dtype)


def actor_joint_history_frame(
    joint_pos_rad: torch.Tensor,
    joint_target_rad: torch.Tensor,
    previous_policy_action: torch.Tensor,
    tip_contact_bits: torch.Tensor,
    active_joint_mask: torch.Tensor,
) -> torch.Tensor:
    r"""构造History30的单个raw frame$[q/\pi,u/\pi,a_{t-1},c_{tip(f(j))}]$。

    返回$[B,16,4]$；ObservationManager/CircularBuffer拥有$H=30$ oldest→latest轴。
    """

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
) -> torch.Tensor:
    r"""构造MVP actor逐JOINT帧$[q/\pi,u/\pi,a_{t-1},c_j,c_{tip(f(j))}]$。

    $c_j$是该JOINT owner自身对object的EMA binary contact；$c_{tip(f(j))}$把所属finger的TIP bit广播到
    depth-major joint slots。二者同时保留，使local TCN既感知当前link支撑，也继承N000已验证的TIP
    release-recontact事件。返回形状`[B,16,5]`，ghost五个通道严格为零。
    """

    _validate_joint_inputs(
        active_joint_mask,
        joint_pos_rad,
        joint_target_rad,
        previous_policy_action,
    )
    if joint_owner_contact_bits.shape != active_joint_mask.shape or joint_owner_contact_bits.dtype != torch.bool:
        raise ValueError("joint owner contact bits must be bool [B,16]")
    current = actor_joint_current(joint_pos_rad, joint_target_rad, previous_policy_action, active_joint_mask)
    own_contact = joint_owner_contact_bits & active_joint_mask  # 本JOINT link的object contact
    finger_tip_contact = broadcast_tip_contact_to_joints(tip_contact_bits, active_joint_mask)
    contacts = torch.stack((own_contact, finger_tip_contact), dim=-1).to(dtype=current.dtype)
    return torch.cat((current, contacts), dim=-1)  # `[B,16,5]`


def actor_owner_contact(owner_contact_bits: torch.Tensor, active_joint_mask: torch.Tensor) -> torch.Tensor:
    r"""返回PALM+JOINT16+TIP4当前binary contact tokens，形状`[B,21,1]`。"""

    _, owner_mask = derive_tip_and_owner_masks(active_joint_mask)
    if owner_contact_bits.shape != owner_mask.shape or owner_contact_bits.dtype != torch.bool:
        raise ValueError("owner contact bits must be bool [B,21]")
    return (owner_contact_bits & owner_mask).to(dtype=torch.float32).unsqueeze(-1)


def actor_joint_limits(soft_joint_limits_rad: torch.Tensor, active_joint_mask: torch.Tensor) -> torch.Tensor:
    r"""返回静态$[q_{min}/\pi,q_{max}/\pi]$，形状$[B,16,2]$，ghost为零。"""

    if soft_joint_limits_rad.shape != (*active_joint_mask.shape, 2):
        raise ValueError("soft joint limits must have shape [B,16,2]")
    if not bool(torch.isfinite(soft_joint_limits_rad).all().item()):
        raise ValueError("soft joint limits must be finite")
    return (soft_joint_limits_rad / math.pi) * active_joint_mask.unsqueeze(-1).to(
        dtype=soft_joint_limits_rad.dtype
    )


def actor_tip_contact(tip_contact_bits: torch.Tensor, active_joint_mask: torch.Tensor) -> torch.Tensor:
    r"""返回TIP-only actor contact$[B,4,1]$，inactive fingertips为零。"""

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
    r"""构造privileged$[q/\pi,\dot q,u/\pi,a_{t-1}]$，形状$[B,16,4]$。"""

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
