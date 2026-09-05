r"""Accepted N000 native trajectory到当前canonical actor packet的无损数据变换。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

N000_JOINT_COUNT = 16
N000_TIP_COUNT = 4
N000_OWNER_COUNT = 21
N000_HISTORY_LENGTH = 30
N000_TEACHER_FRAME_WIDTH = 3 * N000_JOINT_COUNT + N000_TIP_COUNT


@dataclass(frozen=True)
class N000StudentActorPacket:
    r"""与`PalmRotationActorObservation`同shape、但尚未附加N040的监督输入。"""

    jnt_current: torch.Tensor  # `[B,16,5]`，q/u/a/own-contact/TIP-contact
    jnt_history: torch.Tensor  # `[B,30,16,5]`
    jnt_limits: torch.Tensor  # `[B,16,2]`，rad除以pi
    owner_contact: torch.Tensor  # `[B,21,1]`，binary FP32
    jnt_valid: torch.Tensor  # bool `[B,16]`
    tip_valid: torch.Tensor  # bool `[B,4]`
    owner_valid: torch.Tensor  # bool `[B,21]`


def canonical_from_native_indices(
    native_joint_names: Sequence[str],
    canonical_joint_names: Sequence[str],
) -> tuple[int, ...]:
    r"""返回使`native[..., indices]`成为canonical轴的name-based permutation。"""

    native = tuple(str(name) for name in native_joint_names)
    canonical = tuple(str(name) for name in canonical_joint_names)
    if not native or len(set(native)) != len(native):
        raise ValueError("N000 native joint names must be non-empty and unique")
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("N000 canonical joint names must be non-empty and unique")
    if set(native) != set(canonical):
        missing = tuple(name for name in canonical if name not in native)
        extra = tuple(name for name in native if name not in canonical)
        raise ValueError(f"N000 native/canonical joint names disagree: missing={missing}, extra={extra}")
    native_index = {name: index for index, name in enumerate(native)}
    return tuple(native_index[name] for name in canonical)


def native_from_canonical_indices(canonical_from_native: Sequence[int]) -> tuple[int, ...]:
    r"""反演`canonical[...,c]=native[...,p[c]]`，供student canonical action回写native环境轴。"""

    permutation = tuple(int(index) for index in canonical_from_native)
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("canonical-from-native indices must form a complete permutation")
    inverse = [0] * len(permutation)
    for canonical_index, native_index in enumerate(permutation):
        inverse[native_index] = canonical_index
    return tuple(inverse)


def sensor_owner_indices_from_sidecar(
    sidecar: Mapping[str, Any],
    *,
    state_sensor_links: Sequence[str],
    canonical_joint_names: Sequence[str],
    canonical_finger_names: Sequence[str],
) -> tuple[int, ...]:
    r"""按旧GM contact-state link顺序恢复21-owner indices。

    映射只读取sidecar中的joint name、child link、joint type与TIP标记：revolute child归对应JOINT，TIP child
    归对应TIP，fixed root与palm都归PALM。它不按link字符串排序或猜测finger-major/depth-major位置。
    """

    hand_cfg = _mapping(sidecar.get("hand_cfg"), "sidecar['hand_cfg']")
    palm_cfg = _mapping(hand_cfg.get("palm"), "sidecar hand_cfg.palm")
    palm_name = _string(palm_cfg.get("name"), "sidecar palm.name")
    joint_names = tuple(str(name) for name in canonical_joint_names)
    finger_names = tuple(str(name) for name in canonical_finger_names)
    if len(joint_names) != N000_JOINT_COUNT or len(set(joint_names)) != len(joint_names):
        raise ValueError("N000 bridge requires 16 unique canonical joint names")
    if len(finger_names) != N000_TIP_COUNT or len(set(finger_names)) != len(finger_names):
        raise ValueError("N000 bridge requires four unique canonical finger names")
    joint_owner = {name: 1 + index for index, name in enumerate(joint_names)}
    finger_owner = {name: 17 + index for index, name in enumerate(finger_names)}
    link_owner: dict[str, int] = {palm_name: 0}

    fingers = hand_cfg.get("fingers")
    if not isinstance(fingers, Sequence) or isinstance(fingers, (str, bytes)):
        raise TypeError("sidecar hand_cfg.fingers must be a sequence")
    for raw_finger in fingers:
        finger = _mapping(raw_finger, "sidecar finger")
        finger_name = _string(finger.get("name"), "sidecar finger.name")
        if finger_name not in finger_owner:
            raise ValueError(f"sidecar finger {finger_name!r} is absent from canonical finger names")
        joints = finger.get("joints")
        if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)):
            raise TypeError(f"sidecar finger {finger_name!r} joints must be a sequence")
        for raw_joint in joints:
            joint = _mapping(raw_joint, f"sidecar finger {finger_name!r} joint")
            child = _string(joint.get("child"), f"sidecar finger {finger_name!r} joint.child")
            name = _string(joint.get("name"), f"sidecar finger {finger_name!r} joint.name")
            if bool(joint.get("is_tip", False)):
                owner = finger_owner[finger_name]
            elif str(joint.get("joint_type", "")).lower() == "revolute":
                if name not in joint_owner:
                    raise ValueError(f"sidecar revolute joint {name!r} is absent from canonical joint names")
                owner = joint_owner[name]
            else:
                owner = 0  # generated fixed root segments share PALM geometry ownership
            if child in link_owner and link_owner[child] != owner:
                raise ValueError(f"sidecar child link {child!r} maps to conflicting owners")
            link_owner[child] = owner

    state_links = tuple(str(link) for link in state_sensor_links)
    if not state_links or len(set(state_links)) != len(state_links):
        raise ValueError("N000 contact-state links must be non-empty and unique")
    missing_links = tuple(link for link in state_links if link not in link_owner)
    if missing_links:
        raise ValueError(f"N000 contact-state links are absent from sidecar ownership: {missing_links}")
    return tuple(link_owner[link] for link in state_links)


def contact_bits_to_owner(
    contact_bits: torch.Tensor,
    sensor_owner_indices: Sequence[int],
    *,
    owner_count: int = N000_OWNER_COUNT,
) -> torch.Tensor:
    r"""把任意前缀shape的sensor bits以逻辑OR归并到canonical owner轴。"""

    indices = tuple(int(index) for index in sensor_owner_indices)
    if contact_bits.dtype != torch.bool or contact_bits.ndim < 1 or contact_bits.shape[-1] != len(indices):
        raise ValueError("N000 contact bits must be bool with final sensor axis matching owner indices")
    if owner_count < 1 or any(index < 0 or index >= owner_count for index in indices):
        raise ValueError("N000 sensor owner index lies outside the requested owner axis")
    owner = torch.zeros(*contact_bits.shape[:-1], owner_count, dtype=torch.bool, device=contact_bits.device)
    for sensor_index, owner_index in enumerate(indices):
        owner[..., owner_index] |= contact_bits[..., sensor_index]
    return owner


def build_n000_student_actor_packet(
    *,
    teacher_history_native: torch.Tensor,
    own_joint_contact_history: torch.Tensor,
    owner_contact: torch.Tensor,
    soft_joint_limits_native_rad: torch.Tensor,
    canonical_from_native: Sequence[int],
) -> N000StudentActorPacket:
    r"""把同状态N000 observation与runtime contact组装成当前student actor输入。"""

    batch = teacher_history_native.shape[0]
    expected_history = (batch, N000_HISTORY_LENGTH, N000_TEACHER_FRAME_WIDTH)
    if teacher_history_native.shape != expected_history or not teacher_history_native.is_floating_point():
        raise ValueError(f"N000 teacher history must be floating {expected_history}")
    if not bool(torch.isfinite(teacher_history_native).all().item()):
        raise ValueError("N000 teacher history must be finite")
    if own_joint_contact_history.shape != (batch, N000_HISTORY_LENGTH, N000_JOINT_COUNT):
        raise ValueError("N000 own-joint contact history must have shape [B,30,16]")
    if own_joint_contact_history.dtype != torch.bool:
        raise TypeError("N000 own-joint contact history must be bool")
    if owner_contact.shape != (batch, N000_OWNER_COUNT) or owner_contact.dtype != torch.bool:
        raise ValueError("N000 owner contact must be bool [B,21]")
    if soft_joint_limits_native_rad.shape != (batch, N000_JOINT_COUNT, 2):
        raise ValueError("N000 native soft joint limits must have shape [B,16,2]")
    if not bool(torch.isfinite(soft_joint_limits_native_rad).all().item()):
        raise ValueError("N000 native soft joint limits must be finite")
    tensors = (own_joint_contact_history, owner_contact, soft_joint_limits_native_rad)
    if any(value.device != teacher_history_native.device for value in tensors):
        raise ValueError("N000 teacher history, contacts, and limits must share one device")

    permutation = tuple(int(index) for index in canonical_from_native)
    if sorted(permutation) != list(range(N000_JOINT_COUNT)):
        raise ValueError("N000 canonical-from-native mapping must be a permutation of 0..15")
    index = torch.tensor(permutation, dtype=torch.long, device=teacher_history_native.device)
    q_native, u_native, action_native, tip_history = torch.split(
        teacher_history_native,
        [N000_JOINT_COUNT, N000_JOINT_COUNT, N000_JOINT_COUNT, N000_TIP_COUNT],
        dim=-1,
    )
    q = q_native.index_select(-1, index)
    target = u_native.index_select(-1, index)
    action = action_native.index_select(-1, index)
    if not bool(torch.all((tip_history == 0.0) | (tip_history == 1.0)).item()):
        raise ValueError("N000 teacher TIP history must contain exact binary values")
    tip_by_joint = tip_history.repeat(1, 1, N000_JOINT_COUNT // N000_TIP_COUNT)
    jnt_history = torch.stack(
        (q, target, action, own_joint_contact_history.to(dtype=q.dtype), tip_by_joint),
        dim=-1,
    )
    limits_index = index.to(device=soft_joint_limits_native_rad.device)
    limits = soft_joint_limits_native_rad.index_select(1, limits_index) / math.pi
    return N000StudentActorPacket(
        jnt_current=jnt_history[:, -1],
        jnt_history=jnt_history,
        jnt_limits=limits,
        owner_contact=owner_contact.to(dtype=torch.float32).unsqueeze(-1),
        jnt_valid=torch.ones(batch, N000_JOINT_COUNT, dtype=torch.bool, device=q.device),
        tip_valid=torch.ones(batch, N000_TIP_COUNT, dtype=torch.bool, device=q.device),
        owner_valid=torch.ones(batch, N000_OWNER_COUNT, dtype=torch.bool, device=q.device),
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    r"""收窄sidecar动态YAML节点并保留定位信息。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _string(value: Any, name: str) -> str:
    r"""读取非空sidecar标识符。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "N000StudentActorPacket",
    "build_n000_student_actor_packet",
    "canonical_from_native_indices",
    "contact_bits_to_owner",
    "native_from_canonical_indices",
    "sensor_owner_indices_from_sidecar",
]
