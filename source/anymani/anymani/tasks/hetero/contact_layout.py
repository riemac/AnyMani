'Canonical-v1 object-filtered contact ABI and owner-role map. Finger order is index/middle/ring/thumb and joint order is depth-major. The scene declares 24 single-link sensors; shared state stores TIP4, 19 finger non-tip contacts, and PALM1. A root collision belongs to the PALM geometry owner but may retain a finger non-tip reward role through an explicit mapping.'

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mdp.runtime_state import CANONICAL_OWNER_COUNT, derive_tip_and_owner_masks

PHYSX_FINGER_ORDER = ("index", "middle", "ring", "thumb")


@dataclass(frozen=True)
class HeterogeneousContactLayout:
    'Fixed sensor names, link chains, role slices, and owner indices.'

    fingertip_links: tuple[str, ...]
    finger_non_tip_links: tuple[str, ...]
    palm_link: str
    finger_link_chains: tuple[tuple[str, ...], ...]
    fingertip_sensor_names: tuple[str, ...]
    finger_non_tip_sensor_names: tuple[str, ...]
    palm_sensor_name: str
    sensor_owner_indices: tuple[int, ...]  # Storage state order is TIP + non-tip + PALM, mapped to 21 geometry owners.

    @property
    def state_sensor_names(self) -> tuple[str, ...]:
        'Return shared-state order: TIP, finger non-tip, then PALM.'

        return (*self.fingertip_sensor_names, *self.finger_non_tip_sensor_names, self.palm_sensor_name)

    @property
    def scene_sensor_names(self) -> tuple[str, ...]:
        'Return scene declaration order: TIP, PALM, then finger non-tip.'

        return (*self.fingertip_sensor_names, self.palm_sensor_name, *self.finger_non_tip_sensor_names)

    @property
    def scene_sensor_link_pairs(self) -> tuple[tuple[str, str], ...]:
        'Return one-to-one sensor/link pairs in scene order.'

        return (
            *tuple(zip(self.fingertip_sensor_names, self.fingertip_links, strict=True)),
            (self.palm_sensor_name, self.palm_link),
            *tuple(zip(self.finger_non_tip_sensor_names, self.finger_non_tip_links, strict=True)),
        )


def build_canonical_contact_layout() -> HeterogeneousContactLayout:
    'Build the canonical-v1 layout with four TIP, 19 non-tip, and one PALM owner.'

    tip_links = tuple(f"{finger}_tip" for finger in PHYSX_FINGER_ORDER)
    root_links = ("index_root", "middle_root", "ring_root")  # thumb_root has no collision geometry and receives no sensor.
    joint_links = tuple(
        f"{finger}_link_j{depth}" for finger in PHYSX_FINGER_ORDER for depth in range(4)
    )  # Sensors are declared finger-major and mapped explicitly to depth-major owners.
    non_tip_links = (*root_links, *joint_links)
    chains = tuple(
        (
            f"{finger}_root",
            *(f"{finger}_link_j{depth}" for depth in range(4)),
            f"{finger}_tip",
        )
        for finger in PHYSX_FINGER_ORDER
    )
    tip_owner_indices = tuple(17 + finger for finger in range(4))
    root_owner_indices = (0, 0, 0)  # Fixed finger roots belong to the PALM geometry owner.
    joint_owner_indices = tuple(
        1 + depth * 4 + finger for finger in range(4) for depth in range(4)
    )  # sensor finger-major -> owner depth-major
    owner_indices = (*tip_owner_indices, *root_owner_indices, *joint_owner_indices, 0)
    if any(index < 0 or index >= CANONICAL_OWNER_COUNT for index in owner_indices):
        raise AssertionError("canonical contact owner mapping exceeds 21-owner ABI")
    return HeterogeneousContactLayout(
        fingertip_links=tip_links,
        finger_non_tip_links=non_tip_links,
        palm_link="palm",
        finger_link_chains=chains,
        fingertip_sensor_names=tuple(f"contact_{link}" for link in tip_links),
        finger_non_tip_sensor_names=tuple(f"contact_{link}" for link in non_tip_links),
        palm_sensor_name="contact_palm",
        sensor_owner_indices=owner_indices,
    )


def active_contact_sensor_mask(
    active_joint_mask: torch.Tensor,
    layout: HeterogeneousContactLayout,
) -> torch.Tensor:
    'Derive state-order sensor validity with shape [B,24]. TIP follows finger validity; three non-thumb roots follow their fingers; 16 joint links map from finger-major sensor order to the depth-major joint mask. PALM is always valid; ghost/no-collision links remain structural zeros.'

    tip_mask, _ = derive_tip_and_owner_masks(active_joint_mask)
    root_mask = tip_mask[:, :3]
    by_depth_finger = active_joint_mask.reshape(active_joint_mask.shape[0], 4, 4)
    joint_sensor_mask = by_depth_finger.transpose(1, 2).reshape(active_joint_mask.shape[0], 16)
    palm_mask = torch.ones(active_joint_mask.shape[0], 1, dtype=torch.bool, device=active_joint_mask.device)
    sensor_mask = torch.cat((tip_mask, root_mask, joint_sensor_mask, palm_mask), dim=-1)
    if sensor_mask.shape[1] != len(layout.state_sensor_names):
        raise ValueError("derived contact mask disagrees with fixed sensor layout")
    return sensor_mask


def structural_collision_filter_pairs(
    palm_link_name: str,
    finger_link_chains: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, str], ...]:
    'Create undirected palm-finger and same-finger collision-filter pairs. Do not filter between different fingers; cross-finger collisions remain in PhysX.'

    if not palm_link_name or not finger_link_chains or any(not chain for chain in finger_link_chains):
        raise ValueError("structural collision filter requires palm and non-empty finger chains")
    pairs: set[tuple[str, str]] = set()
    for chain in finger_link_chains:
        for link in chain:
            pairs.add(_ordered_pair(palm_link_name, link))
        for left_index, left in enumerate(chain):
            for right in chain[left_index + 1 :]:
                pairs.add(_ordered_pair(left, right))
    return tuple(sorted(pairs))


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    r"""Canonicalize an undirected link pair。"""

    return (left, right) if left <= right else (right, left)


__all__ = [
    "PHYSX_FINGER_ORDER",
    "HeterogeneousContactLayout",
    "active_contact_sensor_mask",
    "build_canonical_contact_layout",
    "structural_collision_filter_pairs",
]
