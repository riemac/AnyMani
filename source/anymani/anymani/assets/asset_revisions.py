"Creates provenance-linked hand revisions as new bundles while retaining the parent assets."

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace

from .asset_schema_core import JointLimitCfg
from .asset_schema_embodiment import HandCfg


@dataclass(frozen=True)
class JointUpperLimitRevision:
    "New source-qualified revision recording which joint upper limits changed."

    finger_name: str
    joint_name: str
    old_upper_rad: float
    new_upper_rad: float


def widen_two_joint_allegro_flexion(
    hand: HandCfg, *, upper_rad: float = 2.23
) -> tuple[HandCfg, tuple[JointUpperLimitRevision, ...]]:
    "Returns a new Allegro revision that widens the declared two-joint flexion limits and preserves its parent identity."

    if not math.isfinite(upper_rad) or upper_rad <= 0.0:
        raise ValueError("requested upper limit must be finite and positive (rad)")
    revised = deepcopy(hand)
    edits: list[JointUpperLimitRevision] = []



    maps = [
        entry["slot_family_map"]
        for key in ("premade_topology", "premade_connectivity")
        if isinstance((entry := hand.metadata.get(key)), dict) and "slot_family_map" in entry
    ]
    if not maps:
        raise ValueError("joint-limit revision requires explicit generated slot_family_map")
    if any(mapping != maps[0] for mapping in maps[1:]):
        raise ValueError("generated slot_family_map declarations disagree")
    family_by_slot = maps[0]

    for finger in revised.fingers:
        if finger.name == "thumb":
            continue
        if finger.name not in family_by_slot:
            raise ValueError(f"missing explicit finger family for slot {finger.name!r}")
        if str(family_by_slot[finger.name]).lower() != "allegro":
            continue
        moving = [joint for joint in finger.joints if joint.joint_type == "revolute"]
        if len(moving) != 2:
            continue
        first, flexion = moving
        if (first.name, first.child, flexion.name, flexion.child) != (
            f"{finger.name}_j0",
            f"{finger.name}_mcp1",
            f"{finger.name}_j1",
            f"{finger.name}_mcp2",
        ):
            raise ValueError(f"two-joint finger {finger.name!r} must explicitly retain MCP1 and MCP2")
        limit = flexion.limit
        if not isinstance(limit, JointLimitCfg):
            raise ValueError(f"flexion joint {flexion.name!r} lacks a typed finite limit")
        if upper_rad <= limit.upper:
            continue
        edits.append(JointUpperLimitRevision(finger.name, flexion.name, limit.upper, float(upper_rad)))
        flexion.limit = replace(limit, upper=float(upper_rad))
    return revised, tuple(edits)


__all__ = ["JointUpperLimitRevision", "widen_two_joint_allegro_flexion"]
