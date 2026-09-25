# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

(
    'Robot runtime configuration package. robots is the embodiment adapter: it '
    'consumes generated hand bundles from assets and lowers them to Isaac Lab '
    'robot/articulation configs. Tasks consume these configs and should not '
    'duplicate spawn/importer details. Keep this package lazy: contract tests '
    'import hand_spawn with Isaac Lab stubs, so eager imports of leap.py would '
    'load real Isaac Lab configs and break the default no-simulator Python test '
    'path.'
)

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_HAND_ANCHOR_POS_E",
    "HandActuatorSpawnCfg",
    "CanonicalRuntimeCfg",
    "HandFrameCfg",
    "HandJointInitCfg",
    "HandSpawnAdapter",
    "HandSpawnCfg",
    "HandUrdfSpawnCfg",
    "LEAP_HAND_CFG",
    "LEAP_HAND_URDF_CFG",
    "LEAP_HAND_URDF_PATH",
    "audit_official_hand_cfg",
    "build_official_hand_cfg",
    "build_official_hand_spawn_audit",
    "resolve_official_joint_indices",
    "resolve_official_link_name",
    "source_joint_to_usd_joint_name",
    "source_link_to_usd_link_name",
]


def __getattr__(name: str) -> Any:
    'Lazily export robot configs and the generated hand-spawn adapter.'

    if name == "LEAP_HAND_CFG":
        from .leap import LEAP_HAND_CFG

        return LEAP_HAND_CFG
    if name in {"LEAP_HAND_URDF_CFG", "LEAP_HAND_URDF_PATH"}:
        from . import leap_urdf

        return getattr(leap_urdf, name)
    if name in {
        "DEFAULT_HAND_ANCHOR_POS_E",
        "HandActuatorSpawnCfg",
        "CanonicalRuntimeCfg",
        "HandFrameCfg",
        "HandJointInitCfg",
        "HandSpawnAdapter",
        "HandSpawnCfg",
        "HandUrdfSpawnCfg",
    }:
        from . import hand_spawn

        return getattr(hand_spawn, name)
    if name in {
        "audit_official_hand_cfg",
        "build_official_hand_cfg",
        "build_official_hand_spawn_audit",
        "resolve_official_joint_indices",
        "resolve_official_link_name",
        "source_joint_to_usd_joint_name",
        "source_link_to_usd_link_name",
    }:
        from . import official_hand_spawn

        return getattr(official_hand_spawn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
