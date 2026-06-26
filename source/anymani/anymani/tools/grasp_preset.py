"""Runtime data class for manual pre-grasp/contact-basin presets.

The calibrator writes human-edited YAML files. Training configs consume those
files through this IsaacLab-free adapter so the experiment cfg remains readable
without hand-copying long joint dictionaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anymani.tools._grasp_preset_helpers import (
    PRESET_DIR,
    PRESET_KIND,
    asset_preset_path,
    fixed_float_tuple,
    generated_asset_latest_preset_path,
    load_yaml_mapping,
    official_leap_latest_preset_path,
    require_mapping,
    safe_preset_slug,
    select_start_preset,
)


@dataclass(frozen=True)
class GraspPreset:
    """Parsed manual pre-grasp/contact-basin preset.

    Attributes are already converted to plain Python tuples/dicts so IsaacLab
    config declarations can consume them directly.
    """

    path: Path
    payload: Mapping[str, Any]
    asset: Mapping[str, Any]
    joint_pos_rad: dict[str, float]
    object_pos_cfg: tuple[float, float, float]
    object_rot_wxyz: tuple[float, float, float, float]
    object_rpy_xyz_rad: tuple[float, float, float] | None

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        expected_hand_source: str | Sequence[str] | None = None,
        expected_hand_ref_contains: str | Sequence[str] | None = None,
    ) -> GraspPreset:
        """Load and validate a preset YAML file.

        Args:
            path: YAML preset path.
            expected_hand_source: Optional allowed value(s) for
                ``asset.hand_source``.
            expected_hand_ref_contains: Optional substring(s) that must appear
                in ``asset.hand_ref``. This catches generated/official mixups.
        """

        preset_path = Path(path).expanduser().resolve()
        payload = load_yaml_mapping(preset_path)
        if payload.get("kind") != PRESET_KIND:
            raise ValueError(
                f"Preset {preset_path} has unsupported kind {payload.get('kind')!r}; "
                f"expected {PRESET_KIND!r}."
            )

        asset = require_mapping(payload.get("asset"), preset_path, "asset")
        _validate_asset_guard(
            preset_path,
            asset,
            expected_hand_source=expected_hand_source,
            expected_hand_ref_contains=expected_hand_ref_contains,
        )

        joint_pos = require_mapping(payload.get("joint_pos_rad"), preset_path, "joint_pos_rad")
        object_pose = require_mapping(payload.get("object_pose_cfg"), preset_path, "object_pose_cfg")
        object_rpy = object_pose.get("rpy_xyz_rad")

        return cls(
            path=preset_path,
            payload=payload,
            asset=asset,
            joint_pos_rad={str(joint_name): float(value) for joint_name, value in joint_pos.items()},
            object_pos_cfg=fixed_float_tuple(object_pose.get("pos"), preset_path, "object_pose_cfg.pos", 3),
            object_rot_wxyz=fixed_float_tuple(object_pose.get("rot_wxyz"), preset_path, "object_pose_cfg.rot_wxyz", 4),
            object_rpy_xyz_rad=fixed_float_tuple(object_rpy, preset_path, "object_pose_cfg.rpy_xyz_rad", 3)
            if object_rpy is not None
            else None,
        )


def _validate_asset_guard(
    path: Path,
    asset: Mapping[str, Any],
    *,
    expected_hand_source: str | Sequence[str] | None,
    expected_hand_ref_contains: str | Sequence[str] | None,
) -> None:
    """Fail fast when a config accidentally points at another asset preset."""

    if expected_hand_source is not None:
        allowed_sources = (expected_hand_source,) if isinstance(expected_hand_source, str) else tuple(expected_hand_source)
        hand_source = asset.get("hand_source")
        if hand_source not in allowed_sources:
            raise ValueError(
                f"Preset {path} belongs to hand_source={hand_source!r}, expected one of {allowed_sources!r}."
            )

    if expected_hand_ref_contains is not None:
        required_parts = (
            (expected_hand_ref_contains,)
            if isinstance(expected_hand_ref_contains, str)
            else tuple(expected_hand_ref_contains)
        )
        hand_ref = str(asset.get("hand_ref", ""))
        missing_parts = tuple(part for part in required_parts if part not in hand_ref)
        if missing_parts:
            raise ValueError(
                f"Preset {path} hand_ref={hand_ref!r} does not contain required substring(s) {missing_parts!r}."
            )


__all__ = [
    "GraspPreset",
    "PRESET_DIR",
    "asset_preset_path",
    "generated_asset_latest_preset_path",
    "official_leap_latest_preset_path",
    "safe_preset_slug",
    "select_start_preset",
]
