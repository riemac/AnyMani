"""Shared helpers for manual grasp preset files.

This module is intentionally pure Python: no IsaacLab imports and no GUI state.
Both the calibrator and training configs can use it without pulling in Kit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

TOOLS_DIR = Path(__file__).resolve().parent
"""Directory containing AnyMani tool scripts."""

PRESET_DIR = TOOLS_DIR / "presets"
"""Root directory for manual grasp presets."""

PRESET_KIND = "anymani_single_asset_grasp_preset"
"""YAML kind tag written by the manual grasp calibrator."""


def asset_preset_path(*parts: str, filename: str = "latest.yaml") -> Path:
    """Return a path inside the asset-scoped preset directory."""

    return PRESET_DIR.joinpath(*parts, filename)


def safe_preset_slug(raw_name: str) -> str:
    """Convert an asset name into a stable directory slug."""

    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw_name.strip())
    return slug or "unknown_asset"


def generated_asset_latest_preset_path(hand_bundle: str | None, *, default_hand_bundle_id: str) -> Path:
    """Return the default latest preset path for a generated hand bundle."""

    if hand_bundle is None:
        asset_slug = Path(default_hand_bundle_id).name
    else:
        asset_slug = Path(hand_bundle).expanduser().name or hand_bundle.rsplit("/", 1)[-1]
    return asset_preset_path("generated_asset", safe_preset_slug(asset_slug))


def official_leap_latest_preset_path() -> Path:
    """Return the default latest preset path for the official LEAP probe."""

    return asset_preset_path("official", "leap")


def load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    """Load a YAML file whose top level must be a mapping."""

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Preset {path} must contain a YAML mapping at top level.")
    return payload


def select_start_preset(
    preset_arg: str | None,
    *,
    default_latest_path: Path,
) -> tuple[Path | None, Mapping[str, Any] | None]:
    """Choose which preset should seed a calibrator session.

    Explicit ``--preset`` must load successfully. Without it, only the current
    asset-scoped ``latest.yaml`` is considered; there is no global fallback.
    """

    if preset_arg is not None:
        preset_path = Path(preset_arg).expanduser().resolve()
        return preset_path, load_yaml_mapping(preset_path)

    if default_latest_path.exists():
        return default_latest_path, load_yaml_mapping(default_latest_path)

    return None, None


def require_mapping(value: Any, path: Path, field_name: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or raise a field-specific error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"Preset {path} field {field_name!r} must be a mapping.")
    return value


def fixed_float_tuple(value: Any, path: Path, field_name: str, length: int) -> tuple[float, ...]:
    """Parse a fixed-length numeric sequence."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"Preset {path} field {field_name!r} must be a list of length {length}.")
    return tuple(float(item) for item in value)


__all__ = [
    "PRESET_DIR",
    "PRESET_KIND",
    "asset_preset_path",
    "fixed_float_tuple",
    "generated_asset_latest_preset_path",
    "load_yaml_mapping",
    "official_leap_latest_preset_path",
    "require_mapping",
    "safe_preset_slug",
    "select_start_preset",
]
