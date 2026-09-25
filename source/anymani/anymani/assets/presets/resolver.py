"Resolves declarative preset names into typed builder configurations."

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..asset_schema_core import PoseCfg
from ..builder.finger_buiders import (
    AllegroFingerBuilderCfg,
    LeapFingerBuilderCfg,
    RegularThumbBuilderCfg,
)
from ..builder.hand_builders import HumanLikeHandBuilderCfg
from ..builder.palm_builders import ComPalmBuilderCfg, SinglePalmBuilderCfg
from .finger_presets import get_finger_builder_preset
from .mount_presets import get_mount_preset
from .palm_presets import get_com_palm_preset, get_single_palm_box_preset


_FINGER_SLOT_NAMES = {"index", "middle", "ring", "little"}


def _to_pose_dict(values: dict[str, Any] | None) -> dict[str, PoseCfg]:

    return {name: PoseCfg.from_value(value) for name, value in (values or {}).items()}


def resolve_palm_builder_cfg(raw: Any) -> Any:
    'Resolves palm builder cfg.'

    if isinstance(raw, (ComPalmBuilderCfg, SinglePalmBuilderCfg)):
        return raw
    if isinstance(raw, str):
        if raw.startswith("com_"):
            return get_com_palm_preset(raw.removeprefix("com_"))
        if raw.startswith("single_box_"):
            return get_single_palm_box_preset(raw.removeprefix("single_box_"))
        raise ValueError(f"Unsupported palm preset string: {raw!r}")
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported palm cfg payload: {raw!r}")

    payload = deepcopy(raw)
    if "preset" in payload:
        return ComPalmBuilderCfg(**payload)
    return SinglePalmBuilderCfg(**payload)


def resolve_finger_builder_cfg(raw: Any) -> Any:
    'Resolves finger builder cfg.'

    if raw is None:
        return None
    if isinstance(raw, (AllegroFingerBuilderCfg, LeapFingerBuilderCfg, RegularThumbBuilderCfg)):
        return raw
    if isinstance(raw, str):
        return get_finger_builder_preset(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported finger cfg payload: {raw!r}")

    payload = deepcopy(raw)
    preset_name = payload.pop("preset_name", payload.pop("preset", None))
    if isinstance(preset_name, str):
        return get_finger_builder_preset(preset_name).replace(**payload)

    thumb_keys = {"lengths", "cmc1_width", "cmc1_height", "cmc1_offset", "non_cmc1_offset"}
    if thumb_keys & set(payload):
        return RegularThumbBuilderCfg(**payload)
    if "fixed_part" in payload:
        return LeapFingerBuilderCfg(**payload)
    return AllegroFingerBuilderCfg(**payload)


def resolve_finger_slot_builder_cfg(raw: Any) -> Any:
    'Resolves finger slot builder cfg.'

    if isinstance(raw, dict) and raw and set(raw).issubset(_FINGER_SLOT_NAMES):
        return {name: resolve_finger_builder_cfg(cfg) for name, cfg in raw.items()}
    return resolve_finger_builder_cfg(raw)


def resolve_human_like_mounts(
    *,
    family: str | None,
    handedness: str | None,
    palm_cfg: Any,
    mount_preset: str | None = None,
    mounts: dict[str, Any] | None = None,
) -> dict[str, PoseCfg]:
    'Resolves human like mounts.'

    candidate_names: list[str] = []
    if mount_preset is not None:
        candidate_names.append(mount_preset)
    if isinstance(palm_cfg, ComPalmBuilderCfg):
        candidate_names.append(f"com_{palm_cfg.preset}")
    if isinstance(palm_cfg, SinglePalmBuilderCfg) and palm_cfg.shape == "box" and family:
        candidate_names.append(f"single_box_{family}")
    if family:
        candidate_names.append(family)

    resolved_from_preset: dict[str, PoseCfg] = {}
    for preset_name in candidate_names:
        try:
            resolved_from_preset = get_mount_preset(preset_name, handedness=handedness or "right")
            break
        except KeyError:
            continue

    return {**resolved_from_preset, **_to_pose_dict(mounts)}


def resolve_human_like_builder_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    'Resolves human like builder kwargs.'

    data = deepcopy(raw)
    if "palm_cfg" in data:
        data["palm_cfg"] = resolve_palm_builder_cfg(data["palm_cfg"])
    if "finger_cfg" in data:
        data["finger_cfg"] = resolve_finger_slot_builder_cfg(data["finger_cfg"])
    if "thumb_cfg" in data:
        data["thumb_cfg"] = resolve_finger_builder_cfg(data["thumb_cfg"])

    mount_preset = data.pop("mount_preset", None)
    data["mounts"] = resolve_human_like_mounts(
        family=data.get("family"),
        handedness=data.get("handedness"),
        palm_cfg=data.get("palm_cfg"),
        mount_preset=mount_preset,
        mounts=data.get("mounts"),
    )
    return data


def make_human_like_builder_cfg(**kwargs: Any) -> HumanLikeHandBuilderCfg:
    'Builds human like builder cfg.'

    return HumanLikeHandBuilderCfg(**resolve_human_like_builder_kwargs(kwargs))


__all__ = [
    "resolve_palm_builder_cfg",
    "resolve_finger_builder_cfg",
    "resolve_finger_slot_builder_cfg",
    "resolve_human_like_mounts",
    "resolve_human_like_builder_kwargs",
    "make_human_like_builder_cfg",
]
