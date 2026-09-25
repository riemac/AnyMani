"Defines stable, explicit named right-hand recipes. The complete HandCfg is mirrored only after palm, mounts, joints, and meshes are assembled."

from __future__ import annotations

from copy import deepcopy
from typing import Any



SINGLE_PALM_ALLEGRO_HAND_PRESET: dict[str, Any] = {
    "name": "single_palm_allegro",
    "family": "allegro",
    "handedness": "right",
    "palm_cfg": "single_box_allegro",
    "finger_cfg": "allegro_non_thumb_v1",
    "thumb_cfg": "allegro_thumb_v1",
}




SINGLE_PALM_LEAP_HAND_PRESET: dict[str, Any] = {
    "name": "single_palm_leap",
    "family": "leap",
    "handedness": "right",
    "palm_cfg": "single_box_leap",  # single-box LEAP palm
    "finger_cfg": "leap_non_thumb_v1",
    "thumb_cfg": "leap_thumb_v1",
}




COM_PALM_ALLEGRO_HAND_PRESET: dict[str, Any] = {
    "name": "com_palm_allegro",
    "family": "allegro",
    "handedness": "right",
    "palm_cfg": "com_allegro",
    "finger_cfg": "allegro_non_thumb_v1",
    "thumb_cfg": "allegro_thumb_v1",
}




COM_PALM_LEAP_HAND_PRESET: dict[str, Any] = {
    "name": "com_palm_leap",
    "family": "leap",
    "handedness": "right",
    "palm_cfg": "com_leap",
    "finger_cfg": "leap_non_thumb_v1",
    "thumb_cfg": "leap_thumb_v1",
}


HAND_PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    "single_palm_allegro": SINGLE_PALM_ALLEGRO_HAND_PRESET,
    "single_palm_leap": SINGLE_PALM_LEAP_HAND_PRESET,
    "com_palm_allegro": COM_PALM_ALLEGRO_HAND_PRESET,
    "com_palm_leap": COM_PALM_LEAP_HAND_PRESET,
}
"Registry of named base-hand builder configurations."


def get_hand_builder_preset_data(name: str) -> dict[str, Any]:
    'Returns hand builder preset data.'

    try:
        return deepcopy(HAND_PRESET_REGISTRY[name])
    except KeyError as exc:
        raise KeyError(f"Unknown hand builder preset: {name!r}") from exc


def make_human_like_builder_cfg_from_preset(preset_name: str, **overrides: Any):
    'Builds human like builder cfg from preset.'

    from .resolver import make_human_like_builder_cfg

    payload = get_hand_builder_preset_data(preset_name)
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return make_human_like_builder_cfg(**payload)


__all__ = [
    "SINGLE_PALM_ALLEGRO_HAND_PRESET",
    "SINGLE_PALM_LEAP_HAND_PRESET",
    "COM_PALM_ALLEGRO_HAND_PRESET",
    "COM_PALM_LEAP_HAND_PRESET",
    "HAND_PRESET_REGISTRY",
    "get_hand_builder_preset_data",
    "make_human_like_builder_cfg_from_preset",
]
