"Exports named palm, finger, mount, and connectivity presets."

from __future__ import annotations

from typing import Any

from .connectivity_presets import (
    FINGER_CONNECTIVITY_PRESET_REGISTRY,
    HAND_CONNECTIVITY_PRESET_REGISTRY,
    NON_THUMB_SLOTS,
    FingerConnectivityPreset,
    HandConnectivityPreset,
    get_default_hand_connectivity_preset_name,
    get_finger_connectivity_preset_data,
    get_hand_connectivity_preset_data,
    list_hand_connectivity_preset_names,
)
from .finger_presets import (
    ALLEGRO_FINGER_PRESET,
    ALLEGRO_THUMB_PRESET,
    FINGER_PRESET_REGISTRY,
    LEAP_FINGER_PRESET,
    LEAP_THUMB_PRESET,
    get_finger_builder_preset,
)
from .hand_presets import (
    COM_PALM_ALLEGRO_HAND_PRESET,
    COM_PALM_LEAP_HAND_PRESET,
    HAND_PRESET_REGISTRY,
    SINGLE_PALM_ALLEGRO_HAND_PRESET,
    SINGLE_PALM_LEAP_HAND_PRESET,
    get_hand_builder_preset_data,
    make_human_like_builder_cfg_from_preset,
)
from .mount_presets import (
    ALLEGRO_MOUNT_PRESET,
    LEAP_MOUNT_PRESET,
    MOUNT_PRESET_REGISTRY,
    get_mount_preset,
)
from .palm_presets import (
    ALLEGRO_SINGLE_PALM_BOX_PRESET,
    COM_PALM_PRESET_DATA,
    LEAP_SINGLE_PALM_BOX_PRESET,
    PALM_PRESET_REGISTRY,
    get_com_palm_preset,
    get_com_palm_preset_data,
    get_single_palm_box_preset,
    get_single_palm_box_preset_data,
)
from .physical_presets import (
    ALLEGRO_NON_THUMB_PHYSICAL_PROFILE,
    ALLEGRO_THUMB_PHYSICAL_PROFILE,
    FINGER_PHYSICAL_PROFILE_REGISTRY,
    LEAP_NON_THUMB_PHYSICAL_PROFILE,
    LEAP_THUMB_PHYSICAL_PROFILE,
    JointPhysicalPreset,
    get_finger_physical_profile,
)
def resolve_palm_builder_cfg(raw: Any) -> Any:
    'Resolves palm builder cfg.'

    from .resolver import resolve_palm_builder_cfg as _impl

    return _impl(raw)


def resolve_finger_builder_cfg(raw: Any) -> Any:
    'Resolves finger builder cfg.'

    from .resolver import resolve_finger_builder_cfg as _impl

    return _impl(raw)


def resolve_finger_slot_builder_cfg(raw: Any) -> Any:
    'Resolves finger slot builder cfg.'

    from .resolver import resolve_finger_slot_builder_cfg as _impl

    return _impl(raw)


def resolve_human_like_mounts(
    *,
    family: str | None,
    handedness: str | None,
    palm_cfg: Any,
    mount_preset: str | None = None,
    mounts: dict[str, Any] | None = None,
):
    'Resolves human like mounts.'

    from .resolver import resolve_human_like_mounts as _impl

    return _impl(
        family=family,
        handedness=handedness,
        palm_cfg=palm_cfg,
        mount_preset=mount_preset,
        mounts=mounts,
    )


def resolve_human_like_builder_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    'Resolves human like builder kwargs.'

    from .resolver import resolve_human_like_builder_kwargs as _impl

    return _impl(raw)


def make_human_like_builder_cfg(**kwargs: Any):
    'Builds human like builder cfg.'

    from .resolver import make_human_like_builder_cfg as _impl

    return _impl(**kwargs)

__all__ = [
    "NON_THUMB_SLOTS",
    "FingerConnectivityPreset",
    "HandConnectivityPreset",
    "FINGER_CONNECTIVITY_PRESET_REGISTRY",
    "HAND_CONNECTIVITY_PRESET_REGISTRY",
    "get_finger_connectivity_preset_data",
    "get_hand_connectivity_preset_data",
    "list_hand_connectivity_preset_names",
    "get_default_hand_connectivity_preset_name",
    "ALLEGRO_FINGER_PRESET",
    "LEAP_FINGER_PRESET",
    "ALLEGRO_THUMB_PRESET",
    "LEAP_THUMB_PRESET",
    "FINGER_PRESET_REGISTRY",
    "get_finger_builder_preset",
    "SINGLE_PALM_ALLEGRO_HAND_PRESET",
    "SINGLE_PALM_LEAP_HAND_PRESET",
    "COM_PALM_ALLEGRO_HAND_PRESET",
    "COM_PALM_LEAP_HAND_PRESET",
    "HAND_PRESET_REGISTRY",
    "get_hand_builder_preset_data",
    "make_human_like_builder_cfg_from_preset",
    "ALLEGRO_MOUNT_PRESET",
    "LEAP_MOUNT_PRESET",
    "MOUNT_PRESET_REGISTRY",
    "get_mount_preset",
    "ALLEGRO_SINGLE_PALM_BOX_PRESET",
    "LEAP_SINGLE_PALM_BOX_PRESET",
    "COM_PALM_PRESET_DATA",
    "PALM_PRESET_REGISTRY",
    "get_single_palm_box_preset",
    "get_single_palm_box_preset_data",
    "get_com_palm_preset",
    "get_com_palm_preset_data",
    "JointPhysicalPreset",
    "LEAP_NON_THUMB_PHYSICAL_PROFILE",
    "LEAP_THUMB_PHYSICAL_PROFILE",
    "ALLEGRO_NON_THUMB_PHYSICAL_PROFILE",
    "ALLEGRO_THUMB_PHYSICAL_PROFILE",
    "FINGER_PHYSICAL_PROFILE_REGISTRY",
    "get_finger_physical_profile",
    "resolve_palm_builder_cfg",
    "resolve_finger_builder_cfg",
    "resolve_finger_slot_builder_cfg",
    "resolve_human_like_mounts",
    "resolve_human_like_builder_kwargs",
    "make_human_like_builder_cfg",
]
