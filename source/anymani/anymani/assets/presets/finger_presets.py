"Defines family-specific finger link and joint recipes."

from __future__ import annotations

from ..units import cm
from ..builder.finger_buiders import (
    AllegroFingerBuilderCfg,
    LeapFingerBuilderCfg,
    RegularFingerBuilderCfg,
    RegularThumbBuilderCfg,
)
from .physical_presets import apply_physical_profile_to_finger_cfg

"Defines finger construction recipes by palm family and anatomical type; joint values align with the proximal-to-distal child-link chain."
ALLEGRO_FINGER_PRESET = AllegroFingerBuilderCfg(
    name="index",
    num_joints=4,
    width=cm(2.7),
    height=cm(2.0),
    length=[cm(1.8), cm(5.4), cm(3.8), cm(2.2)],
    mesh_offsets=[0.0, 0.0, 0.0, cm(-0.6)],
    axes=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    tip={"type": "cs", "radius": cm(1.2), "height": cm(1.0)},
)


"Reviewed Allegro finger builder recipe."
ALLEGRO_THUMB_PRESET = RegularThumbBuilderCfg(
    name="thumb",
    lengths=[cm(4.5), cm(1.7), cm(4.3), cm(4.0)],
    cmc1_width=cm(3.5),
    cmc1_height=cm(3.4),
    width=cm(1.9),
    height=cm(2.7),
    cmc1_offset=(cm(0.9), cm(1.45)),
    non_cmc1_offset=[cm(-0.2), 0.0, cm(-0.9)],
    axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
    tip={"type": "cs", "radius": cm(1.2), "height": cm(1.0)},
)


"Reviewed Allegro thumb builder recipe."
LEAP_FINGER_PRESET = LeapFingerBuilderCfg(
    name="index",
    num_joints=4,
    width=cm(3.4),
    height=cm(2.05),
    length=[cm(3.9), cm(1.5), cm(3.6), cm(2.0)],
    mesh_offsets=[0.0, 0.0, 0.0, 0.0],
    fixed_part=cm(1.3),
    axes=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    tip={"type": "cs", "radius": cm(1.2), "height": cm(1.8)},
)

"Reviewed LEAP finger builder recipe."
LEAP_THUMB_PRESET = RegularThumbBuilderCfg(
    name="thumb",
    lengths=[cm(2.8), cm(1.7), cm(4.7), cm(2.3)],
    cmc1_width=cm(2.30),
    cmc1_height=cm(2.67),
    width=cm(2.3),
    height=cm(3.47),
    cmc1_offset=(0.0, cm(-0.33)),
    non_cmc1_offset=[0.0, 0.0, 0.0],
    axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
    tip={"type": "cs", "radius": cm(1.2), "height": cm(1.8)},
)


"Reviewed LEAP thumb builder recipe."
FINGER_PRESET_REGISTRY: dict[str, RegularFingerBuilderCfg] = {
    "allegro_non_thumb_v1": ALLEGRO_FINGER_PRESET,
    "leap_non_thumb_v1": LEAP_FINGER_PRESET,
    "allegro_thumb_v1": ALLEGRO_THUMB_PRESET,
    "leap_thumb_v1": LEAP_THUMB_PRESET,
}


def get_finger_builder_preset(name: str) -> RegularFingerBuilderCfg:
    'Returns finger builder preset.'

    try:
        cfg = FINGER_PRESET_REGISTRY[name].copy()
    except KeyError as exc:
        raise KeyError(f"Unknown finger builder preset: {name!r}") from exc
    return apply_physical_profile_to_finger_cfg(name, cfg)


__all__ = [
    "ALLEGRO_FINGER_PRESET",
    "LEAP_FINGER_PRESET",
    "ALLEGRO_THUMB_PRESET",
    "LEAP_THUMB_PRESET",
    "FINGER_PRESET_REGISTRY",
    "get_finger_builder_preset",
]
