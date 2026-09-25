"Stores canonical right-hand finger mounts in palm-local meters and radians; left mounts come from strict reflection of the complete hand."

from __future__ import annotations

import math

from ..asset_schema_core import PoseCfg
from ..units import cm, deg, m, rad

ALLEGRO_MOUNT_PRESET: dict[str, PoseCfg] = {
    "index": PoseCfg(pos=(m(0.0), m(0.0435), m(-0.001542)), rpy=(rad(-0.0873), 0.0, 0.0)),
    "middle": PoseCfg(pos=(m(0.0), m(0.0), m(0.0007)), rpy=(0.0, 0.0, 0.0)),
    "ring": PoseCfg(pos=(m(0.0), m(-0.0435), m(-0.001542)), rpy=(rad(0.0873), 0.0, 0.0)),
    "thumb": PoseCfg(pos=(m(-0.0182), m(0.019333), m(-0.045987)), rpy=(0.0, rad(-1.6581), rad(-1.5708))),
}
"Reviewed Allegro palm-local mount recipe."


LEAP_MOUNT_PRESET: dict[str, PoseCfg] = {
    "index": PoseCfg(pos=(m(-0.0070), m(0.0230), m(-0.0187)), rpy=(rad(1.5708), rad(1.5708), 0.0)),
    "middle": PoseCfg(pos=(m(-0.0071), m(-0.0224), m(-0.0187)), rpy=(rad(1.5708), rad(1.5708), 0.0)),
    "ring": PoseCfg(pos=(m(-0.00709), m(-0.0678), m(-0.0187)), rpy=(rad(1.5708), rad(1.5708), 0.0)),
    "thumb": PoseCfg(pos=(m(-0.0693), m(-0.0012), m(-0.0216)), rpy=(0.0, rad(1.5708), 0.0)),
}
"Reviewed LEAP palm-local mount recipe."


LEAP_SINGLE_BOX_MOUNT_PRESET: dict[str, PoseCfg] = {
    "thumb": PoseCfg(pos=(cm(3.7), cm(3.1), cm(1.0)), rpy=(0.0, 0.0, rad(-math.pi / 2.0))),
    "index": PoseCfg(pos=(cm(4.6), cm(8.0), cm(0.8)), rpy=(0.0, 0.0, 0.0)),
    "middle": PoseCfg(pos=(0.0, cm(8.0), cm(0.8)), rpy=(0.0, 0.0, 0.0)),
    "ring": PoseCfg(pos=(cm(-4.6), cm(8.0), cm(0.8)), rpy=(0.0, 0.0, 0.0)),
}
"Reviewed single-palm LEAP mount positions in meters."


ALLEGRO_SINGLE_BOX_MOUNT_PRESET: dict[str, PoseCfg] = {

    "thumb": PoseCfg(pos=(cm(2.45), cm(3.05), cm(-1.45)), rpy=(0.0, 0.0, rad(-1.65806278845))),
    "index": PoseCfg(pos=(cm(4.4), cm(9.44), cm(0.9)), rpy=(0.0, 0.0, deg(-5.0))),
    "middle": PoseCfg(pos=(0.0, cm(9.44), cm(0.9)), rpy=(0.0, 0.0, 0.0)),
    "ring": PoseCfg(pos=(cm(-4.4), cm(9.44), cm(0.9)), rpy=(0.0, 0.0, deg(5.0))),
}
"Reviewed single-palm Allegro mount positions in meters."


MOUNT_PRESET_REGISTRY: dict[str, dict[str, PoseCfg]] = {
    "allegro": ALLEGRO_MOUNT_PRESET,
    "leap": LEAP_MOUNT_PRESET,
    "com_allegro": ALLEGRO_MOUNT_PRESET,
    "single_box_allegro": ALLEGRO_SINGLE_BOX_MOUNT_PRESET,
    "com_leap": LEAP_MOUNT_PRESET,
    "single_box_leap": LEAP_SINGLE_BOX_MOUNT_PRESET,
}
"Registry of palm-local mount recipes."


def get_mount_preset(name: str, *, handedness: str | None = None) -> dict[str, PoseCfg]:
    'Returns mount preset.'

    try:
        preset = MOUNT_PRESET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown mount preset: {name!r}") from exc

    _ = handedness
    return {finger: pose.copy() for finger, pose in preset.items()}


__all__ = [
    "ALLEGRO_MOUNT_PRESET",
    "LEAP_MOUNT_PRESET",
    "ALLEGRO_SINGLE_BOX_MOUNT_PRESET",
    "LEAP_SINGLE_BOX_MOUNT_PRESET",
    "MOUNT_PRESET_REGISTRY",
    "get_mount_preset",
]
