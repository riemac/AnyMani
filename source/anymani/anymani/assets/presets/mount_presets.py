"""hand-level mount preset。

这一层表达的是“finger 根部相对于 palm frame 的挂载位姿”。
它与 finger/palm 自身几何是不同层级的信息，因此应独立存放。

把 mount preset 从 builder 子目录挪出来之后，hand 装配层和 palm preset 层
都可以引用同一份 mount 语义，而不会再制造“某个 builder 私藏一份挂载字典”的耦合。
"""

from __future__ import annotations

from ..asset_schema_core import PoseCfg


ALLEGRO_MOUNT_PRESET: dict[str, PoseCfg] = {
    "index": PoseCfg(pos=(0.0, 0.0435, -0.001542), rpy=(-0.0873, 0.0, 0.0)),
    "middle": PoseCfg(pos=(0.0, 0.0, 0.0007), rpy=(0.0, 0.0, 0.0)),
    "ring": PoseCfg(pos=(0.0, -0.0435, -0.001542), rpy=(0.0873, 0.0, 0.0)),
    "thumb": PoseCfg(pos=(-0.0182, 0.019333, -0.045987), rpy=(0.0, -1.6581, -1.5708)),
}
"""Allegro 挂载点 preset。"""


LEAP_MOUNT_PRESET: dict[str, PoseCfg] = {
    "index": PoseCfg(pos=(-0.0070, 0.0230, -0.0187), rpy=(1.5708, 1.5708, 0.0)),
    "middle": PoseCfg(pos=(-0.0071, -0.0224, -0.0187), rpy=(1.5708, 1.5708, 0.0)),
    "ring": PoseCfg(pos=(-0.00709, -0.0678, -0.0187), rpy=(1.5708, 1.5708, 0.0)),
    "thumb": PoseCfg(pos=(-0.0693, -0.0012, -0.0216), rpy=(0.0, 1.5708, 0.0)),
}
"""LEAP 挂载点 preset。"""


MOUNT_PRESET_REGISTRY: dict[str, dict[str, PoseCfg]] = {
    "allegro": ALLEGRO_MOUNT_PRESET,
    "leap": LEAP_MOUNT_PRESET,
    "com_allegro": ALLEGRO_MOUNT_PRESET,
    "single_box_allegro": ALLEGRO_MOUNT_PRESET,
    "com_leap": LEAP_MOUNT_PRESET,
    "single_box_leap": LEAP_MOUNT_PRESET,
}
"""挂载点 preset 注册表。

这里显式同时保留：

- family 名
- palm recipe 名

这样 hand-level 的 mount 解析就不需要再知道“某个别名最终对应哪个 family”。
"""


def get_mount_preset(name: str) -> dict[str, PoseCfg]:
    r"""按名字返回一份 mount preset 副本。"""

    try:
        preset = MOUNT_PRESET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown mount preset: {name!r}") from exc
    return {finger: pose.copy() for finger, pose in preset.items()}


__all__ = [
    "ALLEGRO_MOUNT_PRESET",
    "LEAP_MOUNT_PRESET",
    "MOUNT_PRESET_REGISTRY",
    "get_mount_preset",
]
