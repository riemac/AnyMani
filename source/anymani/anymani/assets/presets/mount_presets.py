"""hand-level mount preset。

这一层表达的是“finger 根部相对于 palm frame 的挂载位姿”。
它与 finger/palm 自身几何是不同层级的信息，因此应独立存放。

把 mount preset 从 builder 子目录挪出来之后，hand 装配层和 palm preset 层
都可以引用同一份 mount 语义，而不会再制造“某个 builder 私藏一份挂载字典”的耦合。
"""

from __future__ import annotations

import math

from ..asset_schema_core import PoseCfg
from .units import cm, deg, rad


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


LEAP_SINGLE_BOX_MOUNT_PRESET: dict[str, PoseCfg] = {
    "thumb": PoseCfg(pos=(cm(3.7), cm(3.1), cm(1.0)), rpy=(0.0, 0.0, rad(-math.pi / 2.0))),
    "index": PoseCfg(pos=(cm(4.6), cm(8.0), cm(0.8)), rpy=(0.0, 0.0, 0.0)),
    "middle": PoseCfg(pos=(0.0, cm(8.0), cm(0.8)), rpy=(0.0, 0.0, 0.0)),
    "ring": PoseCfg(pos=(cm(-4.6), cm(8.0), cm(0.8)), rpy=(0.0, 0.0, 0.0)),
}
"""Single-box LEAP palm 的显式挂载点 preset（右手基准）。

这组数值不是“从 family 猜”的，而是直接来自用户早期写在
`HumanLikeHandBuilder.build()` TODO 里的 single-palm 记录。
"""


ALLEGRO_SINGLE_BOX_MOUNT_PRESET: dict[str, PoseCfg] = {
    "thumb": PoseCfg(pos=(cm(2.45), cm(3.05), cm(-1.45)), rpy=(0.0, 0.0, rad(-math.pi / 2.0))),
    "index": PoseCfg(pos=(cm(4.4), cm(9.44), cm(0.9)), rpy=(0.0, 0.0, deg(-5.0))),
    "middle": PoseCfg(pos=(0.0, cm(9.44), cm(0.9)), rpy=(0.0, 0.0, 0.0)),
    "ring": PoseCfg(pos=(cm(-4.4), cm(9.44), cm(0.9)), rpy=(0.0, 0.0, deg(5.0))),
}
"""Single-box Allegro palm 的显式挂载点 preset（右手基准）。"""


MOUNT_PRESET_REGISTRY: dict[str, dict[str, PoseCfg]] = {
    "allegro": ALLEGRO_MOUNT_PRESET,
    "leap": LEAP_MOUNT_PRESET,
    "com_allegro": ALLEGRO_MOUNT_PRESET,
    "single_box_allegro": ALLEGRO_SINGLE_BOX_MOUNT_PRESET,
    "com_leap": LEAP_MOUNT_PRESET,
    "single_box_leap": LEAP_SINGLE_BOX_MOUNT_PRESET,
}
"""挂载点 preset 注册表。

这里显式同时保留：

- family 名
- palm recipe 名

这样 hand-level 的 mount 解析就不需要再知道“某个别名最终对应哪个 family”。
"""


_LEFT_THUMB_MIRROR_PRESETS = {"single_box_allegro", "single_box_leap"}
"""当前只有 single-box palm 的左右手镜像规则有明确科研语义说明。

# Question:
真实 family 的 `allegro` / `leap` / `com_*` 左右手挂载规则目前并没有同样明确的
注释来源；因此这轮只对 single-box 系列应用显式左手 thumb 镜像。
"""


def _mirror_thumb_pose_for_left(pose: PoseCfg) -> PoseCfg:
    r"""把 single-box 右手 thumb 挂载点镜像成左手版本。

    规则直接来自用户原始 TODO：

    1. 关于 palm frame，左/右手都保持同一套 `x/y/z` 约定；
    2. thumb 的位置相对 `y-z` 平面对称，因此只翻转 `x`；
    3. yaw 的符号反转，其余约定保持不变。
    """

    return PoseCfg(pos=(-pose.pos[0], pose.pos[1], pose.pos[2]), rpy=(pose.rpy[0], pose.rpy[1], -pose.rpy[2]))


def get_mount_preset(name: str, *, handedness: str = "right") -> dict[str, PoseCfg]:
    r"""按名字返回一份 mount preset 副本，并在需要时做 handedness 修正。"""

    try:
        preset = MOUNT_PRESET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown mount preset: {name!r}") from exc

    mounts = {finger: pose.copy() for finger, pose in preset.items()}
    if handedness == "left" and name in _LEFT_THUMB_MIRROR_PRESETS and "thumb" in mounts:
        mounts["thumb"] = _mirror_thumb_pose_for_left(mounts["thumb"])
    return mounts


__all__ = [
    "ALLEGRO_MOUNT_PRESET",
    "LEAP_MOUNT_PRESET",
    "ALLEGRO_SINGLE_BOX_MOUNT_PRESET",
    "LEAP_SINGLE_BOX_MOUNT_PRESET",
    "MOUNT_PRESET_REGISTRY",
    "get_mount_preset",
]
