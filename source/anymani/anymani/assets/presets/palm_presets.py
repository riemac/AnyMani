"""palm preset 常量、注册表与原始数据表。

和 finger preset 不同，palm 这里分成两类：

1. `single_box_*`：本质上是参数化 palm 的锚点，因此可以直接存成尺寸字典
2. `com_*`：本质上是“真实 hand family 的复合碰撞体 recipe”，builder 需要读取
   其中的 collision/inertial/mount 组织信息来 lower 成 `PalmCfg`

因此本模块一方面提供“对外返回 builder cfg”的工厂函数，另一方面也显式暴露
`COM_PALM_PRESET_DATA` 给 `ComPalmBuilder` 读取原始 recipe。
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..builder.palm_builders import ComPalmBuilderCfg, SinglePalmBuilderCfg


ALLEGRO_SINGLE_PALM_BOX_PRESET: dict[str, Any] = {
    "shape": "box",
    "width": 0.112,
    "length": 0.0944,
    "height": 0.042,
}
"""Allegro 单体 box palm 的参数锚点。"""


LEAP_SINGLE_PALM_BOX_PRESET: dict[str, Any] = {
    "shape": "box",
    "width": 0.12,
    "length": 0.08,
    "height": 0.046,
}
"""LEAP 单体 box palm 的参数锚点。"""


COM_PALM_PRESET_DATA: dict[str, dict[str, Any]] = {
    "allegro": {
        "collisions": [
            {"size": (0.0414, 0.1120, 0.0448), "origin": (-0.0090, 0.0000, -0.0230)},
            {"size": (0.0414, 0.0538, 0.0428), "origin": (-0.0090, -0.0253, -0.0667)},
            {"size": (0.0414, 0.0720, 0.0130), "origin": (-0.0093, -0.00557, -0.08874)},
        ],
        "inertial": {
            "mass": 0.4154,
            "origin": (0.0, 0.0, 0.0),
            "inertia": {"ixx": 1.0e-4, "iyy": 1.0e-4, "izz": 1.0e-4},
        },
        "mount_preset": "allegro",
    },
    "leap": {
        "collisions": [
            {"size": (0.022, 0.026, 0.034), "origin": (-0.009, 0.008, -0.011)},
            {"size": (0.022, 0.026, 0.034), "origin": (-0.009, -0.037, -0.011)},
            {"size": (0.022, 0.026, 0.034), "origin": (-0.00709, -0.0678, -0.0187)},
            {"size": (0.058, 0.020, 0.046), "origin": (-0.066, -0.078, -0.0115), "rpy": (0.0, 0.0, -0.2967)},
            {"size": (0.020, 0.120, 0.030), "origin": (-0.030, -0.035, -0.003)},
            {"size": (0.010, 0.120, 0.020), "origin": (-0.032, -0.035, -0.024), "rpy": (0.0, 0.785, 0.0)},
            {"size": (0.024, 0.116, 0.046), "origin": (-0.048, -0.033, -0.0115)},
            {"size": (0.044, 0.052, 0.046), "origin": (-0.078, -0.053, -0.0115)},
            {"size": (0.004, 0.036, 0.034), "origin": (-0.098, -0.009, -0.006)},
            {"size": (0.044, 0.056, 0.004), "origin": (-0.078, -0.003, 0.010)},
        ],
        "inertial": {
            "mass": 0.237,
            "origin": (0.0, 0.0, 0.0),
            "inertia": {
                "ixx": 3.54094e-4,
                "ixy": -1.193e-6,
                "ixz": -2.445e-6,
                "iyy": 2.60915e-4,
                "iyz": -2.905e-6,
                "izz": 5.29257e-4,
            },
        },
        "mount_preset": "leap",
    },
}
"""复合 palm preset 的原始几何/惯量/挂载 recipe。"""


def get_single_palm_box_preset(name: str) -> "SinglePalmBuilderCfg":
    r"""按名字返回一份单一 box palm preset cfg。"""

    from ..builder.palm_builders import SinglePalmBuilderCfg

    single_preset_registry = {
        "allegro": ALLEGRO_SINGLE_PALM_BOX_PRESET,
        "leap": LEAP_SINGLE_PALM_BOX_PRESET,
    }
    try:
        payload = deepcopy(single_preset_registry[name])
    except KeyError as exc:
        raise KeyError(f"Unknown single palm box preset: {name!r}") from exc
    return SinglePalmBuilderCfg(**payload)


def get_com_palm_preset(name: str) -> "ComPalmBuilderCfg":
    r"""按名字返回一份复合 palm preset cfg。"""

    from ..builder.palm_builders import ComPalmBuilderCfg

    if name not in COM_PALM_PRESET_DATA:
        raise KeyError(f"Unknown composite palm preset: {name!r}")
    return ComPalmBuilderCfg(preset=name)


def get_com_palm_preset_data(name: str) -> dict[str, Any]:
    r"""按名字返回一份复合 palm 的原始 recipe 数据副本。"""

    try:
        return deepcopy(COM_PALM_PRESET_DATA[name])
    except KeyError as exc:
        raise KeyError(f"Unknown composite palm preset data: {name!r}") from exc


PALM_PRESET_REGISTRY = {
    "single_box_allegro": lambda: get_single_palm_box_preset("allegro"),
    "single_box_leap": lambda: get_single_palm_box_preset("leap"),
    "com_allegro": lambda: get_com_palm_preset("allegro"),
    "com_leap": lambda: get_com_palm_preset("leap"),
}
"""掌部 preset 的轻量注册表。"""


__all__ = [
    "ALLEGRO_SINGLE_PALM_BOX_PRESET",
    "LEAP_SINGLE_PALM_BOX_PRESET",
    "COM_PALM_PRESET_DATA",
    "PALM_PRESET_REGISTRY",
    "get_single_palm_box_preset",
    "get_com_palm_preset",
    "get_com_palm_preset_data",
]
