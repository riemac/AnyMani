"""hand-level mount preset。

这一层表达的是“finger 根部相对于 palm frame 的挂载位姿”。
它与 finger/palm 自身几何是不同层级的信息，因此应独立存放。

把 mount preset 从 builder 子目录挪出来之后，hand 装配层和 palm preset 层
都可以引用同一份 mount 语义，而不会再制造“某个 builder 私藏一份挂载字典”的耦合。

# NOTE:
当前这里统一保存 canonical right-hand anchors。hand builder 先用这份数据装配
完整右手，再对目标 left 执行整手严格反射；全部 finger mounts 使用同一数学合同，
mount preset 本体继续保持纯数据表角色。
"""

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
"""Allegro 挂载点 preset。"""


LEAP_MOUNT_PRESET: dict[str, PoseCfg] = {
    "index": PoseCfg(pos=(m(-0.0070), m(0.0230), m(-0.0187)), rpy=(rad(1.5708), rad(1.5708), 0.0)),
    "middle": PoseCfg(pos=(m(-0.0071), m(-0.0224), m(-0.0187)), rpy=(rad(1.5708), rad(1.5708), 0.0)),
    "ring": PoseCfg(pos=(m(-0.00709), m(-0.0678), m(-0.0187)), rpy=(rad(1.5708), rad(1.5708), 0.0)),
    "thumb": PoseCfg(pos=(m(-0.0693), m(-0.0012), m(-0.0216)), rpy=(0.0, rad(1.5708), 0.0)),
}
"""LEAP 挂载点 preset。

# NOTE:
这里保留“真实 family 锚点默认用 m 直写”的风格；
若后续某组 mount 仍来自手工 cm 记录，则允许像 single-box 系列那样显式写 `cm(...)`。
"""


LEAP_SINGLE_BOX_MOUNT_PRESET: dict[str, PoseCfg] = {
    "thumb": PoseCfg(pos=(cm(3.7), cm(3.1), cm(1.0)), rpy=(0.0, 0.0, rad(-math.pi / 2.0))), # 直接从图示理解中得到
    "index": PoseCfg(pos=(cm(4.6), cm(8.0), cm(0.8)), rpy=(0.0, 0.0, 0.0)),
    "middle": PoseCfg(pos=(0.0, cm(8.0), cm(0.8)), rpy=(0.0, 0.0, 0.0)),
    "ring": PoseCfg(pos=(cm(-4.6), cm(8.0), cm(0.8)), rpy=(0.0, 0.0, 0.0)),
}
"""Single-box LEAP palm 的显式挂载点 preset（右手基准）。

这组数值不是“从 family 猜”的，而是直接来自用户早期写在
`HumanLikeHandBuilder.build()` 草案里的 single-palm 记录。
"""


ALLEGRO_SINGLE_BOX_MOUNT_PRESET: dict[str, PoseCfg] = {
    # "thumb": PoseCfg(pos=(cm(2.45), cm(3.05), cm(-1.45)), rpy=(0.0, 0.0, rad(-math.pi / 2.0))), # 原数据
    "thumb": PoseCfg(pos=(cm(2.45), cm(3.05), cm(-1.45)), rpy=(0.0, 0.0, rad(-1.65806278845))), # 语义对齐后的挂载点，计算逻辑见 `AnyMani/source/anymani/anymani/assets/doc/draft/mounts.md`
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


def get_mount_preset(name: str, *, handedness: str | None = None) -> dict[str, PoseCfg]:
    r"""按名字返回一份 canonical mount preset 副本。

    Args:
        name (str): 已注册的 mount preset 名。
        handedness (str | None): 兼容 palm preview 调用签名；preset 始终返回
            canonical right-hand anchors，目标 handedness 由完整 hand lowering 解释。

    Returns:
        dict[str, PoseCfg]: canonical right-hand 语义下的 mount 字典副本。

    Raises:
        KeyError: 当名字未注册时抛出。
    """

    try:
        preset = MOUNT_PRESET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown mount preset: {name!r}") from exc

    _ = handedness  # mount 数据表不单独执行 handedness 变换
    return {finger: pose.copy() for finger, pose in preset.items()}  # 返回副本，避免调用方污染注册表


__all__ = [
    "ALLEGRO_MOUNT_PRESET",
    "LEAP_MOUNT_PRESET",
    "ALLEGRO_SINGLE_BOX_MOUNT_PRESET",
    "LEAP_SINGLE_BOX_MOUNT_PRESET",
    "MOUNT_PRESET_REGISTRY",
    "get_mount_preset",
]
