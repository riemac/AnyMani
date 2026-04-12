r"""hand-level 挂载点 preset 的共享定义。

本模块把原先散落在 `hand_builders.py` 里的 mount preset 提取出来，变成
builder 子目录内可复用的共享常量与查询函数。这样做的直接原因不是“为了抽象”，
而是当前已经有两处 builder 需要消费同一份 hand-level mount 语义：

1. `hand_builders.py`
   - 在装配阶段决定每根 finger 的最终 `mount`
2. `palm_builders.py`
   - 在复合 palm preset 的 metadata 中记录 `finger_mounts`

若继续把 preset 只写在 `hand_builders.py`，`palm_builders.py` 就只能通过
不存在的私有模块导入，最终在测试收集阶段直接炸掉。把它提出来之后，
pre-made 主链的 mount 语义就重新收敛到了单一来源。
"""

from __future__ import annotations

from ..asset_schema_core import PoseCfg


ALLEGRO_MOUNT_PRESET: dict[str, PoseCfg] = {
    "index": PoseCfg(pos=(0.0, 0.0435, -0.001542), rpy=(-0.0873, 0.0, 0.0)),
    "middle": PoseCfg(pos=(0.0, 0.0, 0.0007), rpy=(0.0, 0.0, 0.0)),
    "ring": PoseCfg(pos=(0.0, -0.0435, -0.001542), rpy=(0.0873, 0.0, 0.0)),
    "thumb": PoseCfg(pos=(-0.0182, 0.019333, -0.045987), rpy=(0.0, -1.6581, -1.5708)),
}
"""Allegro 挂载点 preset。

这些值表达的是 Allegro 风格 hand 中：
- index / middle / ring 在 palm 顶缘的挂载位姿
- thumb 在 palm 侧前方的挂载位姿
"""


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

这里显式同时支持：
- family 名
- palm recipe 名

这样 single palm 和 com palm 都可以走同一个 hand-level mount preset 入口。
"""


def get_mount_preset(name: str) -> dict[str, PoseCfg]:
    r"""按名字返回一份挂载点 preset 副本。"""

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
