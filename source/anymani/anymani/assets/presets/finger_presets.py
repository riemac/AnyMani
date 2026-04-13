"""finger preset 常量与注册表。

这里保存的是 pre-made 阶段最核心的一类离散锚点：手指/拇指的几何与关节链参数。

把它们从 `builder/finger_buiders.py` 中搬出来，不是为了“抽象更优雅”，而是为了
把两层职责真正分开：

1. builder 文件只负责解释 canonical cfg 并构建 `FingerCfg`
2. preset 文件只负责保存研究者手工测量/整理出来的离散锚点

这样你以后改 preset 时，不会再被 finger builder 的串联算法细节干扰。
"""

from __future__ import annotations

from ..builder.finger_buiders import (
    AllegroFingerBuilderCfg,
    LeapFingerBuilderCfg,
    RegularFingerBuilderCfg,
    RegularThumbBuilderCfg,
)


ALLEGRO_FINGER_PRESET = AllegroFingerBuilderCfg(
    name="index",
    num_joints=4,
    width=2.7,
    height=2.0,
    length=[1.8, 5.4, 3.8, 2.2],
    mesh_offsets=[0.0, 0.0, -0.6, 0.0],
    axes=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    tip={"type": "cs", "radius": 1.2, "height": 1.0},
)
"""Allegro 非拇指执行型 preset。

这里刻意把几何参数完整展开写死，而不是再套一个更隐式的中间层，
因为科研调 preset 时，最重要的是能直接看到：

- 长度
- 截面尺寸
- mesh 偏移
- 旋转轴
- 指尖类型
"""


LEAP_FINGER_PRESET = LeapFingerBuilderCfg(
    name="index",
    num_joints=4,
    width=3.4,
    height=2.05,
    length=[3.9, 1.5, 3.6, 2.0],
    mesh_offsets=[0.0, 0.0, 0.0, 0.0],
    fixed_part=1.3,
    axes=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    tip={"type": "cs", "radius": 1.2, "height": 1.0},
)
"""LEAP 非拇指执行型 preset。

# Question:
原始设计里，LEAP 的 tip 更贴近 custom `white_tip.obj` 语义；当前 v1 为了
先打通 pre-made 闭环，执行路径仍采用 `cylinder + sphere` primitive tip。
"""


ALLEGRO_THUMB_PRESET = RegularThumbBuilderCfg(
    name="thumb",
    lengths=[4.5, 1.7, 4.3, 4.0],
    cmc1_width=3.5,
    cmc1_height=3.4,
    width=1.9,
    height=2.7,
    cmc1_offset=(0.9, 1.45),
    non_cmc1_offset=[-0.2, 0.0, -0.9],
    axes=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
    tip={"type": "cs", "radius": 1.2, "height": 1.0},
)
"""Allegro 拇指执行型 preset。"""


LEAP_THUMB_PRESET = RegularThumbBuilderCfg(
    name="thumb",
    lengths=[2.8, 1.7, 4.7, 2.3],
    cmc1_width=2.30,
    cmc1_height=2.67,
    width=2.3,
    height=3.47,
    cmc1_offset=(0.0, -0.33),
    non_cmc1_offset=[0.0, 0.0, 0.0],
    axes=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
    tip={"type": "cs", "radius": 1.2, "height": 1.0},
)
"""LEAP 拇指执行型 preset。"""


FINGER_PRESET_REGISTRY: dict[str, RegularFingerBuilderCfg] = {
    "allegro_non_thumb_v1": ALLEGRO_FINGER_PRESET,
    "leap_non_thumb_v1": LEAP_FINGER_PRESET,
    "allegro_thumb_v1": ALLEGRO_THUMB_PRESET,
    "leap_thumb_v1": LEAP_THUMB_PRESET,
}
"""finger preset 的轻量注册表。

这里保留显式字典而不是更重的动态注册机制，原因是科研期的需求更偏向：

- 一眼能读到当前有哪些离散锚点
- 名称稳定，方便 sidecar/provenance 回溯
- 修改时不必追框架魔法
"""


def get_finger_builder_preset(name: str) -> RegularFingerBuilderCfg:
    r"""按名字返回一份 finger builder preset 副本。"""

    try:
        return FINGER_PRESET_REGISTRY[name].copy()
    except KeyError as exc:
        raise KeyError(f"Unknown finger builder preset: {name!r}") from exc


__all__ = [
    "ALLEGRO_FINGER_PRESET",
    "LEAP_FINGER_PRESET",
    "ALLEGRO_THUMB_PRESET",
    "LEAP_THUMB_PRESET",
    "FINGER_PRESET_REGISTRY",
    "get_finger_builder_preset",
]
