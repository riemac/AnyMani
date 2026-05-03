"""finger preset 常量与注册表。

这里保存的是 pre-made 阶段最核心的一类离散锚点：手指/拇指的几何与关节链参数。

把它们从 `builder/finger_buiders.py` 中搬出来，不是为了“抽象更优雅”，而是为了
把两层职责真正分开：

1. builder 文件只负责解释 canonical cfg 并构建 `FingerCfg`
2. preset 文件只负责保存研究者手工测量/整理出来的离散锚点

这样你以后改 preset 时，不会再被 finger builder 的串联算法细节干扰。
"""

from __future__ import annotations

from .units import cm
from ..builder.finger_buiders import (
    AllegroFingerBuilderCfg,
    LeapFingerBuilderCfg,
    RegularFingerBuilderCfg,
    RegularThumbBuilderCfg,
)
from .physical_presets import apply_physical_profile_to_finger_cfg

"""Allegro 非拇指执行型 preset。

这里刻意把几何参数完整展开写死，而不是再套一个更隐式的中间层，
因为科研调 preset 时，最重要的是能直接看到：

- 长度
- 截面尺寸
- mesh 偏移
- 旋转轴
- 指尖类型
"""
ALLEGRO_FINGER_PRESET = AllegroFingerBuilderCfg(
    name="index",
    num_joints=4,
    width=cm(2.7),  # 默认人工测量锚点仍按 cm 录入，更符合手部资产调参直觉
    height=cm(2.0),  # builder 侧最终只看到 SI(m)，这里的 `cm(...)` 只承担显式换算
    length=[cm(1.8), cm(5.4), cm(3.8), cm(2.2)],
    mesh_offsets=[0.0, 0.0, cm(-0.6), 0.0],
    axes=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    tip={"type": "cs", "radius": cm(1.2), "height": cm(1.0)},
)


"""Allegro 拇指执行型 preset。

轴语义严格对齐 `Thumb.png` 中的科研约定：

- CMC1: 绕 $x$
- CMC2: 绕 $y$
- MCP / IP: 绕 $z$
"""
ALLEGRO_THUMB_PRESET = RegularThumbBuilderCfg(
    name="thumb",
    lengths=[cm(4.5), cm(1.7), cm(4.3), cm(4.0)],
    cmc1_width=cm(3.5),
    cmc1_height=cm(3.4),
    width=cm(1.9),
    height=cm(2.7),
    cmc1_offset=(cm(0.9), cm(1.45)),  # (y, z) 偏移
    non_cmc1_offset=[cm(-0.2), 0.0, cm(-0.9)],
    axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
    tip={"type": "cs", "radius": cm(1.2), "height": cm(1.0)},
)


"""LEAP 非拇指执行型 preset。

# Question:
原始设计里，LEAP 的 tip 更贴近 custom `white_tip.obj` 语义；当前 v1 为了
先打通 pre-made 闭环，执行路径仍采用 `cylinder + sphere` primitive tip。
"""
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

"""LEAP 拇指执行型 preset。

本轮虽然不恢复 LEAP 的 custom `leap_cube` tip，但共享的 thumb 链路轴语义仍与
`Thumb.png` 保持一致：CMC1 为 $x$，CMC2 为 $y$，后续为 $z$。
"""
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


"""finger preset 的轻量注册表。

这里保留显式字典而不是更重的动态注册机制，原因是科研期的需求更偏向：

- 一眼能读到当前有哪些离散锚点
- 名称稳定，方便 sidecar/provenance 回溯
- 修改时不必追框架魔法
"""
FINGER_PRESET_REGISTRY: dict[str, RegularFingerBuilderCfg] = {
    "allegro_non_thumb_v1": ALLEGRO_FINGER_PRESET,
    "leap_non_thumb_v1": LEAP_FINGER_PRESET,
    "allegro_thumb_v1": ALLEGRO_THUMB_PRESET,
    "leap_thumb_v1": LEAP_THUMB_PRESET,
}


def get_finger_builder_preset(name: str) -> RegularFingerBuilderCfg:
    r"""按名字返回一份 finger builder preset 副本。

    返回前会注入 official joint physical profile。这样 pre-made 运行时只消费
    已审查的 Python preset，不再读取官方 URDF；同时用户仍然能在
    `finger_presets.py` 里直接看到几何锚点，在 `physical_presets.py` 里直接看到
    关节限位 / effort / velocity / friction 的数值来源。
    """

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
