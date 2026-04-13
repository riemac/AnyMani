"""preset 层统一入口。

这个子包的职责不是“再造一层框架”，而是把原先散落在 builder 里的
preset 常量、注册表和字符串 -> cfg 的自动解析逻辑集中起来。

这样做的核心目的有两个：

1. builder 回到“只处理 canonical cfg / build 算法”的低层职责；
2. preset 作为 pre-made 的主要离散锚点，拥有清晰、独立、可直接维护的家。

也就是说，今后如果你要人工微调：

- finger preset
- palm preset
- mount preset
- human-like hand 的 preset 组装逻辑

优先都应该来这个子包，而不是去碰 builder 文件本体。
"""

from __future__ import annotations

from typing import Any

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
from .units import cm, deg, m, mm, rad


def resolve_palm_builder_cfg(raw: Any) -> Any:
    """延迟导入 resolver，避免 builder -> presets 子模块时触发循环导入。"""

    from .resolver import resolve_palm_builder_cfg as _impl

    return _impl(raw)


def resolve_finger_builder_cfg(raw: Any) -> Any:
    """延迟导入 resolver，避免 package 聚合入口抢先拉起 hand builder。"""

    from .resolver import resolve_finger_builder_cfg as _impl

    return _impl(raw)


def resolve_finger_slot_builder_cfg(raw: Any) -> Any:
    """延迟导入 resolver。"""

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
    """延迟导入 resolver。"""

    from .resolver import resolve_human_like_mounts as _impl

    return _impl(
        family=family,
        handedness=handedness,
        palm_cfg=palm_cfg,
        mount_preset=mount_preset,
        mounts=mounts,
    )


def resolve_human_like_builder_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    """延迟导入 resolver。"""

    from .resolver import resolve_human_like_builder_kwargs as _impl

    return _impl(raw)


def make_human_like_builder_cfg(**kwargs: Any):
    """延迟导入 resolver。"""

    from .resolver import make_human_like_builder_cfg as _impl

    return _impl(**kwargs)

__all__ = [
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
    "m",
    "cm",
    "mm",
    "deg",
    "rad",
    "resolve_palm_builder_cfg",
    "resolve_finger_builder_cfg",
    "resolve_finger_slot_builder_cfg",
    "resolve_human_like_mounts",
    "resolve_human_like_builder_kwargs",
    "make_human_like_builder_cfg",
]
