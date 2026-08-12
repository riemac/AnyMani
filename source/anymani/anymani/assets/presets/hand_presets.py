r"""hand preset 常量、注册表与整手组合辅助。

这个文件保存的不是 palm/finger 的底层几何参数，而是更高一层的
“整手组合语义”：

1. 选哪个 family；
2. 选哪个 palm preset；
3. 非拇指/拇指分别选哪个 finger preset；
4. 默认 handedness 是什么。

# NOTE:
这里刻意采用“显式字典注册表”而不是更复杂的动态注册机制，因为 hand preset
在科研工作流里本质上是 pre-made 的离散锚点：

- 名字要稳定，方便 sidecar / provenance 回溯；
- 组合要一眼能读懂，方便你直接肉眼核对；
- 修改时要像改实验表一样直接，不要追框架魔法。

# NOTE:
当前 hand preset 的 palm、全部 finger mounts、joint chains 与 mesh recipes 都采用
canonical right-hand 语义，只保存一份离散真源。`HumanLikeHandBuilder` 完成整手
装配后，再对完整 `HandCfg` 执行严格 $y$-$z$ 平面反射；因此不需要维护 left preset，
也不会把 handedness 缩减为某一根 thumb mount 的局部特例。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# --- 预设1：single palm allegro hand ---
# 主要由 Allegro 的 single-box palm、对应挂载点，以及 Allegro fingers 构成。
SINGLE_PALM_ALLEGRO_HAND_PRESET: dict[str, Any] = {
    "name": "single_palm_allegro",  # 默认 hand 名，便于 preview / sidecar 直接回溯来源
    "family": "allegro",  # hand family 标签
    "handedness": "right",  # hand preset 当前默认给右手；需要左手时在上层覆盖即可
    "palm_cfg": "single_box_allegro",  # palm 选择 single-box Allegro 几何锚点
    "finger_cfg": "allegro_non_thumb_v1",  # 非拇指统一采用 Allegro v1 preset
    "thumb_cfg": "allegro_thumb_v1",  # 拇指采用 Allegro thumb v1 preset
}


# --- 预设2：single palm leap hand ---
# 主要由 LEAP 的 single-box palm、对应挂载点，以及 LEAP fingers 构成。
SINGLE_PALM_LEAP_HAND_PRESET: dict[str, Any] = {
    "name": "single_palm_leap",  # 默认 hand 名
    "family": "leap",  # hand family 标签
    "handedness": "right",  # 默认右手
    "palm_cfg": "single_box_leap",  # single-box LEAP palm
    "finger_cfg": "leap_non_thumb_v1",  # LEAP 非拇指 preset
    "thumb_cfg": "leap_thumb_v1",  # LEAP 拇指 preset
}


# --- 预设3：com palm allegro hand ---
# 主要由 Allegro 的 composite palm recipe 与 Allegro fingers 构成。
COM_PALM_ALLEGRO_HAND_PRESET: dict[str, Any] = {
    "name": "com_palm_allegro",  # 默认 hand 名
    "family": "allegro",  # hand family 标签
    "handedness": "right",  # 默认右手
    "palm_cfg": "com_allegro",  # composite palm 直接走 `com_allegro`
    "finger_cfg": "allegro_non_thumb_v1",  # Allegro 非拇指 preset
    "thumb_cfg": "allegro_thumb_v1",  # Allegro 拇指 preset
}


# --- 预设4：com palm leap hand ---
# 主要由 LEAP 的 composite palm recipe 与 LEAP fingers 构成。
COM_PALM_LEAP_HAND_PRESET: dict[str, Any] = {
    "name": "com_palm_leap",  # 默认 hand 名
    "family": "leap",  # hand family 标签
    "handedness": "right",  # 默认右手
    "palm_cfg": "com_leap",  # composite palm 直接走 `com_leap`
    "finger_cfg": "leap_non_thumb_v1",  # LEAP 非拇指 preset
    "thumb_cfg": "leap_thumb_v1",  # LEAP 拇指 preset
}


HAND_PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    "single_palm_allegro": SINGLE_PALM_ALLEGRO_HAND_PRESET,
    "single_palm_leap": SINGLE_PALM_LEAP_HAND_PRESET,
    "com_palm_allegro": COM_PALM_ALLEGRO_HAND_PRESET,
    "com_palm_leap": COM_PALM_LEAP_HAND_PRESET,
}
"""整手组合 preset 的轻量注册表。"""


def get_hand_builder_preset_data(name: str) -> dict[str, Any]:
    r"""按名字返回一份 hand preset 原始组合字典副本。

    这里返回“原始组合数据”而不是直接返回 typed cfg，原因是 hand preset 的主要用途有两类：

    1. quick-check / CLI：希望先读到组合长什么样，再决定是否覆写 handedness / name；
    2. recipe / 未来 pre-made：希望把这组离散锚点继续喂给 resolver，走统一的
       `str/dict -> typed cfg` 桥接路径。

    Args:
        name (str): 已注册的 hand preset 名。

    Returns:
        dict[str, Any]: hand preset 的原始组合字典副本。

    Raises:
        KeyError: 当名字未注册时抛出。
    """

    try:
        return deepcopy(HAND_PRESET_REGISTRY[name])  # 返回副本，避免上层覆盖时污染注册表本体
    except KeyError as exc:
        raise KeyError(f"Unknown hand builder preset: {name!r}") from exc


def make_human_like_builder_cfg_from_preset(preset_name: str, **overrides: Any):
    r"""从 hand preset 构造一份 `HumanLikeHandBuilderCfg`。

    这个函数的定位，是把 hand preset 变成与 `make_human_like_builder_cfg(...)`
    平级的“整手组合便利入口”：

    - preset 负责提供一组离散锚点；
    - overrides 负责做少量显式覆盖；
    - 最终仍走统一的 preset resolver，把字符串解析成 typed cfg。

    Args:
        preset_name (str): 已注册的 hand preset 名。
        **overrides (Any): 对 preset 字段的显式覆写，例如 `handedness="left"`、
            `name="allegro_preview_left"` 等。值为 `None` 的键会被忽略。

    Returns:
        HumanLikeHandBuilderCfg: 已解析好的 typed hand builder cfg。
    """

    from .resolver import make_human_like_builder_cfg

    payload = get_hand_builder_preset_data(preset_name)  # 先取出组合锚点
    payload.update({key: value for key, value in overrides.items() if value is not None})  # 再应用显式覆盖
    return make_human_like_builder_cfg(**payload)  # 最终仍收口到统一 resolver


__all__ = [
    "SINGLE_PALM_ALLEGRO_HAND_PRESET",
    "SINGLE_PALM_LEAP_HAND_PRESET",
    "COM_PALM_ALLEGRO_HAND_PRESET",
    "COM_PALM_LEAP_HAND_PRESET",
    "HAND_PRESET_REGISTRY",
    "get_hand_builder_preset_data",
    "make_human_like_builder_cfg_from_preset",
]
