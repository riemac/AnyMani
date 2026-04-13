"""preset 解析层：把用户/recipe 侧的 preset 表达收敛为 builder 可直接消费的 cfg。

这个模块存在的原因，是为了把“builder 只吃 resolved cfg”和“用户仍然可以只写
preset 名”这两件事同时成立：

1. builder 层不再直接解析 preset 字符串
2. recipe / quick-check / 上层脚本仍可以用 preset 名驱动 pre-made 工作流

因此，凡是“字符串 preset -> typed cfg / mounts”的自动桥接，都应尽量收口到这里。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..asset_schema_core import PoseCfg
from ..builder.finger_buiders import (
    AllegroFingerBuilderCfg,
    LeapFingerBuilderCfg,
    RegularThumbBuilderCfg,
)
from ..builder.hand_builders import HumanLikeHandBuilderCfg
from ..builder.palm_builders import ComPalmBuilderCfg, SinglePalmBuilderCfg
from .finger_presets import get_finger_builder_preset
from .mount_presets import get_mount_preset
from .palm_presets import get_com_palm_preset, get_single_palm_box_preset


_FINGER_SLOT_NAMES = {"index", "middle", "ring", "little"}


def _to_pose_dict(values: dict[str, Any] | None) -> dict[str, PoseCfg]:
    r"""把宽松 mount 输入统一规范为 `PoseCfg` 字典。"""

    return {name: PoseCfg.from_value(value) for name, value in (values or {}).items()}


def resolve_palm_builder_cfg(raw: Any) -> Any:
    r"""把 palm preset/配置表达解析成 typed palm builder cfg。"""

    if isinstance(raw, (ComPalmBuilderCfg, SinglePalmBuilderCfg)):
        return raw
    if isinstance(raw, str):
        if raw.startswith("com_"):
            return get_com_palm_preset(raw.removeprefix("com_"))
        if raw.startswith("single_box_"):
            return get_single_palm_box_preset(raw.removeprefix("single_box_"))
        raise ValueError(f"Unsupported palm preset string: {raw!r}")
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported palm cfg payload: {raw!r}")

    payload = deepcopy(raw)
    if "preset" in payload:
        return ComPalmBuilderCfg(**payload)
    return SinglePalmBuilderCfg(**payload)


def resolve_finger_builder_cfg(raw: Any) -> Any:
    r"""把 finger preset/配置表达解析成 typed finger builder cfg。"""

    if isinstance(raw, (AllegroFingerBuilderCfg, LeapFingerBuilderCfg, RegularThumbBuilderCfg)):
        return raw
    if isinstance(raw, str):
        return get_finger_builder_preset(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported finger cfg payload: {raw!r}")

    payload = deepcopy(raw)
    preset_name = payload.pop("preset_name", payload.pop("preset", None))
    if isinstance(preset_name, str):
        return get_finger_builder_preset(preset_name).replace(**payload)

    thumb_keys = {"lengths", "cmc1_width", "cmc1_height", "cmc1_offset", "non_cmc1_offset"}
    if thumb_keys & set(payload):
        return RegularThumbBuilderCfg(**payload)
    if "fixed_part" in payload:
        return LeapFingerBuilderCfg(**payload)
    return AllegroFingerBuilderCfg(**payload)


def resolve_finger_slot_builder_cfg(raw: Any) -> Any:
    r"""解析非拇指 finger 槽位配置。

    这里要区分两种 dict 语义：

    1. 单个 finger cfg 的字段字典
    2. 按 `index/middle/ring/little` 分配的多 finger 映射
    """

    if isinstance(raw, dict) and raw and set(raw).issubset(_FINGER_SLOT_NAMES):
        return {name: resolve_finger_builder_cfg(cfg) for name, cfg in raw.items()}
    return resolve_finger_builder_cfg(raw)


def resolve_human_like_mounts(
    *,
    family: str | None,
    palm_cfg: Any,
    mount_preset: str | None = None,
    mounts: dict[str, Any] | None = None,
) -> dict[str, PoseCfg]:
    r"""为 human-like hand 解析最终挂载点字典。

    这个函数承担的，正是原先 hand builder 内部那段 preset 推断逻辑；
    现在把它挪到 preset 层，是为了让 builder 只看到最终的显式 mount 结果。
    """

    candidate_names: list[str] = []
    if mount_preset is not None:
        candidate_names.append(mount_preset)
    if isinstance(palm_cfg, ComPalmBuilderCfg):
        candidate_names.append(f"com_{palm_cfg.preset}")
    if isinstance(palm_cfg, SinglePalmBuilderCfg) and palm_cfg.shape == "box" and family:
        candidate_names.append(f"single_box_{family}")
    if family:
        candidate_names.append(family)

    resolved_from_preset: dict[str, PoseCfg] = {}
    for preset_name in candidate_names:
        try:
            resolved_from_preset = get_mount_preset(preset_name)
            break
        except KeyError:
            continue

    return {**resolved_from_preset, **_to_pose_dict(mounts)}


def resolve_human_like_builder_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    r"""把 human-like hand 的高层 preset 表达规约成 builder 可直接消费的 kwargs。"""

    data = deepcopy(raw)
    if "palm_cfg" in data:
        data["palm_cfg"] = resolve_palm_builder_cfg(data["palm_cfg"])
    if "finger_cfg" in data:
        data["finger_cfg"] = resolve_finger_slot_builder_cfg(data["finger_cfg"])
    if "thumb_cfg" in data:
        data["thumb_cfg"] = resolve_finger_builder_cfg(data["thumb_cfg"])

    mount_preset = data.pop("mount_preset", None)
    data["mounts"] = resolve_human_like_mounts(
        family=data.get("family"),
        palm_cfg=data.get("palm_cfg"),
        mount_preset=mount_preset,
        mounts=data.get("mounts"),
    )
    return data


def make_human_like_builder_cfg(**kwargs: Any) -> HumanLikeHandBuilderCfg:
    r"""用 preset 层解析规则构造 `HumanLikeHandBuilderCfg`。

    这是给 recipe 层、测试层和后续 quick-check 脚本用的便利入口。
    builder 本体不再吃 preset 字符串；若调用方手里拿的是 preset 名，
    就应该先经过这个函数或同级 resolver。
    """

    return HumanLikeHandBuilderCfg(**resolve_human_like_builder_kwargs(kwargs))


__all__ = [
    "resolve_palm_builder_cfg",
    "resolve_finger_builder_cfg",
    "resolve_finger_slot_builder_cfg",
    "resolve_human_like_mounts",
    "resolve_human_like_builder_kwargs",
    "make_human_like_builder_cfg",
]
