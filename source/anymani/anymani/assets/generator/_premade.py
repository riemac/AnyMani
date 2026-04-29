"""pre-made 相关内部辅助。

这个文件承载的是 `HandGenerator` façade 背后那些**不必继续塞在类体里**的纯 helper：

- pre-made façade 输入规约
- connectivity preset 选择
- hand preset / connectivity preset 的 sample / enumerate 选择
- canonical hand 构建
- connectivity lower
- 递归/平铺输出目录解析

把它们从 `hand_generator.py` 中拆出来的动机非常明确：

1. `HandGenerator / HandGeneratorCfg` 仍然是唯一 façade，不动；
2. 但 façade 文件里不应该再同时堆满大量内部 helper；
3. pre-made 逻辑本身已经够复杂，需要独立成一个可单独阅读的小模块。

# NOTE:
这里的函数全部是“内部 helper”，不是新的用户入口。
真正的用户仍然只应直接面对：

- `HandGeneratorCfg`
- `HandGenerator`
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import random
from pathlib import Path
from typing import Any, Literal

from ..asset_base import HandCfg
from ..asset_builders import HandBuilder
from ..presets.connectivity_presets import (
    _remaining_revolute_count as _connectivity_remaining_revolute_count,
    get_finger_connectivity_preset_data,
    get_hand_connectivity_preset_data,
    list_finger_connectivity_preset_names,
)
from ..presets.hand_presets import get_hand_builder_preset_data, make_human_like_builder_cfg_from_preset
from ..presets.resolver import resolve_finger_builder_cfg
from ._generation_result import HandGenerationResult


_PREMADE_SLOT_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")
_PREMADE_NON_THUMB_SLOT_ORDER: tuple[str, ...] = tuple(slot for slot in _PREMADE_SLOT_ORDER if slot != "thumb")
_PREMADE_SLOT_VARIANT_TOKEN: dict[str, str] = {
    "thumb": "t",
    "index": "i",
    "middle": "m",
    "ring": "r",
    "little": "l",
}
_PREMADE_FINGER_PRESET_BY_FAMILY_AND_KIND: dict[tuple[str, Literal["thumb", "non_thumb"]], str] = {
    ("allegro", "thumb"): "allegro_thumb_v1",
    ("allegro", "non_thumb"): "allegro_non_thumb_v1",
    ("leap", "thumb"): "leap_thumb_v1",
    ("leap", "non_thumb"): "leap_non_thumb_v1",
}


@dataclass(frozen=True)
class PremadeTopologySpec:
    r"""pre-made 阶段的一份显式 topology 规格。

    这里刻意把过去隐含在：

    - `hand preset`
    - `mixed`
    - `missing`

    里的结构信息，收敛成一个内部显式对象。原因是当 mixed / missing 真正进入
    pre-made 主线后，后续的这些动作都需要读同一份 topology 真源：

    - builder 直接造骨架
    - connectivity 只对 surviving slots lower
    - validator / sidecar / 输出目录回溯 topology provenance
    """

    name: str
    anchor_root: str
    topology_kind: Literal["single_family", "missing", "mixed"]
    base_hand_preset: str
    handedness: Literal["left", "right"]
    family: str
    finger_preset_names: dict[str, str]
    surviving_slots: tuple[str, ...]

    def slot_family_map(self) -> dict[str, str]:
        r"""返回每个 surviving slot 当前来自哪个 finger family。"""

        return {
            slot: _finger_family_from_preset_name(preset_name)
            for slot, preset_name in self.finger_preset_names.items()
        }

    def to_metadata(self) -> dict[str, Any]:
        r"""把 topology 规格转成可挂到 HandCfg / sidecar 的稳定 provenance。"""

        return {
            "topology_registry_key": self.name,
            "base_hand_preset": self.base_hand_preset,
            "handedness": self.handedness,
            "topology_kind": self.topology_kind,
            "topology_anchor": self.anchor_root,
            "topology_name": self.name,
            "surviving_slots": list(self.surviving_slots),
            "slot_finger_presets": dict(self.finger_preset_names),
            "slot_family_map": self.slot_family_map(),
        }


def normalize_name_list(values: list[str] | tuple[str, ...] | None, *, field_name: str) -> list[str]:
    r"""把 recipe / YAML 侧的名称列表统一规约为 `list[str]`。

    这次 pre-made façade 的显式设计就是：

    - `hand_presets: list[str]`
    - `connectivity_presets: dict[str, list[str]] | None`

    因而这里不再像上一版那样往 tuple 收，而是显式回到 list。
    这样科研侧在读配置时，看到的形状会和既定契约完全一致。
    """

    if values is None:
        return []
    if isinstance(values, str):
        return [str(values)]
    if isinstance(values, tuple):
        return [str(item) for item in values]
    if isinstance(values, list):
        return [str(item) for item in values]
    raise TypeError(f"{field_name} must be a list/tuple of str or None, got {values!r}")


def normalize_connectivity_mapping(
    values: dict[str, Any] | None,
) -> dict[str, dict[str, list[str]]] | None:
    r"""把 `connectivity_presets` 统一规约为唯一合法的 slot-level 形状。

    当前 `HandGeneratorCfg.connectivity_presets` 的 contract 已明确收敛为：

    `hand_preset -> {slot -> [finger_connectivity_preset_name, ...]}`

    也就是说，研究者直接声明：

    - 哪个 base hand 参与 pre-made；
    - 这个 hand 的每个 slot 允许枚举哪些**已注册 finger connectivity 资产**。

    不再接受旧的 hand-level alias 列表形状，避免配置层再次出现
    “看起来像整手 preset，实际上 lower 成 slot recipe” 的语义歧义。
    """

    if values is None:
        return None
    if not isinstance(values, dict):
        raise TypeError(f"connectivity_presets must be a mapping or None, got {values!r}")

    normalized: dict[str, dict[str, list[str]]] = {}
    for hand_preset_name, preset_names in values.items():
        normalized_key = str(hand_preset_name)
        if not isinstance(preset_names, dict):
            raise TypeError(
                f"connectivity_presets[{hand_preset_name!r}] must be a slot-level mapping "
                f"{{slot -> [finger_connectivity_preset_name, ...]}}, got {preset_names!r}"
            )
        invalid_slots = set(preset_names) - set(_PREMADE_SLOT_ORDER)
        if invalid_slots:
            raise ValueError(
                f"connectivity_presets[{hand_preset_name!r}] has invalid slot keys {sorted(invalid_slots)!r}; "
                f"allowed slots are {_PREMADE_SLOT_ORDER!r}"
            )
        normalized[normalized_key] = {
            str(slot_name): normalize_name_list(
                slot_values,
                field_name=f"connectivity_presets[{hand_preset_name!r}][{slot_name!r}]",
            )
            for slot_name, slot_values in preset_names.items()
        }
    return normalized


def _finger_family_from_preset_name(preset_name: str) -> str:
    r"""从 finger preset 名里读取 family 前缀。"""

    for family in ("allegro", "leap"):
        if preset_name.startswith(f"{family}_"):
            return family
    raise ValueError(f"Cannot infer finger family from preset name {preset_name!r}")


def _slot_finger_kind(slot_name: str) -> Literal["thumb", "non_thumb"]:
    return "thumb" if slot_name == "thumb" else "non_thumb"


def _requested_handednesses(cfg: Any) -> tuple[Literal["left", "right"], ...]:
    r"""把 `HandGeneratorCfg.handedness` lower 成当前 pre-made 要展开的 handedness 集合。"""

    requested = str(getattr(cfg, "handedness", "all"))
    if requested == "all":
        return ("left", "right")
    if requested in {"left", "right"}:
        return (requested,)
    raise ValueError(f"Unsupported handedness request {requested!r}; expected 'left' / 'right' / 'all'.")


def _build_topology_registry_key(
    *,
    base_hand_preset: str,
    handedness: Literal["left", "right"],
    suffix_tokens: tuple[str, ...] = (),
) -> str:
    r"""为内部 topology registry 生成稳定 key。

    # NOTE:
    这里的 key 只服务运行时枚举唯一性与 stable provenance，
    **不是** 最终导出目录名。最终目录名必须等 connectivity lower 完成、每个 slot
    的 surviving revolute DOF 真正确定后，才能按用户约定写成：

    - single / missing: `right_t3_i2_m2_r4`
    - mixed: `right_allegro_t4_leap_i3_m4_r4`
    """

    parts = [base_hand_preset, handedness]
    parts.extend(str(token) for token in suffix_tokens)
    return "__".join(parts)


def _supports_topology_expansion(cfg: Any, *, base_hand_preset_name: str) -> bool:
    r"""判断当前 base hand 是否应展开 missing / mixed topology。

    现在的 pre-made façade 已经只保留 slot-level candidate pool。
    因而 mixed / missing 的展开边界也跟着变得直接：

    - 只要 `Made` 没有被 concrete builder cfg 局部覆写；
    - 那么 base hand 就按照 slot-level candidate pool 正常展开 mixed / missing。
    """

    _ = base_hand_preset_name  # 当前函数保留 hand 参数，是为了调用点语义仍然清楚：判断的是“这个 base hand 能否展开”
    return cfg.Made.class_type is HandBuilder


def _extract_base_topology_spec(
    hand_preset_name: str,
    *,
    handedness: Literal["left", "right"],
) -> PremadeTopologySpec:
    r"""从 hand preset 读出 canonical single-family topology 规格。"""

    hand_preset_data = get_hand_builder_preset_data(hand_preset_name)
    non_thumb_slots = _extract_non_thumb_slots_from_hand_preset(hand_preset_data)
    finger_preset_names: dict[str, str] = {}

    raw_finger_cfg = hand_preset_data.get("finger_cfg")
    if isinstance(raw_finger_cfg, dict):
        for slot_name in non_thumb_slots:
            raw_value = raw_finger_cfg.get(slot_name)
            if not isinstance(raw_value, str):
                raise TypeError(
                    f"Hand preset {hand_preset_name!r} must keep finger_cfg[{slot_name!r}] as a preset string "
                    f"for premade topology enumeration, got {raw_value!r}"
                )
            finger_preset_names[slot_name] = raw_value
    else:
        if not isinstance(raw_finger_cfg, str):
            raise TypeError(
                f"Hand preset {hand_preset_name!r} must keep finger_cfg as a preset string for premade topology "
                f"enumeration, got {raw_finger_cfg!r}"
            )
        for slot_name in non_thumb_slots:
            finger_preset_names[slot_name] = raw_finger_cfg

    raw_thumb_cfg = hand_preset_data.get("thumb_cfg")
    if raw_thumb_cfg is not None:
        if not isinstance(raw_thumb_cfg, str):
            raise TypeError(
                f"Hand preset {hand_preset_name!r} must keep thumb_cfg as a preset string for premade topology "
                f"enumeration, got {raw_thumb_cfg!r}"
            )
        finger_preset_names["thumb"] = raw_thumb_cfg

    surviving_slots = tuple(slot_name for slot_name in _PREMADE_SLOT_ORDER if slot_name in finger_preset_names)
    return PremadeTopologySpec(
        name=_build_topology_registry_key(base_hand_preset=hand_preset_name, handedness=handedness),
        anchor_root=hand_preset_name,
        topology_kind="single_family",
        base_hand_preset=hand_preset_name,
        handedness=handedness,
        family=str(hand_preset_data["family"]),
        finger_preset_names=finger_preset_names,
        surviving_slots=surviving_slots,
    )


def _extract_non_thumb_slots_from_hand_preset(hand_preset_data: dict[str, Any]) -> tuple[str, ...]:
    r"""从 hand preset 原始字典里恢复当前 canonical non-thumb slot 集合。"""

    finger_cfg = hand_preset_data.get("finger_cfg")
    if isinstance(finger_cfg, dict):
        return tuple(slot_name for slot_name in _PREMADE_NON_THUMB_SLOT_ORDER if slot_name in finger_cfg)

    num_non_thumb = int(hand_preset_data.get("num_non_thumb", 3))
    return _PREMADE_NON_THUMB_SLOT_ORDER[:num_non_thumb]


def _build_missing_topology_specs(base_topology: PremadeTopologySpec) -> tuple[PremadeTopologySpec, ...]:
    r"""从 canonical single-family topology 派生“缺失一根 non-thumb”的 pre-made 规格。"""

    if "thumb" not in base_topology.surviving_slots:
        return ()

    non_thumb_slots = [slot_name for slot_name in base_topology.surviving_slots if slot_name != "thumb"]
    if len(non_thumb_slots) < 3:
        return ()  # 当前 missing 首版只处理 4 指 canonical hand -> 缺 1 根 non-thumb 的情形

    specs: list[PremadeTopologySpec] = []
    for missing_slot in non_thumb_slots:
        remaining = {
            slot_name: preset_name
            for slot_name, preset_name in base_topology.finger_preset_names.items()
            if slot_name != missing_slot
        }
        surviving_slots = tuple(slot_name for slot_name in _PREMADE_SLOT_ORDER if slot_name in remaining)
        specs.append(
            PremadeTopologySpec(
                name=_build_topology_registry_key(
                    base_hand_preset=base_topology.base_hand_preset,
                    handedness=base_topology.handedness,
                    suffix_tokens=(f"missing_{missing_slot}",),
                ),
                anchor_root=base_topology.base_hand_preset,
                topology_kind="missing",
                base_hand_preset=base_topology.base_hand_preset,
                handedness=base_topology.handedness,
                family=base_topology.family,
                finger_preset_names=remaining,
                surviving_slots=surviving_slots,
            )
        )
    return tuple(specs)


def _build_mixed_topology_specs(base_topology: PremadeTopologySpec) -> tuple[PremadeTopologySpec, ...]:
    r"""从 canonical topology 派生 mixed-family finger 组合。"""

    slot_order = base_topology.surviving_slots
    if not slot_order:
        return ()

    specs: list[PremadeTopologySpec] = []
    for family_assignment in product(("allegro", "leap"), repeat=len(slot_order)):
        slot_family_map = dict(zip(slot_order, family_assignment))
        if all(current_family == base_topology.family for current_family in slot_family_map.values()):
            continue  # 全部与 palm family 一致时，不再重复 single-family canonical topology

        finger_preset_names = {
            slot_name: _PREMADE_FINGER_PRESET_BY_FAMILY_AND_KIND[(slot_family_map[slot_name], _slot_finger_kind(slot_name))]
            for slot_name in slot_order
        }
        specs.append(
            PremadeTopologySpec(
                name=_format_mixed_topology_name(
                    base_hand_preset=base_topology.base_hand_preset,
                    handedness=base_topology.handedness,
                    slot_family_map=slot_family_map,
                ),
                anchor_root="mixed",
                topology_kind="mixed",
                base_hand_preset=base_topology.base_hand_preset,
                handedness=base_topology.handedness,
                family=base_topology.family,
                finger_preset_names=finger_preset_names,
                surviving_slots=slot_order,
            )
        )
    return tuple(specs)


def _format_mixed_topology_name(
    *,
    base_hand_preset: str,
    handedness: Literal["left", "right"],
    slot_family_map: dict[str, str],
) -> str:
    r"""把 mixed topology 的结构 provenance 写成内部 registry key。"""

    parts = [base_hand_preset, handedness, "mixed"]
    for slot_name in _PREMADE_SLOT_ORDER:
        if slot_name in slot_family_map:
            parts.append(f"{slot_name}_{slot_family_map[slot_name]}")
    return "__".join(parts)


def _build_premade_topology_registry(cfg: Any) -> dict[str, PremadeTopologySpec]:
    r"""构建当前 generator cfg 可见的所有 pre-made topology 规格。"""

    registry: dict[str, PremadeTopologySpec] = {}
    for hand_preset_name in cfg.hand_presets:
        for handedness in _requested_handednesses(cfg):
            base_topology = _extract_base_topology_spec(hand_preset_name, handedness=handedness)
            registry[base_topology.name] = base_topology

            if not _supports_topology_expansion(cfg, base_hand_preset_name=hand_preset_name):
                continue
            if getattr(cfg, "missing", True):
                for spec in _build_missing_topology_specs(base_topology):
                    registry[spec.name] = spec
            if getattr(cfg, "mixed", False):
                for spec in _build_mixed_topology_specs(base_topology):
                    registry[spec.name] = spec
    return registry


def _resolve_premade_topology_spec(cfg: Any, topology_name: str) -> PremadeTopologySpec:
    r"""按名字返回当前 cfg 下的一份 premade topology 规格。"""

    registry = _build_premade_topology_registry(cfg)
    try:
        return registry[topology_name]
    except KeyError as exc:
        raise KeyError(f"Unknown premade topology {topology_name!r}") from exc


def _configured_connectivity_value(
    cfg: Any,
    *,
    base_hand_preset_name: str,
) -> dict[str, list[str]] | None:
    r"""读取某个 base hand 的 connectivity façade 配置。"""

    if cfg.connectivity_presets is None:
        return None
    return cfg.connectivity_presets.get(base_hand_preset_name)


def _build_connectivity_selection_registry(
    cfg: Any,
    *,
    topology: PremadeTopologySpec,
) -> dict[str, dict[str, str]]:
    r"""为某个 topology 展开可用的 slot-level connectivity 选择空间。"""

    configured = _configured_connectivity_value(cfg, base_hand_preset_name=topology.base_hand_preset)
    return _build_slot_level_connectivity_selection_registry(topology=topology, configured_slot_pools=configured)


def _build_slot_level_connectivity_selection_registry(
    *,
    topology: PremadeTopologySpec,
    configured_slot_pools: dict[str, list[str]] | None,
) -> dict[str, dict[str, str]]:
    r"""按 surviving slots 的 candidate pool 做笛卡尔展开。"""

    slot_pools: dict[str, tuple[str, ...]] = {}
    slot_family_map = topology.slot_family_map()
    for slot_name in topology.surviving_slots:
        slot_kind = _slot_finger_kind(slot_name)
        candidate_names = (
            configured_slot_pools.get(slot_name)
            if configured_slot_pools is not None and slot_name in configured_slot_pools
            else list_finger_connectivity_preset_names(
                family=slot_family_map[slot_name],
                finger_kind=slot_kind,
            )
        )
        if not candidate_names:
            raise ValueError(
                f"No connectivity candidates are available for topology {topology.name!r} slot {slot_name!r}"
            )

        compatible: list[str] = []
        for recipe_name in candidate_names:
            recipe = get_finger_connectivity_preset_data(recipe_name)
            if recipe.finger_kind != slot_kind:
                raise ValueError(
                    f"Finger connectivity preset {recipe_name!r} is for {recipe.finger_kind!r}, "
                    f"but topology {topology.name!r} slot {slot_name!r} expects {slot_kind!r}"
                )
            if recipe.family != slot_family_map[slot_name]:
                continue  # slot-level pool 允许同时列出多 family recipe；真正展开到某个 topology 时再按 family 过滤
            compatible.append(recipe_name)
        if not compatible:
            return {}  # 当前 topology 在给定 slot-level pool 下没有合法 connectivity 组合，直接整手跳过
        slot_pools[slot_name] = tuple(compatible)

    registry: dict[str, dict[str, str]] = {}
    ordered_slots = tuple(slot_name for slot_name in _PREMADE_SLOT_ORDER if slot_name in slot_pools)
    for combination in product(*(slot_pools[slot_name] for slot_name in ordered_slots)):
        slot_recipe_names = {
            slot_name: recipe_name for slot_name, recipe_name in zip(ordered_slots, combination)
        }
        registry[_format_slot_level_connectivity_name(slot_recipe_names)] = slot_recipe_names
    return registry


def _format_slot_level_connectivity_name(slot_recipe_names: dict[str, str]) -> str:
    r"""把 slot-level finger connectivity 组合写成显式可读名。"""

    parts: list[str] = []
    for slot_name in _PREMADE_SLOT_ORDER:
        if slot_name not in slot_recipe_names:
            continue
        recipe = get_finger_connectivity_preset_data(slot_recipe_names[slot_name])
        parts.append(f"{slot_name}-{_finger_connectivity_short_code(recipe)}")
    return "__".join(parts)


def _finger_connectivity_short_code(recipe: Any) -> str:
    r"""把一条 finger-level recipe 压成简洁但仍可读的短码。"""

    if not recipe.deleted_joint_suffixes:
        return "full"
    return "drop_" + "_".join(str(token) for token in recipe.deleted_joint_suffixes)


def _slot_revolute_count_from_recipe_name(recipe_name: str) -> int:
    r"""把单个 slot 选中的 connectivity recipe 映射成 surviving revolute DOF 数。"""

    recipe = get_finger_connectivity_preset_data(recipe_name)
    return _connectivity_remaining_revolute_count(recipe)


def _slot_revolute_count_map(slot_recipe_names: dict[str, str]) -> dict[str, int]:
    r"""为当前 topology recipe 组合生成 `{slot -> remaining_revolute}` 映射。"""

    return {
        slot_name: _slot_revolute_count_from_recipe_name(recipe_name)
        for slot_name, recipe_name in slot_recipe_names.items()
    }


def _format_single_family_variant_name(
    *,
    handedness: Literal["left", "right"],
    slot_recipe_names: dict[str, str],
) -> str:
    r"""把 single / missing topology 写成 `<handedness>_t<n>_i<n>_...`。"""

    slot_counts = _slot_revolute_count_map(slot_recipe_names)
    parts = [handedness]
    for slot_name in _PREMADE_SLOT_ORDER:
        if slot_name not in slot_counts:
            continue  # missing 的语义就是“缺 slot 时不写该 token”
        parts.append(f"{_PREMADE_SLOT_VARIANT_TOKEN[slot_name]}{slot_counts[slot_name]}")
    return "_".join(parts)


def _iter_consecutive_family_slot_groups(
    *,
    slot_family_map: dict[str, str],
    surviving_slots: tuple[str, ...],
) -> list[tuple[str, tuple[str, ...]]]:
    r"""按 canonical slot 顺序压缩连续同 family 的 slot 段。"""

    groups: list[tuple[str, tuple[str, ...]]] = []
    current_family: str | None = None
    current_slots: list[str] = []

    for slot_name in _PREMADE_SLOT_ORDER:
        if slot_name not in surviving_slots:
            continue
        family = slot_family_map[slot_name]
        if family == current_family:
            current_slots.append(slot_name)
            continue
        if current_family is not None:
            groups.append((current_family, tuple(current_slots)))
        current_family = family
        current_slots = [slot_name]

    if current_family is not None:
        groups.append((current_family, tuple(current_slots)))
    return groups


def _format_mixed_group_name(topology: PremadeTopologySpec) -> str:
    r"""生成 mixed 第一层 family-composition 分组目录名。"""

    slot_family_map = topology.slot_family_map()
    groups = _iter_consecutive_family_slot_groups(
        slot_family_map=slot_family_map,
        surviving_slots=topology.surviving_slots,
    )

    parts = [topology.family, topology.base_hand_preset]
    previous_family = topology.family
    for family, slot_group in groups:
        if family != previous_family:
            parts.append(family)  # 压缩连续同族段，只在切换时重复写 family 前缀
        parts.extend(slot_group)
        previous_family = family
    return "_".join(parts)


def _format_mixed_variant_name(
    *,
    handedness: Literal["left", "right"],
    slot_family_map: dict[str, str],
    slot_recipe_names: dict[str, str],
) -> str:
    r"""把 mixed topology 写成 `<handedness>_<family>_t<n>_<family>_i<n>_...`。"""

    slot_counts = _slot_revolute_count_map(slot_recipe_names)
    parts = [handedness]
    previous_family: str | None = None

    for slot_name in _PREMADE_SLOT_ORDER:
        if slot_name not in slot_counts:
            continue
        family = slot_family_map[slot_name]
        if family != previous_family:
            parts.append(family)
            previous_family = family
        parts.append(f"{_PREMADE_SLOT_VARIANT_TOKEN[slot_name]}{slot_counts[slot_name]}")
    return "_".join(parts)


def _resolve_topology_output_identity(
    cfg: Any,
    *,
    topology: PremadeTopologySpec,
    connectivity_preset_name: str,
    selected_slot_recipes: dict[str, str],
) -> dict[str, Any]:
    r"""把内部 topology registry key lower 成最终导出目录身份。"""

    if topology.topology_kind == "mixed":
        topology_group_name = _format_mixed_group_name(topology)
        base_variant_name = _format_mixed_variant_name(
            handedness=topology.handedness,
            slot_family_map=topology.slot_family_map(),
            slot_recipe_names=selected_slot_recipes,
        )
    else:
        topology_group_name = topology.base_hand_preset
        base_variant_name = _format_single_family_variant_name(
            handedness=topology.handedness,
            slot_recipe_names=selected_slot_recipes,
        )

    duplicate_selection_names: list[str] = []
    selection_registry = _build_connectivity_selection_registry(cfg, topology=topology)
    for selection_name, candidate_slot_recipes in selection_registry.items():
        if topology.topology_kind == "mixed":
            candidate_base_name = _format_mixed_variant_name(
                handedness=topology.handedness,
                slot_family_map=topology.slot_family_map(),
                slot_recipe_names=candidate_slot_recipes,
            )
        else:
            candidate_base_name = _format_single_family_variant_name(
                handedness=topology.handedness,
                slot_recipe_names=candidate_slot_recipes,
            )
        if candidate_base_name == base_variant_name:
            duplicate_selection_names.append(selection_name)

    duplicate_selection_names = sorted(duplicate_selection_names)
    variant_index = (
        duplicate_selection_names.index(connectivity_preset_name) + 1
        if connectivity_preset_name in duplicate_selection_names
        else 1
    )
    topology_name = (
        base_variant_name
        if len(duplicate_selection_names) <= 1
        else f"{base_variant_name}_{variant_index}"
    )
    return {
        "topology_group_name": topology_group_name,
        "topology_name": topology_name,
        "topology_variant_base_name": base_variant_name,
        "topology_variant_index": variant_index,
    }


def _resolve_slot_level_connectivity_selection(
    hand_cfg: HandCfg,
    *,
    connectivity_preset_name: str,
    hand_preset_name: str | None,
) -> dict[str, str]:
    r"""把 connectivity 名展开成当前 topology 上每个 slot 的 finger recipe。"""

    topology_metadata = _extract_premade_topology_metadata(hand_cfg, hand_preset_name=hand_preset_name)

    try:
        legacy_preset = get_hand_connectivity_preset_data(connectivity_preset_name)
    except KeyError:
        legacy_preset = None
    if legacy_preset is not None:
        return {
            slot_name: legacy_preset.finger_slots[slot_name]
            for slot_name in topology_metadata["surviving_slots"]
            if slot_name in legacy_preset.finger_slots
        }

    slot_family_map = topology_metadata["slot_family_map"]
    resolved: dict[str, str] = {}
    for token in str(connectivity_preset_name).split("__"):
        slot_name, short_code = token.split("-", 1)
        resolved[slot_name] = _resolve_finger_connectivity_name_from_short_code(
            family=slot_family_map[slot_name],
            finger_kind=_slot_finger_kind(slot_name),
            short_code=short_code,
        )
    return resolved


def _resolve_finger_connectivity_name_from_short_code(
    *,
    family: str,
    finger_kind: Literal["thumb", "non_thumb"],
    short_code: str,
) -> str:
    r"""由短码反解真实的 finger connectivity preset 名。"""

    for recipe_name in list_finger_connectivity_preset_names(family=family, finger_kind=finger_kind):
        recipe = get_finger_connectivity_preset_data(recipe_name)
        if _finger_connectivity_short_code(recipe) == short_code:
            return recipe_name
    raise ValueError(
        f"Cannot resolve finger connectivity code {short_code!r} for family={family!r}, finger_kind={finger_kind!r}"
    )


def _extract_premade_topology_metadata(hand_cfg: HandCfg, *, hand_preset_name: str | None) -> dict[str, Any]:
    r"""从 HandCfg metadata 中读取 premade topology provenance。"""

    metadata = dict(hand_cfg.metadata or {})
    topology_metadata = metadata.get("premade_topology")
    if isinstance(topology_metadata, dict):
        return topology_metadata
    if hand_preset_name is None:
        raise ValueError("HandCfg is missing premade_topology metadata")
    return {
        "topology_registry_key": hand_preset_name,
        "base_hand_preset": hand_preset_name,
        "handedness": hand_cfg.handedness,
        "topology_kind": "single_family",
        "topology_anchor": hand_preset_name,
        "topology_name": hand_preset_name,
        "surviving_slots": [slot_name for slot_name in _PREMADE_SLOT_ORDER if slot_name in {"thumb", "index", "middle", "ring"}],
        "slot_finger_presets": {},
        "slot_family_map": {slot_name: hand_cfg.family for slot_name in ("thumb", "index", "middle", "ring")},
    }


def _make_builder_cfg_from_topology(topology: PremadeTopologySpec):
    r"""把 premade topology 规格 lower 成一份可直接 build 的 hand builder cfg。"""

    base_builder_cfg = make_human_like_builder_cfg_from_preset(
        topology.base_hand_preset,
        name=topology.name,
    )
    non_thumb_cfg = {
        slot_name: resolve_finger_builder_cfg(preset_name)
        for slot_name, preset_name in topology.finger_preset_names.items()
        if slot_name != "thumb"
    }
    thumb_cfg = (
        resolve_finger_builder_cfg(topology.finger_preset_names["thumb"])
        if "thumb" in topology.finger_preset_names
        else None
    )
    return base_builder_cfg.replace(
        name=topology.name,
        handedness=topology.handedness,
        finger_cfg=non_thumb_cfg,
        thumb_cfg=thumb_cfg,
    )


def stable_premade_id(*parts: str) -> str:
    r"""为 pre-made 的离散 recipe 组合生成稳定短 ID。

    在 `enumerate` 路径里，我们希望：

    - 同一个 `(base_hand_preset, connectivity_preset)` 组合，多次生成时 ID 稳定；
    - sidecar / output path 能直接靠这组 provenance 回溯；
    - 但目录名又不要长到影响人工浏览。

    因而这里对 provenance 字符串做 md5，再取前 8 位十六进制作为稳定短签名。
    """

    payload = "::".join(parts).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:8]


def resolve_deleted_joint_names(finger: Any, *, deleted_joint_suffixes: tuple[str, ...]) -> tuple[str, ...]:
    r"""把 slot-agnostic 的 delete recipe 展开成当前 finger 上的真实 joint 名。

    例如，当 finger 为 `index`，而 recipe 写的是 `("j2", "j3")` 时，这里会解析成：

    - `index_j2`
    - `index_j3`

    这样 connectivity preset 的科学语义仍然是“显式删除哪些 joint”，
    只是为了避免对 index / middle / ring 重复抄写，允许 recipe 在注册层使用后缀表达。
    """

    joint_name_set = {joint.name for joint in finger.joints}  # 当前 finger 真实存在的 joint 名全集
    resolved: list[str] = []
    for suffix in deleted_joint_suffixes:
        candidate = str(suffix)  # 允许 recipe 直接写完整 joint 名，也允许只写后缀
        if candidate not in joint_name_set and not candidate.startswith(f"{finger.name}_"):
            candidate = f"{finger.name}_{candidate}"  # 把 `j2` 展开成 `index_j2` / `thumb_j2` 这类真实名字
        if candidate not in joint_name_set:
            raise ValueError(
                f"Deleted joint token {suffix!r} cannot be resolved on finger {finger.name!r}; "
                f"available joints are {[joint.name for joint in finger.joints]!r}"
            )
        resolved.append(candidate)
    return tuple(resolved)


def candidate_hand_preset_names(cfg: Any) -> tuple[str, ...]:
    r"""返回当前 generator cfg 可见的 pre-made topology registry key 集合。"""

    return tuple(_build_premade_topology_registry(cfg))


def connectivity_names_for_hand_preset(cfg: Any, *, hand_preset_name: str) -> tuple[str, ...]:
    r"""返回某个 premade topology registry key 允许搭配的 connectivity 名集合。

    这里统一屏蔽“legacy hand-level list”与“new slot-level candidate pool”的差异，
    对 `HandGenerator` 只返回当前 topology 可以真正枚举的连接性名字列表。
    """

    topology = _resolve_premade_topology_spec(cfg, hand_preset_name)
    return tuple(_build_connectivity_selection_registry(cfg, topology=topology))


def resolve_single_premade_selection(cfg: Any) -> tuple[str | None, str | None] | None:
    r"""为 `generate()` 的单样本路径解析一次 topology + connectivity 联合采样。"""

    topology_candidates = candidate_hand_preset_names(cfg)
    if not topology_candidates:
        return None

    available_pairs: list[tuple[str, tuple[str, ...]]] = []
    for topology_name in topology_candidates:
        connectivity_names = connectivity_names_for_hand_preset(cfg, hand_preset_name=topology_name)
        if connectivity_names:
            available_pairs.append((topology_name, connectivity_names))
    if not available_pairs:
        return None

    topology_name, connectivity_names = random.choice(available_pairs)
    return topology_name, random.choice(connectivity_names)


def build_base_hand(cfg: Any, *, hand_preset_name: str | None) -> tuple[HandCfg, str]:
    r"""构建本次样本的 canonical base hand。

    base hand 的来源按以下优先级收敛：

    1. 若 `Made` 已经是具体 builder cfg，则优先使用它；
    2. 否则若给了 `hand_preset_name`，就从 hand preset 解析出 builder cfg；
    3. 两者都没有时，说明当前 cfg 既没有具体 `Made`，也没有 pre-made hand preset，
       这在运行时应视为无效输入。

    这样做的动机，是同时支持两条工作流：

    - 正式 pre-made：`hand_preset -> canonical hand`
    - 科研局部实验：当 `hand_presets` 只有一个锚点时，`Made` 仍可作为局部覆写后的
      concrete builder cfg，帮助你在不改 hand preset 名称的前提下快速调试
    """

    if cfg.Made.class_type is not HandBuilder:
        builder_cfg = cfg.Made  # 显式 `Made` 一旦具体化，就说明用户要以它作为真实基座
    elif hand_preset_name is not None:
        topology = _resolve_premade_topology_spec(cfg, hand_preset_name)
        builder_cfg = _make_builder_cfg_from_topology(topology)
    else:
        raise ValueError("HandGenerator requires a concrete Made cfg or at least one hand preset when using the pre-made facade")

    builder = builder_cfg.class_type(builder_cfg)
    hand_cfg = builder.build()
    if hand_preset_name is not None and cfg.Made.class_type is HandBuilder:
        topology_metadata = _resolve_premade_topology_spec(cfg, hand_preset_name).to_metadata()
        hand_metadata = dict(hand_cfg.metadata)
        hand_metadata["premade_topology"] = topology_metadata
        hand_cfg = hand_cfg.replace(metadata=hand_metadata)
    return hand_cfg, builder_cfg.__class__.__name__


def apply_connectivity_preset(
    cfg: Any,
    hand_cfg: HandCfg,
    *,
    connectivity_preset_name: str,
    hand_preset_name: str | None,
) -> tuple[HandCfg, dict[str, Any]]:
    r"""把 connectivity 选择 lower 成显式的 joint delete + regroup 结果。

    当前入口名仍保留 `connectivity_preset_name`，是为了兼容旧调用点；但真正的语义
    已经允许它表示两类对象：

    1. legacy hand-level alias，例如 `allegro_full`
    2. new slot-level selection，例如 `thumb-full__index-drop_j3__middle-full__ring-full`
    """

    # 局部导入 `JointDeleteMutator`，保留当前 generator 主路径的 fallback import 习惯。
    from .mutate import JointDeleteCfg, JointDeleteMutator

    mutated = hand_cfg.copy()
    topology_metadata = _extract_premade_topology_metadata(mutated, hand_preset_name=hand_preset_name)
    topology = _resolve_premade_topology_spec(
        cfg,
        str(topology_metadata.get("topology_registry_key") or hand_preset_name),
    )
    selected_slot_recipes = _resolve_slot_level_connectivity_selection(
        mutated,
        connectivity_preset_name=connectivity_preset_name,
        hand_preset_name=hand_preset_name,
    )
    per_finger_connectivity: dict[str, Any] = {}

    for finger_name, finger_recipe_name in selected_slot_recipes.items():
        current_finger = next((finger for finger in mutated.fingers if finger.name == finger_name), None)
        if current_finger is None:
            continue  # 当前 hand 若没有这个 slot，就跳过，不对未来 little-finger 扩展设死约束

        finger_recipe = get_finger_connectivity_preset_data(finger_recipe_name)
        deleted_joint_names = resolve_deleted_joint_names(
            current_finger,
            deleted_joint_suffixes=finger_recipe.deleted_joint_suffixes,
        )
        deleted_joint_set = set(deleted_joint_names)  # 便于同时回溯被删 joint 与被删 child-link
        deleted_child_links = [
            str(joint.child)
            for joint in current_finger.joints
            if joint.name in deleted_joint_set
        ]
        remaining_revolute = sum(
            1
            for joint in current_finger.joints
            if joint.joint_type == "revolute" and joint.name not in deleted_joint_set
        )
        per_finger_connectivity[finger_name] = {
            "finger_connectivity_preset": finger_recipe.name,
            "deleted_joint_suffixes": list(finger_recipe.deleted_joint_suffixes),
            "deleted_joints": list(deleted_joint_names),
            "deleted_child_links": deleted_child_links,
            "remaining_revolute": remaining_revolute,
            "regroup_strategy": finger_recipe.regroup_strategy,
            "slot_family": topology_metadata["slot_family_map"].get(finger_name),
            "slot_finger_preset": topology_metadata["slot_finger_presets"].get(finger_name),
        }

        if not deleted_joint_names:
            continue  # full chain 这类 recipe 不需要真正执行 delete

        lowered = JointDeleteMutator(
            JointDeleteCfg(
                target_finger=finger_name,
                deleted_joints=deleted_joint_names,
                regroup_strategy=finger_recipe.regroup_strategy,
                respect_preset=False,  # legality 已由 connectivity registry 定义，这里不再让 generic mutator 额外裁决
                keep_terminal_joint=True,
            )
        ).mutate(mutated)
        if lowered is None:
            raise ValueError(
                f"Failed to lower connectivity preset {connectivity_preset_name!r} on finger {finger_name!r}"
            )
        mutated = lowered

    output_identity = _resolve_topology_output_identity(
        cfg,
        topology=topology,
        connectivity_preset_name=connectivity_preset_name,
        selected_slot_recipes=selected_slot_recipes,
    )

    hand_metadata = dict(mutated.metadata)
    hand_metadata["premade_connectivity"] = {
        "topology_registry_key": topology_metadata["topology_registry_key"],
        "base_hand_preset": topology_metadata["base_hand_preset"],
        "handedness": topology_metadata["handedness"],
        "topology_kind": topology_metadata["topology_kind"],
        "topology_anchor": topology_metadata["topology_anchor"],
        "topology_group_name": output_identity["topology_group_name"],
        "topology_name": output_identity["topology_name"],
        "topology_variant_base_name": output_identity["topology_variant_base_name"],
        "topology_variant_index": output_identity["topology_variant_index"],
        "surviving_slots": topology_metadata["surviving_slots"],
        "slot_family_map": topology_metadata["slot_family_map"],
        "slot_finger_presets": topology_metadata["slot_finger_presets"],
        "connectivity_preset": connectivity_preset_name,
        "selected_slot_recipes": selected_slot_recipes,
        "per_finger": per_finger_connectivity,
    }
    mutated = mutated.replace(metadata=hand_metadata)
    return mutated, {
        "topology_registry_key": topology_metadata["topology_registry_key"],
        "base_hand_preset": topology_metadata["base_hand_preset"],
        "handedness": topology_metadata["handedness"],
        "topology_kind": topology_metadata["topology_kind"],
        "topology_anchor": topology_metadata["topology_anchor"],
        "topology_group_name": output_identity["topology_group_name"],
        "topology_name": output_identity["topology_name"],
        "topology_variant_base_name": output_identity["topology_variant_base_name"],
        "topology_variant_index": output_identity["topology_variant_index"],
        "surviving_slots": topology_metadata["surviving_slots"],
        "slot_family_map": topology_metadata["slot_family_map"],
        "slot_finger_presets": topology_metadata["slot_finger_presets"],
        "connectivity_preset": connectivity_preset_name,
        "selected_slot_recipes": selected_slot_recipes,
        "per_finger_connectivity": per_finger_connectivity,
    }


def resolve_export_root(
    cfg: Any,
    *,
    result: HandGenerationResult,
    run_root: Path | None = None,
) -> Path:
    r"""根据 pre-made provenance 与 `output_layout` 计算本次导出的根目录。

    当前导出器仍保持它一贯的职责边界：

    - `HandExporter` 负责在传入目录下再补一层 `{sample_id}/`
    - `HandGenerator` 负责决定这个“传入目录”到底应该是
      `generated/<timestamp>/...` 下的哪一个 topology 目录

    这样可以在不破坏现有 exporter 结构的前提下，把目录语义仍然收口到
    `HandGeneratorCfg` 这个唯一 façade。
    """

    effective_root = run_root or Path(cfg.output_dir)
    if result.hand_cfg is None or "connectivity_preset" not in result.metadata:
        return effective_root

    if cfg.output_layout == "flat":
        return effective_root / "flat"

    topology_kind = str(result.metadata.get("topology_kind") or "single_family")
    topology_name = str(result.metadata.get("topology_name") or result.metadata.get("connectivity_preset") or result.hand_cfg.family)
    topology_group_name = str(
        result.metadata.get("topology_group_name")
        or result.metadata.get("base_hand_preset")
        or result.hand_cfg.family
    )
    if topology_kind == "mixed":
        return effective_root / "mixed" / topology_group_name / topology_name
    return effective_root / topology_group_name / topology_name


__all__ = [
    "normalize_name_list",
    "normalize_connectivity_mapping",
    "stable_premade_id",
    "resolve_deleted_joint_names",
    "candidate_hand_preset_names",
    "connectivity_names_for_hand_preset",
    "resolve_single_premade_selection",
    "build_base_hand",
    "apply_connectivity_preset",
    "resolve_export_root",
]
