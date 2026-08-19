"""pre-made connectivity 选择、lowering 与导出 provenance。"""

from __future__ import annotations

import random
from itertools import product
from typing import Any, Literal

from ...asset_base import HandCfg
from ...presets.connectivity_presets import (
    get_finger_connectivity_preset_data,
    get_hand_connectivity_preset_data,
    list_finger_connectivity_preset_names,
)
from .identity import resolve_topology_output_identity
from .topology import (
    PremadeTopologySpec,
    candidate_hand_preset_names,
    extract_premade_topology_metadata,
    resolve_premade_topology_spec,
    slot_finger_kind,
)

_PREMADE_SLOT_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")


def resolve_deleted_joint_names(finger: Any, *, deleted_joint_suffixes: tuple[str, ...]) -> tuple[str, ...]:
    r"""把 slot-agnostic 的 delete recipe 展开成当前 finger 上的真实 joint 名。"""

    joint_name_set = {joint.name for joint in finger.joints}
    resolved: list[str] = []
    for suffix in deleted_joint_suffixes:
        candidate = str(suffix)
        if candidate not in joint_name_set and not candidate.startswith(f"{finger.name}_"):
            candidate = f"{finger.name}_{candidate}"
        if candidate not in joint_name_set:
            raise ValueError(
                f"Deleted joint token {suffix!r} cannot be resolved on finger {finger.name!r}; "
                f"available joints are {[joint.name for joint in finger.joints]!r}"
            )
        resolved.append(candidate)
    return tuple(resolved)


def _configured_connectivity_value(
    cfg: Any,
    *,
    base_hand_preset_name: str,
) -> dict[str, list[str]] | None:
    r"""读取某个 base hand 的 connectivity façade 配置。"""

    if cfg.connectivity_presets is None:
        return None
    return cfg.connectivity_presets.get(base_hand_preset_name)


def _finger_connectivity_short_code(recipe: Any) -> str:
    r"""把一条 finger-level recipe 压成简洁但仍可读的短码。"""

    if not recipe.deleted_joint_suffixes:
        return "full"
    return "drop_" + "_".join(str(token) for token in recipe.deleted_joint_suffixes)


def _format_slot_level_connectivity_name(slot_recipe_names: dict[str, str]) -> str:
    r"""把 slot-level finger connectivity 组合写成显式可读名。"""

    parts: list[str] = []
    for slot_name in _PREMADE_SLOT_ORDER:
        if slot_name not in slot_recipe_names:
            continue
        recipe = get_finger_connectivity_preset_data(slot_recipe_names[slot_name])
        parts.append(f"{slot_name}-{_finger_connectivity_short_code(recipe)}")
    return "__".join(parts)


def build_connectivity_selection_registry(
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
        slot_kind = slot_finger_kind(slot_name)
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
                continue
            compatible.append(recipe_name)
        if not compatible:
            return {}
        slot_pools[slot_name] = tuple(compatible)

    registry: dict[str, dict[str, str]] = {}
    ordered_slots = tuple(slot_name for slot_name in _PREMADE_SLOT_ORDER if slot_name in slot_pools)
    for combination in product(*(slot_pools[slot_name] for slot_name in ordered_slots)):
        slot_recipe_names = {slot_name: recipe_name for slot_name, recipe_name in zip(ordered_slots, combination)}
        registry[_format_slot_level_connectivity_name(slot_recipe_names)] = slot_recipe_names
    return registry


def connectivity_names_for_hand_preset(cfg: Any, *, hand_preset_name: str) -> tuple[str, ...]:
    r"""返回某个 premade topology registry key 允许搭配的 connectivity 名集合。"""

    topology = resolve_premade_topology_spec(cfg, hand_preset_name)
    return tuple(build_connectivity_selection_registry(cfg, topology=topology))


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


def _resolve_slot_level_connectivity_selection(
    hand_cfg: HandCfg,
    *,
    connectivity_preset_name: str,
    hand_preset_name: str | None,
) -> dict[str, str]:
    r"""把 connectivity 名展开成当前 topology 上每个 slot 的 finger recipe。"""

    topology_metadata = extract_premade_topology_metadata(hand_cfg, hand_preset_name=hand_preset_name)

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
            finger_kind=slot_finger_kind(slot_name),
            short_code=short_code,
        )
    return resolved


def apply_connectivity_preset(
    cfg: Any,
    hand_cfg: HandCfg,
    *,
    connectivity_preset_name: str,
    hand_preset_name: str | None,
) -> tuple[HandCfg, dict[str, Any]]:
    r"""把 connectivity 选择 lower 成显式的 joint delete + regroup 结果。"""

    from .connectivity_lowering import JointDeleteCfg, JointDeleteMutator

    mutated = hand_cfg.copy()
    topology_metadata = extract_premade_topology_metadata(mutated, hand_preset_name=hand_preset_name)
    topology = resolve_premade_topology_spec(
        cfg,
        str(topology_metadata.get("topology_registry_key") or hand_preset_name),
    )
    selected_slot_recipes = _resolve_slot_level_connectivity_selection(
        mutated,
        connectivity_preset_name=connectivity_preset_name,
        hand_preset_name=hand_preset_name,
    )
    selection_registry = build_connectivity_selection_registry(cfg, topology=topology)
    per_finger_connectivity: dict[str, Any] = {}

    for finger_name, finger_recipe_name in selected_slot_recipes.items():
        current_finger = next((finger for finger in mutated.fingers if finger.name == finger_name), None)
        if current_finger is None:
            continue

        finger_recipe = get_finger_connectivity_preset_data(finger_recipe_name)
        deleted_joint_names = resolve_deleted_joint_names(
            current_finger,
            deleted_joint_suffixes=finger_recipe.deleted_joint_suffixes,
        )
        deleted_joint_set = set(deleted_joint_names)
        deleted_child_links = [str(joint.child) for joint in current_finger.joints if joint.name in deleted_joint_set]
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
            continue

        lowered = JointDeleteMutator(
            JointDeleteCfg(
                target_finger=finger_name,
                deleted_joints=deleted_joint_names,
                regroup_strategy=finger_recipe.regroup_strategy,
                respect_preset=False,
                keep_terminal_joint=True,
            )
        ).mutate(mutated)
        if lowered is None:
            raise ValueError(
                f"Failed to lower connectivity preset {connectivity_preset_name!r} on finger {finger_name!r}"
            )
        mutated = lowered

    output_identity = resolve_topology_output_identity(
        cfg,
        topology=topology,
        connectivity_preset_name=connectivity_preset_name,
        selected_slot_recipes=selected_slot_recipes,
        selection_registry=selection_registry,
    )

    hand_metadata = dict(mutated.metadata)
    hand_metadata["premade_connectivity"] = {
        "topology_registry_key": topology_metadata["topology_registry_key"],
        "base_hand_preset": topology_metadata["base_hand_preset"],
        "handedness": topology_metadata["handedness"],
        "topology_kind": topology_metadata["topology_kind"],
        "family_composition": topology_metadata["family_composition"],
        "missing_slots": topology_metadata["missing_slots"],
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
        "family_composition": topology_metadata["family_composition"],
        "missing_slots": topology_metadata["missing_slots"],
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


__all__ = [
    "apply_connectivity_preset",
    "build_connectivity_selection_registry",
    "candidate_hand_preset_names",
    "connectivity_names_for_hand_preset",
    "resolve_deleted_joint_names",
    "resolve_single_premade_selection",
]
