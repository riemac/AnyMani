"""pre-made 输出 identity 与稳定 ID。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from ...asset_base import HandCfg
from ...presets.connectivity_presets import _remaining_revolute_count as _connectivity_remaining_revolute_count
from ...presets.connectivity_presets import get_finger_connectivity_preset_data
from ..result import HandGenerationResult
from .topology import PremadeTopologySpec


_PREMADE_SLOT_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")
_PREMADE_SLOT_VARIANT_TOKEN: dict[str, str] = {
    "thumb": "t",
    "index": "i",
    "middle": "m",
    "ring": "r",
    "little": "l",
}


def stable_premade_id(*parts: str) -> str:
    r"""为 pre-made 的离散 recipe 组合生成稳定短 ID。"""

    payload = "::".join(parts).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:8]


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
            continue
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
            parts.append(family)
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


def resolve_topology_output_identity(
    cfg: Any,
    *,
    topology: PremadeTopologySpec,
    connectivity_preset_name: str,
    selected_slot_recipes: dict[str, str],
    selection_registry: dict[str, dict[str, str]],
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
    topology_name = base_variant_name if len(duplicate_selection_names) <= 1 else f"{base_variant_name}_{variant_index}"
    return {
        "topology_group_name": topology_group_name,
        "topology_name": topology_name,
        "topology_variant_base_name": base_variant_name,
        "topology_variant_index": variant_index,
    }


def resolve_export_root(
    cfg: Any,
    *,
    result: HandGenerationResult,
    run_root: Path | None = None,
) -> Path:
    r"""根据 pre-made provenance 计算 topology 根目录。"""

    effective_root = run_root or Path(cfg.output_dir)
    if result.hand_cfg is None or "connectivity_preset" not in result.metadata:
        return effective_root

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
    "resolve_export_root",
    "resolve_topology_output_identity",
    "stable_premade_id",
]
