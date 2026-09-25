"Defines explicit finger-chain deletion recipes and hand-level family compositions. JointCfg owns its child-link geometry, so deleting a joint deletes that child-link geometry; recipe names are provenance shorthand while deleted suffixes define execution."

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal





NON_THUMB_SLOTS: tuple[str, ...] = ("index", "middle", "ring")




_CANONICAL_REVOLUTE_COUNT: dict[tuple[str, Literal["non_thumb", "thumb"]], int] = {
    ("allegro", "non_thumb"): 4,
    ("allegro", "thumb"): 4,
    ("leap", "non_thumb"): 4,
    ("leap", "thumb"): 4,
}


@dataclass
class FingerConnectivityPreset:
    "Allowed joint chain and mechanism recipe for one semantic finger slot."

    name: str
    family: str
    finger_kind: Literal["non_thumb", "thumb"]
    deleted_joint_suffixes: tuple[str, ...] = ()
    regroup_strategy: Literal["drop", "merge"] = "drop"
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HandConnectivityPreset:
    "Allowed thumb and non-thumb connectivity recipes for one base-palm family."

    name: str
    family: str
    finger_slots: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _build_finger_connectivity_registry() -> dict[str, FingerConnectivityPreset]:

    return {
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        "allegro_non_thumb_full": FingerConnectivityPreset(
            name="allegro_non_thumb_full",
            family="allegro",
            finger_kind="non_thumb",
            deleted_joint_suffixes=(),
            regroup_strategy="drop",
            note='Allegro non-thumb full chain; retain `j0, j1, j2, j3, tip`.',
        ),
        "allegro_non_thumb_drop_j3": FingerConnectivityPreset(
            name="allegro_non_thumb_drop_j3",
            family="allegro",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j3",),
            regroup_strategy="drop",
            note='Allegro non-thumb without distal `j3`; remove its child-link geometry and do not merge it into the parent segment.',
        ),
        "allegro_non_thumb_drop_j2_j3": FingerConnectivityPreset(
            name="allegro_non_thumb_drop_j2_j3",
            family="allegro",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j2", "j3"),
            regroup_strategy="drop",
            note='Allegro non-thumb without `j2` and `j3`; retain the two proximal joints and tip.',
        ),
        # ------------------------------------------------------------------


        # ------------------------------------------------------------------
        "leap_non_thumb_full": FingerConnectivityPreset(
            name="leap_non_thumb_full",
            family="leap",
            finger_kind="non_thumb",
            deleted_joint_suffixes=(),
            regroup_strategy="drop",
            note='LEAP non-thumb full chain; retain `root_fixed`, `j0` through `j3`, and `tip`.',
        ),
        "leap_non_thumb_drop_j3": FingerConnectivityPreset(
            name="leap_non_thumb_drop_j3",
            family="leap",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j3",),
            regroup_strategy="drop",
            note='LEAP non-thumb without distal `j3`; retain the fixed root and tip.',
        ),
        "leap_non_thumb_drop_j2_j3": FingerConnectivityPreset(
            name="leap_non_thumb_drop_j2_j3",
            family="leap",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j2", "j3"),
            regroup_strategy="drop",
            note='LEAP non-thumb without `j2` and `j3`; retain `root_fixed`, `j0`, `j1`, and `tip`.',
        ),
        "leap_non_thumb_drop_j1_j2_j3": FingerConnectivityPreset(
            name="leap_non_thumb_drop_j1_j2_j3",
            family="leap",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j1", "j2", "j3"),
            regroup_strategy="drop",
            note='LEAP non-thumb without `j1`, `j2`, and `j3`; retain only `root_fixed`, `j0`, and `tip`.',
        ),
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        "allegro_thumb_full": FingerConnectivityPreset(
            name="allegro_thumb_full",
            family="allegro",
            finger_kind="thumb",
            deleted_joint_suffixes=(),
            regroup_strategy="drop",
            note='Allegro thumb full chain; retain `j0`, `j1`, `j2`, `j3`, and `tip`.',
        ),
        "allegro_thumb_drop_j3": FingerConnectivityPreset(
            name="allegro_thumb_drop_j3",
            family="allegro",
            finger_kind="thumb",
            deleted_joint_suffixes=("j3",),
            regroup_strategy="drop",
            note='Allegro thumb without distal `j3`; retain the tip and connect it to the nearest surviving segment.',
        ),
        "leap_thumb_full": FingerConnectivityPreset(
            name="leap_thumb_full",
            family="leap",
            finger_kind="thumb",
            deleted_joint_suffixes=(),
            regroup_strategy="drop",
            note='LEAP thumb full chain; retain `j0`, `j1`, `j2`, `j3`, and `tip`.',
        ),
        "leap_thumb_drop_j3": FingerConnectivityPreset(
            name="leap_thumb_drop_j3",
            family="leap",
            finger_kind="thumb",
            deleted_joint_suffixes=("j3",),
            regroup_strategy="drop",
            note='LEAP thumb without distal `j3`; retain the tip and connect it to the nearest surviving segment.',
        ),
    }




_FINGER_CONNECTIVITY_ENUMERATION_ORDER: dict[tuple[str, Literal["non_thumb", "thumb"]], tuple[str, ...]] = {
    ("allegro", "non_thumb"): (
        "allegro_non_thumb_full",
        "allegro_non_thumb_drop_j3",
        "allegro_non_thumb_drop_j2_j3",
    ),
    ("allegro", "thumb"): (
        "allegro_thumb_full",
        "allegro_thumb_drop_j3",
    ),
    ("leap", "non_thumb"): (
        "leap_non_thumb_full",
        "leap_non_thumb_drop_j3",
        "leap_non_thumb_drop_j2_j3",
        "leap_non_thumb_drop_j1_j2_j3",
    ),
    ("leap", "thumb"): (
        "leap_thumb_full",
        "leap_thumb_drop_j3",
    ),
}


def _remaining_revolute_count(preset: FingerConnectivityPreset) -> int:

    canonical = _CANONICAL_REVOLUTE_COUNT[(preset.family, preset.finger_kind)]
    return canonical - len(preset.deleted_joint_suffixes)


def _build_hand_connectivity_registry() -> dict[str, HandConnectivityPreset]:

    registry: dict[str, HandConnectivityPreset] = {}

    for family in ("allegro", "leap"):
        non_thumb_names = _FINGER_CONNECTIVITY_ENUMERATION_ORDER[(family, "non_thumb")]
        thumb_names = _FINGER_CONNECTIVITY_ENUMERATION_ORDER[(family, "thumb")]

        for thumb_name, index_name, middle_name, ring_name in product(
            thumb_names,
            non_thumb_names,
            non_thumb_names,
            non_thumb_names,
        ):
            thumb_recipe = FINGER_CONNECTIVITY_PRESET_REGISTRY[thumb_name]
            index_recipe = FINGER_CONNECTIVITY_PRESET_REGISTRY[index_name]
            middle_recipe = FINGER_CONNECTIVITY_PRESET_REGISTRY[middle_name]
            ring_recipe = FINGER_CONNECTIVITY_PRESET_REGISTRY[ring_name]

            thumb_dof = _remaining_revolute_count(thumb_recipe)
            index_dof = _remaining_revolute_count(index_recipe)
            middle_dof = _remaining_revolute_count(middle_recipe)
            ring_dof = _remaining_revolute_count(ring_recipe)

            if (
                thumb_recipe.deleted_joint_suffixes == ()
                and index_recipe.deleted_joint_suffixes == ()
                and middle_recipe.deleted_joint_suffixes == ()
                and ring_recipe.deleted_joint_suffixes == ()
            ):
                name = f"{family}_full"
            else:
                name = f"{family}_t{thumb_dof}_i{index_dof}_m{middle_dof}_r{ring_dof}"

            registry[name] = HandConnectivityPreset(
                name=name,
                family=family,
                finger_slots={
                    "thumb": thumb_name,
                    "index": index_name,
                    "middle": middle_name,
                    "ring": ring_name,
                },
                metadata={
                    "thumb_revolute": thumb_dof,
                    "index_revolute": index_dof,
                    "middle_revolute": middle_dof,
                    "ring_revolute": ring_dof,
                    "thumb_deleted_joint_suffixes": list(thumb_recipe.deleted_joint_suffixes),
                    "index_deleted_joint_suffixes": list(index_recipe.deleted_joint_suffixes),
                    "middle_deleted_joint_suffixes": list(middle_recipe.deleted_joint_suffixes),
                    "ring_deleted_joint_suffixes": list(ring_recipe.deleted_joint_suffixes),
                },
            )

    return registry


FINGER_CONNECTIVITY_PRESET_REGISTRY: dict[str, FingerConnectivityPreset] = _build_finger_connectivity_registry()
"Registry of allowed per-finger topology recipes."

HAND_CONNECTIVITY_PRESET_REGISTRY: dict[str, HandConnectivityPreset] = _build_hand_connectivity_registry()
"Registry of named hand-level connectivity combinations."


def get_finger_connectivity_preset_data(name: str) -> FingerConnectivityPreset:
    'Returns finger connectivity preset data.'

    try:
        return deepcopy(FINGER_CONNECTIVITY_PRESET_REGISTRY[name])
    except KeyError as exc:
        raise KeyError(f"Unknown finger connectivity preset: {name!r}") from exc


def get_hand_connectivity_preset_data(name: str) -> HandConnectivityPreset:
    'Returns hand connectivity preset data.'

    try:
        return deepcopy(HAND_CONNECTIVITY_PRESET_REGISTRY[name])
    except KeyError as exc:
        raise KeyError(f"Unknown hand connectivity preset: {name!r}") from exc


def list_hand_connectivity_preset_names(family: str | None = None) -> tuple[str, ...]:
    'Returns hand connectivity preset names.'

    names = [
        name
        for name, preset in HAND_CONNECTIVITY_PRESET_REGISTRY.items()
        if family is None or preset.family == family
    ]
    return tuple(sorted(names))


def list_finger_connectivity_preset_names(
    *,
    family: str | None = None,
    finger_kind: Literal["non_thumb", "thumb"] | None = None,
) -> tuple[str, ...]:
    'Returns finger connectivity preset names.'

    ordered_names: list[str] = []
    seen: set[str] = set()
    for (current_family, current_kind), names in _FINGER_CONNECTIVITY_ENUMERATION_ORDER.items():
        if family is not None and current_family != family:
            continue
        if finger_kind is not None and current_kind != finger_kind:
            continue
        for name in names:
            if name in seen:
                continue
            ordered_names.append(name)
            seen.add(name)
    return tuple(ordered_names)


def get_default_hand_connectivity_preset_name(family: str) -> str:
    'Returns default hand connectivity preset name.'

    return f"{family}_full"


__all__ = [
    "NON_THUMB_SLOTS",
    "FingerConnectivityPreset",
    "HandConnectivityPreset",
    "FINGER_CONNECTIVITY_PRESET_REGISTRY",
    "HAND_CONNECTIVITY_PRESET_REGISTRY",
    "get_finger_connectivity_preset_data",
    "get_hand_connectivity_preset_data",
    "list_finger_connectivity_preset_names",
    "list_hand_connectivity_preset_names",
    "get_default_hand_connectivity_preset_name",
]
