"""pre-made façade 输入规约。"""

from __future__ import annotations

from typing import Any


_PREMADE_SLOT_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")


def normalize_name_list(values: list[str] | tuple[str, ...] | None, *, field_name: str) -> list[str]:
    r"""把 recipe / YAML 侧的名称列表统一规约为 `list[str]`。"""

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
    r"""把 `connectivity_presets` 统一规约为唯一合法的 slot-level 形状。"""

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


__all__ = ["normalize_connectivity_mapping", "normalize_name_list"]
