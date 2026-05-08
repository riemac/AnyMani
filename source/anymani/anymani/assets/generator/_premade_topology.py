"""pre-made topology 规格、registry 与 base hand 构建。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

from ..asset_base import HandCfg
from ..asset_builders import HandBuilder
from ..presets.hand_presets import get_hand_builder_preset_data, make_human_like_builder_cfg_from_preset
from ..presets.resolver import resolve_finger_builder_cfg


_PREMADE_SLOT_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")
_PREMADE_NON_THUMB_SLOT_ORDER: tuple[str, ...] = tuple(slot for slot in _PREMADE_SLOT_ORDER if slot != "thumb")
_PREMADE_FINGER_PRESET_BY_FAMILY_AND_KIND: dict[tuple[str, Literal["thumb", "non_thumb"]], str] = {
    ("allegro", "thumb"): "allegro_thumb_v1",
    ("allegro", "non_thumb"): "allegro_non_thumb_v1",
    ("leap", "thumb"): "leap_thumb_v1",
    ("leap", "non_thumb"): "leap_non_thumb_v1",
}


@dataclass(frozen=True)
class PremadeTopologySpec:
    r"""pre-made 阶段的一份显式 topology 规格。"""

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


def _finger_family_from_preset_name(preset_name: str) -> str:
    r"""从 finger preset 名里读取 family 前缀。"""

    for family in ("allegro", "leap"):
        if preset_name.startswith(f"{family}_"):
            return family
    raise ValueError(f"Cannot infer finger family from preset name {preset_name!r}")


def slot_finger_kind(slot_name: str) -> Literal["thumb", "non_thumb"]:
    r"""把 slot 名 lower 成 thumb/non-thumb 类型。"""

    return "thumb" if slot_name == "thumb" else "non_thumb"


def requested_handednesses(cfg: Any) -> tuple[Literal["left", "right"], ...]:
    r"""把 `HandGeneratorCfg.handedness` lower 成当前 pre-made 要展开的 handedness 集合。"""

    requested = str(getattr(cfg, "handedness", "all"))
    if requested == "all":
        return ("left", "right")
    if requested in {"left", "right"}:
        return (requested,)
    raise ValueError(f"Unsupported handedness request {requested!r}; expected 'left' / 'right' / 'all'.")


def build_topology_registry_key(
    *,
    base_hand_preset: str,
    handedness: Literal["left", "right"],
    suffix_tokens: tuple[str, ...] = (),
) -> str:
    r"""为内部 topology registry 生成稳定 key。"""

    parts = [base_hand_preset, handedness]
    parts.extend(str(token) for token in suffix_tokens)
    return "__".join(parts)


def _supports_topology_expansion(cfg: Any, *, base_hand_preset_name: str) -> bool:
    r"""判断当前 base hand 是否应展开 missing / mixed topology。"""

    _ = base_hand_preset_name
    return cfg.Made.class_type is HandBuilder


def _extract_non_thumb_slots_from_hand_preset(hand_preset_data: dict[str, Any]) -> tuple[str, ...]:
    r"""从 hand preset 原始字典里恢复当前 canonical non-thumb slot 集合。"""

    finger_cfg = hand_preset_data.get("finger_cfg")
    if isinstance(finger_cfg, dict):
        return tuple(slot_name for slot_name in _PREMADE_NON_THUMB_SLOT_ORDER if slot_name in finger_cfg)

    num_non_thumb = int(hand_preset_data.get("num_non_thumb", 3))
    return _PREMADE_NON_THUMB_SLOT_ORDER[:num_non_thumb]


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
        name=build_topology_registry_key(base_hand_preset=hand_preset_name, handedness=handedness),
        anchor_root=hand_preset_name,
        topology_kind="single_family",
        base_hand_preset=hand_preset_name,
        handedness=handedness,
        family=str(hand_preset_data["family"]),
        finger_preset_names=finger_preset_names,
        surviving_slots=surviving_slots,
    )


def _build_missing_topology_specs(base_topology: PremadeTopologySpec) -> tuple[PremadeTopologySpec, ...]:
    r"""从 canonical single-family topology 派生“缺失一根 non-thumb”的 pre-made 规格。"""

    if "thumb" not in base_topology.surviving_slots:
        return ()

    non_thumb_slots = [slot_name for slot_name in base_topology.surviving_slots if slot_name != "thumb"]
    if len(non_thumb_slots) < 3:
        return ()

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
                name=build_topology_registry_key(
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


def _build_mixed_topology_specs(base_topology: PremadeTopologySpec) -> tuple[PremadeTopologySpec, ...]:
    r"""从 canonical topology 派生 mixed-family finger 组合。"""

    slot_order = base_topology.surviving_slots
    if not slot_order:
        return ()

    specs: list[PremadeTopologySpec] = []
    for family_assignment in product(("allegro", "leap"), repeat=len(slot_order)):
        slot_family_map = dict(zip(slot_order, family_assignment))
        if all(current_family == base_topology.family for current_family in slot_family_map.values()):
            continue

        finger_preset_names = {
            slot_name: _PREMADE_FINGER_PRESET_BY_FAMILY_AND_KIND[(slot_family_map[slot_name], slot_finger_kind(slot_name))]
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


def build_premade_topology_registry(cfg: Any) -> dict[str, PremadeTopologySpec]:
    r"""构建当前 generator cfg 可见的所有 pre-made topology 规格。"""

    registry: dict[str, PremadeTopologySpec] = {}
    for hand_preset_name in cfg.hand_presets:
        for handedness in requested_handednesses(cfg):
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


def resolve_premade_topology_spec(cfg: Any, topology_name: str) -> PremadeTopologySpec:
    r"""按名字返回当前 cfg 下的一份 premade topology 规格。"""

    registry = build_premade_topology_registry(cfg)
    try:
        return registry[topology_name]
    except KeyError as exc:
        raise KeyError(f"Unknown premade topology {topology_name!r}") from exc


def extract_premade_topology_metadata(hand_cfg: HandCfg, *, hand_preset_name: str | None) -> dict[str, Any]:
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


def make_builder_cfg_from_topology(topology: PremadeTopologySpec):
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


def candidate_hand_preset_names(cfg: Any) -> tuple[str, ...]:
    r"""返回当前 generator cfg 可见的 pre-made topology registry key 集合。"""

    return tuple(build_premade_topology_registry(cfg))


def build_base_hand(cfg: Any, *, hand_preset_name: str | None) -> tuple[HandCfg, str]:
    r"""构建本次样本的 canonical base hand。"""

    if cfg.Made.class_type is not HandBuilder:
        builder_cfg = cfg.Made
    elif hand_preset_name is not None:
        topology = resolve_premade_topology_spec(cfg, hand_preset_name)
        builder_cfg = make_builder_cfg_from_topology(topology)
    else:
        raise ValueError("HandGenerator requires a concrete Made cfg or at least one hand preset when using the pre-made facade")

    builder = builder_cfg.class_type(builder_cfg)
    hand_cfg = builder.build()
    if hand_preset_name is not None and cfg.Made.class_type is HandBuilder:
        topology_metadata = resolve_premade_topology_spec(cfg, hand_preset_name).to_metadata()
        hand_metadata = dict(hand_cfg.metadata)
        hand_metadata["premade_topology"] = topology_metadata
        hand_cfg = hand_cfg.replace(metadata=hand_metadata)
    return hand_cfg, builder_cfg.__class__.__name__


__all__ = [
    "PremadeTopologySpec",
    "build_base_hand",
    "build_premade_topology_registry",
    "build_topology_registry_key",
    "candidate_hand_preset_names",
    "extract_premade_topology_metadata",
    "make_builder_cfg_from_topology",
    "requested_handednesses",
    "resolve_premade_topology_spec",
    "slot_finger_kind",
]
