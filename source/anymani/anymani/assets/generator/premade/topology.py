"""pre-made topology 规格、registry 与 base hand 构建。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_builders import HandBuilder
from ...presets.hand_presets import get_hand_builder_preset_data, make_human_like_builder_cfg_from_preset
from ...presets.resolver import resolve_finger_builder_cfg

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
    r"""pre-made 阶段的一份显式 topology 规格。

    family composition 与 slot survival 是两个独立物理轴：前者回答存活的
    non-thumb 来自一个还是多个运动学 family，后者回答 canonical palm 上哪些
    finger slot 实际存在。将二者分开后，三指 mixed hand 不再需要新增一个揉合
    两种语义的枚举标签。

    ``topology_kind`` 仅作为历史 sidecar/summary 的派生兼容标签：mixed composition
    始终导出为 ``mixed``，single-family 再按 ``missing_slots`` 区分 ``missing`` 与
    ``single_family``。新代码不得根据该标签反推完整 topology。
    """

    name: str
    anchor_root: str
    family_composition: Literal["single_family", "mixed"]
    missing_slots: tuple[str, ...]
    base_hand_preset: str
    handedness: Literal["left", "right"]
    family: str
    finger_preset_names: dict[str, str]
    surviving_slots: tuple[str, ...]

    def __post_init__(self) -> None:
        r"""验证 composition 与 slot survival 没有产生退化或互相矛盾的 topology。"""

        if len(set(self.missing_slots)) != len(self.missing_slots):
            raise ValueError("premade topology missing_slots must be unique")
        if any(slot not in _PREMADE_NON_THUMB_SLOT_ORDER for slot in self.missing_slots):
            raise ValueError("premade topology may only mark non-thumb slots as missing")
        if set(self.missing_slots) & set(self.surviving_slots):
            raise ValueError("premade topology missing_slots and surviving_slots must be disjoint")
        if tuple(slot for slot in _PREMADE_SLOT_ORDER if slot in self.surviving_slots) != self.surviving_slots:
            raise ValueError("premade topology surviving_slots must follow canonical slot order")

        # Thumb 的 mount 属于 palm canonical 装配；它不参与 mixed 判定，但必须与 palm family 一致。
        slot_family_map = self.slot_family_map()
        if slot_family_map.get("thumb") != self.family:
            raise ValueError("premade topology thumb family must match the base palm family")
        non_thumb_families = {
            slot_family_map[slot] for slot in self.surviving_slots if slot != "thumb"
        }  # 存活 non-thumb 的运动学机制集合，不把同型 thumb 当作 family 证据
        if self.family_composition == "mixed":
            if non_thumb_families != {"allegro", "leap"}:
                raise ValueError("mixed topology requires both Allegro and LEAP surviving non-thumb fingers")
        elif non_thumb_families != {self.family}:
            raise ValueError("single-family topology requires every surviving non-thumb to match the palm family")

    @property
    def topology_kind(self) -> Literal["single_family", "missing", "mixed"]:
        r"""返回旧 sidecar/summary 使用的粗粒度派生标签。"""

        if self.family_composition == "mixed":
            return "mixed"
        return "missing" if self.missing_slots else "single_family"

    def slot_family_map(self) -> dict[str, str]:
        r"""返回每个 surviving slot 当前来自哪个 finger family。"""

        return {
            slot: _finger_family_from_preset_name(preset_name) for slot, preset_name in self.finger_preset_names.items()
        }

    def to_metadata(self) -> dict[str, Any]:
        r"""把 topology 规格转成可挂到 HandCfg / sidecar 的稳定 provenance。"""

        return {
            "topology_registry_key": self.name,
            "base_hand_preset": self.base_hand_preset,
            "handedness": self.handedness,
            "topology_kind": self.topology_kind,
            "family_composition": self.family_composition,
            "missing_slots": list(self.missing_slots),
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
    if requested == "left":
        return ("left",)
    if requested == "right":
        return ("right",)
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
        family_composition="single_family",
        missing_slots=(),
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
                family_composition="single_family",
                missing_slots=(missing_slot,),
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
    r"""从 full 或 missing single-family topology 派生 true mixed 组合。

    `mixed` 的机械语义不是把所有 finger slot 任意换族。thumb mount 由 palm
    family 的 canonical preset 定义，因此 thumb 与 palm 必须共享 family；只有
    index / middle / ring / little 这类 non-thumb slot 参与 LEAP / Allegro
    笛卡尔展开。存活 non-thumb 必须同时含有两个 family；全为 base family 会退化
    为普通 single-family，全为 opposite family 也不因 palm/thumb 标签而冒充 mixed。
    """

    slot_order = base_topology.surviving_slots
    mixed_slot_order = tuple(slot_name for slot_name in slot_order if slot_name != "thumb")
    if not mixed_slot_order:
        return ()

    specs: list[PremadeTopologySpec] = []
    for family_assignment in product(("allegro", "leap"), repeat=len(mixed_slot_order)):
        # thumb family 是 palm mount 的结构边界；non-thumb family assignment 才是 mixed 自由度。
        slot_family_map: dict[str, str] = dict(zip(mixed_slot_order, family_assignment))
        if "thumb" in slot_order:
            slot_family_map["thumb"] = base_topology.family

        # True mixed 必须由存活 non-thumb 本身提供两种运动学 family，thumb 不计入判据。
        if set(family_assignment) != {"allegro", "leap"}:
            continue

        # 恢复 canonical slot 顺序，使 registry key、metadata 与导出目录保持确定性。
        slot_family_map = {slot_name: slot_family_map[slot_name] for slot_name in slot_order}
        finger_preset_names = {
            slot_name: _PREMADE_FINGER_PRESET_BY_FAMILY_AND_KIND[
                (slot_family_map[slot_name], slot_finger_kind(slot_name))
            ]
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
                family_composition="mixed",
                missing_slots=base_topology.missing_slots,
                base_hand_preset=base_topology.base_hand_preset,
                handedness=base_topology.handedness,
                family=base_topology.family,
                finger_preset_names=finger_preset_names,
                surviving_slots=slot_order,
            )
        )
    return tuple(specs)


def build_premade_topology_registry(cfg: Any) -> dict[str, PremadeTopologySpec]:
    r"""构建 full/missing 与 single/mixed 正交组合后的 pre-made topology 规格。"""

    registry: dict[str, PremadeTopologySpec] = {}
    for hand_preset_name in cfg.hand_presets:
        for handedness in requested_handednesses(cfg):
            base_topology = _extract_base_topology_spec(hand_preset_name, handedness=handedness)
            if not _supports_topology_expansion(cfg, base_hand_preset_name=hand_preset_name):
                registry[base_topology.name] = base_topology
                continue

            # Slot survival 先形成 single-family physical templates；mixed family assignment
            # 随后作用于每个 full/missing template，避免两个独立分支漏掉组合空间。
            single_family_specs = [base_topology]
            if getattr(cfg, "missing", True):
                single_family_specs.extend(_build_missing_topology_specs(base_topology))
            for spec in single_family_specs:
                registry[spec.name] = spec
            if getattr(cfg, "mixed", False):
                for single_family_spec in single_family_specs:
                    for mixed_spec in _build_mixed_topology_specs(single_family_spec):
                        registry[mixed_spec.name] = mixed_spec
    return registry


def resolve_premade_topology_spec(cfg: Any, topology_name: str) -> PremadeTopologySpec:
    r"""按名字返回当前 cfg 下的一份 premade topology 规格。"""

    registry = build_premade_topology_registry(cfg)
    try:
        return registry[topology_name]
    except KeyError as exc:
        raise KeyError(f"Unknown premade topology {topology_name!r}") from exc


def extract_premade_topology_metadata(hand_cfg: HandCfg, *, hand_preset_name: str | None) -> dict[str, Any]:
    r"""从 HandCfg metadata 中读取并补齐正交 premade topology provenance。

    2026-08-19 之前的持久化 sidecar 只有粗粒度 ``topology_kind``。这里保留真实
    兼容需要：mixed 直接恢复 family composition；single-family missing 则用 base
    preset 的 canonical slots 与已保存 surviving slots 求集合差。该迁移只补 metadata，
    不修改历史资产文件或改变其 physical identity。
    """

    metadata = dict(hand_cfg.metadata or {})
    topology_metadata = metadata.get("premade_topology")
    if isinstance(topology_metadata, dict):
        normalized = dict(topology_metadata)
        topology_kind = str(normalized.get("topology_kind") or "single_family")
        normalized.setdefault("family_composition", "mixed" if topology_kind == "mixed" else "single_family")
        if "missing_slots" not in normalized:
            missing_slots: tuple[str, ...] = ()
            base_hand_preset = normalized.get("base_hand_preset")
            handedness = normalized.get("handedness")
            surviving_slots = normalized.get("surviving_slots")
            if topology_kind == "missing" and isinstance(base_hand_preset, str) and handedness in {"left", "right"}:
                base_topology = _extract_base_topology_spec(base_hand_preset, handedness=handedness)
                surviving_set = set(surviving_slots) if isinstance(surviving_slots, (tuple, list)) else set()
                missing_slots = tuple(slot for slot in base_topology.surviving_slots if slot not in surviving_set)
            normalized["missing_slots"] = list(missing_slots)
        return normalized
    if hand_preset_name is None:
        raise ValueError("HandCfg is missing premade_topology metadata")
    return {
        "topology_registry_key": hand_preset_name,
        "base_hand_preset": hand_preset_name,
        "handedness": hand_cfg.handedness,
        "topology_kind": "single_family",
        "family_composition": "single_family",
        "missing_slots": [],
        "topology_anchor": hand_preset_name,
        "topology_name": hand_preset_name,
        "surviving_slots": [
            slot_name for slot_name in _PREMADE_SLOT_ORDER if slot_name in {"thumb", "index", "middle", "ring"}
        ],
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
        raise ValueError(
            "HandGenerator requires a concrete Made cfg or at least one hand preset when using the pre-made facade"
        )

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
