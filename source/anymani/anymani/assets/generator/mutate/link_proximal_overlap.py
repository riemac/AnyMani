r"""Joint child-link 近端几何重叠后变异算子。

该算子表达的是 primitive skin 相对 joint frame 的 proximal overhang，而不是
kinematic link length。它允许 LEAP 风格的零重叠几何向 parent 内部延伸，也允许
Allegro 风格的既有 overhang 被缩减，但始终保持 child distal boundary 与下游
joint frame 不变。

对 eligible child $i$，定义：

$$
o_i^0=\max(0,-d_i),
\qquad
o_i'=\operatorname{clip}
\left(o_i^0+\eta_i s_i^\star,\ 0,\ \kappa E_{i-1}^0\right).
$$

其中 $d_i$ 是变异前 primitive proximal boundary，$s_i^\star$ 是可选
`link_scale` 后的 child effective span，$E_{i-1}^0$ 是变异前 parent 净 span。
signed ratio $\eta_i$ 的正负分别表示增加和缩减 overhang；下界 0 禁止制造 gap，
上界 $\kappa E_{i-1}^0$ 禁止 child 侵入超过 parent 的可配置比例。
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import MISSING, dataclass, field
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import Vector2
from .axial_geometry import (
    AxialGeometryEdit,
    ProximalOverlapContribution,
    joint_cross_section,
    joint_primary_length,
    make_axial_geometry_patch_op,
    validate_axial_geometry,
)
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler

_MODE_IDENTITY = "identity"
_MODE_DISTURB = "disturb"
_MODE_HOMOLOGOUS_NON_THUMB = "homologous_non_thumb"
_ALL_SELF_MODES = (_MODE_IDENTITY, _MODE_DISTURB, _MODE_HOMOLOGOUS_NON_THUMB)
_MODE_TOLERANCE = 1e-9
_GEOMETRY_TOLERANCE = 1e-9


@dataclass
class LinkProximalOverlapCfg(MutatorBaseCfg):
    r"""关节 child-link 近端几何重叠配置。

    作用域是每根 finger 实际 surviving chain 中首 active revolute child 之后的
   活动 child。palm-side fixed root、首 active child、fixed tip 与非轴向 geometry
    都不参与；unsupported eligible geometry 直接 fail closed。
    """

    class_type: type[LinkProximalOverlapMutator] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    r"""运行时 mutator 类型绑定，不属于科研采样接口。"""

    overhang_delta_ratio: Vector2 = field(default=MISSING)
    r"""signed additive ratio 区间 $(\eta_{min},\eta_{max})$。

    例如 `(-0.1,0.2)` 表示按缩放后 child span 的 $10\%$ 缩减已有
    overhang，或按 $20\%$ 增加近端侵入。该字段不是对已有 overhang 的
    乘法系数，因此基础 $o_i^0=0$ 的 LEAP link 仍可产生正重叠。
    """

    self_mode: Literal[
        "identity",
        "disturb",
        "homologous_non_thumb",
    ] | dict[str, float] | None = _MODE_DISTURB
    r"""整手级结构模式。

    - `identity`：显式 no-op，只记录 provenance；
    - `disturb`：每个 eligible joint 独立采样 $\eta_i$；
    - `homologous_non_thumb`：non-thumb 按 `(family, semantic suffix)` 共享
      $\eta$，thumb 仍逐 joint 独立；
    - `dict[str,float]`：先按概率选择一个整手级 mode，概率和必须为 1。

    这里不再引入 per-owner Bernoulli。零作用由 signed ratio 经 $o_i'\ge0$
    截断自然产生；整手精确 no-op 由 `identity` 表达。
    """

    max_parent_overlap_ratio: float = 0.5
    r"""最终 overhang 相对 parent 变异前净 span 的上限 $\kappa$。

    物理合同固定为 $0<\kappa\le0.5$，即 child 近端不得侵入超过 parent
    净 span 的一半；配置可以在这个硬上限内收紧。
    """

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    r"""$\eta$ 的基础采样分布。"""

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""ratio 样本越过声明区间时的边界解释。"""

    _active_modes: tuple[str, ...] = field(init=False, default=(), repr=False)
    r"""当前配置中具有正概率的 mode 集合。"""

    def __post_init__(self) -> None:
        r"""绑定 runtime，并在资产生成前拒绝非法概率与几何 cap。"""

        self.class_type = LinkProximalOverlapMutator
        self._active_modes = _resolve_active_modes(self.self_mode)
        if len(self.overhang_delta_ratio) != 2:
            raise ValueError("overhang_delta_ratio must contain exactly two values")
        if not all(math.isfinite(float(value)) for value in self.overhang_delta_ratio):
            raise ValueError("overhang_delta_ratio must contain finite values")
        self.max_parent_overlap_ratio = float(self.max_parent_overlap_ratio)
        if not 0.0 < self.max_parent_overlap_ratio <= 0.5:
            raise ValueError("max_parent_overlap_ratio must satisfy 0 < ratio <= 0.5")


class LinkProximalOverlapMutator(MutatorBase):
    r"""把 signed overhang proposal lowering 成可与 `link_scale` 合成的 patch。"""

    cfg: LinkProximalOverlapCfg

    def __init__(self, cfg: LinkProximalOverlapCfg) -> None:
        r"""绑定一份已经通过概率与 cap 校验的配置。"""

        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""返回单个结构化 sample sampler，避免暴露未消费的备用随机量。"""

        return {"sample": lambda: self._sample_one(target)}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""基于 mother `HandCfg` 规划近端 overhang edit，不立即修改几何。"""

        sample = _normalize_sample_payload(sampled_params, cfg=self.cfg, target=target)
        resolved_mode = str(sample["resolved_self_mode"])
        patch = HandPatch()
        patch.metadata.setdefault("post_mutate_samples", {})
        patch.metadata["post_mutate_samples"]["link_proximal_overlap"] = sample
        patch.metadata["post_mutate_link_proximal_overlap"] = sample
        if resolved_mode == _MODE_IDENTITY:
            return patch

        eligible = list(_iter_eligible_joints(target))
        eligible_names = {joint.name for _, _, joint, _ in eligible}
        joint_delta_ratio = {
            str(joint_name): float(value)
            for joint_name, value in dict(sample.get("joint_delta_ratio", {})).items()
        }
        unknown = sorted(set(joint_delta_ratio) - eligible_names)
        if unknown:
            raise ValueError(f"proximal overlap sample targets ineligible joints: {unknown}")

        for finger_index, joint_index, joint, next_joint in eligible:
            if joint.name not in joint_delta_ratio:
                continue  # 手工 deterministic sample 可只覆盖一个 owner；正式 sampler 会完整生成
            ratio = joint_delta_ratio[joint.name]
            if not math.isfinite(ratio):
                raise ValueError(f"proximal overlap ratio must be finite for {joint.name!r}")

            validate_axial_geometry(joint)
            source_length = joint_primary_length(joint)
            if source_length is None:
                raise ValueError(f"proximal overlap cannot infer axial length for {joint.name!r}")
            primary = joint.collisions[0] if joint.collisions else joint.visuals[0]
            proximal_boundary = float(primary.origin.pos[1]) - source_length / 2.0
            distal_boundary = float(primary.origin.pos[1]) + source_length / 2.0
            if proximal_boundary > _GEOMETRY_TOLERANCE:
                raise ValueError(
                    f"proximal overlap requires touching/overlapping source geometry, got gap "
                    f"{proximal_boundary!r} for {joint.name!r}"
                )

            source_child_span = float(next_joint.origin.pos[1])
            if not math.isclose(
                distal_boundary,
                source_child_span,
                rel_tol=0.0,
                abs_tol=_GEOMETRY_TOLERANCE,
            ):
                raise ValueError(
                    f"child distal geometry/joint span mismatch for {joint.name!r}: "
                    f"geometry={distal_boundary!r}, joint={source_child_span!r}"
                )
            parent_span_before = float(joint.origin.pos[1])

            overlap = ProximalOverlapContribution(
                joint_name=joint.name,
                child_link=str(joint.child),
                delta_ratio=ratio,
                base_overhang=max(0.0, -proximal_boundary),
                source_child_span=source_child_span,
                parent_span_before=parent_span_before,
                max_parent_overlap_ratio=self.cfg.max_parent_overlap_ratio,
            )
            patch.add_op(
                make_axial_geometry_patch_op(
                    AxialGeometryEdit(
                        finger_index=finger_index,
                        joint_index=joint_index,
                        joint_name=joint.name,
                        child_link=str(joint.child),
                        source_length=source_length,
                        source_cross_section=joint_cross_section(joint),
                        keep_center=False,
                        overlap=overlap,
                    )
                )
            )
        return patch

    def _sample_one(self, target: HandCfg) -> dict[str, Any]:
        r"""先解析整手级 mode，再只采样该 mode 真正消费的 ratio。"""

        return self.sample_one_for_mode(target, resolved_mode=_draw_resolved_mode(self.cfg))

    def sample_one_for_mode(self, target: HandCfg, *, resolved_mode: str) -> dict[str, Any]:
        r"""为测试、accepted-mode 重采和正式 proposal 生成结构化 sample。"""

        if resolved_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported link_proximal_overlap mode: {resolved_mode!r}")
        if resolved_mode == _MODE_IDENTITY:
            return {"resolved_self_mode": _MODE_IDENTITY, "joint_delta_ratio": {}}

        sampler = _make_range_sampler(
            self.cfg.overhang_delta_ratio,
            distrib=self.cfg.distrib,
            boundary_policy=self.cfg.boundary_policy,
        )
        eligible = list(_iter_eligible_joints(target))
        if resolved_mode == _MODE_DISTURB:
            return {
                "resolved_self_mode": _MODE_DISTURB,
                "joint_delta_ratio": {joint.name: float(sampler()) for _, _, joint, _ in eligible},
            }

        groups = _resolve_homologous_non_thumb_groups(target, eligible=eligible)
        joint_delta_ratio: dict[str, float] = {}
        homologous_groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for group_key, joint_names in groups.items():
            ratio = float(sampler())
            homologous_groups[group_key] = {"joint_names": list(joint_names), "delta_ratio": ratio}
            for joint_name in joint_names:
                joint_delta_ratio[joint_name] = ratio

        thumb_joint_delta_ratio: OrderedDict[str, float] = OrderedDict()
        for _, _, joint, _ in eligible:
            if not joint.name.startswith("thumb_"):
                continue
            ratio = float(sampler())
            thumb_joint_delta_ratio[joint.name] = ratio
            joint_delta_ratio[joint.name] = ratio
        return {
            "resolved_self_mode": _MODE_HOMOLOGOUS_NON_THUMB,
            "joint_delta_ratio": joint_delta_ratio,
            "homologous_groups": dict(homologous_groups),
            "thumb_joint_delta_ratio": dict(thumb_joint_delta_ratio),
        }


def _iter_eligible_joints(hand: HandCfg):
    r"""遍历每根实际 surviving chain 中首 active child 之后的 revolute joints。"""

    for finger_index, finger in enumerate(hand.fingers):
        active_indices = [
            joint_index
            for joint_index, joint in enumerate(finger.joints)
            if joint.joint_type == "revolute" and not joint.is_tip
        ]
        for joint_index in active_indices[1:]:
            next_index = joint_index + 1
            if next_index >= len(finger.joints):
                raise ValueError(f"eligible joint {finger.joints[joint_index].name!r} has no distal boundary joint")
            yield finger_index, joint_index, finger.joints[joint_index], finger.joints[next_index]


def _normalize_sample_payload(
    sampled_params: dict[str, Any] | None,
    *,
    cfg: LinkProximalOverlapCfg,
    target: HandCfg,
) -> dict[str, Any]:
    r"""把结构化 sample 或手工 joint->ratio payload 规约成统一形状。"""

    params = dict(sampled_params or {})
    raw_sample = params.get("sample")
    if isinstance(raw_sample, dict):
        sample = dict(raw_sample)
        sample.setdefault("joint_delta_ratio", {})
        return sample
    if cfg.self_mode == _MODE_IDENTITY:
        return {"resolved_self_mode": _MODE_IDENTITY, "joint_delta_ratio": {}}
    eligible_names = {joint.name for _, _, joint, _ in _iter_eligible_joints(target)}
    return {
        "resolved_self_mode": _MODE_DISTURB,
        "joint_delta_ratio": {
            str(name): float(value)
            for name, value in params.items()
            if str(name) in eligible_names
        },
    }


def _resolve_active_modes(self_mode: str | dict[str, float] | None) -> tuple[str, ...]:
    r"""验证高层 mode 或 mode mixture，并返回所有正概率项。"""

    if self_mode is None:
        return (_MODE_DISTURB,)
    if isinstance(self_mode, str):
        if self_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported link_proximal_overlap self_mode: {self_mode!r}")
        return (self_mode,)
    if not isinstance(self_mode, dict):
        raise TypeError("link_proximal_overlap.self_mode must be str | dict[str,float] | None")

    active: list[str] = []
    total = 0.0
    for mode_name, probability in self_mode.items():
        if mode_name not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported link_proximal_overlap self_mode key: {mode_name!r}")
        probability = float(probability)
        if probability < 0.0:
            raise ValueError(f"self_mode probability must be non-negative, got {mode_name!r}={probability!r}")
        total += probability
        if probability > _MODE_TOLERANCE:
            active.append(mode_name)
    if not active:
        raise ValueError("self_mode dict must contain at least one positive-probability mode")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"self_mode probabilities must sum to 1.0, got {total!r}")
    return tuple(active)


def _draw_resolved_mode(cfg: LinkProximalOverlapCfg) -> str:
    r"""按配置概率为当前 candidate 选择唯一整手级 mode。"""

    if cfg.self_mode is None:
        return _MODE_DISTURB
    if isinstance(cfg.self_mode, str):
        return cfg.self_mode
    threshold = random.random()
    cumulative = 0.0
    last_mode = _MODE_DISTURB
    for mode_name, probability in cfg.self_mode.items():
        probability = float(probability)
        if probability <= _MODE_TOLERANCE:
            continue
        cumulative += probability
        last_mode = mode_name
        if threshold <= cumulative + _MODE_TOLERANCE:
            return mode_name
    return last_mode


def _resolve_homologous_non_thumb_groups(
    target: HandCfg,
    *,
    eligible: list[tuple[int, int, Any, Any]],
) -> OrderedDict[str, list[str]]:
    r"""按 `(finger family, semantic child suffix)` 解析 non-thumb 同源组。"""

    slot_family_map = _resolve_slot_family_map(target)
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for finger_index, _, joint, _ in eligible:
        finger = target.fingers[finger_index]
        if finger.name == "thumb":
            continue
        family = slot_family_map.get(finger.name)
        if not family:
            raise ValueError(f"missing slot family for finger {finger.name!r}")
        prefix = f"{finger.name}_"
        child_link = str(joint.child)
        if not child_link.startswith(prefix) or len(child_link) == len(prefix):
            raise ValueError(f"cannot resolve semantic suffix from child link {child_link!r}")
        semantic_suffix = child_link[len(prefix) :]
        groups.setdefault(f"{family}:{semantic_suffix}", []).append(joint.name)
    return groups


def _resolve_slot_family_map(target: HandCfg) -> dict[str, str]:
    r"""读取 pre-made slot-family provenance；单 family builder 可用 hand family 回退。"""

    metadata = dict(target.metadata or {})
    for metadata_key in ("premade_topology", "premade_connectivity"):
        payload = metadata.get(metadata_key)
        if isinstance(payload, dict) and isinstance(payload.get("slot_family_map"), dict):
            return {str(slot): str(family) for slot, family in payload["slot_family_map"].items()}
    if target.family in {"allegro", "leap"}:
        return {finger.name: target.family for finger in target.fingers}
    raise ValueError("homologous_non_thumb requires premade slot_family_map for mixed/generic hands")


__all__ = ["LinkProximalOverlapCfg", "LinkProximalOverlapMutator"]
