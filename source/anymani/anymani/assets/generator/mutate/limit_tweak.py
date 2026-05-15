r"""关节限位微调算子：在已有 `HandCfg` 上对 joint limit 做小范围参数修改。

科研语义：
`limit_tweak` 改的是活动关节的合法角域 $[q_{\min}, q_{\max}]$，不是几何
链长，也不是 joint axis。为了保持 post-mutate 不改变 topology，本算子
只生成 limit 字段 patch，并保持 `lower < upper`。

这意味着它在物理上属于“关节运动学约束”的局部再标定，而不是手型
重建；因此它和 `link_scale`、`mount_perturb` 在建模层面是正交的。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
import random
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import JointLimitCfg, Vector2
from ._base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler


_MODE_IDENTITY = "identity"
_MODE_DISTURB = "disturb"
_MODE_HOMOLOGOUS_NON_THUMB = "homologous_non_thumb"

_ALL_SELF_MODES = (
    _MODE_IDENTITY,
    _MODE_DISTURB,
    _MODE_HOMOLOGOUS_NON_THUMB,
)
"""`limit_tweak.self_mode` 当前支持的全部高层 mode。"""

_MODE_TOLERANCE = 1e-9
"""mode 概率求和与正概率判定的数值容差。"""


@dataclass
class LimitTweakCfg(MutatorBaseCfg):
    r"""关节限位微调工具配置。

    这份配置在科研上对应三层语义：

    1. 关节极限角域的合法范围本身要被扰动；
    2. 扰动后的 limit 仍要保持物理合理性，尤其不能翻转成 lower >= upper；
    3. 扰动可以按高层 mode 切换成不同的结构先验，而不只是“每个 joint 各抖各的”。
    """

    class_type: type["LimitTweakMutator"] | None = field(init=False, default=None, repr=False)
    r"""关联的运行时类。"""

    disturb_object: Literal["independent", "shared"] = "independent"
    r"""lower / upper 的耦合方式。

    所有关节限位 $[q_{min}, q_{max}]$ 都有 lower 与 upper 两个边界：

    - `independent`：对 lower 与 upper 分别独立采样；
    - `shared`：lower / upper 共享同一个扰动值，表示整段角域整体平移或整体缩放。
    """

    self_mode: Literal[
        "identity",
        "disturb",
        "homologous_non_thumb",
    ] | dict[str, float] | None = _MODE_DISTURB
    r"""关节限位扰动的高层 mode 选择器。

    支持三种输入语义：

    - `None`：未显式指定，默认落到 `disturb`
    - `str`：固定使用某一个 mode
    - `dict[str, float]`：按概率混合采样一个 mode；概率和必须严格为 1

    预设 mode：

    - `identity`
      显式 no-op：不改任何 limit，只记录 provenance，供 accepted/output 锚点样本保留权重。
    - `disturb`
      当前最通用的 mode：每个 joint 各自按 `disturb_object` 采样 limit 增量。
    - `homologous_non_thumb`
      非拇指 joint 按 `(finger_family, joint_semantic)` 分组共享扰动；thumb 每个 joint 独立。
      例如 mixed topology 中 `allegro:mcp1` 与 `leap:mcp1` 绝不共享，因为它们虽然语义名同为
      `mcp1`，但 family 不同，机械先验并不等价。
    """

    disturb_type: Literal["add", "scale"] = "add"
    r"""扰动类型。默认为添加。

    - `add`：在原有值基础上添加增量
    - `scale`：在原有值基础上按比例缩放
    """

    joint_range: Vector2 | None = None
    r"""关节限位微调范围配置，同时适用于所有活动关节。

    表示在原有 `HandCfg` 的 joint limit 基础上进行微调的数值范围；
    `None` 表示没有可供采样的扰动域。

    - 当 `disturb_type="add"` 时，区间一律按 **radian** 解释；若研究者更习惯用 degree，
      应在配置侧显式写 `deg(...)` 后再传入
    - 当 `disturb_type="scale"` 时，可表示为 `(-0.1, 0.1)`，表示在原有基础上做约 $\pm10\%$
      的相对缩放；最终写回时使用 $q' = q (1 + \delta)$
    """

    clip: dict[str, float] | None = None
    r"""裁剪范围。默认不裁剪。

    - `{"abs": 0.17}`：表示微调后关节限位的绝对增量不超过约 $0.17\text{rad}$；
    - `{"rel": 0.2}`：当前 v1 仍把它解释成“对采样增量本身做 $\pm0.2$ 裁剪”，
      不是相对原始 limit 的精确百分比约束。
    """

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    r"""分布类型。可选正态分布/均匀分布。

    适用于所选中的全部关节对象，但这不表示不同关节每次采样相同；
    每个独立随机变量仍然是独立的，只是共享同一套高层分布假设。
    """

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""关节限位扰动的边界处理策略。

    该字段只规定采样结果超出 `joint_range` 时如何处理，
    不改变扰动类型 `disturb_type`，也不改变基础分布 `distrib`。
    """

    _active_modes: tuple[str, ...] = field(init=False, default=(), repr=False)
    r"""当前 cfg 真正会被采样到的 mode 集合；dict 输入时只保留正概率项。"""

    def __post_init__(self) -> None:
        r"""校验 mode 契约，并补齐运行时类。

        这里选择 fail-fast：一旦某个非 `identity` mode 被启用，却没有 `joint_range`，
        就立刻报错，而不是静默退回 no-op。科研上“不知道是否在采样”比直接失败更危险。
        """

        self.class_type = LimitTweakMutator
        self._active_modes = _resolve_active_modes(self.self_mode)
        if any(mode != _MODE_IDENTITY for mode in self._active_modes) and self.joint_range is None:
            raise ValueError("limit_tweak requires joint_range when any non-identity self_mode is active")


class LimitTweakMutator(MutatorBase):
    r"""把结构化 limit 扰动 lowering 成一次性写回 `joint.limit` 的 patch。"""

    cfg: LimitTweakCfg

    def __init__(self, cfg: LimitTweakCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""返回一个结构化样本生成器。

        与 `mount_perturb` 一样，这里不再把 `limit_tweak` lower 成一堆裸 sampler 键值对。
        原因是高层 mode 一旦引入，同一个 accepted 样本真正消费的随机量集合会随 mode 改变；
        若继续把所有潜在随机量平铺到 sidecar，就会把未消费的备用随机量也暴露出去，
        破坏科研可读性。
        """

        if self.cfg.self_mode == _MODE_IDENTITY and self.cfg.joint_range is None:
            return {"sample": lambda: {"resolved_self_mode": _MODE_IDENTITY}}
        return {"sample": lambda: self._sample_one(target)}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        r"""基于结构化样本 payload 生成 limit patch。

        这里保留一个重要兼容分支：若测试或脚本仍手工传入旧的扁平 `index_j0::lower`
        风格 sampled params，就把它规范化成 `disturb` 模式的结构化样本再统一处理。
        """

        sample = _normalize_sample_payload(sampled_params, self.cfg, target=target)
        resolved_mode = str(sample["resolved_self_mode"])

        patch = HandPatch()
        patch.metadata.setdefault("post_mutate_samples", {})
        patch.metadata["post_mutate_samples"]["limit_tweak"] = sample
        patch.metadata["post_mutate_limit_tweak"] = sample

        if resolved_mode == _MODE_IDENTITY:
            return patch  # `identity` 是显式 no-op mode：不改任何 limit，只保留 provenance

        joint_deltas = dict(sample.get("joint_deltas", {}))
        for finger_index, joint_index, joint in _iter_target_joints(target, None):
            payload = dict(joint_deltas.get(joint.name, {}))
            lower_delta = _clip_delta(float(payload.get("lower", 0.0)), self.cfg.clip)
            upper_delta = _clip_delta(float(payload.get("upper", 0.0)), self.cfg.clip)

            def apply_limit(
                hand: HandCfg,
                *,
                fi=finger_index,
                ji=joint_index,
                dl=lower_delta,
                du=upper_delta,
            ) -> None:
                current = hand.fingers[fi].joints[ji].limit
                if current is None:
                    return

                # `scale` 沿用当前 contract：若上下界共享同一缩放因子，就表示整段角域成比例扩缩；
                # 若 `disturb_object="independent"`，则 lower / upper 各自持有自己的缩放增量。
                if self.cfg.disturb_type == "scale":
                    lower = current.lower * (1.0 + dl)
                    upper = current.upper * (1.0 + (dl if self.cfg.disturb_object == "shared" else du))
                elif self.cfg.disturb_object == "shared":
                    lower = current.lower + dl
                    upper = current.upper + dl
                else:
                    lower = current.lower + dl
                    upper = current.upper + du

                # 物理底线：不允许 lower/upper 翻转；一旦翻转，就收缩到极小合法区间。
                if lower >= upper:
                    center = 0.5 * (lower + upper)
                    lower, upper = center - 1e-4, center + 1e-4

                hand.fingers[fi].joints[ji].limit = JointLimitCfg(
                    lower=lower,
                    upper=upper,
                    effort=current.effort,
                    velocity=current.velocity,
                )

            patch.add(("finger", finger_index, "joint", joint_index, "limit"), apply_limit)
        return patch

    def _sample_one(self, target: HandCfg) -> dict[str, Any]:
        r"""为当前 hand 样本生成一份已经解析好 mode 的结构化随机量。"""

        resolved_mode = _draw_resolved_mode(self.cfg)
        return self.sample_one_for_mode(target, resolved_mode=resolved_mode)

    def sample_one_for_mode(self, target: HandCfg, *, resolved_mode: str) -> dict[str, Any]:
        r"""为 accepted-quota 路径生成指定 mode 的结构化随机量。

        generator 层把 `self_mode=dict` 解释成 accepted/output 分布时，会在 term 内部
        强制重采指定 mode。若只改 mode 名而不重新生成 mode 专属 payload，就会出现
        “forced homologous_non_thumb 但 sample 里没有 group_deltas”这类伪样本。
        """

        if resolved_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported limit_tweak resolved mode: {resolved_mode!r}")
        if resolved_mode == _MODE_IDENTITY:
            return {"resolved_self_mode": _MODE_IDENTITY}

        sampler = _make_range_sampler(
            self.cfg.joint_range,
            distrib=self.cfg.distrib,
            boundary_policy=self.cfg.boundary_policy,
        )

        if resolved_mode == _MODE_DISTURB:
            joint_deltas = {
                joint.name: _sample_delta_pair(sampler, disturb_object=self.cfg.disturb_object)
                for _, _, joint in _iter_target_joints(target, None)
            }
            return {
                "resolved_self_mode": _MODE_DISTURB,
                "joint_deltas": joint_deltas,
            }

        groups = _resolve_homologous_non_thumb_groups(target)
        joint_deltas: dict[str, dict[str, float]] = {}
        homologous_groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for group_key, joint_names in groups.items():
            sampled = _sample_delta_pair(sampler, disturb_object=self.cfg.disturb_object)
            homologous_groups[group_key] = {
                "joint_names": list(joint_names),
                "lower": sampled["lower"],
                "upper": sampled["upper"],
            }
            for joint_name in joint_names:
                joint_deltas[joint_name] = dict(sampled)

        # thumb 不属于 non-thumb 同源组；它在这个 mode 下仍保持每个 joint 独立。
        thumb_joint_deltas: OrderedDict[str, dict[str, float]] = OrderedDict()
        for _, _, joint in _iter_target_joints(target, None):
            if not joint.name.startswith("thumb_"):
                continue
            sampled = _sample_delta_pair(sampler, disturb_object=self.cfg.disturb_object)
            thumb_joint_deltas[joint.name] = sampled
            joint_deltas[joint.name] = dict(sampled)

        return {
            "resolved_self_mode": _MODE_HOMOLOGOUS_NON_THUMB,
            "joint_deltas": joint_deltas,
            "homologous_groups": dict(homologous_groups),
            "thumb_joint_deltas": dict(thumb_joint_deltas),
        }


def _resolve_active_modes(self_mode: str | dict[str, float] | None) -> tuple[str, ...]:
    r"""把 `self_mode` lowering 成真正会被采样到的 mode 集合。"""

    if self_mode is None:
        return (_MODE_DISTURB,)
    if isinstance(self_mode, str):
        if self_mode not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported limit_tweak self_mode: {self_mode!r}")
        return (self_mode,)

    positive_modes: list[str] = []
    total = 0.0
    for mode_name, probability in self_mode.items():
        if mode_name not in _ALL_SELF_MODES:
            raise ValueError(f"unsupported limit_tweak self_mode key: {mode_name!r}")
        prob = float(probability)
        if prob < 0.0:
            raise ValueError(f"limit_tweak.self_mode probability must be non-negative, got {mode_name!r}={prob!r}")
        total += prob
        if prob > _MODE_TOLERANCE:
            positive_modes.append(mode_name)

    if not positive_modes:
        raise ValueError("limit_tweak.self_mode dict must contain at least one positive-probability mode")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_MODE_TOLERANCE):
        raise ValueError(f"limit_tweak.self_mode probabilities must sum to 1, got {total!r}")
    return tuple(positive_modes)


def _draw_resolved_mode(cfg: LimitTweakCfg) -> str:
    r"""按 `self_mode` 为当前样本解析最终 mode。"""

    if cfg.self_mode is None:
        return _MODE_DISTURB
    if isinstance(cfg.self_mode, str):
        return cfg.self_mode

    threshold = random.random()
    cumulative = 0.0
    last_mode = _MODE_DISTURB
    for mode_name, probability in cfg.self_mode.items():
        prob = float(probability)
        if prob <= _MODE_TOLERANCE:
            continue
        cumulative += prob
        last_mode = mode_name
        if threshold <= cumulative + _MODE_TOLERANCE:
            return mode_name
    return last_mode


def _normalize_sample_payload(
    sampled_params: dict[str, Any] | None,
    cfg: LimitTweakCfg,
    *,
    target: HandCfg,
) -> dict[str, Any]:
    r"""把外部传入的 sampled params 统一规约成结构化 `sample`。

    支持两种入口：

    1. 新 contract：`{"sample": {...}}`
    2. 旧兼容：`{"index_j0::lower": ...}` 或 `{"index_j0": ...}`
    """

    sampled = dict(sampled_params or {})
    sample = sampled.get("sample")
    if isinstance(sample, dict):
        return dict(sample)

    if cfg.self_mode == _MODE_IDENTITY:
        return {"resolved_self_mode": _MODE_IDENTITY}

    joint_deltas: dict[str, dict[str, float]] = {}
    for _, _, joint in _iter_target_joints(target, None):
        lower = float(sampled.get(f"{joint.name}::lower", sampled.get(joint.name, 0.0)))
        if cfg.disturb_object == "shared":
            upper = lower
        else:
            upper = float(sampled.get(f"{joint.name}::upper", 0.0))
        joint_deltas[joint.name] = {"lower": lower, "upper": upper}
    return {
        "resolved_self_mode": _MODE_DISTURB,
        "joint_deltas": joint_deltas,
    }


def _sample_delta_pair(sampler, *, disturb_object: str) -> dict[str, float]:
    r"""采样一对 lower/upper 扰动。

    - `shared`：上下界共享同一随机量；
    - `independent`：上下界各自独立采样。
    """

    lower = float(sampler())
    if disturb_object == "shared":
        return {"lower": lower, "upper": lower}
    return {"lower": lower, "upper": float(sampler())}


def _resolve_homologous_non_thumb_groups(target: HandCfg) -> OrderedDict[str, list[str]]:
    r"""解析 `homologous_non_thumb` 模式下的 non-thumb 同源分组。

    分组键定义为：
    $$
    (\text{finger family},\ \text{joint semantic}).
    $$

    这里的 `joint semantic` 直接来自 child link 名，例如：

    - `index_mcp1 \to mcp1`
    - `middle_pip \to pip`

    这样即便 pre-made `joint_delete` 把 joint 名重新压紧成 `j0/j1/...`，child link 仍保留
    `mcp1/mcp2/pip/dip` 这类 anatomy 语义，因此 mode 仍能跨 delete 后拓扑保持可解释。
    """

    slot_family_map = _resolve_slot_family_map(target)
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for finger in target.fingers:
        if finger.name == "thumb":
            continue  # thumb 在这个 mode 下保持每个 joint 独立，不参与 non-thumb 同源组
        family = slot_family_map.get(finger.name)
        if not isinstance(family, str) or not family:
            raise ValueError(
                "limit_tweak.homologous_non_thumb requires premade slot_family_map; "
                f"missing family for finger {finger.name!r}"
            )
        for joint in finger.joints:
            if joint.joint_type != "revolute" or joint.limit is None:
                continue
            semantic = _resolve_joint_semantic(finger_name=finger.name, child_link=str(joint.child))
            group_key = f"{family}:{semantic}"
            groups.setdefault(group_key, []).append(joint.name)
    return groups


def _resolve_slot_family_map(target: HandCfg) -> dict[str, str]:
    r"""从 hand metadata 中读取当前 topology 的 slot -> family 映射。"""

    metadata = dict(target.metadata or {})
    premade_topology = metadata.get("premade_topology")
    if isinstance(premade_topology, dict):
        slot_family_map = premade_topology.get("slot_family_map")
        if isinstance(slot_family_map, dict):
            return {str(slot): str(family) for slot, family in slot_family_map.items()}
    premade_connectivity = metadata.get("premade_connectivity")
    if isinstance(premade_connectivity, dict):
        slot_family_map = premade_connectivity.get("slot_family_map")
        if isinstance(slot_family_map, dict):
            return {str(slot): str(family) for slot, family in slot_family_map.items()}
    raise ValueError("limit_tweak.homologous_non_thumb requires hand metadata with premade slot_family_map")


def _resolve_joint_semantic(*, finger_name: str, child_link: str) -> str:
    r"""从 child link 名中解析 joint semantic suffix。

    例如：

    - `index_mcp1 -> mcp1`
    - `ring_pip -> pip`

    若当前 child link 不满足这一稳定命名 contract，就直接 fail-hard，
    避免把“不知道 semantic 的 joint”错误并入某个同源组。
    """

    prefix = f"{finger_name}_"
    if not child_link.startswith(prefix):
        raise ValueError(
            "limit_tweak.homologous_non_thumb expects child link names to preserve anatomy suffix; "
            f"got finger={finger_name!r}, child_link={child_link!r}"
        )
    semantic = child_link[len(prefix) :]
    if not semantic:
        raise ValueError(
            "limit_tweak.homologous_non_thumb could not parse non-empty joint semantic suffix from "
            f"child_link={child_link!r}"
        )
    return semantic


def _iter_target_joints(hand: HandCfg, target_joints: tuple[str, ...] | None):
    r"""遍历当前 hand 中会被 `limit_tweak` 消费的目标关节。"""

    target_set = set(target_joints or ())
    for finger_index, finger in enumerate(hand.fingers):
        for joint_index, joint in enumerate(finger.joints):
            if joint.joint_type != "revolute" or joint.limit is None:
                continue
            if target_set and joint.name not in target_set:
                continue
            yield finger_index, joint_index, joint


def _clip_delta(delta: float, clip: dict[str, float] | None) -> float:
    r"""按当前 `clip` 约定裁剪一个 limit 扰动量。"""

    if clip is None:
        return delta
    if "abs" in clip:
        bound = abs(float(clip["abs"]))
    elif "rel" in clip:
        bound = abs(float(clip["rel"]))
    else:
        return delta
    return max(-bound, min(bound, delta))


__all__ = ["LimitTweakCfg", "LimitTweakMutator"]
