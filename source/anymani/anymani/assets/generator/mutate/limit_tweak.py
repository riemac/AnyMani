r"""关节限位微调算子：在已有 `HandCfg` 上对 joint limit 做小范围参数修改。

科研语义：
`limit_tweak` 改的是活动关节的合法角域 $[q_{\min},q_{\max}]$，不是几何
链长，也不是 joint axis。为了保持 post-mutate 不改变 topology，本算子
只生成 limit 字段 patch，并保持 `lower < upper`。

这意味着它在物理上属于“关节运动学约束”的局部再标定，而不是手型
重建；因此它和 `link_scale`、`mount_perturb` 在建模层面是正交的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

from ...asset_base import HandCfg
from ...asset_schema_core import JointLimitCfg, Vector2
from ._base import HandPatch, MutatorBase, MutatorBaseCfg, _make_range_sampler


@dataclass
class LimitTweakCfg(MutatorBaseCfg):
    r"""关节限位微调工具配置。

    这份配置在科研上对应两层语义：

    1. 关节极限角域的合法范围本身要被扰动；
    2. 扰动后的 limit 仍要保持物理合理性，尤其不能翻转成 lower >= upper。
    """

    class_type: type["LimitTweakMutator"] | None = field(init=False, default=None, repr=False)
    r"""关联的运行时类。"""

    disturb_unit: Literal["deg", "rad"] = "deg"
    r"""微调范围的单位，默认为度。

    这里记录的是用户输入范围的解释单位，不是导出到 URDF 的单位；
    真正写回时仍要映射到 `JointLimitCfg` 的弧度语义。
    """

    disturb_object: Literal["independent", "shared"] = "independent"
    r"""扰动对象。适用于所有活动关节。

    所有关节限位 $[q_{min},\ q_{max}]$ 都有 lower 与 upper 两个边界：
    - independent: 微调可以独立 (independent) 地对 lower 和 upper 进行。
    - shared: 微调可以共享 (shared) 同一扰动值。
    """

    disturb_type: Literal["add", "scale"] = "add"
    r"""扰动类型。默认为添加。

    - `add`: 在原有值基础上添加
    - `scale`: 在原有值基础上按比例缩放
    """

    joint_range: Vector2 | None = None
    r"""关节限位微调范围配置，同时适用于所有活动关节。

    表示在原有 `HandCfg` 的 joint limit 基础上进行微调的范围；
    `None` 表示不进行操作。

    - 当扰动类型为 `add` 时，可表示为 (-5, 5)。单位为 `deg` 表示原有基础上 ±5度 的范围内扰动
    - 当扰动类型为 `scale` 时，可表示为 (0.9, 1.1)，表示在原有基础上 ±10% 的范围内扰动。
    """

    clip: dict[str, float] | None = None
    r"""裁剪范围。默认不裁剪。

    - {"abs": 10}：表示微调后关节限位的绝对值不超过 10 度（disturb_unit 为 `deg` 时）。
    - {"rel": 0.2}：表示微调后关节限位的相对值不超过 20%。
    """

    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    r"""分布类型。可选正态分布/均匀分布。

    适用于所选中的全部关节对象，但这不表示不同关节每次采样相同；
    每个关节仍然是独立随机变量，只是共享同一套分布假设。

    支持以下两种输入格式：
    1. 字符串简写（使用默认参数）：
       - "uniform"：在 `link_size` 定义的范围内做均匀采样（默认）。
       - "normal"：以原尺寸为中心，`link_size` 定义的范围作为 ±3σ 的区间，做正态分布采样。

    2. 字典详细配置（用于自定义分布参数）：
       - {"type": "normal", "sigma_rule": 1}：使用 1σ 法则（即范围的半宽作为 1σ，分布更平缓，贴近均匀分布）。
       - {"type": "normal", "sigma": 1/3}：直接指定 σ 为各关节限位范围半宽的 1/3, 相当于 3σ 法则。
       - {"type": "uniform"}：等同于 "uniform"。
    """

    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None
    r"""关节限位扰动的边界处理策略。

    该字段只规定采样结果超出 `joint_range` 或 `clip` 所定义边界时如何处理，
    不改变扰动类型 `disturb_type`，也不改变基础分布 `distrib`。

    - ``"none"``：不做额外边界处理，适合均匀分布已经严格落在合法区间内的情形。
    - ``"clip"``：把越界样本裁剪到边界上，实现简单，但会增加边界点的概率质量。
    - ``"truncate"``：直接使用截断分布采样，概率语义更干净。
    - ``"resample"``：拒绝越界样本并重新采样，即 rejection sampling。

    默认值为 ``None`` 时，可由运行时根据 `distrib` 自动选择：
    均匀分布通常等价于 ``"none"``；正态分布通常使用 ``"truncate"`` 或 ``"resample"``。
    """

    _distribution: Any = field(init=False, repr=False)
    """内部解析 disturb / distrib / boundary_policy 后生成的采样对象。

    这个字段只服务运行时，不应该被研究者当成 public API。
    """

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = LimitTweakMutator


class LimitTweakMutator(MutatorBase):
    r"""关节限位微调运行时壳。

    在已构建好的 `HandCfg` 上对目标关节的 `limit.lower` / `limit.upper`
    做小范围参数修改，不改变拓扑和几何。
    """

    cfg: LimitTweakCfg

    def __init__(self, cfg: LimitTweakCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        r"""把 current limit tweak 语义 lowering 成可批量采样变量。

        如果没有 `joint_range`，就表示这一轮不启用 limit 变异，因此返回空表；
        这样比“塞一个零分布”更清楚，因为它明确表达了 term 级别的关闭。
        """

        if self.cfg.joint_range is None:
            return {}
        sample_range = _unit_adjusted_range(self.cfg.joint_range, self.cfg.disturb_unit, self.cfg.disturb_type)
        sampler = _make_range_sampler(
            sample_range,
            distrib=self.cfg.distrib,
            boundary_policy=self.cfg.boundary_policy,
        )
        if self.cfg.disturb_object == "shared":
            return {joint.name: sampler for _, _, joint in _iter_target_joints(target, None)}
        return {
            f"{joint.name}::lower": sampler
            for _, _, joint in _iter_target_joints(target, None)
        } | {
            f"{joint.name}::upper": sampler
            for _, _, joint in _iter_target_joints(target, None)
        }

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        sampled_params = sampled_params or {}
        patch = HandPatch()
        # 这里的 patch 只负责写回 lower / upper，不碰 axis、link geometry
        # 或 joint parent-child 关系。
        for finger_index, joint_index, joint in _iter_target_joints(target, None):
            delta = _clip_delta(float(sampled_params.get(f"{joint.name}::lower", sampled_params.get(joint.name, 0.0))), self.cfg.clip)

            def apply_limit(hand: HandCfg, *, fi=finger_index, ji=joint_index, d=delta) -> None:
                current = hand.fingers[fi].joints[ji].limit
                if current is None:
                    return
                if self.cfg.disturb_type == "scale":
                    lower = current.lower * (1.0 + d)
                    upper = current.upper * (1.0 + d)
                elif self.cfg.disturb_object == "shared":
                    lower = current.lower + d
                    upper = current.upper + d
                else:
                    lower = current.lower + d
                    upper = current.upper + _clip_delta(float(sampled_params.get(f"{joint.name}::upper", 0.0)), self.cfg.clip)
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


def _iter_target_joints(hand: HandCfg, target_joints: tuple[str, ...] | None):
    target_set = set(target_joints or ())
    for finger_index, finger in enumerate(hand.fingers):
        for joint_index, joint in enumerate(finger.joints):
            if joint.joint_type != "revolute" or joint.limit is None:
                continue
            if target_set and joint.name not in target_set:
                continue
            yield finger_index, joint_index, joint


def _unit_adjusted_range(value_range: Vector2, disturb_unit: Literal["deg", "rad"], disturb_type: Literal["add", "scale"]) -> Vector2:
    r"""把用户输入区间解释成运行时实际采样区间。

    对 ``add`` 来说，区间代表绝对角度增量；对 ``scale`` 来说，区间
    本身已经是比例语义，所以无需再做度到弧度的转换。
    """

    if disturb_type == "scale" or disturb_unit == "rad":
        return value_range
    return (math.radians(float(value_range[0])), math.radians(float(value_range[1])))


def _clip_delta(delta: float, clip: dict[str, float] | None) -> float:
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
