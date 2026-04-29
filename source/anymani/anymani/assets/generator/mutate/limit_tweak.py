r"""关节限位微调算子。

科研语义：
`limit_tweak` 改的是活动关节的合法角域 $[q_{\min},q_{\max}]$，不是几何
链长，也不是 joint axis。为了保持 post-mutate 不改变 topology，本算子
只生成 limit 字段 patch，并保持 `lower < upper`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ...asset_base import AssetCfgBase, HandCfg
from ...asset_schema_core import JointLimitCfg
from ._base import HandPatch, MutatorBase
from ._distribution import ScalarDistributionCfg, normalize_distribution


@dataclass
class LimitTweakCfg(AssetCfgBase):
    r"""关节限位微调配置。

    兼容两套表述：
    - 新 quick/test 常用：`mode`、`symmetric`、`delta_distribution`、`clip`；
    - 草稿保留：`disturb_type`、`joint_range`、`distrib`。
    """

    class_type: type["LimitTweakMutator"] | None = None
    target_joints: tuple[str, ...] | None = None
    mode: Literal["absolute", "relative"] = "absolute"
    symmetric: bool = False
    delta_distribution: Any = None
    clip: float | dict[str, float] | None = None

    disturb_unit: Literal["deg", "rad"] = "rad"
    disturb_object: Literal["independent", "shared"] = "independent"
    disturb_type: Literal["add", "scale"] | None = None
    joint_range: tuple[float, float] | None = None
    distrib: Literal["uniform", "normal"] | dict[str, Any] = "uniform"
    boundary_policy: Literal["none", "clip", "truncate", "resample"] | None = None

    def __post_init__(self) -> None:
        self.class_type = LimitTweakMutator
        if isinstance(self.target_joints, list):
            self.target_joints = tuple(self.target_joints)
        if self.disturb_type == "scale":
            self.mode = "relative"
        if self.delta_distribution is None:
            if self.joint_range is not None:
                self.delta_distribution = ScalarDistributionCfg(kind="uniform", low=self.joint_range[0], high=self.joint_range[1])
            else:
                self.delta_distribution = ScalarDistributionCfg(kind="fixed", value=0.0)
        else:
            self.delta_distribution = normalize_distribution(self.delta_distribution)


class LimitTweakMutator(MutatorBase):
    r"""生成 joint limit 的 deferred patch。"""

    cfg: LimitTweakCfg

    def __init__(self, cfg: LimitTweakCfg):
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        return {
            f"{joint.name}::delta": self.cfg.delta_distribution
            for _, _, joint in _iter_target_joints(target, self.cfg.target_joints)
        }

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        sampled_params = sampled_params or {}
        patch = HandPatch()
        for finger_index, joint_index, joint in _iter_target_joints(target, self.cfg.target_joints):
            delta = _clip_delta(float(sampled_params.get(f"{joint.name}::delta", 0.0)), self.cfg.clip)

            def apply_limit(hand: HandCfg, *, fi=finger_index, ji=joint_index, d=delta) -> None:
                current = hand.fingers[fi].joints[ji].limit
                if current is None:
                    return
                if self.cfg.mode == "relative":
                    lower = current.lower * (1.0 + d)
                    upper = current.upper * (1.0 + d)
                elif self.cfg.symmetric:
                    lower = current.lower - d
                    upper = current.upper + d
                else:
                    lower = current.lower + d
                    upper = current.upper + d
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


def _clip_delta(delta: float, clip: float | dict[str, float] | None) -> float:
    if clip is None:
        return delta
    if isinstance(clip, dict):
        if "abs" in clip:
            bound = abs(float(clip["abs"]))
        elif "rel" in clip:
            bound = abs(float(clip["rel"]))
        else:
            return delta
    else:
        bound = abs(float(clip))
    return max(-bound, min(bound, delta))


__all__ = ["LimitTweakCfg", "LimitTweakMutator"]
