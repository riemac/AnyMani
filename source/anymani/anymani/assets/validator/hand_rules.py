r"""整手级验证规则集和验证流水线。

这是验证层的顶层编排者，整合 joint / finger 级验证后再加上手级全局规则。
对应 `资产生产概略.png` 中 `pre-made → validator → HandCfgs` 和
`post-mutate → HandCfgs` 两处 validator 节点。

当前收纳的手级全局规则
----------------------

1. **全局名称唯一性**：joint / link / finger 名称在全手范围内不得重复
   （schema 已做，但 post-mutate 后可能再违规，需重跑）
2. **DOF 总量范围**：整手 DOF 是否在 ``[dof_min, dof_max]`` 内
3. **手指数量范围**：finger 数量是否在 ``[finger_count_min, finger_count_max]`` 内
4. **所有旁链挂载一致性**：所有 finger 的 ``parent_link`` 均等于 ``palm.name``
   （schema 已做，但任何 post-mutate 挂载/结构改动后仍需重跑）
5. **手指间几何 clearance**：
   - pre-made 阶段仍可使用 mount-origin 轻量近似；
   - post-mutate 阶段默认升级为 collision geometry 的 sampled-surface SDF clearance。

设计说明
--------

### 两次校验时机

从 `资产生产概略.png` 可以看到，validator 在流水线里出现了两次：
- 第一次：pre-made 之后（先验产物的基本合法性，再喂给 post-mutate）
- 第二次：post-mutate 之后（验证 mutate 没有破坏全局一致性）

这次重构后，两次虽然仍共用同一个 `HandValidator` 运行时壳，但配置上已经显式拆成：

- `HandValidatorCfg.pre_made`
- `HandValidatorCfg.post_mutate`

### strict 模式

各阶段都保留自己的 ``strict`` 开关：

- `pre_made.strict`
- `post_mutate.strict`

这样调用方可以独立决定哪一阶段的 warnings 要升级成 errors。

### 手指间距计算的阶段语义

pre-made 阶段的拓扑筛选仍可使用最轻量的 mount-origin 近似：

$$
d_{ij} = \|p_i - p_j\|_2 \geq d_{\min}
$$

post-mutate 阶段则使用真实 collision primitive 的 SDF surface clearance：

$$
c(F_i,F_j)=\min\left(
  \min_{\mathbf{x}\in S_{F_i}}\operatorname{SDF}_{F_j}(\mathbf{x}),
  \min_{\mathbf{y}\in S_{F_j}}\operatorname{SDF}_{F_i}(\mathbf{y})
\right).
$$

这个证书只覆盖 `post_mutate_home_pose`，不声称 all-pose / mesh-exact /
trajectory / runtime physics safety。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ..asset_base import AssetCfgBase, HandCfg
from ._base import ValidatorBase, ValidationResult
from .finger_rules import FingerValidatorCfg, FingerValidator
from ._sdf_clearance import SdfClearanceConfig, evaluate_finger_sdf_clearance


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class _HandValidatorStageCfg(AssetCfgBase):
    r"""单个 validator 阶段共享的规则集。

    这里把“整手合法性”里与阶段无关的主体规则抽成共享基类：

    - pre-made 关心它，因为 connectivity lower 后要先筛掉明显不合法的 topology；
    - post-mutate 也关心它，因为连续参数扰动可能再次破坏这些约束。
    """

    dof_min: int | None = 1
    """允许的最小总 DOF；为 ``None`` 时不检查下限。"""

    dof_max: int | None = None
    """允许的最大总 DOF；为 ``None`` 时不检查上限。"""

    finger_count_min: int | None = 3
    """允许的最少手指数；为 ``None`` 时不检查。"""

    finger_count_max: int | None = 4
    """允许的最多手指数；为 ``None`` 时不检查。"""

    require_thumb: bool = True
    """是否强制要求整手必须包含一根逻辑名为 `thumb` 的手指。"""

    thumb_min_revolute_dof: int | None = 3
    """thumb 至少要保留多少个 revolute DOF；为 ``None`` 时不检查。"""

    require_non_thumb_with_min_revolute_dof: int | None = 3
    """是否要求至少存在一根 non-thumb finger，其 revolute DOF 不低于该阈值。"""

    check_global_uniqueness: bool = True
    """是否重跑全局名称唯一性检查（joint / link / finger 名称全手唯一）。"""

    check_mount_consistency: bool = True
    """是否检查所有 finger 的 ``parent_link`` 均等于 ``palm.name``。"""

    check_finger_spacing: bool = True
    """是否检查任意两根手指之间的最小间距。

    # NOTE:
    字段名保持旧 contract，不改成 `check_clearance`，是为了不破坏已有 recipe。
    但 post-mutate 阶段的真实语义已从 mount-origin distance 升级为 SDF clearance。
    """

    min_finger_spacing: float = 0.015
    r"""允许的最小手指间距 / clearance（meter）。

    pre-made 阶段解释为 mount-origin 距离下限；post-mutate SDF 阶段解释为：
    $$
    c(F_i,F_j)\geq\texttt{min\_finger\_spacing}.
    $$
    """

    finger: FingerValidatorCfg = field(default_factory=FingerValidatorCfg)
    """手指级验证配置；hand 验证器内部对每根 finger 跑此配置。"""

    strict: bool = False
    """是否把所有 warnings 升级为 errors（严格模式）。"""


@dataclass
class HandValidatorPreMadeCfg(_HandValidatorStageCfg):
    r"""pre-made 阶段的结构性规则。"""

    check_palm_thumb_binding: bool = True
    """是否检查 palm family 与 thumb family 必须一致。"""


@dataclass
class HandValidatorPostMutateCfg(_HandValidatorStageCfg):
    r"""post-mutate 阶段的几何/参数后验规则。"""

    sdf_surface_samples_per_axis: int = 5
    """SDF surface sampling 密度。

    对 box 是每个面上的规则网格密度；对 cylinder / sphere 会转换成经纬 /
    角向采样。该字段不是“精确度证明”，只是 sampled approximation 的预算。
    """

    sdf_unsupported_geometry_policy: Literal["fail", "warn_skip"] = "fail"
    """unsupported collision geometry 的处理策略。

    默认 ``"fail"``，因为科研 validator 不能把“不知道”伪装成“安全”。
    ``"warn_skip"`` 只用于 exploratory 调试，会写入 `skipped_bodies` 并把
    certificate 标记为 incomplete。
    """

    sdf_threshold_tolerance: float = 1e-9
    """clearance 阈值比较的数值容差，只用于抵消浮点误差。"""

    sdf_device: Literal["auto", "cuda", "cpu"] = "auto"
    """SDF 数值计算设备。

    默认 ``"auto"``：优先尝试 CUDA，若当前环境或几何类型无法走 GPU，则由
    SDF 后端回退 CPU。证书会记录实际使用的 device，避免性能路径变成隐式假设。
    """

    sdf_mesh_backend: Literal["auto", "warp", "trimesh"] = "auto"
    """mesh signed-distance 后端策略。

    默认 ``"auto"``：custom STL/OBJ collision 先走 Warp GPU mesh query；若 Warp
    不可用或运行失败，则回退到 trimesh CPU。显式 ``"warp"`` 用于强制 GPU 路线。
    """

    sdf_mesh_surface_samples: int = 4096
    """每个 mesh collision body 的 surface 采样点数。

    4096 是当前首版的保守科研预算：优先降低 custom fingertip 漏检风险，而不是把
    validator 调成过轻量的 exploratory 检查。
    """


@dataclass
class HandValidatorCfg(AssetCfgBase):
    r"""整手级验证规则配置（显式区分 pre-made / post-mutate 两个阶段）。"""

    PreMadeCfg = HandValidatorPreMadeCfg
    PostMutateCfg = HandValidatorPostMutateCfg

    class_type: type["HandValidator"] | None = None
    """关联的运行时类。"""

    pre_made: HandValidatorPreMadeCfg = field(default_factory=HandValidatorPreMadeCfg)
    """pre-made 阶段规则：结构拓扑、thumb 完整性、palm-thumb family 绑定等。"""

    post_mutate: HandValidatorPostMutateCfg = field(default_factory=HandValidatorPostMutateCfg)
    """post-mutate 阶段规则：几何/参数扰动后的全局一致性闸门。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandValidator


# ============================================================================
#  运行时壳
# ============================================================================


class HandValidator(ValidatorBase):
    r"""整手级验证器（同时是验证流水线）。

    对 `HandCfg` 做层次化验证：先调用 `FingerValidator` 对每根手指跑规则，
    再叠加手级全局规则，最终返回合并后的 `ValidationResult`。
    """

    cfg: HandValidatorCfg

    def __init__(self, cfg: HandValidatorCfg):
        self.cfg = cfg

    def validate(
        self,
        target: HandCfg,
        *,
        stage: str = "post_mutate",
    ) -> ValidationResult:  # type: ignore[override]
        r"""对 `HandCfg` 执行分阶段整手验证。

        Args:
            target (HandCfg): 待验证的整手配置。
            stage (str): 当前验证阶段。

                - ``"pre_made"``：connectivity lower 后的结构性闸门
                - ``"post_mutate"``：post-mutate 之后的后验闸门（默认）

        Returns:
            ValidationResult: 含 finger 级与 hand 级规则的合并结果。
        """

        if stage == "pre_made":
            return self._validate_with_stage_cfg(target, stage_cfg=self.cfg.pre_made, stage=stage)
        if stage == "post_mutate":
            return self._validate_with_stage_cfg(target, stage_cfg=self.cfg.post_mutate, stage=stage)
        raise ValueError(f"Unsupported validation stage {stage!r}; expected 'pre_made' or 'post_mutate'.")

    def validate_pre_made(self, target: HandCfg) -> ValidationResult:
        r"""对 pre-made 产物执行结构性校验。"""

        return self.validate(target, stage="pre_made")

    def validate_post_mutate(self, target: HandCfg) -> ValidationResult:
        r"""对 post-mutate 产物执行后验校验。"""

        return self.validate(target, stage="post_mutate")

    def _validate_with_stage_cfg(
        self,
        target: HandCfg,
        *,
        stage_cfg: _HandValidatorStageCfg,
        stage: str,
    ) -> ValidationResult:
        r"""按给定阶段 cfg 执行一遍完整的层次化整手验证。"""

        result = ValidationResult()
        finger_validator = FingerValidator(stage_cfg.finger)

        for finger in target.fingers:
            result.merge(finger_validator.validate(finger))

        if isinstance(stage_cfg, HandValidatorPreMadeCfg) and stage_cfg.check_palm_thumb_binding:
            self._validate_palm_thumb_binding(target, result=result)

        if stage_cfg.check_global_uniqueness:
            finger_names = [finger.name for finger in target.fingers]
            if len(finger_names) != len(set(finger_names)):
                result.errors.append(f"hand '{target.name}'[{stage}]: duplicate finger names")

            joint_names = [joint.name for joint in target.iter_joints()]
            if len(joint_names) != len(set(joint_names)):
                result.errors.append(f"hand '{target.name}'[{stage}]: duplicate joint names")

            link_names = [target.palm.name] + [joint.child for joint in target.iter_joints()]
            if len(link_names) != len(set(link_names)):
                result.errors.append(f"hand '{target.name}'[{stage}]: duplicate link names")

        dof = target.dof_count
        if stage_cfg.dof_min is not None and dof < stage_cfg.dof_min:
            result.errors.append(f"hand '{target.name}'[{stage}]: dof {dof} < min {stage_cfg.dof_min}")
        if stage_cfg.dof_max is not None and dof > stage_cfg.dof_max:
            result.warnings.append(f"hand '{target.name}'[{stage}]: dof {dof} > max {stage_cfg.dof_max}")

        finger_count = len(target.fingers)
        if stage_cfg.finger_count_min is not None and finger_count < stage_cfg.finger_count_min:
            result.errors.append(
                f"hand '{target.name}'[{stage}]: finger count {finger_count} < min {stage_cfg.finger_count_min}"
            )
        if stage_cfg.finger_count_max is not None and finger_count > stage_cfg.finger_count_max:
            result.errors.append(
                f"hand '{target.name}'[{stage}]: finger count {finger_count} > max {stage_cfg.finger_count_max}"
            )

        thumb_finger = next((finger for finger in target.fingers if finger.name == "thumb"), None)
        if stage_cfg.require_thumb and thumb_finger is None:
            result.errors.append(f"hand '{target.name}'[{stage}]: missing required thumb finger")

        if thumb_finger is not None and stage_cfg.thumb_min_revolute_dof is not None:
            thumb_dof = _revolute_dof_count(thumb_finger)
            if thumb_dof < stage_cfg.thumb_min_revolute_dof:
                result.errors.append(
                    f"hand '{target.name}'[{stage}]: thumb revolute dof {thumb_dof} < min {stage_cfg.thumb_min_revolute_dof}"
                )

        if stage_cfg.require_non_thumb_with_min_revolute_dof is not None:
            threshold = stage_cfg.require_non_thumb_with_min_revolute_dof
            non_thumb_dofs = [
                (finger.name, _revolute_dof_count(finger))
                for finger in target.fingers
                if finger.name != "thumb"
            ]
            if not any(dof >= threshold for _, dof in non_thumb_dofs):
                result.errors.append(
                    f"hand '{target.name}'[{stage}]: expected at least one non-thumb finger with revolute dof >= {threshold}, "
                    f"got {non_thumb_dofs!r}"
                )

        if stage_cfg.check_mount_consistency:
            for finger in target.fingers:
                if finger.parent_link != target.palm.name:
                    result.errors.append(
                        f"finger '{finger.name}'[{stage}] parent_link '{finger.parent_link}' != palm '{target.palm.name}'"
                    )

        if stage_cfg.check_finger_spacing:
            if isinstance(stage_cfg, HandValidatorPostMutateCfg):
                self._validate_post_mutate_sdf_clearance(target, stage_cfg=stage_cfg, result=result)
            else:
                self._validate_mount_origin_spacing(target, stage_cfg=stage_cfg, stage=stage, result=result)

        if stage_cfg.strict:
            result = result.as_strict()
        result.passed = len(result.errors) == 0
        return result

    def _validate_mount_origin_spacing(
        self,
        target: HandCfg,
        *,
        stage_cfg: _HandValidatorStageCfg,
        stage: str,
        result: ValidationResult,
    ) -> None:
        r"""执行旧版 mount-origin 间距近似。

        该规则只比较 finger root frame 的原点：
        $$
        d_{ij}=\|p_i-p_j\|_2.
        $$

        它不能发现“root 原点相距尚可，但 link mesh 因姿态/宽度发生穿膜”的反例，
        因而 post-mutate 默认不再走这条路径。
        """

        mounts = [(finger.name, finger.mount.pos) for finger in target.fingers]
        for idx in range(len(mounts)):
            for jdx in range(idx + 1, len(mounts)):
                name_i, pos_i = mounts[idx]
                name_j, pos_j = mounts[jdx]
                distance = math.sqrt(sum((lhs - rhs) ** 2 for lhs, rhs in zip(pos_i, pos_j)))
                if distance < stage_cfg.min_finger_spacing:
                    result.warnings.append(
                        f"finger spacing '{name_i}'-'{name_j}'[{stage}, mount_origin_approx]: "
                        f"{distance * 100.0:.2f} cm < min {stage_cfg.min_finger_spacing * 100.0:.2f} cm"
                    )

    def _validate_post_mutate_sdf_clearance(
        self,
        target: HandCfg,
        *,
        stage_cfg: HandValidatorPostMutateCfg,
        result: ValidationResult,
    ) -> None:
        r"""执行 post-mutate home-pose sampled SDF clearance 检查。

        该函数只负责规则编排：

        - `_collision_geometry.py` 抽取 collision body；
        - `_sdf_clearance.py` 计算 signed distance 与 certificate；
        - 本函数把 certificate 接回 `ValidationResult`，并决定 error 文案。
        """

        try:
            clearance = evaluate_finger_sdf_clearance(
                target,
                SdfClearanceConfig(
                    min_clearance=stage_cfg.min_finger_spacing,
                    surface_samples_per_axis=stage_cfg.sdf_surface_samples_per_axis,
                    unsupported_policy=stage_cfg.sdf_unsupported_geometry_policy,
                    tolerance=stage_cfg.sdf_threshold_tolerance,
                    device=stage_cfg.sdf_device,
                    mesh_backend=stage_cfg.sdf_mesh_backend,
                    mesh_surface_samples=stage_cfg.sdf_mesh_surface_samples,
                ),
            )
        except (RuntimeError, ValueError) as exc:
            result.errors.append(f"finger spacing sdf[post_mutate]: {exc}")
            result.metadata["finger_spacing_certificate"] = {
                "pose_scope": "post_mutate_home_pose",
                "geometry_scope": "collision_geometry_only",
                "sdf_kind": "sampled_surface_sdf_approx",
                "complete": False,
                "device": stage_cfg.sdf_device,
                "mesh_sdf": {
                    "requested_backend": stage_cfg.sdf_mesh_backend,
                    "actual_backend": "none",
                    "mesh_query_count": 0,
                    "mesh_sample_count": 0,
                    "fallback_events": [],
                },
                "skipped_bodies": [],
                "not_certified": [
                    "all_pose_collision_free",
                    "mesh_exact_clearance",
                    "trajectory_safety",
                    "physics_runtime_safety",
                ],
            }
            return

        certificate = clearance.certificate.to_dict()
        result.metadata["finger_spacing_certificate"] = certificate
        if not clearance.certificate.complete:
            result.errors.append(
                "finger spacing sdf[post_mutate]: incomplete certificate; "
                f"skipped_bodies={certificate.get('skipped_bodies', [])!r}"
            )

        for pair in clearance.violations:
            result.errors.append(
                f"finger spacing '{pair.finger_i}'-'{pair.finger_j}'[post_mutate, sdf_clearance]: "
                f"{pair.clearance * 100.0:.2f} cm < min {stage_cfg.min_finger_spacing * 100.0:.2f} cm "
                f"(i_to_j={pair.direction_i_to_j * 100.0:.2f} cm, "
                f"j_to_i={pair.direction_j_to_i * 100.0:.2f} cm)"
            )

    def _validate_palm_thumb_binding(self, target: HandCfg, *, result: ValidationResult) -> None:
        r"""检查 pre-made topology 是否满足 palm family 与 thumb family 绑定。"""

        thumb_finger = next((finger for finger in target.fingers if finger.name == "thumb"), None)
        if thumb_finger is None:
            return  # `require_thumb` 会负责“有没有拇指”，这里只关心“有拇指时 family 是否匹配”

        metadata = dict(target.metadata or {})
        premade_metadata = metadata.get("premade_connectivity")
        if not isinstance(premade_metadata, dict):
            premade_metadata = metadata.get("premade_topology")
        if not isinstance(premade_metadata, dict):
            return  # 非 pre-made / 无 provenance 的手不强行套这条规则

        slot_family_map = premade_metadata.get("slot_family_map")
        if not isinstance(slot_family_map, dict):
            return

        thumb_family = slot_family_map.get("thumb")
        if thumb_family is None:
            return

        palm_family = target.family
        if str(thumb_family) != str(palm_family):
            result.errors.append(
                f"hand '{target.name}'[pre_made]: palm family {palm_family!r} requires thumb family to match, "
                f"got thumb family {thumb_family!r}"
            )


def _revolute_dof_count(finger) -> int:
    r"""统计一根 finger 当前 surviving 链上的 revolute DOF 数。"""

    return sum(1 for joint in finger.joints if joint.joint_type == "revolute")


__all__ = ["HandValidatorCfg", "HandValidator"]
