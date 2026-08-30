r"""N040 proper-SE(3)-invariant density/Gamma method 配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.config import (
    EntityPermutationCfg,
    FairGradCfg,
    JointConfigurationMeasureCfg,
    JointSignRewriteCfg,
)
from anymani.distill.models.se3_density_material_jacobian_ssl import SE3DensityMaterialJacobianModelCfg
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.targets.material_point_jacobian import MaterialPointRelationJacobianCfg

from .config import DensityMaterialJacobianObjectivesCfg, MaterialPointSamplingCfg


@dataclass(frozen=True)
class SE3CoordinateRewriteCfg:
    r"""每个 asset q-block 的随机 proper-SE(3) coordinate rewrite。

    Translation 只改变 hand-frame origin，单位 m；rotation 从 Haar-SO(3) 采样。Reflection 不属于配置域。
    """

    probability: float = 1.0  # 每个 unique asset evidence row 被重写的概率
    translation_half_extent_m: float = 0.05  # 每轴均匀区间 $[-5,5]$ cm
    seed_offset: int = 93_113  # 与 entity/sign/material/query 随机域分离

    def __post_init__(self) -> None:
        r"""验证概率、平移尺度和随机域。"""

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("SE3 rewrite probability must lie in [0,1]")
        if self.translation_half_extent_m < 0.0 or self.seed_offset < 0:
            raise ValueError("SE3 translation extent and seed_offset must be non-negative")


@dataclass(frozen=True)
class SE3DensityMaterialJacobianMethodCfg:
    r"""N040 source/target/model/objective/augmentation 完整配置。"""

    runtime_type: ClassVar[type | None] = None
    state_measure: JointConfigurationMeasureCfg = field(default_factory=JointConfigurationMeasureCfg)
    representation: GeometryRepresentationCfg = field(default_factory=GeometryRepresentationCfg)
    material_target: MaterialPointRelationJacobianCfg = field(default_factory=MaterialPointRelationJacobianCfg)
    material_sampling: MaterialPointSamplingCfg = field(default_factory=MaterialPointSamplingCfg)
    model: SE3DensityMaterialJacobianModelCfg = field(default_factory=SE3DensityMaterialJacobianModelCfg)
    objectives: DensityMaterialJacobianObjectivesCfg = field(default_factory=DensityMaterialJacobianObjectivesCfg)
    fairgrad: FairGradCfg = field(default_factory=FairGradCfg)
    entity_permutation: EntityPermutationCfg = field(default_factory=EntityPermutationCfg)
    joint_sign_rewrite: JointSignRewriteCfg = field(default_factory=JointSignRewriteCfg)
    se3_coordinate_rewrite: SE3CoordinateRewriteCfg = field(default_factory=SE3CoordinateRewriteCfg)

    def __post_init__(self) -> None:
        r"""验证 state measure 与 canonical density bandwidths。"""

        if self.state_measure.kind != "scrambled_sobol_joint_limits":
            raise ValueError("N040 requires scrambled Sobol joint-limit measure")
        if tuple(self.representation.field.fixed_bandwidths_m) != tuple(self.representation.field.bandwidth_centers_m):
            raise ValueError("N040 canonical fixed density grid must match training centers")


__all__ = ["SE3CoordinateRewriteCfg", "SE3DensityMaterialJacobianMethodCfg"]
