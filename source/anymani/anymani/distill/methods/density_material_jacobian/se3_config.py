"""N040 configuration. It keeps the proper-SE(3) coordinate contract, preserves reflection-sensitive chirality, and retains the published density and material Jacobian objectives."""


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


    probability: float = 1.0
    translation_half_extent_m: float = 0.05
    seed_offset: int = 93_113

    def __post_init__(self) -> None:


        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("SE3 rewrite probability must lie in [0,1]")
        if self.translation_half_extent_m < 0.0 or self.seed_offset < 0:
            raise ValueError("SE3 translation extent and seed_offset must be non-negative")


@dataclass(frozen=True)
class SE3DensityMaterialJacobianMethodCfg:


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


        if self.state_measure.kind != "scrambled_sobol_joint_limits":
            raise ValueError("N040 requires scrambled Sobol joint-limit measure")
        if tuple(self.representation.field.fixed_bandwidths_m) != tuple(self.representation.field.bandwidth_centers_m):
            raise ValueError("N040 canonical fixed density grid must match training centers")


__all__ = ["SE3CoordinateRewriteCfg", "SE3DensityMaterialJacobianMethodCfg"]
