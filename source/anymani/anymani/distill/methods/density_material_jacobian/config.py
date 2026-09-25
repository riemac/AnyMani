"""Density and material Jacobian configuration. Joint angles use radians and geometry uses metres."""


from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.config import (
    EntityPermutationCfg,
    FairGradCfg,
    JointConfigurationMeasureCfg,
    JointSignRewriteCfg,
)
from anymani.distill.models.density_material_jacobian_ssl import DensityMaterialJacobianModelCfg
from anymani.distill.objectives.contracts import ObjectiveTermResult
from anymani.distill.representations.geometry import GeometryRepresentationCfg
from anymani.distill.representations.targets.material_point_jacobian import MaterialPointRelationJacobianCfg


@dataclass(frozen=True)
class MaterialPointSamplingCfg:


    train_active_per_joint: int = 2
    train_zero_per_joint: int = 1
    fixed_active_per_joint: int = 4
    fixed_zero_per_joint: int = 4
    points_per_edge: int = 1
    seed_offset: int = 71_117

    def __post_init__(self) -> None:


        counts = (
            self.train_active_per_joint,
            self.train_zero_per_joint,
            self.fixed_active_per_joint,
            self.fixed_zero_per_joint,
            self.points_per_edge,
        )
        if min(counts) < 1 or self.seed_offset < 0:
            raise ValueError("material-point sampling counts must be positive and seed_offset non-negative")


@dataclass(frozen=True)
class GammaChannelScaleCfg:


    height: float = 0.30
    radius: float = 0.30
    dot: float = 0.13
    chirality: float = 0.13

    @property
    def values(self) -> tuple[float, float, float, float]:


        return (self.height, self.radius, self.dot, self.chirality)

    def __post_init__(self) -> None:


        if min(self.values) <= 0.0:
            raise ValueError("Gamma channel scales must be strictly positive")


@dataclass(frozen=True)
class ObjectiveTermCfg:


    func: ClassVar[Callable[..., ObjectiveTermResult] | None] = None

    def qualified_func_name(self) -> str:


        func = type(self).func
        if func is None:
            raise RuntimeError(f"{type(self).__name__} has not bound its objective function")
        return f"{func.__module__}.{func.__qualname__}"


@dataclass(frozen=True)
class DensityObjectiveCfg(ObjectiveTermCfg):


    name: ClassVar[str] = "density"


@dataclass(frozen=True)
class MaterialJacobianObjectiveCfg(ObjectiveTermCfg):


    name: ClassVar[str] = "material_jacobian"
    channel_scale: GammaChannelScaleCfg = field(default_factory=GammaChannelScaleCfg)


@dataclass(frozen=True)
class DensityMaterialJacobianObjectivesCfg:


    density: DensityObjectiveCfg = field(default_factory=DensityObjectiveCfg)
    material_jacobian: MaterialJacobianObjectiveCfg = field(default_factory=MaterialJacobianObjectiveCfg)

    def enabled(self) -> dict[str, ObjectiveTermCfg]:


        return {"density": self.density, "material_jacobian": self.material_jacobian}


@dataclass(frozen=True)
class DensityMaterialJacobianMethodCfg:


    runtime_type: ClassVar[type | None] = None
    state_measure: JointConfigurationMeasureCfg = field(default_factory=JointConfigurationMeasureCfg)
    representation: GeometryRepresentationCfg = field(default_factory=GeometryRepresentationCfg)
    material_target: MaterialPointRelationJacobianCfg = field(default_factory=MaterialPointRelationJacobianCfg)
    material_sampling: MaterialPointSamplingCfg = field(default_factory=MaterialPointSamplingCfg)
    model: DensityMaterialJacobianModelCfg = field(default_factory=DensityMaterialJacobianModelCfg)
    objectives: DensityMaterialJacobianObjectivesCfg = field(default_factory=DensityMaterialJacobianObjectivesCfg)
    fairgrad: FairGradCfg = field(default_factory=FairGradCfg)
    entity_permutation: EntityPermutationCfg = field(default_factory=EntityPermutationCfg)
    joint_sign_rewrite: JointSignRewriteCfg = field(default_factory=JointSignRewriteCfg)

    def __post_init__(self) -> None:


        if self.state_measure.kind != "scrambled_sobol_joint_limits":
            raise ValueError("joint density/Gamma method requires scrambled Sobol joint-limit measure")
        if tuple(self.representation.field.fixed_bandwidths_m) != tuple(self.representation.field.bandwidth_centers_m):
            raise ValueError("canonical fixed density bandwidth grid must match declared centers")


__all__ = [
    "DensityMaterialJacobianMethodCfg",
    "DensityMaterialJacobianObjectivesCfg",
    "DensityObjectiveCfg",
    "GammaChannelScaleCfg",
    "MaterialJacobianObjectiveCfg",
    "MaterialPointSamplingCfg",
]
