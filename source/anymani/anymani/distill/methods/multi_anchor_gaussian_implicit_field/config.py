"""Shared joint-state, FairGrad, permutation, and sign-rewrite configuration."""


from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from anymani.distill.models.geometry_ssl import GeometrySSLModelCfg
from anymani.distill.objectives.contracts import ObjectiveTermResult
from anymani.distill.representations.geometry import GeometryRepresentationCfg


@dataclass(frozen=True)
class JointConfigurationMeasureCfg:


    kind: str = "scrambled_sobol_joint_limits"


@dataclass(frozen=True)
class JointSignRewriteCfg:


    probability: float = 0.20
    seed_offset: int = 17

    def __post_init__(self) -> None:


        if not 0.0 <= self.probability < 1.0:
            raise ValueError("joint-sign rewrite probability must lie in [0,1)")


@dataclass(frozen=True)
class EntityPermutationCfg:


    enabled: bool = True
    seed_offset: int = 31_337

    def __post_init__(self) -> None:


        if self.seed_offset < 0:
            raise ValueError("entity permutation seed_offset must be non-negative")


@dataclass(frozen=True)
class FairGradCfg:


    algorithm: str = "fairgrad_alpha_1_two_task_analytic_v1"
    near_opposition_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:


        if self.algorithm != "fairgrad_alpha_1_two_task_analytic_v1":
            raise ValueError("unsupported shared-gradient aggregation algorithm")
        if not 0.0 < self.near_opposition_tolerance < 1.0:
            raise ValueError("FairGrad near-opposition tolerance must lie in (0,1)")


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
class KappaObjectiveCfg(ObjectiveTermCfg):


    name: ClassVar[str] = "kappa"


@dataclass(frozen=True)
class MultiAnchorGaussianObjectivesCfg:


    density: DensityObjectiveCfg | None = field(default_factory=DensityObjectiveCfg)
    kappa: KappaObjectiveCfg | None = field(default_factory=KappaObjectiveCfg)

    def enabled(self) -> dict[str, ObjectiveTermCfg]:


        terms = {
            "density": self.density,
            "kappa": self.kappa,
        }
        return {name: config for name, config in terms.items() if config is not None}

    def __post_init__(self) -> None:


        if self.density is None or self.kappa is None:
            raise ValueError("unified multi-anchor method requires both density and kappa objectives")


@dataclass(frozen=True)
class MultiAnchorGaussianMethodCfg:


    runtime_type: ClassVar[type | None] = None
    state_measure: JointConfigurationMeasureCfg = field(default_factory=JointConfigurationMeasureCfg)
    representation: GeometryRepresentationCfg = field(default_factory=GeometryRepresentationCfg)
    model: GeometrySSLModelCfg = field(default_factory=GeometrySSLModelCfg)
    objectives: MultiAnchorGaussianObjectivesCfg = field(default_factory=MultiAnchorGaussianObjectivesCfg)
    fairgrad: FairGradCfg = field(default_factory=FairGradCfg)
    entity_permutation: EntityPermutationCfg = field(default_factory=EntityPermutationCfg)
    joint_sign_rewrite: JointSignRewriteCfg = field(default_factory=JointSignRewriteCfg)

    def __post_init__(self) -> None:


        if self.state_measure.kind != "scrambled_sobol_joint_limits":
            raise ValueError("first-round state measure must be scrambled Sobol over joint limits")
        if not isinstance(self.representation, GeometryRepresentationCfg):
            raise TypeError("multi-anchor method requires GeometryRepresentationCfg")
        if not isinstance(self.model, GeometrySSLModelCfg):
            raise TypeError("multi-anchor method requires GeometrySSLModelCfg")
        if tuple(self.representation.field.fixed_bandwidths_m) != tuple(
            self.representation.field.bandwidth_centers_m
        ):
            raise ValueError("fixed evaluation sigma grid must match the three training centers")


__all__ = [
    "DensityObjectiveCfg",
    "EntityPermutationCfg",
    "FairGradCfg",
    "JointConfigurationMeasureCfg",
    "JointSignRewriteCfg",
    "KappaObjectiveCfg",
    "MultiAnchorGaussianMethodCfg",
    "MultiAnchorGaussianObjectivesCfg",
    "ObjectiveTermCfg",
]
