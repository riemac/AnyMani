r"""Gaussian density + anchor-relational Material-point Jacobian method 配置。"""

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
    r"""每 JOINT 的 descendant/zero edge 预算与 fixed home-surface identity 选择。"""

    train_active_per_joint: int = 2  # 每 joint 每 q-block 两个 descendant owner/material edges
    train_zero_per_joint: int = 1  # 每 joint 每 q-block 一个 PALM/跨指 structural-zero edge
    fixed_active_per_joint: int = 4  # canonical evaluation 的 active material edges
    fixed_zero_per_joint: int = 4  # canonical evaluation 的 structural-zero material edges
    points_per_edge: int = 1  # 每条 owner/JOINT edge 的固定 material identities 数
    seed_offset: int = 71_117  # 与 query/entity/sign 随机域分离

    def __post_init__(self) -> None:
        r"""拒绝空 edge/material 预算或负随机域。"""

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
    r"""四个无量纲 relation sensitivities 的固定数值尺度。

    数值来自 AR-MPJ-001 的 64-asset teacher RMS 邻域，所有 morphology 共享同一尺度；它们只改善
    optimization conditioning，不按资产消除绝对运动学差异。
    """

    height: float = 0.30
    radius: float = 0.30
    dot: float = 0.13
    chirality: float = 0.13

    @property
    def values(self) -> tuple[float, float, float, float]:
        r"""按 target channel contract 返回稳定顺序。"""

        return (self.height, self.radius, self.dot, self.chirality)

    def __post_init__(self) -> None:
        r"""四通道尺度必须严格为正。"""

        if min(self.values) <= 0.0:
            raise ValueError("Gamma channel scales must be strictly positive")


@dataclass(frozen=True)
class ObjectiveTermCfg:
    r"""不进入 OmegaConf 的模块级 objective callable 绑定。"""

    func: ClassVar[Callable[..., ObjectiveTermResult] | None] = None

    def qualified_func_name(self) -> str:
        r"""返回 artifact/checkpoint 使用的完整公式身份。"""

        func = type(self).func
        if func is None:
            raise RuntimeError(f"{type(self).__name__} has not bound its objective function")
        return f"{func.__module__}.{func.__qualname__}"


@dataclass(frozen=True)
class DensityObjectiveCfg(ObjectiveTermCfg):
    r"""完整 owner/query/sigma Gaussian density MSE。"""

    name: ClassVar[str] = "density"


@dataclass(frozen=True)
class MaterialJacobianObjectiveCfg(ObjectiveTermCfg):
    r"""四通道 anchor-relational Material-point Jacobian objective。"""

    name: ClassVar[str] = "material_jacobian"
    channel_scale: GammaChannelScaleCfg = field(default_factory=GammaChannelScaleCfg)


@dataclass(frozen=True)
class DensityMaterialJacobianObjectivesCfg:
    r"""Density/Gamma 两项主任务；首个正式方法不允许关闭任一项。"""

    density: DensityObjectiveCfg = field(default_factory=DensityObjectiveCfg)
    material_jacobian: MaterialJacobianObjectiveCfg = field(default_factory=MaterialJacobianObjectiveCfg)

    def enabled(self) -> dict[str, ObjectiveTermCfg]:
        r"""按正式训练/日志顺序返回两项目标。"""

        return {"density": self.density, "material_jacobian": self.material_jacobian}


@dataclass(frozen=True)
class DensityMaterialJacobianMethodCfg:
    r"""新联合 method 的 source/target/model/objective/augmentation 完整配置。"""

    runtime_type: ClassVar[type | None] = None  # 在 method.py 绑定，避免循环 import
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
        r"""验证 source/model 语义与 canonical density evaluation 网格。"""

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
