r"""多锚点 Gaussian 隐式场方法的强类型装配配置。

一份方法就是
$$
\mathcal M=\left(\mu_q,\mathcal R,f_\theta,\mathcal L,\mathcal A\right):
$$
`state_measure` 定义 $q$ 的测度，`representation` 定义物理真值体系，`model` 是可学习映射，
`objectives` 定义 rho/kappa 两项训练约束，`joint_sign_rewrite` 是方法专属坐标增强。Trainer 预算、
resident window 和 `calibrate_objectives` 阶段不属于本配置。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from anymani.distill.models.geometry_ssl import GeometrySSLModelCfg
from anymani.distill.objectives.contracts import ObjectiveTermResult
from anymani.distill.representations.geometry import GeometryRepresentationCfg


@dataclass(frozen=True)
class JointConfigurationMeasureCfg:
    r"""完整 joint-limit 超矩形上的连续 scrambled Sobol 测度。

    $$
    q\sim\operatorname{Sobol}\left(\prod_{i=1}^{N_J}[q_i^{\min},q_i^{\max}]\right).
    $$

    该测度包含自碰撞构型。首轮不引入在线碰撞筛选；limits 只定义采样域，不进入 encoder。
    """

    kind: str = "scrambled_sobol_joint_limits"


@dataclass(frozen=True)
class JointSignRewriteCfg:
    r"""训练期单 JOINT 坐标符号改写。

    每个 $(asset,q)$ 先以 `probability` 决定是否改写；若选中，恰好翻一个有效 JOINT：
    $$
    q_i'=-q_i,\qquad q_i^{home\prime}=-q_i^{home},\qquad \mathcal S_i'=-\mathcal S_i.
    $$
    density/distance 不变；对应 JOINT 的 $\kappa/g$ 翻号。validation 不做这项增强，另做双前向 parity audit。
    """

    probability: float = 0.20  # 每个 $(asset,q)$ 被选中改写的概率
    seed_offset: int = 17  # 相对 schedule seed 的确定性偏移

    def __post_init__(self) -> None:
        r"""拒绝越出 $[0,1)$ 的改写概率。"""

        if not 0.0 <= self.probability < 1.0:
            raise ValueError("joint-sign rewrite probability must lie in [0,1)")


@dataclass(frozen=True)
class ObjectiveTermCfg:
    r"""无 term-specific 超参的 objective 声明：只有显式无量纲权重。

    `func` 用 ClassVar 绑定模块级 callable，不进入 OmegaConf structured config。
    """

    func: ClassVar[Callable[..., ObjectiveTermResult] | None] = None
    weight: float = 1.0  # $\lambda$；calibration 只记录，不自动改写

    def __post_init__(self) -> None:
        r"""允许权重为零做消融，但拒绝负权重。"""

        if self.weight < 0.0:
            raise ValueError("objective term weight must be non-negative")

    def qualified_func_name(self) -> str:
        r"""返回 artifact 记录用的完整限定名。"""

        func = type(self).func
        if func is None:
            raise RuntimeError(f"{type(self).__name__} has not bound its objective function")
        return f"{func.__module__}.{func.__qualname__}"


@dataclass(frozen=True)
class DensityObjectiveCfg(ObjectiveTermCfg):
    r"""逐 owner/query/sigma 的 Gaussian 密度重建。"""

    name: ClassVar[str] = "density"


@dataclass(frozen=True)
class KappaObjectiveCfg(ObjectiveTermCfg):
    r"""抽样场 Jacobian 元素 $\kappa=\partial d/\partial q_i$ 的监督。"""

    name: ClassVar[str] = "kappa"


@dataclass(frozen=True)
class MultiAnchorGaussianObjectivesCfg:
    r"""rho/kappa 双主 objective 的 typed aggregate；字段为 None 表示显式关闭。"""

    density: DensityObjectiveCfg | None = field(default_factory=DensityObjectiveCfg)
    kappa: KappaObjectiveCfg | None = field(default_factory=KappaObjectiveCfg)

    def enabled(self) -> dict[str, ObjectiveTermCfg]:
        r"""返回开启的 term 名称到配置。"""

        terms = {
            "density": self.density,
            "kappa": self.kappa,
        }
        return {name: config for name, config in terms.items() if config is not None}

    def __post_init__(self) -> None:
        r"""固定 rho/kappa 双主任务与 1:1 normalized vanilla aggregation。"""

        if self.density is None or self.kappa is None:
            raise ValueError("unified multi-anchor method requires both density and kappa objectives")
        if self.density.weight != 1.0 or self.kappa.weight != 1.0:
            raise ValueError("unified multi-anchor method fixes density/kappa normalized weights to 1.0")


@dataclass(frozen=True)
class MultiAnchorGaussianMethodCfg:
    r"""多锚点 Gaussian 隐式场的 representation/model/objectives 聚合配置。"""

    runtime_type: ClassVar[type | None] = None  # 在 method.py 绑定，避免循环 import
    state_measure: JointConfigurationMeasureCfg = field(default_factory=JointConfigurationMeasureCfg)
    representation: GeometryRepresentationCfg = field(default_factory=GeometryRepresentationCfg)
    model: GeometrySSLModelCfg = field(default_factory=GeometrySSLModelCfg)
    objectives: MultiAnchorGaussianObjectivesCfg = field(default_factory=MultiAnchorGaussianObjectivesCfg)
    joint_sign_rewrite: JointSignRewriteCfg = field(default_factory=JointSignRewriteCfg)

    def __post_init__(self) -> None:
        r"""在物化资产前验证方法内部配置类型。"""

        if self.state_measure.kind != "scrambled_sobol_joint_limits":
            raise ValueError("first-round state measure must be scrambled Sobol over joint limits")
        if not isinstance(self.representation, GeometryRepresentationCfg):
            raise TypeError("multi-anchor method requires GeometryRepresentationCfg")
        if not isinstance(self.model, GeometrySSLModelCfg):
            raise TypeError("multi-anchor method requires GeometrySSLModelCfg")
        if tuple(self.representation.field.validation_bandwidths_m) != tuple(
            self.representation.field.bandwidth_centers_m
        ):
            raise ValueError("validation sigma grid must match the three training centers")


__all__ = [
    "DensityObjectiveCfg",
    "JointConfigurationMeasureCfg",
    "JointSignRewriteCfg",
    "KappaObjectiveCfg",
    "MultiAnchorGaussianMethodCfg",
    "MultiAnchorGaussianObjectivesCfg",
    "ObjectiveTermCfg",
]
