r"""多锚点 Gaussian 隐式场方法的强类型装配配置。

一份方法就是
$$
\mathcal M=\left(\mu_q,\mathcal R,f_\theta,\mathcal L,\mathcal A\right):
$$
`state_measure` 定义 $q$ 的测度，`representation` 定义物理真值体系，`model` 是可学习映射，
`objectives` 定义 rho/kappa 两项训练约束，`entity_permutation` 与 `joint_sign_rewrite` 是两个独立
增强随机域。Trainer 预算和 resident window 不属于本配置。
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
class EntityPermutationCfg:
    r"""每个资产 q-block 的真实 PALM/JOINT/TIP entity 轴随机重标号。

    同一资产连续 8 个 q 共用一个双射 $P$；padding slot、JOINT coordinate axis、anchor axis 与 token
    通道轴都不参与置换。``seed_offset`` 与 joint-sign 随机域分离，resume 只需恢复绝对 block 身份。
    """

    enabled: bool = True
    seed_offset: int = 31_337

    def __post_init__(self) -> None:
        """拒绝负 seed domain，避免不同语言的无符号转换产生不一致身份。"""

        if self.seed_offset < 0:
            raise ValueError("entity permutation seed_offset must be non-negative")


@dataclass(frozen=True)
class FairGradCfg:
    r"""两任务 shared encoder 的解析 $\alpha=1$ FairGrad 身份与反向冲突边界。"""

    algorithm: str = "fairgrad_alpha_1_two_task_analytic_v1"
    near_opposition_tolerance: float = 1.0e-6  # 当 $1+\cos(g_\rho,g_\kappa)$ 不超过此值时阻塞 shared

    def __post_init__(self) -> None:
        """只接受当前已验证的解析公式与开区间数值容差。"""

        if self.algorithm != "fairgrad_alpha_1_two_task_analytic_v1":
            raise ValueError("unsupported shared-gradient aggregation algorithm")
        if not 0.0 < self.near_opposition_tolerance < 1.0:
            raise ValueError("FairGrad near-opposition tolerance must lie in (0,1)")


@dataclass(frozen=True)
class ObjectiveTermCfg:
    r"""无 term-specific 超参的 objective 声明。

    `func` 用 ClassVar 绑定模块级 callable，不进入 OmegaConf structured config。
    density 与 $\kappa$ 不再携带固定 scalar task weight；shared 优先级由 FairGrad 梯度几何决定，
    private reader 各自只接收本任务梯度。
    """

    func: ClassVar[Callable[..., ObjectiveTermResult] | None] = None

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
        r"""固定 rho/kappa 双主任务；任务间不再通过 scalar loss 求和。"""

        if self.density is None or self.kappa is None:
            raise ValueError("unified multi-anchor method requires both density and kappa objectives")


@dataclass(frozen=True)
class MultiAnchorGaussianMethodCfg:
    r"""多锚点 Gaussian 隐式场的 representation/model/objectives 聚合配置。"""

    runtime_type: ClassVar[type | None] = None  # 在 method.py 绑定，避免循环 import
    state_measure: JointConfigurationMeasureCfg = field(default_factory=JointConfigurationMeasureCfg)
    representation: GeometryRepresentationCfg = field(default_factory=GeometryRepresentationCfg)
    model: GeometrySSLModelCfg = field(default_factory=GeometrySSLModelCfg)
    objectives: MultiAnchorGaussianObjectivesCfg = field(default_factory=MultiAnchorGaussianObjectivesCfg)
    fairgrad: FairGradCfg = field(default_factory=FairGradCfg)
    entity_permutation: EntityPermutationCfg = field(default_factory=EntityPermutationCfg)
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
    "EntityPermutationCfg",
    "FairGradCfg",
    "JointConfigurationMeasureCfg",
    "JointSignRewriteCfg",
    "KappaObjectiveCfg",
    "MultiAnchorGaussianMethodCfg",
    "MultiAnchorGaussianObjectivesCfg",
    "ObjectiveTermCfg",
]
