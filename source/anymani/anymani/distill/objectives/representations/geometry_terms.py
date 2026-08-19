r"""多锚点场方法的六个独立 objective terms。

每个 term 只拥有一个数学约束及其归约统计，不拥有模型 forward、query/target realization、JVP
调度或 optimizer。``MultiAnchorGaussianMethod`` 提供一次 minibatch 内共享的 context，使派生场、
密度自导数和 paired second forward 不会被多个 term 重复计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch

from anymani.distill.ssl.contracts import AdditiveStatistic, ObjectiveTermResult


class MultiAnchorObjectiveContext(Protocol):
    r"""六项 objective 读取的 method-local、带 autograd 图共享上下文。"""

    density_prediction: torch.Tensor  # `[B,G,N_Q,N_sigma]`
    density_target: torch.Tensor  # 与 prediction 同形状
    density_valid_mask: torch.Tensor  # `[B,G,N_Q]`
    kappa_prediction: torch.Tensor  # `[B,E]`
    kappa_target: torch.Tensor  # `[B,E]`
    edge_valid_mask: torch.Tensor  # `[B,E]`
    field_sensitivity_target: torch.Tensor  # `[B,E,N_sigma]`

    @property
    def derived_field_sensitivity(self) -> torch.Tensor:
        r"""返回共享的解析派生预测 $\hat g^{(\kappa)}$。"""

        ...

    @property
    def auto_field_sensitivity(self) -> torch.Tensor:
        r"""返回共享的密度预测 q-JVP $\hat g^{auto}$。"""

        ...

    @property
    def paired_additive_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""返回零阶偶性和一阶奇性的两个 numerator/denominator 对。"""

        ...


def _masked_square_components(error: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""按 term 声明的有效标量轴返回平方误差 numerator/denominator。"""

    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)  # owner/query 或 edge mask 广播到 sigma 等尾轴
    weight = mask.to(error.dtype).expand_as(error)  # 每个有效物理标量权重为 1
    denominator = weight.sum()  # 当前 term 的有效标量总数
    if int(denominator.detach().item()) == 0:
        raise ValueError("objective term received no valid supervision scalars")
    return torch.sum(weight * error.square()), denominator


@dataclass(frozen=True)
class GeometryObjectiveTermCfg:
    r"""当前 canonical objective term 的共同权重与一次性校准声明。"""

    weight: float = 1.0  # 无量纲；校准结果另存 runtime evidence，不改写本字段
    calibrate: bool = True  # 是否参与 train-only shared-encoder gradient calibration

    def __post_init__(self) -> None:
        r"""允许权重为零做显式消融，但拒绝负权重。"""

        if self.weight < 0.0:
            raise ValueError("objective term weight must be non-negative")


class _GeometryObjectiveTerm:
    r"""保存一个 concrete term 的声明权重；子类只实现 :meth:`evaluate`。"""

    name: ClassVar[str]
    required_nodes: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: GeometryObjectiveTermCfg) -> None:
        r"""绑定冻结配置，不创建参数或运行时缓存。"""

        self.config = config

    @property
    def weight(self) -> float:
        r"""返回本次声明的无量纲 term 权重。"""

        return self.config.weight


class DensityObjectiveTerm(_GeometryObjectiveTerm):
    r"""逐 owner/query/sigma Gaussian density 重建项。"""

    name = "density"

    def evaluate(self, context: MultiAnchorObjectiveContext) -> ObjectiveTermResult:
        r"""计算 density prediction 与特权 target 的有效标量平方误差。"""

        numerator, denominator = _masked_square_components(
            context.density_prediction - context.density_target,
            context.density_valid_mask,
        )
        statistic = AdditiveStatistic(self.name, numerator, denominator)
        return ObjectiveTermResult(self.name, (statistic,), {"loss": statistic.mean})


@dataclass(frozen=True)
class DensityObjectiveTermCfg(GeometryObjectiveTermCfg):
    r"""Density 重建 term 配置。"""

    runtime_type: ClassVar[type[DensityObjectiveTerm]] = DensityObjectiveTerm


class KappaObjectiveTerm(_GeometryObjectiveTerm):
    r"""抽样 owner-query-JOINT edges 上的距离灵敏度项。"""

    name = "kappa"

    def evaluate(self, context: MultiAnchorObjectiveContext) -> ObjectiveTermResult:
        r"""计算 $\hat\kappa$ 与解析距离灵敏度的有效 edge MSE。"""

        numerator, denominator = _masked_square_components(
            context.kappa_prediction - context.kappa_target,
            context.edge_valid_mask,
        )
        statistic = AdditiveStatistic(self.name, numerator, denominator)
        return ObjectiveTermResult(self.name, (statistic,), {"loss": statistic.mean})


@dataclass(frozen=True)
class KappaObjectiveTermCfg(GeometryObjectiveTermCfg):
    r"""距离灵敏度 term 配置。"""

    runtime_type: ClassVar[type[KappaObjectiveTerm]] = KappaObjectiveTerm


class DerivedFieldObjectiveTerm(_GeometryObjectiveTerm):
    r"""由预测 density 与 kappa 解析得到的场灵敏度监督项。"""

    name = "derived_field"
    required_nodes = frozenset({"derived_field"})

    def evaluate(self, context: MultiAnchorObjectiveContext) -> ObjectiveTermResult:
        r"""约束 $\hat g^{(\kappa)}$ 对齐特权场灵敏度。"""

        prediction = context.derived_field_sensitivity
        numerator, denominator = _masked_square_components(
            prediction - context.field_sensitivity_target,
            context.edge_valid_mask,
        )
        statistic = AdditiveStatistic(self.name, numerator, denominator)
        return ObjectiveTermResult(
            self.name,
            (statistic,),
            {"loss": statistic.mean, "prediction": prediction},
        )


@dataclass(frozen=True)
class DerivedFieldObjectiveTermCfg(GeometryObjectiveTermCfg):
    r"""解析派生场灵敏度 term 配置。"""

    runtime_type: ClassVar[type[DerivedFieldObjectiveTerm]] = DerivedFieldObjectiveTerm


class SobolevObjectiveTerm(_GeometryObjectiveTerm):
    r"""同一 density predictor 对物理 q 的自动微分监督项。"""

    name = "sobolev"
    required_nodes = frozenset({"auto_field"})

    def evaluate(self, context: MultiAnchorObjectiveContext) -> ObjectiveTermResult:
        r"""约束固定 query/sigma 下的 $\partial\hat\rho/\partial q$ 对齐教师。"""

        prediction = context.auto_field_sensitivity
        numerator, denominator = _masked_square_components(
            prediction - context.field_sensitivity_target,
            context.edge_valid_mask,
        )
        statistic = AdditiveStatistic(self.name, numerator, denominator)
        return ObjectiveTermResult(
            self.name,
            (statistic,),
            {"loss": statistic.mean, "prediction": prediction},
        )


@dataclass(frozen=True)
class SobolevObjectiveTermCfg(GeometryObjectiveTermCfg):
    r"""Density q-JVP term 配置。"""

    runtime_type: ClassVar[type[SobolevObjectiveTerm]] = SobolevObjectiveTerm


class ChainObjectiveTerm(_GeometryObjectiveTerm):
    r"""解析 kappa 路径与 density 自导数路径之间的一致性项。"""

    name = "chain"
    required_nodes = frozenset({"derived_field", "auto_field"})

    def evaluate(self, context: MultiAnchorObjectiveContext) -> ObjectiveTermResult:
        r"""约束两条预测场灵敏度路径彼此一致。"""

        numerator, denominator = _masked_square_components(
            context.derived_field_sensitivity - context.auto_field_sensitivity,
            context.edge_valid_mask,
        )
        statistic = AdditiveStatistic(self.name, numerator, denominator)
        return ObjectiveTermResult(self.name, (statistic,), {"loss": statistic.mean})


@dataclass(frozen=True)
class ChainObjectiveTermCfg(GeometryObjectiveTermCfg):
    r"""场灵敏度 chain consistency term 配置。"""

    runtime_type: ClassVar[type[ChainObjectiveTerm]] = ChainObjectiveTerm


class PairedParityObjectiveTerm(_GeometryObjectiveTerm):
    r"""真实 joint-sign 坐标改写下零阶偶与一阶奇的成对表征项。"""

    name = "paired"
    required_nodes = frozenset({"paired_forward"})

    def evaluate(self, context: MultiAnchorObjectiveContext) -> ObjectiveTermResult:
        r"""分别保留零阶和一阶 MSE 分母，避免尾 minibatch 改变两支相对权重。"""

        zero_numerator, zero_denominator, first_numerator, first_denominator = context.paired_additive_components
        zero = AdditiveStatistic("zero_order", zero_numerator, zero_denominator)
        first = AdditiveStatistic("first_order", first_numerator, first_denominator)
        return ObjectiveTermResult(self.name, (zero, first), {"loss": zero.mean + first.mean})


@dataclass(frozen=True)
class PairedParityObjectiveTermCfg(GeometryObjectiveTermCfg):
    r"""零阶偶/一阶奇 paired parity term 配置。"""

    runtime_type: ClassVar[type[PairedParityObjectiveTerm]] = PairedParityObjectiveTerm


__all__ = [
    "ChainObjectiveTermCfg",
    "DensityObjectiveTermCfg",
    "DerivedFieldObjectiveTermCfg",
    "GeometryObjectiveTermCfg",
    "KappaObjectiveTermCfg",
    "MultiAnchorObjectiveContext",
    "PairedParityObjectiveTermCfg",
    "SobolevObjectiveTermCfg",
]
