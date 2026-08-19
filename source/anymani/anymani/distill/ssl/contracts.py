r"""Embodiment pretraining 组件装配与跨阶段数据语义合同。

本模块只定义稳定边界，不定义 hand、Gaussian field、网络结构或损失公式。具体配置通过
``ClassVar runtime_type`` 绑定本地运行时；Hydra 只组合 concrete dataclass，最高 façade 不维护
``kind -> constructor`` 注册表，也不解析任何组件内部字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch


@runtime_checkable
class RuntimeBoundCfg(Protocol):
    r"""所有可由最高 façade 构造的 concrete 配置必须满足的最小协议。"""

    runtime_type: ClassVar[type[Any]]  # 不进入 Hydra/YAML，只绑定同一 owner 的 runtime


def build_runtime(config: RuntimeBoundCfg) -> Any:
    r"""只通过 concrete cfg 自己声明的 ``runtime_type`` 构造运行时。

    Args:
        config (CfgT): Hydra compose 后恢复的 concrete dataclass。

    Returns:
        Any: 对应 role runtime；具体类型由调用方的 role contract 收窄。

    Raises:
        TypeError: 配置未声明可调用的 ``runtime_type`` 时抛出。
    """

    runtime_type = getattr(type(config), "runtime_type", None)  # ClassVar 必须来自 concrete cfg 类型
    if runtime_type is None or not callable(runtime_type):
        raise TypeError(f"pretraining config {type(config).__name__} does not declare a callable runtime_type")
    return runtime_type(config)  # 所有 runtime 构造阶段只保存 cfg，不执行 IO/CUDA


@dataclass(frozen=True)
class FeatureSpec:
    r"""部署期 retained 表征向下游声明的实体轴、关节轴和规范变换语义。

    框架要求零阶实体序列；逐 JOINT 一阶序列是可选能力。该对象只描述类型，不携带任何
    learned tensor，也不规定当前方法内部如何产生零阶或一阶表征。
    """

    zero_order_width: int  # 每个 PALM/JOINT/TIP entity 的固定宽度 $D_0$
    first_order_width: int | None  # 每个活动 JOINT 的 $D_1$；None 表示该方法不导出一阶包
    entity_axis: str = "PALM/JOINT/TIP owner sequence"  # 下游不得无条件全局池化掉 owner 来源
    joint_axis: str = "active JOINT sequence"  # 按资产结构模式对齐，不是跨手固定动作槽
    frame_contract: str = "hand frame {h}; in-plane SO(2) gauge"  # reflection/chirality 仍是物理差异
    zero_order_transform: str = "joint-sign even"  # 成对关节坐标改写下保持不变
    first_order_transform: str = "joint-sign odd"  # 仅 first_order_width 非空时有效

    def __post_init__(self) -> None:
        r"""拒绝空表征宽度和无意义的一阶零宽度。"""

        if self.zero_order_width < 1:
            raise ValueError("zero_order_width must be positive")
        if self.first_order_width is not None and self.first_order_width < 1:
            raise ValueError("first_order_width must be positive when first-order features are enabled")


@dataclass(frozen=True)
class AdditiveStatistic:
    r"""一个 loss component 在不等大小 minibatches 间可精确相加的统计量。

    ``numerator / denominator`` 的具体统计单位由 objective term 自己定义。Trainer 只跨本次
    optimizer update 的 minibatches 求和，不解释 owner、query、edge 或 latent channel 轴。
    """

    name: str  # term 内稳定 component 名，例如 density 或 paired/zero_order
    numerator: torch.Tensor  # 与模型计算图相连的标量误差总和
    denominator: torch.Tensor  # 无梯度的有效统计单位数，严格为正

    def __post_init__(self) -> None:
        r"""验证两个量都是有限标量，且 denominator 严格为正。"""

        if self.numerator.ndim != 0 or self.denominator.ndim != 0:
            raise ValueError("additive statistic numerator and denominator must be scalar tensors")
        if not torch.isfinite(self.numerator) or not torch.isfinite(self.denominator):
            raise ValueError("additive statistic values must be finite")
        if float(self.denominator.detach()) <= 0.0:
            raise ValueError("additive statistic denominator must be positive")

    @property
    def mean(self) -> torch.Tensor:
        r"""返回当前统计块的均值，主要用于日志；梯度累计使用合并后的分母。"""

        return self.numerator / self.denominator


@dataclass(frozen=True)
class ObjectiveTermResult:
    r"""一个独立 objective term 的可优化统计量与只读诊断。"""

    name: str  # Hydra mapping 与日志共同使用的稳定 term 名
    components: tuple[AdditiveStatistic, ...]  # 一个 term 可含多个分别归一化后相加的分支
    metrics: dict[str, torch.Tensor]  # 不参与 reduction 语义的附加标量/稠密诊断

    def __post_init__(self) -> None:
        r"""拒绝空 term、重复 component 名和空名称。"""

        if not self.name or not self.components:
            raise ValueError("objective term requires a name and at least one additive component")
        names = tuple(component.name for component in self.components)
        if len(set(names)) != len(names):
            raise ValueError(f"objective term {self.name!r} contains duplicate component names")


__all__ = [
    "AdditiveStatistic",
    "FeatureSpec",
    "ObjectiveTermResult",
    "RuntimeBoundCfg",
    "build_runtime",
]
