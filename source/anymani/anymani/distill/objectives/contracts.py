r"""Objective 可加统计量与 term 结果合同。

这些类型只描述预测与真值比较后的可优化统计，不拥有采样、模型或 optimizer。Trainer 只合并
method 已经归约好的标量，不解释 owner、query 或 edge 轴。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdditiveStatistic:
    r"""一个 loss component 在不等大小 minibatches 间可精确相加的统计量。

    ``numerator / denominator`` 的统计单位由 objective 自己定义。当前方法把每个
    $(asset,q)$ realization 当作一个等权单位，而不是把全部物理标量全局混合。
    """

    name: str  # term 内稳定 component 名，例如 density 或 kappa/active
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
        r"""返回当前统计块的均值，主要用于日志。"""

        return self.numerator / self.denominator


@dataclass(frozen=True)
class ObjectiveTermResult:
    r"""一个独立 objective term 的可优化统计量与只读诊断。"""

    name: str  # 稳定 term 名
    components: tuple[AdditiveStatistic, ...]  # 一个 term 可含多个分别归一化后相加的分支
    metrics: dict[str, torch.Tensor]  # 不参与 reduction 语义的附加诊断

    def __post_init__(self) -> None:
        r"""拒绝空 term、重复 component 名和空名称。"""

        if not self.name or not self.components:
            raise ValueError("objective term requires a name and at least one additive component")
        names = tuple(component.name for component in self.components)
        if len(set(names)) != len(names):
            raise ValueError(f"objective term {self.name!r} contains duplicate component names")


__all__ = ["AdditiveStatistic", "ObjectiveTermResult"]
