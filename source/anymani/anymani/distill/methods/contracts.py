r"""Task-free embodiment method 对 SSL trainer 暴露的窄合同。

本模块不规定所有方法都必须有 source/query/field/target，也不把 representation/model/objectives
写成万能基类字段。Concrete method 对内开放、对外封闭：trainer 只看到 prepare、realize、
forward、reduce、evaluate 与 retained export。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from anymani.distill.objectives.contracts import AdditiveStatistic, ObjectiveTermResult


@dataclass(frozen=True)
class FeatureSpec:
    r"""部署期 retained 表征向下游声明的实体轴、关节轴和规范变换语义。"""

    zero_order_width: int  # 每个 PALM/JOINT/TIP entity 的固定宽度 $D_0$
    first_order_width: int | None  # 每个活动 JOINT 的 $D_1$；None 表示该方法不导出一阶包
    entity_axis: str = "PALM/JOINT/TIP owner sequence"
    joint_axis: str = "active JOINT sequence"
    frame_contract: str = "hand frame {h}; in-plane SO(2) gauge"
    zero_order_transform: str = "joint-sign even"
    first_order_transform: str = "joint-sign odd"

    def __post_init__(self) -> None:
        r"""拒绝空表征宽度和无意义的一阶零宽度。"""

        if self.zero_order_width < 1:
            raise ValueError("zero_order_width must be positive")
        if self.first_order_width is not None and self.first_order_width < 1:
            raise ValueError("first_order_width must be positive when first-order features are enabled")


@dataclass(frozen=True)
class MethodStep:
    r"""一次 method forward 的预测与独立 objective 结果。"""

    objectives: dict[str, ObjectiveTermResult]
    sample_count: int  # 本次 forward 中等权的 $(asset,q)$ 数


@dataclass(frozen=True)
class MethodUpdate:
    r"""一个 optimizer update 的标量损失与可记录均值。"""

    loss: torch.Tensor  # 加权五项总损失，保留计算图
    terms: dict[str, float]  # 各 term 的 $(asset,q)$ 等权均值
    sample_count: int


@runtime_checkable
class EmbodimentMethod(Protocol):
    r"""SSL trainer 需要的最小运行时行为；内部组件由 concrete method 自己管理。"""

    def prepare(self, catalog: Any, *, device: torch.device, dtype: torch.dtype) -> None:
        r"""物化 CPU sources、推导 padding，并建立每资产 q sampler。"""
        ...

    def initialize_samplers(self, *, train_seed: int, validation_seeds: dict[str, int]) -> None:
        r"""为 train/validation 资产建立独立 q sampler；trainer 不直接构造 Sobol。"""
        ...

    def make_independent_samplers(self, sources: Any, *, seed: int) -> Any:
        r"""为独立 q-bank 建立不复用训练 cursor 的 sampler。"""
        ...

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> Any:
        r"""在 Trainer 冻结 device/dtype 后一次性构造 learned model。"""
        ...

    def require_model(self) -> Any:
        r"""返回已初始化模型。"""
        ...

    def realize_minibatch(self, schedule_item: Any, **runtime: Any) -> Any:
        r"""由 schedule item 生成一次 method batch，不把 representation 内部字段交给 trainer。"""
        ...

    def forward_objectives(self, batch: Any, *, step: int, mode: str = "train") -> MethodStep:
        r"""完成一次前向并计算全部开启的 objective terms。"""
        ...

    def reduce_update(self, steps: tuple[MethodStep, ...]) -> MethodUpdate:
        r"""按 $(asset,q)$ 等权合并一个 optimizer update 的 method steps。"""
        ...

    def evaluate(self, batches: tuple[Any, ...]) -> dict[str, float]:
        r"""在固定 bank 上按 $(asset,q)$ 等权聚合，不更新参数。"""
        ...

    def feature_spec(self) -> FeatureSpec:
        r"""返回下游消费的零/一阶合同。"""
        ...

    def retained_state_dict(self) -> dict[str, torch.Tensor]:
        r"""返回只含 retained encoder 的 transfer state。"""
        ...

    def close(self) -> None:
        r"""释放 method 持有的可回收资源。"""
        ...


__all__ = [
    "AdditiveStatistic",
    "EmbodimentMethod",
    "FeatureSpec",
    "MethodStep",
    "MethodUpdate",
    "ObjectiveTermResult",
]
