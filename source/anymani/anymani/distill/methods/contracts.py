r"""Task-free embodiment method 对 SSL trainer 暴露的窄合同。

本模块不规定所有方法都必须有 source/query/field/target，也不把 representation/model/objectives
写成万能基类字段。Concrete method 对内开放、对外封闭：trainer 只看到 prepare、realize、
forward、reduce、evaluate 与 retained export。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
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

    loss: torch.Tensor  # 当前方法全部启用 objective 的加权总损失，保留计算图
    terms: dict[str, float]  # 各 term 的 $(asset,q)$ 等权均值
    sample_count: int
    denominators: dict[str, float] = field(default_factory=dict)  # 完整 minibatch 的有效 pair 计数


@dataclass(frozen=True)
class MethodEvaluationReport:
    r"""Method 固定评估测度返回给 Trainer 的结构化、只读结果。

    ``metrics`` 只包含 checkpoint selection 使用的标量；``strata`` 保存 owner/query/sigma/distance/
    ancestor 等分层充分统计；``ablations`` 只在冻结 best checkpoint 的 final evaluation 中出现。
    Trainer 不解释这些内部物理轴，只负责跨 suite 等权、promotion 和 artifact 写出。
    """

    metrics: dict[str, float]
    strata: dict[str, object]
    ablations: dict[str, object] | None = None


@runtime_checkable
class MethodSplitSession(Protocol):
    r"""一个 split 的 source、sampler 与 resident device state 封装。"""

    @property
    def asset_count(self) -> int:
        r"""返回当前 split 的真实资产数。"""
        ...

    def realize(self, schedule_item: Any, *, schedule: Any, step: int) -> Any:
        r"""按 Trainer 给出的离散 schedule item 产生 opaque method batch。"""
        ...

    def state_dict(self) -> dict[str, object]:
        r"""返回每资产状态采样 cursor。"""
        ...

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        r"""严格恢复每资产状态采样 cursor。"""
        ...

    def close(self) -> None:
        r"""释放 resident device state 与底层 lease。"""
        ...


@runtime_checkable
class EmbodimentMethod(Protocol):
    r"""SSL trainer 需要的最小运行时行为；内部组件由 concrete method 自己管理。"""

    def prepare(self, catalog: Any, *, device: torch.device, dtype: torch.dtype) -> None:
        r"""物化 CPU sources、推导 padding，并建立每资产 q sampler。"""
        ...

    def split_names(self, role: str) -> tuple[str, ...]:
        r"""返回 validation/evaluation 的具名 suites；train 使用空 suite 名。"""
        ...

    def split_asset_count(self, role: str, *, suite: str = "") -> int:
        r"""返回某个具名 split 的真实资产数；空 suite 必须显式返回 0。"""
        ...

    def asset_manifest(self, catalog: Any) -> dict[str, Any]:
        r"""返回 Method 实际 materialization 后的 train/validation/evaluation provenance。"""
        ...

    def open_session(
        self,
        role: str,
        *,
        suite: str = "",
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        max_resident_assets: int,
        window_factory: Any,
    ) -> MethodSplitSession:
        r"""打开封装 source/sampler/resident state 的 split session。"""
        ...

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> Any:
        r"""在 Trainer 冻结 device/dtype 后一次性构造 learned model。"""
        ...

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        r"""返回 optimizer 唯一允许更新的 learned parameters。"""
        ...

    def train_mode(self) -> None:
        r"""切换 learned method 为训练模式。"""
        ...

    def eval_mode(self) -> None:
        r"""切换 learned method 为评估模式。"""
        ...

    def forward_objectives(
        self,
        batch: Any,
        *,
        step: int,
        mode: str = "train",
        microbatch_size: int | None = None,
    ) -> MethodStep:
        r"""完成一次前向并计算全部开启的 objective terms。"""
        ...

    def reduce_update(self, steps: tuple[MethodStep, ...]) -> MethodUpdate:
        r"""按 $(asset,q)$ 等权合并一个 optimizer update 的 method steps。"""
        ...

    def evaluate_session(
        self,
        session: MethodSplitSession,
        schedule: Any,
        *,
        include_ablations: bool = False,
    ) -> MethodEvaluationReport:
        r"""流式执行固定 Method 测度并返回结构化评估结果。"""
        ...

    def analyze_ablations(
        self,
        evidence: Mapping[str, Any],
        *,
        bootstrap_replicates: int,
        seed: int,
    ) -> dict[str, Any]:
        r"""把 Method-owned 配对 ablation evidence 聚合为最终统计报告。"""
        ...

    def feature_spec(self) -> FeatureSpec:
        r"""返回下游消费的零/一阶合同。"""
        ...

    def training_state_dict(self) -> dict[str, torch.Tensor]:
        r"""返回 full resume 所需的完整 learned state。"""
        ...

    def load_training_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        r"""严格恢复完整 learned state。"""
        ...

    def retained_artifact_payload(self, *, metadata: Mapping[str, Any], source_checkpoint: Path) -> dict[str, Any]:
        r"""构造由 concrete Method 拥有语义的 standalone transfer payload。"""
        ...

    def close(self) -> None:
        r"""释放 method 持有的可回收资源。"""
        ...


__all__ = [
    "AdditiveStatistic",
    "EmbodimentMethod",
    "FeatureSpec",
    "MethodStep",
    "MethodEvaluationReport",
    "MethodSplitSession",
    "MethodUpdate",
    "ObjectiveTermResult",
]
