r"""一次 method step 的 typed rho/kappa 目标上下文。

Objective 只读取已组装的预测与真值，不自行重新运行 encoder、decoder 或物理监督源。
"""

from __future__ import annotations

import torch

from anymani.distill.models.geometry_ssl import GeometrySSLForward

from .batch import PaddedOnlineGeometryBatch


class MultiAnchorObjectiveContext:
    r"""rho/kappa objective 共享的 method-local 普通参数梯度上下文。"""

    def __init__(
        self,
        *,
        prediction: GeometrySSLForward,
        batch: PaddedOnlineGeometryBatch,
    ) -> None:
        r"""保存基础 forward 与对应 teacher batch。"""

        self.prediction = prediction
        self.batch = batch

    @property
    def density_prediction(self) -> torch.Tensor:
        r"""返回 density decoder 预测 $\hat\rho$，形状 `[B,G,N_Q,N_\sigma]`。"""

        return self.prediction.density

    @property
    def density_target(self) -> torch.Tensor:
        r"""返回特权 density target $\rho$。"""

        return self.batch.field_targets.density

    @property
    def density_valid_mask(self) -> torch.Tensor:
        r"""返回 owner/query 零阶有效 mask。"""

        return self.batch.field_targets.valid_mask

    @property
    def kappa_prediction(self) -> torch.Tensor:
        r"""返回 sampled edges 上的 $\hat\kappa$，形状 `[B,E]`。"""

        return self.prediction.kappa

    @property
    def kappa_target(self) -> torch.Tensor:
        r"""返回解析距离灵敏度 $\kappa$。"""

        return self.batch.sensitivity_targets.kappa

    @property
    def edge_valid_mask(self) -> torch.Tensor:
        r"""返回一阶有效 mask：active 含光滑性，zero 只含拓扑/face。"""

        return self.batch.sensitivity_targets.valid_mask

    @property
    def active_mask(self) -> torch.Tensor:
        r"""返回 joint-first active/zero 分层；True 为 descendant/active。"""

        return self.batch.sensitivity_targets.active_mask

__all__ = ["MultiAnchorObjectiveContext"]
