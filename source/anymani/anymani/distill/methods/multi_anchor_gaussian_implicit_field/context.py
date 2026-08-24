r"""一次 method step 的 typed 三项目标上下文。

derived field 在一个 minibatch 内至多计算一次。Objective 只读取已组装的预测与真值，
不自行重新运行 encoder、decoder 或物理监督源。
"""

from __future__ import annotations

import torch

from anymani.distill.models.geometry_ssl import GeometrySSLForward

from .batch import PaddedOnlineGeometryBatch


class MultiAnchorObjectiveContext:
    r"""三项 objective 共享的 method-local 普通参数梯度上下文。"""

    def __init__(
        self,
        *,
        prediction: GeometrySSLForward,
        batch: PaddedOnlineGeometryBatch,
    ) -> None:
        r"""保存基础 forward；解析组合场只在首次访问时计算。"""

        self.prediction = prediction
        self.batch = batch
        self._derived_field: torch.Tensor | None = None

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

    @property
    def field_sensitivity_target(self) -> torch.Tensor:
        r"""返回 sampled edges、全部实际 sigma 上的解析 $g$。"""

        return self.batch.sensitivity_targets.field_sensitivity

    @staticmethod
    def _select_owner_queries(
        values: torch.Tensor,
        owner_index: torch.Tensor,
        query_index: torch.Tensor,
    ) -> torch.Tensor:
        r"""从 `[B,G,N_Q,...]` 读取共享或逐样本 sampled edge selectors。"""

        if owner_index.ndim == 1:
            return values[:, owner_index, query_index]
        batch_index = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
        return values[batch_index, owner_index, query_index]

    @property
    def derived_field_sensitivity(self) -> torch.Tensor:
        r"""只计算一次 $\hat g^{(\kappa)}=-(d/\sigma^2)\hat\rho\hat\kappa$。"""

        if self._derived_field is None:
            targets = self.batch.sensitivity_targets
            field = self.batch.field_targets
            selected_density = self._select_owner_queries(
                self.prediction.density,
                targets.owner_index,
                targets.query_index,
            )
            selected_distance = self._select_owner_queries(
                field.distance,
                targets.owner_index,
                targets.query_index,
            )
            inverse_sigma_squared = field.bandwidths.square().reciprocal()
            inverse_sigma_squared = (
                inverse_sigma_squared.view(1, 1, -1)
                if inverse_sigma_squared.ndim == 1
                else inverse_sigma_squared.unsqueeze(1)
            )
            self._derived_field = (
                -selected_distance.unsqueeze(-1)
                * inverse_sigma_squared
                * selected_density
                * self.prediction.kappa.unsqueeze(-1)
            )
        return self._derived_field

__all__ = ["MultiAnchorObjectiveContext"]
