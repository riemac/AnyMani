"""Typed predictions and truth for geometry pretraining objectives."""


from __future__ import annotations

import torch

from anymani.distill.models.geometry_ssl import GeometrySSLForward

from .batch import PaddedOnlineGeometryBatch


class MultiAnchorObjectiveContext:


    def __init__(
        self,
        *,
        prediction: GeometrySSLForward,
        batch: PaddedOnlineGeometryBatch,
    ) -> None:


        self.prediction = prediction
        self.batch = batch

    @property
    def density_prediction(self) -> torch.Tensor:


        return self.prediction.density

    @property
    def density_target(self) -> torch.Tensor:


        return self.batch.field_targets.density

    @property
    def density_valid_mask(self) -> torch.Tensor:


        return self.batch.field_targets.valid_mask

    @property
    def kappa_prediction(self) -> torch.Tensor:


        return self.prediction.kappa

    @property
    def kappa_target(self) -> torch.Tensor:


        return self.batch.sensitivity_targets.kappa

    @property
    def edge_valid_mask(self) -> torch.Tensor:


        return self.batch.sensitivity_targets.valid_mask

    @property
    def active_mask(self) -> torch.Tensor:


        return self.batch.sensitivity_targets.active_mask

__all__ = ["MultiAnchorObjectiveContext"]
