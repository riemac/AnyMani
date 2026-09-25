"""Conditional Gaussian density and distance-sensitivity readers. Density is unitless and uses metric query and bandwidth inputs."""


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ScalarSigmaFiLMDensityDecoderCfg:


    hidden_width: int = 128
    residual_blocks: int = 2
    sigma_reference_m: float = 0.016

    def __post_init__(self) -> None:


        if self.hidden_width < 1 or self.residual_blocks < 1:
            raise ValueError("density decoder hidden width and residual blocks must be positive")
        if self.sigma_reference_m <= 0.0:
            raise ValueError("sigma_reference_m must be strictly positive")


@dataclass(frozen=True)
class DistanceSensitivityDecoderCfg:


    hidden_width: int = 128
    residual_blocks: int = 2
    readout_rank: int = 64
    physical_scale_m: float = 0.1
    initialization_version: str = "symmetric_nonzero_r_minus_quarter_v1"

    def __post_init__(self) -> None:


        if min(self.hidden_width, self.residual_blocks, self.readout_rank) < 1:
            raise ValueError("sensitivity decoder hidden width and residual blocks must be positive")
        if self.physical_scale_m <= 0.0:
            raise ValueError("sensitivity decoder physical_scale_m must be strictly positive")
        if self.initialization_version != "symmetric_nonzero_r_minus_quarter_v1":
            raise ValueError("unsupported sensitivity decoder initialization version")


@dataclass(frozen=True)
class GeometrySSLDecoderCfg:


    density: ScalarSigmaFiLMDensityDecoderCfg = ScalarSigmaFiLMDensityDecoderCfg()
    sensitivity: DistanceSensitivityDecoderCfg = DistanceSensitivityDecoderCfg()


class _FiLMResidualBlock(nn.Module):


    def __init__(self, hidden_width: int, condition_width: int) -> None:


        super().__init__()
        self.normalization = nn.LayerNorm(hidden_width)
        self.modulation = nn.Linear(condition_width, 2 * hidden_width)
        self.update = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
        )

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:


        gamma, beta = self.modulation(condition).chunk(2, dim=-1)
        modulated = self.normalization(hidden) * (1.0 + gamma) + beta
        return hidden + self.update(modulated)


class ConditionalDensityDecoder(nn.Module):


    def __init__(
        self,
        config: ScalarSigmaFiLMDensityDecoderCfg,
        *,
        entity_width: int,
        query_width: int,
    ) -> None:


        super().__init__()
        self.config = config
        if entity_width < 1 or query_width < 1:
            raise ValueError("density decoder latent/query widths must be positive")
        self.entity_width = entity_width
        self.query_width = query_width
        self.query_projection = nn.Linear(
            query_width + 1,
            config.hidden_width,
        )
        self.blocks = nn.ModuleList(
            _FiLMResidualBlock(config.hidden_width, entity_width)
            for _ in range(config.residual_blocks)
        )
        self.output = nn.Linear(config.hidden_width, 1)

    def forward(
        self,
        owner_latent: torch.Tensor,
        query_features: torch.Tensor,
        bandwidths: torch.Tensor,
    ) -> torch.Tensor:


        if owner_latent.ndim != 3 or query_features.ndim != 4:
            raise ValueError("owner_latent/query_features must have shapes [B,G,D] and [B,G,N_Q,D_q]")
        if owner_latent.shape[:2] != query_features.shape[:2]:
            raise ValueError("owner_latent and query_features must share [B,G] axes")
        if owner_latent.shape[-1] != self.entity_width:
            raise ValueError("owner latent width does not match decoder config")
        if query_features.shape[-1] != self.query_width:
            raise ValueError("query feature width does not match decoder config")
        bandwidths = bandwidths.detach()
        if bandwidths.ndim == 1:
            bandwidths = bandwidths.unsqueeze(0).expand(query_features.shape[0], -1)
        if bandwidths.ndim != 2 or bandwidths.shape[0] != query_features.shape[0]:
            raise ValueError("bandwidths must have shape [N_sigma] or [B,N_sigma]")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(bandwidths > 0.0),
            "bandwidths must be strictly positive",
        )  # pyright: ignore[reportPrivateImportUsage]


        sigma_count = bandwidths.shape[1]
        query = query_features.unsqueeze(3).expand(-1, -1, -1, sigma_count, -1)
        log_sigma = torch.log(bandwidths / self.config.sigma_reference_m)
        log_sigma = log_sigma[:, None, None, :, None].expand(
            -1, query_features.shape[1], query_features.shape[2], -1, -1
        )
        hidden = self.query_projection(torch.cat((query, log_sigma), dim=-1))
        condition = owner_latent[:, :, None, None, :].expand(
            -1, -1, query_features.shape[2], sigma_count, -1
        )
        for block in self.blocks:
            hidden = block(hidden, condition)
        return torch.sigmoid(self.output(hidden)).squeeze(-1)


class DistanceSensitivityDecoder(nn.Module):


    def __init__(
        self,
        config: DistanceSensitivityDecoderCfg,
        *,
        entity_width: int,
        query_width: int,
    ) -> None:


        super().__init__()
        self.config = config
        if min(entity_width, query_width) < 1:
            raise ValueError("sensitivity decoder entity/query widths must be positive")
        self.entity_width = entity_width
        self.query_width = query_width
        self.query_projection = nn.Linear(query_width, config.hidden_width)
        self.blocks = nn.ModuleList(
            _FiLMResidualBlock(config.hidden_width, entity_width)
            for _ in range(config.residual_blocks)
        )
        self.row_normalization = nn.LayerNorm(config.hidden_width)
        self.row_projection = nn.Linear(config.hidden_width, config.readout_rank, bias=False)
        self.joint_projection = nn.Linear(entity_width, config.readout_rank, bias=False)
        self._reset_bilinear_parameters()

    def _reset_bilinear_parameters(self) -> None:


        projected_std = self.config.readout_rank ** (-0.25)
        nn.init.normal_(
            self.row_projection.weight,
            mean=0.0,
            std=projected_std / self.config.hidden_width**0.5,
        )
        nn.init.normal_(
            self.joint_projection.weight,
            mean=0.0,
            std=projected_std / self.entity_width**0.5,
        )

    def forward(
        self,
        owner_latent: torch.Tensor,
        joint_latent: torch.Tensor,
        selected_query: torch.Tensor,
    ) -> torch.Tensor:


        if owner_latent.shape != joint_latent.shape or owner_latent.ndim != 3:
            raise ValueError("owner_latent and joint_latent must share [B,E,D] shape")
        if owner_latent.shape[-1] != self.entity_width:
            raise ValueError("owner/JOINT latent width does not match sensitivity decoder")
        if selected_query.shape != (*owner_latent.shape[:2], self.query_width):
            raise ValueError("selected_query must have shape [B,E,D_q]")
        hidden = self.query_projection(selected_query)
        for block in self.blocks:
            hidden = block(hidden, owner_latent)
        row = self.row_projection(self.row_normalization(hidden))
        column = self.joint_projection(joint_latent)
        return (
            self.config.physical_scale_m
            * (row * column).sum(dim=-1)
            / self.config.readout_rank**0.5
        )


__all__ = [
    "ConditionalDensityDecoder",
    "DistanceSensitivityDecoderCfg",
    "DistanceSensitivityDecoder",
    "GeometrySSLDecoderCfg",
    "ScalarSigmaFiLMDensityDecoderCfg",
]
