"""Anchor-relational material Jacobian reader. It predicts height, radius, dot, and chirality channels per radian."""


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AnchorRelationalJacobianDecoderCfg:


    latent_width: int = 128
    relation_width: int = 64
    hidden_width: int = 128

    def __post_init__(self) -> None:


        if min(self.latent_width, self.relation_width, self.hidden_width) < 1:
            raise ValueError("material-point Jacobian decoder widths must be positive")


class AnchorRelationalJacobianDecoder(nn.Module):


    def __init__(self, config: AnchorRelationalJacobianDecoderCfg = AnchorRelationalJacobianDecoderCfg()) -> None:


        super().__init__()
        self.config = config
        self.context_projection = nn.Sequential(
            nn.Linear(2 * config.latent_width, config.relation_width),
            nn.GELU(),
            nn.Linear(config.relation_width, config.relation_width),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.relation_width),
            nn.Linear(config.relation_width, config.hidden_width),
            nn.GELU(),
            nn.Linear(config.hidden_width, 4),
        )

    def forward(
        self,
        owner_latent: torch.Tensor,
        joint_latent: torch.Tensor,
        static_pair_feature: torch.Tensor,
    ) -> torch.Tensor:


        if owner_latent.shape != joint_latent.shape or owner_latent.ndim != 3:
            raise ValueError("owner_latent and joint_latent must have identical [B,E,D] shape")
        if owner_latent.shape[-1] != self.config.latent_width:
            raise ValueError("owner/joint latent width does not match decoder config")
        if (
            static_pair_feature.ndim != 4
            or static_pair_feature.shape[:2] != owner_latent.shape[:2]
            or static_pair_feature.shape[-1] != self.config.relation_width
        ):
            raise ValueError("static_pair_feature must have aligned [B,E,K,D_r] shape")


        edge_context = self.context_projection(
            torch.cat((owner_latent, joint_latent), dim=-1)
        )
        fused = static_pair_feature + edge_context.unsqueeze(-2)
        return self.output_projection(fused)  # `[B,E,K,4]`


@dataclass(frozen=True)
class BilinearAnchorRelationalJacobianDecoderCfg:


    latent_width: int = 128
    relation_width: int = 64
    hidden_width: int = 128  # row query MLP width
    readout_rank: int = 64

    def __post_init__(self) -> None:


        if min(self.latent_width, self.relation_width, self.hidden_width, self.readout_rank) < 1:
            raise ValueError("bilinear material-Jacobian decoder widths/rank must be positive")


class BilinearAnchorRelationalJacobianDecoder(nn.Module):


    def __init__(
        self,
        config: BilinearAnchorRelationalJacobianDecoderCfg = BilinearAnchorRelationalJacobianDecoderCfg(),
    ) -> None:
        super().__init__()
        self.config = config
        self.row_projection = nn.Sequential(
            nn.Linear(config.latent_width + config.relation_width, config.hidden_width),
            nn.GELU(),
            nn.Linear(config.hidden_width, 4 * config.readout_rank),
        )  # `[z_g,f_k] -> [4,R]`
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(config.latent_width),
            nn.Linear(config.latent_width, config.readout_rank, bias=False),
        )
        self.query_residual = nn.Linear(config.relation_width, 4)
        self.scale = float(config.readout_rank) ** -0.5

    def forward(
        self,
        owner_latent: torch.Tensor,
        joint_latent: torch.Tensor,
        static_pair_feature: torch.Tensor,
    ) -> torch.Tensor:


        if owner_latent.shape != joint_latent.shape or owner_latent.ndim != 3:
            raise ValueError("owner_latent and joint_latent must have identical [B,E,D] shape")
        if owner_latent.shape[-1] != self.config.latent_width:
            raise ValueError("owner/joint latent width does not match bilinear decoder config")
        if (
            static_pair_feature.ndim != 4
            or static_pair_feature.shape[:2] != owner_latent.shape[:2]
            or static_pair_feature.shape[-1] != self.config.relation_width
        ):
            raise ValueError("static_pair_feature must have aligned [B,E,K,D_r] shape")
        anchor_count = static_pair_feature.shape[2]
        owner_expanded = owner_latent.unsqueeze(-2).expand(-1, -1, anchor_count, -1)  # `[B,E,K,D]`
        row = self.row_projection(torch.cat((owner_expanded, static_pair_feature), dim=-1))
        row = row.view(*row.shape[:-1], 4, self.config.readout_rank)  # `[B,E,K,4,R]`
        column = self.joint_projection(joint_latent).unsqueeze(-2).unsqueeze(-2)  # `[B,E,1,1,R]`
        interaction = torch.sum(row * column, dim=-1) * self.scale  # `[B,E,K,4]`
        return interaction + self.query_residual(static_pair_feature)


__all__ = [
    "AnchorRelationalJacobianDecoder",
    "AnchorRelationalJacobianDecoderCfg",
    "BilinearAnchorRelationalJacobianDecoder",
    "BilinearAnchorRelationalJacobianDecoderCfg",
]
