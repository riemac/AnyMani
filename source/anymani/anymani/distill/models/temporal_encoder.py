"""Causal tactile history encoders for the 30-frame policy input at 20 Hz."""


from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F


class TactileTemporalConvEncoder(nn.Module):


    history_length: int = 30
    """Causal history: 30 samples at 20 Hz, covering 1.5 seconds."""

    stage_lengths: tuple[int, int, int] = (12, 8, 4)
    """Output length of the three valid temporal convolutions."""

    def __init__(
        self,
        frame_dim: int = 52,
        latent_dim: int = 64,
        hidden_channels: tuple[int, int, int] = (64, 64, 64),
    ) -> None:


        super().__init__()
        if frame_dim <= 0 or latent_dim <= 0:
            raise ValueError(f"frame_dim and latent_dim must be positive, got {frame_dim=} and {latent_dim=}")
        if len(hidden_channels) != 3 or any(channel <= 0 for channel in hidden_channels):
            raise ValueError(f"hidden_channels must contain three positive widths, got {hidden_channels!r}")

        self.frame_dim = int(frame_dim)
        self.latent_dim = int(latent_dim)
        c1, c2, c3 = (int(channel) for channel in hidden_channels)


        self.conv1 = nn.Conv1d(self.frame_dim, c1, kernel_size=9, stride=2, padding=0)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=5, stride=1, padding=0)
        self.activation = nn.ReLU()


        self.projection = nn.Linear(c3 * self.stage_lengths[-1], self.latent_dim)  # `[B,4c_3] -> [B,d_z]`

    @overload
    def temporal_features(
        self,
        history: torch.Tensor,
        *,
        return_stage_lengths: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def temporal_features(
        self,
        history: torch.Tensor,
        *,
        return_stage_lengths: Literal[True],
    ) -> tuple[torch.Tensor, tuple[int, int, int]]: ...

    def temporal_features(
        self,
        history: torch.Tensor,
        *,
        return_stage_lengths: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[int, int, int]]:


        if history.ndim != 3:
            raise ValueError(f"TCN history must have shape [B,30,{self.frame_dim}], got {tuple(history.shape)}")
        if history.shape[1:] != (self.history_length, self.frame_dim):
            raise ValueError(
                f"TCN history must have per-sample shape {(self.history_length, self.frame_dim)}, "
                f"got {tuple(history.shape[1:])}"
            )

        features = history.transpose(1, 2)
        features = F.pad(features, (1, 0), mode="replicate")
        stage1 = self.activation(self.conv1(features))
        stage2 = self.activation(self.conv2(stage1))
        stage3 = self.activation(self.conv3(stage2))
        lengths = (stage1.shape[-1], stage2.shape[-1], stage3.shape[-1])
        if lengths != self.stage_lengths:
            raise RuntimeError(f"TCN stage lengths changed from {self.stage_lengths} to {lengths}")
        if return_stage_lengths:
            return stage3, lengths
        return stage3

    def forward(self, history: torch.Tensor) -> torch.Tensor:


        temporal_map = self.temporal_features(history)
        flattened = temporal_map.flatten(start_dim=1)
        return self.projection(flattened)


class PerJointTactileTemporalEncoder(nn.Module):


    def __init__(
        self,
        *,
        joint_count: int = 16,
        frame_dim: int = 4,
        latent_dim: int = 64,
        hidden_channels: tuple[int, int, int] = (64, 64, 64),
    ) -> None:


        super().__init__()
        if joint_count < 1:
            raise ValueError("per-joint temporal encoder joint_count must be positive")
        self.joint_count = int(joint_count)
        self.frame_dim = int(frame_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = TactileTemporalConvEncoder(
            frame_dim=self.frame_dim,
            latent_dim=self.latent_dim,
            hidden_channels=hidden_channels,
        )

    def forward(self, history: torch.Tensor, joint_valid_mask: torch.Tensor) -> torch.Tensor:


        if history.ndim != 4 or history.shape[1:] != (
            self.encoder.history_length,
            self.joint_count,
            self.frame_dim,
        ):
            raise ValueError(
                "per-joint history must have shape "
                f"[B,{self.encoder.history_length},{self.joint_count},{self.frame_dim}], got {tuple(history.shape)}"
            )
        batch_size = history.shape[0]
        if joint_valid_mask.shape != (batch_size, self.joint_count) or joint_valid_mask.dtype != torch.bool:
            raise ValueError(f"joint_valid_mask must be bool [{batch_size},{self.joint_count}]")


        joint_sequences = history.permute(0, 2, 1, 3).reshape(
            batch_size * self.joint_count,
            self.encoder.history_length,
            self.frame_dim,
        )  # `[BN_J,30,F]`
        temporal = self.encoder(joint_sequences).reshape(
            batch_size,
            self.joint_count,
            self.latent_dim,
        )  # `[B,N_J,D_t]`
        return temporal * joint_valid_mask.unsqueeze(-1).to(dtype=temporal.dtype)


class PerJointHistoryStackEncoder(nn.Module):


    def __init__(self, *, joint_count: int = 16, frame_dim: int = 4, latent_dim: int = 32) -> None:


        super().__init__()
        if min(joint_count, frame_dim, latent_dim) < 1:
            raise ValueError("per-joint stack encoder dimensions must be positive")
        self.joint_count = int(joint_count)
        self.frame_dim = int(frame_dim)
        self.latent_dim = int(latent_dim)
        self.history_length = 30
        stack_width = self.history_length * self.frame_dim
        self.network = nn.Sequential(
            nn.LayerNorm(stack_width),
            nn.Linear(stack_width, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, history: torch.Tensor, joint_valid_mask: torch.Tensor) -> torch.Tensor:


        if history.ndim != 4 or history.shape[1:] != (
            self.history_length,
            self.joint_count,
            self.frame_dim,
        ):
            raise ValueError(
                f"per-joint stack history must have shape [B,30,{self.joint_count},{self.frame_dim}], "
                f"got {tuple(history.shape)}"
            )
        batch_size = history.shape[0]
        if joint_valid_mask.shape != (batch_size, self.joint_count) or joint_valid_mask.dtype != torch.bool:
            raise ValueError(f"joint_valid_mask must be bool [{batch_size},{self.joint_count}]")
        stacked = history.permute(0, 2, 1, 3).reshape(batch_size, self.joint_count, -1)  # `[B,N_J,120]`
        temporal = self.network(stacked)
        return temporal * joint_valid_mask.unsqueeze(-1).to(dtype=temporal.dtype)


class PerJointRawHistoryStack(nn.Module):


    history_length: int = 30
    """The same 30-sample, 1.5-second observation window as the TCN."""

    def __init__(self, *, joint_count: int = 16, frame_dim: int = 5) -> None:


        super().__init__()
        if joint_count < 1 or frame_dim < 1:
            raise ValueError("per-joint raw history dimensions must be positive")
        self.joint_count = int(joint_count)
        self.frame_dim = int(frame_dim)
        self.output_dim = self.history_length * self.frame_dim

    def forward(self, history: torch.Tensor, joint_valid_mask: torch.Tensor) -> torch.Tensor:


        expected = (self.history_length, self.joint_count, self.frame_dim)  # sample-level History30 shape
        if history.ndim != 4 or history.shape[1:] != expected:
            raise ValueError(
                "per-joint raw history must have shape "
                f"[B,{self.history_length},{self.joint_count},{self.frame_dim}], got {tuple(history.shape)}"
            )
        batch_size = history.shape[0]
        if joint_valid_mask.shape != (batch_size, self.joint_count) or joint_valid_mask.dtype != torch.bool:
            raise ValueError(f"joint_valid_mask must be bool [{batch_size},{self.joint_count}]")
        stacked = history.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.joint_count,
            self.output_dim,
        )
        return stacked * joint_valid_mask.unsqueeze(-1).to(dtype=stacked.dtype)


__all__ = [
    "PerJointHistoryStackEncoder",
    "PerJointRawHistoryStack",
    "PerJointTactileTemporalEncoder",
    "TactileTemporalConvEncoder",
]
