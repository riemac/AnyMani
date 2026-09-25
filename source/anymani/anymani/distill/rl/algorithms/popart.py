'Normalize global value targets while preserving the physical Critic output.'

from __future__ import annotations

import torch
from torch import nn


class PopArtValueNormalizer(nn.Module):
    'Normalize return targets while preserving denormalized Critic values.'

    running_mean: torch.Tensor
    running_var: torch.Tensor
    count: torch.Tensor

    def __init__(self, epsilon: float = 1.0e-5) -> None:
        'Initialize the instance.'
        super().__init__()
        if not 0 < epsilon < float("inf"):
            raise ValueError("PopArt epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self.register_buffer("running_mean", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(1, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, value: torch.Tensor, denorm: bool = False, mask=None) -> torch.Tensor:
        'Handle forward.'
        if mask is not None:
            raise ValueError("PopArt statistics require explicit complete rollout returns, not forward masks")
        mean = self.running_mean.to(dtype=value.dtype)
        scale = (self.running_var + self.epsilon).sqrt().to(dtype=value.dtype)
        return value * scale + mean if denorm else (value - mean) / scale

    @torch.no_grad()
    def update_from_returns(self, returns: torch.Tensor, head: nn.Linear) -> dict[str, torch.Tensor]:
        'Update from returns; shapes [B,1].'
        if returns.ndim != 2 or returns.shape[1] != 1 or returns.shape[0] == 0:
            raise ValueError("PopArt returns must be nonempty [B,1]")
        if head.out_features != 1 or head.bias is None:
            raise ValueError("global PopArt requires a scalar Linear head with bias")
        if returns.device != self.running_mean.device or head.weight.device != returns.device:
            raise ValueError("PopArt returns, moments and head must share one device")
        torch._assert_async(torch.isfinite(returns).all(), "PopArt returns must be finite")  # pyright: ignore[reportPrivateImportUsage]
        target = returns.detach().to(dtype=torch.float64)
        old_mean = self.running_mean.clone()
        old_scale = (self.running_var + self.epsilon).sqrt()
        batch_mean = target.mean(dim=0)
        batch_var = target.var(dim=0, unbiased=False)
        batch_count = target.shape[0]
        new_count = self.count + batch_count
        delta = batch_mean - old_mean
        new_mean = old_mean + delta * batch_count / new_count
        new_var = (
            self.running_var * self.count
            + batch_var * batch_count
            + delta.square() * self.count * batch_count / new_count
        ) / new_count
        new_scale = (new_var + self.epsilon).sqrt()


        weight = head.weight.detach().to(dtype=torch.float64)
        bias = head.bias.detach().to(dtype=torch.float64)
        new_weight = (weight * (old_scale / new_scale)).to(dtype=head.weight.dtype)
        new_bias = ((old_scale * bias + old_mean - new_mean) / new_scale).to(dtype=head.bias.dtype)
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.isfinite(new_weight).all() & torch.isfinite(new_bias).all() & torch.isfinite(new_var).all(),
            "PopArt compensation must remain finite",
        )
        weight_error = (new_scale * new_weight.double() - old_scale * weight).abs().max()
        bias_error = (new_scale * new_bias.double() + new_mean - old_scale * bias - old_mean).abs().max()
        head.weight.copy_(new_weight)
        head.bias.copy_(new_bias)
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(new_count)
        return {"weight_error": weight_error, "bias_error": bias_error, "count": new_count.detach()}
