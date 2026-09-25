"""Unitless Gaussian density from unsigned distance in metres and bandwidth in metres."""


from __future__ import annotations

import torch


def gaussian_density_from_distance(distance: torch.Tensor, bandwidths: torch.Tensor) -> torch.Tensor:


    sigma = _broadcast_bandwidths(distance, bandwidths)
    if torch.any(bandwidths <= 0):
        raise ValueError("Gaussian bandwidths must be strictly positive")
    if torch.any(distance < 0):
        raise ValueError("unsigned distance cannot contain negative values")

    sigma_squared = sigma.square()
    squared_distance = distance.unsqueeze(-1).square()
    return torch.exp(-squared_distance / (2.0 * sigma_squared))


def field_sensitivity_from_distance(
    distance: torch.Tensor,
    density: torch.Tensor,
    bandwidths: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:


    sigma = _broadcast_bandwidths(distance, bandwidths)
    if density.shape != sigma.shape:
        raise ValueError(
            "density must have shape [*distance.shape, L], "
            f"got distance={tuple(distance.shape)}, density={tuple(density.shape)}, sigma={tuple(sigma.shape)}"
        )
    if kappa.shape[:-1] != distance.shape:
        raise ValueError(
            "kappa must have shape [*distance.shape, N_J], "
            f"got distance={tuple(distance.shape)}, kappa={tuple(kappa.shape)}"
        )
    if torch.any(bandwidths <= 0):
        raise ValueError("Gaussian bandwidths must be strictly positive")

    inverse_sigma_squared = sigma.square().reciprocal()
    radial_factor = -distance.unsqueeze(-1) * inverse_sigma_squared
    return radial_factor.unsqueeze(-1) * density.unsqueeze(-1) * kappa.unsqueeze(-2)


def _broadcast_bandwidths(distance: torch.Tensor, bandwidths: torch.Tensor) -> torch.Tensor:


    if bandwidths.ndim == 1:
        view_shape = (1,) * distance.ndim + (bandwidths.shape[0],)
        return bandwidths.reshape(view_shape).expand(*distance.shape, -1)
    if bandwidths.ndim == 2 and distance.ndim >= 1 and bandwidths.shape[0] == distance.shape[0]:
        view_shape = (bandwidths.shape[0],) + (1,) * (distance.ndim - 1) + (bandwidths.shape[1],)
        return bandwidths.reshape(view_shape).expand(*distance.shape, -1)
    raise ValueError(
        f"bandwidths must have shape [L] or [B,L] matching distance batch, got {tuple(bandwidths.shape)}"
    )


__all__ = ["field_sensitivity_from_distance", "gaussian_density_from_distance"]
