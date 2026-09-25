"""Additive objective statistics and typed objective results."""


from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdditiveStatistic:


    name: str
    numerator: torch.Tensor
    denominator: torch.Tensor

    def __post_init__(self) -> None:


        if self.numerator.ndim != 0 or self.denominator.ndim != 0:
            raise ValueError("additive statistic numerator and denominator must be scalar tensors")
        if not torch.isfinite(self.numerator) or not torch.isfinite(self.denominator):
            raise ValueError("additive statistic values must be finite")
        if float(self.denominator.detach()) <= 0.0:
            raise ValueError("additive statistic denominator must be positive")

    @property
    def mean(self) -> torch.Tensor:


        return self.numerator / self.denominator


@dataclass(frozen=True)
class ObjectiveTermResult:


    name: str
    components: tuple[AdditiveStatistic, ...]
    metrics: dict[str, torch.Tensor]

    def __post_init__(self) -> None:


        if not self.name or not self.components:
            raise ValueError("objective term requires a name and at least one additive component")
        names = tuple(component.name for component in self.components)
        if len(set(names)) != len(names):
            raise ValueError(f"objective term {self.name!r} contains duplicate component names")


__all__ = ["AdditiveStatistic", "ObjectiveTermResult"]
