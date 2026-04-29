"""Legacy scalar distribution adapter for post-mutate sampling.

`ScalarDistributionCfg` is intentionally a compatibility/helper layer, not the
research-facing abstraction for every mutator. Each mutator may expose richer
fields and lower them into scalar sample variables only at pipeline time.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Literal

from ...asset_base import AssetCfgBase


@dataclass
class ScalarDistributionCfg(AssetCfgBase):
    """Small scalar distribution descriptor used by legacy recipes/tests."""

    kind: Literal["fixed", "uniform", "normal"] = "fixed"
    value: float | None = None
    low: float | None = None
    high: float | None = None
    mean: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()  # type: ignore[assignment]
        if self.kind not in {"fixed", "uniform", "normal"}:
            raise ValueError(f"unsupported scalar distribution kind: {self.kind!r}")
        if self.value is not None:
            self.value = float(self.value)
        if self.low is not None:
            self.low = float(self.low)
        if self.high is not None:
            self.high = float(self.high)
        self.mean = float(self.mean)
        self.sigma = float(self.sigma)
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")


def normalize_distribution(value: Any, *, default: ScalarDistributionCfg | None = None) -> Any:
    """Normalize common shorthand without forcing all configs to use it publicly."""

    if value is None:
        return default if default is not None else ScalarDistributionCfg(kind="fixed", value=0.0)
    if isinstance(value, ScalarDistributionCfg):
        return value
    if isinstance(value, dict):
        return ScalarDistributionCfg(**value)
    if isinstance(value, (int, float)):
        return ScalarDistributionCfg(kind="fixed", value=float(value))
    return value


def sample_scalar_distribution(cfg: Any, *, rng: random.Random | None = None) -> float:
    """Sample one scalar from a legacy/simple distribution descriptor."""

    normalized = normalize_distribution(cfg)
    random_source = rng or random
    if isinstance(normalized, ScalarDistributionCfg):
        if normalized.kind == "fixed":
            return float(0.0 if normalized.value is None else normalized.value)
        if normalized.kind == "uniform":
            low = normalized.low if normalized.low is not None else -1.0
            high = normalized.high if normalized.high is not None else 1.0
            return float(random_source.uniform(low, high))
        if normalized.kind == "normal":
            return float(random_source.gauss(normalized.mean, normalized.sigma))
    if callable(normalized):
        return float(normalized())
    return float(normalized)


def sample_scalar_distribution_batch(cfg: Any, *, batch_size: int, prefer_cuda: bool = True) -> list[float]:
    """Sample a batch, using torch/CUDA when available and applicable."""

    normalized = normalize_distribution(cfg)
    if batch_size <= 0:
        return []
    if not isinstance(normalized, ScalarDistributionCfg):
        return [sample_scalar_distribution(normalized) for _ in range(batch_size)]

    if prefer_cuda:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if normalized.kind == "fixed":
                value = float(0.0 if normalized.value is None else normalized.value)
                tensor = torch.full((batch_size,), value, device=device)
            elif normalized.kind == "uniform":
                low = normalized.low if normalized.low is not None else -1.0
                high = normalized.high if normalized.high is not None else 1.0
                tensor = torch.rand((batch_size,), device=device) * (high - low) + low
            else:
                tensor = torch.randn((batch_size,), device=device) * normalized.sigma + normalized.mean
            return [float(value) for value in tensor.cpu().tolist()]
        except Exception:
            pass

    return [sample_scalar_distribution(normalized) for _ in range(batch_size)]


__all__ = [
    "ScalarDistributionCfg",
    "normalize_distribution",
    "sample_scalar_distribution",
    "sample_scalar_distribution_batch",
]
