"Defines the proposal, patch, and metadata contracts shared by post-mutate operators."

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...asset_base import AssetCfgBase, HandCfg


@dataclass
class MutatorBaseCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."
    class_type: type[MutatorBase] | None = field(init=False, default=None, repr=False)
    "Associated runtime implementation for this configuration class."


@dataclass(frozen=True)
class SampleSpec:
    "Proposal payload and random-sampling metadata for one mutation term."

    name: str
    distribution: Any


@dataclass
class PatchOp:
    "Atomic edit applied to a hand copy after all proposals are resolved."

    path: tuple[Any, ...]
    apply: Callable[[HandCfg], None] = field(repr=False)
    composer: str | None = None
    "Rigid transform with translation in meters and rotation in radians."

    payload: Any = field(default=None, repr=False)
    "Serialized geometry, proposal, or provenance values associated with this record."

    compose: Callable[[PatchOp], PatchOp] | None = field(default=None, repr=False)
    "Rigid transform with translation in meters and rotation in radians."

    finalize_metadata: Callable[[dict[str, Any]], None] | None = field(default=None, repr=False)
    "Whether to include complete generation and geometry provenance in hand.yaml."

    def merged_with(self, other: PatchOp) -> PatchOp:
        "Returns a merged deferred patch without mutating either input."

        if self.path != other.path:
            raise ValueError(f"cannot merge patch paths {self.path!r} and {other.path!r}")
        if self.composer is None or self.composer != other.composer or self.compose is None:
            raise ValueError(f"post-mutate patch conflict at path {self.path!r}")
        return self.compose(other)


@dataclass
class HandPatch:
    "Ordered geometry edits and provenance for one candidate hand."

    ops: list[PatchOp] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, path: tuple[Any, ...], apply: Callable[[HandCfg], None]) -> None:
        self.ops.append(PatchOp(path=path, apply=apply))

    def add_op(self, op: PatchOp) -> None:
        "Appends one atomic edit to the deferred hand patch."

        self.ops.append(op)

    def extend(self, other: HandPatch) -> None:
        self.ops.extend(other.ops)
        self.merge_metadata(other.metadata)

    def merge_metadata(self, metadata: dict[str, Any]) -> None:
        "Combines proposal metadata while preserving the original source samples."

        for key, value in metadata.items():
            if isinstance(self.metadata.get(key), dict) and isinstance(value, dict):
                merged = dict(self.metadata[key])  # type: ignore[index]
                merged.update(value)
                self.metadata[key] = merged
            else:
                self.metadata[key] = value

    def _finalize_metadata(self) -> None:

        for op in self.ops:
            if op.finalize_metadata is not None:
                op.finalize_metadata(self.metadata)

    def apply(self, target: HandCfg) -> HandCfg:
        self._finalize_metadata()
        mutated = target.copy()
        for op in self.ops:
            op.apply(mutated)
        if self.metadata:
            mutated.metadata = {**mutated.metadata, **self.metadata}
        return mutated.replace(fingers=mutated.fingers, palm=mutated.palm, metadata=dict(mutated.metadata))


class MutatorBase:
    "Common declaration, sampling, and patch protocol for mutation operators."

    cfg: MutatorBaseCfg

    def __init__(self, cfg: MutatorBaseCfg) -> None:
        self.cfg = cfg

    def describe_sampling(self, target: HandCfg) -> dict[str, Any]:
        'Summarizes proposal settings and sample outcomes.'

        return {}

    def plan_patch(self, target: HandCfg, sampled_params: dict[str, Any] | None = None) -> HandPatch:
        "Builds a deferred geometry patch from one unmodified source hand; edits are applied only after sampling finishes."

        return HandPatch()

    def mutate(self, target: HandCfg, *, sampled_params: dict[str, Any] | None = None) -> HandCfg | None:
        "Applies the sampled patch to a hand copy and returns its typed result."

        try:
            return self.plan_patch(target, sampled_params=sampled_params).apply(target)
        except Exception:
            return None


def _make_range_sampler(
    value_range: tuple[float, float],
    *,
    distrib: str | dict[str, Any] = "uniform",
    boundary_policy: str | None = None,
) -> Callable[[], float]:

    low, high = float(value_range[0]), float(value_range[1])
    if low > high:
        low, high = high, low



    distrib_type = distrib.get("type", "uniform") if isinstance(distrib, dict) else distrib
    distrib_type = str(distrib_type).lower()
    policy = boundary_policy or ("none" if distrib_type == "uniform" else "clip")

    def _project(sample: float) -> float:




        if policy in {"clip", "truncate", "resample"}:
            return max(low, min(high, float(sample)))
        return float(sample)

    def _sample_uniform() -> float:

        return random.uniform(low, high)

    def _sample_normal() -> float:



        center = 0.5 * (low + high)
        half_width = max(0.5 * abs(high - low), 1e-12)
        if isinstance(distrib, dict) and "sigma" in distrib:
            sigma = float(distrib["sigma"]) * half_width
        else:
            sigma_rule = float(distrib.get("sigma_rule", 3.0)) if isinstance(distrib, dict) else 3.0
            sigma = half_width / max(abs(sigma_rule), 1e-12)
        return _project(random.gauss(center, sigma))

    if distrib_type == "normal":
        return _sample_normal
    if distrib_type == "uniform":
        return _sample_uniform
    raise ValueError(f"unsupported mutate distribution type: {distrib_type!r}")


def _sample_value(distribution: Any) -> Any:

    return distribution() if callable(distribution) else distribution


__all__ = ["HandPatch", "MutatorBase", "MutatorBaseCfg", "PatchOp", "SampleSpec"]
