"""Post-mutate pipeline based on joint sampling and deferred HandCfg patches."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

from ...asset_base import AssetCfgBase, HandCfg
from ._base import HandPatch, MutatorBase
from ._distribution import sample_scalar_distribution_batch


@dataclass
class MutatorTerm(AssetCfgBase):
    """One entry in the post-mutate term container."""

    cfg: Any
    enabled: bool = True


@dataclass
class HandMutatorCfg(AssetCfgBase):
    r"""Open term container for post-mutate.

    Terms may be supplied either through the `terms` dict or as class attributes
    on subclasses, matching the IsaacLab-style cfg pattern used elsewhere.
    """

    class_type: type["HandMutator"] | None = None
    terms: dict[str, MutatorTerm] = field(default_factory=dict)
    order: tuple[str, ...] = ()
    on_reject: Literal["abort", "skip"] = "abort"
    step_validate: bool = False
    prefer_cuda_sampling: bool = True

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = HandMutator
        normalized: dict[str, MutatorTerm] = {}

        # Class-level MutatorTerm declarations are copied first, then instance
        # terms override them. This keeps handwritten research cfgs compact.
        for cls in reversed(type(self).mro()):
            for name, value in cls.__dict__.items():
                if isinstance(value, MutatorTerm):
                    normalized[name] = value.copy()
        for name, term in self.terms.items():
            normalized[name] = term if isinstance(term, MutatorTerm) else MutatorTerm(cfg=term)
        self.terms = normalized

        if isinstance(self.order, list):
            self.order = tuple(self.order)
        if not self.order:
            self.order = tuple(self.terms)

    def has_terms(self) -> bool:
        return any(term.enabled for _, term in self.ordered_terms())

    def ordered_terms(self) -> list[tuple[str, MutatorTerm]]:
        ordered: "OrderedDict[str, MutatorTerm]" = OrderedDict()
        for name in self.order:
            if name in self.terms:
                ordered[name] = self.terms[name]
        for name, term in self.terms.items():
            ordered.setdefault(name, term)
        return [(name, term) for name, term in ordered.items() if term.enabled]


class HandMutator:
    """Runtime that samples all terms jointly, composes patches, and applies once."""

    cfg: HandMutatorCfg

    def __init__(self, cfg: HandMutatorCfg):
        self.cfg = cfg

    def _make_runtime(self, term: MutatorTerm) -> MutatorBase:
        cfg = term.cfg
        runtime_cls = getattr(cfg, "class_type", None)
        if runtime_cls is None:
            raise TypeError(f"mutator cfg {cfg!r} does not define class_type")
        return runtime_cls(cfg)

    def describe_sampling(self, target: HandCfg) -> dict[str, dict[str, Any]]:
        """Return the joint sample plan as term -> local variable -> distribution."""

        plan: dict[str, dict[str, Any]] = {}
        for name, term in self.cfg.ordered_terms():
            runtime = self._make_runtime(term)
            plan[name] = dict(runtime.describe_sampling(target))
        return plan

    def sample_batch(self, target: HandCfg, *, batch_size: int) -> list[dict[str, dict[str, float]]]:
        """Sample `batch_size` independent joint parameter assignments."""

        sample_plan = self.describe_sampling(target)
        batch: list[dict[str, dict[str, float]]] = [
            {term_name: {} for term_name in sample_plan}
            for _ in range(max(int(batch_size), 0))
        ]
        for term_name, distribution_map in sample_plan.items():
            for local_name, distribution in distribution_map.items():
                values = sample_scalar_distribution_batch(
                    distribution,
                    batch_size=len(batch),
                    prefer_cuda=self.cfg.prefer_cuda_sampling,
                )
                for sample, value in zip(batch, values, strict=True):
                    sample[term_name][local_name] = value
        return batch

    def plan_patch(
        self,
        target: HandCfg,
        *,
        sampled_params: dict[str, dict[str, Any]] | None = None,
    ) -> HandPatch:
        """Compose all term patches before mutating any `HandCfg` object."""

        sampled_params = sampled_params or {}
        composed = HandPatch()
        touched_paths: set[tuple[Any, ...]] = set()
        for name, term in self.cfg.ordered_terms():
            runtime = self._make_runtime(term)
            patch = runtime.plan_patch(target, sampled_params=sampled_params.get(name, {}))
            for op in patch.ops:
                if op.path in touched_paths:
                    raise ValueError(f"post-mutate patch conflict at path {op.path!r}")
                touched_paths.add(op.path)
            composed.extend(patch)
        return composed

    def mutate(
        self,
        target: HandCfg,
        *,
        sampled_params: dict[str, dict[str, Any]] | None = None,
    ) -> HandCfg | None:
        """Apply the composed patch once to a deep copy of `target`."""

        try:
            return self.plan_patch(target, sampled_params=sampled_params).apply(target)
        except Exception:
            if self.cfg.on_reject == "skip":
                return None
            raise

    def mutate_batch(
        self,
        target: HandCfg,
        *,
        sampled_batch: list[dict[str, dict[str, Any]]] | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[HandCfg | None, dict[str, dict[str, Any]]]]:
        """Batch helper used by generator refill loops.

        Sampling is batched; object patch application remains Python/dataclass
        based and deterministic.
        """

        if sampled_batch is None:
            sampled_batch = self.sample_batch(target, batch_size=int(batch_size or 1))
        results: list[tuple[HandCfg | None, dict[str, dict[str, Any]]]] = []
        for sampled_params in sampled_batch:
            results.append((self.mutate(target, sampled_params=sampled_params), sampled_params))
        return results


__all__ = ["HandMutatorCfg", "HandMutator", "MutatorTerm"]
