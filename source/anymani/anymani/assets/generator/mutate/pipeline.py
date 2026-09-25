"Applies local post-mutate operators to a hand copy. Joint deletion belongs to pre-made topology lowering, not this mutation pipeline."

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ...asset_base import AssetCfgBase, HandCfg
from ...handedness import lower_hand_to_handedness
from .base import HandPatch, MutatorBase, MutatorBaseCfg, _sample_value


@dataclass
class HandMutatorCfg(AssetCfgBase):
    "Ordered mutator configuration applied to a copy of the original hand."

    class_type: type[HandMutator] | None = field(init=False, default=None, repr=False)
    "Associated runtime implementation for this configuration class."

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.class_type is None:
            self.class_type = HandMutator

    def ordered_terms(self) -> list[tuple[str, MutatorBaseCfg]]:
        "Returns enabled operators in stable application order."

        ordered: OrderedDict[str, MutatorBaseCfg] = OrderedDict()


        for cls in reversed(type(self).mro()):
            for name, value in cls.__dict__.items():
                if name.startswith("_"):
                    continue
                if isinstance(value, MutatorBaseCfg):
                    ordered[name] = value.copy()


        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "class_type":
                continue
            if isinstance(value, MutatorBaseCfg):
                ordered[name] = value

        return list(ordered.items())

    def has_terms(self) -> bool:
        "Reports whether at least one geometry mutator is enabled."

        return bool(self.ordered_terms())

    def to_dict(self) -> dict[str, Any]:
        'Serializes the typed object as a dictionary.'




        return {name: term_cfg for name, term_cfg in self.ordered_terms()}


class HandMutator:
    "Applies a typed geometry proposal to a hand copy and records sampled values."

    cfg: HandMutatorCfg

    def __init__(self, cfg: HandMutatorCfg):
        "Stores the ordered mutation terms; proposals are sampled only when requested."

        self.cfg = cfg

    def _make_runtime(self, cfg: MutatorBaseCfg) -> MutatorBase:

        runtime_cls = getattr(cfg, "class_type", None)
        if runtime_cls is None:
            raise TypeError(f"mutator cfg {cfg!r} does not define class_type")
        return runtime_cls(cfg)

    def describe_sampling(self, target: HandCfg) -> dict[str, dict[str, Any]]:
        'Summarizes proposal settings and sample outcomes.'

        canonical_target = _canonicalize_for_mutation(target)
        plan: dict[str, dict[str, Any]] = {}
        for name, cfg in self.cfg.ordered_terms():
            runtime = self._make_runtime(cfg)
            plan[name] = dict(runtime.describe_sampling(canonical_target))
        return plan

    def sample_batch(self, target: HandCfg, *, batch_size: int) -> list[dict[str, dict[str, Any]]]:
        'Selects a proposal batch for the requested mode.'

        sample_plan = self.describe_sampling(target)
        batch: list[dict[str, dict[str, Any]]] = [
            {term_name: {} for term_name in sample_plan}
            for _ in range(max(int(batch_size), 0))
        ]
        for term_name, distribution_map in sample_plan.items():
            for local_name, distribution in distribution_map.items():
                for sample in batch:
                    sample[term_name][local_name] = _sample_value(distribution)
        return batch

    def plan_patch(
        self,
        target: HandCfg,
        *,
        sampled_params: dict[str, dict[str, Any]] | None = None,
    ) -> HandPatch:
        "Composes enabled operator edits against one original hand before any patch is applied."

        if target.handedness != "right":
            raise ValueError(
                "HandMutator.plan_patch expects canonical right-hand input; "
                "use HandMutator.mutate for handedness-aware post-mutate."
            )
        sampled_params = sampled_params or {}
        composed = HandPatch()
        op_index_by_path: dict[tuple[Any, ...], int] = {}
        for name, cfg in self.cfg.ordered_terms():
            runtime = self._make_runtime(cfg)
            patch = runtime.plan_patch(target, sampled_params=sampled_params.get(name, {}))
            for op in patch.ops:
                existing_index = op_index_by_path.get(op.path)
                if existing_index is None:
                    op_index_by_path[op.path] = len(composed.ops)
                    composed.add_op(op)
                    continue

                composed.ops[existing_index] = composed.ops[existing_index].merged_with(op)
            composed.merge_metadata(patch.metadata)
        return composed

    def mutate(
        self,
        target: HandCfg,
        *,
        sampled_params: dict[str, dict[str, Any]] | None = None,
    ) -> HandCfg | None:
        "Samples enabled operators, applies their patches, and records proposal provenance."

        try:
            target_handedness = target.handedness
            canonical_target = _canonicalize_for_mutation(target)
            canonical_mutated = self.plan_patch(
                canonical_target,
                sampled_params=sampled_params,
            ).apply(canonical_target)
            return lower_hand_to_handedness(canonical_mutated, target_handedness)
        except Exception:
            return None

    def mutate_batch(
        self,
        target: HandCfg,
        *,
        sampled_batch: list[dict[str, dict[str, Any]]] | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[HandCfg | None, dict[str, dict[str, Any]]]]:
        "Runs bounded candidates through the configured mutator pipeline."

        if sampled_batch is None:
            sampled_batch = self.sample_batch(target, batch_size=int(batch_size or 1))
        results: list[tuple[HandCfg | None, dict[str, dict[str, Any]]]] = []
        for sampled_params in sampled_batch:
            results.append((self.mutate(target, sampled_params=sampled_params), sampled_params))
        return results


def _canonicalize_for_mutation(target: HandCfg) -> HandCfg:

    return lower_hand_to_handedness(target, "right")


__all__ = ["HandMutatorCfg", "HandMutator"]
