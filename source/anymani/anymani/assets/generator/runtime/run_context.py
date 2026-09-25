"Defines output roots, run identity, and resource ownership for one generation call."

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..result import HandGenerationResult


@dataclass
class GenerationRunContext:
    "Output root, run identity, deterministic seed, and owned-artifact registry for one invocation."

    root_dir: Path
    summary: dict[str, Any]
    last_rejection_stage: str | None = None
    last_rejection_error_codes: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        cfg: Any,
        *,
        config_dump: dict[str, Any],
    ) -> GenerationRunContext:
        "Creates a new run context with explicit output ownership and seed identity."

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_root = _allocate_run_root(cfg, timestamp=timestamp)
        pre_made_enabled = cfg.mode == "made"
        post_mutate_enabled = cfg.mode == "mutate" and bool(cfg.Mutate.has_terms())
        summary = {
            "run": {
                "timestamp": timestamp,
                "root_dir": str(run_root),
                "mode": cfg.mode,
                "artifact_level": cfg.artifact_level,
                "phases": {
                    "pre_made": pre_made_enabled,
                    "post_mutate": post_mutate_enabled,
                    "combined": False,
                },
            },
            "config": config_dump,
            "stats": {
                "attempted": 0,
                "succeeded": 0,
                "rejected": 0,
                "rejected_by_stage": {},
                "rejected_by_reason": {},
                "by_topology": {},
            },
        }
        context = cls(root_dir=run_root, summary=summary)
        context.write_summary()
        return context

    def write_summary(self) -> None:
        'Writes summary.'

        stats = self.summary["stats"]
        stats["topology_count"] = len(stats["by_topology"])
        summary_path = self.root_dir / "summary.yaml"
        summary_path.write_text(
            yaml.safe_dump(self.summary, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def record_rejection(
        self,
        *,
        stage: str,
        error_codes: tuple[str, ...] = (),
        write_summary: bool = True,
    ) -> None:
        'Records rejection.'

        self.last_rejection_stage = stage
        self.last_rejection_error_codes = canonical_rejection_error_codes(error_codes)
        stats = self.summary["stats"]
        stats["attempted"] += 1
        stats["rejected"] += 1
        rejected_by_stage = dict(stats.get("rejected_by_stage") or {})
        rejected_by_stage[stage] = int(rejected_by_stage.get(stage, 0)) + 1
        stats["rejected_by_stage"] = rejected_by_stage
        reason_key = rejection_reason_key(self.last_rejection_error_codes)
        rejected_by_reason = dict(stats.get("rejected_by_reason") or {})
        rejected_by_reason[reason_key] = int(rejected_by_reason.get(reason_key, 0)) + 1
        stats["rejected_by_reason"] = rejected_by_reason
        if write_summary:
            self.write_summary()

    def record_success(self, result: HandGenerationResult, *, write_summary: bool = True) -> None:
        'Records success.'

        self.last_rejection_stage = None
        self.last_rejection_error_codes = ()
        stats = self.summary["stats"]
        stats["attempted"] += 1
        stats["succeeded"] += 1
        topology_key = result_topology_key(result)
        by_topology = dict(stats.get("by_topology") or {})
        by_topology[topology_key] = int(by_topology.get(topology_key, 0)) + 1
        stats["by_topology"] = by_topology
        if write_summary:
            self.write_summary()


def result_topology_key(result: HandGenerationResult) -> str:
    "Returns the stable topology key used to group run results."

    topology_name = str(result.metadata.get("topology_name") or result.metadata.get("family") or "unknown_topology")
    topology_group_name = str(
        result.metadata.get("topology_group_name")
        or result.metadata.get("base_hand_preset")
        or result.metadata.get("family")
        or "ungrouped"
    )
    family_composition = result.metadata.get("family_composition")
    is_mixed = (
        str(family_composition) == "mixed"
        if family_composition is not None
        else str(result.metadata.get("topology_kind") or "single_family") == "mixed"
    )
    if is_mixed:
        return f"mixed/{topology_group_name}/{topology_name}"
    return f"{topology_group_name}/{topology_name}"


def canonical_rejection_error_codes(error_codes: tuple[str, ...]) -> tuple[str, ...]:
    "Returns stable rejection codes used in run summaries."

    normalized = {str(code).strip() for code in error_codes if str(code).strip()}
    return tuple(sorted(normalized))


def rejection_reason_key(error_codes: tuple[str, ...]) -> str:
    "Maps a candidate rejection to its stable summary key."

    canonical_codes = canonical_rejection_error_codes(error_codes)
    return "+".join(canonical_codes) if canonical_codes else "unclassified"


def _allocate_run_root(cfg: Any, *, timestamp: str) -> Path:

    if cfg.mode == "full":
        raise NotImplementedError(
            "mode='full' is temporarily unsupported. "
            "This migration only covers mode='made' and independent mode='mutate'; "
            "the full pipeline has not been adapted to topology-root export semantics yet."
        )

    if cfg.mode == "mutate":
        if cfg.source_topology_dir is None:
            raise ValueError("mode='mutate' requires 'source_topology_dir'")
        base_root = Path(cfg.source_topology_dir)
    else:
        base_root = Path(cfg.output_dir)

    run_root = base_root / timestamp
    collision_index = 2
    while run_root.exists():
        run_root = base_root / f"{timestamp}_{collision_index:02d}"
        collision_index += 1

    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


__all__ = [
    "GenerationRunContext",
    "canonical_rejection_error_codes",
    "rejection_reason_key",
    "result_topology_key",
]
