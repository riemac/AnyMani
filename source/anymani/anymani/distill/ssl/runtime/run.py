"""Run identity, deterministic setup, and output layout."""


from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar


class PretrainRun:


    def __init__(self, config: PretrainRunCfg) -> None:


        self.config = config

    def prepare_output_dir(self, override: Path | None = None) -> Path:


        output_dir = override
        if output_dir is None:
            experiment_root = Path(self.config.output_dir) / self.config.experiment_name
            explicit = Path(self.config.resume_checkpoint).expanduser() if self.config.resume_checkpoint else None
            if explicit is not None:
                explicit_run_root = explicit.resolve().parent.parent
                if (explicit_run_root / "COMPLETE").is_file() and not self.config.extend_completed_run:
                    raise ValueError("completed training run is immutable; use explicit completed-run extension")

                if not self.config.extend_completed_run or (explicit_run_root / "INCOMPLETE").is_file():
                    output_dir = explicit_run_root
            elif not self.config.new_run and experiment_root.is_dir():
                candidates = tuple(
                    child
                    for child in experiment_root.iterdir()
                    if child.is_dir()
                    and (child / "INCOMPLETE").is_file()
                    and (child / "checkpoints" / "recovery.pt").is_file()
                )
                if len(candidates) > 1:
                    raise RuntimeError(f"multiple incomplete runs require --resume_checkpoint or --new_run: {candidates}")
                if candidates:
                    output_dir = candidates[0]
            if output_dir is None:
                timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                output_dir = experiment_root / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        if not (output_dir / "COMPLETE").exists():
            (output_dir / "INCOMPLETE").write_text("schema=9.0.0\n", encoding="ascii")
        return output_dir

    @staticmethod
    def code_revision() -> str:


        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return result.stdout.strip() or "unknown"

    @staticmethod
    def package_version() -> str:


        for parent in Path(__file__).resolve().parents:
            version_path = parent / "VERSION"
            if version_path.is_file():
                declared = version_path.read_text(encoding="utf-8").strip()
                if declared:
                    return declared
        try:
            return version("anymani")
        except PackageNotFoundError:
            return "editable-unknown"

    def checkpoint_metadata(
        self,
        *,
        geometry_semantics_schema: str,
        dataset_identity: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        declared_objective: Mapping[str, float],
        objective_formula: Mapping[str, str] | None = None,
        fairgrad_formula: Mapping[str, Any] | None = None,
        parameter_partition: Mapping[str, Any] | None = None,
        source_artifact: Mapping[str, Any] | None = None,
        worktree_dirty: bool = False,
        worktree_fingerprint: str = "",
    ) -> Any:


        from anymani.distill.ssl.checkpoint import PretrainCheckpointMetadata

        return PretrainCheckpointMetadata(
            code_revision=self.code_revision(),
            package_version=self.package_version(),
            geometry_semantics_schema=geometry_semantics_schema,
            dataset_identity=dataset_identity,
            resolved_config=resolved_config,
            declared_objective=declared_objective,
            objective_formula=dict(objective_formula or {}),
            fairgrad_formula=dict(fairgrad_formula or {}),
            parameter_partition=dict(parameter_partition or {}),
            source_artifact=dict(source_artifact or {}),
            worktree_dirty=worktree_dirty,
            worktree_fingerprint=worktree_fingerprint,
        )

    @staticmethod
    def save_full_checkpoint(path: Path, **payload: Any) -> None:


        from anymani.distill.ssl.checkpoint import save_pretrain_checkpoint

        save_pretrain_checkpoint(path, **payload)

    @staticmethod
    def save_retained_artifact(path: Path, payload: Mapping[str, Any]) -> None:


        from anymani.distill.ssl.checkpoint import save_retained_artifact

        save_retained_artifact(path, payload)


@dataclass(frozen=True)
class PretrainRunCfg:


    runtime_type: ClassVar[type[PretrainRun]] = PretrainRun
    output_dir: str = "logs/ssl"
    experiment_name: str = "multi_anchor_gaussian"
    resume_checkpoint: str = ""
    new_run: bool = False
    seed: int = 0
    deterministic_algorithms: bool = True
    source_cache_root: str = "logs/ssl/_cache/geometry_source/v2"
    source_cache_mode: str = "readonly"
    allow_worktree_change: bool = False
    extend_completed_run: bool = False
    extension_source_package_version: str = ""

    def __post_init__(self) -> None:


        if not self.output_dir or not self.experiment_name:
            raise ValueError("pretraining run requires output_dir and experiment_name")
        if self.seed < 0:
            raise ValueError("pretraining run seed must be non-negative")
        if self.source_cache_mode not in {"auto", "readonly", "read-write", "off"}:
            raise ValueError("source_cache_mode must be 'auto', 'readonly', 'read-write', or 'off'")
        if self.source_cache_mode != "off" and not self.source_cache_root:
            raise ValueError("source_cache_root is required unless source cache mode is off")
        if self.extend_completed_run and not self.resume_checkpoint:
            raise ValueError("completed-run extension requires an explicit resume_checkpoint")
        if self.extension_source_package_version and not self.extend_completed_run:
            raise ValueError("extension_source_package_version requires completed-run extension")


__all__ = ["PretrainRun", "PretrainRunCfg"]
