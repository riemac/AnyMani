'Record geometry-SSL scalars, runtime events, and dense validation snapshots.'

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from anymani.distill.methods.multi_anchor_gaussian_implicit_field.batch import (
    PaddedOnlineGeometryBatch,
)
from anymani.distill.models.geometry_ssl import GeometrySSLForward


class GeometrySSLRunLogger:
    'Contract for geometry sslrun logger.'

    def __init__(self, output_dir: Path, *, purge_step: int | None = None) -> None:
        'Initialize the instance.'

        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"), purge_step=purge_step)
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.runtime_jsonl_path = output_dir / "runtime.jsonl"

    def continuation_offsets(self) -> dict[str, int]:
        'Handle continuation offsets.'

        self.writer.flush()
        return {
            "metrics_jsonl_bytes": self.jsonl_path.stat().st_size if self.jsonl_path.is_file() else 0,
            "runtime_jsonl_bytes": self.runtime_jsonl_path.stat().st_size if self.runtime_jsonl_path.is_file() else 0,
        }

    def restore_continuation(self, offsets: Mapping[str, int], *, purge_step: int) -> None:
        'Restore continuation.'

        for name, path in (("metrics_jsonl_bytes", self.jsonl_path), ("runtime_jsonl_bytes", self.runtime_jsonl_path)):
            raw_offset = offsets.get(name)
            if not isinstance(raw_offset, int) or raw_offset < 0:
                raise ValueError(f"recovery logger offset {name!r} must be a non-negative integer")
            current_size = path.stat().st_size if path.is_file() else 0
            if current_size < raw_offset:
                raise ValueError(f"recovery logger file {path} is shorter than checkpoint offset")
            if path.is_file():
                with path.open("r+b") as stream:
                    stream.truncate(raw_offset)
        self.writer.close()
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"), purge_step=purge_step)

    def log_runtime_event(self, event: dict[str, Any]) -> None:
        'Handle log runtime event.'

        with self.runtime_jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

        if event.get("event") == "optimizer_update":
            optimizer_update = int(event["optimizer_update"])
            for name in (
                "step_seconds",
                "q_samples_per_second",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
            ):
                value = event.get(name)
                if value is not None:
                    self.writer.add_scalar(f"runtime/{name}", float(value), optimizer_update)

    def log_terms(
        self,
        *,
        optimizer_update: int,
        epoch: int,
        mini_epoch: int,
        minibatch_in_epoch: int,
        global_minibatch: int,
        new_pairs_seen: int,
        pair_uses: int,
        teacher_pairs_realized: int,
        microbatches_consumed: int,
        wall_time_seconds: float,
        split: str,
        terms: dict[str, float],
        denominators: dict[str, float],
        asset_ids: tuple[str, ...],  # shapes [B]
        gradient_groups: dict[str, dict[str, Any]] | None = None,
        batch: PaddedOnlineGeometryBatch | None = None,  # q cursor provenance
        gradient_evidence: dict[str, float] | None = None,
        diagnostic_seconds: float = 0.0,
        diagnostics: dict[str, float] | None = None,
    ) -> None:
        'Record supplied objective scalars and gradient evidence without recomputing targets.'

        scalars = {f"raw/{name}": value for name, value in terms.items()}
        for name, value in scalars.items():
            self.writer.add_scalar(f"{split}_update/{name}", value, optimizer_update)
        for group_name, group in (gradient_groups or {}).items():
            for field_name in ("pre_clip_norm", "post_clip_norm", "clip_ratio"):
                self.writer.add_scalar(
                    f"{split}_gradient/{group_name}/{field_name}",
                    float(group[field_name]),
                    optimizer_update,
                )
        for name, value in (gradient_evidence or {}).items():
            self.writer.add_scalar(f"{split}_z_gradient/{name}", value, optimizer_update)
        for name, value in (diagnostics or {}).items():
            self.writer.add_scalar(f"{split}_diagnostic/{name}", value, optimizer_update)
        if diagnostic_seconds > 0.0:
            self.writer.add_scalar(f"{split}_z_gradient/diagnostic_seconds", diagnostic_seconds, optimizer_update)
        record: dict[str, Any] = {
            "epoch": epoch,
            "mini_epoch": mini_epoch,
            "minibatch_in_epoch": minibatch_in_epoch,
            "global_minibatch": global_minibatch,
            "minibatch_reuse_identity": [global_minibatch, mini_epoch],
            "optimizer_update": optimizer_update,
            "new_pairs_seen": new_pairs_seen,
            "pair_uses": pair_uses,
            "teacher_pairs_realized": teacher_pairs_realized,
            "microbatches_consumed": microbatches_consumed,
            "wall_time_seconds": wall_time_seconds,
            "denominators": dict(denominators),
            "split": split,
            "asset_ids": list(asset_ids),
            "q_index": (
                batch.q_index.detach().cpu().tolist() if batch is not None and batch.q_index is not None else None
            ),
            **scalars,
        }
        if gradient_groups:
            record["gradient_groups"] = gradient_groups
        if gradient_evidence:
            record["z_gradient_evidence"] = dict(gradient_evidence)
            record["z_gradient_diagnostic_seconds"] = diagnostic_seconds
        if diagnostics:
            record["diagnostics"] = dict(diagnostics)
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def finalize_training_metrics(
        self,
        *,
        teacher_baselines: Mapping[str, object],
        expected_optimizer_updates: int,
        lineage_metrics_path: Path | None = None,
    ) -> Path:
        'Handle finalize training metrics.'

        baselines: dict[str, float] = {}
        for name, record in teacher_baselines.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"final teacher baseline lacks mapping for {name}")
            value = record.get("baseline_mse")
            if not isinstance(value, (float, int)) or float(value) <= 0.0:
                raise ValueError(f"final teacher baseline {name}.baseline_mse must be positive")
            baselines[name] = float(value)
        if not baselines:
            raise ValueError("final teacher baselines must contain at least one objective mapping")

        sources = []
        if lineage_metrics_path is not None and lineage_metrics_path.resolve() != self.jsonl_path.resolve():
            sources.append(lineage_metrics_path)
        sources.append(self.jsonl_path)
        by_update: dict[int, dict[str, Any]] = {}
        for source in sources:
            if not source.is_file():
                raise ValueError(f"training metric lineage file does not exist: {source}")
            for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
                record = json.loads(line)
                update = record.get("optimizer_update")
                if not isinstance(update, int):
                    raise ValueError(f"metric record {source}:{line_number} lacks integer optimizer_update")
                if update in by_update:
                    raise ValueError(f"duplicate optimizer_update={update} in training metric lineage")
                by_update[update] = record
        expected = set(range(1, expected_optimizer_updates + 1))
        if set(by_update) != expected:
            missing = sorted(expected - set(by_update))
            unexpected = sorted(set(by_update) - expected)
            raise ValueError(
                f"training metric lineage is not a complete update prefix: missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}"
            )

        output = self.output_dir / "metrics_finalized.jsonl"
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for update in range(1, expected_optimizer_updates + 1):
                raw = by_update[update]
                missing_raw = [name for name in baselines if f"raw/{name}" not in raw]
                if missing_raw:
                    raise ValueError(f"metric record optimizer_update={update} lacks raw terms={missing_raw}")
                normalized = {
                    name: float(raw[f"raw/{name}"]) / baselines[name]
                    for name in baselines
                }
                finalized = {
                    **raw,
                    "final_teacher_baselines": dict(baselines),
                    **{f"normalized/{name}": value for name, value in normalized.items()},
                    **{f"skill/{name}": 1.0 - value for name, value in normalized.items()},
                }
                stream.write(json.dumps(finalized, sort_keys=True) + "\n")
                for name, value in normalized.items():
                    self.writer.add_scalar(f"train_finalized/normalized/{name}", value, update)
                    self.writer.add_scalar(f"train_finalized/skill/{name}", 1.0 - value, update)
        temporary.replace(output)
        return output

    def log_epoch_terms(
        self,
        *,
        epoch: int,
        new_pairs_seen: int,
        pair_uses: int,
        optimizer_updates: int,
        terms: dict[str, float],
    ) -> None:
        'Handle log epoch terms.'

        for name, value in terms.items():
            self.writer.add_scalar(f"train_epoch/{name}", value, new_pairs_seen)
        self.writer.add_scalar("progress/epoch", epoch, new_pairs_seen)
        self.writer.add_scalar("progress/pair_uses", pair_uses, new_pairs_seen)
        self.writer.add_scalar("progress/optimizer_updates", optimizer_updates, new_pairs_seen)

    def log_validation_metrics(
        self,
        *,
        epoch: int,
        new_pairs_seen: int,
        metrics: dict[str, dict[str, float]],
    ) -> None:
        'Handle log validation metrics.'

        for suite, suite_metrics in metrics.items():
            for name, value in suite_metrics.items():
                self.writer.add_scalar(f"validation/{suite}/{name}", value, new_pairs_seen)
        self.writer.add_scalar("validation/epoch", epoch, new_pairs_seen)

    def save_dense_snapshot(
        self,
        *,
        optimizer_update: int,
        split: str,
        prediction: GeometrySSLForward,
        batch: PaddedOnlineGeometryBatch,  # target/mask/asset identity
    ) -> Path:
        'Save dense snapshot.'

        entity_valid = batch.evidence.entity_valid_mask  # shapes [B,26]
        joint_valid = batch.evidence.joint_valid_mask  # shapes [B,20]
        if entity_valid is None or joint_valid is None:
            raise ValueError("dense SSL snapshot requires padded entity/joint validity masks")
        evidence_row_index = batch.evidence_row_index
        if evidence_row_index is not None:
            entity_valid = entity_valid[evidence_row_index]
            joint_valid = joint_valid[evidence_row_index]
            joint_entity_index = batch.evidence.joint_entity_index[evidence_row_index]
        else:
            joint_entity_index = batch.evidence.joint_entity_index
        path = self.output_dir / f"{split}_dense_update_{optimizer_update:08d}.npz"
        np.savez_compressed(
            path,
            asset_ids=np.asarray(batch.asset_ids),  # `[B]` Unicode asset IDs
            q_index=(
                batch.q_index.detach().cpu().numpy()
                if batch.q_index is not None
                else np.full(len(batch.asset_ids), -1, dtype=np.int64)
            ),  # `[B]` asset-local Sobol absolute cursor
            q=batch.q.detach().cpu().numpy(),  # shapes [B,20]; units rad
            query_points_h=batch.queries.query_points_h.detach().cpu().numpy(),  # shapes [B,26,N_Q,3]; units m
            query_stratum=batch.queries.query_stratum.detach().cpu().numpy(),
            adjacent_owner_index=batch.queries.adjacent_owner_index.detach().cpu().numpy(),  # adjacent routing
            workspace_anchor_index=batch.queries.workspace_anchor_index.detach().cpu().numpy(),  # anchor routing
            owner_role=batch.field_targets.owner_role.detach().cpu().numpy(),  # canonical PALM/JOINT/TIP axis
            bandwidths_m=batch.field_targets.bandwidths.detach().cpu().numpy(),
            entities=prediction.latents.entities.detach().cpu().numpy(),  # shapes [B,26,D]
            entity_valid_mask=entity_valid.detach().cpu().numpy(),  # `[B,26]` bool
            joint_valid_mask=joint_valid.detach().cpu().numpy(),  # `[B,20]` bool
            evidence_row_index=(
                evidence_row_index.detach().cpu().numpy()
                if evidence_row_index is not None
                else np.arange(len(batch.asset_ids), dtype=np.int64)
            ),
            anchor_index=(
                batch.anchor_index.detach().cpu().numpy()
                if batch.anchor_index is not None
                else np.zeros(len(batch.asset_ids), dtype=np.int64)
            ),
            field_valid_mask=batch.field_targets.valid_mask.detach().cpu().numpy(),  # `[B,26,N_Q]`
            edge_valid_mask=batch.sensitivity_targets.valid_mask.detach().cpu().numpy(),  # `[B,E]`
            ancestor_mask=batch.sensitivity_targets.ancestor_mask.detach().cpu().numpy(),
            active_mask=batch.sensitivity_targets.active_mask.detach().cpu().numpy(),  # active/structural-zero
            edge_owner_index=batch.sensitivity_targets.owner_index.detach().cpu().numpy(),  # sampled owner
            edge_query_index=batch.sensitivity_targets.query_index.detach().cpu().numpy(),  # sampled query
            edge_joint_index=batch.sensitivity_targets.joint_index.detach().cpu().numpy(),  # sampled JOINT
            joint_entity_index=joint_entity_index.detach().cpu().numpy(),  # `[B,N_J]` JOINT view routing
            closest_point_h_m=batch.sensitivity_targets.closest_point.detach().cpu().numpy(),  # `[B,E,3]`
            closest_source=batch.sensitivity_targets.closest_source.detach().cpu().numpy(),  # owner/face provenance
            uniqueness_margin_m=batch.sensitivity_targets.uniqueness_margin.detach().cpu().numpy(),
            distance_m=batch.field_targets.distance.detach().cpu().numpy(),
            density_prediction=prediction.density.detach().cpu().numpy(),  # `[B,G,N_Q,N_sigma]`
            density_target=batch.field_targets.density.detach().cpu().numpy(),  # teacher density
            kappa_prediction=prediction.kappa.detach().cpu().numpy(),  # Shapes [B,E]; units m/rad.
            kappa_target=batch.sensitivity_targets.kappa.detach().cpu().numpy(),  # Units m/rad.
            density_error=(prediction.density - batch.field_targets.density).detach().cpu().numpy(),
            kappa_error=(prediction.kappa - batch.sensitivity_targets.kappa).detach().cpu().numpy(),  # m/rad
            central_difference=(
                batch.sensitivity_targets.central_difference.detach().cpu().numpy()
                if batch.sensitivity_targets.central_difference is not None
                else np.zeros_like(batch.sensitivity_targets.kappa.detach().cpu().numpy())
            ),
            central_difference_valid_mask=(
                batch.sensitivity_targets.central_difference_valid_mask.detach().cpu().numpy()
                if batch.sensitivity_targets.central_difference_valid_mask is not None
                else np.zeros_like(batch.sensitivity_targets.valid_mask.detach().cpu().numpy())
            ),
            central_difference_plus_face=(
                batch.sensitivity_targets.central_difference_plus_face.detach().cpu().numpy()
                if batch.sensitivity_targets.central_difference_plus_face is not None
                else np.full_like(batch.sensitivity_targets.closest_source.detach().cpu().numpy(), -1)
            ),
            central_difference_minus_face=(
                batch.sensitivity_targets.central_difference_minus_face.detach().cpu().numpy()
                if batch.sensitivity_targets.central_difference_minus_face is not None
                else np.full_like(batch.sensitivity_targets.closest_source.detach().cpu().numpy(), -1)
            ),
        )
        return path

    def close(self) -> None:
        'Close the declared contract.'

        self.writer.flush()
        self.writer.close()


__all__ = ["GeometrySSLRunLogger"]
