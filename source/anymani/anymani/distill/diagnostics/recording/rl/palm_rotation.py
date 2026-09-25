'Record PPO scalars and selected trajectories without recomputing task outcomes.'

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pl

PALM_ROTATION_METRICS_SCHEMA_VERSION = "2.9.0"


PALM_ROTATION_METRICS_SCHEMA: dict[str, Any] = {
    "schema_version": pl.String,
    "identity_digest": pl.String,
    "update": pl.Int64,
    "transitions": pl.Int64,
    "scope": pl.String,
    "scope_index": pl.Int16,
    "dataset_row": pl.Int32,
    "cell_id": pl.Int8,
    "reward_mean": pl.Float64,
    **{
        f"reward_term_{name}": pl.Float64
        for name in (
            "pose_keypoint",
            "orientation_tracking",
            "rotation_progress",
            "goal_success",
            "good_tip_contact",
            "bad_finger_non_tip_contact",
            "speed_band",
            "speed_jitter",
            "off_axis_angular_velocity",
            "object_linear_velocity",
            "joint_pose_anchor",
            "mechanical_power",
            "torque_l2",
            "action_l2",
            "action_rate_l2",
            "failure",
        )
    },
    "goal_count_mean": pl.Float64,  # strict full-pose+position tracking count
    **{
        name: pl.Int64
        for name in (
            "first30_asset_count", "first30_observed_assets", "first30_qualified_assets", "first30_window_count",
            "first30_windows_min", "first30_windows_max", "first30_one_turn_proxy_assets", "first30_two_turn_proxy_assets",
            "first30_policy_start_min", "first30_policy_end_max",
        )
    },
    **{
        name: pl.Float64
        for name in ("first30_net_median", "first30_goal_median", "first30_direction_median", "first30_safe_fraction")
    },
    "adr_position_level_at_update_end": pl.Float64,
    "adr_position_half_width_m_at_update_end": pl.Float64,
    "frontier_count_mean": pl.Float64,
    "frontier_pulse_rate": pl.Float64,
    "max_positive_net_turns_mean": pl.Float64,  # units M
    "net_turns_mean": pl.Float64,
    "drop_rate": pl.Float64,
    "axis_failure_rate": pl.Float64,
    "tip_contact_mean": pl.Float64,
    "palm_contact_rate": pl.Float64,
    "non_tip_contact_rate": pl.Float64,
    "advantage_mean": pl.Float64,
    "advantage_std": pl.Float64,
    "value_error_mean": pl.Float64,
    "return_target_mean": pl.Float64,
    "return_target_std": pl.Float64,
    "value_prediction_mean": pl.Float64,
    "value_prediction_std": pl.Float64,
    "value_error_physical_mean": pl.Float64,
    "value_explained_variance": pl.Float64,
    "value_clip_fraction": pl.Float64,
    "kl_per_active_dof": pl.Float64,
    "clip_fraction": pl.Float64,
    "action_rms": pl.Float64,
    "action_clamp_fraction": pl.Float64,
    "physical_action_rms": pl.Float64,
    "policy_mean_rms": pl.Float64,
    "policy_mean_near_bound_fraction": pl.Float64,
    "base_mean_rms": pl.Float64,
    "residual_rms": pl.Float64,
    "residual_fraction": pl.Float64,
    "direct_mean_rms": pl.Float64,
    "direct_mean_near_bound_fraction": pl.Float64,
    "direct_pre_tanh_derivative_mean": pl.Float64,
    "film_modulation_rms": pl.Float64,
    "candidate_lambda": pl.Float64,
    "actual_lambda": pl.Float64,
    "counterfactual_adr_level": pl.Float64,
    "actor_grad_norm": pl.Float64,
    "critic_grad_norm": pl.Float64,
    "actor_cagrad_relative_gap": pl.Float64,
    "critic_cagrad_relative_gap": pl.Float64,
    "actor_cagrad_worst_projection": pl.Float64,
    "critic_cagrad_worst_projection": pl.Float64,
    "actor_cagrad_iterations": pl.Float64,
    "critic_cagrad_iterations": pl.Float64,
    "popart_weight_error": pl.Float64,
    "popart_bias_error": pl.Float64,
    "popart_count": pl.Float64,
    "gradient_cosine": pl.Float64,
    "actor_probe_grad_norm": pl.Float64,
    "critic_probe_grad_norm": pl.Float64,
    "actor_probe_mean_cosine": pl.Float64,
    "critic_probe_mean_cosine": pl.Float64,
    "actor_probe_negative_pair_fraction": pl.Float64,
    "critic_probe_negative_pair_fraction": pl.Float64,
    "actor_probe_top1_norm_fraction": pl.Float64,
    "critic_probe_top1_norm_fraction": pl.Float64,
    "actor_probe_grad_norm_span": pl.Float64,
    "critic_probe_grad_norm_span": pl.Float64,
    "gradient_probe_seconds": pl.Float64,
    "actor_loss": pl.Float64,
    "critic_loss": pl.Float64,
    "entropy": pl.Float64,
    "policy_sigma": pl.Float64,
    "policy_base_sigma": pl.Float64,
    "recovery_floor_fraction": pl.Float64,
    "actor_rejected_action_cost": pl.Float64,
    "actor_rejected_action_fraction": pl.Float64,
    "actor_base_lr": pl.Float64,
    "actor_residual_lr": pl.Float64,
    "actor_contextual_lr": pl.Float64,
    "critic_lr": pl.Float64,
    "optimizer_microbatches": pl.Float64,
    "optimizer_steps": pl.Float64,
    **{
        name: pl.Float64
        for name in (
            "student_rollout_updates",
            "student_critic_optimizer_steps",
            "student_actor_optimizer_steps",
            "student_warmup_rollouts",
            "student_warmup_critic_optimizer_steps",
            "student_anchor_loss",
            "student_anchor_weighted_loss",
            "student_fk_loss",
            "student_fk_weighted_loss",
        )
    },
    "rollout_sample_count": pl.Float64,
    "completed_episode_count": pl.Float64,
    "terminal_goal_count_mean": pl.Float64,
    "terminal_frontier_count_mean": pl.Float64,
    "terminal_max_positive_net_turns_mean": pl.Float64,
    "terminal_net_turns_mean": pl.Float64,
    "terminal_absolute_path_turns_mean": pl.Float64,
    "terminal_directional_consistency_mean": pl.Float64,
    "terminal_timeout_rate": pl.Float64,
    "terminal_drop_rate": pl.Float64,
    "terminal_axis_failure_rate": pl.Float64,
    "environment_step_seconds": pl.Float64,
    "rollout_policy_seconds": pl.Float64,
    "ppo_update_seconds": pl.Float64,
    "epoch_total_seconds": pl.Float64,
    "steps_per_second": pl.Float64,
    "gpu_memory_bytes": pl.Int64,
    "gpu_memory_allocated_bytes": pl.Int64,
    "gpu_memory_reserved_bytes": pl.Int64,
    "gpu_peak_reserved_bytes": pl.Int64,
    "gpu_driver_free_bytes": pl.Int64,
    "gpu_driver_total_bytes": pl.Int64,
    "process_rss_bytes": pl.Int64,
    "process_peak_rss_bytes": pl.Int64,
    "process_swap_bytes": pl.Int64,
    "system_available_memory_bytes": pl.Int64,
}


def _sha256(path: Path) -> str:
    'Handle sha256.'

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # 1 MiB bounded I/O block
            digest.update(block)
    return digest.hexdigest()


def _normalize_metric_row(
    row: Mapping[str, Any],
    *,
    identity_digest: str,
) -> dict[str, Any]:
    'Normalize metric row.'

    unknown = set(row) - set(PALM_ROTATION_METRICS_SCHEMA)
    if unknown:
        raise ValueError(f"palm-rotation metric row contains unknown fields: {sorted(unknown)}")
    normalized: dict[str, Any] = {
        name: None for name in PALM_ROTATION_METRICS_SCHEMA
    }  # missing scope-specific scalar -> null
    normalized.update(row)
    normalized["schema_version"] = PALM_ROTATION_METRICS_SCHEMA_VERSION
    normalized["identity_digest"] = identity_digest
    if normalized["scope"] not in {"global", "cell", "asset"}:
        raise ValueError("metric scope must be global, cell or asset")
    if int(normalized["update"] or 0) < 1 or int(normalized["transitions"] or 0) < 0:
        raise ValueError("metric update must be positive and transitions non-negative")
    return normalized


class PalmRotationMetricsRecorder:
    'Contract for PALM rotation metrics recorder.'

    def __init__(
        self,
        run_dir: Path | str,
        *,
        identity_digest: str,
        flush_every_updates: int = 50,
    ) -> None:
        'Initialize the instance.'

        if pl.__version__ != "1.32.3":
            raise RuntimeError(f"palm-rotation metrics require polars==1.32.3, got {pl.__version__}")
        if len(identity_digest) != 64:
            raise ValueError("metrics identity_digest must be a SHA-256 hex string")
        if flush_every_updates < 1:
            raise ValueError("flush_every_updates must be positive")
        self.run_dir = Path(run_dir).expanduser()  # run-owned evidence root
        self.shard_dir = self.run_dir / "metrics_shards"  # interrupted-run immutable temporary shards
        self.final_path = self.run_dir / "metrics.parquet"  # completed-run compact scalar table
        self.identity_digest = identity_digest
        self.flush_every_updates = int(flush_every_updates)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._pending: list[dict[str, Any]] = []  # fixed-schema normalized rows not yet durable
        self._pending_first_update: int | None = None
        self._last_recorded_update = 0
        self._next_shard_index = len(tuple(self.shard_dir.glob("metrics-*.parquet")))

    @property
    def pending_row_count(self) -> int:
        'Handle pending row count.'

        return len(self._pending)

    def record(self, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
        'Record the declared contract; shapes [Path].'

        if not rows:
            raise ValueError("metrics record requires at least one row")
        normalized = [_normalize_metric_row(row, identity_digest=self.identity_digest) for row in rows]
        updates = {int(row["update"]) for row in normalized}
        if len(updates) != 1:
            raise ValueError("one metrics record call must contain exactly one update")
        update = updates.pop()
        if update <= self._last_recorded_update:
            raise ValueError("metric updates must be strictly increasing")
        if self._pending_first_update is None:
            self._pending_first_update = update
        self._pending.extend(normalized)
        self._last_recorded_update = update
        if update - self._pending_first_update + 1 >= self.flush_every_updates:
            path = self.flush(reason="cadence")
            return [path] if path is not None else []
        return []

    def flush(self, *, reason: str) -> Path | None:
        'Flush the declared contract.'

        if not self._pending:
            return None
        first_update = int(self._pending_first_update or self._last_recorded_update)
        last_update = self._last_recorded_update
        filename = f"metrics-{self._next_shard_index:06d}-u{first_update:08d}-u{last_update:08d}.parquet"
        destination = self.shard_dir / filename  # immutable shard final path
        temporary = destination.with_suffix(".parquet.tmp")  # same-filesystem atomic source
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"metrics shard path already exists: {destination}")


        frame = pl.DataFrame(self._pending, schema=PALM_ROTATION_METRICS_SCHEMA)
        frame = frame.with_columns(pl.lit(reason).alias("_flush_reason"))  # shard lifecycle evidence
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(destination)  # readers observe either no shard or a complete footer/data file
        self._pending.clear()
        self._pending_first_update = None
        self._next_shard_index += 1
        return destination

    def state_dict(self) -> dict[str, Any]:
        'Handle state dict.'

        if self._pending:
            raise RuntimeError("metrics recorder must flush pending rows before checkpoint")
        shards = sorted(self.shard_dir.glob("metrics-*.parquet"))
        return {
            "schema_version": PALM_ROTATION_METRICS_SCHEMA_VERSION,
            "identity_digest": self.identity_digest,
            "flush_every_updates": self.flush_every_updates,
            "last_recorded_update": self._last_recorded_update,
            "next_shard_index": self._next_shard_index,
            "shards": [{"name": path.name, "sha256": _sha256(path)} for path in shards],
        }

    def load_state_dict(self, state: object) -> None:
        'Load state dict.'

        if not isinstance(state, Mapping) or state.get("schema_version") != PALM_ROTATION_METRICS_SCHEMA_VERSION:
            raise RuntimeError("metrics recorder checkpoint state is missing or incompatible")
        if state.get("identity_digest") != self.identity_digest:
            raise RuntimeError("metrics recorder identity disagrees with checkpoint")
        if int(state.get("flush_every_updates", -1)) != self.flush_every_updates:
            raise RuntimeError("metrics recorder flush cadence disagrees with checkpoint")
        expected_shards = state.get("shards")
        if not isinstance(expected_shards, Sequence):
            raise RuntimeError("metrics recorder checkpoint shard inventory is malformed")
        actual_shards = sorted(self.shard_dir.glob("metrics-*.parquet"))
        actual = [{"name": path.name, "sha256": _sha256(path)} for path in actual_shards]
        if list(expected_shards) != actual:
            raise RuntimeError("metrics recorder shards disagree with checkpoint inventory")
        self._last_recorded_update = int(state.get("last_recorded_update", 0))
        self._next_shard_index = int(state.get("next_shard_index", len(actual_shards)))
        if self._next_shard_index != len(actual_shards):
            raise RuntimeError("metrics recorder next shard index is not contiguous")

    def finalize(self) -> Path:
        'Handle finalize.'

        self.flush(reason="finalize")
        shards = sorted(self.shard_dir.glob("metrics-*.parquet"))
        if not shards:
            raise RuntimeError("cannot finalize an empty metrics recorder")
        temporary = self.final_path.with_suffix(".parquet.tmp")
        if temporary.exists():
            temporary.unlink()

        frame = pl.concat([pl.read_parquet(path) for path in shards], how="diagonal_relaxed")
        frame = frame.sort(("update", "scope", "scope_index"))  # deterministic analysis order
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(self.final_path)
        return self.final_path


def write_selected_trajectories_hdf5(
    path: Path | str,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Path:
    'Write selected trajectories HDF5; shapes [str,np.ndarray].'

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if not arrays or any(not name for name in arrays):
        raise ValueError("trajectory HDF5 requires non-empty named arrays")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = "1.0.0"
        handle.attrs["metadata_json"] = json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        for name, array in arrays.items():
            value = np.asarray(array)
            if value.dtype.kind not in "biuf":
                raise TypeError(f"trajectory array {name!r} must be numeric or bool")
            handle.create_dataset(
                name,
                data=value,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=True,
            )
        handle.flush()
    temporary.replace(destination)
    return destination


__all__ = [
    "PALM_ROTATION_METRICS_SCHEMA",
    "PALM_ROTATION_METRICS_SCHEMA_VERSION",
    "PalmRotationMetricsRecorder",
    "write_selected_trajectories_hdf5",
]
