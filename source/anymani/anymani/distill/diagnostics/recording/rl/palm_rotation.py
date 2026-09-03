r"""MVP80训练的Zstd Parquet标量分片与gzip HDF5轨迹artifact。

训练标量以``scope in {global,cell,asset}``共享固定宽schema。每50 updates或checkpoint前，pending rows
先写同文件系统``.tmp``，完成后原子rename为不可变Parquet shard；run完成时按``(update,scope,scope_index)``
排序并原子发布单个``metrics.parquet``。Recorder只落盘调用方已经计算的事实，不运行环境、模型或梯度。

Selected-checkpoint trajectories保留逐时刻dense arrays，使用HDF5 gzip+shuffle。Parquet服务宽标量查询，
HDF5服务时间序列重算；TensorBoard仅作为global/8-cell在线视图，不是唯一事实源。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pl

PALM_ROTATION_METRICS_SCHEMA_VERSION = "2.0.0"

# 一个固定nullable宽表避免不同scope/flush由inference产生不兼容Parquet schemas。
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
    "goal_count_mean": pl.Float64,
    "net_turns_mean": pl.Float64,
    "drop_rate": pl.Float64,
    "axis_failure_rate": pl.Float64,
    "tip_contact_mean": pl.Float64,
    "palm_contact_rate": pl.Float64,
    "non_tip_contact_rate": pl.Float64,
    "advantage_mean": pl.Float64,
    "advantage_std": pl.Float64,
    "value_error_mean": pl.Float64,
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
    "film_modulation_rms": pl.Float64,
    "candidate_lambda": pl.Float64,
    "actual_lambda": pl.Float64,
    "counterfactual_adr_level": pl.Float64,
    "actor_grad_norm": pl.Float64,
    "critic_grad_norm": pl.Float64,
    "gradient_cosine": pl.Float64,
    "actor_loss": pl.Float64,
    "critic_loss": pl.Float64,
    "entropy": pl.Float64,
    "policy_sigma": pl.Float64,
    "actor_base_lr": pl.Float64,
    "actor_residual_lr": pl.Float64,
    "critic_lr": pl.Float64,
    "optimizer_microbatches": pl.Float64,
    "optimizer_steps": pl.Float64,
    "rollout_sample_count": pl.Float64,
    "completed_episode_count": pl.Float64,
    "terminal_goal_count_mean": pl.Float64,
    "terminal_net_turns_mean": pl.Float64,
    "terminal_absolute_path_turns_mean": pl.Float64,
    "terminal_directional_consistency_mean": pl.Float64,
    "terminal_timeout_rate": pl.Float64,
    "terminal_drop_rate": pl.Float64,
    "terminal_axis_failure_rate": pl.Float64,
    "steps_per_second": pl.Float64,
    "gpu_memory_bytes": pl.Int64,
    "process_rss_bytes": pl.Int64,
    "process_peak_rss_bytes": pl.Int64,
    "process_swap_bytes": pl.Int64,
    "system_available_memory_bytes": pl.Int64,
}


def _sha256(path: Path) -> str:
    r"""流式计算不可变shard或trajectory artifact摘要。"""

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
    r"""把一个global/cell/asset scalar row投影到固定nullable schema。"""

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
    r"""维护run-local pending rows、原子Parquet shards与final compaction state。"""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        identity_digest: str,
        flush_every_updates: int = 50,
    ) -> None:
        r"""构造recorder并验证固定Polars版本和输出目录。

        Args:
            run_dir (Path | str): 当前PPO run根目录。
            identity_digest (str): checkpoint/runtime exact identity SHA-256。
            flush_every_updates (int): 自动flush cadence；正式MVP固定50 updates。
        """

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
        r"""返回尚未写入Parquet的标量行数。"""

        return len(self._pending)

    def record(self, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
        r"""追加一个update的global/cell/asset rows，并在50-update边界自动flush。

        Args:
            rows (Sequence[Mapping[str, Any]]): 同一update的1 global、8 cell和80 asset宽表事实。

        Returns:
            list[Path]: 本次触发flush时发布的单一shard，否则为空。
        """

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
            self._pending_first_update = update  # 当前shard起始update
        self._pending.extend(normalized)
        self._last_recorded_update = update
        if update - self._pending_first_update + 1 >= self.flush_every_updates:
            path = self.flush(reason="cadence")
            return [path] if path is not None else []
        return []

    def flush(self, *, reason: str) -> Path | None:
        r"""把pending rows以Zstd Parquet原子发布；checkpoint前必须调用。

        Args:
            reason (str): ``cadence``、``checkpoint``、``finalize``或``shutdown``等证据标签。

        Returns:
            Path | None: 已发布shard；无pending rows时返回None。
        """

        if not self._pending:
            return None
        first_update = int(self._pending_first_update or self._last_recorded_update)
        last_update = self._last_recorded_update
        filename = f"metrics-{self._next_shard_index:06d}-u{first_update:08d}-u{last_update:08d}.parquet"
        destination = self.shard_dir / filename  # immutable shard final path
        temporary = destination.with_suffix(".parquet.tmp")  # same-filesystem atomic source
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"metrics shard path already exists: {destination}")

        # Fixed schema ensures null-only columns retain intended numeric dtype in every independent shard。
        frame = pl.DataFrame(self._pending, schema=PALM_ROTATION_METRICS_SCHEMA)
        frame = frame.with_columns(pl.lit(reason).alias("_flush_reason"))  # shard lifecycle evidence
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(destination)  # readers observe either no shard or a complete footer/data file
        self._pending.clear()
        self._pending_first_update = None
        self._next_shard_index += 1
        return destination

    def state_dict(self) -> dict[str, Any]:
        r"""返回checkpoint内的recorder continuation state。

        Checkpoint前调用方应先``flush(reason="checkpoint")``，因此pending rows必须为零；保存每个shard SHA
        可在resume时证伪截断、替换或错误run directory。
        """

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
        r"""核对identity与现有shards后恢复append位置。"""

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
        r"""flush最后rows并原子合并全部shards为``metrics.parquet``。"""

        self.flush(reason="finalize")
        shards = sorted(self.shard_dir.glob("metrics-*.parquet"))
        if not shards:
            raise RuntimeError("cannot finalize an empty metrics recorder")
        temporary = self.final_path.with_suffix(".parquet.tmp")
        if temporary.exists():
            temporary.unlink()  # 只清理本finalize调用拥有的同名未发布临时文件
        # Diagonal concat允许未来schema显式增加nullable列；当前所有shards仍由固定schema生成。
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
    r"""原子写selected-checkpoint dense trajectories。

    Args:
        path (Path | str): 目标``.h5``路径。
        arrays (Mapping[str, np.ndarray]): 带trajectory/time轴的dense数值或bool arrays。
        metadata (Mapping[str, Any]): JSON-safe checkpoint/evaluation/identity信息。

    Returns:
        Path: 完整发布的HDF5路径。
    """

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if not arrays or any(not name for name in arrays):
        raise ValueError("trajectory HDF5 requires non-empty named arrays")
    if temporary.exists():
        temporary.unlink()  # 当前目标的旧未发布临时文件不属于有效artifact
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = "1.0.0"
        handle.attrs["metadata_json"] = json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        for name, array in arrays.items():
            value = np.asarray(array)  # 保存调用方已形成的trajectory事实，不重算
            if value.dtype.kind not in "biuf":
                raise TypeError(f"trajectory array {name!r} must be numeric or bool")
            handle.create_dataset(
                name,
                data=value,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=True,
            )  # gzip跨环境可读；shuffle改善连续float/bit压缩率
        handle.flush()
    temporary.replace(destination)
    return destination


__all__ = [
    "PALM_ROTATION_METRICS_SCHEMA",
    "PALM_ROTATION_METRICS_SCHEMA_VERSION",
    "PalmRotationMetricsRecorder",
    "write_selected_trajectories_hdf5",
]
