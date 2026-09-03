r"""MVP80 Parquet分片、resume、final compaction与HDF5轨迹合同。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest
from anymani.distill.diagnostics.recording.rl.palm_rotation import (
    PalmRotationMetricsRecorder,
    write_selected_trajectories_hdf5,
)


def _row(update: int, *, scope: str = "global", scope_index: int = 0) -> dict[str, object]:
    r"""构造最小固定schema row；其余scope-specific metrics应落为null。"""

    return {
        "update": update,
        "transitions": update * 76800,
        "scope": scope,
        "scope_index": scope_index,
        "reward_mean": float(update),
        "net_turns_mean": 0.1 * update,
    }


def test_parquet_shards_resume_and_final_compaction(tmp_path: Path) -> None:
    r"""每2-update原子flush，checkpoint inventory恢复后继续并合并单表。"""

    identity = "a" * 64
    recorder = PalmRotationMetricsRecorder(tmp_path, identity_digest=identity, flush_every_updates=2)
    assert recorder.record([_row(1)]) == []
    shards = recorder.record([_row(2, scope="cell", scope_index=3)])
    assert len(shards) == 1 and shards[0].is_file()
    assert not tuple((tmp_path / "metrics_shards").glob("*.tmp"))
    state = recorder.state_dict()

    resumed = PalmRotationMetricsRecorder(tmp_path, identity_digest=identity, flush_every_updates=2)
    resumed.load_state_dict(state)
    resumed.record([_row(3, scope="asset", scope_index=17)])
    resumed.flush(reason="checkpoint")
    resumed_state = resumed.state_dict()
    assert resumed_state["last_recorded_update"] == 3
    final_path = resumed.finalize()
    frame = pl.read_parquet(final_path).sort("update")
    assert frame["update"].to_list() == [1, 2, 3]
    assert frame["scope"].to_list() == ["global", "cell", "asset"]
    assert frame["identity_digest"].unique().to_list() == [identity]
    assert frame["goal_count_mean"].null_count() == 3


def test_recorder_rejects_unknown_metric_and_tampered_shard(tmp_path: Path) -> None:
    r"""Schema漂移与checkpoint后shard字节替换都必须fail closed。"""

    identity = "b" * 64
    recorder = PalmRotationMetricsRecorder(tmp_path, identity_digest=identity, flush_every_updates=1)
    with pytest.raises(ValueError, match="unknown fields"):
        recorder.record([{**_row(1), "unversioned_metric": 1.0}])
    shard = recorder.record([_row(1)])[0]
    state = recorder.state_dict()
    shard.write_bytes(shard.read_bytes() + b"tamper")
    resumed = PalmRotationMetricsRecorder(tmp_path, identity_digest=identity, flush_every_updates=1)
    with pytest.raises(RuntimeError, match="shards disagree"):
        resumed.load_state_dict(state)


def test_selected_trajectory_hdf5_roundtrip_is_compressed_and_atomic(tmp_path: Path) -> None:
    r"""Dense trajectory保持shape/dtype/metadata，最终目录不残留临时文件。"""

    destination = tmp_path / "evaluation" / "trajectories.h5"
    arrays = {
        "yaw_rad": np.linspace(0.0, 2.0 * np.pi, 24, dtype=np.float32).reshape(2, 12),
        "done": np.zeros((2, 12), dtype=np.bool_),
    }
    path = write_selected_trajectories_hdf5(
        destination,
        arrays=arrays,
        metadata={"checkpoint_update": 320, "identity_digest": "c" * 64},
    )
    assert path == destination and path.is_file()
    assert not destination.with_suffix(".h5.tmp").exists()
    with h5py.File(path, "r") as handle:
        yaw = handle["yaw_rad"]
        done = handle["done"]
        assert isinstance(yaw, h5py.Dataset) and isinstance(done, h5py.Dataset)
        np.testing.assert_array_equal(yaw[:], arrays["yaw_rad"])
        np.testing.assert_array_equal(done[:], arrays["done"])
        assert yaw.compression == "gzip"
        assert handle.attrs["schema_version"] == "1.0.0"
