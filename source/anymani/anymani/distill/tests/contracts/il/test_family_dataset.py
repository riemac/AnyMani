r"""两族离线教师轨迹的数据合同：HDF5 流写、History30 无损重建与质量门。

这些测试只使用 NumPy、h5py 与临时文件，不导入 Isaac Sim、Kit、teacher 或训练器。
每条命题都对应一个会改变科研结论的边界：时间轴不能重复或跳步，H0 必须包含当前帧，
inactive 行不能把 ghost 状态当作监督样本，失败轨迹必须保留且不能进入合格门。
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from anymani.distill.il.family_dataset import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    FamilyTrajectoryWriter,
    episode_replica_split,
    quality_episode_mask,
    read_family_metadata,
    reconstruct_history,
)


def _metadata() -> dict[str, object]:
    r"""构造具有完整 lineage 的可追溯元数据；哈希字符串仅为测试身份哨兵。"""

    return {
        "family": "leap",
        "teacher_checkpoint_sha256": "1" * 64,
        "cohort_sha256": "2" * 64,
        "n040_sha256": "3" * 64,
        "ordered_assets": ["asset-0", "asset-1"],
        "protocol": {"action_mode": "mean", "action_seed": None},
        "actor_abi": {
            "arm": "direct_token",
            "history_encoder": "tcn",
            "history_length": 30,
            "joint_count": 16,
            "owner_count": 21,
            "geometry_width": 128,
            "actor_contact": "tip-only-binary",
            "phase_clock_enabled": False,
            "joint_kinematics_width": 15,
        },
    }


def _static() -> dict[str, np.ndarray]:
    r"""构造两个资产的最小 static evidence；所有槽有效以便独立测试动态合同。"""

    return {
        "actor_jnt_limits": np.tile(np.array([[-1.0, 1.0]], dtype=np.float32), (2, 16, 1)),
        "jnt_valid": np.ones((2, 16), dtype=bool),
        "tip_valid": np.ones((2, 4), dtype=bool),
        "owner_valid": np.ones((2, 21), dtype=bool),
        "shortest_path": np.zeros((2, 21, 21), dtype=np.int16),
        "parent_direction": np.zeros((2, 21, 21), dtype=np.int8),
        "child_direction": np.zeros((2, 21, 21), dtype=np.int8),
        "joint_kinematics": np.zeros((2, 16, 15), dtype=np.float32),
    }


def _initial_history(envs: int = 2) -> np.ndarray:
    r"""返回 H0，最后一帧是 step0 current，数值保持在动作/接触合法区间。"""

    history = np.zeros((envs, 30, 16, 5), dtype=np.float32)
    history[:, :, :, 0] = np.linspace(-0.3, 0.3, 30, dtype=np.float32)[None, :, None]
    history[:, :, :, 1] = 0.05
    history[:, :, :, 2] = -0.05
    return history


def _step_arrays(history: np.ndarray, step: int, active: np.ndarray) -> dict[str, np.ndarray]:
    r"""由 H0 生成一个连续合法 step；测试调用者在需要时覆盖 inactive 行 history。"""

    envs = history.shape[0]
    current = np.zeros((envs, 16, 5), dtype=np.float32)
    if step == 0:
        current[...] = history[:, -1]
    else:
        current[..., 0] = np.float32(0.01 * step)
        current[..., 1] = 0.02
        current[..., 2] = -0.02
    owner_contact = np.zeros((envs, 21, 1), dtype=np.float32)
    teacher_mean = np.full((envs, 16), 0.1, dtype=np.float32)
    behavior_action = np.full((envs, 16), -0.1, dtype=np.float32)
    teacher_mean[~active.astype(bool)] = 0.0
    behavior_action[~active.astype(bool)] = 0.0
    geometry_tokens = np.zeros((envs, 21, 128), dtype=np.float32)
    joint_origin_fk = np.zeros((envs, 16, 3), dtype=np.float32)
    return {
        "jnt_current": current,
        "owner_contact": owner_contact,
        "teacher_mean": teacher_mean,
        "behavior_action": behavior_action,
        "geometry_tokens": geometry_tokens,
        "active": active.astype(np.float32),
        "history": history.copy(),
        "joint_origin_fk": joint_origin_fk,
    }


def _summary(envs: int = 2) -> dict[str, np.ndarray]:
    r"""构造全量 final 统计；第二行故意低于净圈门以验证失败保留。"""

    return {
        "net_turns": np.array([0.6, 0.4], dtype=np.float32)[:envs],
        "path_turns": np.array([0.8, 0.6], dtype=np.float32)[:envs],
        "duration_s": np.full(envs, 30.0, dtype=np.float32),
        "termination_drop": np.zeros(envs, dtype=bool),
        "termination_axis": np.zeros(envs, dtype=bool),
    }


def test_stream_layout_history_and_traceability(tmp_path: Path) -> None:
    r"""流写只保留首个 H0 与连续 current；root/static/frames/samples/final 身份闭合。"""

    path = tmp_path / "family.h5"
    h0 = _initial_history()
    env_asset = np.array([0, 1], dtype=np.int64)
    env_replica = np.array([0, 3], dtype=np.int64)
    with FamilyTrajectoryWriter(
        path,
        _metadata(),
        steps=2,
        env_asset_index=env_asset,
        env_replica_index=env_replica,
        static=_static(),
        initial_history=h0,
        sample_stride=2,
    ) as writer:
        writer.append(step=0, **_step_arrays(h0, 0, np.ones(2, dtype=bool)))
        step_one = _step_arrays(h0, 1, np.ones(2, dtype=bool))
        step_one["history"] = np.concatenate((h0[:, 1:], step_one["jnt_current"][:, None]), axis=1)
        step_one.pop("joint_origin_fk")  # 非采样步允许省略 FK，避免重复存储。
        writer.append(step=1, **step_one)
        writer.finalize(_summary())

    metadata = read_family_metadata(path)
    assert metadata["artifact_type"] == ARTIFACT_TYPE
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["family"] == "leap"
    assert metadata["teacher_checkpoint_sha256"] == "1" * 64
    assert metadata["ordered_assets"] == ["asset-0", "asset-1"]
    assert metadata["protocol"] == {"action_mode": "mean", "action_seed": None}
    assert "checkpoint_sha256" not in metadata and "asset_identities" not in metadata
    assert metadata["completed"] is True
    assert metadata["recorded_steps"] == 2 and metadata["sample_count"] == 1
    with h5py.File(path, "r") as stream:
        assert stream.attrs["artifact_type"] == ARTIFACT_TYPE
        assert stream.attrs["schema_version"] == SCHEMA_VERSION
        assert stream["initial_history"].dtype == np.dtype("float32")
        assert stream["frames/jnt_current"].shape == (2, 2, 16, 5)
        assert stream["frames/owner_contact"].dtype == np.dtype("float32")
        assert stream["frames/active"].dtype == np.dtype("float32")
        assert stream["samples/step_index"].shape == (1,)
        assert stream["samples/teacher_mean"].dtype == np.dtype("float32")
        assert stream["samples/geometry_tokens"].shape == (1, 2, 21, 128)
        assert stream["samples/joint_origin_fk"].dtype == np.dtype("float32")
        assert stream["samples/joint_origin_fk"].shape == (1, 2, 16, 3)
        assert stream["final/quality_episode_mask"][:].tolist() == [True, False]


def test_history_reconstruction_t0_t29_t30_has_no_future_leak() -> None:
    r"""t0 使用 H0，t29 保留 H0 最后一帧，t30 完全由 current1..30 组成。"""

    h0 = _initial_history(envs=1)
    frames = np.zeros((32, 1, 16, 5), dtype=np.float32)
    frames[0, 0] = h0[0, -1]
    for step in range(1, 32):
        frames[step, 0, 0, 0] = step / 100.0
    result = reconstruct_history(
        h0,
        frames,
        np.array([0, 29, 30], dtype=np.int64),
        np.array([0, 0, 0], dtype=np.int64),
    )
    assert np.array_equal(result[0], h0[0])
    assert np.array_equal(result[1], np.concatenate((h0[0, -1:], frames[1:30, 0]), axis=0))
    assert np.array_equal(result[2], frames[1:31, 0])
    future = result[2].copy()
    frames[31, 0, 0, 0] = 0.99
    result_after_future_edit = reconstruct_history(h0, frames, np.array([30]), np.array([0]))
    assert np.array_equal(result_after_future_edit[0], future)


def test_inactive_row_is_not_compared_and_active_mask_cannot_reactivate(tmp_path: Path) -> None:
    r"""inactive 行可携带 padding history；active 只能 true→false，不能伪造 reset 后新回合。"""

    path = tmp_path / "inactive.h5"
    h0 = _initial_history()
    with FamilyTrajectoryWriter(
        path,
        _metadata(),
        steps=3,
        env_asset_index=np.array([0, 1], dtype=np.int64),
        env_replica_index=np.array([0, 1], dtype=np.int64),
        static=_static(),
        initial_history=h0,
        sample_stride=1,
    ) as writer:
        writer.append(step=0, **_step_arrays(h0, 0, np.ones(2, dtype=bool)))
        row = _step_arrays(h0, 1, np.array([True, False]))
        row["history"][0] = np.concatenate((h0[0, 1:], row["jnt_current"][0:1]), axis=0)
        row["history"][1] = 7.0  # inactive padding 不参与 History30 连续性比较。
        row["teacher_mean"][1] = 0.35  # evaluator 已在 reset 后计算动作；inactive 仅由 active mask 排除。
        row["behavior_action"][1] = -0.35
        writer.append(step=1, **row)
        previous_history = row["history"][0].copy()
        row = _step_arrays(h0, 2, np.array([True, False]))
        row["history"][0] = np.concatenate((previous_history[1:], row["jnt_current"][0:1]), axis=0)
        row["history"][1] = -9.0
        row["teacher_mean"][1] = 0.25
        row["behavior_action"][1] = -0.25
        writer.append(step=2, **row)
        writer.finalize(_summary())

    assert read_family_metadata(path)["completed"] is True
    with h5py.File(path, "r") as stream:
        assert stream["frames/active"][1, 1] == 0.0
        assert stream["samples/teacher_mean"][1, 1, 0] == pytest.approx(0.35)
        assert stream["samples/behavior_action"][2, 1, 0] == pytest.approx(-0.25)


def test_active_transition_cannot_reactivate_after_terminal_padding(tmp_path: Path) -> None:
    r"""true→false 后的相同 replica 不得重新变 true，避免跨 reset 拼接两个 episode。"""

    h0 = _initial_history()
    path = tmp_path / "reactivation.h5"
    writer = FamilyTrajectoryWriter(
        path,
        _metadata(),
        steps=3,
        env_asset_index=np.array([0, 1], dtype=np.int64),
        env_replica_index=np.array([0, 1], dtype=np.int64),
        static=_static(),
        initial_history=h0,
        sample_stride=4,
    )
    writer.append(step=0, **_step_arrays(h0, 0, np.ones(2, dtype=bool)))
    writer.append(step=1, **_step_arrays(h0, 1, np.zeros(2, dtype=bool)))
    with pytest.raises(ValueError, match="reactivation|true to false"):
        writer.append(step=2, **_step_arrays(h0, 2, np.ones(2, dtype=bool)))
    writer.close()


def test_step_order_output_no_overwrite_and_incomplete_is_unreadable(tmp_path: Path) -> None:
    r"""重复/遗漏 step fail closed；输出文件使用 create-excl，未 finalize 文件不可读。"""

    path = tmp_path / "incomplete.h5"
    h0 = _initial_history()
    kwargs = dict(
        metadata=_metadata(),
        steps=2,
        env_asset_index=np.array([0, 1], dtype=np.int64),
        env_replica_index=np.array([0, 1], dtype=np.int64),
        static=_static(),
        initial_history=h0,
    )
    writer = FamilyTrajectoryWriter(path, **kwargs)
    writer.append(step=0, **_step_arrays(h0, 0, np.ones(2, dtype=bool)))
    with pytest.raises(ValueError, match="contiguous|step"):
        writer.append(step=2, **_step_arrays(h0, 1, np.ones(2, dtype=bool)))
    writer.close()
    with pytest.raises(RuntimeError, match="incomplete|completed"):
        read_family_metadata(path)
    with pytest.raises(FileExistsError):
        FamilyTrajectoryWriter(path, **kwargs)


def test_invalid_action_shape_ghost_and_mask_are_rejected(tmp_path: Path) -> None:
    r"""动作必须是 [N,16] 的有限 [-1,1] 值；无效 joint 的 ghost 动作必须精确为零。"""

    static = _static()
    static["jnt_valid"][1, -1] = False
    static["owner_valid"][1, 1 + 15] = False
    h0 = _initial_history()
    h0[1, :, -1, :] = 0.0
    path = tmp_path / "invalid.h5"
    writer = FamilyTrajectoryWriter(
        path,
        _metadata(),
        steps=1,
        env_asset_index=np.array([0, 1], dtype=np.int64),
        env_replica_index=np.array([0, 1], dtype=np.int64),
        static=static,
        initial_history=h0,
    )
    values = _step_arrays(h0, 0, np.ones(2, dtype=bool))
    values["teacher_mean"] = np.zeros((2, 15), dtype=np.float32)
    with pytest.raises(ValueError, match="teacher_mean.*shape"):
        writer.append(step=0, **values)
    values = _step_arrays(h0, 0, np.ones(2, dtype=bool))
    values["teacher_mean"][1, -1] = 0.2
    with pytest.raises(ValueError, match="ghost"):
        writer.append(step=0, **values)
    writer.close()


def test_joint_kinematics_and_actor_abi_are_required(tmp_path: Path) -> None:
    r"""15D static kinematics 与 actor ABI 缺一不可，避免旧 geometry 表混入新训练。"""

    h0 = _initial_history()
    static = _static()
    static.pop("joint_kinematics")
    with pytest.raises(ValueError, match="joint_kinematics"):
        FamilyTrajectoryWriter(
            tmp_path / "missing-kinematics.h5",
            _metadata(),
            steps=1,
            env_asset_index=np.array([0, 1], dtype=np.int64),
            env_replica_index=np.array([0, 1], dtype=np.int64),
            static=static,
            initial_history=h0,
        )

    metadata = _metadata()
    metadata["actor_abi"] = {}
    with pytest.raises(ValueError, match="actor_abi"):
        FamilyTrajectoryWriter(
            tmp_path / "missing-abi.h5",
            metadata,
            steps=1,
            env_asset_index=np.array([0, 1], dtype=np.int64),
            env_replica_index=np.array([0, 1], dtype=np.int64),
            static=_static(),
            initial_history=h0,
        )


def test_metadata_uses_only_canonical_lineage_names(tmp_path: Path) -> None:
    r"""旧别名/大小写变体不能被猜测，actor ABI 任一稳定维度变化都必须拒绝。"""

    metadata = _metadata()
    metadata.pop("teacher_checkpoint_sha256")
    metadata["checkpoint_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="teacher_checkpoint_sha256"):
        FamilyTrajectoryWriter(
            tmp_path / "alias.h5",
            metadata,
            steps=1,
            env_asset_index=np.array([0, 1], dtype=np.int64),
            env_replica_index=np.array([0, 1], dtype=np.int64),
            static=_static(),
            initial_history=_initial_history(),
        )

    metadata = _metadata()
    metadata["actor_abi"] = dict(metadata["actor_abi"])
    metadata["actor_abi"]["geometry_width"] = 127  # type: ignore[index]
    with pytest.raises(ValueError, match="actor_abi"):
        FamilyTrajectoryWriter(
            tmp_path / "abi-drift.h5",
            metadata,
            steps=1,
            env_asset_index=np.array([0, 1], dtype=np.int64),
            env_replica_index=np.array([0, 1], dtype=np.int64),
            static=_static(),
            initial_history=_initial_history(),
        )


def test_sample_fk_is_required_and_early_stop_needs_terminal_proof(tmp_path: Path) -> None:
    r"""采样步缺 FK 必须拒绝；不足 nominal steps 只有 summary 的全 env terminal 证明才可封存。"""

    h0 = _initial_history()
    path = tmp_path / "early.h5"
    writer = FamilyTrajectoryWriter(
        path,
        _metadata(),
        steps=4,
        env_asset_index=np.array([0, 1], dtype=np.int64),
        env_replica_index=np.array([0, 1], dtype=np.int64),
        static=_static(),
        initial_history=h0,
        sample_stride=2,
    )
    row = _step_arrays(h0, 0, np.ones(2, dtype=bool))
    row.pop("joint_origin_fk")
    with pytest.raises(ValueError, match="joint_origin_fk|FK"):
        writer.append(step=0, **row)
    writer.close()

    path = tmp_path / "early-proven.h5"
    with FamilyTrajectoryWriter(
        path,
        _metadata(),
        steps=4,
        env_asset_index=np.array([0, 1], dtype=np.int64),
        env_replica_index=np.array([0, 1], dtype=np.int64),
        static=_static(),
        initial_history=h0,
        sample_stride=2,
    ) as writer:
        writer.append(step=0, **_step_arrays(h0, 0, np.zeros(2, dtype=bool)))
        with pytest.raises(ValueError, match="terminated"):
            writer.finalize(_summary())
        early_summary = _summary()
        early_summary.update(
            {
                "duration_s": np.full(2, 1.0, dtype=np.float32),
                "terminated": np.ones(2, dtype=bool),
            }
        )
        writer.finalize(early_summary)
    assert not bool(read_family_metadata(path)["has_qualified_episode"])


def test_quality_gate_preserves_failures_and_episode_split_isolated() -> None:
    r"""完整30秒、净圈与方向、安全旗共同决定 mask；零合格不会被补造。"""

    mask = quality_episode_mask(
        np.array([0.6, 0.6, 0.6, 0.4, 0.6]),
        np.array([0.8, 0.6, 0.8, 0.8, 0.8]),
        np.array([30.0, 29.9998, 30.0, 30.0, 30.0]),
        np.array([False, False, True, False, False]),
        np.array([False, False, False, False, True]),
    )
    assert mask.tolist() == [True, True, False, False, False]
    no_sample = quality_episode_mask(
        np.zeros(2), np.ones(2), np.full(2, 10.0), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool)
    )
    assert no_sample.shape == (2,) and not bool(no_sample.any())

    replica = np.array([0, 3, 4, 7, 8, 11], dtype=np.int64)
    train, validation = episode_replica_split(replica)
    assert set(replica[train].tolist()).isdisjoint(set(replica[validation].tolist()))
    assert np.all(replica[validation] % 4 == 3)
    assert np.all(replica[train] % 4 != 3)
