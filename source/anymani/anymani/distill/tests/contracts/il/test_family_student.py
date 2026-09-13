r"""两族离线共享学生的关键输入、平衡监督与独立 checkpoint 合同。

这些测试只构造纯 PyTorch 张量，不启动 Isaac Sim；训练测试是低量 CPU canary，不能替代冻结物理评价。
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from anymani.distill.il.family_dataset import FamilyTrajectoryWriter
from anymani.distill.il.family_student import (
    FAMILY_STUDENT_ACTOR_ABI,
    FamilyStudentConfig,
    build_family_student,
    export_family_student_torchscript,
    load_family_student,
    save_family_student_checkpoint,
)
from anymani.distill.il.train_family import (
    METRICS_FILENAME,
    TRAINING_REPORT_FILENAME,
    _atomic_append_jsonl,
    _atomic_write_json,
    _load_metrics_records,
    _quality_for_handle,
    _validate_output_state,
    balanced_family_action_mse,
    balanced_mean_action_mse,
    estimate_compact_resident_bytes,
    fit_family_student,
    load_family_sources,
)
from anymani.distill.models.palm_rotation_policy import PalmRotationActorObservation, PalmRotationGeometry


def _batch(batch: int = 4) -> tuple[PalmRotationActorObservation, PalmRotationGeometry, torch.Tensor]:
    r"""构造包含 9/12/16 DoF 的 canonical actor packet 和静态 joint kinematics。"""

    generator = torch.Generator().manual_seed(19)
    jnt_valid = torch.ones(batch, 16, dtype=torch.bool)
    jnt_valid[0, 9:] = False
    if batch > 1:
        jnt_valid[1, 12:] = False
    tip_valid = torch.ones(batch, 4, dtype=torch.bool)
    owner_valid = torch.cat((torch.ones(batch, 1, dtype=torch.bool), jnt_valid, tip_valid), dim=-1)
    current = torch.randn(batch, 16, 5, generator=generator)
    history = torch.randn(batch, 30, 16, 5, generator=generator)
    limits = torch.stack((-torch.ones(batch, 16), torch.ones(batch, 16)), dim=-1)
    contact = torch.rand(batch, 21, 1, generator=generator)
    observation = PalmRotationActorObservation(current, history, limits, contact, jnt_valid, tip_valid, owner_valid)
    tokens = torch.randn(batch, 21, 128, generator=generator)
    graph = torch.zeros(batch, 21, 21, dtype=torch.long)
    geometry = PalmRotationGeometry(tokens, owner_valid, graph, graph.clone(), graph.clone())
    kinematics = torch.randn(batch, 16, 15, generator=generator)
    return observation, geometry, kinematics


def _write_tiny_collection(path: Path, *, action_mode: str, action_seed: int | None) -> None:
    r"""写一个 completed、无合格 episode 的微型 collection，用于验证 source identity 而非训练效果。"""

    static = {
        "actor_jnt_limits": np.tile(np.array([[-1.0, 1.0]], dtype=np.float32), (1, 16, 1)),
        "jnt_valid": np.ones((1, 16), dtype=bool),
        "tip_valid": np.ones((1, 4), dtype=bool),
        "owner_valid": np.ones((1, 21), dtype=bool),
        "shortest_path": np.zeros((1, 21, 21), dtype=np.int16),
        "parent_direction": np.zeros((1, 21, 21), dtype=np.int8),
        "child_direction": np.zeros((1, 21, 21), dtype=np.int8),
        "joint_kinematics": np.zeros((1, 16, 15), dtype=np.float32),
    }
    zeros_current = np.zeros((1, 16, 5), dtype=np.float32)
    zeros_contact = np.zeros((1, 21, 1), dtype=np.float32)
    zeros_action = np.zeros((1, 16), dtype=np.float32)
    zeros_tokens = np.zeros((1, 21, 128), dtype=np.float32)
    zeros_fk = np.zeros((1, 16, 3), dtype=np.float32)
    initial_history = np.zeros((1, 30, 16, 5), dtype=np.float32)
    metadata = {
        "family": "leap",
        "teacher_checkpoint_sha256": "t" * 64,
        "cohort_sha256": "c" * 64,
        "n040_sha256": "n" * 64,
        "ordered_assets": ["asset-0"],
        "protocol": {"action_mode": action_mode, "action_seed": action_seed},
        "actor_abi": FAMILY_STUDENT_ACTOR_ABI,
    }
    with FamilyTrajectoryWriter(
        path,
        metadata,
        steps=1,
        env_asset_index=np.array([0], dtype=np.int64),
        env_replica_index=np.array([0], dtype=np.int64),
        static=static,
        initial_history=initial_history,
        sample_stride=1,
    ) as writer:
        writer.append(
            0,
            jnt_current=zeros_current,
            owner_contact=zeros_contact,
            teacher_mean=zeros_action,
            behavior_action=zeros_action,
            geometry_tokens=zeros_tokens,
            active=np.zeros(1, dtype=np.float32),
            history=initial_history,
            joint_origin_fk=zeros_fk,
        )
        writer.finalize(
            {
                "net_turns": np.zeros(1, dtype=np.float32),
                "path_turns": np.zeros(1, dtype=np.float32),
                "duration_s": np.zeros(1, dtype=np.float32),
                "termination_drop": np.zeros(1, dtype=bool),
                "termination_axis": np.zeros(1, dtype=bool),
            }
        )


def test_actor_abi_and_dof_normalized_loss() -> None:
    r"""动作损失只在有效 joint 上平均，ghost DoF 不改变分母。"""

    assert FAMILY_STUDENT_ACTOR_ABI["arm"] == "direct_token"
    assert FAMILY_STUDENT_ACTOR_ABI["history_length"] == 30
    prediction = torch.zeros(1, 16)
    prediction[0, 0] = 1.0
    target = torch.zeros_like(prediction)
    valid = torch.zeros(1, 16, dtype=torch.bool)
    valid[0, 0] = True
    assert balanced_family_action_mse(prediction, target, valid).item() == pytest.approx(1.0)


def test_balanced_family_and_asset_weights_do_not_follow_data_volume() -> None:
    r"""family→asset 两级等权使一个大资产的重复样本不能主导目标。"""

    prediction = torch.zeros(4, 16)
    prediction[:2] = 1.0
    target = torch.zeros_like(prediction)
    valid = torch.ones_like(prediction, dtype=torch.bool)
    value = balanced_family_action_mse(
        prediction,
        target,
        valid,
        family_ids=("leap", "leap", "leap", "allegro"),
        asset_ids=("a", "a", "b", "c"),
    )
    assert value.item() == pytest.approx(0.25)


def test_balanced_validation_metric_averages_assets_then_families() -> None:
    r"""best 指标先等权同族资产，再等权两族，避免 sample 数最多的族决定 checkpoint。"""

    per_asset = {
        "leap/asset-a": {"samples": 1000, "mean_action_mse": 1.0},
        "leap/asset-b": {"samples": 1, "mean_action_mse": 3.0},
        "allegro/asset-c": {"samples": 2, "mean_action_mse": 2.0},
    }
    assert balanced_mean_action_mse(per_asset) == pytest.approx(2.0)


def test_same_teacher_different_collection_protocols_can_merge(tmp_path: Path) -> None:
    r"""mean 与 sample 共享 teacher SHA 仍是两个合法 collection，并可共同进入 family/asset 测度。"""

    mean_path = tmp_path / "mean.h5"
    sample_path = tmp_path / "sample.h5"
    _write_tiny_collection(mean_path, action_mode="mean", action_seed=None)
    _write_tiny_collection(sample_path, action_mode="sample", action_seed=7)
    bundle = load_family_sources((mean_path, sample_path), max_ram_gib=1.0, batch_size=2)
    assert len(bundle.sources) == 2
    assert bundle.sources[0].source_sha256 == bundle.sources[1].source_sha256
    assert bundle.sources[0].collection_identity != bundle.sources[1].collection_identity


def test_duplicate_collection_identity_is_rejected_even_under_different_path(tmp_path: Path) -> None:
    r"""同一 collection 复制到另一文件名后，dataset/collection identity 仍阻止重复训练分母。"""

    original = tmp_path / "original.h5"
    duplicate = tmp_path / "duplicate.h5"
    _write_tiny_collection(original, action_mode="mean", action_seed=None)
    shutil.copy2(original, duplicate)
    with pytest.raises(ValueError, match="collection identity"):
        load_family_sources((original, duplicate), max_ram_gib=1.0, batch_size=2)


def test_quality_reader_trusts_writer_mask_and_rejects_protocol_mismatch(tmp_path: Path) -> None:
    r"""用 float64 同一 gate 复核：0 圈安全段为 False，优质段漏标和协议漂移都 fail closed。"""

    path = tmp_path / "quality.h5"
    with h5py.File(path, "w") as stream:
        final = stream.create_group("final")
        final.create_dataset("quality_episode_mask", data=[False, False])
        final.create_dataset("net_turns", data=[0.0, 0.0])  # 质量门只用于 admission，不进入 Actor
        final.create_dataset("path_turns", data=[0.8, 0.8])
        final.create_dataset("duration_s", data=[30.0, 30.0])
        final.create_dataset("termination_drop", data=[False, False])
        final.create_dataset("termination_axis", data=[False, False])
        final.create_dataset("policy_step_count", data=[600, 600])
        active = stream.create_group("frames").create_dataset("active", shape=(600, 2), dtype="f4")
        active[...] = 1.0
    with h5py.File(path, "r") as stream:
        assert _quality_for_handle(stream).tolist() == [False, False]

    leaked_good_path = tmp_path / "stored-false-good.h5"
    with h5py.File(leaked_good_path, "w") as stream:
        final = stream.create_group("final")
        final.create_dataset("quality_episode_mask", data=[False])
        final.create_dataset("net_turns", data=[1.0])
        final.create_dataset("path_turns", data=[1.0])
        final.create_dataset("duration_s", data=[30.0])
        final.create_dataset("termination_drop", data=[False])
        final.create_dataset("termination_axis", data=[False])
        final.create_dataset("policy_step_count", data=[600])
        active = stream.create_group("frames").create_dataset("active", shape=(600, 1), dtype="f4")
        active[...] = 1.0
    with h5py.File(leaked_good_path, "r") as stream:
        with pytest.raises(ValueError, match="stored family quality mask"):
            _quality_for_handle(stream)

    short_path = tmp_path / "short.h5"
    with h5py.File(short_path, "w") as stream:
        final = stream.create_group("final")
        final.create_dataset("quality_episode_mask", data=[False])
        final.create_dataset("net_turns", data=[0.0])
        final.create_dataset("path_turns", data=[0.8])
        final.create_dataset("duration_s", data=[2.0])
        final.create_dataset("termination_drop", data=[False])
        final.create_dataset("termination_axis", data=[False])
        active = stream.create_group("frames").create_dataset("active", shape=(2, 1), dtype="f4")
        active[...] = 1.0
    with h5py.File(short_path, "r") as stream:
        assert _quality_for_handle(stream).tolist() == [False]

    bad_path = tmp_path / "stored-true-short.h5"
    with h5py.File(bad_path, "w") as stream:
        final = stream.create_group("final")
        final.create_dataset("quality_episode_mask", data=[True])
        final.create_dataset("net_turns", data=[1.0])
        final.create_dataset("path_turns", data=[1.0])
        final.create_dataset("duration_s", data=[30.0])
        final.create_dataset("termination_drop", data=[False])
        final.create_dataset("termination_axis", data=[False])
        active = stream.create_group("frames").create_dataset("active", shape=(600, 1), dtype="f4")
        active[...] = 1.0
        active[-1, 0] = 0.0
    with h5py.File(bad_path, "r") as stream:
        with pytest.raises(ValueError, match="recomputed gate/protocol"):
            _quality_for_handle(stream)


def test_compact_resident_estimate_for_four_2048_env_cases_needs_no_allocation() -> None:
    r"""真实首批 shape=600×2048、S=150 只保留一次 Z，四 case 的上界仍落在 16 GiB。"""

    one_case = estimate_compact_resident_bytes(
        recorded_steps=600,
        env_count=2048,
        quality_env_count=2048,
        sample_count=150,
        active_pair_count=2048 * 150,
        asset_count=128,
        batch_size=2048,
    )
    assert 3.0 < one_case / 2**30 < 4.5
    assert 4 * one_case < 16 * 2**30


def test_training_report_and_metrics_resume_identity_are_atomic(tmp_path: Path) -> None:
    r"""epoch record 与 report 可独立恢复，并拒绝不同 run identity 的静默覆盖。"""

    output = tmp_path / "run"
    output.mkdir()
    identity = {"dataset_sha256": "d", "n040_sha256": "n", "representation": "no_z"}
    epoch = {"epoch": 1, "update": 2, "processed_samples": 8}
    _atomic_append_jsonl(output / METRICS_FILENAME, epoch)
    _atomic_write_json(
        output / TRAINING_REPORT_FILENAME,
        {
            "artifact_type": "anymani.family_distilled_actor_training_report",
            "schema_version": "1.0.0",
            "run_identity": identity,
            "epochs": [epoch],
            "status": "running",
            "cumulative_wall_seconds": 0.25,
            "best_validation_balanced_mean_action_mse": 0.5,
            "best_epoch": 1,
        },
    )
    checkpoint = output / "last.pt"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    report, records = _validate_output_state(output, resume=checkpoint, run_identity=identity)
    assert report is not None and records == [epoch]
    assert _load_metrics_records(output / METRICS_FILENAME) == [epoch]
    with pytest.raises(ValueError, match="run_identity"):
        _validate_output_state(output, resume=checkpoint, run_identity={**identity, "representation": "n040"})
    with pytest.raises(FileExistsError, match="non-empty"):
        _validate_output_state(output, resume=None, run_identity=identity)


def test_no_z_ignores_token_values_but_keeps_graph_contract() -> None:
    r"""No-Z 仅抹除 geometry token 数值，图关系、mask、limits/history/kinematics 不删除。"""

    observation, geometry, kinematics = _batch()
    torch.manual_seed(5)
    actor = build_family_student("no_z", device="cpu")
    actor.eval()
    changed = PalmRotationGeometry(
        geometry.tokens + 7.0,
        geometry.owner_valid,
        geometry.shortest_path,
        geometry.parent_direction,
        geometry.child_direction,
    )
    first = actor(observation, geometry, joint_kinematics=kinematics).mean
    second = actor(observation, changed, joint_kinematics=kinematics).mean
    assert torch.equal(first, second)
    assert geometry.shortest_path.shape == (observation.jnt_current.shape[0], 21, 21)


def test_checkpoint_round_trip_and_identity_guards(tmp_path: Path) -> None:
    r"""checkpoint 参数往返逐 key 一致，并拒绝错误 schema 与数据源漂移。"""

    actor = build_family_student("n040", device="cpu")
    path = tmp_path / "student.pt"
    save_family_student_checkpoint(
        path,
        actor,
        metadata={"dataset_sha256": "dataset-a", "n040_sha256": "n040-a"},
    )
    restored, metadata = load_family_student(path, device="cpu")
    assert metadata["dataset_sha256"] == "dataset-a"
    assert all(torch.equal(actor.state_dict()[key], restored.state_dict()[key]) for key in actor.state_dict())
    with pytest.raises(ValueError, match="schema"):
        load_family_student(path, device="cpu", expected_schema="0.0.0")
    with pytest.raises(ValueError, match="dataset"):
        load_family_student(path, device="cpu", expected_dataset_sha256="dataset-b")


def test_tiny_cpu_fit_reduces_bc_loss_without_learning_sigma() -> None:
    r"""低量合成监督能降低 BC loss，同时 global_log_std 始终是冻结探索常数。"""

    observation, geometry, kinematics = _batch(batch=2)
    torch.manual_seed(23)
    actor = build_family_student("no_z", device="cpu")
    with torch.no_grad():
        target = actor(observation, geometry, joint_kinematics=kinematics).mean.detach()
    target = target + 0.1 * observation.jnt_current[..., 0]
    valid = observation.jnt_valid
    sigma_before = actor.global_log_std.detach().clone()
    before = balanced_family_action_mse(actor(observation, geometry, joint_kinematics=kinematics).mean.detach(), target, valid).item()
    result = fit_family_student(
        actor,
        batches=((observation, geometry, target, "leap", torch.tensor([0, 0]), valid, kinematics, None),),
        max_updates=8,
        batch_size=2,
        learning_rate=3.0e-3,
        seed=42,
    )
    after = balanced_family_action_mse(actor(observation, geometry, joint_kinematics=kinematics).mean.detach(), target, valid).item()
    assert result["updates"] == 8
    assert after < before
    assert torch.equal(sigma_before, actor.global_log_std.detach())


def test_config_rejects_noncanonical_routes() -> None:
    r"""学生配置锁定 direct-token/tcn/global-sigma/phase-off，不悄悄切换动作 ABI。"""

    with pytest.raises(ValueError, match="local_skip"):
        FamilyStudentConfig(local_skip=True)


def test_torchscript_export_checks_dynamic_batches_and_sidecar(tmp_path: Path) -> None:
    r"""12-tensor动作模块在 B=2/5/17 上与 Python actor 一致，且不覆盖正式输出。"""

    checkpoint = tmp_path / "student.pt"
    output = tmp_path / "student.ts"
    actor = build_family_student("no_z", device="cpu")
    save_family_student_checkpoint(checkpoint, actor, metadata={"dataset_sha256": "d", "n040_sha256": "n"})
    sidecar = export_family_student_torchscript(checkpoint, output)
    assert output.exists() and Path(f"{output}.json").exists()
    assert sidecar["validation_batch_sizes"] == [2, 5, 17]
    assert sidecar["torchscript_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        export_family_student_torchscript(checkpoint, output)
