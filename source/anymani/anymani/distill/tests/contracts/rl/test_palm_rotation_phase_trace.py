r"""Dense trace phase-clock contracts for artifact-only diagnostics.

测试只生成临时 JSON/HDF5 事实文件；不导入 model、runtime 或 teacher，也不启动 Isaac/Kit/GPU。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5py
import numpy as np
import pytest
import torch
from anymani.distill.diagnostics.analysis.rl.palm_rotation_trace import audit_palm_rotation_trace
from anymani.distill.tests.contracts.rl.test_family_teacher_acceptance import (
    _report,
    _write_cohort,
    _write_evaluation,
)


def _phase_contract(period: int = 43) -> dict[str, Any]:
    r"""返回当前 runtime 发布的任意整数周期 phase contract 副本。"""

    return {
        "source": "physical-episode-length-buf",
        "period_policy_steps": period,
        "increment_rad_per_policy_step": 2.0 * math.pi / period,
        "encoding": ["sin", "cos"],
        "encoding_dtype": "float32",
        "reset": "physical-episode-counter-zero-before-returned-observation",
        "transport_key": "phase_clock",
        "actor_adapter": "zero-linear2to128-after-contextual-joints-before-existing-head-norm",
        "critic_adapter": "zero-linear2to896-after-readout-before-existing-value-norm",
    }


def _sha256(path: Path) -> str:
    r"""计算测试 artifact 的实际字节摘要，保持 evaluation 引用闭合。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase_rows(step_count: int, period: int = 43) -> np.ndarray:
    r"""按 producer 的整数取模与 FP32 三角函数重建 `[T,2]` sin/cos。"""

    phase_index = np.arange(step_count, dtype=np.float32) % np.float32(period)
    angle = phase_index * np.float32(2.0 * math.pi / period)
    return np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)


def _write_phase_case(
    root: Path,
    *,
    step_count: int = 30,
    terminate_first_at: int = 5,
    phase_period: int = 43,
    include_dataset: bool = True,
    metadata_mode: str = "full",
    phase_values: np.ndarray | None = None,
) -> Path:
    r"""创建两副本首轨迹，第一副本中途 reset，第二副本在第30步超时。"""

    assert 1 <= terminate_first_at < step_count
    checkpoint = root / "checkpoint.pth"
    checkpoint.write_bytes(b"phase-trace-checkpoint")
    identity: dict[str, Any] = {
        "schema_version": "1.0.0",
        "checkpoint_sha256": _sha256(checkpoint),
        "method_identity_digest": "a" * 64,
        "protocol": {
            "num_assets": 1,
            "replicas_per_asset": 2,
            "policy_steps": step_count,
            "policy_dt_s": 0.05,
            "horizon_s": step_count * 0.05,
            "trace_stride": 1,
            "trace_rewards": True,
            "deterministic_actor_mean": True,
            "first_trajectory_only": True,
        },
    }
    identity["identity_digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    active = np.ones((step_count, 1, 2), dtype=bool)
    active[terminate_first_at:, 0, 0] = False
    drop = np.zeros_like(active)
    drop[terminate_first_at - 1, 0, 0] = True
    timeout = np.zeros_like(active)
    timeout[-1, 0, 1] = True
    post_state_valid = active & ~(drop | timeout)
    duration = np.zeros_like(active, dtype=np.float32)
    for index in range(step_count):
        duration[index, 0, 0] = min(index + 1, terminate_first_at) * 0.05
        duration[index, 0, 1] = (index + 1) * 0.05
    phase = _phase_rows(step_count, phase_period)[:, None, None, :].repeat(2, axis=2)
    phase[terminate_first_at:, 0, 0] = np.asarray((0.25, 0.75), dtype=np.float32)
    if phase_values is not None:
        phase = np.asarray(phase_values, dtype=np.float32)
    trace_arrays: dict[str, np.ndarray] = {
        "policy_step": np.arange(1, step_count + 1, dtype=np.int64),
        "active": active,
        "post_state_valid": post_state_valid,
        "episode_duration_s": duration,
        "net_rotation_rad": np.zeros_like(active, dtype=np.float32),
        "absolute_path_rotation_rad": np.zeros_like(active, dtype=np.float32),
        "goal_success_pulse": np.zeros_like(active),
        "termination_object_out_of_anchor": drop,
        "termination_goal_axis_misaligned": np.zeros_like(active),
        "termination_time_out": timeout,
        "reward_terms_step": np.ones((step_count, 1, 2, 1), dtype=np.float32),
        "reward_step": np.ones_like(active, dtype=np.float32),
    }
    if include_dataset:
        trace_arrays["pre_phase_clock"] = phase
    trajectory_arrays: dict[str, np.ndarray] = {
        "signed_net_turns": np.zeros((1, 2), dtype=np.float32),
        "absolute_path_turns": np.zeros((1, 2), dtype=np.float32),
        "duration_s": np.asarray([[terminate_first_at * 0.05, step_count * 0.05]], dtype=np.float32),
        "diagnostic_step_count": np.asarray([[terminate_first_at, step_count]], dtype=np.float32),
        "goal_count": np.zeros((1, 2), dtype=np.float32),
        "termination_drop": np.asarray([[True, False]], dtype=bool),
        "termination_axis": np.zeros((1, 2), dtype=bool),
        "termination_timeout": np.asarray([[False, True]], dtype=bool),
    }
    trace_metadata: dict[str, Any] = {
        **identity,
        "axes": "time,asset,replica,feature",
        "reward_term_names": ["reward"],
    }
    if metadata_mode == "full":
        trace_metadata["phase_clock"] = _phase_contract(phase_period)
        trace_metadata["pre_phase_clock_semantics"] = "sin/cos from physical episode counter before applied action"
    elif metadata_mode == "clock_only":
        trace_metadata["phase_clock"] = _phase_contract(phase_period)
    elif metadata_mode == "semantics_only":
        trace_metadata["pre_phase_clock_semantics"] = "sin/cos from physical episode counter before applied action"
    elif metadata_mode != "none":
        raise ValueError(f"unknown test metadata mode: {metadata_mode}")
    trace_path = root / "evaluation.trace.h5"
    trajectory_path = root / "evaluation.h5"
    with h5py.File(trace_path, "w") as stream:
        stream.attrs["schema_version"] = "1.0.0"
        stream.attrs["metadata_json"] = json.dumps(trace_metadata)
        for name, values in trace_arrays.items():
            stream.create_dataset(name, data=values)
    with h5py.File(trajectory_path, "w") as stream:
        stream.attrs["schema_version"] = "1.0.0"
        stream.attrs["metadata_json"] = json.dumps(identity)
        for name, values in trajectory_arrays.items():
            stream.create_dataset(name, data=values)
    document = {
        "evaluation_identity": identity,
        "checkpoint": str(checkpoint),
        "trajectory_hdf5": str(trajectory_path),
        "trajectory_hdf5_sha256": _sha256(trajectory_path),
        "step_trace": {"path": str(trace_path), "samples": step_count, "sha256": _sha256(trace_path)},
    }
    evaluation = root / "evaluation.json"
    evaluation.write_text(json.dumps(document), encoding="utf-8")
    return evaluation


def _republish(evaluation: Path) -> None:
    r"""更新临时 HDF5 修改后的外层 SHA，令测试触发语义审计。"""

    document = json.loads(evaluation.read_text(encoding="utf-8"))
    trace_path = Path(document["step_trace"]["path"])
    trajectory_path = Path(document["trajectory_hdf5"])
    document["step_trace"]["sha256"] = _sha256(trace_path)
    document["trajectory_hdf5_sha256"] = _sha256(trajectory_path)
    evaluation.write_text(json.dumps(document), encoding="utf-8")


def test_trace_phase_clock_contract_is_audited_and_reported(tmp_path: Path) -> None:
    r"""正确的前30步 phase、inactive reset 行和 FP32 误差应形成显式证据。"""

    report = audit_palm_rotation_trace(_write_phase_case(tmp_path))
    assert report["status"] == "passed"
    assert report["phase_clock"] == _phase_contract()
    assert report["pre_phase_clock_semantics"] == "sin/cos from physical episode counter before applied action"
    assert report["phase_clock_max_abs_error"] <= 1.0e-6
    assert report["active_samples"] == 35  # 第一个副本5步，第二个副本30步。


@pytest.mark.parametrize("defect", ["offset", "swap", "period"])
def test_trace_phase_clock_rejects_alignment_order_or_period_defects(tmp_path: Path, defect: str) -> None:
    r"""错位一步、交换 sin/cos 或篡改周期均不能通过 phase auditor。"""

    evaluation = _write_phase_case(tmp_path)
    document = json.loads(evaluation.read_text(encoding="utf-8"))
    trace_path = Path(document["step_trace"]["path"])
    with h5py.File(trace_path, "r+") as stream:
        phase_dataset = cast(h5py.Dataset, stream["pre_phase_clock"])
        if defect == "offset":
            values = phase_dataset[:]
            values[0, :, 1] = values[1, :, 1]  # 第1步使用第2步的 phase，active 行必须暴露错位。
            phase_dataset[:] = values
        elif defect == "swap":
            values = phase_dataset[:]
            phase_dataset[:] = values[:, :, :, ::-1]
        else:
            metadata = json.loads(str(stream.attrs["metadata_json"]))
            metadata["phase_clock"]["period_policy_steps"] = 44
            metadata["phase_clock"]["increment_rad_per_policy_step"] = 2.0 * math.pi / 44
            stream.attrs["metadata_json"] = json.dumps(metadata)
    _republish(evaluation)
    with pytest.raises(ValueError, match="phase|clock|contract"):
        audit_palm_rotation_trace(evaluation)


@pytest.mark.parametrize("metadata_mode,include_dataset", [("full", False), ("none", True), ("clock_only", True), ("semantics_only", True)])
def test_trace_phase_clock_requires_metadata_and_dataset_pair(
    tmp_path: Path, metadata_mode: str, include_dataset: bool
) -> None:
    r"""phase metadata 与 pre_phase_clock dataset 任一缺失都必须拒绝。"""

    evaluation = _write_phase_case(tmp_path, metadata_mode=metadata_mode, include_dataset=include_dataset)
    with pytest.raises(ValueError, match="phase|clock|metadata|dataset"):
        audit_palm_rotation_trace(evaluation)


@pytest.mark.parametrize("defect", ["dtype", "shape", "nonfinite"])
def test_trace_phase_clock_checks_float32_shape_and_finite_values(tmp_path: Path, defect: str) -> None:
    r"""phase dataset 必须是 `[T,A,R,2]` float32 且有效样本有限。"""

    evaluation = _write_phase_case(tmp_path)
    document = json.loads(evaluation.read_text(encoding="utf-8"))
    trace_path = Path(document["step_trace"]["path"])
    with h5py.File(trace_path, "r+") as stream:
        phase_dataset = cast(h5py.Dataset, stream["pre_phase_clock"])
        values = phase_dataset[:]
        del stream["pre_phase_clock"]
        if defect == "dtype":
            stream.create_dataset("pre_phase_clock", data=values.astype(np.float64))
        elif defect == "shape":
            stream.create_dataset("pre_phase_clock", data=values[..., :1])
        else:
            values[0, 0, 1, 0] = np.nan
            stream.create_dataset("pre_phase_clock", data=values)
    _republish(evaluation)
    with pytest.raises(ValueError, match="phase|clock|float32|shape|finite"):
        audit_palm_rotation_trace(evaluation)


@pytest.mark.parametrize("period", [2, 44])
def test_trace_phase_clock_accepts_complete_consistent_non43_contract(tmp_path: Path, period: int) -> None:
    r"""任意合法 P>=2（含周期44）的 metadata 与数据一致时都必须通过。"""

    report = audit_palm_rotation_trace(_write_phase_case(tmp_path, phase_period=period))
    assert report["status"] == "passed"
    assert report["phase_clock"] == _phase_contract(period)
    assert report["phase_clock_max_abs_error"] <= 1.0e-6


def test_old_trace_without_phase_metadata_or_dataset_keeps_legacy_behavior(tmp_path: Path) -> None:
    r"""旧 trace 不出现 phase 时仍按原完整性合同通过，报告明确标为无 phase。"""

    report = audit_palm_rotation_trace(_write_phase_case(tmp_path, metadata_mode="none", include_dataset=False))
    assert report["status"] == "passed"
    assert report["phase_clock"] is None
    assert report["phase_clock_max_abs_error"] is None


def _load_acceptance_module() -> Any:
    r"""raw-load family acceptance，避免测试经包初始化引入 runtime/model。"""

    source = Path(__file__).resolve().parents[3] / "diagnostics/analysis/rl/family_teacher_acceptance.py"
    spec = importlib.util.spec_from_file_location("family_teacher_acceptance_phase_contract", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_identity(cohort_sha: str, *, phase: dict[str, Any] | None) -> dict[str, Any]:
    r"""构造 CPU-only checkpoint metadata，policy.phase_clock 为可选字段。"""

    payload: dict[str, Any] = {
        "identity_schema_version": "4.0.0",
        "manifest": {"sha256": cohort_sha, "support_asset_count": 128, "selected_rows": list(range(128))},
        "policy": {
            "actor_contact": "tip-only-binary",
            "action_authority_rad_per_policy_step": 1.0 / 24.0,
        },
        "geometry_provider": {
            "retained_artifact": {
                "schema_version": "5.0.0",
                "artifact_type": "retained_geometry_encoder",
                "sha256": "d" * 64,
            }
        },
    }
    if phase is not None:
        payload["policy"]["phase_clock"] = phase
    payload["identity_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return payload


def test_acceptance_rejects_phase_checkpoint_with_legacy_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""CPU checkpoint声明 phase 时，任何一次评价缺少 phase trace 都不能进入重复门。"""

    acceptance = _load_acceptance_module()
    cohort = tmp_path / "cohort.lock"
    cohort.write_text(json.dumps({"schema_version": "1.2.0"}))
    cohort_sha = _sha256(cohort)
    phase = _phase_contract()
    checkpoint = tmp_path / "phase.pth"
    identity = _checkpoint_identity(cohort_sha, phase=phase)
    torch.save({"anymani_identity": identity, "model": {}}, checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    method_sha = identity["identity_digest"]
    first_trace = tmp_path / "first.trace.h5"
    second_trace = tmp_path / "second.trace.h5"
    first_trajectory = tmp_path / "first.h5"
    second_trajectory = tmp_path / "second.h5"
    for path in (first_trace, second_trace, first_trajectory, second_trajectory):
        path.write_bytes(path.name.encode())

    def evaluation(path: Path, trace: Path, trajectory: Path) -> Path:
        document = {
            "artifact_type": "test",
            "schema_version": "1.0.0",
            "evaluation_identity": {
                "manifest_sha256": cohort_sha,
                "method_identity_digest": method_sha,
                "checkpoint_sha256": checkpoint_sha,
                "protocol": {
                    "trace_stride": 1,
                    "trace_rewards": True,
                },
            },
            "checkpoint": str(checkpoint),
            "trajectory_hdf5": str(trajectory),
            "step_trace": {"path": str(trace), "samples": 1, "sha256": "e" * 64},
        }
        path.write_text(json.dumps(document))
        return path

    first = evaluation(tmp_path / "first.json", first_trace, first_trajectory)
    second = evaluation(tmp_path / "second.json", second_trace, second_trajectory)

    def fake_family_teacher(_name: str, _source: Path) -> Any:
        if _source.name == "family_teacher.py":
            def evaluate_teacher(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return {"artifact_type": "fake", "schema_version": "1.0.0"}

            return SimpleNamespace(evaluate_teacher=evaluate_teacher)

        def audit(_path: Path, *, action_mode: str) -> dict[str, Any]:
            assert action_mode == "mean"
            return {"status": "passed", "phase_clock": None, "phase_clock_max_abs_error": None}

        return SimpleNamespace(audit_palm_rotation_trace=audit)

    monkeypatch.setattr(acceptance, "_load_raw_module", fake_family_teacher)
    with pytest.raises(acceptance.AcceptanceError, match="phase|checkpoint|trace"):
        acceptance.evaluate_repeat_acceptance(
            [first, second],
            cohort,
            expected_method_identity_digest=method_sha,
            expected_checkpoint_sha256=checkpoint_sha,
        )


def test_acceptance_preserves_and_matches_phase_checkpoint_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""两次评价的 trace phase 证书与 CPU checkpoint policy.phase_clock 相同才可通过。"""

    acceptance = _load_acceptance_module()
    cohort, cohort_sha = _write_cohort(tmp_path)
    phase = _phase_contract()
    checkpoint = tmp_path / "phase.pth"
    checkpoint_identity = _checkpoint_identity(cohort_sha, phase=phase)
    torch.save({"anymani_identity": checkpoint_identity, "model": {}}, checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    method_sha = checkpoint_identity["identity_digest"]
    first_trajectory = tmp_path / "first.h5"
    second_trajectory = tmp_path / "second.h5"
    first_trace = tmp_path / "first.trace.h5"
    second_trace = tmp_path / "second.trace.h5"
    for path in (first_trajectory, second_trajectory, first_trace, second_trace):
        path.write_bytes(path.name.encode())
    first = _write_evaluation(
        tmp_path,
        "first.json",
        cohort_sha,
        method=method_sha,
        checkpoint_sha=checkpoint_sha,
        checkpoint_path=checkpoint,
        trajectory=first_trajectory,
        trace=first_trace,
    )
    second = _write_evaluation(
        tmp_path,
        "second.json",
        cohort_sha,
        method=method_sha,
        checkpoint_sha=checkpoint_sha,
        checkpoint_path=checkpoint,
        trajectory=second_trajectory,
        trace=second_trace,
    )

    def fake_loader(_name: str, source: Path) -> Any:
        if source.name == "family_teacher.py":
            def evaluate_teacher(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return _report(set(range(86)))

            return SimpleNamespace(evaluate_teacher=evaluate_teacher)

        def audit(_path: Path, *, action_mode: str) -> dict[str, Any]:
            assert action_mode == "mean"
            return {"status": "passed", "phase_clock": phase, "phase_clock_max_abs_error": 0.0}

        return SimpleNamespace(audit_palm_rotation_trace=audit)

    monkeypatch.setattr(acceptance, "_load_raw_module", fake_loader)
    result = acceptance.evaluate_repeat_acceptance(
        [first, second],
        cohort,
        expected_method_identity_digest=method_sha,
        expected_checkpoint_sha256=checkpoint_sha,
    )
    assert result["identity_evidence"]["phase_clock"] == phase
    assert result["inputs"]["checkpoint_metadata"][0]["policy"]["phase_clock"] == phase
    assert result["single_results"][0]["trace_audit"]["phase_clock"] == phase
