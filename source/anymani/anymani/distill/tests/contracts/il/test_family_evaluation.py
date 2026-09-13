r"""Frozen family student 的跨 runtime TorchScript 评价合同。

测试故意只生成 tiny TorchScript、plain-dict IL checkpoint 与 mock binding；它们验证的是
sidecar/输入 ABI/身份隔离/TF32 生命周期，不把 dummy 输出冒充为 LEAP 或 Allegro 物理能力。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import anymani.distill.il.family_evaluation as family_evaluation
import pytest
import torch

CANONICAL_ACTOR_ABI = {
    "arm": "direct_token",
    "history_encoder": "tcn",
    "history_length": 30,
    "joint_count": 16,
    "owner_count": 21,
    "geometry_width": 128,
    "actor_contact": "tip-only-binary",
    "phase_clock_enabled": False,
    "joint_kinematics_width": 15,
}


class _TinyNoZ(torch.nn.Module):
    r"""只用 current 构造有界动作的 dummy；参数/buffer 供 freeze contract 检测。"""

    def __init__(self, *, no_z: bool) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("state_witness", torch.tensor([1.0]))
        self.no_z = no_z

    def forward(
        self,
        jnt_current: torch.Tensor,
        jnt_history: torch.Tensor,
        jnt_limits: torch.Tensor,
        owner_contact: torch.Tensor,
        jnt_valid: torch.Tensor,
        tip_valid: torch.Tensor,
        owner_valid: torch.Tensor,
        geometry_tokens: torch.Tensor,
        shortest_path: torch.Tensor,
        parent_direction: torch.Tensor,
        child_direction: torch.Tensor,
        joint_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        del jnt_history, jnt_limits, owner_contact, tip_valid, owner_valid
        del shortest_path, parent_direction, child_direction, joint_kinematics
        value = jnt_current[..., 0] * self.gain
        if not self.no_z:
            value = value + geometry_tokens[:, 1:17, 0]
        return value.clamp(-1.0, 1.0) * jnt_valid.to(dtype=torch.float32)


def _checkpoint_config(variant: str = "no_z") -> dict[str, object]:
    r"""构造与独立 IL checkpoint 对齐的 actor config，不依赖 family_student 模块。"""

    return {
        "variant": variant,
        "initial_log_std": -0.5,
        "max_log_std": -0.43,
        "history_encoder": "tcn",
        "history_length": 30,
        "local_skip": False,
        "sigma_mode": "global",
        "phase_clock_enabled": False,
        "joint_kinematics_width": 15,
        "joint_origin_width": 3,
        "link_length_m": 0.1,
    }


def _write_artifacts(tmp_path: Path, *, variant: str = "no_z") -> tuple[Path, Path, Path, dict[str, object]]:
    r"""写出 plain-dict checkpoint、tiny .ts 与严格 sidecar，返回其实际身份。"""

    checkpoint = tmp_path / "student.pt"
    scripted = tmp_path / "student.ts"
    sidecar = tmp_path / "student.ts.json"
    config = _checkpoint_config(variant)
    metadata = {"n040_sha256": "n040-id", "dataset_sha256": "dataset-id"}
    payload: dict[str, object] = {
        "artifact_type": "anymani.family_distilled_actor",
        "schema": "1.0.0",
        "schema_version": "1.0.0",
        "actor_state_dict": {},
        "actor_config": config,
        "variant": variant,
        "representation": variant,
        "actor_abi": dict(CANONICAL_ACTOR_ABI),
        "metadata": metadata,
        "optimizer_state_dict": {},
        "training_state": {"epoch": 3, "update": 7, "processed_samples": 19},
        "rng_state": {},
    }
    torch.save(payload, checkpoint)
    example = tuple(
        (
            torch.zeros(2, 16, 5),
            torch.zeros(2, 30, 16, 5),
            torch.zeros(2, 16, 2),
            torch.zeros(2, 21, 1),
            torch.ones(2, 16, dtype=torch.bool),
            torch.ones(2, 4, dtype=torch.bool),
            torch.ones(2, 21, dtype=torch.bool),
            torch.zeros(2, 21, 128),
            torch.zeros(2, 21, 21, dtype=torch.long),
            torch.zeros(2, 21, 21, dtype=torch.long),
            torch.zeros(2, 21, 21, dtype=torch.long),
            torch.zeros(2, 16, 15),
        )
    )
    traced = cast(torch.jit.ScriptModule, torch.jit.trace(_TinyNoZ(no_z=variant in {"no_z", "fk"}), example, strict=False))
    traced.save(str(scripted))
    sidecar_payload: dict[str, object] = {
        "artifact_type": "anymani.family_distilled_actor_torchscript",
        "schema": "1.0.0",
        "schema_version": "1.0.0",
        "torchscript_sha256": hashlib.sha256(scripted.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "n040_sha256": "n040-id",
        "dataset_sha256": "dataset-id",
        "variant": variant,
        "actor_config": config,
        "input_abi": list(family_evaluation.TORCHSCRIPT_INPUT_ABI),
        "input_shapes": [
            ["B", 16, 5],
            ["B", 30, 16, 5],
            ["B", 16, 2],
            ["B", 21, 1],
            ["B", 16],
            ["B", 4],
            ["B", 21],
            ["B", 21, 128],
            ["B", 21, 21],
            ["B", 21, 21],
            ["B", 21, 21],
            ["B", 16, 15],
        ],
        "input_dtypes": [
            "float32",
            "float32",
            "float32",
            "float32",
            "bool",
            "bool",
            "bool",
            "float32",
            "int64",
            "int64",
            "int64",
            "float32",
        ],
        "output_shape": ["B", 16],
        "precision": {"dtype": "float32", "amp": False, "tf32": False},
        "loaded_file_parity": True,
        "parity_max_abs": 0.0,
        "parity_tolerance": 1.0e-5,
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2) + "\n", encoding="utf-8")
    return checkpoint, scripted, sidecar, sidecar_payload


def _observation(batch: int = 2) -> dict[str, torch.Tensor]:
    r"""构造无 object/asset 输入的 canonical actor observation。"""

    return {
        "actor_jnt_current": torch.zeros(batch, 16, 5),
        "actor_jnt_history": torch.zeros(batch, 30, 16, 5),
        "actor_jnt_limits": torch.stack((-torch.ones(batch, 16), torch.ones(batch, 16)), dim=-1),
        "actor_owner_contact": torch.zeros(batch, 21, 1),
        "jnt_valid": torch.ones(batch, 16, dtype=torch.bool),
        "tip_valid": torch.ones(batch, 4, dtype=torch.bool),
        "owner_valid": torch.ones(batch, 21, dtype=torch.bool),
        "geometry_tokens": torch.zeros(batch, 21, 128),
        "shortest_path": torch.zeros(batch, 21, 21, dtype=torch.int16),
        "parent_direction": torch.zeros(batch, 21, 21, dtype=torch.int16),
        "child_direction": torch.zeros(batch, 21, 21, dtype=torch.int16),
        "object_state": torch.full((batch, 3), 99.0),  # 不得进入 TorchScript 12-input tuple。
    }


def _binding(asset_count: int = 2) -> SimpleNamespace:
    r"""构造 source-assets/canonical-routing mock；真实 FK 数值由测试 monkeypatch 提供。"""

    joint_names = family_evaluation.CANONICAL_HAND_SCHEMA_V1.joint_names
    source_assets = tuple(
        SimpleNamespace(asset_id=f"asset-{index}", geometry_semantics=object()) for index in range(asset_count)
    )
    canonical_artifacts = tuple(
        SimpleNamespace(
            routing=SimpleNamespace(
                source_to_canonical=tuple((f"source-{slot}", joint_names[slot]) for slot in range(16))
            )
        )
        for _ in range(asset_count)
    )
    return SimpleNamespace(
        source_assets=source_assets,
        canonical_artifacts=canonical_artifacts,
        source_member_keys=tuple(f"member-{index}" for index in range(asset_count)),
    )


def _identity(n040: str = "n040-id") -> dict[str, object]:
    r"""构造 teacher/runtime 的最小 direct-token、TIP-only、phase-free identity。"""

    return {
        "identity_schema_version": "4.0.0",
        "identity_digest": "identity",
        "policy": {
            "arm": "direct_token",
            "actor_contact": "tip-only-binary",
            "action_authority_rad_per_policy_step": 1.0 / 24.0,
        },
        "training": {"history_encoder": "tcn", "phase_period_steps": None, "sigma_mode": "global"},
        "geometry_provider": {"retained_artifact": {"sha256": n040}},
        "precision": {"actor_dtype": "float32", "tf32": False},
    }


def _start(
    evaluator: family_evaluation.FrozenFamilyStudent,
    checkpoint_path: Path,
    *,
    observation: dict[str, torch.Tensor] | None = None,
    runtime_identity: dict[str, object] | None = None,
) -> None:
    r"""以 CPU mock binding 启动 student evaluator。"""

    teacher = checkpoint_path.parent / "teacher.pth"
    teacher.write_bytes(b"teacher-reference")
    cohort = checkpoint_path.parent / "cohort.lock"
    cohort.write_bytes(b"cohort-reference")
    evaluator.start(
        checkpoint_path=teacher,
        checkpoint_identity=_identity(),
        runtime_identity=runtime_identity or _identity(),
        binding=_binding(),
        observation=observation or _observation(),
        cohort_path=cohort,
        cohort_members=({}, {}),
        steps=600,
        replicas=1,
    )


def test_sidecar_and_checkpoint_identity_mismatch_fail_closed(tmp_path: Path) -> None:
    r"""artifact/schema/SHA/variant/12ABI/precision/parity 任一漂移都不能加载。"""

    cases: list[tuple[str, object]] = [
        ("artifact_type", "wrong"),
        ("schema_version", "0.0.0"),
        ("torchscript_sha256", "wrong"),
        ("checkpoint_sha256", "wrong"),
        ("n040_sha256", "wrong"),
        ("variant", "n040"),
        ("input_abi", list(family_evaluation.TORCHSCRIPT_INPUT_ABI[:-1])),
        ("loaded_file_parity", False),
    ]
    for key, value in cases:
        case_dir = tmp_path / f"case-{key}"
        case_dir.mkdir()
        checkpoint, scripted, sidecar, _ = _write_artifacts(case_dir)
        altered = json.loads(sidecar.read_text(encoding="utf-8"))
        altered[key] = value
        sidecar.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises((ValueError, RuntimeError), match="sidecar|TorchScript|checkpoint|N040|ABI|parity|schema"):
            family_evaluation.FrozenFamilyStudent(checkpoint, scripted, sidecar)


def test_numeric_loaded_file_parity_evidence_is_accepted(tmp_path: Path) -> None:
    r"""exporter 用 max/tolerance 保存实际 loaded-file parity 时仍按同一通过条件读取。"""

    checkpoint, scripted, sidecar, _ = _write_artifacts(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.pop("loaded_file_parity")
    payload.update({"parity_max_abs": 1.0e-6, "parity_tolerance": 1.0e-5, "ghost_max_abs": 0.0, "range_excess_max_abs": 0.0})
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    family_evaluation.FrozenFamilyStudent(checkpoint, scripted, sidecar)


def test_start_rejects_runtime_n040_and_teacher_route_mismatch(tmp_path: Path) -> None:
    r"""IL N040 必须与当前 runtime 相同，teacher reference 必须保持 1/24、TIP-only、History30、无 phase。"""

    checkpoint, scripted, sidecar, _ = _write_artifacts(tmp_path)
    for field, value in (("n040", "different"), ("phase", 43), ("contact", "all-owner-binary-no-force")):
        runtime = _identity()
        if field == "n040":
            runtime["geometry_provider"] = {"retained_artifact": {"sha256": value}}
        elif field == "phase":
            runtime["training"] = {"history_encoder": "tcn", "phase_period_steps": value, "sigma_mode": "global"}
        else:
            runtime["policy"] = {"arm": "direct_token", "actor_contact": value, "action_authority_rad_per_policy_step": 1.0 / 24.0}
        evaluator = family_evaluation.FrozenFamilyStudent(checkpoint, scripted, sidecar)
        with pytest.raises(ValueError, match="N040|phase|TIP|contact|identity"):
            _start(evaluator, checkpoint, runtime_identity=runtime)


def test_no_z_act_routes_exactly_twelve_inputs_and_ignores_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""No-Z loaded TS 对 geometry token 改动不敏感，且 dummy 只能收到 12 个 actor/kinematics 输入。"""

    checkpoint, scripted, sidecar, _ = _write_artifacts(tmp_path)
    monkeypatch.setattr(
        family_evaluation,
        "build_joint_kinematics_bank",
        lambda semantics, mappings, *, joint_count, dtype: SimpleNamespace(
            features=torch.zeros(len(semantics), joint_count, 15, dtype=dtype)
        ),
    )
    evaluator = family_evaluation.FrozenFamilyStudent(checkpoint, scripted, sidecar)
    observation = _observation()
    _start(evaluator, checkpoint, observation=observation)
    first = evaluator.act(0, observation)
    changed = dict(observation)
    changed["geometry_tokens"] = torch.full((2, 21, 128), 7.0)
    second = evaluator.act(1, changed)
    assert first.dtype == torch.float32 and first.shape == (2, 16)
    assert torch.equal(first, second)
    assert "object_state" in changed and evaluator.metadata["input_abi"] == list(family_evaluation.TORCHSCRIPT_INPUT_ABI)


def test_act_rejects_dtype_shape_and_ghost_violations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""输入必须是声明的 dtype/shape，invalid joint 的 12-input ghost 值必须精确为零。"""

    checkpoint, scripted, sidecar, _ = _write_artifacts(tmp_path)
    monkeypatch.setattr(
        family_evaluation,
        "build_joint_kinematics_bank",
        lambda semantics, mappings, *, joint_count, dtype: SimpleNamespace(
            features=torch.zeros(len(semantics), joint_count, 15, dtype=dtype)
        ),
    )
    evaluator = family_evaluation.FrozenFamilyStudent(checkpoint, scripted, sidecar)
    observation = _observation()
    _start(evaluator, checkpoint, observation=observation)
    wrong_dtype = dict(observation)
    wrong_dtype["actor_jnt_current"] = observation["actor_jnt_current"].double()
    with pytest.raises(ValueError, match="float32|dtype"):
        evaluator.act(0, wrong_dtype)
    wrong_shape = dict(observation)
    wrong_shape["geometry_tokens"] = torch.zeros(2, 20, 128)
    with pytest.raises(ValueError, match="shape"):
        evaluator.act(0, wrong_shape)
    ghost = dict(observation)
    ghost["jnt_valid"] = torch.ones(2, 16, dtype=torch.bool)
    ghost["jnt_valid"][0, -1] = False
    ghost["owner_valid"] = torch.cat(
        (torch.ones(2, 1, dtype=torch.bool), ghost["jnt_valid"], torch.ones(2, 4, dtype=torch.bool)), dim=-1
    )
    ghost["actor_jnt_current"][0, -1, 0] = 0.1
    with pytest.raises(ValueError, match="ghost"):
        evaluator.act(0, ghost)


def test_nonzero_valid_history_and_zero_ghost_history_are_distinguished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实活动关节的历史可以非零，只有 ghost 槽必须为零；避免掩码反转。"""
    checkpoint, scripted, sidecar, _ = _write_artifacts(tmp_path)
    monkeypatch.setattr(
        family_evaluation, "build_joint_kinematics_bank",
        lambda semantics, mappings, *, joint_count, dtype: SimpleNamespace(
            features=torch.zeros(len(semantics), joint_count, 15, dtype=dtype)
        ),
    )
    evaluator = family_evaluation.FrozenFamilyStudent(checkpoint, scripted, sidecar)
    observation = _observation()
    observation["jnt_valid"][:, -1] = False
    observation["owner_valid"][:, 16] = False
    observation["actor_jnt_limits"][:, -1] = 0.0
    observation["actor_jnt_current"][:, :-1, 0] = 0.25
    observation["actor_jnt_history"][:, :, :-1, 0] = 0.25
    _start(evaluator, checkpoint, observation=observation)
    evaluator.act(0, observation)
    evaluator.after_step()
    observation["actor_jnt_history"][0, 0, -1, 0] = 0.1
    with pytest.raises(ValueError, match="actor_jnt_history ghost"):
        evaluator.act(1, observation)


def test_tf32_flags_restore_and_finish_detects_ts_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""act 只在 TorchScript 调用期间关闭 TF32；TS buffer 变化在 finish 被逐值发现。"""

    checkpoint, scripted, sidecar, _ = _write_artifacts(tmp_path)
    monkeypatch.setattr(
        family_evaluation,
        "build_joint_kinematics_bank",
        lambda semantics, mappings, *, joint_count, dtype: SimpleNamespace(
            features=torch.zeros(len(semantics), joint_count, 15, dtype=dtype)
        ),
    )
    evaluator = family_evaluation.FrozenFamilyStudent(checkpoint, scripted, sidecar)
    observation = _observation()
    _start(evaluator, checkpoint, observation=observation)
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    evaluator.act(0, observation)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    torch.backends.cuda.matmul.allow_tf32 = previous_matmul
    torch.backends.cudnn.allow_tf32 = previous_cudnn
    evaluator.after_step()
    evaluator._script.state_witness.fill_(2.0)
    with pytest.raises(RuntimeError, match="buffer|parameter|changed"):
        evaluator.finish()


def test_identity_updates_separate_student_and_teacher_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""identity_updates 提供 JSON-safe student/method、reference teacher、runtime 与 helper 身份。"""

    checkpoint, scripted, sidecar, sidecar_payload = _write_artifacts(tmp_path)
    monkeypatch.setattr(
        family_evaluation,
        "build_joint_kinematics_bank",
        lambda semantics, mappings, *, joint_count, dtype: SimpleNamespace(
            features=torch.zeros(len(semantics), joint_count, 15, dtype=dtype)
        ),
    )
    evaluator = family_evaluation.FrozenFamilyStudent(checkpoint, scripted, sidecar)
    observation = _observation()
    _start(evaluator, checkpoint, observation=observation)
    evaluator.act(0, observation)
    evaluator.after_step()
    updates = evaluator.identity_updates()
    json.dumps(updates, ensure_ascii=False, sort_keys=True)
    assert updates["student_checkpoint_sha256"] == sidecar_payload["checkpoint_sha256"]
    assert updates["torchscript_sha256"] == sidecar_payload["torchscript_sha256"]
    assert len(updates["method_identity_digest"]) == 64
    assert updates["n040_sha256"] == "n040-id"
    assert updates["variant"] == "no_z"
    assert updates["input_abi"] == list(family_evaluation.TORCHSCRIPT_INPUT_ABI)
    assert updates["training_state"] == {"epoch": 3, "update": 7, "processed_samples": 19}
    assert updates["executed_steps"] == 1
    assert updates["reference_teacher"]["checkpoint_path"].endswith("teacher.pth")
    assert updates["reference_teacher"]["checkpoint_sha256"] != updates["student_checkpoint_sha256"]
    assert "cohort_members" not in updates["reference_teacher"]
    assert updates["evaluated_cohort"]["role"] == "student_evaluation_population"
    assert len(updates["evaluated_cohort"]["members"]) == 2
    assert updates["runtime"]["n040_sha256"] == "n040-id"
    assert len(updates["helper_source_sha256"]) == 64
    assert updates["document_updates"]["student_checkpoint_path"].endswith("student.pt")
