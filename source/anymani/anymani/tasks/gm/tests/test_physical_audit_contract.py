r"""Canonical/native physical audit 的纯张量合同测试。

这些测试不启动 Isaac Sim。它们锁定四元数双覆盖、接触时间/冲量单位、matched trace shape
与 $(E_m,S_m)$ 分离语义，避免 runtime probe 形成数据后才发现分析公式不可复算。
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "physical_audit.py"


def _module():
    r"""按文件加载纯张量模块，避免触发 ``tasks.gm`` IsaacLab Gym registry。"""

    spec = importlib.util.spec_from_file_location("gm_physical_audit_contract", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(*, joint_offset: float = 0.0, contact_scale: float = 1.0) -> dict[str, torch.Tensor]:
    r"""构造三步、两关节、两 fingertip 的最小 matched trace。"""

    joint_pos = torch.tensor([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]]) + joint_offset  # ``[T=3,J=2]`` rad
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # $(w,x,y,z)$ 单位旋转
    return {
        "joint_pos_rad": joint_pos,
        "joint_vel_rad_s": joint_pos * 2.0,
        "joint_target_rad": torch.tensor([[0.0, 0.1], [0.15, 0.25], [0.25, 0.35]]),
        "tip_pos_m": torch.zeros(3, 2, 3),
        "tip_quat_wxyz": identity_quat.repeat(3, 2, 1),
        "object_pos_m": torch.zeros(3, 3),
        "object_quat_wxyz": identity_quat.repeat(3, 1),
        "object_lin_vel_m_s": torch.zeros(3, 3),
        "object_ang_vel_rad_s": torch.zeros(3, 3),
        "contact_force_N": torch.tensor([[0.0, 0.0], [0.3, 0.0], [0.5, 0.2]]) * contact_scale,
        "reward_terms": torch.zeros(3, 4),
    }


def test_quaternion_geodesic_respects_double_cover_and_principal_angle() -> None:
    r"""$q$ 与 $-q$ 距离必须为零，绕 z 轴 $90^\circ$ 的距离必须为 $\pi/2$。"""

    module = _module()
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    candidate = torch.tensor(
        [[-1.0, 0.0, 0.0, 0.0], [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]]
    )

    distance = module.quaternion_geodesic_angle_wxyz(identity, candidate)

    torch.testing.assert_close(distance, torch.tensor([0.0, math.pi / 2.0]), atol=1.0e-6, rtol=0.0)


def test_semantic_axis_alignment_does_not_compare_incidental_body_order() -> None:
    r"""Source/canonical finger slot 顺序不同时，candidate 必须按语义 label gather。"""

    module = _module()
    indices = module.candidate_indices_in_reference_semantic_order(
        ("index", "middle", "ring", "thumb"),
        ("thumb", "index", "middle", "ring"),
    )

    assert indices == (1, 2, 3, 0)


def test_semantic_axis_alignment_rejects_missing_or_duplicate_entities() -> None:
    r"""Matched probe 不得通过截断或重复 row 掩盖真实 fingertip 集合差异。"""

    module = _module()
    with pytest.raises(ValueError, match="differ"):
        module.candidate_indices_in_reference_semantic_order(("index", "thumb"), ("index", "ring"))
    with pytest.raises(ValueError, match="unique"):
        module.candidate_indices_in_reference_semantic_order(("index", "index"), ("index", "thumb"))


def test_contact_onset_and_impulse_keep_step_and_physical_units() -> None:
    r"""首次越阈值发生在 step 1，左矩形积分按 channel 返回 N·s。"""

    module = _module()
    forces = torch.tensor([[0.0, 0.0], [0.3, 0.0], [0.5, 0.2]])  # ``[T=3,C=2]`` N

    assert module.first_contact_step(forces, threshold_N=0.25) == 1
    torch.testing.assert_close(module.integrated_contact_impulse_Ns(forces, 0.01), torch.tensor([0.008, 0.002]))


def test_matched_summary_separates_control_input_from_dynamics_error() -> None:
    r"""相同 target 下，joint state offset 形成非零动力学误差但 target 误差严格为零。"""

    module = _module()
    summary = module.summarize_physical_trace_error(
        _trace(),
        _trace(joint_offset=0.01),
        sample_dt_s=0.01,
        contact_threshold_N=0.25,
    )

    assert summary["sample_count"] == 3
    assert summary["joint_pos_abs_max_rad"] == pytest.approx(0.01)
    assert summary["joint_vel_abs_max_rad_s"] == pytest.approx(0.02)
    assert summary["joint_target_abs_max_rad"] == 0.0
    assert summary["contact_onset_abs_delta_steps"] == 0


def test_canonical_error_and_native_variation_remain_distinct() -> None:
    r"""$S_m=0$ 时不以 epsilon 伪造有限 $E_m/S_m$，原始 $E_m$ 仍必须保存。"""

    module = _module()
    result = module.compare_canonical_against_native_repeat(
        _trace(),
        _trace(),
        _trace(joint_offset=0.01),
        sample_dt_s=0.01,
        contact_threshold_N=0.25,
    )

    canonical = result["canonical_error_E"]
    variation = result["native_repeat_variation_S"]
    ratios = result["error_to_native_variation_ratio"]
    assert canonical["joint_pos_abs_max_rad"] == pytest.approx(0.01)
    assert variation["joint_pos_abs_max_rad"] == 0.0
    assert ratios["joint_pos_abs_max_rad"] is None


def test_trace_validation_rejects_misaligned_time_axis() -> None:
    r"""任一字段遗漏时间样本时必须 fail closed，禁止 PyTorch 广播制造伪 matched 数据。"""

    module = _module()
    malformed = _trace()
    malformed["object_pos_m"] = malformed["object_pos_m"][:2]

    with pytest.raises(ValueError, match="share time axis"):
        module.validate_physical_audit_trace(malformed)
