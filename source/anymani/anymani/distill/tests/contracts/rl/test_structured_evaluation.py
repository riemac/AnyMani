r"""Pre-reset snapshot trajectory accumulator与distribution统计合同。"""

from __future__ import annotations

import math

import torch
from anymani.distill.rl.structured_evaluation import FixedDurationTrajectoryAccumulator


def _snapshot(
    *,
    step: int,
    net_rotation: tuple[float, float],
    speed: tuple[float, float],
    tip: tuple[float, float],
    palm: tuple[float, float],
    non_tip: tuple[float, float],
    duration: tuple[float, float],
    axis_failure: tuple[bool, bool] = (False, False),
) -> dict[str, torch.Tensor]:
    r"""构造两环境post-physics snapshot；env0/1分别属于formal rows16/32。"""

    return {
        "valid": torch.ones(2, dtype=torch.bool),
        "step": torch.full((2,), step, dtype=torch.long),
        "dataset_row": torch.tensor((16, 32), dtype=torch.long),
        "axis_speed_rad_s": torch.tensor(speed),
        "net_rotation_rad": torch.tensor(net_rotation),
        "completed_subgoals": torch.tensor((1.0 if net_rotation[0] >= math.pi / 6.0 else 0.0, 0.0)),
        "goal_success_pulse": torch.tensor((net_rotation[0] >= math.pi / 6.0, False)),
        "episode_duration_s": torch.tensor(duration),
        "tip_active_count": torch.tensor(tip),
        "palm_contact": torch.tensor(palm),
        "finger_non_tip_contact": torch.tensor(non_tip),
        "orientation_keypoint_error_m": torch.tensor((0.01, 0.02)),
        "position_error_m": torch.tensor((0.03, 0.04)),
        "termination_object_out_of_anchor": torch.zeros(2, dtype=torch.bool),
        "termination_goal_axis_misaligned": torch.tensor(axis_failure),
        "termination_time_out": torch.zeros(2, dtype=torch.bool),
    }


def test_terminal_record_preserves_pre_reset_speed_contact_and_angle() -> None:
    r"""Env0 terminal后即使runtime buffers清零，已finalize record仍保留terminal frame与两步均值。"""

    accumulator = FixedDurationTrajectoryAccumulator(torch.tensor((16, 32), dtype=torch.long), step_dt=0.05)
    first = _snapshot(
        step=1,
        net_rotation=(0.2, -0.1),
        speed=(4.0, -2.0),
        tip=(2.0, 1.0),
        palm=(1.0, 1.0),
        non_tip=(0.0, 1.0),
        duration=(0.05, 0.05),
    )
    accumulator.add_step(first, torch.tensor((1.0, 2.0)), torch.tensor((False, False)))
    terminal = _snapshot(
        step=2,
        net_rotation=(math.pi / 3.0, -0.2),
        speed=(8.0, -2.0),
        tip=(2.0, 1.0),
        palm=(0.0, 1.0),
        non_tip=(0.0, 1.0),
        duration=(0.1, 0.1),
        axis_failure=(True, False),
    )
    accumulator.add_step(terminal, torch.tensor((-50.0, 2.0)), torch.tensor((True, False)))

    # 模拟ManagerBasedRLEnv在env.step返回前把command/contact row0清零；record不应随引用改变。
    for name, value in terminal.items():
        if name not in {"valid", "dataset_row", "step"}:
            value[0] = False if value.dtype == torch.bool else 0
    accumulator.finish_window(terminal)
    first_record = accumulator.records[0]
    assert math.isclose(first_record["signed_net_rotation_rad"], math.pi / 3.0, rel_tol=1.0e-6)
    assert first_record["signed_axis_speed_sample_mean_rad_s"] == 6.0
    assert first_record["tip_active_count_mean"] == 2.0
    assert first_record["palm_occupancy_fraction"] == 0.5
    assert first_record["termination_goal_axis_misaligned"] is True
    assert first_record["episode_any_success_pulse"] is True
    assert first_record["reached_positive_30deg"] is True


def test_summary_reports_equal_asset_quantiles_and_single_seed_boundary() -> None:
    r"""每个asset先成分布，再等权聚合；trajectory count不伪装成seed uncertainty。"""

    accumulator = FixedDurationTrajectoryAccumulator(torch.tensor((16, 32), dtype=torch.long), step_dt=0.05)
    snapshot = _snapshot(
        step=1,
        net_rotation=(1.0, -1.0),
        speed=(20.0, -20.0),
        tip=(2.0, 0.0),
        palm=(1.0, 1.0),
        non_tip=(0.0, 1.0),
        duration=(0.05, 0.05),
    )
    accumulator.add_step(snapshot, torch.tensor((1.0, 3.0)), torch.tensor((False, False)))
    accumulator.finish_window(snapshot)
    summary = accumulator.summary(requested_steps=1)
    assert summary["trajectory_count"] == 2.0
    assert summary["equal_asset"]["asset_count"] == 2
    assert summary["equal_asset"]["signed_net_rotation_rad"]["median"] == 0.0
    assert summary["equal_asset"]["signed_net_rotation_rad"]["q10"] == 0.0
    assert summary["equal_asset"]["signed_net_rotation_rad"]["cvar10_lower"] == 0.0
    assert summary["uncertainty"]["independent_training_seed_count"] == 1
    assert summary["uncertainty"]["confidence_interval"] is None
