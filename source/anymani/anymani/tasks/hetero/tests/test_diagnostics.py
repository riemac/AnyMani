r"""Terminal-step success、partial-reset isolation与equal-asset reducer合同。"""

from __future__ import annotations

import math

import torch

from anymani.tasks.hetero.mdp.diagnostics import (
    asset_episode_sufficient_statistics,
    equal_asset_metric_from_extras,
)


def test_terminal_statistics_include_success_pulse_and_ignore_nonreset_stale_bits() -> None:
    r"""Reset rows0/2计数；rows1/3的stale dones与success不进入任何asset sum。"""

    extras = asset_episode_sufficient_statistics(
        dataset_row_by_env=torch.tensor((10, 10, 20, 20)),
        reset_env_ids=torch.tensor((0, 2)),
        goal_success_count=torch.tensor((2.0, 100.0, 5.0, 100.0)),
        goal_success_pulse=torch.tensor((True, True, False, True)),
        net_rotation_rad=torch.tensor((math.pi, 99.0, -math.pi, 99.0)),
        positive_net_rotation_turns=torch.tensor((0.5, 99.0, 0.0, 99.0)),
        episode_duration_s=torch.tensor((10.0, 99.0, 20.0, 99.0)),
        termination_bits={
            "object_out_of_anchor": torch.tensor((True, True, False, True)),
            "goal_axis_misaligned": torch.tensor((False, True, True, True)),
            "time_out": torch.tensor((False, True, False, True)),
        },
        horizon_s=120.0,
    )
    assert extras["asset/10/episode_count"] == 1.0
    assert extras["asset/10/goal_success_count_sum"] == 3.0  # count2 + terminal pulse1
    assert extras["asset/20/goal_success_count_sum"] == 5.0
    assert extras["asset/10/termination_object_out_of_anchor_sum"] == 1.0
    assert extras["asset/20/termination_goal_axis_misaligned_sum"] == 1.0
    assert extras["asset/10/termination_time_out_sum"] == 0.0  # stale row1 timeout被隔离
    assert abs(extras["asset/20/net_rotation_turns_signed_sum"] + 0.5) < 1.0e-7


def test_equal_asset_reducer_uses_per_asset_means() -> None:
    r"""Episode counts不等时仍先按asset求均值。"""

    extras = {
        "asset/10/episode_count": 2.0,
        "asset/10/net_rotation_rad_signed_sum": 4.0,
        "asset/20/episode_count": 1.0,
        "asset/20/net_rotation_rad_signed_sum": 10.0,
    }
    assert equal_asset_metric_from_extras(extras, "net_rotation_rad_signed_sum") == 6.0
