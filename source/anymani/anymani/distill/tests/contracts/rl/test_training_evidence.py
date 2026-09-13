r"""不同环境异步终止、跨策略版本与末端右删失的记录合同。"""

from __future__ import annotations

import polars as pl
import torch
from anymani.distill.diagnostics.recording.rl.training_evidence import TrainingEvidence


def test_partial_resets_preserve_individual_rewards_and_policy_versions(tmp_path) -> None:
    r"""结束的回合不混入下一回合；未结束回合单列，奖励积分不按计划时长稀释。"""
    recorder = TrainingEvidence(tmp_path, "a" * 64, torch.tensor([0, 1]), ["rotation", "effort"])

    def snapshot(durations, turns, drop, timeout):
        net = torch.tensor(turns) * (2 * torch.pi)
        return {
            "episode_duration_s": torch.tensor(durations),
            "net_rotation_rad": net,
            "absolute_path_rotation_rad": net.abs(),
            "max_positive_net_rotation_rad": net.clamp_min(0),
            "completed_subgoals": torch.zeros(2),
            "rotation_frontier_count": torch.floor(net.clamp_min(0) / (torch.pi / 6)),
            "termination_object_out_of_anchor": torch.tensor(drop),
            "termination_goal_axis_misaligned": torch.zeros(2, dtype=torch.bool),
            "termination_time_out": torch.tensor(timeout),
        }

    recorder.capture(
        snapshot([0.05, 0.05], [0.05, 0.05], [False, False], [False, False]), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    recorder.capture(
        snapshot([0.1, 0.1], [0.1, 0.1], [True, False], [False, False]), torch.tensor([[-5.0, 0.0], [1.0, 1.0]])
    )
    first = recorder.drain()
    torch.testing.assert_close(first["reward_terms"], torch.tensor([[-2.0, 1.0], [2.0, 2.5]]))
    recorder.policy_version = 30720
    recorder.capture(
        snapshot([0.05, 0.15], [0.02, 0.3], [False, False], [False, True]), torch.tensor([[2.0, 2.0], [1.0, 2.0]])
    )
    recorder.close()
    table = pl.read_parquet(list(tmp_path.glob("*.parquet"))).sort("env_id", "episode_id")
    assert table.height == 3
    assert table["episode_id"].to_list() == [0, 1, 0]
    assert table["censored"].to_list() == [False, True, False]
    assert table["reward_sum/rotation"].to_list() == [-4.0, 2.0, 5.0]
    assert table["reward_sum/effort"].to_list() == [2.0, 2.0, 7.0]
    assert table["policy_version_start"].to_list() == [0, 30720, 0]
    assert table["policy_version_end"].to_list() == [0, 30720, 30720]
    recorder.close()
    assert len(list(tmp_path.glob("*.parquet"))) == 1
