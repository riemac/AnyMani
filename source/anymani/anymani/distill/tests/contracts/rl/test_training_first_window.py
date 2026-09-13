r"""首30秒在到达时结算，之后失败不改写已观察窗口；不启动仿真。"""

from __future__ import annotations

import polars as pl
import pytest
import torch
from anymani.distill.diagnostics.recording.rl.training_evidence import TrainingEvidence


def _snapshot(step: int, *, net: float, goals: int, drop: bool = False, timeout: bool = False) -> dict:
    r"""构造单环境post-physics事实；step为本回合已经完成的20Hz控制步数。"""
    return {
        "episode_duration_s": torch.tensor([step * 0.05]),  # 实际经过秒数，不是计划120秒。
        "net_rotation_rad": torch.tensor([net * 2 * torch.pi]),  # 有符号物理净转。
        "absolute_path_rotation_rad": torch.tensor([abs(net) * 2.5 * torch.pi]),  # 非零时方向性为±.8。
        "max_positive_net_rotation_rad": torch.tensor([max(net, 0) * 2 * torch.pi]),  # 合法正向前沿。
        "completed_subgoals": torch.tensor([float(goals)]),  # 包含当前尚未由command消费的成功pulse。
        "rotation_frontier_count": torch.tensor([float(int(max(net, 0) * 12))]),  # 与目标数分开。
        "termination_object_out_of_anchor": torch.tensor([drop]),  # 真实物理终止。
        "termination_goal_axis_misaligned": torch.tensor([False]),  # 本夹具不额外触发轴失败。
        "termination_time_out": torch.tensor([timeout]),  # 有限时域，提前发生时是窗口删失。
    }


def _recorder(tmp_path) -> TrainingEvidence:
    r"""显式启用独立首30秒产物；原episode目录继续只保存回合结束记录。"""
    return TrainingEvidence(
        tmp_path / "episodes", "a" * 64, torch.tensor([0]), ["progress"],
        first_window_root=tmp_path / "first30",  # 两类证据的统计单位及目录分开。
    )


def test_first30_publishes_before_episode_end_and_preserves_goal_pulse(tmp_path) -> None:
    r"""第600步立即记录15个目标；后续回合进展及晚期掉落不覆盖这份安全窗口。"""
    recorder = _recorder(tmp_path)
    for step in range(1, 601):
        recorder.capture(_snapshot(step, net=step / 600, goals=step // 40), torch.zeros(1, 1))
    recorder.drain(force=True)  # 30秒事件独立flush，不等待120秒回合结束。
    assert not list((tmp_path / "episodes").glob("*.parquet"))
    window = pl.read_parquet(list((tmp_path / "first30").glob("*.parquet")))
    assert window.height == 1
    assert window["goal_count"].to_list() == [15]  # 边界第600步的第15次成功必须被计入。
    assert window["safe"].to_list() == [True]
    assert window["complete"].to_list() == [True]
    assert window["net_turns"][0] == pytest.approx(1.0)
    for step in range(601, 651):
        recorder.capture(_snapshot(step, net=2.0, goals=30, drop=step == 650), torch.zeros(1, 1))
    recorder.close()
    episode = pl.read_parquet(list((tmp_path / "episodes").glob("*.parquet")))
    assert episode["termination_drop"].to_list() == [True]
    assert episode["first30_safe"].to_list() == [True]  # 32.5秒掉落不改写30秒安全事实。
    assert episode["goal_count_first30"].to_list() == [15]
    assert episode["net_turns_first30"][0] == pytest.approx(1.0)
    assert episode["absolute_path_turns_first30"][0] == pytest.approx(1.25)
    assert pl.read_parquet(list((tmp_path / "first30").glob("*.parquet"))).height == 1


def test_early_failure_keeps_raw_prefix_and_next_episode_has_independent_window(tmp_path) -> None:
    r"""早失败不按30/20放大，reset后的新回合重新累计目标及策略版本。"""
    recorder = _recorder(tmp_path)
    for step in range(1, 401):
        if step == 301:
            recorder.policy_version = 61440  # 第一回合跨越一次策略更新。
        recorder.capture(_snapshot(step, net=step / 800, goals=7, drop=step == 400), torch.zeros(1, 1))
    recorder.drain()
    for step in range(1, 601):
        recorder.capture(_snapshot(step, net=step / 300, goals=32), torch.zeros(1, 1))
    recorder.close()
    windows = pl.read_parquet(list((tmp_path / "first30").glob("*.parquet"))).sort("episode_id")
    assert windows["episode_id"].to_list() == [0, 1]
    assert windows["complete"].to_list() == [False, True]
    assert windows["safe"].to_list() == [False, True]
    assert windows["net_turns"].to_list() == pytest.approx([0.5, 2.0])
    assert windows["goal_count"].to_list() == [7, 32]
    assert windows["policy_version_start"].to_list() == [0, 61440]
    assert windows["policy_version_end"].to_list() == [61440, 61440]
    assert recorder.first30_statistics.summary()["first30_window_count"] == 2


@pytest.mark.parametrize("step,drop,timeout,expected", [(100, False, True, 0), (600, True, True, 1)])
def test_timeout_censoring_and_simultaneous_boundary_failure(tmp_path, step, drop, timeout, expected) -> None:
    r"""纯早timeout不进入窗口分母；第600步timeout加物理失败必须算不安全完整窗口。"""
    recorder = _recorder(tmp_path)
    for current in range(1, step + 1):
        recorder.capture(
            _snapshot(current, net=0.5, goals=5, drop=drop and current == step, timeout=timeout and current == step),
            torch.zeros(1, 1),
        )
    recorder.close()
    assert recorder.first30_statistics.summary()["first30_window_count"] == expected
    paths = list((tmp_path / "first30").glob("*.parquet"))
    if expected:
        window = pl.read_parquet(paths)
        assert window["complete"].to_list() == [True]
        assert window["safe"].to_list() == [False]
    else:
        assert not paths  # 不把5秒timeout当作30秒物理失败或成功。
