r"""计划回合长度、partial reset和课程时间归一化合同。"""

from types import SimpleNamespace

import torch

from anymani.tasks.hetero.mdp.episode_horizon import (
    EPISODE_HORIZON_STEPS_ATTR,
    planned_time_out,
    reference_horizon_turns,
    reset_episode_horizon,
)


def test_reference_turns_compare_rates_without_rewarding_early_failure() -> None:
    r"""相同净转速具有相同课程信号；只转一会儿后失败不能借短存活分母获利。"""

    turns = torch.tensor((1.0, 2.0, 4.0, 0.2))
    planned = torch.tensor((30.0, 60.0, 120.0, 30.0))
    actual = reference_horizon_turns(turns, planned, 120.0)
    torch.testing.assert_close(actual, torch.tensor((4.0, 4.0, 4.0, 0.8)))
    assert torch.equal(reference_horizon_turns(turns, torch.full_like(turns, 120), 120), turns)


def test_partial_horizon_reset_and_timeout_use_selected_plans() -> None:
    r"""20–60秒采样只更新指定行；固定时长不消耗随机流。"""

    env = SimpleNamespace(num_envs=256, device="cpu", step_dt=0.05, max_episode_length=1200)
    torch.manual_seed(42)
    reset_episode_horizon(env, torch.arange(256), minimum_seconds=20, maximum_seconds=60)
    lengths = getattr(env, EPISODE_HORIZON_STEPS_ATTR)
    assert lengths.min() >= 400 and lengths.max() <= 1200 and lengths.unique().numel() > 100
    before = lengths.clone()
    rng = torch.get_rng_state().clone()
    reset_episode_horizon(env, torch.tensor((2, 7)), minimum_seconds=30, maximum_seconds=30)
    assert lengths[2] == lengths[7] == 600
    assert torch.equal(lengths[:2], before[:2]) and torch.equal(lengths[8:], before[8:])
    assert torch.equal(rng, torch.get_rng_state())
    env.episode_length_buf = lengths - 1
    assert not planned_time_out(env).any()
    env.episode_length_buf = lengths.clone()
    assert planned_time_out(env).all()
