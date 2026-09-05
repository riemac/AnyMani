r"""8-cell median reward-release的Isaac CurriculumManager adapter。

纯Torch asset/cell状态位于 :mod:`curriculum_state`。本模块只负责在command partial reset前读取刚结束episode
的positive net turns，并把per-env$\lambda_{rew}$暴露给reward与privileged critic；MVP不启用ADR。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from .commands import get_rotation_command
from .curriculum_state import (
    HETERO_REWARD_RELEASE_STATE_ATTR,
    HeterogeneousRewardReleaseState,
    even_median,
    release_from_net_turns,
)
from .episode_horizon import EPISODE_HORIZON_STEPS_ATTR, reference_horizon_turns


class RewardReleaseByAssetMedianCell(ManagerTermBase):
    r"""Isaac curriculum adapter；episode reset不清训练期asset/cell状态。"""

    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        r"""从cfg静态asset/cell/env routing构造state并发布给reward/critic。"""

        super().__init__(cfg, env)
        state = HeterogeneousRewardReleaseState(
            dataset_rows_by_asset=cast(Sequence[int], cfg.params["dataset_rows_by_asset"]),
            cell_ids_by_asset=cast(Sequence[int], cfg.params["cell_ids_by_asset"]),
            asset_index_by_env=cast(Sequence[int], cfg.params["asset_index_by_env"]),
            device=env.device,
        )
        setattr(env, HETERO_REWARD_RELEASE_STATE_ATTR, state)
        floor = float(cast(float, cfg.params.get("release_floor", 0.0)))  # 续训塑形下限，不是已学能力估计
        if not 0.0 <= floor <= 1.0:
            raise ValueError("reward release floor must lie in [0,1]")
        state.cell_lambda.fill_(floor)
        state.env_lambda.fill_(floor)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""Reward curriculum跨episode持续，不随partial reset清零。"""

        _ = env_ids

    def __call__(
        self,
        env,
        env_ids: Sequence[int] | slice,
        command_name: str,
        dataset_rows_by_asset: Sequence[int],
        cell_ids_by_asset: Sequence[int],
        asset_index_by_env: Sequence[int],
        release_start_turns: float = 1.0,
        release_end_turns: float = 2.0,
        ema_alpha: float = 0.05,
        release_floor: float = 0.0,
        reference_seconds: float | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""在reset事件采样下一回合之前，用刚结束回合的计划时长更新课程。

        若启用参考时长，以$N^{ref}=N^+T_{ref}/T_{planned}$输入EMA，失败不缩短分母。EMA仍按每次
        asset-reset cohort更新一次，随机回合会改变其更新频率；这不是按墙钟等价的EMA。``release_floor=1``
        的续训对照固定实际塑形强度，隔离该瞬态；候选EMA仍单独保留，不冒充已达到能力门。
        """

        _ = (dataset_rows_by_asset, cell_ids_by_asset, asset_index_by_env)  # 构造期已冻结
        state = getattr(env, HETERO_REWARD_RELEASE_STATE_ATTR, None)
        if not isinstance(state, HeterogeneousRewardReleaseState):
            raise RuntimeError("heterogeneous reward release state is unavailable")
        ids = (
            torch.arange(env.num_envs, device=env.device)
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        )
        # Manager也在初次/手动reset时调用课程；只统计真正结束的非空episode，保留full resume的EMA状态。
        ids = ids[(env.episode_length_buf[ids] > 0) & env.termination_manager.dones[ids]]
        if ids.numel() > 0:
            command = get_rotation_command(env, command_name)
            turns = command.positive_net_rotation_turns
            if reference_seconds is not None:
                lengths = getattr(env, EPISODE_HORIZON_STEPS_ATTR, None)
                planned_seconds = (
                    lengths.to(dtype=turns.dtype) * float(env.step_dt)
                    if lengths is not None
                    else torch.full_like(turns, float(env.max_episode_length) * float(env.step_dt))
                )
                turns = reference_horizon_turns(turns, planned_seconds, reference_seconds)
            state.update(
                reset_env_ids=ids,
                positive_net_turns_by_env=turns,
                ema_alpha=float(ema_alpha),
                release_start_turns=float(release_start_turns),
                release_end_turns=float(release_end_turns),
            )
        state.cell_lambda.clamp_(min=release_floor)  # 实际系数$\lambda=\max(\lambda_{curriculum},\lambda_{floor})$
        state.env_lambda.copy_(state.cell_lambda[state.cell_ids_by_asset[state.asset_index_by_env]])
        return {
            "lambda_mean": state.cell_lambda.mean().detach(),
            "lambda_min": state.cell_lambda.min().detach(),
            "lambda_max": state.cell_lambda.max().detach(),
            "net_turns_cell_median_mean": state.cell_net_turns_median.mean().detach(),
        }


def reward_release_gain(env) -> torch.Tensor:
    r"""返回每environment实际cell-level$\lambda_{rew}\in[0,1]$。"""

    state = getattr(env, HETERO_REWARD_RELEASE_STATE_ATTR, None)
    if not isinstance(state, HeterogeneousRewardReleaseState):
        return torch.zeros(env.num_envs, device=env.device)
    return state.env_lambda


def reward_release_observation(env) -> torch.Tensor:
    r"""返回privileged critic读取的`[N,1]`实际reward-release coefficient。"""

    return reward_release_gain(env).unsqueeze(-1)


__all__ = [
    "HETERO_REWARD_RELEASE_STATE_ATTR",
    "HeterogeneousRewardReleaseState",
    "RewardReleaseByAssetMedianCell",
    "even_median",
    "release_from_net_turns",
    "reward_release_gain",
    "reward_release_observation",
]
