(
    'Isaac CurriculumManager adapter for 8-cell median reward release. Pure-Torch '
    'asset/cell state lives in curriculum_state. Before command partial reset, '
    "read completed episodes' positive net turns and expose the per-env reward "
    'coefficient to reward and privileged critic terms. MVP leaves ADR disabled.'
)

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
    'Isaac curriculum adapter; episode reset preserves training-time asset/cell state.'

    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        'Build state from static asset/cell/env routing and publish it to reward/critic terms.'

        super().__init__(cfg, env)
        state = HeterogeneousRewardReleaseState(
            dataset_rows_by_asset=cast(Sequence[int], cfg.params["dataset_rows_by_asset"]),
            cell_ids_by_asset=cast(Sequence[int], cfg.params["cell_ids_by_asset"]),
            asset_index_by_env=cast(Sequence[int], cfg.params["asset_index_by_env"]),
            device=env.device,
        )
        setattr(env, HETERO_REWARD_RELEASE_STATE_ATTR, state)
        floor = float(cast(float, cfg.params.get("release_floor", 0.0)))  # Resume shaping floor; this is not an estimate of learned capability.
        if not 0.0 <= floor <= 1.0:
            raise ValueError("reward release floor must lie in [0,1]")
        state.cell_lambda.fill_(floor)
        state.env_lambda.fill_(floor)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        'Reward curriculum persists across episodes and partial resets.'

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
        (
            "Before sampling a new episode, update from the completed episode's planned "
            'duration. Normalize turns as N_ref=N_plus*T_ref/T_planned; failure never '
            'shortens the denominator. Updates occur per asset-reset cohort, so duration '
            'randomization changes wall-clock cadence. A release_floor of 1 pins shaping '
            'during resume comparisons while candidate EMA remains separate.'
        )

        _ = (dataset_rows_by_asset, cell_ids_by_asset, asset_index_by_env)  # Frozen after construction.
        state = getattr(env, HETERO_REWARD_RELEASE_STATE_ATTR, None)
        if not isinstance(state, HeterogeneousRewardReleaseState):
            raise RuntimeError("heterogeneous reward release state is unavailable")
        ids = (
            torch.arange(env.num_envs, device=env.device)
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        )
        # The manager also calls the curriculum on initial/manual reset. Count only completed non-empty episodes and preserve EMA state across full resume.
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
        state.cell_lambda.clamp_(min=release_floor)  # Actual coefficient is max(curriculum coefficient, resume floor).
        state.env_lambda.copy_(state.cell_lambda[state.cell_ids_by_asset[state.asset_index_by_env]])
        return {
            "lambda_mean": state.cell_lambda.mean().detach(),
            "lambda_min": state.cell_lambda.min().detach(),
            "lambda_max": state.cell_lambda.max().detach(),
            "net_turns_cell_median_mean": state.cell_net_turns_median.mean().detach(),
        }


def reward_release_gain(env) -> torch.Tensor:
    'Return the actual per-env cell-level reward coefficient in [0,1].'

    state = getattr(env, HETERO_REWARD_RELEASE_STATE_ATTR, None)
    if not isinstance(state, HeterogeneousRewardReleaseState):
        return torch.zeros(env.num_envs, device=env.device)
    return state.env_lambda


def reward_release_observation(env) -> torch.Tensor:
    'Return the actual reward-release coefficient for the privileged critic, shape [N,1].'

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
