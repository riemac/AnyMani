(
    'Plan episode length at reset and normalize curriculum progress by planned '
    'duration. Early failure does not shorten the denominator. Convert positive '
    'net turns to the reference duration before per-asset EMA/cell-median '
    'updates. Duration randomization is independent of physical ADR.'
)

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

EPISODE_HORIZON_STEPS_ATTR = "_anymani_hetero_episode_horizon_steps"


def reference_horizon_turns(
    turns: torch.Tensor, planned_seconds: torch.Tensor, reference_seconds: float
) -> torch.Tensor:
    'Return N_ref = N_plus * T_ref / T_planned; never divide by actual survival time.'

    if turns.shape != planned_seconds.shape or not math.isfinite(reference_seconds) or reference_seconds <= 0:
        raise ValueError("turns and planned horizons must align and reference duration must be positive")
    if not torch.isfinite(turns).all() or not torch.isfinite(planned_seconds).all() or not (planned_seconds > 0).all():
        raise ValueError("planned episode durations must be finite and positive")
    return turns.clamp_min(0.0) * (reference_seconds / planned_seconds)  # Equal reference durations preserve the original net-turn count.


def reset_episode_horizon(
    env: Any, env_ids: torch.Tensor | Sequence[int] | None, *, minimum_seconds: float, maximum_seconds: float
) -> None:
    (
        'Sample an inclusive integer step limit uniformly for selected envs only; '
        'leave other envs and physical state unchanged. Use ceil(T_min/dt) through '
        'floor(T_max/dt). At 20 Hz, 20-60 seconds means 400-1200 steps. The '
        "configured maximum is storage capacity/fallback, not each episode's sampled "
        'plan.'
    )

    if not 0 < minimum_seconds <= maximum_seconds or not math.isfinite(maximum_seconds):
        raise ValueError("episode duration bounds must satisfy 0 < minimum <= maximum")
    minimum_steps = math.ceil(minimum_seconds / float(env.step_dt) - 1.0e-9)  # Round inward so actual duration meets the declared lower bound.
    maximum_steps = math.floor(maximum_seconds / float(env.step_dt) + 1.0e-9)  # Do not exceed the declared upper bound.
    if minimum_steps < 1 or minimum_steps > maximum_steps:
        raise ValueError("episode duration interval must contain at least one valid policy-step horizon")
    lengths = getattr(env, EPISODE_HORIZON_STEPS_ATTR, None)
    if lengths is None:
        lengths = torch.full((env.num_envs,), maximum_steps, dtype=torch.long, device=env.device)
        setattr(env, EPISODE_HORIZON_STEPS_ATTR, lengths)
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    )
    if minimum_steps == maximum_steps:
        lengths[ids] = maximum_steps  # Fixed duration consumes no extra RNG, preserving comparison with the prior protocol.
    else:
        lengths[ids] = torch.randint(minimum_steps, maximum_steps + 1, (ids.numel(),), device=env.device)


def planned_time_out(env: Any) -> torch.Tensor:
    "Check timeout against this episode's planned policy steps; physical failures remain separate."

    lengths = getattr(env, EPISODE_HORIZON_STEPS_ATTR, None)
    if lengths is None:
        return env.episode_length_buf >= env.max_episode_length  # Shape inference may run before the first reset.
    return env.episode_length_buf >= lengths
