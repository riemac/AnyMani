'Represent optional phase-clock inputs as policy-step features; the default task disables this branch.'

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import torch


def normalize_phase_period_steps(value: int | None) -> int | None:
    'Normalize phase period steps.'

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
        raise ValueError("phase period must be an integer of at least two policy steps")
    return int(value)


def phase_clock_from_episode_steps(episode_steps: torch.Tensor, *, period_steps: int) -> torch.Tensor:
    'Handle phase clock from episode steps.'

    period = normalize_phase_period_steps(period_steps)
    if period is None:
        raise ValueError("phase period is required for an enabled clock")
    if episode_steps.ndim != 1 or episode_steps.dtype not in (torch.int32, torch.int64):
        raise ValueError("episode phase requires a rank-one integer policy-step counter")
    torch._assert_async(torch.all(episode_steps >= 0), "episode phase counter cannot be negative")  # pyright: ignore[reportPrivateImportUsage]
    angle = torch.remainder(episode_steps, period).to(torch.float32) * (2.0 * math.pi / period)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)  # shapes [N,2]


def phase_clock_contract(period_steps: int | None) -> dict[str, Any] | None:
    'Handle phase clock contract.'

    period = normalize_phase_period_steps(period_steps)
    if period is None:
        return None
    return {
        "source": "physical-episode-length-buf",
        "period_policy_steps": period,
        "increment_rad_per_policy_step": 2.0 * math.pi / period,
        "encoding": ["sin", "cos"],
        "encoding_dtype": "float32",
        "reset": "physical-episode-counter-zero-before-returned-observation",
        "transport_key": "phase_clock",
        "actor_adapter": "zero-linear2to128-after-contextual-joints-before-existing-head-norm",
        "critic_adapter": "zero-linear2to896-after-readout-before-existing-value-norm",
    }


def audit_phase_rollout(phases: torch.Tensor, reset_observations: torch.Tensor, *, period_steps: int) -> dict[str, Any]:
    'Handle audit phase rollout; shapes [env,time,2], [e,t].'

    if phases.ndim != 3 or phases.shape[-1] != 2 or phases.dtype != torch.float32:
        raise ValueError("phase rollout must have float32 [environment,time,2] shape")
    if reset_observations.shape != phases.shape[:2] or reset_observations.dtype != torch.bool:
        raise ValueError("phase rollout reset observations must be boolean [environment,time]")
    if phases.device != reset_observations.device or not phases.shape[0] or not phases.shape[1]:
        raise ValueError("phase rollout requires nonempty axes on the same device")
    time_index = torch.arange(phases.shape[1], device=phases.device).expand(phases.shape[:2])
    last_reset = torch.where(reset_observations, time_index, 0).cummax(dim=1).values
    episode_steps = time_index - last_reset
    expected = phase_clock_from_episode_steps(episode_steps.flatten(), period_steps=period_steps).reshape_as(phases)
    maximum_error = float((phases - expected).abs().max().item())
    if not bool(torch.isfinite(phases).all()) or maximum_error > 1.0e-6:
        raise RuntimeError(f"stored phase does not follow physical reset/step timing: max_error={maximum_error}")
    return {
        "status": "passed",
        "environment_count": phases.shape[0],
        "rollout_steps": phases.shape[1],
        "period_policy_steps": period_steps,
        "maximum_abs_error": maximum_error,
        "reset_observation_count": int(reset_observations.sum().item()),
        "semantics": "cold-first-rollout-before-action-phase-and-dones",
    }
