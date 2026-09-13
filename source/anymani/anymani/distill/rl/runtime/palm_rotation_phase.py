r"""由物理回合计数生成可重放的内部相位，不扩展 N040 或任务观测真值。"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import torch


def normalize_phase_period_steps(value: int | None) -> int | None:
    r"""验证周期的 policy-step 单位；None 保留无时钟的旧合同。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
        raise ValueError("phase period must be an integer of at least two policy steps")
    return int(value)


def phase_clock_from_episode_steps(episode_steps: torch.Tensor, *, period_steps: int) -> torch.Tensor:
    r"""读取每个环境的回合年龄，返回无量纲 FP32 ``[sin(phi), cos(phi)]``。

    $$\phi_e=2\pi\frac{n_e\bmod P}{P}.$$

    ``n_e`` 由 Isaac 在一次 policy step 后递增、在物理 reset 时归零。本函数不保存或修改计数；PPO
    重放必须使用 rollout 中已存储的相位，而不能在 forward 中重新推进。先在整数域取模，避免长回合
    先转 FP32 再取模导致周期漂移；sin/cos 本身保留普通 FP32 数值精度。
    """

    period = normalize_phase_period_steps(period_steps)
    if period is None:
        raise ValueError("phase period is required for an enabled clock")
    if episode_steps.ndim != 1 or episode_steps.dtype not in (torch.int32, torch.int64):
        raise ValueError("episode phase requires a rank-one integer policy-step counter")
    torch._assert_async(torch.all(episode_steps >= 0), "episode phase counter cannot be negative")  # pyright: ignore[reportPrivateImportUsage]
    angle = torch.remainder(episode_steps, period).to(torch.float32) * (2.0 * math.pi / period)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)  # `[N,2]`，不依赖物体/触觉/墙钟。


def phase_clock_contract(period_steps: int | None) -> dict[str, Any] | None:
    r"""返回可绑定到方法身份的周期、来源、编码顺序和零适配器语义。"""

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
    r"""核验首次冷启动 rollout 的 ``[env,time,2]`` 时钟及动作前 reset 标记。

    rl_games 在动作前保存 observation 和上一环境步的 done。因此 ``reset_observations[e,t]`` 表示
    该次动作已属于新物理回合，计数应为零；其余时刻相对上一个动作递增一次。先由真实 done 序列重建
    整数回合年龄，再独立与已存 phase 比较，可发现错一拍、时间/环境轴混淆和错误常数时钟。
    """

    if phases.ndim != 3 or phases.shape[-1] != 2 or phases.dtype != torch.float32:
        raise ValueError("phase rollout must have float32 [environment,time,2] shape")
    if reset_observations.shape != phases.shape[:2] or reset_observations.dtype != torch.bool:
        raise ValueError("phase rollout reset observations must be boolean [environment,time]")
    if phases.device != reset_observations.device or not phases.shape[0] or not phases.shape[1]:
        raise ValueError("phase rollout requires nonempty axes on the same device")
    time_index = torch.arange(phases.shape[1], device=phases.device).expand(phases.shape[:2])
    last_reset = torch.where(reset_observations, time_index, 0).cummax(dim=1).values
    episode_steps = time_index - last_reset  # 首个冷启动 observation 的回合年龄固定为零。
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
