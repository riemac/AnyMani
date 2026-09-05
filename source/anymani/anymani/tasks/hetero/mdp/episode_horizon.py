r"""异构旋转的计划回合长度与可比较的课程进度。

计划时长在reset时采样，失败可以提前结束但不缩短课程归一化分母。将净圈换算到参考时长后，再沿用
逐资产EMA与cell中位数，可避免相同转速仅因回合变短而丢失奖励释放。时间随机化独立于物理ADR。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

EPISODE_HORIZON_STEPS_ATTR = "_anymani_hetero_episode_horizon_steps"


def reference_horizon_turns(
    turns: torch.Tensor, planned_seconds: torch.Tensor, reference_seconds: float
) -> torch.Tensor:
    r"""返回$N^{ref}=N^+T_{ref}/T_{planned}$；失败不以实际存活时间作分母。"""

    if turns.shape != planned_seconds.shape or not math.isfinite(reference_seconds) or reference_seconds <= 0:
        raise ValueError("turns and planned horizons must align and reference duration must be positive")
    if not torch.isfinite(turns).all() or not torch.isfinite(planned_seconds).all() or not (planned_seconds > 0).all():
        raise ValueError("planned episode durations must be finite and positive")
    return turns.clamp_min(0.0) * (reference_seconds / planned_seconds)  # 同参考时长时严格保持原净圈数


def reset_episode_horizon(
    env: Any, env_ids: torch.Tensor | Sequence[int] | None, *, minimum_seconds: float, maximum_seconds: float
) -> None:
    r"""按selected env采样闭区间计划上限，不改其他实例，也不改变物体/手的物理状态。

    $L_i\sim\mathcal U_{\mathbb Z}(\lceil T_{min}/\Delta t\rceil,\lfloor T_{max}/\Delta t\rfloor)$，
    20 Hz下20–60秒对应400–1200步。配置上限是存储容量与fallback，不代替每个回合的计划上限。
    """

    if not 0 < minimum_seconds <= maximum_seconds or not math.isfinite(maximum_seconds):
        raise ValueError("episode duration bounds must satisfy 0 < minimum <= maximum")
    minimum_steps = math.ceil(minimum_seconds / float(env.step_dt) - 1.0e-9)  # 向内取整，实际时长不低于声明下限
    maximum_steps = math.floor(maximum_seconds / float(env.step_dt) + 1.0e-9)  # 不越过声明上限
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
        lengths[ids] = maximum_steps  # 固定时长不消费额外RNG，便于与旧协议对照
    else:
        lengths[ids] = torch.randint(minimum_steps, maximum_steps + 1, (ids.numel(),), device=env.device)


def planned_time_out(env: Any) -> torch.Tensor:
    r"""按本回合计划policy steps判断timeout；失败仍由独立物理termination处理。"""

    lengths = getattr(env, EPISODE_HORIZON_STEPS_ATTR, None)
    if lengths is None:
        return env.episode_length_buf >= env.max_episode_length  # 首次reset前的shape推断
    return env.episode_length_buf >= lengths
