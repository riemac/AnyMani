r"""Palm-up rotation task的anchor drop与signed goal-normal terminations。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import get_rotation_command
from .task_math import task_termination_flags

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_out_of_anchor(
    env: ManagerBasedRLEnv,
    command_name: str,
    *,
    drop_distance_m: float = 0.07,
) -> torch.Tensor:
    r"""返回$[\|p_o-p_{anchor}\|\ge0.07\,m]$。"""

    command = get_rotation_command(env, command_name)
    drop, _ = task_termination_flags(
        command.position_error_m,
        command.goal_normal_alignment,
        drop_distance_m=drop_distance_m,
    )
    return drop


def goal_axis_misaligned(
    env: ManagerBasedRLEnv,
    command_name: str,
    *,
    max_axis_angle_deg: float = 45.0,
) -> torch.Tensor:
    r"""返回$[z_o^Tz_g<\cos45^\circ]$，不取absolute value。"""

    command = get_rotation_command(env, command_name)
    _, misaligned = task_termination_flags(
        command.position_error_m,
        command.goal_normal_alignment,
        max_axis_angle_deg=max_axis_angle_deg,
    )
    return misaligned


__all__ = ["goal_axis_misaligned", "object_out_of_anchor"]
