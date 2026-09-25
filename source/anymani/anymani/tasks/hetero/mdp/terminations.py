'Anchor-drop and signed goal-normal terminations for the palm-up rotation task.'

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
    'Return the drop condition: distance from object position to anchor is at least 0.07 m.'

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
    'Return the normal-alignment failure: dot(z_object,z_goal) < cos(45 deg). Do not take the absolute value.'

    command = get_rotation_command(env, command_name)
    _, misaligned = task_termination_flags(
        command.position_error_m,
        command.goal_normal_alignment,
        max_axis_angle_deg=max_axis_angle_deg,
    )
    return misaligned


__all__ = ["goal_axis_misaligned", "object_out_of_anchor"]
