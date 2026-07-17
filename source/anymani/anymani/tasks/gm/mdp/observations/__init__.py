r"""Observation terms for `tasks.gm`.

IsaacLab RL 既然服务于层次通才专家训练阶段，用于训练 specialist policy / teacher，
而该 policy 本身不直接用于 sim2real，那么 teacher obs 可以尽可能携带有用状态、
命令、接触和特权信息。更复杂的 geometry tensor / token 表征仍由 `distill/` 接管；
本 package 只提供 GM MDP 层需要的浅 observation terms。

本 package 保持外部扁平 API：`anymani.tasks.gm.mdp.joint_pos_raw`、
`gm_mdp.fingertip_contact_binary` 等调用不变。
"""

from __future__ import annotations

from .observations_command import reorient_command
from .observations_contact import (
    fingertip_contact_binary,
    fingertip_contact_force,
    tactile_finger_non_tip_bits,
    tactile_palm_force_ema,
    tactile_tip_contact_bits,
    tactile_tip_force_ema,
)
from .observations_geometry import joint_soft_pos_limits
from .observations_priv import object_orientation, object_pos
from .observations_state import (
    joint_pos_limit_normalized,
    joint_pos_raw,
    joint_vel_raw,
    last_action,
    last_processed_action,
)
from .observations_tactile import (
    tactile_rotation_critic_state,
    tactile_rotation_policy_frame,
    tactile_rotation_privileged_task_state,
)

__all__ = [
    "fingertip_contact_binary",
    "fingertip_contact_force",
    "joint_pos_limit_normalized",
    "joint_pos_raw",
    "joint_soft_pos_limits",
    "joint_vel_raw",
    "last_action",
    "last_processed_action",
    "object_orientation",
    "object_pos",
    "reorient_command",
    "tactile_finger_non_tip_bits",
    "tactile_palm_force_ema",
    "tactile_tip_contact_bits",
    "tactile_tip_force_ema",
    "tactile_rotation_critic_state",
    "tactile_rotation_policy_frame",
    "tactile_rotation_privileged_task_state",
]
