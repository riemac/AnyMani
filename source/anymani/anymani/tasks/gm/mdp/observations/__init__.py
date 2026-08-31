r"""Observation terms for `tasks.gm`.

IsaacLab RL 既然服务于层次通才专家训练阶段，用于训练 specialist policy / teacher，
而该 policy 本身不直接用于 sim2real，那么 teacher obs 可以尽可能携带有用状态、
命令、接触和特权信息。更复杂的 geometry tensor / token 表征仍由 `distill/` 接管；
本 package 只提供 GM MDP 层需要的浅 observation terms。

外部仍通过扁平 `gm_mdp.*` API 消费 terms；内部文件按 state、tactile、privileged、geometry
与 command 语义分工，不按某个实验 route 复制 observation 实现。
"""

from __future__ import annotations

from .observations_command import reorient_command
from .observations_geometry import joint_soft_pos_limits
from .observations_priv import (
    adr_actual_state,
    object_goal_task_state,
    object_orientation,
    object_pos,
    reward_release_coefficient,
)
from .observations_state import (
    canonical_active_joint_mask,
    canonical_asset_row,
    canonical_morphology_cell_one_hot,
    joint_pos_limit_normalized,
    joint_pos_raw,
    joint_target,
    joint_vel_raw,
    last_action,
    last_processed_action,
)
from .observations_tactile import (
    finger_non_tip_contact_bits_ema,
    fingertip_contact_binary,
    fingertip_contact_force,
    palm_force_magnitude_ema,
    tip_contact_bits_ema,
    tip_force_magnitude_ema,
)
from .observations_temporal import per_joint_policy_frame

__all__ = [
    "adr_actual_state",
    "canonical_active_joint_mask",
    "canonical_asset_row",
    "canonical_morphology_cell_one_hot",
    "finger_non_tip_contact_bits_ema",
    "fingertip_contact_binary",
    "fingertip_contact_force",
    "joint_pos_limit_normalized",
    "joint_pos_raw",
    "joint_soft_pos_limits",
    "joint_target",
    "joint_vel_raw",
    "last_action",
    "last_processed_action",
    "object_goal_task_state",
    "object_orientation",
    "object_pos",
    "palm_force_magnitude_ema",
    "per_joint_policy_frame",
    "reorient_command",
    "reward_release_coefficient",
    "tip_contact_bits_ema",
    "tip_force_magnitude_ema",
]
