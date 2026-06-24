r"""MDP terms for generalized manipulation tasks.

这里是 `gm` 环境自己的 MDP 语义层。设计原则：

- 能直接复用 Isaac Lab 通用项时，不复制实现；
- 一旦涉及 AnyMani 的 hand asset metadata、same-topology contract、
  object-in-hand 成功判据、morphology-aware observation，就在 `gm/mdp`
  下显式落名；
- 不从旧 `tasks/inhand` 继续借模块名，避免把 LEAP-specific 历史语义带入新主线。
"""

from __future__ import annotations

from .actions.clamped_relative_action import ClampedRelativeJointActionCfg, ClampedRelativeJointPositionAction
from .commands.commands_cfg import ReorientCommandCfg
from .commands.reorient_command import ReorientCommand
from .curriculums import RewardCurriculumByGoalSuccess
from .events import (
    apply_generated_structural_collision_filter,
    generated_structural_collision_filter_pairs,
    record_object_reset_anchor,
)
from .observations import (
    fingertip_contact_binary,
    fingertip_contact_force_h,
    joint_pos_raw,
    joint_soft_pos_limits,
    joint_vel_raw,
    last_processed_action,
    object_pos_h,
    object_rot6d_h,
    reorient_command,
)
from .rewards import (
    AxisDeltaRotationReward,
    action_l2_curriculum,
    action_rate_l2_curriculum,
    bad_non_tip_contact,
    goal_success_bonus,
    good_fingertip_contact,
    keypoint_reorientation_reward,
    reorientation_reward_placeholder,
    termination_penalty_placeholder,
    torque_l2_curriculum,
)
from .terminations import object_falling_placeholder, object_out_of_hand

__all__ = [
    "AxisDeltaRotationReward",
    "ClampedRelativeJointActionCfg",
    "ClampedRelativeJointPositionAction",
    "ReorientCommand",
    "ReorientCommandCfg",
    "RewardCurriculumByGoalSuccess",
    "action_l2_curriculum",
    "action_rate_l2_curriculum",
    "apply_generated_structural_collision_filter",
    "bad_non_tip_contact",
    "fingertip_contact_binary",
    "fingertip_contact_force_h",
    "generated_structural_collision_filter_pairs",
    "goal_success_bonus",
    "good_fingertip_contact",
    "joint_pos_raw",
    "joint_soft_pos_limits",
    "joint_vel_raw",
    "keypoint_reorientation_reward",
    "last_processed_action",
    "object_pos_h",
    "object_rot6d_h",
    "object_falling_placeholder",
    "object_out_of_hand",
    "record_object_reset_anchor",
    "reorient_command",
    "reorientation_reward_placeholder",
    "termination_penalty_placeholder",
    "torque_l2_curriculum",
]
