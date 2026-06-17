r"""Reward terms for `tasks.gm`.

本 package 只承载 generalized in-hand manipulation 的任务奖励与正则项。奖励设计描述
“物体是否完成了手内操作目标”，不在 reward 中编码资产采样偏好；资产 bank 如何采样、
哪些 hand variant 进入训练，属于 `distill` 的训练组织问题。

当前奖励设计对齐 AnyRotate 分组：
$$
r = r_{\text{reorient}} + r_{\text{contact}} + r_{\text{stable}} + r_{\text{terminate}}.
$$

外部 API 保持扁平：`gm_mdp.keypoint_reorientation_reward`、`gm_mdp.good_fingertip_contact`
等调用不变。
"""

from __future__ import annotations

from .rewards_contact import bad_non_tip_contact, good_fingertip_contact
from .rewards_reorient import (
    AxisDeltaRotationReward,
    goal_success_bonus,
    keypoint_reorientation_reward,
    reorientation_reward_placeholder,
)
from .rewards_stable import action_l2_curriculum, action_rate_l2_curriculum, torque_l2_curriculum
from .rewards_terminate import termination_penalty_placeholder

__all__ = [
    "AxisDeltaRotationReward",
    "action_l2_curriculum",
    "action_rate_l2_curriculum",
    "bad_non_tip_contact",
    "goal_success_bonus",
    "good_fingertip_contact",
    "keypoint_reorientation_reward",
    "reorientation_reward_placeholder",
    "termination_penalty_placeholder",
    "torque_l2_curriculum",
]
