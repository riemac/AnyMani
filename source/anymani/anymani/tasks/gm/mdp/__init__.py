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
from .rewards import reorientation_reward_placeholder
from .terminations import object_falling_placeholder

__all__ = [
    "ClampedRelativeJointActionCfg",
    "ClampedRelativeJointPositionAction",
    "object_falling_placeholder",
    "reorientation_reward_placeholder",
]
