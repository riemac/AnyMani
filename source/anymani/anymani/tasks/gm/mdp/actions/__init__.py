r"""`gm` 任务的自定义动作空间模块。

当前仅包含一个动作术语：
    - `ClampedRelativeJointPositionAction`：相对关节位置动作，在 `apply_actions`
      中显式 clamp 目标到 soft joint limits。

该模块的设计原则：
    - 只覆写 IsaacLab 原生缺少的科研语义（joint limit clamp），其余解析逻辑复用基类。
    - 不加 EMA（相对增量自带平滑），不 rescale 到 limits（保持 raw rad 量纲，
      与 obs 侧 $q_i$ 同空间）。
"""

from __future__ import annotations

from .clamped_relative_action import ClampedRelativeJointActionCfg, ClampedRelativeJointPositionAction

__all__ = [
    "ClampedRelativeJointActionCfg",
    "ClampedRelativeJointPositionAction",
]
