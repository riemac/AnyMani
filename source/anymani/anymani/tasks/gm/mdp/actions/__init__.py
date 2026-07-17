r"""`gm` 任务的自定义动作空间模块。

本命名空间承载 generalized manipulation 中可跨 hand / task 复用的 action 原件。
当前主线是声明式 ADR joint-position action：

- `ADRRelativeJointPositionActionCfg`：relative raw-rad delta，支持 `reference="current"|"target"`；
- `ADREMAJointPositionToLimitsActionCfg`：joint-limit absolute target + EMA，支持同一 reference 枚举；
- `ClampedRelativeJointPositionAction`：早期 plain current-relative raw-delta scaffold，保留作历史对照。
"""

from __future__ import annotations

from .adr_joint_actions import (
    ADREMAJointPositionToLimitsAction,
    ADREMAJointPositionToLimitsActionCfg,
    ADRJointAction,
    ADRRelativeJointPositionAction,
    ADRRelativeJointPositionActionCfg,
    PolicyStepADRTargetJointPositionAction,
    PolicyStepADRTargetJointPositionActionCfg,
    compute_ema_joint_command,
    compute_leap_adr_latency_steps,
    compute_relative_joint_command,
)
from .clamped_relative_action import ClampedRelativeJointActionCfg, ClampedRelativeJointPositionAction

__all__ = [
    "ADREMAJointPositionToLimitsAction",
    "ADREMAJointPositionToLimitsActionCfg",
    "ADRJointAction",
    "ADRRelativeJointPositionAction",
    "ADRRelativeJointPositionActionCfg",
    "ClampedRelativeJointActionCfg",
    "ClampedRelativeJointPositionAction",
    "PolicyStepADRTargetJointPositionAction",
    "PolicyStepADRTargetJointPositionActionCfg",
    "compute_ema_joint_command",
    "compute_leap_adr_latency_steps",
    "compute_relative_joint_command",
]
