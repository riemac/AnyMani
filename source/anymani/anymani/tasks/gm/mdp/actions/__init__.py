r"""`gm` 任务的自定义动作空间模块。

本命名空间承载 generalized manipulation 中可跨 hand / task 复用的 action 原件。
当前只保留被LEAP/inhand对照消费的声明式ADR joint-position actions：

- `ADRRelativeJointPositionActionCfg`：relative raw-rad delta，支持 `reference="current"|"target"`；
- `ADREMAJointPositionToLimitsActionCfg`：joint-limit absolute target + EMA，支持同一 reference 枚举；
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

__all__ = [
    "ADREMAJointPositionToLimitsAction",
    "ADREMAJointPositionToLimitsActionCfg",
    "ADRJointAction",
    "ADRRelativeJointPositionAction",
    "ADRRelativeJointPositionActionCfg",
    "PolicyStepADRTargetJointPositionAction",
    "PolicyStepADRTargetJointPositionActionCfg",
    "compute_ema_joint_command",
    "compute_leap_adr_latency_steps",
    "compute_relative_joint_command",
]
