r"""验证器子包：层次化 HandCfg 软约束检查。

工具按验证层级分拆到独立模块：

- ``_base``         → ValidationResult, ValidatorBase（基础协议）
- ``joint_rules``   → JointValidatorCfg, JointValidator（关节级规则）
- ``finger_rules``  → FingerValidatorCfg, FingerValidator（手指级规则 + 关节嵌套）
- ``hand_rules``    → HandValidatorCfg, HandValidator（整手级规则 + 完整层次流水线）

典型用法::

    from anymani.assets.generator.validator import HandValidator, HandValidatorCfg

    cfg = HandValidatorCfg(dof_min=4, dof_max=20, strict=False)
    result = HandValidator(cfg).validate(my_hand_cfg)
    if not result:
        print("Validation failed:", result.errors)
    if result.warnings:
        print("Warnings:", result.warnings)
"""

from ._base import ValidationResult, ValidatorBase
from .finger_rules import FingerValidatorCfg, FingerValidator
from .hand_rules import HandValidatorCfg, HandValidator
from .joint_rules import JointValidatorCfg, JointValidator

__all__ = [
    # 基础
    "ValidationResult",
    "ValidatorBase",
    # 关节级
    "JointValidatorCfg",
    "JointValidator",
    # 手指级
    "FingerValidatorCfg",
    "FingerValidator",
    # 整手级（同时是流水线入口）
    "HandValidatorCfg",
    "HandValidator",
]
