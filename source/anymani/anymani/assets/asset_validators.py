r"""向后兼容的 re-export 桩（已拆分到 generator/validator/ 子包）。

本文件的原始内容已拆分到以下模块：

- ``generator/validator/_base.py``        → ValidationResult, ValidatorBase
- ``generator/validator/joint_rules.py``  → JointValidatorCfg, JointValidator
- ``generator/validator/finger_rules.py`` → FingerValidatorCfg, FingerValidator
- ``generator/validator/hand_rules.py``   → HandValidatorCfg, HandValidator

此文件保留仅为不破坏已有代码的 import 路径。
"""

from .validator._base import ValidationResult, ValidatorBase
from .validator.finger_rules import FingerValidatorCfg, FingerValidator
from .validator.hand_rules import HandValidatorCfg, HandValidator
from .validator.joint_rules import JointValidatorCfg, JointValidator

# 旧接口别名（旧代码直接用 ValidatorCfg）
ValidatorCfg = ValidatorBase

__all__ = [
    "ValidationResult",
    "ValidatorBase",
    "ValidatorCfg",
    "JointValidatorCfg",
    "JointValidator",
    "FingerValidatorCfg",
    "FingerValidator",
    "HandValidatorCfg",
    "HandValidator",
]
