"Defines asset validators contracts used by hand geometry generation and validation."

from .validator._base import ValidationResult, ValidatorBase
from .validator.finger_rules import FingerValidatorCfg, FingerValidator
from .validator.hand_rules import HandValidatorCfg, HandValidator
from .validator.joint_rules import JointValidatorCfg, JointValidator


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
