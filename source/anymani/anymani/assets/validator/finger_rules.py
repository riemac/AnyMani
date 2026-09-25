"Checks finger count, chain completeness, spacing, and anatomical length constraints."

from __future__ import annotations

from dataclasses import dataclass, field

from ..asset_base import AssetCfgBase, FingerCfg
from ._base import ValidationResult, ValidatorBase
from .joint_rules import JointValidator, JointValidatorCfg

# ============================================================================

# ============================================================================


@dataclass
class FingerValidatorCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type[FingerValidator] | None = None
    "Associated runtime implementation for this configuration class."

    min_revolute_dof: int = 1
    "Minimum active revolute joints required for a surviving finger."

    max_joint_depth: int | None = None
    "Maximum active revolute depth allowed by the canonical finger schema."

    check_tip_uniqueness: bool = True
    "Whether this generation or validation condition is active."

    joint: JointValidatorCfg = field(default_factory=JointValidatorCfg)
    "Typed source joint or joint-level physical profile entry."

    strict: bool = False
    "Whether missing or unverified semantic fields cause validation to fail."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerValidator


# ============================================================================

# ============================================================================


class FingerValidator(ValidatorBase):
    "Checks generated geometry against explicit acceptance gates."

    cfg: FingerValidatorCfg

    def __init__(self, cfg: FingerValidatorCfg):
        self.cfg = cfg

    def validate(self, target: FingerCfg) -> ValidationResult:  # type: ignore[override]
        "Runs the configured acceptance gates and returns explicit candidate evidence."

        result = ValidationResult()
        joint_validator = JointValidator(self.cfg.joint)

        for joint in target.joints:
            result.merge(joint_validator.validate(joint))

        revolute_count = sum(1 for joint in target.joints if joint.joint_type == "revolute")
        if revolute_count < self.cfg.min_revolute_dof:
            result.add_error(
                f"finger '{target.name}': revolute dof {revolute_count} < min {self.cfg.min_revolute_dof}",
                code="finger.revolute_dof_below_min",
            )

        if self.cfg.max_joint_depth is not None and len(target.joints) > self.cfg.max_joint_depth:
            result.warnings.append(
                f"finger '{target.name}': depth {len(target.joints)} > max {self.cfg.max_joint_depth}"
            )

        if self.cfg.check_tip_uniqueness:
            tip_count = sum(1 for joint in target.joints if joint.is_tip)
            if tip_count != 1:
                result.warnings.append(
                    f"finger '{target.name}': expected exactly 1 tip joint, got {tip_count}"
                )

        if self.cfg.strict:
            result = result.as_strict()
        result.passed = len(result.errors) == 0
        return result


__all__ = ["FingerValidatorCfg", "FingerValidator"]
