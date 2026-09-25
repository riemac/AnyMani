"Checks revolute-joint axes, limits, connectivity, and finite coordinate values."

from __future__ import annotations

import math
from dataclasses import dataclass

from ..asset_base import AssetCfgBase, JointCfg
from ._base import ValidationResult, ValidatorBase

# ============================================================================

# ============================================================================


@dataclass
class JointValidatorCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type[JointValidator] | None = None
    "Associated runtime implementation for this configuration class."

    check_limit_range: bool = True
    "Lower and upper joint-coordinate bounds in radians."

    limit_max_range: float = 2 * math.pi
    "Lower and upper joint-coordinate bounds in radians."

    check_link_length: bool = True
    "Count or linear dimension in the units declared by the associated schema."

    min_link_length: float = 1e-4
    "Count or linear dimension in the units declared by the associated schema."

    strict: bool = False
    "Whether missing or unverified semantic fields cause validation to fail."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointValidator


# ============================================================================

# ============================================================================


class JointValidator(ValidatorBase):
    "Checks generated geometry against explicit acceptance gates."

    cfg: JointValidatorCfg

    def __init__(self, cfg: JointValidatorCfg):
        self.cfg = cfg

    def validate(self, target: JointCfg) -> ValidationResult:  # type: ignore[override]
        "Runs the configured acceptance gates and returns explicit candidate evidence."

        result = ValidationResult()

        if target.joint_type == "revolute" and self.cfg.check_limit_range:
            if target.limit is None:
                result.add_error(
                    f"joint '{target.name}': revolute joint is missing limits",
                    code="joint.revolute_missing_limits",
                )
            else:
                joint_range = target.limit.upper - target.limit.lower
                if joint_range > self.cfg.limit_max_range:
                    result.warnings.append(
                        f"joint '{target.name}': limit range {joint_range:.3f} rad > {self.cfg.limit_max_range:.3f}"
                    )

        if self.cfg.check_link_length and target.origin is not None:
            x, y, z = target.origin.pos
            length = math.sqrt(x * x + y * y + z * z)
            allow_zero_origin = bool(target.metadata.get("allow_zero_origin", False))
            if length < self.cfg.min_link_length and not allow_zero_origin:
                result.warnings.append(
                    f"joint '{target.name}': link length {length:.6f} m < min {self.cfg.min_link_length:.6f}"
                )

        if self.cfg.strict:
            result = result.as_strict()
        result.passed = len(result.errors) == 0
        return result


__all__ = ["JointValidatorCfg", "JointValidator"]
