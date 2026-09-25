"Defines candidate rejection reasons and validator result records."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Result record with errors, warnings, and the scope of any geometry certificate.

    Clearance metadata states whether checks used the post-mutate home pose and collision geometry, which unsupported bodies were skipped, and what conclusions the result does not support.
    """

    passed: bool = True
    "Whether all required validation gates passed for this candidate."

    errors: list[str] = field(default_factory=list)
    "Explicit candidate rejection or validation errors."

    error_codes: list[str] = field(default_factory=list)
    "Stable validation and generation failure codes."

    warnings: list[str] = field(default_factory=list)
    "Non-fatal geometry or provenance issues returned by validation."

    metadata: dict[str, Any] = field(default_factory=dict)
    "Certificate scope, source posture, unsupported-body, and backend evidence."

    def add_error(self, message: str, *, code: str) -> None:
        "Adds a categorized validation failure to the result record."

        self.errors.append(str(message))
        self.error_codes.append(str(code))
        self.passed = False

    def merge(self, other: ValidationResult) -> ValidationResult:
        "Merges validation results without dropping rejection evidence."

        self.errors.extend(other.errors)
        self.error_codes.extend(other.error_codes)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)
        self.passed = len(self.errors) == 0
        return self

    def as_strict(self) -> ValidationResult:
        "Returns a validator config that rejects unsupported geometry explicitly."

        return ValidationResult(
            passed=len(self.errors) + len(self.warnings) == 0,
            errors=self.errors + self.warnings,
            error_codes=self.error_codes + ["strict.warning_promoted"] * len(self.warnings),
            warnings=[],
            metadata=dict(self.metadata),
        )

    def __bool__(self) -> bool:
        return self.passed


class ValidatorBase:
    "Shared acceptance interface for topology, joint, and geometry validators."

    def validate(self, target: object) -> ValidationResult:
        "Runs the configured acceptance gates and returns explicit candidate evidence."

        raise NotImplementedError


__all__ = ["ValidationResult", "ValidatorBase"]
