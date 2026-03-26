"""Validator-side runtime objects for generated hand assets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .asset_schema_core import AssetCfgBase
from .asset_schema_embodiment import HandCfg

HandRule = Callable[[HandCfg], None]


def reject_mimic_joints(hand: HandCfg) -> None:
    r"""Reject mimic joints in generator v1.

    Args:
        hand (HandCfg): Hand asset to inspect.

    Raises:
        NotImplementedError: If any joint uses `mimic`.
    """

    mimic_joints = [joint.name for joint in hand.iter_joints() if joint.mimic is not None]
    if mimic_joints:
        raise NotImplementedError(
            "Generator v1 does not support automatic mimic-hand generation; "
            f"found mimic joints: {mimic_joints}"
        )


@dataclass
class ValidatorCfg(AssetCfgBase):
    r"""Config for validator runtime objects."""

    class_type: type["Validator"] | None = None
    """Associated validator runtime class."""

    require_complete: bool = True
    """Whether unresolved required fields should fail validation."""

    reject_mimic: bool = True
    """Whether mimic joints should be rejected in v1."""

    rules: list[HandRule] = field(default_factory=list)
    """Additional validation rules."""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = Validator


class Validator:
    r"""Base runtime object for validating generated hands."""

    def __init__(self, cfg: ValidatorCfg):
        self.cfg = cfg

    def _run_builtin_checks(self, hand: HandCfg) -> None:
        if self.cfg.require_complete:
            missing = hand.validate()
            if missing:
                raise ValueError(f"HandCfg contains unresolved required fields: {missing}")
        if self.cfg.reject_mimic:
            reject_mimic_joints(hand)

    def validate(self, hand: HandCfg) -> HandCfg:
        self._run_builtin_checks(hand)
        for rule in self.cfg.rules:
            rule(hand)
        return hand


__all__ = ["HandRule", "ValidatorCfg", "Validator", "reject_mimic_joints"]
