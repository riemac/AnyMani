"""Top-level orchestration for hand-asset generation.

This module intentionally stays thin:

- `asset_schema_*` define the canonical declaration layer
- `asset_builders.py` assembles `HandCfg`
- `asset_validators.py` checks generated hands
- `asset_exporters.py` serializes generated hands

`asset_generator.py` coordinates these subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .asset_builders import HandBuilder, HandBuilderCfg
from .asset_exporters import Exporter, ExporterCfg
from .asset_schema_core import AssetCfgBase
from .asset_schema_embodiment import HandCfg
from .asset_validators import Validator, ValidatorCfg


@dataclass
class AssetGeneratorCfg(AssetCfgBase):
    r"""Top-level config for generator orchestration."""

    class_type: type["AssetGenerator"] | None = None
    """Associated asset-generator runtime class."""

    builder: HandBuilderCfg = field(default_factory=HandBuilderCfg)
    """Top-level hand builder config."""

    validator: ValidatorCfg = field(default_factory=ValidatorCfg)
    """Top-level validator config."""

    exporter: ExporterCfg = field(default_factory=ExporterCfg)
    """Top-level exporter config."""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = AssetGenerator


class AssetGenerator:
    r"""Top-level runtime object that builds, validates and exports hand assets."""

    cfg: AssetGeneratorCfg

    def __init__(self, cfg: AssetGeneratorCfg):
        self.cfg = cfg
        builder_type = cfg.builder.class_type or HandBuilder
        validator_type = cfg.validator.class_type or Validator
        exporter_type = cfg.exporter.class_type or Exporter
        self.builder = builder_type(cfg.builder)
        self.validator = validator_type(cfg.validator)
        self.exporter = exporter_type(cfg.exporter)

    def build(self) -> HandCfg:
        r"""Build one `HandCfg` from the configured builder."""

        return self.builder.build()

    def validate(self, hand: HandCfg) -> HandCfg:
        r"""Validate a built `HandCfg`."""

        return self.validator.validate(hand)

    def export(self, hand: HandCfg) -> dict[str, Any]:
        r"""Export a validated `HandCfg`."""

        return self.exporter.export(hand)

    def generate(self) -> HandCfg:
        r"""Run the full build -> validate -> export pipeline.

        Returns:
            HandCfg: The validated hand object.
        """

        hand = self.build()
        hand = self.validate(hand)
        _ = self.export(hand)
        return hand


__all__ = ["AssetGeneratorCfg", "AssetGenerator"]
