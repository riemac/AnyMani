"Defines asset generator contracts used by hand geometry generation and validation."

from __future__ import annotations

from dataclasses import dataclass, field

from .asset_base import AssetCfgBase
from .asset_builders import HandBuilderCfg
from .asset_exporters import HandExporter
from .asset_validators import HandValidatorCfg


@dataclass
class AssetGeneratorCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["AssetGenerator"] | None = None
    "Associated runtime implementation for this configuration class."

    Build: HandBuilderCfg = field(default_factory=HandBuilderCfg)
    "Pre-made or post-mutate generation stage configuration."

    Validate: HandValidatorCfg = field(default_factory=HandValidatorCfg)
    "Optional structural or geometry acceptance gates for generated candidates."

    Export: type[HandExporter] = HandExporter
    "URDF and sidecar export configuration."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = AssetGenerator


class AssetGenerator:
    "Compatibility facade that dispatches to the modular HandGenerator pipeline."

    cfg: AssetGeneratorCfg

    def __init__(self, cfg: AssetGeneratorCfg):
        self.cfg = cfg

    def generate(self) -> None:
        "Runs the configured generation stages and records their results."

        raise NotImplementedError('AssetGenerator is a scaffold; its generation algorithm is not implemented.')


__all__ = ["AssetGeneratorCfg", "AssetGenerator"]
