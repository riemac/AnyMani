"""Exporter-side runtime objects for generated hand assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_schema_core import AssetCfgBase
from .asset_schema_embodiment import HandCfg


@dataclass
class ExporterCfg(AssetCfgBase):
    r"""Config for exporter runtime objects."""

    class_type: type["Exporter"] | None = None
    """Associated exporter runtime class."""

    output_dir: str | None = None
    """Optional output directory."""

    dump_json: bool = False
    """Whether to dump a JSON snapshot of the `HandCfg`."""

    json_file_name: str = "hand.json"
    """Output file name used when `dump_json=True`."""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = Exporter


class Exporter:
    r"""Base runtime object for serializing generated hands."""

    def __init__(self, cfg: ExporterCfg):
        self.cfg = cfg

    def serialize(self, hand: HandCfg) -> dict[str, Any]:
        r"""Serialize a hand into a plain dictionary."""

        return hand.to_dict()

    def export(self, hand: HandCfg) -> dict[str, Any]:
        r"""Export the hand according to exporter configuration.

        v1 keeps exporter behavior intentionally modest:
        it primarily serializes `HandCfg`, and can optionally dump a JSON
        debug artifact for inspection.
        """

        payload = self.serialize(hand)
        if self.cfg.dump_json and self.cfg.output_dir is not None:
            output_dir = Path(self.cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / self.cfg.json_file_name
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload


__all__ = ["ExporterCfg", "Exporter"]
