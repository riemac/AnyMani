"Writes stable asset identity, provenance, and typed geometry semantics to hand.yaml."

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_schema_geometry import derive_generated_geometry_semantics, geometry_semantics_to_dict
from ..handedness import handedness_contract
from ..validator._finger_length import measure_finger_axial_lengths
from ._base import ExporterBase, ExportResult

# ============================================================================

# ============================================================================


@dataclass
class SidecarCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type[SidecarExporter] | None = None
    "Associated runtime implementation for this configuration class."

    filename: str = "hand.yaml"
    "Stable semantic identifier preserved in the output metadata."

    include_provenance: bool = True
    "Whether to write source and generation identity fields to the sidecar."

    include_finger_stats: bool = True
    "Whether to include per-finger proposal and acceptance counts."

    experiment_tag: str | None = None
    "Human-readable tag attached to one generated run."

    overwrite: bool = True
    "Whether an existing output path may be replaced."

    include_geometry_semantics: bool = True
    "Whether to include the typed geometry contract in hand.yaml."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = SidecarExporter


# ============================================================================

# ============================================================================


class SidecarExporter(ExporterBase):
    "Writes a hand component in the stable URDF and sidecar contract."

    cfg: SidecarCfg

    def __init__(self, cfg: SidecarCfg):
        self.cfg = cfg

    def export(
        self,
        target: HandCfg,
        output_dir: Path,
        extra: dict[str, Any] | None = None,
    ) -> ExportResult:
        "Writes the declared hand component and its metadata to the output bundle."

        doc_extra = dict(extra or {})
        consumed_keys = set()
        doc: dict[str, Any] = {
            "id": doc_extra.get("id", "<unknown>"),
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            "name": target.name,
            "family": target.family,
            "handedness": target.handedness,
            "dof": target.dof_count,
            "finger_count": len(target.fingers),
        }
        expected_handedness_contract = handedness_contract(target=target.handedness)
        metadata_handedness_contract = target.metadata.get("handedness_contract")
        if target.handedness == "left" and metadata_handedness_contract != expected_handedness_contract:
            raise ValueError(
                "left HandCfg must carry a complete strict handedness_contract before sidecar export"
            )
        doc["handedness_contract"] = expected_handedness_contract
        consumed_keys.add("id")

        if self.cfg.include_finger_stats:
            fingers: list[dict[str, Any]] = []
            axial_lengths = {
                measurement.finger_name: measurement
                for measurement in measure_finger_axial_lengths(target)
            }
            for finger in target.fingers:
                measurement = axial_lengths.get(finger.name)
                fingers.append(
                    {
                        "name": finger.name,
                        "joint_count": len(finger.joints),
                        "revolute_dof": finger.dof_count,
                        "total_length_cm": None if measurement is None else round(measurement.axial_length * 100.0, 3),
                    }
                )
            doc["fingers"] = fingers

        if self.cfg.include_provenance:
            doc["provenance"] = {
                "recipe_hash": doc_extra.get("recipe_hash"),
                "seed": doc_extra.get("seed"),
                "experiment_tag": self.cfg.experiment_tag or doc_extra.get("experiment_tag"),
            }
            consumed_keys.update({"recipe_hash", "seed", "experiment_tag"})

        for key, value in doc_extra.items():
            if key not in consumed_keys:
                doc[key] = value

        if self.cfg.include_geometry_semantics:
            geometry_semantics = derive_generated_geometry_semantics(
                target,
                asset_id=str(doc["id"]),
                topology_key=None if doc.get("topology_name") is None else str(doc["topology_name"]),
            )
            doc["geometry_semantics"] = geometry_semantics_to_dict(geometry_semantics)



        doc["hand_cfg"] = target.to_dict()

        out_path = output_dir / self.cfg.filename
        if out_path.exists() and not self.cfg.overwrite:
            return ExportResult(skipped=[out_path])

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return ExportResult(written=[out_path])


__all__ = ["SidecarCfg", "SidecarExporter"]
