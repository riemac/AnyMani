"Coordinates a full hand export and its sidecar metadata."

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from ..asset_base import AssetCfgBase
from ._base import ExporterBase, ExportResult
from .sidecar import SidecarCfg, SidecarExporter
from .urdf_writer import UrdfWriterCfg, UrdfWriter

if TYPE_CHECKING:


    from ..generator.hand_generator import HandGenerationResult


# ============================================================================

# ============================================================================


@dataclass
class HandExporterCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["HandExporter"] | None = None
    "Associated runtime implementation for this configuration class."

    artifact_level: Literal["hand_cfg", "urdf", "bundle"] = "bundle"
    "Whether the pipeline returns a hand configuration or writes a complete asset bundle."

    Urdf: UrdfWriterCfg = field(default_factory=UrdfWriterCfg)
    "URDF export stage configuration and output options."

    Sidecar: SidecarCfg = field(default_factory=SidecarCfg)
    "Stable semantic identifier preserved in the output metadata."

    export_tree_txt: bool = True
    "Whether to write a textual topology tree beside the URDF."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandExporter


# ============================================================================

# ============================================================================


class HandExporter(ExporterBase):
    "Writes a hand component in the stable URDF and sidecar contract."

    cfg: HandExporterCfg

    def __init__(self, cfg: HandExporterCfg):
        self.cfg = cfg

    def export(
        self,
        result: HandGenerationResult,
        output_dir: Path,
        sample_id: str | None = None,
        *,
        nest_sample_dir: bool = True,
        mesh_root_dir: Path | None = None,
    ) -> ExportResult:
        "Writes the declared hand component and its metadata to the output bundle."

        if self.cfg.artifact_level == "hand_cfg":
            return ExportResult()
        if result.hand_cfg is None:
            raise ValueError("HandExporter requires HandGenerationResult.hand_cfg")

        resolved_id = sample_id or str(result.metadata.get("id") or uuid4().hex[:8])
        result.metadata.setdefault("id", resolved_id)


        out_dir = output_dir / resolved_id if nest_sample_dir else output_dir
        combined = ExportResult()

        urdf_result = UrdfWriter(self.cfg.Urdf).export(
            result.hand_cfg,
            out_dir,
            mesh_root_dir=mesh_root_dir,
        )
        combined.merge(urdf_result)
        if urdf_result.written:
            result.urdf_path = urdf_result.written[0]

        sidecar_result = SidecarExporter(self.cfg.Sidecar).export(
            result.hand_cfg,
            out_dir,
            extra={**result.metadata, "id": resolved_id},
        )
        combined.merge(sidecar_result)
        if sidecar_result.written:
            result.sidecar_path = sidecar_result.written[0]

        if self.cfg.artifact_level == "bundle":
            result.render_trees()
            if self.cfg.export_tree_txt and result.tree_txt is not None:
                tree_txt = out_dir / "tree.txt"
                tree_txt.write_text(result.tree_txt, encoding="utf-8")
                combined.written.append(tree_txt)

        return combined


__all__ = ["HandExporterCfg", "HandExporter"]
