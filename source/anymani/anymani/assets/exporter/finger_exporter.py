"Serializes finger links and joints into the hand URDF tree."

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

from ..asset_base import AssetCfgBase, FingerCfg
from ..asset_schema_core import CollisionGeometryCfg, Vector3, VisualGeometryCfg
from ._base import ExporterBase, ExportResult
from .urdf_writer import UrdfWriterCfg, _MeshExportState, _build_joint_elem, _build_link_elem


@dataclass
class FingerExporterCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["FingerExporter"] | None = None
    Urdf: UrdfWriterCfg = field(default_factory=lambda: UrdfWriterCfg(filename="finger.urdf"))
    base_link_name: str = "finger_preview_base"
    base_box_size: Vector3 = (0.008, 0.008, 0.008)

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerExporter


class FingerExporter(ExporterBase):
    "Writes a hand component in the stable URDF and sidecar contract."

    cfg: FingerExporterCfg

    def __init__(self, cfg: FingerExporterCfg):
        self.cfg = cfg

    def export(self, target: FingerCfg, output_dir: Path) -> ExportResult:  # type: ignore[override]
        out_path = output_dir / self.cfg.Urdf.filename
        if out_path.exists() and not self.cfg.Urdf.overwrite:
            return ExportResult(skipped=[out_path])

        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_state = _MeshExportState(
            output_dir=output_dir,
            mesh_dirname=self.cfg.Urdf.canonical_mesh_dirname,
        )
        robot = ET.Element("robot", attrib={"name": target.name})
        robot.append(self._build_base_link(mesh_state=mesh_state))

        parent_name = self.cfg.base_link_name
        for joint in target.joints:
            robot.append(_build_joint_elem(joint, parent_name, self.cfg.Urdf))
            robot.append(
                _build_link_elem(
                    joint.child,
                    joint.inertial,
                    joint.collisions,
                    joint.visuals,
                    self.cfg.Urdf,
                    mesh_state=mesh_state,
                )
            )
            parent_name = joint.child

        ET.indent(robot)
        ET.ElementTree(robot).write(out_path, encoding="unicode", xml_declaration=True)
        return ExportResult(written=[out_path, *mesh_state.written])

    def _build_base_link(self, *, mesh_state: _MeshExportState) -> ET.Element:

        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.base_link_name}_collision",
                geometry={"type": "box", "size": self.cfg.base_box_size},
            )
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.base_link_name}_visual",
                geometry={"type": "box", "size": self.cfg.base_box_size},
            )
        ]
        return _build_link_elem(
            self.cfg.base_link_name,
            None,
            collisions,
            visuals,
            self.cfg.Urdf,
            mesh_state=mesh_state,
        )


__all__ = ["FingerExporterCfg", "FingerExporter"]
