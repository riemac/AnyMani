"""finger 级 quick-check exporter。

与 `JointExporter` 相比，这里要预览的是整根 finger 的串联骨架是否合理：

- 每一段 link 的局部几何
- 每个 joint 的 origin / axis / limit
- tip joint 是否接对

v1 仍遵循已确认的预览语义：

- **纯局部 URDF**
- **中性 stub base**

因此 `FingerCfg.mount` 不在这里展开；mount 关系留给 palm/hand 级 quick-check 观察。
"""

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
    r"""finger 级预览导出配置。"""

    class_type: type["FingerExporter"] | None = None
    Urdf: UrdfWriterCfg = field(default_factory=lambda: UrdfWriterCfg(filename="finger.urdf"))
    base_link_name: str = "finger_preview_base"
    base_box_size: Vector3 = (0.008, 0.008, 0.008)

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerExporter


class FingerExporter(ExporterBase):
    r"""把单根 `FingerCfg` 导出为独立 URDF。"""

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
        r"""构造 finger 局部预览用的中性基座。"""

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
