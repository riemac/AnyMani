"""joint 级 quick-check exporter。

这个导出器的目标不是替代正式整手导出，而是给科研调参提供一个更短的反馈回路：
把单个 `JointCfg` 独立写成可直接打开的 URDF，快速看局部 link 几何、joint 轴和
origin 是否合理。

v1 预览语义采用用户已确认的方案：

- **纯局部 URDF**
- **中性 stub base**

也就是说，这里不会试图恢复 hand/palm 上下文，只保留单 joint 本体。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

from ..asset_base import AssetCfgBase, JointCfg
from ..asset_schema_core import CollisionGeometryCfg, Vector3, VisualGeometryCfg
from ._base import ExporterBase, ExportResult
from .urdf_writer import UrdfWriterCfg, _build_joint_elem, _build_link_elem


@dataclass
class JointExporterCfg(AssetCfgBase):
    r"""joint 级预览导出配置。"""

    class_type: type["JointExporter"] | None = None
    Urdf: UrdfWriterCfg = field(default_factory=lambda: UrdfWriterCfg(filename="joint.urdf"))
    base_link_name: str = "joint_preview_base"
    base_box_size: Vector3 = (0.008, 0.008, 0.008)

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointExporter


class JointExporter(ExporterBase):
    r"""把单个 `JointCfg` 导出为独立 URDF。"""

    cfg: JointExporterCfg

    def __init__(self, cfg: JointExporterCfg):
        self.cfg = cfg

    def export(self, target: JointCfg, output_dir: Path) -> ExportResult:  # type: ignore[override]
        out_path = output_dir / self.cfg.Urdf.filename
        if out_path.exists() and not self.cfg.Urdf.overwrite:
            return ExportResult(skipped=[out_path])

        output_dir.mkdir(parents=True, exist_ok=True)
        robot = ET.Element("robot", attrib={"name": target.name})
        robot.append(self._build_base_link())
        robot.append(_build_joint_elem(target, self.cfg.base_link_name, self.cfg.Urdf))
        robot.append(_build_link_elem(target.child, target.inertial, target.collisions, target.visuals, self.cfg.Urdf))
        ET.indent(robot)
        ET.ElementTree(robot).write(out_path, encoding="unicode", xml_declaration=True)
        return ExportResult(written=[out_path])

    def _build_base_link(self) -> ET.Element:
        r"""构造一个可见但语义中性的 stub base。"""

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
        return _build_link_elem(self.cfg.base_link_name, None, collisions, visuals, self.cfg.Urdf)


__all__ = ["JointExporterCfg", "JointExporter"]
