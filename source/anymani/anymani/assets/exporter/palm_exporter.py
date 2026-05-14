"""palm 级 quick-check exporter。

这个导出器服务的不是“整手闭环”，而是 palm preset 微调时最关键的问题：

1. palm 本体几何对不对
2. finger mounts 分布对不对
3. finger root 的朝向直觉对不对

因此 v1 默认预览语义采用用户确认后的方案：

- **palm 本体**
- **mount marker**
- **短 stub finger roots**

其中 marker 负责告诉你“挂载点在哪里”，stub root 负责告诉你“手指将沿哪个局部方向长出”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

from ..asset_base import AssetCfgBase, PalmCfg
from ..asset_schema_core import CollisionGeometryCfg, PoseCfg, Vector3, VisualGeometryCfg
from ._base import ExporterBase, ExportResult
from .urdf_writer import UrdfWriterCfg, _MeshExportState, _build_fixed_joint, _build_link_elem


@dataclass
class PalmExporterCfg(AssetCfgBase):
    r"""palm 级预览导出配置。"""

    class_type: type["PalmExporter"] | None = None
    Urdf: UrdfWriterCfg = field(default_factory=lambda: UrdfWriterCfg(filename="palm.urdf"))
    show_mount_markers: bool = True
    show_stub_roots: bool = True
    marker_radius: float = 0.003
    stub_size: Vector3 = (0.004, 0.018, 0.004)

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = PalmExporter


class PalmExporter(ExporterBase):
    r"""把单个 `PalmCfg` 导出为独立 URDF。"""

    cfg: PalmExporterCfg

    def __init__(self, cfg: PalmExporterCfg):
        self.cfg = cfg

    def export(self, target: PalmCfg, output_dir: Path) -> ExportResult:  # type: ignore[override]
        out_path = output_dir / self.cfg.Urdf.filename
        if out_path.exists() and not self.cfg.Urdf.overwrite:
            return ExportResult(skipped=[out_path])

        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_state = _MeshExportState(
            output_dir=output_dir,
            mesh_dirname=self.cfg.Urdf.canonical_mesh_dirname,
        )
        robot = ET.Element("robot", attrib={"name": target.name})
        robot.append(
            _build_link_elem(
                target.name,
                target.inertial,
                target.collisions,
                target.visuals,
                self.cfg.Urdf,
                mesh_state=mesh_state,
            )
        )

        for finger_name, mount in self._preview_mounts(target).items():
            preview_link_name = f"{finger_name}_mount_preview_link"
            preview_joint_name = f"{finger_name}_mount_preview_joint"
            robot.append(_build_fixed_joint(preview_joint_name, target.name, preview_link_name, mount))
            robot.append(self._build_mount_preview_link(preview_link_name, mesh_state=mesh_state))

        ET.indent(robot)
        ET.ElementTree(robot).write(out_path, encoding="unicode", xml_declaration=True)
        return ExportResult(written=[out_path, *mesh_state.written])

    def _preview_mounts(self, target: PalmCfg) -> dict[str, PoseCfg]:
        r"""从 `PalmCfg.metadata["finger_mounts"]` 读取 palm 级预览挂载点。"""

        if not isinstance(target.metadata, dict):
            return {}
        finger_mounts = target.metadata.get("finger_mounts")
        if not isinstance(finger_mounts, dict):
            return {}
        return {finger_name: PoseCfg.from_value(pose) for finger_name, pose in finger_mounts.items()}

    def _build_mount_preview_link(self, link_name: str, *, mesh_state: _MeshExportState) -> ET.Element:
        r"""构建一个同时包含 marker 与 stub-root 的预览 link。"""

        collisions: list[CollisionGeometryCfg] = []
        visuals: list[VisualGeometryCfg] = []

        if self.cfg.show_mount_markers:
            marker_geometry = {"type": "sphere", "radius": self.cfg.marker_radius}
            collisions.append(CollisionGeometryCfg(name=f"{link_name}_marker_collision", geometry=marker_geometry))
            visuals.append(VisualGeometryCfg(name=f"{link_name}_marker_visual", geometry=marker_geometry))

        if self.cfg.show_stub_roots:
            stub_origin = PoseCfg(pos=(0.0, self.cfg.stub_size[1] / 2.0, 0.0))
            stub_geometry = {"type": "box", "size": self.cfg.stub_size}
            collisions.append(
                CollisionGeometryCfg(
                    name=f"{link_name}_stub_collision",
                    geometry=stub_geometry,
                    origin=stub_origin,
                )
            )
            visuals.append(
                VisualGeometryCfg(
                    name=f"{link_name}_stub_visual",
                    geometry=stub_geometry,
                    origin=stub_origin,
                )
            )

        return _build_link_elem(link_name, None, collisions, visuals, self.cfg.Urdf, mesh_state=mesh_state)


__all__ = ["PalmExporterCfg", "PalmExporter"]
