r"""导出器子包：HandCfg → 落盘产物（URDF / Sidecar / Tree 文件）。

工具分层到独立模块：

- ``_base``         → ExportResult, ExporterBase（基础协议）
- ``urdf_writer``   → UrdfWriterCfg, UrdfWriter（HandCfg → URDF XML）
- ``joint_exporter``  → JointExporterCfg, JointExporter（JointCfg → standalone URDF）
- ``finger_exporter`` → FingerExporterCfg, FingerExporter（FingerCfg → standalone URDF）
- ``palm_exporter``   → PalmExporterCfg, PalmExporter（PalmCfg → standalone URDF）
- ``sidecar``       → SidecarCfg, SidecarExporter（HandCfg → 元数据 YAML）
- ``hand_exporter`` → HandExporterCfg, HandExporter（编排层，按 artifact_level 调度）

典型用法::

    from anymani.assets.generator.exporter import HandExporter, HandExporterCfg

    cfg = HandExporterCfg(artifact_level="bundle")
    result = HandExporter(cfg).export(generation_result, output_dir=Path("outputs/"))
    print("Written:", result.written)
"""

from ._base import ExportResult, ExporterBase
from .finger_exporter import FingerExporter, FingerExporterCfg
from .hand_exporter import HandExporter, HandExporterCfg
from .joint_exporter import JointExporter, JointExporterCfg
from .palm_exporter import PalmExporter, PalmExporterCfg
from .sidecar import SidecarCfg, SidecarExporter
from .urdf_writer import UrdfWriter, UrdfWriterCfg

__all__ = [
    # 基础
    "ExportResult",
    "ExporterBase",
    # URDF
    "UrdfWriterCfg",
    "UrdfWriter",
    # Layered preview exporters
    "JointExporterCfg",
    "JointExporter",
    "FingerExporterCfg",
    "FingerExporter",
    "PalmExporterCfg",
    "PalmExporter",
    # Sidecar
    "SidecarCfg",
    "SidecarExporter",
    # 编排层（主入口）
    "HandExporterCfg",
    "HandExporter",
]
