r"""向后兼容的 re-export 桩（已拆分到 generator/exporter/ 子包）。

本文件的原始内容已拆分到以下模块：

- ``generator/exporter/_base.py``         → ExportResult, ExporterBase
- ``generator/exporter/urdf_writer.py``   → UrdfWriterCfg, UrdfWriter
- ``generator/exporter/sidecar.py``       → SidecarCfg, SidecarExporter
- ``generator/exporter/hand_exporter.py`` → HandExporterCfg, HandExporter

此文件保留仅为不破坏已有代码的 import 路径。
旧接口 Exporter / JointExporter / FingerExporter / PalmExporter / HandExporter
均通过别名映射到新实现。
"""

from .exporter._base import ExportResult, ExporterBase
from .exporter.finger_exporter import FingerExporter, FingerExporterCfg
from .exporter.hand_exporter import HandExporter, HandExporterCfg
from .exporter.joint_exporter import JointExporter, JointExporterCfg
from .exporter.palm_exporter import PalmExporter, PalmExporterCfg
from .exporter.sidecar import SidecarCfg, SidecarExporter
from .exporter.urdf_writer import UrdfWriter, UrdfWriterCfg

# 旧接口别名（向后兼容）
Exporter = ExporterBase

__all__ = [
    "ExportResult",
    "ExporterBase",
    "Exporter",
    "JointExporterCfg",
    "FingerExporterCfg",
    "PalmExporterCfg",
    "UrdfWriterCfg",
    "UrdfWriter",
    "SidecarCfg",
    "SidecarExporter",
    "HandExporterCfg",
    "HandExporter",
    "JointExporter",
    "FingerExporter",
    "PalmExporter",
]
