r"""手部资产导出层的顶层抽象合同。

本文件只保留共享导出对象与公共基类，真正的 joint / finger / palm / hand
级导出骨架，下沉到 `exporter/` 子目录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .asset_base import AssetCfgBase


@dataclass
class ExportArtifact(AssetCfgBase):
    r"""单个导出产物的声明式描述。"""

    role: str = "artifact"
    """产物角色，例如 `primary_urdf` / `metadata` / `debug_joint`。"""

    path: str = ""
    """产物写出后的相对或绝对路径。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """附带的补充信息，如 lineage、target name、round-trip 标记。"""


@dataclass
class ExportBundle(AssetCfgBase):
    r"""一次导出操作的返回包。"""

    primary: ExportArtifact | None = None
    """主产物，例如整手正式 URDF。"""

    sidecars: list[ExportArtifact] = field(default_factory=list)
    """sidecar 产物，例如 metadata / 索引文件 / 配置摘要。"""

    debug_assets: list[ExportArtifact] = field(default_factory=list)
    """调试向导出件，例如 joint/finger/palm 级自包含资产。"""


@dataclass
class ExporterCfg(AssetCfgBase):
    r"""导出器配置基类。"""

    class_type: type["Exporter"] | None = None
    """关联的导出器运行时类。"""

    output_dir: str = "outputs/assets"
    """导出根目录。默认放在相对工作区可追踪的产物目录下。"""


class Exporter:
    r"""导出器基类。

    `Exporter` 只负责表达“从 canonical asset 导出外部工件”的职责，不在
    这里写任何具体格式细节。
    """

    cfg: ExporterCfg

    def __init__(self, cfg: ExporterCfg):
        self.cfg = cfg

    def export(self, target: AssetCfgBase) -> ExportBundle | None:
        r"""导出资产对象。

        Args:
            target (AssetCfgBase): 待导出的资产对象。

        Returns:
            ExportBundle | None: 当前阶段只保留产物合同，不写正式导出逻辑。
        """
        pass

        # TODO:算法之一（通用导出合同）
        # ────────────────────────────────────────
        # 输入：canonical asset object + exporter cfg。
        # 输出：`ExportBundle`，其中包含 primary / sidecar / debug 三类工件。
        #
        # ── 导出前规范化 ──
        #   1. 统一 target name、输出目录与 lineage tag。
        #   2. 决定这次导出是“正式产物”还是“局部 debug 子资产”。
        #
        # ── 工件组织 ──
        #   1. `primary` 负责正式消费入口（如整手 URDF）。
        #   2. `sidecars` 负责 metadata、索引和 provenance。
        #   3. `debug_assets` 负责局部可视化与 round-trip 调试。
        #
        # IDEA：顶层 `Exporter` 不直接决定 XML/mesh/yaml 的写法；这些细节由下沉到 `exporter/` 的层级文件承载。


from .exporter.finger_exporters import FingerExporter, FingerExporterCfg
from .exporter.hand_exporters import HandExporter, HandExporterCfg
from .exporter.joint_exporters import JointExporter, JointExporterCfg
from .exporter.palm_exporters import PalmExporter, PalmExporterCfg


__all__ = [
    "ExportArtifact",
    "ExportBundle",
    "ExporterCfg",
    "Exporter",
    "JointExporterCfg",
    "JointExporter",
    "FingerExporterCfg",
    "FingerExporter",
    "PalmExporterCfg",
    "PalmExporter",
    "HandExporterCfg",
    "HandExporter",
]
