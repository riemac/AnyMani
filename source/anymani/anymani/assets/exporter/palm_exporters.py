r"""掌级导出器的声明式配置类和运行时类。"""

from __future__ import annotations

from dataclasses import dataclass

from ..asset_base import AssetCfgBase
from ..asset_exporters import ExportBundle, Exporter, ExporterCfg


@dataclass
class PalmExporterCfg(ExporterCfg):
    r"""掌级导出器配置。"""

    class_type: type["PalmExporter"] | None = None
    """关联的掌级导出器类。"""

    write_self_contained_urdf: bool = True
    """是否导出 palm 级自包含 URDF。默认导出。"""

    write_metadata_sidecar: bool = True
    """是否写出 palm 级 metadata。默认开启。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = PalmExporter


class PalmExporter(Exporter):
    r"""掌级导出器。"""

    cfg: PalmExporterCfg

    def __init__(self, cfg: PalmExporterCfg):
        self.cfg = cfg

    def export(self, target: AssetCfgBase) -> ExportBundle | None:
        r"""导出一个掌级资产对象。"""
        pass

        # TODO:算法之一（palm-level debug 导出）
        # ────────────────────────────────────────
        # 输入：预期为 canonical `PalmCfg`。
        # 输出：`ExportBundle`。
        #
        # ── 主产物 ──
        #   1. 导出 palm 级最小 URDF，便于核对 compound collision 与 design frame。
        #
        # ── sidecar ──
        #   1. 记录 mount baseline、preset 名称与 collision-first 摘要。
        #
        # IDEA：掌级 exporter 的核心价值在于让 hand-level mount 问题可被局部化检查。


__all__ = ["PalmExporterCfg", "PalmExporter"]