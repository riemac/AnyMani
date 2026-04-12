r"""手指级导出器的声明式配置类和运行时类。"""

from __future__ import annotations

from dataclasses import dataclass

from ..asset_base import AssetCfgBase
from ..asset_exporters import ExportBundle, Exporter, ExporterCfg


@dataclass
class FingerExporterCfg(ExporterCfg):
    r"""手指级导出器配置。"""

    class_type: type["FingerExporter"] | None = None
    """关联的手指级导出器类。"""

    write_self_contained_urdf: bool = True
    """是否导出 finger 级自包含 URDF。默认导出。"""

    write_metadata_sidecar: bool = True
    """是否写出 finger 级 metadata。默认开启。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerExporter


class FingerExporter(Exporter):
    r"""手指级导出器。"""

    cfg: FingerExporterCfg

    def __init__(self, cfg: FingerExporterCfg):
        self.cfg = cfg

    def export(self, target: AssetCfgBase) -> ExportBundle | None:
        r"""导出一个手指级资产对象。"""
        pass

        # TODO:算法之一（finger-level debug 导出）
        # ────────────────────────────────────────
        # 输入：预期为 canonical `FingerCfg`。
        # 输出：`ExportBundle`。
        #
        # ── 主产物 ──
        #   1. 导出 finger 级最小 URDF，以便快速核对 tip、joint chain 与 mount 语义。
        #
        # ── sidecar ──
        #   1. 记录 joint 顺序、tip link、lineage 与 preset 来源。
        #
        # IDEA：finger-level exporter 是 delete/regroup 调试最直接的观察窗口。


__all__ = ["FingerExporterCfg", "FingerExporter"]