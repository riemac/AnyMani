r"""关节级导出器的声明式配置类和运行时类。

关节级导出器主要服务调试：它不追求完整 hand asset，而追求“单个 joint 是否被正确理解”。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..asset_base import AssetCfgBase
from ..asset_exporters import ExportBundle, Exporter, ExporterCfg


@dataclass
class JointExporterCfg(ExporterCfg):
    r"""关节级导出器配置。"""

    class_type: type["JointExporter"] | None = None
    """关联的关节级导出器类。"""

    write_self_contained_urdf: bool = True
    """是否导出单关节级自包含 URDF。默认导出。"""

    write_metadata_sidecar: bool = True
    """是否同步输出 joint 级 metadata。默认输出。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointExporter


class JointExporter(Exporter):
    r"""关节级导出器。"""

    cfg: JointExporterCfg

    def __init__(self, cfg: JointExporterCfg):
        self.cfg = cfg

    def export(self, target: AssetCfgBase) -> ExportBundle | None:
        r"""导出一个关节级资产对象。"""
        pass

        # TODO:算法之一（joint-level debug 导出）
        # ────────────────────────────────────────
        # 输入：预期为 canonical `JointCfg`。
        # 输出：`ExportBundle`。
        #
        # ── 主产物 ──
        #   1. 若 `write_self_contained_urdf=True`，则以后实现时生成最小 joint-level URDF。
        #
        # ── sidecar ──
        #   1. 记录 role、axis、limit、child-link geometry 摘要。
        #
        # IDEA：joint-level exporter 的目标是帮助核对局部 frame 与几何语义，而不是替代 hand-level 正式导出。


__all__ = ["JointExporterCfg", "JointExporter"]