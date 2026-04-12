r"""整手级导出器的声明式配置类和运行时类。

hand-level exporter 负责正式产物：`URDF + metadata sidecar + debug 子资产`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..asset_base import AssetCfgBase
from ..asset_exporters import ExportBundle, Exporter, ExporterCfg


@dataclass
class HandExporterCfg(ExporterCfg):
    r"""手级导出器配置。"""

    class_type: type["HandExporter"] | None = None
    """关联的手级导出器类。"""

    primary_format: Literal["urdf"] = "urdf"
    """正式主产物格式。当前阶段只保留 `urdf`。"""

    write_metadata_sidecar: bool = True
    """是否输出 metadata sidecar。默认开启。"""

    write_debug_assets: bool = True
    """是否同步输出 joint/finger/palm 级 debug 子资产。默认开启。"""

    round_trip_check: bool = True
    """是否在导出后执行 round-trip 语义自检。默认开启。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandExporter


class HandExporter(Exporter):
    r"""手级导出器。"""

    cfg: HandExporterCfg

    def __init__(self, cfg: HandExporterCfg):
        self.cfg = cfg

    def export(self, target: AssetCfgBase) -> ExportBundle | None:
        r"""导出一个整手级资产对象。"""
        pass

        # TODO:算法之一（正式整手导出）
        # ────────────────────────────────────────
        # 输入
        #   `target` — 预期为 canonical `HandCfg`。
        #   可调参数：
        #     `primary_format="urdf"`
        #     `write_metadata_sidecar`
        #     `write_debug_assets`
        #     `round_trip_check`
        #
        # 输出：`ExportBundle`，其中：
        #   - `primary`      对应正式 URDF
        #   - `sidecars`     对应 metadata / provenance / stable joint order 等
        #   - `debug_assets` 对应 joint/finger/palm 级自包含导出件
        #
        # ── 主产物写出 ──
        #   1. 按 palm -> finger -> joint 的稳定顺序线性化 `HandCfg`。
        #   2. 保留 build/post 阶段已经确定的 frame 语义，不在 exporter 中偷偷重算。
        #
        # ── sidecar 写出 ──
        #   1. 记录 preset 来源、delete/regroup lineage、mount baseline、stable joint order。
        #   2. 为后续 graph / SDF / runtime adapter 提供可追踪元数据。
        #
        # ── debug 子资产 ──
        #   1. 若 `write_debug_assets=True`，则以后实现时同步导出 joint/finger/palm 级最小工件。
        #
        # ── round-trip 自检 ──
        #   1. 若 `round_trip_check=True`，则以后实现时检查：
        #      `HandCfg -> URDF(+metadata) -> reparse` 是否仍保持同一语义。
        #
        # IDEA：hand-level exporter 的目标不是“把 XML 写出来”而已，而是把 canonical hand contract 稳定固化。


__all__ = ["HandExporterCfg", "HandExporter"]