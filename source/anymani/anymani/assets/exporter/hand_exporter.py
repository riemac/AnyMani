r"""整手导出编排层：把 URDF / Sidecar / Tree 文件按 artifact_level 统一调度。

本模块是 `exporter/` 子包对外的唯一入口，对应 `资产生产概略.png` 中
两处 `Exporter` 节点（pre-made 后 / post-mutate 后均可调用）。

分类说明
--------

- **URDF**：主体仿真资产，必须符合 URDF 1.0 标准
- **Sidecar**：元数据与溯源 YAML，与 URDF 同级存放
- **Tree 文件**：可视化调试文件（tree.txt / tree.mmd），由 `render_trees()` 生成

设计说明
--------

### artifact_level 的对应关系

从 `HandGeneratorCfg.artifact_level` 的三个值到导出行为：

+-------------------+------------------+------------------+-------------------+
| artifact_level    | URDF             | Sidecar          | Tree 文件          |
+===================+==================+==================+===================+
| ``hand_cfg``      | ✗                | ✗                | ✗                  |
+-------------------+------------------+------------------+-------------------+
| ``urdf``          | ✓                | ✗                | ✗                  |
+-------------------+------------------+------------------+-------------------+
| ``bundle``        | ✓                | ✓                | ✓                  |
+-------------------+------------------+------------------+-------------------+

``hand_cfg`` 时 `HandExporter` 直接返回空 `ExportResult`（不调用任何子导出器）。

### 目录结构约定

每个产物放在 ``output_dir / {sample_id}/`` 下：

.. code-block::

    outputs/
    └── a3f2c0b1/
        ├── hand.urdf
        ├── hand.yaml     (sidecar)
        ├── tree.txt
        └── tree.mmd

``sample_id`` 来自 ``HandGenerationResult.metadata["id"]``；若无则用 ``uuid4``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from ..asset_base import AssetCfgBase, HandCfg
from ._base import ExporterBase, ExportResult
from .sidecar import SidecarCfg, SidecarExporter
from .urdf_writer import UrdfWriterCfg, UrdfWriter

if TYPE_CHECKING:
    # 避免循环导入；HandGenerationResult 仅在类型注解中使用，
    # 在运行时由调用方传入，不需要在此模块 __init__ 时 import。
    from ..generator.hand_generator import HandGenerationResult


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class HandExporterCfg(AssetCfgBase):
    r"""整手导出编排层配置。"""

    class_type: type["HandExporter"] | None = None
    """关联的运行时类。"""

    artifact_level: Literal["hand_cfg", "urdf", "bundle"] = "bundle"
    """产物粒度；与 ``HandGeneratorCfg.artifact_level`` 保持同步。
    ``hand_cfg`` 时直接返回空结果，``urdf`` 只写 URDF，``bundle`` 写全套。"""

    Urdf: UrdfWriterCfg = field(default_factory=UrdfWriterCfg)
    """URDF 写入器配置。"""

    Sidecar: SidecarCfg = field(default_factory=SidecarCfg)
    """Sidecar YAML 写入器配置。"""

    export_tree_txt: bool = True
    """是否在 ``bundle`` 模式下写出 ASCII 树状文件（tree.txt）。"""

    export_tree_mermaid: bool = True
    """是否在 ``bundle`` 模式下写出 Mermaid 树状文件（tree.mmd）。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandExporter


# ============================================================================
#  运行时壳
# ============================================================================


class HandExporter(ExporterBase):
    r"""整手导出编排器。

    根据 ``cfg.artifact_level`` 决定调用哪些子导出器，并把结果合并到
    ``HandGenerationResult`` 的对应字段。
    """

    cfg: HandExporterCfg

    def __init__(self, cfg: HandExporterCfg):
        self.cfg = cfg

    def export(
        self,
        result: HandGenerationResult,
        output_dir: Path,
        sample_id: str | None = None,
    ) -> ExportResult:
        r"""把 `HandGenerationResult` 中的产物按 artifact_level 写出到磁盘。

        Args:
            result (HandGenerationResult): 待导出的生成结果包。
            output_dir (Path): 根落盘目录；产物放在 ``output_dir / sample_id /`` 下。
            sample_id (str | None): 当前样本 ID；为 ``None`` 时从 result.metadata 或 uuid4 获取。

        Returns:
            ExportResult: 含所有写入/跳过/错误路径的合并结果。
        """

        if self.cfg.artifact_level == "hand_cfg":
            return ExportResult()
        if result.hand_cfg is None:
            raise ValueError("HandExporter requires HandGenerationResult.hand_cfg")

        resolved_id = sample_id or str(result.metadata.get("id") or uuid4().hex[:8])
        result.metadata.setdefault("id", resolved_id)
        out_dir = output_dir / resolved_id
        combined = ExportResult()

        urdf_result = UrdfWriter(self.cfg.Urdf).export(result.hand_cfg, out_dir)
        combined.merge(urdf_result)
        if urdf_result.written:
            result.urdf_path = urdf_result.written[0]

        if self.cfg.artifact_level == "bundle":
            sidecar_result = SidecarExporter(self.cfg.Sidecar).export(
                result.hand_cfg,
                out_dir,
                extra={**result.metadata, "id": resolved_id},
            )
            combined.merge(sidecar_result)
            if sidecar_result.written:
                result.sidecar_path = sidecar_result.written[0]

            result.render_trees()
            if self.cfg.export_tree_txt and result.tree_txt is not None:
                tree_txt = out_dir / "tree.txt"
                tree_txt.write_text(result.tree_txt, encoding="utf-8")
                combined.written.append(tree_txt)
            if self.cfg.export_tree_mermaid and result.tree_mermaid is not None:
                tree_mmd = out_dir / "tree.mmd"
                tree_mmd.write_text(result.tree_mermaid, encoding="utf-8")
                combined.written.append(tree_mmd)

        return combined

        # TODO:算法之一（artifact_level-aware orchestration）
        # ────────────────────────────────────────
        # 输入
        #   result: HandGenerationResult（含 hand_cfg / metadata）
        #   output_dir: Path
        #   sample_id: str | None
        #   cfg.artifact_level: "hand_cfg" | "urdf" | "bundle"
        #
        # 输出：ExportResult
        #
        # ── 快速路径：hand_cfg 模式 ──
        #   if cfg.artifact_level == "hand_cfg":
        #     return ExportResult()
        #
        # ── sample_id 确定 ──
        #   if sample_id is None:
        #     sample_id = result.metadata.get("id") or uuid4().hex[:8]
        #   out_dir = output_dir / sample_id
        #
        # ── 总结果容器 ──
        #   combined = ExportResult()
        #   hand = result.hand_cfg
        #
        # ── URDF 导出（urdf + bundle 均需要）──
        #   urdf_result = UrdfWriter(cfg.Urdf).export(hand, out_dir)
        #   combined.merge(urdf_result)
        #   if urdf_result.written:
        #     result.urdf_path = urdf_result.written[0]
        #
        # ── Sidecar + Tree（仅 bundle 模式）──
        #   if cfg.artifact_level == "bundle":
        #     extra = {**result.metadata, "id": sample_id}
        #     sidecar_result = SidecarExporter(cfg.Sidecar).export(hand, out_dir, extra)
        #     combined.merge(sidecar_result)
        #     if sidecar_result.written:
        #       result.sidecar_path = sidecar_result.written[0]
        #
        #     result.render_trees()   # 确保 tree_txt / tree_mermaid 已生成
        #
        #     if cfg.export_tree_txt and result.tree_txt:
        #       tree_path = out_dir / "tree.txt"
        #       tree_path.write_text(result.tree_txt)
        #       combined.written.append(tree_path)
        #
        #     if cfg.export_tree_mermaid and result.tree_mermaid:
        #       mmd_path = out_dir / "tree.mmd"
        #       mmd_path.write_text(result.tree_mermaid)
        #       combined.written.append(mmd_path)
        #
        # return combined
        #
        # IDEA：HandExporter 的价值是"让调用方不需要知道哪些模式下产生哪些文件"；
        # 所有 artifact_level 的决策都在这里集中处理，GeneratorRunner 的 _save_result()
        # 只需调用 HandExporter.export(result, output_dir) 即可。


__all__ = ["HandExporterCfg", "HandExporter"]
