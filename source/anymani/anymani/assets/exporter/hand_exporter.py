r"""整手导出编排层：把 URDF / Sidecar / Tree 文件按 artifact_level 统一调度。

本模块是 `exporter/` 子包对外的唯一入口，对应 `资产生产概略.png` 中
两处 `Exporter` 节点（pre-made 后 / post-mutate 后均可调用）。

分类说明
--------

- **URDF**：主体仿真资产，必须符合 URDF 1.0 标准
- **Sidecar**：元数据与溯源 YAML，与 URDF 同级存放
- **Tree 文件**：可视化调试文件（tree.txt），由 `render_trees()` 生成

设计说明
--------

### artifact_level 的对应关系

从 `HandGeneratorCfg.artifact_level` 的三个值到导出行为：

+-------------------+------------------+------------------+-------------------+
| artifact_level    | URDF             | Sidecar          | Tree 文件          |
+===================+==================+==================+===================+
| ``hand_cfg``      | ✗                | ✗                | ✗                  |
+-------------------+------------------+------------------+-------------------+
| ``urdf``          | ✓                | ✓                | ✗                  |
+-------------------+------------------+------------------+-------------------+
| ``bundle``        | ✓                | ✓                | ✓                  |
+-------------------+------------------+------------------+-------------------+

``hand_cfg`` 时 `HandExporter` 直接返回空 `ExportResult`（不调用任何子导出器）。
``urdf`` 现在也会写 `hand.yaml`，因为独立 post-mutate 需要稳定的 `hand_cfg`
恢复入口，而不是再从 URDF 逆向解析。

### 目录结构约定

导出器本身不再根据 `mode` 猜目录语义，而是显式接受调用方传入的
`nest_sample_dir` 开关：

- `nest_sample_dir=False`：直接写到 `output_dir/`
- `nest_sample_dir=True`：写到 `output_dir/{sample_id}/`

因此当前两条主工作流分别对应：

.. code-block::

    # pre-made
    generated/<premade_timestamp>/<group>/<topology>/
    ├── hand.urdf
    ├── hand.yaml
    └── tree.txt

    # independent post-mutate
    generated/<premade_timestamp>/<group>/<topology>/<mutate_timestamp>/<sample_id>/
    ├── hand.urdf
    ├── hand.yaml
    └── tree.txt

`sample_id` 仍来自 `HandGenerationResult.metadata["id"]`；若无则退回 `uuid4`。
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
    ``hand_cfg`` 时直接返回空结果，``urdf`` 写 URDF + sidecar，``bundle`` 写全套。"""

    Urdf: UrdfWriterCfg = field(default_factory=UrdfWriterCfg)
    """URDF 写入器配置。"""

    Sidecar: SidecarCfg = field(default_factory=SidecarCfg)
    """Sidecar YAML 写入器配置。"""

    export_tree_txt: bool = True
    """是否在 ``bundle`` 模式下写出 ASCII 树状文件（tree.txt）。"""

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
        *,
        nest_sample_dir: bool = True,
        mesh_root_dir: Path | None = None,
    ) -> ExportResult:
        r"""把 `HandGenerationResult` 中的产物按 artifact_level 写出到磁盘。

        Args:
            result (HandGenerationResult): 待导出的生成结果包。
            output_dir (Path): 根落盘目录。
            sample_id (str | None): 当前样本 ID；为 ``None`` 时从 result.metadata 或 uuid4 获取。
            nest_sample_dir (bool): 是否在 `output_dir` 下再补一层 `sample_id/`。
            mesh_root_dir (Path | None): 当前导出边界共享的 mesh 根目录。为 `None` 时，
                退化为每个样本目录下自带 `meshes/`。

        Returns:
            ExportResult: 含所有写入/跳过/错误路径的合并结果。
        """

        if self.cfg.artifact_level == "hand_cfg":
            return ExportResult()
        if result.hand_cfg is None:
            raise ValueError("HandExporter requires HandGenerationResult.hand_cfg")

        resolved_id = sample_id or str(result.metadata.get("id") or uuid4().hex[:8])  # 逻辑样本 ID 仍稳定写入 sidecar metadata
        result.metadata.setdefault("id", resolved_id)  # 即使 pre-made 直写 topology 根，也保留 sidecar 顶层 `id`

        # pre-made 与 mutate-only 的目录差异在调用方决定；导出器只机械执行“是否再套一层样本目录”。
        out_dir = output_dir / resolved_id if nest_sample_dir else output_dir
        combined = ExportResult()

        urdf_result = UrdfWriter(self.cfg.Urdf).export(
            result.hand_cfg,
            out_dir,
            mesh_root_dir=mesh_root_dir,
        )
        combined.merge(urdf_result)
        if urdf_result.written:
            result.urdf_path = urdf_result.written[0]

        sidecar_result = SidecarExporter(self.cfg.Sidecar).export(
            result.hand_cfg,
            out_dir,
            extra={**result.metadata, "id": resolved_id},
        )
        combined.merge(sidecar_result)
        if sidecar_result.written:
            result.sidecar_path = sidecar_result.written[0]

        if self.cfg.artifact_level == "bundle":
            result.render_trees()
            if self.cfg.export_tree_txt and result.tree_txt is not None:
                tree_txt = out_dir / "tree.txt"
                tree_txt.write_text(result.tree_txt, encoding="utf-8")
                combined.written.append(tree_txt)

        return combined


__all__ = ["HandExporterCfg", "HandExporter"]
