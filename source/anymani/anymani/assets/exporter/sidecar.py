r"""Sidecar 导出器：把 HandCfg 的元数据与溯源信息写为 YAML 文件。

Sidecar 是每个导出产物目录里与 URDF 同级的一个轻量 YAML 文件，记录：

- 整手结构摘要（family / handedness / dof / finger_count）
- 生成溯源（generation config hash / timestamp / random seed）
- 关键参数统计（每根 finger 的总链长、关节限位范围）
- 可追踪字段（id / experiment tag）

设计说明
--------

### 溯源的意义

批量生成 N 个手资产之后，需要能从任意一个 URDF 文件反向追溯到它的
生成参数（pre-made preset / mutate 配置 / 随机种子）。Sidecar 是这条追踪链的
纸面记录，而不是把溯源嵌进 URDF 注释里（那会污染标准格式）。

### 与 RecipeLoader.dump() 的关系

`RecipeLoader.dump()` 序列化的是生成器的**配置**（HandGeneratorCfg）；
`SidecarExporter` 记录的是这次生成的**产物描述**（HandCfg 结构摘要 + 上下文）。
两者互补——前者让你知道"用什么设置生成的"，后者让你知道"生成出了什么"。

典型 Sidecar 输出
-----------------

.. code-block:: yaml

    id: a3f2c0b1
    timestamp: "2026-04-12T14:30:00"
    family: leap
    handedness: right
    dof: 16
    finger_count: 4
    fingers:
      - name: index
        joint_count: 4
        revolute_dof: 4
        total_length_cm: 8.42
      - ...
    provenance:
      recipe_hash: "d4e5f6..."
      seed: 42
      experiment_tag: leap_variant_v1
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import math
from pathlib import Path
from typing import Any
import yaml

from ..asset_base import AssetCfgBase, HandCfg
from ._base import ExporterBase, ExportResult


# ============================================================================
#  配置类
# ============================================================================


@dataclass
class SidecarCfg(AssetCfgBase):
    r"""Sidecar YAML 导出器配置。"""

    class_type: type["SidecarExporter"] | None = None
    """关联的运行时类。"""

    filename: str = "hand.yaml"
    """Sidecar 输出文件名；相对于传入的 output_dir。"""

    include_provenance: bool = True
    """是否在 sidecar 里写入溯源字段（recipe_hash / seed / experiment_tag）。"""

    include_finger_stats: bool = True
    """是否在 sidecar 里写入每根 finger 的统计信息（链长 / DOF 等）。"""

    experiment_tag: str | None = None
    """可选的实验标签；若不为 None 则写入 provenance.experiment_tag。"""

    overwrite: bool = True
    """若目标文件已存在，是否覆盖。``False`` 时记入 skipped 并跳过。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = SidecarExporter


# ============================================================================
#  运行时壳
# ============================================================================


class SidecarExporter(ExporterBase):
    r"""Sidecar YAML 写入器。

    把 `HandCfg` 的结构摘要和溯源信息写出为轻量 YAML 文件，与 URDF 同目录存放。
    """

    cfg: SidecarCfg

    def __init__(self, cfg: SidecarCfg):
        self.cfg = cfg

    def export(
        self,
        target: HandCfg,
        output_dir: Path,
        extra: dict[str, Any] | None = None,
    ) -> ExportResult:
        r"""把 `HandCfg` 的元数据写出为 Sidecar YAML。

        Args:
            target (HandCfg): 待导出的整手配置。
            output_dir (Path): 产物落盘目录；不存在时自动创建。
            extra (dict | None): 调用方传入的额外字段（如 id / seed / recipe_hash）；
                与 Sidecar 结构合并写入。

        Returns:
            ExportResult: 含写入路径或错误信息的结果包。
        """

        doc_extra = dict(extra or {})
        consumed_keys = set()
        doc: dict[str, Any] = {
            "id": doc_extra.get("id", "<unknown>"),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "name": target.name,
            "family": target.family,
            "handedness": target.handedness,
            "dof": target.dof_count,
            "finger_count": len(target.fingers),
        }
        consumed_keys.add("id")

        if self.cfg.include_finger_stats:
            fingers: list[dict[str, Any]] = []
            for finger in target.fingers:
                total_length = sum(
                    math.sqrt(joint.origin.pos[0] ** 2 + joint.origin.pos[1] ** 2 + joint.origin.pos[2] ** 2)
                    for joint in finger.joints
                )
                fingers.append(
                    {
                        "name": finger.name,
                        "joint_count": len(finger.joints),
                        "revolute_dof": finger.dof_count,
                        "total_length_cm": round(total_length * 100.0, 3),
                    }
                )
            doc["fingers"] = fingers

        if self.cfg.include_provenance:
            doc["provenance"] = {
                "recipe_hash": doc_extra.get("recipe_hash"),
                "seed": doc_extra.get("seed"),
                "experiment_tag": self.cfg.experiment_tag or doc_extra.get("experiment_tag"),
            }
            consumed_keys.update({"recipe_hash", "seed", "experiment_tag"})

        for key, value in doc_extra.items():
            if key not in consumed_keys:
                doc[key] = value

        out_path = output_dir / self.cfg.filename
        if out_path.exists() and not self.cfg.overwrite:
            return ExportResult(skipped=[out_path])

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return ExportResult(written=[out_path])

        # TODO:算法之一（HandCfg → Sidecar YAML）
        # ────────────────────────────────────────
        # 输入
        #   target: HandCfg
        #   output_dir: Path
        #   extra: dict（来自 HandGenerationResult.metadata 或 runner 注入的 seed/id 等）
        #   cfg: SidecarCfg
        #
        # 输出：ExportResult
        #
        # ── 构建 sidecar dict ──
        #   import math, datetime
        #   doc = {
        #     "id": extra.get("id", "<unknown>"),
        #     "timestamp": datetime.datetime.now().isoformat(),
        #     "name": target.name,
        #     "family": target.family,
        #     "handedness": target.handedness,
        #     "dof": target.dof_count,
        #     "finger_count": len(target.fingers),
        #   }
        #
        # ── finger 统计（若 include_finger_stats）──
        #   if cfg.include_finger_stats:
        #     doc["fingers"] = []
        #     for f in target.fingers:
        #       total_len = sum(
        #         math.sqrt(sum(x**2 for x in j.origin.pos))
        #         for j in f.joints if j.origin
        #       )
        #       doc["fingers"].append({
        #         "name": f.name,
        #         "joint_count": len(f.joints),
        #         "revolute_dof": f.dof_count,
        #         "total_length_cm": round(total_len * 100, 3),
        #       })
        #
        # ── 溯源字段（若 include_provenance）──
        #   if cfg.include_provenance:
        #     doc["provenance"] = {
        #       "recipe_hash": extra.get("recipe_hash"),
        #       "seed": extra.get("seed"),
        #       "experiment_tag": cfg.experiment_tag or extra.get("experiment_tag"),
        #     }
        #
        # ── 合并 extra 中其余字段 ──
        #   (extra 中已被消费的 key 不重复写入)
        #
        # ── 写入文件 ──
        #   out_path = output_dir / cfg.filename
        #   if out_path.exists() and not cfg.overwrite:
        #     return ExportResult(skipped=[out_path])
        #   output_dir.mkdir(parents=True, exist_ok=True)
        #   import yaml
        #   out_path.write_text(yaml.dump(doc, allow_unicode=True, sort_keys=False))
        #   return ExportResult(written=[out_path])
        #
        # IDEA：Sidecar 最重要的字段是 id + recipe_hash，它们是以后批量数据集
        # 做去重和追溯的基础。建议 id 使用 uuid4 生成，recipe_hash 取
        # HandGeneratorCfg 序列化后的 md5/sha256 前 8 位。


__all__ = ["SidecarCfg", "SidecarExporter"]
