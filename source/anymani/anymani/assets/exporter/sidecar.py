r"""Sidecar 导出器：把 HandCfg 的元数据、溯源信息与完整快照写为 YAML 文件。

Sidecar 是每个导出产物目录里与 URDF 同级的一个 YAML 文件，记录：

- 整手结构摘要（family / handedness / dof / finger_count）
- 生成溯源（generation config hash / timestamp / random seed）
- 关键参数统计（每根 finger 的真实轴向长度、关节限位范围）
- 可追踪字段（id / experiment tag）
- 以及一个可直接恢复的完整 `hand_cfg` 快照

设计说明
--------

### 溯源的意义

批量生成 N 个手资产之后，需要能从任意一个 URDF 文件反向追溯到它的
生成参数（pre-made preset / mutate 配置 / 随机种子）。Sidecar 是这条追踪链的
纸面记录，而不是把溯源嵌进 URDF 注释里（那会污染标准格式）。

### 与 RecipeLoader.dump() 的关系

    `RecipeLoader.dump()` 序列化的是生成器的**配置**（HandGeneratorCfg）；
`SidecarExporter` 记录的是这次生成的**产物描述**（HandCfg 结构摘要 + 上下文）
并且额外保存完整 `hand_cfg` 快照。两者互补——前者让你知道"用什么设置生成的"，
后者让你知道"生成出了什么"，并支持后续恢复。

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

新 sidecar 还可以包含 `geometry_semantics`：它不是从 URDF 逆向猜测的摘要，而是 exporter 在
`HandCfg` 真源仍在内存时写出的完整版本化 owner/kinematic/home/anchor 事实。这样 future
pre-made、post-mutate 和其变体资产在落盘时就共享同一 schema；bank 只负责读取和选择，robots
负责动态 lower，distill 不直接接触 exporter 内部对象。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_schema_geometry import derive_generated_geometry_semantics, geometry_semantics_to_dict
from ..handedness import handedness_contract
from ..validator._finger_length import measure_finger_axial_lengths
from ._base import ExporterBase, ExportResult

# ============================================================================
#  配置类
# ============================================================================


@dataclass
class SidecarCfg(AssetCfgBase):
    r"""Sidecar YAML 导出器配置。"""

    class_type: type[SidecarExporter] | None = None
    """关联的运行时类。"""

    filename: str = "hand.yaml"
    """Sidecar 输出文件名；相对于传入的 output_dir。"""

    include_provenance: bool = True
    """是否在 sidecar 里写入溯源字段（recipe_hash / seed / experiment_tag）。"""

    include_finger_stats: bool = True
    """是否在 sidecar 里写入每根 finger 的统计信息（真实轴向长度 / DOF 等）。"""

    experiment_tag: str | None = None
    """可选的实验标签；若不为 None 则写入 provenance.experiment_tag。"""

    overwrite: bool = True
    """若目标文件已存在，是否覆盖。``False`` 时记入 skipped 并跳过。"""

    include_geometry_semantics: bool = True
    """是否写入供 bank/robots/distill 消费的版本化 owner、运动学、home 与锚点种子语义。"""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = SidecarExporter


# ============================================================================
#  运行时壳
# ============================================================================


class SidecarExporter(ExporterBase):
    r"""Sidecar YAML 写入器。

        把 `HandCfg` 的结构摘要、溯源信息与完整快照写出为 YAML 文件，与 URDF 同目录存放。
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
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            "name": target.name,
            "family": target.family,
            "handedness": target.handedness,
            "dof": target.dof_count,
            "finger_count": len(target.fingers),
        }
        expected_handedness_contract = handedness_contract(target=target.handedness)  # exporter 认可的唯一 generated same-$q$ 合同
        metadata_handedness_contract = target.metadata.get("handedness_contract")
        if target.handedness == "left" and metadata_handedness_contract != expected_handedness_contract:
            raise ValueError(
                "left HandCfg must carry a complete strict handedness_contract before sidecar export"
            )  # 不能仅凭顶层 left 标签伪造“已完成物理反射”的证书
        doc["handedness_contract"] = expected_handedness_contract  # 顶层轻量字段供 HandBank fail-closed gate 直接读取
        consumed_keys.add("id")

        if self.cfg.include_finger_stats:
            fingers: list[dict[str, Any]] = []
            axial_lengths = {
                measurement.finger_name: measurement
                for measurement in measure_finger_axial_lengths(target)
            }  # sidecar 与 validator 共用同一套长度定义，避免“摘要一套、闸门一套”
            for finger in target.fingers:
                measurement = axial_lengths.get(finger.name)  # 当前 finger 的真实轴向长度测量；缺失时显式写空值，而不是偷偷退回旧近似
                fingers.append(
                    {
                        "name": finger.name,
                        "joint_count": len(finger.joints),
                        "revolute_dof": finger.dof_count,
                        "total_length_cm": None if measurement is None else round(measurement.axial_length * 100.0, 3),
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

        if self.cfg.include_geometry_semantics:
            geometry_semantics = derive_generated_geometry_semantics(
                target,
                asset_id=str(doc["id"]),
                topology_key=None if doc.get("topology_name") is None else str(doc["topology_name"]),
            )
            doc["geometry_semantics"] = geometry_semantics_to_dict(geometry_semantics)

        # 这里显式保留完整 `HandCfg` 快照，而不是再要求后续从 URDF 逆向提取。
        # 这样 independent post-mutate 可以直接从 `hand.yaml` 恢复内存对象。
        doc["hand_cfg"] = target.to_dict()

        out_path = output_dir / self.cfg.filename
        if out_path.exists() and not self.cfg.overwrite:
            return ExportResult(skipped=[out_path])

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return ExportResult(written=[out_path])


__all__ = ["SidecarCfg", "SidecarExporter"]
