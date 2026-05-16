"""独立 post-mutate 的来源 topology 恢复辅助。

这个模块现在只做一件事：把 pre-made topology 根目录里的 `hand.yaml`
恢复回 post-mutate 运行时所需的 `HandCfg` 与 provenance。

新的目录 contract 明确固定为：

1. pre-made 基座直接位于 `.../<group>/<topology>/`
2. topology 根目录自己就持有 `hand.yaml`
3. 独立 post-mutate 的新 run 位于
   `.../<group>/<topology>/<mutate_timestamp>/<sample_id>/`

也就是说，恢复阶段不再扫描 sample 子目录，更不再引入 `*_origin`
这种会扭曲 topology 根语义的过渡机制。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ...asset_base import HandCfg


_SIDECAR_FILENAME = "hand.yaml"
_SIDECAR_SUMMARY_KEYS = {
    "id",
    "timestamp",
    "name",
    "family",
    "handedness",
    "dof",
    "finger_count",
    "fingers",
    "provenance",
    "hand_cfg",
    "warnings",
}


@dataclass(frozen=True)
class PostMutateSource:
    r"""独立 post-mutate 当前使用的一份来源 topology。

    Attributes:
        topology_dir: pre-made topology 根目录。
        origin_sidecar_path: topology 根下的 `hand.yaml` 路径。
        origin_sample_id: pre-made 逻辑样本 ID；来自 topology 根 sidecar 顶层 `id`。
        hand_cfg: 由 `hand.yaml.hand_cfg` 恢复出的完整 `HandCfg`。
        metadata: 继续透传给 post-mutate 导出侧的 provenance 元数据。
    """

    topology_dir: Path
    origin_sidecar_path: Path
    origin_sample_id: str
    hand_cfg: HandCfg
    metadata: dict[str, Any]


def load_post_mutate_source(topology_dir: Path | str) -> PostMutateSource:
    r"""从 pre-made topology 根目录恢复 mutate-only 来源。

    Args:
        topology_dir (Path | str): 形如
            `.../generated/<premade_timestamp>/<group>/<topology>/`

    Returns:
        PostMutateSource: 含恢复出的 `HandCfg` 与来源 provenance。
    """

    resolved_topology_dir = Path(topology_dir)  # 调用边界统一先收口到 `Path`
    if not resolved_topology_dir.is_dir():
        raise FileNotFoundError(f"Post-mutate source topology directory does not exist: {resolved_topology_dir}")

    # pre-made contract 已经改成 topology 根直接持有 sidecar，因此恢复入口也固定锚在这里。
    sidecar_path = resolved_topology_dir / _SIDECAR_FILENAME
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            "Independent post-mutate now requires a topology-root sidecar; "
            f"missing {sidecar_path}"
        )

    sidecar_doc = yaml.safe_load(sidecar_path.read_text(encoding="utf-8")) or {}  # topology 根 sidecar 是唯一真源
    if not isinstance(sidecar_doc, dict):
        raise ValueError(f"Sidecar must be a mapping, got {type(sidecar_doc).__name__}: {sidecar_path}")

    hand_cfg_raw = sidecar_doc.get("hand_cfg")
    if not isinstance(hand_cfg_raw, dict):
        raise ValueError(
            f"Sidecar {sidecar_path} is missing top-level 'hand_cfg'; cannot restore independent post-mutate source."
        )

    origin_sample_id = str(sidecar_doc.get("id") or "")  # pre-made 的稳定逻辑 ID 现在只保留在 metadata，而不再占目录层级
    if not origin_sample_id:
        raise ValueError(
            f"Topology-root sidecar {sidecar_path} is missing top-level 'id'; "
            "independent post-mutate requires a stable pre-made sample identifier."
        )

    hand_cfg = _restore_hand_cfg(hand_cfg_raw)  # sidecar snapshot -> 运行时 dataclass
    metadata = {
        key: value
        for key, value in sidecar_doc.items()
        if key not in _SIDECAR_SUMMARY_KEYS
    }
    metadata["source_origin_sample_id"] = origin_sample_id  # 继续显式标出 pre-made 逻辑样本 ID
    metadata["source_origin_topology_dir"] = str(resolved_topology_dir)  # 真实来源语义是 topology 根，而不是旧 sample 目录
    metadata["source_topology_dir"] = str(resolved_topology_dir)  # 供下游 preview / summary 保持沿用

    return PostMutateSource(
        topology_dir=resolved_topology_dir,
        origin_sidecar_path=sidecar_path,
        origin_sample_id=origin_sample_id,
        hand_cfg=hand_cfg,
        metadata=metadata,
    )


__all__ = ["PostMutateSource", "load_post_mutate_source"]


def _restore_hand_cfg(hand_cfg_raw: dict[str, Any]) -> HandCfg:
    r"""把 sidecar 里的 `hand_cfg` 快照恢复成真正的 `HandCfg`。

    # NOTE:
    当前 `AssetCfgBase.to_dict()` 会把 dataclass 递归压平成原生容器，但像
    `BoxGeometryCfg` / `CylinderGeometryCfg` 这样的 geometry 子类，其
    `geometry_type` 是 `ClassVar`，不会自动出现在输出字典里。

    因而 sidecar 中的几何快照常常长成：

    ```yaml
    geometry:
      size: [0.1, 0.2, 0.3]
    ```

    而不是 schema loader 直接期望的：

    ```yaml
    geometry:
      type: box
      size: [0.1, 0.2, 0.3]
    ```

    这里做的恢复不是重新发明 schema，而只是把这些“缺失 type 的几何字典”
    补成 loader 可识别的最小形状。
    """

    if not isinstance(hand_cfg_raw, dict):
        raise TypeError(f"'hand_cfg' must be a mapping, got {type(hand_cfg_raw).__name__}")
    normalized = _rehydrate_geometry_mappings(hand_cfg_raw)  # 先递归补 geometry type，再正式实例化
    return HandCfg(**normalized)


def _rehydrate_geometry_mappings(value: Any) -> Any:
    r"""递归补全所有缺失 `type` 的 geometry 映射。"""

    if isinstance(value, list):
        return [_rehydrate_geometry_mappings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rehydrate_geometry_mappings(item) for item in value)
    if not isinstance(value, dict):
        return value

    normalized = {key: _rehydrate_geometry_mappings(item) for key, item in value.items()}  # 先递归到底层节点
    if "geometry" in normalized and isinstance(normalized["geometry"], dict):
        normalized["geometry"] = _inject_geometry_type(normalized["geometry"])  # 子字段 `geometry` 是最常见缺 type 的位置
    return _inject_geometry_type(normalized)


def _inject_geometry_type(geometry_doc: dict[str, Any]) -> dict[str, Any]:
    r"""按最小启发式为 geometry 映射补上 `type` 字段。"""

    if "type" in geometry_doc or "kind" in geometry_doc:
        return geometry_doc  # 已显式声明几何类型时不再篡改

    normalized = dict(geometry_doc)
    if any(key in normalized for key in ("file_path", "path", "mesh")):
        normalized["type"] = "mesh"  # 有 mesh 路径时优先判成 mesh geometry
        return normalized
    if "size" in normalized:
        normalized["type"] = "box"  # `size=[x,y,z]` 是 box 的最小特征
        return normalized
    if "radius" in normalized and "length" in normalized:
        normalized["type"] = "cylinder"  # radius + length 对应圆柱
        return normalized
    if "radius" in normalized:
        normalized["type"] = "sphere"  # 只有半径时退成球
        return normalized
    return normalized
