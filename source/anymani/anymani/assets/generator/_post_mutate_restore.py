"""独立 post-mutate 的产物定位与 HandCfg 恢复辅助。

这个模块只负责 mutate-only 工作流里那条“从已有 pre-made 产物恢复运行时输入”的链路，
避免把目录扫描、`*_origin` 重命名、sidecar 解析这些细节继续塞回
`hand_generator.py`。

当前首版的边界非常明确：

1. 只接受 **topology 目录** 作为输入，而不是整个时间戳目录；
2. 只依赖现有 `hand.yaml` 中的顶层 `hand_cfg` 快照恢复，不做 URDF 逆向提取；
3. 首次进入该 topology 时，把唯一 pre-made 原始样本目录重命名为 `*_origin`，
   之后所有 post-mutate 产物都与它并列存放。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..asset_base import HandCfg


_SIDECAR_FILENAME = "hand.yaml"
_ORIGIN_SUFFIX = "_origin"
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
    r"""独立 post-mutate 当前使用的一份来源样本。

    Attributes:
        topology_dir: 当前 mutate-only 工作流操作的 topology 目录。
        origin_sample_dir: 被标记为 `*_origin` 的 pre-made 原始样本目录。
        origin_sample_id: 原始样本的逻辑 ID（来自 sidecar 顶层 `id`）。
        hand_cfg: 由 `hand.yaml.hand_cfg` 恢复出的完整 `HandCfg`。
        metadata: 需要继续传给 post-mutate 导出侧的 provenance 元数据。
    """

    topology_dir: Path
    origin_sample_dir: Path
    origin_sample_id: str
    hand_cfg: HandCfg
    metadata: dict[str, Any]


def load_post_mutate_source(topology_dir: Path | str) -> PostMutateSource:
    r"""从一个 topology 目录恢复 mutate-only 的来源样本。

    这里故意把“输入 topology 目录”的语义写死，而不是做过度智能的路径猜测。
    因为当前统一 runner 已经明确指定：独立 post-mutate 入口应以 topology 目录为输入，
    自动找到唯一 pre-made 原始样本并把它改名成 `*_origin`。

    Args:
        topology_dir (Path | str): 例如
            `.../generated/<timestamp>/<group>/<topology_name>/`

    Returns:
        PostMutateSource: 已定位 origin 目录并恢复出 `HandCfg` 的来源描述。
    """

    resolved_topology_dir = Path(topology_dir)
    if not resolved_topology_dir.is_dir():
        raise FileNotFoundError(f"Post-mutate source topology directory does not exist: {resolved_topology_dir}")

    origin_sample_dir = _resolve_origin_sample_dir(resolved_topology_dir)
    sidecar_path = origin_sample_dir / _SIDECAR_FILENAME
    sidecar_doc = yaml.safe_load(sidecar_path.read_text(encoding="utf-8")) or {}
    if not isinstance(sidecar_doc, dict):
        raise ValueError(f"Sidecar must be a mapping, got {type(sidecar_doc).__name__}: {sidecar_path}")

    hand_cfg_raw = sidecar_doc.get("hand_cfg")
    if not isinstance(hand_cfg_raw, dict):
        raise ValueError(
            f"Sidecar {sidecar_path} is missing top-level 'hand_cfg'; cannot restore independent post-mutate source."
        )

    hand_cfg = _restore_hand_cfg(hand_cfg_raw)
    origin_sample_id = str(sidecar_doc.get("id") or origin_sample_dir.name.removesuffix(_ORIGIN_SUFFIX))

    metadata = {
        key: value
        for key, value in sidecar_doc.items()
        if key not in _SIDECAR_SUMMARY_KEYS
    }
    metadata["source_origin_sample_id"] = origin_sample_id
    metadata["source_origin_sample_dir"] = str(origin_sample_dir)
    metadata["source_topology_dir"] = str(resolved_topology_dir)

    return PostMutateSource(
        topology_dir=resolved_topology_dir,
        origin_sample_dir=origin_sample_dir,
        origin_sample_id=origin_sample_id,
        hand_cfg=hand_cfg,
        metadata=metadata,
    )


def _resolve_origin_sample_dir(topology_dir: Path) -> Path:
    r"""在 topology 目录下定位（并在需要时创建）`*_origin` 样本目录。

    规则是：

    1. 若已经存在唯一的 `*_origin/hand.yaml`，直接使用；
    2. 否则若只存在唯一的普通样本目录，则把它重命名为 `*_origin`；
    3. 其余情况（例如有多个普通样本但没有 `*_origin`）一律报错，避免猜错来源。
    """

    candidate_dirs = sorted(
        path
        for path in topology_dir.iterdir()
        if path.is_dir() and (path / _SIDECAR_FILENAME).is_file()
    )
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No sample directory containing '{_SIDECAR_FILENAME}' was found under topology directory {topology_dir}"
        )

    origin_dirs = [path for path in candidate_dirs if path.name.endswith(_ORIGIN_SUFFIX)]
    if origin_dirs:
        if len(origin_dirs) != 1:
            raise ValueError(
                f"Expected exactly one '*{_ORIGIN_SUFFIX}' sample under {topology_dir}, got {origin_dirs!r}"
            )
        return origin_dirs[0]

    if len(candidate_dirs) != 1:
        raise ValueError(
            "Topology directory has multiple sample subdirectories but none is marked as '*_origin'; "
            f"cannot infer which one is the pre-made source: {candidate_dirs!r}"
        )

    source_dir = candidate_dirs[0]
    renamed_dir = source_dir.with_name(f"{source_dir.name}{_ORIGIN_SUFFIX}")
    source_dir.rename(renamed_dir)
    return renamed_dir


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
    normalized = _rehydrate_geometry_mappings(hand_cfg_raw)
    return HandCfg(**normalized)


def _rehydrate_geometry_mappings(value: Any) -> Any:
    r"""递归补全所有缺失 `type` 的 geometry 映射。"""

    if isinstance(value, list):
        return [_rehydrate_geometry_mappings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rehydrate_geometry_mappings(item) for item in value)
    if not isinstance(value, dict):
        return value

    normalized = {key: _rehydrate_geometry_mappings(item) for key, item in value.items()}
    if "geometry" in normalized and isinstance(normalized["geometry"], dict):
        normalized["geometry"] = _inject_geometry_type(normalized["geometry"])
    return _inject_geometry_type(normalized)


def _inject_geometry_type(geometry_doc: dict[str, Any]) -> dict[str, Any]:
    r"""按最小启发式为 geometry 映射补上 `type` 字段。"""

    if "type" in geometry_doc or "kind" in geometry_doc:
        return geometry_doc

    normalized = dict(geometry_doc)
    if any(key in normalized for key in ("file_path", "path", "mesh")):
        normalized["type"] = "mesh"
        return normalized
    if "size" in normalized:
        normalized["type"] = "box"
        return normalized
    if "radius" in normalized and "length" in normalized:
        normalized["type"] = "cylinder"
        return normalized
    if "radius" in normalized:
        normalized["type"] = "sphere"
        return normalized
    return normalized
