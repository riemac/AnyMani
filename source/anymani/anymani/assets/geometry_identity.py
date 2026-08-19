r"""生成期与数据集发布期共用的静态 geometry fingerprint。

该 fingerprint 回答“两个资产是否提供同一份静态可运动碰撞几何”，用于拒绝
post-mutate no-op、同一 variant set 内重复和跨 dataset role 的重复。它覆盖 frame
calibration、完整 kinematic chain、$q_{home}$、owner、collision transforms、primitive
参数与 mesh bytes；显式排除资产 ID、路径、joint limits、动力学参数和 anchor realization。

它不是 Distill ``physical_geometry_hash``：后者还覆盖 lower 后空间旋量、owner union
真实表面与图张量。两层检查分别服务生成期逐槽补抽和训练前最终 leakage gate。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .asset_base import HandCfg
from .asset_schema_geometry import derive_generated_geometry_semantics, geometry_semantics_to_dict


def geometry_fingerprint_from_hand(hand: HandCfg) -> str:
    r"""从已物化最终 collision 几何的 ``HandCfg`` 计算静态几何身份。

    ``asset_id`` 只用于满足 geometry-semantics schema，随后会在 fingerprint 规约中移除。
    mesh 路径允许不同，但其真实文件 bytes 必须可读且会替代路径进入哈希。
    """

    semantics = derive_generated_geometry_semantics(hand, asset_id="__geometry_identity__")
    return geometry_fingerprint_from_semantics(geometry_semantics_to_dict(semantics))


def geometry_fingerprint_from_sidecar(sidecar_path: str | Path) -> str:
    r"""读取 generated sidecar 并计算与路径、ID、limits 和 dynamics 无关的静态几何身份。"""

    resolved_path = Path(sidecar_path).resolve()
    document = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise TypeError(f"hand sidecar must be a mapping: {resolved_path}")
    semantics = document.get("geometry_semantics")
    if not isinstance(semantics, dict):
        raise ValueError(f"hand sidecar lacks geometry_semantics: {resolved_path}")
    return geometry_fingerprint_from_semantics(semantics, sidecar_dir=resolved_path.parent)


def geometry_fingerprint_from_semantics(
    semantics: dict[str, Any],
    *,
    sidecar_dir: Path | None = None,
) -> str:
    r"""把 geometry semantics 规约成 build-time 静态物理 payload 并哈希。

    Args:
        semantics (dict[str, Any]): 已验证或刚从 ``HandCfg`` 推导的 geometry-semantics 文档。
        sidecar_dir (Path | None): 相对 mesh 路径的解析基准；内存 ``HandCfg`` 使用当前路径语义。

    Returns:
        str: 带固定域分隔符的 SHA-256 静态几何身份。
    """

    payload = deepcopy(semantics)
    for field_name in (
        "content_hash",
        "migration_version",
        "source_kind",
        "asset_id",
        "asset_name",
        "topology_key",
        "family",
        "joint_limits_rad",
        "anchor_seeds",
    ):
        payload.pop(field_name, None)

    # 文件位置不是物理属性；以真实 bytes 替换路径，使 mother 与 run-local mesh 可比较。
    components = payload.get("components", ())
    if not isinstance(components, (list, tuple)):
        raise TypeError("geometry_semantics.components must be a sequence")
    for component in components:
        if not isinstance(component, dict):
            raise TypeError("geometry semantics component must be a mapping")
        geometry = component.get("geometry_payload")
        if not isinstance(geometry, dict) or "file_path" not in geometry:
            continue
        raw_path = Path(str(geometry.pop("file_path")))
        mesh_path = raw_path if raw_path.is_absolute() or sidecar_dir is None else sidecar_dir / raw_path
        if not mesh_path.is_file():
            raise FileNotFoundError(f"geometry fingerprint mesh does not exist: {mesh_path}")
        geometry["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    digest = hashlib.sha256()
    digest.update(b"anymani-dataset-build-geometry-v1\0")
    digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "geometry_fingerprint_from_hand",
    "geometry_fingerprint_from_semantics",
    "geometry_fingerprint_from_sidecar",
]
