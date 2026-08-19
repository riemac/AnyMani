r"""Dataset build 阶段的静态 geometry fingerprint。

该 fingerprint 用于拒绝 post-mutate no-op、同一 mother 的重复 variants 和跨 role
静态物理重复。它覆盖 frame calibration、完整 kinematic chain、$q_{home}$、owner、
collision component transforms、primitive 参数与 mesh bytes；显式排除资产 ID、路径、
joint limits、动力学参数和 anchor realization。

它不是 Distill ``physical_geometry_hash``：后者还覆盖 lower 后空间旋量、owner union
真实表面与图张量。两层检查分别服务生成期快速补采与训练前最终 leakage gate。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def geometry_fingerprint_from_sidecar(sidecar_path: str | Path) -> str:
    r"""读取 generated sidecar 并计算与路径/ID/limits 无关的静态几何身份。"""

    resolved_path = Path(sidecar_path).resolve()
    document = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise TypeError(f"hand sidecar must be a mapping: {resolved_path}")
    semantics = document.get("geometry_semantics")
    if not isinstance(semantics, dict):
        raise ValueError(f"hand sidecar lacks geometry_semantics: {resolved_path}")
    return geometry_fingerprint_from_semantics(semantics, sidecar_dir=resolved_path.parent)


def geometry_fingerprint_from_semantics(semantics: dict[str, Any], *, sidecar_dir: Path) -> str:
    r"""把 geometry semantics 规约成 build-time 静态物理 payload 并哈希。"""

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

    # mesh 路径本身不具物理意义；以实际 bytes 身份替换，避免不同 run 的同 mesh 被误判为不同。
    components = payload.get("components", ())
    if not isinstance(components, list):
        raise TypeError("geometry_semantics.components must be a sequence")
    for component in components:
        if not isinstance(component, dict):
            raise TypeError("geometry semantics component must be a mapping")
        geometry = component.get("geometry_payload")
        if not isinstance(geometry, dict) or "file_path" not in geometry:
            continue
        raw_path = Path(str(geometry.pop("file_path")))
        mesh_path = raw_path if raw_path.is_absolute() else sidecar_dir / raw_path
        if not mesh_path.is_file():
            raise FileNotFoundError(f"geometry fingerprint mesh does not exist: {mesh_path}")
        geometry["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    digest = hashlib.sha256()
    digest.update(b"anymani-dataset-build-geometry-v1\0")
    digest.update(encoded)
    return digest.hexdigest()


__all__ = ["geometry_fingerprint_from_semantics", "geometry_fingerprint_from_sidecar"]
