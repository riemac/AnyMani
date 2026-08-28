r"""AnyMani URDF→USD 稳定 cache identity。

IsaacLab 2.3.2 的 ``AssetConverterBase`` 会 hash 主 URDF 与 converter cfg，但不读取 URDF 引用的
mesh bytes。对 2048 个 canonical assets，这会让 mesh 原地更新后继续复用旧 USD。本模块在进入
IsaacLab converter 前计算更完整的目录 key：

$$
k=\operatorname{SHA256}(H_{urdf},\{uri_i,H_{mesh_i}\},C_{converter},V_{Lab},V_{Sim},H_{converter-src},S_{canonical}).
$$

``asset_row`` 是一次 selection 内的策略路由，不改变物理 USD，因此从 canonical identity 中排除。
模块只返回目录路径，不创建目录、不 import Isaac Sim/Kit；实际 lazy hit/miss 仍由 ``UrdfFileCfg``
与 IsaacLab converter 负责。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from anymani.assets.bank.urdf_utils import parse_urdf_mesh_refs

_ROUTING_ONLY_CANONICAL_FIELDS = frozenset({"asset_row", "asset_row_start"})
"""不进入 physical USD identity 的 selection-local routing 字段。"""


def _sha256_file(path: Path) -> str:
    r"""流式计算文件 SHA-256，避免批量 mesh identity 暂存全部 bytes。"""

    digest = hashlib.sha256()  # 单文件内容身份；不混入绝对路径
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)  # 1 MiB chunk，峰值内存与文件大小解耦
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    r"""把 converter cfg 降为确定性 JSON 容器，不保留对象地址。"""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        module = getattr(value, "__module__", "unknown")  # callable 的稳定实现命名空间
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
        return f"{module}.{qualname}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"USD cache identity cannot serialize value of type {type(value).__qualname__}")


def build_urdf_usd_cache_dir(
    *,
    urdf_path: Path,
    converter_config: Mapping[str, Any],
    isaaclab_version: str,
    isaac_sim_version: str,
    converter_implementation_sha256: str,
    canonical_identity: Mapping[str, Any] | None = None,
    cache_root: Path | None = None,
) -> Path:
    r"""计算 mesh-aware、版本化且与 selection row 无关的 USD cache directory。

    Args:
        urdf_path (Path): importer 实际消费的 source/canonical URDF。
        converter_config (Mapping[str, Any]): 移除 ``asset_path/usd_dir/usd_file_name`` 后的
            ``UrdfFileCfg.to_dict()``；只描述会改变输出 USD 的 importer 参数。
        isaaclab_version (str): 当前 IsaacLab Python distribution/version identity。
        isaac_sim_version (str): 当前 Isaac Sim 版本，例如 ``5.1.0``。
        converter_implementation_sha256 (str): 当前 UrdfConverter 实现源码 hash。
        canonical_identity (Mapping[str, Any] | None): schema/artifact identity；routing row 会被排除。
        cache_root (Path | None): 测试或调用方显式 cache 根；默认
            ``${ANYMANI_CACHE_DIR:-~/.cache/anymani}``。

    Returns:
        Path: ``<root>/isaaclab/usd/<sim-version>/<sha256-key>``；本函数不创建目录。
    """

    resolved_urdf = urdf_path.expanduser().resolve(strict=True)  # physical importer input
    mesh_refs = parse_urdf_mesh_refs(resolved_urdf, require_existing=True)  # XML 顺序的 URI→real path
    mesh_dependencies = [
        {
            "raw_uri": ref.raw_uri,
            "sha256": _sha256_file(ref.real_path),
        }
        for ref in mesh_refs
    ]  # URI 和 bytes 都进入 key；同 basename 的不同目录不会碰撞
    canonical = {
        str(key): _json_safe(value)
        for key, value in sorted((canonical_identity or {}).items())
        if key not in _ROUTING_ONLY_CANONICAL_FIELDS
    }  # asset_row 改变只影响 policy routing，不改变 physical USD
    payload = {
        "schema": "anymani.urdf_usd_cache.v1",
        "urdf_sha256": _sha256_file(resolved_urdf),
        "mesh_dependencies": mesh_dependencies,
        "converter_config": _json_safe(converter_config),
        "isaaclab_version": str(isaaclab_version),
        "isaac_sim_version": str(isaac_sim_version),
        "converter_implementation_sha256": str(converter_implementation_sha256),
        "canonical_identity": canonical,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    cache_key = hashlib.sha256(encoded).hexdigest()  # 全输入 identity 的稳定 64-hex directory name
    root = cache_root or Path(os.environ.get("ANYMANI_CACHE_DIR", "~/.cache/anymani"))
    safe_sim_version = str(isaac_sim_version).replace("/", "_")  # 防止版本字符串逃逸目录层级
    return root.expanduser() / "isaaclab" / "usd" / safe_sim_version / cache_key


__all__ = ["build_urdf_usd_cache_dir"]
