r"""URDF 纯解析工具 scaffold。

本模块未来承接从 `tasks/gm/heterogeneous_test_env_cfg.py` 中抽离出来的 IsaacLab-free
XML 解析逻辑，例如：

- `<mesh filename="..."/>` 的 raw URI 收集与路径闭包检查；
- `<visual name="..."><material><color rgba="..."/>` 的 debug color 解析。

这里不导入 IsaacLab / USD / Omni，也不读取 mesh 几何内容。当前只放方法桩，避免
`hand_bank.py` 或 `hand_container.py` 后续变成大文件。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

from .hand_container import UrdfMeshRef, UrdfRgba


def parse_urdf_mesh_refs(
    urdf_path: Path,
    *,
    virtual_mesh_dir: PurePosixPath = PurePosixPath("meshes"),
    require_existing: bool = True,
) -> tuple[UrdfMeshRef, ...]:
    r"""解析 URDF mesh 引用并返回虚拟标准视图中的 mesh refs。

    Args:
        urdf_path (Path): 已解析到真实磁盘的 `hand.urdf` 路径。
        virtual_mesh_dir (PurePosixPath): 虚拟标准视图中 mesh 文件所在目录。

        require_existing (bool): 是否要求每个解析出的真实 mesh 路径必须存在。

    Returns:
        tuple[UrdfMeshRef, ...]: URDF 中出现的 mesh 引用记录，保持 XML 遍历顺序。

    Raises:
        FileNotFoundError: 当 `urdf_path` 或必需 mesh 文件不存在时抛出。
        ValueError: 当遇到当前无法闭合到本地文件的 URI 时抛出。
    """

    resolved_urdf_path = Path(urdf_path).expanduser().resolve(strict=False)
    if not resolved_urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {resolved_urdf_path}")

    mesh_refs, _ = parse_urdf_metadata(
        resolved_urdf_path,
        virtual_mesh_dir=virtual_mesh_dir,
        require_existing=require_existing,
        parse_visual_rgba=False,
    )
    return mesh_refs


def parse_urdf_metadata(
    urdf_path: Path,
    *,
    virtual_mesh_dir: PurePosixPath = PurePosixPath("meshes"),
    require_existing: bool = True,
    parse_visual_rgba: bool = True,
) -> tuple[tuple[UrdfMeshRef, ...], dict[str, UrdfRgba]]:
    r"""一次 XML parse 同时解析 mesh 引用和 named visual RGBA。

    HandContainer 同时需要路径闭包与可视化颜色。两者来自同一份 URDF XML，合并入口
    避免每项资产重复执行 ``ET.parse``，但不读取 mesh 顶点或面片。
    """

    resolved_urdf_path = Path(urdf_path).expanduser().resolve(strict=False)
    if not resolved_urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {resolved_urdf_path}")

    root = ET.parse(resolved_urdf_path).getroot()
    mesh_refs: list[UrdfMeshRef] = []
    for mesh_elem in root.findall(".//mesh"):
        raw_uri = mesh_elem.attrib.get("filename")
        if not raw_uri:
            continue
        real_path = _resolve_mesh_uri(raw_uri, urdf_path=resolved_urdf_path)
        if require_existing and not real_path.is_file():
            raise FileNotFoundError(f"URDF mesh reference {raw_uri!r} does not exist: {real_path}")
        mesh_refs.append(
            UrdfMeshRef(
                raw_uri=raw_uri,
                virtual_path=virtual_mesh_dir / real_path.name,
                real_path=real_path,
            )
        )
    rgba_by_name = _visual_rgba_from_root(root, urdf_path=resolved_urdf_path) if parse_visual_rgba else {}
    return tuple(mesh_refs), rgba_by_name


def parse_urdf_visual_rgba_by_name(urdf_path: Path) -> dict[str, UrdfRgba]:
    r"""只解析 URDF named visual 的 RGBA，不要求无关 mesh URI 可解析。"""

    resolved_urdf_path = Path(urdf_path).expanduser().resolve(strict=False)
    if not resolved_urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {resolved_urdf_path}")
    root = ET.parse(resolved_urdf_path).getroot()
    return _visual_rgba_from_root(root, urdf_path=resolved_urdf_path)


def _visual_rgba_from_root(root: ET.Element, *, urdf_path: Path) -> dict[str, UrdfRgba]:
    r"""从已解析 XML root 读取 named visual color，供组合与独立入口共用。"""

    rgba_by_name: dict[str, UrdfRgba] = {}
    for visual_elem in root.findall(".//visual"):
        visual_name = visual_elem.attrib.get("name")
        if not visual_name:
            continue
        color_elem = visual_elem.find("./material/color")
        if color_elem is None:
            continue
        raw_rgba = color_elem.attrib.get("rgba")
        if raw_rgba is None:
            continue
        rgba_by_name[visual_name] = _parse_rgba(raw_rgba, visual_name=visual_name, urdf_path=urdf_path)
    return rgba_by_name


def _resolve_mesh_uri(raw_uri: str, *, urdf_path: Path) -> Path:
    r"""把 URDF mesh URI 解析为本地真实路径。"""

    if raw_uri.startswith("package://"):
        raise ValueError(f"package:// mesh URI is not supported by HandAssetBank yet: {raw_uri}")
    mesh_path = Path(raw_uri).expanduser()
    if mesh_path.is_absolute():
        return mesh_path.resolve(strict=False)
    return (urdf_path.parent / mesh_path).resolve(strict=False)


def _parse_rgba(raw_rgba: str, *, visual_name: str, urdf_path: Path) -> UrdfRgba:
    r"""解析 URDF color rgba 字符串为四元组。"""

    parts = raw_rgba.split()
    if len(parts) != 4:
        raise ValueError(f"visual {visual_name!r} in {urdf_path} has invalid rgba field: {raw_rgba!r}")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"visual {visual_name!r} in {urdf_path} has non-float rgba field: {raw_rgba!r}") from exc


__all__ = [
    "parse_urdf_metadata",
    "parse_urdf_mesh_refs",
    "parse_urdf_visual_rgba_by_name",
]
