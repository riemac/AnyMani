r"""程序化 mesh 生成与物化工具。

本模块服务于一个非常具体的资产 contract：某些几何在 builder / mutator 阶段
仍然应以少量科研参数表达，但进入 physics closure / validator / exporter 前，
必须变成真实存在的 triangle mesh 文件。

当前首个使用者是 `cs` fingertip：

- 研究接口仍是半径 $r$、圆柱高度 $h$、高半比 $\lambda=h/r$；
- batched IsaacLab articulation 需要它在 schema 上表现为单个 collision body；
- physics closure 需要真实 watertight mesh 来做 `trimesh` 体积分；
- sidecar / URDF 需要记录可恢复、可追踪的最终 mesh 路径。

# NOTE:
这里不是把 `cs` 变成普通 custom tip preset。`cs` 仍是 procedural primitive tip，
只是最终 collision/visual 表达从两个 primitive body 变为一个外表面 mesh body。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from .asset_base import HandCfg, JointCfg
from .asset_schema_core import CollisionGeometryCfg, MeshGeometryCfg, PoseCfg, VisualGeometryCfg

PROCEDURAL_SCHEME = "procedural"
r"""程序化 mesh URI 的 scheme。"""

PROCEDURAL_AUTHORITY = "anymani"
r"""当前项目内部 procedural mesh 的 URI authority。"""

CS_TIP_PATH = "/cs_tip"
r"""`cs` fingertip procedural mesh 的 URI path。"""

DEFAULT_CS_TIP_RADIAL_SEGMENTS = 32
r"""`cs` mesh 圆周离散段数，兼顾 SDF 质量和导出文件体积。"""

DEFAULT_CS_TIP_CAP_RINGS = 8
r"""`cs` 半球帽从赤道到顶点之间的纬线段数。"""

_MIN_POSITIVE_LENGTH = 1e-9
r"""程序化 mesh 参数的最小正长度，防止退化三角面。"""


@dataclass(frozen=True)
class ProceduralCsTipSpec:
    r"""参数化 `cs` fingertip mesh 的最小几何规格。

    外表面定义为：

    1. 底面为 $y=0$ 的圆盘；
    2. 主体为半径 $r$、高度 $h$ 的圆柱，$0\le y\le h$；
    3. 顶部为球心在 $(0,h,0)$、半径 $r$ 的上半球，$h\le y\le h+r$。

    这对应旧 `cylinder + sphere` collision union 的外边界：保留圆柱主体和球帽
    接触形状，但去掉两个 primitive 之间的内部重叠面，使 `trimesh` 能把它当作
    单个 watertight 刚体求体积、质心和惯量。
    """

    radius: float
    r"""圆柱与球帽共享半径 $r$，单位 m。"""

    height: float
    r"""圆柱主体高度 $h$，单位 m。"""

    radial_segments: int = DEFAULT_CS_TIP_RADIAL_SEGMENTS
    r"""圆周离散段数 $N_\theta$。"""

    cap_rings: int = DEFAULT_CS_TIP_CAP_RINGS
    r"""半球帽纬向离散段数 $N_\phi$。"""

    def __post_init__(self) -> None:
        r"""校验 mesh 参数不退化。"""

        if float(self.radius) <= _MIN_POSITIVE_LENGTH:
            raise ValueError(f"cs tip radius must be positive, got {self.radius!r}")
        if float(self.height) <= _MIN_POSITIVE_LENGTH:
            raise ValueError(f"cs tip height must be positive, got {self.height!r}")
        if int(self.radial_segments) < 8:
            raise ValueError(f"cs tip radial_segments must be >= 8, got {self.radial_segments!r}")
        if int(self.cap_rings) < 2:
            raise ValueError(f"cs tip cap_rings must be >= 2, got {self.cap_rings!r}")

    @property
    def ratio(self) -> float:
        r"""返回高半比 $\lambda=h/r$。"""

        return float(self.height) / float(self.radius)


def make_procedural_cs_tip_uri(
    *,
    radius: float,
    height: float,
    radial_segments: int = DEFAULT_CS_TIP_RADIAL_SEGMENTS,
    cap_rings: int = DEFAULT_CS_TIP_CAP_RINGS,
) -> str:
    r"""把 `cs` 几何参数编码成稳定 procedural URI。

    URI 只作为 builder/mutator 到 materializer 之间的中间表示；进入 physics
    closure 前会被替换成真实 OBJ 路径。
    """

    spec = ProceduralCsTipSpec(
        radius=float(radius),
        height=float(height),
        radial_segments=int(radial_segments),
        cap_rings=int(cap_rings),
    )
    query = urlencode(
        {
            "radius": _fmt_float(spec.radius),
            "height": _fmt_float(spec.height),
            "radial_segments": str(spec.radial_segments),
            "cap_rings": str(spec.cap_rings),
        }
    )
    return f"{PROCEDURAL_SCHEME}://{PROCEDURAL_AUTHORITY}{CS_TIP_PATH}?{query}"


def is_procedural_cs_tip_uri(file_path: str) -> bool:
    r"""判断一个 mesh path 是否是 `cs` fingertip procedural URI。"""

    parsed = urlparse(str(file_path))
    return parsed.scheme == PROCEDURAL_SCHEME and parsed.netloc == PROCEDURAL_AUTHORITY and parsed.path == CS_TIP_PATH


def parse_procedural_cs_tip_uri(file_path: str) -> ProceduralCsTipSpec:
    r"""从 procedural URI 解析 `cs` fingertip 参数。"""

    parsed = urlparse(str(file_path))
    if parsed.scheme != PROCEDURAL_SCHEME or parsed.netloc != PROCEDURAL_AUTHORITY or parsed.path != CS_TIP_PATH:
        raise ValueError(f"not an AnyMani procedural cs tip URI: {file_path!r}")
    query = parse_qs(parsed.query)
    return ProceduralCsTipSpec(
        radius=float(_single_query_value(query, "radius")),
        height=float(_single_query_value(query, "height")),
        radial_segments=int(_single_query_value(query, "radial_segments", DEFAULT_CS_TIP_RADIAL_SEGMENTS)),
        cap_rings=int(_single_query_value(query, "cap_rings", DEFAULT_CS_TIP_CAP_RINGS)),
    )


def materialize_procedural_cs_tip_mesh(
    spec: ProceduralCsTipSpec,
    mesh_root_dir: Path,
    *,
    write_enabled: bool = True,
) -> tuple[Path, bool]:
    r"""把一份参数化 `cs` mesh 写入导出边界共享 mesh 目录。

    Args:
        spec: `cs` tip 的几何参数。
        mesh_root_dir: 当前 topology/run 边界共享的 `meshes/` 目录。
        write_enabled: 为 `False` 时只返回应使用的路径，不真正写文件。

    Returns:
        tuple[Path, bool]: 真实 mesh 路径，以及本次调用是否实际写出了新文件。
    """

    mesh_root = Path(mesh_root_dir)
    mesh_path = mesh_root / cs_tip_mesh_filename(spec)
    written = False
    if write_enabled:
        mesh_root.mkdir(parents=True, exist_ok=True)
        if not mesh_path.exists():
            mesh_path.write_text(cs_tip_obj_text(spec), encoding="utf-8")
            written = True
    return mesh_path, written


def materialize_hand_procedural_meshes(
    hand: HandCfg,
    *,
    mesh_root_dir: Path,
    write_enabled: bool = True,
) -> tuple[HandCfg, list[Path]]:
    r"""扫描并物化 `HandCfg` 中所有 procedural / legacy `cs` fingertip mesh。

    这一步必须在 physics closure 之前执行，因为 `asset_physics.py` 的 `trimesh`
    backend 需要读取真实 mesh 文件，而不是读取 builder 阶段的 procedural URI。

    同时这里会迁移历史 sidecar 中的 two-primitive `cs` tip，使 post-mutate 从旧
    topology 根恢复时，也能进入新的 single-mesh schema。
    """

    materialized = hand.copy()
    written_paths: list[Path] = []
    for finger_index, finger in enumerate(materialized.fingers):
        joints: list[JointCfg] = []
        finger_changed = False
        for joint in finger.joints:
            new_joint, written = materialize_joint_procedural_meshes(
                joint,
                mesh_root_dir=mesh_root_dir,
                write_enabled=write_enabled,
            )
            joints.append(new_joint)
            finger_changed = finger_changed or new_joint is not joint
            written_paths.extend(written)
        if finger_changed:
            materialized.fingers[finger_index] = finger.replace(joints=joints)
    if written_paths:
        metadata = dict(materialized.metadata)
        metadata["procedural_mesh_materialization"] = {
            "cs_tip_mesh_count": len({str(path) for path in written_paths}),
            "mesh_root_dir": str(Path(mesh_root_dir)),
        }
        materialized.metadata = metadata
    return materialized.replace(fingers=materialized.fingers, metadata=dict(materialized.metadata)), written_paths


def materialize_joint_procedural_meshes(
    joint: JointCfg,
    *,
    mesh_root_dir: Path,
    write_enabled: bool = True,
) -> tuple[JointCfg, list[Path]]:
    r"""物化单个 joint child-link 上的 procedural `cs` mesh。

    返回原 `joint` 对象表示没有修改；返回新 `JointCfg` 表示该 joint 的 collision /
    visual schema 已经从 procedural URI 或 legacy two-primitive schema 迁移到真实 mesh。
    """

    procedural_spec = _procedural_spec_from_mesh_elements(joint)
    if procedural_spec is not None:
        return _materialize_procedural_mesh_joint(joint, procedural_spec, mesh_root_dir=mesh_root_dir, write_enabled=write_enabled)

    legacy = _legacy_cs_spec_from_joint(joint)
    if legacy is not None:
        spec, origin = legacy
        return _materialize_legacy_cs_joint(joint, spec, origin, mesh_root_dir=mesh_root_dir, write_enabled=write_enabled)

    return joint, []


def cs_tip_mesh_filename(spec: ProceduralCsTipSpec) -> str:
    r"""为 `cs` 参数生成稳定 OBJ 文件名。"""

    payload = "::".join(
        (
            _fmt_float(spec.radius),
            _fmt_float(spec.height),
            str(int(spec.radial_segments)),
            str(int(spec.cap_rings)),
        )
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]
    return f"cs_tip_{digest}_r{_mm_token(spec.radius)}_h{_mm_token(spec.height)}.obj"


def cs_tip_obj_text(spec: ProceduralCsTipSpec) -> str:
    r"""生成 flat-base cylinder + upper-hemisphere `cs` fingertip OBJ 文本。"""

    vertices, faces = _cs_tip_vertices_and_faces(spec)
    lines = [
        "# AnyMani procedural cs fingertip mesh",
        f"# radius_m={_fmt_float(spec.radius)}",
        f"# height_m={_fmt_float(spec.height)}",
        f"# cs_ratio={_fmt_float(spec.ratio)}",
    ]
    for x, y, z in vertices:
        lines.append(f"v {_fmt_float(x)} {_fmt_float(y)} {_fmt_float(z)}")
    for face in faces:
        lines.append("f " + " ".join(str(index) for index in face))
    return "\n".join(lines) + "\n"


def _materialize_procedural_mesh_joint(
    joint: JointCfg,
    spec: ProceduralCsTipSpec,
    *,
    mesh_root_dir: Path,
    write_enabled: bool,
) -> tuple[JointCfg, list[Path]]:
    r"""把 builder 阶段的 procedural URI 写成真实 mesh path。"""

    mesh_path, written = materialize_procedural_cs_tip_mesh(spec, mesh_root_dir, write_enabled=write_enabled)
    collisions = [_replace_procedural_mesh_element(collision, mesh_path=mesh_path) for collision in joint.collisions]
    visuals = [_replace_procedural_mesh_element(visual, mesh_path=mesh_path) for visual in joint.visuals]
    return (
        joint.replace(
            collisions=collisions,
            visuals=visuals,
            inertial=None,
            metadata=_cs_metadata(joint.metadata, spec, mesh_path=mesh_path),
        ),
        [mesh_path] if written else [],
    )


def _materialize_legacy_cs_joint(
    joint: JointCfg,
    spec: ProceduralCsTipSpec,
    origin: PoseCfg,
    *,
    mesh_root_dir: Path,
    write_enabled: bool,
) -> tuple[JointCfg, list[Path]]:
    r"""把历史 `cylinder + sphere` tip schema 迁移成单 mesh schema。"""

    mesh_path, written = materialize_procedural_cs_tip_mesh(spec, mesh_root_dir, write_enabled=write_enabled)
    geometry = {"type": "mesh", "file_path": str(mesh_path), "scale": (1.0, 1.0, 1.0)}
    visual_material = joint.visuals[0].material if joint.visuals else None
    collisions = [
        CollisionGeometryCfg(
            name=f"{joint.name}_mesh_col",
            geometry=geometry,
            origin=origin,
        )
    ]
    visuals = [
        VisualGeometryCfg(
            name=f"{joint.name}_mesh_vis",
            geometry=geometry,
            origin=origin,
            material=visual_material,
        )
    ]
    return (
        joint.replace(
            collisions=collisions,
            visuals=visuals,
            inertial=None,
            metadata=_cs_metadata(joint.metadata, spec, mesh_path=mesh_path, legacy_schema=True),
        ),
        [mesh_path] if written else [],
    )


def _replace_procedural_mesh_element(element, *, mesh_path: Path):
    r"""把单个 procedural mesh element 的 path 替换为真实 OBJ path。"""

    geometry = element.geometry
    if isinstance(geometry, MeshGeometryCfg) and is_procedural_cs_tip_uri(geometry.file_path):
        return element.replace(
            geometry={"type": "mesh", "file_path": str(mesh_path), "scale": (1.0, 1.0, 1.0)}
        )
    return element.copy()


def _procedural_spec_from_mesh_elements(joint: JointCfg) -> ProceduralCsTipSpec | None:
    r"""从 joint 的 mesh collision/visual 元素中读取 procedural `cs` 参数。"""

    for element in [*joint.collisions, *joint.visuals]:
        geometry = element.geometry
        if isinstance(geometry, MeshGeometryCfg) and is_procedural_cs_tip_uri(geometry.file_path):
            return parse_procedural_cs_tip_uri(geometry.file_path)
    return None


def _legacy_cs_spec_from_joint(joint: JointCfg) -> tuple[ProceduralCsTipSpec, PoseCfg] | None:
    r"""识别并解析历史 `cylinder + sphere` 形式的 `cs` fingertip。"""

    if not joint.is_tip or len(joint.collisions) < 2:
        return None
    body, cap = joint.collisions[0], joint.collisions[1]
    body_geometry = body.geometry
    cap_geometry = cap.geometry
    if body_geometry.kind != "cylinder" or cap_geometry.kind != "sphere":
        return None
    radius = float(body_geometry.radius)
    height = float(body_geometry.length)
    if not math.isclose(float(cap_geometry.radius), radius, rel_tol=0.0, abs_tol=1e-12):
        return None
    expected_cap_y = body.origin.pos[1] + height / 2.0
    if not math.isclose(cap.origin.pos[1], expected_cap_y, rel_tol=0.0, abs_tol=1e-9):
        return None
    offset = PoseCfg(
        pos=(body.origin.pos[0], body.origin.pos[1] - height / 2.0, body.origin.pos[2]),
        rpy=cap.origin.rpy,
    )
    return ProceduralCsTipSpec(radius=radius, height=height), offset


def _cs_metadata(
    metadata: dict[str, Any],
    spec: ProceduralCsTipSpec,
    *,
    mesh_path: Path,
    legacy_schema: bool = False,
) -> dict[str, Any]:
    r"""生成 procedural `cs` mesh 的 joint metadata。"""

    return {
        **dict(metadata),
        "tip_type": "cs",
        "procedural_tip_type": "cs",
        "procedural_mesh_kind": "cs_tip",
        "procedural_mesh_schema": "flat_base_cylinder_upper_hemisphere_v1",
        "procedural_mesh_path": str(mesh_path),
        "cs_radius": float(spec.radius),
        "cs_height": float(spec.height),
        "cs_ratio": float(spec.ratio),
        "cs_radial_segments": int(spec.radial_segments),
        "cs_cap_rings": int(spec.cap_rings),
        "legacy_cs_primitive_schema": bool(legacy_schema),
    }


def _cs_tip_vertices_and_faces(spec: ProceduralCsTipSpec) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    r"""离散化 `cs` fingertip 外表面，并返回 OBJ 1-based faces。"""

    radius = float(spec.radius)
    height = float(spec.height)
    radial_segments = int(spec.radial_segments)
    cap_rings = int(spec.cap_rings)
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]  # 底面中心，用于封闭 flat base
    ring_indices: list[list[int]] = []

    def add_ring(*, y: float, ring_radius: float) -> list[int]:
        r"""新增一圈位于 $(x,z)$ 平面的圆环顶点，返回 OBJ 1-based 索引。"""

        indices: list[int] = []
        for index in range(radial_segments):
            theta = 2.0 * math.pi * index / radial_segments
            vertices.append((ring_radius * math.cos(theta), y, ring_radius * math.sin(theta)))
            indices.append(len(vertices))
        return indices

    ring_indices.append(add_ring(y=0.0, ring_radius=radius))
    ring_indices.append(add_ring(y=height, ring_radius=radius))
    for ring_index in range(1, cap_rings):
        phi = 0.5 * math.pi * ring_index / cap_rings
        ring_indices.append(add_ring(y=height + radius * math.sin(phi), ring_radius=radius * math.cos(phi)))
    vertices.append((0.0, height + radius, 0.0))
    top_index = len(vertices)

    faces: list[tuple[int, ...]] = []
    base_center = 1
    base_ring = ring_indices[0]
    for index in range(radial_segments):
        current = base_ring[index]
        nxt = base_ring[(index + 1) % radial_segments]
        faces.append((base_center, current, nxt))  # 底面外法向应朝 $-y$，OBJ 顶点绕序按右手系取反

    for lower, upper in zip(ring_indices[:-1], ring_indices[1:]):
        for index in range(radial_segments):
            lower_current = lower[index]
            lower_next = lower[(index + 1) % radial_segments]
            upper_current = upper[index]
            upper_next = upper[(index + 1) % radial_segments]
            faces.append((lower_current, upper_next, lower_next))  # 侧面外法向为径向 $+r$，需使用 $\partial_y\times\partial_\theta$
            faces.append((lower_current, upper_current, upper_next))  # 同一 quad 的第二个三角形保持一致外法向

    last_ring = ring_indices[-1]
    for index in range(radial_segments):
        current = last_ring[index]
        nxt = last_ring[(index + 1) % radial_segments]
        faces.append((current, top_index, nxt))  # 顶帽外法向朝球面外侧，避免被 trimesh 判定为反向体
    return vertices, faces


def _single_query_value(query: dict[str, list[str]], key: str, default: Any | None = None) -> str:
    r"""从 `parse_qs` 结果中读取单值参数。"""

    values = query.get(key)
    if not values:
        if default is None:
            raise KeyError(f"procedural cs tip URI missing query key: {key}")
        return str(default)
    return str(values[0])


def _fmt_float(value: float) -> str:
    r"""把浮点参数格式化为 URI / OBJ 中稳定可读的短字符串。"""

    return f"{float(value):.12g}"


def _mm_token(value_m: float) -> str:
    r"""把米制长度转成文件名中的毫米 token。"""

    return str(int(round(float(value_m) * 1_000_000.0)))


__all__ = [
    "CS_TIP_PATH",
    "DEFAULT_CS_TIP_CAP_RINGS",
    "DEFAULT_CS_TIP_RADIAL_SEGMENTS",
    "PROCEDURAL_AUTHORITY",
    "PROCEDURAL_SCHEME",
    "ProceduralCsTipSpec",
    "cs_tip_mesh_filename",
    "cs_tip_obj_text",
    "is_procedural_cs_tip_uri",
    "make_procedural_cs_tip_uri",
    "materialize_hand_procedural_meshes",
    "materialize_joint_procedural_meshes",
    "materialize_procedural_cs_tip_mesh",
    "parse_procedural_cs_tip_uri",
]
