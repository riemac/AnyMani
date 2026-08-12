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
import os
import tempfile
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
    r"""扫描并物化 ``HandCfg`` 中所有 procedural 与 handedness mesh。

    这一步必须在 physics closure 之前执行，因为 `asset_physics.py` 的 `trimesh`
    backend 需要读取真实 mesh 文件，而不是读取 builder 阶段的 procedural URI。

    同时这里会：

    1. 迁移历史 sidecar 中的 two-primitive ``cs`` tip；
    2. 对 ``reflected_about_yz=True`` 的普通 custom mesh 烘焙顶点反射与面绕序；
    3. 让 palm、collision、visual 和 joint child-link 在 physics/validator/exporter
       前共同引用同一份最终物理 mesh。
    """

    materialized = hand.copy()  # 不修改 builder/mutator 输入真源
    written_paths: list[Path] = []  # 只记录本次首次发布的新文件，供候选期回滚

    # 当前 palm 主要是 primitive box，但严格整手合同允许未来 custom palm；
    # 因而在同一 physics 前边界处理 palm mesh，不把 handedness 逻辑锁死在 tip。
    palm_collisions, palm_collision_written = _materialize_reflected_mesh_elements(
        materialized.palm.collisions,
        mesh_root_dir=mesh_root_dir,
        write_enabled=write_enabled,
    )
    palm_visuals, palm_visual_written = _materialize_reflected_mesh_elements(
        materialized.palm.visuals,
        mesh_root_dir=mesh_root_dir,
        write_enabled=write_enabled,
    )
    if palm_collisions != materialized.palm.collisions or palm_visuals != materialized.palm.visuals:
        materialized.palm = materialized.palm.replace(
            collisions=palm_collisions,
            visuals=palm_visuals,
        )  # palm collision/visual 引用同一物理 handedness 下的最终 mesh
    written_paths.extend(palm_collision_written)
    written_paths.extend(palm_visual_written)

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
            "written_mesh_count": len({str(path) for path in written_paths}),
            "mesh_root_dir": str(Path(mesh_root_dir)),
        }  # provenance 描述所有 physics 前 mesh，不把 handedness mesh 误称为 cs tip
        materialized.metadata = metadata
    return materialized.replace(fingers=materialized.fingers, metadata=dict(materialized.metadata)), written_paths


def materialize_joint_procedural_meshes(
    joint: JointCfg,
    *,
    mesh_root_dir: Path,
    write_enabled: bool = True,
) -> tuple[JointCfg, list[Path]]:
    r"""物化单个 joint child-link 上的 procedural 与 handedness mesh。

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

    collisions, collision_written = _materialize_reflected_mesh_elements(
        joint.collisions,
        mesh_root_dir=mesh_root_dir,
        write_enabled=write_enabled,
    )
    visuals, visual_written = _materialize_reflected_mesh_elements(
        joint.visuals,
        mesh_root_dir=mesh_root_dir,
        write_enabled=write_enabled,
    )
    written = [*collision_written, *visual_written]  # collision/visual 同源时第二次命中稳定缓存
    if collisions == joint.collisions and visuals == joint.visuals:
        return joint, written  # 当前 joint 没有待物化 handedness mesh

    metadata = dict(joint.metadata)
    metadata["handedness_mesh_materialization"] = {
        "reflection_plane": "local_yz",
        "schema": "vertex_x_negate_reverse_winding_v1",
    }  # sidecar 保留 mesh 手性证书，便于审计最终路径的来源
    return (
        joint.replace(
            collisions=collisions,
            visuals=visuals,
            inertial=None,  # mesh 改变后必须由紧随其后的 physics closure 重建惯量
            metadata=metadata,
        ),
        written,
    )


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
            geometry={
                "type": "mesh",
                "file_path": str(mesh_path),
                "scale": (1.0, 1.0, 1.0),
                "reflected_about_yz": False,
            }
        )
    return element.copy()


def _materialize_reflected_mesh_elements(
    elements: list[Any],
    *,
    mesh_root_dir: Path,
    write_enabled: bool,
) -> tuple[list[Any], list[Path]]:
    r"""把 geometry elements 中待反射的 custom mesh 替换为最终文件。

    Args:
        elements: 同一 palm/joint 下的 collision 或 visual elements。
        mesh_root_dir: 当前 topology/run 共享 ``meshes/`` 根。
        write_enabled: ``False`` 时只解析稳定目标路径。

    Returns:
        tuple[list[Any], list[Path]]: 替换后的 elements 与本次首次发布文件。
    """

    materialized: list[Any] = []  # 保持 element 顺序，避免 owner/index 语义漂移
    written_paths: list[Path] = []  # 同一内容 hash 只会有一次首次发布
    for element in elements:
        geometry = element.geometry
        if not isinstance(geometry, MeshGeometryCfg) or not geometry.reflected_about_yz:
            materialized.append(element.copy())  # primitive/已物化 mesh 保持物理含义
            continue
        if is_procedural_cs_tip_uri(geometry.file_path):
            materialized.append(element.copy())  # 轴对称 cs 由优先级更高的 procedural joint 分支处理
            continue

        reflected_path, written = materialize_reflected_mesh_about_yz(
            geometry.file_path,
            mesh_root_dir=mesh_root_dir,
            write_enabled=write_enabled,
        )  # 顶点反射与 face 反序在 physics closure 前共同完成
        materialized.append(
            element.replace(
                geometry=geometry.replace(
                    file_path=str(reflected_path),
                    reflected_about_yz=False,
                )
            )
        )  # 新文件已经烘焙手性，标记必须清零以防 restore/exporter 二次反射
        if written:
            written_paths.append(reflected_path)
    return materialized, written_paths


def materialize_reflected_mesh_about_yz(
    file_path: str,
    *,
    mesh_root_dir: Path,
    write_enabled: bool = True,
) -> tuple[Path, bool]:
    r"""把 canonical triangle mesh 关于局部 $y$-$z$ 平面反射并原子发布。

    顶点和 triangle face 分别执行：

    $$
    \mathbf v_i'=S\mathbf v_i,\qquad
    S=\operatorname{diag}(-1,1,1),\qquad
    (i,j,k)\mapsto(i,k,j).
    $$

    Face 反序用于抵消 improper transform 的 $\det(S)=-1$，使外法向与 signed
    volume 保持正确。目标文件名由源内容 SHA-256 与反射 schema 共同决定；同内容
    mesh 可跨 collision/visual/worker 复用，同 basename 异内容不会冲突。

    Args:
        file_path (str): canonical custom mesh 的本地 STL/OBJ 路径。
        mesh_root_dir (Path): 当前 topology/run 共享 ``meshes/`` 根目录。
        write_enabled (bool): ``False`` 时只返回稳定目标路径，不实际写文件。

    Returns:
        tuple[Path, bool]: 镜像 mesh 路径，以及本次是否首次发布该文件。

    Raises:
        FileNotFoundError: source mesh 无法解析时抛出。
        ValueError: source 格式不支持，或镜像结果不满足 watertight/正体积合同时抛出。
    """

    source_path = _resolve_local_mesh_path(file_path)  # physics 前必须解析为真实本地文件
    source_bytes = source_path.read_bytes()  # 内容变化会自然生成新 cache key
    digest = hashlib.sha256(b"anymani_yz_reflect_v1\0" + source_bytes).hexdigest()[:16]
    suffix = source_path.suffix.lower()  # 输出格式保持与 canonical source 一致，便于人工巡检
    if suffix not in {".stl", ".obj"}:
        raise ValueError(f"strict handedness reflection currently supports STL/OBJ meshes, got {source_path}")
    target_path = Path(mesh_root_dir) / f"{source_path.stem}_yz_reflect_v1_{digest}{suffix}"
    if target_path.is_file() or not write_enabled:
        return target_path, False  # cache hit 与 dry-run 都不属于本候选的新写文件

    import numpy as np
    import trimesh

    source_mesh = trimesh.load(source_path, force="mesh", process=True)
    if not isinstance(source_mesh, trimesh.Trimesh) or len(source_mesh.vertices) == 0 or len(source_mesh.faces) == 0:
        raise ValueError(f"handedness reflection requires a non-empty triangle mesh: {source_path}")

    vertices = np.asarray(source_mesh.vertices, dtype=np.float64).copy()  # 顶点矩阵 $[N_v,3]$
    vertices[:, 0] *= -1.0  # $\mathbf v'=S\mathbf v$，只翻转 mesh-local $x$
    faces = np.asarray(source_mesh.faces, dtype=np.int64)[:, (0, 2, 1)].copy()  # $(i,j,k)\mapsto(i,k,j)$
    reflected_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not reflected_mesh.is_watertight:
        raise ValueError(f"reflected handedness mesh must be watertight: {source_path}")
    if not reflected_mesh.is_winding_consistent:
        raise ValueError(f"reflected handedness mesh must have consistent winding: {source_path}")
    if float(reflected_mesh.volume) <= 0.0:
        raise ValueError(f"reflected handedness mesh must preserve positive volume: {source_path}")

    exported = reflected_mesh.export(file_type=suffix.removeprefix("."))  # trimesh 统一序列化 STL/OBJ
    payload = exported.encode("utf-8") if isinstance(exported, str) else bytes(exported)
    written = _publish_bytes_once(target_path, payload)  # 并行 worker 中只允许一个首次发布者
    return target_path, written


def _resolve_local_mesh_path(file_path: str) -> Path:
    r"""按 generator/physics 同源边界解析待镜像 custom mesh 路径。"""

    if str(file_path).startswith("package://"):
        raise ValueError(f"handedness mesh materialization requires a local path, got {file_path!r}")
    raw_path = Path(file_path).expanduser()
    if raw_path.is_absolute():
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        return raw_path
    for candidate in (Path.cwd() / raw_path, Path(__file__).resolve().parent / raw_path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Unable to resolve mesh path for handedness reflection: {file_path!r}")


def _publish_bytes_once(target_path: Path, payload: bytes) -> bool:
    r"""用同目录临时文件和原子 hard-link 发布共享 mesh。

    临时文件完整写入后才尝试 ``os.link`` 到稳定目标名。并行 worker 中只有一个
    进程能创建目标 inode；其余进程命中 ``FileExistsError`` 并复用胜者文件，
    因而不会读取半写 target，也不会把复用文件误计入自身 ``written_paths``。
    """

    target_path.parent.mkdir(parents=True, exist_ok=True)  # 同目录保证 hard-link 位于同一文件系统
    temporary_path: Path | None = None  # finally 中只清理本次私有临时 inode
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)  # 先完整序列化，稳定目标路径尚不可见
            temporary.flush()
            os.fsync(temporary.fileno())  # 发布前把用户态缓冲刷新到文件系统
            temporary_path = Path(temporary.name)
        try:
            os.link(temporary_path, target_path)  # 原子 no-clobber 发布
            return True
        except FileExistsError:
            return False  # 另一 worker 已发布同内容 hash 文件，当前调用安全复用
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)  # 临时文件永远不属于资产 bundle


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
