"Materializes procedural tip URIs before validation and physics closure. The cs tip remains a procedural recipe and is exported as one closed collision surface."

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
"Versioned recipe name for a procedural mesh shape."

PROCEDURAL_AUTHORITY = "anymani"
"Identity label for the procedural source of a generated mesh."

CS_TIP_PATH = "/cs_tip"
"Default output path for a procedural cylindrical-segment tip mesh."

DEFAULT_CS_TIP_RADIAL_SEGMENTS = 32
"Default angular tessellation count for a procedural tip."

DEFAULT_CS_TIP_CAP_RINGS = 8
"Default radial layers used to generate tip end caps."

_MIN_POSITIVE_LENGTH = 1e-9
"Smallest positive link dimension accepted by the builders."


@dataclass(frozen=True)
class ProceduralCsTipSpec:
    "Parsed procedural tip dimensions and stable geometry identity."

    radius: float
    "Primitive radius in meters."

    height: float
    "Link cross-section dimension in meters."

    radial_segments: int = DEFAULT_CS_TIP_RADIAL_SEGMENTS
    "Angular tessellation count around a procedural cylinder or tip."

    cap_rings: int = DEFAULT_CS_TIP_CAP_RINGS
    "Number of radial layers used to close procedural mesh end caps."

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

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
        "Returns the declared cylinder height-to-radius ratio."

        return float(self.height) / float(self.radius)


def make_procedural_cs_tip_uri(
    *,
    radius: float,
    height: float,
    radial_segments: int = DEFAULT_CS_TIP_RADIAL_SEGMENTS,
    cap_rings: int = DEFAULT_CS_TIP_CAP_RINGS,
) -> str:
    'Builds procedural cs tip uri.'

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
    "Reports whether a URDF URI uses the supported procedural-tip scheme."

    parsed = urlparse(str(file_path))
    return parsed.scheme == PROCEDURAL_SCHEME and parsed.netloc == PROCEDURAL_AUTHORITY and parsed.path == CS_TIP_PATH


def parse_procedural_cs_tip_uri(file_path: str) -> ProceduralCsTipSpec:
    'Loads and validates procedural cs tip uri.'

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
    'Materializes procedural cs tip mesh.'

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
    'Materializes hand procedural meshes.'

    materialized = hand.copy()
    written_paths: list[Path] = []



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
        )
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
        }
        materialized.metadata = metadata
    return materialized.replace(fingers=materialized.fingers, metadata=dict(materialized.metadata)), written_paths


def materialize_joint_procedural_meshes(
    joint: JointCfg,
    *,
    mesh_root_dir: Path,
    write_enabled: bool = True,
) -> tuple[JointCfg, list[Path]]:
    'Materializes joint procedural meshes.'

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
    written = [*collision_written, *visual_written]
    if collisions == joint.collisions and visuals == joint.visuals:
        return joint, written

    metadata = dict(joint.metadata)
    metadata["handedness_mesh_materialization"] = {
        "reflection_plane": "local_yz",
        "schema": "vertex_x_negate_reverse_winding_v1",
    }
    return (
        joint.replace(
            collisions=collisions,
            visuals=visuals,
            inertial=None,
            metadata=metadata,
        ),
        written,
    )


def cs_tip_mesh_filename(spec: ProceduralCsTipSpec) -> str:
    "Returns the content-derived filename for a procedural tip mesh."

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
    "Serializes the procedural tip surface into OBJ coordinates measured in meters."

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

    materialized: list[Any] = []
    written_paths: list[Path] = []
    for element in elements:
        geometry = element.geometry
        if not isinstance(geometry, MeshGeometryCfg) or not geometry.reflected_about_yz:
            materialized.append(element.copy())
            continue
        if is_procedural_cs_tip_uri(geometry.file_path):
            materialized.append(element.copy())
            continue

        reflected_path, written = materialize_reflected_mesh_about_yz(
            geometry.file_path,
            mesh_root_dir=mesh_root_dir,
            write_enabled=write_enabled,
        )
        materialized.append(
            element.replace(
                geometry=geometry.replace(
                    file_path=str(reflected_path),
                    reflected_about_yz=False,
                )
            )
        )
        if written:
            written_paths.append(reflected_path)
    return materialized, written_paths


def materialize_reflected_mesh_about_yz(
    file_path: str,
    *,
    mesh_root_dir: Path,
    write_enabled: bool = True,
) -> tuple[Path, bool]:
    'Materializes reflected mesh about yz.'

    source_path = _resolve_local_mesh_path(file_path)
    source_bytes = source_path.read_bytes()
    digest = hashlib.sha256(b"anymani_yz_reflect_v1\0" + source_bytes).hexdigest()[:16]
    suffix = source_path.suffix.lower()
    if suffix not in {".stl", ".obj"}:
        raise ValueError(f"strict handedness reflection currently supports STL/OBJ meshes, got {source_path}")
    target_path = Path(mesh_root_dir) / f"{source_path.stem}_yz_reflect_v1_{digest}{suffix}"
    if target_path.is_file() or not write_enabled:
        return target_path, False

    import numpy as np
    import trimesh

    source_mesh = trimesh.load(source_path, force="mesh", process=True)
    if not isinstance(source_mesh, trimesh.Trimesh) or len(source_mesh.vertices) == 0 or len(source_mesh.faces) == 0:
        raise ValueError(f"handedness reflection requires a non-empty triangle mesh: {source_path}")

    vertices = np.asarray(source_mesh.vertices, dtype=np.float64).copy()
    vertices[:, 0] *= -1.0
    faces = np.asarray(source_mesh.faces, dtype=np.int64)[:, (0, 2, 1)].copy()
    reflected_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not reflected_mesh.is_watertight:
        raise ValueError(f"reflected handedness mesh must be watertight: {source_path}")
    if not reflected_mesh.is_winding_consistent:
        raise ValueError(f"reflected handedness mesh must have consistent winding: {source_path}")
    if float(reflected_mesh.volume) <= 0.0:
        raise ValueError(f"reflected handedness mesh must preserve positive volume: {source_path}")

    exported = reflected_mesh.export(file_type=suffix.removeprefix("."))
    payload = exported.encode("utf-8") if isinstance(exported, str) else bytes(exported)
    written = _publish_bytes_once(target_path, payload)
    return target_path, written


def _resolve_local_mesh_path(file_path: str) -> Path:

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

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.link(temporary_path, target_path)
            return True
        except FileExistsError:
            return False
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _procedural_spec_from_mesh_elements(joint: JointCfg) -> ProceduralCsTipSpec | None:

    for element in [*joint.collisions, *joint.visuals]:
        geometry = element.geometry
        if isinstance(geometry, MeshGeometryCfg) and is_procedural_cs_tip_uri(geometry.file_path):
            return parse_procedural_cs_tip_uri(geometry.file_path)
    return None


def _legacy_cs_spec_from_joint(joint: JointCfg) -> tuple[ProceduralCsTipSpec, PoseCfg] | None:

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

    radius = float(spec.radius)
    height = float(spec.height)
    radial_segments = int(spec.radial_segments)
    cap_rings = int(spec.cap_rings)
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    ring_indices: list[list[int]] = []

    def add_ring(*, y: float, ring_radius: float) -> list[int]:
        "Adds one ring of vertices with stable winding to the procedural surface."

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
        faces.append((base_center, current, nxt))

    for lower, upper in zip(ring_indices[:-1], ring_indices[1:]):
        for index in range(radial_segments):
            lower_current = lower[index]
            lower_next = lower[(index + 1) % radial_segments]
            upper_current = upper[index]
            upper_next = upper[(index + 1) % radial_segments]
            faces.append((lower_current, upper_next, lower_next))
            faces.append((lower_current, upper_current, upper_next))

    last_ring = ring_indices[-1]
    for index in range(radial_segments):
        current = last_ring[index]
        nxt = last_ring[(index + 1) % radial_segments]
        faces.append((current, top_index, nxt))
    return vertices, faces


def _single_query_value(query: dict[str, list[str]], key: str, default: Any | None = None) -> str:

    values = query.get(key)
    if not values:
        if default is None:
            raise KeyError(f"procedural cs tip URI missing query key: {key}")
        return str(default)
    return str(values[0])


def _fmt_float(value: float) -> str:

    return f"{float(value):.12g}"


def _mm_token(value_m: float) -> str:

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
