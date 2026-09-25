"Builds signed-distance queries from collision meshes using the selected backend."

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..asset_schema_core import MeshGeometryCfg, Vector3
from ._collision_geometry import apply_inverse_pose, apply_pose

if TYPE_CHECKING:
    from ._collision_geometry import CollisionBodyRecord


MeshSdfBackend = Literal["auto", "warp", "trimesh"]
"Selected signed-distance backend for collision-mesh queries."


@dataclass
class MeshSdfQueryStats:
    "Counts and backend details for mesh signed-distance queries."

    requested_backend: MeshSdfBackend = "auto"
    actual_backend: str = "none"
    mesh_query_count: int = 0
    mesh_sample_count: int = 0
    fallback_events: list[str] = field(default_factory=list)

    def record_backend(self, backend: str) -> None:
        'Records the SDF backend used for this query.'

        if self.actual_backend == "none":
            self.actual_backend = backend
        elif self.actual_backend != backend:
            self.actual_backend = "mixed"

    def to_dict(self) -> dict[str, object]:
        'Serializes the typed object as a dictionary.'

        return {
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "mesh_query_count": self.mesh_query_count,
            "mesh_sample_count": self.mesh_sample_count,
            "fallback_events": list(self.fallback_events),
        }


@dataclass(frozen=True)
class _WarpMeshHandle:
    "Cached Warp mesh state used by the signed-distance service."

    mesh: Any
    points: Any
    indices: Any


_WARP_MESH_CACHE_MAXSIZE = 128
_WARP_MESH_CACHE: OrderedDict[
    tuple[str, str, tuple[float, float, float], str],
    _WarpMeshHandle,
] = OrderedDict()
"Cached Warp mesh handles indexed by exact source mesh identity."


def is_mesh_body(body: CollisionBodyRecord) -> bool:
    "Reports whether a collision body has a local mesh representation."

    return isinstance(body.geometry, MeshGeometryCfg)


def sample_mesh_surface(body: CollisionBodyRecord, *, sample_count: int) -> list[Vector3]:
    'Selects deterministic sample points on the mesh surface.'

    if not isinstance(body.geometry, MeshGeometryCfg):
        raise TypeError(f"sample_mesh_surface expects MeshGeometryCfg, got {type(body.geometry).__name__}")


    mesh_path, mesh_sha256 = _mesh_identity(body.geometry.file_path)
    mesh = _load_checked_trimesh(mesh_path, _scale_tuple(body.geometry.scale), mesh_sha256)


    seed = _stable_seed(body.body_path, mesh_path, mesh_sha256, repr(body.geometry.scale))
    from trimesh.sample import sample_surface

    sampled = sample_surface(mesh, sample_count, seed=seed)
    local_points = sampled[0]


    return [apply_pose(body.world_pose, _to_vector3(point)) for point in local_points]


def signed_distance_to_mesh_body(
    point_world: Vector3,
    body: CollisionBodyRecord,
    *,
    backend: MeshSdfBackend,
    device: str,
    stats: MeshSdfQueryStats | None = None,
) -> float:
    "Queries signed distance from points to one transformed collision mesh."

    return float(
        signed_distance_to_mesh_body_batch(
            [point_world],
            body,
            backend=backend,
            device=device,
            stats=stats,
        )[0]
    )


def signed_distance_to_mesh_body_batch(
    points_world: list[Vector3],
    body: CollisionBodyRecord,
    *,
    backend: MeshSdfBackend,
    device: str,
    stats: MeshSdfQueryStats | None = None,
) -> np.ndarray:
    "Batches signed-distance queries while retaining point and body indices."

    if not isinstance(body.geometry, MeshGeometryCfg):
        raise TypeError(f"signed_distance_to_mesh_body_batch expects MeshGeometryCfg, got {type(body.geometry).__name__}")
    if not points_world:
        return np.empty((0,), dtype=np.float64)


    points_local = np.asarray([apply_inverse_pose(body.world_pose, point) for point in points_world], dtype=np.float32)


    if backend in {"auto", "warp"} and device == "cuda":
        try:
            return _signed_distance_warp(points_local, body.geometry, stats=stats)
        except Exception as exc:
            if backend == "warp":
                raise RuntimeError(f"Warp mesh SDF failed for {body.body_path}: {exc}") from exc
            if stats is not None:
                stats.fallback_events.append(f"{body.body_path}: warp→trimesh fallback because {type(exc).__name__}: {exc}")


    return _signed_distance_trimesh(points_local, body.geometry, stats=stats)


def _signed_distance_trimesh(
    points_local: np.ndarray,
    geometry: MeshGeometryCfg,
    *,
    stats: MeshSdfQueryStats | None,
) -> np.ndarray:

    mesh_path, mesh_sha256 = _mesh_identity(geometry.file_path)
    query = _get_trimesh_proximity_query(mesh_path, _scale_tuple(geometry.scale), mesh_sha256)
    if stats is not None:
        stats.record_backend("trimesh")
        stats.mesh_query_count += 1
    return -np.asarray(query.signed_distance(points_local), dtype=np.float64)


@lru_cache(maxsize=128)
def _get_trimesh_proximity_query(
    file_path: str,
    scale: tuple[float, float, float],
    content_sha256: str,
):

    from trimesh.proximity import ProximityQuery

    return ProximityQuery(_load_checked_trimesh(file_path, scale, content_sha256))


def _signed_distance_warp(
    points_local: np.ndarray,
    geometry: MeshGeometryCfg,
    *,
    stats: MeshSdfQueryStats | None,
) -> np.ndarray:

    wp = _require_warp_cuda()
    device = "cuda:0"
    handle = _get_warp_mesh_handle(geometry, device=device)
    query_points = wp.array(points_local, dtype=wp.vec3, device=device)  # [N, 3] local query points
    distances = wp.zeros(points_local.shape[0], dtype=float, device=device)  # [N] signed distances


    wp.launch(_warp_mesh_signed_distance_kernel, dim=points_local.shape[0], inputs=[handle.mesh.id, query_points, distances], device=device)
    wp.synchronize_device(device)

    if stats is not None:
        stats.record_backend("warp")
        stats.mesh_query_count += 1
    return np.asarray(distances.numpy(), dtype=np.float64)


def _get_warp_mesh_handle(geometry: MeshGeometryCfg, *, device: str) -> _WarpMeshHandle:

    wp = _require_warp_cuda()
    path, mesh_sha256 = _mesh_identity(geometry.file_path)
    scale = _scale_tuple(geometry.scale)
    key = (path, mesh_sha256, scale, device)
    if key in _WARP_MESH_CACHE:
        _WARP_MESH_CACHE.move_to_end(key)
        return _WARP_MESH_CACHE[key]

    mesh = _load_checked_trimesh(path, scale, mesh_sha256)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
    points = wp.array(vertices, dtype=wp.vec3, device=device)
    indices = wp.array(faces, dtype=wp.int32, device=device)
    handle = _WarpMeshHandle(mesh=wp.Mesh(points, indices), points=points, indices=indices)
    _remember_warp_mesh_handle(key, handle)
    return handle


def _remember_warp_mesh_handle(
    key: tuple[str, str, tuple[float, float, float], str],
    handle: _WarpMeshHandle,
) -> None:

    _WARP_MESH_CACHE[key] = handle
    _WARP_MESH_CACHE.move_to_end(key)
    if len(_WARP_MESH_CACHE) > _WARP_MESH_CACHE_MAXSIZE:
        _WARP_MESH_CACHE.popitem(last=False)


@lru_cache(maxsize=128)
def _load_checked_trimesh(
    file_path: str,
    scale: tuple[float, float, float],
    content_sha256: str,
):

    import trimesh

    path = _resolve_mesh_path(file_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != content_sha256:
        raise ValueError(f"mesh bytes changed while constructing SDF cache entry: {path}")
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"mesh SDF expects a triangle mesh, got {type(mesh).__name__}: {path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"mesh SDF got empty mesh: {path}")

    mesh = mesh.copy()
    mesh.apply_scale(scale)
    if not mesh.is_watertight:
        raise ValueError(f"mesh SDF requires watertight mesh after trimesh processing: {path}")
    if not mesh.is_winding_consistent:
        raise ValueError(f"mesh SDF requires winding-consistent mesh after trimesh processing: {path}")
    return mesh


def _mesh_identity(file_path: str) -> tuple[str, str]:

    path = _resolve_mesh_path(file_path)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_mesh_path(file_path: str) -> Path:

    raw_path = Path(file_path).expanduser()
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path.resolve()
    if raw_path.exists():
        return raw_path.resolve()
    assets_root = Path(__file__).resolve().parents[1]
    candidate = assets_root / raw_path
    if candidate.exists():
        return candidate.resolve()
    raise ValueError(f"mesh file does not exist for SDF validator: {file_path!r}")


def _require_warp_cuda():

    try:
        import warp as wp
    except Exception as exc:
        raise RuntimeError("warp-lang is not importable") from exc

    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("Warp CUDA device is unavailable")
    return wp


def _scale_tuple(scale: Vector3) -> tuple[float, float, float]:

    return (float(scale[0]), float(scale[1]), float(scale[2]))


def _stable_seed(*parts: str) -> int:

    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _to_vector3(point: np.ndarray) -> Vector3:

    return (float(point[0]), float(point[1]), float(point[2]))


try:
    import warp as wp

    @wp.kernel
    def _warp_mesh_signed_distance_kernel(
        mesh: wp.uint64,  # pyright: ignore[reportInvalidTypeForm]
        points: wp.array(dtype=wp.vec3),  # pyright: ignore[reportInvalidTypeForm]
        distances: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    ):
        r"""Warp kernel that batches signed-distance queries from points to one mesh."""

        tid = wp.tid()
        point = points[tid]
        query = wp.mesh_query_point(mesh, point, 1.0e8)  # pyright: ignore[reportArgumentType]
        if not query.result:  # pyright: ignore[reportAttributeAccessIssue]
            distances[tid] = 3.4028234663852886e38
            return
        closest = wp.mesh_eval_position(  # pyright: ignore[reportAttributeAccessIssue]
            mesh,
            query.face,  # pyright: ignore[reportAttributeAccessIssue]
            query.u,  # pyright: ignore[reportAttributeAccessIssue]
            query.v,  # pyright: ignore[reportAttributeAccessIssue]
        )
        distance = wp.length(closest - point)
        if query.sign >= 0.0:  # pyright: ignore[reportAttributeAccessIssue]
            distances[tid] = distance
        else:
            distances[tid] = -distance

except Exception:

    _warp_mesh_signed_distance_kernel = None  # pyright: ignore[reportAssignmentType]


__all__ = [
    "MeshSdfBackend",
    "MeshSdfQueryStats",
    "is_mesh_body",
    "sample_mesh_surface",
    "signed_distance_to_mesh_body",
    "signed_distance_to_mesh_body_batch",
]
