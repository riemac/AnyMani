"""Collision unions and home-surface samples assigned to semantic owners."""


from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any

import numpy as np
import trimesh

from anymani.assets.asset_schema_geometry import (
    CollisionComponentSemanticsCfg,
    HandGeometrySemanticsCfg,
)
from anymani.assets.bank import HandContainer

from .kinematics import EmbodimentGeometrySpec


@dataclass(frozen=True)
class OwnerSurfaceRecord:


    owner_id: str
    owner_index: int
    role: str  # palm/joint/tip
    finger_name: str | None
    component_ids: tuple[str, ...]
    surface_mesh: trimesh.Trimesh
    solid_mesh: trimesh.Trimesh | None
    boolean_applied: bool


@dataclass(frozen=True)
class OwnerGeometryCache:


    asset_id: str
    asset_content_hash: str
    boolean_backend: str
    records: tuple[OwnerSurfaceRecord, ...]
    surface_geometry_hash: str = ""
    surface_processing_version: str = "owner-surface-v2"


@dataclass(frozen=True)
class HomeSurfaceSamples:


    owner_ids: tuple[str, ...]
    points_owner_local_m: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray
    sampling_seed: int
    oversample_factor: int


@dataclass(frozen=True)
class WarpOwnerMeshHandle:


    owner_id: str
    mesh: object
    points: object
    indices: object
    source_face_indices: object
    face_altitudes: object
    face_count: int
    surface_audit: WarpSurfaceAudit


@dataclass(frozen=True)
class WarpOwnerGeometryCache:


    asset_id: str
    asset_content_hash: str
    surface_geometry_hash: str
    surface_processing_version: str
    device: str
    handles: tuple[WarpOwnerMeshHandle, ...]


@dataclass(frozen=True)
class WarpSurfaceAudit:


    input_face_count: int
    output_face_count: int
    removed_face_count: int
    input_area_m2: float
    removed_area_m2: float
    removed_area_fraction: float


@dataclass(frozen=True)
class WarpSurfaceView:


    vertices: np.ndarray
    faces: np.ndarray  # `[F_valid,3]` int32
    source_face_indices: np.ndarray
    face_altitudes_m: np.ndarray
    audit: WarpSurfaceAudit


@dataclass(frozen=True)
class OwnerSurfaceSamplingArrays:


    vertices_owner_local_m: tuple[np.ndarray, ...]
    faces: tuple[np.ndarray, ...]
    face_normals_owner_local: tuple[np.ndarray, ...]
    face_area_cdf: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class GeometryIdentity:


    physical_geometry_hash: str
    configuration_domain_hash: str


_WarpCacheKey = tuple[str, str, str]
_WARP_OWNER_CACHE: dict[_WarpCacheKey, WarpOwnerGeometryCache] = {}
"""Reuse each BVH by surface hash, processing version, and device."""

_WARP_OWNER_CACHE_LEASES: dict[_WarpCacheKey, int] = {}
"""Active resident-window leases per GPU cache entry; evict at zero."""

_WARP_OWNER_CACHE_LOCK = RLock()
"""Protect cache and lease metadata; construct BVHs outside this lock."""

def materialize_owner_geometry_cache(
    container: HandContainer,
    spec: EmbodimentGeometrySpec,
) -> OwnerGeometryCache:


    semantics = container.geometry_semantics
    if semantics is None:
        raise ValueError("HandContainer must be resolved with require_geometry_semantics=True")
    if spec.owner_ids != tuple(owner.owner_id for owner in semantics.owners):
        raise ValueError("EmbodimentGeometrySpec owner axis does not match container geometry semantics")
    if spec.component_owner_local_transforms is None:
        raise ValueError("EmbodimentGeometrySpec is missing component_owner_local_transforms")
    if spec.component_owner_local_transforms.shape[0] != len(semantics.components):
        raise ValueError("component transform axis does not match geometry semantics")

    transformed_by_owner: dict[str, list[trimesh.Trimesh]] = {
        owner.owner_id: [] for owner in semantics.owners
    }
    component_index_by_id = {
        component.component_id: component_index
        for component_index, component in enumerate(semantics.components)
    }
    for component_index, component in enumerate(semantics.components):
        mesh = _component_mesh(component, container=container)
        transform = spec.component_owner_local_transforms[component_index].detach().cpu().numpy()
        mesh.apply_transform(transform)  # collision local -> owner reference link
        _require_surface(mesh, context=f"component '{component.component_id}'")
        transformed_by_owner[component.owner_id].append(mesh)

    records: list[OwnerSurfaceRecord] = []
    for owner in semantics.owners:
        component_meshes = transformed_by_owner[owner.owner_id]
        if len(component_meshes) != len(owner.component_ids):
            raise ValueError(f"owner '{owner.owner_id}' component materialization is incomplete")
        all_components_are_solid = all(_is_volume(mesh) for mesh in component_meshes)
        solid_mesh = (
            strict_owner_union(component_meshes, owner_id=owner.owner_id)
            if all_components_are_solid
            else None
        )
        surface_mesh = (
            solid_mesh.copy()
            if solid_mesh is not None
            else _concatenate_owner_surfaces(component_meshes, owner_id=owner.owner_id)
        )
        records.append(
            OwnerSurfaceRecord(
                owner_id=owner.owner_id,
                owner_index=owner.owner_index,
                role=owner.role,
                finger_name=getattr(owner, "finger_name", None),
                component_ids=tuple(owner.component_ids),
                surface_mesh=surface_mesh,
                solid_mesh=solid_mesh,
                boolean_applied=solid_mesh is not None and len(component_meshes) > 1,
            )
        )


    if tuple(component_index_by_id) != tuple(component.component_id for component in semantics.components):
        raise ValueError("component provenance order is inconsistent")
    surface_geometry_hash = _owner_surface_geometry_hash(tuple(records))
    return OwnerGeometryCache(
        asset_id=container.asset_id,
        asset_content_hash=semantics.content_hash,
        boolean_backend=f"manifold3d=={version('manifold3d')}",
        records=tuple(records),
        surface_geometry_hash=surface_geometry_hash,
    )


def materialize_warp_owner_geometry_cache(
    cache: OwnerGeometryCache,
    *,
    device: str = "cuda:0",
    surface_views: tuple[WarpSurfaceView, ...] | None = None,
) -> WarpOwnerGeometryCache:


    surface_hash = cache.surface_geometry_hash or _owner_surface_geometry_hash(cache.records)
    key = (surface_hash, cache.surface_processing_version, device)
    with _WARP_OWNER_CACHE_LOCK:
        existing = _WARP_OWNER_CACHE.get(key)
        if existing is not None:
            _WARP_OWNER_CACHE_LEASES[key] = int(_WARP_OWNER_CACHE_LEASES.get(key, 0)) + 1
            return existing

    try:
        import warp as wp
    except Exception as exc:
        raise RuntimeError("Warp is required for the online owner-surface target backend") from exc
    wp.init()
    resolved_device = wp.get_device(device)
    if resolved_device.is_cuda and not wp.is_cuda_available():
        raise RuntimeError(f"Warp CUDA device is unavailable: {device}")

    if surface_views is not None and len(surface_views) != len(cache.records):
        raise ValueError("precomputed Warp surface view count must match owner records")
    handles: list[WarpOwnerMeshHandle] = []
    for owner_index, record in enumerate(cache.records):
        surface_view = (
            surface_views[owner_index]
            if surface_views is not None
            else prepare_warp_surface_view(record.surface_mesh, owner_id=record.owner_id)
        )
        vertices = surface_view.vertices
        faces_2d = surface_view.faces
        faces = faces_2d.reshape(-1)
        points = wp.array(vertices, dtype=wp.vec3, device=device)
        indices = wp.array(faces, dtype=wp.int32, device=device)
        source_face_indices = wp.array(surface_view.source_face_indices, dtype=wp.int32, device=device)
        face_altitudes = wp.array(
            surface_view.face_altitudes_m,
            dtype=wp.vec3,
            device=device,
        )
        mesh = wp.Mesh(
            points,
            indices,
            support_winding_number=record.owner_id == "palm",
        )
        handles.append(
            WarpOwnerMeshHandle(
                owner_id=record.owner_id,
                mesh=mesh,
                points=points,
                indices=indices,
                source_face_indices=source_face_indices,
                face_altitudes=face_altitudes,
                face_count=len(faces_2d),
                surface_audit=surface_view.audit,
            )
        )
    result = WarpOwnerGeometryCache(
        asset_id=cache.asset_id,
        asset_content_hash=cache.asset_content_hash,
        surface_geometry_hash=surface_hash,
        surface_processing_version=cache.surface_processing_version,
        device=device,
        handles=tuple(handles),
    )
    with _WARP_OWNER_CACHE_LOCK:
        existing = _WARP_OWNER_CACHE.get(key)
        if existing is not None:
            _WARP_OWNER_CACHE_LEASES[key] = int(_WARP_OWNER_CACHE_LEASES.get(key, 0)) + 1
            return existing
        _WARP_OWNER_CACHE[key] = result
        _WARP_OWNER_CACHE_LEASES[key] = 1
        return result


def release_warp_owner_geometry_cache(cache: WarpOwnerGeometryCache) -> bool:


    key = (cache.surface_geometry_hash, cache.surface_processing_version, cache.device)
    with _WARP_OWNER_CACHE_LOCK:
        leases = _WARP_OWNER_CACHE_LEASES.get(key)
        if leases is None:
            raise KeyError("Warp owner geometry cache is not resident or has already been released")
        if leases > 1:
            _WARP_OWNER_CACHE_LEASES[key] = leases - 1
            return False
        _WARP_OWNER_CACHE_LEASES.pop(key, None)
        _WARP_OWNER_CACHE.pop(key, None)
        return True


def warp_owner_geometry_cache_stats() -> dict[str, int]:


    with _WARP_OWNER_CACHE_LOCK:
        return {
            "entry_count": len(_WARP_OWNER_CACHE),
            "owner_mesh_count": sum(len(cache.handles) for cache in _WARP_OWNER_CACHE.values()),
            "lease_count": sum(_WARP_OWNER_CACHE_LEASES.values()),
        }


def geometry_identity(
    semantics: HandGeometrySemanticsCfg,
    spec: EmbodimentGeometrySpec,
    cache: OwnerGeometryCache,
) -> GeometryIdentity:


    if spec.joint_limits is None:
        raise ValueError("geometry identity requires explicit joint limits for configuration-domain provenance")
    if tuple(spec.joint_names) != tuple(semantics.active_joint_names):
        raise ValueError("geometry identity joint axis does not match geometry semantics")
    if tuple(spec.owner_ids) != tuple(owner.owner_id for owner in semantics.owners):
        raise ValueError("geometry identity owner axis does not match geometry semantics")
    surface_hash = cache.surface_geometry_hash or _owner_surface_geometry_hash(cache.records)

    physical = hashlib.sha256()
    physical.update(b"physical-geometry-v1\0")
    _hash_strings(physical, spec.joint_names)
    _hash_strings(physical, spec.owner_ids)
    _hash_strings(physical, tuple(owner.role for owner in semantics.owners))
    _hash_tensor(physical, spec.space_screws, floating=True)
    _hash_tensor(physical, spec.q_home, floating=True)
    _hash_tensor(physical, spec.owner_home_transforms, floating=True)
    _hash_tensor(physical, spec.owner_ancestor_mask, floating=False)
    _hash_tensor(physical, spec.joint_ancestor_mask, floating=False)
    for optional in (
        spec.owner_parent_indices,
        spec.owner_graph_shortest,
        spec.owner_graph_parent,
        spec.owner_graph_child,
        spec.component_owner_indices,
    ):
        _hash_optional_tensor(physical, optional, floating=False)
    _hash_optional_tensor(physical, spec.component_owner_local_transforms, floating=True)
    physical.update(bytes.fromhex(surface_hash))

    domain = hashlib.sha256()
    domain.update(b"configuration-domain-v1\0")
    _hash_strings(domain, spec.joint_names)
    _hash_tensor(domain, spec.joint_limits, floating=True)
    return GeometryIdentity(
        physical_geometry_hash=physical.hexdigest(),
        configuration_domain_hash=domain.hexdigest(),
    )


def prepare_warp_surface_view(
    surface_mesh: trimesh.Trimesh,
    *,
    owner_id: str,
    max_area_loss_fraction: float = 1.0e-8,
) -> WarpSurfaceView:


    if not 0.0 <= max_area_loss_fraction <= 1.0:
        raise ValueError("max_area_loss_fraction must lie in [0,1]")
    _require_surface(surface_mesh, context=f"owner '{owner_id}' surface")
    vertices64 = np.asarray(surface_mesh.vertices, dtype=np.float64)
    faces = np.asarray(surface_mesh.faces, dtype=np.int32)
    triangles64 = vertices64[faces]
    cross64 = np.cross(triangles64[:, 1] - triangles64[:, 0], triangles64[:, 2] - triangles64[:, 0])
    area64 = 0.5 * np.linalg.norm(cross64, axis=-1)
    total_area = float(area64.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError(f"owner '{owner_id}' surface has zero or non-finite total area")

    vertices32 = vertices64.astype(np.float32)
    triangles32 = vertices32[faces]
    edges32 = np.stack(
        (
            triangles32[:, 1] - triangles32[:, 0],
            triangles32[:, 2] - triangles32[:, 1],
            triangles32[:, 0] - triangles32[:, 2],
        ),
        axis=1,
    )
    squared_edge_lengths32 = np.sum(edges32 * edges32, axis=-1)
    doubled_area32 = np.linalg.norm(np.cross(edges32[:, 0], -edges32[:, 2]), axis=-1)
    valid = np.all(squared_edge_lengths32 > 0.0, axis=-1) & (doubled_area32 > 0.0)
    removed_area = float(area64[~valid].sum())
    removed_fraction = removed_area / total_area
    if removed_fraction > max_area_loss_fraction:
        raise ValueError(
            f"owner '{owner_id}' float32 surface area-loss budget exceeded: "
            f"removed_fraction={removed_fraction:.12g}, budget={max_area_loss_fraction:.12g}"
        )
    if not np.any(valid):
        raise ValueError(f"owner '{owner_id}' has no valid float32 triangles after surface filtering")

    valid_triangles32 = triangles32[valid]
    valid_doubled_area32 = doubled_area32[valid]
    opposite_edges32 = np.stack(
        (
            np.linalg.norm(valid_triangles32[:, 2] - valid_triangles32[:, 1], axis=-1),
            np.linalg.norm(valid_triangles32[:, 0] - valid_triangles32[:, 2], axis=-1),
            np.linalg.norm(valid_triangles32[:, 1] - valid_triangles32[:, 0], axis=-1),
        ),
        axis=-1,
    )
    if np.any(opposite_edges32 <= 0.0):
        raise RuntimeError(f"owner '{owner_id}' float32 surface filtering left a zero opposite edge")
    audit = WarpSurfaceAudit(
        input_face_count=len(faces),
        output_face_count=int(valid.sum()),
        removed_face_count=int((~valid).sum()),
        input_area_m2=total_area,
        removed_area_m2=removed_area,
        removed_area_fraction=removed_fraction,
    )
    return WarpSurfaceView(
        vertices=np.ascontiguousarray(vertices32),
        faces=np.ascontiguousarray(faces[valid]),
        source_face_indices=np.ascontiguousarray(np.flatnonzero(valid).astype(np.int32)),
        face_altitudes_m=np.ascontiguousarray(
            (valid_doubled_area32[:, None] / opposite_edges32).astype(np.float32)
        ),
        audit=audit,
    )


def prepare_owner_surface_sampling_arrays(cache: OwnerGeometryCache) -> OwnerSurfaceSamplingArrays:


    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    cdfs: list[np.ndarray] = []
    for record in cache.records:
        surface = record.surface_mesh
        vertex = np.ascontiguousarray(np.asarray(surface.vertices, dtype=np.float64))
        face = np.ascontiguousarray(np.asarray(surface.faces, dtype=np.int32))
        normal = np.ascontiguousarray(np.asarray(surface.face_normals, dtype=np.float64))
        area = np.asarray(surface.area_faces, dtype=np.float64)
        if vertex.ndim != 2 or vertex.shape[1:] != (3,) or face.ndim != 2 or face.shape[1:] != (3,) or np.any(area <= 0.0):
            raise ValueError(f"owner {record.owner_id!r} surface sampling arrays require positive triangles")
        cdf = np.cumsum(area / area.sum(), dtype=np.float64)
        cdf[-1] = 1.0
        vertices.append(vertex)
        faces.append(face)
        normals.append(normal)
        cdfs.append(np.ascontiguousarray(cdf))
    return OwnerSurfaceSamplingArrays(tuple(vertices), tuple(faces), tuple(normals), tuple(cdfs))


def strict_owner_union(meshes: list[trimesh.Trimesh], *, owner_id: str) -> trimesh.Trimesh:


    if not meshes:
        raise ValueError(f"owner '{owner_id}' has no collision solids")
    for component_index, mesh in enumerate(meshes):
        _require_volume(mesh, context=f"owner '{owner_id}' component[{component_index}]")
    if len(meshes) == 1:
        return meshes[0].copy()

    try:
        union = trimesh.boolean.union(meshes, engine="manifold", check_volume=True)
    except Exception as exc:
        raise ValueError(f"strict Boolean union failed for owner '{owner_id}': {exc}") from exc
    if not isinstance(union, trimesh.Trimesh):
        raise ValueError(f"strict Boolean union for owner '{owner_id}' did not return a single Trimesh")
    union.remove_unreferenced_vertices()
    _require_volume(union, context=f"owner '{owner_id}' union")
    return union


def _concatenate_owner_surfaces(meshes: list[trimesh.Trimesh], *, owner_id: str) -> trimesh.Trimesh:


    if not meshes:
        raise ValueError(f"owner '{owner_id}' has no collision surfaces")
    surface = meshes[0].copy() if len(meshes) == 1 else trimesh.util.concatenate(tuple(meshes))
    surface.remove_unreferenced_vertices()
    _require_surface(surface, context=f"owner '{owner_id}' concatenated surface")
    return surface


def sample_owner_home_surfaces(
    cache: OwnerGeometryCache,
    *,
    points_per_owner: int,
    sampling_seed: int,
    oversample_factor: int = 8,
) -> HomeSurfaceSamples:


    if points_per_owner < 1:
        raise ValueError("points_per_owner must be positive")
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be positive")

    sampled_points: list[np.ndarray] = []
    sampled_faces: list[np.ndarray] = []
    sampled_barycentric: list[np.ndarray] = []
    for record in cache.records:
        owner_seed = _stable_owner_seed(sampling_seed, record.owner_id)
        candidate_count = points_per_owner * oversample_factor
        sampled_surface = trimesh.sample.sample_surface(
            record.surface_mesh,
            candidate_count,
            seed=owner_seed,
        )
        candidates = sampled_surface[0]
        candidate_faces = sampled_surface[1]
        selected = _farthest_point_indices(candidates, points_per_owner)
        points = candidates[selected]
        faces = candidate_faces[selected]
        triangles = record.surface_mesh.triangles[faces]
        barycentric = trimesh.triangles.points_to_barycentric(triangles, points)
        sampled_points.append(points)
        sampled_faces.append(faces)
        sampled_barycentric.append(barycentric)

    return HomeSurfaceSamples(
        owner_ids=tuple(record.owner_id for record in cache.records),
        points_owner_local_m=np.stack(sampled_points, axis=0),
        face_indices=np.stack(sampled_faces, axis=0),
        barycentric=np.stack(sampled_barycentric, axis=0),
        sampling_seed=sampling_seed,
        oversample_factor=oversample_factor,
    )


def _component_mesh(
    component: CollisionComponentSemanticsCfg,
    *,
    container: HandContainer,
) -> trimesh.Trimesh:


    payload = component.geometry_payload
    kind = component.geometry_kind
    if kind == "box":
        mesh = trimesh.creation.box(extents=np.asarray(payload["size"], dtype=np.float64))
    elif kind == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=float(payload["radius"]),
            height=float(payload["length"]),
            sections=64,
        )
    elif kind == "elliptic_cylinder":
        mesh = _elliptic_cylinder_mesh(
            radius_x=float(payload["radius_x"]),
            radius_z=float(payload["radius_z"]),
            length=float(payload["length"]),
        )
    elif kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(payload["radius"]))
    elif kind == "mesh":
        mesh_path = _resolve_component_mesh_path(str(payload["file_path"]), container=container)


        loaded = trimesh.load(mesh_path, force="mesh", process=True)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"mesh component '{component.component_id}' did not load as one Trimesh")
        mesh = loaded.copy()
        mesh.apply_scale(np.asarray(payload.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64))
    else:
        raise ValueError(f"unsupported collision geometry kind={kind!r} for '{component.component_id}'")
    _clean_surface_topology(mesh)
    _require_surface(mesh, context=f"component '{component.component_id}'")
    return mesh


def _elliptic_cylinder_mesh(*, radius_x: float, radius_z: float, length: float) -> trimesh.Trimesh:


    mesh = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=64)
    source = mesh.vertices.copy()
    mesh.vertices[:, 0] = radius_x * source[:, 0]
    mesh.vertices[:, 1] = length * source[:, 2]
    mesh.vertices[:, 2] = -radius_z * source[:, 1]
    return mesh


def _resolve_component_mesh_path(raw_path: str, *, container: HandContainer) -> Path:
    if container.portable_mesh_bindings is not None:
        return container.resolve_mesh_locator(raw_path)
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"collision mesh does not exist: {candidate}")
        return candidate

    for mesh_ref in container.mesh_refs:
        if raw_path == mesh_ref.raw_uri or PurePosixPath(raw_path).name == mesh_ref.virtual_path.name:
            return mesh_ref.real_path
    virtual_candidate = PurePosixPath("meshes") / PurePosixPath(raw_path).name
    try:
        return container.real_path(virtual_candidate)
    except KeyError as exc:
        raise FileNotFoundError(f"collision mesh path {raw_path!r} is not part of asset {container.asset_id!r}") from exc


def _clean_surface_topology(mesh: trimesh.Trimesh) -> None:


    mesh.merge_vertices()
    if len(mesh.faces):
        unique = mesh.unique_faces()
        mesh.update_faces(unique)
    if len(mesh.faces):
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
        doubled_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=-1,
        )
        mesh.update_faces(np.isfinite(doubled_area) & (doubled_area > 0.0))
    mesh.remove_unreferenced_vertices()


def _require_surface(mesh: trimesh.Trimesh, *, context: str) -> None:


    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError(f"{context} must be a triangle surface")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"{context} must contain vertices and faces")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"{context} contains non-finite vertices")
    triangles = vertices[faces]
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=-1,
    )
    if not np.all(np.isfinite(doubled_area)) or np.any(doubled_area <= 0.0):
        raise ValueError(f"{context} contains zero-area or non-finite triangles")


def _is_volume(mesh: trimesh.Trimesh) -> bool:


    return bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume)


def _require_volume(mesh: trimesh.Trimesh, *, context: str) -> None:


    if not _is_volume(mesh):
        raise ValueError(
            f"{context} must be a watertight consistently-wound volume; "
            f"watertight={mesh.is_watertight}, winding={mesh.is_winding_consistent}, volume={mesh.is_volume}"
        )


def _owner_surface_geometry_hash(records: tuple[OwnerSurfaceRecord, ...]) -> str:


    digest = hashlib.sha256()
    digest.update(b"owner-surface-v2\0")
    for record in records:
        vertices = np.ascontiguousarray(record.surface_mesh.vertices, dtype="<f8")
        faces = np.ascontiguousarray(record.surface_mesh.faces, dtype="<i4")
        digest.update(record.owner_id.encode("utf-8") + b"\0")
        digest.update(np.asarray(vertices.shape, dtype="<i8").tobytes())
        digest.update(vertices.tobytes())
        digest.update(np.asarray(faces.shape, dtype="<i8").tobytes())
        digest.update(faces.tobytes())
    return digest.hexdigest()


def _hash_strings(digest: Any, values: tuple[str, ...]) -> None:


    digest.update(len(values).to_bytes(8, "little", signed=False))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)


def _hash_tensor(digest: Any, value: Any, *, floating: bool) -> None:


    tensor = value.detach().cpu()
    array = np.asarray(tensor, dtype="<f8" if floating else "<i8")
    array = np.ascontiguousarray(array)
    digest.update(b"f8" if floating else b"i8")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def _hash_optional_tensor(digest: Any, value: Any | None, *, floating: bool) -> None:


    if value is None:
        digest.update(b"\x00")
        return
    digest.update(b"\x01")
    _hash_tensor(digest, value, floating=floating)


def _stable_owner_seed(seed: int, owner_id: str) -> int:


    owner_hash = 2166136261
    for byte in owner_id.encode("utf-8"):
        owner_hash = ((owner_hash ^ byte) * 16777619) & 0xFFFFFFFF
    return (int(seed) ^ owner_hash) & 0xFFFFFFFF


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:


    if count > len(points):
        raise ValueError(f"cannot select {count} points from {len(points)} candidates")
    centroid = points.mean(axis=0)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=-1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_squared_distance = np.sum((points - points[first]) ** 2, axis=-1)
    for output_index in range(1, count):
        next_index = int(np.argmax(minimum_squared_distance))
        selected[output_index] = next_index
        next_distance = np.sum((points - points[next_index]) ** 2, axis=-1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, next_distance)
    return selected


from .anchor_sampling import (  # noqa: E402, I001 - imported after collision cache types to avoid a cycle
    AnchorClassificationStats,
    AnchorRealization,
    AnchorSamples,
    sample_palm_anchor_bank_warp,
    sample_palm_anchor_realization_warp,
    sample_palm_anchor_supports,
)


__all__ = [
    "AnchorClassificationStats",
    "AnchorRealization",
    "HomeSurfaceSamples",
    "AnchorSamples",
    "GeometryIdentity",
    "OwnerGeometryCache",
    "OwnerSurfaceRecord",
    "WarpOwnerGeometryCache",
    "WarpOwnerMeshHandle",
    "WarpSurfaceAudit",
    "WarpSurfaceView",
    "OwnerSurfaceSamplingArrays",
    "materialize_owner_geometry_cache",
    "materialize_warp_owner_geometry_cache",
    "prepare_warp_surface_view",
    "prepare_owner_surface_sampling_arrays",
    "geometry_identity",
    "release_warp_owner_geometry_cache",
    "sample_owner_home_surfaces",
    "sample_palm_anchor_bank_warp",
    "sample_palm_anchor_realization_warp",
    "sample_palm_anchor_supports",
    "strict_owner_union",
    "warp_owner_geometry_cache_stats",
]
