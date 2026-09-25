"""Sample finite physical anchor banks from hand geometry."""


from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import trimesh

from anymani.assets.asset_schema_geometry import HandGeometrySemanticsCfg

from .kinematics import EmbodimentGeometrySpec

_ANCHOR_BOUNDARY_RECHECK_M = 1.0e-6
"""CPU float64 verification band for Warp float32 signs, in meters."""

_ANCHOR_SAMPLING_VERSION = "palm-seed-radial-gaussian-fps-fast-winding-v2"
"""Bank-major proposal and classification version; device is separate provenance."""


@dataclass(frozen=True)
class AnchorSamples:


    anchors_hand_m: np.ndarray
    finger_names: tuple[str, ...]
    seed_ids: tuple[str, ...]
    surface_mask: np.ndarray
    radial_support_radius_m: float
    radial_decay_scale_m: float
    surface_fraction: float
    sampling_seed: int
    algorithm_version: str


@dataclass(frozen=True)
class AnchorRealization:


    bank_index: int
    bank_size: int
    root_seed: int
    derived_seed: int
    samples: AnchorSamples
    realization_hash: str
    sampling_version: str

    def __post_init__(self) -> None:


        if not 0 <= self.bank_index < self.bank_size:
            raise ValueError("anchor realization bank_index must lie in [0,bank_size)")
        if self.derived_seed != self.root_seed + self.bank_index * 1_000_003:
            raise ValueError("anchor realization derived seed does not match bank identity")
        if self.samples.sampling_seed != self.derived_seed:
            raise ValueError("anchor samples seed does not match selected realization")
        if len(self.realization_hash) != 64 or not self.sampling_version:
            raise ValueError("anchor realization requires SHA-256 hash and sampling version")


@dataclass(frozen=True)
class AnchorClassificationStats:


    query_point_count: int
    kernel_launch_count: int
    boundary_recheck_count: int
    boundary_disagreement_count: int
    elapsed_seconds: float


def sample_palm_anchor_supports(
    cache: Any,
    semantics: HandGeometrySemanticsCfg,
    spec: EmbodimentGeometrySpec,
    *,
    anchors_per_finger: int,
    sampling_seed: int,
    radial_support_radius_m: float = 0.05,
    radial_decay_scale_m: float | None = None,
    surface_fraction: float = 0.5,
    _interior_proposals: tuple[np.ndarray, ...] | None = None,
    _algorithm_version: str = "palm-seed-radial-gaussian-fps-v1",
) -> AnchorSamples:


    if anchors_per_finger < 1:
        raise ValueError("anchors_per_finger must be positive")
    radial_decay_scale_m = (
        0.5 * radial_support_radius_m if radial_decay_scale_m is None else radial_decay_scale_m
    )
    if radial_support_radius_m <= 0.0 or not 0.0 < radial_decay_scale_m <= radial_support_radius_m:
        raise ValueError("anchor support radius and radial decay scale must satisfy 0 < tau_a <= R_a")
    if not 0.0 <= surface_fraction <= 1.0:
        raise ValueError("surface_fraction must lie in [0,1]")
    if spec.owner_ids != tuple(owner.owner_id for owner in semantics.owners):
        raise ValueError("anchor semantics/spec owner axes do not match")
    if _interior_proposals is not None and len(_interior_proposals) != len(semantics.anchor_seeds):
        raise ValueError("preclassified interior proposals must align with the anchor seed axis")

    palm_index = next(owner.owner_index for owner in semantics.owners if owner.owner_id == "palm")
    palm_record = cache.records[palm_index]
    palm_transform = spec.owner_home_transforms[palm_index].detach().cpu().numpy()
    inverse_palm = np.linalg.inv(palm_transform)
    hand_rotation = np.asarray(semantics.asset_to_hand_rotation, dtype=np.float64).reshape(3, 3)
    hand_translation = np.asarray(semantics.asset_to_hand_translation_m, dtype=np.float64)

    all_points: list[np.ndarray] = []
    all_finger_names: list[str] = []
    all_seed_ids: list[str] = []
    all_surface_mask: list[bool] = []
    surface_count = int(round(anchors_per_finger * surface_fraction))
    interior_count = anchors_per_finger - surface_count
    for seed_index, seed in enumerate(semantics.anchor_seeds):
        seed_hand = hand_rotation @ np.asarray(seed.position_a_m) + hand_translation
        seed_local = (inverse_palm @ np.append(seed_hand, 1.0))[:3]
        sampled_surface = trimesh.sample.sample_surface(
            palm_record.surface_mesh,
            max(anchors_per_finger * 64, 256),
            seed=_stable_owner_seed(sampling_seed, seed.seed_id),
        )
        local_surface = _within_radius(sampled_surface[0], seed_local, radial_support_radius_m)
        local_surface = _radial_decay_candidates(
            local_surface,
            seed_local,
            radial_decay_scale_m,
            seed=_stable_owner_seed(sampling_seed + 2, seed.seed_id),
        )
        if len(local_surface) < surface_count:
            raise ValueError(
                f"anchor seed '{seed.seed_id}' has only {len(local_surface)} palm surface candidates "
                f"after radial decay within radius {radial_support_radius_m} m; need {surface_count}"
            )
        selected_surface = (
            local_surface[_farthest_point_indices(local_surface, surface_count)]
            if surface_count
            else np.empty((0, 3))
        )

        if interior_count and palm_record.solid_mesh is None:
            raise ValueError(
                "palm interior anchors require OwnerSurfaceRecord.solid_mesh; "
                "an open surface cannot define inside support"
            )
        interior_candidates = max(anchors_per_finger * 64, 256)
        local_interior = (
            _interior_proposals[seed_index]
            if _interior_proposals is not None
            else _sample_interior_support(
                palm_record.solid_mesh,
                seed_local,
                radial_support_radius_m,
                interior_candidates if interior_count else 0,
                seed=_stable_owner_seed(sampling_seed + 1, seed.seed_id),
            )
            if palm_record.solid_mesh is not None
            else np.empty((0, 3))
        )
        local_interior = _radial_decay_candidates(
            local_interior,
            seed_local,
            radial_decay_scale_m,
            seed=_stable_owner_seed(sampling_seed + 3, seed.seed_id),
        )
        if len(local_interior) < interior_count:
            raise ValueError(
                f"anchor seed '{seed.seed_id}' has only {len(local_interior)} palm interior candidates "
                f"after radial decay within radius {radial_support_radius_m} m; need {interior_count}"
            )
        selected_interior = (
            local_interior[_farthest_point_indices(local_interior, interior_count)]
            if interior_count
            else np.empty((0, 3))
        )
        local_points = np.concatenate((selected_surface, selected_interior), axis=0)
        homogeneous = np.concatenate((local_points, np.ones((len(local_points), 1))), axis=1)
        hand_points = (palm_transform @ homogeneous.T).T[:, :3]
        all_points.append(hand_points)
        all_finger_names.extend([seed.finger_name] * anchors_per_finger)
        all_seed_ids.extend([seed.seed_id] * anchors_per_finger)
        all_surface_mask.extend([True] * surface_count + [False] * interior_count)

    return AnchorSamples(
        anchors_hand_m=np.concatenate(all_points, axis=0),
        finger_names=tuple(all_finger_names),
        seed_ids=tuple(all_seed_ids),
        surface_mask=np.asarray(all_surface_mask, dtype=bool),
        radial_support_radius_m=float(radial_support_radius_m),
        radial_decay_scale_m=float(radial_decay_scale_m),
        surface_fraction=float(surface_fraction),
        sampling_seed=int(sampling_seed),
        algorithm_version=_algorithm_version,
    )


def sample_palm_anchor_bank_warp(
    cache: Any,
    semantics: HandGeometrySemanticsCfg,
    spec: EmbodimentGeometrySpec,
    warp_cache: Any,
    *,
    bank_size: int,
    anchors_per_finger: int,
    static_sampling_seed: int,
    radial_support_radius_m: float = 0.05,
    radial_decay_scale_m: float | None = None,
    surface_fraction: float = 0.5,
    boundary_recheck_m: float = _ANCHOR_BOUNDARY_RECHECK_M,
) -> tuple[tuple[AnchorSamples, ...], AnchorClassificationStats]:


    if bank_size < 1 or anchors_per_finger < 1 or boundary_recheck_m < 0.0:
        raise ValueError("anchor bank, per-finger budget and boundary tolerance must be valid")
    surface_hash = cache.surface_geometry_hash
    if not surface_hash:
        from .collision_geometry import _owner_surface_geometry_hash

        surface_hash = _owner_surface_geometry_hash(cache.records)
    if surface_hash != warp_cache.surface_geometry_hash or (
        cache.surface_processing_version != warp_cache.surface_processing_version
    ):
        raise ValueError("CPU owner geometry and Warp cache surface identities differ")
    if tuple(record.owner_id for record in cache.records) != tuple(handle.owner_id for handle in warp_cache.handles):
        raise ValueError("CPU owner geometry and Warp cache owner axes differ")
    palm_index = next(owner.owner_index for owner in semantics.owners if owner.owner_id == "palm")
    palm_record = cache.records[palm_index]
    if warp_cache.handles[palm_index].owner_id != palm_record.owner_id:
        raise ValueError("palm CPU/Warp owner axes differ")

    surface_count = int(round(anchors_per_finger * surface_fraction))
    interior_count = anchors_per_finger - surface_count
    interior_candidate_count = max(anchors_per_finger * 64, 256) if interior_count else 0
    palm_solid = palm_record.solid_mesh
    if interior_count and palm_solid is None:
        raise ValueError("palm interior anchors require a closed solid mesh")

    palm_transform = spec.owner_home_transforms[palm_index].detach().cpu().numpy()
    inverse_palm = np.linalg.inv(palm_transform)
    hand_rotation = np.asarray(semantics.asset_to_hand_rotation, dtype=np.float64).reshape(3, 3)
    hand_translation = np.asarray(semantics.asset_to_hand_translation_m, dtype=np.float64)
    centers = []
    for seed in semantics.anchor_seeds:
        seed_hand = hand_rotation @ np.asarray(seed.position_a_m) + hand_translation
        centers.append((inverse_palm @ np.append(seed_hand, 1.0))[:3])

    jobs: list[tuple[np.ndarray, int]] = []
    for bank_index in range(bank_size):
        sampling_seed = static_sampling_seed + bank_index * 1_000_003
        jobs.extend(
            (center, _stable_owner_seed(sampling_seed + 1, seed.seed_id))
            for center, seed in zip(centers, semantics.anchor_seeds)
        )

    started = perf_counter()
    if interior_candidate_count:
        assert palm_solid is not None
        proposals, query_count, launch_count, recheck_count, disagreement_count = _sample_interior_support_jobs_warp(
            palm_solid,
            warp_cache.handles[palm_index],
            jobs,
            radius=radial_support_radius_m,
            count=interior_candidate_count,
            device=warp_cache.device,
            boundary_recheck_m=boundary_recheck_m,
        )
    else:
        proposals = tuple(np.empty((0, 3), dtype=np.float64) for _ in jobs)
        query_count = launch_count = recheck_count = disagreement_count = 0

    seed_count = len(semantics.anchor_seeds)
    anchor_bank = tuple(
        sample_palm_anchor_supports(
            cache,
            semantics,
            spec,
            anchors_per_finger=anchors_per_finger,
            sampling_seed=static_sampling_seed + bank_index * 1_000_003,
            radial_support_radius_m=radial_support_radius_m,
            radial_decay_scale_m=radial_decay_scale_m,
            surface_fraction=surface_fraction,
            _interior_proposals=proposals[bank_index * seed_count : (bank_index + 1) * seed_count],
            _algorithm_version=f"{_ANCHOR_SAMPLING_VERSION}:{warp_cache.device}",
        )
        for bank_index in range(bank_size)
    )
    return anchor_bank, AnchorClassificationStats(
        query_point_count=query_count,
        kernel_launch_count=launch_count,
        boundary_recheck_count=recheck_count,
        boundary_disagreement_count=disagreement_count,
        elapsed_seconds=perf_counter() - started,
    )


def _anchor_realization_hash(samples: AnchorSamples) -> str:


    digest = hashlib.sha256(b"anymani-anchor-realization-v1\0")
    for array in (samples.anchors_hand_m, samples.surface_mask):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes(order="C"))
    for values in (samples.finger_names, samples.seed_ids):
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    for value in (
        samples.radial_support_radius_m,
        samples.radial_decay_scale_m,
        samples.surface_fraction,
    ):
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    digest.update(int(samples.sampling_seed).to_bytes(8, "little", signed=True))
    digest.update(samples.algorithm_version.encode("utf-8"))
    return digest.hexdigest()


def sample_palm_anchor_realization_warp(
    cache: Any,
    semantics: HandGeometrySemanticsCfg,
    spec: EmbodimentGeometrySpec,
    warp_cache: Any,
    *,
    bank_index: int,
    bank_size: int,
    anchors_per_finger: int,
    static_sampling_seed: int,
    radial_support_radius_m: float = 0.05,
    radial_decay_scale_m: float | None = None,
    surface_fraction: float = 0.5,
    boundary_recheck_m: float = _ANCHOR_BOUNDARY_RECHECK_M,
) -> tuple[AnchorRealization, AnchorClassificationStats]:


    if not 0 <= bank_index < bank_size:
        raise ValueError("selected anchor bank_index must lie in [0,bank_size)")
    derived_seed = int(static_sampling_seed) + int(bank_index) * 1_000_003
    selected, stats = sample_palm_anchor_bank_warp(
        cache,
        semantics,
        spec,
        warp_cache,
        bank_size=1,
        anchors_per_finger=anchors_per_finger,
        static_sampling_seed=derived_seed,
        radial_support_radius_m=radial_support_radius_m,
        radial_decay_scale_m=radial_decay_scale_m,
        surface_fraction=surface_fraction,
        boundary_recheck_m=boundary_recheck_m,
    )
    samples = selected[0]
    return AnchorRealization(
        bank_index=int(bank_index),
        bank_size=int(bank_size),
        root_seed=int(static_sampling_seed),
        derived_seed=derived_seed,
        samples=samples,
        realization_hash=_anchor_realization_hash(samples),
        sampling_version=samples.algorithm_version,
    ), stats


def _sample_interior_support_jobs_warp(
    mesh: trimesh.Trimesh,
    warp_handle: Any,
    jobs: list[tuple[np.ndarray, int]],
    *,
    radius: float,
    count: int,
    device: str,
    boundary_recheck_m: float,
) -> tuple[tuple[np.ndarray, ...], int, int, int, int]:


    if count == 0:
        return tuple(np.empty((0, 3), dtype=np.float64) for _ in jobs), 0, 0, 0, 0
    states = [
        {"rng": np.random.default_rng(seed), "accepted": [], "attempts": 0}
        for _center, seed in jobs
    ]
    query_point_count = 0
    kernel_launch_count = 0
    boundary_recheck_count = 0
    boundary_disagreement_count = 0
    max_attempts = max(10000, count * 10000)
    while True:
        active: list[tuple[int, np.ndarray]] = []
        for job_index, ((center, _seed), state) in enumerate(zip(jobs, states)):
            accepted = state["accepted"]
            accepted_count = sum(len(batch) for batch in accepted)
            if accepted_count >= count:
                continue
            if int(state["attempts"]) >= max_attempts:
                raise ValueError(f"palm solid has fewer than {count} interior candidates for job {job_index}")
            batch_size = max(256, (count - accepted_count) * 32)
            rng = state["rng"]
            candidate = rng.uniform(-radius, radius, size=(batch_size, 3)) + center[None, :]
            candidate = candidate[np.linalg.norm(candidate - center[None, :], axis=-1) <= radius]
            state["attempts"] = int(state["attempts"]) + batch_size
            if len(candidate):
                active.append((job_index, candidate))
        if not active:
            break

        merged = np.concatenate(tuple(candidate for _job, candidate in active), axis=0)
        inside, rechecked, disagreements = _classify_inside_warp(
            mesh,
            warp_handle,
            merged,
            device=device,
            boundary_recheck_m=boundary_recheck_m,
        )
        query_point_count += len(merged)
        kernel_launch_count += 1
        boundary_recheck_count += rechecked
        boundary_disagreement_count += disagreements
        offset = 0
        for job_index, candidate in active:
            stop = offset + len(candidate)
            states[job_index]["accepted"].append(candidate[inside[offset:stop]])
            offset = stop

    results = []
    for job_index, state in enumerate(states):
        accepted_batches = state["accepted"]
        result = np.concatenate(accepted_batches, axis=0) if accepted_batches else np.empty((0, 3), dtype=np.float64)
        if len(result) < count:
            raise ValueError(f"palm solid has only {len(result)} interior candidates for job {job_index}; need {count}")
        results.append(result[:count])
    return tuple(results), query_point_count, kernel_launch_count, boundary_recheck_count, boundary_disagreement_count


def _classify_inside_warp(
    mesh: trimesh.Trimesh,
    warp_handle: Any,
    points: np.ndarray,
    *,
    device: str,
    boundary_recheck_m: float,
) -> tuple[np.ndarray, int, int]:


    if _warp_anchor_signed_distance_kernel is None:
        raise RuntimeError("Warp anchor signed-distance kernel is unavailable")
    import warp as wp

    query_points = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=device)
    signed_distance = wp.zeros(len(points), dtype=wp.float32, device=device)
    wp.launch(
        _warp_anchor_signed_distance_kernel,
        dim=len(points),
        inputs=[  # pyright: ignore[reportAttributeAccessIssue]
            warp_handle.mesh.id,  # pyright: ignore[reportAttributeAccessIssue]
            query_points,
            signed_distance,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    distance = np.asarray(signed_distance.numpy(), dtype=np.float32)
    if np.any(~np.isfinite(distance)) or np.any(distance >= 0.5 * np.finfo(np.float32).max):
        raise RuntimeError("Warp anchor query failed to find a closest palm surface")
    inside = distance < 0.0
    boundary = np.abs(distance) <= boundary_recheck_m
    disagreement_count = 0
    if np.any(boundary):
        cpu_inside = _contains_points_fixed_ray(mesh, np.asarray(points[boundary], dtype=np.float64))
        disagreement_count = int(np.count_nonzero(inside[boundary] != cpu_inside))
        inside[boundary] = cpu_inside
    return inside, int(np.count_nonzero(boundary)), disagreement_count


def _contains_points_fixed_ray(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:


    from trimesh.ray.ray_triangle import RayMeshIntersector
    from trimesh.ray.ray_util import contains_points

    direction = np.asarray([0.4395064455, 0.617598629942, 0.652231566745], dtype=np.float64)
    return contains_points(
        RayMeshIntersector(mesh),
        np.asarray(points, dtype=np.float64),
        check_direction=direction,
    )


def _stable_owner_seed(seed: int, owner_id: str) -> int:


    owner_hash = 2166136261
    for byte in owner_id.encode("utf-8"):
        owner_hash = ((owner_hash ^ byte) * 16777619) & 0xFFFFFFFF
    return (int(seed) ^ owner_hash) & 0xFFFFFFFF


def _within_radius(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:


    return points[np.linalg.norm(points - center[None, :], axis=-1) <= radius]


def _radial_decay_candidates(
    points: np.ndarray,
    center: np.ndarray,
    scale: float,
    *,
    seed: int,
) -> np.ndarray:


    if len(points) == 0:
        return points
    squared_radius = np.sum((points - center[None, :]) ** 2, axis=-1)
    acceptance = np.exp(-squared_radius / (2.0 * scale * scale))
    rng = np.random.default_rng(seed)
    return points[rng.random(len(points)) < acceptance]


def _sample_interior_support(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    radius: float,
    count: int,
    *,
    seed: int,
) -> np.ndarray:


    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(10000, count * 10000)
    while sum(len(batch) for batch in accepted) < count and attempts < max_attempts:
        batch_size = max(256, (count - sum(len(batch) for batch in accepted)) * 32)
        candidate = rng.uniform(-radius, radius, size=(batch_size, 3)) + center[None, :]
        candidate = candidate[np.linalg.norm(candidate - center[None, :], axis=-1) <= radius]
        if len(candidate):
            accepted.append(candidate[mesh.contains(candidate)])
        attempts += batch_size
    result = np.concatenate(accepted, axis=0) if accepted else np.empty((0, 3), dtype=np.float64)
    if len(result) < count:
        raise ValueError(f"palm solid has only {len(result)} interior support candidates; need {count}")
    return result[:count]


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


try:
    import warp as wp

    @wp.kernel
    def _warp_anchor_signed_distance_kernel(
        mesh: wp.uint64,  # pyright: ignore[reportInvalidTypeForm]
        points: wp.array(dtype=wp.vec3),  # pyright: ignore[reportInvalidTypeForm]
        signed_distance: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    ):


        thread = wp.tid()
        point = points[thread]
        query = wp.mesh_query_point_sign_winding_number(  # pyright: ignore[reportArgumentType]
            mesh,
            point,
            wp.float32(1.0e8),  # pyright: ignore[reportArgumentType]
            wp.float32(2.0),  # pyright: ignore[reportArgumentType]
            wp.float32(0.5),  # pyright: ignore[reportArgumentType]
        )
        if not query.result:  # pyright: ignore[reportAttributeAccessIssue]
            signed_distance[thread] = 3.4028234663852886e38
            return
        closest = wp.mesh_eval_position(  # pyright: ignore[reportAttributeAccessIssue]
            mesh,
            query.face,  # pyright: ignore[reportAttributeAccessIssue]
            query.u,  # pyright: ignore[reportAttributeAccessIssue]
            query.v,  # pyright: ignore[reportAttributeAccessIssue]
        )
        signed_distance[thread] = wp.length(closest - point) * query.sign  # pyright: ignore[reportAttributeAccessIssue]

except Exception:
    _warp_anchor_signed_distance_kernel = None  # pyright: ignore[reportAssignmentType]


__all__ = [
    "AnchorClassificationStats",
    "AnchorRealization",
    "AnchorSamples",
    "sample_palm_anchor_bank_warp",
    "sample_palm_anchor_realization_warp",
    "sample_palm_anchor_supports",
]
