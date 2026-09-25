"Batches signed-distance work while preserving per-asset and per-owner indices."

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from ..asset_base import HandCfg
from ..asset_schema_core import (
    BoxGeometryCfg,
    CylinderGeometryCfg,
    EllipticCylinderGeometryCfg,
    MeshGeometryCfg,
    SphereGeometryCfg,
)
from ._collision_geometry import CollisionBodyRecord, extract_finger_collision_bodies
from ._mesh_sdf import MeshSdfQueryStats, signed_distance_to_mesh_body_batch
from ._sdf_clearance import (
    FingerPairClearance,
    SdfClearanceCertificate,
    SdfClearanceConfig,
    SdfClearanceResult,
    _filter_union_surface_points,
    _rotation_matrix_rows,
    evaluate_finger_sdf_clearance,
    sample_body_surface,
)

_PRIMITIVE_KINDS = ("box", "sphere", "cylinder", "elliptic_cylinder")
_MAX_POINT_BODY_COMBINATIONS = 1_000_000


@dataclass(frozen=True)
class _DirectedQuery:
    "One directed finger-pair surface-distance query."

    hand_index: int
    pair_index: int
    direction: int
    points: tuple[tuple[float, float, float], ...]
    target_bodies: tuple[CollisionBodyRecord, ...]
    mesh_stats: MeshSdfQueryStats


@dataclass(frozen=True)
class _PreparedHand:
    "Prepared collision bodies and surface samples for one hand."

    finger_pairs: tuple[tuple[str, str], ...]
    skipped_bodies: tuple[dict[str, str], ...]
    complete: bool
    mesh_stats: MeshSdfQueryStats


def evaluate_finger_sdf_clearance_batch(
    hands: Sequence[HandCfg],
    configs: Sequence[SdfClearanceConfig],
    *,
    borderline_recheck_margin: float = 1.0e-6,
    verify_all_with_scalar: bool = False,
) -> list[SdfClearanceResult]:
    "Batches ordered finger-pair clearance queries for one candidate hand set."

    if len(hands) != len(configs):
        raise ValueError("hands and configs must have identical batch length")
    if not hands:
        return []
    if borderline_recheck_margin < 0.0:
        raise ValueError("borderline_recheck_margin must be non-negative")
    _require_cuda_configs(configs)

    prepared: list[_PreparedHand] = []
    directed_queries: list[_DirectedQuery] = []
    for hand_index, (hand, cfg) in enumerate(zip(hands, configs)):
        hand_record, hand_queries = _prepare_hand_queries(hand_index, hand, cfg)
        prepared.append(hand_record)
        directed_queries.extend(hand_queries)

    directed_clearances = _evaluate_directed_queries(directed_queries)
    results = _reconstruct_results(prepared, configs, directed_queries, directed_clearances)
    final_results: list[SdfClearanceResult] = []
    for hand, cfg, result in zip(hands, configs, results):
        is_borderline = any(
            abs(pair.clearance - cfg.min_clearance) <= borderline_recheck_margin
            for pair in result.certificate_pair_objects
        )
        if verify_all_with_scalar or is_borderline:
            reference = evaluate_finger_sdf_clearance(hand, cfg)
            _assert_scalar_parity(result, reference, atol=1.0e-7)
            final_results.append(reference)
        else:
            final_results.append(result.base_result)
    return final_results


@dataclass(frozen=True)
class _ResultWithPairs:
    "Batch result joined to its queried hand and finger-pair indices."

    base_result: SdfClearanceResult
    certificate_pair_objects: tuple[FingerPairClearance, ...]


def _prepare_hand_queries(
    hand_index: int,
    hand: HandCfg,
    cfg: SdfClearanceConfig,
) -> tuple[_PreparedHand, list[_DirectedQuery]]:

    extraction = extract_finger_collision_bodies(hand, unsupported_policy=cfg.unsupported_policy)
    mesh_stats = MeshSdfQueryStats(requested_backend=cfg.mesh_backend)
    finger_names = [finger.name for finger in hand.fingers]
    finger_pairs: list[tuple[str, str]] = []
    queries: list[_DirectedQuery] = []
    for left_index, finger_i in enumerate(finger_names):
        for finger_j in finger_names[left_index + 1 :]:
            bodies_i = extraction.bodies_by_finger.get(finger_i, [])
            bodies_j = extraction.bodies_by_finger.get(finger_j, [])
            if not bodies_i or not bodies_j:
                continue
            pair_index = len(finger_pairs)
            finger_pairs.append((finger_i, finger_j))
            queries.append(
                _make_directed_query(
                    hand_index=hand_index,
                    pair_index=pair_index,
                    direction=0,
                    source_bodies=bodies_i,
                    target_bodies=bodies_j,
                    cfg=cfg,
                    mesh_stats=mesh_stats,
                )
            )
            queries.append(
                _make_directed_query(
                    hand_index=hand_index,
                    pair_index=pair_index,
                    direction=1,
                    source_bodies=bodies_j,
                    target_bodies=bodies_i,
                    cfg=cfg,
                    mesh_stats=mesh_stats,
                )
            )
    return (
        _PreparedHand(
            finger_pairs=tuple(finger_pairs),
            skipped_bodies=tuple(body.to_dict() for body in extraction.skipped_bodies),
            complete=extraction.complete,
            mesh_stats=mesh_stats,
        ),
        queries,
    )


def _make_directed_query(
    *,
    hand_index: int,
    pair_index: int,
    direction: int,
    source_bodies: list[CollisionBodyRecord],
    target_bodies: list[CollisionBodyRecord],
    cfg: SdfClearanceConfig,
    mesh_stats: MeshSdfQueryStats,
) -> _DirectedQuery:

    candidate_points: list[tuple[float, float, float]] = []
    for body in source_bodies:
        if isinstance(body.geometry, MeshGeometryCfg):
            mesh_stats.mesh_sample_count += 1
        candidate_points.extend(
            sample_body_surface(
                body,
                samples_per_axis=cfg.surface_samples_per_axis,
                mesh_surface_samples=cfg.mesh_surface_samples,
            )
        )
    points = _filter_union_surface_points(
        candidate_points,
        source_bodies,
        mesh_backend=cfg.mesh_backend,
        device="cuda",
        mesh_stats=mesh_stats,
    )
    return _DirectedQuery(
        hand_index=hand_index,
        pair_index=pair_index,
        direction=direction,
        points=tuple(points),
        target_bodies=tuple(target_bodies),
        mesh_stats=mesh_stats,
    )


def _evaluate_directed_queries(queries: Sequence[_DirectedQuery]) -> list[float]:

    import torch

    if not queries:
        return []
    query_min = torch.full((len(queries),), math.inf, dtype=torch.float32, device="cuda")
    for geometry_kind in _PRIMITIVE_KINDS:



        for chunk in _primitive_combination_chunks(queries, geometry_kind=geometry_kind):
            indices, distances = _primitive_chunk_distances(chunk, geometry_kind=geometry_kind)
            query_min.scatter_reduce_(0, indices, distances, reduce="amin", include_self=True)


    for query_index, query in enumerate(queries):
        if not query.points:
            continue
        for body in query.target_bodies:
            if not isinstance(body.geometry, MeshGeometryCfg):
                continue
            distances = signed_distance_to_mesh_body_batch(
                list(query.points),
                body,
                backend=query.mesh_stats.requested_backend,
                device="cuda",
                stats=query.mesh_stats,
            )
            if len(distances):
                mesh_min = torch.tensor(float(distances.min()), dtype=torch.float32, device="cuda")
                query_min[query_index] = torch.minimum(query_min[query_index], mesh_min)
    return [float(value) for value in query_min.detach().cpu().tolist()]


def _primitive_combination_chunks(
    queries: Sequence[_DirectedQuery],
    *,
    geometry_kind: str,
):

    chunk: list[tuple[int, tuple[float, float, float], CollisionBodyRecord]] = []
    for query_index, query in enumerate(queries):
        for body in query.target_bodies:
            if body.geometry_kind != geometry_kind:
                continue
            for point in query.points:
                chunk.append((query_index, point, body))
                if len(chunk) == _MAX_POINT_BODY_COMBINATIONS:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def _primitive_chunk_distances(
    combinations: Sequence[tuple[int, tuple[float, float, float], CollisionBodyRecord]],
    *,
    geometry_kind: str,
):

    import torch

    indices = torch.tensor([item[0] for item in combinations], dtype=torch.int64, device="cuda")
    points = torch.tensor([item[1] for item in combinations], dtype=torch.float32, device="cuda")
    translations = torch.tensor(
        [item[2].world_pose.pos for item in combinations],
        dtype=torch.float32,
        device="cuda",
    )
    rotations = torch.tensor(
        [_rotation_matrix_rows(item[2].world_pose.rpy) for item in combinations],
        dtype=torch.float32,
        device="cuda",
    )
    local = torch.bmm((points - translations).unsqueeze(1), rotations).squeeze(1)
    bodies = [item[2] for item in combinations]
    if geometry_kind == "box":
        geometries = [cast(BoxGeometryCfg, body.geometry) for body in bodies]
        half_size = torch.tensor(
            [[float(value) / 2.0 for value in geometry.size] for geometry in geometries],
            dtype=torch.float32,
            device="cuda",
        )
        q = torch.abs(local) - half_size
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=1)
        inside = torch.minimum(torch.amax(q, dim=1), torch.zeros_like(outside))
        return indices, outside + inside
    if geometry_kind == "sphere":
        geometries = [cast(SphereGeometryCfg, body.geometry) for body in bodies]
        radii = torch.tensor(
            [float(geometry.radius) for geometry in geometries],
            dtype=torch.float32,
            device="cuda",
        )
        return indices, torch.linalg.norm(local, dim=1) - radii
    if geometry_kind == "cylinder":
        geometries = [cast(CylinderGeometryCfg, body.geometry) for body in bodies]
        radii = torch.tensor([float(geometry.radius) for geometry in geometries], dtype=torch.float32, device="cuda")
        half_lengths = torch.tensor(
            [float(geometry.length) / 2.0 for geometry in geometries],
            dtype=torch.float32,
            device="cuda",
        )
        radial = torch.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2) - radii
        axial = torch.abs(local[:, 2]) - half_lengths
        outside = torch.sqrt(torch.clamp(radial, min=0.0) ** 2 + torch.clamp(axial, min=0.0) ** 2)
        inside = torch.minimum(torch.maximum(radial, axial), torch.zeros_like(outside))
        return indices, outside + inside
    if geometry_kind == "elliptic_cylinder":
        geometries = [cast(EllipticCylinderGeometryCfg, body.geometry) for body in bodies]
        radius_x = torch.tensor(
            [float(geometry.radius_x) for geometry in geometries],
            dtype=torch.float32,
            device="cuda",
        )
        radius_z = torch.tensor(
            [float(geometry.radius_z) for geometry in geometries],
            dtype=torch.float32,
            device="cuda",
        )
        half_lengths = torch.tensor(
            [float(geometry.length) / 2.0 for geometry in geometries],
            dtype=torch.float32,
            device="cuda",
        )
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        scaled_radius = torch.sqrt((x / radius_x) ** 2 + (z / radius_z) ** 2)
        radial_norm = torch.sqrt(x * x + z * z)
        safe_norm = torch.clamp(radial_norm, min=1e-12)
        ux, uz = x / safe_norm, z / safe_norm
        directional_boundary = 1.0 / torch.sqrt((ux / radius_x) ** 2 + (uz / radius_z) ** 2)
        center_boundary = torch.minimum(radius_x, radius_z)
        boundary_radius = torch.where(radial_norm <= 1e-12, center_boundary, directional_boundary)
        radial = (scaled_radius - 1.0) * boundary_radius
        axial = torch.abs(y) - half_lengths
        outside = torch.sqrt(torch.clamp(radial, min=0.0) ** 2 + torch.clamp(axial, min=0.0) ** 2)
        inside = torch.minimum(torch.maximum(radial, axial), torch.zeros_like(outside))
        return indices, outside + inside
    raise ValueError(f"unsupported primitive batch geometry kind: {geometry_kind!r}")


def _reconstruct_results(
    prepared: Sequence[_PreparedHand],
    configs: Sequence[SdfClearanceConfig],
    queries: Sequence[_DirectedQuery],
    directed_clearances: Sequence[float],
) -> list[_ResultWithPairs]:

    lookup = {
        (query.hand_index, query.pair_index, query.direction): float(clearance)
        for query, clearance in zip(queries, directed_clearances)
    }
    results: list[_ResultWithPairs] = []
    for hand_index, (record, cfg) in enumerate(zip(prepared, configs)):
        pairs: list[FingerPairClearance] = []
        violations: list[FingerPairClearance] = []
        for pair_index, (finger_i, finger_j) in enumerate(record.finger_pairs):
            forward = lookup[(hand_index, pair_index, 0)]
            backward = lookup[(hand_index, pair_index, 1)]
            pair = FingerPairClearance(
                finger_i=finger_i,
                finger_j=finger_j,
                clearance=min(forward, backward),
                direction_i_to_j=forward,
                direction_j_to_i=backward,
            )
            pairs.append(pair)
            if pair.clearance < cfg.min_clearance - cfg.tolerance:
                violations.append(pair)
        certificate = SdfClearanceCertificate(
            complete=record.complete,
            skipped_bodies=list(record.skipped_bodies),
            min_clearance=cfg.min_clearance,
            device="cuda",
            mesh_sdf=record.mesh_stats.to_dict(),
            pair_clearances=[pair.to_dict() for pair in pairs],
        )
        results.append(
            _ResultWithPairs(
                base_result=SdfClearanceResult(
                    passed=not violations and certificate.complete,
                    certificate=certificate,
                    violations=violations,
                ),
                certificate_pair_objects=tuple(pairs),
            )
        )
    return results


def _assert_scalar_parity(
    batch_result: _ResultWithPairs | SdfClearanceResult,
    scalar_result: SdfClearanceResult,
    *,
    atol: float,
) -> None:

    batch = batch_result.base_result if isinstance(batch_result, _ResultWithPairs) else batch_result
    if batch.passed != scalar_result.passed:
        raise RuntimeError("central GPU batch/scalar pass-fail mismatch")
    batch_certificate = batch.certificate
    scalar_certificate = scalar_result.certificate
    batch_metadata = batch_certificate.to_dict()
    scalar_metadata = scalar_certificate.to_dict()
    batch_metadata.pop("pair_clearances")
    scalar_metadata.pop("pair_clearances")
    if batch_metadata != scalar_metadata:
        raise RuntimeError("central GPU batch/scalar certificate metadata mismatch")
    batch_violations = [(pair.finger_i, pair.finger_j) for pair in batch.violations]
    scalar_violations = [(pair.finger_i, pair.finger_j) for pair in scalar_result.violations]
    if batch_violations != scalar_violations:
        raise RuntimeError("central GPU batch/scalar violation-order mismatch")
    batch_pairs = batch_certificate.pair_clearances
    scalar_pairs = scalar_certificate.pair_clearances
    if len(batch_pairs) != len(scalar_pairs):
        raise RuntimeError("central GPU batch/scalar pair-count mismatch")
    for batch_pair, scalar_pair in zip(batch_pairs, scalar_pairs):
        if (batch_pair["finger_i"], batch_pair["finger_j"]) != (
            scalar_pair["finger_i"],
            scalar_pair["finger_j"],
        ):
            raise RuntimeError("central GPU batch/scalar pair-order mismatch")
        for field_name in ("clearance", "direction_i_to_j", "direction_j_to_i"):
            if not math.isclose(float(batch_pair[field_name]), float(scalar_pair[field_name]), abs_tol=atol):
                raise RuntimeError(f"central GPU batch/scalar {field_name} mismatch")


def _require_cuda_configs(configs: Sequence[SdfClearanceConfig]) -> None:

    if any(cfg.device != "cuda" for cfg in configs):
        raise ValueError("central GPU SDF batch requires device='cuda'")
    if any(cfg.mesh_backend != "warp" for cfg in configs):
        raise ValueError("central GPU SDF batch requires mesh_backend='warp'")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required by central GPU SDF batch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("central GPU SDF batch requires an available CUDA device")


__all__ = ["evaluate_finger_sdf_clearance_batch"]
