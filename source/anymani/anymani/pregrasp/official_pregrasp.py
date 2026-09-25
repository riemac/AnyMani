(
    'Pure-CPU proposal and certificate logic for strict pregrasp on official '
    'Allegro/LEAP hands. Consume hash-validated OfficialHandSemanticsCfg, derive '
    'TIP centers from native collision meshes/FK, and package raw '
    'GoodPregraspMetrics from the main-thread physics session. Protocol: 256 '
    'scrambled Sobol candidates per asset; active joints stay within 10%-90% of '
    'source limits; object xy uses measured PALM collision bounds and z stays '
    'near 33 mm above the palm. After cheap geometry screening, run up to three '
    'rounds of 128 full-physics CEM candidates. Save all '
    'candidates/failures/metrics to NPZ; publish a certificate only when all '
    'eight pass MVP80_STRICT_GOOD_PREGRASP_GATE. Bind native URDF/mesh, '
    'physics/source code, DexCube, controller, and proposal identity; loader '
    'rechecks NPZ bytes, Top-8 rows, and every strict metric.'
)

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from anymani.assets.bank.official_hand import OfficialHandSemanticsCfg, forward_kinematics

from .good_catalog import (
    GOOD_PREGRASP_TOP_K,
    GoodPregraspCandidate,
    GoodPregraspEntry,
    GoodPregraspKey,
    GoodPregraspMember,
    GoodPregraspMetrics,
)
from .mvp80_strict_search import fixed_position_envelope, geometry_score
from .official_identity import (
    OFFICIAL_FINGER_ORDER,
    OFFICIAL_PREGRASP_ARTIFACT_TYPE,
    OFFICIAL_PREGRASP_SCHEMA_VERSION,
    OfficialPregraspIdentity,
    build_official_controller_identity,
    build_official_physics_identity,
    build_official_pregrasp_identity,
    build_official_source_implementation_identity,
    validate_official_pregrasp_for_session,
)
from .strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE, StrictGoodPregraspGate

OFFICIAL_PREGRASP_SOBOL_DIMENSION = 19
'Proposal dimensions: 16 joints, PALM xy, and z clearance.'

OFFICIAL_PREGRASP_SOBOL_CANDIDATES = 256
'Fixed Sobol candidate count per official asset.'

OFFICIAL_PREGRASP_GEOMETRY_TOP_K = 32
'Candidates entering physics after the initial cheap screen.'

OFFICIAL_PREGRASP_CEM_ROUNDS = 3
'Maximum CEM rounds when strict Top-8 is incomplete.'

OFFICIAL_PREGRASP_CEM_CANDIDATES = 128
'Physics candidates per CEM round; all use the same 1 s native gate.'

OFFICIAL_PREGRASP_PHYSICS_STEPS = 20
'20 Hz policy steps: 1 s uses 120 Hz physics with six substeps per step.'

OFFICIAL_PREGRASP_PALM_TOP_CLEARANCE_M = 0.033
'Nominal object-origin z clearance above measured PALM bounds, meters.'

OFFICIAL_PREGRASP_PALM_Z_JITTER_M = 0.002
'Local z jitter for Sobol/CEM proposals, meters.'

_NON_THUMB_FINGERS = ("index", "middle", "ring")
_OWNER_COUNT = 21
_JOINT_COUNT = 16
_SHA256_LENGTH = 64
_OFFICIAL_PREGRASP_PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # AnyMani/source/anymani
_OFFICIAL_PREGRASP_REPO_ROOT = _OFFICIAL_PREGRASP_PACKAGE_ROOT.parents[1]  # AnyMani


def _stable_digest(payload: Mapping[str, Any]) -> str:
    'Compute stable SHA-256 over a JSON-safe identity payload.'

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    'Stream hashes for source, mesh, and NPZ files.'

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    'Validate a lowercase 64-character SHA-256 identity.'

    parsed = str(value)
    if len(parsed) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in parsed):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256")
    return parsed


@dataclass(frozen=True)
class OfficialProposalBatch:
    'Candidate batch proposed by native meshes/FK with cheap geometry evidence.'

    q0_rad: torch.Tensor
    'Actual canonical joint state/target [C,16], rad.'

    object_position_h_m: torch.Tensor
    'Object origin in hand semantic frame [C,3], meters.'

    tip_centers_h_m: torch.Tensor
    'Centers of real native TIP collision meshes [C,4,3], meters.'

    non_thumb_pair: torch.Tensor
    'Selected non-thumb pair [C,2], index/middle/ring.'

    envelope_fingers: tuple[tuple[str, str, str], ...]
    'Selected thumb-plus-two-non-thumb roles per candidate.'

    envelope_tip_center_distances_m: torch.Tensor
    'Distances from selected three-finger envelope to object center [C,3], meters.'

    envelope_sector_min_deg: torch.Tensor
    'Minimum in-plane sector angle of selected three-finger envelope [C], degrees.'

    joint_margin_fraction: torch.Tensor
    'Minimum normalized margin from active joints to nearest source limit [C].'

    cheap_score: torch.Tensor
    'Continuous cheap-geometry ranking score [C].'

    cheap_pass: torch.Tensor
    'Joint/distance/sector cheap-screen pass mask [C].'

    seed_source: tuple[str, ...]
    'Proposal source label, e.g. sobol or explicit legacy_seed.'

    def __post_init__(self) -> None:
        'Validate candidate shapes, finite values, and seed alignment.'

        candidate_count = self.q0_rad.shape[0]
        if self.q0_rad.shape != (candidate_count, _JOINT_COUNT):
            raise ValueError("official q proposals must have shape [C,16]")
        if self.object_position_h_m.shape != (candidate_count, 3):
            raise ValueError("official object proposals must have shape [C,3]")
        if self.tip_centers_h_m.shape != (candidate_count, 4, 3):
            raise ValueError("official TIP centers must have shape [C,4,3]")
        if self.non_thumb_pair.shape != (candidate_count, 2):
            raise ValueError("official envelope pair must have shape [C,2]")
        if self.envelope_tip_center_distances_m.shape != (candidate_count, 3):
            raise ValueError("official envelope distances must have shape [C,3]")
        if self.envelope_sector_min_deg.shape != (candidate_count,):
            raise ValueError("official envelope sector must have shape [C]")
        if self.joint_margin_fraction.shape != (candidate_count,):
            raise ValueError("official joint margin must have shape [C]")
        if self.cheap_score.shape != (candidate_count,):
            raise ValueError("official cheap score must have shape [C]")
        if self.cheap_pass.shape != (candidate_count,):
            raise ValueError("official cheap pass must have shape [C]")
        for value, name in (
            (self.q0_rad, "q0_rad"),
            (self.object_position_h_m, "object_position_h_m"),
            (self.tip_centers_h_m, "tip_centers_h_m"),
            (self.non_thumb_pair, "non_thumb_pair"),
            (self.envelope_tip_center_distances_m, "envelope_tip_center_distances_m"),
            (self.envelope_sector_min_deg, "envelope_sector_min_deg"),
            (self.joint_margin_fraction, "joint_margin_fraction"),
            (self.cheap_score, "cheap_score"),
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"official proposal {name} must be finite")
        if self.non_thumb_pair.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("official envelope pair indices must use an integer tensor")
        if not torch.equal(self.non_thumb_pair, self.non_thumb_pair.to(dtype=torch.long)):
            raise ValueError("official envelope pair indices must be integral")
        if bool(((self.non_thumb_pair < 0) | (self.non_thumb_pair > 2)).any().item()):
            raise ValueError("official envelope pair indices must lie in [0,2]")
        if self.cheap_pass.dtype != torch.bool:
            raise ValueError("official cheap_pass must be boolean")
        if len(self.envelope_fingers) != candidate_count or len(self.seed_source) != candidate_count:
            raise ValueError("official proposal metadata must align with candidate axis")


def official_tip_center_positions_h(
    description: OfficialHandSemanticsCfg,
    q0_rad: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    (
        'Compute hand-frame TIP centers from native collision meshes and FK. Use mesh '
        'vertices, collision origins, and forward_kinematics from '
        'geometry_semantics.components; never use official empty marker frames as '
        'contact surfaces. Build on CPU and let the main thread move results to the '
        'simulator device. Input q0 is canonical [16] or [C,16] rad; return mesh AABB '
        'centers [C,4,3] in hand frame, meters.'
    )

    q_batch = _as_cpu_q_batch(q0_rad)
    components = {component.component_id: component for component in description.geometry_semantics.components}
    tip_owners = {
        cast(str, owner.finger_name): owner
        for owner in description.geometry_semantics.owners
        if owner.role == "tip" and owner.finger_name is not None
    }
    missing = set(OFFICIAL_FINGER_ORDER) - set(tip_owners)
    if missing:
        raise ValueError(f"official description lacks TIP owners: {sorted(missing)}")
    R_ha = np.asarray(description.semantic_R_ha, dtype=np.float64).reshape(3, 3)
    p_ha = np.asarray(description.semantic_p_ha, dtype=np.float64)
    output: list[np.ndarray] = []
    for q_row in q_batch.numpy():
        transforms = {
            name: np.asarray(value, dtype=np.float64).reshape(4, 4)
            for name, value in forward_kinematics(description, q_row).items()
        }
        per_finger: list[np.ndarray] = []
        for finger in OFFICIAL_FINGER_ORDER:
            owner = tip_owners[finger]
            points = _owner_component_points(owner, components, transforms, description)
            center_a = 0.5 * (points.min(axis=0) + points.max(axis=0))
            center_h = R_ha @ center_a + p_ha
            per_finger.append(center_h)
        output.append(np.stack(per_finger, axis=0))
    return torch.from_numpy(np.stack(output, axis=0).astype(np.float32, copy=False))


def official_palm_bounds_h(description: OfficialHandSemanticsCfg) -> tuple[torch.Tensor, torch.Tensor]:
    (
        'Compute hand-frame AABB of actual PALM collision components. Use typed '
        'collision geometry to constrain object xy proposals; do not use preset '
        'bounds or marker visuals.'
    )

    components = {component.component_id: component for component in description.geometry_semantics.components}
    palm_owner = next((owner for owner in description.geometry_semantics.owners if owner.role == "palm"), None)
    if palm_owner is None:
        raise ValueError("official description lacks PALM owner")
    transforms = {
        name: np.asarray(value, dtype=np.float64).reshape(4, 4)
        for name, value in forward_kinematics(description, (0.0,) * _JOINT_COUNT).items()
    }
    points = _owner_component_points(palm_owner, components, transforms, description)
    R_ha = np.asarray(description.semantic_R_ha, dtype=np.float64).reshape(3, 3)
    p_ha = np.asarray(description.semantic_p_ha, dtype=np.float64)
    points_h = points @ R_ha.T + p_ha
    return (
        torch.from_numpy(points_h.min(axis=0).astype(np.float32)),
        torch.from_numpy(points_h.max(axis=0).astype(np.float32)),
    )


def generate_official_sobol_candidates(
    description: OfficialHandSemanticsCfg,
    *,
    candidate_count: int = OFFICIAL_PREGRASP_SOBOL_CANDIDATES,
    seed: int = 20260914,
) -> OfficialProposalBatch:
    (
        'Generate the fixed native Sobol proposal batch. For each active joint, q = '
        'lower + (0.1 + 0.8*u)*(upper-lower). Place object xy inside measured PALM '
        'bounds and z near palm_top+0.033 m. Cheap screening only saves physics '
        'budget; final admission requires a 1 s zero-action replay in '
        'OfficialPalmRotationSession.'
    )

    if candidate_count < 1 or seed < 0:
        raise ValueError("official Sobol candidate_count must be positive and seed non-negative")
    lower = torch.tensor([limits[0] for limits in description.geometry_semantics.joint_limits_rad], dtype=torch.float32)
    upper = torch.tensor([limits[1] for limits in description.geometry_semantics.joint_limits_rad], dtype=torch.float32)
    if lower.shape != (16,) or not bool((lower < upper).all().item()):
        raise ValueError("official joint limits must be ordered [16]")
    sobol = torch.quasirandom.SobolEngine(
        dimension=OFFICIAL_PREGRASP_SOBOL_DIMENSION,
        scramble=True,
        seed=int(seed),
    ).draw(candidate_count)
    span = upper - lower
    q0 = lower + (0.10 + 0.80 * sobol[:, :16]) * span  # Active q is constrained to [lower+0.1*range, upper-0.1*range], rad.
    palm_low, palm_high = official_palm_bounds_h(description)
    xy = palm_low[:2] + sobol[:, 16:18] * (palm_high[:2] - palm_low[:2])  # Object xy stays within real PALM bounds, meters.
    z = (
        palm_high[2]
        + OFFICIAL_PREGRASP_PALM_TOP_CLEARANCE_M
        + (sobol[:, 18] - 0.5) * (2.0 * OFFICIAL_PREGRASP_PALM_Z_JITTER_M)
    )  # Object z is about 33 mm above the palm top, meters.
    object_position = torch.cat((xy, z.unsqueeze(-1)), dim=-1)
    tip_centers = official_tip_center_positions_h(description, q0)
    active_tip_mask = torch.ones(candidate_count, 4, dtype=torch.bool)
    envelope = fixed_position_envelope(tip_centers, active_tip_mask, object_position)
    margin = torch.minimum((q0 - lower) / span, (upper - q0) / span).amin(dim=-1)
    cheap = geometry_score(margin, envelope.tip_center_distances_m, envelope.sector_min_deg)
    cheap_pass = (
        (margin >= MVP80_STRICT_GOOD_PREGRASP_GATE.joint_margin_fraction_min)
        & (envelope.tip_center_distances_m.amax(dim=-1) <= MVP80_STRICT_GOOD_PREGRASP_GATE.tip_center_distance_m_max)
        & (envelope.sector_min_deg >= MVP80_STRICT_GOOD_PREGRASP_GATE.sector_min_deg)
    )
    pair_names = tuple(_pair_to_fingers(pair) for pair in envelope.non_thumb_pair.tolist())
    return OfficialProposalBatch(
        q0_rad=q0,
        object_position_h_m=object_position,
        tip_centers_h_m=tip_centers,
        non_thumb_pair=envelope.non_thumb_pair,
        envelope_fingers=pair_names,
        envelope_tip_center_distances_m=envelope.tip_center_distances_m,
        envelope_sector_min_deg=envelope.sector_min_deg,
        joint_margin_fraction=margin,
        cheap_score=cheap,
        cheap_pass=cheap_pass,
        seed_source=("sobol",) * candidate_count,
    )


def generate_official_cem_candidates(
    description: OfficialHandSemanticsCfg,
    elite_q0_rad: torch.Tensor,
    elite_position_h_m: torch.Tensor,
    *,
    candidate_count: int = OFFICIAL_PREGRASP_CEM_CANDIDATES,
    seed: int = 20260914,
    round_index: int = 0,
) -> OfficialProposalBatch:
    (
        'Generate one 128-item low-rank CEM round from native-physics elites. CEM '
        'changes proposals only, not the strict gate. Sample joints along the first '
        'four elite-covariance directions and clamp to the 10%-90% comfort interval; '
        'keep object position within measured PALM xy bounds and the top-clearance z '
        'band. Every proposal uses the same 1 s native gate.'
    )

    if elite_q0_rad.ndim != 2 or elite_q0_rad.shape[1] != _JOINT_COUNT:
        raise ValueError("official CEM elite q must have shape [E,16]")
    if elite_position_h_m.shape != (elite_q0_rad.shape[0], 3):
        raise ValueError("official CEM elite positions must have shape [E,3]")
    if elite_q0_rad.shape[0] < 2 or candidate_count < 1 or seed < 0 or round_index < 0:
        raise ValueError("official CEM requires at least two elites and positive candidate_count/seed/round_index")
    elite_q0_rad = elite_q0_rad.detach().to(device="cpu", dtype=torch.float32)
    elite_position_h_m = elite_position_h_m.detach().to(device="cpu", dtype=torch.float32)
    lower = torch.tensor([limits[0] for limits in description.geometry_semantics.joint_limits_rad], dtype=torch.float32)
    upper = torch.tensor([limits[1] for limits in description.geometry_semantics.joint_limits_rad], dtype=torch.float32)
    span = (upper - lower).clamp_min(1.0e-8)
    comfort_lower = lower + 0.10 * span
    comfort_upper = upper - 0.10 * span
    centered = elite_q0_rad.float() - elite_q0_rad.float().mean(dim=0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = min(4, vh.shape[0])
    basis = vh[:rank]
    coefficient = centered @ basis.T
    coefficient_std = coefficient.std(dim=0, unbiased=False).clamp(0.002, 0.06) * (0.70**round_index)
    generator = torch.Generator(device="cpu").manual_seed(int(seed + round_index * 1_000_003))
    noise = torch.randn(candidate_count, rank + 3, generator=generator)
    centers = torch.arange(candidate_count, dtype=torch.long) % elite_q0_rad.shape[0]
    q0 = elite_q0_rad[centers].float() + (noise[:, :rank] * coefficient_std) @ basis
    q0 = torch.maximum(torch.minimum(q0, comfort_upper), comfort_lower)
    palm_low, palm_high = official_palm_bounds_h(description)
    position_std = elite_position_h_m.float().std(dim=0, unbiased=False).clamp(
        min=torch.tensor((0.0005, 0.0005, 0.0002)),
        max=torch.tensor((0.006, 0.006, 0.002)),
    ) * (0.70**round_index)
    positions = elite_position_h_m[centers].float() + noise[:, rank:] * position_std
    positions[:, :2] = torch.maximum(torch.minimum(positions[:, :2], palm_high[:2]), palm_low[:2])
    z_low = palm_high[2] + OFFICIAL_PREGRASP_PALM_TOP_CLEARANCE_M - OFFICIAL_PREGRASP_PALM_Z_JITTER_M
    z_high = palm_high[2] + OFFICIAL_PREGRASP_PALM_TOP_CLEARANCE_M + OFFICIAL_PREGRASP_PALM_Z_JITTER_M
    positions[:, 2] = positions[:, 2].clamp(float(z_low), float(z_high))
    tip_centers = official_tip_center_positions_h(description, q0)
    envelope = fixed_position_envelope(tip_centers, torch.ones(candidate_count, 4, dtype=torch.bool), positions)
    margin = torch.minimum((q0 - lower) / span, (upper - q0) / span).amin(dim=-1)
    cheap = geometry_score(margin, envelope.tip_center_distances_m, envelope.sector_min_deg)
    cheap_pass = (
        (margin >= MVP80_STRICT_GOOD_PREGRASP_GATE.joint_margin_fraction_min)
        & (envelope.tip_center_distances_m.amax(dim=-1) <= MVP80_STRICT_GOOD_PREGRASP_GATE.tip_center_distance_m_max)
        & (envelope.sector_min_deg >= MVP80_STRICT_GOOD_PREGRASP_GATE.sector_min_deg)
    )
    return OfficialProposalBatch(
        q0_rad=q0,
        object_position_h_m=positions,
        tip_centers_h_m=tip_centers,
        non_thumb_pair=envelope.non_thumb_pair,
        envelope_fingers=tuple(_pair_to_fingers(pair) for pair in envelope.non_thumb_pair.tolist()),
        envelope_tip_center_distances_m=envelope.tip_center_distances_m,
        envelope_sector_min_deg=envelope.sector_min_deg,
        joint_margin_fraction=margin,
        cheap_score=cheap,
        cheap_pass=cheap_pass,
        seed_source=(f"cem_round_{round_index}",) * candidate_count,
    )


def _as_cpu_q_batch(q0_rad: Sequence[float] | torch.Tensor) -> torch.Tensor:
    'Normalize q input to CPU float64 [C,16].'

    tensor = torch.as_tensor(q0_rad, dtype=torch.float64, device="cpu")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != _JOINT_COUNT or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("official q must be finite with shape [16] or [C,16]")
    return tensor


def _owner_component_points(
    owner: Any,
    components: Mapping[str, Any],
    transforms: Mapping[str, np.ndarray],
    description: OfficialHandSemanticsCfg,
) -> np.ndarray:
    "Lower one owner's typed collision components into raw root frame {a}."

    parts: list[np.ndarray] = []
    for component_id in owner.component_ids:
        component = components.get(component_id)
        if component is None:
            raise ValueError(f"official owner {owner.owner_id!r} references unknown component {component_id!r}")
        carrier_transform = transforms.get(component.carrier_link)
        if carrier_transform is None:
            raise ValueError(f"official component {component_id!r} carrier link is absent from FK")
        local_vertices = _component_vertices(component)
        local_transform = _transform_from_rpy(component.origin_rpy_rad, component.origin_pos_m)
        homogeneous = np.concatenate((local_vertices, np.ones((len(local_vertices), 1))), axis=1)
        parts.append((carrier_transform @ local_transform @ homogeneous.T).T[:, :3])
    if not parts:
        raise ValueError(f"official owner {owner.owner_id!r} has no collision components")
    return np.concatenate(parts, axis=0)


def _component_vertices(component: Any) -> np.ndarray:
    'Read local vertices from an official typed collision component; never use marker geometry.'

    kind = str(component.geometry_kind)
    payload = component.geometry_payload
    if kind == "box":
        size = np.asarray(payload["size"], dtype=np.float64)
        if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ValueError(f"official box component {component.component_id!r} has invalid size")
        return np.asarray(
            [
                (sx * size[0] * 0.5, sy * size[1] * 0.5, sz * size[2] * 0.5)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
    if kind != "mesh":
        raise ValueError(f"official native center has no exact rule for collision kind {kind!r}")
    raw_path = payload.get("resolved_path") or payload.get("file_path")
    if not isinstance(raw_path, str):
        raise ValueError(f"official mesh component {component.component_id!r} lacks resolved path")
    scale = tuple(float(value) for value in payload.get("scale", (1.0, 1.0, 1.0)))
    if len(scale) != 3 or not all(math.isfinite(value) and value > 0.0 for value in scale):
        raise ValueError(f"official mesh component {component.component_id!r} has invalid scale")
    return _load_mesh_vertices(Path(raw_path).expanduser().resolve(strict=False), scale)


@lru_cache(maxsize=64)
def _load_mesh_vertices(path: Path, scale: tuple[float, float, float]) -> np.ndarray:
    'Cache read-only mesh vertices; never write to source artifacts.'

    if not path.is_file():
        raise FileNotFoundError(f"official collision mesh does not exist: {path}")
    import trimesh

    loaded = cast(Any, trimesh.load(path, force="mesh", process=False))
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.all(np.isfinite(vertices)):
        raise ValueError(f"official collision mesh has invalid vertices: {path}")
    return vertices * np.asarray(scale, dtype=np.float64)


def _transform_from_rpy(rpy: Sequence[float], translation: Sequence[float]) -> np.ndarray:
    'Build a 4x4 transform for URDF fixed-axis Rz(yaw)Ry(pitch)Rx(roll).'

    roll, pitch, yaw = (float(value) for value in rpy)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        ),
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tuple(float(value) for value in translation), dtype=np.float64)
    return transform


def _pair_to_fingers(pair: Sequence[int]) -> tuple[str, str, str]:
    'Encode index/middle/ring pair as thumb plus two non-thumb roles.'

    if (
        len(pair) != 2
        or len(set(int(value) for value in pair)) != 2
        or any(int(value) not in range(3) for value in pair)
    ):
        raise ValueError(f"official envelope pair must contain two distinct non-thumb indices, got {pair!r}")
    return ("thumb", _NON_THUMB_FINGERS[int(pair[0])], _NON_THUMB_FINGERS[int(pair[1])])


def official_selection_score(metrics: GoodPregraspMetrics) -> tuple[float, ...]:
    'Return deterministic quality-ranking vector for a strict candidate; larger is better.'

    peak_angular = (
        float(metrics.peak_angular_velocity_rad_s) if metrics.peak_angular_velocity_rad_s is not None else math.inf
    )
    return (
        float(metrics.joint_limit_margin_fraction),
        -max(metrics.envelope_tip_center_distance_m),
        float(metrics.envelope_sector_min_deg),
        -float(metrics.penetration_depth_max_m),
        -float(metrics.object_displacement_max_m),
        -float(metrics.object_tilt_max_deg),
        -float(metrics.peak_linear_velocity_m_s),
        -peak_angular,
        float(metrics.palm_contact_fraction),
    )


@dataclass(frozen=True)
class OfficialPregraspCandidateRecord:
    'Complete candidate/metrics/seed record for one proposal.'

    index: int
    candidate: GoodPregraspCandidate
    metrics: GoodPregraspMetrics
    seed_source: str
    tested: bool = True

    @property
    def strict_pass(self) -> bool:
        'Evaluate the raw metrics with the global strict v5 gate.'

        return MVP80_STRICT_GOOD_PREGRASP_GATE.accepts(self.metrics)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        'Return every strict-gate condition violated by this candidate.'

        return MVP80_STRICT_GOOD_PREGRASP_GATE.violations(self.metrics)


@dataclass(frozen=True)
class OfficialPregraspCertificate:
    'NPZ containing all candidates plus JSON certificate for strict Top-8.'

    identity: OfficialPregraspIdentity
    entry: GoodPregraspEntry
    npz_path: Path
    npz_sha256: str
    rank0_path: Path
    rank0_sha256: str
    top8_npz_indices: tuple[int, ...]
    candidate_count: int
    tested_count: int
    search_report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        'Return a JSON-safe certificate document.'

        return {
            "artifact_type": OFFICIAL_PREGRASP_ARTIFACT_TYPE,
            "schema_version": OFFICIAL_PREGRASP_SCHEMA_VERSION,
            "identity": self.identity.to_dict(),
            "npz_path": str(self.npz_path),
            "npz_sha256": self.npz_sha256,
            "rank0_path": str(self.rank0_path),
            "rank0_sha256": self.rank0_sha256,
            "top8_npz_indices": list(self.top8_npz_indices),
            "candidate_count": self.candidate_count,
            "tested_count": self.tested_count,
            "search_report": dict(self.search_report),
            "entry": self.entry.to_dict(),
        }


def write_official_pregrasp_artifacts(
    output_dir: Path,
    identity: OfficialPregraspIdentity,
    records: Sequence[OfficialPregraspCandidateRecord],
    *,
    search_report: Mapping[str, Any],
    gate: StrictGoodPregraspGate = MVP80_STRICT_GOOD_PREGRASP_GATE,
) -> OfficialPregraspCertificate | None:
    (
        'Atomically write all-candidate NPZ/search report and write a certificate '
        'only when all Top-8 pass. If Top-8 is incomplete, return None but preserve '
        'failed candidates/report so callers can record search failure.'
    )

    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = tuple(sorted(records, key=lambda record: record.index))
    if tuple(record.index for record in ordered) != tuple(range(len(ordered))):
        raise ValueError("official candidate records must have contiguous indices 0..N-1")
    npz_path = output_dir / "candidates.npz"
    _atomic_write_npz(npz_path, _records_to_arrays(ordered))
    npz_sha = _sha256_file(npz_path)
    strict_records = [record for record in ordered if record.tested and record.strict_pass]
    report = dict(search_report)
    report.update(
        {
            "artifact_type": OFFICIAL_PREGRASP_ARTIFACT_TYPE,
            "schema_version": OFFICIAL_PREGRASP_SCHEMA_VERSION,
            "candidate_count": len(ordered),
            "tested_count": sum(record.tested for record in ordered),
            "strict_pass_count": len(strict_records),
            "failure_reason_counts": dict(
                Counter(reason for record in ordered if record.tested for reason in record.reason_codes)
            ),
            "candidate_npz": str(npz_path.name),
            "candidate_npz_sha256": npz_sha,
        }
    )
    _atomic_write_json(output_dir / "search-report.json", report)
    if len(strict_records) < GOOD_PREGRASP_TOP_K:
        return None
    ranked = sorted(
        strict_records, key=lambda record: (official_selection_score(record.metrics), -record.index), reverse=True
    )
    selected = tuple(ranked[:GOOD_PREGRASP_TOP_K])
    key = _build_good_catalog_key(identity)
    members = tuple(
        GoodPregraspMember(
            rank=rank,
            candidate=record.candidate,
            metrics=record.metrics,
            selection_score=official_selection_score(record.metrics),
        )
        for rank, record in enumerate(selected)
    )
    entry = GoodPregraspEntry(key=key, members=members)
    gate.validate_entry(entry)
    rank0_path = output_dir / "rank0.npz"
    _atomic_write_npz(
        rank0_path,
        {
            "q0_rad": np.asarray(selected[0].candidate.q_state_rad, dtype=np.float64),
            "q_target_rad": np.asarray(selected[0].candidate.q_target_rad, dtype=np.float64),
            "object_position_h_m": np.asarray(selected[0].candidate.object_position_h_m, dtype=np.float64),
            "object_orientation_h_wxyz": np.asarray(selected[0].candidate.object_orientation_h_wxyz, dtype=np.float64),
        },
    )
    rank0_sha = _sha256_file(rank0_path)
    _atomic_write_json(
        output_dir / "rank0.json",
        {
            "artifact_type": OFFICIAL_PREGRASP_ARTIFACT_TYPE + ".rank0",
            "schema_version": OFFICIAL_PREGRASP_SCHEMA_VERSION,
            "strict_gate_passed": True,
            "artifact_sha256": rank0_sha,
            "native_source_sha256": identity.source_urdf_sha256,
            "identity_digest": identity.identity_digest,
            "certificate_path": "certificate.json",
            "rank0_npz": rank0_path.name,
            "q0_rad": list(selected[0].candidate.q_state_rad),
            "object_position_h_m": list(selected[0].candidate.object_position_h_m),
        },
    )
    report.update(
        {
            "status": "strict_top8_ready",
            "top8_npz_indices": [record.index for record in selected],
            "rank0_npz": rank0_path.name,
            "rank0_npz_sha256": rank0_sha,
            "rank0": {
                "q0_rad": list(selected[0].candidate.q_state_rad),
                "object_position_h_m": list(selected[0].candidate.object_position_h_m),
            },
        }
    )
    _atomic_write_json(output_dir / "search-report.json", report)
    certificate = OfficialPregraspCertificate(
        identity=identity,
        entry=entry,
        npz_path=npz_path,
        npz_sha256=npz_sha,
        rank0_path=rank0_path,
        rank0_sha256=rank0_sha,
        top8_npz_indices=tuple(record.index for record in selected),
        candidate_count=len(ordered),
        tested_count=sum(record.tested for record in ordered),
        search_report=report,
    )
    _atomic_write_json(output_dir / "certificate.json", certificate.to_dict())
    return certificate


def load_official_pregrasp_certificate(path: Path) -> OfficialPregraspCertificate:
    'Read and revalidate official certificate, NPZ hash, Top-8, and every strict metric.'

    certificate_path = Path(path).expanduser().resolve(strict=True)
    document = json.loads(certificate_path.read_text(encoding="utf-8"))
    if document.get("artifact_type") != OFFICIAL_PREGRASP_ARTIFACT_TYPE:
        raise ValueError("unexpected official pregrasp artifact_type")
    if document.get("schema_version") != OFFICIAL_PREGRASP_SCHEMA_VERSION:
        raise ValueError("unsupported official pregrasp schema_version")
    identity = OfficialPregraspIdentity.from_dict(document["identity"])
    raw_npz_path = Path(str(document["npz_path"]))
    npz_path = raw_npz_path if raw_npz_path.is_absolute() else certificate_path.parent / raw_npz_path
    npz_path = npz_path.resolve(strict=True)
    npz_sha = _sha256_file(npz_path)
    if npz_sha != str(document["npz_sha256"]):
        raise ValueError("official candidate NPZ SHA-256 mismatch")
    raw_rank0_path = Path(str(document["rank0_path"]))
    rank0_path = raw_rank0_path if raw_rank0_path.is_absolute() else certificate_path.parent / raw_rank0_path
    rank0_path = rank0_path.resolve(strict=True)
    rank0_sha = _sha256_file(rank0_path)
    if rank0_sha != str(document["rank0_sha256"]):
        raise ValueError("official rank0 NPZ SHA-256 mismatch")
    try:
        rank0_arrays = np.load(rank0_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read official rank0 NPZ: {exc}") from exc
    rank0_q0: np.ndarray
    rank0_position: np.ndarray
    with rank0_arrays:
        for name, shape in (
            ("q0_rad", (16,)),
            ("q_target_rad", (16,)),
            ("object_position_h_m", (3,)),
            ("object_orientation_h_wxyz", (4,)),
        ):
            if name not in rank0_arrays or tuple(rank0_arrays[name].shape) != shape:
                raise ValueError(f"official rank0 NPZ array {name!r} has invalid shape")
            if not np.isfinite(np.asarray(rank0_arrays[name])).all():
                raise ValueError(f"official rank0 NPZ array {name!r} contains non-finite values")
        if not np.array_equal(rank0_arrays["q0_rad"], rank0_arrays["q_target_rad"]):
            raise ValueError("official rank0 NPZ q0/u0 contract is not exact")
        if not np.array_equal(rank0_arrays["object_orientation_h_wxyz"], np.asarray((1.0, 0.0, 0.0, 0.0))):
            raise ValueError("official rank0 NPZ object is not exactly upright")
        rank0_q0 = np.asarray(rank0_arrays["q0_rad"]).copy()
        rank0_position = np.asarray(rank0_arrays["object_position_h_m"]).copy()
    try:
        arrays = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read official candidate NPZ: {exc}") from exc
    with arrays:
        _validate_npz_arrays(arrays)
        entry = GoodPregraspEntry.from_dict(document["entry"])
        MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)
        if entry.key != _build_good_catalog_key(identity):
            raise ValueError("official certificate key does not close over its identity")
        if int(document["candidate_count"]) != int(arrays["q0_rad"].shape[0]):
            raise ValueError("official certificate candidate_count disagrees with NPZ")
        top8_indices = tuple(int(value) for value in document["top8_npz_indices"])
        if len(top8_indices) != GOOD_PREGRASP_TOP_K or len(set(top8_indices)) != GOOD_PREGRASP_TOP_K:
            raise ValueError("official certificate must reference exactly eight unique NPZ rows")
        if any(index < 0 or index >= arrays["q0_rad"].shape[0] for index in top8_indices):
            raise ValueError("official Top-8 NPZ row index is out of range")
        if not all(bool(arrays["tested"][index]) and bool(arrays["strict_pass"][index]) for index in top8_indices):
            raise ValueError("official certificate Top-8 rows are not tested strict-pass candidates")
        for index in np.flatnonzero(np.asarray(arrays["tested"], dtype=np.bool_)):
            candidate = _candidate_from_npz(arrays, int(index))
            metrics = _metrics_from_npz(arrays, int(index))
            strict_pass = bool(arrays["strict_pass"][index])
            if strict_pass != MVP80_STRICT_GOOD_PREGRASP_GATE.accepts(metrics):
                raise ValueError(f"official NPZ strict_pass disagrees with metrics at row {int(index)}")
            if not candidate.active_joint_mask or candidate.q_state_rad != candidate.q_target_rad:
                raise ValueError(f"official NPZ candidate contract is invalid at row {int(index)}")
        for rank, index in enumerate(top8_indices):
            candidate = _candidate_from_npz(arrays, index)
            metrics = _metrics_from_npz(arrays, index)
            member = entry.members[rank]
            if candidate.to_dict() != member.candidate.to_dict():
                raise ValueError(f"official Top-8 candidate row {index} disagrees with JSON rank {rank}")
            if not _metrics_close(metrics, member.metrics):
                raise ValueError(f"official Top-8 metrics row {index} disagrees with JSON rank {rank}")
            if not MVP80_STRICT_GOOD_PREGRASP_GATE.accepts(metrics):
                raise ValueError(f"official Top-8 metrics row {index} fails strict gate")
            expected_score = official_selection_score(metrics)
            if len(member.selection_score) != len(expected_score) or not np.allclose(
                np.asarray(member.selection_score, dtype=np.float64),
                np.asarray(expected_score, dtype=np.float64),
                atol=2e-6,
                rtol=0.0,
            ):
                raise ValueError(f"official Top-8 selection score disagrees with metrics at rank {rank}")
            if rank == 0:
                if not np.array_equal(
                    rank0_q0,
                    np.asarray(candidate.q_state_rad, dtype=np.float64),
                ) or not np.array_equal(
                    rank0_position,
                    np.asarray(candidate.object_position_h_m, dtype=np.float64),
                ):
                    raise ValueError("official rank0 NPZ disagrees with certificate rank 0")
        tested_count = int(np.asarray(arrays["tested"], dtype=np.bool_).sum())
    if tested_count != int(document["tested_count"]):
        raise ValueError("official certificate tested_count disagrees with NPZ")
    return OfficialPregraspCertificate(
        identity=identity,
        entry=entry,
        npz_path=npz_path,
        npz_sha256=npz_sha,
        rank0_path=rank0_path,
        rank0_sha256=rank0_sha,
        top8_npz_indices=top8_indices,
        candidate_count=int(document["candidate_count"]),
        tested_count=tested_count,
        search_report=dict(document["search_report"]),
    )


def _build_good_catalog_key(identity: OfficialPregraspIdentity) -> GoodPregraspKey:
    'Lower official identity into the existing schema-3 exact key.'

    canonical_schema_digest = _stable_digest(
        {
            "schema": "official-native-16d",
            "joint_count": 16,
            "finger_order": list(OFFICIAL_FINGER_ORDER),
            "storage": "description.joints canonical depth-major",
        }
    )
    routing_digest = _stable_digest(
        {
            "canonical_source_joint_names": [
                str(item["source_joint_name"]) for item in identity.controller_identity["joint_mapping"]
            ],
            "canonical_slots": list(OFFICIAL_FINGER_ORDER),
            "usd_joint_names": [str(item["usd_joint_name"]) for item in identity.controller_identity["joint_mapping"]],
        }
    )
    return GoodPregraspKey(
        asset_id=identity.asset_id,
        source_content_hash=identity.source_urdf_sha256,
        physical_geometry_hash=identity.source_content_hash,
        canonical_schema_digest=canonical_schema_digest,
        routing_digest=routing_digest,
        object_asset_id=identity.object_asset_id,
        object_asset_sha256=identity.object_asset_sha256,
        object_scale=identity.object_scale,
        physics_identity_digest=_stable_digest(identity.physics_identity),
        generation_identity_digest=_stable_digest(identity.generation_identity),
    )


def _records_to_arrays(records: Sequence[OfficialPregraspCandidateRecord]) -> dict[str, np.ndarray]:
    'Lower all candidate records to NPZ arrays without object pickle.'

    count = len(records)
    # Candidate state/pose stays float64 so the JSON rank payload and the NPZ row
    # round-trip to the same Python floats; physical metrics may remain compact
    # float32 below because the loader compares them with an explicit tolerance.
    q = np.asarray([record.candidate.q_state_rad for record in records], dtype=np.float64).reshape(count, 16)
    position = np.asarray([record.candidate.object_position_h_m for record in records], dtype=np.float64).reshape(
        count, 3
    )
    masks = np.asarray([record.candidate.active_joint_mask for record in records], dtype=np.bool_).reshape(count, 16)
    pair = np.asarray(
        [
            [
                _NON_THUMB_FINGERS.index(record.metrics.envelope_fingers[1]),
                _NON_THUMB_FINGERS.index(record.metrics.envelope_fingers[2]),
            ]
            for record in records
        ],
        dtype=np.int64,
    ).reshape(count, 2)
    owner = np.asarray([record.metrics.owner_contact_fraction for record in records], dtype=np.float32).reshape(
        count, 21
    )
    return {
        "q0_rad": q,
        "q_target_rad": q.copy(),
        "active_joint_mask": masks,
        "object_position_h_m": position,
        "object_orientation_h_wxyz": np.tile(np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64), (count, 1)),
        "tested": np.asarray([record.tested for record in records], dtype=np.bool_),
        "strict_pass": np.asarray([record.tested and record.strict_pass for record in records], dtype=np.bool_),
        "seed_source_code": np.asarray([_seed_source_code(record.seed_source) for record in records], dtype=np.int64),
        "candidate_index": np.asarray([record.index for record in records], dtype=np.int64),
        "envelope_pair": pair,
        "joint_limit_margin_fraction": np.asarray(
            [record.metrics.joint_limit_margin_fraction for record in records], dtype=np.float32
        ),
        "envelope_sector_min_deg": np.asarray(
            [record.metrics.envelope_sector_min_deg for record in records], dtype=np.float32
        ),
        "envelope_tip_center_distance_m": np.asarray(
            [record.metrics.envelope_tip_center_distance_m for record in records], dtype=np.float32
        ).reshape(count, 3),
        "penetration_depth_max_m": np.asarray(
            [record.metrics.penetration_depth_max_m for record in records], dtype=np.float32
        ),
        "object_displacement_max_m": np.asarray(
            [record.metrics.object_displacement_max_m for record in records], dtype=np.float32
        ),
        "object_tilt_max_deg": np.asarray([record.metrics.object_tilt_max_deg for record in records], dtype=np.float32),
        "peak_linear_velocity_m_s": np.asarray(
            [record.metrics.peak_linear_velocity_m_s for record in records], dtype=np.float32
        ),
        "peak_off_axis_angular_velocity_rad_s": np.asarray(
            [record.metrics.peak_off_axis_angular_velocity_rad_s for record in records], dtype=np.float32
        ),
        "peak_angular_velocity_rad_s": np.asarray(
            [
                record.metrics.peak_angular_velocity_rad_s
                if record.metrics.peak_angular_velocity_rad_s is not None
                else np.nan
                for record in records
            ],
            dtype=np.float32,
        ),
        "peak_angular_velocity_present": np.asarray(
            [record.metrics.peak_angular_velocity_rad_s is not None for record in records], dtype=np.bool_
        ),
        "palm_contact_fraction": np.asarray(
            [record.metrics.palm_contact_fraction for record in records], dtype=np.float32
        ),
        "owner_contact_fraction": owner,
    }


def _candidate_from_npz(arrays: Mapping[str, Any], index: int) -> GoodPregraspCandidate:
    'Restore and validate q0=u0, upright pose, and active mask from an NPZ row.'

    return GoodPregraspCandidate(
        q_state_rad=tuple(float(value) for value in arrays["q0_rad"][index]),
        q_target_rad=tuple(float(value) for value in arrays["q_target_rad"][index]),
        active_joint_mask=tuple(bool(value) for value in arrays["active_joint_mask"][index]),
        object_position_h_m=cast(
            tuple[float, float, float], tuple(float(value) for value in arrays["object_position_h_m"][index])
        ),
        object_orientation_h_wxyz=cast(
            tuple[float, float, float, float],
            tuple(float(value) for value in arrays["object_orientation_h_wxyz"][index]),
        ),
    )


def _metrics_from_npz(arrays: Mapping[str, Any], index: int) -> GoodPregraspMetrics:
    'Restore raw GoodPregraspMetrics from an NPZ row.'

    pair = tuple(int(value) for value in arrays["envelope_pair"][index])
    return GoodPregraspMetrics(
        joint_limit_margin_fraction=float(arrays["joint_limit_margin_fraction"][index]),
        envelope_fingers=_pair_to_fingers(pair),
        envelope_sector_min_deg=float(arrays["envelope_sector_min_deg"][index]),
        envelope_tip_center_distance_m=cast(
            tuple[float, float, float],
            tuple(float(value) for value in arrays["envelope_tip_center_distance_m"][index]),
        ),
        penetration_depth_max_m=float(arrays["penetration_depth_max_m"][index]),
        object_displacement_max_m=float(arrays["object_displacement_max_m"][index]),
        object_tilt_max_deg=float(arrays["object_tilt_max_deg"][index]),
        peak_linear_velocity_m_s=float(arrays["peak_linear_velocity_m_s"][index]),
        peak_off_axis_angular_velocity_rad_s=float(arrays["peak_off_axis_angular_velocity_rad_s"][index]),
        palm_contact_fraction=float(arrays["palm_contact_fraction"][index]),
        owner_contact_fraction=tuple(float(value) for value in arrays["owner_contact_fraction"][index]),
        peak_angular_velocity_rad_s=(
            float(arrays["peak_angular_velocity_rad_s"][index])
            if bool(arrays["peak_angular_velocity_present"][index])
            else None
        ),
    )


def _metrics_close(left: GoodPregraspMetrics, right: GoodPregraspMetrics) -> bool:
    'Compare NPZ float32 row with JSON metrics while preserving strict fields/units.'

    left_dict, right_dict = left.to_dict(), right.to_dict()
    for key, left_value in left_dict.items():
        right_value = right_dict[key]
        if isinstance(left_value, list):
            if left_value and isinstance(left_value[0], str):
                if left_value != right_value:
                    return False
            elif not np.allclose(
                np.asarray(left_value, dtype=np.float64),
                np.asarray(right_value, dtype=np.float64),
                atol=2e-6,
                rtol=0.0,
            ):
                return False
        elif isinstance(left_value, float):
            if not math.isclose(left_value, float(right_value), abs_tol=2e-6, rel_tol=0.0):
                return False
        elif left_value != right_value:
            return False
    return True


def _validate_npz_arrays(arrays: Any) -> None:
    'Validate all-candidate NPZ shapes, dtypes, indices, and finiteness of tested rows.'

    required = {
        "q0_rad",
        "q_target_rad",
        "active_joint_mask",
        "object_position_h_m",
        "object_orientation_h_wxyz",
        "tested",
        "strict_pass",
        "seed_source_code",
        "candidate_index",
        "envelope_pair",
        "joint_limit_margin_fraction",
        "envelope_sector_min_deg",
        "envelope_tip_center_distance_m",
        "penetration_depth_max_m",
        "object_displacement_max_m",
        "object_tilt_max_deg",
        "peak_linear_velocity_m_s",
        "peak_off_axis_angular_velocity_rad_s",
        "peak_angular_velocity_rad_s",
        "palm_contact_fraction",
        "owner_contact_fraction",
        "peak_angular_velocity_present",
    }
    missing = required - set(arrays.files)
    if missing:
        raise ValueError(f"official candidate NPZ lacks arrays: {sorted(missing)}")
    count = int(arrays["q0_rad"].shape[0])
    expected_shapes = {
        "q0_rad": (count, 16),
        "q_target_rad": (count, 16),
        "active_joint_mask": (count, 16),
        "object_position_h_m": (count, 3),
        "object_orientation_h_wxyz": (count, 4),
        "tested": (count,),
        "strict_pass": (count,),
        "seed_source_code": (count,),
        "candidate_index": (count,),
        "envelope_pair": (count, 2),
        "envelope_tip_center_distance_m": (count, 3),
        "owner_contact_fraction": (count, 21),
        "peak_angular_velocity_present": (count,),
    }
    for name, shape in expected_shapes.items():
        if tuple(arrays[name].shape) != shape:
            raise ValueError(f"official candidate NPZ array {name!r} has shape {arrays[name].shape}, expected {shape}")
    if not np.array_equal(np.asarray(arrays["candidate_index"], dtype=np.int64), np.arange(count)):
        raise ValueError("official candidate NPZ indices are not contiguous")
    tested = np.asarray(arrays["tested"], dtype=np.bool_)
    finite_names = (
        "q0_rad",
        "q_target_rad",
        "object_position_h_m",
        "object_orientation_h_wxyz",
        "joint_limit_margin_fraction",
        "envelope_sector_min_deg",
        "envelope_tip_center_distance_m",
        "penetration_depth_max_m",
        "object_displacement_max_m",
        "object_tilt_max_deg",
        "peak_linear_velocity_m_s",
        "peak_off_axis_angular_velocity_rad_s",
        "palm_contact_fraction",
        "owner_contact_fraction",
    )
    for name in finite_names:
        if not np.isfinite(np.asarray(arrays[name])[tested]).all():
            raise ValueError(f"official candidate NPZ tested array {name!r} contains non-finite values")
    peak_present = np.asarray(arrays["peak_angular_velocity_present"], dtype=np.bool_)
    if np.any(tested & ~peak_present):
        # GoodPregraspMetrics allows ``None`` for failed/raw rows, but a tested official
        # candidate must carry the total angular velocity required by strict v5.
        raise ValueError("tested official NPZ row lacks total angular velocity")
    if np.any(np.asarray(arrays["strict_pass"], dtype=np.bool_)[~tested]):
        raise ValueError("untested official NPZ row cannot be strict_pass")


def _seed_source_code(seed_source: str) -> int:
    'Encode proposal seed provenance using integer arrays, never object/pickle.'

    if seed_source == "sobol":
        return 0
    if seed_source.startswith("cem_round_"):
        return 1 + int(seed_source.removeprefix("cem_round_"))
    if seed_source == "legacy_seed":
        return 100
    raise ValueError(f"unknown official proposal seed source {seed_source!r}")


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    'Atomically publish NPZ in the target directory; interruption cannot leave a partial candidate file.'

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **dict(arrays))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    'Atomically write report/certificate as canonical JSON plus replace.'

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


__all__ = [
    "OFFICIAL_FINGER_ORDER",
    "OFFICIAL_PREGRASP_ARTIFACT_TYPE",
    "OFFICIAL_PREGRASP_CEM_CANDIDATES",
    "OFFICIAL_PREGRASP_CEM_ROUNDS",
    "OFFICIAL_PREGRASP_GEOMETRY_TOP_K",
    "OFFICIAL_PREGRASP_PALM_TOP_CLEARANCE_M",
    "OFFICIAL_PREGRASP_PHYSICS_STEPS",
    "OFFICIAL_PREGRASP_SCHEMA_VERSION",
    "OFFICIAL_PREGRASP_SOBOL_CANDIDATES",
    "OfficialPregraspCandidateRecord",
    "OfficialPregraspCertificate",
    "OfficialPregraspIdentity",
    "OfficialProposalBatch",
    "build_official_controller_identity",
    "build_official_physics_identity",
    "build_official_pregrasp_identity",
    "build_official_source_implementation_identity",
    "generate_official_cem_candidates",
    "generate_official_sobol_candidates",
    "load_official_pregrasp_certificate",
    "official_palm_bounds_h",
    "official_selection_score",
    "official_tip_center_positions_h",
    "validate_official_pregrasp_for_session",
    "write_official_pregrasp_artifacts",
]
