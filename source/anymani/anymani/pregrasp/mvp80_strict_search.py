(
    'Pure-Torch geometry proposals, strict gates, and low-rank CEM for MVP80 '
    'pregrasp. Initial proposals use 13 scrambled Sobol dimensions: N000/midpoint '
    'blend (1), depth synergies (4), finger synergies (4), opposition mix (1), '
    'in-plane offsets (2), and palm-height clearance (1). CEM does not fit 16 '
    'joint slots independently: derive four PCA directions from physically '
    'screened elites and combine them with 3D object position, giving seven '
    'refinement variables per asset.'
)

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from .strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE, StrictGoodPregraspGate

STRICT_SOBOL_DIMENSION = 13
STRICT_CEM_JOINT_RANK = 4
PROPOSAL_TIP_CLEARANCE_M = 0.060  # Cheap collision-free proxy lower bound; publication still depends on real penetration gate.
CEM_PROPOSAL_CENTER_COUNTS = (24, 20, 16, 12, 8, 8, 6, 6, 4, 4, 4, 4, 3, 3, 3, 3)
'Allocate 128 CEM proposals by descending elite quality so strong contact modes receive more local samples.'
CEM_PHYSICS_CENTER_COUNTS = (8, 6, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
'Reserve 32 full-physics trials while covering the first 13 elites to reduce premature single-mode collapse.'
N000_CANONICAL_Q = (
    0.0,
    0.0,
    0.0,
    0.88,
    -0.61000001,
    -0.12,
    0.56,
    1.73000002,
    1.05999994,
    1.17999995,
    1.51999998,
    0.71999997,
    0.93000001,
    0.57999998,
    0.44,
    1.63,
)  # depth-major index/middle/ring/thumb，rad


def stable_asset_stream_key(*identity_parts: str) -> int:
    (
        'Map source/physical identity to a deterministic positive integer stream key. '
        'Dataset row is config-local and may differ across PPO/SSL manifests, so '
        'derive cohort proposals from immutable identity. Keep key in [1,2^31-1] so '
        'seed+key*104729 stays in a conservative backend range; reordering the cohort '
        "does not change an asset's proposals."
    )

    parts = tuple(str(part).strip() for part in identity_parts)  # Ordered identity fields; order is part of the random-stream protocol.
    if not parts or any(not part for part in parts):
        raise ValueError("asset stream key requires non-empty identity parts")
    payload = b"\0".join(part.encode("utf-8") for part in parts)  # NUL cannot occur in SHA/ID fields and separates them unambiguously.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1) + 1  # Range [1,2^31-1].


@dataclass(frozen=True)
class EnvelopeResult:
    'Select the thumb-plus-two-non-thumb envelope and object position for each batch.'

    object_position_h_m: torch.Tensor  # `[B,3]`
    non_thumb_pair: torch.Tensor  # long`[B,2]`，0=index/1=middle/2=ring
    tip_center_distances_m: torch.Tensor  # `[B,3]`，thumb first
    sector_min_deg: torch.Tensor  # `[B]`


def deepest_contact_normal_from_buffers(
    normals: torch.Tensor,
    separations: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    *,
    environment_count: int,
    body_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        "Recover each env's deepest penetration and normal from PhysX packed "
        'contacts. This pure-Python helper imports no Isaac/Omni and supports direct '
        'contract checks of env/body/filter grouping and start/count interpretation.'
    )

    flat_counts = counts.reshape(-1).long()
    flat_starts = starts.reshape(-1).long()
    if environment_count < 1 or body_count < 1 or flat_counts.numel() % (environment_count * body_count):
        raise ValueError("contact buffer groups disagree with environment/body cardinality")
    group_ids = torch.repeat_interleave(torch.arange(flat_counts.numel(), device=flat_counts.device), flat_counts)
    if group_ids.numel() == 0:
        return (
            torch.zeros(environment_count, dtype=torch.float32, device=flat_counts.device),
            torch.zeros(environment_count, 3, dtype=torch.float32, device=flat_counts.device),
        )
    block_starts = flat_counts.cumsum(0) - flat_counts
    offsets = torch.arange(group_ids.numel(), device=group_ids.device) - block_starts.repeat_interleave(flat_counts)
    buffer_indices = flat_starts[group_ids] + offsets
    contact_separation = separations.reshape(-1).index_select(0, buffer_indices)
    contact_normal = normals.reshape(-1, 3).index_select(0, buffer_indices)
    groups_per_env = flat_counts.numel() // environment_count
    contact_env = torch.div(group_ids, groups_per_env, rounding_mode="floor")
    minimum = torch.full(
        (environment_count,), torch.inf, dtype=contact_separation.dtype, device=contact_separation.device
    )
    minimum.scatter_reduce_(0, contact_env, contact_separation, reduce="amin", include_self=True)
    contact_order = torch.arange(contact_env.numel(), device=contact_env.device)
    sentinel = torch.full_like(contact_order, contact_order.numel())
    candidates = torch.where(contact_separation == minimum[contact_env], contact_order, sentinel)
    selected = torch.full((environment_count,), contact_order.numel(), dtype=torch.long, device=contact_env.device)
    selected.scatter_reduce_(0, contact_env, candidates, reduce="amin", include_self=True)
    penetrating = torch.isfinite(minimum) & (minimum < 0.0) & (selected < contact_order.numel())
    depth = torch.where(penetrating, -minimum, torch.zeros_like(minimum))
    output_normal = torch.zeros(environment_count, 3, dtype=contact_normal.dtype, device=contact_normal.device)
    output_normal[penetrating] = contact_normal[selected[penetrating]]
    return depth, output_normal


def sobol_bank(
    asset_keys: tuple[int, ...],
    *,
    candidate_count: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    (
        'Generate a separate scrambled Sobol sequence per stable asset key, invariant '
        'to cohort order. Return [A,C,13], approximately inside the open unit '
        'interval.'
    )

    if candidate_count < 1 or not asset_keys or len(set(asset_keys)) != len(asset_keys):
        raise ValueError("Sobol bank requires positive candidates and unique asset keys")
    rows = []
    for asset_key in asset_keys:
        engine = torch.quasirandom.SobolEngine(
            dimension=STRICT_SOBOL_DIMENSION,
            scramble=True,
            seed=int(seed + asset_key * 104729),
        )  # Asset-local scramble makes proposals independent of cohort order/prefix.
        rows.append(engine.draw(candidate_count, dtype=torch.float32))
    return torch.stack(rows, dim=0).to(device)  # `[A,C,13]`


def initial_joint_candidates(
    lower: torch.Tensor,
    upper: torch.Tensor,
    active_mask: torch.Tensor,
    sobol: torch.Tensor,
    *,
    margin_fraction: float = 0.11,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        'Generate comfort-domain joint states from N000/midpoint and depth/finger '
        'synergies. Inputs lower/upper [A,16] rad, active_mask bool [A,16], Sobol '
        '[A,C,13]. Keep the configured margin on each side (strict v5 uses 10%). '
        'Return canonical q [A,C,16] and actual minimum margin [A,C].'
    )

    if lower.shape != upper.shape or lower.shape != active_mask.shape or lower.shape[-1] != 16:
        raise ValueError("joint candidate limits/mask must share shape [A,16]")
    if sobol.shape[:1] != lower.shape[:1] or sobol.shape[-1] != STRICT_SOBOL_DIMENSION:
        raise ValueError("joint candidate Sobol bank must have shape [A,C,13]")
    if not 0.0 < margin_fraction < 0.5:
        raise ValueError("joint comfort margin must lie in (0,0.5)")
    span = (upper - lower).clamp_min(1.0e-6)  # `[A,16]`，rad
    comfortable_lower = lower + margin_fraction * span
    comfortable_upper = upper - margin_fraction * span
    midpoint = 0.5 * (lower + upper)
    template = torch.tensor(N000_CANONICAL_Q, dtype=lower.dtype, device=lower.device).expand_as(lower)
    template = torch.maximum(torch.minimum(template, comfortable_upper), comfortable_lower)

    # Blend w in [0.45,0.95]; expand remaining perturbations by depth/finger synergies to depth-major 16 slots.
    blend = (0.45 + 0.50 * sobol[..., 0]).unsqueeze(-1)  # `[A,C,1]`
    base = blend * template.unsqueeze(1) + (1.0 - blend) * midpoint.unsqueeze(1)
    depth_offset = (sobol[..., 1:5] - 0.5) * 0.24  # Four depth synergies in [-0.12,0.12] rad.
    finger_offset = (sobol[..., 5:9] - 0.5) * 0.16  # Four finger synergies in [-0.08,0.08] rad.
    offsets = depth_offset.unsqueeze(-1) + finger_offset.unsqueeze(-2)  # `[A,C,4 depth,4 finger]`
    q = base + offsets.reshape(*base.shape[:-1], 16)
    q = torch.maximum(torch.minimum(q, comfortable_upper.unsqueeze(1)), comfortable_lower.unsqueeze(1))
    q = q * active_mask.unsqueeze(1)  # Ghost slots remain exactly zero.
    normalized_margin = torch.minimum(
        (q - lower.unsqueeze(1)) / span.unsqueeze(1),
        (upper.unsqueeze(1) - q) / span.unsqueeze(1),
    )
    margin = torch.where(active_mask.unsqueeze(1), normalized_margin, torch.inf).amin(dim=-1)
    return q, margin


def _minimum_sector_degrees(vectors_xy: torch.Tensor) -> torch.Tensor:
    'Compute the minimum undirected in-plane pair angle among three tips relative to object center.'

    normalized = vectors_xy / torch.linalg.vector_norm(vectors_xy, dim=-1, keepdim=True).clamp_min(1.0e-8)
    pair_indices = ((0, 1), (0, 2), (1, 2))
    cosines = torch.stack(
        [torch.sum(normalized[:, left] * normalized[:, right], dim=-1) for left, right in pair_indices],
        dim=-1,
    )
    return torch.rad2deg(torch.acos(cosines.clamp(-1.0, 1.0))).amin(dim=-1)


def _select_pair(
    tip_positions_h: torch.Tensor,
    active_tip_mask: torch.Tensor,
    candidate_positions_h: torch.Tensor,
) -> EnvelopeResult:
    'Choose the best combined non-thumb pair under the 10 cm/30-degree gate.'

    pair_table = torch.tensor(((0, 1), (0, 2), (1, 2)), dtype=torch.long, device=tip_positions_h.device)
    pair_distances: list[torch.Tensor] = []
    pair_sectors: list[torch.Tensor] = []
    pair_validity: list[torch.Tensor] = []
    for pair in pair_table:
        fingers = torch.cat((tip_positions_h[:, 3:4], tip_positions_h[:, pair]), dim=1)  # thumb first
        distances = torch.linalg.vector_norm(fingers - candidate_positions_h.unsqueeze(1), dim=-1)
        sector = _minimum_sector_degrees(fingers[:, :, :2] - candidate_positions_h[:, None, :2])
        valid = active_tip_mask[:, 3] & active_tip_mask[:, pair[0]] & active_tip_mask[:, pair[1]]
        pair_distances.append(distances)
        pair_sectors.append(sector)
        pair_validity.append(valid)
    distances = torch.stack(pair_distances, dim=1)  # `[B,3 pairs,3 fingers]`
    sectors = torch.stack(pair_sectors, dim=1)  # `[B,3]`
    validity = torch.stack(pair_validity, dim=1)
    minimum_distance = distances.amin(dim=-1)
    geometric_pass = (
        (distances.amax(dim=-1) <= 0.10)
        & (minimum_distance >= PROPOSAL_TIP_CLEARANCE_M)
        & (sectors >= 30.0)
        & validity
    )
    clearance_deficit = torch.clamp(PROPOSAL_TIP_CLEARANCE_M - minimum_distance, min=0.0)
    score = (
        10.0 * geometric_pass.float()
        + sectors / 180.0
        - distances.amax(dim=-1) / 0.10
        - 5.0 * clearance_deficit / PROPOSAL_TIP_CLEARANCE_M
    )
    score = torch.where(validity, score, torch.full_like(score, -torch.inf))
    if bool((~torch.isfinite(score).any(dim=-1)).any().item()):
        raise RuntimeError("one asset lacks thumb plus two active non-thumb fingertips")
    best = score.argmax(dim=-1)
    rows = torch.arange(best.shape[0], device=best.device)
    return EnvelopeResult(candidate_positions_h, pair_table[best], distances[rows, best], sectors[rows, best])


def initial_envelope(
    tip_positions_h: torch.Tensor,
    active_tip_mask: torch.Tensor,
    sobol: torch.Tensor,
) -> EnvelopeResult:
    'Generate object centers from real TIP forward kinematics and Sobol opposition mix/offset; select the best envelope.'

    if sobol.shape != (tip_positions_h.shape[0], STRICT_SOBOL_DIMENSION):
        raise ValueError("initial envelope requires one 13D Sobol vector per candidate")
    default_xy = torch.tensor((0.00578, 0.08511), dtype=tip_positions_h.dtype, device=tip_positions_h.device)
    # Use the mean active non-thumb tip location for a coarse opposition center; score each final pair strictly.
    non_thumb_weight = active_tip_mask[:, :3].unsqueeze(-1).to(tip_positions_h.dtype)
    non_thumb_mean = (tip_positions_h[:, :3, :2] * non_thumb_weight).sum(dim=1) / non_thumb_weight.sum(
        dim=1
    ).clamp_min(1.0)
    opposition_xy = 0.5 * (tip_positions_h[:, 3, :2] + non_thumb_mean)
    mix = (0.55 + 0.40 * sobol[:, 9]).unsqueeze(-1)  # Opposition fraction in [0.55,0.95].
    center_xy = mix * opposition_xy + (1.0 - mix) * default_xy
    signed_xy = 2.0 * sobol[:, 10:12] - 1.0
    center_xy += signed_xy.sign() * signed_xy.abs().pow(3) * torch.tensor((0.040, 0.030), device=sobol.device)
    # Cubic mapping concentrates proposals near opposition while retaining +/-4/3 cm tails for wide palms/root collisions.
    center_xy = torch.stack(
        (center_xy[:, 0].clamp(-0.060, 0.060), center_xy[:, 1].clamp(0.030, 0.140)), dim=-1
    )
    center_z = 0.059 + (sobol[:, 12:13] - 0.5) * 0.004  # First-step penetration-free drop distance in [0.057,0.061] m.
    position = torch.cat((center_xy, center_z), dim=-1)
    return _select_pair(tip_positions_h, active_tip_mask, position)


def fixed_position_envelope(
    tip_positions_h: torch.Tensor,
    active_tip_mask: torch.Tensor,
    object_position_h_m: torch.Tensor,
) -> EnvelopeResult:
    'Choose the best valid three-finger pair for CEM-proposed object positions.'

    if object_position_h_m.shape != (tip_positions_h.shape[0], 3):
        raise ValueError("CEM object positions must have shape [B,3]")
    return _select_pair(tip_positions_h, active_tip_mask, object_position_h_m)


def geometry_score(
    joint_margin: torch.Tensor,
    distances: torch.Tensor,
    sector_deg: torch.Tensor,
) -> torch.Tensor:
    'Return strict-first continuous score for cheap-geometry Top-32 ranking.'

    strict = (
        (joint_margin >= MVP80_STRICT_GOOD_PREGRASP_GATE.joint_margin_fraction_min)
        & (distances.amax(dim=-1) <= MVP80_STRICT_GOOD_PREGRASP_GATE.tip_center_distance_m_max)
        & (sector_deg >= MVP80_STRICT_GOOD_PREGRASP_GATE.sector_min_deg)
    )
    return (
        100.0 * strict.float()
        + 2.0 * joint_margin
        + sector_deg / 180.0
        - distances.amax(dim=-1) / MVP80_STRICT_GOOD_PREGRASP_GATE.tip_center_distance_m_max
        - 5.0
        * torch.clamp(PROPOSAL_TIP_CLEARANCE_M - distances.amin(dim=-1), min=0.0)
        / PROPOSAL_TIP_CLEARANCE_M
    )


def strict_pass_mask(
    *,
    joint_margin: torch.Tensor,
    distances: torch.Tensor,
    sector_deg: torch.Tensor,
    penetration_m: torch.Tensor,
    displacement_m: torch.Tensor,
    tilt_deg: torch.Tensor,
    peak_linear_m_s: torch.Tensor,
    peak_angular_rad_s: torch.Tensor,
    palm_fraction: torch.Tensor,
    gate: StrictGoodPregraspGate = MVP80_STRICT_GOOD_PREGRASP_GATE,
) -> torch.Tensor:
    'Apply the unique strict predicate to same-shaped physical metric tensors.'

    return (
        (joint_margin >= gate.joint_margin_fraction_min)
        & (distances.amax(dim=-1) <= gate.tip_center_distance_m_max)
        & (sector_deg >= gate.sector_min_deg)
        & (penetration_m <= gate.penetration_depth_m_max)
        & (displacement_m <= gate.object_displacement_m_max)
        & (tilt_deg <= gate.object_tilt_deg_max)
        & (peak_linear_m_s <= gate.peak_linear_velocity_m_s_max)
        & (peak_angular_rad_s <= gate.peak_angular_velocity_rad_s_max)
        & (palm_fraction >= gate.palm_contact_fraction_min)
    )


def normalized_gate_violation(
    *,
    joint_margin: torch.Tensor,
    distances: torch.Tensor,
    sector_deg: torch.Tensor,
    penetration_m: torch.Tensor,
    displacement_m: torch.Tensor,
    tilt_deg: torch.Tensor,
    peak_linear_m_s: torch.Tensor,
    peak_angular_rad_s: torch.Tensor,
    palm_fraction: torch.Tensor,
    gate: StrictGoodPregraspGate = MVP80_STRICT_GOOD_PREGRASP_GATE,
) -> torch.Tensor:
    'Return dimensionless summed gate excess for CEM elite ordering; zero means strict pass.'

    epsilon = torch.finfo(joint_margin.dtype).eps
    violation = torch.clamp(gate.joint_margin_fraction_min - joint_margin, min=0.0) / max(
        gate.joint_margin_fraction_min, epsilon
    )
    violation += torch.clamp(distances.amax(dim=-1) - gate.tip_center_distance_m_max, min=0.0) / max(
        gate.tip_center_distance_m_max, epsilon
    )
    violation += torch.clamp(gate.sector_min_deg - sector_deg, min=0.0) / max(gate.sector_min_deg, epsilon)
    upper_terms = (
        (penetration_m, gate.penetration_depth_m_max),
        (displacement_m, gate.object_displacement_m_max),
        (tilt_deg, gate.object_tilt_deg_max),
        (peak_linear_m_s, gate.peak_linear_velocity_m_s_max),
        (peak_angular_rad_s, gate.peak_angular_velocity_rad_s_max),
    )
    for value, maximum in upper_terms:
        violation += torch.clamp(value - maximum, min=0.0) / max(maximum, epsilon)
    violation += torch.clamp(gate.palm_contact_fraction_min - palm_fraction, min=0.0) / max(
        gate.palm_contact_fraction_min, epsilon
    )
    return violation


def low_rank_cem_candidates(
    elite_q: torch.Tensor,
    elite_position: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    candidate_count: int,
    seed: int,
    round_index: int,
    asset_keys: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        "Fit 4D PCA from each asset's physical elites and sample a 7D mixture CEM. A "
        'mean across contact modes may be unstable, so choose an elite center per '
        'proposal, then perturb along four shared PCA directions and a 3D position '
        'scale. Return q [A,C,16], p_ho [A,C,3], and elite index [A,C].'
    )

    if elite_q.ndim != 3 or elite_q.shape[-1] != 16 or elite_position.shape != (*elite_q.shape[:2], 3):
        raise ValueError("CEM elites must have shapes [A,E,16] and [A,E,3]")
    asset_count, elite_count, _ = elite_q.shape
    if elite_count < 2 or candidate_count < 1 or lower.shape != (asset_count, 16):
        raise ValueError("CEM requires at least two elites and aligned asset limits")
    if asset_keys is not None and len(asset_keys) != asset_count:
        raise ValueError("CEM asset keys must align with the asset axis")
    q_output = torch.zeros(asset_count, candidate_count, 16, device=elite_q.device, dtype=elite_q.dtype)
    position_output = torch.zeros(asset_count, candidate_count, 3, device=elite_q.device, dtype=elite_q.dtype)
    center_output = torch.zeros(asset_count, candidate_count, dtype=torch.long, device=elite_q.device)
    span = (upper - lower).clamp_min(1.0e-6)
    comfortable_lower = lower + 0.11 * span  # Keep 1% numerical/search margin beyond the 10% hard gate.
    comfortable_upper = upper - 0.11 * span
    shrink = 0.70**round_index  # Narrow successive rounds but retain a minimum exploration std.

    for asset_index in range(asset_count):
        active = active_mask[asset_index].to(dtype=elite_q.dtype)
        q_values = elite_q[asset_index] * active  # `[E,16]`
        q_mean = q_values.mean(dim=0)
        centered = (q_values - q_mean) * active
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        rank = min(STRICT_CEM_JOINT_RANK, vh.shape[0])
        basis = vh[:rank] * active.unsqueeze(0)  # `[R,16]` elite covariance principal directions
        coefficients = centered @ basis.T
        coefficient_std = coefficients.std(dim=0, unbiased=False).clamp(0.008, 0.06) * shrink
        position_values = elite_position[asset_index]
        position_std = position_values.std(dim=0, unbiased=False)
        minimum_std = torch.tensor((0.00075, 0.00075, 0.00040), device=elite_q.device)
        maximum_std = torch.tensor((0.006, 0.006, 0.002), device=elite_q.device)
        position_std = torch.maximum(position_std, minimum_std)
        position_std = torch.minimum(position_std, maximum_std) * shrink

        asset_key = asset_index if asset_keys is None else int(asset_keys[asset_index])
        generator = torch.Generator(device=elite_q.device).manual_seed(
            int(seed + round_index * 1_000_003 + asset_key * 104729)
        )  # Caller supplies a manifest row (legacy) or physical identity key (cohort).
        noise = torch.randn(candidate_count, rank + 3, generator=generator, device=elite_q.device)
        if elite_count == len(CEM_PROPOSAL_CENTER_COUNTS) and candidate_count == sum(CEM_PROPOSAL_CENTER_COUNTS):
            center_indices = torch.repeat_interleave(
                torch.arange(elite_count, device=elite_q.device),
                torch.tensor(CEM_PROPOSAL_CENTER_COUNTS, device=elite_q.device),
            )
        else:
            center_indices = torch.arange(candidate_count, device=elite_q.device) % elite_count
        q_samples = q_values[center_indices] + (noise[:, :rank] * coefficient_std) @ basis
        q_samples = torch.maximum(torch.minimum(q_samples, comfortable_upper[asset_index]), comfortable_lower[asset_index])
        q_output[asset_index] = q_samples * active
        positions = position_values[center_indices] + noise[:, rank:] * position_std
        positions[:, 0] = positions[:, 0].clamp(-0.060, 0.060)
        positions[:, 1] = positions[:, 1].clamp(0.030, 0.140)
        positions[:, 2] = positions[:, 2].clamp(0.055, 0.065)
        position_output[asset_index] = positions
        center_output[asset_index] = center_indices
    return q_output, position_output, center_output


__all__ = [
    "EnvelopeResult",
    "CEM_PHYSICS_CENTER_COUNTS",
    "CEM_PROPOSAL_CENTER_COUNTS",
    "N000_CANONICAL_Q",
    "PROPOSAL_TIP_CLEARANCE_M",
    "STRICT_CEM_JOINT_RANK",
    "STRICT_SOBOL_DIMENSION",
    "fixed_position_envelope",
    "deepest_contact_normal_from_buffers",
    "geometry_score",
    "initial_envelope",
    "initial_joint_candidates",
    "low_rank_cem_candidates",
    "normalized_gate_violation",
    "sobol_bank",
    "stable_asset_stream_key",
    "strict_pass_mask",
]
