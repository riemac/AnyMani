from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from anymani.distill.representations.sources.kinematics import (
    EmbodimentGeometrySpec,
    forward_owner_transforms,
    forward_owner_transforms_and_spatial_screws,
    selected_point_jacobian,
    transform_owner_points,
)
from anymani.distill.ssl.config_store import compose_evaluation_cfg
from anymani.distill.ssl.contracts import build_runtime
from anymani.distill.ssl.runtime.sampling import FixedAssetQSchedule
from anymani.distill.ssl.runtime.scheduler import ResidentGeometryAssetWindow

LENGTH_SCALE_M = 0.1
FINITE_DIFFERENCE_RAD = 1.0e-6
DISTANCE_EPSILON_M = 1.0e-9
PLANE_RADIUS_EPSILON_M = 1.0e-9


class ScalarAccumulator:
    r"""流式保存误差矩、极值与有界 quantile 样本。"""

    def __init__(self, *, sample_budget: int = 200_000) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_square = 0.0
        self.maximum = 0.0
        self.sample_budget = int(sample_budget)
        self.samples: list[torch.Tensor] = []
        self.sample_count = 0

    def update(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        x = values.detach()
        if mask is not None:
            try:
                mask = torch.broadcast_to(mask, x.shape)
            except RuntimeError as error:
                raise ValueError(
                    f"statistic mask shape {tuple(mask.shape)} cannot broadcast to values {tuple(x.shape)}"
                ) from error
            x = x[mask]
        x = x.reshape(-1).double().cpu()
        if x.numel() == 0:
            return
        self.count += int(x.numel())
        self.sum += float(x.sum())
        self.sum_square += float(x.square().sum())
        self.maximum = max(self.maximum, float(x.abs().max()))
        if self.sample_count < self.sample_budget:
            remaining = self.sample_budget - self.sample_count
            if x.numel() > remaining:
                stride = max(1, math.ceil(x.numel() / remaining))
                x = x[::stride][:remaining]
            self.samples.append(x)
            self.sample_count += int(x.numel())

    def report(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0}
        sample = torch.cat(self.samples) if self.samples else torch.empty(0, dtype=torch.float64)
        report: dict[str, float | int] = {
            "count": self.count,
            "mean": self.sum / self.count,
            "rms": math.sqrt(self.sum_square / self.count),
            "max_abs": self.maximum,
        }
        if sample.numel():
            absolute = sample.abs()
            report.update(
                {
                    "q50_abs": float(torch.quantile(absolute, 0.50)),
                    "q90_abs": float(torch.quantile(absolute, 0.90)),
                    "q99_abs": float(torch.quantile(absolute, 0.99)),
                }
            )
        return report


class ProbeStatistics:
    r"""AR-MPJ-001 的全部物理统计容器。"""

    def __init__(self) -> None:
        names = (
            "active_distance_target_m_per_rad",
            "zero_distance_target_m_per_rad",
            "active_relation_height_per_rad",
            "active_relation_radius_per_rad",
            "active_relation_dot_per_rad",
            "active_relation_chirality_per_rad",
            "zero_relation_per_rad",
            "point_jacobian_fd_error_m_per_rad",
            "distance_fd_error_m_per_rad",
            "relation_height_fd_error_per_rad",
            "relation_radius_fd_error_per_rad",
            "relation_dot_fd_error_per_rad",
            "relation_chirality_fd_error_per_rad",
            "se3_distance_error_m_per_rad",
            "se3_relation_error_per_rad",
            "reflection_distance_even_error_m_per_rad",
            "reflection_relation_parity_error_per_rad",
            "joint_sign_point_position_error_m",
            "joint_sign_point_jacobian_error_m_per_rad",
            "joint_sign_distance_parity_error_m_per_rad",
            "joint_sign_relation_parity_error_per_rad",
            "radial_gram_min_eigenvalue",
            "radial_gram_condition",
            "relation_gram_min_eigenvalue",
            "relation_gram_condition",
            "relation_direction_gram_min_eigenvalue",
            "relation_direction_gram_condition",
            "material_anchor_distance_m",
            "material_anchor_plane_radius_m",
            "anchor_center_plane_radius_m",
        )
        self.values = {name: ScalarAccumulator() for name in names}
        self.asset_ids: set[str] = set()
        self.row_count = 0
        self.edge_count = 0
        self.active_edge_count = 0
        self.zero_edge_count = 0
        self.finite_difference_edge_count = 0
        self.distance_singular_count = 0
        self.relation_radius_singular_count = 0
        self.radial_rank_deficient_count = 0
        self.relation_rank_deficient_count = 0
        self.relation_direction_rank_deficient_count = 0
        self.target_only_seconds = 0.0
        self.target_scalar_count = 0
        self.existing_pipeline_realize_seconds = 0.0
        self.audit_seconds = 0.0

    def report(self, *, requested_assets: int, q_per_asset: int) -> dict[str, Any]:
        return {
            "case": "AR-MPJ-001",
            "population": {
                "requested_assets": requested_assets,
                "unique_assets": len(self.asset_ids),
                "q_per_asset": q_per_asset,
                "rows": self.row_count,
                "material_joint_edges": self.edge_count,
                "active_edges": self.active_edge_count,
                "structural_zero_edges": self.zero_edge_count,
                "finite_difference_edges": self.finite_difference_edge_count,
            },
            "singularities": {
                "distance_below_epsilon": self.distance_singular_count,
                "plane_radius_below_epsilon": self.relation_radius_singular_count,
            },
            "conditioning": {
                "radial_rank_deficient": self.radial_rank_deficient_count,
                "relation_rank_deficient": self.relation_rank_deficient_count,
                "relation_direction_rank_deficient": self.relation_direction_rank_deficient_count,
            },
            "runtime": {
                "target_only_seconds": self.target_only_seconds,
                "target_scalar_count": self.target_scalar_count,
                "target_scalars_per_second": self.target_scalar_count / max(self.target_only_seconds, 1.0e-12),
                "existing_density_kappa_pipeline_realize_seconds": self.existing_pipeline_realize_seconds,
                "physical_audit_seconds": self.audit_seconds,
            },
            "statistics": {name: value.report() for name, value in self.values.items()},
            "constants": {
                "length_scale_m": LENGTH_SCALE_M,
                "finite_difference_rad": FINITE_DIFFERENCE_RAD,
                "distance_epsilon_m": DISTANCE_EPSILON_M,
                "plane_radius_epsilon_m": PLANE_RADIUS_EPSILON_M,
            },
        }


def _relation_values(
    points_h: torch.Tensor,
    anchors_h: torch.Tensor,
    palm_normal_h: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    r"""返回与 retained point-anchor frontend 同语义的四个 q-dependent 标量。"""

    center = anchors_h.mean(dim=0)
    anchor_centered = anchors_h - center
    relation = points_h[:, None, :] - anchors_h[None, :, :]
    relation_height = torch.sum(relation * palm_normal_h, dim=-1)
    anchor_height = torch.sum(anchor_centered * palm_normal_h, dim=-1)
    relation_plane = relation - relation_height[..., None] * palm_normal_h
    anchor_plane = anchor_centered - anchor_height[..., None] * palm_normal_h
    relation_radius = torch.linalg.vector_norm(relation_plane, dim=-1)
    dot = torch.sum(relation_plane * anchor_plane[None, :, :], dim=-1)
    chirality = torch.sum(
        torch.cross(relation_plane, anchor_plane[None, :, :].expand_as(relation_plane), dim=-1)
        * palm_normal_h,
        dim=-1,
    )
    values = torch.stack(
        (
            relation_height / LENGTH_SCALE_M,
            relation_radius / LENGTH_SCALE_M,
            dot / (LENGTH_SCALE_M**2),
            chirality / (LENGTH_SCALE_M**2),
        ),
        dim=-1,
    )
    geometry = {
        "relation": relation,
        "relation_plane": relation_plane,
        "relation_radius": relation_radius,
        "anchor_plane": anchor_plane,
    }
    return values, geometry


def _relation_jacobian(
    point_jacobian_h: torch.Tensor,
    geometry: dict[str, torch.Tensor],
    palm_normal_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""把 raw material-point Jacobian 投影到四个 invariant relation channels。"""

    relation_plane = geometry["relation_plane"]
    relation_radius = geometry["relation_radius"]
    anchor_plane = geometry["anchor_plane"]
    point_height_velocity = torch.sum(point_jacobian_h * palm_normal_h, dim=-1)
    point_plane_velocity = point_jacobian_h - point_height_velocity[:, None] * palm_normal_h
    valid_radius = relation_radius > PLANE_RADIUS_EPSILON_M
    radial_direction = relation_plane / relation_radius.clamp_min(PLANE_RADIUS_EPSILON_M)[..., None]
    height = point_height_velocity[:, None].expand_as(relation_radius) / LENGTH_SCALE_M
    radius = torch.sum(radial_direction * point_plane_velocity[:, None, :], dim=-1) / LENGTH_SCALE_M
    dot = torch.sum(point_plane_velocity[:, None, :] * anchor_plane[None, :, :], dim=-1) / (LENGTH_SCALE_M**2)
    chirality = torch.sum(
        torch.cross(
            point_plane_velocity[:, None, :].expand(-1, anchor_plane.shape[0], -1),
            anchor_plane[None, :, :].expand(point_jacobian_h.shape[0], -1, -1),
            dim=-1,
        )
        * palm_normal_h,
        dim=-1,
    ) / (LENGTH_SCALE_M**2)
    return torch.stack((height, radius, dot, chirality), dim=-1), valid_radius


def _distance_jacobian(
    point_jacobian_h: torch.Tensor,
    relation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""返回每个 anchor radial projection、距离与非奇异 mask。"""

    distance = torch.linalg.vector_norm(relation, dim=-1)
    valid = distance > DISTANCE_EPSILON_M
    direction = relation / distance.clamp_min(DISTANCE_EPSILON_M)[..., None]
    target = torch.sum(direction * point_jacobian_h[:, None, :], dim=-1)
    return target, distance, valid


def _gram_report(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""返回每个 edge 的 normalized Gram 最小特征值、条件数和数值秩。"""

    gram = torch.matmul(rows.transpose(-1, -2), rows) / max(1, rows.shape[-2])
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    minimum = eigenvalues[..., 0]
    maximum = eigenvalues[..., -1]
    condition = maximum / minimum.clamp_min(1.0e-15)
    tolerance = maximum * 1.0e-10
    rank = torch.sum(eigenvalues > tolerance[..., None], dim=-1)
    return minimum, condition, rank


def _conditioning(
    geometry: dict[str, torch.Tensor],
    palm_normal_h: torch.Tensor,
) -> dict[str, torch.Tensor]:
    r"""审计 radial-only 与四关系测量对三维 J 的可观测条件。"""

    relation = geometry["relation"]
    relation_distance = torch.linalg.vector_norm(relation, dim=-1)
    radial_rows = relation / relation_distance.clamp_min(DISTANCE_EPSILON_M)[..., None]

    relation_plane = geometry["relation_plane"]
    relation_radius = geometry["relation_radius"]
    radial_plane_rows = relation_plane / relation_radius.clamp_min(PLANE_RADIUS_EPSILON_M)[..., None]
    anchor_plane_scaled = geometry["anchor_plane"] / LENGTH_SCALE_M
    tangent_rows = torch.cross(
        anchor_plane_scaled,
        palm_normal_h.expand_as(anchor_plane_scaled),
        dim=-1,
    )
    height_rows = palm_normal_h.view(1, 1, 3).expand(relation.shape[0], relation.shape[1], 3)
    dot_rows = anchor_plane_scaled.view(1, anchor_plane_scaled.shape[0], 3).expand_as(relation)
    tangent_rows = tangent_rows.view(1, tangent_rows.shape[0], 3).expand_as(relation)
    relation_rows = torch.cat((height_rows, radial_plane_rows, dot_rows, tangent_rows), dim=1)

    directional_parts = [height_rows, radial_plane_rows]
    for rows in (dot_rows, tangent_rows):
        norm = torch.linalg.vector_norm(rows, dim=-1, keepdim=True)
        directional_parts.append(rows / norm.clamp_min(1.0e-12))
    directional_rows = torch.cat(directional_parts, dim=1)

    radial_min, radial_condition, radial_rank = _gram_report(radial_rows)
    relation_min, relation_condition, relation_rank = _gram_report(relation_rows)
    direction_min, direction_condition, direction_rank = _gram_report(directional_rows)
    return {
        "radial_min": radial_min,
        "radial_condition": radial_condition,
        "radial_rank": radial_rank,
        "relation_min": relation_min,
        "relation_condition": relation_condition,
        "relation_rank": relation_rank,
        "direction_min": direction_min,
        "direction_condition": direction_condition,
        "direction_rank": direction_rank,
    }


def _proper_rotation(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.tensor((0.31, -0.72, 0.62), device=device, dtype=dtype)
    axis = axis / torch.linalg.vector_norm(axis)
    theta = torch.tensor(0.83, device=device, dtype=dtype)
    skew = torch.tensor(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        device=device,
        dtype=dtype,
    )
    identity = torch.eye(3, device=device, dtype=dtype)
    return identity + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)


def _rewrite_spec(spec: EmbodimentGeometrySpec, sign: torch.Tensor) -> EmbodimentGeometrySpec:
    limits = spec.joint_limits
    if limits is not None:
        rewritten_limits = torch.where(
            sign[:, None] > 0,
            limits,
            torch.stack((-limits[:, 1], -limits[:, 0]), dim=-1),
        )
    else:
        rewritten_limits = None
    return replace(
        spec,
        space_screws=spec.space_screws * sign[:, None],
        q_home=spec.q_home * sign,
        joint_limits=rewritten_limits,
    )


def _expand_material_edges(
    local_home_points: torch.Tensor,
    owner_index: torch.Tensor,
    joint_index: torch.Tensor,
    *,
    q_index: int,
    points_per_edge: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""为每条 sampled owner/joint edge 选择多个 persistent owner-local surface points。"""

    home_count = local_home_points.shape[1]
    offsets = torch.linspace(0, home_count - 1, points_per_edge, device=owner_index.device).round().long()
    edge_id = torch.arange(owner_index.numel(), device=owner_index.device)
    base = (13 * edge_id + 7 * joint_index + 3 * int(q_index)) % home_count
    point_index = (base[:, None] + offsets[None, :]) % home_count
    expanded_owner = owner_index[:, None].expand(-1, points_per_edge).reshape(-1)
    expanded_joint = joint_index[:, None].expand(-1, points_per_edge).reshape(-1)
    expanded_point = point_index.reshape(-1)
    local_points = local_home_points[expanded_owner, expanded_point]
    return expanded_owner, expanded_joint, local_points


def _finite_difference(
    spec: EmbodimentGeometrySpec,
    q: torch.Tensor,
    owner_index: torch.Tensor,
    joint_index: torch.Tensor,
    local_points: torch.Tensor,
    anchors_h: torch.Tensor,
    palm_normal_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""以 edge-batched q± 追踪同一个 owner-local material point。"""

    edge_count = owner_index.numel()
    repeated_q = q.expand(edge_count, -1).clone()
    batch_index = torch.arange(edge_count, device=q.device)
    plus = repeated_q.clone()
    minus = repeated_q.clone()
    plus[batch_index, joint_index] += FINITE_DIFFERENCE_RAD
    minus[batch_index, joint_index] -= FINITE_DIFFERENCE_RAD
    if spec.joint_limits is None:
        valid = torch.ones(edge_count, device=q.device, dtype=torch.bool)
    else:
        lower = spec.joint_limits[joint_index, 0]
        upper = spec.joint_limits[joint_index, 1]
        valid = (minus[batch_index, joint_index] >= lower) & (plus[batch_index, joint_index] <= upper)
    plus_transform = forward_owner_transforms(spec, plus)
    minus_transform = forward_owner_transforms(spec, minus)
    batched_owner = owner_index[:, None]
    batched_points = local_points[:, None, :]
    point_plus = transform_owner_points(plus_transform, batched_owner, batched_points)[:, 0]
    point_minus = transform_owner_points(minus_transform, batched_owner, batched_points)[:, 0]
    point_fd = (point_plus - point_minus) / (2.0 * FINITE_DIFFERENCE_RAD)
    distance_plus, geometry_plus = _relation_values(point_plus, anchors_h, palm_normal_h)
    distance_minus, geometry_minus = _relation_values(point_minus, anchors_h, palm_normal_h)
    relation_fd = (distance_plus - distance_minus) / (2.0 * FINITE_DIFFERENCE_RAD)
    radial_plus = torch.linalg.vector_norm(geometry_plus["relation"], dim=-1)
    radial_minus = torch.linalg.vector_norm(geometry_minus["relation"], dim=-1)
    radial_fd = (radial_plus - radial_minus) / (2.0 * FINITE_DIFFERENCE_RAD)
    return point_fd, radial_fd, relation_fd, valid


def _process_row(
    stats: ProbeStatistics,
    *,
    asset_id: str,
    q: torch.Tensor,
    q_index: int,
    spec: EmbodimentGeometrySpec,
    local_home_points: torch.Tensor,
    anchors_h: torch.Tensor,
    palm_normal_h: torch.Tensor,
    owner_index: torch.Tensor,
    joint_index: torch.Tensor,
    points_per_edge: int,
    gauge_audit: bool,
) -> None:
    audit_started = perf_counter()
    torch.cuda.synchronize(q.device)
    target_started = perf_counter()
    expanded_owner, expanded_joint, local_points = _expand_material_edges(
        local_home_points,
        owner_index,
        joint_index,
        q_index=q_index,
        points_per_edge=points_per_edge,
    )
    transforms, current_screws = forward_owner_transforms_and_spatial_screws(spec, q)
    points_h = transform_owner_points(transforms, expanded_owner, local_points)[0]
    point_jacobian = selected_point_jacobian(
        spec,
        q,
        expanded_owner,
        expanded_joint,
        local_points,
        owner_transforms=transforms,
        current_spatial_screws=current_screws,
    )[0]
    relation_values, geometry = _relation_values(points_h, anchors_h, palm_normal_h)
    del relation_values
    distance_target, distance, distance_valid = _distance_jacobian(point_jacobian, geometry["relation"])
    relation_target, radius_valid = _relation_jacobian(point_jacobian, geometry, palm_normal_h)
    ancestor = spec.owner_ancestor_mask[expanded_owner, expanded_joint]
    torch.cuda.synchronize(q.device)
    stats.target_only_seconds += perf_counter() - target_started
    stats.target_scalar_count += int(expanded_owner.numel() * anchors_h.shape[0] * 5)

    stats.asset_ids.add(asset_id)
    stats.row_count += 1
    stats.edge_count += int(expanded_owner.numel())
    stats.active_edge_count += int(ancestor.sum())
    stats.zero_edge_count += int((~ancestor).sum())
    stats.distance_singular_count += int((~distance_valid).sum())
    stats.relation_radius_singular_count += int((~radius_valid).sum())
    stats.values["material_anchor_distance_m"].update(distance)
    stats.values["material_anchor_plane_radius_m"].update(geometry["relation_radius"])
    stats.values["anchor_center_plane_radius_m"].update(
        torch.linalg.vector_norm(geometry["anchor_plane"], dim=-1)
    )
    stats.values["active_distance_target_m_per_rad"].update(distance_target, ancestor[:, None] & distance_valid)
    stats.values["zero_distance_target_m_per_rad"].update(distance_target, (~ancestor)[:, None] & distance_valid)
    for channel, name in enumerate(("height", "radius", "dot", "chirality")):
        channel_valid = radius_valid if channel == 1 else torch.ones_like(radius_valid)
        stats.values[f"active_relation_{name}_per_rad"].update(
            relation_target[..., channel],
            ancestor[:, None] & channel_valid,
        )
    stats.values["zero_relation_per_rad"].update(relation_target, (~ancestor)[:, None, None])

    conditioning = _conditioning(geometry, palm_normal_h)
    stats.values["radial_gram_min_eigenvalue"].update(conditioning["radial_min"])
    stats.values["radial_gram_condition"].update(conditioning["radial_condition"])
    stats.values["relation_gram_min_eigenvalue"].update(conditioning["relation_min"])
    stats.values["relation_gram_condition"].update(conditioning["relation_condition"])
    stats.values["relation_direction_gram_min_eigenvalue"].update(conditioning["direction_min"])
    stats.values["relation_direction_gram_condition"].update(conditioning["direction_condition"])
    stats.radial_rank_deficient_count += int((conditioning["radial_rank"] < 3).sum())
    stats.relation_rank_deficient_count += int((conditioning["relation_rank"] < 3).sum())
    stats.relation_direction_rank_deficient_count += int((conditioning["direction_rank"] < 3).sum())

    point_fd, distance_fd, relation_fd, fd_valid = _finite_difference(
        spec,
        q,
        expanded_owner,
        expanded_joint,
        local_points,
        anchors_h,
        palm_normal_h,
    )
    stats.finite_difference_edge_count += int(fd_valid.sum())
    stats.values["point_jacobian_fd_error_m_per_rad"].update(point_jacobian - point_fd, fd_valid[:, None])
    stats.values["distance_fd_error_m_per_rad"].update(
        distance_target - distance_fd,
        fd_valid[:, None] & distance_valid,
    )
    for channel, name in enumerate(("height", "radius", "dot", "chirality")):
        channel_valid = fd_valid[:, None] & (radius_valid if channel == 1 else torch.ones_like(radius_valid))
        stats.values[f"relation_{name}_fd_error_per_rad"].update(
            relation_target[..., channel] - relation_fd[..., channel],
            channel_valid,
        )

    rotation = _proper_rotation(q.device, q.dtype)
    translation = torch.tensor((0.021, -0.037, 0.014), device=q.device, dtype=q.dtype)
    rotated_points = points_h @ rotation.T + translation
    rotated_anchors = anchors_h @ rotation.T + translation
    rotated_jacobian = point_jacobian @ rotation.T
    rotated_normal = palm_normal_h @ rotation.T
    _, rotated_geometry = _relation_values(rotated_points, rotated_anchors, rotated_normal)
    rotated_distance, _, _ = _distance_jacobian(rotated_jacobian, rotated_geometry["relation"])
    rotated_relation, _ = _relation_jacobian(rotated_jacobian, rotated_geometry, rotated_normal)
    stats.values["se3_distance_error_m_per_rad"].update(rotated_distance - distance_target)
    stats.values["se3_relation_error_per_rad"].update(rotated_relation - relation_target)

    reflection = torch.diag(torch.tensor((-1.0, 1.0, 1.0), device=q.device, dtype=q.dtype))
    reflected_points = points_h @ reflection.T
    reflected_anchors = anchors_h @ reflection.T
    reflected_jacobian = point_jacobian @ reflection.T
    reflected_normal = palm_normal_h @ reflection.T
    _, reflected_geometry = _relation_values(reflected_points, reflected_anchors, reflected_normal)
    reflected_distance, _, _ = _distance_jacobian(reflected_jacobian, reflected_geometry["relation"])
    reflected_relation, _ = _relation_jacobian(reflected_jacobian, reflected_geometry, reflected_normal)
    relation_parity = torch.tensor((1.0, 1.0, 1.0, -1.0), device=q.device, dtype=q.dtype)
    stats.values["reflection_distance_even_error_m_per_rad"].update(reflected_distance - distance_target)
    stats.values["reflection_relation_parity_error_per_rad"].update(
        reflected_relation - relation_target * relation_parity
    )

    if gauge_audit:
        sign = torch.where(
            torch.arange(spec.space_screws.shape[0], device=q.device) % 2 == 0,
            torch.tensor(-1.0, device=q.device, dtype=q.dtype),
            torch.tensor(1.0, device=q.device, dtype=q.dtype),
        )
        rewritten_spec = _rewrite_spec(spec, sign)
        rewritten_q = q * sign
        rewritten_transforms, rewritten_screws = forward_owner_transforms_and_spatial_screws(
            rewritten_spec,
            rewritten_q,
        )
        rewritten_points = transform_owner_points(rewritten_transforms, expanded_owner, local_points)[0]
        rewritten_jacobian = selected_point_jacobian(
            rewritten_spec,
            rewritten_q,
            expanded_owner,
            expanded_joint,
            local_points,
            owner_transforms=rewritten_transforms,
            current_spatial_screws=rewritten_screws,
        )[0]
        expected_sign = sign[expanded_joint]
        _, rewritten_geometry = _relation_values(rewritten_points, anchors_h, palm_normal_h)
        rewritten_distance, _, _ = _distance_jacobian(rewritten_jacobian, rewritten_geometry["relation"])
        rewritten_relation, _ = _relation_jacobian(rewritten_jacobian, rewritten_geometry, palm_normal_h)
        stats.values["joint_sign_point_position_error_m"].update(rewritten_points - points_h)
        stats.values["joint_sign_point_jacobian_error_m_per_rad"].update(
            rewritten_jacobian - point_jacobian * expected_sign[:, None]
        )
        stats.values["joint_sign_distance_parity_error_m_per_rad"].update(
            rewritten_distance - distance_target * expected_sign[:, None]
        )
        stats.values["joint_sign_relation_parity_error_per_rad"].update(
            rewritten_relation - relation_target * expected_sign[:, None, None]
        )
    torch.cuda.synchronize(q.device)
    stats.audit_seconds += perf_counter() - audit_started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fixed-material anchor Jacobian targets.")
    parser.add_argument("--assets", type=int, default=64)
    parser.add_argument("--q-per-asset", type=int, default=2)
    parser.add_argument("--assets-per-minibatch", type=int, default=8)
    parser.add_argument("--points-per-edge", type=int, default=4)
    parser.add_argument("--gauge-assets", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/autoresearch/material_point_jacobian/AR-MPJ-001/report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.assets, args.q_per_asset, args.assets_per_minibatch, args.points_per_edge) < 1:
        raise ValueError("asset, q, minibatch and material-point counts must be positive")
    torch.manual_seed(20260830)
    cfg = compose_evaluation_cfg(config_ref="geometry_ssl_multitask_representation_v0_7_5")
    device = torch.device("cuda:0")
    catalog = build_runtime(cfg.data).resolve_evaluation()
    method = build_runtime(cfg.method)
    method.configure_source_artifacts(
        root=cfg.evaluation.source_cache_root,
        mode="readonly",
        dataset_manifest_sha256=str(catalog.dataset.source_sha256),
        producer_device=str(device),
        role="evaluation",
    )
    method.prepare(catalog, role="evaluation", device=device, dtype=torch.float32)
    session = method.open_session(
        "evaluation",
        suite="unseen_variant_set",
        seed=cfg.run.seed + cfg.evaluation.evaluation_seed_offset,
        device=device,
        dtype=torch.float32,
        max_resident_assets=args.assets_per_minibatch,
        window_factory=ResidentGeometryAssetWindow,
    )
    schedule = FixedAssetQSchedule(
        args.assets,
        q_per_asset=args.q_per_asset,
        assets_per_minibatch=args.assets_per_minibatch,
        q_per_asset_per_minibatch=args.q_per_asset,
        max_resident_assets=args.assets_per_minibatch,
    )
    stats = ProbeStatistics()
    overall_started = perf_counter()
    step = 0
    try:
        while not schedule.complete:
            torch.cuda.synchronize(device)
            realize_started = perf_counter()
            batch = session.realize(schedule.next(), schedule=schedule, step=step)
            torch.cuda.synchronize(device)
            stats.existing_pipeline_realize_seconds += perf_counter() - realize_started
            evidence_rows = batch.evidence_row_index
            if evidence_rows is None:
                evidence_rows = torch.arange(batch.q.shape[0], device=device)
            sampling_role = batch.sensitivity_targets.sampling_role
            if sampling_role is None:
                raise ValueError("probe requires edge sampling_role to distinguish real edges from padding")
            q_indices = (
                batch.q_index
                if batch.q_index is not None
                else torch.arange(batch.q.shape[0], device=device, dtype=torch.long)
            )
            resident: dict[str, Any] = session.window._resident
            for row, asset_id in enumerate(batch.asset_ids):
                state = resident[asset_id]
                spec = state.spec.to(device=device, dtype=torch.float64)
                joint_count = spec.space_screws.shape[0]
                q = batch.q[row : row + 1, :joint_count].to(dtype=torch.float64)
                evidence_row = int(evidence_rows[row])
                anchor_mask = batch.evidence.anchor_valid_mask[evidence_row]
                anchors_h = batch.evidence.anchors[evidence_row, anchor_mask].to(dtype=torch.float64)
                palm_normal_h = batch.evidence.palm_normal[evidence_row].to(dtype=torch.float64)
                local_home_points = torch.as_tensor(
                    state.source.home_surface.points_owner_local_m,
                    device=device,
                    dtype=torch.float64,
                )
                real_edge = sampling_role[row] >= 0
                owner_index = batch.sensitivity_targets.owner_index[row, real_edge]
                joint_index = batch.sensitivity_targets.joint_index[row, real_edge]
                _process_row(
                    stats,
                    asset_id=asset_id,
                    q=q,
                    q_index=int(q_indices[row]),
                    spec=spec,
                    local_home_points=local_home_points,
                    anchors_h=anchors_h,
                    palm_normal_h=palm_normal_h,
                    owner_index=owner_index,
                    joint_index=joint_index,
                    points_per_edge=args.points_per_edge,
                    gauge_audit=len(stats.asset_ids) <= args.gauge_assets,
                )
            step += 1
    finally:
        session.close()
        method.close()

    report = stats.report(requested_assets=args.assets, q_per_asset=args.q_per_asset)
    torch.cuda.synchronize(device)
    report["runtime"].update(
        {
            "overall_seconds": perf_counter() - overall_started,
            "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
