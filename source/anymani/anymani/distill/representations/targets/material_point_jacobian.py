"""Anchor-relational material Jacobian targets. The four channels are height, radius, dot, and chirality; values are normalized by a shared length scale and expressed per radian."""


from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from anymani.distill.representations.sources.kinematics import (
    EmbodimentGeometrySpec,
    forward_owner_transforms_and_spatial_screws,
    selected_point_jacobian,
    transform_owner_points,
)

RELATION_CHANNELS = ("height", "radius", "dot", "chirality")
"""Fixed physical channel order of the relation target."""


@dataclass(frozen=True)
class MaterialPointRelationJacobianCfg:


    length_scale_m: float = 0.1
    distance_epsilon_m: float = 1.0e-9
    plane_radius_epsilon_m: float = 1.0e-9

    def __post_init__(self) -> None:


        if self.length_scale_m <= 0.0:
            raise ValueError("length_scale_m must be strictly positive")
        if self.distance_epsilon_m <= 0.0 or self.plane_radius_epsilon_m <= 0.0:
            raise ValueError("distance and plane-radius epsilons must be strictly positive")


@dataclass(frozen=True)
class MaterialPointAnchorJacobianMeasurements:


    distance_m: torch.Tensor
    distance_sensitivity_m_per_rad: torch.Tensor
    relation_values: torch.Tensor
    relation_sensitivity_per_rad: torch.Tensor
    distance_valid_mask: torch.Tensor
    radius_valid_mask: torch.Tensor

    def __post_init__(self) -> None:


        if self.distance_m.ndim != 3:
            raise ValueError("distance_m must have shape [B,E,K]")
        measurement_shape = self.distance_m.shape  # `[B,E,K]`
        for name, value in (
            ("distance_sensitivity_m_per_rad", self.distance_sensitivity_m_per_rad),
            ("distance_valid_mask", self.distance_valid_mask),
            ("radius_valid_mask", self.radius_valid_mask),
        ):
            if value.shape != measurement_shape:
                raise ValueError(f"{name} must have shape [B,E,K]={tuple(measurement_shape)}")
        relation_shape = (*measurement_shape, len(RELATION_CHANNELS))  # `[B,E,K,4]`
        if self.relation_values.shape != relation_shape or self.relation_sensitivity_per_rad.shape != relation_shape:
            raise ValueError(f"relation tensors must have shape {relation_shape}")
        if self.distance_valid_mask.dtype != torch.bool or self.radius_valid_mask.dtype != torch.bool:
            raise TypeError("distance/radius validity masks must use torch.bool")


@dataclass(frozen=True)
class MaterialPointRelationJacobianTarget(MaterialPointAnchorJacobianMeasurements):


    material_points_h_m: torch.Tensor
    point_jacobian_h_m_per_rad: torch.Tensor
    owner_index: torch.Tensor
    joint_index: torch.Tensor
    ancestor_mask: torch.Tensor
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:


        super().__post_init__()
        batch_size, edge_count = self.distance_m.shape[:2]
        edge_shape = (batch_size, edge_count)
        if self.material_points_h_m.shape != (*edge_shape, 3):
            raise ValueError("material_points_h_m must have shape [B,E,3]")
        if self.point_jacobian_h_m_per_rad.shape != (*edge_shape, 3):
            raise ValueError("point_jacobian_h_m_per_rad must have shape [B,E,3]")
        for name, selector in (
            ("owner_index", self.owner_index),
            ("joint_index", self.joint_index),
            ("ancestor_mask", self.ancestor_mask),
        ):
            if selector.shape != edge_shape:
                raise ValueError(f"{name} must have shape [B,E]={edge_shape}")
        if self.owner_index.dtype != torch.long or self.joint_index.dtype != torch.long:
            raise TypeError("owner_index and joint_index must use torch.long")
        if self.ancestor_mask.dtype != torch.bool:
            raise TypeError("ancestor_mask must use torch.bool")


        structural_zero = ~self.ancestor_mask
        if (
            torch.any(self.point_jacobian_h_m_per_rad[structural_zero] != 0)
            or torch.any(self.distance_sensitivity_m_per_rad[structural_zero] != 0)
            or torch.any(self.relation_sensitivity_per_rad[structural_zero] != 0)
        ):
            raise ValueError("non-ancestor material-point Jacobian targets must be exactly zero")
        if self.provenance and (
            self.provenance.get("frame") != "h"
            or self.provenance.get("distance_unit") != "m"
            or self.provenance.get("joint_unit") != "rad"
            or self.provenance.get("material_identity") != "fixed_owner_local_home_surface_point"
        ):
            raise ValueError("material-point target provenance does not match the fixed owner-local contract")


@dataclass(frozen=True)
class _RelationGeometry:


    relation: torch.Tensor
    relation_height: torch.Tensor
    relation_plane: torch.Tensor
    relation_radius: torch.Tensor
    anchor_plane: torch.Tensor


def _validate_measurement_inputs(
    material_points_h_m: torch.Tensor,
    point_jacobian_h_m_per_rad: torch.Tensor,
    anchors_h_m: torch.Tensor,
    palm_normal_h: torch.Tensor,
) -> None:


    if material_points_h_m.ndim != 3 or material_points_h_m.shape[-1] != 3:
        raise ValueError("material_points_h_m must have shape [B,E,3]")
    if point_jacobian_h_m_per_rad.shape != material_points_h_m.shape:
        raise ValueError("point_jacobian_h_m_per_rad must share [B,E,3] with material points")
    if anchors_h_m.ndim != 2 or anchors_h_m.shape[-1] != 3 or anchors_h_m.shape[0] < 1:
        raise ValueError("anchors_h_m must have non-empty shape [K,3]")
    if palm_normal_h.shape != (3,):
        raise ValueError("palm_normal_h must have shape [3]")
    tensors = (material_points_h_m, point_jacobian_h_m_per_rad, anchors_h_m, palm_normal_h)
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("material points, Jacobians, anchors and palm normal must be floating-point")
    if any(tensor.device != material_points_h_m.device for tensor in tensors[1:]):
        raise ValueError("material points, Jacobians, anchors and palm normal must share device")
    if any(tensor.dtype != material_points_h_m.dtype for tensor in tensors[1:]):
        raise ValueError("material points, Jacobians, anchors and palm normal must share dtype")
    normal_norm = torch.linalg.vector_norm(palm_normal_h)
    if not torch.allclose(normal_norm, torch.ones_like(normal_norm), atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("palm_normal_h must be a unit vector")


def _relation_geometry(
    material_points_h_m: torch.Tensor,
    anchors_h_m: torch.Tensor,
    palm_normal_h: torch.Tensor,
) -> _RelationGeometry:


    anchor_center = anchors_h_m.mean(dim=0)
    anchor_centered = anchors_h_m - anchor_center
    relation = material_points_h_m.unsqueeze(-2) - anchors_h_m
    relation_height = torch.sum(relation * palm_normal_h, dim=-1)
    anchor_height = torch.sum(anchor_centered * palm_normal_h, dim=-1)
    relation_plane = relation - relation_height.unsqueeze(-1) * palm_normal_h
    anchor_plane = anchor_centered - anchor_height.unsqueeze(-1) * palm_normal_h
    relation_radius = torch.linalg.vector_norm(relation_plane, dim=-1)
    return _RelationGeometry(relation, relation_height, relation_plane, relation_radius, anchor_plane)


def measure_material_point_anchor_jacobian(
    material_points_h_m: torch.Tensor,
    point_jacobian_h_m_per_rad: torch.Tensor,
    anchors_h_m: torch.Tensor,
    palm_normal_h: torch.Tensor,
    config: MaterialPointRelationJacobianCfg = MaterialPointRelationJacobianCfg(),
) -> MaterialPointAnchorJacobianMeasurements:


    _validate_measurement_inputs(material_points_h_m, point_jacobian_h_m_per_rad, anchors_h_m, palm_normal_h)
    geometry = _relation_geometry(material_points_h_m, anchors_h_m, palm_normal_h)
    length_scale = float(config.length_scale_m)
    quadratic_scale = length_scale * length_scale


    distance_m = torch.linalg.vector_norm(geometry.relation, dim=-1)
    distance_valid = distance_m > config.distance_epsilon_m
    distance_direction = geometry.relation / distance_m.clamp_min(config.distance_epsilon_m).unsqueeze(-1)
    distance_sensitivity = torch.sum(
        distance_direction * point_jacobian_h_m_per_rad.unsqueeze(-2),
        dim=-1,
    )


    anchor_plane = geometry.anchor_plane
    anchor_plane_batched = anchor_plane.view(1, 1, anchor_plane.shape[0], 3)  # `[1,1,K,3]`
    dot = torch.sum(geometry.relation_plane * anchor_plane_batched, dim=-1)
    chirality = torch.sum(
        torch.cross(geometry.relation_plane, anchor_plane_batched.expand_as(geometry.relation_plane), dim=-1)
        * palm_normal_h,
        dim=-1,
    )
    relation_values = torch.stack(
        (
            geometry.relation_height / length_scale,
            geometry.relation_radius / length_scale,
            dot / quadratic_scale,
            chirality / quadratic_scale,
        ),
        dim=-1,
    )


    point_height_velocity = torch.sum(
        point_jacobian_h_m_per_rad * palm_normal_h,
        dim=-1,
    )
    point_plane_velocity = (
        point_jacobian_h_m_per_rad - point_height_velocity.unsqueeze(-1) * palm_normal_h
    )
    radius_valid = geometry.relation_radius > config.plane_radius_epsilon_m
    radial_direction = geometry.relation_plane / geometry.relation_radius.clamp_min(
        config.plane_radius_epsilon_m
    ).unsqueeze(-1)
    height_sensitivity = point_height_velocity.unsqueeze(-1).expand_as(distance_m) / length_scale
    radius_sensitivity = torch.sum(
        radial_direction * point_plane_velocity.unsqueeze(-2),
        dim=-1,
    ) / length_scale
    dot_sensitivity = torch.sum(
        point_plane_velocity.unsqueeze(-2) * anchor_plane_batched,
        dim=-1,
    ) / quadratic_scale
    chirality_sensitivity = torch.sum(
        torch.cross(
            point_plane_velocity.unsqueeze(-2).expand_as(geometry.relation_plane),
            anchor_plane_batched.expand_as(geometry.relation_plane),
            dim=-1,
        )
        * palm_normal_h,
        dim=-1,
    ) / quadratic_scale
    relation_sensitivity = torch.stack(
        (height_sensitivity, radius_sensitivity, dot_sensitivity, chirality_sensitivity),
        dim=-1,
    )

    return MaterialPointAnchorJacobianMeasurements(
        distance_m=distance_m,
        distance_sensitivity_m_per_rad=distance_sensitivity,
        relation_values=relation_values,
        relation_sensitivity_per_rad=relation_sensitivity,
        distance_valid_mask=distance_valid,
        radius_valid_mask=radius_valid,
    )


def _expanded_selector(selector: torch.Tensor, *, batch_size: int, name: str) -> torch.Tensor:


    if selector.ndim == 1:
        return selector.unsqueeze(0).expand(batch_size, -1)
    if selector.ndim == 2 and selector.shape[0] == batch_size:
        return selector
    raise ValueError(f"{name} must have shape [E] or [B,E] sharing q batch size")


def generate_material_point_relation_jacobian_targets(
    spec: EmbodimentGeometrySpec,
    q: torch.Tensor,
    owner_index: torch.Tensor,
    joint_index: torch.Tensor,
    local_material_points_m: torch.Tensor,
    anchors_h_m: torch.Tensor,
    palm_normal_h: torch.Tensor,
    config: MaterialPointRelationJacobianCfg = MaterialPointRelationJacobianCfg(),
    *,
    owner_transforms: torch.Tensor | None = None,
    current_spatial_screws: torch.Tensor | None = None,
) -> MaterialPointRelationJacobianTarget:


    if owner_index.shape != joint_index.shape or owner_index.ndim not in {1, 2}:
        raise ValueError("owner_index and joint_index must have identical [E] or [B,E] shape")
    batch_size = q.shape[0]
    owner_selector = _expanded_selector(owner_index, batch_size=batch_size, name="owner_index")  # `[B,E]`
    joint_selector = _expanded_selector(joint_index, batch_size=batch_size, name="joint_index")  # `[B,E]`


    if owner_transforms is None or current_spatial_screws is None:
        computed_transforms, computed_screws = forward_owner_transforms_and_spatial_screws(spec, q)
        owner_transforms = computed_transforms if owner_transforms is None else owner_transforms
        current_spatial_screws = computed_screws if current_spatial_screws is None else current_spatial_screws


    material_points_h_m = transform_owner_points(
        owner_transforms,
        owner_selector,
        local_material_points_m,
    )
    point_jacobian_h_m_per_rad = selected_point_jacobian(
        spec,
        q,
        owner_selector,
        joint_selector,
        local_material_points_m,
        owner_transforms=owner_transforms,
        current_spatial_screws=current_spatial_screws,
    )
    measurements = measure_material_point_anchor_jacobian(
        material_points_h_m,
        point_jacobian_h_m_per_rad,
        anchors_h_m,
        palm_normal_h,
        config,
    )
    ancestor_mask = spec.owner_ancestor_mask[owner_selector, joint_selector]

    return MaterialPointRelationJacobianTarget(
        distance_m=measurements.distance_m,
        distance_sensitivity_m_per_rad=measurements.distance_sensitivity_m_per_rad,
        relation_values=measurements.relation_values,
        relation_sensitivity_per_rad=measurements.relation_sensitivity_per_rad,
        distance_valid_mask=measurements.distance_valid_mask,
        radius_valid_mask=measurements.radius_valid_mask,
        material_points_h_m=material_points_h_m,
        point_jacobian_h_m_per_rad=point_jacobian_h_m_per_rad,
        owner_index=owner_selector,
        joint_index=joint_selector,
        ancestor_mask=ancestor_mask,
        provenance={
            "frame": "h",
            "distance_unit": "m",
            "joint_unit": "rad",
            "relation_unit": "dimensionless",
            "relation_sensitivity_unit": "rad^-1",
            "relation_channels": ",".join(RELATION_CHANNELS),
            "material_identity": "fixed_owner_local_home_surface_point",
            "anchor_motion": "fixed_palm_support",
        },
    )


__all__ = [
    "RELATION_CHANNELS",
    "MaterialPointAnchorJacobianMeasurements",
    "MaterialPointRelationJacobianCfg",
    "MaterialPointRelationJacobianTarget",
    "generate_material_point_relation_jacobian_targets",
    "measure_material_point_anchor_jacobian",
]
