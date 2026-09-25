"""Typed geometry field samples, masks, and provenance."""


from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum

import torch


class QueryStratum(IntEnum):


    WORKSPACE = 0
    OWNER_SHELL = 1
    ADJACENT = 2


class SensitivityOwnerCategory(IntEnum):


    SELF = 0
    SAME_FINGER_TIP = 1
    OTHER_DESCENDANT = 2
    PALM = 3
    SAME_FINGER_UPSTREAM = 4
    OTHER_FINGER_JOINT = 5
    OTHER_FINGER_TIP = 6
    FALLBACK = 7


class SensitivitySamplingRole(IntEnum):


    ACTIVE_OWNER_SHELL = 0
    ACTIVE_CONTEXT = 1
    STRUCTURAL_ZERO = 2


@dataclass(frozen=True)
class FieldTargetBatch:


    query_points: torch.Tensor
    query_stratum: torch.Tensor
    distance: torch.Tensor
    density: torch.Tensor
    valid_mask: torch.Tensor
    owner_role: torch.Tensor
    bandwidths: torch.Tensor
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:


        if self.query_points.ndim != 4 or self.query_points.shape[-1] != 3:
            raise ValueError(f"query_points must have shape [B,G,N_Q,3], got {tuple(self.query_points.shape)}")
        base_shape = self.query_points.shape[:-1]
        if self.query_stratum.shape != base_shape:
            raise ValueError("query_stratum must have shape [B,G,N_Q]")
        if self.distance.shape != base_shape or self.valid_mask.shape != base_shape:
            raise ValueError("distance and valid_mask must have shape [B,G,N_Q]")
        if self.bandwidths.ndim not in {1, 2}:
            raise ValueError("bandwidths must have shape [L] or [B,L]")
        if self.bandwidths.ndim == 2 and self.bandwidths.shape[0] != base_shape[0]:
            raise ValueError("sampled bandwidths [B,L] must share B with query targets")
        bandwidth_count = self.bandwidths.shape[-1]
        if self.density.shape != (*base_shape, bandwidth_count):
            raise ValueError("density must have shape [B,G,N_Q,N_sigma]")
        if torch.any(self.bandwidths <= 0.0):
            raise ValueError("bandwidths must be strictly positive")
        if self.owner_role.shape not in {(base_shape[1],), (base_shape[0], base_shape[1])}:
            raise ValueError("owner_role must have shape [G] or [B,G]")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must use torch.bool")
        if torch.any(self.query_stratum < int(QueryStratum.WORKSPACE)) or torch.any(
            self.query_stratum > int(QueryStratum.ADJACENT)
        ):
            raise ValueError("query_stratum contains an unknown physical stratum")
        if self.provenance.get("frame") != "h" or self.provenance.get("length_unit") != "m":
            raise ValueError("FieldTargetBatch provenance must declare frame='h' and length_unit='m'")


@dataclass(frozen=True)
class SensitivityTargetBatch:


    owner_index: torch.Tensor
    query_index: torch.Tensor
    joint_index: torch.Tensor
    ancestor_mask: torch.Tensor
    active_mask: torch.Tensor
    closest_point: torch.Tensor
    closest_source: torch.Tensor
    uniqueness_margin: torch.Tensor
    kappa: torch.Tensor
    field_sensitivity: torch.Tensor
    valid_mask: torch.Tensor  # `[B,E]`
    owner_category: torch.Tensor | None = None
    query_stratum: torch.Tensor | None = None
    fallback_category: torch.Tensor | None = None
    sampling_role: torch.Tensor | None = None  # active-shell / active-context / structural-zero
    central_difference: torch.Tensor | None = None
    central_difference_valid_mask: torch.Tensor | None = None
    central_difference_plus_face: torch.Tensor | None = None
    central_difference_minus_face: torch.Tensor | None = None
    central_difference_elapsed_seconds: float = 0.0
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:


        if self.owner_index.ndim not in {1, 2}:
            raise ValueError("edge selectors must have shape [E] or [B,E]")
        edge_count = self.owner_index.shape[-1]
        selector_shape = self.owner_index.shape
        for name, selector in (
            ("owner_index", self.owner_index),
            ("query_index", self.query_index),
            ("joint_index", self.joint_index),
            ("ancestor_mask", self.ancestor_mask),
            ("active_mask", self.active_mask),
        ):
            if selector.shape != selector_shape:
                raise ValueError(f"{name} must share selector shape {selector_shape}, got {tuple(selector.shape)}")
        for name, label in (
            ("owner_category", self.owner_category),
            ("query_stratum", self.query_stratum),
            ("fallback_category", self.fallback_category),
            ("sampling_role", self.sampling_role),
        ):
            if label is not None and (label.shape != selector_shape or label.dtype != torch.long):
                raise ValueError(f"{name} must have long selector shape {selector_shape}")
        if self.closest_point.ndim != 3 or self.closest_point.shape[1:] != (edge_count, 3):
            raise ValueError("closest_point must have shape [B,E,3]")
        batch_size = self.closest_point.shape[0]
        if self.owner_index.ndim == 2 and self.owner_index.shape[0] != batch_size:
            raise ValueError("batched edge selectors must share B with closest_point")
        edge_shape = (batch_size, edge_count)
        for name, value in (
            ("closest_source", self.closest_source),
            ("uniqueness_margin", self.uniqueness_margin),
            ("kappa", self.kappa),
            ("valid_mask", self.valid_mask),
        ):
            if value.shape != edge_shape:
                raise ValueError(f"{name} must have shape [B,E]={edge_shape}, got {tuple(value.shape)}")
        for name, value in (
            ("central_difference", self.central_difference),
            ("central_difference_valid_mask", self.central_difference_valid_mask),
            ("central_difference_plus_face", self.central_difference_plus_face),
            ("central_difference_minus_face", self.central_difference_minus_face),
        ):
            if value is not None and value.shape != edge_shape:
                raise ValueError(f"{name} must have shape [B,E]={edge_shape}")
        if self.central_difference_valid_mask is not None and self.central_difference_valid_mask.dtype != torch.bool:
            raise TypeError("central_difference_valid_mask must use torch.bool")
        if self.central_difference_elapsed_seconds < 0.0:
            raise ValueError("central_difference_elapsed_seconds must be non-negative")
        if self.field_sensitivity.ndim != 3 or self.field_sensitivity.shape[:2] != edge_shape:
            raise ValueError("field_sensitivity must have shape [B,E,L]")
        if self.ancestor_mask.dtype != torch.bool or self.active_mask.dtype != torch.bool or self.valid_mask.dtype != torch.bool:
            raise TypeError("ancestor_mask, active_mask and valid_mask must use torch.bool")
        if torch.any(self.active_mask != self.ancestor_mask):
            raise ValueError("active_mask must match ancestor_mask: active edges are kinematic descendants")

        nonancestor = ~self.ancestor_mask
        if self.ancestor_mask.ndim == 1:
            invalid_kappa = self.kappa[:, nonancestor]
            invalid_field = self.field_sensitivity[:, nonancestor]
        else:
            invalid_kappa = self.kappa[nonancestor]
            invalid_field = self.field_sensitivity[nonancestor]
        if torch.any(invalid_kappa != 0) or torch.any(invalid_field != 0):
            raise ValueError("non-ancestor sensitivity targets must be exactly zero")
        if self.provenance and (
            self.provenance.get("frame") != "h"
            or self.provenance.get("distance_unit") != "m"
            or self.provenance.get("joint_unit") != "rad"
        ):
            raise ValueError("SensitivityTargetBatch provenance must declare frame='h', distance_unit='m', joint_unit='rad'")


__all__ = [
    "FieldTargetBatch",
    "QueryStratum",
    "SensitivityOwnerCategory",
    "SensitivitySamplingRole",
    "SensitivityTargetBatch",
]
