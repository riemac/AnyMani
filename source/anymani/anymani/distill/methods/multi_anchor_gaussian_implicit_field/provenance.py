"""Physical source and split provenance for geometry pretraining."""


from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from anymani.distill.representations.queries.spatial_sampling import SURFACE_QUERY_SAMPLING_VERSION
from anymani.distill.representations.sources.anchor_sampling import AnchorSamples
from anymani.distill.representations.sources.collision_geometry import HomeSurfaceSamples, OwnerGeometryCache


def validate_asset_manifest_isolation(manifest: Mapping[str, Any]) -> None:


    raw_evaluation = manifest.get("evaluation", {})
    if not isinstance(raw_evaluation, Mapping):
        raise ValueError("asset manifest evaluation field must be a mapping")

    roles: dict[str, list[Mapping[str, Any]]] = {}
    raw_train = manifest.get("train", ())
    if not isinstance(raw_train, (tuple, list)) or not all(isinstance(record, Mapping) for record in raw_train):
        raise ValueError("asset manifest train field must be a sequence of mappings")
    roles["train"] = list(raw_train)


    for suite_name, raw_records in raw_evaluation.items():
        if not isinstance(raw_records, (tuple, list)) or not all(isinstance(record, Mapping) for record in raw_records):
            raise ValueError(f"asset manifest evaluation suite {suite_name!r} must be a sequence of mappings")
        roles[f"evaluation.{suite_name}"] = list(raw_records)
    for identity_name in ("content_hash", "physical_geometry_hash"):
        owners: dict[str, str] = {}
        conflicts: set[str] = set()
        for role, records in roles.items():
            for record in records:
                identity = str(record.get(identity_name, ""))
                if not identity:
                    raise ValueError(f"asset manifest record lacks {identity_name}")
                previous = owners.setdefault(identity, role)
                if previous != role:
                    conflicts.add(identity)
        if conflicts:
            label = "content hashes" if identity_name == "content_hash" else "physical geometry hashes"
            raise ValueError(f"{label} leak across train/held-out roles: {sorted(conflicts)}")


def anchor_realization_record(anchors: AnchorSamples | None) -> dict[str, str]:


    if anchors is None:
        return {
            "anchor_realization_hash": "",
            "anchor_sampling_version": "",
            "anchor_sampling_seed": "",
            "anchor_count": "",
            "anchor_support_radius_m": "",
            "anchor_radial_decay_scale_m": "",
            "anchor_surface_fraction": "",
        }
    digest = hashlib.sha256()
    digest.update(b"anymani-anchor-realization-v1\0")
    for array in (anchors.anchors_hand_m, anchors.surface_mask):
        contiguous = array.copy(order="C")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    for values in (anchors.finger_names, anchors.seed_ids):
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    scalar_provenance = (
        anchors.algorithm_version,
        str(anchors.sampling_seed),
        repr(anchors.radial_support_radius_m),
        repr(anchors.radial_decay_scale_m),
        repr(anchors.surface_fraction),
    )
    for value in scalar_provenance:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return {
        "anchor_realization_hash": digest.hexdigest(),
        "anchor_sampling_version": anchors.algorithm_version,
        "anchor_sampling_seed": str(anchors.sampling_seed),
        "anchor_count": str(len(anchors.anchors_hand_m)),
        "anchor_support_radius_m": repr(anchors.radial_support_radius_m),
        "anchor_radial_decay_scale_m": repr(anchors.radial_decay_scale_m),
        "anchor_surface_fraction": repr(anchors.surface_fraction),
    }


def home_surface_realization_record(
    home_surface: HomeSurfaceSamples | None,
    geometry_cache: OwnerGeometryCache | None,
) -> dict[str, str]:


    if home_surface is None or geometry_cache is None:
        return {
            "home_surface_realization_hash": "",
            "home_surface_sampling_seed": "",
            "home_surface_oversample_factor": "",
            "boolean_backend": "",
            "surface_geometry_hash": "",
            "surface_processing_version": "",
            "surface_query_sampling_version": "",
        }
    digest = hashlib.sha256()
    digest.update(b"anymani-home-surface-realization-v1\0")
    for array in (home_surface.points_owner_local_m, home_surface.face_indices, home_surface.barycentric):
        contiguous = array.copy(order="C")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    for owner_id in home_surface.owner_ids:
        encoded = owner_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    for value in (str(home_surface.sampling_seed), str(home_surface.oversample_factor)):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return {
        "home_surface_realization_hash": digest.hexdigest(),
        "home_surface_sampling_seed": str(home_surface.sampling_seed),
        "home_surface_oversample_factor": str(home_surface.oversample_factor),
        "boolean_backend": geometry_cache.boolean_backend,
        "surface_geometry_hash": geometry_cache.surface_geometry_hash,
        "surface_processing_version": geometry_cache.surface_processing_version,
        "surface_query_sampling_version": SURFACE_QUERY_SAMPLING_VERSION,
    }


__all__ = [
    "anchor_realization_record",
    "home_surface_realization_record",
    "validate_asset_manifest_isolation",
]
