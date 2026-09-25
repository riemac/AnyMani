'Pure-Python routing contract for schema-3 pregrasp catalogs. The reset config supports one catalog root, so a generated multi-family union is assembled into one index first; each member route then selects its frozen generation identity. Validate routes from cohort selection and never infer or rewrite keys from catalog hits. Legacy cohorts without routes retain the historical single-generation path.'

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREGRASP_ROUTING_SCHEMA_VERSION = "1.0.0"
'Schema version for per-member generation and catalog routes.'

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PregraspGenerationRoute:
    'Forward generation and catalog route for one generated cohort member.'

    asset_id: str  # Generated asset identity for this cohort member.
    cohort_index: int  # Dense position on the generated 256-member axis.
    source_content_hash: str  # Must match member.configuration_domain_hash.
    physical_geometry_hash: str  # Must match the canonical parent physical identity.
    canonical_schema_digest: str  # Must match the canonical parent schema digest.
    generation_identity_digest: str  # Exact generation identity used by the schema-3 key.
    physics_identity_digest: str  # Exact physics identity used by the schema-3 key.
    catalog_root: str  # Combined schema-3 catalog root; one reset config can use only one root.
    catalog_source_family: str  # Route provenance: LEAP or revised Allegro.


def resolve_pregrasp_generation_routes(
    selection: Mapping[str, Any],
    *,
    members: Sequence[Mapping[str, Any]],
    expected_physics_identity_digest: str,
) -> tuple[PregraspGenerationRoute, ...] | None:
    'Resolve and validate the ordered member routes from a frozen cohort selection. Inputs are the selection and aligned member identities plus the expected strict physics digest. Return routes in member order, or None for a legacy selection. Reject missing/mismatched identities, invalid hashes, multiple catalog roots, or a missing catalog index.'

    raw_routes = selection.get("pregrasp_generation_by_asset_id")
    if raw_routes is None:
        return None  # Legacy cohorts without per-member routes retain the single-generation behavior.
    if not isinstance(raw_routes, Mapping) or not raw_routes:
        raise ValueError("pregrasp_generation_by_asset_id must be a non-empty mapping")
    if len(raw_routes) != len(members):
        raise ValueError(f"pregrasp generation route count {len(raw_routes)} does not cover member axis {len(members)}")
    expected_ids = {str(member.get("asset_id", "")) for member in members}
    if len(expected_ids) != len(members):
        raise ValueError("cohort member axis contains duplicate or empty asset_id values")
    if any(not asset_id for asset_id in expected_ids) or set(str(key) for key in raw_routes) != expected_ids:
        raise ValueError("pregrasp generation routes must cover exactly every member asset_id")
    if not _SHA256.fullmatch(str(expected_physics_identity_digest)):
        raise ValueError("expected pregrasp physics identity must be a lowercase SHA-256")

    routes: list[PregraspGenerationRoute] = []
    catalog_roots: set[str] = set()
    for expected_index, member in enumerate(members):
        asset_id = str(member.get("asset_id", ""))
        raw_route = raw_routes.get(asset_id)
        if not isinstance(raw_route, Mapping):
            raise TypeError(f"pregrasp generation route for {asset_id!r} must be a mapping")
        member_index = int(member.get("cohort_index", -1))
        route_asset_id = str(raw_route.get("asset_id", ""))
        route_index = int(raw_route.get("cohort_index", -1))
        if member_index != expected_index or route_asset_id != asset_id or route_index != expected_index:
            raise ValueError(f"pregrasp route identity/index disagrees for member {expected_index}")
        for field in (
            "source_content_hash",
            "physical_geometry_hash",
            "canonical_schema_digest",
            "generation_identity_digest",
            "physics_identity_digest",
        ):
            if _SHA256.fullmatch(str(raw_route.get(field, ""))) is None:
                raise ValueError(f"pregrasp route {asset_id!r} has invalid {field}")
        source_content_hash = str(raw_route["source_content_hash"])
        physical_geometry_hash = str(raw_route["physical_geometry_hash"])
        canonical_schema_digest = str(raw_route["canonical_schema_digest"])
        if source_content_hash != str(member.get("configuration_domain_hash", "")):
            raise ValueError(f"pregrasp route source identity disagrees for member {expected_index}")
        if physical_geometry_hash != str(member.get("physical_geometry_hash", "")):
            raise ValueError(f"pregrasp route physical identity disagrees for member {expected_index}")
        if canonical_schema_digest != str(member.get("canonical_schema_digest", "")):
            raise ValueError(f"pregrasp route canonical schema disagrees for member {expected_index}")
        physics_digest = str(raw_route["physics_identity_digest"])
        if physics_digest != str(expected_physics_identity_digest):
            raise ValueError(f"pregrasp route physics identity disagrees for member {expected_index}")
        catalog_root = str(raw_route.get("catalog_root", "")).strip()
        if not catalog_root:
            raise ValueError(f"pregrasp route catalog_root is empty for member {expected_index}")
        resolved_catalog = str(Path(catalog_root).expanduser().resolve(strict=False))
        if not (Path(resolved_catalog) / "index.json").is_file():
            raise FileNotFoundError(f"pregrasp route catalog index does not exist: {resolved_catalog}")
        catalog_roots.add(resolved_catalog)
        source_family = str(raw_route.get("catalog_source_family", "")).strip()
        if source_family not in {"leap", "allegro"}:
            raise ValueError(f"pregrasp route source family is invalid for member {expected_index}")
        member_family = str(member.get("family", "")).strip()
        if member_family and member_family != source_family:
            raise ValueError(f"pregrasp route source family disagrees for member {expected_index}")
        routes.append(
            PregraspGenerationRoute(
                asset_id=asset_id,
                cohort_index=expected_index,
                source_content_hash=source_content_hash,
                physical_geometry_hash=physical_geometry_hash,
                canonical_schema_digest=canonical_schema_digest,
                generation_identity_digest=str(raw_route["generation_identity_digest"]),
                physics_identity_digest=physics_digest,
                catalog_root=resolved_catalog,
                catalog_source_family=source_family,
            )
        )
    if len(catalog_roots) != 1:
        raise ValueError(
            f"current GoodPregraspResetCfg supports one catalog root; per-source route roots={sorted(catalog_roots)!r}"
        )
    return tuple(routes)


def validate_pregrasp_generation_routes_against_catalog(
    routes: Sequence[PregraspGenerationRoute],
    *,
    catalog_keys: Sequence[Mapping[str, Any]],
) -> None:
    'Compare catalog keys against route-first asset identities to reject generation changes. Catalog keys provide consistency evidence only; they never define or replace a route. Compare source, physical, canonical, physics, and generation identity before building the reset config.'

    key_by_asset: dict[str, Mapping[str, Any]] = {}
    for key in catalog_keys:
        asset_id = str(key.get("asset_id", ""))
        if not asset_id or asset_id in key_by_asset:
            raise ValueError(f"catalog keys contain duplicate/empty asset_id {asset_id!r}")
        key_by_asset[asset_id] = key
    route_ids = {route.asset_id for route in routes}
    if len(route_ids) != len(routes):
        raise ValueError("generation routes contain duplicate asset_id values")
    if route_ids != set(key_by_asset):
        raise ValueError("catalog key asset coverage disagrees with generation route coverage")
    fields = (
        "source_content_hash",
        "physical_geometry_hash",
        "canonical_schema_digest",
        "physics_identity_digest",
        "generation_identity_digest",
    )
    for route in routes:
        key = key_by_asset[route.asset_id]
        route_values = {
            "source_content_hash": route.source_content_hash,
            "physical_geometry_hash": route.physical_geometry_hash,
            "canonical_schema_digest": route.canonical_schema_digest,
            "physics_identity_digest": route.physics_identity_digest,
            "generation_identity_digest": route.generation_identity_digest,
        }
        for field in fields:
            if str(key.get(field, "")) != route_values[field]:
                raise ValueError(f"catalog {field} disagrees with route for asset {route.asset_id!r}")


__all__ = [
    "PREGRASP_ROUTING_SCHEMA_VERSION",
    "PregraspGenerationRoute",
    "resolve_pregrasp_generation_routes",
    "validate_pregrasp_generation_routes_against_catalog",
]
