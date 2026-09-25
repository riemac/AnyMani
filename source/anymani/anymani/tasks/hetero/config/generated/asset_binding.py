'Ordered binding from formal PPO data to canonical scene and pregrasp identity. This module uses only neutral assets and robot interfaces; dataset rows are provenance, scene routing uses selection-local prototype indices, and cache queries use canonical physical identity.'

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from anymani.assets.bank.cohort import load_hand_asset_cohort
from anymani.assets.bank.dataset import HandAssetDataset
from anymani.assets.bank.hand_container import HandContainer
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.assets.bank.prepared_train import resolve_prepared_train
from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1, CanonicalHandArtifact
from anymani.pregrasp import (
    AtomicPregraspCache,
    GoodPregraspCatalog,
    GoodPregraspKey,
    PregraspCoverage,
    PregraspRecord,
    PregraspTier,
    active_mask_digest,
    tier_satisfies,
)
from anymani.pregrasp.revalidation import validate_revalidation_generation_identity
from anymani.pregrasp.schema import stable_digest
from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE
from anymani.publication.paper_data import PaperAssetBundle, PaperAssetView
from anymani.robots.hand_spawn import CanonicalRuntimeCfg, HandSpawnAdapter, HandSpawnCfg, HandUrdfSpawnCfg
from anymani.robots.visual_material_policy import generated_hand_visual_materials_enabled

from ...contact_layout import HeterogeneousContactLayout, build_canonical_contact_layout
from ...mdp.events import (
    GoodPregraspAssetBinding,
    GoodPregraspResetCfg,
    PregraspAssetBinding,
    PregraspResetCfg,
)
from ...mdp.runtime_state import PregraspRuntimeIdentity
from .cohort_good_pregrasp_identity import (
    COHORT_GOOD_PREGRASP_CATALOG_ROOT,
    COHORT_GOOD_PREGRASP_GENERATION_DIGEST,
    COHORT_GOOD_PREGRASP_OBJECT_SCALE,
    COHORT_GOOD_PREGRASP_PHYSICS_DIGEST,
    COHORT_GOOD_PREGRASP_REQUIRE_STRICT,
)
from .good_pregrasp_identity import (
    GOOD_PREGRASP_CATALOG_ROOT,
    GOOD_PREGRASP_GENERATION_DIGEST,
    GOOD_PREGRASP_OBJECT_SCALE,
    GOOD_PREGRASP_PHYSICS_DIGEST,
    GOOD_PREGRASP_REQUIRE_STRICT,
)
from .pregrasp_identity import DEX_CUBE_SHA256, FormalPregraspCatalogIdentity
from .pregrasp_routing import (
    PregraspGenerationRoute,
    resolve_pregrasp_generation_routes,
    validate_pregrasp_generation_routes_against_catalog,
)

FORMAL_PPO_ASSET_COUNT = 2048
PPO_DATASET_PATH = (
    resolve_anymani_root() / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo.yaml"
)
DEFAULT_PREGRASP_CACHE_ROOT = resolve_anymani_root() / "outputs/pregrasp/schema_v2/formal-cache-v2"
DEFAULT_GOOD_PREGRASP_CATALOG_ROOT = resolve_anymani_root() / GOOD_PREGRASP_CATALOG_ROOT
DEFAULT_COHORT_GOOD_PREGRASP_CATALOG_ROOT = resolve_anymani_root() / COHORT_GOOD_PREGRASP_CATALOG_ROOT


def selected_formal_dataset_rows() -> tuple[int, ...]:
    'Read the explicit canary row selection for this process; default to the formal 2048-row axis.'

    raw = os.environ.get("ANYMANI_HETERO_ASSET_ROWS", "").strip()
    rows = (
        tuple(range(FORMAL_PPO_ASSET_COUNT))
        if not raw
        else tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    )
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("ANYMANI_HETERO_ASSET_ROWS must contain unique formal rows")
    if any(row < 0 or row >= FORMAL_PPO_ASSET_COUNT for row in rows):
        raise ValueError("heterogeneous formal dataset row lies outside [0,2047]")
    return rows


def _morphology_cell_id(artifact: CanonicalHandArtifact) -> int:
    'Map handedness, three or four tips, and three or four thumb DoF to diagnostic cell IDs 0 through 7.'

    routing = artifact.routing
    handedness_offset = 0 if routing.handedness == "left" else 4
    tip_offset = 0 if sum(routing.active_tip_mask) == 3 else 2
    thumb_dof = sum(routing.active_joint_mask[index] for index in (3, 7, 11, 15))
    thumb_offset = 0 if thumb_dof == 3 else 1
    return handedness_offset + tip_offset + thumb_offset


def _cohort_member_family(member: Any) -> str:
    'Recover family from pure-group provenance and cross-check it against the generation route.'

    group_name = str(member.provenance.group_name)
    if group_name == "single_palm_leap":
        return "leap"
    if group_name == "single_palm_allegro":
        return "allegro"
    raise ValueError(f"per-member pregrasp generation route requires a pure family group, got {group_name!r}")


def _resolve_paper_cohort(
    name: str,
    *,
    view_lock_path: str,
    configured_lock_path: str,
) -> tuple[Any, Mapping[str, Any], Path]:
    """Resolve a verified bundle cohort or its child runtime view without loading parent datasets."""
    data_dir = Path(os.environ.get("ANYMANI_DATA_DIR", resolve_anymani_root() / "data")).expanduser().resolve()
    bundle = PaperAssetBundle(data_dir / "assets")
    if view_lock_path:
        view: PaperAssetView = bundle.resolve_view(view_lock_path)
        if view.cohort_name != name:
            raise ValueError("paper runtime view belongs to a different requested cohort")
        cohort = view.cohort
        metadata = view.to_dict()
    else:
        cohort = bundle.resolve(name, ready_only=True)
        manifest_cohort = bundle.cohort(name)
        metadata = {
            "cohort_name": name,
            "runtime_cohort_id": cohort.cohort_id,
            "runtime_lock_path": str(cohort.lock_path),
            "runtime_lock_sha256": cohort.lock_sha256,
            "parent_cohort_id": cohort.cohort_id,
            "parent_lock_sha256": cohort.lock_sha256,
            "parent_nominal_count": int(manifest_cohort["nominal_count"]),
            "parent_ready_count": int(manifest_cohort["ready_count"]),
            "selected_parent_member_indices": list(range(len(cohort.members))),
            "selected_source_member_keys": list(cohort.source_keys),
            "selected_asset_ids": [member.asset_id for member in cohort.members],
        }
    resolved_lock = cohort.lock_path.expanduser().resolve(strict=True)
    if configured_lock_path and Path(configured_lock_path).expanduser().resolve(strict=True) != resolved_lock:
        raise ValueError("evaluator cohort lock differs from the verified publication runtime axis")
    os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(resolved_lock)

    catalog_root = bundle.catalog_root(name).resolve(strict=True)
    configured_catalog = os.environ.get("ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT", "").strip()
    if configured_catalog and Path(configured_catalog).expanduser().resolve(strict=True) != catalog_root:
        raise ValueError("configured good-pregrasp catalog differs from the verified paper bundle")
    os.environ["ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT"] = str(catalog_root)
    return cohort, metadata, catalog_root


@dataclass(frozen=True)
class GeneratedAssetBinding:
    'Ordered source, canonical, runtime, and pregrasp bindings for one selection.'

    dataset_rows: tuple[int, ...]  # Selection-local runtime integer axis; legacy mode matches formal row order exactly.
    source_rows: tuple[int, ...]  # Original row in the parent train manifest; provenance only.
    source_member_keys: tuple[str, ...]  # Source-qualified alias#row distinguishes colliding row numbers across manifests.
    cohort_id: str  # Resolved cohort ID; legacy row mode uses formal-ppo-row-selection.
    cohort_lock_sha256: str  # Exact member-level lock bytes; empty in legacy row mode.
    source_manifest_sha256s: tuple[tuple[str, str], ...]  # Alias/SHA pairs, sorted by alias.
    source_assets: tuple[HandContainer, ...]
    hand_spawn_cfg: HandSpawnCfg
    hand_adapter: HandSpawnAdapter
    canonical_artifacts: tuple[CanonicalHandArtifact, ...]
    active_joint_masks: tuple[tuple[bool, ...], ...]
    runtime_identities: tuple[PregraspRuntimeIdentity, ...]
    morphology_cell_ids: tuple[int, ...]
    contact_layout: HeterogeneousContactLayout
    dataset_sha256: str
    pregrasp_generation_identity: Mapping[str, Any] | None = None  # Explicit cohort revalidation protocol; None retains the historical Sobol/CEM identity.
    pregrasp_generation_routes: tuple[PregraspGenerationRoute, ...] = ()
    # Union cohorts store routes per generated member; legacy cohorts leave this empty.
    paper_runtime_view: Mapping[str, Any] | None = None
    paper_pregrasp_catalog_root: Path | None = None

    @property
    def asset_count(self) -> int:
        'Number of selection-local prototypes, A.'

        return len(self.dataset_rows)

    def asset_index_by_env(self, num_envs: int) -> tuple[int, ...]:
        'Round-robin prototype index for each environment: k_e = e mod A.'

        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        return tuple(env_id % self.asset_count for env_id in range(num_envs))

    def active_joint_mask_by_env(self, num_envs: int) -> tuple[tuple[bool, ...], ...]:
        'Static active-joint mask expanded through scene routing to shape [N,16].'

        return tuple(self.active_joint_masks[index] for index in self.asset_index_by_env(num_envs))

    def dataset_row_by_env(self, num_envs: int) -> tuple[int, ...]:
        'Formal dataset row expanded through scene routing for diagnostics only.'

        return tuple(self.dataset_rows[index] for index in self.asset_index_by_env(num_envs))

    def build_pregrasp_reset_cfg(
        self,
        *,
        num_envs: int,
        object_scale: float,
        minimum_tier: PregraspTier,
        catalog_identity: FormalPregraspCatalogIdentity,
        exact_tier: PregraspTier | None = None,
        cache_root: Path = DEFAULT_PREGRASP_CACHE_ROOT,
    ) -> PregraspResetCfg:
        'Choose the unique matching basin record for each prototype and build its exact reset config.'

        bindings = tuple(
            _select_pregrasp_binding(
                cache_root=cache_root,
                runtime_identity=runtime_identity,
                object_scale=object_scale,
                minimum_tier=minimum_tier,
                exact_tier=exact_tier,
                catalog_identity=catalog_identity,
            )
            for runtime_identity in self.runtime_identities
        )
        frame = self.hand_spawn_cfg.frame
        return PregraspResetCfg(
            cache_root=str(cache_root.resolve()),
            bindings=bindings,
            asset_index_by_env=self.asset_index_by_env(num_envs),
            semantic_R_ha=tuple(float(value) for value in frame.semantic_R_ha),
            semantic_p_ha=(
                float(frame.semantic_p_ha[0]),
                float(frame.semantic_p_ha[1]),
                float(frame.semantic_p_ha[2]),
            ),
            minimum_tier=minimum_tier,
            require_basin=True,
        )

    def build_good_pregrasp_reset_cfg(
        self,
        *,
        num_envs: int,
        rank: int = 0,
        catalog_root: Path | None = None,
    ) -> GoodPregraspResetCfg:
        'Build the scale-1.1 schema-3 Top-K reset binding for each prototype. Construct keys forward from scene source, physical, canonical, routing, object, physics, and generation identities; never accept a generation merely because a catalog lookup succeeds.'

        if self.pregrasp_generation_routes and not self.cohort_lock_sha256:
            raise ValueError("per-member pregrasp routes require a frozen cohort lock")
        if self.pregrasp_generation_identity is not None and not self.cohort_lock_sha256:
            raise ValueError("explicit pregrasp revalidation requires a frozen cohort lock")
        if self.cohort_lock_sha256:
            if self.pregrasp_generation_routes:
                if self.pregrasp_generation_identity is not None:
                    raise ValueError("per-member pregrasp routes cannot be combined with one revalidation protocol")
                if len(self.pregrasp_generation_routes) != len(self.source_assets):
                    raise ValueError("per-member pregrasp routes must align with source asset axis")
                route_root = Path(self.pregrasp_generation_routes[0].catalog_root).resolve(strict=False)
                resolved_catalog_root = (catalog_root or route_root).resolve(strict=False)
                if resolved_catalog_root != route_root or any(
                    Path(route.catalog_root).resolve(strict=False) != route_root
                    for route in self.pregrasp_generation_routes
                ):
                    raise ValueError("current good-pregrasp reset requires one shared routed catalog root")
            else:
                resolved_catalog_root = (
                    catalog_root
                    or self.paper_pregrasp_catalog_root
                    or DEFAULT_COHORT_GOOD_PREGRASP_CATALOG_ROOT
                )
                if catalog_root is not None and self.paper_pregrasp_catalog_root is not None:
                    if catalog_root.expanduser().resolve() != self.paper_pregrasp_catalog_root.resolve():
                        raise ValueError("requested catalog differs from the verified paper bundle")
            object_scale = COHORT_GOOD_PREGRASP_OBJECT_SCALE
            physics_digest = COHORT_GOOD_PREGRASP_PHYSICS_DIGEST
            generation_digest = COHORT_GOOD_PREGRASP_GENERATION_DIGEST
            require_strict = COHORT_GOOD_PREGRASP_REQUIRE_STRICT
            if self.pregrasp_generation_identity is not None:
                # Accept only explicit revalidation protocols; physics and the strict gate stay fixed. Never infer an accepted identity from a catalog hit.
                protocol = validate_revalidation_generation_identity(dict(self.pregrasp_generation_identity))
                if protocol["physics_identity_digest"] != physics_digest:
                    raise ValueError("cohort revalidation cannot change the fixed pregrasp physics")
                if protocol["strict_gate_digest"] != MVP80_STRICT_GOOD_PREGRASP_GATE.digest:
                    raise ValueError("cohort revalidation cannot relax the strict pregrasp gate")
                generation_digest = stable_digest(protocol)  # Revalidation uses a new key and never renames an old certificate.
        else:
            resolved_catalog_root = catalog_root or DEFAULT_GOOD_PREGRASP_CATALOG_ROOT
            object_scale = GOOD_PREGRASP_OBJECT_SCALE
            physics_digest = GOOD_PREGRASP_PHYSICS_DIGEST
            generation_digest = GOOD_PREGRASP_GENERATION_DIGEST
            require_strict = GOOD_PREGRASP_REQUIRE_STRICT
        if self.pregrasp_generation_routes:
            # Manual binding construction must not defer route-order or physical-identity errors until a catalog miss.
            # The canonical artifact is the current scene identity source of truth.
            for asset_index, (route, source_asset, artifact) in enumerate(
                zip(self.pregrasp_generation_routes, self.source_assets, self.canonical_artifacts, strict=True)
            ):
                if route.cohort_index != asset_index or route.asset_id != source_asset.asset_id:
                    raise ValueError(f"pregrasp route asset/index disagrees at binding index {asset_index}")
                if (
                    route.source_content_hash != artifact.source_content_hash
                    or route.physical_geometry_hash != artifact.physical_geometry_hash
                    or route.canonical_schema_digest != artifact.schema_digest
                ):
                    raise ValueError(f"pregrasp route canonical identity disagrees at binding index {asset_index}")
                if route.physics_identity_digest != physics_digest:
                    raise ValueError(f"pregrasp route physics identity disagrees at binding index {asset_index}")
        bindings = tuple(
            GoodPregraspAssetBinding.from_key(
                GoodPregraspKey(
                    asset_id=source_asset.asset_id,
                    source_content_hash=artifact.source_content_hash,
                    physical_geometry_hash=artifact.physical_geometry_hash,
                    canonical_schema_digest=artifact.schema_digest,
                    routing_digest=active_mask_digest(artifact.routing.active_joint_mask),
                    object_asset_id="DexCube",
                    object_asset_sha256=DEX_CUBE_SHA256,
                    object_scale=object_scale,
                    physics_identity_digest=physics_digest,
                    generation_identity_digest=(
                        self.pregrasp_generation_routes[asset_index].generation_identity_digest
                        if self.pregrasp_generation_routes
                        else generation_digest
                    ),
                ),
                runtime_identity=runtime_identity,
            )
            for asset_index, (source_asset, artifact, runtime_identity) in enumerate(
                zip(
                    self.source_assets,
                    self.canonical_artifacts,
                    self.runtime_identities,
                    strict=True,
                )
            )
        )
        frame = self.hand_spawn_cfg.frame
        return GoodPregraspResetCfg(
            catalog_root=str(resolved_catalog_root.resolve()),
            bindings=bindings,
            asset_index_by_env=self.asset_index_by_env(num_envs),
            semantic_R_ha=tuple(float(value) for value in frame.semantic_R_ha),
            semantic_p_ha=(
                float(frame.semantic_p_ha[0]),
                float(frame.semantic_p_ha[1]),
                float(frame.semantic_p_ha[2]),
            ),
            rank=rank,
            require_strict=require_strict,
        )


def build_generated_asset_binding(dataset_rows: tuple[int, ...] | None = None) -> GeneratedAssetBinding:
    'Resolve a member-level cohort lock or the legacy formal rows, then lower canonical assets. A cohort lock defines runtime order; source-qualified parent rows remain provenance. Without a lock, retain the original 2048-row PPO selection and explicit row-subset behavior.'

    cohort_path = os.environ.get("ANYMANI_HETERO_COHORT_LOCK", "").strip()
    paper_cohort_name = os.environ.get("ANYMANI_HETERO_PAPER_COHORT", "").strip()
    paper_view_lock_path = os.environ.get("ANYMANI_HETERO_PAPER_VIEW_LOCK", "").strip()
    pregrasp_generation_routes: tuple[PregraspGenerationRoute, ...] = ()
    paper_runtime_view: Mapping[str, Any] | None = None
    paper_pregrasp_catalog_root: Path | None = None
    if paper_cohort_name:
        if dataset_rows is not None or os.environ.get("ANYMANI_HETERO_ASSET_ROWS", "").strip():
            raise ValueError("paper cohort is mutually exclusive with explicit formal dataset rows")
        cohort, paper_runtime_view, paper_pregrasp_catalog_root = _resolve_paper_cohort(
            paper_cohort_name,
            view_lock_path=paper_view_lock_path,
            configured_lock_path=cohort_path,
        )
        cohort_path = str(cohort.lock_path)
        rows = tuple(range(len(cohort.members)))
        source_rows = tuple(member.source_row for member in cohort.members)
        source_member_keys = cohort.source_keys
        cohort_id = cohort.cohort_id
        cohort_lock_sha256 = cohort.lock_sha256
        source_manifest_sha256s = tuple(
            (alias, source.manifest_sha256) for alias, source in sorted(cohort.sources.items())
        )
        source_assets = cohort.assets
        effective_dataset_sha256 = cohort.lock_sha256
        raw_generation = cohort.selection.get("pregrasp_generation_identity")
        if raw_generation is not None and not isinstance(raw_generation, Mapping):
            raise ValueError("paper pregrasp_generation_identity must be an explicit protocol mapping")
        pregrasp_generation_identity = None if raw_generation is None else dict(raw_generation)
        member_payloads = tuple(
            {
                "cohort_index": member.cohort_index,
                "asset_id": member.asset_id,
                "family": _cohort_member_family(member),
                "configuration_domain_hash": member.configuration_domain_hash,
                "physical_geometry_hash": member.physical_geometry_hash,
                "canonical_schema_digest": member.canonical_schema_digest,
            }
            for member in cohort.members
        )
        resolved_routes = resolve_pregrasp_generation_routes(
            cohort.selection,
            members=member_payloads,
            expected_physics_identity_digest=COHORT_GOOD_PREGRASP_PHYSICS_DIGEST,
        )
        pregrasp_generation_routes = resolved_routes or ()
        if pregrasp_generation_routes:
            catalog = GoodPregraspCatalog(paper_pregrasp_catalog_root)
            catalog_keys = tuple(entry.key.to_dict() for entry in catalog.read_entries())
            validate_pregrasp_generation_routes_against_catalog(
                pregrasp_generation_routes,
                catalog_keys=catalog_keys,
            )
    elif cohort_path:
        if dataset_rows is not None or os.environ.get("ANYMANI_HETERO_ASSET_ROWS", "").strip():
            raise ValueError("cohort lock is mutually exclusive with explicit formal dataset rows")
        cohort = load_hand_asset_cohort(cohort_path, require_geometry_semantics=True)
        rows = tuple(range(len(cohort.members)))  # Dense local k=0,...,A-1 prototype axis for transport and diagnostics.
        source_rows = tuple(member.source_row for member in cohort.members)
        source_member_keys = cohort.source_keys
        cohort_id = cohort.cohort_id
        cohort_lock_sha256 = cohort.lock_sha256
        source_manifest_sha256s = tuple(
            (alias, source.manifest_sha256) for alias, source in sorted(cohort.sources.items())
        )
        source_assets = cohort.assets
        effective_dataset_sha256 = cohort.lock_sha256  # binding identity follows exact selected membership/order
        raw_generation = cohort.selection.get("pregrasp_generation_identity")
        if raw_generation is not None and not isinstance(raw_generation, Mapping):
            raise ValueError("cohort pregrasp_generation_identity must be an explicit protocol mapping")
        pregrasp_generation_identity = None if raw_generation is None else dict(raw_generation)
        member_payloads = tuple(
            {
                "cohort_index": member.cohort_index,
                "asset_id": member.asset_id,
                "family": _cohort_member_family(member),
                "configuration_domain_hash": member.configuration_domain_hash,
                "physical_geometry_hash": member.physical_geometry_hash,
                "canonical_schema_digest": member.canonical_schema_digest,
            }
            for member in cohort.members
        )
        resolved_routes = resolve_pregrasp_generation_routes(
            cohort.selection,
            members=member_payloads,
            expected_physics_identity_digest=COHORT_GOOD_PREGRASP_PHYSICS_DIGEST,
        )
        pregrasp_generation_routes = resolved_routes or ()
        if pregrasp_generation_routes:
            # Resolve the route first, then verify a catalog record against the same exact key.
            # Never infer a route from a catalog hit; reject a wrong generation before adapter or scene setup.
            catalog = GoodPregraspCatalog(Path(pregrasp_generation_routes[0].catalog_root))
            catalog_keys = tuple(entry.key.to_dict() for entry in catalog.read_entries())
            validate_pregrasp_generation_routes_against_catalog(
                pregrasp_generation_routes,
                catalog_keys=catalog_keys,
            )
    else:
        rows = selected_formal_dataset_rows() if dataset_rows is None else tuple(dataset_rows)
        if not rows or len(set(rows)) != len(rows) or any(row < 0 or row >= FORMAL_PPO_ASSET_COUNT for row in rows):
            raise ValueError("generated asset binding rows must be unique formal indices")
        dataset = HandAssetDataset.from_yaml(PPO_DATASET_PATH)
        partition, _ = resolve_prepared_train(dataset, require_geometry_semantics=True)
        if len(partition.assets) != FORMAL_PPO_ASSET_COUNT:
            raise ValueError(f"formal PPO train must contain 2048 assets, got {len(partition.assets)}")
        source_assets = tuple(partition.assets[row] for row in rows)
        source_rows = rows
        source_member_keys = tuple(f"ppo#{row}" for row in rows)
        cohort_id = "formal-ppo-row-selection"
        cohort_lock_sha256 = ""
        source_manifest_sha256s = (("ppo", dataset.source_sha256),)
        effective_dataset_sha256 = dataset.source_sha256
        pregrasp_generation_identity = None  # Legacy dataset-row selection preserves its original generation protocol.
    spawn_cfg = HandSpawnCfg(
        urdf=HandUrdfSpawnCfg(
            activate_contact_sensors=True,
            use_stable_usd_cache=True,
            force_usd_conversion=False,
        ),
        canonical_runtime=CanonicalRuntimeCfg(
            enabled=True,
            output_root="outputs",
            schema_version=CANONICAL_HAND_SCHEMA_V1.version,
            validate_artifact=True,
        ),
        asset_routing="round_robin",
        restore_visual_materials=generated_hand_visual_materials_enabled(
            override_env="ANYMANI_HETERO_RESTORE_VISUAL_MATERIALS"
        ),  # Restore URDF colors for GUI or video; skip them in headless runs unless explicitly enabled.
        validate_same_schema=True,
    )
    adapter = HandSpawnAdapter(spawn_cfg, resolved_assets=source_assets)
    artifacts = adapter.canonical_artifacts
    if cohort_path:
        for member, artifact in zip(cohort.members, artifacts, strict=True):
            if member.configuration_domain_hash and member.configuration_domain_hash != artifact.source_content_hash:
                raise RuntimeError(
                    f"cohort member {member.cohort_index} configuration-domain hash disagrees with canonical artifact"
                )
            if member.physical_geometry_hash and member.physical_geometry_hash != artifact.physical_geometry_hash:
                raise RuntimeError(
                    f"cohort member {member.cohort_index} physical geometry hash disagrees with canonical lowering"
                )
            if member.canonical_schema_digest and member.canonical_schema_digest != artifact.schema_digest:
                raise RuntimeError(
                    f"cohort member {member.cohort_index} canonical schema digest disagrees with runtime artifact"
                )
    active_masks = tuple(tuple(bool(value) for value in artifact.routing.active_joint_mask) for artifact in artifacts)
    runtime_identities = tuple(
        PregraspRuntimeIdentity(
            source_content_hash=artifact.source_content_hash,
            physical_geometry_hash=artifact.physical_geometry_hash,
            canonical_schema_digest=artifact.schema_digest,
            routing_digest=active_mask_digest(artifact.routing.active_joint_mask),
        )
        for artifact in artifacts
    )
    return GeneratedAssetBinding(
        dataset_rows=rows,
        source_rows=source_rows,
        source_member_keys=source_member_keys,
        cohort_id=cohort_id,
        cohort_lock_sha256=cohort_lock_sha256,
        source_manifest_sha256s=source_manifest_sha256s,
        source_assets=source_assets,
        hand_spawn_cfg=spawn_cfg,
        hand_adapter=adapter,
        canonical_artifacts=artifacts,
        active_joint_masks=active_masks,
        runtime_identities=runtime_identities,
        morphology_cell_ids=tuple(_morphology_cell_id(artifact) for artifact in artifacts),
        contact_layout=build_canonical_contact_layout(),
        dataset_sha256=effective_dataset_sha256,
        pregrasp_generation_identity=pregrasp_generation_identity,
        pregrasp_generation_routes=pregrasp_generation_routes,
        paper_runtime_view=paper_runtime_view,
        paper_pregrasp_catalog_root=paper_pregrasp_catalog_root,
    )


def _select_pregrasp_binding(
    *,
    cache_root: Path,
    runtime_identity: PregraspRuntimeIdentity,
    object_scale: float,
    minimum_tier: PregraspTier,
    exact_tier: PregraspTier | None,
    catalog_identity: FormalPregraspCatalogIdentity,
) -> PregraspAssetBinding:
    'Select the unique record that matches runtime identity, scale, and basin tier.'

    cache = AtomicPregraspCache(cache_root)
    matches: list[PregraspRecord] = []
    for entry in cache.load_index().entries:
        if not entry.scale_min <= object_scale <= entry.scale_max:
            continue
        if entry.coverage != PregraspCoverage.BASIN or not tier_satisfies(entry.tier, minimum_tier):
            continue
        if exact_tier is not None and entry.tier != exact_tier:
            continue
        document = json.loads(cache.payload_path(entry).read_text(encoding="utf-8"))
        record = PregraspRecord.from_dict(cast(dict, document))
        key = record.lookup_key
        try:
            catalog_identity.validate_lookup_key(key)
        except ValueError:
            continue
        try:
            runtime_identity.validate_lookup_key(key)
        except ValueError:
            continue
        matches.append(record)
    if len(matches) != 1:
        raise RuntimeError(
            "pregrasp catalog must resolve exactly one basin record for "
            f"physical={runtime_identity.physical_geometry_hash} scale={object_scale} "
            f"minimum_tier={minimum_tier.value} exact_tier={exact_tier.value if exact_tier else None}, "
            f"got {len(matches)}"
        )
    return PregraspAssetBinding.from_lookup_key(
        matches[0].lookup_key,
        requested_scale=object_scale,
        runtime_identity=runtime_identity,
    )


__all__ = [
    "DEFAULT_COHORT_GOOD_PREGRASP_CATALOG_ROOT",
    "DEFAULT_PREGRASP_CACHE_ROOT",
    "DEFAULT_GOOD_PREGRASP_CATALOG_ROOT",
    "DEX_CUBE_SHA256",
    "FORMAL_PPO_ASSET_COUNT",
    "GeneratedAssetBinding",
    "build_generated_asset_binding",
    "selected_formal_dataset_rows",
]
