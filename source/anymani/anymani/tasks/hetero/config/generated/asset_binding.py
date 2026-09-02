r"""Formal PPO dataset到canonical scene/pregrasp identity的唯一有序asset binding。

本模块只调用中立``assets``与``robots``接口，不import旧任务或``distill``。Formal dataset row是provenance；
scene routing使用selection-local prototype index；cache query使用canonical artifact的physical identity。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from anymani.assets.bank.dataset import HandAssetDataset
from anymani.assets.bank.hand_container import HandContainer
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.assets.bank.prepared_train import resolve_prepared_train
from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1, CanonicalHandArtifact
from anymani.pregrasp import (
    AtomicPregraspCache,
    PregraspCoverage,
    PregraspRecord,
    PregraspTier,
    active_mask_digest,
    tier_satisfies,
)
from anymani.robots.hand_spawn import CanonicalRuntimeCfg, HandSpawnAdapter, HandSpawnCfg, HandUrdfSpawnCfg

from ...contact_layout import HeterogeneousContactLayout, build_canonical_contact_layout
from ...mdp.events import PregraspAssetBinding, PregraspResetCfg
from ...mdp.runtime_state import PregraspRuntimeIdentity
from .pregrasp_identity import DEX_CUBE_SHA256, FormalPregraspCatalogIdentity

FORMAL_PPO_ASSET_COUNT = 2048
PPO_DATASET_PATH = (
    resolve_anymani_root() / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo.yaml"
)
DEFAULT_PREGRASP_CACHE_ROOT = resolve_anymani_root() / "outputs/pregrasp/schema_v2/formal-cache-v2"


def selected_formal_dataset_rows() -> tuple[int, ...]:
    r"""解析当前进程的显式canary rows；缺省为formal 2048轴。"""

    raw = os.environ.get("ANYMANI_HETERO_ASSET_ROWS", "").strip()
    rows = tuple(range(FORMAL_PPO_ASSET_COUNT)) if not raw else tuple(
        int(item.strip()) for item in raw.split(",") if item.strip()
    )
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("ANYMANI_HETERO_ASSET_ROWS must contain unique formal rows")
    if any(row < 0 or row >= FORMAL_PPO_ASSET_COUNT for row in rows):
        raise ValueError("heterogeneous formal dataset row lies outside [0,2047]")
    return rows


def _morphology_cell_id(artifact: CanonicalHandArtifact) -> int:
    r"""返回handedness×3/4 tips×thumb 3/4 DoF的固定0..7诊断cell。"""

    routing = artifact.routing
    handedness_offset = 0 if routing.handedness == "left" else 4
    tip_offset = 0 if sum(routing.active_tip_mask) == 3 else 2
    thumb_dof = sum(routing.active_joint_mask[index] for index in (3, 7, 11, 15))
    thumb_offset = 0 if thumb_dof == 3 else 1
    return handedness_offset + tip_offset + thumb_offset


@dataclass(frozen=True)
class GeneratedAssetBinding:
    r"""同序source/canonical/runtime/pregrasp装配结果。"""

    dataset_rows: tuple[int, ...]
    source_assets: tuple[HandContainer, ...]
    hand_spawn_cfg: HandSpawnCfg
    hand_adapter: HandSpawnAdapter
    canonical_artifacts: tuple[CanonicalHandArtifact, ...]
    active_joint_masks: tuple[tuple[bool, ...], ...]
    runtime_identities: tuple[PregraspRuntimeIdentity, ...]
    morphology_cell_ids: tuple[int, ...]
    contact_layout: HeterogeneousContactLayout
    dataset_sha256: str

    @property
    def asset_count(self) -> int:
        r"""返回selection-local prototype数量$A$。"""

        return len(self.dataset_rows)

    def asset_index_by_env(self, num_envs: int) -> tuple[int, ...]:
        r"""复制MultiAssetSpawner round-robin规则$k_e=e\bmod A$。"""

        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        return tuple(env_id % self.asset_count for env_id in range(num_envs))

    def active_joint_mask_by_env(self, num_envs: int) -> tuple[tuple[bool, ...], ...]:
        r"""按scene routing展开full$[N,16]$ static mask。"""

        return tuple(self.active_joint_masks[index] for index in self.asset_index_by_env(num_envs))

    def dataset_row_by_env(self, num_envs: int) -> tuple[int, ...]:
        r"""按scene routing展开formal row，仅供diagnostics标签。"""

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
        r"""为每个prototype选择唯一matching basin record并构造exact reset cfg。"""

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


def build_generated_asset_binding(dataset_rows: tuple[int, ...] | None = None) -> GeneratedAssetBinding:
    r"""解析formal train、选择rows并由单一HandSpawnAdapter lower canonical artifacts。"""

    rows = selected_formal_dataset_rows() if dataset_rows is None else tuple(dataset_rows)
    if not rows or len(set(rows)) != len(rows) or any(row < 0 or row >= FORMAL_PPO_ASSET_COUNT for row in rows):
        raise ValueError("generated asset binding rows must be unique formal indices")
    dataset = HandAssetDataset.from_yaml(PPO_DATASET_PATH)
    partition, _ = resolve_prepared_train(dataset, require_geometry_semantics=True)
    if len(partition.assets) != FORMAL_PPO_ASSET_COUNT:
        raise ValueError(f"formal PPO train must contain 2048 assets, got {len(partition.assets)}")
    source_assets = tuple(partition.assets[row] for row in rows)
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
        restore_visual_materials=False,
        validate_same_schema=True,
    )
    adapter = HandSpawnAdapter(spawn_cfg, resolved_assets=source_assets)
    artifacts = adapter.canonical_artifacts
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
        source_assets=source_assets,
        hand_spawn_cfg=spawn_cfg,
        hand_adapter=adapter,
        canonical_artifacts=artifacts,
        active_joint_masks=active_masks,
        runtime_identities=runtime_identities,
        morphology_cell_ids=tuple(_morphology_cell_id(artifact) for artifact in artifacts),
        contact_layout=build_canonical_contact_layout(),
        dataset_sha256=dataset.source_sha256,
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
    r"""从已提交index中选择唯一runtime-identity/scale/tier basin record。"""

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
    "DEFAULT_PREGRASP_CACHE_ROOT",
    "DEX_CUBE_SHA256",
    "FORMAL_PPO_ASSET_COUNT",
    "GeneratedAssetBinding",
    "build_generated_asset_binding",
    "selected_formal_dataset_rows",
]
