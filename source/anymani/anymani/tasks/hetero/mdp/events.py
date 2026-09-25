(
    'Fail-closed identity-keyed pregrasp reset. Resolve provider data, '
    'tier/coverage/scale, and tensor preflight before PhysX writes. Write actual '
    'q_s, PD target q_t, and world object pose; publish a full-size sidecar for '
    'ActionManager.reset. Static env-to-asset routing selects exact lookup keys; '
    'dataset row is not part of the key.'
)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from isaaclab.assets import Articulation, RigidObject

from anymani.pregrasp import (
    MVP80_STRICT_GOOD_PREGRASP_GATE,
    FilePregraspProvider,
    GoodPregraspCatalog,
    GoodPregraspEntry,
    GoodPregraspKey,
    PregraspLookupKey,
    PregraspQuery,
    PregraspRecord,
    PregraspTier,
)
from anymani.pregrasp.isaac_runtime import hand_semantic_pose_w, object_pose_w_from_hand

from ..contact_layout import structural_collision_filter_pairs
from .adr import perturb_reset_position
from .runtime_state import (
    CANONICAL_JOINT_COUNT,
    HETERO_PREGRASP_STATE_ATTR,
    HeterogeneousPregraspState,
    PregraspRuntimeIdentity,
    ResolvedPregraspBatch,
    normalize_env_ids,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedEnv


def apply_structural_collision_filter(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None,
    *,
    robot_prim_path: str,
    palm_link_name: str,
    finger_link_chains: Sequence[Sequence[str]],
) -> None:
    'Write structural collision filters with FilteredPairsAPI during prestartup.'

    _ = env_ids  # Run stage-level prestartup operations once, not for episode subsets.
    from pxr import Sdf, Usd, UsdPhysics

    if "{ENV_REGEX_NS}" not in robot_prim_path:
        raise ValueError("robot_prim_path must contain {ENV_REGEX_NS}")
    stage = env.scene.stage
    pairs = structural_collision_filter_pairs(
        palm_link_name, tuple(tuple(str(link) for link in chain) for chain in finger_link_chains)
    )
    link_names = sorted({name for pair in pairs for name in pair})
    directed_edges = 0
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        for env_path in env.scene.env_prim_paths:
            robot_path = robot_prim_path.replace("{ENV_REGEX_NS}", str(env_path))
            paths = {name: f"{robot_path}/{name}" for name in link_names}
            missing = [name for name, path in paths.items() if not stage.GetPrimAtPath(path).IsValid()]
            if missing:
                raise RuntimeError(f"canonical structural collision links are missing: {missing}")
            for left, right in pairs:
                for source, target in ((paths[left], paths[right]), (paths[right], paths[left])):
                    source_prim = stage.GetPrimAtPath(source)
                    api = UsdPhysics.FilteredPairsAPI.Apply(source_prim)
                    if not api:
                        raise RuntimeError(f"cannot apply FilteredPairsAPI to {source}")
                    relationship = api.GetFilteredPairsRel() or api.CreateFilteredPairsRel()
                    target_path = Sdf.Path(target)
                    if target_path not in set(relationship.GetTargets()):
                        relationship.AddTarget(target_path)
                        directed_edges += 1
    setattr(
        env,
        "_anymani_hetero_structural_collision_stats",
        {"link_pairs": len(pairs), "directed_edges": directed_edges},
    )


def lock_ghost_joint_limits(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    robot_name: str = "robot",
) -> None:
    (
        'Set inactive canonical position limits exactly to [0,0] and clear '
        'default/target. Keep importer velocity limits finite and positive because '
        'PhysX treats zero as a continuous braking constraint.'
    )

    ids = normalize_env_ids(env_ids, num_envs=env.num_envs, device=env.device)
    robot = cast(Articulation, env.scene[robot_name])
    if robot.num_joints != CANONICAL_JOINT_COUNT:
        raise ValueError("ghost lock requires canonical 16-joint articulation")
    active = torch.tensor(active_joint_mask_by_env, dtype=torch.bool, device=env.device)
    if active.shape != (env.num_envs, CANONICAL_JOINT_COUNT):
        raise ValueError("ghost lock requires full [num_envs,16] active mask")
    limits = robot.data.joint_pos_limits[ids].clone()
    selected_active = active[ids]
    limits[..., 0] = torch.where(selected_active, limits[..., 0], torch.zeros_like(limits[..., 0]))
    limits[..., 1] = torch.where(selected_active, limits[..., 1], torch.zeros_like(limits[..., 1]))
    robot.write_joint_position_limit_to_sim(
        limits, env_ids=ids, warn_limit_violation=False  # type: ignore[arg-type]
    )
    selected_default = robot.data.default_joint_pos[ids]
    default = torch.where(selected_active, selected_default, torch.zeros_like(selected_default))
    robot.data.default_joint_pos[ids] = default
    robot.set_joint_position_target(default, env_ids=ids)  # type: ignore[arg-type]


def validate_formal_object_physics(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    expected_physics_identity: dict[str, object],
    object_name: str = "object",
) -> None:
    (
        'Check runtime PhysX mass/inertia against the formal scale probe on startup. '
        'Scene config owns density/material/solver and imported USD bytes are '
        'validated; this event checks the resulting PhysX state. DexCube fixed mass '
        'at scale 1.2: m=0.2160000056 kg and each principal inertia is '
        '1.8662401999e-4 kg m2. Fail before first reset on asset/importer drift.'
    )

    _ = env_ids  # Object properties are shared across envs; read the full view to detect prototype mismatches.
    object_asset = cast(RigidObject, env.scene[object_name])
    masses = object_asset.root_physx_view.get_masses().to(device=env.device).reshape(env.num_envs, -1)
    inertias = object_asset.root_physx_view.get_inertias().to(device=env.device).reshape(env.num_envs, -1)
    expected_mass = float(cast(float, expected_physics_identity["object_observed_mass_kg"]))
    expected_principal = torch.tensor(
        cast(list[float], expected_physics_identity["object_observed_principal_inertia_kg_m2"]),
        dtype=inertias.dtype,
        device=inertias.device,
    )
    expected_inertia = torch.zeros_like(inertias)
    expected_inertia[:, (0, 4, 8)] = expected_principal
    mass_error = torch.max(torch.abs(masses - expected_mass))
    inertia_error = torch.max(torch.abs(inertias - expected_inertia))
    if float(mass_error.item()) > 1.0e-7 or float(inertia_error.item()) > 1.0e-9:
        raise RuntimeError(
            "runtime DexCube mass/inertia disagree with formal pregrasp identity: "
            f"mass_error={float(mass_error.item()):.3e}, inertia_error={float(inertia_error.item()):.3e}"
        )
    setattr(
        env,
        "_anymani_formal_object_physics_validation",
        {
            "mass_error_kg": float(mass_error.item()),
            "inertia_error_kg_m2": float(inertia_error.item()),
        },
    )


@dataclass(frozen=True)
class PregraspAssetBinding:
    (
        'Exact lookup key and object scale for one runtime asset prototype. Isaac '
        'configclass deep-copies event params, so store canonical JSON rather than an '
        'immutable mapping proxy; restore and validate the same identity at '
        'construction and execution.'
    )

    lookup_key_json: str  # Complete JSON-safe, deepcopy-safe lookup document.
    requested_scale: float  # Actual prestartup absolute scale for this scene prototype.
    runtime_identity: PregraspRuntimeIdentity  # Generated from scene asset binding; never infer it from cache.

    def __post_init__(self) -> None:
        'Reject invalid JSON/identity and non-finite or non-positive scale.'

        if not torch.isfinite(torch.tensor(self.requested_scale)) or self.requested_scale <= 0.0:
            raise ValueError("pregrasp binding scale must be finite and positive")
        document = json.loads(self.lookup_key_json)
        if not isinstance(document, dict):
            raise ValueError("pregrasp lookup JSON must contain an object")
        lookup_key = PregraspLookupKey.from_dict(document)  # Validate SHA, finite values, and identity at config construction.
        self.runtime_identity.validate_lookup_key(lookup_key)  # Prevent a valid cache key from binding a different physical scene asset.

    @classmethod
    def from_lookup_key(
        cls,
        lookup_key: PregraspLookupKey,
        *,
        requested_scale: float,
        runtime_identity: PregraspRuntimeIdentity,
    ) -> PregraspAssetBinding:
        'Create a deterministic JSON transport binding from a validated key.'

        return cls(
            lookup_key_json=json.dumps(
                lookup_key.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
            ),
            requested_scale=requested_scale,
            runtime_identity=runtime_identity,
        )

    def resolve_lookup_key(self) -> PregraspLookupKey:
        'Restore and revalidate the exact lookup key at the event boundary.'

        document = json.loads(self.lookup_key_json)
        if not isinstance(document, dict):  # After post-init, only external mutation can violate this invariant.
            raise ValueError("pregrasp lookup JSON must contain an object")
        return PregraspLookupKey.from_dict(document)


@dataclass(frozen=True)
class PregraspResetCfg:
    (
        'Static cache, routing, asset, and hand-frame contract. asset_index_by_env is '
        'scene routing metadata, not a policy input or cache identity; each env '
        'selects a binding, and the exact PregraspLookupKey determines the lookup.'
    )

    cache_root: str  # Production AtomicPregraspCache root.
    bindings: tuple[PregraspAssetBinding, ...]  # One exact query per scene asset prototype.
    asset_index_by_env: tuple[int, ...]  # Static env-to-prototype routing, length N.
    semantic_R_ha: tuple[float, ...]  # Row-major R_ha, 9 values.
    semantic_p_ha: tuple[float, float, float]  # Palm translation p_ha, meters.
    robot_name: str = "robot"  # scene articulation key
    object_name: str = "object"  # scene rigid object key
    minimum_tier: PregraspTier = PregraspTier.CONTACT_BASIN  # Contact is the production default.
    require_basin: bool = True  # Point-only mode is excluded from training reset by default.

    def __post_init__(self) -> None:
        'Validate routing, frame, and tier; reject empty bindings and invalid prototype indices.'

        if not self.cache_root or not self.bindings or not self.asset_index_by_env:
            raise ValueError("pregrasp reset requires cache root, bindings and env routing")
        if len(self.semantic_R_ha) != 9 or len(self.semantic_p_ha) != 3:
            raise ValueError("hand semantic calibration must contain 9 rotation and 3 translation values")
        if any(index < 0 or index >= len(self.bindings) for index in self.asset_index_by_env):
            raise ValueError("asset_index_by_env references a missing pregrasp binding")
        object.__setattr__(self, "minimum_tier", PregraspTier(self.minimum_tier))


@dataclass(frozen=True)
class GoodPregraspAssetBinding:
    'Transport for a schema-3 exact key for one runtime prototype.'

    key_json: str  # Canonical key JSON safe for configclass/deepcopy.
    runtime_identity: PregraspRuntimeIdentity  # Physical hand identity supplied by the scene.

    def __post_init__(self) -> None:
        'Restore the key and validate actual hand identity during construction.'

        document = json.loads(self.key_json)
        if not isinstance(document, dict):
            raise ValueError("good-pregrasp key JSON must contain an object")
        self.runtime_identity.validate_good_key(GoodPregraspKey.from_dict(document))

    @classmethod
    def from_key(
        cls,
        key: GoodPregraspKey,
        *,
        runtime_identity: PregraspRuntimeIdentity,
    ) -> GoodPregraspAssetBinding:
        'Create stable JSON transport from a validated key.'

        return cls(
            key_json=json.dumps(key.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False),
            runtime_identity=runtime_identity,
        )

    def resolve_key(self) -> GoodPregraspKey:
        'Restore the exact key at the reset call boundary.'

        document = json.loads(self.key_json)
        if not isinstance(document, dict):
            raise ValueError("good-pregrasp key JSON must contain an object")
        return GoodPregraspKey.from_dict(document)


@dataclass(frozen=True)
class GoodPregraspResetCfg:
    'ManagerBased reset config for the Top-8 good-pregrasp catalog.'

    catalog_root: str  # Schema-3 GoodPregraspCatalog root.
    bindings: tuple[GoodPregraspAssetBinding, ...]  # One exact key per scene prototype.
    asset_index_by_env: tuple[int, ...]  # Static env-to-prototype routing [N].
    semantic_R_ha: tuple[float, ...]  # Row-major R_ha.
    semantic_p_ha: tuple[float, float, float]  # Palm translation p_ha, meters.
    rank: int = 0  # MVP uses Top-1/rank-0 only.
    require_strict: bool = False  # Replay strict v5 hard gates per entry; keep historical v4 catalogs readable.
    robot_name: str = "robot"
    object_name: str = "object"

    def __post_init__(self) -> None:
        'Validate catalog, routing, frame, and shared candidate rank.'

        if not self.catalog_root or not self.bindings or not self.asset_index_by_env:
            raise ValueError("good-pregrasp reset requires catalog root, bindings and env routing")
        if len(self.semantic_R_ha) != 9 or len(self.semantic_p_ha) != 3:
            raise ValueError("good-pregrasp hand calibration must contain 9 rotation and 3 translation values")
        if self.rank < 0:
            raise ValueError("good-pregrasp reset rank must be non-negative")
        if any(index < 0 or index >= len(self.bindings) for index in self.asset_index_by_env):
            raise ValueError("good-pregrasp env routing references a missing prototype binding")


def _resolve_records(
    *,
    config: PregraspResetCfg,
    selected_asset_indices: Sequence[int],
) -> list[PregraspRecord]:
    (
        'Resolve all unique bindings, then expand records in selected-env order. Any '
        'miss, insufficient tier, point-only entry, or corrupt payload raises a typed '
        'provider error before the caller writes robot, object, or sidecar state.'
    )

    provider = FilePregraspProvider(Path(config.cache_root))  # Revalidate index/payload at every reset; never cache bad results.
    resolved_by_asset: dict[int, PregraspRecord] = {}
    for asset_index in sorted(set(selected_asset_indices)):
        binding = config.bindings[asset_index]
        resolution = provider.resolve(
            PregraspQuery(
                lookup_key=binding.resolve_lookup_key(),
                requested_scale=binding.requested_scale,
                min_tier=config.minimum_tier,
                require_basin=config.require_basin,
            )
        )
        resolved_by_asset[asset_index] = resolution.record
    return [resolved_by_asset[index] for index in selected_asset_indices]  # Preserve the exact order of env_ids.


def _resolve_good_entries(
    *,
    env: ManagerBasedEnv,
    config: GoodPregraspResetCfg,
    selected_asset_indices: Sequence[int],
) -> list[GoodPregraspEntry]:
    'Resolve and validate the full asset axis at first reset; later partial resets gather from memory.'

    cache_attr = "_anymani_good_pregrasp_entry_cache"
    cache_identity = (
        config.catalog_root,
        tuple(binding.key_json for binding in config.bindings),
        bool(config.require_strict),
    )
    cache = getattr(env, cache_attr, None)
    if not isinstance(cache, tuple) or len(cache) != 2 or cache[0] != cache_identity:
        keys = tuple(binding.resolve_key() for binding in config.bindings)
        for binding, key in zip(config.bindings, keys, strict=True):
            binding.runtime_identity.validate_good_key(key)
        entries = GoodPregraspCatalog(Path(config.catalog_root)).resolve_many(keys)
        if config.require_strict:
            for entry in entries:
                MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)
        cache = (cache_identity, entries)
        setattr(env, cache_attr, cache)  # Immutable catalog/preload cache; preserve it across episode resets.
    entries = cast(tuple[GoodPregraspEntry, ...], cache[1])
    return [entries[index] for index in selected_asset_indices]


def _install_resolved_pregrasp_batch(
    env: ManagerBasedEnv,
    ids: torch.Tensor,
    batch: ResolvedPregraspBatch,
    *,
    semantic_R_ha: Sequence[float],
    semantic_p_ha: Sequence[float],
    robot_name: str,
    object_name: str,
) -> None:
    'Apply one reset batch after joint-limit, frame, and sidecar preflight.'

    robot = cast(Articulation, env.scene[robot_name])
    object_asset = cast(RigidObject, env.scene[object_name])
    if robot.num_joints != CANONICAL_JOINT_COUNT:
        raise ValueError("heterogeneous pregrasp reset requires canonical 16-joint transport")

    # Require q0/u0 within each selected env's soft limits; the batch contract already proves ghosts are zero.
    limits = robot.data.soft_joint_pos_limits[ids]  # `[K,16,2]`，rad
    lower, upper = limits[..., 0], limits[..., 1]
    tolerance = 1.0e-6  # Tolerate only FP32 serialization-boundary error.
    if bool(((batch.q_state_rad < lower - tolerance) | (batch.q_state_rad > upper + tolerance)).any().item()):
        raise ValueError("pregrasp q_state lies outside runtime soft joint limits")
    if bool(((batch.q_target_rad < lower - tolerance) | (batch.q_target_rad > upper + tolerance)).any().item()):
        raise ValueError("pregrasp q_target lies outside runtime soft joint limits")

    # Transform each hand-frame object pose into the shared world scene: T_wo = T_wh * T_ho.
    hand_pos_w, hand_quat_w = hand_semantic_pose_w(
        robot.data.root_pos_w[ids],
        robot.data.root_quat_w[ids],
        semantic_R_ha,
        semantic_p_ha,
    )
    object_pos_w, object_quat_w = object_pose_w_from_hand(
        hand_pos_w,
        hand_quat_w,
        batch.object_position_h_m,
        batch.object_quat_h_wxyz,
    )
    object_pos_w = perturb_reset_position(env, ids, object_pos_w, hand_quat_w)  # Apply the new distribution explicitly; command reset captures its offset in the new anchor.
    object_pose_w = torch.cat((object_pos_w, object_quat_w), dim=-1)  # `[K,7]`
    zero_joint_velocity = torch.zeros_like(batch.q_state_rad)  # Initial joint velocity qdot_0 = 0 rad/s.
    zero_object_velocity = torch.zeros(ids.numel(), 6, device=env.device)  # World twist [K,6] is zero.

    # Preflight sidecar types and shapes too, so errors cannot leave PhysX partially updated.
    existing_sidecar = getattr(env, HETERO_PREGRASP_STATE_ATTR, None)
    if existing_sidecar is None:
        sidecar = HeterogeneousPregraspState(num_envs=env.num_envs, device=env.device)
    elif isinstance(existing_sidecar, HeterogeneousPregraspState):
        sidecar = existing_sidecar
    else:
        raise RuntimeError("environment pregrasp sidecar attribute has incompatible type")
    if sidecar.num_envs != env.num_envs or sidecar.device != torch.device(env.device):
        raise RuntimeError("environment pregrasp sidecar disagrees with scene shape/device")

    # After all fallible checks, write joint state/target and object pose/velocity consecutively.
    robot.write_joint_state_to_sim(batch.q_state_rad, zero_joint_velocity, env_ids=ids)  # type: ignore[arg-type]
    robot.set_joint_position_target(batch.q_target_rad, env_ids=ids)  # type: ignore[arg-type]
    robot.set_joint_velocity_target(zero_joint_velocity, env_ids=ids)  # type: ignore[arg-type]
    object_asset.write_root_pose_to_sim(object_pose_w, env_ids=ids)  # type: ignore[arg-type]
    object_asset.write_root_velocity_to_sim(zero_object_velocity, env_ids=ids)  # type: ignore[arg-type]
    sidecar.install(ids, batch)
    if existing_sidecar is None:
        setattr(env, HETERO_PREGRASP_STATE_ATTR, sidecar)


def reset_from_pregrasp_cache(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    config: PregraspResetCfg,
) -> None:
    (
        'Apply an exact provider result to partial-reset robot/object rows and '
        'preload sidecar. Complete provider, routing, shape, and joint-limit checks '
        'before any PhysX write. Raise PregraspProviderError on '
        'identity/scale/tier/coverage failures with zero writes; raise ValueError for '
        'invalid runtime routing, shape, limits, or candidates.'
    )

    ids = normalize_env_ids(env_ids, num_envs=env.num_envs, device=env.device)  # Device-local indices [K].
    if len(config.asset_index_by_env) != env.num_envs:
        raise ValueError("pregrasp env routing length disagrees with ManagerBased scene")
    selected_asset_indices = [config.asset_index_by_env[index] for index in ids.detach().cpu().tolist()]

    # Validate provider/schema before asset access or writes; a cache miss has zero reset side effects.
    records = _resolve_records(config=config, selected_asset_indices=selected_asset_indices)
    batch = ResolvedPregraspBatch.from_records(records, device=env.device)  # Joint state [K,16] plus hand-frame object pose.
    _install_resolved_pregrasp_batch(
        env,
        ids,
        batch,
        semantic_R_ha=config.semantic_R_ha,
        semantic_p_ha=config.semantic_p_ha,
        robot_name=config.robot_name,
        object_name=config.object_name,
    )


def reset_from_good_pregrasp_catalog(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    config: GoodPregraspResetCfg,
) -> None:
    'Apply a partial good-pregrasp reset from a schema-3 exact Top-K entry.'

    ids = normalize_env_ids(env_ids, num_envs=env.num_envs, device=env.device)
    if len(config.asset_index_by_env) != env.num_envs:
        raise ValueError("good-pregrasp env routing disagrees with ManagerBased scene")
    selected_asset_indices = [config.asset_index_by_env[index] for index in ids.detach().cpu().tolist()]
    entries = _resolve_good_entries(env=env, config=config, selected_asset_indices=selected_asset_indices)
    batch = ResolvedPregraspBatch.from_good_entries(
        entries,
        rank=config.rank,
        device=env.device,
    )
    _install_resolved_pregrasp_batch(
        env,
        ids,
        batch,
        semantic_R_ha=config.semantic_R_ha,
        semantic_p_ha=config.semantic_p_ha,
        robot_name=config.robot_name,
        object_name=config.object_name,
    )


__all__ = [
    "GoodPregraspAssetBinding",
    "GoodPregraspResetCfg",
    "PregraspAssetBinding",
    "PregraspResetCfg",
    "reset_from_pregrasp_cache",
    "reset_from_good_pregrasp_catalog",
    "validate_formal_object_physics",
]
