r"""Fail-closed identity-keyed pregrasp reset event for heterogeneous ManagerBased environments。

Provider解析、tier/coverage/scale验证与所有tensor preflight必须在任何PhysX写入前完成。Reset分别写actual
$\mathbf q_s$、PD preload$\mathbf q_t$和$T_{wo}=T_{wh}T_{ho}$，随后发布full-size sidecar供
``ActionManager.reset``恢复target。Dataset row不参与查询；env到asset binding的静态routing只选择exact key。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from isaaclab.assets import Articulation, RigidObject

from anymani.pregrasp import FilePregraspProvider, PregraspLookupKey, PregraspQuery, PregraspRecord, PregraspTier
from anymani.pregrasp.isaac_runtime import hand_semantic_pose_w, object_pose_w_from_hand

from ..contact_layout import structural_collision_filter_pairs
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
    r"""在prestartup阶段以双向``FilteredPairsAPI``写结构碰撞过滤。"""

    _ = env_ids  # stage级prestartup操作不按episode subset重复
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
    r"""把inactive canonical position limits写为精确$[0,0]$并清default/target。

    Velocity limits保持importer的有限正值；PhysX会把零velocity limit解释为持续制动约束。
    """

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


@dataclass(frozen=True)
class PregraspAssetBinding:
    r"""一个runtime asset prototype所需的exact lookup key与absolute object scale。

    Isaac``configclass``会deepcopy EventTerm params；``PregraspLookupKey``内部的immutable mapping proxy不可
    pickle。因此配置层保存canonical JSON text，构造与执行两端都严格恢复并重验同一identity。
    """

    lookup_key_json: str  # JSON-safe/deepcopy-safe的完整lookup document
    requested_scale: float  # scene中该prototype的实际prestartup absolute scale
    runtime_identity: PregraspRuntimeIdentity  # 由scene asset binding独立生成，不从cache反推

    def __post_init__(self) -> None:
        r"""拒绝坏JSON、坏identity与non-finite/non-positive scale。"""

        if not torch.isfinite(torch.tensor(self.requested_scale)) or self.requested_scale <= 0.0:
            raise ValueError("pregrasp binding scale must be finite and positive")
        document = json.loads(self.lookup_key_json)
        if not isinstance(document, dict):
            raise ValueError("pregrasp lookup JSON must contain an object")
        lookup_key = PregraspLookupKey.from_dict(document)  # 配置创建时立即执行SHA/finite/identity验证
        self.runtime_identity.validate_lookup_key(lookup_key)  # 防止合法cache key误绑另一physical scene asset

    @classmethod
    def from_lookup_key(
        cls,
        lookup_key: PregraspLookupKey,
        *,
        requested_scale: float,
        runtime_identity: PregraspRuntimeIdentity,
    ) -> PregraspAssetBinding:
        r"""由严格key创建确定性JSON transport binding。"""

        return cls(
            lookup_key_json=json.dumps(
                lookup_key.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
            ),
            requested_scale=requested_scale,
            runtime_identity=runtime_identity,
        )

    def resolve_lookup_key(self) -> PregraspLookupKey:
        r"""在event调用边界恢复并重新验证exact lookup key。"""

        document = json.loads(self.lookup_key_json)
        if not isinstance(document, dict):  # post-init后只可能由外部非法mutation触发
            raise ValueError("pregrasp lookup JSON must contain an object")
        return PregraspLookupKey.from_dict(document)


@dataclass(frozen=True)
class PregraspResetCfg:
    r"""Pregrasp event的cache、routing、asset和hand-frame静态配置。

    ``asset_index_by_env``是scene routing metadata，不进入policy observation或cache identity。其第$i$项选择
    ``bindings[k]``；真正命中仍由binding内的exact :class:`PregraspLookupKey`决定。
    """

    cache_root: str  # production AtomicPregraspCache根目录
    bindings: tuple[PregraspAssetBinding, ...]  # 每个scene asset prototype一个exact query
    asset_index_by_env: tuple[int, ...]  # 长度$N$的静态env-to-prototype routing
    semantic_R_ha: tuple[float, ...]  # row-major$R_{ha}$，长度9
    semantic_p_ha: tuple[float, float, float]  # $p_{ha}$，单位m
    robot_name: str = "robot"  # scene articulation key
    object_name: str = "object"  # scene rigid object key
    minimum_tier: PregraspTier = PregraspTier.CONTACT_BASIN  # 正式主线默认contact
    require_basin: bool = True  # point-only默认不可进入训练reset

    def __post_init__(self) -> None:
        r"""验证routing、frame和tier，不允许空binding或非法prototype index。"""

        if not self.cache_root or not self.bindings or not self.asset_index_by_env:
            raise ValueError("pregrasp reset requires cache root, bindings and env routing")
        if len(self.semantic_R_ha) != 9 or len(self.semantic_p_ha) != 3:
            raise ValueError("hand semantic calibration must contain 9 rotation and 3 translation values")
        if any(index < 0 or index >= len(self.bindings) for index in self.asset_index_by_env):
            raise ValueError("asset_index_by_env references a missing pregrasp binding")
        object.__setattr__(self, "minimum_tier", PregraspTier(self.minimum_tier))


def _resolve_records(
    *,
    config: PregraspResetCfg,
    selected_asset_indices: Sequence[int],
) -> list[PregraspRecord]:
    r"""先解析全部unique bindings，再按selected env顺序展开records。

    任一miss、tier不足、point-only或corrupt payload都会在返回前抛typed provider error，因此caller尚未写入任何
    robot/object/sidecar state，不会产生半完成partial reset。
    """

    provider = FilePregraspProvider(Path(config.cache_root))  # 每次reset重验index/payload，不缓存坏结果
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
    return [resolved_by_asset[index] for index in selected_asset_indices]  # 与env_ids顺序严格一致


def reset_from_pregrasp_cache(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    config: PregraspResetCfg,
) -> None:
    r"""按exact provider结果partial-reset robot、object与preload sidecar。

    Args:
        env (ManagerBasedEnv): 拥有canonical articulation与DexCube rigid object的环境。
        env_ids (Sequence[int] | torch.Tensor | None): reset rows；``None``表示全部环境。
        config (PregraspResetCfg): exact cache/routing/frame contract。

    Raises:
        PregraspProviderError: 任一selected binding无法满足identity/scale/tier/coverage；此时零PhysX写入。
        ValueError: routing、shape、joint limits或candidate tensor不符合runtime scene。
    """

    ids = normalize_env_ids(env_ids, num_envs=env.num_envs, device=env.device)  # device-local$[K]$
    if len(config.asset_index_by_env) != env.num_envs:
        raise ValueError("pregrasp env routing length disagrees with ManagerBased scene")
    selected_asset_indices = [config.asset_index_by_env[index] for index in ids.detach().cpu().tolist()]

    # Provider与schema验证先于asset访问/写入，cache miss保证零reset副作用。
    records = _resolve_records(config=config, selected_asset_indices=selected_asset_indices)
    batch = ResolvedPregraspBatch.from_records(records, device=env.device)  # $[K,16]$+hand-frame pose
    robot = cast(Articulation, env.scene[config.robot_name])
    object_asset = cast(RigidObject, env.scene[config.object_name])
    if robot.num_joints != CANONICAL_JOINT_COUNT:
        raise ValueError("heterogeneous pregrasp reset requires canonical 16-joint transport")

    # Actual与target都必须处于selected env自己的soft limits；ghost已由batch合同验证为零。
    limits = robot.data.soft_joint_pos_limits[ids]  # $[K,16,2]$，单位rad
    lower, upper = limits[..., 0], limits[..., 1]
    tolerance = 1.0e-6  # 仅容忍FP32序列化边界，不放宽物理joint limit
    if bool(((batch.q_state_rad < lower - tolerance) | (batch.q_state_rad > upper + tolerance)).any().item()):
        raise ValueError("pregrasp q_state lies outside runtime soft joint limits")
    if bool(((batch.q_target_rad < lower - tolerance) | (batch.q_target_rad > upper + tolerance)).any().item()):
        raise ValueError("pregrasp q_target lies outside runtime soft joint limits")

    # Frame chain严格使用$T_{wh}=T_{wa}T_{ah}$和$T_{wo}=T_{wh}T_{ho}$。
    hand_pos_w, hand_quat_w = hand_semantic_pose_w(
        robot.data.root_pos_w[ids],
        robot.data.root_quat_w[ids],
        config.semantic_R_ha,
        config.semantic_p_ha,
    )
    object_pos_w, object_quat_w = object_pose_w_from_hand(
        hand_pos_w,
        hand_quat_w,
        batch.object_position_h_m,
        batch.object_quat_h_wxyz,
    )
    object_pose_w = torch.cat((object_pos_w, object_quat_w), dim=-1)  # $[K,7]$，world actor/root pose
    zero_joint_velocity = torch.zeros_like(batch.q_state_rad)  # reset$\dot q_s=0$ rad/s
    zero_object_velocity = torch.zeros(ids.numel(), 6, device=env.device)  # world CoM twist$[K,6]=0$

    # Sidecar类型/shape同样在PhysX写入前验证；新对象暂不挂到env，避免preflight失败留下半状态。
    existing_sidecar = getattr(env, HETERO_PREGRASP_STATE_ATTR, None)
    if existing_sidecar is None:
        sidecar = HeterogeneousPregraspState(num_envs=env.num_envs, device=env.device)
    elif isinstance(existing_sidecar, HeterogeneousPregraspState):
        sidecar = existing_sidecar
    else:
        raise RuntimeError("environment pregrasp sidecar attribute has incompatible type")
    if sidecar.num_envs != env.num_envs or sidecar.device != torch.device(env.device):
        raise RuntimeError("environment pregrasp sidecar disagrees with scene shape/device")

    # 所有可能失败的identity/shape/frame preflight完成后，再执行四个selected-row simulator writes。
    # IsaacLab注解写Sequence，但writer实现执行``env_ids[:,None]``；必须保留device Tensor实参。
    robot.write_joint_state_to_sim(
        batch.q_state_rad, zero_joint_velocity, env_ids=ids  # type: ignore[arg-type]
    )
    robot.set_joint_position_target(batch.q_target_rad, env_ids=ids)  # type: ignore[arg-type]
    robot.set_joint_velocity_target(zero_joint_velocity, env_ids=ids)  # type: ignore[arg-type]
    object_asset.write_root_pose_to_sim(object_pose_w, env_ids=ids)  # type: ignore[arg-type]
    object_asset.write_root_velocity_to_sim(zero_object_velocity, env_ids=ids)  # type: ignore[arg-type]

    # Sidecar是event与后续ActionManager.reset之间的commit marker；partial rows以外保持不变。
    sidecar.install(ids, batch)
    if existing_sidecar is None:
        setattr(env, HETERO_PREGRASP_STATE_ATTR, sidecar)  # tensor install完成后才发布新sidecar


__all__ = [
    "PregraspAssetBinding",
    "PregraspResetCfg",
    "reset_from_pregrasp_cache",
]
