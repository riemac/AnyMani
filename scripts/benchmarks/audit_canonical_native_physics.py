#!/usr/bin/env python3
r"""运行同一 generated hand 的 native/native-repeat/canonical matched physical probe。

该脚本回答 Q005 的两层问题：无接触模式检查 canonical ghost schema 是否保持真实 active joint 与
fingertip 局部动力学；scripted-contact 模式让独立动态 cube 接触 matched tip/palm/non-tip body，比较
contact onset、累计冲量与 object pose/twist。三个 articulation 在同一个 Isaac Sim / PhysX scene 中同时运行：

* ``native_reference`` 与 ``native_repeat`` 使用同一 source ``HandContainer``；二者给出数值基线 $S_m$；
* ``canonical`` 使用同一 container 的 ``CanonicalRuntimeCfg(enabled=True)``；它给出误差 $E_m$；
* 三只 fixed-base hand 沿 env x 轴分开放置，记录时把 fingertip position 转回各自 root-relative frame；
* self-collision、ADR 与随机 reset 始终关闭；只有 contact phase 安装 object-filtered sensors 与独立 cube；
* 三侧写入相同 source-joint $q_0,\dot q_0,u_t$，canonical ghost state/target 始终固定为零。

本 probe 不声明物理等价阈值。输出 ``trace.npz`` 保留可事后复算的 dense arrays，``summary.json``
保存 target/limit identity、$E_m$、$S_m$ 与 $E_m/S_m$。接触与物体字段只在 scripted-contact
模式可解释；reward 尚未组装 task manager，因此两种模式均保存显式结构零，不形成 reward 结论。

运行示例：

```bash
source /home/hac/isaac/env_isaaclab/bin/activate
PYTHONPATH=source/anymani python scripts/benchmarks/audit_canonical_native_physics.py \
  --asset_path source/anymani/anymani/assets/generated/2026-08-16_14-55-19/single_palm_allegro/right_t4_i4_m4_r4 \
  --output_dir logs/benchmarks/heterogeneous_rl/canonical_native_audit/right_t4_i4_m4_r4 \
  --headless
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    r"""解析资产、轨迹与 AppLauncher 参数。

    Returns:
        argparse.Namespace: ``asset_path``/``output_dir`` 与固定时域参数；IsaacLab 参数由
            :class:`AppLauncher` 追加。
    """

    parser = argparse.ArgumentParser(description="Audit one generated hand in native and canonical PhysX schemas.")
    parser.add_argument("--asset_path", type=Path, required=True, help="Source generated-hand bundle root.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Unique local evidence directory.")
    parser.add_argument("--settle_steps", type=int, default=8, help="Physics steps holding q_home before excitation.")
    parser.add_argument("--response_steps", type=int, default=60, help="Physics steps after the target step input.")
    parser.add_argument("--target_delta_rad", type=float, default=0.05, help="Alternating source-joint step magnitude.")
    parser.add_argument(
        "--phase",
        choices=("no_contact", "scripted_contact"),
        default="no_contact",
        help="Audit active step response or a dynamic cube approaching the first semantic fingertip.",
    )
    parser.add_argument("--contact_start_offset_m", type=float, default=0.04)
    parser.add_argument("--contact_approach_speed_m_s", type=float, default=0.20)
    parser.add_argument("--contact_cube_size_m", type=float, default=0.02)
    parser.add_argument(
        "--contact_role",
        choices=("tip", "palm", "finger_non_tip"),
        default="tip",
        help="Semantic collision role targeted by scripted contact.",
    )
    parser.add_argument("--contact_role_index", type=int, default=0, help="Index within tip/non-tip role axis.")
    parser.add_argument(
        "--ghost_actuator_mode",
        choices=("inherited", "zero"),
        default="inherited",
        help="Keep production-v0 ghost drive properties or zero them as an isolated actuator probe.",
    )
    parser.add_argument(
        "--ghost_lock_mode",
        choices=("zero", "position_zero", "velocity_zero", "epsilon", "importer"),
        default="zero",
        help="Isolate position/velocity locks; epsilon uses +/- ghost_limit_rad and importer velocity.",
    )
    parser.add_argument(
        "--ghost_limit_rad",
        type=float,
        default=1.0e-3,
        help="Symmetric ghost position deadband used only by --ghost_lock_mode epsilon.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.settle_steps < 0 or args.response_steps <= 0:
        parser.error("settle_steps must be non-negative and response_steps must be positive")
    if not math.isfinite(args.target_delta_rad) or args.target_delta_rad <= 0.0:
        parser.error("target_delta_rad must be finite and positive")
    if not math.isfinite(args.ghost_limit_rad) or args.ghost_limit_rad <= 0.0:
        parser.error("ghost_limit_rad must be finite and positive")
    contact_values = (args.contact_start_offset_m, args.contact_approach_speed_m_s, args.contact_cube_size_m)
    if any(not math.isfinite(value) or value <= 0.0 for value in contact_values):
        parser.error("contact offset, approach speed and cube size must be finite and positive")
    if args.contact_role_index < 0:
        parser.error("contact_role_index must be non-negative")
    return args


ARGS = _parse_args()
"""AppLauncher 之前可用的纯 CLI 配置。"""

APP_LAUNCHER = AppLauncher(ARGS)
"""Isaac Sim application owner；所有 USD/PhysX imports 必须位于其后。"""

SIMULATION_APP = APP_LAUNCHER.app
"""当前 Kit application；对象存活覆盖整个 probe，进程退出由父 benchmark 回收。"""


import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from anymani.assets.asset_sidecar import restore_hand_cfg_snapshot  # noqa: E402
from anymani.assets.bank import HandBank, HandBankCfg  # noqa: E402
from anymani.assets.bank.path_utils import resolve_bank_path  # noqa: E402
from anymani.distill.diagnostics.recording.rl import record_optional_rl_phase  # noqa: E402
from anymani.robots.hand_spawn import (  # noqa: E402
    CanonicalRuntimeCfg,
    HandFrameCfg,
    HandJointInitCfg,
    HandSpawnAdapter,
    HandSpawnCfg,
    HandUrdfSpawnCfg,
)
from anymani.tasks.gm.contact_sensors import (  # noqa: E402
    build_contact_sensor_layout_from_assets,
    make_contact_sensor_cfg,
)
from anymani.tasks.gm.physical_audit import (  # noqa: E402
    PHYSICAL_AUDIT_SCHEMA_VERSION,
    candidate_indices_in_reference_semantic_order,
    compare_canonical_against_native_repeat,
)
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

_ROOT_X_BY_MODE = {
    "native_reference": -0.6,
    "native_repeat": 0.0,
    "canonical": 0.6,
}
"""三只 hand 的 env-frame x anchor，单位 m；分离后不会互相发生刚体接触。"""


def _resolve_source_asset(asset_path: Path):
    r"""通过 HandBank 唯一入口解析一个带 typed geometry semantics 的 source bundle。

    Args:
        asset_path (Path): generated mother/variant bundle root，可为仓库相对或绝对路径。

    Returns:
        HandContainer: 经过 sidecar、URDF mesh closure 与 geometry semantics 验证的单资产。
    """

    resolved_path = resolve_bank_path(asset_path)  # shell cwd 无关的 AnyMani-root anchored 路径
    selection = HandBank(
        HandBankCfg(
            source_mode="mixed",
            selection_mode="explicit",
            containers=(str(resolved_path),),
            validate_mesh_relpaths=True,
            parse_visual_rgba=False,
            require_geometry_semantics=True,
        )
    ).resolve()
    if len(selection.assets) != 1:
        raise RuntimeError(f"physical audit requires exactly one source asset, got {len(selection.assets)}")
    return selection.assets[0]


def _source_home_by_joint(container: Any) -> dict[str, float]:
    r"""恢复并投影 source active-joint $q_{home}$ 到 typed hard limits。

    部分历史 mother 使用全零 $q_{home}$，但真实 URDF interval 可能不包含 0。IsaacLab 在 PhysX
    view 初始化前先验证 ``ArticulationCfg.init_state``，因此 matched probe 的共同初态定义为：

    $$
    q_{0,j}=\Pi_{[l_j,u_j]}(q_{home,j}).
    $$

    该投影同时写入 native 与 canonical active joints，不改变两种表示之间的比较命题。

    Args:
        container (HandContainer): 已严格解析的 source asset。

    Returns:
        dict[str, float]: source joint name 到共同合法初态 $q_0$（rad）的映射。
    """

    semantics = container.geometry_semantics
    if semantics is None:
        raise ValueError("physical audit source asset must expose typed geometry semantics")
    names = tuple(str(name) for name in semantics.active_joint_names)  # source generalized-coordinate identity
    values = tuple(float(value) for value in semantics.q_home_rad)  # 与 names 同序，任务参考 rad
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError("geometry semantics q_home must identify each source active joint exactly once")

    hand_cfg_raw = container.sidecar.get("hand_cfg")
    if not isinstance(hand_cfg_raw, dict):
        raise ValueError("physical audit source sidecar lacks typed hand_cfg")
    hand_cfg = restore_hand_cfg_snapshot(hand_cfg_raw)  # source joint limits 的 typed 真源
    limits_by_name = {
        joint.name: joint.limit
        for finger in hand_cfg.fingers
        for joint in finger.joints
        if joint.joint_type == "revolute"
    }
    projected: dict[str, float] = {}
    for name, value in zip(names, values, strict=True):
        limit = limits_by_name.get(name)
        if limit is None:
            raise ValueError(f"source active joint {name!r} lacks typed hard limits")
        lower = float(getattr(limit, "lower"))  # typed ``JointLimitCfg.lower``，rad
        upper = float(getattr(limit, "upper"))  # typed ``JointLimitCfg.upper``，rad
        projected[name] = min(max(value, lower), upper)  # $\Pi_{[l_j,u_j]}(q_{home,j})$
    return projected


def _make_spawn_cfg(container: Any, *, canonical: bool, root_x_m: float) -> HandSpawnCfg:
    r"""构造仅表示方式不同的 fixed-base hand spawn 配置。

    Args:
        container (HandContainer): 三侧共享的 source physical asset。
        canonical (bool): ``False`` 直接 spawn source；``True`` lower 到 canonical v1。
        root_x_m (float): 当前实例在 env frame 中的 x 平移，单位 m。

    Returns:
        HandSpawnCfg: 相同 URDF importer、actuator和 frame orientation 的声明式配置。
    """

    source_home = _source_home_by_joint(container)  # native 初始位置；canonical adapter 会改写为全局 boot pose
    return HandSpawnCfg(
        bank=HandBankCfg(),  # resolved_assets 由 adapter 显式注入，因此不触发第二次 bank discovery
        frame=HandFrameCfg(anchor_p_eh=(float(root_x_m), 0.0, 0.5)),
        joint_init=HandJointInitCfg(joint_pos=source_home),
        urdf=HandUrdfSpawnCfg(
            fix_base=True,
            merge_fixed_joints=False,
            use_stable_usd_cache=True,
            self_collision=False,
            activate_contact_sensors=ARGS.phase == "scripted_contact",
        ),
        canonical_runtime=CanonicalRuntimeCfg(enabled=canonical, output_root="outputs"),
        asset_routing="round_robin",
        restore_visual_materials=False,
        validate_same_schema=True,
    )


def _aligned_tip_body_names(
    source_container: Any,
    canonical_adapter: HandSpawnAdapter,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    r"""返回 native 与 canonical 中按同一 semantic finger slot 对齐的真实 fingertip names。"""

    native_layout = build_contact_sensor_layout_from_assets((source_container,), validate_all_assets=True)
    canonical_layout = build_contact_sensor_layout_from_assets(
        canonical_adapter.selection.assets,
        validate_all_assets=True,
    )
    native_slots = tuple(name.partition("_")[0] for name in native_layout.fingertip_link_names)
    canonical_pairs = tuple((name.partition("_")[0], name) for name in canonical_layout.fingertip_link_names)
    canonical_active_pairs = tuple(pair for pair in canonical_pairs if pair[0] in set(native_slots))
    canonical_active_slots = tuple(slot for slot, _ in canonical_active_pairs)
    canonical_order = candidate_indices_in_reference_semantic_order(native_slots, canonical_active_slots)
    canonical_names = tuple(canonical_active_pairs[index][1] for index in canonical_order)
    return tuple(native_layout.fingertip_link_names), canonical_names


def _joint_child_by_name(container: Any) -> dict[str, str]:
    r"""从 resolved sidecar 的 typed hand snapshot 恢复 revolute/fixed joint→child body identity。"""

    hand_cfg_raw = container.sidecar.get("hand_cfg")
    if not isinstance(hand_cfg_raw, dict):
        raise ValueError(f"asset {container.asset_id!r} lacks typed hand_cfg for contact body routing")
    hand_cfg = restore_hand_cfg_snapshot(hand_cfg_raw)
    return {joint.name: str(joint.child) for finger in hand_cfg.fingers for joint in finger.joints}


def _aligned_contact_target_body_names(
    source_container: Any,
    canonical_adapter: HandSpawnAdapter,
) -> tuple[str, str]:
    r"""按 tip/palm/finger-non-tip 角色返回一对 source/canonical collision body names。"""

    native_layout = build_contact_sensor_layout_from_assets(
        (source_container,),
        validate_all_assets=True,
        collision_only=ARGS.contact_role == "finger_non_tip",
    )  # non-tip 角色只允许选择真实 collision-bearing body
    canonical_container = canonical_adapter.selection.assets[0]
    if ARGS.contact_role == "palm":
        if ARGS.contact_role_index != 0:
            raise ValueError("palm role contains exactly one body at index 0")
        canonical_layout = build_contact_sensor_layout_from_assets((canonical_container,), validate_all_assets=True)
        return native_layout.palm_link_name, canonical_layout.palm_link_name
    if ARGS.contact_role == "tip":
        native_names, canonical_names = _aligned_tip_body_names(source_container, canonical_adapter)
        if ARGS.contact_role_index >= len(native_names):
            raise ValueError(f"tip role index {ARGS.contact_role_index} exceeds real tip count {len(native_names)}")
        return native_names[ARGS.contact_role_index], canonical_names[ARGS.contact_role_index]

    source_joint_by_child = {child: joint for joint, child in _joint_child_by_name(source_container).items()}
    source_to_canonical = dict(canonical_adapter.canonical_artifacts[0].routing.source_to_canonical)
    canonical_child_by_joint = _joint_child_by_name(canonical_container)
    routable_non_tip_pairs: list[tuple[str, str]] = []
    for source_body in native_layout.finger_non_tip_link_names:
        source_joint = source_joint_by_child.get(source_body)
        canonical_joint = source_to_canonical.get(source_joint) if source_joint is not None else None
        canonical_body = canonical_child_by_joint.get(canonical_joint) if canonical_joint is not None else None
        if canonical_body is not None:
            routable_non_tip_pairs.append((source_body, canonical_body))
    if ARGS.contact_role_index >= len(routable_non_tip_pairs):
        raise ValueError(
            f"finger_non_tip role index {ARGS.contact_role_index} exceeds active routable body count "
            f"{len(routable_non_tip_pairs)}"
        )
    return routable_non_tip_pairs[ARGS.contact_role_index]


def _make_probe_object_cfg(*, prim_path: str) -> RigidObjectCfg:
    r"""构造无重力动态接触 cube；初始远离 hand，runtime 再按 tip pose 写入 approach state。"""

    cube_size = float(ARGS.contact_cube_size_m)  # 三轴相同的 cube edge length，m
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(cube_size, cube_size, cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )


def _make_scene_cfg(
    container: Any,
) -> tuple[
    InteractiveSceneCfg,
    HandSpawnAdapter,
    HandSpawnAdapter,
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    r"""建立包含两个 native instances 与一个 canonical instance 的单-env scene。

    Args:
        container (HandContainer): 三个 articulation 共享的 source bundle。

    Returns:
        tuple: scene cfg、native/canonical adapter，以及 mode→object/sensor scene keys。
    """

    native_reference_adapter = HandSpawnAdapter(
        _make_spawn_cfg(container, canonical=False, root_x_m=_ROOT_X_BY_MODE["native_reference"]),
        resolved_assets=(container,),
    )
    native_repeat_adapter = HandSpawnAdapter(
        _make_spawn_cfg(container, canonical=False, root_x_m=_ROOT_X_BY_MODE["native_repeat"]),
        resolved_assets=(container,),
    )
    canonical_adapter = HandSpawnAdapter(
        _make_spawn_cfg(container, canonical=True, root_x_m=_ROOT_X_BY_MODE["canonical"]),
        resolved_assets=(container,),
    )

    # 三个 asset keys 形成三个独立 PhysX articulation views；空间平移不进入 root-relative trace。
    native_reference_cfg = native_reference_adapter.build_articulation_cfg(
        prim_path="{ENV_REGEX_NS}/NativeReference"
    )
    native_repeat_cfg = native_repeat_adapter.build_articulation_cfg(prim_path="{ENV_REGEX_NS}/NativeRepeat")
    canonical_cfg = canonical_adapter.build_articulation_cfg(prim_path="{ENV_REGEX_NS}/Canonical")

    @configclass
    class CanonicalNativeAuditSceneCfg(InteractiveSceneCfg):
        r"""三个 fixed-base hand articulation 的最小无接触审计 scene。"""

        native_reference: ArticulationCfg = native_reference_cfg
        native_repeat: ArticulationCfg = native_repeat_cfg
        canonical: ArticulationCfg = canonical_cfg

    scene_cfg = CanonicalNativeAuditSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    object_keys: dict[str, str] = {}
    sensor_keys: dict[str, str] = {}
    target_body_names: dict[str, str] = {}
    if ARGS.phase == "scripted_contact":
        native_target_body, canonical_target_body = _aligned_contact_target_body_names(container, canonical_adapter)
        target_body_names = {
            "native_reference": native_target_body,
            "native_repeat": native_target_body,
            "canonical": canonical_target_body,
        }
        robot_prim_by_mode = {
            "native_reference": "{ENV_REGEX_NS}/NativeReference",
            "native_repeat": "{ENV_REGEX_NS}/NativeRepeat",
            "canonical": "{ENV_REGEX_NS}/Canonical",
        }
        for mode in target_body_names:
            object_key = f"{mode}_object"
            sensor_key = f"{mode}_contact"
            object_prim_path = f"{{ENV_REGEX_NS}}/{mode.title().replace('_', '')}Object"
            setattr(scene_cfg, object_key, _make_probe_object_cfg(prim_path=object_prim_path))
            setattr(
                scene_cfg,
                sensor_key,
                make_contact_sensor_cfg(
                    target_body_names[mode],
                    robot_prim_path=robot_prim_by_mode[mode],
                    object_prim_path=object_prim_path,
                    history_length=0,
                    track_air_time=False,
                    track_friction_forces=True,
                    max_contact_data_count_per_prim=16,
                ),
            )
            object_keys[mode] = object_key
            sensor_keys[mode] = sensor_key
    return scene_cfg, native_reference_adapter, canonical_adapter, object_keys, sensor_keys, target_body_names


def _write_root_and_joint_state(
    scene: InteractiveScene,
    robots: MappingLike,
    *,
    source_home: dict[str, float],
    source_joint_names: tuple[str, ...],
    canonical_joint_names: tuple[str, ...],
) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
    r"""把三侧写入相同 source-joint home state，并返回 joint routing 与 full targets。

    Args:
        scene (InteractiveScene): 已在 ``sim.reset()`` 后初始化的 scene。
        robots (MappingLike): mode 到 :class:`Articulation` 的映射。
        source_home (dict[str, float]): source joint home angles，单位 rad。
        source_joint_names (tuple[str, ...]): 审计共同 source joint 轴 $J$。
        canonical_joint_names (tuple[str, ...]): 与 source 轴一一对应的 canonical joint 名称。

    Returns:
        tuple[dict[str,list[int]],dict[str,torch.Tensor]]: 各 mode 的 active joint IDs 与 full home target。
    """

    # 按 runtime joint names 识别 active slots；CLI 在生产零锁与隔离候选之间切换。
    canonical_robot = robots["canonical"]
    canonical_active_names = set(canonical_joint_names)
    canonical_active_mask = torch.tensor(
        [name in canonical_active_names for name in canonical_robot.joint_names],
        device=canonical_robot.device,
        dtype=torch.bool,
    )  # ``[J_canonical=16]``，严格沿实际 PhysX joint row order
    canonical_env_ids = torch.tensor([0], device=canonical_robot.device, dtype=torch.long)  # PhysX row selector
    physx_env_ids = cast(Sequence[int], canonical_env_ids)  # IsaacLab runtime 接受 Tensor，stub 仍声明 Sequence
    if ARGS.ghost_lock_mode in {"zero", "position_zero", "epsilon"}:
        canonical_limits = canonical_robot.data.joint_pos_limits.clone()  # ``[1,16,2]``，rad
        ghost_half_width = 0.0 if ARGS.ghost_lock_mode != "epsilon" else float(ARGS.ghost_limit_rad)
        canonical_limits[:, ~canonical_active_mask, 0] = -ghost_half_width  # ghost lower limit，rad
        canonical_limits[:, ~canonical_active_mask, 1] = ghost_half_width  # ghost upper limit，rad
        canonical_robot.write_joint_position_limit_to_sim(
            canonical_limits,
            joint_ids=list(range(canonical_robot.num_joints)),
            env_ids=physx_env_ids,
            warn_limit_violation=False,
        )
    if ARGS.ghost_lock_mode in {"zero", "velocity_zero"}:
        canonical_velocity_limits = canonical_robot.data.joint_vel_limits.clone()  # ``[1,16]``，rad/s
        canonical_velocity_limits[:, ~canonical_active_mask] = 0.0
        canonical_robot.write_joint_velocity_limit_to_sim(
            canonical_velocity_limits,
            joint_ids=list(range(canonical_robot.num_joints)),
            env_ids=physx_env_ids,
        )

    # Implicit actuator cfg 默认匹配 ``.*``，会把 active stiffness/damping/armature/friction 同样施加到 ghost。
    # ``zero`` 只移除这些非物理负载；位置/速度 limit 由独立 ``ghost_lock_mode`` 控制。
    ghost_joint_ids = torch.nonzero(~canonical_active_mask, as_tuple=False).flatten().tolist()
    if ARGS.ghost_actuator_mode == "zero" and ghost_joint_ids:
        canonical_robot.write_joint_stiffness_to_sim(0.0, joint_ids=ghost_joint_ids, env_ids=physx_env_ids)
        canonical_robot.write_joint_damping_to_sim(0.0, joint_ids=ghost_joint_ids, env_ids=physx_env_ids)
        canonical_robot.write_joint_effort_limit_to_sim(0.0, joint_ids=ghost_joint_ids, env_ids=physx_env_ids)
        canonical_robot.write_joint_armature_to_sim(0.0, joint_ids=ghost_joint_ids, env_ids=physx_env_ids)
        canonical_robot.write_joint_friction_coefficient_to_sim(
            0.0,
            joint_dynamic_friction_coeff=0.0,
            joint_viscous_friction_coeff=0.0,
            joint_ids=ghost_joint_ids,
            env_ids=physx_env_ids,
        )

    joint_ids_by_mode: dict[str, list[int]] = {}
    full_home_by_mode: dict[str, torch.Tensor] = {}
    for mode, robot in robots.items():
        names = canonical_joint_names if mode == "canonical" else source_joint_names
        joint_ids, resolved_names = robot.find_joints(list(names), preserve_order=True)
        if tuple(resolved_names) != names:
            raise RuntimeError(f"{mode} active joint routing mismatch: expected={names}, resolved={resolved_names}")
        joint_ids_by_mode[mode] = joint_ids

        # Full state 保留每个 articulation 自己的 generalized axis；只有 matched active source slots写 q_home。
        joint_pos = torch.zeros_like(robot.data.default_joint_pos)  # ``[1,J_mode]``，rad；canonical ghost=0
        joint_vel = torch.zeros_like(robot.data.default_joint_vel)  # ``[1,J_mode]``，rad/s
        source_home_tensor = torch.tensor(
            [source_home[name] for name in source_joint_names],
            device=robot.device,
            dtype=joint_pos.dtype,
        )  # ``[J]``，三侧完全相同的物理 generalized coordinates
        joint_pos[:, joint_ids] = source_home_tensor

        root_state = robot.data.default_root_state.clone()  # cfg 中的 env-local $T_{ea}$ 与零 root velocity
        root_state[:, :3] += scene.env_origins  # $p_{wa}=p_{we}+p_{ea}$
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        full_home_by_mode[mode] = joint_pos.clone()  # actuator target 与真实初态一致，避免隐式启动脉冲
    scene.reset()
    return joint_ids_by_mode, full_home_by_mode


class MappingLike(dict[str, Articulation]):
    r"""给 Pyright 与科研读者声明 mode→Articulation 的固定 runtime 映射。"""


def _tip_ids_by_mode(
    robots: MappingLike,
    *,
    source_container: Any,
    canonical_adapter: HandSpawnAdapter,
) -> dict[str, list[int]]:
    r"""按 source/canonical sidecar 的显式 ``is_tip`` 语义解析真实 fingertip bodies。

    Args:
        robots (MappingLike): 三个 runtime articulations。
        source_container (HandContainer): native contact topology 真源。
        canonical_adapter (HandSpawnAdapter): canonical derived sidecar 真源。

    Returns:
        dict[str, list[int]]: 各 mode 按相同 semantic finger 顺序排列的 body IDs。
    """

    native_names, canonical_names_in_native_order = _aligned_tip_body_names(source_container, canonical_adapter)

    ids_by_mode: dict[str, list[int]] = {}
    for mode, robot in robots.items():
        names = canonical_names_in_native_order if mode == "canonical" else native_names
        body_ids, resolved_names = robot.find_bodies(list(names), preserve_order=True)
        if tuple(resolved_names) != tuple(names):
            raise RuntimeError(f"{mode} fingertip routing mismatch: expected={names}, resolved={resolved_names}")
        ids_by_mode[mode] = body_ids
    return ids_by_mode


def _empty_trace_lists() -> dict[str, list[torch.Tensor]]:
    r"""分配一条 phase-A trace 的 append-only GPU tensor lists。"""

    return {
        "joint_pos_rad": [],
        "joint_vel_rad_s": [],
        "joint_target_rad": [],
        "tip_pos_m": [],
        "tip_quat_wxyz": [],
        "object_pos_m": [],
        "object_quat_wxyz": [],
        "object_lin_vel_m_s": [],
        "object_ang_vel_rad_s": [],
        "contact_force_N": [],
        "reward_terms": [],
    }


def _capture_sample(
    traces: dict[str, dict[str, list[torch.Tensor]]],
    robots: MappingLike,
    *,
    joint_ids_by_mode: dict[str, list[int]],
    tip_ids_by_mode: dict[str, list[int]],
    active_targets_by_mode: dict[str, torch.Tensor],
    scene: InteractiveScene | None = None,
    objects_by_mode: dict[str, RigidObject] | None = None,
    sensor_keys_by_mode: dict[str, str] | None = None,
) -> None:
    r"""采集 active joint、root-relative tip，以及可选 object/contact post-physics state。"""

    for mode, robot in robots.items():
        joint_ids = joint_ids_by_mode[mode]  # common source-joint axis $J$
        tip_ids = tip_ids_by_mode[mode]  # common semantic fingertip axis $K$
        trace = traces[mode]
        trace["joint_pos_rad"].append(robot.data.joint_pos[0, joint_ids].detach().clone())
        trace["joint_vel_rad_s"].append(robot.data.joint_vel[0, joint_ids].detach().clone())
        trace["joint_target_rad"].append(active_targets_by_mode[mode].detach().clone())

        # 三侧仅 root translation 不同；减去各自 root position 后得到共同 `{a}`-relative tip origin。
        root_position_w = robot.data.root_pos_w[0]  # ``[3]``，m
        tip_position_root = robot.data.body_pos_w[0, tip_ids] - root_position_w[None, :]  # ``[K,3]``，m
        trace["tip_pos_m"].append(tip_position_root.detach().clone())
        trace["tip_quat_wxyz"].append(robot.data.body_quat_w[0, tip_ids].detach().clone())

        if objects_by_mode is None or sensor_keys_by_mode is None or scene is None:
            # Phase A 不存在 object/contact；结构零只保持跨阶段 artifact schema，不形成任务结论。
            trace["object_pos_m"].append(torch.zeros(3, device=robot.device, dtype=robot.data.joint_pos.dtype))
            trace["object_quat_wxyz"].append(
                torch.tensor((1.0, 0.0, 0.0, 0.0), device=robot.device, dtype=robot.data.joint_pos.dtype)
            )
            trace["object_lin_vel_m_s"].append(torch.zeros(3, device=robot.device, dtype=robot.data.joint_pos.dtype))
            trace["object_ang_vel_rad_s"].append(torch.zeros(3, device=robot.device, dtype=robot.data.joint_pos.dtype))
            trace["contact_force_N"].append(torch.zeros(1, device=robot.device, dtype=robot.data.joint_pos.dtype))
        else:
            object_asset = objects_by_mode[mode]
            trace["object_pos_m"].append((object_asset.data.root_pos_w[0] - root_position_w).detach().clone())
            trace["object_quat_wxyz"].append(object_asset.data.root_quat_w[0].detach().clone())
            trace["object_lin_vel_m_s"].append(object_asset.data.root_lin_vel_w[0].detach().clone())
            trace["object_ang_vel_rad_s"].append(object_asset.data.root_ang_vel_w[0].detach().clone())

            # 与任务 contact bit 一致：normal+friction 后先逐 pair 求模，再在非 batch 维取最大。
            sensor = scene[sensor_keys_by_mode[mode]]
            force_w = getattr(sensor.data, "force_matrix_w", None)
            if force_w is None:
                force_w = getattr(sensor.data, "net_forces_w", None)
            if force_w is None:
                raise RuntimeError(f"contact sensor {sensor_keys_by_mode[mode]!r} lacks force tensors")
            total_force_w = torch.nan_to_num(force_w, nan=0.0)
            friction_w = getattr(sensor.data, "friction_forces_w", None)
            if friction_w is not None:
                total_force_w = total_force_w + torch.nan_to_num(friction_w, nan=0.0)
            pair_magnitude = torch.linalg.vector_norm(total_force_w, dim=-1)
            contact_magnitude = pair_magnitude.reshape(pair_magnitude.shape[0], -1).amax(dim=-1)
            trace["contact_force_N"].append(contact_magnitude.detach().clone())  # ``[C=1]``，N
        trace["reward_terms"].append(torch.zeros(1, device=robot.device, dtype=robot.data.joint_pos.dtype))


def _initialize_scripted_contact_objects(
    robots: MappingLike,
    objects_by_mode: dict[str, RigidObject],
    *,
    contact_body_ids_by_mode: dict[str, int],
    palm_body_ids_by_mode: dict[str, int],
) -> None:
    r"""沿 palm→target body 的 outward direction 放置 cube，再向目标 body 入射。"""

    device = robots["canonical"].device
    dtype = robots["canonical"].data.joint_pos.dtype
    fallback_direction_w = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)  # palm role 的手心法向近似
    identity_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
    for mode, robot in robots.items():
        target_body_id = contact_body_ids_by_mode[mode]  # matched tip/palm/non-tip runtime body row
        target_body_position_w = robot.data.body_pos_w[0, target_body_id]
        palm_position_w = robot.data.body_pos_w[0, palm_body_ids_by_mode[mode]]
        radial_direction_w = target_body_position_w - palm_position_w  # palm center→target body，world frame
        radial_norm = torch.linalg.vector_norm(radial_direction_w)
        approach_direction_w = (
            radial_direction_w / radial_norm if float(radial_norm.item()) > 1.0e-6 else fallback_direction_w
        )  # 从 hand 外侧沿 $-d$ 入射；palm 自身退回 +z
        approach_offset_w = approach_direction_w * float(ARGS.contact_start_offset_m)  # m
        approach_velocity_w = -approach_direction_w * float(ARGS.contact_approach_speed_m_s)  # m/s
        object_asset = objects_by_mode[mode]
        root_pose = torch.cat((target_body_position_w + approach_offset_w, identity_quat)).unsqueeze(0)  # ``[1,7]``
        root_velocity = torch.cat((approach_velocity_w, torch.zeros_like(approach_velocity_w))).unsqueeze(0)  # ``[1,6]``
        object_asset.write_root_pose_to_sim(root_pose)
        object_asset.write_root_velocity_to_sim(root_velocity)


def _stack_traces(traces: dict[str, dict[str, list[torch.Tensor]]]) -> dict[str, dict[str, torch.Tensor]]:
    r"""把 append-only sample lists lower 成 ``[T,...]`` tensors。"""

    return {
        mode: {field: torch.stack(samples, dim=0) for field, samples in trace.items()}
        for mode, trace in traces.items()
    }


def _write_artifacts(
    output_dir: Path,
    *,
    traces: dict[str, dict[str, torch.Tensor]],
    summary: dict[str, object],
) -> None:
    r"""保存 dense NPZ 与 JSON-safe summary；所有 arrays 保留原始单位。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{mode}__{field}": tensor.detach().cpu().numpy()
        for mode, trace in traces.items()
        for field, tensor in trace.items()
    }
    np.savez_compressed(output_dir / "trace.npz", **arrays)  # dense raw evidence，可用新阈值事后重算
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run() -> dict[str, object]:
    r"""执行 matched no-contact step response 并发布可审计结果。"""

    record_optional_rl_phase("physical_audit_asset_resolve", "start", asset_path=str(ARGS.asset_path))
    source_container = _resolve_source_asset(ARGS.asset_path)
    source_home = _source_home_by_joint(source_container)  # source joint identity→rad
    record_optional_rl_phase(
        "physical_audit_asset_resolve",
        "complete",
        asset_id=source_container.asset_id,
        source_dof=len(source_home),
    )

    record_optional_rl_phase("physical_audit_scene_construct", "start")
    (
        scene_cfg,
        native_adapter,
        canonical_adapter,
        object_keys_by_mode,
        sensor_keys_by_mode,
        contact_body_names_by_mode,
    ) = _make_scene_cfg(source_container)
    canonical_artifact = canonical_adapter.canonical_artifacts[0]
    routing = canonical_artifact.routing
    source_joint_names = tuple(source_name for source_name, _ in routing.source_to_canonical)
    canonical_joint_names = tuple(canonical_name for _, canonical_name in routing.source_to_canonical)
    if set(source_joint_names) != set(source_home):
        raise RuntimeError("canonical routing does not cover source q_home joint identity")

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, render_interval=1, device="cuda:0"))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    sim_dt_s = float(sim.get_physics_dt())  # $1/120$ s，contact impulse/trajectory time base
    robots = MappingLike(
        native_reference=scene["native_reference"],
        native_repeat=scene["native_repeat"],
        canonical=scene["canonical"],
    )
    objects_by_mode = (
        {mode: cast(RigidObject, scene[key]) for mode, key in object_keys_by_mode.items()}
        if object_keys_by_mode
        else None
    )
    record_optional_rl_phase("physical_audit_scene_construct", "complete")

    joint_ids_by_mode, full_home_by_mode = _write_root_and_joint_state(
        scene,
        robots,
        source_home=source_home,
        source_joint_names=source_joint_names,
        canonical_joint_names=canonical_joint_names,
    )
    tip_ids_by_mode = _tip_ids_by_mode(
        robots,
        source_container=source_container,
        canonical_adapter=canonical_adapter,
    )
    contact_body_ids_by_mode: dict[str, int] = {}
    palm_body_ids_by_mode: dict[str, int] = {}
    for mode, body_name in contact_body_names_by_mode.items():
        body_ids, resolved_names = robots[mode].find_bodies([body_name], preserve_order=True)
        if len(body_ids) != 1 or resolved_names != [body_name]:
            raise RuntimeError(f"{mode} contact body routing failed for {body_name!r}: {resolved_names}")
        contact_body_ids_by_mode[mode] = body_ids[0]
        palm_ids, palm_names = robots[mode].find_bodies(["palm"], preserve_order=True)
        if len(palm_ids) != 1 or palm_names != ["palm"]:
            raise RuntimeError(f"{mode} palm body routing failed: {palm_names}")
        palm_body_ids_by_mode[mode] = palm_ids[0]

    # 先比较 active limits，再在三侧交集内构造完全相同的 alternating step target。
    limits_by_mode = {
        mode: robot.data.joint_pos_limits[0, joint_ids_by_mode[mode]].detach().clone()
        for mode, robot in robots.items()
    }  # 每项 ``[J,2]``，rad
    lower_intersection = torch.stack([limits[:, 0] for limits in limits_by_mode.values()]).amax(dim=0)
    upper_intersection = torch.stack([limits[:, 1] for limits in limits_by_mode.values()]).amin(dim=0)
    if torch.any(lower_intersection > upper_intersection):
        raise RuntimeError("native/canonical active joint-limit intersection is empty")
    source_home_tensor = torch.tensor(
        [source_home[name] for name in source_joint_names],
        device=robots["canonical"].device,
        dtype=robots["canonical"].data.joint_pos.dtype,
    )
    alternating_sign = torch.where(
        torch.arange(len(source_joint_names), device=source_home_tensor.device) % 2 == 0,
        1.0,
        -1.0,
    )
    active_step_target = torch.clamp(
        source_home_tensor + alternating_sign * float(ARGS.target_delta_rad),
        min=lower_intersection,
        max=upper_intersection,
    )  # ``[J]``，三侧共享且位于所有 active hard limits 内

    # q_home settle 只消除 importer/drive 初始化差异，不进入响应 trace。
    for _ in range(int(ARGS.settle_steps)):
        for mode, robot in robots.items():
            robot.set_joint_position_target(full_home_by_mode[mode])
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt_s)

    traces = {mode: _empty_trace_lists() for mode in robots}
    home_active_targets = {
        mode: full_home_by_mode[mode][0, joint_ids_by_mode[mode]].detach().clone() for mode in robots
    }
    if ARGS.phase == "scripted_contact":
        if objects_by_mode is None or set(objects_by_mode) != set(robots) or set(sensor_keys_by_mode) != set(robots):
            raise RuntimeError("scripted contact scene lacks one object/sensor per matched articulation")
        _initialize_scripted_contact_objects(
            robots,
            objects_by_mode,
            contact_body_ids_by_mode=contact_body_ids_by_mode,
            palm_body_ids_by_mode=palm_body_ids_by_mode,
        )
        active_targets_by_mode = home_active_targets  # contact probe 固定 q_home，不叠加主动 finger motion
        phase_name = "physical_audit_scripted_contact"
        applied_target = source_home_tensor
    else:
        active_targets_by_mode = {mode: active_step_target.clone() for mode in robots}
        phase_name = "physical_audit_no_contact_response"
        applied_target = active_step_target

    _capture_sample(
        traces,
        robots,
        joint_ids_by_mode=joint_ids_by_mode,
        tip_ids_by_mode=tip_ids_by_mode,
        active_targets_by_mode=active_targets_by_mode,
        scene=scene if objects_by_mode is not None else None,
        objects_by_mode=objects_by_mode,
        sensor_keys_by_mode=sensor_keys_by_mode if objects_by_mode is not None else None,
    )  # $t=0^-$，step input 或 object approach 前共同状态

    # Phase A 下发 joint target step；Phase B 则 hold q_home，让相同 incoming object 产生首次接触和响应。
    record_optional_rl_phase(
        phase_name,
        "start",
        response_steps=int(ARGS.response_steps),
        target_delta_rad=float(ARGS.target_delta_rad) if ARGS.phase == "no_contact" else 0.0,
    )
    for _ in range(int(ARGS.response_steps)):
        for mode, robot in robots.items():
            robot.set_joint_position_target(applied_target[None, :], joint_ids=joint_ids_by_mode[mode])
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt_s)
        _capture_sample(
            traces,
            robots,
            joint_ids_by_mode=joint_ids_by_mode,
            tip_ids_by_mode=tip_ids_by_mode,
            active_targets_by_mode=active_targets_by_mode,
            scene=scene if objects_by_mode is not None else None,
            objects_by_mode=objects_by_mode,
            sensor_keys_by_mode=sensor_keys_by_mode if objects_by_mode is not None else None,
        )
    record_optional_rl_phase(phase_name, "complete")

    stacked = _stack_traces(traces)
    comparison = compare_canonical_against_native_repeat(
        stacked["native_reference"],
        stacked["native_repeat"],
        stacked["canonical"],
        sample_dt_s=sim_dt_s,
        contact_threshold_N=0.25,
    )
    canonical_robot = robots["canonical"]
    canonical_active_names = set(canonical_joint_names)
    canonical_mask = torch.tensor(
        [name in canonical_active_names for name in canonical_robot.joint_names],
        device=canonical_robot.device,
        dtype=torch.bool,
    )  # 实际 PhysX joint order 下的 active/ghost selector
    ghost_joint_pos = canonical_robot.data.joint_pos[0, ~canonical_mask].abs()  # ``[G]``，G 可为 0
    ghost_joint_vel = canonical_robot.data.joint_vel[0, ~canonical_mask].abs()  # ``[G]``，rad/s
    ghost_target = canonical_robot.data.joint_pos_target[0, ~canonical_mask].abs()  # ``[G]``，rad
    ghost_joint_pos_max = float(ghost_joint_pos.max().item()) if ghost_joint_pos.numel() else 0.0
    ghost_joint_vel_max = float(ghost_joint_vel.max().item()) if ghost_joint_vel.numel() else 0.0
    ghost_target_max = float(ghost_target.max().item()) if ghost_target.numel() else 0.0
    active_limit_pair_error = max(
        float(torch.max(torch.abs(limits_by_mode["native_reference"] - limits_by_mode["native_repeat"])).item()),
        float(torch.max(torch.abs(limits_by_mode["native_reference"] - limits_by_mode["canonical"])).item()),
    )

    summary: dict[str, object] = {
        "schema_version": PHYSICAL_AUDIT_SCHEMA_VERSION,
        "phase": "scripted_contact" if ARGS.phase == "scripted_contact" else "no_contact_step_response",
        "asset_id": source_container.asset_id,
        "asset_path": str(resolve_bank_path(ARGS.asset_path)),
        "source_dof": len(source_joint_names),
        "source_joint_names": list(source_joint_names),
        "canonical_joint_names": list(canonical_joint_names),
        "native_tip_body_names": [robots["native_reference"].body_names[index] for index in tip_ids_by_mode["native_reference"]],
        "canonical_tip_body_names": [robots["canonical"].body_names[index] for index in tip_ids_by_mode["canonical"]],
        "sim_dt_s": sim_dt_s,
        "settle_steps": int(ARGS.settle_steps),
        "response_steps": int(ARGS.response_steps),
        "requested_target_delta_rad": float(ARGS.target_delta_rad),
        "ghost_actuator_mode": str(ARGS.ghost_actuator_mode),
        "ghost_lock_mode": str(ARGS.ghost_lock_mode),
        "ghost_limit_rad": float(ARGS.ghost_limit_rad),
        "actual_target_delta_rad": (applied_target - source_home_tensor).detach().cpu().tolist(),
        "active_limit_pair_abs_max_error_rad": active_limit_pair_error,
        "canonical_ghost_joint_pos_abs_max_rad": ghost_joint_pos_max,
        "canonical_ghost_joint_vel_abs_max_rad_s": ghost_joint_vel_max,
        "canonical_ghost_target_abs_max_rad": ghost_target_max,
        "comparison": comparison,
        "contact_start_offset_m": float(ARGS.contact_start_offset_m),
        "contact_approach_speed_m_s": float(ARGS.contact_approach_speed_m_s),
        "contact_cube_size_m": float(ARGS.contact_cube_size_m),
        "contact_role": str(ARGS.contact_role),
        "contact_role_index": int(ARGS.contact_role_index),
        "contact_body_names": contact_body_names_by_mode,
        "contact_force_peak_N": {
            mode: float(trace["contact_force_N"].max().item()) for mode, trace in stacked.items()
        },
        "interpretation_boundary": (
            "Reward arrays remain structural zeros; scripted-contact evidence covers one dynamic cube and one real tip."
            if ARGS.phase == "scripted_contact"
            else "Phase-A object/contact/reward arrays are structural zeros; only joint/tip/limit/ghost metrics are evidence."
        ),
    }
    _write_artifacts(ARGS.output_dir.expanduser().resolve(), traces=stacked, summary=summary)
    return summary


def main() -> int:
    r"""运行 probe 并确保 Kit teardown 不覆盖真实失败退出码。"""

    try:
        summary = _run()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()  # os._exit 前把完整错误交给 parent benchmark stderr
        return 1


if __name__ == "__main__":
    EXIT_CODE = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Kit ``close()`` 的异常退出钩子可能先终止解释器并返回 0；父 benchmark 负责进程组回收。
    os._exit(EXIT_CODE)  # 直接传播 Python assertion/exception，不让 teardown 覆盖真实退出码
