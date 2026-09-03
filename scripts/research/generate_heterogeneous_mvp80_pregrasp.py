r"""为80手MVP生成DexCube scale-1.1的Top-8 palm-supported good pregrasps。

每项资产使用32个并行环境，连续评估8批，共256个候选。候选joint configuration来自N000
canonical pregrasp、joint-limit midpoint及二者blend，再叠加确定性小扰动。Object center的$x/y$不是
跨手固定魔数，而由候选$q$下实际thumb TIP与两根non-thumb TIP形成的opposition midpoint推导；N000
hand-frame anchor只作为掌面内的保守收缩中心。Object orientation严格为hand-frame upright identity。

每个候选以$q_0=u_0$、零joint/object velocity走训练同物理配置的1 s cold reset。Hard acceptance只检查
关节余量、三指包络、穿透、位移、倾斜、速度尖峰与PALM support；TIP/JOINT contact只记录，不作为准入门。
每项资产按词典序物理质量保存Top-8，MVP runtime固定消费rank-0。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, cast

import yaml

ASSETS_PER_RUN = 80
CANDIDATES_PER_BATCH = 32
DEFAULT_BATCH_COUNT = 16
PHYSICS_STEPS = 120  # 1 s × 120 Hz
POLICY_SUBSTEPS = 6  # 20 Hz contact/diagnostic sampling
EARLY_PEAK_STEPS = 24  # 前0.2 s速度尖峰
TAIL_POLICY_SAMPLES = 10  # 最后0.5 s PALM support

N000_CANONICAL_Q = (
    0.0,
    0.0,
    0.0,
    0.88,
    -0.61000001,
    -0.12,
    0.56,
    1.73000002,
    1.05999994,
    1.17999995,
    1.51999998,
    0.71999997,
    0.93000001,
    0.57999998,
    0.44,
    1.63,
)  # depth-major index/middle/ring/thumb，rad

DEFAULT_SELECTION = Path(
    "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80_candidates.yaml"
)
DEFAULT_CATALOG = Path("outputs/pregrasp/catalogs/heterogeneous_rotation_mvp80_dexcube_s1p1_v4")
DEFAULT_EVIDENCE = Path("outputs/pregrasp/search/heterogeneous_rotation_mvp80_dexcube_s1p1_v4")


def _parse_args() -> argparse.Namespace:
    r"""解析selection、catalog、candidate批数与可复现seed。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--batch-count", type=int, default=DEFAULT_BATCH_COUNT)
    parser.add_argument(
        "--asset-limit",
        type=int,
        default=None,
        help="Development-only ordered prefix; formal generation leaves this unset and uses all 80 assets.",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.batch_count < 1:
        parser.error("--batch-count must be positive")
    if args.asset_limit is not None and not 1 <= args.asset_limit <= ASSETS_PER_RUN:
        parser.error("--asset-limit must lie in [1,80]")
    return args


ARGS = _parse_args()
SELECTION_DOCUMENT = yaml.safe_load(ARGS.selection.read_text(encoding="utf-8"))
FORMAL_SELECTED_ROWS = tuple(int(row) for row in SELECTION_DOCUMENT["initial_selected_rows"])
if len(FORMAL_SELECTED_ROWS) != ASSETS_PER_RUN or len(set(FORMAL_SELECTED_ROWS)) != ASSETS_PER_RUN:
    raise ValueError("MVP pregrasp generation requires exactly 80 unique initial selection rows")
SELECTED_ROWS = FORMAL_SELECTED_ROWS[: ARGS.asset_limit] if ARGS.asset_limit is not None else FORMAL_SELECTED_ROWS
ASSET_COUNT = len(SELECTED_ROWS)  # 正式为80；development smoke可取有序前缀
NUM_ENVS = ASSET_COUNT * CANDIDATES_PER_BATCH
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in SELECTED_ROWS)
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(NUM_ENVS)

from isaaclab.app import AppLauncher  # noqa: E402  # scene routing必须在task imports前冻结

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def _tip_positions_h(
    tip_positions_w,
    hand_position_w,
    hand_quaternion_w,
):
    r"""把四个TIP reference-link origins从world变换到hand semantic frame。

    Args:
        tip_positions_w: `[N,4,3]`，world frame TIP positions，单位m。
        hand_position_w: `[N,3]`，$p_{wh}$，单位m。
        hand_quaternion_w: `[N,4]`，$q_{wh}$，wxyz。

    Returns:
        Tensor: `[N,4,3]` hand-frame TIP positions，单位m。
    """

    import isaaclab.utils.math as math_utils

    offsets_w = tip_positions_w - hand_position_w.unsqueeze(1)  # TIP相对hand origin的world向量，m
    repeated_inverse = hand_quaternion_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
    return math_utils.quat_apply_inverse(repeated_inverse, offsets_w.reshape(-1, 3)).reshape(-1, 4, 3)


def _minimum_sector_degrees(vectors_xy):
    r"""计算三根finger相对object center的最小无向面内夹角。

    ``vectors_xy``形状为`[N,3,2]`。三个pair角度均位于$[0,180]$ degree；较大的minimum表示三根指头
    没有退化到同一窄sector。
    """

    import torch

    normalized = vectors_xy / torch.linalg.vector_norm(vectors_xy, dim=-1, keepdim=True).clamp_min(1.0e-8)
    pair_indices = ((0, 1), (0, 2), (1, 2))
    cosines = torch.stack(
        [torch.sum(normalized[:, left] * normalized[:, right], dim=-1) for left, right in pair_indices],
        dim=-1,
    )
    return torch.rad2deg(torch.acos(cosines.clamp(-1.0, 1.0))).amin(dim=-1)


def _derive_object_candidates(
    tip_positions_h,
    active_tip_mask,
    xy_noise,
    joint_margin,
):
    r"""为每个$q$选择thumb＋两根non-thumb包络并生成hand-specific object center。

    三个non-thumb pair为(index,middle)、(index,ring)、(middle,ring)。每个pair先形成
    $c_{opp}=\frac12(p_{thumb}+\frac12(p_a+p_b))$，再向N000掌内anchor收缩35%，并加入最多1 cm的
    deterministic XY扰动。Pair选择优先joint margin、minimum sector与较短TIP-center距离。

    Returns:
        tuple: object positions `[N,3]`、选中non-thumb indices `[N,2]`、三指距离`[N,3]`、sector`[N]`。
    """

    import torch

    pair_table = torch.tensor(((0, 1), (0, 2), (1, 2)), dtype=torch.long, device=tip_positions_h.device)
    default_xy = torch.tensor((0.00578, 0.08511), dtype=tip_positions_h.dtype, device=tip_positions_h.device)
    pair_centers: list[torch.Tensor] = []
    pair_distances: list[torch.Tensor] = []
    pair_sectors: list[torch.Tensor] = []
    pair_validity: list[torch.Tensor] = []
    for pair in pair_table:
        non_thumb = tip_positions_h[:, pair]  # `[N,2,3]` selected opposing fingertips
        thumb = tip_positions_h[:, 3:4]  # `[N,1,3]` canonical thumb TIP
        opposition_xy = 0.5 * (thumb[:, 0, :2] + non_thumb[:, :, :2].mean(dim=1))
        center_xy = 0.35 * opposition_xy + 0.65 * default_xy + xy_noise
        center_xy = torch.stack(
            (
                center_xy[:, 0].clamp(-0.035, 0.035),
                center_xy[:, 1].clamp(0.040, 0.105),
            ),
            dim=-1,
        )  # common palm support rectangle内的hand-specific center
        center = torch.cat(
            (center_xy, torch.full((center_xy.shape[0], 1), 0.054, device=center_xy.device)),
            dim=-1,
        )  # scale-1.1 DexCube center高度，hand frame m
        fingers = torch.cat((thumb, non_thumb), dim=1)  # `[N,3,3]` thumb first
        distances = torch.linalg.vector_norm(fingers - center.unsqueeze(1), dim=-1)  # center distance，m
        sector = _minimum_sector_degrees(fingers[:, :, :2] - center_xy.unsqueeze(1))
        valid = active_tip_mask[:, 3] & active_tip_mask[:, pair[0]] & active_tip_mask[:, pair[1]]
        pair_centers.append(center)
        pair_distances.append(distances)
        pair_sectors.append(sector)
        pair_validity.append(valid)

    centers = torch.stack(pair_centers, dim=1)  # `[N,3 pairs,3]`
    distances = torch.stack(pair_distances, dim=1)  # `[N,3 pairs,3 fingers]`
    sectors = torch.stack(pair_sectors, dim=1)  # `[N,3 pairs]`
    validity = torch.stack(pair_validity, dim=1)  # bool`[N,3 pairs]`
    scores = 2.0 * joint_margin.unsqueeze(-1) + sectors / 180.0 - distances.amax(dim=-1) / 0.1
    scores = torch.where(validity, scores, torch.full_like(scores, -torch.inf))
    if bool((~torch.isfinite(scores).any(dim=-1)).any().item()):
        raise RuntimeError("one selected MVP asset lacks thumb plus two active non-thumb fingertips")
    best = scores.argmax(dim=-1)  # 每environment最优finger pair
    rows = torch.arange(best.shape[0], device=best.device)
    return centers[rows, best], pair_table[best], distances[rows, best], sectors[rows, best]


def _canonical_candidate_q(lower, upper, active_mask, random_values, proposal_index):
    r"""从N000/midpoint/blend family构造$q_0=u_0$候选。

    每四个candidate循环一次N000/midpoint blend；随后对active joints加入$U(-0.1,0.1)$ rad扰动并clip。
    Ghost joints最终严格为0。
    """

    import torch

    midpoint = 0.5 * (lower + upper)  # `[N,16]` soft-limit midpoint，rad
    template = torch.tensor(N000_CANONICAL_Q, dtype=lower.dtype, device=lower.device).expand_as(lower)
    template = torch.maximum(torch.minimum(template, upper), lower)  # 按每资产limits裁剪N000 role seed
    blend_weights = torch.tensor((0.55, 0.70, 0.85, 1.0), dtype=lower.dtype, device=lower.device)
    blend = blend_weights[proposal_index % 4].unsqueeze(-1)  # N000 template在candidate中的权重$w$
    q = blend * template + (1.0 - blend) * midpoint
    q += (random_values - 0.5) * 0.2 * active_mask  # $\delta q\in[-0.1,0.1]$ rad
    q = torch.maximum(torch.minimum(q, upper), lower) * active_mask
    span = (upper - lower).clamp_min(1.0e-6)
    margin = torch.minimum((q - lower) / span, (upper - q) / span)
    margin = torch.where(active_mask.bool(), margin, torch.inf).amin(dim=-1)  # 最近active limit的归一化余量
    return q, margin


def _physics_identity_digest(*, object_scale: float, cube_sha256: str) -> str:
    r"""核对generator实际object identity并返回runtime共享physics摘要。"""

    from anymani.tasks.hetero.config.generated.good_pregrasp_identity_v4 import (
        GOOD_PREGRASP_OBJECT_SCALE,
        GOOD_PREGRASP_PHYSICS_DIGEST,
    )
    from anymani.tasks.hetero.config.generated.pregrasp_identity import DEX_CUBE_SHA256

    if object_scale != GOOD_PREGRASP_OBJECT_SCALE or cube_sha256 != DEX_CUBE_SHA256:
        raise ValueError("generator object scale/bytes disagree with good-pregrasp runtime identity")
    return GOOD_PREGRASP_PHYSICS_DIGEST


def _generation_identity_digest(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    r"""核对CLI预算并读取generator/runtime共同拥有的v4数值协议。"""

    from anymani.tasks.hetero.config.generated.good_pregrasp_identity_v4 import (
        GOOD_PREGRASP_CANDIDATE_COUNT,
        GOOD_PREGRASP_GENERATION_DIGEST,
        GOOD_PREGRASP_GENERATION_IDENTITY,
        GOOD_PREGRASP_SEED,
    )

    if args.seed != GOOD_PREGRASP_SEED:
        raise ValueError("formal good-pregrasp generation requires the shared deterministic seed")
    if args.batch_count * CANDIDATES_PER_BATCH != GOOD_PREGRASP_CANDIDATE_COUNT:
        raise ValueError("formal good-pregrasp generation requires the shared candidate budget")
    return GOOD_PREGRASP_GENERATION_DIGEST, dict(GOOD_PREGRASP_GENERATION_IDENTITY)


def main() -> int:
    r"""运行8批cold-reset search、发布逐资产Top-8并保存压缩候选证据。"""

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    import numpy as np
    import torch
    from anymani.pregrasp import active_mask_digest
    from anymani.pregrasp.good_catalog import (
        GoodPregraspCandidate,
        GoodPregraspCatalog,
        GoodPregraspEntry,
        GoodPregraspKey,
        GoodPregraspMember,
        GoodPregraspMetrics,
    )
    from anymani.pregrasp.isaac_runtime import (
        contact_penetration_depth_per_env,
        hand_semantic_pose_w,
        object_pose_h_from_world,
        object_pose_w_from_hand,
    )
    from anymani.tasks.hetero.config.generated.pregrasp_harness_env_cfg import GeneratedPregraspHarnessEnvCfg
    from anymani.tasks.hetero.config.generated.scene import (
        ASSET_BINDING,
        CONTACT_LAYOUT,
        RESOLVED_DEX_CUBE_SHA256,
    )
    from anymani.tasks.hetero.contact_sensors import sensor_contact_magnitude
    from anymani.tasks.hetero.mdp.runtime_state import derive_tip_and_owner_masks
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

    if ASSET_BINDING.dataset_rows != SELECTED_ROWS or ASSET_BINDING.asset_count != ASSET_COUNT:
        raise RuntimeError("scene asset binding disagrees with MVP80 selection")
    generation_digest, generation_identity = _generation_identity_digest(ARGS)
    physics_digest = _physics_identity_digest(object_scale=1.1, cube_sha256=RESOLVED_DEX_CUBE_SHA256)
    device = torch.device("cuda:0")
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)

    # Scene使用80 assets round-robin，每项32 replicas；object scale必须在PhysX startup前覆盖为1.1。
    cfg = GeneratedPregraspHarnessEnvCfg()
    cfg.seed = ARGS.seed
    object_spawn = cast(sim_utils.UsdFileCfg, cfg.scene.object.spawn)
    object_spawn.scale = (1.1, 1.1, 1.1)
    runtime_env = ManagerBasedRLEnv(cfg=cfg)
    runtime_env.sim._app_control_on_stop_handle = None
    runtime_env.reset()
    robot = cast(Articulation, runtime_env.scene["robot"])
    object_asset = cast(RigidObject, runtime_env.scene["object"])
    tip_body_ids, _ = robot.find_bodies(list(CONTACT_LAYOUT.fingertip_links), preserve_order=True)
    if len(tip_body_ids) != 4:
        raise RuntimeError("good-pregrasp generator requires four canonical TIP body slots")
    sensors = {
        name: cast(ContactSensor, runtime_env.scene[name])
        for name in CONTACT_LAYOUT.state_sensor_names
    }
    sensor_owner_indices = torch.tensor(CONTACT_LAYOUT.sensor_owner_indices, dtype=torch.long, device=device)
    active_mask = torch.tensor(ASSET_BINDING.active_joint_mask_by_env(NUM_ENVS), dtype=torch.bool, device=device)
    active_float = active_mask.to(dtype=torch.float32)
    active_tip_mask, active_owner_mask = derive_tip_and_owner_masks(active_mask)
    env_ids = torch.arange(NUM_ENVS, dtype=torch.long, device=device)
    local_asset = env_ids % ASSET_COUNT  # round-robin physical prototype index
    candidate_slot = env_ids // ASSET_COUNT  # 0..31 within current batch
    limits = robot.data.soft_joint_pos_limits
    lower, upper = limits[..., 0], limits[..., 1]

    # 全256候选的随机源一次生成，batch切分不改变$q/xy$序列。
    total_candidates = ARGS.batch_count * CANDIDATES_PER_BATCH
    random_bank = torch.stack(
        tuple(
            torch.rand(
                total_candidates,
                18,
                generator=torch.Generator(device=device).manual_seed(ARGS.seed + int(dataset_row) * 104729),
                device=device,
            )
            for dataset_row in SELECTED_ROWS
        ),
        dim=0,
    )  # 每资产随机流只依赖formal row；cohort顺序/大小不改变16 joint + 2 XY候选

    # CPU evidence tensors按[asset,candidate,...]保存；GPU只保留当前32 replicas/asset的物理状态。
    q_bank = torch.zeros(ASSET_COUNT, total_candidates, 16, dtype=torch.float32)
    position_bank = torch.zeros(ASSET_COUNT, total_candidates, 3, dtype=torch.float32)
    pair_bank = torch.zeros(ASSET_COUNT, total_candidates, 2, dtype=torch.int64)
    distance_bank = torch.zeros(ASSET_COUNT, total_candidates, 3, dtype=torch.float32)
    metric_names = (
        "joint_margin",
        "sector_deg",
        "penetration_m",
        "displacement_m",
        "tilt_deg",
        "peak_linear_m_s",
        "peak_off_axis_angular_rad_s",
        "palm_fraction",
    )
    metrics_bank = {name: torch.zeros(ASSET_COUNT, total_candidates) for name in metric_names}
    owner_contact_bank = torch.zeros(ASSET_COUNT, total_candidates, 21)
    passed_bank = torch.zeros(ASSET_COUNT, total_candidates, dtype=torch.bool)

    frame = ASSET_BINDING.hand_spawn_cfg.frame
    policy_sample_count = PHYSICS_STEPS // POLICY_SUBSTEPS
    for batch_index in range(ARGS.batch_count):
        global_candidate = candidate_slot + batch_index * CANDIDATES_PER_BATCH  # `[N]` 0..255 per asset
        random_values = random_bank[local_asset, global_candidate]
        q, joint_margin = _canonical_candidate_q(
            lower,
            upper,
            active_float,
            random_values[:, :16],
            global_candidate,
        )

        # 先把object移出手部并写入$q$，推进一个physics step以刷新真实TIP body transforms。
        far_pose = object_asset.data.default_root_state[:, :7].clone()
        far_pose[:, :3] = runtime_env.scene.env_origins + torch.tensor((0.0, 0.0, 1.5), device=device)
        far_pose[:, 3] = 1.0
        far_pose[:, 4:7] = 0.0
        object_asset.write_root_pose_to_sim(far_pose)
        object_asset.write_root_velocity_to_sim(torch.zeros(NUM_ENVS, 6, device=device))
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        runtime_env.scene.write_data_to_sim()
        runtime_env.sim.step(render=False)
        runtime_env.scene.update(runtime_env.physics_dt)

        # 当前$q$下的真实TIP positions决定每只hand自己的opposition center与有效non-thumb pair。
        hand_pos_w, hand_quat_w = hand_semantic_pose_w(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            frame.semantic_R_ha,
            frame.semantic_p_ha,
        )
        tips_h = _tip_positions_h(robot.data.body_pos_w[:, tip_body_ids], hand_pos_w, hand_quat_w)
        xy_noise = (random_values[:, 16:18] - 0.5) * 0.01  # $[-0.5,0.5]$ cm hand-frame扰动
        object_pos_h, selected_pair, tip_distances, sector_deg = _derive_object_candidates(
            tips_h,
            active_tip_mask,
            xy_noise,
            joint_margin,
        )
        object_quat_h = torch.zeros(NUM_ENVS, 4, device=device)
        object_quat_h[:, 0] = 1.0  # exact upright identity quaternion
        object_pos_w, object_quat_w = object_pose_w_from_hand(
            hand_pos_w,
            hand_quat_w,
            object_pos_h,
            object_quat_h,
        )

        # 重新写入candidate exact cold state；$q_0=u_0$且所有速度为零。
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        object_asset.write_root_pose_to_sim(torch.cat((object_pos_w, object_quat_w), dim=-1))
        object_asset.write_root_velocity_to_sim(torch.zeros(NUM_ENVS, 6, device=device))
        initial_pos_w = object_pos_w.clone()

        displacement_max = torch.zeros(NUM_ENVS, device=device)
        tilt_max = torch.zeros(NUM_ENVS, device=device)
        peak_linear = torch.zeros(NUM_ENVS, device=device)
        peak_angular = torch.zeros(NUM_ENVS, device=device)
        penetration_max = torch.zeros(NUM_ENVS, device=device)
        force_ema = torch.zeros(NUM_ENVS, len(CONTACT_LAYOUT.state_sensor_names), device=device)
        owner_contact_count = torch.zeros(NUM_ENVS, 21, device=device)
        palm_tail_count = torch.zeros(NUM_ENVS, device=device)
        for physics_step in range(PHYSICS_STEPS):
            robot.set_joint_position_target(q)  # 所有120 Hz substeps保持同一$q_0$ target
            runtime_env.scene.write_data_to_sim()
            runtime_env.sim.step(render=False)
            runtime_env.scene.update(runtime_env.physics_dt)
            displacement = torch.linalg.vector_norm(object_asset.data.root_pos_w - initial_pos_w, dim=-1)
            displacement_max = torch.maximum(displacement_max, displacement)
            _, object_quat_current_h = object_pose_h_from_world(
                hand_pos_w,
                hand_quat_w,
                object_asset.data.root_pos_w,
                object_asset.data.root_quat_w,
            )
            object_z_h = math_utils.quat_apply(
                object_quat_current_h,
                torch.tensor((0.0, 0.0, 1.0), device=device).expand(NUM_ENVS, -1),
            )
            tilt = torch.rad2deg(torch.acos(object_z_h[:, 2].clamp(-1.0, 1.0)))
            tilt_max = torch.maximum(tilt_max, tilt)
            if physics_step < EARLY_PEAK_STEPS:
                peak_linear = torch.maximum(
                    peak_linear,
                    torch.linalg.vector_norm(object_asset.data.root_lin_vel_w, dim=-1),
                )
                angular_velocity_h = math_utils.quat_apply_inverse(
                    hand_quat_w,
                    object_asset.data.root_ang_vel_w,
                )  # hand-frame object angular velocity$[N,3]$，rad/s
                peak_angular = torch.maximum(
                    peak_angular,
                    torch.linalg.vector_norm(angular_velocity_h[:, :2], dim=-1),
                )  # 只限制会导致倾倒/弹飞的off-axis分量；目标轴yaw不进入hard gate
            if (physics_step + 1) % POLICY_SUBSTEPS:
                continue

            # N000 contact语义：每个sensor先求object-filtered force magnitude，再在20 Hz做EMA和0.25 N门。
            raw_force = torch.stack(
                [sensor_contact_magnitude(runtime_env, name) for name in CONTACT_LAYOUT.state_sensor_names],
                dim=-1,
            )
            force_ema = 0.5 * raw_force + 0.5 * force_ema
            sensor_bits = force_ema > 0.25
            owner_bits = torch.zeros(NUM_ENVS, 21, dtype=torch.int64, device=device)
            owner_bits.scatter_reduce_(
                1,
                sensor_owner_indices.reshape(1, -1).expand(NUM_ENVS, -1),
                sensor_bits.to(dtype=torch.int64),
                reduce="amax",
                include_self=True,
            )
            owner_contact_count += owner_bits.to(dtype=torch.float32) * active_owner_mask
            policy_index = (physics_step + 1) // POLICY_SUBSTEPS - 1
            if policy_index >= policy_sample_count - TAIL_POLICY_SAMPLES:
                palm_tail_count += sensor_bits[:, -1].to(dtype=torch.float32)  # dedicated PALM sensor，不含root owners
            for sensor in sensors.values():
                penetration_max = torch.maximum(
                    penetration_max,
                    contact_penetration_depth_per_env(sensor, runtime_env.physics_dt),
                )

        owner_fraction = owner_contact_count / float(policy_sample_count)
        palm_fraction = palm_tail_count / float(TAIL_POLICY_SAMPLES)
        passed = (
            (tip_distances.amax(dim=-1) <= 0.125)
            & (sector_deg >= 10.0)
            & (penetration_max <= 0.0005)
            & (displacement_max <= 0.015)
            & (tilt_max <= 10.0)
            & (peak_linear <= 0.5)
            & (peak_angular <= 8.0)
            & (palm_fraction >= 0.5)
        )

        # Scatter回[asset,candidate] CPU evidence轴；local_asset/candidate_slot构成双射。
        asset_cpu = local_asset.cpu()
        candidate_cpu = global_candidate.cpu()
        q_bank[asset_cpu, candidate_cpu] = q.detach().cpu()
        position_bank[asset_cpu, candidate_cpu] = object_pos_h.detach().cpu()
        pair_bank[asset_cpu, candidate_cpu] = selected_pair.detach().cpu()
        distance_bank[asset_cpu, candidate_cpu] = tip_distances.detach().cpu()
        metrics_bank["joint_margin"][asset_cpu, candidate_cpu] = joint_margin.detach().cpu()
        metrics_bank["sector_deg"][asset_cpu, candidate_cpu] = sector_deg.detach().cpu()
        metrics_bank["penetration_m"][asset_cpu, candidate_cpu] = penetration_max.detach().cpu()
        metrics_bank["displacement_m"][asset_cpu, candidate_cpu] = displacement_max.detach().cpu()
        metrics_bank["tilt_deg"][asset_cpu, candidate_cpu] = tilt_max.detach().cpu()
        metrics_bank["peak_linear_m_s"][asset_cpu, candidate_cpu] = peak_linear.detach().cpu()
        metrics_bank["peak_off_axis_angular_rad_s"][asset_cpu, candidate_cpu] = peak_angular.detach().cpu()
        metrics_bank["palm_fraction"][asset_cpu, candidate_cpu] = palm_fraction.detach().cpu()
        owner_contact_bank[asset_cpu, candidate_cpu] = owner_fraction.detach().cpu()
        passed_bank[asset_cpu, candidate_cpu] = passed.detach().cpu()
        print(
            {
                "batch": batch_index,
                "candidate_range": [batch_index * 32, (batch_index + 1) * 32 - 1],
                "passed": int(passed.sum().item()),
                "total": NUM_ENVS,
            },
            flush=True,
        )

    # 压缩保存所有候选数值，后续阈值/排序分析无需重跑PhysX。
    evidence_root = ARGS.evidence.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        evidence_root / "candidates.npz",
        dataset_rows=np.asarray(SELECTED_ROWS, dtype=np.int64),
        q_state_rad=q_bank.numpy(),
        object_position_h_m=position_bank.numpy(),
        envelope_pair_indices=pair_bank.numpy(),
        envelope_tip_center_distance_m=distance_bank.numpy(),
        passed=passed_bank.numpy(),
        owner_contact_fraction=owner_contact_bank.numpy(),
        **{name: values.numpy() for name, values in metrics_bank.items()},
    )

    # 每资产按hard pass后词典序质量选择Top-8，并发布schema-3 catalog payload。
    catalog = GoodPregraspCatalog(ARGS.catalog.resolve())
    finger_names = ("index", "middle", "ring", "thumb")
    published: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for asset_index, dataset_row in enumerate(SELECTED_ROWS):
        passed_indices = torch.nonzero(passed_bank[asset_index], as_tuple=False).flatten().tolist()

        def candidate_score(candidate_index: int) -> tuple[float, ...]:
            r"""Lexicographic Top-8 score：舒适/包络优先，随后最小动态瞬态。"""

            return (
                float(metrics_bank["palm_fraction"][asset_index, candidate_index]),
                -float(metrics_bank["displacement_m"][asset_index, candidate_index]),
                -float(metrics_bank["tilt_deg"][asset_index, candidate_index]) / 10.0,
                -float(metrics_bank["peak_linear_m_s"][asset_index, candidate_index]),
                -float(metrics_bank["peak_off_axis_angular_rad_s"][asset_index, candidate_index]) / 8.0,
                -float(metrics_bank["penetration_m"][asset_index, candidate_index]),
                float(metrics_bank["joint_margin"][asset_index, candidate_index]),
                float(metrics_bank["sector_deg"][asset_index, candidate_index]) / 180.0,
                -float(distance_bank[asset_index, candidate_index].max()),
            )

        ranked = sorted(passed_indices, key=candidate_score, reverse=True)
        if len(ranked) < 8:
            failed.append(
                {
                    "dataset_row": dataset_row,
                    "asset_id": ASSET_BINDING.source_assets[asset_index].asset_id,
                    "passed_candidates": len(ranked),
                }
            )
            continue
        artifact = ASSET_BINDING.canonical_artifacts[asset_index]
        source_asset = ASSET_BINDING.source_assets[asset_index]
        members = []
        for rank, candidate_index in enumerate(ranked[:8]):
            pair = pair_bank[asset_index, candidate_index].tolist()
            envelope_fingers = ("thumb", finger_names[pair[0]], finger_names[pair[1]])
            candidate_position = position_bank[asset_index, candidate_index].tolist()
            candidate_distances = distance_bank[asset_index, candidate_index].tolist()
            candidate = GoodPregraspCandidate(
                q_state_rad=tuple(float(value) for value in q_bank[asset_index, candidate_index].tolist()),
                q_target_rad=tuple(float(value) for value in q_bank[asset_index, candidate_index].tolist()),
                active_joint_mask=tuple(bool(value) for value in ASSET_BINDING.active_joint_masks[asset_index]),
                object_position_h_m=(
                    float(candidate_position[0]),
                    float(candidate_position[1]),
                    float(candidate_position[2]),
                ),
            )
            metrics = GoodPregraspMetrics(
                joint_limit_margin_fraction=float(metrics_bank["joint_margin"][asset_index, candidate_index]),
                envelope_fingers=envelope_fingers,
                envelope_sector_min_deg=float(metrics_bank["sector_deg"][asset_index, candidate_index]),
                envelope_tip_center_distance_m=(
                    float(candidate_distances[0]),
                    float(candidate_distances[1]),
                    float(candidate_distances[2]),
                ),
                penetration_depth_max_m=float(metrics_bank["penetration_m"][asset_index, candidate_index]),
                object_displacement_max_m=float(metrics_bank["displacement_m"][asset_index, candidate_index]),
                object_tilt_max_deg=float(metrics_bank["tilt_deg"][asset_index, candidate_index]),
                peak_linear_velocity_m_s=float(metrics_bank["peak_linear_m_s"][asset_index, candidate_index]),
                peak_off_axis_angular_velocity_rad_s=float(
                    metrics_bank["peak_off_axis_angular_rad_s"][asset_index, candidate_index]
                ),
                palm_contact_fraction=float(metrics_bank["palm_fraction"][asset_index, candidate_index]),
                owner_contact_fraction=tuple(
                    float(value) for value in owner_contact_bank[asset_index, candidate_index].tolist()
                ),
            )
            members.append(
                GoodPregraspMember(
                    rank=rank,
                    candidate=candidate,
                    metrics=metrics,
                    selection_score=candidate_score(candidate_index),
                )
            )
        key = GoodPregraspKey(
            asset_id=source_asset.asset_id,
            source_content_hash=artifact.source_content_hash,
            physical_geometry_hash=artifact.physical_geometry_hash,
            canonical_schema_digest=artifact.schema_digest,
            routing_digest=active_mask_digest(artifact.routing.active_joint_mask),
            object_asset_id="DexCube",
            object_asset_sha256=RESOLVED_DEX_CUBE_SHA256,
            object_scale=1.1,
            physics_identity_digest=physics_digest,
            generation_identity_digest=generation_digest,
        )
        index_entry = catalog.publish(GoodPregraspEntry(key=key, members=tuple(members)))
        published.append(
            {
                "dataset_row": dataset_row,
                "asset_id": source_asset.asset_id,
                "passed_candidates": len(ranked),
                "key_digest": index_entry.key_digest,
                "entry_digest": index_entry.entry_digest,
            }
        )

    summary = {
        "artifact_type": "anymani.good_pregrasp.generation_summary",
        "schema_version": "1.0.0",
        "selection_path": str(ARGS.selection),
        "dataset_rows": list(SELECTED_ROWS),
        "object": {"asset_id": "DexCube", "scale": 1.1, "orientation_h_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "generation_identity": generation_identity,
        "generation_identity_digest": generation_digest,
        "physics_identity_digest": physics_digest,
        "candidate_count_per_asset": total_candidates,
        "published_count": len(published),
        "failed_count": len(failed),
        "published": published,
        "failed": failed,
        "catalog_root": str(ARGS.catalog.resolve()),
        "candidate_npz": str((evidence_root / "candidates.npz").resolve()),
    }
    summary_path = evidence_root / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(
        {
            "summary": str(summary_path),
            "published": len(published),
            "failed": len(failed),
        },
        flush=True,
    )
    runtime_env.close()
    return 0 if not failed else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
