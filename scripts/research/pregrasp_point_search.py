r"""一个或多个heterogeneous hand在一个absolute scale下搜索/认证pregrasp points。

Grid/random/CEM产生nominal proposals；verify分别恢复actual joint state与PD preload target；basin从可复现中心点
施加$\delta q_s/\delta T_{ho}$/object twist扰动。每个trial固定PD hold 6 s，并以20 Hz记录TIP/palm/finger-
non-tip contact、object drift/velocity/orientation、penetration、joint margin、tracking与effort。

输出保存全部candidate摘要与每资产最佳``coverage=point``的 :class:`PregraspRecord`。Basin portfolio只形成
local-perturbation充分统计；仍须结合独立scale stress构造 :class:`ScaleCertificate`后才能写production cache。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any, cast

Q_PROPOSAL_NAMES = (
    "zero",
    "n000_template",
    "limit_midpoint",
    "limit_quarter",
    "limit_three_quarter",
    "template_plus_0p2",
    "template_minus_0p2",
    "template_distal_plus_0p35",
)  # 先保留互不相同的保守proposal families，后续可按failure evidence扩展

OBJECT_OFFSETS_H_M = (
    (0.0, 0.0, 0.0),
    (-0.02, 0.0, 0.0),
    (0.02, 0.0, 0.0),
    (0.0, -0.02, 0.0),
    (0.0, 0.02, 0.0),
    (0.0, 0.0, -0.015),
    (0.0, 0.0, 0.015),
)  # 全部位于hand semantic frame，单位m

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
)  # N000 manual pregrasp映射到canonical depth-major index/middle/ring/thumb，单位rad


def _parse_args() -> argparse.Namespace:
    r"""解析单scale point-search身份与输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, choices=(1.1, 1.2, 1.25), required=True)
    parser.add_argument("--rows", type=str, default="0,16", help="One or more distinct formal dataset rows.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--portfolio", choices=("grid", "random", "refine", "verify", "cem", "basin"), default="grid"
    )
    parser.add_argument("--random-candidates", type=int, default=128)
    parser.add_argument("--seed-artifact", type=Path, default=None, help="Point-search artifact for refine portfolio.")
    parser.add_argument(
        "--seed-frontier",
        choices=("gate", "contact", "support", "selected"),
        default="gate",
        help="Saved per-asset candidate list used by verify/basin portfolios.",
    )
    parser.add_argument("--refine-q-radius", type=float, default=0.1)
    parser.add_argument("--refine-position-radius", type=float, default=0.01)
    parser.add_argument("--refine-yaw-deg", type=float, default=15.0)
    parser.add_argument("--basin-linear-velocity-radius", type=float, default=0.01)
    parser.add_argument("--basin-angular-velocity-radius", type=float, default=0.1)
    parser.add_argument(
        "--basin-min-tier",
        choices=("support_basin", "contact_basin"),
        default="contact_basin",
        help="Complete point tier counted as a successful local perturbation trial.",
    )
    parser.add_argument("--elite-count", type=int, default=16)
    parser.add_argument("--cem-std-scale", type=float, default=1.0)
    return parser.parse_args()


def _build_gate():
    r"""构造进入search identity的首轮显式认证门。"""

    from anymani.tasks.hetero.config.generated.pregrasp_identity import FORMAL_PREGRASP_GATE

    return FORMAL_PREGRASP_GATE


def _point_metric_dict(metrics: Any) -> dict[str, Any]:
    r"""给selection report增加最常用排序字段，完整schema仍保存在record中。"""

    return {
        "tip_ge_2_fraction": metrics.tip_ge_2_fraction,
        "tip_ge_3_fraction": metrics.tip_ge_3_fraction,
        "tip_active_count_mean": metrics.tip_active_count_mean,
        "finger_non_tip_occupancy_fraction": metrics.finger_non_tip_occupancy_fraction,
        "palm_occupancy_fraction": metrics.palm_occupancy_fraction,
        "object_anchor_distance_max_m": metrics.object_anchor_distance_max_m,
        "penetration_depth_max_m": metrics.penetration_depth_max_m,
        "joint_limit_margin_min_rad": metrics.joint_limit_margin_min_rad,
    }


def _seed_q_target(record: dict[str, Any]) -> list[float]:
    r"""读取proposal/controller target；旧point-search文档仅作为本轮CEM seed使用。

    Production schema-2.1不会走这个兼容分支，provider仍会严格拒绝缺少``q_target_rad``的payload。
    """

    candidate = record["candidate"]  # 研究artifact内嵌的candidate document
    values = candidate["q_target_rad"] if "q_target_rad" in candidate else candidate["q_rad"]
    return list(values)  # 2.0分析seed只有单一q坐标


def _seed_q_state(record: dict[str, Any]) -> list[float]:
    r"""读取actual reset state；只对当前CEM分析seed兼容schema-2.0的``q_rad``。"""

    candidate = record["candidate"]  # 研究artifact内嵌的candidate document
    values = candidate["q_state_rad"] if "q_state_rad" in candidate else candidate["q_rad"]
    return list(values)  # 2.1后actual与target不再混用


def _stored_point_gate_rank(point: dict[str, Any], gate: Any) -> tuple[float, ...]:
    r"""由已保存point summary恢复“contact优先、门距最小”的refinement rank。"""

    metrics = point["record"]["metrics"]
    gate_distance = (
        max(0.0, gate.min_tip_ge_2_fraction - float(metrics["tip_ge_2_fraction"]))
        + 10.0
        * max(
            0.0,
            float(metrics["finger_non_tip_occupancy_fraction"]) - gate.max_finger_non_tip_fraction,
        )
        + max(0.0, float(metrics["object_anchor_distance_max_m"]) / gate.max_anchor_distance_m - 1.0)
        + max(
            0.0,
            float(metrics["object_angular_velocity_rms_rad_s"]) / gate.max_angular_velocity_rms_rad_s - 1.0,
        )
        + max(
            0.0,
            float(metrics["object_orientation_drift_max_rad"]) / gate.max_object_orientation_drift_rad - 1.0,
        )
        + max(
            0.0,
            float(metrics["target_tracking_error_rms_rad"]) / gate.max_target_tracking_error_rms_rad - 1.0,
        )
    )
    return (
        float(float(metrics["tip_ge_2_fraction"]) >= gate.min_tip_ge_2_fraction),
        -float(len(point["reason_codes"])),
        -gate_distance,
        float(metrics["tip_ge_2_fraction"]),
    )


def _cem_score(point: dict[str, Any], gate: Any) -> tuple[float, ...]:
    r"""按contact可行性优先的词典序选择CEM elites。

    第一轴先保留满足2-TIP persistence的候选，第二轴再减少non-tip；只有在相同contact层内才比较
    support物理门距。这样CEM不会因non-tip惩罚较大而回退到零接触support点。
    """

    metrics = point["record"]["metrics"]
    physical_violation = max(
        0.0, float(metrics["object_anchor_distance_max_m"]) / gate.max_anchor_distance_m - 1.0
    )
    physical_violation += max(
        0.0,
        float(metrics["object_angular_velocity_rms_rad_s"]) / gate.max_angular_velocity_rms_rad_s - 1.0,
    )
    physical_violation += max(
        0.0,
        float(metrics["object_orientation_drift_max_rad"]) / gate.max_object_orientation_drift_rad - 1.0,
    )
    physical_violation += max(
        0.0,
        float(metrics["target_tracking_error_rms_rad"]) / gate.max_target_tracking_error_rms_rad - 1.0,
    )
    physical_violation += max(
        0.0,
        float(metrics["penetration_depth_max_m"]) / gate.max_penetration_depth_m - 1.0,
    )
    if bool(metrics["dropped"]):
        physical_violation += 10.0
    return (
        float(float(metrics["tip_ge_2_fraction"]) >= gate.min_tip_ge_2_fraction),
        -float(metrics["finger_non_tip_occupancy_fraction"]),
        -physical_violation,
        float(metrics["tip_ge_2_fraction"]),
        float(metrics["tip_active_count_mean"]),
    )


def main() -> int:
    r"""并行评估112个point并为两只hand输出最佳schema-2 point record。"""

    args = _parse_args()
    rows = tuple(int(item.strip()) for item in args.rows.split(",") if item.strip())
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("--rows must contain one or more distinct dataset rows")
    os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in rows)
    refinement_records: dict[int, dict[str, Any]] = {}
    cem_points: dict[int, list[dict[str, Any]]] = {}
    if args.portfolio in {"refine", "verify", "cem", "basin"}:
        if args.seed_artifact is None:
            raise ValueError("refine/verify/cem/basin portfolio requires --seed-artifact")
        seed_document = json.loads(args.seed_artifact.read_text())
        if args.portfolio == "cem":
            if args.elite_count < 2 or args.cem_std_scale <= 0.0:
                raise ValueError("CEM requires elite-count>=2 and positive std scale")
            seed_gate = _build_gate()
            for row in rows:
                row_points = [point for point in seed_document["points"] if int(point["dataset_row"]) == row]
                if len(row_points) < args.elite_count:
                    raise ValueError("CEM seed artifact has fewer points than elite-count")
                cem_points[row] = sorted(row_points, key=lambda point: _cem_score(point, seed_gate), reverse=True)[
                    : args.elite_count
                ]
        else:
            source_key = (
                "contact_frontier"
                if args.portfolio == "refine" and args.seed_frontier != "support"
                else {
                    "gate": "gate_frontier",
                    "contact": "contact_frontier",
                    "support": "support_frontier",
                    "selected": "selected",
                }[args.seed_frontier]
            )
            if source_key in seed_document:
                refinement_records = {
                    int(item["dataset_row"]): dict(item["record"])
                    for item in seed_document[source_key]
                    if int(item["dataset_row"]) in rows
                }
            elif args.seed_frontier == "support":
                refinement_records = {
                    row: dict(
                        max(
                            (
                                point
                                for point in seed_document["points"]
                                if int(point["dataset_row"]) == row and point["tier"] != "rejected"
                            ),
                            key=lambda point: (
                                float(point["record"]["metrics"]["joint_limit_margin_min_rad"]),
                                -float(point["record"]["metrics"]["target_tracking_error_rms_rad"]),
                                -float(point["record"]["metrics"]["object_anchor_distance_max_m"]),
                            ),
                        )["record"]
                    )
                    for row in rows
                }
            else:
                # 早期point-search artifact尚无gate_frontier字段时，从完整strict records确定性恢复。
                seed_gate = _build_gate()
                refinement_records = {
                    row: dict(
                        max(
                            (point for point in seed_document["points"] if int(point["dataset_row"]) == row),
                            key=lambda point: _stored_point_gate_rank(point, seed_gate),
                        )["record"]
                    )
                    for row in rows
                }
            if set(refinement_records) != set(rows):
                raise ValueError("refinement artifact must provide one contact frontier for every requested row")
            if min(args.refine_q_radius, args.refine_position_radius, args.refine_yaw_deg) <= 0.0:
                raise ValueError("refinement radii must be positive")
            if args.portfolio == "basin" and min(
                args.basin_linear_velocity_radius, args.basin_angular_velocity_radius
            ) <= 0.0:
                raise ValueError("basin velocity radii must be positive")

    # Scene routing与ghost mask在task config import时冻结，因此先确定完整round-robin环境轴。
    asset_count = len(rows)  # 每个requested physical asset拥有独立prototype
    point_count_per_asset = (
        len(Q_PROPOSAL_NAMES) * len(OBJECT_OFFSETS_H_M)
        if args.portfolio == "grid"
        else (1 if args.portfolio == "verify" else int(args.random_candidates))
    )
    if point_count_per_asset < 1:
        raise ValueError("candidate count per asset must be positive")
    num_envs = asset_count * point_count_per_asset  # MultiAssetSpawner按$e\bmod A$路由candidate replicas
    os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(num_envs)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    env = None
    try:
        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        import torch
        from anymani.pregrasp import (
            PregraspCandidate,
            PregraspCoverage,
            PregraspLookupKey,
            PregraspMetrics,
            active_mask_digest,
            certify_pregrasp,
        )
        from anymani.pregrasp.isaac_runtime import (
            contact_penetration_depth_per_env,
            file_sha256,
            hand_semantic_pose_w,
            object_pose_h_from_world,
            object_pose_w_from_hand,
        )
        from anymani.tasks.hetero.config.generated.pregrasp_harness_env_cfg import (
            GeneratedPregraspHarnessEnvCfg,
        )
        from anymani.tasks.hetero.config.generated.pregrasp_identity import formal_physics_identity
        from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING, CONTACT_LAYOUT
        from anymani.tasks.hetero.contact_sensors import sensor_contact_magnitude
        from isaaclab.assets import Articulation, RigidObject
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.sensors import ContactSensor
        from isaaclab.utils.assets import retrieve_file_path

        # 每个scale由独立prestartup scene拥有；replicate_physics=False允许per-prototype scale/physics view。
        cfg = GeneratedPregraspHarnessEnvCfg()
        cfg.scene.replicate_physics = False
        cfg.seed = args.seed
        object_spawn = cast(sim_utils.UsdFileCfg, cfg.scene.object.spawn)
        object_spawn.scale = (args.scale, args.scale, args.scale)
        if object_spawn.rigid_props is None:
            raise RuntimeError("DexCube spawn must expose rigid-body solver properties")
        runtime_env = ManagerBasedRLEnv(cfg=cfg)
        env = runtime_env
        runtime_env.sim._app_control_on_stop_handle = None
        env.reset()
        robot = cast(Articulation, runtime_env.scene["robot"])
        object_asset = cast(RigidObject, runtime_env.scene["object"])

        # 构造每个env的canonical active mask与soft-limit-aware q proposal。
        active_rows = torch.tensor(ASSET_BINDING.active_joint_masks, dtype=torch.bool, device=runtime_env.device)
        active = active_rows[torch.arange(num_envs, device=runtime_env.device) % asset_count]
        limits = robot.data.soft_joint_pos_limits
        lower, upper = limits[:, :, 0], limits[:, :, 1]
        span = upper - lower
        template = torch.tensor(N000_CANONICAL_Q, dtype=torch.float32, device=runtime_env.device).expand(num_envs, -1)
        q_target = torch.zeros(num_envs, 16, dtype=torch.float32, device=runtime_env.device)
        proposal_index = torch.arange(num_envs, device=runtime_env.device) // asset_count
        if args.portfolio == "grid":
            q_index = proposal_index // len(OBJECT_OFFSETS_H_M)
            q_target[q_index == 1] = template[q_index == 1]
            q_target[q_index == 2] = 0.5 * (lower[q_index == 2] + upper[q_index == 2])
            q_target[q_index == 3] = lower[q_index == 3] + 0.25 * span[q_index == 3]
            q_target[q_index == 4] = lower[q_index == 4] + 0.75 * span[q_index == 4]
            q_target[q_index == 5] = template[q_index == 5] + 0.2
            q_target[q_index == 6] = template[q_index == 6] - 0.2
            q_target[q_index == 7] = template[q_index == 7]
            q_target[q_index == 7, 4:] += 0.35  # depth1–3 flexion proposal
            q_names = [Q_PROPOSAL_NAMES[int(index)] for index in q_index.tolist()]
            search_q_proposals = list(Q_PROPOSAL_NAMES)
        elif args.portfolio == "random":
            # 四种形态无关seed交错出现，再施加AnyRotate式active-joint $U(-0.3,0.3)$ rad proposal。
            q_index = proposal_index % 4
            q_target[q_index == 0] = template[q_index == 0]
            q_target[q_index == 1] = 0.5 * (lower[q_index == 1] + upper[q_index == 1])
            q_target[q_index == 2] = lower[q_index == 2] + 0.75 * span[q_index == 2]
            q_target[q_index == 3] = template[q_index == 3]
            q_target[q_index == 3, 4:] += 0.35
            generator = torch.Generator(device=runtime_env.device)
            generator.manual_seed(args.seed)
            q_target += (torch.rand(q_target.shape, generator=generator, device=runtime_env.device) * 0.6 - 0.3) * active
            random_seed_names = ("template", "midpoint", "three_quarter", "distal_plus_0p35")
            q_names = [f"random_{random_seed_names[int(index)]}" for index in q_index.tolist()]
            search_q_proposals = [f"random_{name}" for name in random_seed_names]
        elif args.portfolio in {"refine", "verify", "basin"}:
            # 每只asset从已保存frontier出发；target代表PD预载，basin只扰动actual reset state。
            seed_q_rows = torch.tensor(
                [_seed_q_target(refinement_records[row]) for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )
            q_target = seed_q_rows[torch.arange(num_envs, device=runtime_env.device) % asset_count]
            if args.portfolio == "refine":
                generator = torch.Generator(device=runtime_env.device)
                generator.manual_seed(args.seed)
                q_target += (
                    (torch.rand(q_target.shape, generator=generator, device=runtime_env.device) * 2.0 - 1.0)
                    * float(args.refine_q_radius)
                    * active
                )
            q_index = torch.zeros_like(proposal_index)
            q_names = [
                f"{args.portfolio}_{refinement_records[rows[env_id % asset_count]]['candidate']['seed_source']}"
                for env_id in range(num_envs)
            ]
            search_q_proposals = sorted(set(q_names))
        else:
            # CEM在每只asset自己的elite population上拟合diagonal Gaussian；ghost维保持零。
            elite_q = torch.tensor(
                [[_seed_q_target(point["record"]) for point in cem_points[row]] for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )
            elite_mean = elite_q.mean(dim=1)
            elite_std = elite_q.std(dim=1, unbiased=False).clamp(min=0.01, max=0.2) * float(args.cem_std_scale)
            local_asset_ids = torch.arange(num_envs, device=runtime_env.device) % asset_count
            generator = torch.Generator(device=runtime_env.device)
            generator.manual_seed(args.seed)
            q_target = elite_mean[local_asset_ids] + torch.randn(
                q_target.shape, generator=generator, device=runtime_env.device
            ) * elite_std[local_asset_ids]
            q_index = torch.zeros_like(proposal_index)
            q_names = [f"cem_generation_from_{args.seed_artifact.name}" for _ in range(num_envs)]
            search_q_proposals = ["cem_diagonal_gaussian"]
        q_target = torch.maximum(torch.minimum(q_target, upper), lower) * active
        q_state = q_target.clone()  # proposal search从零预载开始；settle后actual state另行保存
        if args.portfolio == "verify":
            q_state = torch.tensor(
                [_seed_q_state(refinement_records[row]) for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )
            q_state = torch.maximum(torch.minimum(q_state, upper), lower) * active
        elif args.portfolio == "basin":
            seed_q_state_rows = torch.tensor(
                [_seed_q_state(refinement_records[row]) for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )
            local_asset_ids = torch.arange(num_envs, device=runtime_env.device) % asset_count
            generator = torch.Generator(device=runtime_env.device)
            generator.manual_seed(args.seed)
            q_perturbation = (
                (torch.rand(q_target.shape, generator=generator, device=runtime_env.device) * 2.0 - 1.0)
                * float(args.refine_q_radius)
                * active
            )  # $\delta q_s\sim U[-r_q,r_q]$，controller target保持nominal preload
            center_mask = proposal_index == 0  # 每只asset的首个trial是零扰动中心复核
            q_perturbation[center_mask] = 0.0
            q_state = seed_q_state_rows[local_asset_ids] + q_perturbation
            q_state = torch.maximum(torch.minimum(q_state, upper), lower) * active

        # 从default object pose构造确定性$T_{ho}$，不继承reset event随机yaw。
        frame = ASSET_BINDING.hand_spawn_cfg.frame
        hand_pos_w, hand_quat_w = hand_semantic_pose_w(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            frame.semantic_R_ha,
            frame.semantic_p_ha,
        )
        default_object_pos_w = object_asset.data.default_root_state[:, :3] + runtime_env.scene.env_origins
        default_object_quat_w = object_asset.data.default_root_state[:, 3:7]
        base_pos_h, base_quat_h = object_pose_h_from_world(
            hand_pos_w,
            hand_quat_w,
            default_object_pos_w,
            default_object_quat_w,
        )
        if args.portfolio == "grid":
            offset_index = proposal_index % len(OBJECT_OFFSETS_H_M)
            offsets_h = torch.tensor(OBJECT_OFFSETS_H_M, dtype=torch.float32, device=runtime_env.device)[offset_index]
            candidate_pos_h = base_pos_h + offsets_h
            candidate_quat_h = base_quat_h
        elif args.portfolio == "random":
            # Position range覆盖原7-point grid并加入连续邻域；yaw只改变hand-frame object orientation。
            generator = torch.Generator(device=runtime_env.device)
            generator.manual_seed(args.seed + 1)
            unit = torch.rand((num_envs, 4), generator=generator, device=runtime_env.device) * 2.0 - 1.0
            offsets_h = unit[:, :3] * torch.tensor((0.03, 0.04, 0.02), device=runtime_env.device)
            yaw = unit[:, 3] * math.pi
            yaw_axis = torch.zeros(num_envs, 3, device=runtime_env.device)
            yaw_axis[:, 2] = 1.0
            yaw_quat_h = math_utils.quat_from_angle_axis(yaw, yaw_axis)
            candidate_pos_h = base_pos_h + offsets_h
            candidate_quat_h = math_utils.quat_mul(yaw_quat_h, base_quat_h)
            offset_index = torch.full_like(proposal_index, -1)
        elif args.portfolio == "refine":
            seed_pos_h = torch.tensor(
                [refinement_records[row]["candidate"]["object_position_h_m"] for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )[torch.arange(num_envs, device=runtime_env.device) % asset_count]
            seed_quat_h = torch.tensor(
                [refinement_records[row]["candidate"]["object_orientation_wxyz"] for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )[torch.arange(num_envs, device=runtime_env.device) % asset_count]
            generator = torch.Generator(device=runtime_env.device)
            generator.manual_seed(args.seed + 1)
            unit = torch.rand((num_envs, 4), generator=generator, device=runtime_env.device) * 2.0 - 1.0
            offsets_h = unit[:, :3] * float(args.refine_position_radius)
            yaw = unit[:, 3] * math.radians(float(args.refine_yaw_deg))
            yaw_axis = torch.zeros(num_envs, 3, device=runtime_env.device)
            yaw_axis[:, 2] = 1.0
            candidate_pos_h = seed_pos_h + offsets_h
            candidate_quat_h = math_utils.quat_mul(math_utils.quat_from_angle_axis(yaw, yaw_axis), seed_quat_h)
            offset_index = torch.full_like(proposal_index, -2)
        elif args.portfolio == "verify":
            candidate_pos_h = torch.tensor(
                [refinement_records[row]["candidate"]["object_position_h_m"] for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )
            candidate_quat_h = torch.tensor(
                [refinement_records[row]["candidate"]["object_orientation_wxyz"] for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )
            offsets_h = torch.zeros_like(candidate_pos_h)
            offset_index = torch.full_like(proposal_index, -3)
        elif args.portfolio == "basin":
            local_asset_ids = torch.arange(num_envs, device=runtime_env.device) % asset_count
            seed_pos_h = torch.tensor(
                [refinement_records[row]["candidate"]["object_position_h_m"] for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )[local_asset_ids]
            seed_quat_h = torch.tensor(
                [refinement_records[row]["candidate"]["object_orientation_wxyz"] for row in rows],
                dtype=torch.float32,
                device=runtime_env.device,
            )[local_asset_ids]
            generator = torch.Generator(device=runtime_env.device)
            generator.manual_seed(args.seed + 1)
            offsets_h = (
                torch.rand((num_envs, 3), generator=generator, device=runtime_env.device) * 2.0 - 1.0
            ) * float(args.refine_position_radius)  # $\delta p_h\sim U[-r_p,r_p]^3$，单位m
            rotation_vector_h = (
                torch.rand((num_envs, 3), generator=generator, device=runtime_env.device) * 2.0 - 1.0
            ) * math.radians(float(args.refine_yaw_deg))  # 三轴rotation-vector扰动，不把local SO(3)退化为yaw-only
            center_mask = proposal_index == 0
            offsets_h[center_mask] = 0.0
            rotation_vector_h[center_mask] = 0.0
            rotation_angle = torch.linalg.vector_norm(rotation_vector_h, dim=-1)
            rotation_axis = rotation_vector_h / rotation_angle.clamp_min(1.0e-12).unsqueeze(-1)
            rotation_axis[center_mask, 2] = 1.0  # 零角度时axis任意；选hand $+z$保持finite quaternion API
            perturbation_quat_h = math_utils.quat_from_angle_axis(rotation_angle, rotation_axis)
            candidate_pos_h = seed_pos_h + offsets_h
            candidate_quat_h = math_utils.quat_mul(perturbation_quat_h, seed_quat_h)
            offset_index = torch.full_like(proposal_index, -5)
        else:
            elite_pos = torch.tensor(
                [
                    [point["record"]["candidate"]["object_position_h_m"] for point in cem_points[row]]
                    for row in rows
                ],
                dtype=torch.float32,
                device=runtime_env.device,
            )
            elite_quat = torch.tensor(
                [
                    [point["record"]["candidate"]["object_orientation_wxyz"] for point in cem_points[row]]
                    for row in rows
                ],
                dtype=torch.float32,
                device=runtime_env.device,
            )
            # Quaternion先相对每组首个elite统一双覆盖符号，再归一化mean；额外yaw维持小范围探索。
            reference = elite_quat[:, :1]
            signs = torch.where((elite_quat * reference).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
            mean_quat = torch.nn.functional.normalize((elite_quat * signs).mean(dim=1), dim=-1)
            mean_pos = elite_pos.mean(dim=1)
            std_pos = elite_pos.std(dim=1, unbiased=False).clamp(min=0.002, max=0.02) * float(args.cem_std_scale)
            local_asset_ids = torch.arange(num_envs, device=runtime_env.device) % asset_count
            generator = torch.Generator(device=runtime_env.device)
            generator.manual_seed(args.seed + 1)
            candidate_pos_h = mean_pos[local_asset_ids] + torch.randn(
                (num_envs, 3), generator=generator, device=runtime_env.device
            ) * std_pos[local_asset_ids]
            yaw = (torch.rand(num_envs, generator=generator, device=runtime_env.device) * 2.0 - 1.0) * math.radians(10.0)
            yaw_axis = torch.zeros(num_envs, 3, device=runtime_env.device)
            yaw_axis[:, 2] = 1.0
            candidate_quat_h = math_utils.quat_mul(
                math_utils.quat_from_angle_axis(yaw, yaw_axis), mean_quat[local_asset_ids]
            )
            offsets_h = candidate_pos_h - mean_pos[local_asset_ids]
            offset_index = torch.full_like(proposal_index, -4)
        candidate_pos_w, candidate_quat_w = object_pose_w_from_hand(
            hand_pos_w,
            hand_quat_w,
            candidate_pos_h,
            candidate_quat_h,
        )

        # Basin在hand frame均匀施加线/角速度扰动，其余portfolio从静止点开始。
        initial_velocity_h = torch.zeros(num_envs, 6, device=runtime_env.device)
        if args.portfolio == "basin":
            generator = torch.Generator(device=runtime_env.device)
            generator.manual_seed(args.seed + 2)
            unit_velocity = torch.rand((num_envs, 6), generator=generator, device=runtime_env.device) * 2.0 - 1.0
            initial_velocity_h[:, :3] = unit_velocity[:, :3] * float(args.basin_linear_velocity_radius)
            initial_velocity_h[:, 3:] = unit_velocity[:, 3:] * float(args.basin_angular_velocity_radius)
            initial_velocity_h[proposal_index == 0] = 0.0  # 首trial维持中心点零扰动对照
        initial_velocity_w = torch.cat(
            (
                math_utils.quat_apply(hand_quat_w, initial_velocity_h[:, :3]),
                math_utils.quat_apply(hand_quat_w, initial_velocity_h[:, 3:]),
            ),
            dim=-1,
        )

        # 写入actual reset state、PD preload target与object state；后续6 s不执行policy action。
        robot.write_joint_state_to_sim(q_state, torch.zeros_like(q_state))
        robot.set_joint_position_target(q_target)
        object_asset.write_root_pose_to_sim(torch.cat((candidate_pos_w, candidate_quat_w), dim=-1))
        object_asset.write_root_velocity_to_sim(initial_velocity_w)
        initial_object_pos_w = candidate_pos_w.clone()
        initial_object_quat_w = candidate_quat_w.clone()

        # 共享EMA严格使用task contract的alpha=0.5与0.25 N阈值；不依赖common_step_counter。
        tip_ema = torch.zeros(num_envs, 4, device=runtime_env.device)
        non_tip_ema = torch.zeros(
            num_envs, len(CONTACT_LAYOUT.finger_non_tip_sensor_names), device=runtime_env.device
        )
        palm_ema = torch.zeros(num_envs, device=runtime_env.device)
        tip_ge2_samples, tip_ge3_samples, tip_count_samples = [], [], []
        tip_bit_samples, non_tip_bit_samples = [], []
        non_tip_samples, palm_samples, tip_distance_samples = [], [], []
        anchor_samples, linear_sq_samples, angular_sq_samples = [], [], []
        orientation_samples, tracking_sq_samples, effort_sq_samples = [], [], []
        penetration_max = torch.zeros(num_envs, device=runtime_env.device)
        tip_body_ids, _ = robot.find_bodies(
            list(CONTACT_LAYOUT.fingertip_links), preserve_order=True
        )
        all_sensor_names = (
            *CONTACT_LAYOUT.fingertip_sensor_names,
            *CONTACT_LAYOUT.finger_non_tip_sensor_names,
            CONTACT_LAYOUT.palm_sensor_name,
        )
        sensors = {name: cast(ContactSensor, runtime_env.scene[name]) for name in all_sensor_names}

        # AnyRotate的120 policy steps在本任务20 Hz下等于6 s；每步包含6个120 Hz physics substeps。
        for physics_step in range(120 * 6):
            robot.set_joint_position_target(q_target)
            runtime_env.scene.write_data_to_sim()
            runtime_env.sim.step(render=False)
            runtime_env.scene.update(runtime_env.physics_dt)
            if (physics_step + 1) % 6:
                continue
            tip_force = torch.stack(
                [
                    sensor_contact_magnitude(runtime_env, name)
                    for name in CONTACT_LAYOUT.fingertip_sensor_names
                ],
                dim=-1,
            )
            non_tip_force = torch.stack(
                [
                    sensor_contact_magnitude(runtime_env, name)
                    for name in CONTACT_LAYOUT.finger_non_tip_sensor_names
                ],
                dim=-1,
            )
            palm_force = sensor_contact_magnitude(runtime_env, CONTACT_LAYOUT.palm_sensor_name)
            tip_ema = 0.5 * tip_force + 0.5 * tip_ema
            non_tip_ema = 0.5 * non_tip_force + 0.5 * non_tip_ema
            palm_ema = 0.5 * palm_force + 0.5 * palm_ema
            tip_bits = tip_ema > 0.25
            tip_count = tip_bits.sum(dim=-1).float()
            tip_ge2_samples.append((tip_count >= 2.0).float())
            tip_ge3_samples.append((tip_count >= 3.0).float())
            tip_count_samples.append(tip_count)
            tip_bit_samples.append(tip_bits.float())
            non_tip_bits = non_tip_ema > 0.25
            non_tip_bit_samples.append(non_tip_bits.float())
            non_tip_samples.append(non_tip_bits.any(dim=-1).float())
            palm_samples.append((palm_ema > 0.25).float())
            tip_distance_samples.append(
                torch.linalg.vector_norm(
                    robot.data.body_pos_w[:, tip_body_ids] - object_asset.data.root_pos_w.unsqueeze(1), dim=-1
                ).mean(dim=-1)
            )
            anchor_samples.append(torch.linalg.vector_norm(object_asset.data.root_pos_w - initial_object_pos_w, dim=-1))
            linear_sq_samples.append(object_asset.data.root_lin_vel_w.square().sum(dim=-1))
            angular_sq_samples.append(object_asset.data.root_ang_vel_w.square().sum(dim=-1))
            orientation_samples.append(math_utils.quat_error_magnitude(initial_object_quat_w, object_asset.data.root_quat_w))
            active_count = active.sum(dim=-1).clamp_min(1)
            tracking_sq_samples.append(((robot.data.joint_pos - q_target) * active).square().sum(dim=-1) / active_count)
            effort_sq_samples.append((robot.data.computed_torque * active).square().sum(dim=-1) / active_count)
            for sensor in sensors.values():
                penetration_max = torch.maximum(
                    penetration_max,
                    contact_penetration_depth_per_env(sensor, runtime_env.physics_dt),
                )

        # 前4 s允许proposal在重力/接触下settle；后2 s的40个samples定义point稳定接触与rate metrics。
        # Drop、最大漂移、orientation excursion和penetration仍覆盖完整6 s，不能用tail掩盖早期失稳。
        gate_start = 80
        tip_ge2 = torch.stack(tip_ge2_samples)[gate_start:].mean(dim=0)
        tip_ge3 = torch.stack(tip_ge3_samples)[gate_start:].mean(dim=0)
        tip_count_mean = torch.stack(tip_count_samples)[gate_start:].mean(dim=0)
        tip_sensor_fraction = torch.stack(tip_bit_samples)[gate_start:].mean(dim=0)
        non_tip_sensor_fraction = torch.stack(non_tip_bit_samples)[gate_start:].mean(dim=0)
        non_tip_fraction = torch.stack(non_tip_samples)[gate_start:].mean(dim=0)
        palm_fraction = torch.stack(palm_samples)[gate_start:].mean(dim=0)
        tip_distance_mean = torch.stack(tip_distance_samples)[gate_start:].mean(dim=0)
        anchor_max = torch.stack(anchor_samples).amax(dim=0)
        linear_rms = torch.sqrt(torch.stack(linear_sq_samples)[gate_start:].mean(dim=0))
        angular_rms = torch.sqrt(torch.stack(angular_sq_samples)[gate_start:].mean(dim=0))
        orientation_max = torch.stack(orientation_samples).amax(dim=0)
        tracking_rms = torch.sqrt(torch.stack(tracking_sq_samples)[gate_start:].mean(dim=0))
        effort_rms = torch.sqrt(torch.stack(effort_sq_samples)[gate_start:].mean(dim=0))
        settled_q = robot.data.joint_pos.detach() * active  # AnyRotate语义：保存settle后的实际joint state
        settled_pos_h, settled_quat_h = object_pose_h_from_world(
            hand_pos_w,
            hand_quat_w,
            object_asset.data.root_pos_w,
            object_asset.data.root_quat_w,
        )
        lower_margin = torch.where(active, settled_q - lower, torch.inf).amin(dim=-1)
        upper_margin = torch.where(active, upper - settled_q, torch.inf).amin(dim=-1)
        joint_margin = torch.minimum(lower_margin, upper_margin)
        finite = torch.isfinite(
            torch.stack(
                (
                    tip_ge2,
                    tip_ge3,
                    tip_count_mean,
                    non_tip_fraction,
                    palm_fraction,
                    tip_distance_mean,
                    anchor_max,
                    linear_rms,
                    angular_rms,
                    orientation_max,
                    tracking_rms,
                    effort_rms,
                    penetration_max,
                ),
                dim=-1,
            )
        ).all(dim=-1)

        # Physics lookup identity保存scale-invariant policy；每scale actual snapshot留在search artifact。
        object_local_path = Path(retrieve_file_path(str(object_spawn.usd_path))).resolve()
        cube_sha256 = file_sha256(object_local_path)
        gate = _build_gate()
        search_identity = {
            "algorithm": f"hetero-point-{args.portfolio}-v3",
            "seed": args.seed,
            "q_proposals": search_q_proposals,
            "object_offsets_h_m": [list(offset) for offset in OBJECT_OFFSETS_H_M] if args.portfolio == "grid" else None,
            "random_candidate_count": point_count_per_asset if args.portfolio == "random" else None,
            "basin_trial_count": point_count_per_asset if args.portfolio == "basin" else None,
            "random_q_offset_rad": [-0.3, 0.3] if args.portfolio == "random" else None,
            "random_object_offset_h_m": (
                {"x": [-0.03, 0.03], "y": [-0.04, 0.04], "z": [-0.02, 0.02]}
                if args.portfolio == "random"
                else None
            ),
            "random_yaw_rad": [-math.pi, math.pi] if args.portfolio == "random" else None,
            "refinement_source_sha256": (
                file_sha256(args.seed_artifact)
                if args.portfolio in {"refine", "verify", "cem", "basin"}
                else None
            ),
            "refinement_q_offset_rad": (
                [-args.refine_q_radius, args.refine_q_radius]
                if args.portfolio in {"refine", "basin"}
                else None
            ),
            "refinement_object_offset_h_m": (
                [-args.refine_position_radius, args.refine_position_radius]
                if args.portfolio in {"refine", "basin"}
                else None
            ),
            "refinement_yaw_rad": (
                [-math.radians(args.refine_yaw_deg), math.radians(args.refine_yaw_deg)]
                if args.portfolio == "refine"
                else None
            ),
            "basin_rotation_vector_h_rad": (
                [-math.radians(args.refine_yaw_deg), math.radians(args.refine_yaw_deg)]
                if args.portfolio == "basin"
                else None
            ),
            "basin_linear_velocity_h_m_s": (
                [-args.basin_linear_velocity_radius, args.basin_linear_velocity_radius]
                if args.portfolio == "basin"
                else None
            ),
            "basin_angular_velocity_h_rad_s": (
                [-args.basin_angular_velocity_radius, args.basin_angular_velocity_radius]
                if args.portfolio == "basin"
                else None
            ),
            "basin_center_control_trials_per_asset": 1 if args.portfolio == "basin" else None,
            "basin_minimum_tier": args.basin_min_tier if args.portfolio == "basin" else None,
            "cem_elite_count": args.elite_count if args.portfolio == "cem" else None,
            "cem_std_scale": args.cem_std_scale if args.portfolio == "cem" else None,
            "settle_policy_steps": 120,
            "settle_prefix_policy_steps": gate_start,
            "certification_tail_policy_steps": 120 - gate_start,
            "physics_substeps_per_policy_step": 6,
        }
        physics_identity = formal_physics_identity(object_scale=args.scale, cube_sha256=cube_sha256)
        point_records = []
        ranked_by_asset: list[list[tuple[tuple[float, ...], int, Any]]] = [[] for _ in range(asset_count)]
        frontier_by_asset: list[list[tuple[tuple[float, ...], int, Any]]] = [[] for _ in range(asset_count)]
        gate_frontier_by_asset: list[list[tuple[tuple[float, ...], int, Any]]] = [
            [] for _ in range(asset_count)
        ]
        support_frontier_by_asset: list[list[tuple[tuple[float, ...], int, Any]]] = [
            [] for _ in range(asset_count)
        ]
        tier_rank = {"rejected": 0, "support_basin": 1, "contact_basin": 2, "gravity_robust": 3}
        for env_id in range(num_envs):
            local_asset = env_id % asset_count
            artifact = ASSET_BINDING.canonical_artifacts[local_asset]
            source_asset = ASSET_BINDING.source_assets[local_asset]
            metrics = PregraspMetrics(
                finite=bool(finite[env_id].item()),
                dropped=bool(anchor_max[env_id].item() >= 0.07),
                penetration_depth_max_m=float(penetration_max[env_id].item()),
                tip_ge_2_fraction=float(tip_ge2[env_id].item()),
                tip_ge_3_fraction=float(tip_ge3[env_id].item()),
                tip_active_count_mean=float(tip_count_mean[env_id].item()),
                palm_occupancy_fraction=float(palm_fraction[env_id].item()),
                finger_non_tip_occupancy_fraction=float(non_tip_fraction[env_id].item()),
                tip_object_center_distance_mean_m=float(tip_distance_mean[env_id].item()),
                object_anchor_distance_max_m=float(anchor_max[env_id].item()),
                object_linear_velocity_rms_m_s=float(linear_rms[env_id].item()),
                object_angular_velocity_rms_rad_s=float(angular_rms[env_id].item()),
                object_orientation_drift_max_rad=float(orientation_max[env_id].item()),
                joint_limit_margin_min_rad=float(joint_margin[env_id].item()),
                target_tracking_error_rms_rad=float(tracking_rms[env_id].item()),
                joint_effort_rms_N_m=float(effort_rms[env_id].item()),
            )
            candidate = PregraspCandidate(
                q_state_rad=tuple(float(value) for value in settled_q[env_id].tolist()),
                q_target_rad=tuple(float(value) for value in q_target[env_id].tolist()),
                active_joint_mask=tuple(bool(value) for value in active[env_id].tolist()),
                object_position_h_m=(
                    float(settled_pos_h[env_id, 0].item()),
                    float(settled_pos_h[env_id, 1].item()),
                    float(settled_pos_h[env_id, 2].item()),
                ),
                object_orientation_wxyz=(
                    float(settled_quat_h[env_id, 0].item()),
                    float(settled_quat_h[env_id, 1].item()),
                    float(settled_quat_h[env_id, 2].item()),
                    float(settled_quat_h[env_id, 3].item()),
                ),
                object_scale=args.scale,
                seed_source=(
                    f"{q_names[env_id]}+offset-{int(offset_index[env_id].item())}"
                ),
            )
            lookup_key = PregraspLookupKey(
                asset_id=source_asset.asset_id,
                source_content_hash=artifact.source_content_hash,
                physical_geometry_hash=artifact.physical_geometry_hash,
                canonical_schema_digest=artifact.schema_digest,
                routing_digest=active_mask_digest(candidate.active_joint_mask),
                cube_asset_id="DexCube",
                cube_asset_sha256=cube_sha256,
                support_mode="palm_supported",
                gate_digest=gate.digest,
                physics_identity=physics_identity,
                search_identity=search_identity,
            )
            record = certify_pregrasp(
                lookup_key=lookup_key,
                candidate=candidate,
                metrics=metrics,
                gate=gate,
                coverage=PregraspCoverage.POINT,
                scale_certificate=None,
            )
            rank = (
                float(tier_rank[record.tier.value]),
                metrics.tip_ge_2_fraction,
                metrics.tip_active_count_mean,
                -metrics.finger_non_tip_occupancy_fraction,
                -metrics.object_anchor_distance_max_m,
                metrics.joint_limit_margin_min_rad,
            )
            ranked_by_asset[local_asset].append((rank, env_id, record))
            frontier_rank = (
                metrics.tip_ge_2_fraction,
                metrics.tip_active_count_mean,
                -metrics.finger_non_tip_occupancy_fraction,
                -metrics.target_tracking_error_rms_rad,
                -metrics.object_anchor_distance_max_m,
                -metrics.penetration_depth_max_m,
            )
            frontier_by_asset[local_asset].append((frontier_rank, env_id, record))
            gate_distance = (
                max(0.0, gate.min_tip_ge_2_fraction - metrics.tip_ge_2_fraction)
                + 10.0 * max(0.0, metrics.finger_non_tip_occupancy_fraction - gate.max_finger_non_tip_fraction)
                + max(0.0, metrics.object_anchor_distance_max_m / gate.max_anchor_distance_m - 1.0)
                + max(0.0, metrics.object_angular_velocity_rms_rad_s / gate.max_angular_velocity_rms_rad_s - 1.0)
                + max(
                    0.0,
                    metrics.object_orientation_drift_max_rad / gate.max_object_orientation_drift_rad - 1.0,
                )
                + max(0.0, metrics.target_tracking_error_rms_rad / gate.max_target_tracking_error_rms_rad - 1.0)
            )
            gate_rank = (
                float(metrics.tip_ge_2_fraction >= gate.min_tip_ge_2_fraction),
                -float(len(record.reason_codes)),
                -gate_distance,
                metrics.tip_ge_2_fraction,
            )
            gate_frontier_by_asset[local_asset].append((gate_rank, env_id, record))
            support_rank = (
                float(record.tier.value != "rejected"),
                metrics.joint_limit_margin_min_rad,
                -metrics.target_tracking_error_rms_rad,
                -metrics.object_anchor_distance_max_m,
            )
            support_frontier_by_asset[local_asset].append((support_rank, env_id, record))
            point_records.append(
                {
                    "env_id": env_id,
                    "dataset_row": int(ASSET_BINDING.dataset_rows[local_asset]),
                    "q_proposal": q_names[env_id],
                    "object_offset_index": int(offset_index[env_id].item()),
                    "initial_q_state_rad": [float(value) for value in q_state[env_id].tolist()],
                    "initial_object_position_h_m": [float(value) for value in candidate_pos_h[env_id].tolist()],
                    "initial_object_orientation_h_wxyz": [
                        float(value) for value in candidate_quat_h[env_id].tolist()
                    ],
                    "initial_linear_velocity_h_m_s": [
                        float(value) for value in initial_velocity_h[env_id, :3].tolist()
                    ],
                    "initial_angular_velocity_h_rad_s": [
                        float(value) for value in initial_velocity_h[env_id, 3:].tolist()
                    ],
                    "tier": record.tier.value,
                    "reason_codes": list(record.reason_codes),
                    "tip_sensor_occupancy_fraction": [
                        float(value) for value in tip_sensor_fraction[env_id].tolist()
                    ],
                    "finger_non_tip_sensor_occupancy_fraction": [
                        float(value) for value in non_tip_sensor_fraction[env_id].tolist()
                    ],
                    "record": record.to_dict(),
                    **_point_metric_dict(metrics),
                }
            )

        selected_records = []
        frontier_records = []
        gate_frontier_records = []
        support_frontier_records = []
        for local_asset, candidates in enumerate(ranked_by_asset):
            _, env_id, record = max(candidates, key=lambda item: item[0])
            selected_records.append(
                {
                    "dataset_row": int(ASSET_BINDING.dataset_rows[local_asset]),
                    "selected_env_id": env_id,
                    "record": record.to_dict(),
                }
            )
            _, frontier_env_id, frontier_record = max(frontier_by_asset[local_asset], key=lambda item: item[0])
            frontier_records.append(
                {
                    "dataset_row": int(ASSET_BINDING.dataset_rows[local_asset]),
                    "frontier_env_id": frontier_env_id,
                    "tip_sensor_occupancy_fraction": [
                        float(value) for value in tip_sensor_fraction[frontier_env_id].tolist()
                    ],
                    "finger_non_tip_sensor_occupancy_fraction": [
                        float(value) for value in non_tip_sensor_fraction[frontier_env_id].tolist()
                    ],
                    "record": frontier_record.to_dict(),
                }
            )
            _, gate_env_id, gate_record = max(gate_frontier_by_asset[local_asset], key=lambda item: item[0])
            gate_frontier_records.append(
                {
                    "dataset_row": int(ASSET_BINDING.dataset_rows[local_asset]),
                    "gate_frontier_env_id": gate_env_id,
                    "tip_sensor_occupancy_fraction": [
                        float(value) for value in tip_sensor_fraction[gate_env_id].tolist()
                    ],
                    "finger_non_tip_sensor_occupancy_fraction": [
                        float(value) for value in non_tip_sensor_fraction[gate_env_id].tolist()
                    ],
                    "record": gate_record.to_dict(),
                }
            )
            _, support_env_id, support_record = max(
                support_frontier_by_asset[local_asset], key=lambda item: item[0]
            )
            support_frontier_records.append(
                {
                    "dataset_row": int(ASSET_BINDING.dataset_rows[local_asset]),
                    "support_frontier_env_id": support_env_id,
                    "record": support_record.to_dict(),
                }
            )

        actual_mass = object_asset.root_physx_view.get_masses().detach().cpu().reshape(num_envs, -1)
        actual_inertia = object_asset.root_physx_view.get_inertias().detach().cpu().reshape(num_envs, -1)
        basin_summary = []
        if args.portfolio == "basin":
            accepted_tiers = (
                {"support_basin", "contact_basin", "gravity_robust"}
                if args.basin_min_tier == "support_basin"
                else {"contact_basin", "gravity_robust"}
            )  # support盆只要求完整物理门；contact盆额外要求TIP/non-tip门
            for local_asset, row in enumerate(rows):
                asset_points = [point for point in point_records if int(point["dataset_row"]) == row]
                tier_successes = sum(point["tier"] in accepted_tiers for point in asset_points)
                basin_summary.append(
                    {
                        "dataset_row": row,
                        "trials": len(asset_points),
                        "minimum_tier": args.basin_min_tier,
                        "tier_successes": tier_successes,
                        "tier_success_fraction": tier_successes / len(asset_points),
                        "required_success_fraction": gate.min_basin_success_fraction,
                        "passed": tier_successes / len(asset_points) >= gate.min_basin_success_fraction,
                        "center_trial_env_id": local_asset,
                        "center_trial_tier": point_records[local_asset]["tier"],
                    }
                )
        output = {
            "artifact_type": "anymani.pregrasp.point_search",
            "schema_version": "3.0.0",
            "dataset_rows": list(rows),
            "scale": args.scale,
            "gate": gate.to_dict(),
            "gate_digest": gate.digest,
            "cube_sha256": cube_sha256,
            "physics_identity": physics_identity,
            "actual_object_mass_kg": actual_mass[:asset_count].tolist(),
            "actual_object_inertia_kg_m2": actual_inertia[:asset_count].tolist(),
            "candidate_count_per_asset": point_count_per_asset,
            "portfolio": args.portfolio,
            "selected": selected_records,
            "contact_frontier": frontier_records,
            "gate_frontier": gate_frontier_records,
            "support_frontier": support_frontier_records,
            "basin_summary": basin_summary if args.portfolio == "basin" else None,
            "points": point_records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "scale": args.scale,
                    "selected_tiers": [item["record"]["tier"] for item in selected_records],
                    "tier_histogram": {
                        tier: sum(point["tier"] == tier for point in point_records)
                        for tier in ("rejected", "support_basin", "contact_basin", "gravity_robust")
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        return 2
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
