r"""生成MVP80 DexCube scale-1.1 strict v5 Top-8 good-pregrasp catalog。

每资产先生成256个13维scrambled-Sobol几何提案，仅选Top-32进入训练同路径的1 s cold-reset物理筛选。
若任一资产不足8个严格候选，从其已测physical elites拟合4D joint-PCA + 3D object-position CEM，最多
追加3轮；每轮生成128个提案、cheap geometry筛到Top-32再做物理验证。门限始终固定，不按资产手调：

- active joint margin ≥10%；三指TIP-center距离≤10 cm；面内sector≥30°；
- penetration≤0.5 mm；1 s位移≤5 mm；倾角≤10°；
- 前0.2 s线速度≤0.25 m/s、**总**角速度≤2 rad/s；后0.5 s PALM support≥50%。

只有80只资产各自至少8个候选通过同一谓词时才发布catalog；否则只保存候选证据与失败分解并返回非零。
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
CANDIDATES_PER_PHYSICS_BATCH = 32
PHYSICS_STEPS = 120  # 1 s × 120 Hz
POLICY_SUBSTEPS = 6  # 20 Hz contact samples
EARLY_PEAK_STEPS = 24  # first 0.2 s
TAIL_POLICY_SAMPLES = 10  # final 0.5 s
DEFAULT_SELECTION = Path(
    "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80_candidates.yaml"
)
DEFAULT_CATALOG = Path("outputs/pregrasp/catalogs/heterogeneous_rotation_mvp80_dexcube_s1p1_v5")
DEFAULT_EVIDENCE = Path("outputs/pregrasp/search/heterogeneous_rotation_mvp80_dexcube_s1p1_v5")


def _parse_args() -> argparse.Namespace:
    r"""解析formal输出、development prefix和固定CEM轮数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--asset-limit", type=int, default=None, help="Development-only ordered prefix; never publish.")
    parser.add_argument("--rows", type=str, default=None, help="Development-only comma-separated selected rows.")
    parser.add_argument(
        "--publish-selection",
        action="store_true",
        help="Publish an explicit --rows selection only when it contains exactly 80 frozen candidate rows.",
    )
    parser.add_argument("--max-cem-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.asset_limit is not None and not 1 <= args.asset_limit <= ASSETS_PER_RUN:
        parser.error("--asset-limit must lie in [1,80]")
    if args.asset_limit is not None and args.rows is not None:
        parser.error("--asset-limit and --rows are mutually exclusive")
    if args.publish_selection and (args.rows is None or args.asset_limit is not None):
        parser.error("--publish-selection requires --rows and forbids --asset-limit")
    if not 0 <= args.max_cem_rounds <= 3:
        parser.error("--max-cem-rounds must lie in [0,3]")
    return args


ARGS = _parse_args()
SELECTION_DOCUMENT = yaml.safe_load(ARGS.selection.read_text(encoding="utf-8"))
FORMAL_SELECTED_ROWS = tuple(int(row) for row in SELECTION_DOCUMENT["initial_selected_rows"])
if len(FORMAL_SELECTED_ROWS) != ASSETS_PER_RUN or len(set(FORMAL_SELECTED_ROWS)) != ASSETS_PER_RUN:
    raise ValueError("strict MVP pregrasp generation requires exactly 80 unique initial selection rows")
FROZEN_CANDIDATE_ROWS = {
    int(side["row"])
    for cell in SELECTION_DOCUMENT["cells"]
    for pair in cell["candidate_pairs"]
    for side in (pair["left"], pair["right"])
}
if ARGS.rows is not None:
    SELECTED_ROWS = tuple(int(value.strip()) for value in ARGS.rows.split(",") if value.strip())
    if not SELECTED_ROWS or len(set(SELECTED_ROWS)) != len(SELECTED_ROWS):
        raise ValueError("--rows must contain unique formal selection rows")
    if not set(SELECTED_ROWS).issubset(FROZEN_CANDIDATE_ROWS):
        raise ValueError("development --rows must be drawn from the frozen pair-candidate manifest")
else:
    SELECTED_ROWS = FORMAL_SELECTED_ROWS[: ARGS.asset_limit] if ARGS.asset_limit is not None else FORMAL_SELECTED_ROWS
DEVELOPMENT_RUN = ARGS.asset_limit is not None or ARGS.rows is not None
ASSET_COUNT = len(SELECTED_ROWS)
if ARGS.publish_selection and ASSET_COUNT != ASSETS_PER_RUN:
    raise ValueError("--publish-selection requires exactly 80 rows")
NUM_ENVS = ASSET_COUNT * CANDIDATES_PER_PHYSICS_BATCH
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in SELECTED_ROWS)
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(NUM_ENVS)

from isaaclab.app import AppLauncher  # noqa: E402  # asset/env routing必须先冻结

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def main() -> int:
    r"""执行Sobol/Top-32/CEM/strict physical筛选并全有或全无地发布formal catalog。"""

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
        deepest_contact_normal_per_env,
        hand_semantic_pose_w,
        object_pose_h_from_world,
        object_pose_w_from_hand,
    )
    from anymani.pregrasp.mvp80_strict_search import (
        fixed_position_envelope,
        geometry_score,
        initial_envelope,
        initial_joint_candidates,
        low_rank_cem_candidates,
        normalized_gate_violation,
        sobol_bank,
        strict_pass_mask,
    )
    from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE
    from anymani.tasks.hetero.config.generated.pregrasp_harness_env_cfg import GeneratedPregraspHarnessEnvCfg
    from anymani.tasks.hetero.config.generated.scene import (
        ASSET_BINDING,
        CONTACT_LAYOUT,
        RESOLVED_DEX_CUBE_SHA256,
    )
    from anymani.tasks.hetero.config.generated.strict_good_pregrasp_identity import (
        STRICT_GOOD_PREGRASP_CEM_CANDIDATES,
        STRICT_GOOD_PREGRASP_CEM_ELITES,
        STRICT_GOOD_PREGRASP_CEM_ROUNDS,
        STRICT_GOOD_PREGRASP_GENERATION_DIGEST,
        STRICT_GOOD_PREGRASP_GENERATION_IDENTITY,
        STRICT_GOOD_PREGRASP_OBJECT_SCALE,
        STRICT_GOOD_PREGRASP_PHYSICS_DIGEST,
        STRICT_GOOD_PREGRASP_PHYSICS_TOP_K,
        STRICT_GOOD_PREGRASP_SEED,
        STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES,
    )
    from anymani.tasks.hetero.contact_sensors import sensor_contact_magnitude
    from anymani.tasks.hetero.mdp.runtime_state import derive_tip_and_owner_masks
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

    # Formal invocation必须与runtime候选身份逐值一致；development prefix只允许减少资产，不改算法数字。
    if ARGS.seed != STRICT_GOOD_PREGRASP_SEED or ARGS.max_cem_rounds != STRICT_GOOD_PREGRASP_CEM_ROUNDS:
        if not DEVELOPMENT_RUN or ARGS.publish_selection:
            raise ValueError("formal strict generation requires shared seed and three CEM rounds")
    if ASSET_BINDING.dataset_rows != SELECTED_ROWS or ASSET_BINDING.asset_count != ASSET_COUNT:
        raise RuntimeError("scene asset binding disagrees with strict selection")
    if STRICT_GOOD_PREGRASP_OBJECT_SCALE != 1.1:
        raise RuntimeError("strict generation identity must fix DexCube scale 1.1")

    device = torch.device("cuda:0")
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)
    cfg = GeneratedPregraspHarnessEnvCfg()
    cfg.seed = ARGS.seed
    object_spawn = cast(sim_utils.UsdFileCfg, cfg.scene.object.spawn)
    object_spawn.scale = (1.1, 1.1, 1.1)
    runtime_env = ManagerBasedRLEnv(cfg=cfg)
    runtime_env.sim._app_control_on_stop_handle = None

    try:
        runtime_env.reset()
        robot = cast(Articulation, runtime_env.scene["robot"])
        object_asset = cast(RigidObject, runtime_env.scene["object"])
        tip_body_ids, _ = robot.find_bodies(list(CONTACT_LAYOUT.fingertip_links), preserve_order=True)
        if len(tip_body_ids) != 4:
            raise RuntimeError("strict generator requires four canonical TIP body slots")
        sensors = {
            name: cast(ContactSensor, runtime_env.scene[name]) for name in CONTACT_LAYOUT.state_sensor_names
        }
        sensor_owner_indices = torch.tensor(CONTACT_LAYOUT.sensor_owner_indices, dtype=torch.long, device=device)
        active_by_asset = torch.tensor(ASSET_BINDING.active_joint_masks, dtype=torch.bool, device=device)
        active_tip_by_asset, active_owner_by_asset = derive_tip_and_owner_masks(active_by_asset)
        active_tip_by_env = active_tip_by_asset.repeat(CANDIDATES_PER_PHYSICS_BATCH, 1)
        active_owner_by_env = active_owner_by_asset.repeat(CANDIDATES_PER_PHYSICS_BATCH, 1)
        limits = robot.data.soft_joint_pos_limits[:ASSET_COUNT]
        lower_by_asset, upper_by_asset = limits[..., 0].clone(), limits[..., 1].clone()
        frame = ASSET_BINDING.hand_spawn_cfg.frame

        def env_major(values: torch.Tensor) -> torch.Tensor:
            r"""把`[A,32,...]`转成scene slot-major`[32*A,...]`。"""

            return values.transpose(0, 1).reshape(NUM_ENVS, *values.shape[2:])

        def asset_major(values: torch.Tensor) -> torch.Tensor:
            r"""把scene`[32*A,...]`恢复为`[A,32,...]`。"""

            return values.reshape(CANDIDATES_PER_PHYSICS_BATCH, ASSET_COUNT, *values.shape[1:]).transpose(0, 1)

        def joint_margin(q: torch.Tensor) -> torch.Tensor:
            r"""计算`[A,C,16]`候选的最近active normalized limit margin。"""

            span = (upper_by_asset - lower_by_asset).clamp_min(1.0e-6).unsqueeze(1)
            margin = torch.minimum(
                (q - lower_by_asset.unsqueeze(1)) / span,
                (upper_by_asset.unsqueeze(1) - q) / span,
            )
            return torch.where(active_by_asset.unsqueeze(1), margin, torch.inf).amin(dim=-1)

        def realize_geometry(
            q_proposals: torch.Tensor,
            *,
            sobol_values: torch.Tensor | None = None,
            direct_positions: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            r"""用direct joint write+一次physics刷新TIP FK，计算全部cheap geometry指标。"""

            candidate_count = q_proposals.shape[1]
            if candidate_count % CANDIDATES_PER_PHYSICS_BATCH != 0:
                raise ValueError("geometry proposal count must be divisible by 32")
            if (sobol_values is None) == (direct_positions is None):
                raise ValueError("geometry realization requires exactly one object proposal representation")
            output = {
                "q": q_proposals,
                "joint_margin": joint_margin(q_proposals),
                "position": torch.zeros(ASSET_COUNT, candidate_count, 3, device=device),
                "pair": torch.zeros(ASSET_COUNT, candidate_count, 2, dtype=torch.long, device=device),
                "distances": torch.zeros(ASSET_COUNT, candidate_count, 3, device=device),
                "sector_deg": torch.zeros(ASSET_COUNT, candidate_count, device=device),
            }
            for start in range(0, candidate_count, CANDIDATES_PER_PHYSICS_BATCH):
                stop = start + CANDIDATES_PER_PHYSICS_BATCH
                q_env = env_major(q_proposals[:, start:stop])

                # Object先移到远处，direct q write后推进一步只为刷新articulation body transforms。
                far_pose = object_asset.data.default_root_state[:, :7].clone()
                far_pose[:, :3] = runtime_env.scene.env_origins + torch.tensor((0.0, 0.0, 1.5), device=device)
                far_pose[:, 3] = 1.0
                far_pose[:, 4:7] = 0.0
                object_asset.write_root_pose_to_sim(far_pose)
                object_asset.write_root_velocity_to_sim(torch.zeros(NUM_ENVS, 6, device=device))
                robot.write_joint_state_to_sim(q_env, torch.zeros_like(q_env))
                robot.set_joint_position_target(q_env)
                runtime_env.scene.write_data_to_sim()
                runtime_env.sim.step(render=False)
                runtime_env.scene.update(runtime_env.physics_dt)
                hand_pos_w, hand_quat_w = hand_semantic_pose_w(
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    frame.semantic_R_ha,
                    frame.semantic_p_ha,
                )
                offsets_w = robot.data.body_pos_w[:, tip_body_ids] - hand_pos_w.unsqueeze(1)
                inverse = hand_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
                tips_h = math_utils.quat_apply_inverse(inverse, offsets_w.reshape(-1, 3)).reshape(-1, 4, 3)
                if sobol_values is not None:
                    latent_env = env_major(sobol_values[:, start:stop])
                    envelope = initial_envelope(tips_h, active_tip_by_env, latent_env)
                else:
                    if direct_positions is None:
                        raise RuntimeError("direct CEM positions disappeared")
                    position_env = env_major(direct_positions[:, start:stop])
                    envelope = fixed_position_envelope(tips_h, active_tip_by_env, position_env)
                output["position"][:, start:stop] = asset_major(envelope.object_position_h_m)
                output["pair"][:, start:stop] = asset_major(envelope.non_thumb_pair)
                output["distances"][:, start:stop] = asset_major(envelope.tip_center_distances_m)
                output["sector_deg"][:, start:stop] = asset_major(envelope.sector_min_deg)
            return output

        def gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            r"""按每资产独立Top-K indices gather任意尾部shape tensor。"""

            view = indices.reshape(*indices.shape, *([1] * (values.ndim - 2)))
            expanded = view.expand(*indices.shape, *values.shape[2:])
            return torch.gather(values, 1, expanded)

        def geometry_top32(geometry: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            r"""初始256个Sobol proposals按strict-first score选择32项做物理筛选。"""

            score = geometry_score(geometry["joint_margin"], geometry["distances"], geometry["sector_deg"])
            # 初始穿透与自由落距对z形成窄折中；0.059 m只作统一proposal prior，最终仍由物理门决定。
            score -= torch.abs(geometry["position"][..., 2] - 0.059) / 0.002
            indices = torch.topk(score, k=STRICT_GOOD_PREGRASP_PHYSICS_TOP_K, dim=1, largest=True).indices
            selected = {name: gather_candidates(value, indices) for name, value in geometry.items()}
            selected["source_index"] = indices
            return selected

        def physical_screen(selected: dict[str, torch.Tensor], *, stage: int) -> dict[str, torch.Tensor]:
            r"""以$q_0=u_0$、零速度执行1 s cold reset并计算strict dynamic metrics。"""

            q_env = env_major(selected["q"])
            position_env = env_major(selected["position"])
            hand_pos_w, hand_quat_w = hand_semantic_pose_w(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                frame.semantic_R_ha,
                frame.semantic_p_ha,
            )
            object_quat_h = torch.zeros(NUM_ENVS, 4, device=device)
            object_quat_h[:, 0] = 1.0
            object_pos_w, object_quat_w = object_pose_w_from_hand(
                hand_pos_w,
                hand_quat_w,
                position_env,
                object_quat_h,
            )
            robot.write_joint_state_to_sim(q_env, torch.zeros_like(q_env))
            robot.set_joint_position_target(q_env)
            object_asset.write_root_pose_to_sim(torch.cat((object_pos_w, object_quat_w), dim=-1))
            object_asset.write_root_velocity_to_sim(torch.zeros(NUM_ENVS, 6, device=device))
            initial_pos_w = object_pos_w.clone()

            displacement_max = torch.zeros(NUM_ENVS, device=device)
            tilt_max = torch.zeros(NUM_ENVS, device=device)
            peak_linear = torch.zeros(NUM_ENVS, device=device)
            peak_total_angular = torch.zeros(NUM_ENVS, device=device)
            peak_off_axis_angular = torch.zeros(NUM_ENVS, device=device)
            penetration_max = torch.zeros(NUM_ENVS, device=device)
            penetration_by_sensor = torch.zeros(
                NUM_ENVS, len(CONTACT_LAYOUT.state_sensor_names), device=device
            )  # 诊断最大穿透来自PALM、哪一JOINT或TIP owner
            initial_penetration = torch.zeros(NUM_ENVS, device=device)
            initial_deepest_normal_w = torch.zeros(NUM_ENVS, 3, device=device)
            force_ema = torch.zeros(NUM_ENVS, len(CONTACT_LAYOUT.state_sensor_names), device=device)
            owner_contact_count = torch.zeros(NUM_ENVS, 21, device=device)
            palm_tail_count = torch.zeros(NUM_ENVS, device=device)
            policy_sample_count = PHYSICS_STEPS // POLICY_SUBSTEPS
            for physics_step in range(PHYSICS_STEPS):
                robot.set_joint_position_target(q_env)
                runtime_env.scene.write_data_to_sim()
                runtime_env.sim.step(render=False)
                runtime_env.scene.update(runtime_env.physics_dt)
                displacement = torch.linalg.vector_norm(object_asset.data.root_pos_w - initial_pos_w, dim=-1)
                displacement_max = torch.maximum(displacement_max, displacement)
                _, quaternion_h = object_pose_h_from_world(
                    hand_pos_w,
                    hand_quat_w,
                    object_asset.data.root_pos_w,
                    object_asset.data.root_quat_w,
                )
                object_z_h = math_utils.quat_apply(
                    quaternion_h,
                    torch.tensor((0.0, 0.0, 1.0), device=device).expand(NUM_ENVS, -1),
                )
                tilt = torch.rad2deg(torch.acos(object_z_h[:, 2].clamp(-1.0, 1.0)))
                tilt_max = torch.maximum(tilt_max, tilt)
                if physics_step < EARLY_PEAK_STEPS:
                    linear = torch.linalg.vector_norm(object_asset.data.root_lin_vel_w, dim=-1)
                    angular_h = math_utils.quat_apply_inverse(hand_quat_w, object_asset.data.root_ang_vel_w)
                    peak_linear = torch.maximum(peak_linear, linear)
                    peak_total_angular = torch.maximum(
                        peak_total_angular, torch.linalg.vector_norm(angular_h, dim=-1)
                    )
                    peak_off_axis_angular = torch.maximum(
                        peak_off_axis_angular, torch.linalg.vector_norm(angular_h[:, :2], dim=-1)
                    )

                policy_boundary = (physics_step + 1) % POLICY_SUBSTEPS == 0
                if physics_step == 0 or policy_boundary:
                    for sensor_index, sensor in enumerate(sensors.values()):
                        if physics_step == 0:
                            sensor_penetration, sensor_normal_w = deepest_contact_normal_per_env(
                                sensor, runtime_env.physics_dt
                            )
                            deeper = sensor_penetration > initial_penetration
                            initial_penetration = torch.maximum(initial_penetration, sensor_penetration)
                            initial_deepest_normal_w[deeper] = sensor_normal_w[deeper]
                        else:
                            sensor_penetration = contact_penetration_depth_per_env(sensor, runtime_env.physics_dt)
                        penetration_by_sensor[:, sensor_index] = torch.maximum(
                            penetration_by_sensor[:, sensor_index], sensor_penetration
                        )
                    penetration_max = penetration_by_sensor.amax(dim=-1)
                if not policy_boundary:
                    continue
                raw_force = torch.stack(
                    [sensor_contact_magnitude(runtime_env, name) for name in CONTACT_LAYOUT.state_sensor_names], dim=-1
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
                owner_contact_count += owner_bits.float() * active_owner_by_env
                policy_index = (physics_step + 1) // POLICY_SUBSTEPS - 1
                if policy_index >= policy_sample_count - TAIL_POLICY_SAMPLES:
                    palm_tail_count += sensor_bits[:, -1].float()

            result = dict(selected)
            initial_deepest_normal_h = math_utils.quat_apply_inverse(
                hand_quat_w, initial_deepest_normal_w
            )  # PhysX sensor→filter normal语义保持原符号，仅转换到hand frame
            result.update(
                {
                    "penetration_m": asset_major(penetration_max),
                    "penetration_by_sensor_m": asset_major(penetration_by_sensor),
                    "initial_penetration_m": asset_major(initial_penetration),
                    "initial_deepest_normal_h": asset_major(initial_deepest_normal_h),
                    "displacement_m": asset_major(displacement_max),
                    "tilt_deg": asset_major(tilt_max),
                    "peak_linear_m_s": asset_major(peak_linear),
                    "peak_angular_rad_s": asset_major(peak_total_angular),
                    "peak_off_axis_angular_rad_s": asset_major(peak_off_axis_angular),
                    "palm_fraction": asset_major(palm_tail_count / float(TAIL_POLICY_SAMPLES)),
                    "owner_contact_fraction": asset_major(owner_contact_count / float(policy_sample_count)),
                    "stage": torch.full(
                        (ASSET_COUNT, CANDIDATES_PER_PHYSICS_BATCH), stage, dtype=torch.int16, device=device
                    ),
                }
            )
            result["passed"] = strict_pass_mask(
                joint_margin=result["joint_margin"],
                distances=result["distances"],
                sector_deg=result["sector_deg"],
                penetration_m=result["penetration_m"],
                displacement_m=result["displacement_m"],
                tilt_deg=result["tilt_deg"],
                peak_linear_m_s=result["peak_linear_m_s"],
                peak_angular_rad_s=result["peak_angular_rad_s"],
                palm_fraction=result["palm_fraction"],
            )
            result["violation"] = normalized_gate_violation(
                joint_margin=result["joint_margin"],
                distances=result["distances"],
                sector_deg=result["sector_deg"],
                penetration_m=result["penetration_m"],
                displacement_m=result["displacement_m"],
                tilt_deg=result["tilt_deg"],
                peak_linear_m_s=result["peak_linear_m_s"],
                peak_angular_rad_s=result["peak_angular_rad_s"],
                palm_fraction=result["palm_fraction"],
            )
            return result

        def concatenate(results: list[dict[str, torch.Tensor]], name: str) -> torch.Tensor:
            r"""沿每资产candidate轴连接各physical stages的同名tensor。"""

            return torch.cat([result[name] for result in results], dim=1)

        def physical_quality(results: list[dict[str, torch.Tensor]]) -> torch.Tensor:
            r"""Strict pass优先，再按总violation、support、comfort和瞬态稳定性排列elite。"""

            passed = concatenate(results, "passed").float()
            violation = concatenate(results, "violation")
            palm = concatenate(results, "palm_fraction")
            margin = concatenate(results, "joint_margin")
            displacement = concatenate(results, "displacement_m")
            angular = concatenate(results, "peak_angular_rad_s")
            return 1000.0 * passed - 100.0 * violation + palm + margin - displacement / 0.005 - angular / 2.0

        # Stage 0：256 Sobol proposals -> geometry Top-32 -> full physics。
        sobol = sobol_bank(
            SELECTED_ROWS,
            candidate_count=STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES,
            seed=ARGS.seed,
            device=device,
        )
        q_initial, _ = initial_joint_candidates(
            lower_by_asset,
            upper_by_asset,
            active_by_asset,
            sobol,
            margin_fraction=0.11,
        )
        initial_geometry = realize_geometry(q_initial, sobol_values=sobol)
        initial_selected = geometry_top32(initial_geometry)
        initial_selected["refinement_target"] = torch.ones(
            ASSET_COUNT, CANDIDATES_PER_PHYSICS_BATCH, dtype=torch.bool, device=device
        )
        results = [physical_screen(initial_selected, stage=0)]
        counts = concatenate(results, "passed").sum(dim=1)
        print({"stage": "sobol_top32", "passed_assets": int((counts >= 8).sum()), "min_passed": int(counts.min())})

        # Stages 1..3：只把尚不足8项的assets标为eligible，其他env仅保持vectorized scene shape。
        rounds_executed = 0
        for round_index in range(1, ARGS.max_cem_rounds + 1):
            failing = counts < 8
            if not bool(failing.any().item()):
                break
            quality = physical_quality(results)
            elite_indices = torch.topk(
                quality,
                k=min(STRICT_GOOD_PREGRASP_CEM_ELITES, quality.shape[1]),
                dim=1,
                largest=True,
            ).indices
            q_all = concatenate(results, "q")
            position_all = concatenate(results, "position")
            initial_depth_all = concatenate(results, "initial_penetration_m")
            initial_normal_all = concatenate(results, "initial_deepest_normal_h")
            elite_q = gather_candidates(q_all, elite_indices)
            elite_position = gather_candidates(position_all, elite_indices)
            elite_depth = gather_candidates(initial_depth_all, elite_indices)
            elite_normal = gather_candidates(initial_normal_all, elite_indices)
            displacement_all = concatenate(results, "displacement_m")
            elite_displacement = gather_candidates(displacement_all, elite_indices)
            correction_distance = torch.where(
                elite_depth > 0.0,
                1.10 * elite_depth + 0.00025,
                torch.zeros_like(elite_depth),
            )  # measured overlap外加0.25 mm clearance
            elite_position = elite_position - elite_normal * correction_distance.unsqueeze(-1)
            settle_feedback = torch.where(
                elite_depth <= 0.0005,
                torch.clamp(elite_displacement - 0.0045, min=0.0, max=0.0030),
                torch.zeros_like(elite_displacement),
            )
            elite_position[..., 2] -= settle_feedback  # 无穿透但落距偏大时降低proposal center
            elite_position[..., 0].clamp_(-0.060, 0.060)
            elite_position[..., 1].clamp_(0.030, 0.140)
            elite_position[..., 2].clamp_(0.055, 0.065)
            q_cem, position_cem, _ = low_rank_cem_candidates(
                elite_q,
                elite_position,
                lower_by_asset,
                upper_by_asset,
                active_by_asset,
                candidate_count=STRICT_GOOD_PREGRASP_CEM_CANDIDATES,
                seed=ARGS.seed,
                round_index=round_index,
                asset_keys=SELECTED_ROWS,
            )
            # 已找到strict或violation≤0.35的近门态时，下一轮前96项做微扰复验，扩展窄稳定盆而不增加预算。
            passed_so_far = concatenate(results, "passed")
            violation_so_far = concatenate(results, "violation")
            q_so_far = concatenate(results, "q")
            position_so_far = concatenate(results, "position")
            span = (upper_by_asset - lower_by_asset).clamp_min(1.0e-6)
            comfortable_lower = lower_by_asset + 0.11 * span
            comfortable_upper = upper_by_asset - 0.11 * span
            for asset_index in torch.nonzero(failing, as_tuple=False).flatten().tolist():
                basin_indices = torch.nonzero(
                    passed_so_far[asset_index] | (violation_so_far[asset_index] <= 0.35), as_tuple=False
                ).flatten()
                if basin_indices.numel() == 0:
                    continue
                basin_indices = basin_indices[
                    torch.argsort(violation_so_far[asset_index, basin_indices], descending=False)
                ][:16]
                exploit_count = min(96, STRICT_GOOD_PREGRASP_CEM_CANDIDATES)
                center_indices = basin_indices[
                    torch.arange(exploit_count, device=device) % basin_indices.numel()
                ]
                generator = torch.Generator(device=device).manual_seed(
                    ARGS.seed + round_index * 10_000_019 + int(SELECTED_ROWS[asset_index]) * 104729
                )
                q_noise = torch.randn(exploit_count, 16, generator=generator, device=device) * 5.0e-4
                q_exploit = q_so_far[asset_index, center_indices] + q_noise * active_by_asset[asset_index]
                q_exploit = torch.maximum(
                    torch.minimum(q_exploit, comfortable_upper[asset_index]), comfortable_lower[asset_index]
                ) * active_by_asset[asset_index]
                position_noise = torch.randn(exploit_count, 3, generator=generator, device=device)
                position_noise *= torch.tensor((5.0e-5, 5.0e-5, 2.5e-5), device=device)
                position_exploit = position_so_far[asset_index, center_indices].clone()
                selected_displacement = displacement_all[asset_index, center_indices]
                selected_depth = initial_depth_all[asset_index, center_indices]
                lower_for_settle = (~passed_so_far[asset_index, center_indices]) & (selected_depth <= 0.0005)
                position_exploit[:, 2] -= torch.where(
                    lower_for_settle,
                    torch.clamp(selected_displacement - 0.0045, min=0.0, max=0.0015),
                    torch.zeros_like(selected_displacement),
                )
                position_exploit += position_noise
                q_cem[asset_index, :exploit_count] = q_exploit
                position_cem[asset_index, :exploit_count] = position_exploit
            cem_geometry = realize_geometry(q_cem, direct_positions=position_cem)
            # CEM预算中的128项全部执行物理筛选；32-env/asset scene连续处理四个candidate slices。
            for start in range(0, STRICT_GOOD_PREGRASP_CEM_CANDIDATES, CANDIDATES_PER_PHYSICS_BATCH):
                stop = start + CANDIDATES_PER_PHYSICS_BATCH
                cem_selected = {name: value[:, start:stop] for name, value in cem_geometry.items()}
                cem_selected["source_index"] = torch.arange(start, stop, device=device).unsqueeze(0).expand(
                    ASSET_COUNT, -1
                )
                cem_selected["refinement_target"] = failing.unsqueeze(1).expand(
                    -1, CANDIDATES_PER_PHYSICS_BATCH
                )
                screened = physical_screen(cem_selected, stage=round_index)
                screened["passed"] &= screened["refinement_target"]
                results.append(screened)
            rounds_executed = round_index
            counts = concatenate(results, "passed").sum(dim=1)
            print(
                {
                    "stage": f"cem_{round_index}_full128",
                    "target_assets": int(failing.sum()),
                    "passed_assets": int((counts >= 8).sum()),
                    "min_passed": int(counts.min()),
                },
                flush=True,
            )

        # Candidate evidence先落盘；即使未覆盖80手，也保留固定预算下的最早失败边界。
        evidence_root = ARGS.evidence.resolve()
        evidence_root.mkdir(parents=True, exist_ok=True)
        evidence_names = (
            "q",
            "position",
            "pair",
            "distances",
            "joint_margin",
            "sector_deg",
            "penetration_m",
            "penetration_by_sensor_m",
            "initial_penetration_m",
            "initial_deepest_normal_h",
            "displacement_m",
            "tilt_deg",
            "peak_linear_m_s",
            "peak_angular_rad_s",
            "peak_off_axis_angular_rad_s",
            "palm_fraction",
            "owner_contact_fraction",
            "passed",
            "violation",
            "stage",
            "source_index",
            "refinement_target",
        )
        evidence = {name: concatenate(results, name).detach().cpu().numpy() for name in evidence_names}
        np.savez_compressed(
            evidence_root / "candidates.npz",
            dataset_rows=np.asarray(SELECTED_ROWS, dtype=np.int64),
            **evidence,
        )

        passed_all = concatenate(results, "passed")
        stage_all = concatenate(results, "stage")
        failed_assets = []
        for asset_index, count in enumerate(passed_all.sum(dim=1).tolist()):
            if count < 8:
                failed_assets.append(
                    {
                        "asset_index": asset_index,
                        "dataset_row": SELECTED_ROWS[asset_index],
                        "asset_id": ASSET_BINDING.source_assets[asset_index].asset_id,
                        "strict_passed_candidates": int(count),
                        "cell_id": ASSET_BINDING.morphology_cell_ids[asset_index],
                    }
                )

        asset_candidate_counts = [
            {
                "asset_index": asset_index,
                "dataset_row": SELECTED_ROWS[asset_index],
                "asset_id": ASSET_BINDING.source_assets[asset_index].asset_id,
                "cell_id": ASSET_BINDING.morphology_cell_ids[asset_index],
                "strict_passed_candidates": int(passed_all[asset_index].sum().item()),
                "passed_by_stage": {
                    str(stage): int((passed_all[asset_index] & (stage_all[asset_index] == stage)).sum().item())
                    for stage in range(rounds_executed + 1)
                },
            }
            for asset_index in range(ASSET_COUNT)
        ]

        # 逐gate统计所有eligible physical candidates，帮助区分proposal不足与具体动态门失败。
        target = concatenate(results, "refinement_target")
        gate_failures = {
            "joint_margin": int(((concatenate(results, "joint_margin") < 0.10) & target).sum()),
            "tip_center_distance": int(((concatenate(results, "distances").amax(dim=-1) > 0.10) & target).sum()),
            "sector": int(((concatenate(results, "sector_deg") < 30.0) & target).sum()),
            "penetration": int(((concatenate(results, "penetration_m") > 0.0005) & target).sum()),
            "displacement": int(((concatenate(results, "displacement_m") > 0.005) & target).sum()),
            "tilt": int(((concatenate(results, "tilt_deg") > 10.0) & target).sum()),
            "peak_linear_velocity": int(((concatenate(results, "peak_linear_m_s") > 0.25) & target).sum()),
            "peak_total_angular_velocity": int(
                ((concatenate(results, "peak_angular_rad_s") > 2.0) & target).sum()
            ),
            "palm_contact_fraction": int(((concatenate(results, "palm_fraction") < 0.5) & target).sum()),
        }

        published: list[dict[str, Any]] = []
        entries: list[GoodPregraspEntry] = []
        if not failed_assets and ASSET_COUNT == ASSETS_PER_RUN and (not DEVELOPMENT_RUN or ARGS.publish_selection):
            finger_names = ("index", "middle", "ring", "thumb")
            all_values = {name: concatenate(results, name) for name in evidence_names}
            quality = physical_quality(results)
            for asset_index, dataset_row in enumerate(SELECTED_ROWS):
                passed_indices = torch.nonzero(all_values["passed"][asset_index], as_tuple=False).flatten()
                ranked = passed_indices[torch.argsort(quality[asset_index, passed_indices], descending=True)][:8]
                members: list[GoodPregraspMember] = []
                for rank, index_tensor in enumerate(ranked):
                    candidate_index = int(index_tensor.item())
                    pair = all_values["pair"][asset_index, candidate_index].tolist()
                    q_values = all_values["q"][asset_index, candidate_index].tolist()
                    position = all_values["position"][asset_index, candidate_index].tolist()
                    distances = all_values["distances"][asset_index, candidate_index].tolist()
                    metrics = GoodPregraspMetrics(
                        joint_limit_margin_fraction=float(all_values["joint_margin"][asset_index, candidate_index]),
                        envelope_fingers=("thumb", finger_names[pair[0]], finger_names[pair[1]]),
                        envelope_sector_min_deg=float(all_values["sector_deg"][asset_index, candidate_index]),
                        envelope_tip_center_distance_m=(
                            float(distances[0]),
                            float(distances[1]),
                            float(distances[2]),
                        ),
                        penetration_depth_max_m=float(all_values["penetration_m"][asset_index, candidate_index]),
                        object_displacement_max_m=float(all_values["displacement_m"][asset_index, candidate_index]),
                        object_tilt_max_deg=float(all_values["tilt_deg"][asset_index, candidate_index]),
                        peak_linear_velocity_m_s=float(all_values["peak_linear_m_s"][asset_index, candidate_index]),
                        peak_off_axis_angular_velocity_rad_s=float(
                            all_values["peak_off_axis_angular_rad_s"][asset_index, candidate_index]
                        ),
                        palm_contact_fraction=float(all_values["palm_fraction"][asset_index, candidate_index]),
                        owner_contact_fraction=tuple(
                            float(value)
                            for value in all_values["owner_contact_fraction"][asset_index, candidate_index].tolist()
                        ),
                        peak_angular_velocity_rad_s=float(
                            all_values["peak_angular_rad_s"][asset_index, candidate_index]
                        ),
                    )
                    candidate = GoodPregraspCandidate(
                        q_state_rad=tuple(float(value) for value in q_values),
                        q_target_rad=tuple(float(value) for value in q_values),
                        active_joint_mask=tuple(bool(value) for value in ASSET_BINDING.active_joint_masks[asset_index]),
                        object_position_h_m=(float(position[0]), float(position[1]), float(position[2])),
                    )
                    score = (
                        metrics.palm_contact_fraction,
                        metrics.joint_limit_margin_fraction,
                        metrics.envelope_sector_min_deg / 180.0,
                        -max(metrics.envelope_tip_center_distance_m),
                        -metrics.object_displacement_max_m,
                        -metrics.object_tilt_max_deg / 10.0,
                        -metrics.peak_linear_velocity_m_s,
                        -float(metrics.peak_angular_velocity_rad_s or 0.0),
                        -metrics.penetration_depth_max_m,
                    )
                    members.append(GoodPregraspMember(rank=rank, candidate=candidate, metrics=metrics, selection_score=score))
                artifact = ASSET_BINDING.canonical_artifacts[asset_index]
                source_asset = ASSET_BINDING.source_assets[asset_index]
                key = GoodPregraspKey(
                    asset_id=source_asset.asset_id,
                    source_content_hash=artifact.source_content_hash,
                    physical_geometry_hash=artifact.physical_geometry_hash,
                    canonical_schema_digest=artifact.schema_digest,
                    routing_digest=active_mask_digest(artifact.routing.active_joint_mask),
                    object_asset_id="DexCube",
                    object_asset_sha256=RESOLVED_DEX_CUBE_SHA256,
                    object_scale=1.1,
                    physics_identity_digest=STRICT_GOOD_PREGRASP_PHYSICS_DIGEST,
                    generation_identity_digest=STRICT_GOOD_PREGRASP_GENERATION_DIGEST,
                )
                entry = GoodPregraspEntry(key=key, members=tuple(members))
                MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)
                entries.append(entry)

            # 全部80 entries先在内存验证完成，再开始atomic per-entry/index发布。
            catalog = GoodPregraspCatalog(ARGS.catalog.resolve())
            for dataset_row, entry in zip(SELECTED_ROWS, entries, strict=True):
                index_entry = catalog.publish(entry)
                published.append(
                    {
                        "dataset_row": dataset_row,
                        "asset_id": entry.key.asset_id,
                        "strict_passed_candidates": int(
                            passed_all[SELECTED_ROWS.index(dataset_row)].sum().item()
                        ),
                        "key_digest": index_entry.key_digest,
                        "entry_digest": index_entry.entry_digest,
                    }
                )

        summary = {
            "artifact_type": "anymani.good_pregrasp.strict_generation_summary",
            "schema_version": "1.0.0",
            "selection_path": str(ARGS.selection),
            "dataset_rows": list(SELECTED_ROWS),
            "object": {"asset_id": "DexCube", "scale": 1.1, "orientation_h_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "generation_identity": STRICT_GOOD_PREGRASP_GENERATION_IDENTITY,
            "generation_identity_digest": STRICT_GOOD_PREGRASP_GENERATION_DIGEST,
            "physics_identity_digest": STRICT_GOOD_PREGRASP_PHYSICS_DIGEST,
            "strict_gate_digest": MVP80_STRICT_GOOD_PREGRASP_GATE.digest,
            "sobol_proposals_per_asset": STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES,
            "initial_physics_candidates_per_asset": STRICT_GOOD_PREGRASP_PHYSICS_TOP_K,
            "cem_physics_candidates_per_target_asset_round": STRICT_GOOD_PREGRASP_CEM_CANDIDATES,
            "cem_rounds_executed": rounds_executed,
            "eligible_physical_candidates": int(target.sum().item()),
            "strict_passed_candidates": int(passed_all.sum().item()),
            "covered_asset_count": int((passed_all.sum(dim=1) >= 8).sum().item()),
            "published_count": len(published),
            "failed_count": len(failed_assets),
            "failed": failed_assets,
            "asset_candidate_counts": asset_candidate_counts,
            "gate_failure_counts": gate_failures,
            "published": published,
            "catalog_root": str(ARGS.catalog.resolve()) if published else None,
            "candidate_npz": str((evidence_root / "candidates.npz").resolve()),
            "formal_all_80_top8_passed": len(published) == 80,
        }
        summary_path = evidence_root / "summary.json"
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(summary_path)
        print(
            {
                "summary": str(summary_path),
                "covered_assets": summary["covered_asset_count"],
                "published": len(published),
                "failed": len(failed_assets),
            },
            flush=True,
        )
        if DEVELOPMENT_RUN:
            return 0 if not failed_assets else 3  # development prefix不发布formal catalog
        return 0 if len(published) == ASSETS_PER_RUN else 3
    finally:
        runtime_env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)  # Kit shutdown完成后再发布科学coverage退出码
