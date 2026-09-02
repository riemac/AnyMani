r"""两只mirror hand在一个absolute cube scale下的pregrasp物理接口probe。

本脚本不搜索candidate、不写cache，也不修改task MDP。它只验证P0001后续认证器依赖的底层事实：

* scale必须在scene startup前写入object spawn cfg；
* canonical artifact与真实cube bytes能形成稳定identity；
* object/hand实际mass、inertia和COM可从PhysX view读取；
* hand-frame $T_{ho}$ 与world pose可以无损往返；
* object-filtered contact view能返回normal force与separation；
* implicit actuator的computed/applied effort在一次policy step后可读取。

默认formal dataset rows ``0,16`` 是同一``t3_i3_m2_r2`` topology的left/right mirror pair。三个scale
``1.1/1.2/1.25``应分别启动独立进程，避免同一PhysX scene中修改prestartup-only scale。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import cast


def _parse_args() -> argparse.Namespace:
    r"""解析单scene的两个dataset rows与absolute cube scale。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, choices=(1.1, 1.2, 1.25), required=True)
    parser.add_argument("--rows", type=str, default="0,16", help="Exactly two formal dataset rows.")
    parser.add_argument("--output", type=Path, required=True, help="Project-local JSON evidence path.")
    return parser.parse_args()


def main() -> int:
    r"""启动一个scale scene并输出所有身份、frame、contact与effort事实。"""

    args = _parse_args()
    rows = tuple(int(item.strip()) for item in args.rows.split(",") if item.strip())
    if len(rows) != 2 or len(set(rows)) != 2:
        raise ValueError("--rows must contain exactly two distinct dataset rows")
    os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in rows)
    os.environ["ANYMANI_HETERO_NUM_ENVS"] = "2"

    # AppLauncher必须早于IsaacLab/pxr/runtime imports；每个scale由一个独立进程拥有SimulationContext。
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    env = None
    try:
        import isaaclab.sim as sim_utils
        import torch
        from anymani.pregrasp.isaac_runtime import (
            contact_penetration_depth_per_env,
            contact_separation_summary,
            file_sha256,
            hand_semantic_pose_w,
            object_pose_h_from_world,
            object_pose_w_from_hand,
        )
        from anymani.tasks.hetero.config.generated.pregrasp_harness_env_cfg import (
            GeneratedPregraspHarnessEnvCfg,
        )
        from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING, CONTACT_LAYOUT
        from isaaclab.assets import Articulation, RigidObject
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.sensors import ContactSensor
        from isaaclab.utils.assets import retrieve_file_path

        # Scale是absolute USD scale；普通episode event不能改变已经创建的collision geometry。
        cfg = GeneratedPregraspHarnessEnvCfg()
        cfg.scene.replicate_physics = False
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

        # 一次零action policy step触发action term、actuator、ContactSensor与effort buffers的正式生命周期。
        zero_action = torch.zeros(2, robot.num_joints, device=runtime_env.device)
        env.step(zero_action)

        # 解析Nucleus/object路径为真实本地bytes；URL本身不足以作为cache identity。
        object_source = str(object_spawn.usd_path)
        object_local_path = Path(retrieve_file_path(object_source)).resolve()
        object_sha256 = file_sha256(object_local_path)

        # 从$T_{wa}$与静态$T_{ah}=T_{ha}^{-1}$构造hand semantic world pose。
        frame = ASSET_BINDING.hand_spawn_cfg.frame
        pos_wh, quat_wh = hand_semantic_pose_w(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            frame.semantic_R_ha,
            frame.semantic_p_ha,
        )
        pos_ho, quat_ho = object_pose_h_from_world(
            pos_wh,
            quat_wh,
            object_asset.data.root_pos_w,
            object_asset.data.root_quat_w,
        )
        reconstructed_pos, reconstructed_quat = object_pose_w_from_hand(
            pos_wh,
            quat_wh,
            pos_ho,
            quat_ho,
        )
        position_roundtrip_error = torch.linalg.vector_norm(reconstructed_pos - object_asset.data.root_pos_w, dim=-1)
        quaternion_alignment = torch.abs(torch.sum(reconstructed_quat * object_asset.data.root_quat_w, dim=-1))
        if torch.any(position_roundtrip_error > 1.0e-6) or torch.any(1.0 - quaternion_alignment > 1.0e-6):
            raise RuntimeError("hand-frame T_ho roundtrip exceeds 1e-6 pose tolerance")

        # 24个object-filtered sensor逐项读取detailed contact separation；palm合法且单独报告。
        sensor_names = (
            *CONTACT_LAYOUT.fingertip_sensor_names,
            *CONTACT_LAYOUT.finger_non_tip_sensor_names,
            CONTACT_LAYOUT.palm_sensor_name,
        )
        contact_summary = {}
        penetration_per_env = torch.zeros(2, device=runtime_env.device)
        for sensor_name in sensor_names:
            sensor = cast(ContactSensor, runtime_env.scene[sensor_name])
            contact_summary[sensor_name] = contact_separation_summary(sensor, runtime_env.physics_dt)
            penetration_per_env = torch.maximum(
                penetration_per_env,
                contact_penetration_depth_per_env(sensor, runtime_env.physics_dt),
            )

        # 实际PhysX mass/inertia在scale与所有startup setter完成后读取，不能只用cfg或default density推断。
        object_mass = object_asset.root_physx_view.get_masses().detach().cpu()
        object_inertia = object_asset.root_physx_view.get_inertias().detach().cpu()
        object_com = object_asset.root_physx_view.get_coms().detach().cpu()
        hand_mass = robot.root_physx_view.get_masses().detach().cpu()
        hand_inertia = robot.root_physx_view.get_inertias().detach().cpu()

        # computed/applied torque来自actuator lifecycle；implicit actuator下是项目可用的近似effort证据。
        computed_torque = robot.data.computed_torque.detach().cpu()
        applied_torque = robot.data.applied_torque.detach().cpu()
        artifacts = []
        for dataset_row, source_asset, artifact in zip(
            ASSET_BINDING.dataset_rows,
            ASSET_BINDING.source_assets,
            ASSET_BINDING.canonical_artifacts,
        ):
            artifacts.append(
                {
                    "dataset_row": int(dataset_row),
                    "asset_id": source_asset.asset_id,
                    "source_content_hash": artifact.source_content_hash,
                    "physical_geometry_hash": artifact.physical_geometry_hash,
                    "canonical_schema_digest": artifact.schema_digest,
                    "active_joint_mask": list(artifact.routing.active_joint_mask),
                }
            )

        summary = {
            "artifact_type": "anymani.pregrasp.physics_probe",
            "schema_version": "1.0.0",
            "dataset_rows": list(rows),
            "scale": args.scale,
            "object_source": object_source,
            "object_local_path": str(object_local_path),
            "object_sha256": object_sha256,
            "object_mass_kg": object_mass.reshape(2, -1).tolist(),
            "object_inertia_kg_m2": object_inertia.reshape(2, -1).tolist(),
            "object_com": object_com.reshape(2, -1).tolist(),
            "hand_mass_kg": hand_mass.reshape(2, -1).tolist(),
            "hand_inertia_kg_m2": hand_inertia.reshape(2, -1).tolist(),
            "candidate_object_position_h_m": pos_ho.detach().cpu().tolist(),
            "candidate_object_orientation_h_wxyz": quat_ho.detach().cpu().tolist(),
            "frame_roundtrip_position_error_m": position_roundtrip_error.detach().cpu().tolist(),
            "frame_roundtrip_quaternion_one_minus_abs_dot": (1.0 - quaternion_alignment).detach().cpu().tolist(),
            "computed_torque_rms_N_m": torch.sqrt(computed_torque.square().mean(dim=-1)).tolist(),
            "applied_torque_rms_N_m": torch.sqrt(applied_torque.square().mean(dim=-1)).tolist(),
            "contact_sensors": contact_summary,
            "penetration_depth_max_per_env_m": penetration_per_env.detach().cpu().tolist(),
            "canonical_artifacts": artifacts,
            "physics_identity": {
                "physics_dt_s": runtime_env.physics_dt,
                "policy_dt_s": runtime_env.step_dt,
                "solver_position_iterations": object_spawn.rigid_props.solver_position_iteration_count,
                "solver_velocity_iterations": object_spawn.rigid_props.solver_velocity_iteration_count,
                "contact_force_threshold_N": 0.25,
                "contact_ema_alpha": 0.5,
                "effort_source": "implicit_actuator_computed_and_applied_torque",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True), flush=True)
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
