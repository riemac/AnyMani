r"""在原N000资产上校准manual pregrasp与zero-q控制的数值接触/稳定性。"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import cast


def _parse_args() -> argparse.Namespace:
    r"""解析6 s正/负控制证据输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    r"""以两个同资产env做manual-default与zero-q六秒PD hold。"""

    args = _parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    env = None
    try:
        import anymani.tasks.gm  # noqa: F401
        import gymnasium as gym
        import isaaclab.utils.math as math_utils
        import torch
        from anymani.tasks.gm.config.single_asset.single_asset_env_cfg import GM_SINGLE_ASSET_CONTACT_LAYOUT
        from anymani.tasks.gm.config.single_asset.tactile_rotation_env_cfg import GmTactileRotationHistory30EnvCfg
        from anymani.tasks.gm.contact_sensors import sensor_total_force_w
        from isaaclab.assets import Articulation, RigidObject
        from isaaclab.envs import ManagerBasedRLEnv

        cfg = GmTactileRotationHistory30EnvCfg()
        cfg.scene.num_envs = 2
        env = gym.make("AnyMani-GM-SingleAsset-TactileRotation-History30Obs-v0", cfg=cfg)
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        env.reset()
        robot = cast(Articulation, runtime_env.scene["robot"])
        object_asset = cast(RigidObject, runtime_env.scene["object"])
        tip_body_ids, _ = robot.find_bodies(list(GM_SINGLE_ASSET_CONTACT_LAYOUT.fingertip_link_names), preserve_order=True)
        q_target = robot.data.default_joint_pos.clone()
        q_target[1].zero_()  # negative control；manual candidate保留env0 default preset
        robot.write_joint_state_to_sim(q_target, torch.zeros_like(q_target))
        robot.set_joint_position_target(q_target)
        initial_pos = object_asset.data.root_pos_w.clone()
        tip_ge2 = []
        bad = []
        palm = []
        drift = []
        tip_distance = []
        linear_sq = []
        angular_sq = []
        orientation = []
        tracking_sq = []
        effort_sq = []
        initial_quat = object_asset.data.root_quat_w.clone()
        for step in range(120 * 6):
            robot.set_joint_position_target(q_target)
            runtime_env.scene.write_data_to_sim()
            runtime_env.sim.step(render=False)
            runtime_env.scene.update(runtime_env.physics_dt)
            if (step + 1) % 6:
                continue
            tip_force = torch.stack(
                [torch.linalg.vector_norm(sensor_total_force_w(runtime_env, name), dim=-1) for name in GM_SINGLE_ASSET_CONTACT_LAYOUT.fingertip_sensor_names],
                dim=-1,
            )
            non_tip_force = torch.stack(
                [torch.linalg.vector_norm(sensor_total_force_w(runtime_env, name), dim=-1) for name in GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_non_tip_sensor_names],
                dim=-1,
            )
            palm_force = torch.linalg.vector_norm(
                sensor_total_force_w(runtime_env, GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_sensor_name), dim=-1
            )
            tip_ge2.append(((tip_force > 0.25).sum(dim=-1) >= 2).float())
            bad.append((non_tip_force > 0.25).any(dim=-1).float())
            palm.append((palm_force > 0.25).float())
            drift.append(torch.linalg.vector_norm(object_asset.data.root_pos_w - initial_pos, dim=-1))
            linear_sq.append(object_asset.data.root_lin_vel_w.square().sum(dim=-1))
            angular_sq.append(object_asset.data.root_ang_vel_w.square().sum(dim=-1))
            orientation.append(math_utils.quat_error_magnitude(initial_quat, object_asset.data.root_quat_w))
            tracking_sq.append((robot.data.joint_pos - q_target).square().mean(dim=-1))
            effort_sq.append(robot.data.computed_torque.square().mean(dim=-1))
            tip_distance.append(
                torch.linalg.vector_norm(
                    robot.data.body_pos_w[:, tip_body_ids] - object_asset.data.root_pos_w.unsqueeze(1),
                    dim=-1,
                ).mean(dim=-1)
            )
        summary = {
            "manual": {
                "tip_ge2_fraction": float(torch.stack(tip_ge2)[:, 0].mean().item()),
                "finger_non_tip_fraction": float(torch.stack(bad)[:, 0].mean().item()),
                "palm_fraction": float(torch.stack(palm)[:, 0].mean().item()),
                "drift_max_m": float(torch.stack(drift)[:, 0].amax().item()),
                "tip_center_distance_mean_m": float(torch.stack(tip_distance)[:, 0].mean().item()),
                "linear_velocity_rms_m_s": float(torch.sqrt(torch.stack(linear_sq)[:, 0].mean()).item()),
                "angular_velocity_rms_rad_s": float(torch.sqrt(torch.stack(angular_sq)[:, 0].mean()).item()),
                "orientation_drift_max_rad": float(torch.stack(orientation)[:, 0].amax().item()),
                "tracking_error_rms_rad": float(torch.sqrt(torch.stack(tracking_sq)[:, 0].mean()).item()),
                "effort_rms_N_m": float(torch.sqrt(torch.stack(effort_sq)[:, 0].mean()).item()),
            },
            "zero": {
                "tip_ge2_fraction": float(torch.stack(tip_ge2)[:, 1].mean().item()),
                "finger_non_tip_fraction": float(torch.stack(bad)[:, 1].mean().item()),
                "palm_fraction": float(torch.stack(palm)[:, 1].mean().item()),
                "drift_max_m": float(torch.stack(drift)[:, 1].amax().item()),
                "tip_center_distance_mean_m": float(torch.stack(tip_distance)[:, 1].mean().item()),
                "linear_velocity_rms_m_s": float(torch.sqrt(torch.stack(linear_sq)[:, 1].mean()).item()),
                "angular_velocity_rms_rad_s": float(torch.sqrt(torch.stack(angular_sq)[:, 1].mean()).item()),
                "orientation_drift_max_rad": float(torch.stack(orientation)[:, 1].amax().item()),
                "tracking_error_rms_rad": float(torch.sqrt(torch.stack(tracking_sq)[:, 1].mean()).item()),
                "effort_rms_N_m": float(torch.sqrt(torch.stack(effort_sq)[:, 1].mean()).item()),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
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
