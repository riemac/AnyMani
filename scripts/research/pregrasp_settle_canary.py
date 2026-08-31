r"""八组16-asset palm-supported reset三控制物理canary。

同一canonical scene中，每个formal asset row有三个独立env：template replica写入N000人工非零seed，
zero replica保持全零q，unsupported replica把cube抬高15 cm形成真实掉落负控。脚本绕过policy，只持续
下发固定PD target并读取object-filtered contact/物体状态，形成无视觉数值证据。
"""

from __future__ import annotations

import json
import os
import traceback
from collections import Counter
from dataclasses import asdict
from typing import cast

BALANCED_DATASET_ROWS = (416, 417, 352, 353, 0, 1, 64, 65, 432, 433, 368, 369, 16, 17, 80, 81)
"""handedness×3/4TIP×thumb3/4DoF每组两个formal dataset rows。"""

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
)
"""N000 manual preset映射到canonical depth-major `index,middle,ring,thumb` JOINT axis，单位rad。"""


def main() -> int:
    r"""运行固定PD hold并比较非零seed与zero-q negative control。"""

    os.environ["ANYMANI_HETEROGENEOUS_ASSET_ROWS"] = ",".join(str(row) for row in BALANCED_DATASET_ROWS)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    env = None
    try:
        import anymani.tasks.gm  # noqa: F401
        import gymnasium as gym
        import torch
        from anymani.pregrasp.schema import PregraspAcceptanceCfg, PregraspMetrics, evaluate_pregrasp
        from anymani.tasks.gm.config.heterogeneous_asset.asset_runtime import (
            HETEROGENEOUS_ACTIVE_MASK_ROWS,
            HETEROGENEOUS_CONTACT_LAYOUT,
            HETEROGENEOUS_SOURCE_DATASET_ROWS,
        )
        from anymani.tasks.gm.config.heterogeneous_asset.tactile_rotation_env_cfg import (
            HeterogeneousTactileRotationEnvCfg,
        )
        from anymani.tasks.gm.contact_sensors import sensor_total_force_w
        from isaaclab.assets import Articulation, RigidObject
        from isaaclab.envs import ManagerBasedRLEnv

        unique_assets = len(BALANCED_DATASET_ROWS)
        cfg = HeterogeneousTactileRotationEnvCfg()
        cfg.scene.num_envs = unique_assets * 3  # template、zero-q、unsupported-object三组同序replicas
        env = gym.make("AnyMani-GM-HeterogeneousAsset-TactileRotation-v0", cfg=cfg)
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        env.reset()
        robot = cast(Articulation, runtime_env.scene["robot"])
        object_asset = cast(RigidObject, runtime_env.scene["object"])
        tip_body_ids, _ = robot.find_bodies(list(HETEROGENEOUS_CONTACT_LAYOUT.fingertip_link_names), preserve_order=True)
        active = torch.tensor(
            [
                *HETEROGENEOUS_ACTIVE_MASK_ROWS,
                *HETEROGENEOUS_ACTIVE_MASK_ROWS,
                *HETEROGENEOUS_ACTIVE_MASK_ROWS,
            ],
            device=runtime_env.device,
            dtype=torch.bool,
        )
        candidate_seed = torch.tensor(N000_CANONICAL_Q, device=runtime_env.device).expand(unique_assets, -1)
        q_target = torch.zeros(unique_assets * 3, 16, device=runtime_env.device)
        q_target[:unique_assets] = candidate_seed * active[:unique_assets]
        limits = robot.data.soft_joint_pos_limits
        q_target = torch.maximum(torch.minimum(q_target, limits[:, :, 1]), limits[:, :, 0])
        q_target = q_target * active.to(dtype=q_target.dtype)
        q_velocity = torch.zeros_like(q_target)
        robot.write_joint_state_to_sim(q_target, q_velocity)
        robot.set_joint_position_target(q_target)
        object_pose = object_asset.data.root_pose_w.clone()
        object_pose[2 * unique_assets :, 2] += 0.15  # unsupported负控：cube离开palm后自由下落
        object_asset.write_root_pose_to_sim(object_pose)
        object_asset.write_root_velocity_to_sim(torch.zeros(unique_assets * 3, 6, device=runtime_env.device))
        initial_object_pos = object_pose[:, :3].clone()

        snapshots = 20  # 120Hz physics下每6步采一次，共1 s palm-supported settle
        tip_ge_2 = []
        non_tip_bad = []
        palm_contact = []
        anchor_distance = []
        linear_speed_sq = []
        angular_speed_sq = []
        tracking_error_sq = []
        tip_center_distance = []
        for physics_step in range(snapshots * 6):
            robot.set_joint_position_target(q_target)
            runtime_env.scene.write_data_to_sim()
            runtime_env.sim.step(render=False)
            runtime_env.scene.update(runtime_env.physics_dt)
            if (physics_step + 1) % 6:
                continue
            tip_force = torch.stack(
                [torch.linalg.vector_norm(sensor_total_force_w(runtime_env, name), dim=-1) for name in HETEROGENEOUS_CONTACT_LAYOUT.fingertip_sensor_names],
                dim=-1,
            )
            non_tip_force = torch.stack(
                [torch.linalg.vector_norm(sensor_total_force_w(runtime_env, name), dim=-1) for name in HETEROGENEOUS_CONTACT_LAYOUT.finger_non_tip_sensor_names],
                dim=-1,
            )
            palm_force = torch.linalg.vector_norm(
                sensor_total_force_w(runtime_env, HETEROGENEOUS_CONTACT_LAYOUT.palm_sensor_name), dim=-1
            )
            tip_ge_2.append(((tip_force > 0.25).sum(dim=-1) >= 2).float())
            non_tip_bad.append((non_tip_force > 0.25).any(dim=-1).float())
            palm_contact.append((palm_force > 0.25).float())
            anchor_distance.append(torch.linalg.vector_norm(object_asset.data.root_pos_w - initial_object_pos, dim=-1))
            linear_speed_sq.append(object_asset.data.root_lin_vel_w.square().sum(dim=-1))
            angular_speed_sq.append(object_asset.data.root_ang_vel_w.square().sum(dim=-1))
            tracking_error_sq.append(((robot.data.joint_pos - q_target) * active).square().sum(dim=-1) / active.sum(dim=-1))
            tip_center_distance.append(
                torch.linalg.vector_norm(
                    robot.data.body_pos_w[:, tip_body_ids] - object_asset.data.root_pos_w.unsqueeze(1),
                    dim=-1,
                ).mean(dim=-1)
            )

        tip_fraction = torch.stack(tip_ge_2).mean(dim=0)
        bad_fraction = torch.stack(non_tip_bad).mean(dim=0)
        palm_fraction = torch.stack(palm_contact).mean(dim=0)
        drift_max = torch.stack(anchor_distance).amax(dim=0)
        lin_rms = torch.sqrt(torch.stack(linear_speed_sq).mean(dim=0))
        ang_rms = torch.sqrt(torch.stack(angular_speed_sq).mean(dim=0))
        tracking_rms = torch.sqrt(torch.stack(tracking_error_sq).mean(dim=0))
        tip_distance_mean = torch.stack(tip_center_distance).mean(dim=0)
        lower_margin = torch.where(active, q_target - limits[:, :, 0], torch.inf).amin(dim=-1)
        upper_margin = torch.where(active, limits[:, :, 1] - q_target, torch.inf).amin(dim=-1)
        limit_margin = torch.minimum(lower_margin, upper_margin)
        finite = torch.isfinite(
            torch.stack(
                (tip_fraction, bad_fraction, palm_fraction, tip_distance_mean, drift_max, lin_rms, ang_rms, tracking_rms),
                dim=-1,
            )
        ).all(dim=-1)

        config = PregraspAcceptanceCfg()
        records = []
        reason_counts = {"template": Counter(), "zero": Counter(), "unsupported": Counter()}
        accepted_counts = {"template": 0, "zero": 0, "unsupported": 0}
        for env_index in range(unique_assets * 3):
            control = "template" if env_index < unique_assets else "zero" if env_index < 2 * unique_assets else "unsupported"
            local_row = env_index % unique_assets
            metrics = PregraspMetrics(
                finite=bool(finite[env_index].item()),
                dropped=bool(drift_max[env_index].item() >= 0.07),
                penetrated=False,
                tip_ge_2_fraction=float(tip_fraction[env_index].item()),
                tip_active_count_mean=float(tip_fraction[env_index].item() * 2.0),
                palm_occupancy_fraction=float(palm_fraction[env_index].item()),
                finger_non_tip_occupancy_fraction=float(bad_fraction[env_index].item()),
                tip_object_center_distance_mean_m=float(tip_distance_mean[env_index].item()),
                object_anchor_distance_max_m=float(drift_max[env_index].item()),
                object_linear_velocity_rms_m_s=float(lin_rms[env_index].item()),
                object_angular_velocity_rms_rad_s=float(ang_rms[env_index].item()),
                joint_limit_margin_min_rad=float(limit_margin[env_index].item()),
                target_tracking_error_rms_rad=float(tracking_rms[env_index].item()),
            )
            reasons = evaluate_pregrasp(metrics, config)
            if not reasons:
                accepted_counts[control] += 1
            reason_counts[control].update(reasons)
            records.append(
                {
                    "control": control,
                    "dataset_row": int(HETEROGENEOUS_SOURCE_DATASET_ROWS[local_row]),
                    "local_asset_row": local_row,
                    "accepted": not reasons,
                    "reason_codes": list(reasons),
                    "metrics": asdict(metrics),
                }
            )

        summary = {
            "artifact_type": "anymani.pregrasp.canary_summary",
            "balanced_dataset_rows": list(HETEROGENEOUS_SOURCE_DATASET_ROWS),
            "template_accepted": accepted_counts["template"],
            "zero_accepted": accepted_counts["zero"],
            "unsupported_accepted": accepted_counts["unsupported"],
            "template_reason_counts": dict(reason_counts["template"]),
            "zero_reason_counts": dict(reason_counts["zero"]),
            "unsupported_reason_counts": dict(reason_counts["unsupported"]),
            "physics_steps": snapshots * 6,
            "records": records,
        }
        print(
            "SUMMARY "
            + json.dumps(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"records"}
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        if accepted_counts["unsupported"] != 0:
            raise RuntimeError("unsupported-object negative control produced false-positive accepted pregrasps")
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
