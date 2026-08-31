r"""固定per-asset q后搜索稳定、TIP更可达的DexCube contact-basin平移。"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import cast

BALANCED_DATASET_ROWS = (416, 417, 352, 353, 0, 1, 64, 65, 432, 433, 368, 369, 16, 17, 80, 81)
OFFSETS_E_M = (
    (0.0, 0.0, 0.0),
    (-0.02, 0.0, 0.0),
    (0.02, 0.0, 0.0),
    (0.0, -0.02, 0.0),
    (0.0, 0.02, 0.0),
    (0.0, -0.04, 0.0),
    (0.0, 0.04, 0.0),
    (0.0, 0.0, 0.015),
    (0.0, 0.0, -0.015),
)


def main() -> int:
    r"""并行settle 16×9 object offsets并写v3 reset manifest。"""

    os.environ["ANYMANI_HETEROGENEOUS_ASSET_ROWS"] = ",".join(str(row) for row in BALANCED_DATASET_ROWS)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    env = None
    try:
        import anymani.tasks.gm  # noqa: F401
        import gymnasium as gym
        import torch
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

        q_manifest_path = Path("outputs/pregrasp/easy_tier_balanced16_scale1p2.json")
        q_manifest = json.loads(q_manifest_path.read_text())
        if tuple(q_manifest["dataset_rows"]) != tuple(HETEROGENEOUS_SOURCE_DATASET_ROWS):
            raise ValueError("q manifest rows disagree with balanced object-offset search")
        selected_q = torch.tensor(q_manifest["selected_q_rad"], dtype=torch.float32)

        assets = len(BALANCED_DATASET_ROWS)
        candidates = len(OFFSETS_E_M)
        cfg = HeterogeneousTactileRotationEnvCfg()
        cfg.scene.num_envs = assets * candidates
        env = gym.make("AnyMani-GM-HeterogeneousAsset-TactileRotation-v0", cfg=cfg)
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        env.reset()
        robot = cast(Articulation, runtime_env.scene["robot"])
        object_asset = cast(RigidObject, runtime_env.scene["object"])
        tip_body_ids, _ = robot.find_bodies(list(HETEROGENEOUS_CONTACT_LAYOUT.fingertip_link_names), preserve_order=True)
        active = torch.tensor(
            [mask for _ in range(candidates) for mask in HETEROGENEOUS_ACTIVE_MASK_ROWS],
            dtype=torch.bool,
            device=runtime_env.device,
        )
        q = selected_q.to(runtime_env.device).repeat(candidates, 1) * active
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        object_pose = object_asset.data.root_pose_w.clone()
        for candidate_index, offset in enumerate(OFFSETS_E_M):
            rows = slice(candidate_index * assets, (candidate_index + 1) * assets)
            object_pose[rows, :3] += torch.tensor(offset, device=runtime_env.device)
        object_asset.write_root_pose_to_sim(object_pose)
        object_asset.write_root_velocity_to_sim(torch.zeros(assets * candidates, 6, device=runtime_env.device))
        initial_pos = object_pose[:, :3].clone()

        tip_distances = []
        tip_active = []
        bad_contacts = []
        drift = []
        linear_sq = []
        angular_sq = []
        for step in range(120):
            robot.set_joint_position_target(q)
            runtime_env.scene.write_data_to_sim()
            runtime_env.sim.step(render=False)
            runtime_env.scene.update(runtime_env.physics_dt)
            if (step + 1) % 6:
                continue
            tip_distances.append(
                torch.linalg.vector_norm(
                    robot.data.body_pos_w[:, tip_body_ids] - object_asset.data.root_pos_w.unsqueeze(1), dim=-1
                ).mean(dim=-1)
            )
            tip_force = torch.stack(
                [
                    torch.linalg.vector_norm(sensor_total_force_w(runtime_env, name), dim=-1)
                    for name in HETEROGENEOUS_CONTACT_LAYOUT.fingertip_sensor_names
                ],
                dim=-1,
            )
            tip_active.append((tip_force > 0.25).float().sum(dim=-1))
            non_tip_force = torch.stack(
                [
                    torch.linalg.vector_norm(sensor_total_force_w(runtime_env, name), dim=-1)
                    for name in HETEROGENEOUS_CONTACT_LAYOUT.finger_non_tip_sensor_names
                ],
                dim=-1,
            )
            bad_contacts.append((non_tip_force > 0.25).any(dim=-1).float())
            drift.append(torch.linalg.vector_norm(object_asset.data.root_pos_w - initial_pos, dim=-1))
            linear_sq.append(object_asset.data.root_lin_vel_w.square().sum(dim=-1))
            angular_sq.append(object_asset.data.root_ang_vel_w.square().sum(dim=-1))

        tip_distance = torch.stack(tip_distances).mean(dim=0)
        tip_active_mean = torch.stack(tip_active).mean(dim=0)
        bad_fraction = torch.stack(bad_contacts).mean(dim=0)
        drift_max = torch.stack(drift).amax(dim=0)
        linear_rms = torch.sqrt(torch.stack(linear_sq).mean(dim=0))
        angular_rms = torch.sqrt(torch.stack(angular_sq).mean(dim=0))
        stable = (drift_max <= 0.025) & (linear_rms <= 0.05) & (angular_rms <= 0.5)

        selected_offsets = []
        records = []
        for asset_index, dataset_row in enumerate(HETEROGENEOUS_SOURCE_DATASET_ROWS):
            env_ids = [candidate * assets + asset_index for candidate in range(candidates)]
            viable = [env_id for env_id in env_ids if bool(stable[env_id].item())]
            pool = viable or env_ids
            best = min(
                pool,
                key=lambda env_id: (
                    -float(tip_active_mean[env_id].item()),
                    float(tip_distance[env_id].item()),
                    float(bad_fraction[env_id].item()),
                ),
            )
            candidate_index = best // assets
            selected_offsets.append(list(OFFSETS_E_M[candidate_index]))
            records.append(
                {
                    "dataset_row": int(dataset_row),
                    "offset_index": candidate_index,
                    "offset_e_m": list(OFFSETS_E_M[candidate_index]),
                    "stable": bool(stable[best].item()),
                    "tip_active_count_mean": float(tip_active_mean[best].item()),
                    "tip_center_distance_mean_m": float(tip_distance[best].item()),
                    "finger_non_tip_fraction": float(bad_fraction[best].item()),
                    "drift_max_m": float(drift_max[best].item()),
                }
            )
        output = dict(q_manifest)
        output["parent_manifest"] = str(q_manifest_path)
        output["selected_object_offset_e_m"] = selected_offsets
        output["object_offset_candidates_e_m"] = [list(offset) for offset in OFFSETS_E_M]
        output["object_offset_records"] = records
        output_path = Path("outputs/pregrasp/easy_tier_balanced16_scale1p2_v3.json")
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "stable_selected": sum(record["stable"] for record in records),
                    "nonzero_offsets": sum(any(abs(value) > 0 for value in offset) for offset in selected_offsets),
                    "tip_active_mean": sum(record["tip_active_count_mean"] for record in records) / assets,
                    "tip_distance_mean_m": sum(record["tip_center_distance_mean_m"] for record in records) / assets,
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
