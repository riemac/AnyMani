r"""为八组16个easy-tier资产搜索稳定且TIP更接近DexCube的per-asset q reset。"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import cast

BALANCED_DATASET_ROWS = (416, 417, 352, 353, 0, 1, 64, 65, 432, 433, 368, 369, 16, 17, 80, 81)

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

CANDIDATE_NAMES = (
    "zero",
    "n000_template",
    "limit_midpoint",
    "limit_quarter",
    "limit_three_quarter",
    "template_plus_0p2",
    "template_minus_0p2",
    "template_distal_plus_0p35",
)


def main() -> int:
    r"""并行settle 128个候选，按stable→TIP distance→non-tip→limit margin词典序选择。"""

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

        assets = len(BALANCED_DATASET_ROWS)
        candidates = len(CANDIDATE_NAMES)
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
        limits = robot.data.soft_joint_pos_limits
        lower, upper = limits[:, :, 0], limits[:, :, 1]
        span = upper - lower
        template = torch.tensor(N000_CANONICAL_Q, device=runtime_env.device).expand(assets, -1)
        q = torch.zeros(assets * candidates, 16, device=runtime_env.device)
        q[assets : 2 * assets] = template
        q[2 * assets : 3 * assets] = 0.5 * (lower[2 * assets : 3 * assets] + upper[2 * assets : 3 * assets])
        q[3 * assets : 4 * assets] = lower[3 * assets : 4 * assets] + 0.25 * span[3 * assets : 4 * assets]
        q[4 * assets : 5 * assets] = lower[4 * assets : 5 * assets] + 0.75 * span[4 * assets : 5 * assets]
        q[5 * assets : 6 * assets] = template + 0.2
        q[6 * assets : 7 * assets] = template - 0.2
        q[7 * assets : 8 * assets] = template
        q[7 * assets : 8 * assets, 4:] += 0.35  # depth1–3增加flexion proposal
        q = torch.maximum(torch.minimum(q, upper), lower) * active
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        initial_pos = object_asset.data.root_pos_w.clone()

        tip_distances = []
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
                    robot.data.body_pos_w[:, tip_body_ids] - object_asset.data.root_pos_w.unsqueeze(1),
                    dim=-1,
                ).mean(dim=-1)
            )
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
        bad_fraction = torch.stack(bad_contacts).mean(dim=0)
        drift_max = torch.stack(drift).amax(dim=0)
        linear_rms = torch.sqrt(torch.stack(linear_sq).mean(dim=0))
        angular_rms = torch.sqrt(torch.stack(angular_sq).mean(dim=0))
        dropped = drift_max >= 0.07
        stable = (~dropped) & (drift_max <= 0.025) & (linear_rms <= 0.05) & (angular_rms <= 0.5)
        lower_margin = torch.where(active, q - lower, torch.inf).amin(dim=-1)
        upper_margin = torch.where(active, upper - q, torch.inf).amin(dim=-1)
        margin = torch.minimum(lower_margin, upper_margin)

        records = []
        selected_q = []
        for asset_index, dataset_row in enumerate(HETEROGENEOUS_SOURCE_DATASET_ROWS):
            env_ids = [candidate * assets + asset_index for candidate in range(candidates)]
            viable = [env_id for env_id in env_ids if bool(stable[env_id].item())]
            pool = viable or env_ids
            best = min(
                pool,
                key=lambda env_id: (
                    float(tip_distance[env_id].item()),
                    float(bad_fraction[env_id].item()),
                    -float(margin[env_id].item()),
                ),
            )
            candidate_index = best // assets
            selected_q.append([float(value) for value in q[best].tolist()])
            records.append(
                {
                    "dataset_row": int(dataset_row),
                    "local_asset_row": asset_index,
                    "candidate_index": candidate_index,
                    "candidate_name": CANDIDATE_NAMES[candidate_index],
                    "stable": bool(stable[best].item()),
                    "tip_center_distance_mean_m": float(tip_distance[best].item()),
                    "finger_non_tip_fraction": float(bad_fraction[best].item()),
                    "drift_max_m": float(drift_max[best].item()),
                    "linear_velocity_rms_m_s": float(linear_rms[best].item()),
                    "angular_velocity_rms_rad_s": float(angular_rms[best].item()),
                    "joint_limit_margin_min_rad": float(margin[best].item()),
                }
            )
        output = {
            "artifact_type": "anymani.pregrasp.easy_tier_manifest",
            "schema_version": "1.0.0",
            "cube_scale": 1.2,
            "dataset_rows": list(HETEROGENEOUS_SOURCE_DATASET_ROWS),
            "candidate_names": list(CANDIDATE_NAMES),
            "selected_q_rad": selected_q,
            "records": records,
        }
        output_path = Path("outputs/pregrasp/easy_tier_balanced16_scale1p2.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "stable_selected": sum(record["stable"] for record in records),
                    "candidate_histogram": {
                        name: sum(record["candidate_name"] == name for record in records) for name in CANDIDATE_NAMES
                    },
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
