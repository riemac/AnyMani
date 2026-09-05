r"""Balanced16 structured environment的10 s zero-action稳定性与equal-asset诊断canary。

Zero action保持pregrasp PD targets，不代表学习策略。该probe只验证16 unique assets能够共同reset/step，并让新的
signed rotation、subgoal、termination、contact与per-asset sum/count统计从同一环境闭环输出。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, cast

BALANCED_16_ROWS = (416, 417, 352, 353, 0, 1, 64, 65, 432, 433, 368, 369, 16, 17, 80, 81)
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in BALANCED_16_ROWS)
os.environ["ANYMANI_HETERO_MIN_PREGRASP_TIER"] = "support_basin"
os.environ["ANYMANI_HETERO_LOG_ASSET_METRICS"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def _parse_args() -> argparse.Namespace:
    r"""解析canary policy steps与durable output。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)  # 20 Hz下10 s
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hetero/evaluation/balanced16-zero-action-10s.json"),
    )
    return parser.parse_args()


def main() -> int:
    r"""运行16-env zero-action canary并保存per-asset/equal-asset metrics。"""

    import anymani.tasks.hetero  # noqa: F401
    import gymnasium as gym
    import torch
    from anymani.tasks.hetero.config.generated.tactile_rotation_env_cfg import (
        ASSET_BINDING,
        GeneratedHeterogeneousTactileRotationEnvCfg,
    )
    from anymani.tasks.hetero.mdp.commands import HeterogeneousRotationCommand
    from anymani.tasks.hetero.mdp.contact_state import HETERO_CONTACT_STATE_ATTR, HeterogeneousContactState
    from anymani.tasks.hetero.mdp.diagnostics import equal_asset_metric_from_extras
    from anymani.tasks.hetero.mdp.runtime_state import HETERO_PREGRASP_STATE_ATTR, HeterogeneousPregraspState
    from isaaclab.envs import ManagerBasedRLEnv

    args = _parse_args()
    if args.steps < 1:
        raise ValueError("canary steps must be positive")
    env = gym.make(
        "AnyMani-Hetero-Generated-TactileRotation-v0",
        cfg=GeneratedHeterogeneousTactileRotationEnvCfg(),
    )
    try:
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        observation_raw, _ = env.reset()
        observation = cast(dict[str, Any], observation_raw)
        if ASSET_BINDING.dataset_rows != BALANCED_16_ROWS or runtime_env.num_envs != len(BALANCED_16_ROWS):
            raise AssertionError("balanced16 scene routing disagrees with requested formal rows")
        if tuple(observation["policy"]["jnt_history"].shape) != (16, 30, 16, 4):
            raise AssertionError("balanced16 History30 structured shape is wrong")

        reward_sum = torch.zeros(16, device=runtime_env.device)
        termination_count = torch.zeros(16, device=runtime_env.device)
        tip_count_sum = torch.zeros(16, device=runtime_env.device)
        palm_occupancy_sum = torch.zeros(16, device=runtime_env.device)
        non_tip_occupancy_sum = torch.zeros(16, device=runtime_env.device)
        for _ in range(args.steps):
            action = torch.zeros(16, 16, device=runtime_env.device)
            _, reward_raw, terminated_raw, truncated_raw, _ = env.step(action)
            reward = cast(torch.Tensor, reward_raw)
            terminated = cast(torch.Tensor, terminated_raw)
            truncated = cast(torch.Tensor, truncated_raw)
            if not bool(torch.isfinite(reward).all().item()):
                raise AssertionError("balanced16 reward became non-finite")
            reward_sum += reward
            termination_count += (terminated | truncated).to(dtype=torch.float32)
            contact = getattr(runtime_env, HETERO_CONTACT_STATE_ATTR)
            if not isinstance(contact, HeterogeneousContactState):
                raise AssertionError("balanced16 env lacks shared contact state")
            tip_count_sum += contact.tip_bits.sum(dim=-1).to(dtype=torch.float32)
            palm_occupancy_sum += contact.palm_bits[:, 0].to(dtype=torch.float32)
            non_tip_occupancy_sum += contact.finger_non_tip_bits.any(dim=-1).to(dtype=torch.float32)

        command = runtime_env.command_manager.get_term("goal_pose")
        if not isinstance(command, HeterogeneousRotationCommand):
            raise AssertionError("balanced16 env lacks heterogeneous rotation command")
        signed_net_rotation_rad = command.net_rotation_rad.clone()
        goal_success_count = command.goal_success_count.clone()
        pregrasp = getattr(runtime_env, HETERO_PREGRASP_STATE_ATTR)
        if not isinstance(pregrasp, HeterogeneousPregraspState) or not bool(pregrasp.valid.all().item()):
            raise AssertionError("balanced16 pregrasp sidecar is incomplete")

        # Force a diagnostics-only partial reset of all current episodes to obtain per-asset sum/count.
        runtime_env._reset_idx(list(range(runtime_env.num_envs)))
        manager_extras = dict(runtime_env.extras["log"])
        command_prefix = "Metrics/goal_pose/"
        terminal_extras = {
            key.removeprefix(command_prefix): value
            for key, value in manager_extras.items()
            if key.startswith(command_prefix)
        }
        equal_asset_signed_rad = equal_asset_metric_from_extras(
            terminal_extras, "net_rotation_rad_signed_sum"
        )
        equal_asset_subgoals = equal_asset_metric_from_extras(terminal_extras, "goal_success_count_sum")
        per_asset = []
        for local_index, dataset_row in enumerate(BALANCED_16_ROWS):
            prefix = f"asset/{dataset_row}"
            if terminal_extras.get(f"{prefix}/episode_count") != 1.0:
                raise AssertionError(f"asset {dataset_row} terminal count is not exactly one")
            per_asset.append(
                {
                    "dataset_row": dataset_row,
                    "record_digest": pregrasp.record_digests[local_index],
                    "signed_net_rotation_rad": float(signed_net_rotation_rad[local_index].item()),
                    "goal_success_count": float(goal_success_count[local_index].item()),
                    "reward_sum": float(reward_sum[local_index].item()),
                    "termination_count": float(termination_count[local_index].item()),
                    "tip_active_count_mean": float((tip_count_sum[local_index] / args.steps).item()),
                    "palm_occupancy_fraction": float((palm_occupancy_sum[local_index] / args.steps).item()),
                    "finger_non_tip_occupancy_fraction": float(
                        (non_tip_occupancy_sum[local_index] / args.steps).item()
                    ),
                    "termination_object_out_of_anchor": terminal_extras[
                        f"{prefix}/termination_object_out_of_anchor_sum"
                    ],
                    "termination_goal_axis_misaligned": terminal_extras[
                        f"{prefix}/termination_goal_axis_misaligned_sum"
                    ],
                    "termination_time_out": terminal_extras[f"{prefix}/termination_time_out_sum"],
                }
            )

        evidence = {
            "artifact_type": "anymani.hetero.structured_evaluation_canary",
            "schema_version": "1.0.0",
            "gym_id": "AnyMani-Hetero-Generated-TactileRotation-v0",
            "probe_policy": "zero_action_hold_pregrasp_target",
            "steps": args.steps,
            "duration_s": args.steps * float(runtime_env.step_dt),
            "dataset_rows": list(BALANCED_16_ROWS),
            "equal_asset_signed_net_rotation_rad": equal_asset_signed_rad,
            "equal_asset_goal_success_count": equal_asset_subgoals,
            "per_asset": per_asset,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "duration_s": evidence["duration_s"],
                    "equal_asset_signed_net_rotation_rad": equal_asset_signed_rad,
                    "equal_asset_goal_success_count": equal_asset_subgoals,
                    "termination_count": float(termination_count.sum().item()),
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
