r"""新`tasks/hetero` 2-asset structured ManagerBased环境的bounded runtime smoke。

该脚本固定formal rows0/16与support minimum tier，使left row0 support basin和right row16 contact basin在同一
round-robin scene中运行。它验证named observation tree、History30、partial reset、20-step finite rollout和
command metrics；不衡量策略能力。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, cast

os.environ["ANYMANI_HETERO_ASSET_ROWS"] = "0,16"
os.environ["ANYMANI_HETERO_MIN_PREGRASP_TIER"] = "support_basin"

from isaaclab.app import AppLauncher  # noqa: E402  # selection必须先于task config import固定

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def _parse_args() -> argparse.Namespace:
    r"""解析bounded steps与durable output。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hetero/runtime-smokes/structured-env-rows-0-16.json"),
    )
    return parser.parse_args()


def _assert_finite_tree(value: Any, *, path: str = "obs") -> None:
    r"""递归验证nested tensor observation全部finite。"""

    import torch

    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise AssertionError(f"{path} contains non-finite values")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, path=f"{path}.{key}")
        return
    raise AssertionError(f"{path} contains unsupported value type {type(value).__name__}")


def main() -> int:
    r"""运行2-asset full reset、step与partial-history smoke。"""

    import anymani.tasks.hetero  # noqa: F401  # 注册唯一新Gym ID
    import gymnasium as gym
    import torch
    from anymani.tasks.hetero.config.generated.pregrasp_identity import FORMAL_SEARCH_PROTOCOL_DIGEST
    from anymani.tasks.hetero.config.generated.scene import (
        FORMAL_PREGRASP_IDENTITY,
        RESOLVED_DEX_CUBE_PATH,
        RESOLVED_DEX_CUBE_SHA256,
    )
    from anymani.tasks.hetero.config.generated.tactile_rotation_env_cfg import (
        ASSET_BINDING,
        GeneratedHeterogeneousTactileRotationEnvCfg,
    )
    from anymani.tasks.hetero.mdp.commands import HeterogeneousRotationCommand
    from anymani.tasks.hetero.mdp.runtime_state import HETERO_PREGRASP_STATE_ATTR, HeterogeneousPregraspState
    from isaaclab.envs import ManagerBasedRLEnv

    args = _parse_args()
    if args.steps < 2:
        raise ValueError("structured smoke requires at least two steps")
    cfg = GeneratedHeterogeneousTactileRotationEnvCfg()
    env = gym.make("AnyMani-Hetero-Generated-TactileRotation-v0", cfg=cfg)
    try:
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        observation_raw, _ = env.reset()
        observation = cast(dict[str, Any], observation_raw)
        _assert_finite_tree(observation)
        if not isinstance(observation.get("policy"), dict) or not isinstance(observation.get("critic"), dict):
            raise AssertionError("policy/critic groups must remain non-concatenated dictionaries")
        policy = observation["policy"]
        critic = observation["critic"]
        expected_policy_shapes = {
            "palm_valid": (2, 1),
            "jnt_current": (2, 16, 3),
            "jnt_history": (2, 30, 16, 4),
            "jnt_limits": (2, 16, 2),
            "tip_contact": (2, 4, 1),
            "jnt_valid": (2, 16),
            "tip_valid": (2, 4),
            "owner_valid": (2, 21),
        }
        expected_critic_shapes = {
            "palm_valid": (2, 1),
            "jnt_state": (2, 16, 4),
            "owner_contact": (2, 21, 2),
            "obj": (2, 1, 15),
            "task": (2, 1, 8),
            "jnt_valid": (2, 16),
            "tip_valid": (2, 4),
            "owner_valid": (2, 21),
        }
        if {name: tuple(value.shape) for name, value in policy.items()} != expected_policy_shapes:
            raise AssertionError("policy structured shapes disagree with O^a contract")
        if {name: tuple(value.shape) for name, value in critic.items()} != expected_critic_shapes:
            raise AssertionError("critic structured shapes disagree with O^c contract")
        forbidden_keys = {"asset_row", "morphology_cell", "all_link_contact", "obj", "task"}
        if forbidden_keys & set(policy):
            raise AssertionError("actor observation exposes privileged/provenance fields")

        # CircularBuffer第一次append应把真实reset frame重复填满全部30个位置。
        reset_history = policy["jnt_history"]
        if not torch.equal(reset_history, reset_history[:, -1:].expand_as(reset_history)):
            raise AssertionError("History30 reset prefix did not repeat the real reset frame")
        sidecar = getattr(runtime_env, HETERO_PREGRASP_STATE_ATTR)
        if not isinstance(sidecar, HeterogeneousPregraspState) or not bool(sidecar.valid.all().item()):
            raise AssertionError("structured env reset did not resolve both pregrasp rows")
        physics_validation = getattr(runtime_env, "_anymani_formal_object_physics_validation", None)
        if not isinstance(physics_validation, dict):
            raise AssertionError("startup did not validate actual DexCube mass/inertia")

        reward_sum = torch.zeros(runtime_env.num_envs, device=runtime_env.device)
        termination_count = torch.zeros(runtime_env.num_envs, device=runtime_env.device)
        first_step_history = None
        command = runtime_env.command_manager.get_term("goal_pose")
        if not isinstance(command, HeterogeneousRotationCommand):
            raise AssertionError("new environment did not instantiate hetero command")
        for step in range(args.steps):
            # 小幅确定性动作形成可检查history shift，同时不把random policy当能力实验。
            action = torch.zeros(runtime_env.num_envs, 16, device=runtime_env.device)
            action[:, :4] = 0.1 if step == 0 else 0.0
            observation_raw, reward_raw, terminated_raw, truncated_raw, _ = env.step(action)
            observation = cast(dict[str, Any], observation_raw)
            reward = cast(torch.Tensor, reward_raw)
            terminated = cast(torch.Tensor, terminated_raw)
            truncated = cast(torch.Tensor, truncated_raw)
            _assert_finite_tree(observation)
            if not bool(torch.isfinite(reward).all().item()):
                raise AssertionError("structured env reward became non-finite")
            reward_sum += reward
            termination_count += (terminated | truncated).to(dtype=torch.float32)
            snapshot = command.post_physics_evaluation_snapshot
            if not bool(snapshot["valid"].all().item()) or not bool(
                (snapshot["step"] == int(runtime_env.common_step_counter)).all().item()
            ):
                raise AssertionError("reward did not capture a current pre-reset evaluation snapshot")
            if step == 0:
                first_step_history = observation["policy"]["jnt_history"].clone()
                if torch.equal(first_step_history[:, -1], reset_history[:, -1]):
                    raise AssertionError("History30 did not append the changed policy frame")

        if first_step_history is None:
            raise AssertionError("structured smoke never produced first-step history")

        # 乱序之外的单row partial reset：row0重复新frame，row1保留旧history并正常shift一次。
        history_before = observation["policy"]["jnt_history"].clone()
        snapshot_before_reset = {
            name: value[0].clone() for name, value in command.post_physics_evaluation_snapshot.items()
        }
        runtime_env._reset_idx([0])
        observation_after = cast(dict[str, dict[str, torch.Tensor]], runtime_env.observation_manager.compute(update_history=True))
        history_after = observation_after["policy"]["jnt_history"]
        if not torch.equal(history_after[0], history_after[0, -1:].expand_as(history_after[0])):
            raise AssertionError("partial reset did not refill selected History30 row")
        if not torch.equal(history_after[1, :-1], history_before[1, 1:]):
            raise AssertionError("partial reset destroyed non-selected history instead of shifting once")
        if any(
            not torch.equal(command.post_physics_evaluation_snapshot[name][0], value)
            for name, value in snapshot_before_reset.items()
        ):
            raise AssertionError("automatic reset semantics cleared the pre-reset evaluation snapshot")
        evidence = {
            "artifact_type": "anymani.hetero.structured_env_smoke",
            "schema_version": "1.0.0",
            "gym_id": "AnyMani-Hetero-Generated-TactileRotation-v0",
            "dataset_rows": list(ASSET_BINDING.dataset_rows),
            "num_envs": runtime_env.num_envs,
            "steps": args.steps,
            "policy_shapes": {name: list(shape) for name, shape in expected_policy_shapes.items()},
            "critic_shapes": {name: list(shape) for name, shape in expected_critic_shapes.items()},
            "history_reset_repeat_passed": True,
            "history_partial_reset_passed": True,
            "pre_reset_evaluation_snapshot_passed": True,
            "reward_sum": [float(value) for value in reward_sum.tolist()],
            "termination_count": [float(value) for value in termination_count.tolist()],
            "signed_net_rotation_rad": [float(value) for value in command.net_rotation_rad.tolist()],
            "goal_success_count": [float(value) for value in command.goal_success_count.tolist()],
            "pregrasp_record_digests": list(sidecar.record_digests),
            "formal_pregrasp_identity": {
                "cube_local_path": str(RESOLVED_DEX_CUBE_PATH),
                "cube_sha256": RESOLVED_DEX_CUBE_SHA256,
                "gate_digest": FORMAL_PREGRASP_IDENTITY.gate_digest,
                "physics_identity": dict(FORMAL_PREGRASP_IDENTITY.physics_identity),
                "search_protocol_digest": FORMAL_SEARCH_PROTOCOL_DIGEST,
                "runtime_physics_validation": physics_validation,
            },
            "actor_contains_only_tip_contact": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), **evidence}, sort_keys=True))
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
