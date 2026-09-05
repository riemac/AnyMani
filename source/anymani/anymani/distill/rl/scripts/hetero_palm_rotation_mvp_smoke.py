r"""两手或完整80手scale-1.1 PalmRotation MVP真实reset/zero-action smoke。

Smoke默认使用final MVP首对rows 1966/1976，验证schema-3 rank-0$q_0=u_0$ reset、strict upright object、
own-JOINT＋TIP History30、all-owner binary contact、privileged reward-release与N000完整reward lifecycle。
默认只跑rows 416/432；``--all80``检查每只最终资产的1 s zero-action cold-reset存活。它不更新policy参数，
即使80手均存活也只证明reset/task/network合同，不形成旋转学习能力结论。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, cast

import yaml

parser = argparse.ArgumentParser(description="Smoke the palm-rotation MVP reset/task/network path.")
parser.add_argument("--all80", action="store_true", help="Use every row from the final MVP80 manifest.")
parser.add_argument("--steps", type=int, default=20, help="Zero-action policy steps; 20 steps equal 1 second.")
args = parser.parse_args()

if args.all80:
    manifest_path = Path(
        "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    ROWS = tuple(int(row) for row in manifest["selected_rows"])
else:
    ROWS = (1966, 1976)
if len(set(ROWS)) != len(ROWS):
    raise ValueError("runtime smoke requires unique selected dataset rows")
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in ROWS)
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(len(ROWS))

from isaaclab.app import AppLauncher  # noqa: E402  # routing必须在task import前冻结

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def main() -> int:
    r"""执行一次reset与20个zero-action policy steps并发布结构化结果。"""

    import anymani.tasks.hetero  # noqa: F401  # 注册MVP Gym ID
    import gymnasium as gym
    import torch
    from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1
    from anymani.distill.models.palm_rotation_policy import (
        PalmRotationActorCritic,
        PalmRotationActorObservation,
        PalmRotationCriticObservation,
    )
    from anymani.distill.rl.runtime.palm_rotation_geometry import build_palm_rotation_bf16_geometry_provider
    from anymani.tasks.hetero.config.generated.palm_rotation_mvp_env_cfg import (
        ASSET_BINDING,
        GeneratedPalmRotationMvpEnvCfg,
    )
    from anymani.tasks.hetero.mdp.curriculum_state import (
        HETERO_REWARD_RELEASE_STATE_ATTR,
        HeterogeneousRewardReleaseState,
    )
    from anymani.tasks.hetero.mdp.runtime_state import HETERO_PREGRASP_STATE_ATTR, HeterogeneousPregraspState
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    cfg = GeneratedPalmRotationMvpEnvCfg()
    env: Any = gym.make("AnyMani-Hetero-Generated-PalmRotation-MVP-v0", cfg=cfg)
    try:
        runtime = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime.sim._app_control_on_stop_handle = None
        observation, _ = env.reset()
        policy = observation["policy"]
        critic = observation["critic"]
        num_envs = len(ROWS)
        expected_policy_shapes = {
            "jnt_current": (num_envs, 16, 5),
            "jnt_history": (num_envs, 30, 16, 5),
            "jnt_limits": (num_envs, 16, 2),
            "owner_contact": (num_envs, 21, 1),
            "jnt_valid": (num_envs, 16),
            "tip_valid": (num_envs, 4),
            "owner_valid": (num_envs, 21),
        }
        if {name: tuple(value.shape) for name, value in policy.items()} != expected_policy_shapes:
            raise AssertionError("MVP policy observation shapes disagree with structured contract")
        if tuple(critic["reward_release"].shape) != (num_envs, 1):
            raise AssertionError("MVP critic reward-release shape mismatch")
        sidecar = getattr(runtime, HETERO_PREGRASP_STATE_ATTR, None)
        if not isinstance(sidecar, HeterogeneousPregraspState) or not bool(sidecar.valid.all().item()):
            raise AssertionError("MVP task did not install schema-3 good-pregrasp state")
        if not torch.equal(sidecar.q_state_rad, sidecar.q_target_rad):
            raise AssertionError("MVP rank-0 reset requires q_state == q_target")
        upright = torch.tensor((1.0, 0.0, 0.0, 0.0), device=runtime.device).expand(num_envs, -1)
        if not torch.equal(sidecar.object_quat_h_wxyz, upright):
            raise AssertionError("MVP object reset is not exact hand-frame upright")
        reward_release = getattr(runtime, HETERO_REWARD_RELEASE_STATE_ATTR, None)
        if not isinstance(reward_release, HeterogeneousRewardReleaseState):
            raise AssertionError("MVP reward-release curriculum state is unavailable")
        if not torch.equal(reward_release.env_lambda, torch.zeros_like(reward_release.env_lambda)):
            raise AssertionError("ADR-0 MVP reward release must start at zero")

        # Canonical policy tensors必须按joint name映射到PhysX native轴；shape相同不能证明语义顺序相同。
        robot = cast(Articulation, runtime.scene["robot"])
        action_term: Any = runtime.action_manager.get_term("hand_joint_pos")
        canonical_names = tuple(CANONICAL_HAND_SCHEMA_V1.joint_names)
        native_names = tuple(robot.joint_names)
        action_names = tuple(action_term._joint_names)
        action_ids = tuple(int(value) for value in action_term._joint_ids)
        if set(native_names) != set(canonical_names) or set(action_names) != set(canonical_names):
            raise AssertionError("runtime/action joints do not contain the exact canonical-v1 joint-name set")
        canonical_index = {name: index for index, name in enumerate(canonical_names)}
        expected_native_q = torch.stack(
            [sidecar.q_state_rad[:, canonical_index[name]] for name in native_names],
            dim=-1,
        )  # canonical sidecar按name重排为robot.data native轴
        reset_q_native_max_error = float(torch.max(torch.abs(robot.data.joint_pos - expected_native_q)).item())

        rewards = []
        done_count = 0
        done_by_env = torch.zeros(num_envs, dtype=torch.long, device=runtime.device)
        for _ in range(int(args.steps)):
            observation, reward, terminated, truncated, _ = env.step(
                torch.zeros(num_envs, 16, device=runtime.device)
            )
            if not bool(torch.isfinite(reward).all().item()):
                raise AssertionError("MVP zero-action reward became non-finite")
            rewards.append(reward.detach().cpu())
            done = terminated | truncated
            done_count += int(done.sum().item())
            done_by_env += done.long()

        # 实际task tensors进入四层BF16 N040、zero-init residual actor与两层LN-c critic。
        actor_observation = PalmRotationActorObservation.from_task_dict(observation["policy"])
        critic_observation = PalmRotationCriticObservation.from_task_dict(observation["critic"])
        provider = build_palm_rotation_bf16_geometry_provider(ASSET_BINDING, device=runtime.device)
        prototype_index = torch.tensor(
            ASSET_BINDING.asset_index_by_env(runtime.num_envs),
            dtype=torch.long,
            device=runtime.device,
        )
        geometry = provider.resolve(prototype_index, actor_observation)
        package = PalmRotationActorCritic(residual_enabled=True).to(runtime.device)
        actor_output = package.actor(actor_observation, geometry)
        value = package.critic(critic_observation, geometry)
        if geometry.tokens.dtype != torch.float32 or actor_output.mean.dtype != torch.float32 or value.dtype != torch.float32:
            raise AssertionError("encoder-only BF16 boundary must restore FP32 policy/value outputs")
        if not torch.equal(actor_output.mean, actor_output.base_mean):
            raise AssertionError("zero-init action residual must exactly equal base policy")
        synthetic_loss = actor_output.mean.square().mean() + value.square().mean()
        synthetic_loss.backward()
        actor_ids, critic_ids = package.trainable_parameter_sets()
        if not actor_ids.isdisjoint(critic_ids):
            raise AssertionError("MVP actor and critic parameters must remain disjoint")
        summary = {
            "artifact_type": "anymani.hetero.palm_rotation_mvp_smoke",
            "schema_version": "1.0.0",
            "dataset_rows": list(ASSET_BINDING.dataset_rows),
            "num_envs": runtime.num_envs,
            "object_scale": 1.1,
            "policy_shapes": {name: list(value.shape) for name, value in observation["policy"].items()},
            "critic_shapes": {name: list(value.shape) for name, value in observation["critic"].items()},
            "rank0_equal_state_target": True,
            "rank0_upright": True,
            "initial_reward_release": 0.0,
            "joint_order_audit": {
                "canonical_joint_names": list(canonical_names),
                "robot_native_joint_names": list(native_names),
                "action_joint_names": list(action_names),
                "action_joint_ids": list(action_ids),
                "robot_native_equals_canonical": native_names == canonical_names,
                "action_equals_canonical": action_names == canonical_names,
                "reset_q_native_max_error_rad": reset_q_native_max_error,
            },
            "steps": int(args.steps),
            "done_count": done_count,
            "done_assets": [
                {"asset_index": index, "dataset_row": ROWS[index], "done_count": int(count)}
                for index, count in enumerate(done_by_env.detach().cpu().tolist())
                if count > 0
            ],
            "reward_mean": float(torch.stack(rewards).mean().item()),
            "network": {
                "actor_parameters": sum(parameter.numel() for parameter in package.actor.parameters()),
                "critic_parameters": sum(parameter.numel() for parameter in package.critic.parameters()),
                "n040_parameters": sum(parameter.numel() for parameter in provider.provider.encoder.parameters()),
                "n040_resolve_calls": provider.resolve_call_count,
                "n040_compute": "bf16_encoder_only",
                "geometry_dtype": str(geometry.tokens.dtype),
                "actor_dtype": str(actor_output.mean.dtype),
                "critic_dtype": str(value.dtype),
                "zero_residual_exact": True,
                "synthetic_loss": float(synthetic_loss.detach().item()),
            },
        }
        suffix = "mvp80" if args.all80 else "rows-1966-1976"
        output = Path(f"outputs/hetero/runtime-smokes/palm-rotation-mvp-{suffix}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(json.dumps(summary, sort_keys=True), flush=True)
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
