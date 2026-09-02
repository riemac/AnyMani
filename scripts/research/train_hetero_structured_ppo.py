r"""同一row16上运行exact support/contact tier的matched structured PPO。

脚本固定四层FP32 N040、D128 gated-pool actor、独立masked-pooling critic与standard GAE/clipped PPO。两臂只由
``--tier``改变pregrasp exact record；模型seed、action-noise stream、minibatch stream、环境数与预算由CLI完整记录。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast


def _parse_args() -> argparse.Namespace:
    r"""解析matched run identity与有界训练预算。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("support_basin", "contact_basin"), required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


ARGS = _parse_args()
if min(ARGS.num_envs, ARGS.updates, ARGS.horizon, ARGS.epochs, ARGS.minibatches, ARGS.eval_steps) < 1:
    raise ValueError("all matched PPO counts must be positive")
if (ARGS.num_envs * ARGS.horizon) % ARGS.minibatches != 0:
    raise ValueError("num_envs*horizon must be divisible by minibatches")
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = "16"
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(ARGS.num_envs)
os.environ["ANYMANI_HETERO_MIN_PREGRASP_TIER"] = ARGS.tier
os.environ["ANYMANI_HETERO_EXACT_PREGRASP_TIER"] = ARGS.tier
os.environ["ANYMANI_HETERO_LOG_ASSET_METRICS"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def _accumulate_terminal_extras(target: dict[str, float], info: dict[str, Any]) -> None:
    r"""只累加当前done step的command-local per-asset sum/count。"""

    log = info.get("log", {})
    if not isinstance(log, dict):
        return
    prefix = "Metrics/goal_pose/"
    for key, value in log.items():
        if key.startswith(prefix) and "/asset/" not in f"/{key}":
            continue
        if key.startswith(prefix + "asset/"):
            local_key = key.removeprefix(prefix)
            target[local_key] = target.get(local_key, 0.0) + float(value)


def _evaluate(
    env,
    runtime,
    prototype_index,
    *,
    steps: int,
) -> tuple[dict[str, float], dict[str, object]]:
    r"""执行deterministic actor-mean fixed-duration evaluation并返回metrics和最后observation。"""

    import torch
    from anymani.distill.models.structured_heterogeneous import StructuredActorObservation
    from anymani.distill.rl.structured_transport import StructuredRlTransport
    from anymani.tasks.hetero.mdp.commands import HeterogeneousRotationCommand
    from anymani.tasks.hetero.mdp.contact_state import HETERO_CONTACT_STATE_ATTR, HeterogeneousContactState

    observation_raw, _ = env.reset()
    observation = cast(dict[str, object], observation_raw)
    runtime.eval()
    reward_sum = 0.0
    done_count = 0.0
    signed_speed_sum = 0.0
    absolute_speed_sum = 0.0
    tip_count_sum = 0.0
    palm_sum = 0.0
    non_tip_sum = 0.0
    terminal_extras: dict[str, float] = {}
    with torch.inference_mode():
        for _ in range(steps):
            transport = StructuredRlTransport.from_nested_observation(observation, prototype_index)
            actor_observation = StructuredActorObservation.from_task_dict(transport.policy_storage())
            geometry = runtime.resolve_geometry(prototype_index, actor_observation)
            actions = runtime.actor_forward(actor_observation, geometry).mean
            next_observation, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            reward_sum += float(rewards.mean().item())
            done_count += float(dones.to(dtype=torch.float32).sum().item())
            if bool(dones.any().item()):
                _accumulate_terminal_extras(terminal_extras, cast(dict[str, Any], info))
            command = env.unwrapped.command_manager.get_term("goal_pose")
            if not isinstance(command, HeterogeneousRotationCommand):
                raise RuntimeError("evaluation command type mismatch")
            signed_speed_sum += float(command.axis_speed_rad_s.mean().item())
            absolute_speed_sum += float(command.axis_speed_rad_s.abs().mean().item())
            contact = getattr(env.unwrapped, HETERO_CONTACT_STATE_ATTR)
            if not isinstance(contact, HeterogeneousContactState):
                raise RuntimeError("evaluation contact state is missing")
            tip_count_sum += float(contact.tip_bits.sum(dim=-1).to(torch.float32).mean().item())
            palm_sum += float(contact.palm_bits.to(torch.float32).mean().item())
            non_tip_sum += float(contact.finger_non_tip_bits.any(dim=-1).to(torch.float32).mean().item())
            observation = cast(dict[str, object], next_observation)

    # 把未自然终止的partial trajectories作为固定evaluation窗口收尾并取得sum/count。
    env.unwrapped._reset_idx(list(range(env.unwrapped.num_envs)))
    _accumulate_terminal_extras(terminal_extras, cast(dict[str, Any], env.unwrapped.extras))
    prefix = "asset/16"
    episode_count = terminal_extras.get(f"{prefix}/episode_count", 0.0)
    if episode_count <= 0.0:
        raise RuntimeError("evaluation produced no per-asset trajectory statistics")
    subgoal_sum = terminal_extras.get(f"{prefix}/goal_success_count_sum", 0.0)
    full_turn_sum = terminal_extras.get(f"{prefix}/reached_positive_full_turn_sum", 0.0)
    metrics = {
        "trajectory_count": episode_count,
        "reward_mean_per_step": reward_sum / steps,
        "signed_axis_speed_mean_rad_s": signed_speed_sum / steps,
        "absolute_axis_speed_mean_rad_s": absolute_speed_sum / steps,
        "signed_net_rotation_rad_mean": terminal_extras.get(f"{prefix}/net_rotation_rad_signed_sum", 0.0)
        / episode_count,
        "signed_net_rotation_turns_mean": terminal_extras.get(
            f"{prefix}/net_rotation_turns_signed_sum", 0.0
        )
        / episode_count,
        "subgoals_per_trajectory": subgoal_sum / episode_count,
        "subgoal_throughput_per_env_s": subgoal_sum / (env.unwrapped.num_envs * steps * env.unwrapped.step_dt),
        "positive_full_turn_fraction": full_turn_sum / episode_count,
        "drop_fraction": terminal_extras.get(f"{prefix}/termination_object_out_of_anchor_sum", 0.0)
        / episode_count,
        "axis_failure_fraction": terminal_extras.get(f"{prefix}/termination_goal_axis_misaligned_sum", 0.0)
        / episode_count,
        "timeout_fraction": terminal_extras.get(f"{prefix}/termination_time_out_sum", 0.0) / episode_count,
        "tip_active_count_mean": tip_count_sum / steps,
        "palm_occupancy_fraction": palm_sum / steps,
        "finger_non_tip_occupancy_fraction": non_tip_sum / steps,
        "raw_done_count": done_count,
    }
    observation_raw, _ = env.reset()
    return metrics, cast(dict[str, object], observation_raw)


def _save_checkpoint(
    path: Path,
    *,
    runtime,
    actor_optimizer,
    critic_optimizer,
    provider_identity: dict[str, Any],
    pregrasp_record_digest: str,
    ppo_cfg,
    update: int,
) -> None:
    r"""保存model/optimizers/identity/config与update，不嵌入nonpersistent evidence bank。"""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "1.0.0",
            "model": runtime.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "provider_identity": provider_identity,
            "pregrasp_record_digest": pregrasp_record_digest,
            "ppo_config": asdict(ppo_cfg),
            "update": update,
        },
        path,
    )


def main() -> int:
    r"""运行initial eval、matched PPO updates、strict checkpoint restore与final eval。"""

    import random

    import anymani.tasks.hetero  # noqa: F401
    import gymnasium as gym
    import numpy as np
    import torch
    from anymani.distill.models.heterogeneous_policy import StructuredActorCriticPackage
    from anymani.distill.models.structured_heterogeneous import StructuredActorObservation
    from anymani.distill.rl.runtime.structured_geometry import build_structured_retained_geometry_provider
    from anymani.distill.rl.structured_ppo import StructuredPpoCfg, collect_rollout, update_ppo
    from anymani.distill.rl.structured_runtime import StructuredHeterogeneousRuntime
    from anymani.distill.rl.structured_transport import StructuredRlTransport
    from anymani.pregrasp import PregraspRecord
    from anymani.tasks.hetero.config.generated.tactile_rotation_env_cfg import (
        ASSET_BINDING,
        GeneratedHeterogeneousTactileRotationEnvCfg,
    )
    from anymani.tasks.hetero.mdp.runtime_state import HETERO_PREGRASP_STATE_ATTR, HeterogeneousPregraspState
    from isaaclab.envs import ManagerBasedRLEnv

    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)
    run_dir = ARGS.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = GeneratedHeterogeneousTactileRotationEnvCfg()
    env_cfg.seed = ARGS.seed
    env = gym.make("AnyMani-Hetero-Generated-TactileRotation-v0", cfg=env_cfg)
    try:
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        observation_raw, _ = env.reset()
        observation = cast(dict[str, object], observation_raw)
        sidecar = getattr(runtime_env, HETERO_PREGRASP_STATE_ATTR)
        if not isinstance(sidecar, HeterogeneousPregraspState):
            raise RuntimeError("training env did not install pregrasp sidecar")
        record_digests = {digest for digest in sidecar.record_digests if digest is not None}
        if len(record_digests) != 1:
            raise RuntimeError("single-asset matched run must use one pregrasp record")
        pregrasp_record_digest = next(iter(record_digests))
        cache_payload = Path("outputs/pregrasp/schema_v2/cache/records") / f"{pregrasp_record_digest}.json"
        pregrasp_record = PregraspRecord.from_dict(json.loads(cache_payload.read_text()))
        if pregrasp_record.tier.value != ARGS.tier:
            raise RuntimeError("runtime pregrasp exact tier disagrees with matched arm")

        provider = build_structured_retained_geometry_provider(ASSET_BINDING, device=runtime_env.device)
        # Environment construction may consume RNG; reseed immediately before model init for matched weights。
        torch.manual_seed(ARGS.seed)
        package = StructuredActorCriticPackage().to(runtime_env.device)
        runtime = StructuredHeterogeneousRuntime(provider, cast(StructuredActorCriticPackage, package)).to(
            runtime_env.device
        )
        ppo_cfg = StructuredPpoCfg(
            horizon=ARGS.horizon,
            epochs=ARGS.epochs,
            minibatches=ARGS.minibatches,
        )
        actor_optimizer = torch.optim.Adam(package.actor.parameters(), lr=ppo_cfg.actor_learning_rate)
        critic_optimizer = torch.optim.Adam(package.critic.parameters(), lr=ppo_cfg.critic_learning_rate)
        action_generator = torch.Generator(device=runtime_env.device).manual_seed(ARGS.seed + 1000)
        minibatch_generator = torch.Generator(device=runtime_env.device).manual_seed(ARGS.seed + 2000)
        prototype_index = torch.tensor(
            ASSET_BINDING.asset_index_by_env(runtime_env.num_envs),
            dtype=torch.long,
            device=runtime_env.device,
        )

        initial_evaluation, observation = _evaluate(
            env, runtime, prototype_index, steps=ARGS.eval_steps
        )
        log_path = run_dir / "updates.jsonl"
        update_records = []
        started = time.perf_counter()
        for update in range(1, ARGS.updates + 1):
            rollout, observation, rollout_metrics = collect_rollout(
                env,
                observation,
                runtime,
                prototype_index,
                ppo_cfg,
                action_generator=action_generator,
            )
            update_metrics = update_ppo(
                runtime,
                rollout,
                actor_optimizer,
                critic_optimizer,
                ppo_cfg,
                generator=minibatch_generator,
            )
            record = {
                "update": update,
                **rollout_metrics,
                **update_metrics,
                "coordination_scale": float(package.actor.coordination_scale.detach().item()),
                "global_log_std": float(package.actor.global_log_std.detach().item()),
            }
            update_records.append(record)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

        checkpoint_path = run_dir / "checkpoint_final.pt"
        _save_checkpoint(
            checkpoint_path,
            runtime=runtime,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            provider_identity=provider.identity,
            pregrasp_record_digest=pregrasp_record_digest,
            ppo_cfg=ppo_cfg,
            update=ARGS.updates,
        )

        # Mutate then strict-load，证明checkpoint确实恢复actor/critic/provider namespaces与输出。
        transport = StructuredRlTransport.from_nested_observation(observation, prototype_index)
        actor_observation = StructuredActorObservation.from_task_dict(transport.policy_storage())
        with torch.inference_mode():
            context = runtime.resolve_geometry(prototype_index, actor_observation)
            expected_mean = runtime.actor_forward(actor_observation, context).mean.clone()
        first_actor_parameter = next(package.actor.parameters())
        with torch.no_grad():
            first_actor_parameter.add_(1.0)
        checkpoint = torch.load(checkpoint_path, map_location=runtime_env.device, weights_only=True)
        if checkpoint["provider_identity"] != provider.identity:
            raise RuntimeError("checkpoint provider identity mismatch before restore")
        if checkpoint["pregrasp_record_digest"] != pregrasp_record_digest:
            raise RuntimeError("checkpoint pregrasp identity mismatch before restore")
        runtime.load_state_dict(checkpoint["model"], strict=True)
        with torch.inference_mode():
            restored_mean = runtime.actor_forward(actor_observation, context).mean
        if not torch.equal(restored_mean, expected_mean):
            raise RuntimeError("strict checkpoint restore did not recover exact actor output")

        final_evaluation, _ = _evaluate(env, runtime, prototype_index, steps=ARGS.eval_steps)
        summary = {
            "artifact_type": "anymani.hetero.structured_ppo_run",
            "schema_version": "1.0.0",
            "tier": ARGS.tier,
            "pregrasp_record_digest": pregrasp_record_digest,
            "provider_identity": provider.identity,
            "seed": ARGS.seed,
            "num_envs": ARGS.num_envs,
            "updates": ARGS.updates,
            "transitions": ARGS.num_envs * ARGS.horizon * ARGS.updates,
            "ppo_config": asdict(ppo_cfg),
            "network": {
                "actor": "gated_pool_d128",
                "critic": "masked_pool_d128",
                "precision": "fp32",
                "n040_layers": 4,
                "actor_parameters": sum(parameter.numel() for parameter in package.actor.parameters()),
                "critic_parameters": sum(parameter.numel() for parameter in package.critic.parameters()),
            },
            "initial_evaluation": initial_evaluation,
            "final_evaluation": final_evaluation,
            "last_update": update_records[-1],
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint_path),
            "checkpoint_strict_restore_passed": True,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"summary": str(summary_path), **final_evaluation}, sort_keys=True), flush=True)
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
