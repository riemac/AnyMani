r"""同一row16上运行exact support/contact tier的matched structured PPO。

脚本固定四层FP32 N040、D128 gated-pool actor、独立masked-pooling critic与standard GAE/clipped PPO。两臂只由
``--tier``改变pregrasp exact record；模型seed、action-noise stream、minibatch stream、环境数与预算由CLI完整记录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
    parser.add_argument("--initial-log-std", type=float, default=-0.5)
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


RUN_SOURCE_FILES = (
    "scripts/research/train_hetero_structured_ppo.py",
    "source/anymani/anymani/tasks/hetero/config/generated/pregrasp_identity.py",
    "source/anymani/anymani/tasks/hetero/config/generated/tactile_rotation_env_cfg.py",
    "source/anymani/anymani/tasks/hetero/mdp/actions.py",
    "source/anymani/anymani/tasks/hetero/mdp/commands.py",
    "source/anymani/anymani/tasks/hetero/mdp/contact_state.py",
    "source/anymani/anymani/tasks/hetero/mdp/diagnostics.py",
    "source/anymani/anymani/tasks/hetero/mdp/rewards.py",
    "source/anymani/anymani/tasks/hetero/mdp/terminations.py",
    "source/anymani/anymani/distill/models/heterogeneous_policy.py",
    "source/anymani/anymani/distill/rl/structured_evaluation.py",
    "source/anymani/anymani/distill/rl/structured_ppo.py",
    "source/anymani/anymani/distill/rl/structured_runtime.py",
)
r"""改变matched estimand、前向、PPO或evaluation的最小source identity集合。"""


def _file_sha256(path: Path) -> str:
    r"""计算artifact/source bytes SHA-256，不信任路径或mtime。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, Any]:
    r"""记录Git commit、相关source digests及aggregate digest。"""

    source_digests = {path: _file_sha256(Path(path)) for path in RUN_SOURCE_FILES}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "--", *RUN_SOURCE_FILES], check=False
    ).returncode
    if dirty != 0:
        raise RuntimeError("matched PPO source files must be committed before formal execution")
    aggregate = hashlib.sha256(
        json.dumps(source_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"git_commit": commit, "source_files": source_digests, "source_bundle_digest": aggregate}


def _clone_state_tree(value: Any) -> Any:
    r"""递归复制optimizer state，防止后续原位mutation污染expected fixture。"""

    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_tree(item) for item in value)
    return value


def _state_trees_equal(left: Any, right: Any) -> bool:
    r"""递归比较model-independent optimizer scalar/tensor state。"""

    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_state_trees_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_state_trees_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _mutate_optimizer_state(optimizer: Any) -> None:
    r"""原位扰动Adam moments/steps，证明load_state_dict而非未变化假阳性。"""

    import torch

    mutated = 0
    with torch.no_grad():
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.add_(1.0)
                    mutated += 1
    if mutated == 0:
        raise RuntimeError("formal PPO optimizer has no state tensors to test restore")


def _evaluate(
    env,
    runtime,
    prototype_index,
    *,
    steps: int,
) -> tuple[dict[str, Any], dict[str, object], list[dict[str, Any]]]:
    r"""只读pre-reset snapshot执行deterministic fixed-duration trajectory evaluation。"""

    import torch
    from anymani.distill.models.structured_heterogeneous import StructuredActorObservation
    from anymani.distill.rl.structured_evaluation import FixedDurationTrajectoryAccumulator
    from anymani.distill.rl.structured_transport import StructuredRlTransport
    from anymani.tasks.hetero.mdp.commands import HeterogeneousRotationCommand

    observation_raw, _ = env.reset()
    observation = cast(dict[str, object], observation_raw)
    runtime.eval()
    command = env.unwrapped.command_manager.get_term("goal_pose")
    if not isinstance(command, HeterogeneousRotationCommand):
        raise RuntimeError("evaluation command type mismatch")
    dataset_rows = torch.tensor(command.cfg.dataset_row_by_env, dtype=torch.long, device=env.unwrapped.device)
    accumulator = FixedDurationTrajectoryAccumulator(dataset_rows, step_dt=float(env.unwrapped.step_dt))
    latest_snapshot: dict[str, torch.Tensor] | None = None
    # Env managers会持久化/原地更新physics buffers；使用no_grad而非InferenceMode，避免生成不可变inference tensor。
    with torch.no_grad():
        for _ in range(steps):
            transport = StructuredRlTransport.from_nested_observation(observation, prototype_index)
            actor_observation = StructuredActorObservation.from_task_dict(transport.policy_storage())
            geometry = runtime.resolve_geometry(prototype_index, actor_observation)
            actions = runtime.actor_forward(actor_observation, geometry).mean
            next_observation, rewards, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated
            latest_snapshot = command.post_physics_evaluation_snapshot
            if not bool((latest_snapshot["step"] == int(env.unwrapped.common_step_counter)).all().item()):
                raise RuntimeError("evaluation snapshot was not captured on the returned physics step")
            accumulator.add_step(latest_snapshot, rewards, dones)
            observation = cast(dict[str, object], next_observation)

    if latest_snapshot is None:
        raise RuntimeError("evaluation produced no post-physics snapshot")
    accumulator.finish_window(latest_snapshot)
    metrics = accumulator.summary(requested_steps=steps)
    observation_raw, _ = env.reset()
    return metrics, cast(dict[str, object], observation_raw), accumulator.records


def _save_checkpoint(
    path: Path,
    *,
    runtime,
    actor_optimizer,
    critic_optimizer,
    provider_identity: dict[str, Any],
    pregrasp_record_digest: str,
    ppo_cfg,
    run_identity: dict[str, Any],
    update: int,
) -> None:
    r"""保存model/optimizers/identity/config与update，不嵌入nonpersistent evidence bank。"""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "2.0.0",
            "model": runtime.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "provider_identity": provider_identity,
            "pregrasp_record_digest": pregrasp_record_digest,
            "ppo_config": asdict(ppo_cfg),
            "run_identity": run_identity,
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
    from anymani.distill.models.heterogeneous_policy import StructuredActorCfg, StructuredActorCriticPackage
    from anymani.distill.models.structured_heterogeneous import StructuredActorObservation
    from anymani.distill.rl.runtime.structured_geometry import build_structured_retained_geometry_provider
    from anymani.distill.rl.structured_evaluation import REQUIRED_SNAPSHOT_FIELDS, write_trajectory_jsonl
    from anymani.distill.rl.structured_ppo import StructuredPpoCfg, collect_rollout, update_ppo
    from anymani.distill.rl.structured_runtime import StructuredHeterogeneousRuntime
    from anymani.distill.rl.structured_transport import StructuredRlTransport
    from anymani.pregrasp import AtomicPregraspCache, PregraspRecord
    from anymani.pregrasp.schema import stable_digest
    from anymani.tasks.hetero.config.generated.asset_binding import DEFAULT_PREGRASP_CACHE_ROOT
    from anymani.tasks.hetero.config.generated.pregrasp_identity import (
        FORMAL_PREGRASP_GATE,
        FORMAL_SEARCH_PROTOCOL_DIGEST,
    )
    from anymani.tasks.hetero.config.generated.tactile_rotation_env_cfg import (
        ASSET_BINDING,
        GeneratedHeterogeneousTactileRotationEnvCfg,
    )
    from anymani.tasks.hetero.mdp.actions import POLICY_STEP_AUTHORITY_RAD
    from anymani.tasks.hetero.mdp.runtime_state import HETERO_PREGRASP_STATE_ATTR, HeterogeneousPregraspState
    from isaaclab.envs import ManagerBasedRLEnv

    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)
    run_dir = ARGS.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    source_identity = _source_identity()
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
        cache_payload = DEFAULT_PREGRASP_CACHE_ROOT / "records" / f"{pregrasp_record_digest}.json"
        pregrasp_record = PregraspRecord.from_dict(json.loads(cache_payload.read_text()))
        if pregrasp_record.tier.value != ARGS.tier:
            raise RuntimeError("runtime pregrasp exact tier disagrees with matched arm")
        cache_index = AtomicPregraspCache(DEFAULT_PREGRASP_CACHE_ROOT).load_index()

        provider = build_structured_retained_geometry_provider(ASSET_BINDING, device=runtime_env.device)
        # Environment construction may consume RNG; reseed immediately before model init for matched weights。
        torch.manual_seed(ARGS.seed)
        package = StructuredActorCriticPackage(
            actor_cfg=StructuredActorCfg(initial_log_std=ARGS.initial_log_std)
        ).to(runtime_env.device)
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
        launch_arguments = {
            "tier": ARGS.tier,
            "num_envs": ARGS.num_envs,
            "updates": ARGS.updates,
            "horizon": ARGS.horizon,
            "epochs": ARGS.epochs,
            "minibatches": ARGS.minibatches,
            "seed": ARGS.seed,
            "initial_log_std": ARGS.initial_log_std,
            "eval_steps": ARGS.eval_steps,
            "run_dir": str(run_dir),
            "argv": list(sys.argv),
        }
        matched_task_contract = {
            "gym_id": "AnyMani-Hetero-Generated-TactileRotation-v0",
            "dataset_rows": list(ASSET_BINDING.dataset_rows),
            "source_content_hash": pregrasp_record.lookup_key.source_content_hash,
            "physical_geometry_hash": pregrasp_record.lookup_key.physical_geometry_hash,
            "canonical_schema_digest": pregrasp_record.lookup_key.canonical_schema_digest,
            "routing_digest": pregrasp_record.lookup_key.routing_digest,
            "cube_asset_sha256": pregrasp_record.lookup_key.cube_asset_sha256,
            "object_scale": pregrasp_record.candidate.object_scale,
            "gate_digest": FORMAL_PREGRASP_GATE.digest,
            "physics_identity_digest": stable_digest(pregrasp_record.lookup_key.physics_identity),
            "search_protocol_digest": FORMAL_SEARCH_PROTOCOL_DIGEST,
            "action": {
                "authority_rad_per_policy_step": POLICY_STEP_AUTHORITY_RAD,
                "physics_substeps": 6,
                "target_update_count_per_policy_step": 1,
            },
            "reward_weights": {
                "pose_keypoint": 1.0,
                "rotation_progress": 5.0,
                "goal_success": 10.0,
                "good_tip_contact": 0.1,
                "bad_finger_non_tip_contact": -0.2,
                "failure": -50.0,
            },
            "termination": {"drop_distance_m": 0.07, "max_axis_angle_deg": 45.0, "horizon_s": 120.0},
            "evaluation": {
                "mode": "deterministic_actor_mean_fixed_duration",
                "lifecycle": "task_post_physics_pre_reset_snapshot",
                "snapshot_fields": list(REQUIRED_SNAPSHOT_FIELDS),
                "eval_steps": ARGS.eval_steps,
            },
        }
        run_identity = {
            "source": source_identity,
            "launch_arguments": launch_arguments,
            "matched_task_contract": matched_task_contract,
            "matched_task_contract_digest": stable_digest(matched_task_contract),
            "arm": {
                "tier": ARGS.tier,
                "pregrasp_record_digest": pregrasp_record_digest,
                "pregrasp_lookup_digest": pregrasp_record.lookup_key.digest,
                "search_identity_digest": stable_digest(pregrasp_record.lookup_key.search_identity),
            },
            "formal_cache_index_digest": cache_index.digest,
            "provider_identity_digest": provider.identity["identity_digest"],
        }

        initial_evaluation, observation, initial_trajectories = _evaluate(
            env, runtime, prototype_index, steps=ARGS.eval_steps
        )
        initial_trajectory_path = run_dir / "initial_evaluation_trajectories.jsonl"
        initial_trajectory_sha256 = write_trajectory_jsonl(initial_trajectory_path, initial_trajectories)
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
            run_identity=run_identity,
            update=ARGS.updates,
        )
        checkpoint_sha256 = _file_sha256(checkpoint_path)

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
        if checkpoint["run_identity"] != run_identity:
            raise RuntimeError("checkpoint run identity mismatch before restore")
        expected_actor_optimizer = _clone_state_tree(actor_optimizer.state_dict())
        expected_critic_optimizer = _clone_state_tree(critic_optimizer.state_dict())
        _mutate_optimizer_state(actor_optimizer)
        _mutate_optimizer_state(critic_optimizer)
        runtime.load_state_dict(checkpoint["model"], strict=True)
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if not _state_trees_equal(actor_optimizer.state_dict(), expected_actor_optimizer):
            raise RuntimeError("checkpoint did not restore exact actor Adam state")
        if not _state_trees_equal(critic_optimizer.state_dict(), expected_critic_optimizer):
            raise RuntimeError("checkpoint did not restore exact critic Adam state")
        with torch.inference_mode():
            restored_mean = runtime.actor_forward(actor_observation, context).mean
        if not torch.equal(restored_mean, expected_mean):
            raise RuntimeError("strict checkpoint restore did not recover exact actor output")

        final_evaluation, _, final_trajectories = _evaluate(env, runtime, prototype_index, steps=ARGS.eval_steps)
        final_trajectory_path = run_dir / "final_evaluation_trajectories.jsonl"
        final_trajectory_sha256 = write_trajectory_jsonl(final_trajectory_path, final_trajectories)
        summary = {
            "artifact_type": "anymani.hetero.structured_ppo_run",
            "schema_version": "2.0.0",
            "tier": ARGS.tier,
            "pregrasp_record_digest": pregrasp_record_digest,
            "provider_identity": provider.identity,
            "run_identity": run_identity,
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
                "initial_log_std": ARGS.initial_log_std,
                "actor_parameters": sum(parameter.numel() for parameter in package.actor.parameters()),
                "critic_parameters": sum(parameter.numel() for parameter in package.critic.parameters()),
            },
            "initial_evaluation": initial_evaluation,
            "initial_evaluation_trajectories": {
                "path": str(initial_trajectory_path),
                "sha256": initial_trajectory_sha256,
                "count": len(initial_trajectories),
            },
            "final_evaluation": final_evaluation,
            "final_evaluation_trajectories": {
                "path": str(final_trajectory_path),
                "sha256": final_trajectory_sha256,
                "count": len(final_trajectories),
            },
            "last_update": update_records[-1],
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_strict_restore_passed": True,
            "optimizer_checkpoint_restore_passed": True,
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
