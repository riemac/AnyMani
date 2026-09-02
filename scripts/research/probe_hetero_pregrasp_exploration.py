r"""固定未训练near-hold actor，扫描support/contact pregrasp的action-noise survival。

该probe不训练。每个sigma独立重置多批16-step sequences，使用同一actor seed和standard-Normal noise stream，测量
axis/drop failure、TIP/palm/non-tip contact与signed speed，判断初始exploration distribution是否离开认证盆。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import cast


def _parse_args() -> argparse.Namespace:
    r"""解析tier、sigma grid与probe预算。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("support_basin", "contact_basin"), required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--sigmas", default="0,0.1,0.3,0.6065306597")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = _parse_args()
if min(ARGS.num_envs, ARGS.sequences, ARGS.horizon) < 1:
    raise ValueError("exploration probe counts must be positive")
SIGMAS = tuple(float(value) for value in ARGS.sigmas.split(","))
if not SIGMAS or any(not math.isfinite(value) or value < 0.0 for value in SIGMAS):
    raise ValueError("exploration sigmas must be finite and non-negative")
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = "16"
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(ARGS.num_envs)
os.environ["ANYMANI_HETERO_MIN_PREGRASP_TIER"] = ARGS.tier
os.environ["ANYMANI_HETERO_EXACT_PREGRASP_TIER"] = ARGS.tier

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def main() -> int:
    r"""运行sigma×sequence survival grid并保存durable artifact。"""

    import anymani.tasks.hetero  # noqa: F401
    import gymnasium as gym
    import torch
    from anymani.distill.models.heterogeneous_policy import StructuredActorCriticPackage
    from anymani.distill.models.structured_heterogeneous import StructuredActorObservation
    from anymani.distill.rl.runtime.structured_geometry import build_structured_retained_geometry_provider
    from anymani.distill.rl.structured_runtime import StructuredHeterogeneousRuntime
    from anymani.distill.rl.structured_transport import StructuredRlTransport
    from anymani.tasks.hetero.config.generated.tactile_rotation_env_cfg import (
        ASSET_BINDING,
        GeneratedHeterogeneousTactileRotationEnvCfg,
    )
    from anymani.tasks.hetero.mdp.commands import HeterogeneousRotationCommand
    from anymani.tasks.hetero.mdp.contact_state import HETERO_CONTACT_STATE_ATTR, HeterogeneousContactState
    from anymani.tasks.hetero.mdp.runtime_state import HETERO_PREGRASP_STATE_ATTR, HeterogeneousPregraspState
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = GeneratedHeterogeneousTactileRotationEnvCfg()
    env_cfg.seed = ARGS.seed
    env = gym.make("AnyMani-Hetero-Generated-TactileRotation-v0", cfg=env_cfg)
    try:
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        _ = env.reset()
        pregrasp = getattr(runtime_env, HETERO_PREGRASP_STATE_ATTR)
        if not isinstance(pregrasp, HeterogeneousPregraspState):
            raise RuntimeError("exploration probe lacks pregrasp sidecar")
        record_digests = {value for value in pregrasp.record_digests if value is not None}
        if len(record_digests) != 1:
            raise RuntimeError("exploration probe must use one exact pregrasp record")

        provider = build_structured_retained_geometry_provider(ASSET_BINDING, device=runtime_env.device)
        torch.manual_seed(ARGS.seed)
        package = StructuredActorCriticPackage().to(runtime_env.device)
        runtime = StructuredHeterogeneousRuntime(provider, cast(StructuredActorCriticPackage, package)).to(
            runtime_env.device
        )
        runtime.eval()
        prototype_index = torch.zeros(ARGS.num_envs, dtype=torch.long, device=runtime_env.device)
        sigma_results = []
        with torch.no_grad():
            for sigma in SIGMAS:
                generator = torch.Generator(device=runtime_env.device).manual_seed(ARGS.seed + 3000)
                axis_trajectory_count = 0.0
                drop_trajectory_count = 0.0
                raw_axis_count = 0.0
                raw_drop_count = 0.0
                tip_sum = 0.0
                palm_sum = 0.0
                non_tip_sum = 0.0
                signed_speed_sum = 0.0
                sample_count = 0
                for _ in range(ARGS.sequences):
                    observation_raw, _ = env.reset()
                    observation = cast(dict[str, object], observation_raw)
                    any_axis = torch.zeros(ARGS.num_envs, dtype=torch.bool, device=runtime_env.device)
                    any_drop = torch.zeros_like(any_axis)
                    for _ in range(ARGS.horizon):
                        transport = StructuredRlTransport.from_nested_observation(observation, prototype_index)
                        actor_observation = StructuredActorObservation.from_task_dict(transport.policy_storage())
                        geometry = runtime.resolve_geometry(prototype_index, actor_observation)
                        mean = runtime.actor_forward(actor_observation, geometry).mean
                        noise = torch.randn(
                            mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
                        )
                        actions = (mean + sigma * noise) * actor_observation.jnt_valid.to(dtype=mean.dtype)
                        next_observation, _, _, _, _ = env.step(actions)
                        axis = runtime_env.termination_manager.get_term("goal_axis_misaligned")
                        drop = runtime_env.termination_manager.get_term("object_out_of_anchor")
                        any_axis |= axis
                        any_drop |= drop
                        raw_axis_count += float(axis.to(torch.float32).sum().item())
                        raw_drop_count += float(drop.to(torch.float32).sum().item())
                        contact = getattr(runtime_env, HETERO_CONTACT_STATE_ATTR)
                        if not isinstance(contact, HeterogeneousContactState):
                            raise RuntimeError("exploration probe contact state is missing")
                        tip_sum += float(contact.tip_bits.sum(dim=-1).to(torch.float32).mean().item())
                        palm_sum += float(contact.palm_bits.to(torch.float32).mean().item())
                        non_tip_sum += float(
                            contact.finger_non_tip_bits.any(dim=-1).to(torch.float32).mean().item()
                        )
                        command = runtime_env.command_manager.get_term("goal_pose")
                        if not isinstance(command, HeterogeneousRotationCommand):
                            raise RuntimeError("exploration probe command type mismatch")
                        signed_speed_sum += float(command.axis_speed_rad_s.mean().item())
                        sample_count += 1
                        observation = cast(dict[str, object], next_observation)
                    axis_trajectory_count += float(any_axis.to(torch.float32).sum().item())
                    drop_trajectory_count += float(any_drop.to(torch.float32).sum().item())
                trajectory_count = ARGS.num_envs * ARGS.sequences
                step_env_count = trajectory_count * ARGS.horizon
                sigma_results.append(
                    {
                        "sigma": sigma,
                        "axis_failure_trajectory_fraction": axis_trajectory_count / trajectory_count,
                        "drop_trajectory_fraction": drop_trajectory_count / trajectory_count,
                        "axis_failure_step_fraction": raw_axis_count / step_env_count,
                        "drop_step_fraction": raw_drop_count / step_env_count,
                        "tip_active_count_mean": tip_sum / sample_count,
                        "palm_occupancy_fraction": palm_sum / sample_count,
                        "finger_non_tip_occupancy_fraction": non_tip_sum / sample_count,
                        "signed_axis_speed_mean_rad_s": signed_speed_sum / sample_count,
                    }
                )

        artifact = {
            "artifact_type": "anymani.hetero.pregrasp_exploration_survival",
            "schema_version": "1.0.0",
            "tier": ARGS.tier,
            "pregrasp_record_digest": next(iter(record_digests)),
            "provider_identity_digest": provider.identity["identity_digest"],
            "seed": ARGS.seed,
            "num_envs": ARGS.num_envs,
            "sequences_per_sigma": ARGS.sequences,
            "horizon": ARGS.horizon,
            "trajectories_per_sigma": ARGS.num_envs * ARGS.sequences,
            "actor_initialization": {"mean_head_gain": 0.01, "global_log_std": -0.5},
            "results": sigma_results,
        }
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        ARGS.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(ARGS.output), "tier": ARGS.tier, "results": sigma_results}, sort_keys=True))
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
