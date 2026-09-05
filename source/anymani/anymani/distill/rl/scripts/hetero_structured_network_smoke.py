r"""2-asset structured env到N040 actor/critic/masked distribution的端到端非训练smoke。

该probe只执行一次synthetic backward与一次sampled environment step，不创建optimizer、不更新参数，也不进入PPO。
它证明task named tensors、prototype routing、frozen current-q Z、actor/critic和概率mask能够在同一runtime闭合。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("ANYMANI_HETERO_ASSET_ROWS", "0,16")
os.environ.setdefault("ANYMANI_HETERO_MIN_PREGRASP_TIER", "support_basin")

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def _parse_args() -> argparse.Namespace:
    r"""解析device与durable output。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hetero/runtime-smokes/structured-network-rows-0-16.json"),
    )
    return parser.parse_args()


def main() -> int:
    r"""运行actual observation transport、network、backward与sampled step。"""

    import anymani.tasks.hetero  # noqa: F401
    import gymnasium as gym
    import torch
    from anymani.distill.models.heterogeneous_policy import StructuredActorCriticPackage
    from anymani.distill.models.structured_heterogeneous import (
        StructuredActorObservation,
        StructuredCriticObservation,
    )
    from anymani.distill.rl.runtime.structured_geometry import build_structured_retained_geometry_provider
    from anymani.distill.rl.structured_masked_distribution import (
        masked_entropy,
        masked_negative_log_probability,
        masked_normal_parameters,
        masked_sample,
    )
    from anymani.distill.rl.structured_runtime import StructuredHeterogeneousRuntime
    from anymani.distill.rl.structured_transport import StructuredRlTransport
    from anymani.tasks.hetero.config.generated.tactile_rotation_env_cfg import (
        ASSET_BINDING,
        GeneratedHeterogeneousTactileRotationEnvCfg,
    )
    from isaaclab.envs import ManagerBasedRLEnv

    args = _parse_args()
    device = torch.device(args.device)
    env = gym.make(
        "AnyMani-Hetero-Generated-TactileRotation-v0",
        cfg=GeneratedHeterogeneousTactileRotationEnvCfg(),
    )
    try:
        runtime_env = cast(ManagerBasedRLEnv, env.unwrapped)
        runtime_env.sim._app_control_on_stop_handle = None
        observation_raw, _ = env.reset()
        observation = cast(dict[str, object], observation_raw)
        prototype_index = torch.tensor(
            ASSET_BINDING.asset_index_by_env(runtime_env.num_envs), dtype=torch.long, device=device
        )
        transport = StructuredRlTransport.from_nested_observation(observation, prototype_index)
        actor_observation = StructuredActorObservation.from_task_dict(transport.policy_storage())
        critic_observation = StructuredCriticObservation.from_task_dict(transport.critic_storage())
        if not torch.allclose(
            actor_observation.jnt_current[..., 0], actor_observation.jnt_history[:, -1, :, 0], atol=1.0e-7
        ):
            raise AssertionError("current q and latest History30 q disagree")

        provider = build_structured_retained_geometry_provider(ASSET_BINDING, device=device)
        package = StructuredActorCriticPackage().to(device)
        network = StructuredHeterogeneousRuntime(provider, cast(StructuredActorCriticPackage, package)).to(device)
        network.train()
        geometry = network.resolve_geometry(prototype_index, actor_observation)
        actor_output = network.actor_forward(actor_observation, geometry)
        critic_output = network.critic_forward(critic_observation, geometry)
        parameters = masked_normal_parameters(
            actor_output.mean, actor_output.log_std, actor_observation.jnt_valid
        )
        actions = masked_sample(parameters, torch.zeros_like(parameters.mean))
        neglogp = masked_negative_log_probability(actions, parameters)
        entropy = masked_entropy(parameters)
        if not torch.equal(actions[~actor_observation.jnt_valid], torch.zeros_like(actions[~actor_observation.jnt_valid])):
            raise AssertionError("masked network emitted nonzero ghost action")

        synthetic_loss = (
            actor_output.mean[actor_observation.jnt_valid].square().mean()
            + critic_output.value.square().mean()
            - 0.01 * entropy.mean()
        )
        synthetic_loss.backward()
        if any(parameter.grad is not None for parameter in provider.parameters()):
            raise AssertionError("frozen N040 received gradients")
        actor_grad_squares = [
            parameter.grad.detach().square().sum()
            for parameter in package.actor.parameters()
            if parameter.grad is not None
        ]
        critic_grad_squares = [
            parameter.grad.detach().square().sum()
            for parameter in package.critic.parameters()
            if parameter.grad is not None
        ]
        if not actor_grad_squares or not critic_grad_squares:
            raise AssertionError("structured actor/critic received no gradients")
        actor_grad_norm = torch.sqrt(torch.stack(actor_grad_squares).sum())
        critic_grad_norm = torch.sqrt(torch.stack(critic_grad_squares).sum())
        if not bool(torch.isfinite(actor_grad_norm).item()) or not bool(torch.isfinite(critic_grad_norm).item()):
            raise AssertionError("structured actor/critic gradient norm is non-finite")

        next_observation, reward_raw, terminated_raw, truncated_raw, _ = env.step(actions.detach())
        reward = cast(torch.Tensor, reward_raw)
        terminated = cast(torch.Tensor, terminated_raw)
        truncated = cast(torch.Tensor, truncated_raw)
        if not bool(torch.isfinite(reward).all().item()):
            raise AssertionError("sampled structured network step produced non-finite reward")
        state_keys = tuple(network.state_dict())
        evidence = {
            "artifact_type": "anymani.hetero.structured_network_smoke",
            "schema_version": "1.0.0",
            "dataset_rows": list(ASSET_BINDING.dataset_rows),
            "provider_identity_digest": provider.identity["identity_digest"],
            "geometry_shape": list(geometry.tokens.tokens.shape),
            "actor_mean_shape": list(actor_output.mean.shape),
            "critic_value_shape": list(critic_output.value.shape),
            "global_log_std_numel": actor_output.log_std.numel(),
            "negative_log_probability": [float(value) for value in neglogp.tolist()],
            "entropy": [float(value) for value in entropy.tolist()],
            "synthetic_loss": float(synthetic_loss.item()),
            "actor_grad_norm": float(actor_grad_norm.item()),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "provider_has_grad": False,
            "sampled_step_reward": [float(value) for value in reward.tolist()],
            "sampled_step_done": [bool(value) for value in (terminated | truncated).tolist()],
            "next_observation_groups": sorted(cast(dict[str, Any], next_observation)),
            "checkpoint_namespaces": {
                "geometry_provider": sum(key.startswith("geometry_provider.") for key in state_keys),
                "actor": sum(key.startswith("policy.actor.") for key in state_keys),
                "critic": sum(key.startswith("policy.critic.") for key in state_keys),
            },
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
