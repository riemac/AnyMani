r"""Frozen-N040 structured actor/critic的有界direct PPO rollout与update。

该实现遵循标准GAE与clipped PPO，不改环境/概率科学语义。Rollout每environment step按current q重算冻结Z，并把
对应Z与named observations存入on-policy buffer；由于encoder严格冻结，PPO epochs复用该rollout Z不会stale。
Actor/critic使用独立optimizer，joint mask始终进入Normal likelihood、entropy、KL与bounds归约。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from anymani.distill.models.structured_heterogeneous import (
    GeometryTokenBatch,
    StructuredActorObservation,
    StructuredCriticObservation,
)
from anymani.distill.rl.structured_masked_distribution import (
    MaskedNormalParameters,
    clipped_ppo_actor_loss,
    masked_bound_loss,
    masked_entropy,
    masked_negative_log_probability,
    masked_normal_parameters,
    masked_policy_kl,
    masked_sample,
)
from anymani.distill.rl.structured_runtime import StructuredHeterogeneousRuntime
from anymani.distill.rl.structured_transport import StructuredRlTransport


@dataclass(frozen=True)
class StructuredPpoCfg:
    r"""Matched小cohort的PPO超参数。"""

    horizon: int = 16
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    entropy_coefficient: float = 0.0
    value_coefficient: float = 1.0
    bounds_coefficient: float = 1.0e-4
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        r"""验证PPO概率、次数与学习率。"""

        if min(self.horizon, self.epochs, self.minibatches) < 1:
            raise ValueError("PPO horizon/epochs/minibatches must be positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("PPO gamma/lambda must lie in valid probability intervals")
        if min(self.clip_epsilon, self.actor_learning_rate, self.critic_learning_rate, self.max_grad_norm) <= 0.0:
            raise ValueError("PPO clip/lr/grad norm must be positive")


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""计算time-major GAE$[T,N]$与returns。

    $$
    \delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t,\qquad
    A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}.
    $$
    """

    if rewards.ndim != 2 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("GAE rewards/values/dones must share [T,N] shape")
    if bootstrap_value.shape != rewards.shape[1:]:
        raise ValueError("GAE bootstrap value must have shape [N]")
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    for time_index in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[time_index].to(dtype=rewards.dtype)
        delta = rewards[time_index] + gamma * nonterminal * next_value - values[time_index]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[time_index] = next_advantage
        next_value = values[time_index]
    return advantages, advantages + values


@dataclass
class StructuredRollout:
    r"""Time-major named observations、frozen Z与PPO sufficient tensors。"""

    actor_terms: dict[str, torch.Tensor]  # each$[T,N,...]$
    critic_terms: dict[str, torch.Tensor]
    geometry_tokens: torch.Tensor  # $[T,N,21,128]$
    owner_valid: torch.Tensor  # bool$[T,N,21]$
    actions: torch.Tensor  # $[T,N,16]$
    negative_log_probability: torch.Tensor  # $[T,N]$
    old_mean: torch.Tensor  # $[T,N,16]$
    old_log_std: torch.Tensor  # $[T,N,16]$
    values: torch.Tensor  # $[T,N]$
    rewards: torch.Tensor  # $[T,N]$
    dones: torch.Tensor  # bool$[T,N]$
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    @property
    def time_env_shape(self) -> tuple[int, int]:
        r"""返回$(T,N)$。"""

        return tuple(self.rewards.shape)  # type: ignore[return-value]

    def flatten(self) -> dict[str, Any]:
        r"""把$[T,N,...]$规约为on-policy sample轴$[TN,...]$，不flatten term内部axes。"""

        if self.advantages is None or self.returns is None:
            raise RuntimeError("rollout advantages/returns must be computed before flatten")
        time_steps, num_envs = self.time_env_shape

        def flatten_tensor(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(time_steps * num_envs, *value.shape[2:])

        return {
            "actor_terms": {name: flatten_tensor(value) for name, value in self.actor_terms.items()},
            "critic_terms": {name: flatten_tensor(value) for name, value in self.critic_terms.items()},
            "geometry_tokens": flatten_tensor(self.geometry_tokens),
            "owner_valid": flatten_tensor(self.owner_valid),
            "actions": flatten_tensor(self.actions),
            "old_neglogp": flatten_tensor(self.negative_log_probability),
            "old_mean": flatten_tensor(self.old_mean),
            "old_log_std": flatten_tensor(self.old_log_std),
            "old_values": flatten_tensor(self.values),
            "advantages": flatten_tensor(self.advantages),
            "returns": flatten_tensor(self.returns),
        }


def _stack_term_steps(steps: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    r"""把同名observation term steps堆成time-major tensors。"""

    if not steps:
        raise ValueError("cannot stack empty structured rollout terms")
    names = set(steps[0])
    if any(set(step) != names for step in steps):
        raise ValueError("rollout steps disagree on structured term names")
    return {name: torch.stack([step[name] for step in steps], dim=0) for name in names}


def collect_rollout(
    env: Any,
    observation: Mapping[str, object],
    runtime: StructuredHeterogeneousRuntime,
    prototype_index: torch.Tensor,
    cfg: StructuredPpoCfg,
    *,
    action_generator: torch.Generator,
) -> tuple[StructuredRollout, Mapping[str, object], dict[str, float]]:
    r"""采集一个on-policy horizon并计算GAE，不更新参数。"""

    actor_steps: list[dict[str, torch.Tensor]] = []
    critic_steps: list[dict[str, torch.Tensor]] = []
    geometry_steps: list[torch.Tensor] = []
    owner_mask_steps: list[torch.Tensor] = []
    action_steps: list[torch.Tensor] = []
    neglogp_steps: list[torch.Tensor] = []
    mean_steps: list[torch.Tensor] = []
    log_std_steps: list[torch.Tensor] = []
    value_steps: list[torch.Tensor] = []
    reward_steps: list[torch.Tensor] = []
    done_steps: list[torch.Tensor] = []
    reward_sum = 0.0
    done_sum = 0.0
    current_observation = observation
    runtime.eval()
    for _ in range(cfg.horizon):
        transport = StructuredRlTransport.from_nested_observation(current_observation, prototype_index)
        actor_observation = StructuredActorObservation.from_task_dict(transport.policy_storage())
        critic_observation = StructuredCriticObservation.from_task_dict(transport.critic_storage())
        with torch.no_grad():
            geometry = runtime.resolve_geometry(prototype_index, actor_observation)
            actor_output = runtime.actor_forward(actor_observation, geometry)
            critic_output = runtime.critic_forward(critic_observation, geometry)
            parameters = masked_normal_parameters(
                actor_output.mean, actor_output.log_std, actor_observation.jnt_valid
            )
            noise = torch.randn(
                parameters.mean.shape,
                dtype=parameters.mean.dtype,
                device=parameters.mean.device,
                generator=action_generator,
            )
            actions = masked_sample(parameters, noise)
            neglogp = masked_negative_log_probability(actions, parameters)
        next_observation, rewards, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        actor_steps.append({name: value.detach() for name, value in transport.policy_terms.items()})
        critic_steps.append({name: value.detach() for name, value in transport.critic_terms.items()})
        geometry_steps.append(geometry.tokens.tokens.detach())
        owner_mask_steps.append(geometry.tokens.owner_valid.detach())
        action_steps.append(actions.detach())
        neglogp_steps.append(neglogp.detach())
        mean_steps.append(parameters.mean.detach())
        log_std_steps.append(parameters.log_std.detach())
        value_steps.append(critic_output.value.detach())
        reward_steps.append(rewards.detach())
        done_steps.append(dones.detach())
        reward_sum += float(rewards.mean().item())
        done_sum += float(dones.to(dtype=torch.float32).mean().item())
        current_observation = next_observation

    next_transport = StructuredRlTransport.from_nested_observation(current_observation, prototype_index)
    next_actor = StructuredActorObservation.from_task_dict(next_transport.policy_storage())
    next_critic = StructuredCriticObservation.from_task_dict(next_transport.critic_storage())
    with torch.no_grad():
        next_geometry = runtime.resolve_geometry(prototype_index, next_actor)
        bootstrap = runtime.critic_forward(next_critic, next_geometry).value
    rollout = StructuredRollout(
        actor_terms=_stack_term_steps(actor_steps),
        critic_terms=_stack_term_steps(critic_steps),
        geometry_tokens=torch.stack(geometry_steps),
        owner_valid=torch.stack(owner_mask_steps),
        actions=torch.stack(action_steps),
        negative_log_probability=torch.stack(neglogp_steps),
        old_mean=torch.stack(mean_steps),
        old_log_std=torch.stack(log_std_steps),
        values=torch.stack(value_steps),
        rewards=torch.stack(reward_steps),
        dones=torch.stack(done_steps),
    )
    rollout.advantages, rollout.returns = generalized_advantage_estimate(
        rollout.rewards,
        rollout.values,
        rollout.dones,
        bootstrap,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )
    return rollout, current_observation, {
        "rollout_reward_mean": reward_sum / cfg.horizon,
        "rollout_done_fraction": done_sum / cfg.horizon,
    }


def update_ppo(
    runtime: StructuredHeterogeneousRuntime,
    rollout: StructuredRollout,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    cfg: StructuredPpoCfg,
    *,
    generator: torch.Generator,
) -> dict[str, float]:
    r"""对一个rollout执行matched clipped PPO epochs并返回诊断均值。"""

    flat = rollout.flatten()
    sample_count = flat["actions"].shape[0]
    if sample_count % cfg.minibatches != 0:
        raise ValueError("rollout sample count must be divisible by minibatches")
    minibatch_size = sample_count // cfg.minibatches
    advantages = flat["advantages"]
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-8)
    metrics: dict[str, list[float]] = {
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "bounds_loss": [],
        "kl": [],
        "clip_fraction": [],
        "actor_grad_norm": [],
        "critic_grad_norm": [],
    }
    runtime.train()
    for _ in range(cfg.epochs):
        permutation = torch.randperm(sample_count, generator=generator, device=flat["actions"].device)
        for start in range(0, sample_count, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            actor_observation = StructuredActorObservation.from_task_dict(
                {name: value[indices] for name, value in flat["actor_terms"].items()}
            )
            critic_observation = StructuredCriticObservation.from_task_dict(
                {name: value[indices] for name, value in flat["critic_terms"].items()}
            )
            geometry = GeometryTokenBatch(flat["geometry_tokens"][indices], flat["owner_valid"][indices])

            actor_output = runtime.policy.actor(actor_observation, geometry)
            parameters = masked_normal_parameters(
                actor_output.mean, actor_output.log_std, actor_observation.jnt_valid
            )
            new_neglogp = masked_negative_log_probability(flat["actions"][indices], parameters)
            actor_loss_samples = clipped_ppo_actor_loss(
                new_neglogp,
                flat["old_neglogp"][indices],
                advantages[indices],
                clip_epsilon=cfg.clip_epsilon,
            )
            entropy = masked_entropy(parameters).mean()
            bounds = masked_bound_loss(parameters.mean, actor_observation.jnt_valid).mean()
            actor_loss = (
                actor_loss_samples.mean()
                - cfg.entropy_coefficient * entropy
                + cfg.bounds_coefficient * bounds
            )
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(runtime.policy.actor.parameters(), cfg.max_grad_norm)
            actor_optimizer.step()

            value = runtime.policy.critic(critic_observation, geometry).value
            critic_loss = cfg.value_coefficient * 0.5 * (value - flat["returns"][indices]).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(runtime.policy.critic.parameters(), cfg.max_grad_norm)
            critic_optimizer.step()

            with torch.no_grad():
                old_log_std = flat["old_log_std"][indices]
                old_parameters = MaskedNormalParameters(
                    mean=flat["old_mean"][indices],
                    log_std=old_log_std,
                    sigma=torch.exp(old_log_std),
                    active_mask=actor_observation.jnt_valid,
                )
                kl = masked_policy_kl(parameters, old_parameters).mean()
                ratio = torch.exp(flat["old_neglogp"][indices] - new_neglogp)
                clip_fraction = ((ratio - 1.0).abs() > cfg.clip_epsilon).to(torch.float32).mean()
            metrics["actor_loss"].append(float(actor_loss.detach().item()))
            metrics["critic_loss"].append(float(critic_loss.detach().item()))
            metrics["entropy"].append(float(entropy.detach().item()))
            metrics["bounds_loss"].append(float(bounds.detach().item()))
            metrics["kl"].append(float(kl.item()))
            metrics["clip_fraction"].append(float(clip_fraction.item()))
            metrics["actor_grad_norm"].append(float(actor_grad.item()))
            metrics["critic_grad_norm"].append(float(critic_grad.item()))
    return {name: sum(values) / len(values) for name, values in metrics.items()}


__all__ = [
    "StructuredPpoCfg",
    "StructuredRollout",
    "collect_rollout",
    "generalized_advantage_estimate",
    "update_ppo",
]
