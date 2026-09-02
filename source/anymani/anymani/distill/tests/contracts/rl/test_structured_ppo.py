r"""Structured direct PPO的GAE、rollout flatten与独立optimizer合同。"""

from __future__ import annotations

import torch
from anymani.distill.models.heterogeneous_policy import StructuredActorCriticPackage
from anymani.distill.models.structured_heterogeneous import (
    GeometryTokenBatch,
    StructuredActorObservation,
    StructuredCriticObservation,
)
from anymani.distill.rl.structured_masked_distribution import (
    masked_negative_log_probability,
    masked_normal_parameters,
)
from anymani.distill.rl.structured_ppo import (
    StructuredPpoCfg,
    StructuredRollout,
    generalized_advantage_estimate,
    update_ppo,
)
from anymani.distill.rl.structured_runtime import StructuredHeterogeneousRuntime
from torch import nn


class _FrozenProvider(nn.Module):
    r"""Update-only test double；PPO minibatches复用rollout Z，不调用resolve。"""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    @property
    def identity(self) -> dict[str, str]:
        return {"identity_digest": "test"}


def _observations(batch: int):
    r"""构造full-valid actor/critic/geometry fixtures。"""

    joint = torch.ones(batch, 16, dtype=torch.bool)
    tip = torch.ones(batch, 4, dtype=torch.bool)
    owner = torch.ones(batch, 21, dtype=torch.bool)
    actor = StructuredActorObservation(
        torch.randn(batch, 16, 3),
        torch.randn(batch, 30, 16, 4),
        torch.randn(batch, 16, 2),
        torch.zeros(batch, 4, 1),
        joint,
        tip,
        owner,
    )
    critic = StructuredCriticObservation(
        torch.randn(batch, 16, 4),
        torch.randn(batch, 21, 2),
        torch.randn(batch, 1, 15),
        torch.randn(batch, 1, 8),
        joint,
        tip,
        owner,
    )
    geometry = GeometryTokenBatch(torch.randn(batch, 21, 128), owner)
    return actor, critic, geometry


def test_gae_stops_bootstrap_at_done_boundary() -> None:
    r"""Terminal row不使用下一value，nonterminal row使用bootstrap。"""

    rewards = torch.tensor(((1.0, 1.0), (1.0, 1.0)))
    values = torch.zeros_like(rewards)
    dones = torch.tensor(((False, False), (True, False)))
    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        dones,
        torch.tensor((10.0, 10.0)),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.equal(advantages[:, 0], torch.tensor((2.0, 1.0)))
    assert torch.equal(advantages[:, 1], torch.tensor((12.0, 11.0)))
    assert torch.equal(returns, advantages)


def test_synthetic_rollout_updates_disjoint_actor_and_critic_parameters() -> None:
    r"""Two-step rollout经过PPO epochs后两套参数均变化且frozen provider不变。"""

    torch.manual_seed(23)
    time_steps, num_envs = 2, 4
    actor, critic, geometry = _observations(time_steps * num_envs)
    package = StructuredActorCriticPackage()
    runtime = StructuredHeterogeneousRuntime(_FrozenProvider(), package)  # type: ignore[arg-type]
    with torch.no_grad():
        actor_output = package.actor(actor, geometry)
        critic_output = package.critic(critic, geometry)
        distribution = masked_normal_parameters(actor_output.mean, actor_output.log_std, actor.jnt_valid)
        actions = distribution.mean.clone()
        neglogp = masked_negative_log_probability(actions, distribution)

    def time_shape(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(time_steps, num_envs, *value.shape[1:])

    rollout = StructuredRollout(
        actor_terms={
            "jnt_current": time_shape(actor.jnt_current),
            "jnt_history": time_shape(actor.jnt_history),
            "jnt_limits": time_shape(actor.jnt_limits),
            "tip_contact": time_shape(actor.tip_contact),
            "jnt_valid": time_shape(actor.jnt_valid),
            "tip_valid": time_shape(actor.tip_valid),
            "owner_valid": time_shape(actor.owner_valid),
        },
        critic_terms={
            "jnt_state": time_shape(critic.jnt_state),
            "owner_contact": time_shape(critic.owner_contact),
            "obj": time_shape(critic.obj),
            "task": time_shape(critic.task),
            "jnt_valid": time_shape(critic.jnt_valid),
            "tip_valid": time_shape(critic.tip_valid),
            "owner_valid": time_shape(critic.owner_valid),
        },
        geometry_tokens=time_shape(geometry.tokens),
        owner_valid=time_shape(geometry.owner_valid),
        actions=time_shape(actions),
        negative_log_probability=time_shape(neglogp),
        old_mean=time_shape(distribution.mean),
        old_log_std=time_shape(distribution.log_std),
        values=time_shape(critic_output.value),
        rewards=torch.randn(time_steps, num_envs),
        dones=torch.zeros(time_steps, num_envs, dtype=torch.bool),
        advantages=torch.randn(time_steps, num_envs),
        returns=torch.randn(time_steps, num_envs),
    )
    actor_before = {name: value.detach().clone() for name, value in package.actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in package.critic.state_dict().items()}
    actor_optimizer = torch.optim.Adam(package.actor.parameters(), lr=3.0e-4)
    critic_optimizer = torch.optim.Adam(package.critic.parameters(), lr=3.0e-4)
    cfg = StructuredPpoCfg(horizon=2, epochs=2, minibatches=2)
    metrics = update_ppo(
        runtime,
        rollout,
        actor_optimizer,
        critic_optimizer,
        cfg,
        generator=torch.Generator().manual_seed(5),
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(not torch.equal(value, actor_before[name]) for name, value in package.actor.state_dict().items())
    assert any(not torch.equal(value, critic_before[name]) for name, value in package.critic.state_dict().items())
    assert runtime.geometry_provider.anchor.grad is None  # type: ignore[attr-defined]
