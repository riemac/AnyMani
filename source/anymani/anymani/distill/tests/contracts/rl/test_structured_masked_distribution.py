r"""Structured masked Normal与synthetic PPO backward合同。"""

from __future__ import annotations

import torch
from anymani.distill.models.heterogeneous_policy import StructuredActorCriticPackage
from anymani.distill.models.structured_heterogeneous import (
    GeometryTokenBatch,
    StructuredActorObservation,
    StructuredCriticObservation,
)
from anymani.distill.rl.structured_masked_distribution import (
    clipped_ppo_actor_loss,
    masked_bound_loss,
    masked_entropy,
    masked_negative_log_probability,
    masked_normal_parameters,
    masked_policy_kl,
    masked_sample,
)


def _fixture(batch: int = 4):
    r"""构造可微actor/critic/geometry输入。"""

    torch.manual_seed(19)
    joint = torch.tensor(
        (True, True, True, True, True, True, True, True, True, False, False, True, False, False, False, False)
    ).repeat(batch, 1)
    tip = torch.ones(batch, 4, dtype=torch.bool)
    owner = torch.cat((torch.ones(batch, 1, dtype=torch.bool), joint, tip), dim=-1)
    actor = StructuredActorObservation(
        torch.randn(batch, 16, 3),
        torch.randn(batch, 30, 16, 4),
        torch.randn(batch, 16, 2),
        torch.randint(0, 2, (batch, 4, 1)).float(),
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


def test_sample_logprob_entropy_kl_and_bounds_ignore_ghost_dimensions() -> None:
    r"""Ghost action/mean任意变化不进入联合概率、KL或bounds，sample ghost严格0。"""

    _, _, geometry = _fixture(batch=3)
    mask = geometry.owner_valid[:, 1:17]
    mean = torch.randn(3, 16)
    global_log_std = torch.tensor(-0.3)
    parameters = masked_normal_parameters(mean, global_log_std, mask)
    noise = torch.randn_like(mean)
    sample = masked_sample(parameters, noise)
    assert torch.equal(sample[~mask], torch.zeros_like(sample[~mask]))
    neglogp = masked_negative_log_probability(sample, parameters)
    entropy = masked_entropy(parameters)
    bound = masked_bound_loss(parameters.mean, mask)

    poisoned_mean = mean.clone()
    poisoned_mean[~mask] = 1.0e6
    poisoned = masked_normal_parameters(poisoned_mean, global_log_std, mask)
    poisoned_actions = sample.clone()
    poisoned_actions[~mask] = -1.0e6
    assert torch.equal(masked_negative_log_probability(poisoned_actions, poisoned), neglogp)
    assert torch.equal(masked_entropy(poisoned), entropy)
    assert torch.equal(masked_bound_loss(poisoned.mean, mask), bound)
    assert torch.allclose(masked_policy_kl(poisoned, parameters), torch.zeros(3), atol=2.0e-5)


def test_synthetic_masked_ppo_actor_critic_backward_is_finite() -> None:
    r"""PPO surrogate+entropy+bounds+value loss对actor/critic产生finite且分离的gradients。"""

    actor_obs, critic_obs, geometry = _fixture()
    package = StructuredActorCriticPackage()
    actor_output = package.actor(actor_obs, geometry)
    critic_output = package.critic(critic_obs, geometry)
    current = masked_normal_parameters(actor_output.mean, actor_output.log_std, actor_obs.jnt_valid)
    actions = masked_sample(current, torch.randn_like(current.mean))
    new_neglogp = masked_negative_log_probability(actions, current)
    old = masked_normal_parameters(
        (actor_output.mean.detach() + 0.01) * actor_obs.jnt_valid,
        actor_output.log_std.detach() + 0.02,
        actor_obs.jnt_valid,
    )
    old_neglogp = masked_negative_log_probability(actions, old).detach()
    advantage = torch.tensor((1.0, -0.5, 0.25, -1.0))
    actor_loss = clipped_ppo_actor_loss(new_neglogp, old_neglogp, advantage).mean()
    entropy_loss = -0.01 * masked_entropy(current).mean()
    bounds_loss = 0.01 * masked_bound_loss(current.mean, actor_obs.jnt_valid).mean()
    value_target = torch.linspace(-1.0, 1.0, 4)
    critic_loss = 0.5 * (critic_output.value - value_target).square().mean()
    total = actor_loss + entropy_loss + bounds_loss + critic_loss
    total.backward()
    assert torch.isfinite(total)
    actor_gradients = [parameter.grad for parameter in package.actor.parameters() if parameter.requires_grad]
    critic_gradients = [parameter.grad for parameter in package.critic.parameters() if parameter.requires_grad]
    assert all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in actor_gradients)
    assert all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in critic_gradients)
    assert package.actor.global_log_std.grad is not None
