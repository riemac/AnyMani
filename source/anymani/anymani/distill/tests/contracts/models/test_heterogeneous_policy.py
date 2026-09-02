r"""轻量structured actor candidates与独立scalar critic核心合同。"""

from __future__ import annotations

from dataclasses import replace

import torch
from anymani.distill.models.heterogeneous_policy import (
    CoordinationKind,
    StructuredActorCfg,
    StructuredActorCriticPackage,
    StructuredCriticCfg,
    StructuredHeterogeneousActor,
    StructuredHeterogeneousCritic,
)
from anymani.distill.models.structured_heterogeneous import (
    GeometryTokenBatch,
    StructuredActorObservation,
    StructuredCriticObservation,
)


def _inputs(batch: int = 3) -> tuple[StructuredActorObservation, StructuredCriticObservation, GeometryTokenBatch]:
    r"""构造10-DoF、4-TIP、21-owner structured fixtures。"""

    torch.manual_seed(7)
    joint = torch.tensor(
        (True, True, True, True, True, True, True, True, True, False, False, True, False, False, False, False)
    ).repeat(batch, 1)
    tip = torch.ones(batch, 4, dtype=torch.bool)
    owner = torch.cat((torch.ones(batch, 1, dtype=torch.bool), joint, tip), dim=-1)
    actor = StructuredActorObservation(
        jnt_current=torch.randn(batch, 16, 3),
        jnt_history=torch.randn(batch, 30, 16, 4),
        jnt_limits=torch.randn(batch, 16, 2),
        tip_contact=torch.randint(0, 2, (batch, 4, 1)).float(),
        jnt_valid=joint,
        tip_valid=tip,
        owner_valid=owner,
    )
    critic = StructuredCriticObservation(
        jnt_state=torch.randn(batch, 16, 4),
        owner_contact=torch.randn(batch, 21, 2),
        obj=torch.randn(batch, 1, 15),
        task=torch.randn(batch, 1, 8),
        jnt_valid=joint,
        tip_valid=tip,
        owner_valid=owner,
    )
    geometry = GeometryTokenBatch(tokens=torch.randn(batch, 21, 128), owner_valid=owner)
    return actor, critic, geometry


def _poison_ghosts(
    actor: StructuredActorObservation,
    critic: StructuredCriticObservation,
    geometry: GeometryTokenBatch,
) -> tuple[StructuredActorObservation, StructuredCriticObservation, GeometryTokenBatch]:
    r"""把所有invalid JOINT/owner slots置大finite poison。"""

    current = actor.jnt_current.clone()
    current[~actor.jnt_valid] = 1.0e4
    history = actor.jnt_history.clone()
    history_mask = actor.jnt_valid[:, None, :, None].expand_as(history)
    history[~history_mask] = -1.0e4
    limits = actor.jnt_limits.clone()
    limits[~actor.jnt_valid] = 1.0e4
    actor_poisoned = replace(actor, jnt_current=current, jnt_history=history, jnt_limits=limits)

    joint_state = critic.jnt_state.clone()
    joint_state[~critic.jnt_valid] = -1.0e4
    owner_contact = critic.owner_contact.clone()
    owner_contact[~critic.owner_valid] = 1.0e4
    critic_poisoned = replace(critic, jnt_state=joint_state, owner_contact=owner_contact)
    tokens = geometry.tokens.clone()
    tokens[~geometry.owner_valid] = -1.0e4
    return actor_poisoned, critic_poisoned, GeometryTokenBatch(tokens=tokens, owner_valid=geometry.owner_valid)


def test_actor_candidates_output_shared_scalar_distribution_and_critic_scalar() -> None:
    r"""三类actor均输出$[B,16]$mean+单scalar logstd，critic输出$[B]$。"""

    actor_obs, critic_obs, geometry = _inputs()
    candidates: tuple[tuple[CoordinationKind, int], ...] = (
        ("local", 128),
        ("gated_pool", 128),
        ("cross_attention", 96),
    )
    for coordination, width in candidates:
        actor = StructuredHeterogeneousActor(
            StructuredActorCfg(hidden_width=width, temporal_width=32, coordination=coordination)
        )
        output = actor(actor_obs, geometry)
        assert output.mean.shape == (3, 16)
        assert output.log_std.shape == () and output.log_std.numel() == 1
        assert float(output.log_std.item()) == -0.5
        assert float(output.mean.abs().max().item()) < 0.1  # gain0.01保证初始策略近似hold pregrasp
        assert torch.equal(output.mean[~actor_obs.jnt_valid], torch.zeros_like(output.mean[~actor_obs.jnt_valid]))
    critic = StructuredHeterogeneousCritic(StructuredCriticCfg(hidden_width=128))
    assert critic(critic_obs, geometry).value.shape == (3,)


def test_actor_critic_parameters_are_disjoint_and_privilege_cannot_change_actor() -> None:
    r"""Critic object/task变化不进入actor；两套trainable parameters无交集。"""

    actor_obs, critic_obs, geometry = _inputs()
    package = StructuredActorCriticPackage()
    actor_ids, critic_ids = package.trainable_parameter_sets()
    assert actor_ids.isdisjoint(critic_ids)
    mean_before = package.actor(actor_obs, geometry).mean
    privileged_changed = replace(critic_obs, obj=critic_obs.obj + 100.0, task=critic_obs.task - 100.0)
    _ = package.critic(privileged_changed, geometry)
    mean_after = package.actor(actor_obs, geometry).mean
    assert torch.equal(mean_before, mean_after)


def test_poisoned_ghosts_do_not_change_actor_or_critic_outputs() -> None:
    r"""大finite poison对active mean与hand value零影响。"""

    actor_obs, critic_obs, geometry = _inputs()
    poisoned_actor, poisoned_critic, poisoned_geometry = _poison_ghosts(actor_obs, critic_obs, geometry)
    package = StructuredActorCriticPackage()
    actor_clean = package.actor(actor_obs, geometry).mean
    actor_poison = package.actor(poisoned_actor, poisoned_geometry).mean
    critic_clean = package.critic(critic_obs, geometry).value
    critic_poison = package.critic(poisoned_critic, poisoned_geometry).value
    assert torch.allclose(actor_clean, actor_poison, atol=1.0e-6, rtol=0.0)
    assert torch.allclose(critic_clean, critic_poison, atol=1.0e-6, rtol=0.0)


def test_joint_permutation_is_actor_equivariant_and_critic_invariant() -> None:
    r"""无absolute-slot embedding时，JOINT合法置换只置换actor输出且不改变critic scalar。"""

    actor_obs, critic_obs, geometry = _inputs(batch=2)
    permutation = torch.tensor((3, 1, 7, 0, 5, 2, 8, 6, 4, 9, 10, 11, 12, 13, 14, 15))
    actor_permuted = replace(
        actor_obs,
        jnt_current=actor_obs.jnt_current[:, permutation],
        jnt_history=actor_obs.jnt_history[:, :, permutation],
        jnt_limits=actor_obs.jnt_limits[:, permutation],
        jnt_valid=actor_obs.jnt_valid[:, permutation],
        owner_valid=torch.cat(
            (actor_obs.owner_valid[:, :1], actor_obs.jnt_valid[:, permutation], actor_obs.tip_valid), dim=-1
        ),
    )
    critic_permuted = replace(
        critic_obs,
        jnt_state=critic_obs.jnt_state[:, permutation],
        owner_contact=torch.cat(
            (
                critic_obs.owner_contact[:, :1],
                critic_obs.owner_contact[:, 1:17][:, permutation],
                critic_obs.owner_contact[:, 17:21],
            ),
            dim=1,
        ),
        jnt_valid=critic_obs.jnt_valid[:, permutation],
        owner_valid=torch.cat(
            (critic_obs.owner_valid[:, :1], critic_obs.jnt_valid[:, permutation], critic_obs.tip_valid), dim=-1
        ),
    )
    geometry_permuted = GeometryTokenBatch(
        tokens=torch.cat(
            (geometry.tokens[:, :1], geometry.tokens[:, 1:17][:, permutation], geometry.tokens[:, 17:21]), dim=1
        ),
        owner_valid=actor_permuted.owner_valid,
    )
    actor = StructuredHeterogeneousActor()
    critic = StructuredHeterogeneousCritic()
    mean = actor(actor_obs, geometry).mean
    mean_permuted = actor(actor_permuted, geometry_permuted).mean
    value = critic(critic_obs, geometry).value
    value_permuted = critic(critic_permuted, geometry_permuted).value
    assert torch.allclose(mean_permuted, mean[:, permutation], atol=1.0e-6, rtol=0.0)
    assert torch.allclose(value_permuted, value, atol=1.0e-6, rtol=0.0)


def test_history_and_geometry_have_active_nonzero_gradient_but_ghost_gradient_is_zero() -> None:
    r"""Actor真实消费History30与$Z^e$，同时ghost input gradients严格为零。"""

    actor_obs, _, geometry = _inputs(batch=2)
    history = actor_obs.jnt_history.clone().requires_grad_(True)
    tokens = geometry.tokens.clone().requires_grad_(True)
    differentiable_obs = replace(actor_obs, jnt_history=history)
    differentiable_geometry = GeometryTokenBatch(tokens=tokens, owner_valid=geometry.owner_valid)
    actor = StructuredHeterogeneousActor()
    loss = actor(differentiable_obs, differentiable_geometry).mean.sum()
    loss.backward()
    assert history.grad is not None and float(history.grad[actor_obs.jnt_valid[:, None, :, None].expand_as(history)].abs().sum()) > 0.0
    assert tokens.grad is not None and float(tokens.grad[geometry.owner_valid].abs().sum()) > 0.0
    assert torch.equal(history.grad[~actor_obs.jnt_valid[:, None, :, None].expand_as(history)], torch.zeros_like(history.grad[~actor_obs.jnt_valid[:, None, :, None].expand_as(history)]))
    assert torch.equal(tokens.grad[~geometry.owner_valid], torch.zeros_like(tokens.grad[~geometry.owner_valid]))


def test_checkpoint_roundtrip_is_exact() -> None:
    r"""Actor/critic namespacedstate_dict roundtrip保持bitwise outputs。"""

    actor_obs, critic_obs, geometry = _inputs(batch=2)
    original = StructuredActorCriticPackage()
    restored = StructuredActorCriticPackage()
    restored.load_state_dict(original.state_dict(), strict=True)
    assert torch.equal(original.actor(actor_obs, geometry).mean, restored.actor(actor_obs, geometry).mean)
    assert torch.equal(original.critic(critic_obs, geometry).value, restored.critic(critic_obs, geometry).value)
