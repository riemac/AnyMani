'Compute full Actor and Critic task gradients per asset without writing parameter gradients.'

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import torch
from torch.func import functional_call, grad_and_value, vmap

from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationCriticObservation,
    PalmRotationDirectActor,
    PalmRotationGeometry,
    expanded_policy_log_std,
)
from anymani.distill.rl.algorithms.action_regularization import tip_silent_rejected_action_cost
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games
from anymani.distill.rl.structured_masked_distribution import masked_bound_loss

prefer_local_rl_games()
from rl_games.common import common_losses  # noqa: E402

from anymani.distill.rl.runtime.palm_rotation_network import PalmRotationMaskedContinuousModel  # noqa: E402


def per_asset_ppo_gradients(
    package: PalmRotationActorCritic,
    observation: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    asset_count: int,
    entropy_noise: torch.Tensor,
    chunk_size: int = 8,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.002,
    bounds_coef: float = 1.0e-4,
    rejected_action_weight: float = 0.0,
    critic_coef: float = 4.0,
    clip_value: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    'Handle per asset PPO gradients; shapes [A,*parameter_shape], [B,16].'
    if not isinstance(package.actor, PalmRotationDirectActor):
        raise ValueError("per-asset functional PPO currently requires a direct actor")
    if asset_count < 1 or chunk_size < 1:
        raise ValueError("asset_count and chunk_size must be positive")
    if not math.isfinite(rejected_action_weight) or rejected_action_weight < 0:
        raise ValueError("rejected action weight must be finite and nonnegative")
    labels = observation["prototype_index"].reshape(-1).long()
    count = labels.numel()
    if count == 0 or count % asset_count:
        raise ValueError("task batch must contain equal nonzero samples for all assets")
    samples = count // asset_count
    expected = torch.arange(asset_count, device=labels.device).repeat_interleave(samples)
    order = labels.argsort(stable=True)
    if not torch.equal(labels[order], expected):
        raise ValueError("task batch is missing assets or is not asset-balanced")


    masks = {name: observation[name] for name in ("jnt_valid", "tip_valid", "owner_valid")}
    actor = PalmRotationActorObservation(
        observation["actor_jnt_current"],
        observation["actor_jnt_history"],
        observation["actor_jnt_limits"],
        observation["actor_owner_contact"],
        **masks,
    )
    masks = {name: getattr(actor, name) for name in masks}
    critic = PalmRotationCriticObservation(
        observation["critic_jnt_state"],
        observation["critic_owner_contact"],
        observation["critic_obj"],
        observation["critic_task"],
        observation["critic_reward_release"],
        **masks,
    )
    geometry = PalmRotationGeometry(
        observation["geometry_tokens"],
        actor.owner_valid,
        observation["shortest_path"].long(),
        observation["parent_direction"].long(),
        observation["child_direction"].long(),
    )
    if entropy_noise.shape != batch["actions"].shape or entropy_noise.shape != (count, 16):
        raise ValueError("entropy noise and actions must share [B,16]")
    for name in ("actions", "old_logp_actions", "advantages", "old_values", "returns"):
        if batch[name].shape[0] != count:
            raise ValueError(f"PPO field {name} does not share the sample axis")

    grouped: dict[str, torch.Tensor] = {}
    for prefix, obj in (("a", actor), ("c", critic), ("g", geometry)):
        for field in fields(obj):
            value = getattr(obj, field.name)
            grouped[f"{prefix}/{field.name}"] = value[order].reshape(asset_count, samples, *value.shape[1:])
    for name in ("actions", "old_logp_actions", "advantages", "old_values", "returns"):
        value = batch[name]
        grouped[name] = value[order].reshape(asset_count, samples, *value.shape[1:])
    grouped["entropy_noise"] = entropy_noise[order].reshape(asset_count, samples, 16)
    phase_enabled = bool(getattr(package.actor, "phase_clock_enabled", False))
    if phase_enabled != bool(getattr(package.critic, "phase_clock_enabled", False)):
        raise ValueError("Actor and Critic phase-clock configuration must agree")
    if phase_enabled:
        phase = observation.get("phase_clock")
        if not isinstance(phase, torch.Tensor) or phase.shape != (count, 2):
            raise ValueError("phase-clock task gradients require stored [B,2] phase samples")
        if phase.dtype != actor.jnt_current.dtype or phase.device != actor.jnt_current.device:
            raise ValueError("phase-clock samples must share Actor dtype and device")
        torch._assert_async(torch.isfinite(phase).all(), "phase-clock samples must be finite")  # pyright: ignore[reportPrivateImportUsage]
        grouped["phase_clock"] = phase[order].reshape(asset_count, samples, 2)

    def view(cls, prefix: str, data: Mapping[str, torch.Tensor]):
        'Handle view.'
        result = object.__new__(cls)
        for field in fields(cls):
            object.__setattr__(result, field.name, data[f"{prefix}/{field.name}"])
        return result

    def objective(actor_parameters, critic_parameters, data):
        'Handle objective.'
        actor_obs = view(PalmRotationActorObservation, "a", data)
        critic_obs = view(PalmRotationCriticObservation, "c", data)
        geo = view(PalmRotationGeometry, "g", data)
        forward_kwargs: dict[str, Any] = {"_validated": True}
        if phase_enabled:
            forward_kwargs["phase_clock"] = data["phase_clock"]
        output = functional_call(package.actor, actor_parameters, (actor_obs, geo), forward_kwargs)
        values = functional_call(package.critic, critic_parameters, (critic_obs, geo), forward_kwargs)[:, None]
        mu = output.mean
        logstd = expanded_policy_log_std(output.log_std, mu, actor_obs.jnt_valid)
        sigma = logstd.exp()
        mask = actor_obs.jnt_valid.to(mu.dtype)
        active_count = mask.sum(-1).clamp_min(1)
        distribution = PalmRotationMaskedContinuousModel.Network
        neglogp = (distribution._squashed_per_joint_neglogp(data["actions"], mu, sigma, logstd) * mask).sum(-1)
        latent = distribution._action_to_latent(mu) + sigma * data["entropy_noise"]
        entropy_action = latent.tanh() * mask
        entropy = (distribution._squashed_per_joint_neglogp(entropy_action, mu, sigma, logstd) * mask).sum(
            -1
        ) / active_count
        actor_loss = common_losses.actor_loss(data["old_logp_actions"], neglogp, data["advantages"], True, clip_epsilon)
        bounds = masked_bound_loss(mu, actor_obs.jnt_valid)
        actor_objective = actor_loss - entropy_coef * entropy + bounds_coef * bounds
        rejected_cost = torch.zeros_like(actor_loss)
        rejected_fraction = torch.zeros_like(actor_loss)
        if rejected_action_weight > 0.0:
            rejected_cost, rejected_fraction = tip_silent_rejected_action_cost(
                mean=mu,
                target_normalized=actor_obs.jnt_current[..., 1],
                limits_normalized=actor_obs.jnt_limits,
                joint_valid=actor_obs.jnt_valid,
                tip_contact=actor_obs.owner_contact[:, 17:21, 0],
                tip_valid=actor_obs.tip_valid,
            )
            actor_objective = actor_objective + rejected_action_weight * rejected_cost
        critic_vector = common_losses.critic_loss(
            None, data["old_values"], values, clip_epsilon, data["returns"], clip_value
        )
        critic_objective = 0.5 * critic_coef * critic_vector.reshape(-1)
        aux = {
            "mus": mu,
            "sigmas": sigma,
            "values": values,
            "prev_neglogp": neglogp,
            "entropy": entropy,
            "actor_loss_vector": actor_loss,
            "bounds_loss_vector": bounds,
            "critic_loss_vector": critic_vector,
            "actor_objective_vector": actor_objective,
            "actor_rejected_action_cost": rejected_cost,
            "actor_rejected_action_fraction": rejected_fraction,
            "critic_objective_vector": critic_objective,
        }
        return actor_objective.mean() + critic_objective.mean(), aux


    gradient_function = grad_and_value(objective, argnums=(0, 1), has_aux=True)
    (actor_gradients, critic_gradients), (_, auxiliary) = vmap(
        gradient_function,
        in_dims=(None, None, 0),
        randomness="error",
        chunk_size=chunk_size,
    )(
        {name: value.detach() for name, value in package.actor.named_parameters()},
        {name: value.detach() for name, value in package.critic.named_parameters()},
        grouped,
    )
    inverse = order.argsort()
    flattened = {name: value.flatten(0, 1)[inverse].detach() for name, value in auxiliary.items()}
    return actor_gradients, critic_gradients, flattened
