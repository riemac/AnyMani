"""Effective recovery sigma distribution contracts for structured palm-rotation PPO."""

from __future__ import annotations

import pytest
import torch
from anymani.distill.rl.algorithms.policy_statistics import mean_preserving_squashed_kl
from anymani.distill.rl.algorithms.task_gradients import per_asset_ppo_gradients
from anymani.distill.rl.palm_rotation_ppo import (
    PalmRotationMaskedContinuousModel,
    PalmRotationPpoAgent,
    PalmRotationRlGamesBuilder,
)
from anymani.distill.rl.runtime.palm_rotation_network import PalmRotationMaskedContinuousModel as _ProductionModel
from anymani.distill.rl.runtime.palm_rotation_vecenv import (
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_FLOAT_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
)
from anymani.distill.tests.contracts.models.test_palm_rotation_recovery_exploration import (
    _gate_observation,
)


def _input_shapes() -> dict[str, tuple[int, ...]]:
    """Return the unchanged sample-level structured Dict ABI."""

    return {**PALM_ROTATION_FLOAT_SHAPES, **PALM_ROTATION_BOOL_SHAPES, **PALM_ROTATION_INT16_SHAPES}


def _network_observation() -> dict[str, torch.Tensor]:
    """Lower the controlled actor packet into one complete PPO observation mapping."""

    actor = _gate_observation()
    batch = actor.jnt_current.shape[0]
    graph = torch.zeros(batch, 21, 21, dtype=torch.int16)
    return {
        "actor_jnt_current": actor.jnt_current.clone(),
        "actor_jnt_history": actor.jnt_history.clone(),
        "actor_jnt_limits": actor.jnt_limits.clone(),
        "actor_owner_contact": actor.owner_contact.clone(),
        "critic_jnt_state": torch.zeros(batch, 16, 4),
        "critic_owner_contact": torch.zeros(batch, 21, 2),
        "critic_obj": torch.zeros(batch, 1, 15),
        "critic_task": torch.zeros(batch, 1, 8),
        "critic_reward_release": torch.zeros(batch, 1),
        "jnt_valid": actor.jnt_valid.clone(),
        "tip_valid": actor.tip_valid.clone(),
        "owner_valid": actor.owner_valid.clone(),
        "geometry_tokens": torch.zeros(batch, 21, 128),
        "shortest_path": graph.clone(),
        "parent_direction": graph.clone(),
        "child_direction": graph.clone(),
        "prototype_index": torch.zeros(batch, 1, dtype=torch.int16),
    }


def _model(recovery_sigma_floor: float | None) -> PalmRotationMaskedContinuousModel.Network:
    """Build the production masked model with the explicit network recovery-floor key."""

    builder = PalmRotationRlGamesBuilder()
    builder.load(
        {
            "palm_rotation": {
                "arm": "direct_token",
                "history_encoder": "tcn",
                "sigma_mode": "global",
                "initial_log_std": -0.5,
                "max_log_std": -0.43,
                "base_action_limit": 0.8,
                "recovery_sigma_floor": recovery_sigma_floor,
            },
            "anymani_identity": {"identity_digest": "recovery-distribution-contract"},
        }
    )
    return PalmRotationMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": _input_shapes(),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    )


def test_network_config_passes_floor_without_changing_learned_parameter_keys() -> None:
    """The network key reaches Actor and creates no recovery-specific learned parameter."""

    model = _model(0.64)
    actor = model.a2c_network.package.actor
    assert actor.recovery_sigma_floor == pytest.approx(0.64)
    assert not any("recovery" in name for name, _ in actor.named_parameters())
    assert not any("recovery" in name for name in actor.state_dict())


def test_rollout_and_update_reuse_effective_sigma_and_recompute_old_action_logprob() -> None:
    """Stored rollout actions must receive the identical effective sigma on PPO re-forward."""

    model = _model(0.64).eval()
    observation = _network_observation()
    rollout = model({"obs": observation, "prev_actions": None, "is_train": False})
    actions = rollout["actions"].detach()
    update = model({"obs": observation, "prev_actions": actions, "is_train": True})

    actor = model.a2c_network.package.actor
    actor_observation = _gate_observation()
    gate = actor.recovery_exploration_mask(actor_observation)
    base_sigma = actor.global_log_std.detach().exp()
    expected_sigma = torch.full((1, 16), 0.64)
    expected_sigma[~gate & actor_observation.jnt_valid] = base_sigma
    expected_sigma[~actor_observation.jnt_valid] = 1.0  # expanded ghost Normal is neutral and masked later.
    torch.testing.assert_close(rollout["sigmas"], expected_sigma, rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(update["sigmas"], rollout["sigmas"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(update["mus"], rollout["mus"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(update["prev_neglogp"], rollout["neglogpacs"], rtol=0.0, atol=1.0e-6)


def test_entropy_uses_the_same_effective_sigma_as_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """With fixed zero noise, production entropy equals the squashed density using returned sigma."""

    model = _model(0.64).eval()
    observation = _network_observation()

    def zero_rsample(normal: torch.distributions.Normal, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Make the production Monte Carlo entropy sample deterministic for the equality check."""

        del sample_shape
        return normal.loc

    monkeypatch.setattr(torch.distributions.Normal, "rsample", zero_rsample)
    result = model({"obs": observation, "prev_actions": torch.zeros(1, 16), "is_train": True})
    mean = result["mus"]
    sigma = result["sigmas"]
    active = observation["jnt_valid"]
    logstd = torch.log(sigma)
    latent = _ProductionModel.Network._action_to_latent(mean)
    entropy_action = torch.tanh(latent) * active.to(dtype=mean.dtype)
    per_joint = _ProductionModel.Network._squashed_per_joint_neglogp(entropy_action, mean, sigma, logstd)
    expected = (per_joint * active.to(dtype=mean.dtype)).sum(dim=-1) / active.sum(dim=-1).clamp_min(1.0)
    torch.testing.assert_close(result["entropy"], expected, rtol=0.0, atol=1.0e-6)


def test_effective_sigma_kl_is_self_zero_and_ghost_neutral() -> None:
    """State-dependent sigma remains compatible with active-DoF KL and ghost masking."""

    model = _model(0.64).eval()
    observation = _network_observation()
    result = model({"obs": observation, "prev_actions": None, "is_train": False})
    mean = result["mus"]
    sigma = result["sigmas"]
    active = observation["jnt_valid"]
    baseline = PalmRotationPpoAgent.masked_policy_kl(mean, sigma, mean, sigma, active)
    direct = mean_preserving_squashed_kl(mean, sigma, mean, sigma, active)
    torch.testing.assert_close(baseline, torch.zeros_like(baseline), rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(direct, torch.zeros_like(direct), rtol=0.0, atol=1.0e-12)

    poisoned_mean = mean.clone()
    poisoned_sigma = sigma.clone()
    poisoned_mean[~active] = float("nan")
    poisoned_sigma[~active] = 0.0
    changed = PalmRotationPpoAgent.masked_policy_kl(poisoned_mean, poisoned_sigma, mean, sigma, active)
    torch.testing.assert_close(changed, baseline, rtol=0.0, atol=0.0)


def test_functional_task_gradient_path_keeps_recovery_sigma_rule() -> None:
    """The torch.func functional Actor path must not bypass the raw-observation floor gate."""

    model = _model(0.64).eval()
    observation = _network_observation()
    rollout = model({"obs": observation, "prev_actions": None, "is_train": False})
    actions = rollout["actions"].detach()
    data = {
        "actions": actions,
        "old_logp_actions": rollout["neglogpacs"].detach(),
        "advantages": torch.ones(1),
        "old_values": rollout["values"].detach(),
        "returns": rollout["values"].detach(),
    }
    _actor_gradients, _critic_gradients, auxiliary = per_asset_ppo_gradients(
        model.a2c_network.package,
        observation,
        data,
        asset_count=1,
        entropy_noise=torch.zeros_like(actions),
        chunk_size=1,
    )
    torch.testing.assert_close(auxiliary["sigmas"], rollout["sigmas"], rtol=0.0, atol=1.0e-6)

