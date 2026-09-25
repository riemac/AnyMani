'Adapt rl_games continuous actions to the active-joint ABI; ghost joints contribute zero actions and no probability loss.'

from __future__ import annotations

from typing import Any

import torch
from rl_games.algos_torch import a2c_continuous, model_builder, models, players, torch_ext
from rl_games.torch_runner import Runner


ANYMANI_CHECKPOINT_IDENTITY_KEY = "anymani_identity"

def validate_anymani_checkpoint_identity(
    *,
    runtime_identity: dict[str, Any],
    checkpoint_identity: object,
) -> None:
    'Reject checkpoint restore when data, asset ordering, or frozen geometry identity differs.'

    if not isinstance(checkpoint_identity, dict):
        raise RuntimeError("heterogeneous AnyMani checkpoint is missing required anymani_identity metadata")
    compared_fields = tuple(sorted(set(runtime_identity) | set(checkpoint_identity)))
    mismatched = [field for field in compared_fields if runtime_identity.get(field) != checkpoint_identity.get(field)]
    if mismatched:
        runtime_digest = runtime_identity.get("identity_digest", "missing")
        checkpoint_digest = checkpoint_identity.get("identity_digest", "missing")
        raise RuntimeError(
            "AnyMani checkpoint identity mismatch before model restore: "
            f"fields={mismatched}, runtime_digest={runtime_digest}, checkpoint_digest={checkpoint_digest}"
        )






class AnyManiMaskedContinuousModel(models.BaseModel):
    'Wrap the rl_games continuous policy with active-joint probability and action masks.'

    def __init__(self, network) -> None:
        super().__init__("a2c")
        self.network_builder = network

    class Network(models.BaseModelNetwork):
        'Adapt the policy network to the masked continuous-action output contract.'

        def __init__(self, a2c_network, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.a2c_network = a2c_network

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def forward(self, input_dict: dict[str, torch.Tensor]):
            is_train = input_dict.get("is_train", True)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            mu, logstd, value, states = self.a2c_network(input_dict)
            active_mask = self.a2c_network.last_active_joint_mask
            if active_mask is None:
                raise RuntimeError("canonical masked network did not expose active joint mask")
            sigma = torch.exp(logstd)
            distribution = torch.distributions.Normal(mu, sigma, validate_args=False)
            active_float = active_mask.to(dtype=mu.dtype)
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            if is_train:
                previous_actions = input_dict["prev_actions"]
                prev_neglogp = self._masked_neglogp(previous_actions, mu, sigma, logstd, active_float)
                entropy = (distribution.entropy() * active_float).sum(dim=-1) / active_count
                return {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
            selected_action = distribution.sample() * active_float
            neglogp = self._masked_neglogp(selected_action, mu, sigma, logstd, active_float)
            return {
                "neglogpacs": torch.squeeze(neglogp),
                "values": self.denorm_value(value),
                "actions": selected_action,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }

        @staticmethod
        def _masked_neglogp(
            actions: torch.Tensor,
            mu: torch.Tensor,
            sigma: torch.Tensor,
            logstd: torch.Tensor,
            active_float: torch.Tensor,
        ) -> torch.Tensor:
            'Sum Normal negative log probability over active joints.'

            per_joint = (
                0.5 * ((actions - mu) / sigma).square()
                + 0.5 * torch.log(torch.as_tensor(2.0 * torch.pi, device=actions.device, dtype=actions.dtype))
                + logstd
            )
            return (per_joint * active_float).sum(dim=-1)


class AnyManiMaskedPpoAgent(a2c_continuous.A2CAgent):
    'Apply active-joint KL and regularization reductions in the PPO agent.'

    def _runtime_identity(self) -> dict[str, Any] | None:
        'Read the Actor network identity used for checkpoint validation.'

        network = getattr(self.model, "a2c_network", None)
        identity = getattr(network, "anymani_identity", None)
        return identity if isinstance(identity, dict) else None

    def get_full_state_weights(self) -> dict[str, Any]:
        'Return full state weights.'

        state = super().get_full_state_weights()
        identity = self._runtime_identity()
        if identity is not None:
            state[ANYMANI_CHECKPOINT_IDENTITY_KEY] = identity
        return state

    def set_full_state_weights(self, weights: dict[str, Any], set_epoch: bool = True) -> None:
        'Validate the run identity before restoring model and optimizer state.'

        runtime_identity = self._runtime_identity()
        if runtime_identity is not None:
            validate_anymani_checkpoint_identity(
                runtime_identity=runtime_identity,
                checkpoint_identity=weights.get(ANYMANI_CHECKPOINT_IDENTITY_KEY),
            )
        super().set_full_state_weights(weights, set_epoch=set_epoch)

    @staticmethod
    def masked_policy_kl(
        current_mu: torch.Tensor,
        current_sigma: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        'Compute per-sample Gaussian KL averaged over active joints.'

        c1 = torch.log(old_sigma / current_sigma + 1.0e-5)
        c2 = (current_sigma.square() + (old_mu - current_mu).square()) / (2.0 * (old_sigma.square() + 1.0e-5))
        per_joint = c1 + c2 - 0.5
        weights = active_mask.to(dtype=per_joint.dtype)
        return (per_joint * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def calc_gradients(self, input_dict) -> None:
        'Run the upstream PPO update and replace KL reduction with the active-joint mean.'

        super().calc_gradients(input_dict)
        a_loss, c_loss, entropy, _kl, last_lr, lr_mul, mu, sigma, b_loss = self.train_result
        active_mask = getattr(self.model.a2c_network, "last_active_joint_mask", None)
        if not isinstance(active_mask, torch.Tensor) or active_mask.shape != mu.shape:
            raise RuntimeError("canonical PPO update did not expose the active joint mask")
        with torch.no_grad():
            kl = self.masked_policy_kl(
                mu,
                sigma,
                input_dict["mu"],
                input_dict["sigma"],
                active_mask,
            ).mean()
        self.train_result = (a_loss, c_loss, entropy, kl, last_lr, lr_mul, mu, sigma, b_loss)

    def bound_loss(self, mu: torch.Tensor) -> torch.Tensor:
        'Average action-bound penalties over active joints.'

        if self.bounds_loss_coef is None:
            return torch.zeros(mu.shape[0], device=mu.device)
        mask = getattr(self.model.a2c_network, "last_active_joint_mask", None)
        if not isinstance(mask, torch.Tensor) or mask.shape != mu.shape:
            inherited = super().bound_loss(mu)
            return inherited if isinstance(inherited, torch.Tensor) else torch.zeros(mu.shape[0], device=mu.device)
        weights = mask.to(dtype=mu.dtype)
        soft_bound = 1.1
        high = torch.clamp_min(mu - soft_bound, 0.0).square()
        low = torch.clamp_max(mu + soft_bound, 0.0).square()
        return ((high + low) * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def reg_loss(self, mu: torch.Tensor) -> torch.Tensor:
        'Average action regularization over active joints.'

        if self.bounds_loss_coef is None:
            return torch.zeros(mu.shape[0], device=mu.device)
        mask = getattr(self.model.a2c_network, "last_active_joint_mask", None)
        if not isinstance(mask, torch.Tensor) or mask.shape != mu.shape:
            inherited = super().reg_loss(mu)
            return inherited if isinstance(inherited, torch.Tensor) else torch.zeros(mu.shape[0], device=mu.device)
        weights = mask.to(dtype=mu.dtype)
        return (mu.square() * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


class AnyManiMaskedPpoPlayer(players.PpoPlayerContinuous):
    'Validate run identity before restoring a masked PPO player.'

    def restore(self, fn: str) -> None:
        'Restore the declared contract.'

        checkpoint = torch_ext.load_checkpoint(fn)
        network = getattr(self.model, "a2c_network", None)
        runtime_identity = getattr(network, "anymani_identity", None)
        if isinstance(runtime_identity, dict):
            validate_anymani_checkpoint_identity(
                runtime_identity=runtime_identity,
                checkpoint_identity=checkpoint.get(ANYMANI_CHECKPOINT_IDENTITY_KEY),
            )
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        env_state = checkpoint.get("env_state")
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)


class AnyManiMaskedRunner(Runner):
    'Register the AnyMani masked PPO agent and player with rl_games.'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.algo_factory.register_builder(
            "anymani_masked_ppo",
            lambda **factory_kwargs: AnyManiMaskedPpoAgent(**factory_kwargs),
        )
        self.player_factory.register_builder(
            "anymani_masked_ppo",
            lambda **factory_kwargs: AnyManiMaskedPpoPlayer(**factory_kwargs),
        )


def register_anymani_masked_ppo() -> None:
    'Register the active-joint continuous model wrapper with rl_games.'

    model_builder.register_model("anymani_masked_continuous", AnyManiMaskedContinuousModel)


__all__ = [
    "ANYMANI_CHECKPOINT_IDENTITY_KEY",
    "ANYMANI_MASKED_PPO_ALGO_KEY",
    "AnyManiMaskedPpoAgent",
    "AnyManiMaskedPpoPlayer",
    "AnyManiMaskedRunner",
    "register_anymani_masked_ppo",
    "validate_anymani_checkpoint_identity",
]


ANYMANI_MASKED_PPO_ALGO_KEY = "anymani_masked_ppo"
