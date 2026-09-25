'Adapt named palm-rotation observations to the rl_games Actor and privileged Critic.'

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import torch
from torch import nn

from anymani.distill.models.family_rotation_student_actor_critic import (
    FamilyRotationStudentActorCritic,
    FamilyStudentRlConfig,
)
from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationCriticObservation,
    PalmRotationGeometry,
    expanded_policy_log_std,
)
from anymani.distill.rl.algorithms.popart import PopArtValueNormalizer

from ..masked_ppo import AnyManiMaskedContinuousModel
from .palm_rotation_phase import normalize_phase_period_steps
from .palm_rotation_student import FamilyStudentAuxiliaryHooks, FamilyStudentRlStats, FamilyStudentWarmupController
from .palm_rotation_vecenv import (
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
    palm_rotation_float_shapes,
)


def denormalize_value_readonly(model: Any, value: torch.Tensor) -> torch.Tensor:
    'Handle denormalize value readonly; shapes [M,1]; units M.'

    if not bool(getattr(model, "normalize_value", False)):
        return value
    normalizer = getattr(model, "value_mean_std", None)
    if not isinstance(normalizer, nn.Module):
        raise TypeError("normalized value model must expose an nn.Module value_mean_std")
    was_training = bool(normalizer.training)
    normalizer.eval()
    try:
        return model.denorm_value(value)
    finally:
        normalizer.train(was_training)


class PalmRotationRlGamesBuilder:
    'Contract for PALM rotation RL games builder.'

    def __init__(self, **kwargs: Any) -> None:
        'Initialize the instance.'

        _ = kwargs
        self.params: dict[str, Any] = {}  # YAML network mapping

    def load(self, params: dict[str, Any]) -> None:
        'Load the declared contract.'

        self.params = params

    def build(self, name: str, **kwargs: Any) -> PalmRotationRlGamesNetwork:
        'Build the declared contract.'

        _ = name
        return PalmRotationRlGamesNetwork(self.params, **kwargs)


class PalmRotationRlGamesNetwork(nn.Module):
    'Contract for PALM rotation RL games network.'

    def __init__(self, params: Mapping[str, Any], **kwargs: Any) -> None:
        'Initialize the instance.'

        super().__init__()
        actions_num = int(kwargs.pop("actions_num"))
        input_shape = kwargs.pop("input_shape")  # Dict[str, sample shape]
        self.value_size = int(kwargs.pop("value_size", 1))  # hand-level scalar value
        self.num_seqs = int(kwargs.pop("num_seqs", 1))
        if actions_num != 16 or self.value_size != 1:
            raise ValueError("palm-rotation PPO requires 16 canonical actions and scalar value")
        if not isinstance(input_shape, Mapping):
            raise TypeError("palm-rotation PPO requires a Dict observation space")
        network_cfg = params.get("palm_rotation", {})
        if not isinstance(network_cfg, Mapping):
            raise TypeError("palm-rotation network config must be a mapping")
        # Student mode is opt-in: an explicit IL artifact path is the only accepted training source.
        # Legacy callers without these keys keep the old exact input shape and old Actor implementation.
        student_checkpoint_raw = network_cfg.get("student_checkpoint", network_cfg.get("family_student_checkpoint"))
        self.student_mode = bool(network_cfg.get("student_enabled", False) or student_checkpoint_raw is not None)
        self.student_variant = network_cfg.get("student_variant", network_cfg.get("family_student_variant"))
        if self.student_variant is not None and self.student_variant not in {"n040", "no_z", "fk"}:
            raise ValueError("student_variant must be n040, no_z, or fk")
        student_rl_payload = network_cfg.get("student_rl", {})
        if student_rl_payload is None:
            student_rl_payload = {}
        if not isinstance(student_rl_payload, Mapping):
            raise TypeError("student_rl network config must be a mapping")
        self.student_rl_config = self._student_rl_config(student_rl_payload) if self.student_mode else None
        self.student_include_fk_target = bool(network_cfg.get("student_include_fk_target", False))
        if self.student_mode and self.student_variant == "fk" and not self.student_include_fk_target:
            raise ValueError("FK student network requires student_include_fk_target=True")
        if self.student_include_fk_target and not self.student_mode:
            raise ValueError("student_include_fk_target requires student_enabled=True")
        if self.student_include_fk_target and self.student_variant not in {None, "fk"}:
            raise ValueError("student_include_fk_target is only valid for the FK student variant")
        raw_anchor_paths = network_cfg.get("student_anchor_dataset_paths", ())
        if raw_anchor_paths is None:
            raw_anchor_paths = ()
        if isinstance(raw_anchor_paths, (str, bytes)):
            raw_anchor_paths = (raw_anchor_paths,)
        if not isinstance(raw_anchor_paths, (tuple, list)):
            raise TypeError("student_anchor_dataset_paths must be a path sequence")
        self.student_anchor_dataset_paths = tuple(str(path) for path in raw_anchor_paths)
        raw_anchor_hashes = network_cfg.get("student_anchor_source_hashes", {})
        if raw_anchor_hashes is None:
            raw_anchor_hashes = {}
        if not isinstance(raw_anchor_hashes, Mapping):
            raise TypeError("student_anchor_source_hashes must be a path-to-SHA mapping")
        self.student_anchor_source_hashes = {
            str(path): str(digest) for path, digest in raw_anchor_hashes.items()
        }
        if any(len(digest) != 64 for digest in self.student_anchor_source_hashes.values()):
            raise ValueError("student_anchor_source_hashes values must be 64-character SHA-256 strings")
        self.student_anchor_seed = int(network_cfg.get("student_anchor_seed", 42))
        self.student_anchor_batch_size = int(network_cfg.get("student_anchor_batch_size", 2048))
        self.student_anchor_microbatch_size = int(network_cfg.get("student_anchor_microbatch_size", 256))
        self.phase_period_steps = normalize_phase_period_steps(network_cfg.get("phase_period_steps"))
        expected_shapes = {
            **palm_rotation_float_shapes(
                self.phase_period_steps,
                student=self.student_mode,
                include_fk_target=self.student_include_fk_target,
            ),
            **PALM_ROTATION_BOOL_SHAPES,
            **PALM_ROTATION_INT16_SHAPES,
        }
        normalized_shapes = {key: tuple(int(dim) for dim in shape) for key, shape in input_shape.items()}
        if normalized_shapes != expected_shapes:
            missing = sorted(set(expected_shapes) - set(normalized_shapes))
            extra = sorted(set(normalized_shapes) - set(expected_shapes))
            wrong = sorted(
                key
                for key in set(expected_shapes) & set(normalized_shapes)
                if expected_shapes[key] != normalized_shapes[key]
            )
            raise ValueError(f"palm-rotation observation ABI mismatch: missing={missing}, extra={extra}, wrong={wrong}")

        arm_raw = "direct_token" if self.student_mode else str(network_cfg.get("arm", "residual"))
        if arm_raw not in {"base", "residual", "direct", "direct_token"}:
            raise ValueError(f"unsupported palm-rotation actor arm: {arm_raw!r}")
        self.arm = cast(Literal["base", "residual", "direct", "direct_token"], arm_raw)
        initial_log_std = float(network_cfg.get("initial_log_std", -0.5))
        max_log_std = float(network_cfg.get("max_log_std", -0.43))  # N000 early-budget exploration ceiling
        base_action_limit = float(network_cfg.get("base_action_limit", 0.8))
        history_encoder_raw = str(network_cfg.get("history_encoder", "tcn"))
        if history_encoder_raw not in {"tcn", "raw_stack"}:
            raise ValueError(f"unsupported palm-rotation history encoder: {history_encoder_raw!r}")
        if self.student_mode and history_encoder_raw != "tcn":
            raise ValueError("family student Actor ABI fixes history_encoder=tcn")
        if self.student_mode and str(network_cfg.get("sigma_mode", "global")) != "global":
            raise ValueError("family student Actor ABI fixes sigma_mode=global")
        history_encoder = cast(Literal["tcn", "raw_stack"], history_encoder_raw)
        if self.student_mode:
            if student_checkpoint_raw is None:
                raise ValueError("student_enabled requires an explicit student_checkpoint policy.pt artifact")
            if self.phase_period_steps is not None:
                raise ValueError("family student Actor ABI requires phase_clock_enabled=False")
            configured_device = None
            global_config = params.get("config")
            if isinstance(global_config, Mapping):
                configured_device = global_config.get("device")
            self.package = FamilyRotationStudentActorCritic.from_checkpoint(
                student_checkpoint_raw,
                variant=self.student_variant,
                critic_seed=student_rl_payload.get("critic_seed", 0),
                rl_config=self.student_rl_config,
                expected_n040_sha256=network_cfg.get("expected_n040_sha256"),
                expected_dataset_sha256=network_cfg.get("expected_dataset_sha256"),
                expected_source_sha256=network_cfg.get("expected_source_sha256"),
                device=configured_device,
            )
            self.student_variant = self.package.variant
            if self.student_variant == "fk" and not self.student_include_fk_target:
                raise ValueError("FK student checkpoint requires joint_origin_target transport")
            if self.student_include_fk_target and self.student_variant != "fk":
                raise ValueError("joint_origin_target transport requires an FK student checkpoint")
            assert self.student_rl_config is not None  # student branch created it above
            self.student_auxiliary_hooks = FamilyStudentAuxiliaryHooks.from_config(self.student_rl_config)
            self.student_warmup = FamilyStudentWarmupController(
                self.student_rl_config.critic_warmup_rollouts,
                optimizer_steps_per_rollout=self.student_rl_config.optimizer_steps_per_rollout,
            )
            self.student_stats = FamilyStudentRlStats(
                warmup_rollouts=self.student_rl_config.critic_warmup_rollouts,
                optimizer_steps_per_rollout=self.student_rl_config.optimizer_steps_per_rollout,
            )
            self.student_initialization_metadata = dict(self.package.initialization_metadata)
            self.student_initialization_metadata["stats"] = self.student_stats.state_dict()
            if self.student_anchor_batch_size < 1 or self.student_anchor_microbatch_size < 1:
                raise ValueError("student anchor batch and microbatch sizes must be positive")
        else:
            self.package = PalmRotationActorCritic(
                arm=self.arm,
                initial_log_std=initial_log_std,
                max_log_std=max_log_std,
                base_action_limit=base_action_limit,
                history_encoder=history_encoder,
                sigma_mode=network_cfg.get("sigma_mode", "global"),
                recovery_sigma_floor=network_cfg.get("recovery_sigma_floor"),
                phase_clock_enabled=self.phase_period_steps is not None,
            )
        compile_mode_raw = network_cfg.get("compile_mode")
        if compile_mode_raw not in {None, "default", "reduce-overhead"}:
            raise ValueError(f"unsupported palm-rotation compile mode: {compile_mode_raw!r}")
        self.compile_mode = None if compile_mode_raw is None else str(compile_mode_raw)
        self._actor_forward: Callable[..., Any] = self.package.actor.forward
        self._critic_forward: Callable[..., torch.Tensor] = self.package.critic.forward
        if self.compile_mode is not None:
            self._actor_forward = torch.compile(self._actor_forward, mode=self.compile_mode)
            self._critic_forward = torch.compile(self._critic_forward, mode=self.compile_mode)

        identity = params.get("anymani_identity")
        if not isinstance(identity, dict):
            raise ValueError("palm-rotation network requires a JSON-safe AnyMani runtime identity")
        self.anymani_identity = identity  # checkpoint pre-load identity gate
        self.last_active_joint_mask: torch.Tensor | None = None  # shapes [B,16]
        self.last_residual_mean: torch.Tensor | None = None
        self.last_direct_mean: torch.Tensor | None = None
        self.last_film_modulation_rms: torch.Tensor | None = None  # shapes [B,16]
        self.last_fk_prediction: torch.Tensor | None = None
        self.last_fk_target: torch.Tensor | None = None

    @staticmethod
    def _student_rl_config(payload: Mapping[str, Any]) -> FamilyStudentRlConfig:
        'Handle student RL config.'

        aliases = {
            "horizon": "horizon_length",
            "num_minibatches": "minibatches",
            "accumulation_steps": "gradient_accumulation_steps",
            "epochs": "mini_epochs",
            "actor_lr": "actor_learning_rate",
            "critic_lr": "critic_learning_rate",
        }
        derived = {
            "critic_warmup_rollouts",
            "optimizer_steps_per_rollout",
            "warmup_critic_optimizer_steps",
            "actor_rollout_updates",
            "rollout_horizon",
            "num_minibatches",
            "accumulation_steps",
            "epochs",
        }

        values = {
            aliases.get(key, key): value
            for key, value in payload.items()
            if key not in {"critic_seed", *derived}
        }
        allowed = set(FamilyStudentRlConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown student_rl configuration keys {sorted(unknown)}")
        return FamilyStudentRlConfig(**values)

    def is_rnn(self) -> bool:
        'Check rnn.'

        return False

    def get_default_rnn_state(self) -> None:
        'Return default rnn state.'

    def get_aux_loss(self) -> None:
        'Return aux loss.'


        return

    def get_student_auxiliary_hooks(self) -> FamilyStudentAuxiliaryHooks | None:
        'Return student auxiliary hooks.'

        return getattr(self, "student_auxiliary_hooks", None)

    def get_student_warmup_controller(self) -> FamilyStudentWarmupController | None:
        'Return student warmup controller.'

        return getattr(self, "student_warmup", None)

    def get_value_layer(self) -> nn.Module:
        'Return value layer.'

        return self.package.critic.value_head

    def actor_parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        'Handle Actor parameter groups.'

        actor = self.package.actor
        if self.student_mode:

            base = [parameter for parameter in actor.parameters() if parameter.requires_grad]
            if not base:
                raise RuntimeError("student Actor optimizer has no trainable parameters")
            return base, []
        contextual_modules: tuple[nn.Module, ...] = (
            actor.geometry_adapter,
            actor.owner_contact_projection,
            actor.palm_dynamic_projection,
            actor.joint_dynamic_projection,
            actor.tip_dynamic_projection,
            actor.global_backbone,
            actor.direct_head if self.arm in {"direct", "direct_token"} else actor.residual_head,  # type: ignore[union-attr]
        )
        contextual_ids = {id(parameter) for module in contextual_modules for parameter in module.parameters()}
        if actor.conditional_sigma_head is not None:
            contextual_ids.update(id(parameter) for parameter in actor.conditional_sigma_head.parameters())
        phase_adapter = getattr(actor, "phase_contextual_adapter", None)
        if phase_adapter is not None:
            contextual_ids.update(id(parameter) for parameter in phase_adapter.parameters())
        base = [
            parameter for parameter in actor.parameters() if id(parameter) not in contextual_ids
        ]  # temporal/local trunk
        contextual = [
            parameter for parameter in actor.parameters() if id(parameter) in contextual_ids
        ]  # graph action path
        if {id(parameter) for parameter in base} & {id(parameter) for parameter in contextual}:
            raise RuntimeError("actor local/contextual optimizer groups overlap")
        if len(base) + len(contextual) != len(list(actor.parameters())):
            raise RuntimeError("actor optimizer groups do not cover all parameters")
        return base, contextual

    def forward(self, input_dict: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        'Handle forward; shapes [B,16], [B,1].'

        observation = input_dict.get("obs")
        if not isinstance(observation, Mapping):
            raise TypeError("palm-rotation network expects a named observation mapping")


        geometry = PalmRotationGeometry(
            tokens=observation["geometry_tokens"].float(),  # FP32 `[B,21,128]`
            owner_valid=observation["owner_valid"].bool(),
            shortest_path=observation["shortest_path"].long(),  # int16 storage -> exact embedding indices
            parent_direction=observation["parent_direction"].long(),
            child_direction=observation["child_direction"].long(),
        )
        actor_observation = PalmRotationActorObservation(
            jnt_current=observation["actor_jnt_current"].float(),
            jnt_history=observation["actor_jnt_history"].float(),
            jnt_limits=observation["actor_jnt_limits"].float(),
            owner_contact=observation["actor_owner_contact"].float(),
            jnt_valid=observation["jnt_valid"].bool(),
            tip_valid=observation["tip_valid"].bool(),
            owner_valid=observation["owner_valid"].bool(),
        )
        critic_observation = PalmRotationCriticObservation(
            jnt_state=observation["critic_jnt_state"].float(),
            owner_contact=observation["critic_owner_contact"].float(),
            obj=observation["critic_obj"].float(),
            task=observation["critic_task"].float(),
            reward_release=observation["critic_reward_release"].float(),
            jnt_valid=observation["jnt_valid"].bool(),
            tip_valid=observation["tip_valid"].bool(),
            owner_valid=observation["owner_valid"].bool(),
        )

        if self.student_mode:
            # shapes [B,16,15]
            joint_kinematics = observation["joint_kinematics"].float()
            actor_output = self._actor_forward(
                actor_observation,
                geometry,
                joint_kinematics=joint_kinematics,
            )
            value = self._critic_forward(critic_observation, geometry).unsqueeze(-1)
            fk_prediction = getattr(actor_output, "fk_prediction", None)
            self.last_fk_prediction = fk_prediction if isinstance(fk_prediction, torch.Tensor) else None
            self.last_fk_target = (
                observation["joint_origin_target"].float().detach()
                if "joint_origin_target" in observation
                else None
            )
        elif self.phase_period_steps is None:
            actor_output = self._actor_forward(actor_observation, geometry)
            value = self._critic_forward(critic_observation, geometry).unsqueeze(-1)
        else:
            phase = observation["phase_clock"].float()
            actor_output = self._actor_forward(actor_observation, geometry, phase_clock=phase)
            value = self._critic_forward(critic_observation, geometry, phase_clock=phase).unsqueeze(-1)
        self.last_active_joint_mask = actor_observation.jnt_valid  # probability/entropy/KL ghost mask
        residual_mean = getattr(actor_output, "residual_mean", None)
        direct_mean = getattr(actor_output, "direct_mean", None)
        self.last_residual_mean = residual_mean.detach() if isinstance(residual_mean, torch.Tensor) else None
        self.last_direct_mean = direct_mean.detach() if isinstance(direct_mean, torch.Tensor) else None
        self.last_film_modulation_rms = actor_output.film_modulation_rms.detach()
        logstd = expanded_policy_log_std(actor_output.log_std, actor_output.mean, actor_observation.jnt_valid)
        return actor_output.mean, logstd, value, None


class PalmRotationMaskedContinuousModel(AnyManiMaskedContinuousModel):
    'Provide the masked tanh-squashed action distribution and its per-joint likelihood.'

    class Network(AnyManiMaskedContinuousModel.Network):
        'Contract for network.'

        _ACTION_EPS = 1.0e-6

        def __init__(self, a2c_network, **kwargs: Any) -> None:
            'Initialize the instance.'
            super().__init__(a2c_network, **kwargs)
            identity = getattr(a2c_network, "anymani_identity", {})
            training = identity.get("training", {}) if isinstance(identity, Mapping) else {}
            mode = training.get("value_normalization", "rms")
            if mode not in {"rms", "popart"}:
                raise ValueError("unknown value normalization strategy")
            if mode == "popart":
                if not self.normalize_value:
                    raise ValueError("PopArt requires value normalization")
                self.value_mean_std = PopArtValueNormalizer()

        @classmethod
        def _action_to_latent(cls, action: torch.Tensor) -> torch.Tensor:
            'Handle action to latent.'

            bounded = action.clamp(min=-1.0 + cls._ACTION_EPS, max=1.0 - cls._ACTION_EPS)
            return torch.atanh(bounded)

        @classmethod
        def _squashed_per_joint_neglogp(
            cls,
            actions: torch.Tensor,
            action_mean: torch.Tensor,
            sigma: torch.Tensor,
            logstd: torch.Tensor,
        ) -> torch.Tensor:
            'Handle squashed per JOINT neglogp.'

            latent_action = cls._action_to_latent(actions)
            latent_mean = cls._action_to_latent(action_mean)  # Dimensionless latent action coordinate.
            normal_neglogp = (
                0.5 * ((latent_action - latent_mean) / sigma).square() + logstd + 0.5 * math.log(2.0 * math.pi)
            )
            log_jacobian = torch.log((1.0 - actions.square()).clamp_min(cls._ACTION_EPS))
            return normal_neglogp + log_jacobian

        def forward(self, input_dict: dict[str, Any]) -> dict[str, torch.Tensor | None]:
            'Handle forward.'

            is_train = bool(input_dict.get("is_train", True))
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            action_mean, logstd, value, states = self.a2c_network(input_dict)
            active_mask = self.a2c_network.last_active_joint_mask
            if not isinstance(active_mask, torch.Tensor):
                raise RuntimeError("palm-rotation squashed policy did not expose active-joint mask")
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.all(action_mean.abs() <= 1.0 + 1.0e-6),
                "palm-rotation deterministic action mean escaped [-1,1]",
            )
            active_float = active_mask.to(dtype=action_mean.dtype)
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            logstd = expanded_policy_log_std(logstd, action_mean, active_mask)
            sigma = torch.exp(logstd)
            latent_mean = self._action_to_latent(action_mean)
            distribution = torch.distributions.Normal(latent_mean, sigma, validate_args=False)

            if is_train:
                previous_actions = input_dict.get("prev_actions")
                if not isinstance(previous_actions, torch.Tensor):
                    raise RuntimeError("squashed PPO update requires bounded previous actions")
                per_joint_neglogp = self._squashed_per_joint_neglogp(
                    previous_actions,
                    action_mean,
                    sigma,
                    logstd,
                )
                prev_neglogp = (per_joint_neglogp * active_float).sum(dim=-1)
                entropy_latent = distribution.rsample()  # current-policy Monte Carlo differential entropy sample
                entropy_action = torch.tanh(entropy_latent) * active_float
                entropy_per_joint = self._squashed_per_joint_neglogp(
                    entropy_action,
                    action_mean,
                    sigma,
                    logstd,
                )
                entropy = (entropy_per_joint * active_float).sum(dim=-1) / active_count
                result: dict[str, torch.Tensor | None] = {
                    "prev_neglogp": prev_neglogp,
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": action_mean,
                    "sigmas": sigma,
                }
            else:
                latent_action = distribution.sample()
                selected_action = torch.tanh(latent_action) * active_float
                per_joint_neglogp = self._squashed_per_joint_neglogp(
                    selected_action,
                    action_mean,
                    sigma,
                    logstd,
                )
                result = {
                    "neglogpacs": (per_joint_neglogp * active_float).sum(dim=-1),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": action_mean,
                    "sigmas": sigma,
                }
            if self.a2c_network.arm in {"direct", "direct_token"}:
                direct = getattr(self.a2c_network, "last_direct_mean", None)
                if not isinstance(direct, torch.Tensor):
                    raise RuntimeError("palm-rotation direct actor did not expose its bounded mean")
                result["direct_means"] = direct
            else:
                residual = getattr(self.a2c_network, "last_residual_mean", None)
                if not isinstance(residual, torch.Tensor):
                    raise RuntimeError("palm-rotation residual/base actor did not expose bounded residual")
                result["residuals"] = residual

            film = getattr(self.a2c_network, "last_film_modulation_rms", None)
            if not isinstance(film, torch.Tensor):
                raise RuntimeError("palm-rotation actor did not expose geometry FiLM diagnostics")
            result["film_modulations"] = film  # shapes [B,16]
            if bool(getattr(self.a2c_network, "student_mode", False)):

                result["student_fk_prediction"] = getattr(self.a2c_network, "last_fk_prediction", None)
                result["student_fk_target"] = getattr(self.a2c_network, "last_fk_target", None)
            return result
