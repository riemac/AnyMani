'Train the heterogeneous palm rotation policy with masked continuous actions and a frozen N040 encoder.'

from __future__ import annotations

import gc
import hashlib
import math
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.common import common_losses
from torch import nn
from torch.nn.utils import clip_grad_norm_

from anymani.distill.diagnostics.recording.rl.optimization_evidence import write_optimization_evidence
from anymani.distill.diagnostics.recording.rl.palm_rotation import (
    PalmRotationMetricsRecorder,
    write_selected_trajectories_hdf5,
)

from .algorithms.action_regularization import tip_silent_rejected_action_cost
from .algorithms.cagrad import combine_task_gradients
from .algorithms.policy_statistics import mean_preserving_squashed_kl
from .algorithms.popart import PopArtValueNormalizer
from .algorithms.ppo_batch import (
    bounded_adaptive_learning_rate,
    normalize_advantages_per_asset,
    stratified_asset_permutation,
)
from .algorithms.task_gradients import per_asset_ppo_gradients
from .masked_ppo import (
    AnyManiMaskedPpoAgent,
    AnyManiMaskedPpoPlayer,
    AnyManiMaskedRunner,
    register_anymani_masked_ppo,
)
from .runtime import palm_rotation_diagnostics as diagnostics
from .runtime import palm_rotation_probes as probes
from .runtime.palm_rotation_diagnostics import (
    PalmRotationPpoDiagnostics,
    rollout_policy_mechanism_metrics,
)
from .runtime.palm_rotation_experience import EnvMajorExperienceBuffer
from .runtime.palm_rotation_network import (
    PalmRotationMaskedContinuousModel,
    PalmRotationRlGamesBuilder,
    PalmRotationRlGamesNetwork,
    denormalize_value_readonly,
)
from .runtime.palm_rotation_optimizer_init import (
    load_named_optimizer_state,
    load_optimizer_parameter_names,
    optimizer_parameter_names,
)
from .runtime.palm_rotation_phase import audit_phase_rollout
from .runtime.palm_rotation_student import (
    FamilyStudentAnchorSampler,
    FamilyStudentRlStats,
    resolve_family_student_anchor_paths,
    resolve_family_student_anchor_source_hashes,
)
from .runtime.palm_rotation_warm_start import (
    load_actor_init_checkpoint,
    load_critic_init_checkpoint,
    should_load_actor_init_checkpoint,
)

PALM_ROTATION_PPO_ALGO = "anymani_palm_rotation_ppo"
PALM_ROTATION_NETWORK = "anymani_palm_rotation"
CRITIC_OPTIMIZER_KEY = "anymani_critic_optimizer"
DIAGNOSTICS_RECORDER_KEY = "anymani_metrics_recorder"
TRAINING_CONTINUATION_KEY = "anymani_training_continuation"
OPTIMIZER_PARAMETER_NAMES_KEY = "anymani_optimizer_parameter_names"


def validate_gradient_probe_compile_compatibility(
    compile_mode: str | None,
    probe_frequency: int,
    full_gradient_shadow_frequency: int = 0,
) -> None:
    'Validate gradient probe compile compatibility.'

    if probe_frequency < 0 or full_gradient_shadow_frequency < 0:
        raise ValueError("gradient probe frequencies must be non-negative")
    if compile_mode is not None and (probe_frequency > 0 or full_gradient_shadow_frequency > 0):
        raise ValueError(
            "head-gradient probe requires eager actor/critic forward; disable --torch_compile or the probe"
        )


class PalmRotationPpoAgent(AnyManiMaskedPpoAgent):
    'Run separate Actor and privileged Critic optimizers with active-joint PPO statistics.'

    def __init__(self, base_name: str, params: dict[str, Any]) -> None:
        'Initialize the instance.'

        super().__init__(base_name, params)
        if self.has_central_value:
            raise ValueError(
                "palm-rotation custom package already owns the privileged critic; duplicate CV is forbidden"
            )
        if self.mixed_precision:
            raise ValueError("actor, critic, PPO losses and optimizers must remain FP32")
        if self.multi_gpu:
            raise ValueError("MVP80 dual-optimizer agent currently supports one GPU only")
        self.diagnostics = PalmRotationPpoDiagnostics()
        network = self.model.a2c_network
        if not isinstance(network, PalmRotationRlGamesNetwork):
            raise TypeError("palm-rotation PPO agent received an incompatible network")
        self.student_mode = bool(network.student_mode)
        self.student_warmup = getattr(network, "student_warmup", None)
        self.student_stats = getattr(network, "student_stats", None)
        self.student_anchor_sampler: FamilyStudentAnchorSampler | None = None
        self._student_rollout_phase: str | None = None
        self._student_actor_requires_grad: dict[int, bool] = {}
        self._student_anchor_microbatch_size = int(getattr(network, "student_anchor_microbatch_size", 256))
        self._student_pending_anchor_loss = torch.zeros((), device=self.ppo_device)
        self._student_pending_fk_loss = torch.zeros((), device=self.ppo_device)
        self._student_anchor_dataset_paths: tuple[Path, ...] = ()
        self._student_anchor_source_hashes: dict[str, str] = {}
        self.student_anchor_source_data_hash: str | None = None
        if self.student_mode:
            if self.student_warmup is None or self.student_stats is None:
                raise RuntimeError("student network must expose warmup and stats hooks")
            if not hasattr(network.package, "anchor_actor"):
                raise RuntimeError("student package must retain the initial frozen BC Actor for anchor loss")
            student_config = network.student_rl_config
            if student_config is None:
                raise RuntimeError("student network must expose the fixed PPO candidate config")
            if str(self.config.get("lr_schedule", "identity")) != "identity":
                raise ValueError("student PPO requires the constant identity learning-rate scheduler")

            self.config.update(
                {
                    "learning_rate": student_config.actor_learning_rate,
                    "adaptive_lr_max": student_config.actor_learning_rate,
                    "residual_learning_rate": student_config.actor_learning_rate,
                    "contextual_learning_rate": student_config.actor_learning_rate,
                    "critic_learning_rate": student_config.critic_learning_rate,
                    "gamma": student_config.gamma,
                    "tau": student_config.gae_lambda,
                    "e_clip": student_config.clip_epsilon,
                    "grad_norm": student_config.grad_norm,
                    "entropy_coef": student_config.entropy_coef,
                }
            )
            self.entropy_coef = student_config.entropy_coef
            self.e_clip = student_config.clip_epsilon
            self.grad_norm = student_config.grad_norm
            self.gamma = student_config.gamma
            self.tau = student_config.gae_lambda
            self.last_lr = student_config.actor_learning_rate
            explicit_anchor_paths_value: Any = getattr(network, "student_anchor_dataset_paths", ())
            explicit_anchor_paths = tuple(str(path) for path in explicit_anchor_paths_value)
            if explicit_anchor_paths:
                self._student_anchor_dataset_paths = tuple(Path(path).expanduser().resolve() for path in explicit_anchor_paths)
                configured_anchor_hashes = getattr(network, "student_anchor_source_hashes", {})
                if configured_anchor_hashes:
                    self._student_anchor_source_hashes = {
                        str(Path(path).expanduser().resolve()): str(digest)
                        for path, digest in configured_anchor_hashes.items()
                    }
            else:
                self._student_anchor_dataset_paths = resolve_family_student_anchor_paths()
                self._student_anchor_source_hashes = resolve_family_student_anchor_source_hashes()

        self.value_normalization_mode = str(self.config.get("value_normalization", "rms"))
        if self.value_normalization_mode not in {"rms", "popart"}:
            raise ValueError("value_normalization must be rms or popart")
        if self.value_normalization_mode == "popart":
            if not self.normalize_value:
                raise ValueError("PopArt requires normalize_value=True")
            if not isinstance(self.model.value_mean_std, PopArtValueNormalizer):
                raise TypeError("model identity and agent PopArt configuration disagree")
            if self.value_mean_std is not self.model.value_mean_std:
                raise RuntimeError("agent/model value normalizer aliases disagree")
        self._last_popart_compensation: dict[str, torch.Tensor] | None = None
        self.gradient_aggregation = str(self.config.get("gradient_aggregation", "mean"))
        self.rejected_action_weight = float(self.config.get("rejected_action_weight", 0.0))
        if not math.isfinite(self.rejected_action_weight) or self.rejected_action_weight < 0:
            raise ValueError("rejected action weight must be finite and nonnegative")
        self.cagrad_c = float(self.config.get("cagrad_c", 0.4))
        self.cagrad_task_chunk = int(self.config.get("cagrad_task_chunk", 128))
        if self.gradient_aggregation not in {"mean", "cagrad"} or not 0 <= self.cagrad_c < 1:
            raise ValueError("invalid gradient aggregation or CAGrad protection radius")
        if self.student_mode and self.gradient_aggregation != "mean":
            raise ValueError("student PPO requires mean gradient aggregation until kinematics-aware CAGrad is implemented")
        if self.cagrad_task_chunk < 1:
            raise ValueError("cagrad_task_chunk must be positive")
        if self.gradient_aggregation == "cagrad" and (
            int(self.config.get("gradient_probe_frequency", 0))
            or int(self.config.get("full_gradient_shadow_frequency", 0))
        ):
            raise ValueError("CAGrad already computes full task gradients; disable legacy gradient probes")
        self._cagrad_actor_accumulator: dict[str, torch.Tensor] | None = None
        self._cagrad_critic_accumulator: dict[str, torch.Tensor] | None = None
        self._last_cagrad_diagnostics: dict[str, torch.Tensor] = {}


        actor_init_path = str(self.config.get("actor_init_checkpoint", "")).strip()
        runtime_identity = network.anymani_identity
        training_identity = runtime_identity.get("training") if isinstance(runtime_identity, Mapping) else None
        warm_start = training_identity.get("actor_warm_start") if isinstance(training_identity, Mapping) else None
        load_actor_init = should_load_actor_init_checkpoint(
            actor_init_path=actor_init_path,
            warm_start=warm_start,
            full_checkpoint_resume=bool(self.config.get("full_checkpoint_resume", False)),
        )
        if load_actor_init:
            if not isinstance(warm_start, Mapping):
                raise RuntimeError("Actor initialization requires its inspected warm-start identity")
            requested_sigma = warm_start.get("actor_init_sigma") if isinstance(warm_start, Mapping) else None
            phase_adaptation = bool(warm_start.get("phase_clock_adaptation"))
            loaded_keys = load_actor_init_checkpoint(
                network.package.actor,
                actor_init_path,
                expected_checkpoint_sha256=str(warm_start["checkpoint_sha256"]),  # type: ignore[index]
                actor_init_sigma=float(requested_sigma) if requested_sigma is not None else None,
                allow_phase_clock_adaptation=phase_adaptation,
            )
            if len(loaded_keys) != int(warm_start["loaded_tensor_count"]):  # type: ignore[index]
                raise RuntimeError("actor-init loaded tensor count disagrees with inspected identity")
            if isinstance(warm_start, Mapping) and bool(warm_start.get("initialize_critic", False)):
                load_critic_init_checkpoint(
                    self.model, actor_init_path, expected_checkpoint_sha256=str(warm_start["checkpoint_sha256"]),
                    allow_phase_clock_adaptation=phase_adaptation,
                )
            print(
                {
                    "actor_initialization": "loaded-before-first-rollout",
                    "requested_sigma": requested_sigma,
                    "actual_log_std": float(network.package.actor.global_log_std.detach().item()),
                    "actual_sigma": float(network.package.actor.global_log_std.detach().exp().item()),
                },
                flush=True,
            )

        base_parameters, contextual_parameters = network.actor_parameter_groups()  # disjoint actor groups
        critic_parameters = list(network.package.critic.parameters())
        actor_ids = {id(parameter) for parameter in (*base_parameters, *contextual_parameters)}
        critic_ids = {id(parameter) for parameter in critic_parameters}
        if actor_ids & critic_ids:
            raise RuntimeError("actor and critic optimizer parameters overlap")



        self._base_lr_reference = float(self.config["learning_rate"])
        self._base_lr_ceiling = float(self.config.get("adaptive_lr_max", self._base_lr_reference))
        if self._base_lr_ceiling != self._base_lr_reference:
            raise ValueError("MVP adaptive_lr_max must equal the declared actor base learning-rate anchor")
        self.actor_arm = network.arm
        self.phase_period_steps = network.phase_period_steps
        secondary_lr_key = (
            "contextual_learning_rate" if self.actor_arm in {"direct", "direct_token"} else "residual_learning_rate"
        )
        self._secondary_lr_ratio = float(self.config.get(secondary_lr_key, 1.0e-4)) / self._base_lr_reference
        self._secondary_group_name = (
            "actor_contextual_direct" if self.actor_arm in {"direct", "direct_token"} else "actor_global_residual"
        )
        self._critic_lr_ratio = float(self.config.get("critic_learning_rate", 5.0e-4)) / self._base_lr_reference
        self._gradient_accumulation_steps = int(self.config.get("gradient_accumulation_steps", 1))
        if self._gradient_accumulation_steps < 1 or self.num_minibatches % self._gradient_accumulation_steps != 0:
            raise ValueError("gradient accumulation must divide the number of stratified activation minibatches")
        self._student_optimizer_steps_per_rollout = (
            self.num_minibatches // self._gradient_accumulation_steps * self.mini_epochs_num
        )
        if self.student_mode:
            assert self.student_warmup is not None and self.student_stats is not None
            expected_steps = self.student_warmup.state.optimizer_steps_per_rollout
            if expected_steps != self._student_optimizer_steps_per_rollout:
                if self.student_warmup.state.rollout_updates != 0:
                    raise ValueError(
                        "student optimizer steps per rollout changed after checkpoint progress started"
                    )
                self.student_warmup.state.optimizer_steps_per_rollout = self._student_optimizer_steps_per_rollout
                self.student_stats.optimizer_steps_per_rollout = self._student_optimizer_steps_per_rollout
        fused = torch.device(self.ppo_device).type == "cuda"
        actor_groups = (
            [{"params": base_parameters, "lr": self.last_lr, "name": "actor_base"}]
            if self.student_mode
            else [
                {"params": base_parameters, "lr": self.last_lr, "name": "actor_base"},
                {
                    "params": contextual_parameters,
                    "lr": self.last_lr * self._secondary_lr_ratio,
                    "name": self._secondary_group_name,
                },
            ]
        )
        self.optimizer = torch.optim.Adam(
            actor_groups,
            eps=1.0e-8,
            weight_decay=self.weight_decay,
            fused=fused,
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=self.last_lr * self._critic_lr_ratio,
            eps=1.0e-8,
            weight_decay=self.weight_decay,
            fused=fused,
        )  # independent critic optimizer
        self.asset_count = int(self.config.get("asset_count", 80))
        self.advantage_normalization_scope = str(self.config.get("advantage_normalization_scope", "global"))
        if self.advantage_normalization_scope not in {"global", "per_asset_rollout"}:
            raise ValueError("advantage_normalization_scope must be global or per_asset_rollout")
        if not self.normalize_advantage:
            raise ValueError("palm-rotation PPO requires normalize_advantage for a declared normalization scope")
        self.last_stratified_permutation: torch.Tensor | None = None  # diagnostics/test evidence
        self.last_advantage_asset_means: torch.Tensor | None = None  # full-rollout raw GAE mean `[A]`
        self.last_advantage_asset_stds: torch.Tensor | None = None  # full-rollout raw GAE sample std `[A]`
        identity = self._runtime_identity()
        if not isinstance(identity, dict) or not isinstance(identity.get("identity_digest"), str):
            raise RuntimeError("palm-rotation diagnostics require the exact runtime identity")
        self.metrics_recorder = PalmRotationMetricsRecorder(
            self.experiment_dir,
            identity_digest=identity["identity_digest"],
            flush_every_updates=int(self.config.get("diagnostics_flush_updates", 50)),
        )  # run-owned Parquet shard lifecycle
        self._optimization_count = torch.zeros(self.asset_count, device=self.ppo_device)  # shapes [A]
        common_optimization_fields = (
            "advantage",
            "advantage_square",
            "value_error",
            "return_target",
            "return_target_square",
            "value_prediction",
            "value_prediction_square",
            "value_residual_square",
            "value_error_physical",
            "value_clip_fraction",
            "kl",
            "clip_fraction",
            "action_rms",
            "policy_mean_rms",
            "policy_mean_near_bound_fraction",
            "film_modulation_rms",
        )
        self._mechanism_metric_fields = (
            ("direct_mean_rms", "direct_mean_near_bound_fraction", "direct_pre_tanh_derivative_mean")
            if self.actor_arm in {"direct", "direct_token"}
            else ("base_mean_rms", "residual_rms", "residual_fraction")
        )
        self._optimization_sums = {
            name: torch.zeros(self.asset_count, device=self.ppo_device)
            for name in (*common_optimization_fields, *self._mechanism_metric_fields)
        }
        self._gradient_probe_per_asset: dict[str, torch.Tensor] | None = None
        self._gradient_probe_global: dict[str, float] | None = None
        self._optimizer_step_count = 0
        self._optimizer_microbatch_count = 0
        self._gradient_microbatch_index = 0
        self._optimizer_scalar_sums = {
            name: torch.zeros((), dtype=torch.float32, device=self.ppo_device)
            for name in (
                "actor_loss",
                "critic_loss",
                "entropy",
                "policy_sigma",
                "policy_base_sigma",
                "recovery_floor_fraction",
                "actor_rejected_action_cost",
                "actor_rejected_action_fraction",
                "actor_grad_norm",
                "critic_grad_norm",
                "actor_cagrad_relative_gap",
                "critic_cagrad_relative_gap",
                "actor_cagrad_worst_projection",
                "critic_cagrad_worst_projection",
                "actor_cagrad_iterations",
                "critic_cagrad_iterations",
                "student_anchor_loss",
                "student_anchor_weighted_loss",
                "student_fk_loss",
                "student_fk_weighted_loss",
            )
        }


        if load_actor_init and isinstance(warm_start, Mapping) and warm_start.get("initialize_optimizers", False):
            source = torch.load(actor_init_path, map_location="cpu", weights_only=False)
            names_path = self.config.get("actor_init_optimizer_names")
            named_source = None
            if names_path is not None:
                named_source = load_optimizer_parameter_names(
                    names_path,
                    expected_checkpoint_sha256=str(warm_start["checkpoint_sha256"]),
                    expected_ledger_sha256=str(warm_start["optimizer_name_ledger_sha256"]),
                )
            if warm_start.get("phase_clock_adaptation") and named_source is None:
                raise ValueError("phase adapter optimizer initialization requires a bound parameter-name ledger")
            named_reports = {}
            for optimizer, key in ((self.optimizer, "optimizer"), (self.critic_optimizer, CRITIC_OPTIMIZER_KEY)):
                if named_source is not None:
                    module = network.package.actor if key == "optimizer" else network.package.critic
                    new_name = "phase_contextual_adapter.weight" if key == "optimizer" else "phase_readout_adapter.weight"
                    named_reports[key] = load_named_optimizer_state(
                        optimizer, source[key], named_source[key], dict(module.named_parameters()),
                        allowed_new_names=(new_name,) if warm_start.get("phase_clock_adaptation") else (),
                    )
                    continue
                current_groups, source_groups = optimizer.state_dict()["param_groups"], source[key]["param_groups"]
                if len(current_groups) != len(source_groups) or any(
                    len(current["params"]) != len(saved["params"]) or current.get("name") != saved.get("name")
                    for current, saved in zip(current_groups, source_groups, strict=True)
                ):
                    raise ValueError("optimizer initialization requires the same ordered parameter groups")
                optimizer.load_state_dict(source[key])
            continuation = source[TRAINING_CONTINUATION_KEY]
            self.entropy_coef = float(continuation["entropy_coef"])
            self.update_lr(float(continuation["last_lr"]))
            random.setstate(continuation["python_random_state"])
            np.random.set_state(continuation["numpy_random_state"])
            torch.set_rng_state(continuation["torch_cpu_rng_state"])
            if torch.cuda.is_available() and continuation["torch_cuda_rng_states"]:
                torch.cuda.set_rng_state_all(continuation["torch_cuda_rng_states"])
            print({"learner_initialization": "actor-critic-popart-adam-rng-inherited",
                   "source_epoch": int(source["epoch"]), "new_run_counters": "restart-at-zero"}, flush=True)
            if named_reports:
                actor_phase = getattr(network.package.actor, "phase_contextual_adapter", None)
                critic_phase = getattr(network.package.critic, "phase_readout_adapter", None)
                self._named_initialization_audit = {
                    "source_checkpoint_sha256": str(warm_start["checkpoint_sha256"]),
                    "optimizer_name_ledger_sha256": str(warm_start["optimizer_name_ledger_sha256"]),
                    "optimizers": named_reports,
                    "value_rms_initial_count": float(self.model.value_mean_std.count.item()),
                    "actor_phase_adapter_initial_max_abs": float(actor_phase.weight.detach().abs().max().item()) if actor_phase is not None else None,
                    "critic_phase_adapter_initial_max_abs": float(critic_phase.weight.detach().abs().max().item()) if critic_phase is not None else None,
                }
                print({"named_learner_initialization": self._named_initialization_audit}, flush=True)

    def _ensure_student_anchor_sampler(self) -> FamilyStudentAnchorSampler:
        'Ensure student anchor sampler.'

        if not bool(getattr(self, "student_mode", False)):
            raise RuntimeError("student anchor sampler requested by legacy PPO agent")
        if self.student_anchor_sampler is not None:
            return self.student_anchor_sampler
        network = self.model.a2c_network
        dataset_paths = self._student_anchor_dataset_paths
        source_hashes = self._student_anchor_source_hashes or None
        # Default final-shared-student lock hashes are verified against current bytes; a resume must fail
        # closed if an HDF5 source was replaced under the same path.  Explicit sources are hashed here too,
        # so their identity becomes part of the continuation state rather than a path-only convention.
        compute_source_hash = True
        self.student_anchor_sampler = FamilyStudentAnchorSampler(
            dataset_paths,
            batch_size=int(getattr(network, "student_anchor_batch_size", 2048)),
            seed=int(getattr(network, "student_anchor_seed", self.config.get("seed", 42))),
            source_hashes=cast(Mapping[str | Path, str] | None, source_hashes),
            compute_source_hash=compute_source_hash,
        )
        self.student_anchor_source_data_hash = self.student_anchor_sampler.source_data_hash
        return self.student_anchor_sampler

    def _set_student_actor_requires_grad(self, enabled: bool) -> None:
        'Handle set student Actor requires grad.'

        if not bool(getattr(self, "student_mode", False)):
            return
        actor = self.model.a2c_network.package.actor
        if not self._student_actor_requires_grad:
            self._student_actor_requires_grad = {
                id(parameter): bool(parameter.requires_grad) for parameter in actor.parameters()
            }
        for parameter in actor.parameters():
            original = self._student_actor_requires_grad[id(parameter)]
            parameter.requires_grad_(enabled and original)

    def _begin_student_rollout(self) -> None:
        'Begin student rollout.'

        if not bool(getattr(self, "student_mode", False)):
            return
        assert self.student_warmup is not None and self.student_stats is not None
        self._student_rollout_phase = self.student_warmup.begin_rollout(
            optimizer_steps_per_rollout=self._student_optimizer_steps_per_rollout
        )
        self._student_pending_anchor_loss = torch.zeros((), device=self.ppo_device)
        self._student_pending_fk_loss = torch.zeros((), device=self.ppo_device)

    def _finish_student_rollout(self, *, success: bool) -> None:
        'Finish student rollout.'

        if not bool(getattr(self, "student_mode", False)):
            return
        assert self.student_warmup is not None and self.student_stats is not None
        try:
            if success:
                self.student_warmup.finish_rollout()
                self.student_stats.record_rollout_update()
                if self.student_stats.rollout_updates != self.student_warmup.state.rollout_updates:
                    raise RuntimeError("student warmup/stats rollout counters diverged")
                self._update_student_initialization_metadata()
        finally:
            self.student_warmup.abort_rollout()
            self._set_student_actor_requires_grad(True)
            self._student_rollout_phase = None

    def _update_student_initialization_metadata(self) -> None:
        'Update student initialization metadata.'

        if not bool(getattr(self, "student_mode", False)):
            return
        assert self.student_warmup is not None and self.student_stats is not None
        network = self.model.a2c_network
        metadata = getattr(network, "student_initialization_metadata", None)
        if not isinstance(metadata, dict):
            return
        metadata["warmup"] = self.student_warmup.metadata()
        metadata["stats"] = self.student_stats.state_dict()
        if self.student_anchor_sampler is not None:
            metadata["anchor_sampler"] = {
                "source_data_hash": self.student_anchor_sampler.source_data_hash,
                "sampled_batches": self.student_anchor_sampler.sampled_batches,
                "sampled_samples": self.student_anchor_sampler.sampled_samples,
            }

    def _student_anchor_backward(self) -> None:
        'Handle student anchor backward.'

        if not bool(getattr(self, "student_mode", False)) or self._student_rollout_phase != "actor_critic":
            return
        assert self.student_warmup is not None and self.student_stats is not None
        network = self.model.a2c_network
        hooks = network.get_student_auxiliary_hooks()
        if hooks is None:
            raise RuntimeError("student Actor phase requires auxiliary hooks")
        sampler = self._ensure_student_anchor_sampler()
        anchor_cpu = sampler.sample()
        batch_size = anchor_cpu.target.shape[0]
        if batch_size != int(getattr(network, "student_anchor_batch_size", 2048)):
            raise RuntimeError("student anchor sampler returned an unexpected batch size")
        # `anchor_cpu.target` is retained only as source provenance; the promised target is this run's
        # frozen initial BC Actor mean, so no teacher-center label can silently become the RL anchor.
        microbatch_size = min(self._student_anchor_microbatch_size, batch_size)
        weighted_loss = torch.zeros((), device=self.ppo_device)
        for start in range(0, batch_size, microbatch_size):
            stop = min(start + microbatch_size, batch_size)
            chunk = anchor_cpu.select(slice(start, stop)).to(self.ppo_device)
            with torch.no_grad():
                il_mean = network.package.initial_bc_mean(
                    chunk.observation,
                    chunk.geometry,
                    chunk.joint_kinematics,
                )
            output = network.package.actor(
                chunk.observation,
                chunk.geometry,
                joint_kinematics=chunk.joint_kinematics,
            )
            chunk_loss = hooks.anchor_loss(output.mean, il_mean, chunk.joint_valid)
            fraction = float(stop - start) / float(batch_size)
            (hooks.anchor_weight * chunk_loss * fraction).backward()
            weighted_loss = weighted_loss + chunk_loss.detach() * fraction
        self._student_pending_anchor_loss = weighted_loss.detach()
        self._optimizer_scalar_sums["student_anchor_loss"].add_(weighted_loss.detach())
        self._optimizer_scalar_sums["student_anchor_weighted_loss"].add_(
            weighted_loss.detach() * hooks.anchor_weight
        )

    def init_tensors(self) -> None:
        'Initialize tensors.'

        batch = self.num_agents * self.num_actors
        if self.config.get("env_major_rollout_storage", False):
            if self.is_rnn:
                raise ValueError("env-major experience currently supports the non-RNN Palm policy path")
            info = {"num_actors": self.num_actors, "horizon_length": self.horizon_length,
                    "has_central_value": self.has_central_value, "use_action_masks": self.use_action_masks}
            self.experience_buffer = EnvMajorExperienceBuffer(self.env_info, info, self.ppo_device)
            self.init_current_rewards(batch, (batch, self.value_size))
            self.update_list = ["actions", "neglogpacs", "values", "mus", "sigmas"]
            self.tensor_list = self.update_list + ["obses", "states", "dones"]
            print("[STORAGE] Env-major backing enabled: rollout flatten uses views, not full observation copies.", flush=True)
        else:
            super().init_tensors()
        mechanism_key = "direct_means" if self.actor_arm in {"direct", "direct_token"} else "residuals"
        channels = (mechanism_key, "film_modulations")
        for name in channels:
            if isinstance(self.experience_buffer, EnvMajorExperienceBuffer):
                value = self.experience_buffer.side_channel(16)
            else:
                value = torch.zeros(self.horizon_length, batch, 16, dtype=torch.float32, device=self.ppo_device)
            self.experience_buffer.tensor_dict[name] = value  # shapes [H,N,16]
        self.update_list.append(mechanism_key)
        self.tensor_list.append(mechanism_key)
        self.update_list.append("film_modulations")
        self.tensor_list.append("film_modulations")

    def update_lr(self, lr: float) -> None:
        'Update lr.'

        current = bounded_adaptive_learning_rate(float(lr), self._base_lr_ceiling)
        self.last_lr = current
        for group in self.optimizer.param_groups:
            group["lr"] = current if group.get("name") == "actor_base" else current * self._secondary_lr_ratio
        for group in self.critic_optimizer.param_groups:
            group["lr"] = current * self._critic_lr_ratio

    def _assert_actor_learning_rate_ratio(self) -> None:
        'Handle assert Actor learning rate ratio.'

        groups = {str(group.get("name")): float(group["lr"]) for group in self.optimizer.param_groups}
        expected_base = float(self.last_lr)
        if bool(getattr(self, "student_mode", False)):
            if abs(groups.get("actor_base", -1.0) - expected_base) > 1.0e-12:
                raise RuntimeError(f"student Actor LR drifted before optimizer step: {groups}")
            return
        expected_secondary = expected_base * self._secondary_lr_ratio
        if abs(groups.get("actor_base", -1.0) - expected_base) > 1.0e-12:
            raise RuntimeError(f"actor base LR drifted before optimizer step: {groups}")
        if abs(groups.get(self._secondary_group_name, -1.0) - expected_secondary) > 1.0e-12:
            raise RuntimeError(f"actor contextual LR ratio drifted before optimizer step: {groups}")

    def train_actor_critic(self, input_dict: dict[str, Any]):
        'Train Actor Critic.'

        self.set_train()
        self.calc_gradients(input_dict)
        return self.train_result

    @staticmethod
    def masked_policy_kl(
        current_mu: torch.Tensor,
        current_sigma: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        'Handle masked policy KL.'

        return mean_preserving_squashed_kl(
            current_mu,
            current_sigma,
            old_mu,
            old_sigma,
            active_mask,
            action_epsilon=PalmRotationMaskedContinuousModel.Network._ACTION_EPS,
        )

    def _gradient_probe_parameters(self) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
        'Handle gradient probe parameters.'

        return probes.gradient_probe_parameters(self)

    @staticmethod
    def _per_asset_gradient_matrix(
        objective: torch.Tensor,
        labels: torch.Tensor,
        parameters: tuple[nn.Parameter, ...],
        *,
        asset_count: int,
    ) -> torch.Tensor:
        'Handle per asset gradient matrix.'

        return probes.per_asset_gradient_matrix(objective, labels, parameters, asset_count=asset_count)

    def _run_gradient_probe(
        self,
        *,
        actor_objective: torch.Tensor,
        critic_objective: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        'Run gradient probe.'

        return probes.run_gradient_probe(
            self, actor_objective=actor_objective, critic_objective=critic_objective, labels=labels
        )

    def _run_full_actor_gradient_shadow(
        self,
        *,
        global_objective: torch.Tensor,
        per_asset_objective: torch.Tensor,
        labels: torch.Tensor,
        replica_halves: torch.Tensor,
    ) -> None:
        'Run full Actor gradient shadow.'

        return probes.run_full_actor_gradient_shadow(
            self,
            global_objective=global_objective,
            per_asset_objective=per_asset_objective,
            labels=labels,
            replica_halves=replica_halves,
        )

    @staticmethod
    def _index_dataset_value(value: Any, indices: torch.Tensor) -> Any:
        'Handle index dataset value.'

        if isinstance(value, dict):
            return {key: tensor[indices] for key, tensor in value.items()}  # named experience tensors
        return value[indices] if isinstance(value, torch.Tensor) else value

    def prepare_dataset(self, batch_dict: dict[str, Any]) -> None:
        'Prepare dataset.'

        raw_returns = batch_dict.get("returns")
        raw_values = batch_dict.get("values")
        if not isinstance(raw_returns, torch.Tensor) or not isinstance(raw_values, torch.Tensor):
            raise RuntimeError("palm-rotation rollout lacks raw return/value tensors")
        rollout_observation = batch_dict.get("obses")
        if not isinstance(rollout_observation, Mapping) or "prototype_index" not in rollout_observation:
            raise RuntimeError("palm-rotation rollout lacks prototype labels for advantage normalization")
        rollout_labels = rollout_observation["prototype_index"].reshape(-1).long()  # shapes [B]
        raw_advantages = (raw_returns - raw_values).sum(dim=1)  # shapes [B]
        if raw_advantages.numel() % self.horizon_length != 0:
            raise RuntimeError("flattened rollout does not divide into complete environment trajectories")
        phase_period = getattr(self, "phase_period_steps", None)
        if ("phase_clock" in rollout_observation) != (phase_period is not None):
            raise RuntimeError("rollout phase clock and agent configuration disagree")
        if phase_period is not None and not hasattr(self, "_phase_transport_witness"):
            phase = rollout_observation["phase_clock"].detach().reshape(-1, self.horizon_length, 2)
            resets = batch_dict["dones"].detach().bool().reshape(-1, self.horizon_length)
            report = audit_phase_rollout(phase, resets, period_steps=phase_period)
            witness = Path(self.nn_dir).parent / "phase_rollout_witness.h5"
            write_selected_trajectories_hdf5(
                witness, arrays={"phase_clock": phase.cpu().numpy(), "reset_observation": resets.cpu().numpy()},
                metadata={**report, "axes": "environment,time,feature", "method_identity_digest": self.model.a2c_network.anymani_identity["identity_digest"]},
            )
            self._phase_transport_witness = {**report, "path": str(witness), "sha256": hashlib.sha256(witness.read_bytes()).hexdigest()}
        flat_index = torch.arange(raw_advantages.numel(), device=raw_advantages.device)
        environment_index = torch.div(flat_index, self.horizon_length, rounding_mode="floor")  # env-major flatten axis
        expected_labels = environment_index.remainder(self.asset_count)
        if not bool(torch.equal(rollout_labels, expected_labels)):
            raise RuntimeError("rollout prototype labels disagree with env-major round-robin routing")
        replica_index = torch.div(environment_index, self.asset_count, rounding_mode="floor")
        replica_halves = replica_index.remainder(2)
        mechanism_key = "direct_means" if self.actor_arm in {"direct", "direct_token"} else "residuals"
        mechanism = batch_dict.get(mechanism_key)  # detached arm-specific rollout diagnostic`[B,16]`
        if not isinstance(mechanism, torch.Tensor):
            raise RuntimeError(f"palm-rotation rollout is missing {mechanism_key} side-channel")
        rollout_mean = batch_dict.get("mus")  # shapes [B,16]
        if not isinstance(rollout_mean, torch.Tensor) or rollout_mean.shape != mechanism.shape:
            raise RuntimeError("palm-rotation rollout mean and mechanism side-channel shapes disagree")
        frozen_rollout_mean = rollout_mean.detach().clone()
        if isinstance(self.value_mean_std, PopArtValueNormalizer):
            head = self.model.a2c_network.package.critic.value_head[-1]
            if not isinstance(head, nn.Linear):
                raise TypeError("PopArt requires the structured Critic scalar Linear head")
            self._last_popart_compensation = self.value_mean_std.update_from_returns(raw_returns, head)

        super().prepare_dataset(batch_dict)
        global_advantages = self.dataset.values_dict.get("advantages")
        if not isinstance(global_advantages, torch.Tensor) or global_advantages.shape != raw_advantages.shape:
            raise RuntimeError("upstream global advantage tensor disagrees with rollout GAE shape")
        per_asset_advantages, asset_means, asset_stds = normalize_advantages_per_asset(
            raw_advantages,
            rollout_labels,
            asset_count=self.asset_count,
        )  # shapes [B], [A]
        self.dataset.values_dict["global_advantages"] = global_advantages.detach()
        self.dataset.values_dict["per_asset_advantages"] = per_asset_advantages.detach()
        self.dataset.values_dict["replica_halves"] = replica_halves.detach()
        self.dataset.values_dict["advantages"] = (
            per_asset_advantages if self.advantage_normalization_scope == "per_asset_rollout" else global_advantages
        )
        self.last_advantage_asset_means = asset_means.detach()
        self.last_advantage_asset_stds = asset_stds.detach()
        self.dataset.values_dict["raw_returns"] = raw_returns.detach()  # shapes [B,1]
        self.dataset.values_dict["raw_values"] = raw_values.detach()  # shapes [B,1]
        self.dataset.values_dict["rollout_mu"] = frozen_rollout_mean
        self.dataset.values_dict[mechanism_key] = mechanism
        film_modulations = batch_dict.get("film_modulations")
        if not isinstance(film_modulations, torch.Tensor) or film_modulations.shape != mechanism.shape:
            raise RuntimeError("palm-rotation rollout is missing geometry FiLM side-channel")
        self.dataset.values_dict["film_modulations"] = film_modulations  # detached mechanism diagnostic
        observation = self.dataset.values_dict.get("obs")
        if not isinstance(observation, dict) or "prototype_index" not in observation:
            raise RuntimeError("stratified PPO requires prototype_index in cached observations")
        permutation = stratified_asset_permutation(
            observation["prototype_index"],
            asset_count=self.asset_count,
            minibatch_count=self.num_minibatches,
        )  # shapes [B]
        if permutation.numel() != self.batch_size or self.minibatch_size * self.num_minibatches != self.batch_size:
            raise RuntimeError("stratified permutation disagrees with rl_games batch geometry")
        self.dataset.values_dict = {
            key: self._index_dataset_value(value, permutation) for key, value in self.dataset.values_dict.items()
        }
        self.last_stratified_permutation = permutation.detach()
        audit_frequency = int(self.config.get("optimization_audit_frequency", 0))
        if audit_frequency > 0 and (int(self.epoch_num) == 1 or int(self.epoch_num) % audit_frequency == 0):
            destination = Path(self.nn_dir).parent
            segment = str(self.config["evidence_segment_id"])
            write_optimization_evidence(
                destination / "optimization_audits" / segment / f"pre-update-{int(self.epoch_num):06d}.pt",
                {
                    "artifact_type": "palm-rotation-optimization-audit",
                    "schema_version": "1.0.0",
                    "capture_phase": "post-dataset-prepare-before-optimizer",
                    "segment_id": segment,
                    "update": int(self.epoch_num),
                    "policy_version": int(self.frame),
                    "identity": self.model.a2c_network.anymani_identity,
                    "model": self.model.state_dict(),
                    "actor_optimizer": self.optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "dataset": self.dataset.values_dict,
                    "permutation": permutation,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "loss_configuration": {
                        key: self.config.get(key)
                        for key in (
                            "e_clip",
                            "entropy_coef",
                            "bounds_loss_coef",
                            "critic_coef",
                            "clip_value",
                            "gradient_aggregation",
                            "cagrad_c",
                            "cagrad_task_chunk",
                            "gradient_accumulation_steps",
                        )
                    },
                },
            )

        if self.config.get("release_rollout_batch", False):


            step_time = batch_dict["step_time"]
            batch_dict.clear()
            batch_dict["step_time"] = step_time
            self._release_unused_rollout_cache = True
        if bool(getattr(self, "student_mode", False)) and self._student_rollout_phase == "critic_only_warmup":
            self._set_student_actor_requires_grad(False)

    def calc_gradients(self, input_dict: dict[str, Any]) -> None:
        'Handle calc gradients.'

        if getattr(self, "_release_unused_rollout_cache", False):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._release_unused_rollout_cache = False
        value_predictions = input_dict["old_values"]  # rollout normalized values`[M,1]`
        old_neglogp = input_dict["old_logp_actions"]  # masked action negative log probability`[M]`
        advantage = input_dict["advantages"]  # identity-selected global/per-asset GAE`[M]`
        global_advantages = input_dict.get("global_advantages")
        per_asset_advantages = input_dict.get("per_asset_advantages")
        replica_halves = input_dict.get("replica_halves")  # even/odd runtime replica split `[M]`
        if not all(
            isinstance(value, torch.Tensor) for value in (global_advantages, per_asset_advantages, replica_halves)
        ):
            raise RuntimeError("PPO minibatch lacks advantage-scope or replica-half shadow tensors")
        kl_reference_mu = input_dict["mu"]  # shapes [M,16]; units M
        kl_reference_sigma = input_dict["sigma"]  # shapes [M,16]; units M
        rollout_mean = input_dict.get("rollout_mu")  # shapes [M,16]; units M
        if not isinstance(rollout_mean, torch.Tensor):
            raise RuntimeError("PPO minibatch lacks immutable rollout policy means")
        returns = input_dict["returns"]  # normalized return targets`[M,1]`
        raw_returns = input_dict.get("raw_returns")  # shapes [M,1]; units M
        raw_values = input_dict.get("raw_values")  # shapes [M,1]; units M
        if not isinstance(raw_returns, torch.Tensor) or not isinstance(raw_values, torch.Tensor):
            raise RuntimeError("PPO minibatch lacks raw return/value diagnostics")
        actions = input_dict["actions"]  # sampled canonical actions`[M,16]`
        observation = self._preproc_obs(input_dict["obs"])
        labels = observation["prototype_index"].reshape(-1).long()
        network = self.model.a2c_network  # validated PalmRotationRlGamesNetwork
        if self.student_mode and self._student_rollout_phase is None:
            raise RuntimeError("student PPO calc_gradients requires a rollout-level phase lock")
        if self.student_mode:
            assert self.student_warmup is not None and self.student_stats is not None
        student_warmup = self.student_warmup
        student_stats = self.student_stats
        student_actor_enabled = not self.student_mode or self._student_rollout_phase == "actor_critic"
        task_actor_gradients = task_critic_gradients = None
        if self.gradient_aggregation == "cagrad":
            network = self.model.a2c_network
            task_actor_gradients, task_critic_gradients, result = per_asset_ppo_gradients(
                network.package,
                observation,
                input_dict,
                asset_count=self.asset_count,
                entropy_noise=torch.randn_like(actions),
                chunk_size=self.cagrad_task_chunk,
                clip_epsilon=self.e_clip,
                entropy_coef=self.entropy_coef,
                bounds_coef=float(self.bounds_loss_coef or 0.0),
                rejected_action_weight=self.rejected_action_weight,
                critic_coef=self.critic_coef,
                clip_value=self.clip_value,
            )
            network.last_active_joint_mask = observation["jnt_valid"].bool()
        else:
            result = self.model({"is_train": True, "prev_actions": actions, "obs": observation})
        new_neglogp = result["prev_neglogp"]  # masked active-joint likelihood
        values = result["values"]  # privileged critic prediction`[M,1]`
        entropy = result["entropy"]  # mean entropy per active DoF`[M]`
        mu = result["mus"]  # current actor means`[M,16]`
        sigma = result["sigmas"]  # shapes [M,16]; units M
        student_fk_loss = torch.zeros((), dtype=mu.dtype, device=mu.device)
        if self.student_mode and student_actor_enabled:
            fk_prediction = result.get("student_fk_prediction")
            fk_target = result.get("student_fk_target")
            if network.student_variant == "fk":
                if not isinstance(fk_prediction, torch.Tensor) or not isinstance(fk_target, torch.Tensor):
                    raise RuntimeError("FK student PPO requires online parsed joint-origin target and prediction")
                hooks = network.get_student_auxiliary_hooks()
                if hooks is None:
                    raise RuntimeError("FK student PPO requires auxiliary hooks")
                student_fk_loss = hooks.fk_loss(fk_prediction, fk_target, observation["jnt_valid"].bool())
            elif fk_prediction is not None or fk_target is not None:
                raise RuntimeError("non-FK student variant received an FK auxiliary transport")
        actor = self.model.a2c_network.package.actor
        base_sigma = actor.global_log_std.detach().exp()
        recovery_floor_fraction = sigma.new_zeros(())
        if actor.recovery_sigma_floor is not None:
            valid = observation["jnt_valid"].bool()
            recovery_floor_fraction = ((sigma.detach() > base_sigma + 1e-6) & valid).sum() / valid.sum()


        rejected_cost = torch.zeros_like(advantage)
        rejected_fraction = torch.zeros_like(advantage)
        if self.rejected_action_weight > 0.0:
            rejected_cost, rejected_fraction = tip_silent_rejected_action_cost(
                mean=mu,
                target_normalized=observation["actor_jnt_current"][..., 1],
                limits_normalized=observation["actor_jnt_limits"],
                joint_valid=observation["jnt_valid"],
                tip_contact=observation["actor_owner_contact"][:, 17:21, 0],
                tip_valid=observation["tip_valid"],
            )

        actor_loss_vector = self.actor_loss_func(old_neglogp, new_neglogp, advantage, self.ppo, self.e_clip)
        bounds_loss_vector = self.bound_loss(mu)  # active-DoF mean bounds penalty
        actor_objective_vector = (
            actor_loss_vector - entropy * self.entropy_coef + bounds_loss_vector * self.bounds_loss_coef
        )  # shapes [M]; units M
        actor_terms, _ = torch_ext.apply_masks(
            [actor_loss_vector.unsqueeze(1), entropy.unsqueeze(1), bounds_loss_vector.unsqueeze(1)],
            None,
        )
        actor_loss, entropy_loss, bounds_loss = actor_terms
        actor_objective = actor_loss - entropy_loss * self.entropy_coef + bounds_loss * self.bounds_loss_coef
        if self.rejected_action_weight > 0.0:
            weighted_rejection = self.rejected_action_weight * rejected_cost
            actor_objective_vector = actor_objective_vector + weighted_rejection
            actor_objective = actor_objective + weighted_rejection.mean()
        if self.student_mode and student_actor_enabled:
            hooks = network.get_student_auxiliary_hooks()
            if hooks is None:
                raise RuntimeError("student PPO requires anchor/FK auxiliary hooks")
            actor_objective_vector = actor_objective_vector + hooks.fk_weight * student_fk_loss
            actor_objective = actor_objective + hooks.fk_weight * student_fk_loss
            self._student_pending_fk_loss.add_(student_fk_loss.detach())
            self._optimizer_scalar_sums["student_fk_loss"].add_(student_fk_loss.detach())
            self._optimizer_scalar_sums["student_fk_weighted_loss"].add_(
                student_fk_loss.detach() * hooks.fk_weight
            )


        critic_vector = common_losses.critic_loss(
            self.model,
            value_predictions,
            values,
            self.e_clip,
            returns,
            self.clip_value,
        )
        critic_terms, _ = torch_ext.apply_masks([critic_vector], None)
        critic_loss = critic_terms[0]
        critic_objective = 0.5 * self.critic_coef * critic_loss
        critic_objective_vector = 0.5 * self.critic_coef * critic_vector.reshape(-1)  # `[M]`


        finite_forward = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "bounds_loss": bounds_loss,
            "actor_objective": actor_objective,
            "rejected_action_cost": rejected_cost,
            "critic_objective": critic_objective,
            "student_fk_loss": student_fk_loss,
            "mu": mu,
            "sigma": sigma,
            "value": values,
        }
        for name, value in finite_forward.items():
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.isfinite(value).all(),
                f"palm-rotation PPO produced non-finite {name}",
            )


        full_gradient_shadow_frequency = int(self.config.get("full_gradient_shadow_frequency", 0))
        if (
            full_gradient_shadow_frequency > 0
            and int(self.epoch_num) % full_gradient_shadow_frequency == 0
            and self._gradient_microbatch_index == 0
        ):
            common_actor_regularizer = -entropy * self.entropy_coef + bounds_loss_vector * self.bounds_loss_coef
            if self.rejected_action_weight > 0.0:
                common_actor_regularizer = common_actor_regularizer + self.rejected_action_weight * rejected_cost
            global_actor_objective = (
                self.actor_loss_func(
                    old_neglogp,
                    new_neglogp,
                    cast(torch.Tensor, global_advantages),
                    self.ppo,
                    self.e_clip,
                )
                + common_actor_regularizer
            )
            per_asset_actor_objective = (
                self.actor_loss_func(
                    old_neglogp,
                    new_neglogp,
                    cast(torch.Tensor, per_asset_advantages),
                    self.ppo,
                    self.e_clip,
                )
                + common_actor_regularizer
            )
            self._run_full_actor_gradient_shadow(
                global_objective=global_actor_objective,
                per_asset_objective=per_asset_actor_objective,
                labels=labels,
                replica_halves=cast(torch.Tensor, replica_halves),
            )


        gradient_probe_frequency = int(self.config.get("gradient_probe_frequency", 0))
        if (
            gradient_probe_frequency > 0
            and int(self.epoch_num) % gradient_probe_frequency == 0
            and self._gradient_microbatch_index == 0
        ):
            self._run_gradient_probe(
                actor_objective=actor_objective_vector,
                critic_objective=critic_objective_vector,
                labels=labels,
            )


        accumulation_offset = self._gradient_microbatch_index % self._gradient_accumulation_steps
        if accumulation_offset == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            if self.student_mode and student_actor_enabled:
                self._student_anchor_backward()
        if self.gradient_aggregation == "cagrad":
            if task_actor_gradients is None or task_critic_gradients is None:
                raise RuntimeError("CAGrad task gradients are missing")

            if accumulation_offset == 0:
                self._cagrad_actor_accumulator = {
                    name: value.div_(self._gradient_accumulation_steps) for name, value in task_actor_gradients.items()
                }
                self._cagrad_critic_accumulator = {
                    name: value.div_(self._gradient_accumulation_steps) for name, value in task_critic_gradients.items()
                }
            else:
                assert self._cagrad_actor_accumulator is not None and self._cagrad_critic_accumulator is not None
                for name, value in task_actor_gradients.items():
                    self._cagrad_actor_accumulator[name].add_(value, alpha=1 / self._gradient_accumulation_steps)
                for name, value in task_critic_gradients.items():
                    self._cagrad_critic_accumulator[name].add_(value, alpha=1 / self._gradient_accumulation_steps)
        else:
            if student_actor_enabled:
                (actor_objective / self._gradient_accumulation_steps).backward()
            (critic_objective / self._gradient_accumulation_steps).backward()
        if not self.truncate_grads:
            raise RuntimeError("palm-rotation PPO requires independent actor/critic gradient clipping")
        accumulation_boundary = accumulation_offset + 1 == self._gradient_accumulation_steps
        if accumulation_boundary:
            self._assert_actor_learning_rate_ratio()
            if self.gradient_aggregation == "cagrad":
                assert self._cagrad_actor_accumulator is not None and self._cagrad_critic_accumulator is not None
                actor_direction, actor_diagnostics = combine_task_gradients(
                    self._cagrad_actor_accumulator, c=self.cagrad_c
                )
                critic_direction, critic_diagnostics = combine_task_gradients(
                    self._cagrad_critic_accumulator, c=self.cagrad_c
                )
                for name, parameter in network.package.actor.named_parameters():
                    parameter.grad = actor_direction[name]
                for name, parameter in network.package.critic.named_parameters():
                    parameter.grad = critic_direction[name]
                self._last_cagrad_diagnostics = {
                    **{f"actor/{name}": value for name, value in actor_diagnostics.items()},
                    **{f"critic/{name}": value for name, value in critic_diagnostics.items()},
                }
                for side, diagnostic in (("actor", actor_diagnostics), ("critic", critic_diagnostics)):
                    for name in ("relative_gap", "worst_projection", "iterations"):
                        self._optimizer_scalar_sums[f"{side}_cagrad_{name}"].add_(diagnostic[name].float())
                self._cagrad_actor_accumulator = self._cagrad_critic_accumulator = None
            actor_grad_norm = (
                clip_grad_norm_(network.package.actor.parameters(), self.grad_norm)
                if student_actor_enabled or not self.student_mode
                else torch.zeros((), dtype=torch.float32, device=self.ppo_device)
            )
            critic_grad_norm = clip_grad_norm_(
                network.package.critic.parameters(), self.grad_norm
            )
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.isfinite(actor_grad_norm) & torch.isfinite(critic_grad_norm),
                "palm-rotation PPO produced non-finite actor or critic gradient norm",
            )

            self.critic_optimizer.step()
            if self.student_mode:
                assert student_warmup is not None and student_stats is not None
                student_warmup.on_critic_step()
                student_stats.record_critic_update()
            if student_actor_enabled:
                self.optimizer.step()
                network.package.actor.project_exploration_parameters()
                if self.student_mode:
                    assert student_warmup is not None and student_stats is not None
                    student_warmup.on_actor_step()
                    student_stats.record_actor_update()
            self._optimizer_step_count += 1
            self._optimizer_scalar_sums["actor_grad_norm"].add_(actor_grad_norm.detach().float())
            self._optimizer_scalar_sums["critic_grad_norm"].add_(critic_grad_norm.detach().float())
            if self.student_mode:
                fk_mean = self._student_pending_fk_loss / float(self._gradient_accumulation_steps)
                assert student_stats is not None
                student_stats.record_auxiliary(
                    anchor_loss=float(self._student_pending_anchor_loss.detach().item()),
                    fk_loss=float(fk_mean.detach().item()),
                )
                self._student_pending_anchor_loss.zero_()
                self._student_pending_fk_loss.zero_()
        self._gradient_microbatch_index += 1


        self._optimizer_microbatch_count += 1
        microbatch_scalars = {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": entropy_loss.detach(),
            "policy_sigma": (sigma.detach() * observation["jnt_valid"]).sum() / observation["jnt_valid"].sum(),
            "policy_base_sigma": base_sigma,
            "recovery_floor_fraction": recovery_floor_fraction,
            "actor_rejected_action_cost": rejected_cost.detach().mean(),
            "actor_rejected_action_fraction": rejected_fraction.detach().mean(),
        }
        for name, value in microbatch_scalars.items():
            self._optimizer_scalar_sums[name].add_(value.float())


        active_mask = network.last_active_joint_mask
        if not isinstance(active_mask, torch.Tensor) or active_mask.shape != mu.shape:
            raise RuntimeError("palm-rotation network did not expose active-joint mask")
        with torch.no_grad():
            kl_per_sample = self.masked_policy_kl(
                mu.detach(), sigma.detach(), kl_reference_mu, kl_reference_sigma, active_mask
            )
            kl = kl_per_sample.mean()
            ratio = torch.exp(old_neglogp - new_neglogp.detach())  # PPO importance ratio`[M]`
            clip_fraction = (torch.abs(ratio - 1.0) > self.e_clip).float()  # clipped sample indicator
            active_float = active_mask.float()
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            action_rms = torch.sqrt((actions.square() * active_float).sum(dim=-1) / active_count)
            value_prediction_physical = denormalize_value_readonly(self.model, values.detach()).reshape(-1)
            return_target_physical = raw_returns.reshape(-1)
            value_residual_physical = value_prediction_physical - return_target_physical
            value_clip_fraction = (
                (values.detach().reshape(-1) - value_predictions.reshape(-1)).abs() > self.e_clip
            ).float()
            mechanism_key = "direct_means" if self.actor_arm in {"direct", "direct_token"} else "residuals"
            mechanism = input_dict.get(mechanism_key)
            if not isinstance(mechanism, torch.Tensor):
                raise RuntimeError(f"PPO minibatch lacks {mechanism_key} diagnostics")
            mechanism_metrics = rollout_policy_mechanism_metrics(
                rollout_mean,
                mechanism,
                active_mask,
                actor_arm=cast(Literal["base", "residual", "direct", "direct_token"], self.actor_arm),
            )
            film_modulations = input_dict.get("film_modulations")
            if not isinstance(film_modulations, torch.Tensor) or film_modulations.shape != actions.shape:
                raise RuntimeError("PPO minibatch geometry FiLM diagnostics disagree with action shape")
            film_modulation_rms = (film_modulations * active_float).sum(dim=-1) / active_count
            optimization_values = {
                "advantage": advantage.detach(),
                "advantage_square": advantage.detach().square(),
                "value_error": torch.abs(values.detach().squeeze(-1) - returns.squeeze(-1)),
                "return_target": return_target_physical,
                "return_target_square": return_target_physical.square(),
                "value_prediction": value_prediction_physical,
                "value_prediction_square": value_prediction_physical.square(),
                "value_residual_square": value_residual_physical.square(),
                "value_error_physical": value_residual_physical.abs(),
                "value_clip_fraction": value_clip_fraction,
                "kl": kl_per_sample,
                "clip_fraction": clip_fraction,
                "action_rms": action_rms,
                "policy_mean_rms": mechanism_metrics["policy_mean_rms"],
                "policy_mean_near_bound_fraction": mechanism_metrics["policy_mean_near_bound_fraction"],
                "film_modulation_rms": film_modulation_rms,
            }
            optimization_values.update(
                {name: mechanism_metrics[name] for name in self._mechanism_metric_fields}
            )
            self._optimization_count.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
            for name, per_sample in optimization_values.items():
                self._optimization_sums[name].scatter_add_(0, labels, per_sample.float())
        self.diagnostics.mini_batch(
            self,
            {
                "values": value_predictions,
                "returns": returns,
                "new_neglogp": new_neglogp,
                "old_neglogp": old_neglogp,
                "masks": None,
            },
            self.e_clip,
            0,
        )
        if self.student_mode:
            network.last_fk_prediction = None
            network.last_fk_target = None
        self.train_result = (
            actor_loss.detach(),
            critic_loss.detach(),
            entropy_loss.detach(),
            kl.detach(),
            self.last_lr,
            1.0,
            mu.detach(),
            sigma.detach(),
            bounds_loss.detach(),
        )

    def _reset_optimization_metrics(self) -> None:
        'Reset optimization metrics.'

        self._optimization_count.zero_()
        for total in self._optimization_sums.values():
            total.zero_()
        self._optimizer_step_count = 0
        self._optimizer_microbatch_count = 0
        self._gradient_microbatch_index = 0
        self.optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        for name in self._optimizer_scalar_sums:
            self._optimizer_scalar_sums[name].zero_()
        self._gradient_probe_per_asset = None
        self._gradient_probe_global = None
        if bool(getattr(self, "student_mode", False)):
            self._student_pending_anchor_loss.zero_()
            self._student_pending_fk_loss.zero_()

    def _drain_optimizer_scalars(self) -> dict[str, float]:
        'Handle drain optimizer scalars.'

        return diagnostics.drain_optimizer_scalars(self)

    def _drain_optimization_metrics(self) -> dict[str, torch.Tensor]:
        'Handle drain optimization metrics.'

        return diagnostics.drain_optimization_metrics(self)

    @staticmethod
    def _mean_fields(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, float]:
        'Handle mean fields.'

        return diagnostics.mean_fields(rows, fields)

    def _record_update_metrics(self, epoch_result: tuple[Any, ...]) -> None:
        'Record update metrics.'

        return diagnostics.record_update_metrics(self, epoch_result)

    def train_epoch(self):
        'Train epoch.'

        if hasattr(self, "_resume_last_mean_rewards"):
            self.last_mean_rewards = float(self._resume_last_mean_rewards)
            del self._resume_last_mean_rewards
        self._reset_optimization_metrics()
        self._begin_student_rollout()
        try:
            result = super().train_epoch()
            self._finish_student_rollout(success=True)
            if self.config.get("release_rollout_batch", False):
                self.dataset.update_values_dict(None)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self._record_update_metrics(result)
            return result
        except BaseException:
            self._finish_student_rollout(success=False)
            raise

    def train(self):
        'Train the declared contract.'
        try:
            result = super().train()
        except BaseException:
            self.metrics_recorder.flush(reason="shutdown")
            raise
        else:
            self.metrics_recorder.finalize()
            return result
        finally:
            if self.student_anchor_sampler is not None:
                self.student_anchor_sampler.close()

    def write_stats(
        self,
        total_time,
        epoch_num,
        step_time,
        play_time,
        update_time,
        actor_losses,
        critic_losses,
        entropies,
        kls,
        last_lr,
        lr_mul,
        frame,
        scaled_time,
        scaled_play_time,
        curr_frames,
    ) -> None:
        'Write stats.'

        super().write_stats(
            total_time,
            epoch_num,
            step_time,
            play_time,
            update_time,
            actor_losses,
            critic_losses,
            entropies,
            kls,
            last_lr,
            lr_mul,
            frame,
            scaled_time,
            scaled_play_time,
            curr_frames,
        )
        cadence = int(self.config.get("evaluation_frequency", 320))
        if cadence > 0 and int(epoch_num) % cadence == 0:
            path = f"{self.nn_dir}/evaluation_{self.config['name']}_ep_{int(epoch_num):05d}"
            self.save(path)  # full identity/model/dual-optimizer/curriculum/diagnostic state

    def get_full_state_weights(self) -> dict[str, Any]:
        'Return full state weights.'

        self.metrics_recorder.flush(reason="checkpoint")
        wrapper = getattr(self.vec_env, "env", None)
        evidence = getattr(wrapper, "training_evidence", None)
        if evidence is not None:
            evidence.drain(force=True)
        state = super().get_full_state_weights()
        state[CRITIC_OPTIMIZER_KEY] = self.critic_optimizer.state_dict()
        network = self.model.a2c_network
        state[OPTIMIZER_PARAMETER_NAMES_KEY] = {
            "optimizer": optimizer_parameter_names(self.optimizer, dict(network.package.actor.named_parameters())),
            CRITIC_OPTIMIZER_KEY: optimizer_parameter_names(self.critic_optimizer, dict(network.package.critic.named_parameters())),
        }
        if hasattr(self, "_named_initialization_audit"):
            state["anymani_named_initialization_audit"] = self._named_initialization_audit
        if hasattr(self, "_phase_transport_witness"):
            state["anymani_phase_transport_witness"] = self._phase_transport_witness
        state[DIAGNOSTICS_RECORDER_KEY] = self.metrics_recorder.state_dict()  # shard inventory/append cursor
        state[TRAINING_CONTINUATION_KEY] = {
            "schema_version": "1.0.0",
            "last_lr": float(self.last_lr),
            "entropy_coef": float(self.entropy_coef),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        if self.student_mode:
            assert self.student_warmup is not None and self.student_stats is not None
            sampler = self._ensure_student_anchor_sampler()
            self._update_student_initialization_metadata()
            state["anymani_student_continuation"] = {
                "schema_version": "1.0.0",
                "rollout_phase": self.student_warmup.phase,
                "warmup": self.student_warmup.metadata(),
                "stats": self.student_stats.state_dict(),
                "anchor_dataset_paths": [str(path) for path in self._student_anchor_dataset_paths],
                "anchor_source_hashes": dict(
                    self.student_anchor_sampler.source_hashes
                    if self.student_anchor_sampler is not None
                    else self._student_anchor_source_hashes
                ),
                "anchor_source_data_hash": sampler.source_data_hash,
                "anchor_microbatch_size": self._student_anchor_microbatch_size,
                "anchor_sampler": (
                    sampler.state_dict()
                ),
                "initial_actor_checkpoint_sha256": self.model.a2c_network.package.initialization_metadata.get(
                    "student_checkpoint_sha256"
                ),
                "initial_actor_checkpoint_path": self.model.a2c_network.package.initialization_metadata.get(
                    "student_checkpoint"
                ),
                "auxiliary": self.model.a2c_network.package.initialization_metadata.get("auxiliary", {}),
            }
        return state

    def save(self, filename: str) -> None:
        'Save the declared contract.'

        destination = Path(filename if filename.endswith(".pth") else f"{filename}.pth")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")  # `<name>.pth.tmp`
        if temporary.exists():
            temporary.unlink()
        state = self.get_full_state_weights()
        torch.save(state, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(destination)

    def set_full_state_weights(self, weights: dict[str, Any], set_epoch: bool = True) -> None:
        'Handle set full state weights.'

        if CRITIC_OPTIMIZER_KEY not in weights:
            raise RuntimeError("palm-rotation checkpoint is missing independent critic optimizer state")
        if DIAGNOSTICS_RECORDER_KEY not in weights:
            raise RuntimeError("palm-rotation checkpoint is missing metrics recorder state")
        continuation = weights.get(TRAINING_CONTINUATION_KEY)
        if not isinstance(continuation, Mapping) or continuation.get("schema_version") != "1.0.0":
            raise RuntimeError("palm-rotation checkpoint is missing exact training continuation state")
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        if "anymani_named_initialization_audit" in weights:
            self._named_initialization_audit = dict(weights["anymani_named_initialization_audit"])
        if "anymani_phase_transport_witness" in weights:
            self._phase_transport_witness = dict(weights["anymani_phase_transport_witness"])
        self.critic_optimizer.load_state_dict(weights[CRITIC_OPTIMIZER_KEY])
        self.metrics_recorder.load_state_dict(weights[DIAGNOSTICS_RECORDER_KEY])
        self.last_lr = float(continuation["last_lr"])
        self.entropy_coef = float(continuation["entropy_coef"])
        self.update_lr(self.last_lr)
        random.setstate(continuation["python_random_state"])
        np.random.set_state(continuation["numpy_random_state"])
        torch.set_rng_state(continuation["torch_cpu_rng_state"])
        cuda_states = continuation.get("torch_cuda_rng_states", [])
        if torch.cuda.is_available() and cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        if self.student_mode:
            assert self.student_warmup is not None and self.student_stats is not None
            continuation_student = weights.get("anymani_student_continuation")
            if not isinstance(continuation_student, Mapping) or continuation_student.get("schema_version") != "1.0.0":
                raise RuntimeError("student PPO checkpoint is missing student continuation state")
            warmup = continuation_student.get("warmup")
            stats = continuation_student.get("stats")
            if not isinstance(warmup, Mapping) or not isinstance(stats, Mapping):
                raise RuntimeError("student PPO checkpoint continuation lacks warmup/stats metadata")
            self.student_warmup.load_metadata(warmup)
            if continuation_student.get("rollout_phase") != self.student_warmup.phase:
                raise RuntimeError("student PPO checkpoint rollout phase disagrees with warmup counters")
            self.student_stats = FamilyStudentRlStats.from_state_dict(stats)
            if (
                self.student_stats.rollout_updates != self.student_warmup.state.rollout_updates
                or self.student_stats.critic_optimizer_steps != self.student_warmup.state.critic_optimizer_steps
                or self.student_stats.actor_optimizer_steps != self.student_warmup.state.actor_optimizer_steps
                or self.student_stats.warmup_rollouts != self.student_warmup.state.warmup_rollouts
                or self.student_stats.optimizer_steps_per_rollout
                != self.student_warmup.state.optimizer_steps_per_rollout
            ):
                raise RuntimeError("student PPO checkpoint warmup and stats counters disagree")
            self.model.a2c_network.student_stats = self.student_stats
            current_auxiliary = self.model.a2c_network.package.initialization_metadata.get("auxiliary", {})
            if continuation_student.get("auxiliary", {}) != current_auxiliary:
                raise RuntimeError("student PPO checkpoint auxiliary configuration disagrees")
            current_initial_sha = self.model.a2c_network.package.initialization_metadata.get(
                "student_checkpoint_sha256"
            )
            if continuation_student.get("initial_actor_checkpoint_sha256") != current_initial_sha:
                raise RuntimeError("student PPO checkpoint initial Actor SHA disagrees")
            current_initial_path = self.model.a2c_network.package.initialization_metadata.get("student_checkpoint")
            if continuation_student.get("initial_actor_checkpoint_path") != current_initial_path:
                raise RuntimeError("student PPO checkpoint initial Actor path disagrees")
            saved_paths = continuation_student.get("anchor_dataset_paths", ())
            if not isinstance(saved_paths, (tuple, list)) or not saved_paths:
                raise RuntimeError("student PPO checkpoint anchor dataset paths are malformed")
            if tuple(str(path) for path in saved_paths) != tuple(str(path) for path in self._student_anchor_dataset_paths):
                raise RuntimeError("student PPO checkpoint anchor dataset paths disagree")
            saved_hashes = continuation_student.get("anchor_source_hashes", {})
            sampler_state = continuation_student.get("anchor_sampler")
            if not isinstance(saved_hashes, Mapping) or not saved_hashes:
                raise RuntimeError("student PPO checkpoint anchor source hashes are missing")
            if not isinstance(sampler_state, Mapping):
                raise RuntimeError("student PPO checkpoint anchor sampler state is missing")
            sampler = self._ensure_student_anchor_sampler()
            current_hashes = sampler.source_hashes
            if continuation_student.get("anchor_source_data_hash") != sampler.source_data_hash:
                raise RuntimeError("student PPO checkpoint anchor source data identity disagrees")
            if int(continuation_student.get("anchor_microbatch_size", -1)) != self._student_anchor_microbatch_size:
                raise RuntimeError("student PPO checkpoint anchor microbatch size disagrees")
            sampler.load_state_dict(sampler_state)
            if dict(saved_hashes) != current_hashes:
                raise RuntimeError("student PPO checkpoint anchor source hashes disagree")
            self._update_student_initialization_metadata()
        self._resume_last_mean_rewards = float(weights.get("last_mean_rewards", -1.0e9))


class PalmRotationPpoRunner(AnyManiMaskedRunner):
    'Contract for PALM rotation PPO runner.'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        'Initialize the instance.'

        super().__init__(*args, **kwargs)
        self.algo_factory.register_builder(
            PALM_ROTATION_PPO_ALGO,
            lambda **factory_kwargs: PalmRotationPpoAgent(**factory_kwargs),
        )
        self.player_factory.register_builder(
            PALM_ROTATION_PPO_ALGO,
            lambda **factory_kwargs: AnyManiMaskedPpoPlayer(**factory_kwargs),
        )


def register_palm_rotation_ppo() -> None:
    'Handle register PALM rotation PPO.'

    register_anymani_masked_ppo()
    model_builder.register_model("anymani_palm_rotation_masked_continuous", PalmRotationMaskedContinuousModel)
    model_builder.register_network(PALM_ROTATION_NETWORK, PalmRotationRlGamesBuilder)


__all__ = [
    "CRITIC_OPTIMIZER_KEY",
    "DIAGNOSTICS_RECORDER_KEY",
    "PALM_ROTATION_NETWORK",
    "PALM_ROTATION_PPO_ALGO",
    "TRAINING_CONTINUATION_KEY",
    "bounded_adaptive_learning_rate",
    "denormalize_value_readonly",
    "normalize_advantages_per_asset",
    "validate_gradient_probe_compile_compatibility",
    "PalmRotationPpoAgent",
    "PalmRotationPpoRunner",
    "PalmRotationMaskedContinuousModel",
    "PalmRotationRlGamesBuilder",
    "PalmRotationRlGamesNetwork",
    "register_palm_rotation_ppo",
    "stratified_asset_permutation",
]
