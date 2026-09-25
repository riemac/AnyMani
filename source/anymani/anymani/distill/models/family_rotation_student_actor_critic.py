"""Actor-critic wrapper for the family student. It keeps the actor observation separate from privileged critic state and gives the actor and critic disjoint parameters."""


from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from anymani.distill.il.family_student import build_family_student, load_family_student
from anymani.distill.models.family_rotation_policy import (
    FAMILY_ROTATION_VARIANTS,
    FamilyRotationActorOutput,
    FamilyRotationStudentActor,
    FamilyRotationVariant,
)
from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorObservation,
    PalmRotationCriticObservation,
    PalmRotationGeometry,
    PalmRotationStructuredCritic,
)

FamilyStudentVariant = FamilyRotationVariant


@dataclass(frozen=True)
class FamilyStudentRlConfig:


    num_envs: int = 1024
    horizon_length: int = 30
    minibatches: int = 8
    gradient_accumulation_steps: int = 2
    mini_epochs: int = 5
    actor_learning_rate: float = 1.0e-5
    critic_learning_rate: float = 3.0e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    grad_norm: float = 1.0
    entropy_coef: float = 0.0
    rl_log_std: float = math.log(0.10)
    freeze_log_std: bool = True
    critic_warmup_updates: int = 20
    max_updates: int = 180
    anchor_weight: float = 0.1
    fk_weight: float = 0.1

    def __post_init__(self) -> None:


        integer_fields = {
            "num_envs": self.num_envs,
            "horizon_length": self.horizon_length,
            "minibatches": self.minibatches,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mini_epochs": self.mini_epochs,
            "critic_warmup_updates": self.critic_warmup_updates,
            "max_updates": self.max_updates,
        }
        if any(not isinstance(value, int) or value < 1 for value in integer_fields.values()):
            raise ValueError(f"student RL integer configuration must be positive: {integer_fields}")
        if self.minibatches % self.gradient_accumulation_steps:
            raise ValueError("student RL accumulation steps must divide minibatches")
        finite_fields = {
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "grad_norm": self.grad_norm,
            "entropy_coef": self.entropy_coef,
            "rl_log_std": self.rl_log_std,
            "anchor_weight": self.anchor_weight,
            "fk_weight": self.fk_weight,
        }
        if any(not math.isfinite(value) for value in finite_fields.values()):
            raise ValueError(f"student RL numeric configuration must be finite: {finite_fields}")
        if self.actor_learning_rate != 1.0e-5 or self.critic_learning_rate != 3.0e-4:
            raise ValueError("student RL candidate fixes actor_learning_rate=1e-5 and critic_learning_rate=3e-4")
        if self.gamma <= 0.0 or self.gamma >= 1.0 or self.gae_lambda < 0.0 or self.gae_lambda > 1.0:
            raise ValueError("student RL gamma/gae_lambda lie outside their probability ranges")
        if self.clip_epsilon <= 0.0 or self.grad_norm <= 0.0:
            raise ValueError("student RL clipping and grad_norm must be positive")
        if self.entropy_coef != 0.0 or not self.freeze_log_std:
            raise ValueError("student RL keeps entropy_coef=0 and freezes the explicit RL log_std")
        if self.rl_log_std != math.log(0.10):
            raise ValueError("student RL fixes the frozen exploration scale at log(0.10)")
        if self.anchor_weight < 0.0 or self.fk_weight < 0.0:
            raise ValueError("student RL auxiliary weights must be non-negative")
        if self.max_updates < self.critic_warmup_updates:
            raise ValueError("student RL max_updates must include the complete Critic warmup rollout budget")

    @property
    def rollout_horizon(self) -> int:


        return self.horizon_length

    @property
    def num_minibatches(self) -> int:


        return self.minibatches

    @property
    def epochs(self) -> int:


        return self.mini_epochs

    @property
    def actor_lr(self) -> float:


        return self.actor_learning_rate

    @property
    def critic_lr(self) -> float:


        return self.critic_learning_rate

    @property
    def clip(self) -> float:


        return self.clip_epsilon

    @property
    def accumulation_steps(self) -> int:


        return self.gradient_accumulation_steps

    @property
    def critic_warmup_rollouts(self) -> int:


        return self.critic_warmup_updates

    @property
    def optimizer_steps_per_rollout(self) -> int:


        return self.minibatches // self.gradient_accumulation_steps * self.mini_epochs

    @property
    def warmup_critic_optimizer_steps(self) -> int:


        return self.critic_warmup_rollouts * self.optimizer_steps_per_rollout

    @property
    def actor_rollout_updates(self) -> int:


        return self.max_updates - self.critic_warmup_rollouts

    def as_dict(self) -> dict[str, object]:


        return {
            "num_envs": self.num_envs,
            "horizon_length": self.horizon_length,
            "minibatches": self.minibatches,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mini_epochs": self.mini_epochs,
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "clip_epsilon": float(self.clip_epsilon),
            "grad_norm": float(self.grad_norm),
            "entropy_coef": float(self.entropy_coef),
            "rl_log_std": float(self.rl_log_std),
            "freeze_log_std": bool(self.freeze_log_std),
            "critic_warmup_updates": self.critic_warmup_updates,
            "critic_warmup_rollouts": self.critic_warmup_rollouts,
            "optimizer_steps_per_rollout": self.optimizer_steps_per_rollout,
            "warmup_critic_optimizer_steps": self.warmup_critic_optimizer_steps,
            "actor_rollout_updates": self.actor_rollout_updates,
            "max_updates": self.max_updates,
            "anchor_weight": float(self.anchor_weight),
            "fk_weight": float(self.fk_weight),
        }


@dataclass(frozen=True)
class FamilyStudentActorCriticOutput:


    actor: FamilyRotationActorOutput
    value: torch.Tensor
    fk_target: torch.Tensor | None = None

    @property
    def mean(self) -> torch.Tensor:


        return self.actor.mean

    @property
    def log_std(self) -> torch.Tensor:


        return self.actor.log_std

    @property
    def fk_prediction(self) -> torch.Tensor | None:


        return self.actor.fk_prediction


def _checkpoint_sha256(path: Path) -> str:


    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dedicated_critic_context(seed: int | None) -> Any:


    if seed is None:
        return nullcontext()
    # Critic modules are intentionally initialized on CPU first.  Keeping the fork CPU-only is important:
    # asking ``fork_rng`` to enumerate CUDA devices would create a CUDA context during an otherwise CPU
    # contract test.  The package performs no CUDA-side parameter initialization before the final optional
    # ``to(device)`` call, so CPU RNG isolation is the relevant construction boundary.
    return torch.random.fork_rng(devices=[])


class FamilyRotationStudentActorCritic(nn.Module):


    def __init__(
        self,
        variant: FamilyStudentVariant | None = None,
        *,
        student_checkpoint: str | Path | None = None,
        actor: FamilyRotationStudentActor | None = None,
        critic_seed: int | None = 0,
        rl_config: FamilyStudentRlConfig | None = None,
        expected_n040_sha256: str | None = None,
        expected_dataset_sha256: str | None = None,
        expected_source_sha256: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:


        super().__init__()
        if student_checkpoint is not None and actor is not None:
            raise ValueError("student_checkpoint and actor are mutually exclusive")
        checkpoint_path = Path(student_checkpoint).expanduser().resolve() if student_checkpoint is not None else None
        if checkpoint_path is not None and checkpoint_path.suffix.lower() in {".ts", ".torchscript"}:
            raise ValueError("TorchScript FrozenFamilyStudent wrapper cannot be trained; use policy.pt actor_state_dict")
        if variant is not None and variant not in FAMILY_ROTATION_VARIANTS:
            raise ValueError(f"student variant must be one of {FAMILY_ROTATION_VARIANTS}, got {variant!r}")
        config = rl_config or FamilyStudentRlConfig()


        if checkpoint_path is not None:
            loaded_actor, student_metadata = load_family_student(
                checkpoint_path,
                device="cpu",
                expected_variant=variant,
                expected_n040_sha256=expected_n040_sha256,
                expected_dataset_sha256=expected_dataset_sha256,
                expected_source_sha256=expected_source_sha256,
            )
            actor = loaded_actor
            resolved_variant = loaded_actor.variant
            checkpoint_digest = _checkpoint_sha256(checkpoint_path)
        else:
            resolved_variant = variant or "n040"
            actor = actor or build_family_student(resolved_variant, device="cpu")
            student_metadata = {}
            checkpoint_digest = None
        if not isinstance(actor, FamilyRotationStudentActor):
            raise TypeError("family student RL package requires FamilyRotationStudentActor")
        if actor.variant != resolved_variant:
            raise ValueError(f"student actor variant {actor.variant!r} disagrees with package {resolved_variant!r}")
        if actor.global_log_std.ndim != 0:
            raise ValueError("family student RL currently requires one global scalar log_std")


        anchor_actor = copy.deepcopy(actor)
        anchor_actor.eval()
        for parameter in anchor_actor.parameters():
            parameter.requires_grad_(False)


        il_log_std = float(actor.global_log_std.detach().item())
        with torch.no_grad():
            actor.global_log_std.fill_(config.rl_log_std)
        actor.global_log_std.requires_grad_(False)


        with _dedicated_critic_context(critic_seed):
            if critic_seed is not None:
                torch.manual_seed(int(critic_seed))
            critic = PalmRotationStructuredCritic()

        self.actor = actor
        self.anchor_actor = anchor_actor
        self.critic = critic
        self.variant: FamilyStudentVariant = resolved_variant
        self.rl_config = config
        self.critic_seed = critic_seed
        self.student_checkpoint = str(checkpoint_path) if checkpoint_path is not None else None
        self.student_metadata = dict(student_metadata)
        self.initialization_metadata: dict[str, object] = {
            "artifact_type": "anymani.family_student_rl_initialization",
            "variant": self.variant,
            "student_checkpoint": self.student_checkpoint,
            "student_checkpoint_sha256": checkpoint_digest,
            "anchor_actor": "initial_BC_actor_frozen_for_train_split_anchor_only",
            "il_log_std": il_log_std,
            "rl_log_std": float(config.rl_log_std),
            "rl_sigma": float(math.exp(config.rl_log_std)),
            "log_std_frozen": True,
            "critic_seed": critic_seed,
            "critic_initialization": "dedicated_seed_fork_rng",
            "candidate": config.as_dict(),
            "warmup": {
                "hook": "family_student_critic_warmup_v2",
                "critic_warmup_rollouts": config.critic_warmup_rollouts,
                "critic_warmup_updates": config.critic_warmup_rollouts,
                "optimizer_steps_per_rollout": config.optimizer_steps_per_rollout,
                "warmup_critic_optimizer_steps": config.warmup_critic_optimizer_steps,
                "actor_rollout_updates": config.actor_rollout_updates,
                "rollout_updates": 0,
                "critic_optimizer_steps": 0,
                "actor_optimizer_steps": 0,
                "critic_updates": 0,
                "actor_updates": 0,
                "phase": "critic_only_warmup",
            },
            "auxiliary": {
                "anchor_weight": config.anchor_weight,
                "fk_weight": config.fk_weight,
                "fk_target_source": "raw_q_and_parsed_joint_kinematics",
            },
        }
        if "source_code_hash" in self.student_metadata:
            self.initialization_metadata["student_source_code_hash"] = self.student_metadata["source_code_hash"]


        if device is not None:
            self.to(device=device, dtype=torch.float32)
        self._assert_parameter_disjoint()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        variant: FamilyStudentVariant | None = None,
        critic_seed: int | None = 0,
        rl_config: FamilyStudentRlConfig | None = None,
        expected_n040_sha256: str | None = None,
        expected_dataset_sha256: str | None = None,
        expected_source_sha256: str | None = None,
        device: torch.device | str | None = None,
    ) -> FamilyRotationStudentActorCritic:


        return cls(
            variant,
            student_checkpoint=path,
            critic_seed=critic_seed,
            rl_config=rl_config,
            expected_n040_sha256=expected_n040_sha256,
            expected_dataset_sha256=expected_dataset_sha256,
            expected_source_sha256=expected_source_sha256,
            device=device,
        )

    def _assert_parameter_disjoint(self) -> None:


        actor_parameters = tuple(self.actor.parameters())
        anchor_parameters = tuple(self.anchor_actor.parameters())
        critic_parameters = tuple(self.critic.parameters())
        actor_ids = {id(parameter) for parameter in actor_parameters}
        anchor_ids = {id(parameter) for parameter in anchor_parameters}
        critic_ids = {id(parameter) for parameter in critic_parameters}
        if actor_ids & critic_ids or actor_ids & anchor_ids or critic_ids & anchor_ids:
            raise RuntimeError("family student Actor/Critic/anchor parameter objects overlap")
        actor_storage = {parameter.data_ptr() for parameter in actor_parameters}
        anchor_storage = {parameter.data_ptr() for parameter in anchor_parameters}
        critic_storage = {parameter.data_ptr() for parameter in critic_parameters}
        if actor_storage & critic_storage or actor_storage & anchor_storage or critic_storage & anchor_storage:
            raise RuntimeError("family student Actor/Critic/anchor parameter storage overlaps")

    def trainable_parameter_sets(self) -> tuple[set[int], set[int]]:


        return (
            {id(parameter) for parameter in self.actor.parameters() if parameter.requires_grad},
            {id(parameter) for parameter in self.critic.parameters() if parameter.requires_grad},
        )

    @property
    def student_actor(self) -> FamilyRotationStudentActor:


        return self.actor

    @property
    def structured_critic(self) -> PalmRotationStructuredCritic:


        return self.critic

    @torch.no_grad()
    def initial_bc_mean(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
        joint_kinematics: torch.Tensor,
    ) -> torch.Tensor:


        return self.anchor_actor(observation, geometry, joint_kinematics=joint_kinematics).mean

    def train(self, mode: bool = True) -> FamilyRotationStudentActorCritic:


        super().train(mode)
        self.anchor_actor.eval()
        return self

    def actor_parameters(self) -> list[nn.Parameter]:


        return [parameter for parameter in self.actor.parameters() if parameter.requires_grad]

    def critic_parameters(self) -> list[nn.Parameter]:


        return [parameter for parameter in self.critic.parameters() if parameter.requires_grad]

    def build_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:


        actor_parameters = self.actor_parameters()
        critic_parameters = self.critic_parameters()
        if not actor_parameters or not critic_parameters:
            raise RuntimeError("family student RL optimizers require nonempty Actor and Critic parameter sets")
        actor_ids, critic_ids = self.trainable_parameter_sets()
        if actor_ids & critic_ids:
            raise RuntimeError("family student optimizer parameter sets overlap")
        actor_optimizer = torch.optim.Adam(
            [{"params": actor_parameters, "lr": self.rl_config.actor_learning_rate, "name": "student_actor"}],
            eps=1.0e-8,
        )
        critic_optimizer = torch.optim.Adam(
            [{"params": critic_parameters, "lr": self.rl_config.critic_learning_rate, "name": "student_critic"}],
            eps=1.0e-8,
        )
        return actor_optimizer, critic_optimizer

    def forward(
        self,
        actor_observation: PalmRotationActorObservation,
        critic_observation: PalmRotationCriticObservation,
        geometry: PalmRotationGeometry,
        joint_kinematics: torch.Tensor,
        *,
        fk_target: torch.Tensor | None = None,
    ) -> FamilyStudentActorCriticOutput:


        batch = actor_observation.jnt_current.shape[0]
        if critic_observation.jnt_state.shape[0] != batch or geometry.tokens.shape[0] != batch:
            raise ValueError("family student Actor/Critic/geometry batch sizes disagree")
        expected_kinematics = (batch, 16, 15)
        if tuple(joint_kinematics.shape) != expected_kinematics:
            raise ValueError(f"joint_kinematics must have shape {expected_kinematics}, got {tuple(joint_kinematics.shape)}")
        if joint_kinematics.dtype != actor_observation.jnt_current.dtype:
            raise ValueError("joint_kinematics dtype must match actor observation dtype")
        if joint_kinematics.device != actor_observation.jnt_current.device:
            raise ValueError("joint_kinematics device must match actor observation device")
        if fk_target is not None:
            expected_fk = (batch, 16, 3)
            if tuple(fk_target.shape) != expected_fk:
                raise ValueError(f"fk_target must have shape {expected_fk}, got {tuple(fk_target.shape)}")
            if fk_target.dtype != actor_observation.jnt_current.dtype or fk_target.device != joint_kinematics.device:
                raise ValueError("fk_target dtype/device must match actor observation and joint_kinematics")
            torch._assert_async(  # type: ignore[reportPrivateImportUsage]
                torch.isfinite(fk_target).all(), "fk_target must contain finite parsed FK values"
            )
            fk_target = fk_target.detach()
        actor_output = self.actor(actor_observation, geometry, joint_kinematics=joint_kinematics)
        value = self.critic(critic_observation, geometry)
        torch._assert_async(  # type: ignore[reportPrivateImportUsage]
            torch.isfinite(value).all(), "family student privileged Critic produced non-finite value"
        )
        return FamilyStudentActorCriticOutput(actor=actor_output, value=value, fk_target=fk_target)

    def forward_from_named(
        self,
        observation: Mapping[str, torch.Tensor],
        *,
        fk_target: torch.Tensor | None = None,
    ) -> FamilyStudentActorCriticOutput:


        if fk_target is None and "joint_origin_target" in observation:
            fk_target = observation["joint_origin_target"].float()

        geometry = PalmRotationGeometry(
            tokens=observation["geometry_tokens"].float(),
            owner_valid=observation["owner_valid"].bool(),
            shortest_path=observation["shortest_path"].long(),
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
        return self.forward(
            actor_observation,
            critic_observation,
            geometry,
            observation["joint_kinematics"].float(),
            fk_target=fk_target,
        )


FamilyStudentActorCritic = FamilyRotationStudentActorCritic
FamilyStudentActorCriticPackage = FamilyRotationStudentActorCritic
FamilyRotationStudentActorCriticOutput = FamilyStudentActorCriticOutput


def build_family_student_actor_critic(
    variant: FamilyStudentVariant | None = None,
    *,
    student_checkpoint: str | Path | None = None,
    critic_seed: int | None = 0,
    rl_config: FamilyStudentRlConfig | None = None,
    expected_n040_sha256: str | None = None,
    expected_dataset_sha256: str | None = None,
    expected_source_sha256: str | None = None,
    device: torch.device | str | None = None,
) -> FamilyRotationStudentActorCritic:


    return FamilyRotationStudentActorCritic(
        variant,
        student_checkpoint=student_checkpoint,
        critic_seed=critic_seed,
        rl_config=rl_config,
        expected_n040_sha256=expected_n040_sha256,
        expected_dataset_sha256=expected_dataset_sha256,
        expected_source_sha256=expected_source_sha256,
        device=device,
    )


build_family_student_rl_package = build_family_student_actor_critic


__all__ = [
    "FamilyStudentActorCritic",
    "FamilyStudentActorCriticOutput",
    "FamilyStudentActorCriticPackage",
    "FamilyStudentRlConfig",
    "FamilyStudentVariant",
    "FamilyRotationStudentActorCritic",
    "FamilyRotationStudentActorCriticOutput",
    "build_family_student_actor_critic",
    "build_family_student_rl_package",
]
