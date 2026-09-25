'Connect frozen family-student policies and their kinematic auxiliary targets to PPO.'

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from anymani.distill.models.family_rotation_student_actor_critic import (
    FamilyStudentActorCriticOutput,
    FamilyStudentRlConfig,
)
from anymani.distill.representations.sources.joint_frames import JointKinematicsBank, build_joint_kinematics_bank

from .student_anchor import (
    FamilyStudentAnchorBatch,
    FamilyStudentAnchorSampler,
    _counter,
    resolve_family_student_anchor_paths,
    resolve_family_student_anchor_source_hashes,
)


class JointOriginTargetProvider(Protocol):
    'Contract for JOINT origin target provider; shapes [B,16,3].'

    def __call__(
        self,
        q_rad: torch.Tensor,
        asset_index: torch.Tensor,
        joint_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        'Return joint-origin targets in metres.'

        ...


def build_joint_origin_target(
    bank: JointKinematicsBank,
    q_rad: torch.Tensor,
    asset_index: torch.Tensor,
) -> torch.Tensor:
    'Build JOINT origin target; shapes [B,16,3].'

    if not isinstance(bank, JointKinematicsBank):
        raise TypeError("joint-origin target requires JointKinematicsBank parsed from typed semantics")
    if q_rad.ndim != 2 or tuple(q_rad.shape[1:]) != (16,):
        raise ValueError(f"q_rad must have shape [B,16], got {tuple(q_rad.shape)}")
    if asset_index.shape != (q_rad.shape[0],) or asset_index.dtype != torch.long:
        raise ValueError("asset_index must be int64 with one entry per q batch row")
    if q_rad.device != bank.features.device or asset_index.device != q_rad.device:
        raise ValueError("q_rad, asset_index and kinematics bank must share a device")

    origins = bank.joint_origins(q_rad.to(dtype=bank.features.dtype), asset_index)
    return origins.to(dtype=q_rad.dtype)


def binding_joint_kinematics_bank(
    binding: Any,
    *,
    device: torch.device | str = "cpu",
) -> JointKinematicsBank:
    'Handle binding JOINT kinematics bank; shapes [A,16,15].'

    source_assets = getattr(binding, "source_assets", None)
    canonical_artifacts = getattr(binding, "canonical_artifacts", None)
    if not isinstance(source_assets, Sequence) or not isinstance(canonical_artifacts, Sequence):
        raise ValueError("student kinematics binding must expose source_assets/canonical_artifacts sequences")
    if not source_assets or len(source_assets) != len(canonical_artifacts):
        raise ValueError("student kinematics source and canonical axes must be nonempty and aligned")


    from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1

    slot_by_name = {name: index for index, name in enumerate(CANONICAL_HAND_SCHEMA_V1.joint_names)}
    semantics: list[Any] = []
    mappings: list[dict[str, int]] = []
    for asset_index, (source, artifact) in enumerate(zip(source_assets, canonical_artifacts, strict=True)):
        semantic = getattr(source, "geometry_semantics", None)
        routing = getattr(artifact, "routing", None)
        pairs = getattr(routing, "source_to_canonical", None)
        if semantic is None or not isinstance(pairs, Sequence):
            raise ValueError(f"binding asset {asset_index} lacks typed semantics/source_to_canonical routing")
        mapping: dict[str, int] = {}
        for pair in pairs:
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(f"binding asset {asset_index} has malformed source_to_canonical pair")
            source_name, canonical_name = pair
            if not isinstance(source_name, str) or not isinstance(canonical_name, str):
                raise ValueError("source_to_canonical names must be strings")
            if canonical_name not in slot_by_name:
                raise ValueError(f"canonical joint name {canonical_name!r} is absent from schema v1")
            if source_name in mapping or slot_by_name[canonical_name] in mapping.values():
                raise ValueError(f"binding asset {asset_index} repeats a joint routing slot")
            mapping[source_name] = slot_by_name[canonical_name]
        semantics.append(semantic)
        mappings.append(mapping)


    try:
        bank64 = build_joint_kinematics_bank(semantics, mappings, joint_count=16, dtype=torch.float64)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"cannot build student joint kinematics bank: {error}") from error
    expected = (len(source_assets), 16, 15)
    if tuple(bank64.features.shape) != expected:
        raise ValueError(f"student kinematics bank shape {tuple(bank64.features.shape)} != expected {expected}")
    if not bool(torch.isfinite(bank64.features).all().item()):
        raise ValueError("student kinematics bank contains non-finite parsed features")
    return bank64.to(device)


@dataclass
class FamilyStudentWarmupState:
    'Contract for family student warmup state; units M.'

    rollout_updates: int = 0
    critic_optimizer_steps: int = 0
    actor_optimizer_steps: int = 0
    warmup_rollouts: int = 20
    optimizer_steps_per_rollout: int = 20

    def __post_init__(self) -> None:
        'Validate the declared contract.'

        values = (
            self.rollout_updates,
            self.critic_optimizer_steps,
            self.actor_optimizer_steps,
            self.warmup_rollouts,
            self.optimizer_steps_per_rollout,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("student warmup counters must be non-negative integers")
        if self.warmup_rollouts < 1 or self.optimizer_steps_per_rollout < 1:
            raise ValueError("student warmup rollout/optimizer budgets must be positive")
        if self.actor_optimizer_steps > self.critic_optimizer_steps:
            raise ValueError("student Actor optimizer steps cannot exceed Critic optimizer steps")

    @property
    def actor_updates_enabled(self) -> bool:
        'Handle Actor updates enabled.'

        return self.rollout_updates >= self.warmup_rollouts

    @property
    def critic_updates(self) -> int:
        'Handle Critic updates.'

        return self.critic_optimizer_steps

    @property
    def actor_updates(self) -> int:
        'Handle Actor updates.'

        return self.actor_optimizer_steps

    @property
    def critic_warmup_updates(self) -> int:
        'Handle Critic warmup updates.'

        return self.warmup_rollouts

    @property
    def warmup_critic_optimizer_steps(self) -> int:
        'Handle warmup Critic optimizer steps.'

        return self.warmup_rollouts * self.optimizer_steps_per_rollout

    @property
    def critic_warmup_optimizer_steps(self) -> int:
        'Handle Critic warmup optimizer steps.'

        return self.warmup_critic_optimizer_steps

    @property
    def phase(self) -> str:
        'Handle phase.'

        return "actor_critic" if self.actor_updates_enabled else "critic_only_warmup"

    def record_critic_update(self, count: int = 1) -> None:
        'Record Critic update.'

        if count < 1:
            raise ValueError("critic update count must be positive")
        self.critic_optimizer_steps += int(count)

    def record_actor_update(self, count: int = 1) -> None:
        'Record Actor update.'

        if count < 1:
            raise ValueError("actor update count must be positive")
        if not self.actor_updates_enabled:
            raise RuntimeError(
                f"student Actor update is disabled during Critic warmup rollout "
                f"({self.rollout_updates}/{self.warmup_rollouts})"
            )
        self.actor_optimizer_steps += int(count)

    def record_rollout_update(self) -> None:
        'Record rollout update.'

        if self.critic_updates <= 0 or self.critic_updates % self.optimizer_steps_per_rollout:
            raise RuntimeError("student rollout completion requires a whole logical Critic optimizer budget")
        next_rollout = self.rollout_updates + 1
        if self.critic_updates != next_rollout * self.optimizer_steps_per_rollout:
            raise RuntimeError("student rollout Critic steps disagree with completed rollout count")
        expected_actor = max(next_rollout - self.warmup_rollouts, 0) * self.optimizer_steps_per_rollout
        if self.actor_updates != expected_actor:
            raise RuntimeError("student rollout Actor steps disagree with completed rollout phase")
        self.rollout_updates += 1

    def expected_phase_for_next_rollout(self) -> str:
        'Handle expected phase for next rollout.'

        return "actor_critic" if self.rollout_updates >= self.warmup_rollouts else "critic_only_warmup"

    def state_dict(self) -> dict[str, object]:
        'Handle state dict.'

        return {
            "rollout_updates": self.rollout_updates,
            "critic_optimizer_steps": self.critic_optimizer_steps,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "warmup_rollouts": self.warmup_rollouts,
            "optimizer_steps_per_rollout": self.optimizer_steps_per_rollout,
            "warmup_critic_optimizer_steps": self.warmup_critic_optimizer_steps,

            "critic_updates": self.critic_optimizer_steps,
            "actor_updates": self.actor_optimizer_steps,
            "critic_warmup_updates": self.warmup_rollouts,
            "phase": self.phase,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> FamilyStudentWarmupState:
        'Handle from state dict.'

        required = {"phase"}
        missing = required - set(state)
        if missing:
            raise ValueError(f"student warmup state misses keys {sorted(missing)}")
        has_new = {
            "rollout_updates",
            "critic_optimizer_steps",
            "actor_optimizer_steps",
            "warmup_rollouts",
            "optimizer_steps_per_rollout",
        }.issubset(state)
        if has_new:
            values = {
                "rollout_updates": _counter(state["rollout_updates"], "rollout_updates"),
                "critic_optimizer_steps": _counter(state["critic_optimizer_steps"], "critic_optimizer_steps"),
                "actor_optimizer_steps": _counter(state["actor_optimizer_steps"], "actor_optimizer_steps"),
                "warmup_rollouts": _counter(state["warmup_rollouts"], "warmup_rollouts"),
                "optimizer_steps_per_rollout": _counter(
                    state["optimizer_steps_per_rollout"], "optimizer_steps_per_rollout"
                ),
            }
        elif {"critic_updates", "actor_updates", "critic_warmup_updates"}.issubset(state):


            values = {
                "rollout_updates": _counter(state["critic_updates"], "critic_updates"),
                "critic_optimizer_steps": _counter(state["critic_updates"], "critic_updates"),
                "actor_optimizer_steps": _counter(state["actor_updates"], "actor_updates"),
                "warmup_rollouts": _counter(state["critic_warmup_updates"], "critic_warmup_updates"),
                "optimizer_steps_per_rollout": 1,
            }
        else:
            raise ValueError("student warmup state misses dual rollout/optimizer counter keys")
        result = cls(
            **values,
        )
        if has_new:
            expected_critic_steps = result.rollout_updates * result.optimizer_steps_per_rollout
            expected_actor_steps = max(result.rollout_updates - result.warmup_rollouts, 0) * result.optimizer_steps_per_rollout
            if result.critic_optimizer_steps != expected_critic_steps:
                raise ValueError("student warmup Critic optimizer counter disagrees with completed rollout count")
            if result.actor_optimizer_steps != expected_actor_steps:
                raise ValueError("student warmup Actor optimizer counter disagrees with completed rollout count")
        if state["phase"] != result.phase:
            raise ValueError("student warmup state phase disagrees with update counters")
        return result


class FamilyStudentWarmupController:
    'Contract for family student warmup controller.'

    def __init__(
        self,
        warmup_rollouts: int = 20,
        *,
        optimizer_steps_per_rollout: int = 20,
        warmup_updates: int | None = None,
    ) -> None:
        'Initialize the instance.'

        if warmup_updates is not None:
            if warmup_rollouts != 20 and warmup_rollouts != warmup_updates:
                raise ValueError("warmup_rollouts and legacy warmup_updates disagree")
            warmup_rollouts = warmup_updates
        self.state = FamilyStudentWarmupState(
            warmup_rollouts=warmup_rollouts,
            optimizer_steps_per_rollout=optimizer_steps_per_rollout,
        )
        self._active_phase: str | None = None
        self._active_critic_steps = 0
        self._active_actor_steps = 0

    @property
    def critic_only(self) -> bool:
        'Handle Critic only.'

        return self.phase == "critic_only_warmup"

    @property
    def phase(self) -> str:
        'Handle phase.'

        return self._active_phase or self.state.expected_phase_for_next_rollout()

    def begin_rollout(self, *, optimizer_steps_per_rollout: int | None = None) -> str:
        'Begin rollout.'

        if self._active_phase is not None:
            raise RuntimeError("student warmup rollout is already active")
        if optimizer_steps_per_rollout is not None:
            if optimizer_steps_per_rollout < 1:
                raise ValueError("optimizer_steps_per_rollout must be positive")
            if optimizer_steps_per_rollout != self.state.optimizer_steps_per_rollout:
                if self.state.rollout_updates != 0:
                    raise ValueError("optimizer_steps_per_rollout cannot change after student training starts")
                self.state.optimizer_steps_per_rollout = int(optimizer_steps_per_rollout)
        self._active_phase = self.state.expected_phase_for_next_rollout()
        self._active_critic_steps = 0
        self._active_actor_steps = 0
        return self._active_phase

    def on_critic_step(self, count: int = 1) -> None:
        'Handle on Critic step.'

        if self._active_phase is None:
            raise RuntimeError("student Critic step requires begin_rollout")
        self.state.record_critic_update(count)
        self._active_critic_steps += int(count)

    def on_critic_optimizer_step(self, count: int = 1) -> None:
        'Handle on Critic optimizer step.'

        self.on_critic_step(count)

    def on_actor_step(self, count: int = 1) -> None:
        'Handle on Actor step.'

        if self._active_phase is None:
            raise RuntimeError("student Actor step requires begin_rollout")
        if self._active_phase != "actor_critic":
            raise RuntimeError("student Actor step is forbidden in this rollout's Critic-only phase")
        self.state.record_actor_update(count)
        self._active_actor_steps += int(count)

    def on_actor_optimizer_step(self, count: int = 1) -> None:
        'Handle on Actor optimizer step.'

        self.on_actor_step(count)

    def finish_rollout(self) -> None:
        'Finish rollout.'

        if self._active_phase is None:
            raise RuntimeError("student warmup rollout is not active")
        expected = self.state.optimizer_steps_per_rollout
        if self._active_critic_steps != expected:
            raise RuntimeError(
                f"student rollout Critic steps {self._active_critic_steps} != expected {expected}"
            )
        expected_actor = 0 if self._active_phase == "critic_only_warmup" else expected
        if self._active_actor_steps != expected_actor:
            raise RuntimeError(
                f"student rollout Actor steps {self._active_actor_steps} != expected {expected_actor}"
            )
        self.state.record_rollout_update()
        self._active_phase = None
        self._active_critic_steps = 0
        self._active_actor_steps = 0

    def abort_rollout(self) -> None:
        'Handle abort rollout.'

        self._active_phase = None
        self._active_critic_steps = 0
        self._active_actor_steps = 0

    def metadata(self) -> dict[str, object]:
        'Handle metadata.'

        return {
            "hook": "family_student_critic_warmup_v2",
            "phase_lock": self.phase,
            **self.state.state_dict(),
        }

    def load_metadata(self, metadata: Mapping[str, object]) -> None:
        'Load metadata.'

        if metadata.get("hook") not in {"family_student_critic_warmup_v1", "family_student_critic_warmup_v2"}:
            raise ValueError("student warmup hook metadata identity mismatch")
        saved_warmup = metadata.get("warmup_rollouts")
        saved_steps = metadata.get("optimizer_steps_per_rollout")
        if saved_warmup is not None and _counter(saved_warmup, "warmup_rollouts") != self.state.warmup_rollouts:
            raise ValueError("student warmup rollout budget disagrees with the instantiated candidate")
        if saved_steps is not None and _counter(saved_steps, "optimizer_steps_per_rollout") != self.state.optimizer_steps_per_rollout:
            raise ValueError("student optimizer steps per rollout disagrees with the instantiated candidate")
        self.state = FamilyStudentWarmupState.from_state_dict(metadata)
        if metadata.get("phase_lock", self.state.phase) != self.state.phase:
            raise ValueError("student warmup metadata phase lock disagrees with counters")


@dataclass
class FamilyStudentRlStats:
    'Contract for family student RL stats.'

    rollout_updates: int = 0
    critic_optimizer_steps: int = 0
    actor_optimizer_steps: int = 0
    warmup_rollouts: int = 20
    optimizer_steps_per_rollout: int = 20
    anchor_loss_sum: float = 0.0
    fk_loss_sum: float = 0.0

    def __post_init__(self) -> None:
        'Validate the declared contract.'

        integer_values = (
            self.rollout_updates,
            self.critic_optimizer_steps,
            self.actor_optimizer_steps,
            self.warmup_rollouts,
            self.optimizer_steps_per_rollout,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integer_values):
            raise ValueError("student RL stats counters must be non-negative integers")
        if self.warmup_rollouts < 1 or self.optimizer_steps_per_rollout < 1:
            raise ValueError("student RL stats warmup/optimizer budget must be positive")
        if self.actor_optimizer_steps > self.critic_optimizer_steps:
            raise ValueError("student Actor optimizer steps cannot exceed Critic optimizer steps")
        if not math.isfinite(self.anchor_loss_sum) or not math.isfinite(self.fk_loss_sum):
            raise ValueError("student RL auxiliary loss sums must be finite")

    @property
    def phase(self) -> str:
        'Handle phase.'

        return "actor_critic" if self.rollout_updates >= self.warmup_rollouts else "critic_only_warmup"

    @property
    def critic_updates(self) -> int:
        'Handle Critic updates.'

        return self.critic_optimizer_steps

    @property
    def actor_updates(self) -> int:
        'Handle Actor updates.'

        return self.actor_optimizer_steps

    @property
    def critic_warmup_updates(self) -> int:
        'Handle Critic warmup updates.'

        return self.warmup_rollouts

    @property
    def warmup_critic_optimizer_steps(self) -> int:
        'Handle warmup Critic optimizer steps.'

        return self.warmup_rollouts * self.optimizer_steps_per_rollout

    @property
    def critic_warmup_optimizer_steps(self) -> int:
        'Handle Critic warmup optimizer steps.'

        return self.warmup_critic_optimizer_steps

    def record_critic_update(self, count: int = 1) -> None:
        'Record Critic update.'

        if count < 1:
            raise ValueError("critic update count must be positive")
        self.critic_optimizer_steps += int(count)

    def record_actor_update(self, count: int = 1) -> None:
        'Record Actor update.'

        if count < 1:
            raise ValueError("actor update count must be positive")
        if self.phase != "actor_critic":
            raise RuntimeError("student Actor stats cannot advance before Critic warmup completes")
        self.actor_optimizer_steps += int(count)

    def record_rollout_update(self) -> None:
        'Record rollout update.'

        expected = self.optimizer_steps_per_rollout
        if self.critic_optimizer_steps < expected or self.critic_optimizer_steps % expected:
            raise RuntimeError("student RL stats require whole Critic optimizer budgets per rollout")
        next_rollout = self.rollout_updates + 1
        if self.critic_optimizer_steps != next_rollout * expected:
            raise RuntimeError("student RL stats Critic steps disagree with completed rollout count")
        expected_actor = max(next_rollout - self.warmup_rollouts, 0) * expected
        if self.actor_optimizer_steps != expected_actor:
            raise RuntimeError("student RL stats Actor steps disagree with completed rollout phase")
        self.rollout_updates += 1

    def record_auxiliary(self, *, anchor_loss: float = 0.0, fk_loss: float = 0.0) -> None:
        'Record auxiliary.'

        if not math.isfinite(anchor_loss) or not math.isfinite(fk_loss) or anchor_loss < 0.0 or fk_loss < 0.0:
            raise ValueError("student auxiliary losses must be finite and non-negative")
        self.anchor_loss_sum += float(anchor_loss)
        self.fk_loss_sum += float(fk_loss)

    def state_dict(self) -> dict[str, object]:
        'Handle state dict.'

        return {
            "rollout_updates": self.rollout_updates,
            "critic_optimizer_steps": self.critic_optimizer_steps,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "warmup_rollouts": self.warmup_rollouts,
            "optimizer_steps_per_rollout": self.optimizer_steps_per_rollout,
            "warmup_critic_optimizer_steps": self.warmup_critic_optimizer_steps,

            "critic_warmup_updates": self.warmup_rollouts,
            "critic_updates": self.critic_optimizer_steps,
            "actor_updates": self.actor_optimizer_steps,
            "anchor_loss_sum": self.anchor_loss_sum,
            "fk_loss_sum": self.fk_loss_sum,
            "phase": self.phase,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> FamilyStudentRlStats:
        'Handle from state dict.'

        required = {"anchor_loss_sum", "fk_loss_sum", "phase"}
        missing = required - set(state)
        if missing:
            raise ValueError(f"student RL stats state misses keys {sorted(missing)}")
        if {
            "rollout_updates",
            "critic_optimizer_steps",
            "actor_optimizer_steps",
            "warmup_rollouts",
            "optimizer_steps_per_rollout",
        }.issubset(state):
            has_new = True
            values = {
                "rollout_updates": _counter(state["rollout_updates"], "rollout_updates"),
                "critic_optimizer_steps": _counter(state["critic_optimizer_steps"], "critic_optimizer_steps"),
                "actor_optimizer_steps": _counter(state["actor_optimizer_steps"], "actor_optimizer_steps"),
                "warmup_rollouts": _counter(state["warmup_rollouts"], "warmup_rollouts"),
                "optimizer_steps_per_rollout": _counter(
                    state["optimizer_steps_per_rollout"], "optimizer_steps_per_rollout"
                ),
            }
        elif {"critic_warmup_updates", "critic_updates", "actor_updates"}.issubset(state):
            has_new = False
            values = {
                "rollout_updates": _counter(state["critic_updates"], "critic_updates"),
                "critic_optimizer_steps": _counter(state["critic_updates"], "critic_updates"),
                "actor_optimizer_steps": _counter(state["actor_updates"], "actor_updates"),
                "warmup_rollouts": _counter(state["critic_warmup_updates"], "critic_warmup_updates"),
                "optimizer_steps_per_rollout": 1,
            }
        else:
            raise ValueError("student RL stats state misses dual rollout/optimizer counter keys")
        result = cls(
            **values,
            anchor_loss_sum=float(cast(Any, state["anchor_loss_sum"])),
            fk_loss_sum=float(cast(Any, state["fk_loss_sum"])),
        )
        if has_new:
            expected_critic_steps = result.rollout_updates * result.optimizer_steps_per_rollout
            expected_actor_steps = max(result.rollout_updates - result.warmup_rollouts, 0) * result.optimizer_steps_per_rollout
            if result.critic_optimizer_steps != expected_critic_steps:
                raise ValueError("student RL stats Critic counter disagrees with rollout count")
            if result.actor_optimizer_steps != expected_actor_steps:
                raise ValueError("student RL stats Actor counter disagrees with rollout count")
        if state["phase"] != result.phase:
            raise ValueError("student RL stats phase disagrees with update counters")
        return result


@dataclass(frozen=True)
class FamilyStudentAuxiliaryHooks:
    'Contract for family student auxiliary hooks.'

    anchor_weight: float = 0.1
    fk_weight: float = 0.1
    link_length_m: float = 0.1

    def __post_init__(self) -> None:
        'Validate the declared contract.'

        if not math.isfinite(self.anchor_weight) or self.anchor_weight < 0.0:
            raise ValueError("anchor_weight must be finite and non-negative")
        if not math.isfinite(self.fk_weight) or self.fk_weight < 0.0:
            raise ValueError("fk_weight must be finite and non-negative")
        if self.link_length_m != 0.1:
            raise ValueError("FK auxiliary link_length_m is fixed at 0.1 m")

    @classmethod
    def from_config(cls, config: FamilyStudentRlConfig) -> FamilyStudentAuxiliaryHooks:
        'Handle from config.'

        return cls(anchor_weight=config.anchor_weight, fk_weight=config.fk_weight)

    @staticmethod
    def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        'Handle masked mse.'

        if prediction.shape != target.shape or prediction.shape[:-1] != valid.shape:
            raise ValueError("auxiliary prediction/target/mask shapes must be [B,16,D]/[B,16]")
        if prediction.ndim != 3 or valid.dtype != torch.bool:
            raise ValueError("auxiliary prediction must be rank-3 and valid mask must be bool")
        if target.dtype != prediction.dtype or target.device != prediction.device or valid.device != prediction.device:
            raise ValueError("auxiliary prediction, target and mask must share dtype/device domains")
        weight = valid.unsqueeze(-1).to(dtype=prediction.dtype)
        per_sample = ((prediction - target).square() * weight).sum(dim=(-1, -2))
        per_sample = per_sample / (valid.sum(dim=-1).clamp_min(1).to(dtype=prediction.dtype) * prediction.shape[-1])
        return per_sample.mean()

    def anchor_loss(
        self,
        actor_mean: torch.Tensor,
        il_mean: torch.Tensor,
        joint_valid: torch.Tensor,
    ) -> torch.Tensor:
        'Handle anchor loss.'

        if actor_mean.shape != il_mean.shape or actor_mean.ndim != 2 or joint_valid.shape != actor_mean.shape:
            raise ValueError("anchor mean/mask shapes must be [B,16]")
        return self._masked_mse(actor_mean.unsqueeze(-1), il_mean.unsqueeze(-1), joint_valid)

    def fk_loss(
        self,
        fk_prediction: torch.Tensor | None,
        fk_target: torch.Tensor,
        joint_valid: torch.Tensor,
    ) -> torch.Tensor:
        'Handle FK loss.'

        if fk_prediction is None:
            raise ValueError("FK auxiliary target was supplied but Actor has no fk_prediction head")
        prediction = fk_prediction
        target = fk_target.detach() / self.link_length_m
        return self._masked_mse(prediction, target, joint_valid)

    def compose(
        self,
        base_actor_loss: torch.Tensor,
        output: FamilyStudentActorCriticOutput,
        joint_valid: torch.Tensor,
        *,
        il_mean: torch.Tensor | None = None,
        fk_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        'Handle compose.'

        if base_actor_loss.ndim != 0:
            raise ValueError("base_actor_loss must be a scalar before auxiliary composition")
        zero = base_actor_loss.new_zeros(())
        if fk_target is None:
            fk_target = getattr(output, "fk_target", None)
        anchor = self.anchor_loss(output.actor.mean, il_mean, joint_valid) if il_mean is not None else zero
        fk = self.fk_loss(output.actor.fk_prediction, fk_target, joint_valid) if fk_target is not None else zero
        total = base_actor_loss + self.anchor_weight * anchor + self.fk_weight * fk
        return total, {
            "anchor_loss": anchor,
            "fk_loss": fk,
            "weighted_anchor_loss": self.anchor_weight * anchor,
            "weighted_fk_loss": self.fk_weight * fk,
        }


def student_candidate_config(**overrides: object) -> FamilyStudentRlConfig:
    'Handle student candidate config.'

    aliases = {
        "horizon": "horizon_length",
        "num_minibatches": "minibatches",
        "accumulation_steps": "gradient_accumulation_steps",
        "epochs": "mini_epochs",
        "actor_lr": "actor_learning_rate",
        "critic_lr": "critic_learning_rate",
    }
    overrides = {aliases.get(key, key): value for key, value in overrides.items()}
    allowed = {
        "num_envs",
        "horizon_length",
        "minibatches",
        "gradient_accumulation_steps",
        "mini_epochs",
        "actor_learning_rate",
        "critic_learning_rate",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "grad_norm",
        "entropy_coef",
        "rl_log_std",
        "freeze_log_std",
        "critic_warmup_updates",
        "max_updates",
        "anchor_weight",
        "fk_weight",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise TypeError(f"unknown student RL candidate fields {sorted(unknown)}")
    return FamilyStudentRlConfig(**overrides)  # type: ignore[arg-type]


__all__ = [
    "FamilyStudentAuxiliaryHooks",
    "FamilyStudentAnchorBatch",
    "FamilyStudentAnchorSampler",
    "FamilyStudentRlStats",
    "FamilyStudentWarmupController",
    "FamilyStudentWarmupState",
    "JointOriginTargetProvider",
    "binding_joint_kinematics_bank",
    "build_joint_origin_target",
    "resolve_family_student_anchor_paths",
    "resolve_family_student_anchor_source_hashes",
    "student_candidate_config",
]
