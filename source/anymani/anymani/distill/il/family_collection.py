"""Stream teacher observations, actions and geometry tokens into HDF5. Preserve the actor weights and the before-action alignment of every label."""

from __future__ import annotations
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import numpy as np
import torch
from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1
from anymani.distill.il.family_dataset import FamilyTrajectoryWriter
from anymani.distill.representations.sources.joint_frames import build_joint_kinematics_bank

ACTOR_ABI = {
    "arm": "direct_token",
    "history_encoder": "tcn",
    "history_length": 30,
    "joint_count": 16,
    "owner_count": 21,
    "geometry_width": 128,
    "actor_contact": "tip-only-binary",
    "phase_clock_enabled": False,
    "joint_kinematics_width": 15,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


class FrozenFamilyCollection:
    """Collect aligned teacher inputs and labels without updating the teacher."""

    def __init__(
        self, output: Path, *, family: str, mode: str = "mean", seed: int | None = None, sample_stride: int = 4
    ) -> None:
        if family not in {"leap_right", "allegro_right"}:
            raise ValueError("family collection requires leap_right or allegro_right")
        if mode not in {"mean", "sample"} or (mode == "sample") != (seed is not None):
            raise ValueError("sample mode needs an action seed; mean mode must not specify one")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("action seed must be a nonnegative 63-bit integer")
        if output.exists() or sample_stride < 1:
            raise ValueError("collector needs a new output path and positive sample stride")
        self.output, self.family, self.mode, self.seed = (output, family, mode, seed)
        self.sample_stride = sample_stride
        self.writer: FamilyTrajectoryWriter | None = None
        self.metadata: dict[str, Any] = {}
        self.active: torch.Tensor | None = None
        self.generator: torch.Generator | None = None
        self.prototype_index: torch.Tensor | None = None
        self.kinematics: Any = None
        self._step = 0
        self.executed_steps = 0

    def start(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_identity: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
        binding: Any,
        observation: Mapping[str, torch.Tensor],
        actor: Any,
        cohort_path: Path,
        cohort_members: Sequence[Mapping[str, Any]],
        steps: int,
        replicas: int,
    ) -> None:
        if self.writer is not None:
            raise RuntimeError("collector start may only occur once")
        self.actor = actor
        self.initial_actor_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
        policy, training = (checkpoint_identity["policy"], checkpoint_identity["training"])
        if policy.get("arm") != "direct_token" or policy.get("actor_contact") != "tip-only-binary":
            raise ValueError("collector requires a frozen DirectToken TIP-only teacher")
        if training.get("history_encoder", "tcn") != "tcn" or training.get("phase_period_steps") is not None:
            raise ValueError("collector requires phase-free TCN History30")
        if training.get("sigma_mode", "global") != "global" or training.get("recovery_sigma_floor") is not None:
            raise ValueError("initial family collection requires the original global teacher distribution")
        count = len(binding.source_assets)
        if count != len(cohort_members) or observation["actor_jnt_current"].shape[0] != count * replicas:
            raise ValueError("collector cohort/environment axes disagree")
        expected_group = {"leap_right": "single_palm_leap", "allegro_right": "single_palm_allegro"}[self.family]
        for member, source in zip(cohort_members, binding.source_assets, strict=True):
            if member["provenance"]["group_name"] != expected_group or source.geometry_semantics.handedness != "right":
                raise ValueError("collector source is not the declared pure right-hand family")
        provider = runtime_identity["geometry_provider"]
        retained = provider["retained_artifact"]
        if retained["sha256"] != checkpoint_identity["geometry_provider"]["retained_artifact"]["sha256"]:
            raise ValueError("collector cannot replace the frozen N040 encoder")
        slot_by_name = {name: i for i, name in enumerate(CANONICAL_HAND_SCHEMA_V1.joint_names)}
        mappings = [
            {source: slot_by_name[target] for source, target in a.routing.source_to_canonical}
            for a in binding.canonical_artifacts
        ]
        semantics = [source.geometry_semantics for source in binding.source_assets]
        cpu_bank = build_joint_kinematics_bank(semantics, mappings, dtype=torch.float64)
        device = observation["actor_jnt_current"].device
        self.kinematics = cpu_bank.to(device)
        self.prototype_index = torch.arange(count * replicas, device=device) % count
        self.active = torch.ones(count * replicas, dtype=torch.bool, device=device)
        if self.mode == "sample":
            assert self.seed is not None
            self.generator = torch.Generator(device=device).manual_seed(self.seed)
        env_assets = np.arange(count * replicas, dtype=np.int64) % count
        env_replicas = np.arange(count * replicas, dtype=np.int64) // count
        static = {"joint_kinematics": cpu_bank.features.float().numpy()}
        for name in (
            "actor_jnt_limits",
            "jnt_valid",
            "tip_valid",
            "owner_valid",
            "shortest_path",
            "parent_direction",
            "child_direction",
        ):
            values = _numpy(observation[name])
            if not np.array_equal(values, values[:count][env_assets]):
                raise ValueError(f"static {name} differs across replicas of the same asset")
            static[name] = values[:count].copy()
        if not np.array_equal(static["jnt_valid"].astype(bool), cpu_bank.valid.numpy()):
            raise ValueError("FK static joint mask differs from the actual Actor mask")
        ordered_assets = [
            {
                "asset_index": i,
                "asset_id": source.asset_id,
                "source_urdf_path": str(source.urdf_path),
                "source_urdf_sha256": _sha256(source.urdf_path),
                "source_member_key": binding.source_member_keys[i],
                "canonical_physical_geometry_hash": artifact.physical_geometry_hash,
                "n040_input_fingerprint": provider["physical_geometry_hashes"][i],
                "configuration_domain_hash": artifact.source_content_hash,
                "source_geometry_semantics_hash": source.geometry_semantics.content_hash,
                "base_design_group": member["provenance"]["group_name"] + "/" + member["provenance"]["mother_name"],
                "source_provenance": dict(member["provenance"]),
            }
            for i, (source, artifact, member) in enumerate(
                zip(binding.source_assets, binding.canonical_artifacts, cohort_members, strict=True)
            )
        ]
        self.metadata = {
            "family": self.family,
            "teacher_checkpoint": str(checkpoint_path.resolve()),
            "teacher_checkpoint_sha256": _sha256(checkpoint_path),
            "teacher_method_identity_digest": checkpoint_identity["identity_digest"],
            "runtime_identity_digest": runtime_identity["identity_digest"],
            "cohort_path": str(cohort_path.resolve()),
            "cohort_sha256": _sha256(cohort_path),
            "n040_sha256": retained["sha256"],
            "actor_abi": dict(ACTOR_ABI),
            "ordered_assets": ordered_assets,
            "geometry_cache_dtype": "float32",
            "target": "bounded_teacher_mean_before_action",
            "teacher_actor_freeze_contract": "all parameters and buffers must remain bitwise equal before finalize",
            "fk_target": "current_joint_frame_origin_in_hand_frame_metres",
            "kinematic_length_scale_m": 0.1,
            "fk_compute_dtype": "float64",
            "protocol": {
                "action_mode": self.mode,
                "action_seed": self.seed,
                "steps": steps,
                "replicas": replicas,
                "sample_stride": self.sample_stride,
                "policy_dt_s": 0.05,
                "first_trajectory_only": True,
                "adr_enabled": False,
            },
            "collector_source_sha256": _sha256(Path(__file__)),
        }
        self.writer = FamilyTrajectoryWriter(
            self.output,
            self.metadata,
            steps=steps,
            env_asset_index=env_assets,
            env_replica_index=env_replicas,
            static=static,
            initial_history=_numpy(observation["actor_jnt_history"]),
            sample_stride=self.sample_stride,
        )

    def act(
        self, step: int, observation: Mapping[str, torch.Tensor], mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        if self.writer is None or self.active is None or step != self._step:
            raise RuntimeError("collector action requires a started, sequential rollout")
        action = mean
        if self.mode == "sample":
            from anymani.distill.rl.palm_rotation_ppo import PalmRotationMaskedContinuousModel

            location = PalmRotationMaskedContinuousModel.Network._action_to_latent(mean)
            valid = observation["jnt_valid"].bool()
            sigma = torch.exp(torch.where(valid, log_std.expand_as(mean), torch.zeros_like(mean)))
            sample = torch.normal(location, sigma, generator=self.generator)
            action = torch.tanh(sample) * valid.to(dtype=mean.dtype)
        fk = None
        if step % self.sample_stride == 0:
            fk = _numpy(
                self.kinematics.joint_origins(
                    (observation["actor_jnt_current"][..., 0] * torch.pi).double(), self.prototype_index
                ).float()
            )
        self.writer.append(
            step,
            jnt_current=_numpy(observation["actor_jnt_current"]),
            owner_contact=_numpy(observation["actor_owner_contact"]),
            teacher_mean=_numpy(mean),
            behavior_action=_numpy(action.clamp(-1, 1)),
            geometry_tokens=_numpy(observation["geometry_tokens"]),
            active=_numpy(self.active),
            history=_numpy(observation["actor_jnt_history"]),
            joint_origin_fk=fk,
        )
        self._step += 1
        return action

    def after_step(self, done: torch.Tensor) -> None:
        if self.active is None:
            raise RuntimeError("collector has not started")
        self.active &= ~done.bool()
        self.executed_steps += 1

    def finish(self, **summary: torch.Tensor) -> None:
        if self.writer is None or self.active is None:
            raise RuntimeError("collector has not started")
        current_state = self.actor.state_dict()
        if any((not torch.equal(value, current_state[name]) for name, value in self.initial_actor_state.items())):
            raise RuntimeError("teacher Actor parameters or buffers changed during collection")
        values = {name: _numpy(value) for name, value in summary.items()}
        values["terminated"] = _numpy(~self.active)
        self.writer.finalize(values)
        self.writer.close()
        self.metadata["data_path"] = str(self.output.resolve())
        self.metadata["data_sha256"] = _sha256(self.output)
        self.metadata["teacher_actor_parameters_frozen_verified"] = True

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
