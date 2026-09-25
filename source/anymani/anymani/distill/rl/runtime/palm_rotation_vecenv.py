'Define the named environment-to-policy tensor ABI for canonical joints, owners, and rollout state.'

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from gym import spaces
from rl_games.common.vecenv import IVecEnv

from anymani.distill.diagnostics.recording.rl.training_evidence import TrainingEvidence
from anymani.distill.models.palm_rotation_policy import PalmRotationActorObservation
from anymani.tasks.hetero.mdp.curriculum_state import (
    HETERO_REWARD_RELEASE_STATE_ATTR,
    HeterogeneousRewardReleaseState,
)

from .palm_rotation_phase import normalize_phase_period_steps, phase_clock_from_episode_steps

if TYPE_CHECKING:
    from anymani.distill.models.palm_rotation_policy import PalmRotationGeometry

    from .palm_rotation_student import JointOriginTargetProvider


class PalmRotationGeometryProvider(Protocol):
    'Contract for PALM rotation geometry provider.'

    resolve_call_count: int

    def to(self, device: torch.device | str) -> PalmRotationGeometryProvider:
        'Handle to.'

        ...

    def resolve(
        self,
        prototype_index: torch.Tensor,
        actor_observation: PalmRotationActorObservation,
    ) -> PalmRotationGeometry:
        'Resolve the declared contract.'

        ...



PALM_ROTATION_FLOAT_SHAPES: dict[str, tuple[int, ...]] = {
    "actor_jnt_current": (16, 5),
    "actor_jnt_history": (30, 16, 5),  # values 1.5 s, 20 Hz; units s, Hz
    "actor_jnt_limits": (16, 2),
    "actor_owner_contact": (21, 1),  # canonical PALM/JOINT/TIP axis
    "critic_jnt_state": (16, 4),
    "critic_owner_contact": (21, 2),
    "critic_obj": (1, 15),
    "critic_task": (1, 8),  # privileged command/progress state
    "critic_reward_release": (1,),
    "geometry_tokens": (21, 128),
}



PALM_ROTATION_STUDENT_EXTRA_FLOAT_SHAPES: dict[str, tuple[int, ...]] = {
    "joint_kinematics": (16, 15),  # values 0.1m; units m
}


PALM_ROTATION_STUDENT_FLOAT_SHAPES = {
    **PALM_ROTATION_FLOAT_SHAPES,
    **PALM_ROTATION_STUDENT_EXTRA_FLOAT_SHAPES,
}


PALM_ROTATION_FK_TARGET_FLOAT_SHAPES: dict[str, tuple[int, ...]] = {
    "joint_origin_target": (16, 3),  # units m
}


PALM_ROTATION_BOOL_SHAPES: dict[str, tuple[int, ...]] = {
    "jnt_valid": (16,),  # Boolean mask on the canonical JOINT axis.
    "tip_valid": (4,),
    "owner_valid": (21,),  # Boolean mask on the PALM/JOINT/TIP axis.
}


PALM_ROTATION_INT16_SHAPES: dict[str, tuple[int, ...]] = {
    "shortest_path": (21, 21),
    "parent_direction": (21, 21),  # parent-directed relation bucket
    "child_direction": (21, 21),  # child-directed relation bucket
    "prototype_index": (1,),  # Asset index range: [0,79].
}


def palm_rotation_float_shapes(
    phase_period_steps: int | None = None,
    *,
    student: bool = False,
    include_fk_target: bool = False,
) -> dict[str, tuple[int, ...]]:
    'Return named FP32 transport shapes for standard, student, and optional phase-clock observations.'

    shapes = dict(PALM_ROTATION_FLOAT_SHAPES)
    if student:
        shapes.update(PALM_ROTATION_STUDENT_EXTRA_FLOAT_SHAPES)
    if include_fk_target:
        if not student:
            raise ValueError("FK target transport requires student=True")
        shapes.update(PALM_ROTATION_FK_TARGET_FLOAT_SHAPES)
    if normalize_phase_period_steps(phase_period_steps) is not None:
        shapes["phase_clock"] = (2,)
    return shapes


def palm_rotation_student_float_shapes(*, include_fk_target: bool = False) -> dict[str, tuple[int, ...]]:
    'Return the twelve-input student observation shapes and optional FK-target shape.'

    return palm_rotation_float_shapes(student=True, include_fk_target=include_fk_target)


def palm_rotation_observation_space(
    clip_observations: float,
    *,
    phase_period_steps: int | None = None,
    student: bool = False,
    include_fk_target: bool = False,
) -> spaces.Dict:
    'Build a Dict observation space with FP32, boolean, and int16 fields on their named axes.'

    clip = float(clip_observations)
    observation_spaces: dict[str, spaces.Space] = {}


    for name, shape in palm_rotation_float_shapes(
        phase_period_steps,
        student=student,
        include_fk_target=include_fk_target,
    ).items():

        bound = np.inf if name in {"geometry_tokens", "joint_kinematics", "joint_origin_target"} else (
            1.0 if name == "phase_clock" else clip
        )
        observation_spaces[name] = spaces.Box(-bound, bound, shape=shape, dtype=np.float32)


    for name, shape in PALM_ROTATION_BOOL_SHAPES.items():
        observation_spaces[name] = spaces.Box(False, True, shape=shape, dtype=np.bool_)


    for name, shape in PALM_ROTATION_INT16_SHAPES.items():
        observation_spaces[name] = spaces.Box(0, np.iinfo(np.int16).max, shape=shape, dtype=np.int16)
    return spaces.Dict(observation_spaces)


class PalmRotationRlGamesVecEnv(IVecEnv):
    'Transport named task observations, N040 tokens, active masks, and rollout facts to rl_games.'

    def __init__(
        self,
        env: Any,
        *,
        geometry_provider: PalmRotationGeometryProvider,
        prototype_index: torch.Tensor,
        rl_device: torch.device | str,
        clip_observations: float,
        clip_actions: float,
        phase_period_steps: int | None = None,
        student: bool = False,
        student_variant: str | None = None,
        joint_kinematics: torch.Tensor | None = None,
        joint_kinematics_bank: Any | None = None,
        joint_origin_target_provider: JointOriginTargetProvider | None = None,
    ) -> None:
        'Initialize the instance; shapes [A,16,15], [N,16,15].'

        self.env = env
        self._rl_device = torch.device(rl_device)
        self._sim_device = torch.device(env.unwrapped.device)
        self._clip_observations = float(clip_observations)  # raw dynamic observation clip
        self._clip_actions = float(clip_actions)
        self.geometry_provider = geometry_provider.to(self._rl_device)  # FP32 master + scoped BF16 encoder
        self.prototype_index = prototype_index.to(self._rl_device, dtype=torch.long)  # `[N]` fixed routing
        if self.prototype_index.shape != (env.unwrapped.num_envs,):
            raise ValueError("prototype_index must align one-to-one with vectorized environments")
        self.phase_period_steps = normalize_phase_period_steps(phase_period_steps)

        if joint_kinematics is not None and joint_kinematics_bank is not None:
            raise ValueError("joint_kinematics and joint_kinematics_bank are mutually exclusive")
        if joint_kinematics is None and joint_kinematics_bank is not None:
            joint_kinematics = getattr(joint_kinematics_bank, "features", None)
        self.student_mode = bool(student or joint_kinematics is not None)
        self.student_variant = student_variant
        if self.student_mode and joint_kinematics is None:
            raise ValueError("student rollout requires static joint_kinematics with source shape [A,16,15]")
        if student_variant is not None and not self.student_mode:
            raise ValueError("student_variant requires student=True and static joint_kinematics")
        if student_variant is not None and student_variant not in {"n040", "no_z", "fk"}:
            raise ValueError("student_variant must be n040, no_z, or fk")
        if self.student_mode and student_variant == "fk" and joint_origin_target_provider is None:
            raise ValueError("FK student rollout requires an explicit joint_origin_target_provider")
        if joint_origin_target_provider is not None and not self.student_mode:
            raise ValueError("joint_origin_target_provider requires student=True")
        if joint_origin_target_provider is not None and student_variant not in {None, "fk"}:
            raise ValueError("joint_origin_target_provider is only valid for the FK student variant")
        if joint_kinematics is not None:
            if not isinstance(joint_kinematics, torch.Tensor):
                raise TypeError("joint_kinematics must be a torch.Tensor or a bank exposing .features")
            expected = (self._asset_count if hasattr(self, "_asset_count") else int(prototype_index.max().item()) + 1, 16, 15)
            # `_asset_count` is computed below; this early check validates rank/width without duplicating routing.
            if joint_kinematics.ndim != 3 or tuple(joint_kinematics.shape[1:]) != expected[1:]:
                raise ValueError(f"joint_kinematics bank must have shape [A,16,15], got {tuple(joint_kinematics.shape)}")
            if joint_kinematics.dtype not in {torch.float32, torch.float64}:
                raise ValueError(f"joint_kinematics bank must be FP32/FP64 source values, got {joint_kinematics.dtype}")
            if not bool(torch.isfinite(joint_kinematics).all().item()):
                raise ValueError("joint_kinematics bank must contain finite values")
            self.joint_kinematics_bank = joint_kinematics.to(self._rl_device, dtype=torch.float32)
        else:
            self.joint_kinematics_bank = None
        self.joint_origin_target_provider = joint_origin_target_provider
        self.include_fk_target = joint_origin_target_provider is not None
        self._observation_space = palm_rotation_observation_space(
            self._clip_observations,
            phase_period_steps=self.phase_period_steps,
            student=self.student_mode,
            include_fk_target=self.include_fk_target,
        )
        self._asset_count = int(self.prototype_index.max().item()) + 1
        if self.joint_kinematics_bank is not None and self.joint_kinematics_bank.shape[0] != self._asset_count:
            raise ValueError(
                f"joint_kinematics bank asset axis {self.joint_kinematics_bank.shape[0]} "
                f"must equal prototype asset count {self._asset_count}"
            )
        self.training_evidence: TrainingEvidence | None = None
        self._rollout_count = torch.zeros(self._asset_count, device=self._rl_device)  # samples per asset/update
        self._rollout_sums = {
            name: torch.zeros(self._asset_count, device=self._rl_device)
            for name in (
                "reward_mean",
                "goal_count_mean",
                "frontier_count_mean",
                "frontier_pulse_rate",
                "max_positive_net_turns_mean",
                "net_turns_mean",
                "drop_rate",
                "axis_failure_rate",
                "tip_contact_mean",
                "palm_contact_rate",
                "non_tip_contact_rate",
                "action_clamp_fraction",
                "physical_action_rms",
            )
        }
        self._current_joint_valid: torch.Tensor | None = None  # shapes [N,16]
        self._last_action_clamp_fraction = torch.zeros(self.num_envs, device=self._rl_device)  # shapes [N]
        self._last_physical_action_rms = torch.zeros(self.num_envs, device=self._rl_device)  # shapes [N]
        self._terminal_count = torch.zeros(
            self._asset_count, device=self._rl_device
        )  # completed episodes per asset/update
        self._terminal_sums = {
            name: torch.zeros(self._asset_count, device=self._rl_device)
            for name in (
                "terminal_goal_count_mean",
                "terminal_frontier_count_mean",
                "terminal_max_positive_net_turns_mean",
                "terminal_net_turns_mean",
                "terminal_absolute_path_turns_mean",
                "terminal_directional_consistency_mean",
                "terminal_timeout_rate",
                "terminal_drop_rate",
                "terminal_axis_failure_rate",
            )
        }

    @property
    def unwrapped(self) -> Any:
        'Handle unwrapped.'

        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        'Handle num envs.'

        return int(self.unwrapped.num_envs)

    @property
    def observation_space(self) -> spaces.Dict:
        'Handle observation space.'

        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        'Handle action space.'

        shape = tuple(self.unwrapped.single_action_space.shape)  # sample-level canonical action shape
        return spaces.Box(-self._clip_actions, self._clip_actions, shape=shape, dtype=np.float32)

    def get_number_of_agents(self) -> int:
        'Return number of agents.'

        return 1

    def get_env_info(self) -> dict[str, Any]:
        'Return env info.'

        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,  # `[16]` canonical transport
            "state_space": None,
            "value_size": 1,
            "agents": 1,
        }

    def seed(self, seed: int = -1) -> int:
        'Handle seed.'

        return int(self.unwrapped.seed(seed))

    def reset(self) -> dict[str, dict[str, torch.Tensor]]:
        'Reset the declared contract.'

        observation, _ = self.env.reset()
        return {"obs": self._transport(observation)}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, dict[str, Any]]:
        'Handle step; units Hz.'

        if self._current_joint_valid is None:
            raise RuntimeError("palm-rotation action step requires a preceding structured observation")
        sampled_actions = actions.detach().to(self._rl_device, dtype=torch.float32)  # shapes [N,16]
        if sampled_actions.shape != self._current_joint_valid.shape:
            raise RuntimeError("sampled actions and active-joint mask disagree")
        active_float = self._current_joint_valid.to(dtype=sampled_actions.dtype)
        active_count = active_float.sum(dim=-1).clamp_min(1.0)
        clipped_actions = sampled_actions.clamp(-self._clip_actions, self._clip_actions)
        changed = (clipped_actions != sampled_actions).to(dtype=sampled_actions.dtype) * active_float
        self._last_action_clamp_fraction = changed.sum(dim=-1) / active_count
        self._last_physical_action_rms = torch.sqrt(
            (clipped_actions.square() * active_float).sum(dim=-1) / active_count
        )
        physical_actions = clipped_actions.to(self._sim_device)
        observation, reward, terminated, truncated, extras = self.env.step(physical_actions)  # units Hz
        extras = {
            key: value.to(self._rl_device, non_blocking=True) if hasattr(value, "to") else value
            for key, value in extras.items()
        }
        if "log" in extras:
            extras["episode"] = extras.pop("log")
        done = (terminated | truncated).to(self._rl_device)  # Boolean episode-end signal.
        self._record_rollout_step(reward.to(self._rl_device))
        return {"obs": self._transport(observation)}, reward.to(self._rl_device), done, extras

    def close(self) -> None:
        'Close the declared contract.'

        if self.training_evidence is not None:
            self.training_evidence.close()
        self.env.close()

    def configure_training_evidence(self, root: Path, identity_digest: str) -> None:
        'Handle configure training evidence.'
        self.training_evidence = TrainingEvidence(
            root / "episodes",
            identity_digest,
            self.prototype_index,
            self.unwrapped.reward_manager.active_terms,
            policy_dt_s=float(self.unwrapped.step_dt),
            first_window_root=root / "first30",
        )

    def _float(self, value: torch.Tensor, *, clip: bool = True) -> torch.Tensor:
        'Handle float.'

        result = value.to(self._rl_device, dtype=torch.float32)  # raw task tensor -> FP32 policy side
        return result.clamp(-self._clip_observations, self._clip_observations) if clip else result

    def _transport(self, observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        'Handle transport.'

        policy = observation.get("policy")  # actor-only named raw observation
        critic = observation.get("critic")  # privileged named raw observation
        if not isinstance(policy, Mapping) or not isinstance(critic, Mapping):
            raise TypeError("palm-rotation task must expose named policy and critic observation groups")


        joint_valid = policy["jnt_valid"].to(self._rl_device, dtype=torch.bool)  # `[N,16]`
        tip_valid = policy["tip_valid"].to(self._rl_device, dtype=torch.bool)  # `[N,4]`
        owner_valid = policy["owner_valid"].to(self._rl_device, dtype=torch.bool)  # `[N,21]`
        self._current_joint_valid = joint_valid
        for name, expected in (("jnt_valid", joint_valid), ("tip_valid", tip_valid), ("owner_valid", owner_valid)):
            actual = critic[name].to(self._rl_device, dtype=torch.bool)
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.all(actual == expected),
                f"actor/critic {name} masks disagree at rollout transport",
            )


        actor_observation = PalmRotationActorObservation(
            jnt_current=self._float(policy["jnt_current"]),  # `[N,16,5]`
            jnt_history=self._float(policy["jnt_history"]),  # `[N,30,16,5]`
            jnt_limits=self._float(policy["jnt_limits"]),  # `[N,16,2]`
            owner_contact=self._float(policy["owner_contact"]),  # `[N,21,1]` binary
            jnt_valid=joint_valid,
            tip_valid=tip_valid,
            owner_valid=owner_valid,
        )
        geometry = self.geometry_provider.resolve(self.prototype_index, actor_observation)  # one BF16 encoder call


        transported = {
            "actor_jnt_current": actor_observation.jnt_current,
            "actor_jnt_history": actor_observation.jnt_history,
            "actor_jnt_limits": actor_observation.jnt_limits,
            "actor_owner_contact": actor_observation.owner_contact,
            "critic_jnt_state": self._float(critic["jnt_state"]),
            "critic_owner_contact": self._float(critic["owner_contact"]),
            "critic_obj": self._float(critic["obj"]),
            "critic_task": self._float(critic["task"]),
            "critic_reward_release": self._float(critic["reward_release"]),
            "jnt_valid": joint_valid,
            "tip_valid": tip_valid,
            "owner_valid": owner_valid,
            "geometry_tokens": geometry.tokens,  # shapes [N,21,128]
            "shortest_path": geometry.shortest_path.to(torch.int16),
            "parent_direction": geometry.parent_direction.to(torch.int16),
            "child_direction": geometry.child_direction.to(torch.int16),
            "prototype_index": self.prototype_index.to(torch.int16).unsqueeze(-1),  # `[N,1]` sampling label
        }
        if self.student_mode:

            assert self.joint_kinematics_bank is not None
            transported["joint_kinematics"] = self.joint_kinematics_bank[self.prototype_index]  # `[N,16,15]`, FP32
            if self.joint_origin_target_provider is not None:

                q_raw = policy["jnt_current"].to(self._rl_device, dtype=torch.float32)
                q_rad = q_raw[..., 0] * torch.pi  # shapes [N,16]; units radians
                fk_target = self.joint_origin_target_provider(
                    q_rad,
                    self.prototype_index,
                    transported["joint_kinematics"],
                )
                if not isinstance(fk_target, torch.Tensor):
                    raise TypeError("joint_origin_target_provider must return a torch.Tensor")
                expected_fk = (self.num_envs, 16, 3)
                if tuple(fk_target.shape) != expected_fk:
                    raise ValueError(f"joint origin target must have shape {expected_fk}, got {tuple(fk_target.shape)}")
                if fk_target.dtype != torch.float32 or fk_target.device != self._rl_device:
                    raise ValueError("joint origin target must be FP32 on the RL device")
                torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                    torch.isfinite(fk_target).all(),
                    "joint origin target provider returned non-finite values",
                )
                transported["joint_origin_target"] = fk_target.detach()
        if self.phase_period_steps is not None:
            episode_steps = self.unwrapped.episode_length_buf
            if not isinstance(episode_steps, torch.Tensor) or episode_steps.shape != (self.num_envs,):
                raise ValueError("phase clock requires one physical episode counter per environment")
            transported["phase_clock"] = phase_clock_from_episode_steps(
                episode_steps.to(self._rl_device), period_steps=self.phase_period_steps
            )  # Sine and cosine components lie in [-1,1].
        return transported

    def _record_rollout_step(self, reward: torch.Tensor) -> None:
        'Record rollout step.'

        command = self.unwrapped.command_manager.get_term("goal_pose")  # N000 moving-subgoal command
        snapshot = getattr(command, "post_physics_evaluation_snapshot", None)
        if not isinstance(snapshot, dict):
            raise RuntimeError("palm-rotation rollout requires a valid post-physics evaluation snapshot")
        if self.training_evidence is not None:


            weighted = self.unwrapped.reward_manager._step_reward * float(self.unwrapped.step_dt)
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                ((weighted.sum(-1) - reward.reshape(-1)).abs() <= 1e-4 + 1e-5 * reward.reshape(-1).abs()).all(),
                "reward term evidence does not reconstruct environment reward",
            )
            self.training_evidence.capture(snapshot, weighted)
        valid = snapshot.get("valid")
        if not isinstance(valid, torch.Tensor):
            raise RuntimeError("palm-rotation post-physics snapshot lacks a tensor validity certificate")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            valid.all(),
            "palm-rotation rollout requires a valid post-physics evaluation snapshot",
        )
        two_pi = 2.0 * torch.pi  # signed net rad -> physical turns
        values = {
            "reward_mean": reward.reshape(-1).float(),
            "goal_count_mean": snapshot["completed_subgoals"].to(self._rl_device).float(),
            "frontier_count_mean": snapshot["rotation_frontier_count"].to(self._rl_device).float(),
            "frontier_pulse_rate": snapshot["rotation_frontier_pulse"].to(self._rl_device).float(),
            "max_positive_net_turns_mean": (
                snapshot["max_positive_net_rotation_rad"].to(self._rl_device).float() / two_pi
            ),
            "net_turns_mean": snapshot["net_rotation_rad"].to(self._rl_device).float() / two_pi,
            "drop_rate": snapshot["termination_object_out_of_anchor"].to(self._rl_device).float(),
            "axis_failure_rate": snapshot["termination_goal_axis_misaligned"].to(self._rl_device).float(),
            "tip_contact_mean": snapshot["tip_active_count"].to(self._rl_device).float(),
            "palm_contact_rate": snapshot["palm_contact"].to(self._rl_device).float(),
            "non_tip_contact_rate": snapshot["finger_non_tip_contact"].to(self._rl_device).float(),
            "action_clamp_fraction": self._last_action_clamp_fraction,
            "physical_action_rms": self._last_physical_action_rms,
        }  # shapes [N]
        ones = torch.ones_like(self.prototype_index, dtype=torch.float32)
        self._rollout_count.scatter_add_(0, self.prototype_index, ones)
        for name, value in values.items():
            if value.shape != self.prototype_index.shape:
                raise RuntimeError(f"rollout metric {name} does not align with environment axis")
            self._rollout_sums[name].scatter_add_(0, self.prototype_index, value)


        terminal_drop = snapshot["termination_object_out_of_anchor"].to(self._rl_device).bool()
        terminal_axis = snapshot["termination_goal_axis_misaligned"].to(self._rl_device).bool()
        terminal_timeout = snapshot["termination_time_out"].to(self._rl_device).bool()
        terminal = terminal_drop | terminal_axis | terminal_timeout
        labels = self.prototype_index[terminal]
        net_turns = snapshot["net_rotation_rad"].to(self._rl_device).float()[terminal] / two_pi
        path_turns = snapshot["absolute_path_rotation_rad"].to(self._rl_device).float()[terminal] / two_pi
        directional = torch.clamp(net_turns, min=0.0) / path_turns.clamp_min(torch.finfo(torch.float32).eps)
        terminal_values = {
            "terminal_goal_count_mean": snapshot["completed_subgoals"].to(self._rl_device).float()[terminal],
            "terminal_frontier_count_mean": snapshot["rotation_frontier_count"].to(self._rl_device).float()[terminal],
            "terminal_max_positive_net_turns_mean": (
                snapshot["max_positive_net_rotation_rad"].to(self._rl_device).float()[terminal] / two_pi
            ),
            "terminal_net_turns_mean": net_turns,
            "terminal_absolute_path_turns_mean": path_turns,
            "terminal_directional_consistency_mean": directional.clamp(max=1.0),
            "terminal_timeout_rate": terminal_timeout[terminal].float(),
            "terminal_drop_rate": terminal_drop[terminal].float(),
            "terminal_axis_failure_rate": terminal_axis[terminal].float(),
        }
        self._terminal_count.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        for name, value in terminal_values.items():
            self._terminal_sums[name].scatter_add_(0, labels, value)

    def drain_rollout_metrics(self) -> dict[str, torch.Tensor]:
        'Handle drain rollout metrics; shapes [str,torch.Tensor], [A].'

        if bool((self._rollout_count <= 0).any().item()):
            raise RuntimeError("cannot drain incomplete per-asset rollout metrics")
        if bool((self._rollout_count != self._rollout_count[0]).any().item()):
            raise RuntimeError("rollout metric counts are not equal across assets")
        result = {name: (total / self._rollout_count).detach().cpu() for name, total in self._rollout_sums.items()}
        result["rollout_sample_count"] = self._rollout_count.detach().cpu().clone()
        result["completed_episode_count"] = self._terminal_count.detach().cpu().clone()
        terminal_denominator = self._terminal_count.clamp_min(1.0)
        for name, total in self._terminal_sums.items():
            result[name] = (total / terminal_denominator).detach().cpu()
        if self.training_evidence is not None:
            evidence = self.training_evidence.drain()
            for name in ("pose_keypoint", "orientation_tracking"):
                result[f"reward_term_{name}"] = torch.zeros_like(result["rollout_sample_count"])
            for index, name in enumerate(self.training_evidence.reward_names):
                result[f"reward_term_{name}"] = evidence["reward_terms"][:, index]
        adr = getattr(self.unwrapped, "_hetero_position_adr", None)
        if adr is not None:
            labels = self.prototype_index.reshape(-1).long()
            counts = torch.bincount(labels, minlength=self._asset_count).clamp_min(1)
            for name, values in (("adr_position_level_at_update_end", adr.level.float()),
                                 ("adr_position_half_width_m_at_update_end", adr.half_width)):
                totals = torch.zeros(self._asset_count, device=values.device).scatter_add_(0, labels, values)
                result[name] = (totals / counts).detach().cpu()
        self._rollout_count.zero_()
        for total in self._rollout_sums.values():
            total.zero_()
        self._terminal_count.zero_()
        for total in self._terminal_sums.values():
            total.zero_()
        return result

    def get_env_state(self) -> dict[str, Any]:
        'Return env state.'

        result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "prototype_index": self.prototype_index.detach().cpu(),  # environment-to-asset routing certificate
            "n040_resolve_call_count": int(self.geometry_provider.resolve_call_count),  # diagnostic continuity
        }
        curriculum = getattr(self.unwrapped, HETERO_REWARD_RELEASE_STATE_ATTR, None)
        if isinstance(curriculum, HeterogeneousRewardReleaseState):
            result["reward_release"] = curriculum.state_dict()
        adr = getattr(self.unwrapped, "_hetero_position_adr", None)
        if adr is not None:
            result["object_position_adr"] = adr.state_dict()
        if self.training_evidence is not None and self.training_evidence.first30_statistics is not None:
            result["first30_statistics"] = self.training_evidence.first30_statistics.state_dict()
        return result

    def set_env_state(self, state: object) -> None:
        'Handle set env state.'

        if state is None:
            return
        if not isinstance(state, Mapping) or state.get("schema_version") != "1.0.0":
            raise RuntimeError("palm-rotation checkpoint environment state is missing or incompatible")
        restored_routing = torch.as_tensor(state["prototype_index"], dtype=torch.long, device=self._rl_device)
        if not torch.equal(restored_routing, self.prototype_index):
            raise RuntimeError("checkpoint prototype routing disagrees with current 80-asset environment")
        self.geometry_provider.resolve_call_count = int(state.get("n040_resolve_call_count", 0))
        curriculum = getattr(self.unwrapped, HETERO_REWARD_RELEASE_STATE_ATTR, None)
        if isinstance(curriculum, HeterogeneousRewardReleaseState):
            curriculum.load_state_dict(state.get("reward_release"))  # fail closed on rows/cells/tensor shapes
        if "object_position_adr" in state:
            from anymani.tasks.hetero.mdp.adr import get_position_adr
            adr = get_position_adr(self.unwrapped)
            if adr is None:
                raise RuntimeError("checkpoint position ADR is enabled but runtime component is disabled")
            adr.load_state_dict(state["object_position_adr"])
        if "first30_statistics" in state:
            if self.training_evidence is None or self.training_evidence.first30_statistics is None:
                raise RuntimeError("checkpoint first30 statistics require an enabled training recorder")
            self.training_evidence.first30_statistics.load_state_dict(state["first30_statistics"])


    def set_train_info(self, env_frames: int, *args: Any, **kwargs: Any) -> None:
        'Handle set train info.'

        if self.training_evidence is not None:
            self.training_evidence.policy_version = int(env_frames)
        _ = (args, kwargs)


class PalmRotationRlGamesGpuEnv(IVecEnv):
    'Contract for PALM rotation RL games GPU env.'

    def __init__(self, config_name: str, num_actors: int, *, env: PalmRotationRlGamesVecEnv) -> None:
        'Initialize the instance.'

        _ = (config_name, num_actors)
        self.env = env

    def step(self, actions: torch.Tensor):
        'Handle step.'

        return self.env.step(actions)

    def reset(self):
        'Reset the declared contract.'

        return self.env.reset()

    def get_number_of_agents(self) -> int:
        'Return number of agents.'

        return self.env.get_number_of_agents()

    def get_env_info(self) -> dict[str, Any]:
        'Return env info.'

        return self.env.get_env_info()

    def set_train_info(self, env_frames: int, *args: Any, **kwargs: Any) -> None:
        'Handle set train info.'

        self.env.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self) -> dict[str, Any]:
        'Return env state.'

        return self.env.get_env_state()

    def set_env_state(self, state: object) -> None:
        'Handle set env state.'

        self.env.set_env_state(state)

    def drain_rollout_metrics(self) -> dict[str, torch.Tensor]:
        'Handle drain rollout metrics.'

        return self.env.drain_rollout_metrics()


__all__ = [
    "PALM_ROTATION_BOOL_SHAPES",
    "PALM_ROTATION_FK_TARGET_FLOAT_SHAPES",
    "PALM_ROTATION_FLOAT_SHAPES",
    "PALM_ROTATION_INT16_SHAPES",
    "PALM_ROTATION_STUDENT_FLOAT_SHAPES",
    "PALM_ROTATION_STUDENT_EXTRA_FLOAT_SHAPES",
    "PalmRotationRlGamesGpuEnv",
    "PalmRotationRlGamesVecEnv",
    "palm_rotation_observation_space",
    "palm_rotation_student_float_shapes",
]
