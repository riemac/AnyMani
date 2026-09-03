r"""Fixed hand-$+z$、30°moving-subgoal rotation command与signed progress唯一owner。

Reward与termination在普通command compute之前读取post-physics state，因此所有consumer调用
``ensure_post_physics_progress_updated``，以``common_step_counter``保证每policy step只累计一次。Episode success
只推进subgoal，不终止；terminal step上的success pulse会在command reset日志前计入completed subgoals。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .contact_state import HeterogeneousContactState
from .diagnostics import asset_episode_sufficient_statistics
from .task_math import (
    axis_angle_from_quaternion_wxyz,
    goal_errors_and_success,
    hand_axis_to_world,
    moving_goal_quaternion,
    projected_space_rotation_delta,
    quaternion_inverse_wxyz,
    quaternion_multiply_wxyz,
    quaternion_to_matrix_wxyz,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _identity_quaternion(num_envs: int, device: torch.device | str) -> torch.Tensor:
    r"""构造$(w,x,y,z)$ identity quaternion batch$[N,4]$。"""

    quaternion = torch.zeros(num_envs, 4, device=device)
    quaternion[:, 0] = 1.0
    return quaternion


class HeterogeneousRotationCommand(CommandTerm):
    r"""固定hand semantic$+z$的continuous rotation command。

    ``net_rotation_rad``与``net_rotation_turns``均保留符号；正向能力另存
    ``positive_net_rotation_turns``，避免把clamped path误命名为net。Command tensor是hand-frame goal log error
    $\phi_h\in\mathbb R^3$，但首版actor可不观察它并持续执行固定方向rotation primitive。
    """

    cfg: HeterogeneousRotationCommandCfg

    def __init__(self, cfg: HeterogeneousRotationCommandCfg, env: ManagerBasedRLEnv) -> None:
        r"""分配goal、anchor、signed progress与honest episode metric buffers。"""

        super().__init__(cfg, env)
        self._env = env
        self.object = cast(RigidObject, env.scene[cfg.object_name])
        self.robot = cast(Articulation, env.scene[cfg.robot_name])
        self.semantic_R_ha = torch.tensor(cfg.semantic_R_ha, dtype=torch.float32, device=self.device).reshape(3, 3)
        identity = torch.eye(3, device=self.device)
        if not torch.allclose(self.semantic_R_ha @ self.semantic_R_ha.T, identity, atol=1.0e-4, rtol=0.0):
            raise ValueError("semantic_R_ha must be orthonormal")
        if abs(float(torch.det(self.semantic_R_ha).item()) - 1.0) > 1.0e-4:
            raise ValueError("semantic_R_ha must have determinant +1")

        axis_h = torch.tensor(cfg.fixed_axis_h, dtype=torch.float32, device=self.device).reshape(1, 3)
        axis_norm = torch.linalg.vector_norm(axis_h, dim=-1, keepdim=True)
        if bool((axis_norm < 1.0e-12).any().item()):
            raise ValueError("fixed hand-frame rotation axis must be non-zero")
        self.axis_h = (axis_h / axis_norm).repeat(self.num_envs, 1)  # 固定$\hat k^h=+z$
        self.axis_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_quat_w = _identity_quaternion(self.num_envs, self.device)
        self.goal_error_so3_h = torch.zeros(self.num_envs, 3, device=self.device)
        self.position_anchor_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_quat_w = _identity_quaternion(self.num_envs, self.device)
        self.has_previous = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_progress_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        self.delta_psi = torch.zeros(self.num_envs, device=self.device)  # signed rad/policy step
        self.net_rotation_rad = torch.zeros(self.num_envs, device=self.device)  # signed$\Psi$
        self.absolute_path_rotation_rad = torch.zeros(self.num_envs, device=self.device)  # $\sum_t|\Delta\psi_t|$
        self.net_rotation_turns = torch.zeros(self.num_envs, device=self.device)  # signed$\Psi/(2\pi)$
        self.positive_net_rotation_turns = torch.zeros(self.num_envs, device=self.device)
        self.reached_positive_full_turn = torch.zeros(self.num_envs, device=self.device)
        self.axis_speed_rad_s = torch.zeros(self.num_envs, device=self.device)
        self.axis_speed_ema_rad_s = torch.zeros(self.num_envs, device=self.device)
        self.orientation_keypoint_error_m = torch.zeros(self.num_envs, device=self.device)
        self.position_error_m = torch.zeros(self.num_envs, device=self.device)
        self.goal_normal_alignment = torch.ones(self.num_envs, device=self.device)
        self.goal_success_pulse = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_success_count = torch.zeros(self.num_envs, device=self.device)
        self.subgoal_throughput_per_horizon_s = torch.zeros(self.num_envs, device=self.device)

        # RewardManager最后一个term在任何automatic reset之前覆盖这些full-env tensors。Command/event reset不清它们，
        # 因而env.step返回后done rows仍对应terminal physics frame，而不是新episode的零值。
        dataset_rows = (
            torch.tensor(cfg.dataset_row_by_env, dtype=torch.long, device=self.device)
            if len(cfg.dataset_row_by_env) == self.num_envs
            else torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        )
        self.post_physics_evaluation_snapshot: dict[str, torch.Tensor] = {
            "valid": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "step": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "dataset_row": dataset_rows,
            "axis_speed_rad_s": torch.zeros(self.num_envs, device=self.device),
            "net_rotation_rad": torch.zeros(self.num_envs, device=self.device),
            "absolute_path_rotation_rad": torch.zeros(self.num_envs, device=self.device),
            "completed_subgoals": torch.zeros(self.num_envs, device=self.device),
            "goal_success_pulse": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "episode_duration_s": torch.zeros(self.num_envs, device=self.device),
            "tip_active_count": torch.zeros(self.num_envs, device=self.device),
            "palm_contact": torch.zeros(self.num_envs, device=self.device),
            "finger_non_tip_contact": torch.zeros(self.num_envs, device=self.device),
            "orientation_keypoint_error_m": torch.zeros(self.num_envs, device=self.device),
            "position_error_m": torch.zeros(self.num_envs, device=self.device),
            "termination_object_out_of_anchor": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "termination_goal_axis_misaligned": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "termination_time_out": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

        self.metrics.update(
            {
                "rotation/delta_psi_rad": self.delta_psi,
                "rotation/net_rotation_rad": self.net_rotation_rad,
                "rotation/absolute_path_rotation_rad": self.absolute_path_rotation_rad,
                "rotation/net_rotation_turns_signed": self.net_rotation_turns,
                "rotation/positive_net_rotation_turns": self.positive_net_rotation_turns,
                "rotation/reached_positive_full_turn": self.reached_positive_full_turn,
                "rotation/axis_speed_rad_s": self.axis_speed_rad_s,
                "rotation/axis_speed_ema_rad_s": self.axis_speed_ema_rad_s,
                "pose/orientation_keypoint_error_m": self.orientation_keypoint_error_m,
                "pose/position_error_m": self.position_error_m,
                "pose/goal_normal_alignment_signed": self.goal_normal_alignment,
                "task/goal_success_count": self.goal_success_count,
                "task/subgoal_throughput_per_horizon_s": self.subgoal_throughput_per_horizon_s,
            }
        )
        ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._capture_reset_state(ids)
        self._resample_command(ids)

    @property
    def command(self) -> torch.Tensor:
        r"""返回hand-frame relative goal log error$\phi_h$，形状$[N,3]$。"""

        return self.goal_error_so3_h

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        r"""记录旧episode并从pregrasp写入后的真实object state建立新anchor/goal。

        ManagerBased reset在reward之后、普通command compute之前发生。若terminal step同时达到subgoal，pulse尚未
        被``_update_command``消费，因此这里先计数，保证throughput不漏掉最后一个成功。
        """

        ids = self._as_ids(env_ids)
        self.goal_success_count[ids] += self.goal_success_pulse[ids].to(dtype=self.goal_success_count.dtype)
        self.subgoal_throughput_per_horizon_s[ids] = self.goal_success_count[ids] / float(self.cfg.horizon_s)
        asset_extras = self._asset_episode_extras(ids)  # 必须在super清零metrics前读取terminal subset
        extras = super().reset(env_ids)
        extras.update(asset_extras)
        self._capture_reset_state(ids)
        self._refresh_goal_state(ids)
        return extras

    def _asset_episode_extras(self, ids: torch.Tensor) -> dict[str, float]:
        r"""输出per-asset terminal sum/count充分统计量，供外部equal-asset reducer。

        只读取当前reset rows和当前TerminationManager snapshot，不使用会混入其他env stale dones的manager
        aggregate。Terminal-step success pulse已在caller中加入``goal_success_count``。
        """

        if not self.cfg.log_asset_metrics:
            return {}
        if len(self.cfg.dataset_row_by_env) != self.num_envs:
            raise RuntimeError("asset diagnostics require one formal dataset row per environment")
        dataset_rows = torch.tensor(self.cfg.dataset_row_by_env, dtype=torch.long, device=self.device)
        duration_s = self._env.episode_length_buf.to(dtype=torch.float32) * float(self._env.step_dt)
        termination = {
            name: self._env.termination_manager.get_term(name)
            for name in ("object_out_of_anchor", "goal_axis_misaligned", "time_out")
        }
        # Caller已经把terminal pulse加入goal_success_count，故传入全False pulse避免重复计数。
        return asset_episode_sufficient_statistics(
            dataset_row_by_env=dataset_rows,
            reset_env_ids=ids,
            goal_success_count=self.goal_success_count,
            goal_success_pulse=torch.zeros_like(self.goal_success_pulse),
            net_rotation_rad=self.net_rotation_rad,
            positive_net_rotation_turns=self.positive_net_rotation_turns,
            episode_duration_s=duration_s,
            termination_bits=termination,
            horizon_s=float(self.cfg.horizon_s),
        )

    def capture_post_physics_evaluation_snapshot(
        self,
        contact: HeterogeneousContactState,
        termination_bits: dict[str, torch.Tensor],
    ) -> None:
        r"""冻结当前post-physics、pre-reset frame的trajectory充分状态。

        调用点必须位于RewardManager最后一个term：TerminationManager已计算当前failure bits，contact rewards已刷新
        20 Hz EMA，而ManagerBasedRLEnv尚未执行``scene.reset``、pregrasp event或command reset。Snapshot保持full
        environment axis；下一step覆盖它，automatic reset本身不得清除。
        """

        required_terms = ("object_out_of_anchor", "goal_axis_misaligned", "time_out")
        if set(termination_bits) != set(required_terms):
            raise ValueError("evaluation snapshot requires exact drop/axis/timeout termination terms")
        if any(bits.shape != (self.num_envs,) for bits in termination_bits.values()):
            raise ValueError("evaluation snapshot termination bits must share the environment axis")
        self.ensure_post_physics_progress_updated()
        contact.ensure_updated(self._env)
        snapshot = self.post_physics_evaluation_snapshot
        snapshot["valid"].fill_(True)
        snapshot["step"].fill_(int(self._env.common_step_counter))
        snapshot["axis_speed_rad_s"].copy_(self.axis_speed_rad_s)
        snapshot["net_rotation_rad"].copy_(self.net_rotation_rad)
        snapshot["absolute_path_rotation_rad"].copy_(self.absolute_path_rotation_rad)
        snapshot["completed_subgoals"].copy_(
            self.goal_success_count + self.goal_success_pulse.to(dtype=self.goal_success_count.dtype)
        )
        snapshot["goal_success_pulse"].copy_(self.goal_success_pulse)
        snapshot["episode_duration_s"].copy_(
            self._env.episode_length_buf.to(dtype=torch.float32) * float(self._env.step_dt)
        )
        snapshot["tip_active_count"].copy_(contact.tip_bits.sum(dim=-1).to(dtype=torch.float32))
        snapshot["palm_contact"].copy_(contact.palm_bits[:, 0].to(dtype=torch.float32))
        snapshot["finger_non_tip_contact"].copy_(
            contact.finger_non_tip_bits.any(dim=-1).to(dtype=torch.float32)
        )
        snapshot["orientation_keypoint_error_m"].copy_(self.orientation_keypoint_error_m)
        snapshot["position_error_m"].copy_(self.position_error_m)
        for term_name in required_terms:
            snapshot[f"termination_{term_name}"].copy_(termination_bits[term_name])

    def ensure_post_physics_progress_updated(self) -> None:
        r"""按common-step stamp幂等刷新signed progress、goal error与success pulse。"""

        step = int(self._env.common_step_counter)
        update_mask = self.last_progress_step != step
        if not bool(update_mask.any().item()):
            return
        ids = update_mask.nonzero(as_tuple=False).flatten()
        self.axis_w[ids] = hand_axis_to_world(
            self.axis_h[ids], self.robot.data.root_quat_w[ids], self.semantic_R_ha
        )
        current_quaternion = self.object.data.root_quat_w
        valid = update_mask & self.has_previous
        self.delta_psi[ids] = 0.0
        if bool(valid.any().item()):
            self.delta_psi[valid] = projected_space_rotation_delta(
                self.previous_quat_w[valid], current_quaternion[valid], self.axis_w[valid]
            )
        self.net_rotation_rad[ids] += self.delta_psi[ids]
        self.absolute_path_rotation_rad[ids] += self.delta_psi[ids].abs()
        self.net_rotation_turns[ids] = self.net_rotation_rad[ids] / (2.0 * math.pi)
        self.positive_net_rotation_turns[ids] = torch.clamp(self.net_rotation_rad[ids], min=0.0) / (2.0 * math.pi)
        self.reached_positive_full_turn[ids] = (self.net_rotation_rad[ids] >= 2.0 * math.pi).to(torch.float32)
        self.axis_speed_rad_s[ids] = self.delta_psi[ids] / float(self._env.step_dt)
        alpha = 1.0 - math.exp(-float(self._env.step_dt) / float(self.cfg.speed_ema_time_constant_s))
        self.axis_speed_ema_rad_s[ids] = (
            (1.0 - alpha) * self.axis_speed_ema_rad_s[ids] + alpha * self.axis_speed_rad_s[ids]
        )
        self.previous_quat_w[ids] = current_quaternion[ids].detach()
        self.has_previous[ids] = True
        self._refresh_goal_state(ids)
        self.last_progress_step[ids] = step

    def _update_metrics(self) -> None:
        r"""CommandManager metric hook复用当前post-physics快照。"""

        self.ensure_post_physics_progress_updated()

    def _update_command(self) -> None:
        r"""Reward消费success pulse后，从当前object pose推进下一30°goal。"""

        success_ids = self.goal_success_pulse.nonzero(as_tuple=False).flatten()
        if success_ids.numel() == 0:
            return
        self.goal_success_count[success_ids] += 1.0
        self.subgoal_throughput_per_horizon_s[success_ids] = self.goal_success_count[success_ids] / float(
            self.cfg.horizon_s
        )
        self._resample_command(success_ids)
        self.command_counter[success_ids] += 1
        self.time_left[success_ids] = self.cfg.resampling_time_range[1]
        self.goal_success_pulse[success_ids] = False

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        r"""从当前pose左乘固定space rotation，不从旧goal累积。"""

        ids = self._as_ids(env_ids)
        if ids.numel() == 0:
            return
        self.axis_w[ids] = hand_axis_to_world(
            self.axis_h[ids], self.robot.data.root_quat_w[ids], self.semantic_R_ha
        )
        self.goal_quat_w[ids] = moving_goal_quaternion(
            self.object.data.root_quat_w[ids], self.axis_w[ids], subgoal_angle_rad=float(self.cfg.subgoal_angle_rad)
        )
        self._refresh_goal_state(ids)

    def _capture_reset_state(self, ids: torch.Tensor) -> None:
        r"""捕获world position anchor并清selected episode progress，不碰其它rows。"""

        self.position_anchor_w[ids] = self.object.data.root_pos_w[ids].detach()
        self.previous_quat_w[ids] = self.object.data.root_quat_w[ids].detach()
        self.has_previous[ids] = True
        self.delta_psi[ids] = 0.0
        self.net_rotation_rad[ids] = 0.0
        self.absolute_path_rotation_rad[ids] = 0.0
        self.net_rotation_turns[ids] = 0.0
        self.positive_net_rotation_turns[ids] = 0.0
        self.reached_positive_full_turn[ids] = 0.0
        self.axis_speed_rad_s[ids] = 0.0
        self.axis_speed_ema_rad_s[ids] = 0.0
        self.goal_success_pulse[ids] = False
        self.goal_success_count[ids] = 0.0
        self.subgoal_throughput_per_horizon_s[ids] = 0.0
        self.last_progress_step[ids] = int(self._env.common_step_counter)

    def _refresh_goal_state(self, ids: torch.Tensor) -> None:
        r"""刷新strict success双门、signed normal alignment和hand-frame goal log error。"""

        if ids.numel() == 0:
            return
        orientation_error, position_error, alignment, success = goal_errors_and_success(
            self.object.data.root_pos_w[ids],
            self.object.data.root_quat_w[ids],
            self.position_anchor_w[ids],
            self.goal_quat_w[ids],
            keypoint_radius_m=float(self.cfg.keypoint_radius_m),
            orientation_threshold_m=float(self.cfg.orientation_success_threshold_m),
            position_threshold_m=float(self.cfg.position_success_threshold_m),
        )
        self.orientation_keypoint_error_m[ids] = orientation_error
        self.position_error_m[ids] = position_error
        self.goal_normal_alignment[ids] = alignment
        self.goal_success_pulse[ids] = success
        error_quaternion_w = quaternion_multiply_wxyz(
            self.goal_quat_w[ids], quaternion_inverse_wxyz(self.object.data.root_quat_w[ids])
        )
        error_vector_w = axis_angle_from_quaternion_wxyz(error_quaternion_w)
        rotation_wa = quaternion_to_matrix_wxyz(self.robot.data.root_quat_w[ids])
        rotation_hw = self.semantic_R_ha.unsqueeze(0) @ rotation_wa.transpose(-1, -2)
        self.goal_error_so3_h[ids] = torch.einsum("bij,bj->bi", rotation_hw, error_vector_w)

    def _as_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        r"""把CommandTerm full/partial selection统一为device LongTensor。"""

        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)


def get_rotation_command(env: ManagerBasedRLEnv, command_name: str) -> HeterogeneousRotationCommand:
    r"""解析并幂等刷新指定rotation command。"""

    term = env.command_manager.get_term(command_name)
    if not isinstance(term, HeterogeneousRotationCommand):
        raise TypeError(f"command {command_name!r} is not HeterogeneousRotationCommand")
    term.ensure_post_physics_progress_updated()
    return term


@configclass
class HeterogeneousRotationCommandCfg(CommandTermCfg):
    r"""Palm-up DexCube固定hand$+z$、30°moving-subgoal command配置。"""

    class_type: type = HeterogeneousRotationCommand
    object_name: str = "object"
    robot_name: str = "robot"
    fixed_axis_h: tuple[float, float, float] = (0.0, 0.0, 1.0)
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    subgoal_angle_rad: float = math.pi / 6.0
    keypoint_radius_m: float = 0.05
    orientation_success_threshold_m: float = 0.005
    position_success_threshold_m: float = 0.025
    speed_ema_time_constant_s: float = 0.25
    horizon_s: float = 120.0  # fixed-horizon throughput分母
    dataset_row_by_env: tuple[int, ...] = ()  # diagnostics-only formal row labels
    log_asset_metrics: bool = False  # training默认不生成per-asset dynamic keys
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)  # 只由reset/success换goal
    debug_vis: bool = False

    def __post_init__(self) -> None:
        r"""静态拒绝非法frame、axis、angle、distance与time constants。"""

        if len(self.semantic_R_ha) != 9:
            raise ValueError("semantic_R_ha must contain nine row-major values")
        if math.sqrt(sum(value * value for value in self.fixed_axis_h)) < 1.0e-12:
            raise ValueError("fixed_axis_h must be non-zero")
        if not 0.0 < self.subgoal_angle_rad < math.pi:
            raise ValueError("subgoal angle must lie in (0,pi)")
        positive = (
            self.keypoint_radius_m,
            self.orientation_success_threshold_m,
            self.position_success_threshold_m,
            self.speed_ema_time_constant_s,
            self.horizon_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("command distance/time parameters must be finite and positive")


__all__ = [
    "HeterogeneousRotationCommand",
    "HeterogeneousRotationCommandCfg",
    "get_rotation_command",
]
