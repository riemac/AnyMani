r"""Palm-supported tactile rotation 的 episode-level 科研诊断状态。

该状态不参与 observation、reward 或 termination，只把已有 post-physics buffers 转成带单位的
episode statistics。每个 env 独立累积，`common_step_counter` 保证同一 policy step 最多更新一次；
CommandTerm reset 在输出 metrics 后只清本次结束的 env，避免跨 episode 与 partial-reset 污染。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast

import torch

from .adr_state import ADR_STATE_SLICES, get_gm_adr_state
from .tactile_contact_state import get_tactile_contact_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_MEAN_METRIC_NAMES = (
    "rotation/axis_speed_mean_rad_s",
    "rotation/axis_speed_abs_mean_rad_s",
    "pose/anchor_distance_mean_m",
    "pose/orientation_keypoint_error_mean_m",
    "action/policy_delta_rms_per_s",
    "action/executed_delta_rms_per_s",
    "action/target_delta_rms_rad_s",
    "action/target_tracking_error_rms_rad",
    "contact/tip_active_count_mean",
    "contact/palm_occupancy_fraction",
    "contact/finger_non_tip_occupancy_fraction",
    "contact/tip_force_ema_mean_N",
    "contact/palm_force_ema_mean_N",
)
r"""逐 policy step 求 episode mean 的 metric keys；名称显式携带聚合与物理单位。"""


class _TactileActionTerm(Protocol):
    r"""Diagnostics 消费的最小 action-term tensor surface。"""

    raw_actions: torch.Tensor  # wrapper-clamped policy action，`[B,16]`
    executed_actions: torch.Tensor  # ADR latency/noise 后实际执行 action，`[B,16]`
    current_targets: torch.Tensor  # policy-step recurrent joint target，`[B,16]`，rad


class GmTactileEpisodeDiagnostics:
    r"""一次读取 command/action/contact/ADR state，维护每个 env 的 episode summary。"""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        fingertip_sensor_names: Sequence[str],
        finger_non_tip_sensor_names: Sequence[str],
        palm_sensor_name: str,
        action_name: str = "hand_joint_pos",
        ema_alpha: float = 0.5,
        force_threshold: float = 0.25,
    ) -> None:
        r"""分配诊断 buffer，并锁定 contact/action schema。

        Args:
            env (ManagerBasedRLEnv): 共享 vectorized environment。
            fingertip_sensor_names (Sequence[str]): 4 个 tip sensors 的 canonical order。
            finger_non_tip_sensor_names (Sequence[str]): 19 个 finger non-tip sensors。
            palm_sensor_name (str): neutral palm support sensor。
            action_name (str): policy-step target action term 名称。
            ema_alpha (float): 与 actor/reward 共用的 contact EMA 系数。
            force_threshold (float): contact bit 阈值，单位 N。
        """

        self.num_envs = int(env.num_envs)  # vectorized batch size $B$
        self.device = env.device  # 所有在线统计保留在 simulation device
        self.action_name = str(action_name)
        self.fingertip_sensor_names = tuple(str(name) for name in fingertip_sensor_names)
        self.finger_non_tip_sensor_names = tuple(str(name) for name in finger_non_tip_sensor_names)
        self.palm_sensor_name = str(palm_sensor_name)
        self.ema_alpha = float(ema_alpha)
        self.force_threshold = float(force_threshold)

        # 所有公开 metrics 均为 `[B]`；CommandTermBase.reset 会对结束 env subset 求均值并写 extras。
        metric_names = (*_MEAN_METRIC_NAMES, "rotation/off_axis_ang_vel_rms_rad_s", "pose/anchor_distance_max_m")
        metric_names += (
            "task/episode_duration_s",
            "task/sampled_horizon_s",
            "adr/actual_object_scale",
            "adr/actual_object_mass_kg",
            "adr/actual_com_offset_norm_m",
            "adr/actual_object_static_friction",
            "adr/actual_object_dynamic_friction",
            "adr/actual_object_restitution",
            "adr/actual_hand_static_friction",
            "adr/actual_hand_dynamic_friction",
            "adr/actual_hand_restitution",
            "adr/actual_joint_stiffness_mean",
            "adr/actual_joint_damping_mean",
            "adr/actual_action_noise_std",
            "adr/actual_latency_steps_mean",
            "adr/actual_wrench_gate_fraction",
            "adr/actual_max_linear_acceleration_m_s2",
            "adr/actual_fraction",
        )
        metric_names += tuple(f"termination/{name}_fraction" for name in env.termination_manager.active_terms)
        self.metrics = {name: torch.zeros(self.num_envs, device=self.device) for name in metric_names}

        # Running sums 只服务 episode mean；RMS 与 max 使用独立充分统计量。
        self._sums = {name: torch.zeros(self.num_envs, device=self.device) for name in _MEAN_METRIC_NAMES}
        self._off_axis_square_sum = torch.zeros(self.num_envs, device=self.device)  # $\sum_t\|\omega_{\perp,t}\|^2$
        self.step_count = torch.zeros(self.num_envs, device=self.device)  # episode 内已积累 policy steps
        self.last_update_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        # Action differences 需要上一 policy-step snapshot；width 在第一次 reset 时从 action term 解析。
        action_term = cast(_TactileActionTerm, env.action_manager.get_term(self.action_name))
        self._previous_policy_action = torch.zeros_like(action_term.raw_actions)  # `[B,16]`，无量纲
        self._previous_executed_action = torch.zeros_like(action_term.executed_actions)  # `[B,16]`，无量纲
        self._previous_target = torch.zeros_like(action_term.current_targets)  # `[B,16]`，rad

    def ensure_updated(
        self,
        env: ManagerBasedRLEnv,
        *,
        axis_w: torch.Tensor,
        axis_speed: torch.Tensor,
        object_ang_vel_w: torch.Tensor,
        joint_position: torch.Tensor,
        position_error: torch.Tensor,
        orientation_keypoint_error: torch.Tensor,
    ) -> None:
        r"""在当前 policy step 对未更新 env 累积一次动力学/action/contact summary。"""

        step = int(env.common_step_counter)
        update_mask = self.last_update_step != step  # reset env 当前 stamp 已处理，下一 physics step 才进入
        if not torch.any(update_mask):
            return
        action_term = cast(_TactileActionTerm, env.action_manager.get_term(self.action_name))
        contact = get_tactile_contact_state(
            env,
            self.fingertip_sensor_names,
            self.finger_non_tip_sensor_names,
            self.palm_sensor_name,
            self.ema_alpha,
            self.force_threshold,
        )
        dt = float(env.step_dt)  # policy period，当前为 $0.05\ s$

        # Object angular velocity 分解为 command 轴向与离轴分量。
        parallel_speed = torch.sum(object_ang_vel_w * axis_w, dim=-1)  # $\omega_\parallel=k^T\omega$，rad/s
        off_axis = object_ang_vel_w - parallel_speed[:, None] * axis_w  # $\omega_\perp$，`[B,3]`
        off_axis_square = torch.sum(off_axis.square(), dim=-1)  # $\|\omega_\perp\|_2^2$

        # Action/target rate 先对 16 joints 做 RMS，再按 episode policy steps 求 mean。
        policy_delta_rate = torch.mean((action_term.raw_actions - self._previous_policy_action).square(), dim=-1).sqrt() / dt
        executed_delta_rate = (
            torch.mean((action_term.executed_actions - self._previous_executed_action).square(), dim=-1).sqrt() / dt
        )
        target_delta_rate = torch.mean((action_term.current_targets - self._previous_target).square(), dim=-1).sqrt() / dt
        target_tracking_error = torch.mean((action_term.current_targets - joint_position).square(), dim=-1).sqrt()

        # 每个 key 的 instantaneous value 都有明确单位；只对 update_mask 行累积。
        instantaneous = {
            "rotation/axis_speed_mean_rad_s": axis_speed,
            "rotation/axis_speed_abs_mean_rad_s": torch.abs(axis_speed),
            "pose/anchor_distance_mean_m": position_error,
            "pose/orientation_keypoint_error_mean_m": orientation_keypoint_error,
            "action/policy_delta_rms_per_s": policy_delta_rate,
            "action/executed_delta_rms_per_s": executed_delta_rate,
            "action/target_delta_rms_rad_s": target_delta_rate,
            "action/target_tracking_error_rms_rad": target_tracking_error,
            "contact/tip_active_count_mean": contact.tip_bits.float().sum(dim=-1),
            "contact/palm_occupancy_fraction": contact.palm_bits.float().squeeze(-1),
            "contact/finger_non_tip_occupancy_fraction": contact.finger_non_tip_bits.float().mean(dim=-1),
            "contact/tip_force_ema_mean_N": contact.tip_force_ema.mean(dim=-1),
            "contact/palm_force_ema_mean_N": contact.palm_force_ema.squeeze(-1),
        }
        self.step_count[update_mask] += 1.0
        denominator = torch.clamp(self.step_count, min=1.0)  # `[B]`，只用于稳定除法
        for name, value in instantaneous.items():
            self._sums[name][update_mask] += value[update_mask]
            self.metrics[name][update_mask] = self._sums[name][update_mask] / denominator[update_mask]

        # RMS 与 maximum 不能由普通 mean summary 反推，单独维护充分统计量。
        self._off_axis_square_sum[update_mask] += off_axis_square[update_mask]
        self.metrics["rotation/off_axis_ang_vel_rms_rad_s"][update_mask] = torch.sqrt(
            self._off_axis_square_sum[update_mask] / denominator[update_mask]
        )
        self.metrics["pose/anchor_distance_max_m"][update_mask] = torch.maximum(
            self.metrics["pose/anchor_distance_max_m"][update_mask], position_error[update_mask]
        )

        # 提交本 step action snapshots，供下一 policy step 计算有限差分。
        self._previous_policy_action[update_mask] = action_term.raw_actions[update_mask]
        self._previous_executed_action[update_mask] = action_term.executed_actions[update_mask]
        self._previous_target[update_mask] = action_term.current_targets[update_mask]
        self.last_update_step[update_mask] = step

    def capture_terminal(self, env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor) -> None:
        r"""在 CommandTerm reset logging 前冻结 episode duration 与精确 termination causes。"""

        ids = _env_ids_tensor(env, env_ids)
        self.metrics["task/episode_duration_s"][ids] = env.episode_length_buf[ids].float() * float(env.step_dt)
        for term_name in env.termination_manager.active_terms:
            self.metrics[f"termination/{term_name}_fraction"][ids] = env.termination_manager.get_term(term_name)[ids].float()

    def reset(self, env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        r"""清指定 env 的 episode accumulators，并记录新 episode 的 actual ADR snapshot。"""

        ids = _env_ids_tensor(env, env_ids)
        for value in self._sums.values():
            value[ids] = 0.0
        self._off_axis_square_sum[ids] = 0.0
        self.step_count[ids] = 0.0
        for value in self.metrics.values():
            value[ids] = 0.0

        # ActionManager 已在 CommandManager 前 reset；以下值正是新 episode 的 $a_0=0,u_0=q_0$。
        action_term = cast(_TactileActionTerm, env.action_manager.get_term(self.action_name))
        self._previous_policy_action[ids] = action_term.raw_actions[ids]
        self._previous_executed_action[ids] = action_term.executed_actions[ids]
        self._previous_target[ids] = action_term.current_targets[ids]
        self.last_update_step[ids] = int(env.common_step_counter)
        self._record_actual_adr_snapshot(env, ids)

    def _record_actual_adr_snapshot(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
        r"""把新 episode 固定/采样的 48D ADR state 降成可解释 scalar diagnostics。"""

        values = get_gm_adr_state(env).values[env_ids]  # `[K,48]` actual values，不读取 nominal endpoints
        self.metrics["adr/actual_object_scale"][env_ids] = values[:, ADR_STATE_SLICES["scale"]].mean(dim=-1)
        self.metrics["adr/actual_object_mass_kg"][env_ids] = values[:, ADR_STATE_SLICES["mass"]].mean(dim=-1)
        self.metrics["adr/actual_com_offset_norm_m"][env_ids] = torch.linalg.norm(
            values[:, ADR_STATE_SLICES["com"]], dim=-1
        )
        object_material = values[:, ADR_STATE_SLICES["object_material"]]  # `[K,3] = (mu_s,mu_d,e)`
        hand_material = values[:, ADR_STATE_SLICES["hand_contact_material"]]  # `[K,3]`
        self.metrics["adr/actual_object_static_friction"][env_ids] = object_material[:, 0]
        self.metrics["adr/actual_object_dynamic_friction"][env_ids] = object_material[:, 1]
        self.metrics["adr/actual_object_restitution"][env_ids] = object_material[:, 2]
        self.metrics["adr/actual_hand_static_friction"][env_ids] = hand_material[:, 0]
        self.metrics["adr/actual_hand_dynamic_friction"][env_ids] = hand_material[:, 1]
        self.metrics["adr/actual_hand_restitution"][env_ids] = hand_material[:, 2]
        self.metrics["adr/actual_joint_stiffness_mean"][env_ids] = values[:, ADR_STATE_SLICES["stiffness"]].mean(dim=-1)
        self.metrics["adr/actual_joint_damping_mean"][env_ids] = values[:, ADR_STATE_SLICES["damping"]].mean(dim=-1)
        self.metrics["adr/actual_action_noise_std"][env_ids] = values[:, ADR_STATE_SLICES["action_noise"]].mean(dim=-1)
        self.metrics["adr/actual_latency_steps_mean"][env_ids] = values[:, ADR_STATE_SLICES["latency_steps"]].mean(dim=-1)
        self.metrics["adr/actual_wrench_gate_fraction"][env_ids] = values[:, ADR_STATE_SLICES["wrench_gate"]].mean(dim=-1)
        self.metrics["adr/actual_max_linear_acceleration_m_s2"][env_ids] = values[
            :, ADR_STATE_SLICES["max_acceleration"]
        ].mean(dim=-1)
        self.metrics["adr/actual_fraction"][env_ids] = values[:, ADR_STATE_SLICES["fraction"]].mean(dim=-1)

        sampled_horizon = getattr(env, "leap_adr_episode_lengths", None)
        if isinstance(sampled_horizon, torch.Tensor):
            self.metrics["task/sampled_horizon_s"][env_ids] = sampled_horizon[env_ids].float() * float(env.step_dt)


def _env_ids_tensor(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
) -> torch.Tensor:
    r"""把 manager partial-reset ids 统一为 env-device LongTensor。"""

    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()


__all__ = ["GmTactileEpisodeDiagnostics"]
