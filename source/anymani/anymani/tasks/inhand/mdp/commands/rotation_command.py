# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""连续旋转命令生成器。

该模块实现一个基于固定轴的连续重定向（Continuous Reorientation）命令项。
命令在环境重置时采样初始姿态，并在达到当前目标后沿同一轴持续累积旋转。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import ContinuousRotationCommandCfg
    from .commands_cfg import RelativeSO3CommandCfg


# 预定义的世界坐标系旋转轴映射
_AXIS_MAP = {
    "x": torch.tensor([1.0, 0.0, 0.0]),
    "x_axis": torch.tensor([1.0, 0.0, 0.0]),
    "y": torch.tensor([0.0, 1.0, 0.0]),
    "y_axis": torch.tensor([0.0, 1.0, 0.0]),
    "z": torch.tensor([0.0, 0.0, 1.0]),
    "z_axis": torch.tensor([0.0, 0.0, 1.0]),
}


class ContinuousRotationCommand(CommandTerm):
    """连续旋转命令项。

    Note
    ----
    命令在世界坐标系下定义：
      - 旋转轴 ``n_w`` 固定在世界坐标系。
      - 每次成功后更新的目标姿态为 ``q_target ← Δq ⊗ q_target``，其中
        ``Δq`` 是绕 ``n_w`` 旋转 ``Δθ`` 的四元数。
    """

    cfg: ContinuousRotationCommandCfg

    def __init__(self, cfg: ContinuousRotationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object = env.scene[cfg.asset_name]

        # 在环境坐标系保留位置命令，保持与现有观察构建兼容
        init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device) # 与手托起物体的设计有关
        self.pos_command_e = self.object.data.default_root_state[:, :3] + init_pos_offset # 环境坐标系  
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins

        # 目标姿态缓冲（世界坐标系四元数）
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0

        # 每个环境固定的旋转轴、角度增量以及累计统计量
        self.rotation_axis_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.delta_angle = torch.full((self.num_envs,), cfg.delta_angle, device=self.device)
        self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, device=self.device)

        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cumulative_rotation"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)

    def __str__(self) -> str:
        msg = "ContinuousRotationCommand:\n"
        msg += f"\t命令维度: {tuple(self.command.shape[1:])}\n"
        msg += f"\t旋转轴: {self.cfg.rotation_axis}\n"
        msg += f"\t角度增量: {self.cfg.delta_angle} rad"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """返回目标位姿 (pos_e, quat_w)。"""

        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)  # 位置是环境坐标系，姿态是世界坐标系，在命令项中返回的是该值

    def _update_metrics(self):
        """更新日志指标。"""

        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )
        self.metrics["cumulative_rotation"] = self.cumulative_rotation
        self.metrics["consecutive_success"] = self.success_counter

    def _resample_command(self, env_ids: Sequence[int]):
        """重置命令并采样新的初始目标姿态。"""

        if len(env_ids) == 0:
            return

        axis_key = self.cfg.rotation_axis.lower()
        if axis_key not in _AXIS_MAP:
            raise ValueError(
                f"不支持的旋转轴 '{self.cfg.rotation_axis}'. 支持项为: {sorted(set(_AXIS_MAP.keys()))}."
            )

        axis_vec = _AXIS_MAP[axis_key].to(self.device)
        self.rotation_axis_w[env_ids] = axis_vec

        # 在物体当前姿态基础上叠加 delta_angle，作为重置后的初始目标姿态
        # 这样重置后的第一个目标与后续目标保持一致的角度增量
        axis_batch = axis_vec.repeat(len(env_ids), 1)
        delta_quat = math_utils.quat_from_angle_axis(self.delta_angle[env_ids], axis_batch)
        base_quat = self.object.data.root_quat_w[env_ids]
        self.quat_command_w[env_ids] = math_utils.quat_mul(delta_quat, base_quat)
        if self.cfg.make_quat_unique:
            self.quat_command_w[env_ids] = math_utils.quat_unique(self.quat_command_w[env_ids])

        self.cumulative_rotation[env_ids] = 0.0
        self.success_counter[env_ids] = 0.0
        self.metrics["orientation_error"][env_ids] = 0.0
        self.metrics["cumulative_rotation"][env_ids] = 0.0
        self.metrics["consecutive_success"][env_ids] = 0.0

    def _update_command(self):
        """根据成功判定沿固定轴增量旋转目标姿态。"""

        if not self.cfg.update_goal_on_success:
            return

        success_mask = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        success_ids = success_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(success_ids) == 0:
            return

        # 成功后沿同一轴推进固定角度，形成连续的目标序列
        delta = torch.full((len(success_ids),), self.cfg.delta_angle, device=self.device)
        delta_quat = math_utils.quat_from_angle_axis(delta, self.rotation_axis_w[success_ids])
        updated = math_utils.quat_mul(delta_quat, self.quat_command_w[success_ids])
        if self.cfg.make_quat_unique:
            updated = math_utils.quat_unique(updated)
        self.quat_command_w[success_ids] = updated

        self.cumulative_rotation[success_ids] += self.cfg.delta_angle
        self.success_counter[success_ids] += 1.0
        self.command_counter[success_ids] += 1
        max_time = self.cfg.resampling_time_range[1]
        self.time_left[success_ids] = max_time

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError("ContinuousRotationCommand 尚未实现调试可视化。")

    def _debug_vis_callback(self, event):
        raise NotImplementedError("ContinuousRotationCommand 尚未实现调试可视化。")


class RelativeSO3Command(CommandTerm):
    """so(3) 相对增量指令命令项（rotvec）。

    该命令项为 in-hand 旋转任务提供一个 **3 维 so(3) 参考指令** ``phi_ref_e``，并在内部维护
    目标姿态 ``quat_command_w``，以便复用现有的 `generated_commands` 观测与基于目标四元数的奖励。

    两种工作模式（与方案文档对齐）：

    - ``fixed_goal``：
        在重采样时冻结目标： ``R_g ← exp(phi_ref) R_c(t0)``。
        之后目标不变，误差 ``phi_err`` 将随控制收敛；当角误差小于阈值时重采样。

    - ``rolling_goal``：
        每个 timestep 更新目标： ``R_g(t) ← exp(phi_ref) R_c(t)``。
        此时误差恒等于 ``phi_ref``，便于部署阶段用固定指令持续旋转。

    坐标系约定：
        当前实现将 ``phi_ref`` 视为 **环境坐标系 {e}** 下的旋转向量（rotvec）。
        对应的目标四元数 ``quat_command_w`` 与物体根姿态的世界系表示一致（root_quat_w）。
    """

    cfg: RelativeSO3CommandCfg

    def __init__(self, cfg: RelativeSO3CommandCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.object = env.scene[cfg.asset_name]

        # NOTE:
        #   用户侧需求：不希望物体位置到处漂移，因此希望命令项同时给出“目标位置约束”。
        #   于是 `command` 属性从 3D 扩展为 6D：
        #       (pos_command_e, phi_ref_e)
        #   - 前 3 维 pos_command_e：环境系 {e} 下的位置目标，用于位置约束
        #   - 后 3 维 phi_ref_e：环境系 {e} 下的 so(3) 指令 rotvec，用于相对旋转
        #
        #   但 policy 观测侧仍只应看到 3 维 so(3) 指令：通过 `so3_command` 观测项取后 3 维。
        #
        #   同时仍保留 quat_command_w 等 buffer 作为 *内部状态*：
        #   - fixed_goal 需要用冻结的目标姿态计算误差并触发成功重采样
        #   - debug/metrics（以及后续可选的可视化）可能复用这些量
        init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)
        self.pos_command_e = self.object.data.default_root_state[:, :3] + init_pos_offset
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins

        # --- buffers ---
        # 目标姿态（世界系四元数）
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0
        # so(3) 指令（环境系 rotvec）。注意：policy 观测直接读取该量。
        self.phi_ref_e = torch.zeros(self.num_envs, 3, device=self.device)

        # --- logging / metrics buffers ---
        self.success_counter = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_rotation_cmd = torch.zeros(self.num_envs, device=self.device)
        # 以角速度积分近似的“实际旋转量”（episode 内累计），用于轨迹质量评估/过滤
        self.cumulative_rotation_actual = torch.zeros(self.num_envs, device=self.device)

        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cumulative_rotation_cmd"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cumulative_rotation_actual"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["so3_command_norm"] = torch.zeros(self.num_envs, device=self.device)

        # 初次采样
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)

    def __str__(self) -> str:
        msg = "RelativeSO3Command:\n"
        msg += f"\t命令维度: {tuple(self.command.shape[1:])} (pos_command_e, phi_ref_e)\n"
        msg += f"\tmode: {self.cfg.mode}\n"
        msg += f"\ttheta_range: [{self.cfg.theta_min}, {self.cfg.theta_max}] rad"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """返回目标位置（环境系）+ so(3) 指令 rotvec（环境系 {e}）。

        形状: (num_envs, 6)

        Note:
            - 前 3 维为目标位置 pos_command_e，用于约束物体位置
            - 后 3 维为 so(3) 指令 phi_ref_e，表示期望的相对旋转
            - policy 观测侧通过 `so3_command` 观测项仅读取后 3 维
        """

        return torch.cat((self.pos_command_e, self.phi_ref_e), dim=-1)

    # ---------------------------------------------------------------------
    # lifecycle hooks
    # ---------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """重置额外统计量。

        注意：`CommandTerm.reset()` 会调用 `_resample()`，我们不能在 `_resample_command()`
        中无条件清零 `success_counter`，否则 fixed_goal 成功触发的重采样会把计数抹掉。
        因此把“按 episode 重置”的逻辑放在这里。
        """

        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.success_counter[env_ids] = 0.0
        self.cumulative_rotation_cmd[env_ids] = 0.0
        self.cumulative_rotation_actual[env_ids] = 0.0
        return extras

    def _update_metrics(self):
        # orientation error (rad)
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )
        # position error (m)
        self.metrics["position_error"] = torch.norm(self.object.data.root_pos_w - self.pos_command_w, dim=1)

        self.metrics["consecutive_success"] = self.success_counter
        self.metrics["cumulative_rotation_cmd"] = self.cumulative_rotation_cmd
        self.metrics["cumulative_rotation_actual"] = self.cumulative_rotation_actual
        self.metrics["so3_command_norm"] = torch.linalg.norm(self.phi_ref_e, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """采样新的 so(3) 增量指令，并根据 mode 更新/初始化目标姿态。"""

        if len(env_ids) == 0:
            return

        # --- sample phi_ref_e on S^2 with uniform angle ---
        phi = self._sample_rotvec(len(env_ids), self.cfg.theta_min, self.cfg.theta_max)
        self.phi_ref_e[env_ids] = phi

        # 将 rotvec -> quaternion (delta rotation)
        delta_quat = self._quat_from_rotvec(phi)
        base_quat = self.object.data.root_quat_w[env_ids]
        self.quat_command_w[env_ids] = math_utils.quat_mul(delta_quat, base_quat)
        if self.cfg.make_quat_unique:
            self.quat_command_w[env_ids] = math_utils.quat_unique(self.quat_command_w[env_ids])

    def _update_command(self):
        mode = self.cfg.mode.lower()

        # 记录 episode 内“实际旋转量”：用 |omega| * dt 近似积分（对所有模式都成立）
        # 说明：这里不区分方向，仅统计旋转强度/幅值。
        omega_norm = torch.linalg.norm(self.object.data.root_ang_vel_w, dim=-1)
        self.cumulative_rotation_actual += omega_norm * float(self._env.step_dt)

        if mode == "rolling_goal":
            # rolling goal: keep phi_ref fixed, but update the goal quaternion every step
            delta_quat = self._quat_from_rotvec(self.phi_ref_e)
            self.quat_command_w[:] = math_utils.quat_mul(delta_quat, self.object.data.root_quat_w)
            if self.cfg.make_quat_unique:
                self.quat_command_w[:] = math_utils.quat_unique(self.quat_command_w)
            return

        if mode != "fixed_goal":
            raise ValueError(
                f"RelativeSO3Command.cfg.mode must be 'fixed_goal' or 'rolling_goal', got: {self.cfg.mode}"
            )

        # fixed goal: resample when orientation error is sufficiently small
        if not self.cfg.update_goal_on_success:
            return

        success_mask = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        success_ids = success_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(success_ids) == 0:
            return

        # 统计“完成了一次增量指令”的累计旋转量（以 ||phi_ref|| 近似旋转角度）
        self.cumulative_rotation_cmd[success_ids] += torch.linalg.norm(self.phi_ref_e[success_ids], dim=-1)
        self.success_counter[success_ids] += 1.0

        # 触发重采样（同时递增 command_counter，并重置 time_left）
        self._resample(success_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # 参照 IsaacLab 的 InHandReOrientationCommand：用一个 marker 显示内部 goal pose。
        # 注意：即使 command() 只暴露 phi_ref_e，我们仍可用内部 quat_command_w 做可视化。
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "goal_pose_visualizer"):
            return

        marker_pos = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        marker_quat = self.quat_command_w
        self.goal_pose_visualizer.visualize(translations=marker_pos, orientations=marker_quat)

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------

    def _sample_rotvec(self, n: int, theta_min: float, theta_max: float) -> torch.Tensor:
        """采样 rotvec: axis ~ Uniform(S^2), theta ~ Uniform([theta_min, theta_max]).

        采样策略来自方案文档：先高斯采样再归一化得到轴向，角度范围由配置控制。
        """

        # axis: normalize Gaussian vector (avoid division by 0)
        axis = torch.randn((n, 3), device=self.device)
        axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-6)
        axis = axis / axis_norm

        # angle
        theta = torch.empty((n,), device=self.device)
        theta.uniform_(float(theta_min), float(theta_max))

        return axis * theta.unsqueeze(-1)

    def _quat_from_rotvec(self, rotvec: torch.Tensor) -> torch.Tensor:
        """将 rotvec (N,3) 转为 delta quaternion (N,4)。"""

        angle = torch.linalg.norm(rotvec, dim=-1)
        # handle near-zero rotation: axis can be arbitrary when angle=0
        axis = torch.zeros_like(rotvec)
        nonzero = angle > 1e-6
        axis[nonzero] = rotvec[nonzero] / angle[nonzero].unsqueeze(-1)
        axis[~nonzero] = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        return math_utils.quat_from_angle_axis(angle, axis)
