r"""Fixed-axis palm-supported tactile rotation command 与实际旋转状态 owner。

本 command 与旧 `ReorientCommand` 独立：旧类训练 random-axis reorientation；本类固定 hand semantic
法向轴并把连续旋转拆成 30 degree moving subgoals。它也是相邻姿态进度的唯一 owner：

$$
\Delta\psi_t=\log(R_tR_{t-1}^{-1})^{\vee\mathsf T}\hat k^w,
\qquad \Psi_t=\sum_j\Delta\psi_j.
$$

`ManagerBasedRLEnv` 在普通 command update 前先计算 termination/reward，因此
`ensure_post_physics_progress_updated()` 以 `common_step_counter` 加戳，允许本 step 第一个 consumer
立即刷新，后续 consumer 复用同一快照。普通 `_update_command()` 只在 reward 完成后推进 goal。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..tactile_diagnostics_state import GmTactileEpisodeDiagnostics
    from .commands_cfg import TactileRotationCommandCfg


def projected_space_rotation_delta(
    previous_quat_w: torch.Tensor,
    current_quat_w: torch.Tensor,
    axis_w: torch.Tensor,
) -> torch.Tensor:
    r"""计算相邻 object 姿态绕 world-frame 有向空间轴的未裁剪增量。

    使用矩阵构造 $R_tR_{t-1}^{-1}$，因此输入 quaternion 同时翻转符号时输出不变。
    这里采用 $SO(3)$ principal logarithm；20 Hz 下物体单步旋转远小于 $\pi$，不会跨
    branch cut。返回值允许为负，往返运动会在净旋转中相消。

    Args:
        previous_quat_w (torch.Tensor): 上一 policy frame 姿态 `[B,4]`，`(w,x,y,z)`。
        current_quat_w (torch.Tensor): 当前 post-physics 姿态 `[B,4]`。
        axis_w (torch.Tensor): world-frame 有向单位轴 `[B,3]`。

    Returns:
        torch.Tensor: signed projected delta `[B]`，单位 rad，未裁剪。
    """

    previous_rot_w = math_utils.matrix_from_quat(previous_quat_w)  # $R_{t-1}$
    current_rot_w = math_utils.matrix_from_quat(current_quat_w)  # $R_t$
    delta_rot_w = current_rot_w @ previous_rot_w.transpose(-1, -2)  # space increment $R_tR_{t-1}^{-1}$
    delta_quat_w = math_utils.quat_from_matrix(delta_rot_w)  # matrix route 消除 input quaternion 符号双覆盖
    delta_rotvec_w = math_utils.axis_angle_from_quat(delta_quat_w)  # principal $\log(\Delta R)^\vee$
    normalized_axis_w = axis_w / (torch.linalg.norm(axis_w, dim=-1, keepdim=True) + 1.0e-8)
    return torch.sum(delta_rotvec_w * normalized_axis_w, dim=-1)  # `[B]`，有向轴向真实转角


def orientation_keypoint_distance_from_quats(
    current_quat_w: torch.Tensor,
    goal_quat_w: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    r"""计算中心对齐的六轴向 orientation-only keypoint 平均距离。"""

    keypoints_o = torch.tensor(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=current_quat_w.dtype,
        device=current_quat_w.device,
    )
    current_rot_w = math_utils.matrix_from_quat(current_quat_w)  # `[B,3,3]`
    goal_rot_w = math_utils.matrix_from_quat(goal_quat_w)  # `[B,3,3]`
    current_points = torch.einsum("bij,kj->bki", current_rot_w, keypoints_o)  # center-aligned $R_or_i$
    goal_points = torch.einsum("bij,kj->bki", goal_rot_w, keypoints_o)  # center-aligned $R_gr_i$
    return torch.linalg.norm(current_points - goal_points, dim=-1).mean(dim=-1)  # `[B]`，单位 m


class TactileRotationCommand(CommandTerm):
    r"""固定 `{h}` z 轴、30 degree moving-subgoal tactile rotation command。"""

    cfg: TactileRotationCommandCfg

    def __init__(self, cfg: TactileRotationCommandCfg, env: ManagerBasedRLEnv):
        r"""分配 goal、anchor、rotation progress 和 speed buffers。"""

        super().__init__(cfg, env)
        self._env = env  # lazy post-physics update 需要 `common_step_counter` 与 `step_dt`
        self.object = env.scene[cfg.asset_name]  # object pose/velocity canonical source
        self.robot = env.scene[cfg.robot_asset_name]  # `{a}->{w}` hand root pose
        self.semantic_R_ha = torch.tensor(cfg.semantic_R_ha, dtype=torch.float32, device=self.device).reshape(3, 3)
        identity = torch.eye(3, dtype=torch.float32, device=self.device)
        det = torch.det(self.semantic_R_ha)
        ortho_error = torch.linalg.norm(self.semantic_R_ha @ self.semantic_R_ha.T - identity)
        if torch.abs(det - 1.0) > 1.0e-3 or ortho_error > 1.0e-3:
            raise ValueError(
                "TactileRotationCommandCfg.semantic_R_ha must be an SO(3) matrix; "
                f"got det={float(det):.6f}, orthogonality_error={float(ortho_error):.6f}."
            )

        axis_h = torch.tensor(cfg.fixed_axis_h, dtype=torch.float32, device=self.device).reshape(1, 3)
        axis_norm = torch.linalg.norm(axis_h, dim=-1, keepdim=True)
        if torch.any(axis_norm < 1.0e-6):
            raise ValueError("TactileRotationCommandCfg.fixed_axis_h must be non-zero.")
        self.axis_h = (axis_h / axis_norm).repeat(self.num_envs, 1)  # `[B,3]`，baseline 固定 `(0,0,1)`
        self.axis_w = torch.zeros(self.num_envs, 3, device=self.device)  # `[B,3]`，随 hand root pose 变换
        self.goal_quat_w = _identity_quat(self.num_envs, self.device)  # `[B,4]`，当前 30 degree subgoal
        self.error_so3_h = torch.zeros(self.num_envs, 3, device=self.device)  # command-facing local error
        self.position_anchor_w = torch.zeros(self.num_envs, 3, device=self.device)  # episode reset object position
        self.previous_quat_w = _identity_quat(self.num_envs, self.device)  # progress 唯一上一姿态缓存
        self.has_previous = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_progress_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        self.delta_psi = torch.zeros(self.num_envs, device=self.device)  # 当前 policy step 未裁剪 signed rad
        self.net_rotation_rad = torch.zeros(self.num_envs, device=self.device)  # episode signed $\Psi$
        self.net_rotation_turns = torch.zeros(self.num_envs, device=self.device)  # positive competence turns
        self.axis_speed = torch.zeros(self.num_envs, device=self.device)  # 未裁剪瞬时 rad/s
        self.axis_speed_ema = torch.zeros(self.num_envs, device=self.device)  # $\tau=0.25$s low-pass speed
        self.orientation_keypoint_error = torch.zeros(self.num_envs, device=self.device)  # center-aligned m
        self.position_error = torch.zeros(self.num_envs, device=self.device)  # anchor distance m
        self.goal_normal_alignment = torch.ones(self.num_envs, device=self.device)  # signed $z_o^Tz_g$
        self.goal_success_pulse = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_success_count = torch.zeros(self.num_envs, device=self.device)
        self.diagnostics: GmTactileEpisodeDiagnostics | None = None  # managers 完整构造后再解析 contact/termination

        self.metrics["delta_psi"] = self.delta_psi
        self.metrics["net_rotation_rad"] = self.net_rotation_rad
        self.metrics["net_rotation_turns"] = self.net_rotation_turns
        self.metrics["axis_speed"] = self.axis_speed
        self.metrics["axis_speed_ema"] = self.axis_speed_ema
        self.metrics["orientation_keypoint_error"] = self.orientation_keypoint_error
        self.metrics["position_error"] = self.position_error
        self.metrics["goal_normal_alignment"] = self.goal_normal_alignment
        self.metrics["goal_success_count"] = self.goal_success_count

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._capture_reset_state(env_ids)  # constructor 后 observation 不读取未初始化 anchor/previous pose
        self._resample_command(env_ids)  # 第一组 30 degree goal 从当前 object orientation 左乘生成

    @property
    def command(self) -> torch.Tensor:
        r"""返回 hand-frame relative goal rotvec；部署 actor 可选择完全不观察该 tensor。"""

        return self.error_so3_h  # `[B,3]`，只满足 CommandManager 接口，不进入固定 52D actor frame

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        r"""记录刚结束 episode metrics，随后清 partial env 的 anchor/progress 并生成首个 goal。"""

        ids = self._as_env_id_tensor(slice(None) if env_ids is None else env_ids)
        diagnostics = self._ensure_diagnostics_initialized()
        if diagnostics is not None:
            diagnostics.capture_terminal(self._env, ids)  # 必须先冻结 terminal causes，再由 super 取 subset mean
        cell_extras = self._morphology_cell_extras(ids)  # 必须在super清零per-env metrics前分组
        extras = super().reset(env_ids)  # 先记录旧 episode metrics；内部会按当前 pose resample goal
        extras.update(cell_extras)
        self._capture_reset_state(ids)  # object reset event 已执行，此处读取本 episode 真实 reset pose
        self._refresh_goal_errors(_mask_from_ids(self.num_envs, ids, self.device))  # 新 anchor 下同步 success/termination 双门
        if diagnostics is not None:
            diagnostics.reset(self._env, ids)  # super 已清旧 metrics；此处写入新 episode actual ADR snapshot
        return extras

    def _morphology_cell_extras(self, ids: torch.Tensor) -> dict[str, float]:
        r"""在reset subset内按固定八组聚合关键episode metrics。

        Cell metadata只用于logging，不进入actor。某次reset没有该cell时不生成key，避免把缺样本误写为0。
        """

        cell_ids = getattr(self._env, "_anymani_morphology_cell_id", None)
        if not isinstance(cell_ids, torch.Tensor) or cell_ids.shape != (self.num_envs,):
            return {}
        labels = (
            "left_tips3_thumb3dof",
            "left_tips3_thumb4dof",
            "left_tips4_thumb3dof",
            "left_tips4_thumb4dof",
            "right_tips3_thumb3dof",
            "right_tips3_thumb4dof",
            "right_tips4_thumb3dof",
            "right_tips4_thumb4dof",
        )
        metric_names = (
            "goal_success_count",
            "net_rotation_turns",
            "position_error",
            "contact/tip_active_count_mean",
            "contact/finger_non_tip_occupancy_fraction",
            "termination/object_out_of_anchor_fraction",
        )
        extras: dict[str, float] = {}
        reset_cells = cell_ids[ids]
        for cell_id, label in enumerate(labels):
            member_ids = ids[reset_cells == cell_id]
            extras[f"cell/{label}/episode_count"] = float(member_ids.numel())
            for metric_name in metric_names:
                metric = self.metrics.get(metric_name)
                if isinstance(metric, torch.Tensor):
                    extras[f"cell/{label}/{metric_name}_sum"] = (
                        float(metric[member_ids].sum().item()) if member_ids.numel() > 0 else 0.0
                    )
        return extras

    def ensure_post_physics_progress_updated(self, env: ManagerBasedRLEnv | None = None) -> None:
        r"""幂等刷新本 policy step 的 actual rotation、speed、success 与 termination buffers。"""

        env = self._env if env is None else env
        step = int(env.common_step_counter)
        update_mask = self.last_progress_step != step
        if not torch.any(update_mask):
            return

        self.axis_w[update_mask] = self._vector_h_to_w(self.axis_h[update_mask], update_mask.nonzero().flatten())
        current_quat_w = self.object.data.root_quat_w  # `[B,4]`，post-physics object orientation
        valid = update_mask & self.has_previous
        self.delta_psi[update_mask] = 0.0
        if torch.any(valid):
            self.delta_psi[valid] = projected_space_rotation_delta(
                self.previous_quat_w[valid], current_quat_w[valid], self.axis_w[valid]
            )
        self.net_rotation_rad[update_mask] += self.delta_psi[update_mask]  # 未裁剪 signed accumulation
        self.net_rotation_turns[update_mask] = torch.clamp(self.net_rotation_rad[update_mask], min=0.0) / (2.0 * math.pi)
        self.axis_speed[update_mask] = self.delta_psi[update_mask] / float(env.step_dt)  # rad/s
        speed_alpha = 1.0 - math.exp(-float(env.step_dt) / float(self.cfg.speed_ema_time_constant_s))
        self.axis_speed_ema[update_mask] = (
            (1.0 - speed_alpha) * self.axis_speed_ema[update_mask] + speed_alpha * self.axis_speed[update_mask]
        )

        self.previous_quat_w[update_mask] = current_quat_w[update_mask].detach()
        self.has_previous[update_mask] = True
        self._refresh_goal_errors(update_mask)
        self.goal_success_pulse[update_mask] = (
            (self.orientation_keypoint_error[update_mask] < float(self.cfg.orientation_keypoint_success_threshold))
            & (self.position_error[update_mask] < float(self.cfg.position_success_threshold))
        )
        diagnostics = self._ensure_diagnostics_initialized()
        if diagnostics is not None:
            diagnostics.ensure_updated(
                env,
                axis_w=self.axis_w,
                axis_speed=self.axis_speed,
                object_ang_vel_w=self.object.data.root_ang_vel_w,
                joint_position=self.robot.data.joint_pos,
                position_error=self.position_error,
                orientation_keypoint_error=self.orientation_keypoint_error,
            )
        self.last_progress_step[update_mask] = step

    def _update_metrics(self) -> None:
        r"""普通 command hook 复用 reward 阶段已生成的 snapshot；若无人读取则在此补刷新。"""

        self.ensure_post_physics_progress_updated(self._env)

    def _update_command(self) -> None:
        r"""reward 计算完成后消费 success pulse，并从当前 object pose 生成下一 30 degree goal。"""

        success_ids = self.goal_success_pulse.nonzero(as_tuple=False).flatten()
        if success_ids.numel() == 0:
            return
        self.goal_success_count[success_ids] += 1.0
        self._resample_command(success_ids)  # 不从旧 goal 累乘，避免执行误差积累到 reference
        self.command_counter[success_ids] += 1
        self.time_left[success_ids] = self.cfg.resampling_time_range[1]
        self.goal_success_pulse[success_ids] = False  # 防御重复 `_update_command`；reward 已消费本 step pulse

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        r"""从当前 object orientation 左乘固定 30 degree space rotation，生成 moving goal。"""

        ids = self._as_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        self.axis_w[ids] = self._vector_h_to_w(self.axis_h[ids], ids)
        angle = torch.full((ids.numel(),), float(self.cfg.subgoal_angle), device=self.device)
        delta_quat_w = math_utils.quat_from_angle_axis(angle, self.axis_w[ids])  # $\Exp([k^w]\pi/6)$
        current_quat_w = self.object.data.root_quat_w[ids]
        self.goal_quat_w[ids] = math_utils.quat_mul(delta_quat_w, current_quat_w)  # left multiplication
        if self.cfg.make_quat_unique:
            self.goal_quat_w[ids] = math_utils.quat_unique(self.goal_quat_w[ids])
        self._refresh_goal_errors(_mask_from_ids(self.num_envs, ids, self.device))

    def _capture_reset_state(self, env_ids: torch.Tensor) -> None:
        r"""捕获本 episode 不变的位置 anchor，并清 actual-rotation lifecycle buffers。"""

        self.position_anchor_w[env_ids] = self.object.data.root_pos_w[env_ids].detach()
        self.previous_quat_w[env_ids] = self.object.data.root_quat_w[env_ids].detach()
        self.has_previous[env_ids] = True  # 下一 physics step 的 delta 从 reset pose 开始
        self.delta_psi[env_ids] = 0.0
        self.net_rotation_rad[env_ids] = 0.0
        self.net_rotation_turns[env_ids] = 0.0
        self.axis_speed[env_ids] = 0.0
        self.axis_speed_ema[env_ids] = 0.0
        self.goal_success_pulse[env_ids] = False
        self.goal_success_count[env_ids] = 0.0
        self.last_progress_step[env_ids] = int(self._env.common_step_counter)  # reset observation 当前 stamp 不累计伪 delta

    def _refresh_goal_errors(self, update_mask: torch.Tensor) -> None:
        r"""更新 success 双门、policy command error 与 signed normal alignment。"""

        if not torch.any(update_mask):
            return
        current_quat_w = self.object.data.root_quat_w[update_mask]
        goal_quat_w = self.goal_quat_w[update_mask]
        self.orientation_keypoint_error[update_mask] = orientation_keypoint_distance_from_quats(
            current_quat_w, goal_quat_w, float(self.cfg.keypoint_radius)
        )
        self.position_error[update_mask] = torch.linalg.norm(
            self.object.data.root_pos_w[update_mask] - self.position_anchor_w[update_mask], dim=-1
        )
        current_rot_w = math_utils.matrix_from_quat(current_quat_w)
        goal_rot_w = math_utils.matrix_from_quat(goal_quat_w)
        self.goal_normal_alignment[update_mask] = torch.sum(current_rot_w[:, :, 2] * goal_rot_w[:, :, 2], dim=-1)

        quat_error_w = math_utils.quat_mul(goal_quat_w, math_utils.quat_inv(current_quat_w))  # $R_gR_o^{-1}$
        error_so3_w = math_utils.axis_angle_from_quat(quat_error_w)
        ids = update_mask.nonzero(as_tuple=False).flatten()
        self.error_so3_h[update_mask] = self._vector_w_to_h(error_so3_w, ids)

    def _vector_h_to_w(self, vector_h: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        r"""按 $v^a=R_{ha}^Tv^h$、$v^w=R_{wa}v^a$ 把 hand semantic vector 转 world。"""

        vector_a = vector_h @ self.semantic_R_ha  # row-vector form of $R_{ha}^T$
        return math_utils.quat_apply(self.robot.data.root_quat_w[env_ids], vector_a)

    def _vector_w_to_h(self, vector_w: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        r"""把 world vector 旋回 hand semantic frame。"""

        vector_a = math_utils.quat_apply_inverse(self.robot.data.root_quat_w[env_ids], vector_w)
        return vector_a @ self.semantic_R_ha.T

    def _as_env_id_tensor(self, env_ids: Sequence[int] | slice | torch.Tensor) -> torch.Tensor:
        r"""把 CommandTerm 的 list/slice/tensor env ids 统一到 env-device LongTensor。"""

        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _ensure_diagnostics_initialized(self) -> GmTactileEpisodeDiagnostics | None:
        r"""在所有 env managers 可用后初始化 episode diagnostics，并注册其 metric tensors。"""

        diagnostics = getattr(self, "diagnostics", None)  # tensor-only command contracts 绕过正式 constructor
        if diagnostics is not None:
            return self.diagnostics
        cfg = self.cfg
        if not (
            getattr(cfg, "diagnostics_fingertip_sensor_names", ())
            and getattr(cfg, "diagnostics_finger_non_tip_sensor_names", ())
            and getattr(cfg, "diagnostics_palm_sensor_name", "")
        ):
            return None
        if not hasattr(self._env, "termination_manager") or not hasattr(self._env, "action_manager"):
            return None  # CommandTerm constructor 阶段 managers 可能尚未完成装配
        from ..tactile_diagnostics_state import GmTactileEpisodeDiagnostics

        self.diagnostics = GmTactileEpisodeDiagnostics(
            self._env,
            fingertip_sensor_names=cfg.diagnostics_fingertip_sensor_names,
            finger_non_tip_sensor_names=cfg.diagnostics_finger_non_tip_sensor_names,
            palm_sensor_name=cfg.diagnostics_palm_sensor_name,
            action_name=cfg.diagnostics_action_name,
            ema_alpha=cfg.diagnostics_contact_ema_alpha,
            force_threshold=cfg.diagnostics_contact_force_threshold,
        )
        self.metrics.update(self.diagnostics.metrics)  # CommandTermBase.reset 统一输出并清 partial env rows
        return self.diagnostics


def ensure_post_physics_progress_updated(env: ManagerBasedRLEnv, command_name: str) -> TactileRotationCommand:
    r"""MDP consumer adapter：解析 tactile command，并幂等刷新当前 post-physics state。"""

    command_term = env.command_manager.get_term(command_name)
    if not isinstance(command_term, TactileRotationCommand):
        raise TypeError(f"Command '{command_name}' must be TactileRotationCommand, got {type(command_term).__name__}.")
    command_term.ensure_post_physics_progress_updated(env)
    return command_term


def _identity_quat(num_envs: int, device: torch.device | str) -> torch.Tensor:
    r"""构造 Isaac Lab `(w,x,y,z)` identity quaternion batch。"""

    quat = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
    quat[:, 0] = 1.0
    return quat


def _mask_from_ids(num_envs: int, env_ids: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    r"""把 partial env ids 转成 `[B]` bool mask，供同一批 buffer 一致更新。"""

    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask[env_ids] = True
    return mask


__all__ = [
    "TactileRotationCommand",
    "ensure_post_physics_progress_updated",
    "orientation_keypoint_distance_from_quats",
    "projected_space_rotation_delta",
]
