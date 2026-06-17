r"""Executable `gm` axis + SO(3) reorientation command.

本文件把已经与用户对齐的 command 数学语义落成 Isaac Lab `CommandTerm`：
command term 持有目标姿态、旋转轴、SO(3) 误差和 success 统计，使 observation、
reward、curriculum 都读取同一个状态源，而不是在多个 MDP 项中反向猜目标。

核心命令语义：

$$
\mathbf{c}_t =
\left[\hat\omega^{\{h\}},\ \phi_e^{\{h\}}\right]
\in \mathbb{R}^{6},
\qquad
\phi_e = \log\left(R_{goal}R_{current}^{-1}\right)
$$

其中：
    - $\hat\omega^{\{h\}}$：hand semantic frame `{h}` 下的有向单位轴；
    - $\phi_e^{\{h\}}$：space error $\log(R_gR_o^{-1})$ 表达到 `{h}` 后的
      so(3) 向量；
    - policy-facing command 只返回 `[axis_h, error_so3_h]`；
    - reward / termination / curriculum 读取 command term 内部 buffer。

DONE(已合意的第一版设计):
    - teacher 第一版假设 hand root / hand orientation 在 episode 内静态，不做
      gravity-invariant / moving-hand 训练。
    - 使用语义对齐矩阵 $R_{ha}$，含义为 $v^{\{h\}}=R_{ha}v^{\{a\}}$；
      generated assets 默认 $R_{ha}=I$。
    - 默认 `axis_resample_mode="subgoal"`：每个 subgoal 成功后重采样 axis + theta。
    - reset / object falling 后随 env reset 重新采样。
    - axis 有向，theta 只采正幅值；默认 `theta_range=(π/6, π/2)`，下限大于
      success threshold，避免刚采样就成功。
    - 新 goal 以当前 object 姿态为基准左乘：$R_g=\exp([\hat\omega]\theta)R_o$。
    - success 默认用 $SO(3)$ geodesic angle threshold；keypoint success 只作为
      reward/cfg 备选。
    - debug visualization 包括两类 marker：goal object + axis arrow。goal object
      显示当前 subgoal 的目标姿态；axis arrow 显示 axis + error-so(3) command
      中的 axis 方向。
    - `goal_marker_pos_h=(0,0,0.25)`：goal object / axis arrow 同中心，位于
      hand semantic frame `{h}` 的 +z 方向 25cm；该位置只服务可视化，不参与
      reward / termination。
    - axis arrow 固定长度（默认 0.15m），只表达 axis 方向，不表达 $\theta$ 大小；
      $\theta$ 由 goal object 的目标姿态体现。

DONE(已兑现的 buffer / metric):
    - `axis_h`: `[B,3]`，policy-facing 有向单位轴，坐标系 `{h}`。
    - `axis_e`: `[B,3]`，reward-facing 有向单位轴，坐标系 `{e}` / `{w}`。
    - `theta`: `[B]`，当前 subgoal 角度，单位 rad。
    - `goal_quat_w`: `[B,4]`，目标 object orientation，IsaacLab `(w,x,y,z)`。
    - `error_so3_h`: `[B,3]`，policy-facing space error，坐标系 `{h}`。
    - `error_so3_e`: `[B,3]`，reward-facing space error，坐标系 `{e}` / `{w}`。
    - `metrics["orientation_error"]`: `[B]`，$\|\phi_e\|$，单位 rad。
    - `metrics["keypoint_error"]`: `[B]`，orientation-only 六轴向 keypoint distance，单位 m。
    - `metrics["goal_success_count"]`: `[B]`，当前 episode 已完成的 subgoal 数。
    - `metrics["axis_progress"]`: `[B]`，当前 episode 累计未裁剪轴向进度，单位 rad。

DONE(Isaac Lab hook 语义):
    - `reset(env_ids)`：清空 per-episode metrics / axis-progress cache，并触发初始采样。
    - `_resample_command(env_ids)`：根据 `axis_resample_mode` 采样 axis/theta，并生成
      新 goal；不要由 reward 反推 goal。
    - `_update_metrics()`：更新 error buffers 和 logging metrics；不得在这里改变 goal。
    - `_update_command()`：检测 success，递增 `goal_success_count`，再触发 subgoal resample。

DONE(debug visualization 语义):
    - `_set_debug_vis_impl(debug_vis)`：创建 / 隐藏 goal object marker。
    - `_debug_vis_callback(event)`：重复 visualize 当前 `goal_quat_w` buffer；位置由
      `goal_marker_pos_h` 经同一个 `semantic_R_ha` / hand-root pose 转到 world。
    - goal object marker 的 orientation 使用 `goal_quat_w`，代表当前 subgoal
      期望物体达到的真实目标姿态。
    - marker 只在 debug/play 中服务人的视觉理解，不应进入 observation、reward、
      termination，也不应被误解为位置目标。

TODO(debug visualization):
    axis arrow marker 仍预留：它的 orientation 应把 marker 局部 +x 方向旋到当前
    `axis_e`；该 axis 就是 `[axis_h, error_so3_h]` 中的 axis 经 `{h}->{e}` 变换后
    的方向。第一版先补 LEAP 风格 goal object marker，避免为箭头姿态引入额外实现风险。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import ReorientCommandCfg


class ReorientCommand(CommandTerm):
    r"""Axis + SO(3) reorientation command term.

    每个 env 维护一个离散 subgoal：从当前 object 姿态出发，沿 hand semantic
    frame `{h}` 中采样的有向单位轴旋转正角度 $\theta$。policy 只看到：

    $$
    c_t=[\hat\omega^{\{h\}},\phi_e^{\{h\}}]\in\mathbb{R}^6,
    $$

    其中 $\phi_e=\log(R_gR_o^{-1})$ 是 left-increment space error。reward 与
    curriculum 读取内部 `goal_quat_w` / `goal_success_count` 等 buffer，保证目标
    姿态只有一个真实来源。
    """

    cfg: ReorientCommandCfg

    def __init__(self, cfg: ReorientCommandCfg, env: ManagerBasedRLEnv):
        r"""初始化 command buffers。

        Args:
            cfg (ReorientCommandCfg): command 配置。
            env (ManagerBasedRLEnv): Isaac Lab env。
        """

        super().__init__(cfg, env)
        self.object = env.scene[cfg.asset_name]  # 被操作物体，提供当前 $R_o$
        self.robot = env.scene[cfg.robot_asset_name]  # 手部 articulation，提供 `{a}` root 姿态

        # `semantic_R_ha` 是 row-major $R_{ha}$，即 $v^h=R_{ha}v^a$。
        self.semantic_R_ha = torch.tensor(cfg.semantic_R_ha, dtype=torch.float32, device=self.device).reshape(3, 3)
        det = torch.det(self.semantic_R_ha)  # 旋转矩阵行列式，理想为 1
        ortho_err = torch.linalg.norm(self.semantic_R_ha @ self.semantic_R_ha.T - torch.eye(3, device=self.device))
        if torch.abs(det - 1.0) > 1.0e-3 or ortho_err > 1.0e-3:
            raise ValueError(
                "ReorientCommandCfg.semantic_R_ha must be an SO(3) matrix; "
                f"got det={float(det):.6f}, orthogonality_error={float(ortho_err):.6f}."
            )

        self.axis_h = torch.zeros(self.num_envs, 3, device=self.device)  # `[B,3]`，policy-facing 轴
        self.axis_e = torch.zeros(self.num_envs, 3, device=self.device)  # `[B,3]`，world/env 轴
        self.theta = torch.zeros(self.num_envs, device=self.device)  # `[B]`，当前 subgoal 角度 rad
        self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)  # `[B,4]`，目标姿态 quaternion
        self.goal_quat_w[:, 0] = 1.0  # 单位 quaternion，避免 reset 前未初始化
        self.error_so3_e = torch.zeros(self.num_envs, 3, device=self.device)  # `[B,3]`，world/env error rotvec
        self.error_so3_h = torch.zeros(self.num_envs, 3, device=self.device)  # `[B,3]`，hand semantic error rotvec

        self.goal_success_count = torch.zeros(self.num_envs, device=self.device)  # episode 内完成 subgoal 数
        self.axis_progress = torch.zeros(self.num_envs, device=self.device)  # episode 内累计目标角度进度 rad

        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["keypoint_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_success_count"] = self.goal_success_count
        self.metrics["axis_progress"] = self.axis_progress

        env_ids = torch.arange(self.num_envs, device=self.device)  # 初始化所有 env 的第一组 subgoal
        self._resample_command(env_ids)

    @property
    def command(self) -> torch.Tensor:
        r"""返回 policy-facing command `[axis_h, error_so3_h]`。

        Returns:
            torch.Tensor: command tensor，形状 `[num_envs, 6]`，前三维是 hand frame
            有向单位轴，后三维是 hand frame SO(3) error rotvec，单位 rad。
        """

        return torch.cat((self.axis_h, self.error_so3_h), dim=-1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        r"""按 episode 重置 success/progress 统计并采样初始 subgoal。

        Args:
            env_ids (Sequence[int] | None): 需要 reset 的 env ids；`None` 表示全部。

        Returns:
            dict[str, float]: Isaac Lab command manager 日志 extras。
        """

        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.goal_success_count[env_ids] = 0.0  # episode 级统计，success-driven resample 不清零
        self.axis_progress[env_ids] = 0.0  # episode 内累计目标角度，reset 时归零
        return extras

    def _update_metrics(self):
        r"""更新 SO(3) error 与日志指标。

        这里不改变 goal，只计算当前 $R_gR_o^{-1}$ 的 logarithm，避免 metrics hook
        与 command update hook 之间产生隐式状态副作用。
        """

        current_quat_w = self.object.data.root_quat_w  # `[B,4]`，当前 object 姿态
        quat_error = math_utils.quat_mul(self.goal_quat_w, math_utils.quat_inv(current_quat_w))  # $R_gR_o^{-1}$
        if self.cfg.make_quat_unique:
            quat_error = math_utils.quat_unique(quat_error)  # 只影响 log 连续性，不改变 SO(3) 姿态
        self.error_so3_e[:] = math_utils.axis_angle_from_quat(quat_error)  # `[B,3]`，world/env rotvec
        self.error_so3_h[:] = self._vector_e_to_h(self.error_so3_e)  # `[B,3]`，hand semantic rotvec

        orientation_error = torch.linalg.norm(self.error_so3_e, dim=-1)  # `[B]`，rad
        self.metrics["orientation_error"][:] = orientation_error
        self.metrics["keypoint_error"][:] = self._orientation_keypoint_distance()
        self.metrics["goal_success_count"] = self.goal_success_count
        self.metrics["axis_progress"] = self.axis_progress

    def _resample_command(self, env_ids: Sequence[int]):
        r"""为指定 env 采样 axis/theta 并生成新目标姿态。

        Args:
            env_ids (Sequence[int]): 需要采样新 subgoal 的 env ids。
        """

        env_ids = self._as_env_id_tensor(env_ids)  # 支持 list / tensor / slice(None)
        if env_ids.numel() == 0:
            return

        # episode 模式下，success-driven resample 只换 theta，不换 axis；reset 后 command_counter 为 0 时才采 axis。
        should_sample_axis = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
        if self.cfg.axis_resample_mode == "episode":
            should_sample_axis = self.command_counter[env_ids] == 0  # reset 初采样换轴；success 后只换 theta
        if self.cfg.axis_mode == "fixed":
            should_sample_axis[:] = True  # fixed axis 每次都写入同一个值，保证配置修改后状态不陈旧

        if should_sample_axis.any():
            target_ids = env_ids[should_sample_axis]
            self.axis_h[target_ids] = self._sample_axis_h(target_ids.numel())  # `[K,3]`，hand frame 单位轴

        theta_min, theta_max = self.cfg.theta_range
        self.theta[env_ids] = math_utils.sample_uniform(theta_min, theta_max, (env_ids.numel(),), self.device)  # rad
        self.axis_e[env_ids] = self._vector_h_to_e(self.axis_h[env_ids], env_ids)  # `{h}` 轴转 `{e}`/world

        delta_quat = math_utils.quat_from_angle_axis(self.theta[env_ids], self.axis_e[env_ids])  # $\exp(\hat\omega\theta)$
        base_quat = self.object.data.root_quat_w[env_ids]  # 当前 object 姿态 $R_o$
        self.goal_quat_w[env_ids] = math_utils.quat_mul(delta_quat, base_quat)  # $R_g=\exp([\omega]\theta)R_o$
        if self.cfg.make_quat_unique:
            self.goal_quat_w[env_ids] = math_utils.quat_unique(self.goal_quat_w[env_ids])

        # 立即刷新被 resample env 的 command error，避免 reset 后第一帧 policy 看到旧误差。
        quat_error = math_utils.quat_mul(self.goal_quat_w[env_ids], math_utils.quat_inv(base_quat))
        self.error_so3_e[env_ids] = math_utils.axis_angle_from_quat(quat_error)
        self.error_so3_h[env_ids] = self._vector_e_to_h(self.error_so3_e[env_ids], env_ids)

    def _update_command(self):
        r"""检测成功并触发下一 subgoal。

        Isaac Lab 调用顺序是 `_update_metrics()` 后 `_update_command()`，因此这里
        可以直接使用刚更新的 `metrics["orientation_error"]`。
        """

        success_mask = self.metrics["orientation_error"] < float(self.cfg.orientation_success_threshold)
        success_ids = success_mask.nonzero(as_tuple=False).flatten()
        if success_ids.numel() == 0:
            return

        self.goal_success_count[success_ids] += 1.0  # curriculum 读取的 episode-level 成功数
        self.axis_progress[success_ids] += self.theta[success_ids]  # 累计目标角度，单位 rad
        self._resample_command(success_ids)  # 默认 subgoal 模式会采新 axis + theta
        self.command_counter[success_ids] += 1  # success-driven subgoal 也计入 command counter
        self.time_left[success_ids] = self.cfg.resampling_time_range[1]  # 继续禁用时间驱动重采样

    def _set_debug_vis_impl(self, debug_vis: bool):
        r"""创建或隐藏目标姿态 marker。

        该 marker 对齐 LEAP 参考实现的“虚拟目标物体”语义：它只展示 command term
        内部的目标姿态 $R_g$，不参与物理碰撞、不改变 reward，也不作为位置目标。
        """

        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)  # USD goal cube marker
            self.goal_pose_visualizer.set_visibility(True)  # 由 CommandTerm callback 每帧写入 pose
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)  # 保留对象但隐藏，便于后续重新打开

    def _debug_vis_callback(self, event):
        r"""把当前目标姿态 buffer 写到虚拟目标物体 marker。

        位置使用 `goal_marker_pos_h`，即 hand semantic frame `{h}` 中的纯显示偏置；
        姿态使用 `goal_quat_w`，即当前 subgoal 真实要求的 object orientation。
        """

        if not hasattr(self, "goal_pose_visualizer"):
            return

        marker_offset_h = torch.tensor(self.cfg.goal_marker_pos_h, dtype=torch.float32, device=self.device).reshape(1, 3)  # $p^h_{marker}$，只服务可视化
        marker_offset_a = marker_offset_h @ self.semantic_R_ha  # $p^a_{marker}=R_{ha}^\top p^h$ 的 row-vector 写法
        marker_offset_w = math_utils.quat_apply(self.robot.data.root_quat_w, marker_offset_a.repeat(self.num_envs, 1))  # $R_{wa}p^a$
        marker_pos_w = self.robot.data.root_pos_w + marker_offset_w  # `[B,3]`，每个 env 中 marker 的世界位置
        self.goal_pose_visualizer.visualize(translations=marker_pos_w, orientations=self.goal_quat_w)  # 姿态目标来自 command 单一真源

    def _as_env_id_tensor(self, env_ids: Sequence[int] | slice | torch.Tensor) -> torch.Tensor:
        r"""把 Isaac Lab 传入的 env id 表达统一成 LongTensor。"""

        if isinstance(env_ids, slice):
            if env_ids == slice(None):
                return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _sample_axis_h(self, count: int) -> torch.Tensor:
        r"""在 hand semantic frame `{h}` 中采样单位旋转轴。

        Args:
            count (int): 需要采样的轴数量。

        Returns:
            torch.Tensor: 单位轴，形状 `[count,3]`。
        """

        if self.cfg.axis_mode == "fixed":
            axis = torch.tensor(self.cfg.fixed_axis_h, dtype=torch.float32, device=self.device).reshape(1, 3)
            norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
            if torch.any(norm < 1.0e-6):
                raise ValueError("ReorientCommandCfg.fixed_axis_h must be non-zero.")
            return (axis / norm).repeat(count, 1)  # `[count,3]`，固定有向轴

        axis = torch.randn(count, 3, device=self.device)  # 高斯归一化给 $S^2$ 上均匀方向
        return axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + 1.0e-6)

    def _vector_h_to_e(self, vec_h: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        r"""把 hand semantic frame `{h}` 向量转到 env/world frame。

        采用配置约定 $v^h=R_{ha}v^a$，因此 $v^a=R_{ha}^\top v^h$；再由 robot
        root 姿态 $R_{ea}$ 转到 `{e}` / `{w}`。当前 cloned env 与 world 只差平移，
        旋转同向，因此 vector 表达可视为 world frame。
        """

        root_quat_w = self.robot.data.root_quat_w if env_ids is None else self.robot.data.root_quat_w[env_ids]  # `[B/K,4]`，robot root `{a}` 到 world
        vec_a = vec_h @ self.semantic_R_ha  # row vector: $v^a = v^h R_{ha}$
        return math_utils.quat_apply(root_quat_w, vec_a)  # $v^e = R_{ea}v^a$

    def _vector_e_to_h(self, vec_e: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        r"""把 env/world frame 向量转回 hand semantic frame `{h}`。"""

        root_quat_w = self.robot.data.root_quat_w if env_ids is None else self.robot.data.root_quat_w[env_ids]  # `[B/K,4]`，robot root `{a}` 到 world
        vec_a = math_utils.quat_apply_inverse(root_quat_w, vec_e)  # $v^a=R_{ea}^{-1}v^e$
        return vec_a @ self.semantic_R_ha.T  # row vector: $v^h=v^aR_{ha}^T$

    def _orientation_keypoint_distance(self) -> torch.Tensor:
        r"""计算 command metric 使用的 orientation-only 六轴向 keypoint distance。"""

        radius = float(self.cfg.keypoint_radius)  # meter，AnyRotate 5cm 数值锚点
        keypoints_o = torch.tensor(
            [[radius, 0.0, 0.0], [-radius, 0.0, 0.0], [0.0, radius, 0.0], [0.0, -radius, 0.0], [0.0, 0.0, radius], [0.0, 0.0, -radius]],
            dtype=torch.float32,
            device=self.device,
        )
        current_rot_w = math_utils.matrix_from_quat(self.object.data.root_quat_w)  # `[B,3,3]`，当前 $R_o$
        goal_rot_w = math_utils.matrix_from_quat(self.goal_quat_w)  # `[B,3,3]`，目标 $R_g$
        current_points = torch.einsum("bij,kj->bki", current_rot_w, keypoints_o)  # `[B,6,3]`，$R_op_i$
        goal_points = torch.einsum("bij,kj->bki", goal_rot_w, keypoints_o)  # `[B,6,3]`，$R_gp_i$
        return torch.linalg.norm(current_points - goal_points, dim=-1).mean(dim=-1)  # `[B]`，单位 m


__all__ = ["ReorientCommand"]
