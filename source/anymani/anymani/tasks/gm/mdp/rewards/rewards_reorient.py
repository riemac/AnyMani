r"""Reorientation reward terms for GM in-hand manipulation.

`r_reorient` 是主任务项，采用随机重定向子目标来训练可复合的连续手内旋转 primitive。
最终绕 `{h}` 轴连续旋转可视为多个重定向子目标的复合：
$$
T_{ab}T_{bc}=T_{ac},\qquad R_g=\exp([\hat\omega]\theta)R_o.
$$

DONE(本轮已合意的 reward 语义): 第一版以 AnyRotate 风格的 keypoint distance reward
为主，但 command / success 的数学语义默认仍采用 $SO(3)$ geodesic threshold；keypoints
使用 object body frame `{o}` 下的 $\pm x,\pm y,\pm z$ 六个轴向点，半径默认 $5\,\text{cm}$。

TODO(tactile rotation replacement semantics):
    新 single-asset tactile rotation baseline 不再使用本文件当前的 orientation-only dense reward
    作为主 pose 项。rotation group 应组合：

    $$
    r_{rotation}
    =
    \lambda_{kp}r_{kp}^{full-pose}
    +
    \lambda_{rot}r_{rot}^{axis-delta}
    +
    \lambda_{goal}r_{goal}.
    $$

    `r_kp` 替代旧 official 的独立 position/orientation dense terms；`r_rot` 读取 command-owned
    actual delta angle，reward 内裁剪到 0.025 rad；`r_goal` 使用 orientation-only keypoint
    threshold 5 mm 与 anchor position threshold 25 mm 的双门，第一版 impulse 权重为 10。

    当前 `AxisDeltaRotationReward` 自己缓存上一姿态，与新 command-owned progress contract 冲突。
    build 阶段应迁移 consumer 后出清该重复状态 owner，不得让两个版本长期并存。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from ..commands.tactile_rotation_command import ensure_post_physics_progress_updated
from .rewards_common import (
    full_pose_keypoint_distances,
    impulse_to_rate,
    normalized_keypoint_kernel,
    orientation_keypoint_distance,
    resolve_axis_e,
    resolve_goal_quat_w,
)


def reorientation_reward_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""临时占位：保持早期 cfg 可导入，但不提供真实任务奖励。

    TODO: 正式训练不能使用该项。env cfg 应使用 `keypoint_reorientation_reward`、
    `AxisDeltaRotationReward`、`goal_success_bonus` 等明确 reward terms。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。

    Returns:
        torch.Tensor: 全零 reward，形状 `[num_envs]`。
    """

    return torch.zeros(env.num_envs, device=env.device)  # 占位项，不改变任何训练梯度信号


def keypoint_reorientation_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    keypoint_radius: float = 0.05,
    curve_sharpness: float = 50.0,
    curve_bias: float = 2.0,
) -> torch.Tensor:
    r"""AnyRotate 风格的 orientation-only keypoint distance reward。

    第一版使用 `{o}` 下六个轴向 keypoints，并只比较姿态：
    $$
    d_{kp}=\frac{1}{6}\sum_{i=1}^{6}\left\|R_o p_i^{\{o\}}-R_g p_i^{\{o\}}\right\|_2.
    $$

    reward 曲线采用 AnyRotate Appendix B 的 squashed distance reward 思路，写成归一化版本：
    $$
    r_{kp}=\frac{2+b}{\exp(a d_{kp})+b+\exp(-a d_{kp})}.
    $$

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): 提供目标姿态的 command term 名称。
        object_cfg (SceneEntityCfg): object asset 配置，默认 `SceneEntityCfg("object")`。
        keypoint_radius (float): 六轴向 keypoints 半径，单位 m，默认 $0.05$。
        curve_sharpness (float): 曲线陡峭度 $a$，默认 $50$。
        curve_bias (float): 曲线偏置 $b$，默认 $2.0$。

    Returns:
        torch.Tensor: keypoint reorientation reward，形状 `[num_envs]`。
    """

    # 解析 object 当前姿态与 command 内部目标姿态，二者均为 world quaternion `(w,x,y,z)`。
    asset: RigidObject = env.scene[object_cfg.name]
    current_quat_w = asset.data.root_quat_w  # `[B,4]`，当前 object orientation
    goal_quat_w = resolve_goal_quat_w(env, command_name)  # `[B,4]`，目标 object orientation
    distance = orientation_keypoint_distance(current_quat_w, goal_quat_w, radius=keypoint_radius)  # $d_{kp}$，单位 m

    # 指数项做上界裁剪，避免极大姿态误差时 `exp(a d)` 数值溢出。
    x = torch.clamp(float(curve_sharpness) * distance, min=0.0, max=30.0)  # $a d_{kp}$，无量纲
    denominator = torch.exp(x) + float(curve_bias) + torch.exp(-x)  # $\exp(x)+b+\exp(-x)$
    numerator = 2.0 + float(curve_bias)  # 归一化常数，使 $d_{kp}=0$ 时 $r_{kp}=1$
    return numerator / denominator  # `[B]`，有界于 $(0,1]$ 的主姿态 reward


def goal_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    success_mode: Literal["so3", "keypoint", "both"] = "so3",
    orientation_success_threshold: float | None = None,
    keypoint_success_threshold: float = 0.02,
    keypoint_radius: float = 0.05,
) -> torch.Tensor:
    r"""重定向子目标成功 bonus。

    默认采用 $SO(3)$ geodesic threshold：
    $$
    \theta_e=\left\|\log(R_gR_o^{-1})\right\|_2,\qquad \theta_e<\theta_{th}.
    $$

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): 提供目标姿态和阈值配置的 command term 名称。
        object_cfg (SceneEntityCfg): object asset 配置。
        success_mode (Literal["so3", "keypoint", "both"]): 成功判据模式。
        orientation_success_threshold (float | None): SO(3) 角误差阈值，单位 rad。
        keypoint_success_threshold (float): keypoint distance 阈值，单位 m。
        keypoint_radius (float): keypoint 半径，单位 m。

    Returns:
        torch.Tensor: 成功 bonus 指示，形状 `[num_envs]`，值为 0/1 float。
    """

    # 解析 object 当前姿态与 command 目标姿态。
    asset: RigidObject = env.scene[object_cfg.name]
    current_quat_w = asset.data.root_quat_w  # `[B,4]`，当前姿态
    goal_quat_w = resolve_goal_quat_w(env, command_name)  # `[B,4]`，目标姿态

    # 若阈值未显式传入，则读取 command cfg；读取失败时使用 command cfg 中讨论过的 $\pi/12$ 默认值。
    command_term = env.command_manager.get_term(command_name)
    if orientation_success_threshold is None:
        orientation_success_threshold = float(
            getattr(command_term.cfg, "orientation_success_threshold", math.pi / 12.0)
        )
    resolved_orientation_threshold = float(orientation_success_threshold)  # 已排除 `None`，收窄给静态检查

    dtheta = math_utils.quat_error_magnitude(goal_quat_w, current_quat_w)  # $\theta_e$，形状 `[B]`
    so3_success = dtheta <= resolved_orientation_threshold  # `[B]`，SO(3) 成功指示
    keypoint_distance = orientation_keypoint_distance(current_quat_w, goal_quat_w, radius=keypoint_radius)  # `[B]`，m
    keypoint_success = keypoint_distance <= float(keypoint_success_threshold)  # `[B]`，keypoint 成功指示

    # 声明式选择成功判据，便于后续 ablation / cfg 切换。
    if success_mode == "so3":
        success = so3_success
    elif success_mode == "keypoint":
        success = keypoint_success
    elif success_mode == "both":
        success = so3_success & keypoint_success
    else:
        raise ValueError(f"Unsupported success_mode: {success_mode}.")

    return success.float()  # RewardManager 需要 float reward，而不是 bool tensor


class AxisDeltaRotationReward(ManagerTermBase):
    r"""沿 command axis 的单步实际旋转增量奖励。

    AnyRotate 中 `r_rot` 的核心思想是奖励物体绕目标轴持续前进，而不是只在到达离散 goal
    时给稀疏成功信号。本项目的 command 是空间轴左乘：
    $$
    R_g=\exp([\hat\omega]\theta)R_o.
    $$
    因此实际旋转进度也应使用 left-increment：
    $$
    \Delta\phi_t=\log(R_tR_{t-1}^{-1}),\qquad
    r_{rot}=\operatorname{clip}(\Delta\phi_t^\top\hat\omega,-c,c).
    $$

    该项是 stateful reward term，因为它需要缓存上一帧 object orientation。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        r"""初始化上一帧姿态缓存。"""

        super().__init__(cfg, env)
        self._prev_quat_w = torch.zeros(env.num_envs, 4, device=env.device)  # `[B,4]`，上一帧 object orientation
        self._prev_quat_w[:, 0] = 1.0  # 单位 quaternion `(1,0,0,0)`，避免未初始化 NaN
        self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # `[B]`，是否已有上一帧

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""在 env reset 时清空上一帧姿态缓存。"""

        if env_ids is None:
            self._has_prev[:] = False  # 全部 env 下一步 reward 置零并重新对齐缓存
        else:
            self._has_prev[env_ids] = False  # 只清空被 reset 的 env

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        clip_value: float = 0.025,
    ) -> torch.Tensor:
        r"""计算沿 command axis 的 clipped SO(3) left-increment。

        Args:
            env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
            command_name (str): 提供空间轴 `axis_e` / `axis_w` 的 command term 名称。
            object_cfg (SceneEntityCfg): object asset 配置。
            clip_value (float): 单步旋转增量裁剪阈值，单位 rad；AnyRotate 使用 $0.025$。

        Returns:
            torch.Tensor: 轴向旋转进度 reward，形状 `[num_envs]`。
        """

        # 读取当前 object 姿态与 command 空间轴；二者都在 `{e}` / `{w}` 语义下比较。
        asset: RigidObject = env.scene[object_cfg.name]
        current_quat_w = asset.data.root_quat_w  # `[B,4]`，当前 object orientation
        axis_e = resolve_axis_e(env, command_name)  # `[B,3]`，单位空间轴 $\hat\omega$

        valid = self._has_prev.clone()  # `[B]`，clone 防止后续更新影响本步 mask
        prev_quat_w = self._prev_quat_w.clone()  # `[B,4]`，上一帧姿态快照
        current_rot_w = math_utils.matrix_from_quat(current_quat_w)  # $R_t$，形状 `[B,3,3]`
        prev_rot_w = math_utils.matrix_from_quat(prev_quat_w)  # $R_{t-1}$，形状 `[B,3,3]`
        delta_rot_w = current_rot_w @ prev_rot_w.transpose(-1, -2)  # $R_tR_{t-1}^{-1}$
        delta_quat_w = math_utils.quat_from_matrix(delta_rot_w)  # `[B,4]`，delta quaternion
        delta_rotvec_w = math_utils.axis_angle_from_quat(delta_quat_w)  # `[B,3]`，so(3) 向量，rad

        progress = torch.sum(delta_rotvec_w * axis_e, dim=-1)  # `[B]`，$\Delta\phi_t^\top\hat\omega$，rad
        progress = torch.clamp(progress, -float(clip_value), float(clip_value))  # clipped progress reward
        progress = torch.where(valid, progress, torch.zeros_like(progress))  # reset 后首帧不计入伪进度

        self._prev_quat_w[:] = current_quat_w.detach()  # `[B,4]`，缓存当前姿态
        self._has_prev[:] = True  # 下一步开始所有未 reset env 均有有效上一帧
        return progress


def tactile_full_pose_keypoint_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    keypoint_radius: float = 0.05,
    curve_sharpness: float = 50.0,
    curve_bias: float = 2.0,
) -> torch.Tensor:
    r"""返回 current object pose 到 `(anchor, moving goal)` 的 normalized dense reward。"""

    command = ensure_post_physics_progress_updated(env, command_name)
    object_asset: RigidObject = env.scene[object_cfg.name]
    distances = full_pose_keypoint_distances(
        current_pos_w=object_asset.data.root_pos_w,
        current_quat_w=object_asset.data.root_quat_w,
        goal_pos_w=command.position_anchor_w,
        goal_quat_w=command.goal_quat_w,
        radius=keypoint_radius,
    )
    return normalized_keypoint_kernel(distances, curve_sharpness, curve_bias)  # continuous bounded rate


def tactile_axis_delta_rotation_rate(
    env: ManagerBasedRLEnv,
    command_name: str,
    clip_value: float = 0.025,
) -> torch.Tensor:
    r"""把 command-owned signed delta 裁剪后转成 policy-frequency-invariant reward rate。"""

    command = ensure_post_physics_progress_updated(env, command_name)
    clipped_delta = torch.clamp(command.delta_psi, -float(clip_value), float(clip_value))  # reward-only clip
    return impulse_to_rate(clipped_delta, env.step_dt)  # metric/curriculum 仍读取未裁剪 `net_rotation_rad`


def tactile_goal_success_impulse(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""返回 success 双门的一步 impulse rate；command hook 会在 reward 后推进 goal。"""

    command = ensure_post_physics_progress_updated(env, command_name)
    return impulse_to_rate(command.goal_success_pulse.float(), env.step_dt)


__all__ = [
    "AxisDeltaRotationReward",
    "goal_success_bonus",
    "keypoint_reorientation_reward",
    "reorientation_reward_placeholder",
    "tactile_axis_delta_rotation_rate",
    "tactile_full_pose_keypoint_reward",
    "tactile_goal_success_impulse",
]
