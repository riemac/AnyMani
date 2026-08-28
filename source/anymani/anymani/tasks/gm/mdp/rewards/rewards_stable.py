r"""Stability and regularization reward terms for GM in-hand manipulation.

TODO(tactile rotation stable group):
    新 baseline 在 AnyRotate 的 pose/work/torque 约束上增加 object speed band、speed jitter、
    off-axis angular velocity、linear velocity 与 action-rate。完整组与 contact group 共同乘
    `lambda_rew`，避免训练初期被“稳定握住但不旋转”的局部最优吸收。

    目标轴速度使用物理时间常数而不是固定离散系数：

    $$
    \alpha_{\omega}
    =
    1-\exp
    \left(
    -\frac{\Delta t_{policy}}{0.25\ \mathrm{s}}
    \right),
    $$

    $$
    \bar\omega_t
    =
    (1-\alpha_{\omega})\bar\omega_{t-1}
    +
    \alpha_{\omega}\omega_{\parallel,t}.
    $$

    速度目标区间为 0.6--0.833 rad/s；区间外距离平方为 speed penalty，瞬时速度相对 EMA
    的残差平方为 jitter penalty。另惩罚非目标轴角速度与 object linear velocity。

    pose anchor 必须是本 episode reset 后实际抓取关节姿态：

    $$
    r_{pose}=-\|q-q_{anchor}\|_2.
    $$

    work 与 torque 使用物理闭合定义：

    $$
    r_{work}
    =
    -\sum_i|\tau_i\dot q_i|,
    \qquad
    r_{torque}
    =
    -\sum_i\tau_i^2.
    $$

    work 是功率 rate，经 RewardManager 乘 `dt` 后近似机械能量。不得使用依赖关节零位的
    `tau^T u`。action L2 和 action-rate 对无量纲 policy command 计算。

    AnyRotate 初值只锚定 pose/work/torque 的相对量级 0.5/0.1/0.05；新增 stable term 的
    权重必须先用旧 checkpoint rollout 标定。每项都要记录 raw rate、weighted rate 与
    episode integral，不能只记录 total reward。
"""

from __future__ import annotations

import isaaclab.envs.mdp as isaac_mdp
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ..canonical_runtime import masked_mean
from ..commands.tactile_rotation_command import ensure_post_physics_progress_updated
from .rewards_common import curriculum_gain


def action_l2_curriculum(
    env: ManagerBasedRLEnv,
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Curriculum-gated action L2 regularizer。

    本项目动作项 `ClampedRelativeJointPositionAction` 已将每步 raw rad delta 通过 `scale=0.1`
    约束在温和范围内，并在下发前 clamp 到 soft joint limits。因而第一版严格模仿 AnyRotate，
    把 action 正则放到 adaptive curriculum 后释放。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: gated action L2 penalty source，形状 `[num_envs]`；外部配置负权重。
    """

    action = env.action_manager.action  # `[B,A]`，policy-facing normalized action
    active_mask = getattr(env, "_anymani_canonical_active_joint_mask", None)
    if isinstance(active_mask, torch.Tensor) and active_mask.shape == action.shape:
        penalty = masked_mean(action.square(), active_mask)  # 按 active joint 数均值，ghost 不计入
    else:
        penalty = isaac_mdp.action_l2(env)
    return penalty * curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)


def action_rate_l2_curriculum(
    env: ManagerBasedRLEnv,
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Curriculum-gated action-rate L2 regularizer。

    该项惩罚相邻 policy action 的变化率，主要用于抑制高频抖动。由于相对增量动作本身
    已有限幅，默认也放入 curriculum；若训练早期仍出现动作爆炸，可单独把本项
    `lambda_floor` 调到 $0.02\sim0.1$。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: gated action-rate L2 penalty source，形状 `[num_envs]`。
    """

    delta = env.action_manager.action - env.action_manager.prev_action  # `[B,A]`，相邻 policy action 差
    active_mask = getattr(env, "_anymani_canonical_active_joint_mask", None)
    if isinstance(active_mask, torch.Tensor) and active_mask.shape == delta.shape:
        penalty = masked_mean(delta.square(), active_mask)  # inactive slot 不形成高频动作惩罚
    else:
        penalty = isaac_mdp.action_rate_l2(env)
    return penalty * curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)


def torque_l2_curriculum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Curriculum-gated torque L2 penalty source。

    对齐 AnyRotate 的 torque penalty：
    $$
    r_{torque}=\|\tau\|_2^2.
    $$

    本函数返回正值 penalty source，实际惩罚由 `RewardsCfg` 负权重实现。若当前 actuator
    backend 没有暴露 `computed_torque`，返回 0 并保留接口。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        asset_cfg (SceneEntityCfg): robot articulation 配置。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: gated torque L2 penalty source，形状 `[num_envs]`。
    """

    # 读取 articulation 的 controller torque；不同 actuator backend 可能不存在该字段。
    asset: Articulation = env.scene[asset_cfg.name]
    torque = getattr(asset.data, "computed_torque", None)  # `[B,d]`，控制器计算力矩，单位 N·m
    if torque is None:
        return torch.zeros(env.num_envs, device=env.device)  # 保留接口，不因 backend 差异中断脚手架

    active_mask = getattr(env, "_anymani_canonical_active_joint_mask", None)
    if isinstance(active_mask, torch.Tensor):
        active_mask = active_mask[:, asset_cfg.joint_ids]
        penalty = masked_mean(torque.square(), active_mask)  # 只按 active joint 平均 $\tau_i^2$
    else:
        penalty = torch.sum(torque**2, dim=-1)  # `[B]`，$\|\tau\|_2^2$
    return penalty * curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)


def object_axis_speed_band_curriculum(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_min: float = 0.6,
    speed_max: float = 0.833,
) -> torch.Tensor:
    r"""惩罚低通轴向速度落在目标区间外的平方距离。

    $$p_{speed}=[\max(0,\omega_{min}-\bar\omega)]^2+
    [\max(0,\bar\omega-\omega_{max})]^2.$$

    返回正 penalty source；env cfg 使用 `weight=-0.5`。
    """

    if float(speed_max) <= float(speed_min):
        raise ValueError(f"speed_max must exceed speed_min, got {speed_min}, {speed_max}.")
    command = ensure_post_physics_progress_updated(env, command_name)
    below = torch.clamp(float(speed_min) - command.axis_speed_ema, min=0.0)
    above = torch.clamp(command.axis_speed_ema - float(speed_max), min=0.0)
    return (below.square() + above.square()) * curriculum_gain(env, 0.0, 1.0)


def object_axis_speed_jitter_curriculum(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""惩罚瞬时轴向速度相对 0.25 s EMA 的残差平方。"""

    command = ensure_post_physics_progress_updated(env, command_name)
    jitter = (command.axis_speed - command.axis_speed_ema).square()
    return jitter * curriculum_gain(env, 0.0, 1.0)


def object_off_axis_ang_vel_curriculum(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    r"""惩罚 object world angular velocity 中正交于目标轴的分量平方。"""

    command = ensure_post_physics_progress_updated(env, command_name)
    object_asset: RigidObject = env.scene[object_cfg.name]
    angular_velocity_w = object_asset.data.root_ang_vel_w
    parallel = torch.sum(angular_velocity_w * command.axis_w, dim=-1, keepdim=True) * command.axis_w
    off_axis = angular_velocity_w - parallel
    return torch.sum(off_axis.square(), dim=-1) * curriculum_gain(env, 0.0, 1.0)


def object_lin_vel_l2_curriculum(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    r"""惩罚 object 线速度平方范数，抑制 palm 上滑移/弹跳。"""

    object_asset: RigidObject = env.scene[object_cfg.name]
    return torch.sum(object_asset.data.root_lin_vel_w.square(), dim=-1) * curriculum_gain(env, 0.0, 1.0)


def joint_pose_anchor_l2_curriculum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""惩罚当前手姿相对本 episode reset 后真实抓取姿态的 L2 距离。"""

    asset: Articulation = env.scene[asset_cfg.name]
    current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    anchor = getattr(env, "_gm_robot_reset_joint_anchor", None)
    if not isinstance(anchor, torch.Tensor) or anchor.shape != current.shape:
        anchor = asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # event 未安装时仅用于显式 smoke fallback
    error = current - anchor
    active_mask = getattr(env, "_anymani_canonical_active_joint_mask", None)
    if isinstance(active_mask, torch.Tensor):
        error = error * active_mask[:, asset_cfg.joint_ids].to(dtype=error.dtype)
    penalty = torch.linalg.norm(error, dim=-1)  # $\|q-q_{anchor}\|_2$，不是平方范数
    return penalty * curriculum_gain(env, 0.0, 1.0)


def joint_mechanical_power_curriculum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""返回 $\sum_i|\tau_i\dot q_i|$ 功率 penalty rate，单位 W。"""

    asset: Articulation = env.scene[asset_cfg.name]
    torque = getattr(asset.data, "computed_torque", None)
    if torque is None:
        return torch.zeros(env.num_envs, device=env.device)
    torque = torque[:, asset_cfg.joint_ids]
    joint_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    power_values = torch.abs(torque * joint_velocity)  # `[B,J]`，$|\tau_i\dot q_i|$，单位 W
    active_mask = getattr(env, "_anymani_canonical_active_joint_mask", None)
    power = (
        masked_mean(power_values, active_mask[:, asset_cfg.joint_ids])
        if isinstance(active_mask, torch.Tensor)
        else torch.sum(power_values, dim=-1)
    )
    return power * curriculum_gain(env, 0.0, 1.0)


__all__ = [
    "action_l2_curriculum",
    "action_rate_l2_curriculum",
    "joint_mechanical_power_curriculum",
    "joint_pose_anchor_l2_curriculum",
    "object_axis_speed_band_curriculum",
    "object_axis_speed_jitter_curriculum",
    "object_lin_vel_l2_curriculum",
    "object_off_axis_ang_vel_curriculum",
    "torque_l2_curriculum",
]
