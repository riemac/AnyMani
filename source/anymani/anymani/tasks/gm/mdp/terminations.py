r"""Termination terms for `tasks.gm`.

本模块只定义 episode 边界，不定义 reward。当前第一版只启用两个边界：

1. IsaacLab 内置 `time_out`；
2. 本文件的 `object_out_of_hand`：object 偏离 reset anchor 太远，视为离手 / 掉落。

DONE(已合意):
    - 不做 `max_success_count` termination；成功数只作为 metric / curriculum 进度。
    - 不默认做 axis deviation termination；它只保留给未来 fixed-axis continuous
      rotation 的可选 metric / termination。
    - 不做 joint limit / 卡死 termination；episode 较短时先由 timeout 兜底。
    - object 离手判据采用 3D L2 distance，默认阈值 `fall_dist=0.12m`。
    - anchor 优先使用 reset/events 记录的 object 初始位置；若 events 尚未写入，
      fallback 到 object default root state。

TODO(single-asset tactile rotation termination):
    新 baseline 的位置边界收紧到相对 episode reset anchor 的 0.07 m，并增加有符号 axis
    alignment termination：

    $$
    \|p_o^h-p_{anchor}^h\|_2\ge0.07\ \mathrm{m},
    $$

    $$
    z_o^{\{h\}\mathsf{T}}z_g^{\{h\}}<\cos(45^\circ).
    $$

    不得使用 absolute dot。绕目标 z 轴的 180 degree yaw 是合法旋转；应失败的是绕横轴
    翻面后 object 法向反向或偏离超过 45 degree。success 只推进 30 degree subgoal，不终止
    episode。timeout 继续使用 LEAP sampled full horizon。

    当前位置 helper 的默认 0.12 m 属于旧 GM probe，不能静默改默认值影响已有任务；新 env
    通过独立 cfg 显式传入 0.07 m。axis termination 应新增独立 term，并从 command 读取
    goal/axis buffer，避免 termination 自己重建 task frame。
"""

from __future__ import annotations

import math

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .commands.tactile_rotation_command import ensure_post_physics_progress_updated


def _resolve_object_reset_anchor_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    r"""解析 object 离手判据使用的 reset anchor。

    termination 需要知道“object 本来应该待在哪附近”。这个位置不应硬编码成
    goal marker 位置，也不应从 reward 反推。优先级如下：

    1. `env._gm_object_reset_anchor_w`：events/reset 已记录的 world-frame anchor；
    2. `env._gm_object_reset_anchor_e`：events/reset 已记录的 env-frame anchor；
    3. `object.data.default_root_state[:, :3]`：scene 中 object 的默认 env-frame 初始位置。

    第 1/2 项是为后续 events 随机化预留的接口；当前 events 还没实现时，第 3 项
    作为 fallback，使 termination 合同可先落地。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        object_cfg (SceneEntityCfg): object asset 配置。

    Returns:
        torch.Tensor: world-frame anchor，形状 `[num_envs, 3]`，单位 m。
    """

    # 优先使用 events/reset 显式记录的 world-frame anchor；未来 reset 随机化时最稳
    anchor_w = getattr(env, "_gm_object_reset_anchor_w", None)  # `[B,3]`，world-frame reset anchor
    if isinstance(anchor_w, torch.Tensor):
        return anchor_w.to(device=env.device)  # 确保 device 与 env 一致

    # 次优先使用 env-frame anchor；转换到 world frame 只需加每个 env 的 origin
    anchor_e = getattr(env, "_gm_object_reset_anchor_e", None)  # `[B,3]`，env-frame reset anchor
    if isinstance(anchor_e, torch.Tensor):
        return anchor_e.to(device=env.device) + env.scene.env_origins  # `[B,3]`，world-frame anchor

    # fallback：使用 object cfg/default root state 中的初始 env-frame 位置
    object_asset: RigidObject = env.scene[object_cfg.name]  # 被操作物体
    default_anchor_e = object_asset.data.default_root_state[:, :3]  # `[B,3]`，object 默认 env-frame root pos
    return default_anchor_e + env.scene.env_origins  # `[B,3]`，world-frame anchor


def object_out_of_hand(
    env: ManagerBasedRLEnv,
    fall_dist: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    r"""Object 离手 / 掉落 termination。

    判据是 object root position 相对 reset anchor 的 3D 欧氏距离：

    $$
    d_o = \left\|p_o^{\{w\}} - p_{anchor}^{\{w\}}\right\|_2,
    \qquad
    done = d_o > d_{fall}
    $$

    这里不用 goal marker 位置，因为 goal marker 只是可视化，不是位置目标；也不
    只看 world z，因为后续 `{h}` 语义和手姿态变化会让“掉落方向”不再等价于
    world z 负方向。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        fall_dist (float): 允许 object 偏离 reset anchor 的最大距离，单位 m，默认 0.12。
        object_cfg (SceneEntityCfg): object asset 配置。

    Returns:
        torch.Tensor: bool tensor，形状 `[num_envs]`。
    """

    # 当前 object root position，world frame，形状 `[B,3]`
    object_asset: RigidObject = env.scene[object_cfg.name]  # 被操作物体
    object_pos_w = object_asset.data.root_pos_w  # `[B,3]`，当前 object world position

    # reset/default anchor，world frame，形状 `[B,3]`
    anchor_w = _resolve_object_reset_anchor_w(env, object_cfg)  # `[B,3]`，object 应待在的附近

    # 3D L2 distance；返回 bool，不在 termination 函数里写额外 metric 副作用
    distance = torch.linalg.norm(object_pos_w - anchor_w, dim=-1)  # `[B]`，单位 m
    return distance > float(fall_dist)  # `[B]`，True 表示该 env 应 reset


def object_falling_placeholder(env: ManagerBasedRLEnv, fall_dist: float) -> torch.Tensor:
    r"""Backward-compatible alias for `object_out_of_hand`.

    TODO:
        旧 cfg 中仍引用该 placeholder 名字。后续正式出清时，应把 env cfg 全部改为
        `object_out_of_hand` 并删除本 alias。
    """

    return object_out_of_hand(env=env, fall_dist=fall_dist)


def tactile_object_out_of_anchor(
    env: ManagerBasedRLEnv,
    command_name: str,
    fall_dist: float = 0.07,
) -> torch.Tensor:
    r"""检测 object root 是否偏离 tactile command 的 episode reset anchor 至少 7 cm。"""

    command = ensure_post_physics_progress_updated(env, command_name)
    return command.position_error >= float(fall_dist)  # plan 使用闭边界 `>=0.07 m`


def tactile_goal_axis_misaligned(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_angle_deg: float = 45.0,
) -> torch.Tensor:
    r"""检测 signed object/goal normal alignment；不得对 dot product 取 absolute。

    $z_o^{w\mathsf T}z_g^w<\cos(45^\circ)$ 时终止。绕合法 z 轴 yaw 不改变 normal；
    横轴翻面产生负 dot 并立即失败。
    """

    command = ensure_post_physics_progress_updated(env, command_name)
    threshold = math.cos(math.radians(float(max_angle_deg)))
    return command.goal_normal_alignment < threshold


def adr_randomized_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""按每个 env reset 时采样的 20--120 s full horizon 触发 timeout。"""

    episode_lengths = getattr(env, "leap_adr_episode_lengths", None)
    if not isinstance(episode_lengths, torch.Tensor):
        return env.episode_length_buf >= env.max_episode_length - 1
    return env.episode_length_buf >= episode_lengths - 1


__all__ = [
    "adr_randomized_time_out",
    "object_falling_placeholder",
    "object_out_of_hand",
    "tactile_goal_axis_misaligned",
    "tactile_object_out_of_anchor",
]
