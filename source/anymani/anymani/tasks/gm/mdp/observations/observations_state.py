r"""State observation terms for GM in-hand manipulation.

DONE(state obs): 关节本体感受 (proprioception)，逐步变化的动态量，属于 obs mdp。

符号约定：$q_i$ 为第 $i$ 个关节角 (rad)，$\dot q_i$ 为关节角速度 (rad/s)，
$q_i^{\min}, q_i^{\max}$ 为该关节的 soft 限位 (rad)；下标 $i$ 遍历 surviving revolute joints。
坐标系语义见 `gm/AGENTS.md` 的 `{a} -> {h}` 约定。

NOTE(设计决策，已与用户对齐): 关节位置统一采用 **raw rad 表征** $q_i$，
而非 IsaacLab 默认的 `joint_pos_limit_normalized`（即 $q_i^{\text{norm}}$）。
该决策依据 `Research/总体/层次通才策略训练.md` 的 state obs 小组划分。

其中归一化变换定义为（IsaacLab `scale_transform` 语义，本项目刻意不采用）：
$$
q_i^{\text{norm}} = \frac{2\,(q_i - q_i^{\min})}{q_i^{\max} - q_i^{\min}} - 1 \in [-1, 1]
$$

为何用 $q_i$ 而非 $q_i^{\text{norm}}$：

1. 跨 variant 语义不变性：资产由同一建模约定生成，home position 近似共面；
   同一关节 raw rad 在不同 post-mutate variant、乃至真实 leap/allegro URDF
   对齐到 `{h}` 后语义一致。
2. post-mutate 只变 joint limit $[q_i^{\min},q_i^{\max}]$，不变零位/轴向语义；
   $q_i$ 是跨 variant / sim2sim 的不变量，归一化值会抹掉真实构型差异。
3. 恢复代价非对称：若用 $q_i^{\text{norm}}$ 还原 $q_i$，需要网络拟合
   $q_i^{\text{norm}}\cdot(q_i^{\max}-q_i^{\min})$ 这类乘性算子。
4. 数值尺度友好：raw 关节角通常落在温和有界区间（约 $[-0.8,1.5]$ rad），适合 PPO。

本模块实现的 state obs：

- `q_raw`: `asset.data.joint_pos[:, joint_ids]`，单位 rad；
- `dq_raw`: `asset.data.joint_vel[:, joint_ids]`，单位 rad/s；
- `last_action`: 上一步实际下发的 raw rad delta $\Delta_{t-1}$，单位 rad。

NOTE(last_action 与动作空间的耦合): 动作空间已确定为 raw rad delta（方案 C，
`ClampedRelativeJointPositionAction`），故 last action 也应在 raw rad 空间。
IsaacLab 内置 `isaac_mdp.last_action` 返回 `raw_actions`，即 policy NN 输出在
scale/clip 前的值，不是实际下发的 `processed_actions`。因此本模块读取 action term 的
`processed_actions`，使 last_action 与 $q_i$ 处于同物理量纲。

边界：$q_i^{\min}/q_i^{\max}$ 是时间常量，属于形态 / geometry 量，不放入本 state
模块，避免 history_length $H>1$ 时把静态量重复堆叠 $H$ 次。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_raw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""读取 hand articulation 的 raw joint position $q$。

    这是 `gm` teacher state obs 的主线关节位置项，刻意不使用 IsaacLab 的
    `joint_pos_limit_normalized`。数学语义为：
    $$
    \mathbf{q}_t = [q_{1,t},\dots,q_{n,t}] \in \mathbb{R}^{n},
    $$
    单位为 rad，索引顺序由 `SceneEntityCfg.joint_ids` 与 action joint schema 决定。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        asset_cfg (SceneEntityCfg): robot articulation 与 joint 子集配置。

    Returns:
        torch.Tensor: raw joint position，形状 `[num_envs, num_joints]`，单位 rad。
    """

    # `asset_cfg` 在 ObservationManager 初始化时会解析 joint_names → joint_ids；preserve_order=True 要求顺序一致。
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]  # $q_i$，raw rad，不随 limit 归一化


def joint_vel_raw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""读取 hand articulation 的 raw joint velocity $\dot q$。

    速度与位置同属 state obs 的动态本体感受项：
    $$
    \dot{\mathbf{q}}_t = [\dot q_{1,t},\dots,\dot q_{n,t}] \in \mathbb{R}^{n}.
    $$

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        asset_cfg (SceneEntityCfg): robot articulation 与 joint 子集配置。

    Returns:
        torch.Tensor: raw joint velocity，形状 `[num_envs, num_joints]`，单位 rad/s。
    """

    # 不减 default_joint_vel；手内操作关心当前真实角速度，而不是相对默认速度。
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]  # $\dot q_i$，raw rad/s


def last_processed_action(
    env: ManagerBasedRLEnv,
    action_name: str = "hand_joint_pos",
) -> torch.Tensor:
    r"""读取上一帧实际下发的 raw rad delta action。

    IsaacLab 内置 `last_action(action_name=...)` 返回 `raw_actions`，即 policy 网络输出
    在 scale / clip 前的无量纲值。`gm` 的动作空间已经固定为 raw relative delta：
    $$
    \Delta_t = a_t^{\mathrm{raw}}\,s \quad (\mathrm{rad}),
    $$
    因此 state obs 中的 last action 必须读取 action term 的 `processed_actions`，
    才与 $q$、$\dot q$ 和 soft limits 处在同一个物理量纲系统内。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        action_name (str): `ActionManager` 中的 action term 名称，默认 `hand_joint_pos`。

    Returns:
        torch.Tensor: 上一步 processed action，形状 `[num_envs, num_joints]`，单位 rad。

    Raises:
        RuntimeError: 若 action term 不暴露 `processed_actions`，说明 obs/action 合同不匹配。
    """

    # 与 `ClampedRelativeJointPositionAction` 直接耦合：processed_actions 正是 target 更新使用的 $\Delta_t$。
    action_term = env.action_manager.get_term(action_name)
    processed_actions = getattr(action_term, "processed_actions", None)  # `[B,N_j]`，rad delta
    if not isinstance(processed_actions, torch.Tensor):
        raise RuntimeError(
            f"Action term '{action_name}' must expose processed_actions for gm raw-rad last_action obs."
        )
    return processed_actions


__all__ = ["joint_pos_raw", "joint_vel_raw", "last_processed_action"]
