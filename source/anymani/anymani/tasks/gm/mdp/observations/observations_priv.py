r"""Privileged observation terms for GM teacher policies.

本模块放置 teacher / specialist policy 可以读取、但不一定要求 student 或 sim2real
策略可观测的 object 状态。当前单资产 MDP 核验阶段优先让 teacher 学会任务，
因此 policy obs 可以显式读取 object pose；后续 distillation / student 阶段再决定
哪些项保留、压缩或遮蔽。

本项目主线是手型泛化，但保留物体泛化接口。物体泛化必须在 teacher 训练阶段引入
多元物体资产并通过 privileged info 显式条件化策略。本段定义从仿真器提取哪些 raw
physical values，交给 `distill/models` 侧的 object token encoder 投影为 `[OBJ]` 全局 token。

物体表征路线（已与用户对齐）：当前阶段采用扩展 HORA 路线：

- teacher 显式喂 raw physical properties（mass, scale, friction, COM 等），不做 HORA 风格
  的压缩嵌入，避免嵌入与具体手型策略耦合；
- 策略侧由 `distill/models` 将 raw values 投影为 `[OBJ]` token，并在 self-attention 池中与
  joint tokens 交互；
- `[OBJ]` token 的 projection 模块未来可额外接收手形态特征，使物体表征对当前 hand
  embodiment 有感知；
- 若几何形状多样性成为瓶颈，`[OBJ]` token 可升级为多个静态 mesh token（BPS / 几何描述符，
  离线预计算）。

DONE(object pose obs):
    当前已经落地 object pose 的 hand-frame 版本：

    - `object_pos_h`: object root 相对 hand semantic frame `{h}` 的位置 $p_o^h$，
      单位 m；
    - `object_rot6d_h`: object frame `{o}` 相对 hand semantic frame `{h}` 的旋转
      $R_{ho}$ 前两列，按 Zhou 6D continuous rotation representation 拼成 6 维，
      避免裸 quaternion 的 $q/-q$ 双覆盖，也避免 9D 矩阵冗余。

TODO(privileged physics obs): 物体物理属性属于 teacher-only 信息，sim2real 不可得。
未来可从仿真器提取约 21 维 raw physical values：

- object mass $m$；
- object scale $(s_x,s_y,s_z)$；
- object friction $\mu$；
- object COM offset $(dx,dy,dz)$；
- object pose $(x,y,z,q_w,q_x,q_y,q_z)$；
- object velocity $(v_x,v_y,v_z,\omega_x,\omega_y,\omega_z)$。

数据来源包括 `RigidObject.root_physx_view`、`object.data.root_pos_w`、
`object.data.root_quat_w`、`object.data.root_lin_vel_w` 与 `object.data.root_ang_vel_w`。
所有值均为 SI 物理量，不依赖手构型，满足跨手型泛化的表征解耦需求。

具体投影维度、是否拆分为多个 token、是否加 position embedding，由
`distill/models` 的 Specialist Policy Transformer 设计决定。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _semantic_R_ha_tensor(env: ManagerBasedRLEnv, semantic_R_ha: tuple[float, ...]) -> torch.Tensor:
    r"""把配置层 row-major $R_{ha}$ 转成 runtime tensor。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env，用于确定 device。
        semantic_R_ha (tuple[float, ...]): row-major 9 元组，语义为列向量约定下
            $v^h = R_{ha} v^a$。

    Returns:
        torch.Tensor: 旋转矩阵 $R_{ha}$，形状 `[3, 3]`，位于 env device。
    """

    return torch.tensor(semantic_R_ha, dtype=torch.float32, device=env.device).reshape(3, 3)  # $R_{ha}$


def object_pos_h(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> torch.Tensor:
    r"""读取 object root 在 hand semantic frame `{h}` 下的位置。

    该项描述“物体相对手在哪里”，而不是 world frame 中的绝对位置：
    $$
    p_o^h = R_{ha} R_{aw} (p_o^w - p_a^w),
    $$
    其中 `{a}` 是 hand raw asset/root frame，`{h}` 是 hand semantic frame。
    在当前 fixed hand anchor 的 single-asset 阶段，reset 后该项应接近标定台
    导出的 contact basin，例如 $(0.02, 0.08, 0.06)$。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        object_cfg (SceneEntityCfg): object rigid body 配置。
        robot_cfg (SceneEntityCfg): hand articulation 配置。
        semantic_R_ha (tuple[float, ...]): $R_{ha}$，row-major 9 元组。

    Returns:
        torch.Tensor: object 相对 hand 的位置，形状 `[num_envs, 3]`，单位 m。
    """

    object_asset: RigidObject = env.scene[object_cfg.name]  # object root pose 来源
    robot_asset: Articulation = env.scene[robot_cfg.name]  # hand root pose 来源
    rel_pos_w = object_asset.data.root_pos_w - robot_asset.data.root_pos_w  # $p_o^w-p_a^w$，`[B,3]`
    rel_pos_a = math_utils.quat_apply_inverse(robot_asset.data.root_quat_w, rel_pos_w)  # $R_{aw}(p_o^w-p_a^w)$
    R_ha = _semantic_R_ha_tensor(env, semantic_R_ha)  # `{a}->{h}` 语义对齐矩阵
    return rel_pos_a @ R_ha.T  # row-vector 写法：$p_o^h = p_o^a R_{ha}^T$


def object_rot6d_h(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> torch.Tensor:
    r"""读取 object orientation 在 hand semantic frame `{h}` 下的 6D 连续旋转表示。

    返回的是 object body frame `{o}` 相对 hand semantic frame `{h}` 的姿态：
    $$
    R_{ho} = R_{ha} R_{aw} R_{wo}.
    $$
    teacher policy 不直接读取裸 quaternion，而读取 Zhou 6D continuous rotation
    representation：取 $R_{ho}$ 的前两列并按列拼接，
    $$
    r^{6D}_{ho} = [R_{ho}[:,0],\ R_{ho}[:,1]] \in \mathbb{R}^6.
    $$
    第三列可由前两列正交化后叉乘恢复；对 policy 而言 6D 足以表达姿态，且比
    9D rotation matrix 少 3 维冗余。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        object_cfg (SceneEntityCfg): object rigid body 配置。
        robot_cfg (SceneEntityCfg): hand articulation 配置。
        semantic_R_ha (tuple[float, ...]): $R_{ha}$，row-major 9 元组。

    Returns:
        torch.Tensor: $R_{ho}$ 前两列的列向量拼接，形状 `[num_envs, 6]`。
    """

    object_asset: RigidObject = env.scene[object_cfg.name]  # object orientation $R_{wo}$
    robot_asset: Articulation = env.scene[robot_cfg.name]  # hand root orientation $R_{wa}$
    R_ha = _semantic_R_ha_tensor(env, semantic_R_ha).unsqueeze(0)  # `[1,3,3]`，广播到所有 env
    R_wa = math_utils.matrix_from_quat(robot_asset.data.root_quat_w)  # `[B,3,3]`，hand root `{a}->{w}`
    R_aw = R_wa.transpose(-1, -2)  # `[B,3,3]`，world/env 到 hand raw asset `{a}`
    R_wo = math_utils.matrix_from_quat(object_asset.data.root_quat_w)  # `[B,3,3]`，object `{o}->{w}`
    R_ho = R_ha @ R_aw @ R_wo  # `[B,3,3]`，object `{o}` 在 hand semantic `{h}` 中的姿态
    return torch.cat((R_ho[:, :, 0], R_ho[:, :, 1]), dim=-1)  # `[B,6]`，列 0/1 按 Zhou 6D 拼接


__all__ = ["object_pos_h", "object_rot6d_h"]
