# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的观测函数

提供sim和real都能使用的观测值
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class body_twists(ManagerTermBase):
    r"""获取一个或多个 se3 动作项末端的速度旋量并拼接返回（直接类实现）。

    用法示例：

    - ObservationsCfg 中配置 `ObsTerm(func=leap_mdp.body_twists, params={"action_names": ["se3"], "asset_cfg": SceneEntityCfg("robot")})`
    - 若未指定 `action_names`，默认遍历全部 se3Action 动作项并按顺序拼接 `(num_envs, 6 * N)` 的 twist。
    - 观测管理器实例化时将 `cfg.params` 传入本类，调用阶段仅传入 `env`（无需额外参数）。
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv") -> None:
        """初始化 twist_body 观测项。
        
        Args:
            cfg: 观测项配置，包含 params 字典
            env: 强化学习环境实例
        """
        super().__init__(cfg, env)
        # 导入 se3Action 类用于类型检查
        from . import actions as se3

        # 获取资产配置，默认为 "robot"
        self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[self._asset_cfg.name]

        # 获取要读取的动作项名称列表
        action_names = cfg.params.get("action_names")
        # 如果未指定，则使用所有活跃的动作项
        candidate_names = list(env.action_manager.active_terms) if action_names is None else list(action_names)

        # 收集所有 se3Action 动作项的信息
        term_infos = []
        for name in candidate_names:
            term = env.action_manager.get_term(name)
            # 检查是否为 se3Action 类型
            if not isinstance(term, se3.se3Action):
                if action_names is not None:
                    raise ValueError(f"动作项 '{name}' 不是 se3Action 类型，而是 {type(term)}")
                continue
            # 存储动作项的关键信息
            info = {
                "name": name,
                "body_idx": term.body_idx,  # 末端执行器的 body 索引
                "Ad_batch": None,  # 伴随变换矩阵（如果需要坐标变换）
            }
            # 如果动作项使用了坐标变换，预计算伴随矩阵
            if term.is_xform and term.Ad_bprime_b is not None:
                info["Ad_batch"] = term.Ad_bprime_b.unsqueeze(0).expand(env.num_envs, -1, -1)
            term_infos.append(info)

        # 确保至少找到一个有效的 se3Action
        if len(term_infos) == 0:
            raise ValueError("未找到任何 se3Action 动作项供 twist_body 读取。")

        self._term_infos = term_infos

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        """计算并返回所有 se3Action 末端的速度旋量。
        
        Args:
            env: 强化学习环境实例
            
        Returns:
            拼接后的速度旋量张量，形状为 (num_envs, 6 * N)，其中 N 是 se3Action 的数量
        """
        twists = []
        for info in self._term_infos:
            body_idx = info["body_idx"]

            # 获取末端执行器在世界坐标系下的速度
            lin_vel_w = self._asset.data.body_lin_vel_w[:, body_idx]
            ang_vel_w = self._asset.data.body_ang_vel_w[:, body_idx]
            body_quat_w = self._asset.data.body_quat_w[:, body_idx]

            # 将速度从世界坐标系转换到 body 坐标系
            lin_vel_b = quat_rotate_inverse(body_quat_w, lin_vel_w)
            ang_vel_b = quat_rotate_inverse(body_quat_w, ang_vel_w)

            # 组合成速度旋量 [角速度, 线速度]
            twist_b = torch.cat([ang_vel_b, lin_vel_b], dim=1)

            # 如果需要，应用伴随变换到目标坐标系
            if info["Ad_batch"] is not None:
                twist_b = torch.matmul(info["Ad_batch"], twist_b.unsqueeze(-1)).squeeze(-1)

            twists.append(twist_b)

        # 拼接所有动作项的速度旋量
        return torch.cat(twists, dim=1)