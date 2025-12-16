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
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class body_twists(ManagerTermBase):
    r"""获取一个或多个 se3 动作项末端的速度旋量并拼接返回（直接类实现）。

    用法示例：

    - ObservationsCfg 中配置 `ObsTerm(func=leap_mdp.body_twists, params={"action_names": ["se3"], "asset_cfg": SceneEntityCfg("robot")})`
    - 若未指定 `action_names`，默认遍历全部 se3Action 动作项并按顺序拼接 `(num_envs, 6 * N)` 的 twist。
    - 观测管理器实例化时将 `cfg.params` 传入本类，调用阶段仅传入 `env`（无需额外参数）。

    坐标系约定：
        - 默认在刚体坐标系 {b} 下表达旋量（use_body_frame=True）。
        - 若 use_body_frame=False，则在世界坐标系 {w} 下表达旋量。

    Note:
        若动作项使用虚拟 Xform（term.is_xform=True），这里会缓存其伴随矩阵 ``Ad_bprime_b``。
        该伴随矩阵用于在 {b} 下将旋量转换到虚拟帧 {b'}。
        因此当 use_body_frame=False（输出 {w}）时，当前实现不会应用该伴随变换，避免产生错误的帧混用。
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

        # 是否归一化（默认True）
        self._normalize = cfg.params.get("normalize", True)

        # 观测旋量是否在刚体坐标系 {b} 下表达（默认True）。若为 False，则在世界坐标系 {w} 下表达。
        self._use_body_frame = cfg.params.get("use_body_frame", True)

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
                # 存储缩放参数
                "ang_scale": getattr(term, "_angular_scale", 1.0),
                "ang_bias": getattr(term, "_angular_bias", 0.0),
                "lin_scale": getattr(term, "_linear_scale", 1.0),
                "lin_bias": getattr(term, "_linear_bias", 0.0),
            }
            # 如果动作项使用了坐标变换，预计算伴随矩阵
            if term.is_xform and term.Ad_bprime_b is not None:
                info["Ad_batch"] = term.Ad_bprime_b.unsqueeze(0).expand(env.num_envs, -1, -1)
            term_infos.append(info)

        # 确保至少找到一个有效的 se3Action
        if len(term_infos) == 0:
            raise ValueError("未找到任何 se3Action 动作项供 twist_body 读取。")

        self._term_infos = term_infos

    def __call__(self, env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg,
                 use_body_frame: False, action_names= ["index_se3", "middle_se3", "ring_se3", "thumb_se3"]) -> torch.Tensor:
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

            # 根据 use_body_frame 选择输出旋量的表达坐标系
            if self._use_body_frame:
                # 将速度从世界坐标系转换到 body 坐标系 {b}
                lin_vel = quat_apply_inverse(body_quat_w, lin_vel_w)
                ang_vel = quat_apply_inverse(body_quat_w, ang_vel_w)
            else:
                # 直接使用世界坐标系 {w} 下的速度
                lin_vel = lin_vel_w
                ang_vel = ang_vel_w

            # 归一化
            if self._normalize:
                ang_vel = (ang_vel - info["ang_bias"]) / info["ang_scale"]
                lin_vel = (lin_vel - info["lin_bias"]) / info["lin_scale"]

            # 组合成速度旋量 [角速度, 线速度]
            twist = torch.cat([ang_vel, lin_vel], dim=1)

            # 如果需要，应用伴随变换到目标坐标系（仅在 {b} 下成立）
            if info["Ad_batch"] is not None and self._use_body_frame:
                twist = (info["Ad_batch"] @ twist.unsqueeze(-1)).squeeze(-1)

            twists.append(twist)

        # 拼接所有动作项的速度旋量
        return torch.cat(twists, dim=1)