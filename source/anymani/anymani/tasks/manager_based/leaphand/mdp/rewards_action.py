# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""LeapHand与动作相关的奖励函数。主要由动作及其导数（如加速度、加加度）决定，用于约束策略的行为流形（Behavior Manifold），使其符合物理可行性或特定风格。

简言之，该奖励文件决定 “怎么做”，是通用（Task-Agnostic）的。

主要包括动作平滑性、能量消耗等。

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from .utils import math as math_leap
# from isaaclab.markers import VisualizationMarkers
# from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def torque_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """关节力矩平方和，用于约束动作的用力大小。"""

    asset: Articulation = env.scene[asset_cfg.name]
    torque = getattr(asset.data, "computed_torque", None)

    if torque is None:
        return torch.zeros(env.num_envs, device=env.device)

    return torch.sum(torque ** 2, dim=-1)

###
# se3动作正则化项
# 遵循 Modern Robotics 约定，主要数学计算调用`source/leaphand/leaphand/tasks/manager_based/leaphand/mdp/utils/math.py`
# 遵守符号约定 {w}, {e}, {s}, {b}, {b'}分别为世界坐标系，环境坐标系，基坐标系，末端坐标系，虚拟坐标系
###

class jacobian_manipulability(ManagerTermBase):
    r"""刚体雅可比矩阵的可操作度指标（Yoshikawa manipulability）。

    - 遍历所有已激活的 se3Action（或指定列表），计算其雅可比 :math:`J_{b}` 或 :math:`J_{b'}`（虚拟 Xform 时已通过伴随变换转换）。
    - 使用 :math:`m=\sqrt{\det(JJ^T)}` 作为操作度，数值上对小奇异值做截断保护。
    - 多个动作项取平均值（等权）作为最终奖励信号。

    用法：
    ``RewTerm(func=mdp.jacobian_manipulability, weight=..., params={"action_names": ["se3"]})``
    未指定 ``action_names`` 时默认遍历所有 se3Action。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        from . import actions as se3

        action_names = cfg.params.get("action_names")
        candidate_names = list(env.action_manager.active_terms) if action_names is None else list(action_names)

        terms = []
        for name in candidate_names:
            term = env.action_manager.get_term(name)
            if isinstance(term, se3.se3Action):
                terms.append(term)

        if len(terms) == 0:
            raise ValueError("jacobian_manipulability: 未找到 se3Action 动作项，请检查配置的 action_names 或动作管理器激活状态。")

        self._se3_terms = terms

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_names: list[str] | None = None,
    ) -> torch.Tensor:
        # action_names 参数仅用于满足框架签名要求，实际使用 self._se3_terms
        metrics = []
        for term in self._se3_terms:
            jac = term._get_jacobian()  # (num_envs, 6, nj)
            m = math_leap.manipulability(jac)  # (num_envs,)
            metrics.append(m)

        return torch.stack(metrics, dim=0).mean(dim=0)

class se3_kinetic_energy(ManagerTermBase):
    r"""se(3) 动作对应末端动能：:math:`E = \tfrac{1}{2} V_b^T G_b V_b`。

    - 使用真实刚体的质心速度旋量 :math:`V_b=[\omega_b^T, v_b^T]^T`（{b} 表示，质心参考点）。
    - 空间惯量 :math:`G_b = \mathrm{diag}(I_b, m I_3)`，从 USD 解析的质量与惯性张量构造，按环境批次缓存。
    - 多个 se3Action 时取平均动能。

    用法：
    ``RewTerm(func=mdp.se3_kinetic_energy, weight=..., params={"action_names": ["se3"]})``；未指定 ``action_names`` 时默认所有 se3Action。
    若 USD 未提供惯性数据，将退化为 0.5‖V‖^2 近似。
    
    .. warning::
        该奖励项依赖 USD 中的质量和惯性张量数据。若 USD 未提供这些物理属性，
        将使用单位惯量矩阵作为退化近似，可能导致动能计算不准确。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        from . import actions as se3

        # 获取资产（默认名称为 "robot"）
        robot_name = cfg.params.get("robot_name", "robot")
        self._asset: Articulation = env.scene[robot_name]

        # 解析动作项名称列表
        action_names = cfg.params.get("action_names")
        candidate_names = list(env.action_manager.active_terms) if action_names is None else list(action_names)

        # 筛选出所有 se3Action 类型的动作项
        terms = []
        for name in candidate_names:
            term = env.action_manager.get_term(name)
            if isinstance(term, se3.se3Action):
                terms.append(term)

        if len(terms) == 0:
            raise ValueError("se3_kinetic_energy: 未找到 se3Action 动作项，请检查配置的 action_names 或动作管理器激活状态。")

        self._se3_terms = terms

        # 预构造空间惯量矩阵 G_b = diag(I_b, m I3)（按动作项/刚体缓存）
        mass = getattr(self._asset.data, "default_mass", None)
        inertia_vec = getattr(self._asset.data, "default_inertia", None)
        self._inertia_map = {}
        
        # 标记是否使用了退化近似
        self._using_fallback = False

        for term in self._se3_terms:
            b_idx = term.body_idx
            if mass is not None and inertia_vec is not None:
                # 从 USD 数据构造真实的空间惯量矩阵
                m = mass[:, b_idx].to(env.device)  # (num_envs,) 质量标量
                I_vals = inertia_vec[:, b_idx].to(env.device)  # (num_envs, 9) 惯性张量展平向量
                
                # 重构 3x3 惯性张量矩阵（列主序）
                I_mat = torch.stack(
                    [
                        torch.stack([I_vals[:, 0], I_vals[:, 3], I_vals[:, 6]], dim=-1),
                        torch.stack([I_vals[:, 1], I_vals[:, 4], I_vals[:, 7]], dim=-1),
                        torch.stack([I_vals[:, 2], I_vals[:, 5], I_vals[:, 8]], dim=-1),
                    ],
                    dim=-2,
                )  # (num_envs, 3, 3)

                # 构造 6x6 空间惯量矩阵 G_b = diag(I_b, m*I_3)
                G = torch.zeros(env.num_envs, 6, 6, device=env.device, dtype=I_mat.dtype)
                G[:, :3, :3] = I_mat  # 转动惯量部分
                G[:, 3:, 3:] = m.view(-1, 1, 1) * torch.eye(3, device=env.device, dtype=I_mat.dtype)  # 平动惯量部分
            else:
                # 缺省时退化为单位惯量（警告：物理意义不准确）
                if not self._using_fallback:
                    import warnings
                    warnings.warn(
                        "se3_kinetic_energy: USD 未提供质量或惯性数据，使用单位惯量矩阵作为退化近似。"
                        "动能计算可能不准确，建议在 USD 中正确设置物理属性。",
                        UserWarning
                    )
                    self._using_fallback = True
                G = torch.eye(6, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1, 1)

            self._inertia_map[term] = G

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_names: list[str] | None = None,
        robot_name: str = "robot",
    ) -> torch.Tensor:
        # action_names 和 robot_name 参数仅用于满足框架签名要求
        # 实际使用 self._se3_terms 和 self._asset
        energies = []
        for term in self._se3_terms:
            b_idx = term.body_idx

            # 获取刚体质心在世界坐标系下的速度
            lin_w = self._asset.data.body_lin_vel_w[:, b_idx]  # (num_envs, 3) 线速度
            ang_w = self._asset.data.body_ang_vel_w[:, b_idx]  # (num_envs, 3) 角速度
            quat_w = self._asset.data.body_quat_w[:, b_idx]    # (num_envs, 4) 姿态四元数

            # 将速度从世界坐标系转换到刚体坐标系 {b}
            lin_b = math_utils.quat_apply_inverse(quat_w, lin_w)
            ang_b = math_utils.quat_apply_inverse(quat_w, ang_w)
            twist_b = torch.cat([ang_b, lin_b], dim=1)  # (num_envs, 6) 速度旋量 [ω, v]

            # 计算动能 E = 0.5 * V^T G V
            G = self._inertia_map[term]
            energy = 0.5 * torch.einsum("bi,bij,bj->b", twist_b, G, twist_b)
            energies.append(energy)

        # 多个动作项时取平均动能
        return torch.stack(energies, dim=0).mean(dim=0)

class se3_action_smooth(ManagerTermBase):
    r"""se(3) 动作平滑性：惩罚相邻时间步旋量指令的跳变。

        - 支持两种来源：默认使用动作项的 ``raw_actions``（更符合 IsaacLab 自带 action_rate 奖励的 RL 正则习惯）；
            若 ``params.use_processed=True``，则使用 ``processed_actions``（物理量级更贴近实际旋量）。
        - 计算 :math:`\|V_t - V_{t-1}\|_p`（默认 L2），多个动作项取平均。
    - 首次调用或刚 reset 时返回 0，并更新历史。

    用法：``RewTerm(func=mdp.se3_action_smooth, weight=..., params={"action_names": ["se3"], "norm": 2})``。
    """
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        """初始化 se3 动作平滑性奖励项。

        Args:
            cfg: 奖励项配置，包含动作名称、范数类型等参数。
            env: 管理器基础强化学习环境实例。
        """
        super().__init__(cfg, env)
        from . import actions as se3

        # 从配置中获取动作名称列表
        action_names = cfg.params.get("action_names")
        # 从配置中获取范数类型
        self.norm = cfg.params.get("norm", 2)

        # 确定使用原始动作还是处理后的动作
        self._use_processed = bool(cfg.params.get("use_processed", False))

        # 获取候选动作项名称列表
        candidate_names = list(env.action_manager.active_terms) if action_names is None else list(action_names)

        # 筛选出所有 se3Action 类型的动作项
        terms = []
        for name in candidate_names:
            term = env.action_manager.get_term(name)
            if isinstance(term, se3.se3Action):
                terms.append(term)

        if len(terms) == 0:
            raise ValueError("se3_action_smooth: 未找到 se3Action 动作项，请检查配置的 action_names 或动作管理器激活状态。")

        # 保存 se3 动作项列表
        self._se3_terms = terms
        # 为每个动作项创建历史动作缓存（用于计算差分）
        self._prev: dict[object, torch.Tensor] = {term: torch.zeros(env.num_envs, 6, device=env.device) for term in terms}
        # 为每个动作项创建初始化标志（首次调用时跳过惩罚计算）
        self._initialized: dict[object, bool] = {term: False for term in terms}
        self._initialized: dict[object, bool] = {term: False for term in terms}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_names: list[str] | None = None,
        use_processed: bool = False,
        norm: int = 2,
    ) -> torch.Tensor:
        """计算动作平滑性惩罚。

        Args:
            env: 管理器基础强化学习环境实例。
            action_names: 动作名称列表（参数仅用于满足框架签名要求）。
            use_processed: 是否使用处理后的动作（参数仅用于满足框架签名要求）。
            norm: 范数类型（参数仅用于满足框架签名要求）。

        Returns:
            动作平滑性惩罚值，形状为 (num_envs,)。
        """
        # 注意：action_names, use_processed, norm 参数仅用于满足框架签名要求
        # 实际使用 self._se3_terms, self._use_processed, self.norm（已在 __init__ 中缓存）
        penalties = []
        for term in self._se3_terms:
            # 获取当前时间步的动作（原始或处理后）
            curr = term.processed_actions if self._use_processed else term.raw_actions  # (num_envs, 6)
            
            # 首次调用时初始化历史缓存，不计算惩罚
            if not self._initialized[term]:
                self._prev[term][:] = curr
                self._initialized[term] = True
                penalties.append(torch.zeros(env.num_envs, device=env.device))
                continue

            # 计算当前动作与上一时间步动作的差分
            diff = curr - self._prev[term]
            # 更新历史缓存
            self._prev[term][:] = curr

            # 根据指定的范数类型计算惩罚值
            if self.norm == 1:
                # L1 范数：绝对值之和
                pen = diff.abs().sum(dim=1)
            elif self.norm == float("inf"):
                # L∞ 范数：最大绝对值
                pen = diff.abs().max(dim=1).values
            else:
                # L2 范数（默认）：欧几里得距离
                pen = torch.linalg.vector_norm(diff, ord=2, dim=1)

            penalties.append(pen)

        # 多个动作项时取平均惩罚值
        return torch.stack(penalties, dim=0).mean(dim=0)

    def reset(self, env_ids: Sequence[int] | None = None):
        """在环境重置时同步平滑性缓存。

        Args:
            env_ids: 需要重置的环境索引列表。若为 None，则重置所有环境。
        """
        if env_ids is None:
            # 重置所有环境的历史缓存和初始化标志
            for term in self._se3_terms:
                self._prev[term].zero_()
                self._initialized[term] = False
        else:
            # 重置指定环境的历史缓存和初始化标志
            for term in self._se3_terms:
                self._prev[term][env_ids] = 0.0
                self._initialized[term] = False
