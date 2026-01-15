# 该文件为 se(3) 动作项的实现文件，用于定义和管理 se(3) 动作项的具体行为。
# 主要思路参考 `/home/hac/isaac/AnyRotate/source/leaphand/leaphand/ideas/idea.ipynb`
# 实现可参考 `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`
# NOTE：坐标系统一约定：{w}-World坐标系， {e}-Env坐标系， {s}-Base/Root坐标系
# {b}-End Effector坐标系（USD关节链中的最后一层刚体的坐标），{b'}-虚拟Xform坐标系（人为设置的指尖坐标系）
# 旋量、雅可比等均遵循 Modern Robotics 的约定
# 主要数学工具调用 `source/leaphand/leaphand/tasks/manager_based/leaphand/mdp/utils/math.py`

from __future__ import annotations

import logging
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from ..utils import math as math_leap
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class se3Action(ActionTerm):
    r"""se(3) 动作项基础实现类。

    本类实现任务空间的增量控制。策略输出 6 维“几何旋量”(geometric twist)
    :math:`\mathcal{V}=[\omega^T, v^T]^T`（角速度在前、线速度在后），并映射到关节空间。

    这里区分两个概念：

    1) **参考点 (reference point)**：线速度 :math:`v` 对应的点。
    2) **参考系 (reference frame)**：\(\omega, v\) 的坐标表达基。

    坐标系符号约定：
        - {w}: 世界坐标系 (world)
        - {s}: 机器人根/基坐标系 (root/base)
        - {b}: 末端真实刚体坐标系 (end-effector rigid body)
        - {b'}: 末端虚拟 Xform 坐标系 (virtual fingertip frame)

    配置项语义（与论文/笔记中一致）：
        - ``cfg.is_xform=False``:
            - 参考点：{b} 的原点
            - 若 ``cfg.use_body_frame=True``：动作为 :math:`\mathcal{V}_b`（参考系 {b}）
            - 若 ``cfg.use_body_frame=False``：动作为 :math:`\mathcal{V}_s`（参考系 {s}，参考点仍为 {b}）
        - ``cfg.is_xform=True``:
            - 参考点：{b'} 的原点
            - 若 ``cfg.use_body_frame=True``：动作为 :math:`\mathcal{V}_{b'}`（参考系 {b'}）
            - 若 ``cfg.use_body_frame=False``：动作为 :math:`\mathcal{V}_{s}^{b'}`（参考系 {s}，参考点为 {b'}）

    Notes:
        - 物理引擎返回的雅可比是“几何雅可比”(linear velocity of a point + angular velocity)，
          因此本实现对 {w}->{b}/{s} 的变换仅做旋转（不引入额外平移耦合项）。
        - 当 ``cfg.is_xform=True`` 且 ``cfg.use_body_frame=False`` 时，旧实现会将 {b'} 的伴随变换
          与 {w} 表达的雅可比混用；本实现显式构造 :math:`J_s` 在参考点 {b'} 上的版本，避免帧混用。
    """

    cfg: actions_cfg.se3ActionCfg
    """动作项配置。"""
    _asset: Articulation
    """应用动作项的关节机器人资产。"""

    def __init__(self, cfg: actions_cfg.se3ActionCfg, env: ManagerBasedEnv) -> None:
        """初始化 se(3) 动作项。

        Args:
            cfg: se(3) 动作项配置。
            env: 管理器基础强化学习环境实例。

        Raises:
            ValueError: 当末端 body 不存在或存在多个匹配时。
            RuntimeError: 当 is_xform=True 但无法获取虚拟Xform到真实刚体的变换时。
        """
        # 初始化基类
        super().__init__(cfg, env)

        # 解析目标末端 body
        if self.cfg.is_xform:
            # 如果是虚拟 Xform，需要找到其父刚体
            if self.cfg.parent is not None:
                # 用户显式指定了父刚体名称
                parent_body_ids, parent_body_names = self._asset.find_bodies(self.cfg.parent)
                if len(parent_body_ids) != 1:
                    raise ValueError(
                        f"配置的 parent 刚体名称 '{self.cfg.parent}' 匹配到 {len(parent_body_ids)} 个刚体：{parent_body_names}。"
                        f"期望恰好 1 个匹配。"
                    )
                self._parent_body_idx = parent_body_ids[0]
                self._parent_body_name = parent_body_names[0]
            else:
                # 自动推断：查找 target 的父 prim（必须是刚体）
                self._parent_body_name, self._parent_body_idx = self._find_parent_rigid_body()
                logger.info(
                    f"自动推断父刚体：{self._parent_body_name} [idx={self._parent_body_idx}]"
                )

            # 计算从父刚体{b}到虚拟Xform{b'}的固定变换 T_bb'
            # 这个变换在初始化时计算一次即可，因为它是固定的几何关系
            self._T_bb_prime = self._compute_xform_offset_transform()
            
            # 预计算伴随变换矩阵（根据技术路线选择使用哪个）
            # Ad_{T_bb'}: 用于将旋量从 {b'} 转到 {b} (旋量变换策略)
            self._Ad_b_bprime = math_leap.adjoint_transform(self._T_bb_prime)
            # Ad_{T_b'b}: 用于将雅可比从 {b} 转到 {b'} (雅可比变换策略)
            self._Ad_bprime_b = math_leap.adjoint_transform(
                math_leap.inverse_transform(self._T_bb_prime)
            )

            # 缓存偏移向量（在 {b} 下表达），用于构造 {s} 下 b' 参考点的雅可比
            self._p_bb_prime = self._T_bb_prime[:3, 3].clone()

            # 实际通过physx API可获取的雅可比是父刚体
            self._body_idx = self._parent_body_idx
            self._body_name = self._parent_body_name

            # 确定使用的技术路线
            strategy = "雅可比变换" if self.cfg.use_xform_jacobian else "旋量变换"
            logger.info(
                f"se(3) 动作项 '{self.__class__.__name__}' 配置为虚拟Xform模式：\n"
                f"  虚拟末端: {self.cfg.target}\n"
                f"  父刚体: {self._body_name} [idx={self._body_idx}]\n"
                f"  技术路线: {strategy}\n"
                f"  固定偏移已计算"
            )
        else:
            # 直接使用真实刚体
            body_ids, body_names = self._asset.find_bodies(self.cfg.target)
            if len(body_ids) != 1:
                raise ValueError(
                    f"末端刚体名称 '{self.cfg.target}' 匹配到 {len(body_ids)} 个刚体：{body_names}。"
                    f"期望恰好 1 个匹配。"
                )
            self._body_idx = body_ids[0]
            self._body_name = body_names[0]
            self._T_bb_prime = None  # 不需要偏移变换
            self._Ad_b_bprime = None  # 不需要旋量变换
            self._Ad_bprime_b = None  # 不需要雅可比变换
            self._p_bb_prime = None

            logger.info(
                f"se(3) 动作项 '{self.__class__.__name__}' 配置为真实刚体模式：\n"
                f"  末端刚体: {self._body_name} [idx={self._body_idx}]"
            )

        # 解析控制的关节（通常是单根手指链或手臂）
        # 这里假设 cfg 中会有 joint_names 配置（类似于 JointAction）
        # 如果没有，则默认控制所有关节
        if hasattr(self.cfg, "joint_names") and self.cfg.joint_names is not None:
            joint_ids, self._joint_names = self._asset.find_joints(
                self.cfg.joint_names, preserve_order=getattr(self.cfg, "preserve_order", True)
            )
        else:
            # 默认控制所有关节
            joint_ids = slice(None)
            self._joint_names = self._asset.joint_names

        self._joint_ids = self._normalize_joint_indices(joint_ids)
        self._num_joints = len(self._joint_ids)
        self._prepare_jacobian_indices()

        logger.info(
            f"控制关节: {self._joint_names} [{self._joint_ids}]\n"
            f"关节总数: {self._num_joints}"
        )

        # 创建动作缓冲区
        # 输入动作维度是 6 (旋量维度：3角速度 + 3线速度)
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        # 处理后的动作是缩放后的旋量（经过scaling和clamping）
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)

        # 解析角速度和线速度的限制，并计算缩放系数
        self._parse_velocity_limits()

        # 创建内部状态缓冲区
        # _joint_pos_target: 存储当前的目标位置（用于增量更新）
        # _joint_vel_target: 存储当前的目标速度（用于前馈控制）
        self._joint_pos_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._joint_vel_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # 初始化为默认位置，实际运行前会被 reset() 覆盖为当前位置
        self._joint_pos_target[:] = self._asset.data.default_joint_pos[:, self._joint_ids]

        # 获取环境的物理步长（用于速度到位置的积分）
        self._dt = self._env.step_dt

        logger.info(f"环境物理步长: {self._dt} s")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:  # 不同于关节空间动作，这里动作维度始终为6，关节空间的动作维度依赖于具体关节数量
        """动作维度，始终为 6 (se(3) 旋量的维度)。"""
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:  # 这里的raw_actions存储的就是来自rl_games或其他rl库归一化的[-1,1]动作值
        """原始动作张量（旋量）。形状为 (num_envs, 6)。"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """处理后的动作张量（缩放后的旋量命令）。形状为 (num_envs, 6)。"""
        return self._processed_actions

    @property
    def body_idx(self) -> int:
        """末端刚体索引（若为虚拟Xform则为其父刚体索引）。"""
        return self._body_idx

    @property
    def is_xform(self) -> bool:
        """是否使用虚拟Xform作为末端。"""
        return self.cfg.is_xform

    @property
    def Ad_bprime_b(self) -> torch.Tensor | None:
        """从真实刚体{b}到虚拟Xform{b'}的伴随变换矩阵。仅当 is_xform=True 时有效。"""
        return self._Ad_bprime_b if self.cfg.is_xform else None

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """动作项的 IO 描述符。

        包含动作维度、数据类型、动作类型等元信息。
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "se3Action"
        self._IO_descriptor.body_name = self._body_name
        self._IO_descriptor.body_idx = self._body_idx
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.is_xform = self.cfg.is_xform
        self._IO_descriptor.use_pd = self.cfg.use_pd
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        r"""处理输入的旋量动作，映射到关节空间。

        处理流程：

        1. 存储原始旋量动作
        2. 应用缩放映射：将 [-1, 1] 的动作映射到物理速度范围 [min, max]
        3. 应用速度限制（clamp，作为安全网）
        4. 如果是虚拟 Xform，应用伴随变换
        5. 获取当前雅可比矩阵
        6. 计算雅可比伪逆
        7. 计算关节速度增量：:math:`\Delta\dot{\theta} = J^+ \mathcal{V}_b`
        8. 如果 use_pd=True，积分为位置目标；否则直接作为速度目标

        Args:
            actions: 输入的旋量动作。形状为 (num_envs, 6)，
                     前3维为角速度 :math:`\omega_b`，后3维为线速度 :math:`v_b`。
                     假设输入范围为 [-1, 1]。
        """
        # 1. 存储原始动作
        self._raw_actions[:] = actions

        # 2. 应用缩放映射 (ax + b)
        # actions: [angular (3), linear (3)]
        if self._angular_vel_limits is not None:
            scaled_angular = actions[:, :3] * self._angular_scale + self._angular_bias
        else:
            scaled_angular = actions[:, :3]

        if self._linear_vel_limits is not None:
            scaled_linear = actions[:, 3:] * self._linear_scale + self._linear_bias
        else:
            scaled_linear = actions[:, 3:]
            
        twist = torch.cat([scaled_angular, scaled_linear], dim=1)

        # 3. 应用速度限制 (Safety Clamp)
        twist_limited = self._apply_velocity_limits(twist)

        # 3.5 可选：对旋量命令做额外过滤（例如 EMA 平滑）
        twist_filtered = self._filter_twist(twist_limited)

        # 存储处理后的旋量（用于奖励项和观察项）。
        # 注意：这里存储的是“最终用于 IK 的旋量命令（物理量级）”。
        self._processed_actions[:] = twist_filtered

        # 4-6. 选择一致的 (Jacobian, Twist) 组合并映射到关节速度
        if self.cfg.is_xform and self.cfg.use_body_frame and (not self.cfg.use_xform_jacobian):
            # V_{b'} --Ad_{T_bb'}--> V_b，然后用 J_b
            twist_for_ik = torch.matmul(self._Ad_b_bprime, twist_filtered.unsqueeze(-1)).squeeze(-1)
            jacobian = self._get_jacobian()  # 返回 J_b (use_xform_jacobian=False)
        else:
            # 其他情况：twist 与 _get_jacobian() 的参考系/参考点一致
            twist_for_ik = twist_filtered
            jacobian = self._get_jacobian()

        jacobian_inv = self._compute_jacobian_inverse(jacobian)
        joint_vel = (jacobian_inv @ twist_for_ik.unsqueeze(-1)).squeeze(-1)

        # 7. 计算目标状态
        # 无论是否使用 PD，我们都计算下一时刻的目标位置
        # theta_target(t+1) = theta_actual(t) + joint_vel(t) * dt
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_pos_target[:] = current_joint_pos + joint_vel * self._dt
        self._joint_vel_target[:] = joint_vel

        # 8. 应用关节限位
        if self.cfg.use_joint_limits:
            # 获取软关节限位
            # shape: (num_envs, num_controlled_joints, 2)
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            # 截断目标位置
            self._joint_pos_target[:] = torch.clamp(
                self._joint_pos_target, min=limits[..., 0], max=limits[..., 1]
            )

    def apply_actions(self):
        """将处理后的动作应用到机器人。

        根据 use_pd 配置，发送位置目标或速度目标到关节控制器。
        """
        if self.cfg.use_pd:
            # PD 控制模式：同时发送位置目标和速度前馈
            # tau = Kp * (pos_target - pos) + Kd * (vel_target - vel)
            self._asset.set_joint_position_target(self._joint_pos_target, joint_ids=self._joint_ids)
            self._asset.set_joint_velocity_target(self._joint_vel_target, joint_ids=self._joint_ids)
        else:
            # 纯位置控制模式：仅发送位置目标，速度目标设为0（仅利用阻尼）
            self._asset.set_joint_position_target(self._joint_pos_target, joint_ids=self._joint_ids)
            self._asset.set_joint_velocity_target(torch.zeros_like(self._joint_vel_target), joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """重置动作项状态。

        Args:
            env_ids: 需要重置的环境索引。如果为 None，则重置所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)

        # 重置原始动作
        self._raw_actions[env_ids] = 0.0

        # 重置目标位置为当前位置，避免跳变
        # 注意：分步索引以避免当 env_ids 和 joint_ids 都是列表时的广播错误
        self._joint_pos_target[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        # 重置目标速度为 0
        self._joint_vel_target[env_ids] = 0.0

        # 子类可覆写/扩展 reset 行为（例如重置滤波器状态）

    def _filter_twist(self, twist: torch.Tensor) -> torch.Tensor:
        """过滤处理后的旋量命令。

        默认实现为恒等映射。子类可实现 EMA / 限幅等滤波操作。

        Args:
            twist: 处理后的旋量（物理量级），形状 (num_envs, 6)。

        Returns:
            过滤后的旋量，形状同输入。
        """
        return twist

    """
    Helper methods.
    """

    def _find_parent_rigid_body(self) -> tuple[str, int]:
        """自动推断虚拟 Xform 的父刚体。

        从 target（虚拟 Xform）向上遍历 USD prim 层级，找到第一个具有 RigidBodyAPI 的 prim。

        Returns:
            tuple[str, int]: (父刚体名称, 父刚体在 body_names 中的索引)

        Raises:
            RuntimeError: 当无法找到虚拟 Xform 或其父刚体时。
            ValueError: 当推断出的父刚体不在机器人的 body_names 列表中时。
        """
        from isaaclab.sim.utils import get_current_stage
        from pxr import UsdPhysics
        import isaaclab.sim as sim_utils

        stage = get_current_stage()

        # 获取第一个环境的 prim 路径
        env_prim_path = sim_utils.find_first_matching_prim(self._asset.cfg.prim_path).GetPath().pathString

        # 首先需要找到 target Xform 在 USD 中的完整路径
        # 由于不知道 target 的完整路径，我们需要在机器人子树中搜索
        target_prim = self._find_prim_by_name(stage, env_prim_path, self.cfg.target)

        if target_prim is None or not target_prim.IsValid():
            raise RuntimeError(
                f"无法在机器人 prim '{env_prim_path}' 下找到名为 '{self.cfg.target}' 的 Xform。"
            )

        logger.info(f"找到虚拟 Xform: {target_prim.GetPrimPath()}")

        # 向上遍历层级，找到第一个刚体
        current_prim = target_prim.GetParent()
        parent_body_prim = None

        while current_prim.IsValid():
            # 检查当前 prim 是否有 RigidBodyAPI
            if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                parent_body_prim = current_prim
                break

            # 如果到达机器人根节点或更高层级，停止搜索
            if str(current_prim.GetPrimPath()) == env_prim_path:
                break

            current_prim = current_prim.GetParent()

        if parent_body_prim is None:
            raise RuntimeError(
                f"无法找到虚拟 Xform '{self.cfg.target}' 的父刚体。"
                f"请检查 USD 层级结构，或显式指定 'parent' 参数。"
            )

        # 提取刚体名称（相对于机器人根的路径最后一段）
        parent_prim_path = str(parent_body_prim.GetPrimPath())
        # 例如 "/World/envs/env_0/Robot/fingertip" -> "fingertip"
        parent_body_name = parent_prim_path.replace(env_prim_path + "/", "").split("/")[-1]

        # 验证并获取刚体索引
        body_ids, body_names = self._asset.find_bodies(parent_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"推断出的父刚体名称 '{parent_body_name}' 在机器人中匹配到 {len(body_ids)} 个刚体：{body_names}。"
                f"期望恰好 1 个匹配。请检查 USD 结构或显式指定 'parent' 参数。"
            )

        return body_names[0], body_ids[0]

    def _find_prim_by_name(self, stage, root_path: str, target_name: str):
        """在 USD stage 中搜索指定名称的 prim。

        Args:
            stage: USD Stage 对象。
            root_path: 搜索的根路径。
            target_name: 目标 prim 的名称。

        Returns:
            找到的 Usd.Prim 或 None。
        """
        from pxr import Usd

        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            return None

        # 深度优先搜索
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == target_name:
                return prim

        return None

    def _compute_xform_offset_transform(self) -> torch.Tensor:
        """计算虚拟 Xform 相对于父刚体的固定变换 T_bb'。

        Returns:
            齐次变换矩阵。形状为 (4, 4)，表示从父刚体坐标系到虚拟Xform坐标系的变换。

        Raises:
            RuntimeError: 当无法从 USD Stage 获取变换时。
        """
        from isaaclab.sim.utils import get_current_stage
        from pxr import UsdGeom
        import isaaclab.sim as sim_utils

        stage = get_current_stage()

        # 获取第一个环境的 prim 路径
        env_prim_path = sim_utils.find_first_matching_prim(self._asset.cfg.prim_path).GetPath().pathString

        # 动态搜索父刚体和虚拟 Xform 的 USD prim
        parent_prim = self._find_prim_by_name(stage, env_prim_path, self._parent_body_name)
        xform_prim = self._find_prim_by_name(stage, env_prim_path, self.cfg.target)

        if parent_prim is None or not parent_prim.IsValid():
            raise RuntimeError(
                f"无法在机器人 prim '{env_prim_path}' 下找到父刚体 '{self._parent_body_name}'。"
            )

        if xform_prim is None or not xform_prim.IsValid():
            raise RuntimeError(
                f"无法在机器人 prim '{env_prim_path}' 下找到虚拟 Xform '{self.cfg.target}'。"
            )

        logger.info(
            f"计算相对变换:\n"
            f"  父刚体路径: {parent_prim.GetPrimPath()}\n"
            f"  Xform路径: {xform_prim.GetPrimPath()}"
        )

        # 获取世界坐标系下的位姿
        xform_cache = UsdGeom.XformCache()

        parent_transform_w = xform_cache.GetLocalToWorldTransform(parent_prim)
        xform_transform_w = xform_cache.GetLocalToWorldTransform(xform_prim)

        # 计算相对变换: T_bb' = T_b^{-1} * T_{b'}
        parent_transform_inv = parent_transform_w.GetInverse()
        relative_transform = parent_transform_inv * xform_transform_w

        # 转换为 PyTorch tensor (4x4 齐次变换矩阵)
        import numpy as np

        transform_np = np.array(relative_transform, dtype=np.float32).T  # USD uses row-major, PyTorch column-major
        T_bb_prime = torch.tensor(transform_np, device=self.device, dtype=torch.float32)

        logger.info(
            f"虚拟Xform相对变换 T_{{bb'}} 计算完成:\n"
            f"  平移: {T_bb_prime[:3, 3]}\n"
            f"  旋转矩阵:\n{T_bb_prime[:3, :3]}"
        )

        return T_bb_prime

    def _get_jacobian(self) -> torch.Tensor:
        """获取与当前动作定义一致的几何雅可比矩阵。

        Returns:
            torch.Tensor: 雅可比矩阵，形状为 (num_envs, 6, num_joints)。其含义与 cfg 绑定：
                - is_xform=False:
                    - use_body_frame=True  -> J_b (参考点 {b}，参考系 {b})
                    - use_body_frame=False -> J_s (参考点 {b}，参考系 {s})
                - is_xform=True:
                    - use_body_frame=True & use_xform_jacobian=False  -> J_b (参考点 {b}，参考系 {b})
                    - use_body_frame=True & use_xform_jacobian=True   -> J_{b'} (参考点 {b'}，参考系 {b'})
                    - use_body_frame=False                            -> J_s^{b'} (参考点 {b'}，参考系 {s})

        Notes:
            - PhysX 返回的几何雅可比在世界坐标系 {w} 下表达且速度顺序为 [v; w]。
              本实现统一转换为 [w; v]。
            - {w}->{b}/{s} 的变换仅做旋转（保持参考点不变）。
        """
        jac_w = self._get_parent_body_jacobian_world()

        if self.cfg.use_body_frame:
            jac_frame = self._rotate_jacobian_to_frame(jac_w, frame="body")
        else:
            jac_frame = self._rotate_jacobian_to_frame(jac_w, frame="root")

        # 参考点处理
        if not self.cfg.is_xform:
            return jac_frame

        # is_xform=True
        if self.cfg.use_body_frame:
            if self.cfg.use_xform_jacobian:
                # J_{b'} = Ad_{T_{b'b}} @ J_b
                Ad = self._Ad_bprime_b
                return Ad.unsqueeze(0).expand(jac_frame.shape[0], -1, -1) @ jac_frame
            # use_xform_jacobian=False 时，process_actions 会走 V_{b'}->V_b 并使用 J_b
            return jac_frame

        # use_body_frame=False: J_s^{b'}，参考点从 {b} 平移到 {b'}，参考系仍为 {s}
        r_s = self._compute_offset_in_root_frame()
        r_skew = math_leap.skew_symmetric(r_s)  # (N, 3, 3)
        jac_out = jac_frame.clone()
        jac_out[:, 3:, :] = jac_frame[:, 3:, :] + torch.bmm(r_skew, jac_frame[:, :3, :])
        return jac_out

    def _get_parent_body_jacobian_world(self) -> torch.Tensor:
        """从 PhysX 读取父刚体（或真实刚体）的雅可比并转换为 {w} 下 [w; v] 顺序。"""
        all_jacobians = self._asset.root_physx_view.get_jacobians()
        if self._jacobi_body_idx < 0:
            raise ValueError(
                f"试图访问 body_idx={self._body_idx} 的雅可比矩阵，但固定基座偏移后索引 < 0。"
            )
        jacobian_physx = all_jacobians[:, self._jacobi_body_idx, :, :][:, :, self._jacobi_joint_ids]
        # PhysX: [v; w] -> project: [w; v]
        return torch.cat([jacobian_physx[:, 3:, :], jacobian_physx[:, :3, :]], dim=1)

    def _rotate_jacobian_to_frame(self, jacobian_world: torch.Tensor, frame: str) -> torch.Tensor:
        """将 {w} 下的雅可比旋转到指定参考系（保持参考点不变）。

        Args:
            jacobian_world: (N, 6, nj) in {w}.
            frame: "body" -> {b}, "root" -> {s}.
        """
        if frame not in ("body", "root"):
            raise ValueError(f"Unsupported frame: {frame}")

        if frame == "body":
            quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        else:
            quat_w = self._asset.data.root_quat_w

        quat_w_wxyz = math_utils.convert_quat(quat_w, to="wxyz")
        R_fw = math_utils.matrix_from_quat(math_utils.quat_inv(quat_w_wxyz))  # world -> frame

        jac = jacobian_world.clone()
        jac[:, :3, :] = torch.bmm(R_fw, jacobian_world[:, :3, :])
        jac[:, 3:, :] = torch.bmm(R_fw, jacobian_world[:, 3:, :])
        return jac

    def _compute_offset_in_root_frame(self) -> torch.Tensor:
        """计算 b->b' 的偏移向量在 {s} 下的表达。

        Returns:
            torch.Tensor: r_s, shape (num_envs, 3).
        """
        if not self.cfg.is_xform or self._p_bb_prime is None:
            raise RuntimeError("Offset is only defined when is_xform=True")

        # R_sb = R_sw * R_wb
        body_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_quat_w = self._asset.data.root_quat_w
        body_quat_w_wxyz = math_utils.convert_quat(body_quat_w, to="wxyz")
        root_quat_w_wxyz = math_utils.convert_quat(root_quat_w, to="wxyz")
        R_wb = math_utils.matrix_from_quat(body_quat_w_wxyz)
        R_sw = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w_wxyz))
        R_sb = torch.bmm(R_sw, R_wb)

        p = self._p_bb_prime.view(1, 3, 1).expand(self.num_envs, -1, -1)
        return torch.bmm(R_sb, p).squeeze(-1)

    def _normalize_joint_indices(self, joint_ids) -> list[int]:
        """将关节索引统一为列表形式。"""

        if isinstance(joint_ids, slice):
            return list(range(self._asset.num_joints))[joint_ids]

        if isinstance(joint_ids, torch.Tensor):
            return joint_ids.tolist()

        if isinstance(joint_ids, np.ndarray):
            return joint_ids.astype(int).tolist()

        return list(joint_ids)

    def _prepare_jacobian_indices(self) -> None:
        """根据基座类型解析 PhysX 雅可比的索引。"""

        if self._asset.is_fixed_base:
            # 固定基座：PhysX 雅可比不包含基座刚体，索引需要减 1
            self._jacobi_body_idx = self._body_idx - 1
            if self._jacobi_body_idx < 0:
                raise ValueError(
                    "固定基座场景下，目标刚体索引为 0，无法在 PhysX Jacobian 中找到对应条目。"
                )
            self._jacobi_joint_ids = self._joint_ids  # 关节索引保持不变
        else:
            # 浮动基座：PhysX 雅可比包含 6 个虚拟自由度（基座位姿）
            self._jacobi_body_idx = self._body_idx  # 刚体索引保持不变
            self._jacobi_joint_ids = [idx + 6 for idx in self._joint_ids]  # 关节索引需要偏移 6

    def _compute_jacobian_inverse(self, jacobian: torch.Tensor) -> torch.Tensor:
        """计算雅可比矩阵的伪逆。

        该方法在基类中使用 Moore-Penrose 伪逆。子类可以重写此方法以使用其他方法
        （如 DLS）。

        Args:
            jacobian: 雅可比矩阵。形状为 (num_envs, 6, num_joints)。

        Returns:
            雅可比伪逆矩阵。形状为 (num_envs, num_joints, 6)。
        """
        return math_leap.pseudo_inv(jacobian)

    def _parse_velocity_limits(self):
        """解析和处理角速度与线速度的限制配置。

        根据配置创建上下界张量，用于后续的 clamp 操作。
        同时计算缩放系数 (scale) 和偏移量 (bias)，用于将 [-1, 1] 的动作映射到 [min, max]。
        公式: y = x * scale + bias
        scale = (max - min) / 2
        bias = (max + min) / 2
        """
        # 解析角速度限制
        if self.cfg.angular_limits is not None:
            if isinstance(self.cfg.angular_limits, (int, float)):
                # 单个值：对所有分量使用相同的限制 [-pi/k, pi/k]
                k = self.cfg.angular_limits
                limit = torch.pi / k
                self._angular_vel_limits = torch.tensor(
                    [-limit, -limit, -limit, limit, limit, limit], device=self.device
                ).view(2, 3)  # shape: (2, 3) for [lower, upper]
            else:
                # 三个分量的独立限制
                kx, ky, kz = self.cfg.angular_limits
                self._angular_vel_limits = torch.tensor(
                    [-torch.pi / kx, -torch.pi / ky, -torch.pi / kz, torch.pi / kx, torch.pi / ky, torch.pi / kz],
                    device=self.device,
                ).view(2, 3)
            
            # 计算角速度缩放参数
            self._angular_scale = (self._angular_vel_limits[1] - self._angular_vel_limits[0]) / 2.0
            self._angular_bias = (self._angular_vel_limits[1] + self._angular_vel_limits[0]) / 2.0
        else:
            self._angular_vel_limits = None
            self._angular_scale = 1.0
            self._angular_bias = 0.0

        # 解析线速度限制
        if self.cfg.linear_limits is not None:
            sqrt_3 = torch.sqrt(torch.tensor(3.0, device=self.device))
            if isinstance(self.cfg.linear_limits, (int, float)):
                # 单个值：对所有分量使用相同的限制
                v = self.cfg.linear_limits
                limit = sqrt_3 * v
                self._linear_vel_limits = torch.tensor(
                    [-limit, -limit, -limit, limit, limit, limit], device=self.device
                ).view(2, 3)
            else:
                # 三个分量的独立限制
                vx, vy, vz = self.cfg.linear_limits
                self._linear_vel_limits = torch.tensor(
                    [
                        -sqrt_3 * vx,
                        -sqrt_3 * vy,
                        -sqrt_3 * vz,
                        sqrt_3 * vx,
                        sqrt_3 * vy,
                        sqrt_3 * vz,
                    ],
                    device=self.device,
                ).view(2, 3)
            
            # 计算线速度缩放参数
            self._linear_scale = (self._linear_vel_limits[1] - self._linear_vel_limits[0]) / 2.0
            self._linear_bias = (self._linear_vel_limits[1] + self._linear_vel_limits[0]) / 2.0
        else:
            self._linear_vel_limits = None
            self._linear_scale = 1.0
            self._linear_bias = 0.0

        logger.info(
            f"速度限制与缩放设置:\n"
            f"  角速度限制: {self._angular_vel_limits if self._angular_vel_limits is not None else 'None'}\n"
            f"  角速度缩放: scale={self._angular_scale}, bias={self._angular_bias}\n"
            f"  线速度限制: {self._linear_vel_limits if self._linear_vel_limits is not None else 'None'}\n"
            f"  线速度缩放: scale={self._linear_scale}, bias={self._linear_bias}"
        )

    def _apply_velocity_limits(self, twist: torch.Tensor) -> torch.Tensor:
        """对旋量应用速度限制。

        Args:
            twist: 输入旋量。形状为 (num_envs, 6)，前3维为角速度，后3维为线速度。

        Returns:
            限制后的旋量。形状与输入相同。
        """
        twist_limited = twist.clone()

        # 限制角速度
        if self._angular_vel_limits is not None:
            twist_limited[:, :3] = torch.clamp(
                twist[:, :3], min=self._angular_vel_limits[0], max=self._angular_vel_limits[1]
            )

        # 限制线速度
        if self._linear_vel_limits is not None:
            twist_limited[:, 3:] = torch.clamp(
                twist[:, 3:], min=self._linear_vel_limits[0], max=self._linear_vel_limits[1]
            )

        return twist_limited


class se3dlsAction(se3Action):
    r"""se(3) 动作项 DLS（Damped Least Squares）实现类。

    该类继承自 :class:`se3Action`，使用阻尼最小二乘法计算雅可比逆，
    提供更好的数值稳定性，特别是在接近奇异位形时。

    DLS 方法通过在优化目标中加入正则化项来避免小奇异值导致的数值爆炸：

    .. math::

        J_{dls}^{\dagger} = J^T (JJ^T + \lambda^2 I)^{-1}

    其中 :math:`\lambda` 是阻尼系数，由配置 :attr:`cfg.damping` 指定。
    """

    cfg: actions_cfg.se3dlsActionsCfg
    """DLS 动作项配置。"""

    def _compute_jacobian_inverse(self, jacobian: torch.Tensor) -> torch.Tensor:
        """使用 DLS 方法计算雅可比伪逆。

        Args:
            jacobian: 雅可比矩阵。形状为 (num_envs, 6, num_joints)。

        Returns:
            DLS 伪逆矩阵。形状为 (num_envs, num_joints, 6)。
        """
        return math_leap.dls_cholesky_inv(jacobian, self.cfg.damping)


class se3dlsEmaAction(se3dlsAction):
    r"""se(3) 动作项 DLS + EMA 平滑实现类。

    EMA 作用于 se(3) 旋量命令（物理量级），以抑制策略输出的高频抖动。
    """

    cfg: actions_cfg.se3dlsEmaActionsCfg

    def __init__(self, cfg: actions_cfg.se3dlsEmaActionsCfg, env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self._ema_twist = torch.zeros(self.num_envs, 6, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        # 重置 EMA 状态为 0，避免跨 episode 泄露
        self._ema_twist[env_ids] = 0.0

    def _filter_twist(self, twist: torch.Tensor) -> torch.Tensor:
        # EMA: V_ema = alpha * V + (1-alpha) * V_ema_prev
        alpha = self.cfg.alpha
        if alpha >= 1.0:
            # 快路径：不做平滑
            self._ema_twist[:] = twist
            return twist
        self._ema_twist[:] = alpha * twist + (1.0 - alpha) * self._ema_twist
        return self._ema_twist


class se3wdlsAction(se3dlsAction):
    """se(3) 动作项 WDLS（Weighted Damped Least Squares）实现类。

    Notes:
        当前实现采用与 :class:`se3dlsAction` 相同的 DLS 伪逆计算逻辑。
        该类的存在主要用于与配置类 `se3wdlsActionsCfg` 对齐，避免任务包导入时
        因符号缺失导致 gym 注册失败。
    """

    cfg: actions_cfg.se3wdlsActionsCfg

    def _compute_jacobian_inverse(self, jacobian: torch.Tensor) -> torch.Tensor:
        return math_leap.dls_cholesky_inv(jacobian, self.cfg.damping)


class se3adlsAction(se3dlsAction):
    """se(3) 动作项 ADLS（Adaptive Damped Least Squares）实现类。

    采用选择性阻尼 (Selective Damping) 的 DLS 伪逆：仅对接近奇异的方向施加阻尼。
    """

    cfg: actions_cfg.se3adlsActionsCfg

    def _compute_jacobian_inverse(self, jacobian: torch.Tensor) -> torch.Tensor:
        return math_leap.dls_inv(
            J=jacobian,
            damping=self.cfg.damping,
            method="svd",
            singular_threshold=self.cfg.singular_threshold,
            selective=True,
        )


class se3awdlsAction(se3dlsAction):
    """se(3) 动作项 AWDLS（Adaptive Weighted Damped Least Squares）实现类。

    Notes:
        当前实现复用 ADLS 的选择性阻尼逻辑；权重项 (W_q/W_x) 的显式加权尚未引入。
        该类用于保证 `se3awdlsActionsCfg` 在导入时可用。
    """

    cfg: actions_cfg.se3awdlsActionsCfg

    def _compute_jacobian_inverse(self, jacobian: torch.Tensor) -> torch.Tensor:
        return math_leap.dls_inv(
            J=jacobian,
            damping=self.cfg.damping,
            method="svd",
            singular_threshold=self.cfg.singular_threshold,
            selective=True,
        )