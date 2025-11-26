# TODO:该文件为se(3)动作项的实现文件，用于定义和管理se(3)动作项的具体行为。
# 主要思路参考 `/home/hac/isaac/AnyRotate/source/leaphand/leaphand/ideas/idea.ipynb`
# 实现可参考 `/home/hac/isaac/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`
# 有一个问题，idea.ipynb里的se(3)动作似乎天生对应的就是相对动作（relative action），所以没必要设置所谓的绝对动作项，当然相对项既然是默认的，也没必要命名为Relative

from __future__ import annotations

import logging
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from ..utils import math as math_leap

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class se3Action(ActionTerm):
    r"""se(3) 动作项基础实现类。

    该类实现了基于 se(3) 李代数旋量的动作空间。旋量 :math:`\mathcal{V}_b = [\omega_b^T, v_b^T]^T`
    表示末端坐标系 {b} 的瞬时速度，通过雅可比伪逆映射到关节空间：

    .. math::

        \theta(t+1) = \theta(t) + J_b^+(\theta(t)) \cdot \mathcal{V}_b(t) \cdot \Delta t

    主要特性：

    1. **任务空间表征**：直接控制末端的旋转和平移速度，更符合手指末端操作的直觉
    2. **坐标系无关**：旋量在局部坐标系 {b} 下定义，有利于策略泛化
    3. **虚拟 Xform 支持**：通过伴随变换支持虚拟设置的指尖坐标系
    4. **速度限制**：对角速度和线速度分别设置物理合理的上限

    .. note::
        该类使用 Moore-Penrose 伪逆计算雅可比逆。对于需要数值稳定性的场景，
        建议使用 :class:`se3dlsAction` 的 DLS (Damped Least Squares) 变体。
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

            # 计算从父刚体{b_raw}到虚拟Xform{b_can}的固定变换 T_raw_to_can
            # 这个变换在初始化时计算一次即可，因为它是固定的几何关系
            self._T_bb_prime = self._compute_xform_offset_transform()

            # 计算伴随变换矩阵 Ad_{T_raw_to_can}
            # 旋量变换: V_raw = Ad * V_can
            self._adjoint_matrix = math_leap.adjoint_transform(self._T_bb_prime)

            # 实际用于雅可比的是父刚体
            self._body_idx = self._parent_body_idx
            self._body_name = self._parent_body_name

            logger.info(
                f"se(3) 动作项 '{self.__class__.__name__}' 配置为虚拟Xform模式：\n"
                f"  虚拟末端: {self.cfg.target}\n"
                f"  父刚体: {self._body_name} [idx={self._body_idx}]\n"
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
            self._adjoint_matrix = None

            logger.info(
                f"se(3) 动作项 '{self.__class__.__name__}' 配置为真实刚体模式：\n"
                f"  末端刚体: {self._body_name} [idx={self._body_idx}]"
            )

        # 解析控制的关节（通常是单根手指链或手臂）
        # NOTE: 这里假设 cfg 中会有 joint_names 配置（类似于 JointAction）
        # 如果没有，则默认控制所有关节
        if hasattr(self.cfg, "joint_names") and self.cfg.joint_names is not None:
            self._joint_ids, self._joint_names = self._asset.find_joints(
                self.cfg.joint_names, preserve_order=getattr(self.cfg, "preserve_order", True)
            )
        else:
            # 默认控制所有关节
            self._joint_ids = slice(None)
            self._joint_names = self._asset.joint_names

        self._num_joints = len(self._joint_ids) if isinstance(self._joint_ids, (list, tuple)) else self._asset.num_joints

        logger.info(
            f"控制关节: {self._joint_names} [{self._joint_ids}]\n"
            f"关节总数: {self._num_joints}"
        )

        # 创建动作缓冲区
        # 输入动作维度是 6 (旋量维度：3角速度 + 3线速度)
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        # 处理后的动作是关节空间的目标（位置或速度，取决于use_pd配置）
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

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
        """处理后的动作张量（关节目标）。形状为 (num_envs, num_joints)。"""
        return self._processed_actions

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

        # 4. 如果是虚拟 Xform，应用伴随变换
        if self.cfg.is_xform:
            # V_raw = Ad_{T_raw_to_can} * V_can
            twist_body = torch.matmul(self._adjoint_matrix, twist_limited.unsqueeze(-1)).squeeze(-1)
        else:
            twist_body = twist_limited

        # 4. 获取当前的雅可比矩阵
        jacobian = self._get_jacobian()  # shape: (num_envs, 6, num_joints)

        # 5. 计算雅可比伪逆
        jacobian_inv = self._compute_jacobian_inverse(jacobian)  # shape: (num_envs, num_joints, 6)

        # 6. 计算关节空间的速度增量
        # delta_joint_vel = J^+ @ twist_body
        delta_joint_vel = torch.matmul(jacobian_inv, twist_body.unsqueeze(-1)).squeeze(-1)  # (num_envs, num_joints)

        # 7. 计算目标状态
        # 无论是否使用 PD，我们都计算下一时刻的目标位置
        # theta_target(t+1) = theta_actual(t) + delta_joint_vel(t) * dt
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_pos_target[:] = current_joint_pos + delta_joint_vel * self._dt
        self._joint_vel_target[:] = delta_joint_vel

        # 8. 应用关节限位
        if self.cfg.use_joint_limits:
            # 获取软关节限位
            # shape: (num_envs, num_controlled_joints, 2)
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            # 截断目标位置
            self._joint_pos_target[:] = torch.clamp(
                self._joint_pos_target, min=limits[..., 0], max=limits[..., 1]
            )

        # 更新 processed_actions (主要用于日志记录，存储位置目标)
        self._processed_actions[:] = self._joint_pos_target

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
        """获取当前末端刚体相对于控制关节的雅可比矩阵。

        Returns:
            几何雅可比矩阵。形状为 (num_envs, 6, num_joints)。
            前 3 行对应线速度，后 3 行对应角速度。
        """
        # 从 PhysX 获取所有刚体的雅可比矩阵
        # shape: (num_envs, num_bodies, 6, num_total_joints)
        all_jacobians = self._asset.root_physx_view.get_jacobians()

        # 提取目标 body 的雅可比
        # 根据基座是否固定，调整 body 索引和关节索引
        if self._asset.is_fixed_base:
            # 固定基座：Jacobian 不包含基座，索引偏移 -1
            jacobian_idx = self._body_idx - 1
            jacobi_joint_ids = self._joint_ids
        else:
            # 浮动基座：Jacobian 包含基座，索引不变
            # 但关节索引需要偏移 6 (跳过浮动基座的 6 个自由度)
            jacobian_idx = self._body_idx
            if isinstance(self._joint_ids, slice):
                # 如果是 slice(None)，则需要手动构造列表或处理
                # 这里简化处理，假设 slice(None) 意味着所有驱动关节
                # 对于浮动基座，通常不建议使用 slice(None) 除非明确知道含义
                # 为安全起见，我们重新获取所有关节索引
                all_joint_ids = list(range(self._asset.num_joints))
                jacobi_joint_ids = [i + 6 for i in all_joint_ids]
            else:
                jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # 提取对应控制关节的部分
        # shape: (num_envs, 6, num_joints)
        # 注意：get_jacobians() 返回的是 (num_envs, num_bodies, 6, num_dofs)
        # 其中 num_dofs 包含浮动基座的 6 DoF (如果存在)
        
        if jacobian_idx < 0:
             raise ValueError(f"试图获取固定基座 (index={self._body_idx}) 的雅可比矩阵，这是不允许的。")

        jacobian_physx = all_jacobians[:, jacobian_idx, :, jacobi_joint_ids]

        # PhysX 返回的雅可比矩阵顺序为 [线速度 v; 角速度 w]
        # 而本类使用的旋量定义为 [角速度 w; 线速度 v]
        # 因此需要交换前3行和后3行
        jacobian = torch.cat([jacobian_physx[:, 3:, :], jacobian_physx[:, :3, :]], dim=1)

        return jacobian

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
        return math_leap.dls_inv(jacobian, self.cfg.damping)