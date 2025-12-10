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

    该类实现了基于 se(3) 李代数旋量的动作空间。旋量 :math:`\mathcal{V}_b = [\omega_b^T, v_b^T]^T`
    表示末端坐标系 {b} 的瞬时速度，通过雅可比伪逆映射到关节空间：

    .. math::

        \theta(t+1) = \theta(t) + J_b^+(\theta(t)) \cdot \mathcal{V}_b(t) \cdot \Delta t

    主要特性：

    1. **任务空间表征**：直接控制末端的旋转和平移速度，更符合手指末端操作的直觉
    2. **坐标系无关**：旋量在局部坐标系 {b}(或 {b'}) 下定义，有利于策略泛化
    3. **虚拟 Xform 支持**：通过伴随变换支持虚拟设置的指尖坐标系 {b'}
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

        # 存储处理后的旋量（用于奖励项和观察项）
        self._processed_actions[:] = twist_limited

        # 4-6. 根据技术路线选择处理流程
        if self.cfg.is_xform and not self.cfg.use_xform_jacobian:
            # 技术路线1（默认）：变换旋量策略 (AnyRotate-temp 的方式)
            # V_b = Ad_{T_bb'} @ V_b'
            twist_body = torch.matmul(self._Ad_b_bprime, twist_limited.unsqueeze(-1)).squeeze(-1)
            # 获取真实刚体的雅可比 J_b
            jacobian = self._get_jacobian()  # shape: (num_envs, 6, num_joints)
            # 计算雅可比伪逆 J_b^+
            jacobian_inv = self._compute_jacobian_inverse(jacobian)
            # 计算关节速度：dθ = J_b^+ @ V_b
            joint_vel = (jacobian_inv @ twist_body.unsqueeze(-1)).squeeze(-1)
        elif self.cfg.is_xform and self.cfg.use_xform_jacobian:
            # 技术路线2：变换雅可比策略
            # 获取真实刚体的雅可比 J_b
            jacobian_body = self._get_jacobian()  # shape: (num_envs, 6, num_joints)
            # 变换到虚拟帧：J_b' = Ad_{T_b'b} @ J_b
            Ad_batch = self._Ad_bprime_b.unsqueeze(0).expand(jacobian_body.shape[0], -1, -1)
            jacobian_prime = Ad_batch @ jacobian_body
            # 计算雅可比伪逆 J_b'^+
            jacobian_inv = self._compute_jacobian_inverse(jacobian_prime)
            # 计算关节速度：dθ = J_b'^+ @ V_b'
            joint_vel = (jacobian_inv @ twist_limited.unsqueeze(-1)).squeeze(-1)
        else:
            # 非虚拟 Xform 模式：直接使用真实刚体
            jacobian = self._get_jacobian()
            jacobian_inv = self._compute_jacobian_inverse(jacobian)
            joint_vel = (jacobian_inv @ twist_limited.unsqueeze(-1)).squeeze(-1)

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
        """获取以 {w} 或 {b} 表示的几何雅可比矩阵（根据 use_body_frame 配置）。
        
        该方法执行以下变换流程：
        1. 从 PhysX 获取原始雅可比矩阵（世界坐标系表示）
        2. 调整速度分量顺序（PhysX: [v; w] → 本项目: [w; v]）
        3. (可选) 通过旋转矩阵将雅可比从世界坐标系 {w} 转换到刚体坐标系 {b}
        
        Returns:
            torch.Tensor: 几何雅可比矩阵。形状为 (num_envs, 6, num_joints)。
                         - 当 use_body_frame=True 时：表示 J_b，映射到刚体坐标系
                         - 当 use_body_frame=False 时：表示 J_w，映射到世界坐标系
        
        Raises:
            ValueError: 当固定基座场景下雅可比索引计算错误时。
        """
        # 1. 获取 PhysX 提供的所有刚体的雅可比矩阵
        # shape: (num_envs, num_bodies, 6, num_dofs)
        all_jacobians = self._asset.root_physx_view.get_jacobians()

        # 2. 验证雅可比索引的有效性
        if self._jacobi_body_idx < 0:
            raise ValueError(
                f"试图访问 body_idx={self._body_idx} 的雅可比矩阵，但固定基座偏移后索引 < 0。"
            )

        # 3. 提取目标刚体和控制关节对应的雅可比子矩阵
        jacobian_physx = all_jacobians[:, self._jacobi_body_idx, :, :][:, :, self._jacobi_joint_ids]  # shape: (num_envs, 6, num_dofs)

        # 4. 调整速度分量顺序
        # PhysX 约定: [v_x, v_y, v_z, w_x, w_y, w_z]^T
        # 本项目约定（Modern Robotics）: [w_x, w_y, w_z, v_x, v_y, v_z]^T
        jacobian_world = torch.cat([jacobian_physx[:, 3:, :], jacobian_physx[:, :3, :]], dim=1)  # shape: (num_envs, 6, num_dofs)

        # 5. (可选) 从世界坐标系 {w} 转换到刚体坐标系 {b}
        if self.cfg.use_body_frame:
            # 获取刚体在世界坐标系下的姿态(仅需要旋转)
            body_quat_w = self._asset.data.body_quat_w[:, self._body_idx]  # shape: (num_envs, 4)
            
            # 计算从世界坐标系到刚体坐标系的旋转矩阵 R_bw = R(q_b^{-1})
            body_quat_w_wxyz = math_utils.convert_quat(body_quat_w, to="wxyz")  # 转换为 (w, x, y, z) 格式
            R_bw = math_utils.matrix_from_quat(math_utils.quat_inv(body_quat_w_wxyz))  # (num_envs, 3, 3)
            
            # 应用旋转变换到雅可比的角速度和线速度部分
            # J_b = R_bw @ J_w (对每个速度分量独立旋转)
            jacobian_output = jacobian_world.clone()
            jacobian_output[:, :3, :] = torch.bmm(R_bw, jacobian_world[:, :3, :])  # 角速度部分
            jacobian_output[:, 3:, :] = torch.bmm(R_bw, jacobian_world[:, 3:, :])  # 线速度部分
        else:
            # 直接返回世界坐标系下的雅可比
            jacobian_output = jacobian_world

        return jacobian_output

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
            self._jacobi_body_idx = self._body_idx - 1
            if self._jacobi_body_idx < 0:
                raise ValueError(
                    "固定基座场景下，目标刚体索引为 0，无法在 PhysX Jacobian 中找到对应条目。"
                )
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [idx + 6 for idx in self._joint_ids]

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
    

class se3wdlsAction(se3dlsAction):
    r"""se(3) 动作项 WDLS（Weighted Damped Least Squares）实现类。

    该类继承自 :class:`se3dlsAction`，通过引入任务空间权重矩阵 :math:`W_x` 和关节空间权重矩阵 :math:`W_q`，
    解决标准DLS存在的量纲不一致和任务优先级缺失问题。

    **数学形式：**

    加权DLS求解以下优化问题：

    .. math::

        \min_{\dot{\theta}} \| W_x (J_b \dot{\theta} - \mathcal{V}_b) \|^2 + \lambda^2 \| W_q \dot{\theta} \|^2

    通过变量代换 :math:`\dot{\phi} = W_q \dot{\theta}`，:math:`\tilde{J}_b = W_x J_b W_q^{-1}`，
    :math:`\tilde{\mathcal{V}}_b = W_x \mathcal{V}_b`，可转化为标准DLS形式：

    .. math::

        \min_{\dot{\phi}} \| \tilde{J}_b \dot{\phi} - \tilde{\mathcal{V}}_b \|^2 + \lambda^2 \| \dot{\phi} \|^2

    解为：

    .. math::

        \dot{\theta} = W_q^{-1} \tilde{J}_b^T (\tilde{J}_b \tilde{J}_b^T + \lambda^2 I)^{-1} W_x \mathcal{V}_b

    **物理意义：**

    1. **量纲归一化**：通过 :math:`W_x = \mathrm{diag}(w_\omega I_3, w_v I_3)` 统一角速度（rad/s）和线速度（m/s）的量级
    2. **关节限位回避**：通过 :math:`W_q = \mathrm{diag}(w_1, \dots, w_n)` 惩罚接近限位的关节运动
    3. **性能指标优化**：使用加权雅可比计算的操作度更能反映实际任务执行能力

    Reference:
        - Modern Robotics, Section 6.2: Numerical Inverse Kinematics
        - Sciavicco & Siciliano (2000). "Modelling and Control of Robot Manipulators"
    """

    cfg: actions_cfg.se3wdlsActionsCfg
    """WDLS 动作项配置。"""

    def __init__(self, cfg: actions_cfg.se3wdlsActionsCfg, env: ManagerBasedEnv) -> None:
        """初始化 WDLS 动作项。

        Args:
            cfg: WDLS 动作项配置。
            env: 管理器基础强化学习环境实例。

        Raises:
            ValueError: 当权重矩阵维度不匹配时。
        """
        # 调用父类初始化
        super().__init__(cfg, env)

        # 初始化任务空间权重矩阵 W_x (6x6)
        if self.cfg.W_x is not None:
            W_x_list = list(self.cfg.W_x) if isinstance(self.cfg.W_x, (tuple, list)) else [self.cfg.W_x] * 6
            if len(W_x_list) != 6:
                raise ValueError(
                    f"W_x 权重矩阵应为 6 个元素，实际收到 {len(W_x_list)} 个。"
                )
            self._W_x = torch.diag(torch.tensor(W_x_list, device=self.device, dtype=torch.float32))
        else:
            # 默认：单位矩阵（不加权）
            self._W_x = torch.eye(6, device=self.device, dtype=torch.float32)

        # 初始化关节空间权重矩阵 W_q (num_joints x num_joints)
        if self.cfg.W_q is not None:
            W_q_list = list(self.cfg.W_q) if isinstance(self.cfg.W_q, (tuple, list)) else [self.cfg.W_q] * self._num_joints
            if len(W_q_list) != self._num_joints:
                raise ValueError(
                    f"W_q 权重矩阵应为 {self._num_joints} 个元素（匹配关节数），实际收到 {len(W_q_list)} 个。"
                )
            self._W_q = torch.diag(torch.tensor(W_q_list, device=self.device, dtype=torch.float32))
        else:
            # 默认：单位矩阵（不加权）
            self._W_q = torch.eye(self._num_joints, device=self.device, dtype=torch.float32)

        # 预计算 W_q 的逆矩阵（对角矩阵求逆直接对角线元素取倒数）
        self._W_q_inv = torch.diag(1.0 / torch.diag(self._W_q))

        logger.info(
            f"WDLS 权重矩阵初始化:\n"
            f"  W_x (任务空间): {torch.diag(self._W_x).tolist()}\n"
            f"  W_q (关节空间): {torch.diag(self._W_q).tolist()}"
        )

    def process_actions(self, actions: torch.Tensor):
        r"""处理输入的旋量动作，使用加权DLS映射到关节空间。

        处理流程（与 se3Action 不同）：

        1. 存储原始旋量动作
        2. 应用缩放和限制
        3. 计算加权旋量：:math:`\tilde{\mathcal{V}}_b = W_x \mathcal{V}_b`
        4. 获取当前雅可比矩阵 :math:`J_b`
        5. 计算加权雅可比：:math:`\tilde{J}_b = W_x J_b W_q^{-1}`
        6. 计算加权DLS伪逆
        7. 计算关节速度增量：:math:`\dot{\theta} = W_q^{-1} \tilde{J}_b^+ \tilde{\mathcal{V}}_b`
        8. 积分为位置目标

        Args:
            actions: 输入的旋量动作。形状为 (num_envs, 6)。
        """
        # 1-2. 存储原始动作、应用缩放和限制（与基类相同）
        self._raw_actions[:] = actions

        # 应用缩放映射
        if self._angular_vel_limits is not None:
            scaled_angular = actions[:, :3] * self._angular_scale + self._angular_bias
        else:
            scaled_angular = actions[:, :3]

        if self._linear_vel_limits is not None:
            scaled_linear = actions[:, 3:] * self._linear_scale + self._linear_bias
        else:
            scaled_linear = actions[:, 3:]

        twist = torch.cat([scaled_angular, scaled_linear], dim=1)
        twist_limited = self._apply_velocity_limits(twist)

        # 存储处理后的旋量（用于奖励项和观察项）
        self._processed_actions[:] = twist_limited

        # 3. 计算加权旋量：tilde_V_b = W_x @ V_b
        # W_x 形状: (6, 6), twist_limited 形状: (num_envs, 6)
        # 扩展 W_x 以支持批量操作
        W_x_batch = self._W_x.unsqueeze(0).expand(self.num_envs, -1, -1)  # (num_envs, 6, 6)
        tilde_twist = (W_x_batch @ twist_limited.unsqueeze(-1)).squeeze(-1)  # (num_envs, 6)

        # 4. 获取当前的雅可比矩阵 J_b
        jacobian = self._get_jacobian()  # (num_envs, 6, num_joints)

        # 5. 计算加权雅可比：tilde_J_b = W_x @ J_b @ W_q^{-1}
        # W_q_inv 形状: (num_joints, num_joints)
        W_q_inv_batch = self._W_q_inv.unsqueeze(0).expand(self.num_envs, -1, -1)  # (num_envs, num_joints, num_joints)
        tilde_jacobian = W_x_batch @ jacobian @ W_q_inv_batch  # (num_envs, 6, num_joints)

        # 6. 计算加权雅可比的DLS伪逆
        # tilde_J_b^+ = tilde_J_b^T (tilde_J_b tilde_J_b^T + lambda^2 I)^{-1}
        tilde_jacobian_inv = math_leap.dls_cholesky_inv(tilde_jacobian, self.cfg.damping)  # (num_envs, num_joints, 6)

        # 7. 计算关节空间的速度增量（考虑权重）
        # 7. 计算关节空间的速度增量（考虑权重）
        # dot_theta = W_q^{-1} @ tilde_J_b^+ @ tilde_V_b
        joint_vel_weighted = (tilde_jacobian_inv @ tilde_twist.unsqueeze(-1)).squeeze(-1)  # (num_envs, num_joints)
        joint_vel = (W_q_inv_batch @ joint_vel_weighted.unsqueeze(-1)).squeeze(-1)  # (num_envs, num_joints)

        # 8. 计算目标状态 - 先读取当前关节位置，避免未定义的变量错误
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_pos_target[:] = current_joint_pos + joint_vel * self._dt
        self._joint_vel_target[:] = joint_vel

        # 应用关节限位
        if self.cfg.use_joint_limits:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            self._joint_pos_target[:] = torch.clamp(
                self._joint_pos_target, min=limits[..., 0], max=limits[..., 1]
            )

class se3adlsAction(se3dlsAction):
    r"""se(3) 动作项 ADLS（Adaptive Damped Least Squares）实现类。

    该类继承自 :class:`se3dlsAction`，通过引入自适应阻尼机制，根据当前雅可比矩阵的操作度动态调整阻尼系数，
    在远离奇异位形时保持高精度，在接近奇异位形时提升稳定性。

    **数学形式：**

    标准DLS求解公式：

    .. math::

        \dot{\theta} = J_b^T (J_b J_b^T + \lambda^2 I)^{-1} \mathcal{V}_b

    自适应阻尼根据操作度 :math:`m` 动态调整：

    .. math::

        \lambda(m) = \lambda_{max} \left( 1 - \frac{m}{m_0} \right)^2

    其中：
    - :math:`m = \sqrt{\det(J_b J_b^T)}` 为 Yoshikawa 操作度
    - :math:`m_0` 为预设的最大操作度阈值
    - :math:`\lambda_{max}` 为最大阻尼系数

    **物理意义：**

    1. **自适应性**：当操作度 :math:`m \approx m_0` 时，系统远离奇异位形，:math:`\lambda \approx 0`，解接近伪逆，精度高
    2. **奇异回避**：当操作度 :math:`m \to 0` 时，系统接近奇异位形，:math:`\lambda \to \lambda_{max}`，增强稳定性
    3. **平滑过渡**：二次方衰减保证阻尼变化的平滑性，避免控制抖动

    Reference:
        - Nakamura & Hanafusa (1986). "Inverse Kinematic Solutions With Singularity Robustness for Robot Manipulator Control"
        - Modern Robotics, Section 6.2: Numerical Inverse Kinematics
    """

    cfg: actions_cfg.se3adlsActionsCfg
    """ADLS 动作项配置。"""

    def __init__(self, cfg: actions_cfg.se3adlsActionsCfg, env: ManagerBasedEnv) -> None:
        """初始化 ADLS 动作项。

        Args:
            cfg: ADLS 动作项配置。
            env: 管理器基础强化学习环境实例。
        """
        # 调用父类初始化（会设置 self.cfg.damping 作为 lambda_max）
        super().__init__(cfg, env)

        self._singular_threshold = self.cfg.singular_threshold
        self._lambda_max = self.cfg.damping  # 继承自 se3dlsActionsCfg

        logger.info(
            f"ADLS 自适应阻尼参数初始化:\n"
            f"  最小奇异值阈值 epsilon: {self._singular_threshold}\n"
            f"  最大阻尼系数 λ_max: {self._lambda_max}"
        )

    def _compute_jacobian_inverse(self, jacobian: torch.Tensor) -> torch.Tensor:
        r"""计算雅可比矩阵的自适应DLS伪逆 (选择性阻尼版本)。

        该方法重写父类的静态阻尼计算，引入基于奇异值的选择性阻尼机制。
        只对接近奇异的方向施加阻尼，健康方向保持高响应。

        **算法流程：**

        1. 调用选择性阻尼版本的 DLS 伪逆
        2. 内部对每个奇异值独立判断: 若 :math:`\sigma_i < \epsilon` 则施加阻尼

        **数学原理：**

        .. math::

            \lambda_i = \begin{cases}
            0 & \text{if } \sigma_i \ge \epsilon \\
            \lambda_{max} \left(1 - \frac{\sigma_i}{\epsilon}\right)^2 & \text{if } \sigma_i < \epsilon
            \end{cases}

        Args:
            jacobian: 雅可比矩阵。形状为 (num_envs, 6, num_joints)。

        Returns:
            自适应DLS伪逆矩阵。形状为 (num_envs, num_joints, 6)。
        """
        # 调用选择性阻尼版本的 DLS 伪逆
        jacobian_inv = math_leap.dls_svd_inv(
            J=jacobian,
            damping=self._lambda_max,
            singular_threshold=self._singular_threshold,
            selective=True,
        )

        return jacobian_inv

class se3awdlsAction(se3wdlsAction):
    r"""se(3) 动作项 AWDLS（Adaptive Weighted Damped Least Squares）实现类。

    该类继承自 :class:`se3wdlsAction`，结合了加权DLS的量纲归一化能力和自适应DLS的奇异回避能力，
    是最完整的DLS变体实现。

    **数学形式：**

    加权DLS优化问题：

    .. math::

        \min_{\dot{\theta}} \| W_x (J_b \dot{\theta} - \mathcal{V}_b) \|^2 + \lambda^2 \| W_q \dot{\theta} \|^2

    解为：

    .. math::

        \dot{\theta} = W_q^{-1} \tilde{J}_b^T (\tilde{J}_b \tilde{J}_b^T + \lambda(m)^2 I)^{-1} W_x \mathcal{V}_b

    其中：
    - 加权雅可比：:math:`\tilde{J}_b = W_x J_b W_q^{-1}`
    - 自适应阻尼：:math:`\lambda(m) = \lambda_{max} (1 - m / m_0)^2`
    - 加权操作度：:math:`m = \sqrt{\det(\tilde{J}_b \tilde{J}_b^T)}`

    **物理意义：**

    1. **量纲统一**：:math:`W_x` 统一角速度和线速度的物理量纲，使优化目标更合理
    2. **关节限位回避**：:math:`W_q` 惩罚接近关节限位的运动，提升安全性
    3. **自适应奇异回避**：:math:`\lambda(m)` 在奇异位形附近自动增大，保证求解稳定性
    4. **最优性能**：结合三种机制，在保证精度的同时最大化鲁棒性

    Reference:
        - Nakamura & Hanafusa (1986). "Inverse Kinematic Solutions With Singularity Robustness"
        - Chan & Lawrence (1988). "General Inverse Kinematics with the Error Damped Pseudoinverse"
        - Modern Robotics, Section 6.2
    """

    cfg: actions_cfg.se3awdlsActionsCfg
    """AWDLS 动作项配置。"""

    def __init__(self, cfg: actions_cfg.se3awdlsActionsCfg, env: ManagerBasedEnv) -> None:
        """初始化 AWDLS 动作项。

        Args:
            cfg: AWDLS 动作项配置。
            env: 管理器基础强化学习环境实例。
        """
        # 调用父类初始化（会设置权重矩阵 W_x 和 W_q）
        super().__init__(cfg, env)

        self._singular_threshold = self.cfg.singular_threshold
        self._lambda_max = self.cfg.damping  # 继承自 se3dlsActionsCfg

        logger.info(
            f"AWDLS 自适应加权阻尼参数初始化:\n"
            f"  任务空间权重 W_x: {torch.diag(self._W_x).tolist()}\n"
            f"  关节空间权重 W_q: {torch.diag(self._W_q).tolist()}\n"
            f"  最小奇异值阈值 epsilon: {self._singular_threshold}\n"
            f"  最大阻尼系数 λ_max: {self._lambda_max}"
        )

    def process_actions(self, actions: torch.Tensor):
        # FIXME:实现逻辑没有错，但注意复用_compute_jacobian_inverse，重构是可考虑
        r"""处理输入的旋量动作，使用自适应加权DLS映射到关节空间。

        该方法结合了 se3wdlsAction 的加权机制和 se3adlsAction 的自适应阻尼，
        实现最完整的DLS变体。

        **关键区别：**
        
        - 奇异值计算使用加权雅可比 :math:`\tilde{J}_b` 而非原始雅可比 :math:`J_b`
        - 自适应阻尼基于加权奇异值，更真实反映任务空间的可控性

        **算法流程：**

        1. 应用缩放和限制
        2. 计算加权旋量：:math:`\tilde{\mathcal{V}}_b = W_x \mathcal{V}_b`
        3. 获取雅可比矩阵 :math:`J_b`
        4. 计算加权雅可比：:math:`\tilde{J}_b = W_x J_b W_q^{-1}`
        5. 计算最小奇异值：:math:`\sigma_{min}`
        6. 计算自适应阻尼：:math:`\lambda = \lambda_{max} (1 - \sigma_{min} / \epsilon)^2`
        7. 计算加权DLS伪逆（使用自适应阻尼）
        8. 计算关节速度并积分为位置目标

        Args:
            actions: 输入的旋量动作。形状为 (num_envs, 6)。
        """
        # 1-2. 存储原始动作、应用缩放和限制（与基类相同）
        self._raw_actions[:] = actions

        # 应用缩放映射
        if self._angular_vel_limits is not None:
            scaled_angular = actions[:, :3] * self._angular_scale + self._angular_bias
        else:
            scaled_angular = actions[:, :3]

        if self._linear_vel_limits is not None:
            scaled_linear = actions[:, 3:] * self._linear_scale + self._linear_bias
        else:
            scaled_linear = actions[:, 3:]

        twist = torch.cat([scaled_angular, scaled_linear], dim=1)
        twist_limited = self._apply_velocity_limits(twist)

        # 存储处理后的旋量（用于奖励项和观察项）
        self._processed_actions[:] = twist_limited

        # 3. 计算加权旋量：tilde_V_b = W_x @ V_b
        W_x_batch = self._W_x.unsqueeze(0).expand(self.num_envs, -1, -1)  # (num_envs, 6, 6)
        tilde_twist = (W_x_batch @ twist_limited.unsqueeze(-1)).squeeze(-1)  # (num_envs, 6)

        # 4. 获取当前的雅可比矩阵 J_b
        jacobian = self._get_jacobian()  # (num_envs, 6, num_joints)

        # 5. 计算加权雅可比：tilde_J_b = W_x @ J_b @ W_q^{-1}
        W_q_inv_batch = self._W_q_inv.unsqueeze(0).expand(self.num_envs, -1, -1)  # (num_envs, num_joints, num_joints)
        tilde_jacobian = W_x_batch @ jacobian @ W_q_inv_batch  # (num_envs, 6, num_joints)

        # 6. 执行 SVD 分解 (一次性完成,避免在 dls_svd_inv 中重复)
        U, S, Vh = torch.linalg.svd(tilde_jacobian, full_matrices=False)

        # 7. 计算加权雅可比的自适应DLS伪逆 (选择性阻尼版本)
        # 传入预计算的 SVD 结果,避免重复分解
        tilde_jacobian_inv = math_leap.dls_svd_inv(
            U=U,
            S=S,
            Vh=Vh,
            damping=self._lambda_max,
            singular_threshold=self._singular_threshold,
            selective=True,
        )  # (num_envs, num_joints, 6)

        # 8. 计算关节空间的速度增量（考虑权重）
        # dot_theta = W_q^{-1} @ tilde_J_dls^+ @ tilde_V_b
        joint_vel_weighted = (tilde_jacobian_inv @ tilde_twist.unsqueeze(-1)).squeeze(-1)  # (num_envs, num_joints)
        joint_vel = (W_q_inv_batch @ joint_vel_weighted.unsqueeze(-1)).squeeze(-1)  # (num_envs, num_joints)

        # 9. 计算目标状态
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_pos_target[:] = current_joint_pos + joint_vel * self._dt
        self._joint_vel_target[:] = joint_vel

        # 应用关节限位
        if self.cfg.use_joint_limits:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            self._joint_pos_target[:] = torch.clamp(
                self._joint_pos_target, min=limits[..., 0], max=limits[..., 1]
            )