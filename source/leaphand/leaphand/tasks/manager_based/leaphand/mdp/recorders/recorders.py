"""录制器定义：用于行为克隆(BC)的数据采集。

该模块在关节空间策略(joint-space policy)rollout时记录必要数据，用于训练SE(3)空间策略。
数据采集环境为 `inhand_base_env_cfg.py`（关节动作），通过rl_games play.py自动触发录制。

**核心功能**：
    1. FingerTipSe3TwistRecorder: 记录指尖SE(3)旋量和雅可比矩阵
       - actions_se3_twist: 指尖速度旋量 (N, 24)，通过刚体速度计算
       - jacobian: 雅可比矩阵 (N, 4, 6, 4)，从PhysX直接获取
    
    2. BCMetadataRecorder: 记录固定参数（作为HDF5 attributes）
       - 伴随矩阵 Ad_{T_bb'}: 虚拟Xform到刚体系的变换 (4, 6, 6)
       - 阻尼系数 λ, 控制周期 Δt, Affine参数等

**BC训练对齐** (参考 `learning.ipynb`):
    标签对齐：$\\Delta\\theta^E$（关节增量）
    损失函数：
        $$\\min_\\theta \\mathcal{L}_{BC} = \\mathbb{E} \\left[ \\| J_{dls}^+ Ad_{T_{bb'}} affine(\\pi(o_T))\\Delta t - \\Delta\\theta^E \\|^2 \\right]$$
    
    其中：
        - $\\Delta\\theta^E$: 从ActionStateRecorder.actions读取（关节增量）
        - $J_b$: 从本录制器的jacobian字段读取
        - $Ad_{T_{bb'}}$, λ, Δt: 从BCMetadataRecorder的metadata读取
        - affine参数: 同样从metadata读取

**技术细节**：
    1. 旋量计算：
       - 修正COM速度到刚体原点: $v_{origin} = v_{com} + \\omega \\times r_{com \\rightarrow origin}$
       - 世界系→刚体系: $V_b = R_{bw} V_w$
       - (可选) 伴随变换: $V_{b'} = Ad_{T_{b'b}} V_b$
    
    2. 雅可比提取：
       - 直接从PhysX获取: `root_physx_view.get_jacobians()`
       - 动态解析关节索引: 支持配置joint mapping
       - 坐标系变换: 世界系→刚体系
    
    3. 自包含设计：
       - 不依赖ActionManager中的se3Action
       - 可在pure joint-space环境(inhand_base_env_cfg)中使用

**HDF5数据结构**：
    demo_0/
      ├── actions: (T, 16)              # 关节动作（ActionStateRecorder）
      ├── observations: (T, obs_dim)     # 观测数据（ActionStateRecorder）
      ├── states/joint_pos: (T, 16)     # 关节位置（ActionStateRecorder）
      ├── actions_se3_twist: (T, 24)    # 指尖旋量（本录制器）
      ├── jacobian: (T, 4, 6, 4)        # 雅可比矩阵（本录制器）
      └── attrs/bc_params/              # 固定参数（BCMetadataRecorder）
          ├── adjoint_matrices: (4, 6, 6)
          ├── damping: float
          ├── dt: float
          └── ...

参考文档：`AnyRotate/source/leaphand/leaphand/ideas/learning.ipynb`
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils import math as math_utils
from ..utils import math as math_leap


class FingerTipSe3TwistRecorder(RecorderTerm):
    """记录四指指尖SE(3)旋量和雅可比矩阵。

    **记录内容**：
        - actions_se3_twist: 指尖速度旋量，形状 (num_envs, 24)
          按手指顺序依次为: 食指、中指、无名指、拇指
          每根手指6维 ([ω; v]: 角速度3 + 线速度3)
        
        - jacobian: 指尖雅可比矩阵，形状 (num_envs, 4, 6, 4)
          4根手指 × 6维旋量 × 4个关节/手指
          在刚体坐标系 {b} 下表示，配合Ad可变换到 {b'}

    **旋量计算流程** (record_post_step):
        1. 提取世界系速度: `body_ang_vel_w`, `body_vel_w` (COM速度)
        2. 修正COM→原点: $v_{origin} = v_{com} + \\omega \\times r_{com \\rightarrow origin}$
        3. 转换到刚体系: $V_b = R_{bw} V_w$
        4. (可选) 伴随变换: $V_{b'} = Ad_{T_{b'b}} V_b$ (如果配置了finger_xform_names)
        5. 打平为24维向量

    **雅可比计算流程** (_compute_jacobians_from_articulation):
        1. 从PhysX获取: `root_physx_view.get_jacobians()` → (N, num_bodies, 6, num_dofs)
        2. 动态解析关节索引: 通过 `find_joints(joint_names_list)` 获取每根手指的关节ID
        3. 调整速度顺序: PhysX [v; ω] → [ω; v]
        4. 坐标系变换: 世界系 → 刚体系 (旋转矩阵 R_bw)
        5. 提取子矩阵: 每根手指对应的 (6, 4) 雅可比

    **配置参数**：
        - finger_body_names: 实际指尖刚体名称 (用于读取速度)
        - finger_xform_names: (可选) 虚拟指尖Xform名称 (用于伴随变换)
        - fingers: 手指关节映射字典 {finger_name: [joint_names]}
        - preserve_order: 是否保持关节顺序一致性

    **设计要点**：
        - 自包含实现：不依赖ActionManager中的se3Action
        - 动态关节映射：支持不同的命名约定和关节顺序
        - 精确旋量定义：遵循Modern Robotics的旋量定义（参考点为刚体原点）
    
    Note:
        返回的数据将被RecorderManager保存到HDF5的时序datasets中（每个时间步一条记录）。
        固定参数（Ad, λ等）由BCMetadataRecorder负责，存储为episode attributes。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 记录指尖刚体名称，需与 URDF/美元场景中的命名保持一致
        self._finger_body_names = (
            list(cfg.finger_body_names)
            if getattr(cfg, "finger_body_names", None)
            else [
                "fingertip",  # 食指
                "fingertip_2",  # 中指
                "fingertip_3",  # 无名指
                "thumb_fingertip",  # 拇指
            ]
        )

        # 可选：虚拟指尖坐标系 {b'} 名称（与动作项的 target 对齐）
        self._finger_xform_names = (
            list(cfg.finger_xform_names) if getattr(cfg, "finger_xform_names", None) else None
        )
        self._finger_parent_names = (
            list(cfg.finger_parent_names) if getattr(cfg, "finger_parent_names", None) else None
        )
        
        # 手指关节名称映射配置
        self._fingers_joint_mapping = getattr(cfg, "fingers", None) or {
            "index": ["a_1", "a_0", "a_2", "a_3"],
            "middle": ["a_5", "a_4", "a_6", "a_7"],
            "ring": ["a_9", "a_8", "a_10", "a_11"],
            "thumb": ["a_12", "a_13", "a_14", "a_15"]
        }
        self._preserve_order = getattr(cfg, "preserve_order", True)
        
        # 建立刚体名称到手指名称的映射（用于查找关节）
        self._body_to_finger_map = {
            "fingertip": "index",
            "fingertip_2": "middle",
            "fingertip_3": "ring",
            "thumb_fingertip": "thumb"
        }

        # 解析刚体索引，后续直接用索引从张量中切片
        self._asset = self._env.scene["robot"]
        body_ids, body_names = self._asset.find_bodies(self._finger_body_names)
        if len(body_ids) != len(self._finger_body_names):
            raise ValueError(
                f"指尖数量不匹配，期望 {len(self._finger_body_names)} 个，实际 {len(body_ids)}: {body_names}"
            )
        self._body_ids = body_ids
        self._body_names = body_names

        # 预计算虚拟坐标系的伴随矩阵 Ad_{T_b'b}，用于将 V_b → V_{b'}
        self._Ad_bprime_b = None
        if self._finger_xform_names is not None:
            if len(self._finger_xform_names) != len(self._finger_body_names):
                raise ValueError(
                    "finger_xform_names 长度必须与 finger_body_names 一致，确保每个指尖都有对应的虚拟坐标系。"
                )
            if self._finger_parent_names is not None and len(self._finger_parent_names) != len(
                self._finger_body_names
            ):
                raise ValueError(
                    "finger_parent_names 长度必须与 finger_body_names 一致，或保持为 None 使用默认父刚体。"
                )

            self._Ad_bprime_b = []
            for idx, target_xform in enumerate(self._finger_xform_names):
                parent_name = (
                    self._finger_parent_names[idx]
                    if self._finger_parent_names is not None
                    else self._body_names[idx]
                )
                T_bb_prime = self._compute_xform_offset_transform(parent_name, target_xform)
                Ad_bprime_b = math_leap.adjoint_transform(math_leap.inverse_transform(T_bb_prime))
                self._Ad_bprime_b.append(Ad_bprime_b.to(self._env.device))

    def record_post_step(self):
        # ============ 第1步：提取世界系下的速度和位姿 ============
        ang_w = self._asset.data.body_ang_vel_w[:, self._body_ids]  # (N, 4, 3) 角速度
        lin_com_w = self._asset.data.body_vel_w[:, self._body_ids]  # (N, 4, 3) 质心线速度
        quat_w = self._asset.data.body_quat_w[:, self._body_ids]    # (N, 4, 4) xyzw格式

        # ============ 第2步：修正线速度从质心到刚体坐标系原点 ============
        # IsaacLab的body_vel_w返回的是刚体质心的线速度，但SE(3)旋量定义中
        # 线速度分量v是参考点(刚体坐标系原点)的速度，两者关系为：
        #   v_origin = v_com + ω × r_{com→origin}
        # 其中 r_{com→origin} = pos_origin - pos_com (世界系表示)
        body_pos_w = self._asset.data.body_pos_w[:, self._body_ids]      # 刚体坐标系原点位置
        body_com_pos_w = self._asset.data.body_com_pos_w[:, self._body_ids]  # 质心位置
        r_com_to_origin_w = body_pos_w - body_com_pos_w  # (N, 4, 3)

        # 计算修正后的线速度: v_origin = v_com + ω × r
        lin_w = lin_com_w + torch.cross(ang_w, r_com_to_origin_w, dim=-1)  # (N, 4, 3)

        # ============ 第3步：将速度从世界系转换到刚体系 ============
        quat_wxyz = math_utils.convert_quat(quat_w, to="wxyz")
        R_bw = math_utils.matrix_from_quat(math_utils.quat_inv(quat_wxyz))  # (N, 4, 3, 3)

        # 旋转变换: v_b = R_bw @ v_w
        ang_b = torch.einsum("...ij,...j->...i", R_bw, ang_w)
        lin_b = torch.einsum("...ij,...j->...i", R_bw, lin_w)

        # ============ 第4步：拼接旋量(角速度在前，线速度在后) ============
        twists = torch.cat([ang_b, lin_b], dim=-1)  # (N, 4, 6)

        # ============ 第5步：若使用虚拟Xform，应用伴随变换 {b} → {b'} ============
        # 伴随变换公式: V_{b'} = Ad_{T_{b'b}} @ V_b
        # 其中 Ad_{T_{b'b}} = Ad_{T_{bb'}^{-1}}
        if self._Ad_bprime_b is not None:
            twists_prime = []
            for idx, Ad in enumerate(self._Ad_bprime_b):
                twist_b = twists[:, idx]  # (N, 6)
                twist_bprime = torch.einsum("ij,bj->bi", Ad, twist_b)
                twists_prime.append(twist_bprime)
            twists = torch.stack(twists_prime, dim=1)  # (N, 4, 6)

        # ============ 第6步：打平为24维向量 ============
        twists_flat = twists.reshape(twists.shape[0], -1)  # (N, 24)

        # ============ 第7步：记录雅可比矩阵(BC训练时需要) ============
        # 从机器人资产直接计算雅可比矩阵（自包含实现，不依赖ActionManager）
        # 注意：这里记录的是 J_b (刚体系雅可比)，训练时需要配合Ad进行变换
        jacobians = self._compute_jacobians_from_articulation()  # (N, 4, 6, num_joints_per_finger)

        # 返回两个记录项：旋量和雅可比
        return {
            "actions_se3_twist": twists_flat,      # (N, 24) SE(3)旋量标签
            "jacobian": jacobians,                  # (N, 4, 6, num_joints_per_finger) 雅可比矩阵
        }

    def _compute_xform_offset_transform(self, parent_body_name: str, xform_name: str) -> torch.Tensor:
        """计算父刚体 {b} 到虚拟指尖 {b'} 的固定变换 T_bb'。

        该偏移与环境数量无关，只需在初始化时计算一次。
        """
        from isaaclab.sim.utils import get_current_stage
        from pxr import UsdGeom
        import isaaclab.sim as sim_utils

        stage = get_current_stage()

        # 取第一个环境的 prim 路径作为搜索根
        env_prim_path = sim_utils.find_first_matching_prim(self._asset.cfg.prim_path).GetPath().pathString

        parent_prim = self._find_prim_by_name(stage, env_prim_path, parent_body_name)
        xform_prim = self._find_prim_by_name(stage, env_prim_path, xform_name)

        if parent_prim is None or not parent_prim.IsValid():
            raise RuntimeError(f"无法在 '{env_prim_path}' 下找到父刚体 '{parent_body_name}'。")
        if xform_prim is None or not xform_prim.IsValid():
            raise RuntimeError(f"无法在 '{env_prim_path}' 下找到虚拟指尖 Xform '{xform_name}'。")

        xform_cache = UsdGeom.XformCache()
        parent_transform_w = xform_cache.GetLocalToWorldTransform(parent_prim)
        xform_transform_w = xform_cache.GetLocalToWorldTransform(xform_prim)

        parent_transform_inv = parent_transform_w.GetInverse()
        relative_transform = parent_transform_inv * xform_transform_w

        transform_np = np.array(relative_transform, dtype=np.float32).T
        return torch.tensor(transform_np, device=self._env.device, dtype=torch.float32)

    def _find_prim_by_name(self, stage, root_path: str, target_name: str):
        """在 USD stage 中从 root_path 子树里搜索名称匹配的 prim。"""
        from pxr import Usd

        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            return None

        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == target_name:
                return prim
        return None

    def _compute_jacobians_from_articulation(self) -> torch.Tensor:
        """直接从机器人资产计算四根手指的雅可比矩阵（自包含实现）。

        该方法不依赖 ActionManager 中的 se3Action，而是直接从 PhysX API 获取雅可比，
        确保在 joint-space policy 环境(inhand_base_env_cfg)中也能正常工作。

        算法流程：
            1. 从 PhysX 获取所有刚体的雅可比矩阵 (世界坐标系表示)
            2. 根据配置的关节名称解析每根手指的关节索引
            3. 调整速度分量顺序 (PhysX: [v; ω] → 本项目: [ω; v])
            4. (可选) 通过旋转矩阵将雅可比从世界系 {w} 转到刚体系 {b}
            5. 提取每根手指对应的子矩阵

        Returns:
            雅可比矩阵张量，形状为 (num_envs, 4, 6, num_joints_per_finger)。
            - 4: 四根手指 (index, middle, ring, thumb)
            - 6: 旋量维度 ([ω; v])
            - num_joints_per_finger: 每根手指的关节数 (LeapHand中为4)

        Raises:
            ValueError: 当固定基座场景下雅可比索引计算错误时。

        Note:
            - 返回的雅可比在刚体坐标系 {b} 下表示（与 se3Action 的 use_body_frame=True 一致）
            - 如果虚拟Xform存在，这里返回的仍是父刚体 {b} 的雅可比
            - 训练BC时需配合伴随矩阵 Ad_{T_bb'} 进行变换
        """
        # 1. 从 PhysX 获取所有刚体的雅可比矩阵
        # shape: (num_envs, num_bodies, 6, num_dofs)
        all_jacobians = self._asset.root_physx_view.get_jacobians()

        jacobians_list = []
        for body_name in self._body_names:
            # 获取刚体索引
            body_ids, _ = self._asset.find_bodies(body_name)
            if len(body_ids) != 1:
                raise ValueError(f"刚体名称 '{body_name}' 匹配到 {len(body_ids)} 个刚体，期望恰好 1 个。")
            body_idx = body_ids[0]

            # 2. 根据刚体名称查找对应手指并获取关节名称列表
            finger_name = self._body_to_finger_map.get(body_name)
            if finger_name is None:
                raise ValueError(f"未知的指尖刚体名称: {body_name}")
            
            joint_names_list = self._fingers_joint_mapping.get(finger_name)
            if joint_names_list is None:
                raise ValueError(f"未配置手指 '{finger_name}' 的关节名称映射")
            
            # 通过关节名称获取关节索引
            joint_ids, _ = self._asset.find_joints(joint_names_list, preserve_order=self._preserve_order)
            
            # 3. 调整 PhysX 雅可比索引（考虑固定基座情况）
            if self._asset.is_fixed_base:
                # 固定基座：PhysX 雅可比不包含基座刚体，索引需要减 1
                jacobi_body_idx = body_idx - 1
                if jacobi_body_idx < 0:
                    raise ValueError(
                        f"固定基座场景下，刚体索引为 0，无法在 PhysX Jacobian 中找到对应条目。"
                    )
                jacobi_joint_ids = joint_ids  # 关节索引保持不变
            else:
                # 浮动基座：PhysX 雅可比包含 6 个虚拟自由度（基座位姿）
                jacobi_body_idx = body_idx
                jacobi_joint_ids = [idx + 6 for idx in joint_ids]  # 关节索引偏移 6

            # 4. 提取目标刚体和关节对应的雅可比子矩阵
            jacobian_physx = all_jacobians[:, jacobi_body_idx, :, :][:, :, jacobi_joint_ids]  # (N, 6, num_joints)

            # 5. 调整速度分量顺序：PhysX [v; ω] → 本项目 [ω; v]
            jacobian_world = torch.cat([jacobian_physx[:, 3:, :], jacobian_physx[:, :3, :]], dim=1)  # (N, 6, num_joints)

            # 6. (可选) 从世界坐标系 {w} 转换到刚体坐标系 {b}
            # 这里默认转换到刚体系（与 se3Action 的 use_body_frame=True 一致）
            body_quat_w = self._asset.data.body_quat_w[:, body_idx]  # (N, 4) in xyzw format
            body_quat_w_wxyz = math_utils.convert_quat(body_quat_w, to="wxyz")  # 转换为 (w, x, y, z) 格式
            R_bw = math_utils.matrix_from_quat(math_utils.quat_inv(body_quat_w_wxyz))  # (N, 3, 3)

            # 应用旋转变换到雅可比的角速度和线速度部分
            jacobian_body = jacobian_world.clone()
            jacobian_body[:, :3, :] = torch.bmm(R_bw, jacobian_world[:, :3, :])  # 角速度部分
            jacobian_body[:, 3:, :] = torch.bmm(R_bw, jacobian_world[:, 3:, :])  # 线速度部分

            jacobians_list.append(jacobian_body)

        # 7. 拼接所有手指的雅可比矩阵
        jacobians = torch.stack(jacobians_list, dim=1)  # (N, 4, 6, num_joints_per_finger)
        return jacobians

    def _find_prim_by_name(self, stage, root_path: str, target_name: str):
        """在 USD stage 中从 root_path 子树里搜索名称匹配的 prim。"""
        from pxr import Usd

        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            return None

        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == target_name:
                return prim
        return None


class BCMetadataRecorder(RecorderTerm):
    """记录BC训练所需的固定参数metadata。
    
    该Recorder在每个episode reset后记录一次固定参数，这些参数在整个episode中保持不变：
        - 伴随矩阵 Ad_{T_bb'}: 虚拟指尖坐标系到刚体系的变换 (4, 6, 6)
        - 阻尼系数 λ: DLS逆运动学的阻尼参数
        - 控制周期 Δt: 仿真步长 × decimation
        - Affine参数: 关节和SE(3)动作的缩放/偏置参数
    
    **使用场景**：
        1. 在 joint-space 环境(inhand_base_env_cfg)中：
           - 仅记录 dt 和 joint_action_scale
           - SE(3)相关参数(Ad, se3_affine)从se3环境配置中获取（训练时硬编码或动态加载）
        
        2. 在 se3-space 环境(inhand_se3_env_cfg)中：
           - 记录完整参数（Ad, λ, dt, joint/se3 affine）
           - 用于验证或调试
    
    **HDF5存储结构**：
        demo_0/attrs/bc_params/
          ├── dt: float                            # 控制周期
          ├── joint_action_scale: float            # 关节动作缩放系数
          ├── adjoint_matrices: np.ndarray (4, 6, 6)  # 伴随矩阵（可选）
          ├── se3_angular_scale: np.ndarray (4,)   # SE(3)角速度缩放（可选）
          ├── se3_angular_bias: np.ndarray (4,)    # SE(3)角速度偏置（可选）
          ├── se3_linear_scale: np.ndarray (4,)    # SE(3)线速度缩放（可选）
          └── se3_linear_bias: np.ndarray (4,)     # SE(3)线速度偏置（可选）

    Note:
        - **阻尼系数 λ（DLS damping）不记录在metadata中**。它是数值稳定性的训练超参数，建议在BC训练时由用户在训练配置中指定（例如：trainer_config['damping'] = 0.01）。
    
    **BC训练数据流**：
        1. 关节策略rollout → 记录 Δθ_label (actions)
        2. 反推等效SE(3)动作（训练时计算）：
           ```python
           V_b = J_b @ (Δθ / Δt)                      # 通过雅可比映射
           V_b_prime = Ad^{-1} @ V_b                  # 伴随变换（如果使用虚拟Xform）
           a_se3 = InverseAffine(V_b_prime)           # 逆仿射到规范化动作
           ```
        3. 用于监督SE(3)策略训练
    
    Note:
        - record_post_reset返回的dict会被RecorderManager转换为episode attributes
        - 训练时通过 `episode.metadata['bc_params']` 访问这些参数
        - 在joint-space环境中，SE(3)参数为None，训练时需要从se3_env_cfg动态加载
    """
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset = self._env.scene["robot"]
        
        # 缓存固定参数（在初始化时提取一次）
        self._metadata_cache = None
    
    def record_post_reset(self, env_ids: list | None):
        """在episode reset后记录固定参数metadata。
        
        Args:
            env_ids: 重置的环境ID列表。由于metadata是固定的，这里忽略env_ids
        
        Returns:
            tuple: (key, value) 其中key为存储路径，value为metadata字典
        
        Note:
            在joint-space环境中，SE(3)相关参数(Ad, λ, se3_affine)不可用，
            此时仅记录基础参数(dt, joint_action_scale)。
            BC训练时需要从se3环境配置中动态加载这些参数。
        """
        # 如果已经缓存，直接返回（避免重复计算）
        if self._metadata_cache is not None:
            return "bc_params", self._metadata_cache
        
        action_manager = self._env.action_manager
        
        # 计算控制周期 Δt = dt * decimation
        dt = self._env.sim.cfg.dt * self._env.cfg.decimation
        
        # 提取关节动作affine参数
        joint_action_scale = self._extract_joint_action_scale(action_manager)
        
        # 构建基础metadata（总是可用）
        metadata = {
            "dt": float(dt),
            "joint_action_scale": float(joint_action_scale),
        }
        
        # 尝试提取SE(3)动作的固定参数（仅在se3环境中可用）
        try:
            se3_actions = self._get_se3_actions_ordered(action_manager)
            
            # 提取伴随矩阵 (4, 6, 6)
            adjoint_matrices = self._extract_adjoint_matrices(se3_actions)
            
            # 提取SE(3) affine参数
            se3_params = self._extract_se3_affine_params(se3_actions)
            
            # 添加SE(3)相关参数（不包含阻尼 lambda）
            # 注：阻尼系数 lambda 应由训练时超参数提供，故不写入metadata
            metadata.update({
                "adjoint_matrices": adjoint_matrices.cpu().numpy(),  # (4, 6, 6)
                **se3_params,  # se3_angular_scale, se3_angular_bias等
            })
        except (RuntimeError, KeyError, TypeError) as e:
            # 在joint-space环境中，se3Action不存在，这是预期行为
            # 仅记录一条警告，不中断执行
            import isaaclab.utils as utils
            utils.carb_log_warning(
                f"BCMetadataRecorder: SE(3)参数不可用（可能在joint-space环境中）。"
                f"仅记录基础参数。错误信息: {e}"
            )
        
        # 缓存以避免重复计算
        self._metadata_cache = metadata
        
        return "bc_params", metadata
    
    def _get_se3_actions_ordered(self, action_manager) -> list:
        """按照固定顺序获取4根手指的se3Action。
        
        顺序：index_se3, middle_se3, ring_se3, thumb_se3
        """
        from ..actions import se3_actions as se3
        
        action_names = ["index_se3", "middle_se3", "ring_se3", "thumb_se3"]
        se3_actions = []
        
        for name in action_names:
            if name not in action_manager.active_terms:
                raise RuntimeError(
                    f"SE(3)动作项 '{name}' 未在动作管理器中找到。"
                    "BC数据采集需要完整的4根手指se3Action配置。"
                )
            action_term = action_manager.get_term(name)
            if not isinstance(action_term, se3.se3Action):
                raise TypeError(f"动作项 '{name}' 不是se3Action类型。")
            se3_actions.append(action_term)
        
        return se3_actions
    
    def _extract_adjoint_matrices(self, se3_actions: list) -> torch.Tensor:
        """提取4根手指的伴随矩阵。
        
        Returns:
            torch.Tensor: 形状为 (4, 6, 6) 的伴随矩阵
        """
        adjoint_matrices = []
        for action in se3_actions:
            if action.is_xform and action.Ad_bprime_b is not None:
                # 如果使用虚拟Xform，记录 Ad_{T_{bb'}}（注意这是逆变换）
                # action.Ad_bprime_b 实际上是 Ad_{T_b'b}，我们需要的是 Ad_{T_bb'}
                # 根据伴随性质：Ad_{T^{-1}} = Ad_{T}^{-1}
                Ad_bprime_b = action.Ad_bprime_b
                # 计算逆伴随：Ad_{T_bb'} = Ad_{T_b'b}^{-1}
                Ad_b_bprime = torch.linalg.inv(Ad_bprime_b)
                adjoint_matrices.append(Ad_b_bprime)
            else:
                # 如果不使用虚拟Xform，伴随矩阵为单位矩阵
                adjoint_matrices.append(torch.eye(6, device=self._env.device))
        
        return torch.stack(adjoint_matrices)  # (4, 6, 6)
    
    def _extract_joint_action_scale(self, action_manager) -> float:
        """提取关节动作的缩放系数。
        
        假设使用 RelativeJointPositionAction，其scale参数即为缩放系数。
        """
        # 查找关节动作项（通常命名为 "hand_joint_pos" 或类似）
        for name in action_manager.active_terms:
            action_term = action_manager.get_term(name)
            # 检查是否为关节位置动作
            if hasattr(action_term, 'cfg') and hasattr(action_term.cfg, 'scale'):
                # RelativeJointPositionActionCfg 有 scale 属性
                return action_term.cfg.scale
        
        # 如果未找到，返回默认值 1/24 (根据inhand_base_env_cfg.py)
        return 1.0 / 24.0
    
    def _extract_se3_affine_params(self, se3_actions: list) -> dict:
        """提取SE(3)动作的affine参数（缩放和偏置）。
        
        Returns:
            dict: 包含4根手指的angular和linear的scale/bias参数
                {
                    "se3_angular_scale": (4,),
                    "se3_angular_bias": (4,),
                    "se3_linear_scale": (4,),
                    "se3_linear_bias": (4,),
                }
        """
        angular_scales = []
        angular_biases = []
        linear_scales = []
        linear_biases = []
        
        for action in se3_actions:
            # 从action对象提取内部缓存的affine参数
            angular_scales.append(getattr(action, '_angular_scale', 1.0))
            angular_biases.append(getattr(action, '_angular_bias', 0.0))
            linear_scales.append(getattr(action, '_linear_scale', 1.0))
            linear_biases.append(getattr(action, '_linear_bias', 0.0))
        
        return {
            "se3_angular_scale": np.array(angular_scales, dtype=np.float32),
            "se3_angular_bias": np.array(angular_biases, dtype=np.float32),
            "se3_linear_scale": np.array(linear_scales, dtype=np.float32),
            "se3_linear_bias": np.array(linear_biases, dtype=np.float32),
        }
