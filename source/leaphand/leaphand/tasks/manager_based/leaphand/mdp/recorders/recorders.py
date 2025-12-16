"""录制器定义：将关节策略执行后的指尖速度转为 SE(3) 旋量标签。

该模块为行为克隆(BC)准备标签：在每个仿真步结束时读取四个指尖的刚体速度，
将世界系速度转换到刚体系后拼接为 24 维 SE(3) 旋量(actions_se3_twist)。

TODO：目的是记录具有良好表现 joint-space policy (`inhand_base_env_cfg.py`训练) rollout 时的数据，用于给 se3 policy 进行模仿学习

模仿学习的预想数据对齐（不保真，个人预想）：

主要是旋量这个物理量的定义和对齐问题，关节策略的输出是关节位置，无法直接对应旋量动作指令，故而有一个转换过程：

E[||J_b dtheta - Ad_T_bb' V_b'] = E[||V_b - Ad_T_bb' pi(obs)]（等等，这个对齐好像搞错了）

- dtheta: 关节角速度，IsaacLab 可通过API直接获得。来自关节策略
- J_b: {b}系下指尖雅可比矩阵（参考点和参考系都位于{b})，可通过IsaacLab API获得几何雅可比 J_g (参考点位于{b}，参考末端位于{b'})，然后经伴随变换 J_b = [R 0;0 R] J_g。来自关节策略
- J_b dtheta: 计算得到的真实速度

写到这里的时候我发现主要问题了，观察对齐和动作对齐错位了

"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils import math as math_utils
from ..utils import math as math_leap


class FingerTipSe3TwistRecorder(RecorderTerm):
    # REVIEW：这里感觉有问题，因为se3动作输出的旋量是{b'}的，但这里似乎把刚体系{b}的速度记录下来了？
    # 按照设想，BC需要构建的数据集是{st,at,rt,st+1}，但这里好像只有record_post_step()，这是否合理？
    # 除了观察Observation，应该还有Command吧？
    """记录四指指尖在刚体系下的旋量(ω_b, v_b)。

    - 旋量按手指顺序依次为: 食指、中指、无名指、拇指。
    - 每根手指 6 维 (角速度3 + 线速度3)，最终输出形状 (num_envs, 24)。
    - 返回键为 ``actions_se3_twist``，方便与默认动作记录区分。
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
        # 提取世界系下的角速度/线速度和姿态
        ang_w = self._asset.data.body_ang_vel_w[:, self._body_ids]  # (N, 4, 3)
        lin_w = self._asset.data.body_vel_w[:, self._body_ids]      # (N, 4, 3)
        quat_w = self._asset.data.body_quat_w[:, self._body_ids]    # (N, 4, 4) xyzw

        # 将姿态转为旋转矩阵 R_bw，用于世界系 -> 刚体系 速度变换
        quat_wxyz = math_utils.convert_quat(quat_w, to="wxyz")
        R_bw = math_utils.matrix_from_quat(math_utils.quat_inv(quat_wxyz))  # (N, 4, 3, 3)

        # 速度从世界系转换到刚体系: v_b = R_bw @ v_w
        ang_b = torch.einsum("...ij,...j->...i", R_bw, ang_w)
        lin_b = torch.einsum("...ij,...j->...i", R_bw, lin_w)

        # 拼接旋量并按手指顺序打平
        twists = torch.cat([ang_b, lin_b], dim=-1)  # (N, 4, 6)

        # 若配置了虚拟指尖坐标系 {b'}，将旋量从真实刚体帧 {b} 变换到虚拟帧 {b'}
        if self._Ad_bprime_b is not None:
            twists_prime = []
            for idx, Ad in enumerate(self._Ad_bprime_b):
                twist_b = twists[:, idx]  # (N, 6)
                twist_bprime = torch.einsum("ij,bj->bi", Ad, twist_b)
                twists_prime.append(twist_bprime)
            twists = torch.stack(twists_prime, dim=1)
        twists_flat = twists.reshape(twists.shape[0], -1)

        return "actions_se3_twist", twists_flat

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
