"""特权观测（Privileged Observations）

提供仅仿真可用的特权信息，如力矩等
提供在训练中常被随机化的物理与执行器参数（如关节刚度/阻尼、关节摩擦/电枢、刚体质量）的紧凑型统计作为 Critic 的输入。

NOTE:
        - 数值表示：除“材质摩擦/恢复”外，其余均以“当前值/默认值”的缩放比作为输入；缩放比更能直接暴露域随机化的尺度因子。
        - 统计聚合：对逐关节参数做 [mean, std] 聚合，得到稳定的小维度输入，避免与关节数量强绑定。
        - 安全回退：若底层API缺失，返回0以保持训练健壮性。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

###
# -特权信息
###


def goal_quat_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "goal_pose",
    make_quat_unique: bool = True,
) -> torch.Tensor:
    """物体当前姿态与目标姿态的四元数差（方向误差）。

    Args:
        env: ManagerBasedRLEnv - 环境实例
        asset_cfg: SceneEntityCfg - 物体资产配置
        command_name: str - 命令项名称（用于获取目标姿态）
        make_quat_unique: bool - 是否对四元数进行归一化处理

    Returns:
        (num_envs, 4) 张量，四元数形式的姿态差（从当前姿态到目标姿态的旋转）

    NOTE:
        - 计算公式：quat_diff = quat_target ⊗ quat_current^(-1)
        - 该四元数表示"从当前姿态旋转到目标姿态所需的旋转"
        - 如果 make_quat_unique=True，确保 w 分量非负（去除符号歧义）
    """
    import isaaclab.utils.math as math_utils

    # 获取物体当前姿态
    obj: RigidObject = env.scene[asset_cfg.name]
    current_quat = obj.data.root_quat_w  # (num_envs, 4) in (w, x, y, z)

    # 获取目标姿态（从命令管理器）
    # goal_pose 通常是 (pos, quat)，我们取后4维作为目标四元数
    goal_pose = env.command_manager.get_command(command_name)
    target_quat = goal_pose[:, -4:]  # (num_envs, 4) in (w, x, y, z)

    # 计算四元数差：quat_diff = target ⊗ current^(-1)
    current_quat_inv = math_utils.quat_inv(current_quat)
    quat_diff = math_utils.quat_mul(target_quat, current_quat_inv)

    # 归一化处理（确保 w 分量非负）
    if make_quat_unique:
        quat_diff = math_utils.quat_unique(quat_diff)

    return quat_diff


def object_mass_scale(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    scale_range: tuple[float, float] = (0.25, 1.2),
) -> torch.Tensor:
    """物体质量缩放系数（相对默认质量）。

    该项与 EventCfg 中 `mdp.randomize_rigid_body_mass(operation='scale')` 保持一致：
    返回的量近似等同于被采样的 scale（并映射到 [-1, 1]）。

    Args:
        env: 训练环境
        asset_cfg: 物体资产配置
        scale_range: 质量缩放范围 (min, max)

    Returns:
        (num_envs, 1) 张量，范围约为 [-1, 1]。
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    try:
        masses = obj.root_physx_view.get_masses()
        # expected shapes: (N, 1) for rigid object
        if masses.ndim == 2:
            masses = masses[:, 0]
        default_mass = obj.data.default_mass
        if default_mass.ndim == 2:
            default_mass = default_mass[:, 0]
        # move to env device
        masses = masses.to(device=env.device, dtype=torch.float)
        default_mass = default_mass.to(device=env.device, dtype=torch.float).clamp(min=1e-6)

        ratio = masses / default_mass
        lo, hi = float(scale_range[0]), float(scale_range[1])
        ratio = torch.clamp(ratio, lo, hi)
        # map to [-1, 1]
        ratio = (2.0 * ratio - (hi + lo)) / (hi - lo)
        return ratio.unsqueeze(-1)
    except Exception:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float)


def object_com_offset(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    com_range: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """物体 COM（质心）偏移（归一化）。

    该项用于为 RMA 提供“外参/物体属性”的一部分。PhysX 可能返回 (N, 3) 的 COM 位置或 (N, 7) 的
    [pos(3), quat(4)]，这里仅取位置部分并按给定范围映射到 [-1, 1]。

    Args:
        env: 训练环境
        asset_cfg: 物体资产配置
        com_range: 每轴范围字典，如 {"x":(-0.01,0.01), "y":..., "z":...}

    Returns:
        (num_envs, 3) 张量。
    """
    if com_range is None:
        com_range = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}

    obj: RigidObject = env.scene[asset_cfg.name]
    try:
        coms = obj.root_physx_view.get_coms()
        if coms.ndim != 2 or coms.shape[-1] not in (3, 7):
            return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float)
        pos = coms[:, 0:3] if coms.shape[-1] == 7 else coms
        pos = pos.to(device=env.device, dtype=torch.float)

        lows = torch.tensor(
            [com_range.get("x", (0.0, 0.0))[0], com_range.get("y", (0.0, 0.0))[0], com_range.get("z", (0.0, 0.0))[0]],
            device=env.device,
            dtype=torch.float,
        )
        highs = torch.tensor(
            [com_range.get("x", (0.0, 0.0))[1], com_range.get("y", (0.0, 0.0))[1], com_range.get("z", (0.0, 0.0))[1]],
            device=env.device,
            dtype=torch.float,
        )
        denom = (highs - lows).clamp(min=1e-6)
        pos = torch.clamp(pos, lows, highs)
        pos = (2.0 * pos - highs - lows) / denom
        return pos
    except Exception:
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float)


def object_material_properties(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    static_friction_range: tuple[float, float] = (0.2, 1.0),
    dynamic_friction_range: tuple[float, float] = (0.15, 0.6),
    restitution_range: tuple[float, float] = (0.0, 0.1),
) -> torch.Tensor:
    """物体材质参数（static/dynamic friction, restitution）的归一化观测。

    Returns:
        (num_envs, 3) 张量，按给定范围映射到 [-1, 1]。
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    try:
        mats = obj.root_physx_view.get_material_properties()
        # possible shapes:
        # - (N, 3)
        # - (N, max_shapes, 3)
        if mats.ndim == 3:
            mats = mats.mean(dim=1)
        if mats.ndim != 2 or mats.shape[-1] != 3:
            return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float)

        mats = mats.to(device=env.device, dtype=torch.float)
        lows = torch.tensor(
            [static_friction_range[0], dynamic_friction_range[0], restitution_range[0]],
            device=env.device,
            dtype=torch.float,
        )
        highs = torch.tensor(
            [static_friction_range[1], dynamic_friction_range[1], restitution_range[1]],
            device=env.device,
            dtype=torch.float,
        )
        denom = (highs - lows).clamp(min=1e-6)
        mats = torch.clamp(mats, lows, highs)
        mats = (2.0 * mats - highs - lows) / denom
        return mats
    except Exception:
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float)
