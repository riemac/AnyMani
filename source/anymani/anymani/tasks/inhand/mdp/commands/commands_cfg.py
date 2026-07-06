# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from pickle import NONE

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .rotation_command import ContinuousRotationCommand, OfficialContinuousRotationCommand


@configclass
class ContinuousRotationCommandCfg(CommandTermCfg):
    r"""连续旋转命令配置。

    该配置对应 a51c666 黄金 tactile 版的 command 语义：目标姿态不是从大范围随机采样，
    而是在成功后沿固定世界轴小步推进：
    $$
    Q_g^{k+1}=R_{\text{axis}}(\Delta\theta)Q_g^k,
    \qquad \Delta\theta=\pi/8\ \text{by default}.
    $$

    `command()` 输出 7D pose-like tensor `[p_g^e, Q_g^w]`，用于恢复历史 actor/critic
    的 quaternion goal-pose 观测，而不是当前 SO(3) rotvec 指令。
    """

    class_type: type = ContinuousRotationCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    asset_name: str = MISSING
    """参与重定向的物体在场景中的名称。"""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """相对于物体默认根姿态的位置偏移。"""

    rotation_axis: str = "z"
    """连续旋转所围绕的世界坐标系轴（x/y/z）。"""

    delta_angle: float = math.pi / 8.0
    """每次成功后的增量旋转角度（单位: rad）。"""

    make_quat_unique: bool = True
    """是否将目标四元数约束为唯一表示。"""

    orientation_success_threshold: float = NONE  # 改为MISSING会报错，因为它要求必须提供值
    """判定完成当前目标姿态的角度阈值（单位: rad）。"""

    update_goal_on_success: bool = True
    """是否在成功达到目标后沿轴继续更新目标。"""

    # NOTE:
    #   Play 模式会打开 debug_vis。历史 ContinuousRotationCommand 没有可视化实现，
    #   这里补齐 marker 配置，使 tactile golden 训练/评估共用同一个 command 类。
    goal_marker_pos_e: tuple[float, float, float] = (-0.2, -0.45, 0.68)
    """目标姿态 marker 的固定显示位置（环境坐标系 {e}），只用于可视化。"""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
            ),
        },
    )
    """目标姿态可视化 marker 配置，当前作为 inhand continuous rotation 命令的通用 marker。"""

    def __post_init__(self):
        """初始化后处理，根据 delta_angle 自动计算成功阈值（5%容差）"""
        # 如果未提供成功阈值，则根据 delta_angle 自动计算
        if self.orientation_success_threshold == NONE:
            # 参考 DirectRLEnv 实现，允许约 0.2rad 的姿态误差，同时兼容更大旋转步长
            self.orientation_success_threshold = max(0.2, self.delta_angle / 2.0)
        # print(f"成功阈值: {self.orientation_success_threshold}")


@configclass
class OfficialContinuousRotationCommandCfg(ContinuousRotationCommandCfg):
    r"""官方 LEAP 连续 z 轴重定向命令配置。

    该配置在 `ContinuousRotationCommandCfg` 的基础上只补一件事：
    当前小目标是否完成，不仅要求姿态误差足够小，还要求物体位置仍留在掌心附近。
    官方阈值为：
    $$
    \|p_o^e - p_g^e\|_2 \le 0.025.
    $$
    """

    class_type: type = OfficialContinuousRotationCommand
    position_success_threshold: float = 0.025
