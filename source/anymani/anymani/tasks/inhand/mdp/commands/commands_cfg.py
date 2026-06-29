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

from .rotation_command import ContinuousRotationCommand, RelativeSO3Command


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
    """目标姿态可视化 marker 配置，外观对齐当前 RelativeSO3Command 的 marker。"""

    def __post_init__(self):
        """初始化后处理，根据 delta_angle 自动计算成功阈值（5%容差）"""
        # 如果未提供成功阈值，则根据 delta_angle 自动计算
        if self.orientation_success_threshold == NONE:
            # 参考 DirectRLEnv 实现，允许约 0.2rad 的姿态误差，同时兼容更大旋转步长
            self.orientation_success_threshold = max(0.2, self.delta_angle / 2.0)
        # print(f"成功阈值: {self.orientation_success_threshold}")


@configclass
class RelativeSO3CommandCfg(CommandTermCfg):
    """so(3) 相对增量指令（rotvec）命令配置。

    该配置与 :class:`~anymani.tasks.inhand.mdp.commands.rotation_command.RelativeSO3Command` 配套。

    设计动机：
        - 训练阶段：用 ``fixed_goal`` 将目标冻结，使误差可收敛；
        - 部署阶段：用 ``rolling_goal`` 保持指令恒定，实现持续旋转。

    Note:
        - ``theta_max`` 不建议取到 π，以规避 rotvec/quat 表示在 π 附近的数值奇异。
        - 当前实现默认在环境坐标系 {e} 下解释 rotvec。
    """

    class_type: type = RelativeSO3Command
    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    asset_name: str = MISSING
    """参与旋转指令的物体在场景中的名称。"""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """相对于物体默认根姿态的位置偏移（用于保持物体在掌心上方生成）。"""

    theta_min: float = 0.0
    """采样的最小旋转角（单位: rad）。"""

    theta_max: float = math.pi / 2.0
    """采样的最大旋转角（单位: rad）。"""

    mode: str = "fixed_goal"
    """指令模式：`fixed_goal` 或 `rolling_goal`。"""

    make_quat_unique: bool = True
    """是否将目标四元数约束为唯一表示（实部为正）。"""

    orientation_success_threshold: float = NONE
    """fixed_goal 下的成功阈值（单位: rad）。rolling_goal 下仅用于日志/指标。"""

    update_goal_on_success: bool = True
    """fixed_goal：是否在成功时重采样新指令。rolling_goal：该字段将被忽略。"""

    # NOTE:
    #   对齐参考实现（LEAP_Hand_Isaac_Lab）：
    #   目标 marker 并不叠加到真实物体位置上方，也不使用半透明/改色材质。
    #   它被放置在每个环境原点附近的一个固定位置，用于“展示目标旋转”。
    #
    #   该位置是环境坐标系 {e} 下的常量，最终可视化时会加上 env_origins 变为世界坐标系。
    goal_marker_pos_e: tuple[float, float, float] = (-0.2, -0.45, 0.68)
    """目标姿态 marker 的固定位置（环境坐标系 {e}）。"""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                # 对齐 LEAP：使用 DexCube 的原生外观（不改色/不改透明度），并使用相同缩放。
                scale=(1.2, 1.2, 1.2),
            ),
        },
    )
    """目标姿态可视化 marker 配置（默认使用 DexCube）。"""

    def __post_init__(self):
        # --- basic validation ---
        if not (0.0 <= float(self.theta_min) < float(self.theta_max)):
            raise ValueError(
                f"RelativeSO3CommandCfg requires 0 <= theta_min < theta_max, got: {self.theta_min}, {self.theta_max}"
            )
        # 避免接近 π 的数值不稳定（rotvec 在 π 附近存在等价类，学习信号更噪）
        if float(self.theta_max) >= math.pi:
            raise ValueError(
                f"RelativeSO3CommandCfg.theta_max must be < pi for numerical stability, got: {self.theta_max}"
            )

        mode = str(self.mode).lower()
        if mode not in {"fixed_goal", "rolling_goal"}:
            raise ValueError(f"RelativeSO3CommandCfg.mode must be 'fixed_goal' or 'rolling_goal', got: {self.mode}")
        self.mode = mode

        # default success threshold: keep consistent with existing tasks (0.2 rad)
        if self.orientation_success_threshold == NONE:
            self.orientation_success_threshold = 0.2
