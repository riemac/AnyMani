# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

import math
from pickle import NONE

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg




@configclass
class ReorientCommandCfg(CommandTermCfg):
    r"""TODO:重定向命令配置

    TOAGENT:流程说明不要删，但可以重述或润色

    预计采用 axis + error so(3)，即相对姿态增量（rotvec）的命令语义，$\hat\omega\theta\in\mathbb{R}^3$，或 $[\omega]\theta\in so(3)$。

    流程：
    1. 随机采样 axis-theta，即 Modern Robotics 中所谓的轴角对-$\hat\omega\theta\in\mathbb{R}^3$
    2. 获取该时刻的目标姿态 $R_{goal} = \exp([\hat\omega]\theta)R_{current}$。注意，这里是左乘，因为物体是绕 {h} 坐标系旋转，而非 {o} 自身。
    3. 获得 error so(3)，即当前姿态与目标姿态的相对旋转 $R_{error} = R_{goal}R_{current}^{-1}$，$ \hat\omega_e\theta_e=\text{Log}(R_{error}) $
    4. 如果 $\theta_e < \theta_{th}$ 阈值，则认为成功，进入下一个目标采样。

    命令项 axis 对应的就是 $\hat\omega$，在重定向过程中固定，直到进入下一个目标采样。error so(3) 对应的就是 $\hat\omega_e\theta_e$，在重定向过程中不断更新，直到 $\theta_e < \theta_{th}$。
    它们都位于欧式空间，拼接成 $\mathbb{R}^6$，比直接输入四元数/矩阵应该更适合 RL 学习。
    """

    class_type: type = ReorientCommand

    debug_vis: bool = False
    """是否启用目标姿态物块 marker 可视化。

    这是 Isaac Lab ``CommandTermCfg`` 的标准 debug visualization 开关。
    在 ``ReorientCommandCfg`` 中，它特指是否显示内部维护的目标姿态物块；
    训练默认关闭，play / review 时可打开。
    """

    theta_range: tuple[float, float] = (0.0, math.pi/2)
    r"""so(3) 命令向量的 norm 范围，即期望的旋转增量范围（单位: rad）。"""

    orientation_success_threshold: float = math.pi/12  # 改为MISSING会报错，因为它要求必须提供值
    """判定完成当前目标姿态的角度阈值（单位: rad）。"""

    axis_mode: Literal["random", "fixed"] = "random"
    """旋转轴采样模式。random: 每次采样一个随机轴；fixed: 固定轴（默认为 z 轴）。"""

    fixed_axis_h: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """``fixed`` 模式下的固定旋转轴，位于 hand semantic frame ``{h}``。

    例如 x/y/z 分别对应 ``(1, 0, 0)``、``(0, 1, 0)``、``(0, 0, 1)``；
    反向旋转用负轴，如 ``(0, 0, -1)``。实现时应自动归一化，但零向量应显式报错。
    """

    goal_marker_pos_e: tuple[float, float, float] = (-0.2, -0.45, 0.68)
    """目标姿态 marker 在环境坐标系 ``{e}`` 下的固定显示位置。

    该位置只服务可视化，不参与目标位置约束；实现时应加上 ``env_origins`` 转为世界坐标。
    """

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
            ),
        },
    )
    """目标姿态 marker 的 USD / scale / prim_path 配置。

    ``debug_vis`` 决定是否显示，本文段只决定显示成什么物块。
    """

    resampling_time_range: tuple[float, float] = (1e6, 1e6)
    """Isaac Lab CommandTerm 生命周期字段：近似禁用时间驱动的自动重采样。

    `ReorientCommand` 的科研语义是 reset / success-driven subgoal resampling，
    不是每隔固定秒数换目标。因此这里给一个极大值，避免 time-left 机制干扰。
    """

    def __post_init__(self):
        """初始化后处"""
