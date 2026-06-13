import math
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .reorient_command import ReorientCommand


@configclass
class ReorientCommandCfg(CommandTermCfg):
    r"""Axis + SO(3) 重定向命令配置。

    TOAGENT:流程说明不要删，但可以重述或润色

    当前采用 axis + error so(3)，即相对姿态增量（rotvec）的命令语义：

    $$
    \mathbf{c}_t = [\hat\omega^{\{h\}},\ \phi_e^{\{h\}}] \in \mathbb{R}^6,
    \qquad
    \phi_e = \log(R_{goal}R_{current}^{-1})
    $$

    流程：
    1. 在 hand semantic frame `{h}` 中采样有向单位轴 $\hat\omega^{\{h\}}$，
       并采样正幅值 $\theta>0$。`(-axis,+theta)` 已等价于 `(axis,-theta)`，
       因此第一版不让 theta 再带符号，避免同一目标双重编码。
    2. 通过语义对齐矩阵 $R_{ha}$ 把 `{h}` 轴转到 raw asset/root frame `{a}`，
       再经 robot root 姿态转到运行时 `{e}`：
       $$
       v^{\{h\}} = R_{ha}v^{\{a\}},\quad
       v^{\{e\}} = R_{ea}R_{ha}^{\top}v^{\{h\}}
       $$
       generated assets 第一版默认 $R_{ha}=I$。
    3. 获取目标姿态：
       $$
       R_{goal} = \exp([\hat\omega^{\{e\}}]\theta)R_{current}
       $$
       注意这里是左乘，因为物体是绕 `{h}` / `{e}` 中的空间轴旋转，而非绕 `{o}` 自身。
    4. 获得 error so(3)：
       $$
       R_{error}=R_{goal}R_{current}^{-1},\qquad
       \phi_e=\log(R_{error})
       $$
    5. 如果 $\theta_e=\|\phi_e\|<\theta_{th}$，则认为当前 subgoal 成功，
       `goal_success_count += 1`，并按 `axis_resample_mode` 进入下一个目标采样。

    DONE(当前默认):
        - `axis_resample_mode="subgoal"`：每个成功 subgoal 后重采样 axis + theta。
        - `theta_range=(pi/6, pi/2)`：下限显式大于成功阈值，避免刚采样就成功。
        - policy obs 只暴露 `[axis_h, error_so3_h]`；reward / termination / curriculum
          读取内部 buffer：`goal_quat_w`、`axis_e`、`error_so3_e`、
          `goal_success_count`。
    """

    class_type: type = ReorientCommand
    """DONE: 指向可运行的 `ReorientCommand`。

    该 command term 负责维护 `goal_quat_w`、`axis_h/e`、`error_so3_h/e` 与
    `goal_success_count`，使 observation / reward / curriculum 不再从 6D command
    反推内部目标姿态。
    """

    asset_name: str = "object"
    """被操作物体在 scene 中的名字；默认与 `GmInHandSceneCfg.object` 对齐。"""

    robot_asset_name: str = "robot"
    """手部 articulation 在 scene 中的名字；用于读取 root pose，完成 `{a}->{e}` 变换。"""

    debug_vis: bool = False
    """是否启用 command 可视化。

    这是 Isaac Lab ``CommandTermCfg`` 的标准 debug visualization 开关。
    在 ``ReorientCommandCfg`` 中，它包括两类 marker：

    1. goal object marker：显示当前 subgoal 的目标物体姿态；
    2. axis arrow marker：显示 axis + error-so(3) command 中的 axis 方向。

    训练默认关闭，play / review 时可打开。
    """

    theta_range: tuple[float, float] = (math.pi / 6, math.pi / 2)
    r"""subgoal 旋转幅值范围（单位: rad）。

    默认下限 $\pi/6$ 大于默认 success threshold $\pi/12$，避免目标刚采样
    就已经满足成功条件，从而虚增 `goal_success_count` 并污染 reward curriculum。
    """

    orientation_success_threshold: float = math.pi/12  # 改为MISSING会报错，因为它要求必须提供值
    """判定完成当前目标姿态的角度阈值（单位: rad）。"""

    axis_mode: Literal["random", "fixed"] = "random"
    """旋转轴采样模式。random: 每次采样一个随机轴；fixed: 固定轴（默认为 z 轴）。"""

    axis_resample_mode: Literal["subgoal", "episode"] = "subgoal"
    """随机轴生命周期。

    - `subgoal`：默认。每个 subgoal 成功后重新采样 axis + theta，训练任意局部
      重定向 primitive。
    - `episode`：每个 episode reset 时采样 axis，episode 内只换 theta；更接近
      固定轴连续旋转消融。

    若 `axis_mode="fixed"`，该字段不会改变 axis 值，只影响文义记录。
    """

    fixed_axis_h: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """``fixed`` 模式下的固定旋转轴，位于 hand semantic frame ``{h}``。

    例如 x/y/z 分别对应 ``(1, 0, 0)``、``(0, 1, 0)``、``(0, 0, 1)``；
    反向旋转用负轴，如 ``(0, 0, -1)``。实现时应自动归一化，但零向量应显式报错。
    """

    semantic_R_ha: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    r"""语义对齐矩阵 $R_{ha}$，row-major 9 个 float。

    语义为：
    $$
    v^{\{h\}} = R_{ha} v^{\{a\}}
    $$

    generated assets 默认 raw asset/root frame `{a}` 与 hand semantic frame `{h}`
    对齐，因此 $R_{ha}=I$。真实 LEAP / Allegro URDF 后续可通过人工视觉校准
    或配置项填写该矩阵。

    只配置旋转，不配置平移：command 只关心旋转轴和 orientation error，位置
    语义暂由 reset / termination 的 task anchor 单独处理。
    """

    make_quat_unique: bool = False
    """是否把 goal quaternion 标准化到实部非负。

    该字段只影响可视化 / 日志的 quaternion 连续性，不改变物理姿态。policy obs
    不直接吃裸四元数，因此默认 False。
    """

    keypoint_radius: float = 0.05
    r"""command metric `keypoint_error` 使用的六轴向 keypoint 半径（单位: m）。

    与 AnyRotate 使用的 $5\,\text{cm}$ 数值锚点对齐。该 metric 只用于日志 / 消融，
    不改变默认 success-driven goal update。
    """

    goal_marker_pos_h: tuple[float, float, float] = (0.0, 0.0, 0.25)
    """目标姿态 marker 在 hand semantic frame ``{h}`` 下的固定显示位置。

    语义：从 hand/root anchor 出发沿 `{h}` z 正向放在手正上方 25cm。该位置
    只服务可视化，不参与任务目标、奖励或终止条件。实现时应统一消费
    `semantic_R_ha`，先把该 offset 从 `{h}` 转到 `{a}` / `{e}`，再加 robot root
    position 或 env origin 得到 world position。

    NOTE:
        第一版 generated assets 默认 root frame 近似 hand semantic anchor，故可
        将该 offset 视为相对 robot root 的 hand-frame 偏移。未来如果需要严格
        palm anchor，可额外引入 `marker_anchor_body_name`，但当前不做。
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

    axis_arrow_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/reorient_axis"
    )
    """axis arrow marker 的 USD / scale / prim_path 配置。

    IsaacLab 的 arrow marker 默认沿 marker 局部 +x 方向。未来实现时，应构造
    一个 quaternion，将 +x 旋到当前 command axis 的 world/env 表达 `axis_e`，
    并在 `goal_marker_pos_h` 对应的 world 位置显示该箭头。

    该箭头只表达 axis 方向，不表达 theta 大小；theta 由 goal object marker 的
    目标姿态体现。
    """

    axis_arrow_length: float = 0.15
    """axis arrow 的固定显示长度（单位: m）。

    实现时可通过修改 marker scale 的 x 分量控制长度。默认 15cm 只作为
    play/debug 可视化 preset，不进入 reward / observation。
    """

    resampling_time_range: tuple[float, float] = (1e6, 1e6)
    """Isaac Lab CommandTerm 生命周期字段：近似禁用时间驱动的自动重采样。

    `ReorientCommand` 的科研语义是 reset / success-driven subgoal resampling，
    不是每隔固定秒数换目标。因此这里给一个极大值，避免 time-left 机制干扰。
    """

    def __post_init__(self):
        r"""配置合法性检查。

        这里做轻量静态检查；$R_{ha}$ 的正交性由 `ReorientCommand` 在 torch
        device 上进一步检查。
        """

        theta_min, theta_max = self.theta_range
        if theta_min <= 0.0 or theta_max <= theta_min:
            raise ValueError(f"theta_range must satisfy 0 < min < max, got {self.theta_range}.")
        if theta_min <= self.orientation_success_threshold:
            raise ValueError(
                "theta_range[0] should be larger than orientation_success_threshold to avoid trivially successful "
                f"subgoals, got theta_min={theta_min}, threshold={self.orientation_success_threshold}."
            )
        if len(self.semantic_R_ha) != 9:
            raise ValueError(f"semantic_R_ha must contain 9 row-major floats, got {len(self.semantic_R_ha)}.")
