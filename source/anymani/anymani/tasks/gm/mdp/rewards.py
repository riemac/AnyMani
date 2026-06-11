r"""Reward terms for `tasks.gm`.

本模块只承载 generalized in-hand manipulation 的任务奖励与正则项。奖励设计
应描述“物体是否完成了手内操作目标”，不要在 reward 中偷偷编码资产采样
偏好；资产 bank 如何采样、哪些 hand variant 进入训练，属于 `distill` 的训练
组织问题，不属于 `gm/mdp/rewards.py`。

当前奖励设计对齐 AnyRotate 的分组：

$$
r = r_{\text{reorient}} + r_{\text{contact}} + r_{\text{stable}} + r_{\text{terminate}}
$$

其中 `r_reorient` 是主任务项，采用“随机重定向子目标”来训练可复合的连续
手内旋转 primitive；最终绕 `{h}` 轴连续旋转可视为多个重定向子目标的复合：

$$
T_{ab} T_{bc} = T_{ac}, \qquad
R_{g} = \exp([\hat\omega] \theta) R_{o}
$$

DONE(本轮已合意的 reward 语义):
    1. `r_reorient` 第一版以 AnyRotate 风格的 keypoint distance reward 为主，
       但 command / success 的数学语义默认仍采用 $SO(3)$ geodesic threshold。
    2. keypoints 第一版使用 object body frame `{o}` 下的 $\pm x,\pm y,\pm z$
       六个轴向点，半径默认 $5\,\text{cm}$，只衡量姿态，不惩罚物体中心平移。
    3. 加入 axis delta rotation reward：
       $\operatorname{clip}(\log(R_t R_{t-1}^{-1}) \cdot \hat\omega, -c, c)$，
       与 command 的“空间轴左乘”语义一致。
    4. `r_contact` 中 good fingertip contact 使用二值 $n_{tip}\ge k$，默认
       `min_contacts=2`；bad non-tip contact 通过 cfg 显式传入 sensor names，
       reward 不猜 asset schema。
    5. contact / stable / action 正则项走 adaptive reward curriculum；每个项
       保留 `lambda_floor` 可配置。默认 good contact floor 为 `0.05`，其他
       bad contact / pose / torque / work / action 正则为 `0.0`。

TODO(仍待后续讨论 / 实现收敛):
    - command term 需要显式暴露 `goal_quat_w`、`axis_e`、`goal_success_count`
      等 buffer / metric；不要让 reward 反向猜 command 的内部状态。
    - keypoint reward 的 AnyRotate 曲线目前采用归一化写法；若后续需要严格
      复现实验，应逐项核对论文 Appendix B 的未归一化常数。
    - 物体位置保持与掉落应进入独立 reward / termination，不混进姿态 keypoint
      reward，避免把“手内自然漂移”错误惩罚成姿态失败。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import isaaclab.envs.mdp as isaac_mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


# ==================
# shared helpers
# ==================


def _resolve_goal_quat_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""从 command term 中解析目标姿态 $R_g$ 的四元数表达。

    `gm` 的 command 观测可以是 axis + error-so(3) 的 6D 张量，但 reward 计算
    keypoint distance / SO(3) success 时需要内部目标姿态。因此目标姿态应由
    command term 以 buffer 形式显式暴露，而不是让 reward 从 6D command 中反推。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): command manager 中的 command term 名称。

    Returns:
        torch.Tensor: 目标姿态四元数，形状 `[num_envs, 4]`，约定为 Isaac Lab
        的 `(w, x, y, z)`。

    Raises:
        RuntimeError: 当 command term 没有暴露目标四元数，且 command tensor 也
        不是 legacy pose-like `[pos, quat]` 形式时抛出。
    """

    # 优先读取 command term 的内部 buffer：这是 `gm` 后续 ReorientCommand 应兑现的契约
    command_term = env.command_manager.get_term(command_name)  # command term 实例，承载目标姿态 buffer
    for attr_name in ("goal_quat_w", "quat_command_w"):
        goal_quat_w = getattr(command_term, attr_name, None)  # `[B,4]`，目标姿态 quaternion
        if isinstance(goal_quat_w, torch.Tensor):
            return goal_quat_w  # 直接返回显式 buffer，避免从观测向量反解内部状态

    # 兼容 IsaacLab 官方 inhand 的 legacy command：command = `[pos_e, quat_w]`
    command = env.command_manager.get_command(command_name)  # 可能是 `[B,7]` pose command，也可能是 `[B,6]` axis+error
    if isinstance(command, torch.Tensor) and command.shape[-1] >= 7:
        return command[:, -4:]  # legacy pose command 的最后四维是目标 quaternion

    # 如果走到这里，说明 command / reward 契约尚未补齐，必须显式失败而不是静默训练错 reward
    raise RuntimeError(
        f"Command '{command_name}' must expose `goal_quat_w` / `quat_command_w`, "
        "or return a legacy pose-like command tensor with final 4 quaternion dims. "
        f"Got command shape: {getattr(command, 'shape', None)}."
    )


def _resolve_axis_e(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""从 command term 中解析空间旋转轴 $\hat\omega^{\{e\}}$。

    本项目的 command 语义锚定在 hand semantic frame `{h}`，例如最终测试的
    z 轴旋转应解释为 $\hat\omega^{\{h\}}=(0,0,1)$。但 reward 计算实际姿态
    增量 $\log(R_t R_{t-1}^{-1})$ 时使用的是世界 / 环境系下的旋转矩阵，因此
    投影轴必须已经由 command term 转到 `{e}` 或 `{w}`。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): command manager 中的 command term 名称。

    Returns:
        torch.Tensor: 单位旋转轴，形状 `[num_envs, 3]`，坐标系为 `{e}` / `{w}`。

    Raises:
        RuntimeError: 当 command term 没有暴露空间轴时抛出。
    """

    # reward 不应自行猜 `{h}->{e}` 对齐；该变换属于 command term 的职责
    command_term = env.command_manager.get_term(command_name)  # command term 实例，理应持有轴向 buffer
    for attr_name in ("axis_e", "axis_w", "axis_command_e", "axis_command_w"):
        axis_e = getattr(command_term, attr_name, None)  # `[B,3]`，空间旋转轴
        if isinstance(axis_e, torch.Tensor):
            return axis_e / (torch.linalg.norm(axis_e, dim=-1, keepdim=True) + 1.0e-6)  # 单位化，防止配置误差

    # 明确要求 command term 暴露空间轴，避免把 hand frame 轴误当成 world frame 轴
    raise RuntimeError(
        f"Command '{command_name}' must expose `axis_e` / `axis_w` for axis-progress reward. "
        "Reward terms intentionally do not infer `{h}->{e}` alignment."
    )


def _six_axis_keypoints_o(device: torch.device | str, radius: float) -> torch.Tensor:
    r"""生成 AnyRotate 风格的六轴向 object keypoints。

    六个点定义在 object body frame `{o}` 中：

    $$
    \mathcal{P}_o = \{\pm r\mathbf{e}_x,\ \pm r\mathbf{e}_y,\ \pm r\mathbf{e}_z\}
    $$

    第一版只用姿态差异：比较 $R_o p_i$ 与 $R_g p_i$，不加 object center，
    因此不会把物体在手内的小幅平移直接惩罚进姿态 reward。

    Args:
        device (torch.device | str): 输出张量所在设备。
        radius (float): keypoint 半径，单位 m；AnyRotate 使用 $5\,\text{cm}$。

    Returns:
        torch.Tensor: keypoints，形状 `[6, 3]`，坐标系 `{o}`。
    """

    # 这里每次构造的成本极小；若后续 object mesh keypoints 增多，可缓存为 module-level 常量或 scene buffer
    return torch.tensor(
        [
            [radius, 0.0, 0.0],   # $+r\mathbf{e}_x$，object 局部 x 正向点
            [-radius, 0.0, 0.0],  # $-r\mathbf{e}_x$，object 局部 x 反向点
            [0.0, radius, 0.0],   # $+r\mathbf{e}_y$，object 局部 y 正向点
            [0.0, -radius, 0.0],  # $-r\mathbf{e}_y$，object 局部 y 反向点
            [0.0, 0.0, radius],   # $+r\mathbf{e}_z$，object 局部 z 正向点
            [0.0, 0.0, -radius],  # $-r\mathbf{e}_z$，object 局部 z 反向点
        ],
        dtype=torch.float32,
        device=device,
    )


def _orientation_keypoint_distance(
    current_quat_w: torch.Tensor,
    goal_quat_w: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    r"""计算 orientation-only keypoint distance。

    对每个 object-local keypoint $p_i^{\{o\}}$，只比较旋转后的方向：

    $$
    d_{kp} = \frac{1}{N}\sum_{i=1}^{N}
    \left\| R_o p_i^{\{o\}} - R_g p_i^{\{o\}} \right\|_2
    $$

    注意这里故意不加 object center $x_o$ / goal center $x_g$，因此该项只塑造
    姿态误差；位置保持、掉落、离手等语义应由独立 term 处理。

    Args:
        current_quat_w (torch.Tensor): 当前物体姿态，形状 `[B,4]`，`(w,x,y,z)`。
        goal_quat_w (torch.Tensor): 目标物体姿态，形状 `[B,4]`，`(w,x,y,z)`。
        radius (float): keypoint 半径，单位 m。

    Returns:
        torch.Tensor: 平均 keypoint distance，形状 `[B]`，单位 m。
    """

    # 将 quaternion 变为旋转矩阵，矩阵语义为 $R\in SO(3)$，形状 `[B,3,3]`
    current_rot_w = math_utils.matrix_from_quat(current_quat_w)  # 当前物体姿态 $R_o$
    goal_rot_w = math_utils.matrix_from_quat(goal_quat_w)        # 目标物体姿态 $R_g$

    # 生成 `{o}` 下六个轴向点；同一物体半径在所有 env 共享，形状 `[6,3]`
    keypoints_o = _six_axis_keypoints_o(device=current_quat_w.device, radius=radius)  # $p_i^{\{o\}}$

    # 左乘旋转矩阵得到 orientation-only keypoints，形状 `[B,6,3]`
    current_points = torch.einsum("bij,kj->bki", current_rot_w, keypoints_o)  # $R_o p_i$
    goal_points = torch.einsum("bij,kj->bki", goal_rot_w, keypoints_o)        # $R_g p_i$

    # 对 6 个 keypoints 求平均 L2 距离，单位 m；该值越小表示姿态越接近
    return torch.linalg.norm(current_points - goal_points, dim=-1).mean(dim=-1)  # $d_{kp}$，形状 `[B]`


def _curriculum_gain(
    env: ManagerBasedRLEnv,
    lambda_floor: float,
    lambda_max: float = 1.0,
    lambda_attr_name: str = "_gm_reward_curriculum_lambda",
    default_lambda: float = 1.0,
) -> torch.Tensor:
    r"""读取全局 reward curriculum 系数，并映射为某个 reward term 的门控系数。

    全局 curriculum term 负责维护一个标量 $\lambda_{global}\in[0,1]$，表示
    “策略整体完成重定向子目标的成熟度”。每个 reward 项再通过自己的
    `lambda_floor` 决定早期是否保留一点塑形信号：

    $$
    \lambda_i = \lambda_{floor,i}
      + (\lambda_{max,i}-\lambda_{floor,i})\lambda_{global}
    $$

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        lambda_floor (float): 当前 reward 项的最小门控系数。
        lambda_max (float): 当前 reward 项最终释放后的最大门控系数。
        lambda_attr_name (str): curriculum term 写入 env 的属性名。
        default_lambda (float): 若 curriculum 尚未启用，默认视作全开还是全关。

    Returns:
        torch.Tensor: 每个 env 的门控系数，形状 `[num_envs]`。
    """

    # 若没有显式 curriculum，默认不让 reward 静默消失；训练 cfg 可通过启用 curriculum 覆盖该行为
    global_lambda = getattr(env, lambda_attr_name, None)  # 标量或 `[B]`，由 `RewardCurriculumByGoalSuccess` 写入
    if global_lambda is None:
        global_lambda = torch.tensor(default_lambda, device=env.device, dtype=torch.float32)  # 默认全开，防止漏配导致 0 reward
    elif not isinstance(global_lambda, torch.Tensor):
        global_lambda = torch.tensor(float(global_lambda), device=env.device, dtype=torch.float32)  # Python float → device tensor
    else:
        global_lambda = global_lambda.to(device=env.device, dtype=torch.float32)  # 确保 device / dtype 与 env 一致

    # 裁剪到 $[0,1]$，避免 curriculum 数值误差把 reward 权重推到预期范围外
    global_lambda = torch.clamp(global_lambda, 0.0, 1.0)  # $\lambda_{global}$
    gain = float(lambda_floor) + (float(lambda_max) - float(lambda_floor)) * global_lambda  # $\lambda_i$

    # RewardManager 期望返回 `[B]`；如果 curriculum 是全局标量，则扩展到所有 env
    if gain.ndim == 0:
        gain = gain.expand(env.num_envs)  # `[B]`，所有 env 共用同一全局 release 系数
    return gain


def _sensor_contact_indicator(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    r"""判断某个显式 ContactSensor 是否与 object 发生有效接触。

    `ContactSensorCfg` 应在 scene 中显式挂到单个 body prim，例如某个 fingertip
    或某个 non-tip link，并设置：

    ```python
    filter_prim_paths_expr=["{ENV_REGEX_NS}/object"]
    track_friction_forces=True  # 若希望接触力包含切向摩擦分量
    ```

    IsaacLab 的 filtered contact 数据约定：
        - `force_matrix_w`: normal force，形状 `[B, body, filter, 3]`；
        - `friction_forces_w`: friction force，形状 `[B, body, filter, 3]`；
        - `net_forces_w`: 未过滤的 normal force，形状 `[B, body, 3]`。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        sensor_name (str): scene 中 ContactSensor 的名字。
        force_threshold (float): 判断接触的力阈值，单位 N。

    Returns:
        torch.Tensor: bool tensor，形状 `[num_envs]`。
    """

    # 显式从 scene 取 sensor；reward 不根据 hand.yaml 推断 prim path，避免跨模块耦合
    sensor = env.scene[sensor_name]  # ContactSensor；名字由 cfg 显式传入
    force_w = getattr(sensor.data, "force_matrix_w", None)  # filtered normal force，优先使用 object-filtered 数据

    # 如果没有 filtered force，则退回 net normal force；这种情况不适合 bad/non-tip 精确判断，但可作为容错
    if force_w is None:
        force_w = sensor.data.net_forces_w  # `[B,body,3]`，未按 object 过滤的 normal force
    if force_w is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' does not expose force data.")

    # 把 NaN 接触点/摩擦项视为无接触贡献；IsaacLab 在无 filtered pair contact 时可能填 NaN
    total_force_w = torch.nan_to_num(force_w, nan=0.0)  # normal force，形状 `[B,...,3]`
    friction_w = getattr(sensor.data, "friction_forces_w", None)  # filtered friction force，若未启用则为 None
    if friction_w is not None:
        total_force_w = total_force_w + torch.nan_to_num(friction_w, nan=0.0)  # normal + friction，总接触力近似

    # 对 body/filter 维度取最大力幅值；只要该 sensor 任意接触 pair 超阈值，就视为该部位接触
    force_norm = torch.linalg.norm(total_force_w, dim=-1)  # `[B,...]`，力幅值 $\|F\|_2$
    if force_norm.ndim > 1:
        force_norm = force_norm.amax(dim=tuple(range(1, force_norm.ndim)))  # `[B]`，sensor 内最大接触强度

    return force_norm > float(force_threshold)  # `[B]`，二值接触指示


# ==================
# $r_{reorient}$
# ==================


def reorientation_reward_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""临时占位：保持当前 `GmRewardsCfg` 可导入，但不提供真实任务奖励。

    TODO:
        正式训练不能使用该项。后续应在 env cfg 中切换到
        `keypoint_reorientation_reward`、`AxisDeltaRotationReward`、
        `goal_success_bonus` 等明确 reward terms。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。

    Returns:
        torch.Tensor: 全零 reward，形状 `[num_envs]`。
    """

    return torch.zeros(env.num_envs, device=env.device)  # 占位项，不改变任何训练梯度信号


def keypoint_reorientation_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    keypoint_radius: float = 0.05,
    curve_sharpness: float = 50.0,
    curve_bias: float = 2.0,
) -> torch.Tensor:
    r"""AnyRotate 风格的 orientation-only keypoint distance reward。

    第一版使用 `{o}` 下六个轴向 keypoints，并只比较姿态：

    $$
    d_{kp} = \frac{1}{6}\sum_{i=1}^{6}
    \left\| R_o p_i^{\{o\}} - R_g p_i^{\{o\}} \right\|_2
    $$

    reward 曲线采用 AnyRotate Appendix B 的 squashed distance reward 思路，
    这里写成归一化版本，使 $d_{kp}=0$ 时 reward 为 1：

    $$
    r_{kp} = \frac{2+b}{\exp(a d_{kp}) + b + \exp(-a d_{kp})}
    $$

    其中 $a=50, b=2.0$ 是 AnyRotate 文中的数值锚点。该写法保留“距离越小
    奖励越大且有界”的形状，同时让 reward weight 更容易解释。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): 提供目标姿态的 command term 名称。
        object_cfg (SceneEntityCfg): object asset 配置，默认 `SceneEntityCfg("object")`。
        keypoint_radius (float): 六轴向 keypoints 半径，单位 m，默认 $0.05$。
        curve_sharpness (float): 曲线陡峭度 $a$，默认 $50$。
        curve_bias (float): 曲线偏置 $b$，默认 $2.0$。

    Returns:
        torch.Tensor: keypoint reorientation reward，形状 `[num_envs]`。
    """

    # 解析 object 当前姿态与 command 内部目标姿态，二者均为 world quaternion `(w,x,y,z)`
    asset: RigidObject = env.scene[object_cfg.name]  # 被操作物体
    current_quat_w = asset.data.root_quat_w  # `[B,4]`，当前 object orientation
    goal_quat_w = _resolve_goal_quat_w(env, command_name)  # `[B,4]`，目标 object orientation

    # 计算 orientation-only keypoint distance；该距离不惩罚 object center 平移
    distance = _orientation_keypoint_distance(current_quat_w, goal_quat_w, radius=keypoint_radius)  # $d_{kp}$，单位 m

    # 指数项做上界裁剪，避免极大姿态误差时 `exp(a d)` 数值溢出；30 对应约 $e^{30}$，已足够接近 0 reward
    x = torch.clamp(float(curve_sharpness) * distance, min=0.0, max=30.0)  # $a d_{kp}$，无量纲
    denominator = torch.exp(x) + float(curve_bias) + torch.exp(-x)  # $\exp(x)+b+\exp(-x)$
    numerator = 2.0 + float(curve_bias)  # 归一化常数，使 $d_{kp}=0$ 时 $r_{kp}=1$

    return numerator / denominator  # `[B]`，有界于 $(0,1]$ 的主姿态 reward


def goal_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    success_mode: Literal["so3", "keypoint", "both"] = "so3",
    orientation_success_threshold: float | None = None,
    keypoint_success_threshold: float = 0.02,
    keypoint_radius: float = 0.05,
) -> torch.Tensor:
    r"""重定向子目标成功 bonus。

    默认采用 $SO(3)$ geodesic threshold：

    $$
    \theta_e = \left\|\log(R_g R_o^{-1})\right\|_2,
    \qquad \theta_e < \theta_{th}
    $$

    同时保留 keypoint / both 两种模式，便于后续声明式 cfg 切换。用户当前
    个人倾向是 `so3`，因为它与 axis + error-so(3) command 语义最一致。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): 提供目标姿态和阈值配置的 command term 名称。
        object_cfg (SceneEntityCfg): object asset 配置。
        success_mode (Literal["so3", "keypoint", "both"]): 成功判据模式。
        orientation_success_threshold (float | None): SO(3) 角误差阈值，单位 rad；
            若为 `None`，优先读取 command cfg 的 `orientation_success_threshold`。
        keypoint_success_threshold (float): keypoint distance 阈值，单位 m。
        keypoint_radius (float): keypoint 半径，单位 m。

    Returns:
        torch.Tensor: 成功 bonus 指示，形状 `[num_envs]`，值为 0/1 float。
    """

    # 解析 object 当前姿态与 command 目标姿态
    asset: RigidObject = env.scene[object_cfg.name]  # 被操作物体
    current_quat_w = asset.data.root_quat_w  # `[B,4]`，当前姿态
    goal_quat_w = _resolve_goal_quat_w(env, command_name)  # `[B,4]`，目标姿态

    # 若阈值未显式传入，则读取 command cfg；读取失败时使用 command cfg 中讨论过的 $\pi/12$ 默认值
    command_term = env.command_manager.get_term(command_name)  # command term，用于访问 cfg
    if orientation_success_threshold is None:
        orientation_success_threshold = getattr(command_term.cfg, "orientation_success_threshold", math.pi / 12.0)

    # SO(3) geodesic error；`quat_error_magnitude` 返回轴角向量的 L2 norm，即角度 rad
    dtheta = math_utils.quat_error_magnitude(goal_quat_w, current_quat_w)  # $\theta_e$，形状 `[B]`
    so3_success = dtheta <= float(orientation_success_threshold)  # `[B]`，SO(3) 成功指示

    # Keypoint success 只衡量姿态 keypoints，不惩罚 center translation
    keypoint_distance = _orientation_keypoint_distance(current_quat_w, goal_quat_w, radius=keypoint_radius)  # `[B]`，单位 m
    keypoint_success = keypoint_distance <= float(keypoint_success_threshold)  # `[B]`，keypoint 成功指示

    # 声明式选择成功判据，便于后续 ablation / cfg 切换
    if success_mode == "so3":
        success = so3_success  # 当前推荐默认：保持 command 数学语义纯净
    elif success_mode == "keypoint":
        success = keypoint_success  # AnyRotate-style success 判据备选
    elif success_mode == "both":
        success = so3_success & keypoint_success  # 更严格的双判据，可能拖慢早期训练
    else:
        raise ValueError(f"Unsupported success_mode: {success_mode}.")

    return success.float()  # RewardManager 需要 float reward，而不是 bool tensor


class AxisDeltaRotationReward(ManagerTermBase):
    r"""沿 command axis 的单步实际旋转增量奖励。

    AnyRotate 中 `r_rot` 的核心思想是奖励物体绕目标轴持续前进，而不是只在
    到达离散 goal 时给稀疏成功信号。本项目的 command 是空间轴左乘：

    $$
    R_g = \exp([\hat\omega] \theta) R_o
    $$

    因此实际旋转进度也应使用 left-increment：

    $$
    \Delta\phi_t = \log(R_t R_{t-1}^{-1}) \in \mathbb{R}^3,
    \qquad
    r_{rot} = \operatorname{clip}(\Delta\phi_t^\top\hat\omega,
    -c, c)
    $$

    该项是 stateful reward term，因为它需要缓存上一帧 object orientation。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        r"""初始化上一帧姿态缓存。

        Args:
            cfg: Isaac Lab `RewardTermCfg`，由 RewardManager 注入。
            env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        """

        # 父类保存 cfg/env；实际参数由 __call__ 每步从 RewardManager 传入
        super().__init__(cfg, env)

        # 初始化为单位 quaternion，随后第一次 __call__ 会用当前 object 姿态覆盖
        self._prev_quat_w = torch.zeros(env.num_envs, 4, device=env.device)  # `[B,4]`，上一帧 object orientation
        self._prev_quat_w[:, 0] = 1.0  # 单位 quaternion `(1,0,0,0)`，避免未初始化 NaN
        self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # `[B]`，是否已有有效上一帧

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""在 env reset 时清空上一帧姿态缓存。

        Args:
            env_ids (Sequence[int] | None): reset 的 env ids；`None` 表示全部 env。
        """

        # reset 后第一帧不应产生伪造的 delta rotation，因此只清空有效标记
        if env_ids is None:
            self._has_prev[:] = False  # 全部 env 下一步 reward 置零并重新对齐缓存
        else:
            self._has_prev[env_ids] = False  # 只清空被 reset 的 env

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        clip_value: float = 0.025,
    ) -> torch.Tensor:
        r"""计算沿 command axis 的 clipped SO(3) left-increment。

        Args:
            env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
            command_name (str): 提供空间轴 `axis_e` / `axis_w` 的 command term 名称。
            object_cfg (SceneEntityCfg): object asset 配置。
            clip_value (float): 单步旋转增量裁剪阈值，单位 rad；AnyRotate 使用 $0.025$。

        Returns:
            torch.Tensor: 轴向旋转进度 reward，形状 `[num_envs]`。
        """

        # 读取当前 object 姿态与 command 空间轴；二者都在 `{e}` / `{w}` 语义下比较
        asset: RigidObject = env.scene[object_cfg.name]  # 被操作物体
        current_quat_w = asset.data.root_quat_w  # `[B,4]`，当前 object orientation
        axis_e = _resolve_axis_e(env, command_name)  # `[B,3]`，单位空间轴 $\hat\omega$

        # 第一次调用或 reset 后没有上一帧姿态，此时 delta reward 应为 0，并建立缓存
        valid = self._has_prev.clone()  # `[B]`，clone 防止后续更新影响本步 mask
        prev_quat_w = self._prev_quat_w.clone()  # `[B,4]`，上一帧姿态快照

        # 将 quaternion 转为旋转矩阵，构造 left-increment $R_t R_{t-1}^{-1}$
        current_rot_w = math_utils.matrix_from_quat(current_quat_w)  # $R_t$，形状 `[B,3,3]`
        prev_rot_w = math_utils.matrix_from_quat(prev_quat_w)        # $R_{t-1}$，形状 `[B,3,3]`
        delta_rot_w = current_rot_w @ prev_rot_w.transpose(-1, -2)   # $R_t R_{t-1}^{-1}$

        # 矩阵 → quaternion → axis-angle vector，得到 $\Delta\phi_t=\log(R_tR_{t-1}^{-1})$
        delta_quat_w = math_utils.quat_from_matrix(delta_rot_w)  # `[B,4]`，delta quaternion
        delta_rotvec_w = math_utils.axis_angle_from_quat(delta_quat_w)  # `[B,3]`，so(3) 向量，单位 rad

        # 投影到 command axis，并裁剪到 AnyRotate 的单步进度范围
        progress = torch.sum(delta_rotvec_w * axis_e, dim=-1)  # `[B]`，$\Delta\phi_t^\top\hat\omega$，单位 rad
        progress = torch.clamp(progress, -float(clip_value), float(clip_value))  # clipped progress reward
        progress = torch.where(valid, progress, torch.zeros_like(progress))  # reset 后首帧不计入伪进度

        # 更新缓存，为下一步计算 left-increment 做准备；detach 避免 reward manager 保存图引用
        self._prev_quat_w[:] = current_quat_w.detach()  # `[B,4]`，缓存当前姿态
        self._has_prev[:] = True  # 下一步开始所有未 reset env 均有有效上一帧

        return progress


# ==================
# $r_{contact}$
# ==================


def good_fingertip_contact(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    min_contacts: int = 2,
    force_threshold: float = 1.0,
    use_curriculum: bool = True,
    lambda_floor: float = 0.05,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Good Contact reward：鼓励至少 `min_contacts` 个指尖与物体接触。

    对齐 AnyRotate Appendix B 的二值接触项：

    $$
    r_{gc} =
    \begin{cases}
    1, & n_{tip-contact} \ge k \\
    0, & \text{otherwise}
    \end{cases}
    $$

    默认 `lambda_floor=0.05`，含义是：训练一开始也给一个很弱的多指接触
    提示，但不会像 full contact reward 那样压过重定向主任务。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        sensor_names (Sequence[str]): 指尖 ContactSensor 名称列表。
        min_contacts (int): 至少接触的指尖数，默认 2。
        force_threshold (float): 判断接触的力阈值，单位 N。
        use_curriculum (bool): 是否乘以 adaptive reward curriculum 系数。
        lambda_floor (float): curriculum 早期下限，默认 0.05。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: reward，形状 `[num_envs]`。
    """

    # 逐个显式 sensor 统计是否接触；reward 不依赖 hand topology / link metadata 自动推断
    contact_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)  # `[B]`，接触指尖数
    for sensor_name in sensor_names:
        contact_count += _sensor_contact_indicator(env, sensor_name, force_threshold).int()  # 每个 sensor 贡献 0/1

    # AnyRotate 风格二值 good contact：至少 k 个指尖接触即给 1
    reward = (contact_count >= int(min_contacts)).float()  # `[B]`，$r_{gc}$

    # curriculum 只调节该项权重，不改变二值接触判据本身
    if use_curriculum:
        reward = reward * _curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)  # $\lambda_{gc} r_{gc}$
    return reward


def bad_non_tip_contact(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    force_threshold: float = 0.5,
    use_curriculum: bool = True,
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Bad Contact penalty indicator：检测任意非指尖部位是否接触物体。

    对齐 AnyRotate Appendix B 的非指尖接触惩罚项：

    $$
    r_{bc} =
    \begin{cases}
    1, & n_{non-tip-contact} > 0 \\
    0, & \text{otherwise}
    \end{cases}
    $$

    本函数返回正值 indicator，实际惩罚由 `RewardsCfg` 中的负权重实现。
    默认 `lambda_floor=0`，即早期不惩罚 palm / link 辅助接触，避免策略在尚未
    学会重定向前被过早约束到狭窄行为流形。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        sensor_names (Sequence[str]): 非指尖 ContactSensor 名称列表。
        force_threshold (float): 判断接触的力阈值，单位 N。
        use_curriculum (bool): 是否乘以 adaptive reward curriculum 系数。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: penalty indicator，形状 `[num_envs]`。
    """

    # 统计是否存在任何非指尖部位接触；只要有一个 sensor 超阈值，就触发 bad contact
    any_bad_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)  # `[B]`，是否有非指尖接触
    for sensor_name in sensor_names:
        any_bad_contact |= _sensor_contact_indicator(env, sensor_name, force_threshold)  # OR 聚合所有 non-tip sensors

    # 返回正值 indicator，外部配置负 weight 后成为惩罚
    penalty = any_bad_contact.float()  # `[B]`，$r_{bc}$

    # curriculum 默认严格 dead-zone：global lambda 为 0 时，该惩罚完全关闭
    if use_curriculum:
        penalty = penalty * _curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)  # $\lambda_{bc} r_{bc}$
    return penalty


# ==================
# $r_{stable}$
# ==================


def action_l2_curriculum(
    env: ManagerBasedRLEnv,
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Curriculum-gated action L2 regularizer。

    本项目动作项 `ClampedRelativeJointPositionAction` 已将每步 raw rad delta 通过
    `scale=0.1` 约束在温和范围内，并在下发前 clamp 到 soft joint limits。因而
    第一版可严格模仿 AnyRotate，把 action 正则也放到 adaptive curriculum 后释放。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: gated action L2 penalty source，形状 `[num_envs]`；外部配置负权重。
    """

    return isaac_mdp.action_l2(env) * _curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)


def action_rate_l2_curriculum(
    env: ManagerBasedRLEnv,
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Curriculum-gated action-rate L2 regularizer。

    该项惩罚相邻 policy action 的变化率，主要用于抑制高频抖动。由于相对
    增量动作本身已有限幅，默认也放入 curriculum；若训练早期仍出现动作爆炸，
    可单独把本项 `lambda_floor` 调到 $0.02\sim0.1$。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: gated action-rate L2 penalty source，形状 `[num_envs]`。
    """

    return isaac_mdp.action_rate_l2(env) * _curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)


def torque_l2_curriculum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_floor: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    r"""Curriculum-gated torque L2 penalty source。

    对齐 AnyRotate 的 torque penalty：

    $$
    r_{torque} = \|\tau\|_2^2
    $$

    本函数返回正值 penalty source，实际惩罚由 `RewardsCfg` 负权重实现。
    若当前 actuator backend 没有暴露 `computed_torque`，返回 0 并保留接口。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        asset_cfg (SceneEntityCfg): robot articulation 配置。
        lambda_floor (float): curriculum 早期下限，默认 0.0。
        lambda_max (float): curriculum 完全释放后的上限，默认 1.0。

    Returns:
        torch.Tensor: gated torque L2 penalty source，形状 `[num_envs]`。
    """

    # 读取 articulation 的 controller torque；不同 actuator backend 可能不存在该字段
    asset: Articulation = env.scene[asset_cfg.name]  # robot articulation
    torque = getattr(asset.data, "computed_torque", None)  # `[B,d]`，控制器计算力矩，单位 N·m
    if torque is None:
        return torch.zeros(env.num_envs, device=env.device)  # 保留接口，不因 backend 差异中断脚手架

    # 对所有动作关节求平方和；若未来只惩罚 actuated joint，应通过 SceneEntityCfg joint_ids 细化
    penalty = torch.sum(torque**2, dim=-1)  # `[B]`，$\|\tau\|_2^2$
    return penalty * _curriculum_gain(env, lambda_floor=lambda_floor, lambda_max=lambda_max)


# ==================
# $r_{terminate}$
# ==================


def termination_penalty_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""Termination penalty 占位项。

    AnyRotate 将 object falling / rotation-axis deviation 等终止条件写成
    `r_terminate`。在 Isaac Lab ManagerBasedRLEnv 中，更干净的做法通常是：
    终止逻辑放在 `terminations.py`，reward 侧只在需要时读取 termination term
    的 bool indicator 并乘负权重。

    TODO:
        等 `gm/mdp/terminations.py` 的掉落、离手、axis deviation 判据稳定后，
        再决定是否需要显式 penalty reward；不要提前把 termination 语义复制
        到 reward 里形成双源漂移。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。

    Returns:
        torch.Tensor: 全零 penalty source，形状 `[num_envs]`。
    """

    return torch.zeros(env.num_envs, device=env.device)  # 当前仅占位，不改变训练语义


__all__ = [
    "AxisDeltaRotationReward",
    "action_l2_curriculum",
    "action_rate_l2_curriculum",
    "bad_non_tip_contact",
    "goal_success_bonus",
    "good_fingertip_contact",
    "keypoint_reorientation_reward",
    "reorientation_reward_placeholder",
    "termination_penalty_placeholder",
    "torque_l2_curriculum",
]
