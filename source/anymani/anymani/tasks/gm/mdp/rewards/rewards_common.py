r"""Shared reward helpers for GM in-hand manipulation.

这些 helper 承载 reward 项之间共享的 command buffer 解析、orientation keypoint 距离、
以及 adaptive reward curriculum 系数。它们不是外部 MDP API，但需要集中维护，避免
reorient/contact/stable 子模块各自复制 command / curriculum 语义。

TODO(full-pose keypoint primitive):
    tactile rotation baseline 的稠密 pose reward 需要在现有 orientation-only helper 之外增加
    full-pose 六轴 keypoint distance：

    $$
    x_i^{\{h\}}
    =
    \left\|
    p_o^{\{h\}}+R_{ho}r_i^{\{o\}}-
    \left(p_{anchor}^{\{h\}}+R_{hg}r_i^{\{o\}}\right)
    \right\|_2.
    $$

    normalized kernel 为：

    $$
    r_{kp}
    =
    \frac{1}{6}
    \sum_i
    \frac{4}
    {\exp(50x_i^{\{h\}})+2+\exp(-50x_i^{\{h\}})}.
    $$

    full-pose helper 服务 dense reward；orientation-only helper 服务 success orientation gate。
    二者不得合并成一个含糊的 `keypoint_distance`。

TODO(reward time semantics):
    helper 应显式区分持续 rate 与离散 impulse。RewardManager 会乘 `step_dt`：连续状态项直接
    返回 rate；delta-rotation 返回 clipped angle 除以 `step_dt`；goal / termination impulse
    返回 indicator 除以 `step_dt`。禁止在新组合式 reward 外层整体抵消 `dt`。
"""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv


def resolve_goal_quat_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""从 command term 中解析目标姿态 $R_g$ 的四元数表达。

    `gm` 的 command 观测可以是 axis + error-so(3) 的 6D 张量，但 reward 计算 keypoint
    distance / SO(3) success 时需要内部目标姿态。因此目标姿态应由 command term 以 buffer
    形式显式暴露，而不是让 reward 从 6D command 中反推。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): command manager 中的 command term 名称。

    Returns:
        torch.Tensor: 目标姿态四元数，形状 `[num_envs, 4]`，约定为 Isaac Lab 的 `(w,x,y,z)`。

    Raises:
        RuntimeError: 当 command term 没有显式暴露目标四元数时抛出。
    """

    # 优先读取 command term 的内部 buffer：这是 `gm` ReorientCommand 应兑现的契约。
    command_term = env.command_manager.get_term(command_name)
    for attr_name in ("goal_quat_w", "quat_command_w"):
        goal_quat_w = getattr(command_term, attr_name, None)  # `[B,4]`，目标姿态 quaternion
        if isinstance(goal_quat_w, torch.Tensor):
            return goal_quat_w

    raise RuntimeError(
        f"Command '{command_name}' must expose `goal_quat_w` / `quat_command_w` as canonical goal state. "
        "GM reward terms intentionally do not infer goal quaternion from the policy-facing command tensor, "
        "because `command_output` may omit quaternion or reorder fields."
    )


def resolve_axis_e(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""从 command term 中解析空间旋转轴 $\hat\omega^{\{e\}}$。

    本项目的 command 语义锚定在 hand semantic frame `{h}`。但 reward 计算实际姿态
    增量 $\log(R_t R_{t-1}^{-1})$ 时使用世界 / 环境系旋转矩阵，因此投影轴必须已经由
    command term 转到 `{e}` 或 `{w}`。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        command_name (str): command manager 中的 command term 名称。

    Returns:
        torch.Tensor: 单位旋转轴，形状 `[num_envs, 3]`，坐标系为 `{e}` / `{w}`。

    Raises:
        RuntimeError: 当 command term 没有暴露空间轴时抛出。
    """

    # reward 不应自行猜 `{h}->{e}` 对齐；该变换属于 command term 的职责。
    command_term = env.command_manager.get_term(command_name)
    for attr_name in ("axis_e", "axis_w", "axis_command_e", "axis_command_w"):
        axis_e = getattr(command_term, attr_name, None)  # `[B,3]`，空间旋转轴
        if isinstance(axis_e, torch.Tensor):
            return axis_e / (torch.linalg.norm(axis_e, dim=-1, keepdim=True) + 1.0e-6)

    raise RuntimeError(
        f"Command '{command_name}' must expose `axis_e` / `axis_w` for axis-progress reward. "
        "Reward terms intentionally do not infer `{h}->{e}` alignment."
    )


def six_axis_keypoints_o(device: torch.device | str, radius: float) -> torch.Tensor:
    r"""生成 AnyRotate 风格的六轴向 object keypoints。

    六个点定义在 object body frame `{o}` 中：
    $$
    \mathcal{P}_o = \{\pm r\mathbf{e}_x,\ \pm r\mathbf{e}_y,\ \pm r\mathbf{e}_z\}.
    $$

    第一版只用姿态差异：比较 $R_o p_i$ 与 $R_g p_i$，不加 object center，
    因此不会把物体在手内的小幅平移直接惩罚进姿态 reward。

    Args:
        device (torch.device | str): 输出张量所在设备。
        radius (float): keypoint 半径，单位 m；AnyRotate 使用 $5\,\text{cm}$。

    Returns:
        torch.Tensor: keypoints，形状 `[6, 3]`，坐标系 `{o}`。
    """

    # 这里每次构造的成本极小；若后续 object mesh keypoints 增多，可缓存为 scene buffer。
    return torch.tensor(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=torch.float32,
        device=device,
    )


def orientation_keypoint_distance(
    current_quat_w: torch.Tensor,
    goal_quat_w: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    r"""计算 orientation-only keypoint distance。

    对每个 object-local keypoint $p_i^{\{o\}}$，只比较旋转后的方向：
    $$
    d_{kp} = \frac{1}{N}\sum_{i=1}^{N}
    \left\| R_o p_i^{\{o\}} - R_g p_i^{\{o\}} \right\|_2.
    $$

    注意这里故意不加 object center $x_o$ / goal center $x_g$，因此该项只塑造姿态误差；
    位置保持、掉落、离手等语义应由独立 term 处理。

    Args:
        current_quat_w (torch.Tensor): 当前物体姿态，形状 `[B,4]`，`(w,x,y,z)`。
        goal_quat_w (torch.Tensor): 目标物体姿态，形状 `[B,4]`，`(w,x,y,z)`。
        radius (float): keypoint 半径，单位 m。

    Returns:
        torch.Tensor: 平均 keypoint distance，形状 `[B]`，单位 m。
    """

    # quaternion → rotation matrix，矩阵语义为 $R\in SO(3)$，形状 `[B,3,3]`。
    current_rot_w = math_utils.matrix_from_quat(current_quat_w)  # 当前物体姿态 $R_o$
    goal_rot_w = math_utils.matrix_from_quat(goal_quat_w)  # 目标物体姿态 $R_g$
    keypoints_o = six_axis_keypoints_o(device=current_quat_w.device, radius=radius)  # `{o}` 下六个轴向点

    # 左乘旋转矩阵得到 orientation-only keypoints，形状 `[B,6,3]`。
    current_points = torch.einsum("bij,kj->bki", current_rot_w, keypoints_o)  # $R_o p_i$
    goal_points = torch.einsum("bij,kj->bki", goal_rot_w, keypoints_o)  # $R_g p_i$
    return torch.linalg.norm(current_points - goal_points, dim=-1).mean(dim=-1)  # $d_{kp}$，形状 `[B]`


def full_pose_keypoint_distances(
    current_pos_w: torch.Tensor,
    current_quat_w: torch.Tensor,
    goal_pos_w: torch.Tensor,
    goal_quat_w: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    r"""计算六个 object-local keypoints 的 full-pose 逐点距离。

    虽然任务公式写在 `{h}`，同一刚体旋转同时作用于 current/goal points 时欧氏距离不变，
    因此 runtime 可直接在 `{w}` 计算，避免重复 hand-frame 变换：

    $$
    x_i=\left\|p_o^w+R_{wo}r_i^o-(p_{anchor}^w+R_{wg}r_i^o)\right\|_2.
    $$

    Returns:
        torch.Tensor: 六个逐点距离 `[B,6]`，单位 m；不在这里提前平均。
    """

    keypoints_o = six_axis_keypoints_o(current_quat_w.device, radius).to(dtype=current_quat_w.dtype)
    current_rot_w = math_utils.matrix_from_quat(current_quat_w)
    goal_rot_w = math_utils.matrix_from_quat(goal_quat_w)
    current_points_w = current_pos_w[:, None, :] + torch.einsum("bij,kj->bki", current_rot_w, keypoints_o)
    goal_points_w = goal_pos_w[:, None, :] + torch.einsum("bij,kj->bki", goal_rot_w, keypoints_o)
    return torch.linalg.norm(current_points_w - goal_points_w, dim=-1)  # `[B,6]`


def normalized_keypoint_kernel(
    distances: torch.Tensor,
    curve_sharpness: float = 50.0,
    curve_bias: float = 2.0,
) -> torch.Tensor:
    r"""对逐 keypoint 距离应用归一化 AnyRotate logistic kernel 后求均值。

    $$
    r_{kp}=\frac1K\sum_i\frac{2+b}{\exp(ax_i)+b+\exp(-ax_i)}.
    $$

    baseline $b=2$，所以 numerator 为 4，确保所有距离为零时 reward 精确为 1。
    """

    x = torch.clamp(float(curve_sharpness) * distances, min=0.0, max=30.0)  # 防止大距离 exp overflow
    per_keypoint = (2.0 + float(curve_bias)) / (torch.exp(x) + float(curve_bias) + torch.exp(-x))
    return per_keypoint.mean(dim=-1)  # `[B]`，$(0,1]`


def impulse_to_rate(value: torch.Tensor, step_dt: float) -> torch.Tensor:
    r"""把离散 step impulse 转成 RewardManager 可积分的 rate。

    RewardManager 最终乘 `step_dt`，所以返回 $I/\Delta t$ 可使一次事件的 episode integral
    恒为 $I$，不随 20/30 Hz policy frequency 改变。
    """

    if float(step_dt) <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}.")
    return value / float(step_dt)


def curriculum_gain(
    env: ManagerBasedRLEnv,
    lambda_floor: float,
    lambda_max: float = 1.0,
    lambda_attr_name: str = "_gm_reward_curriculum_lambda",
    default_lambda: float = 1.0,
) -> torch.Tensor:
    r"""读取全局 reward curriculum 系数，并映射为某个 reward term 的门控系数。

    全局 curriculum term 负责维护标量 $\lambda_{global}\in[0,1]$，表示“策略整体完成
    重定向子目标的成熟度”。每个 reward 项再通过自己的 `lambda_floor` 决定早期是否
    保留塑形信号：
    $$
    \lambda_i = \lambda_{floor,i} + (\lambda_{max,i}-\lambda_{floor,i})\lambda_{global}.
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

    # 若没有显式 curriculum，默认不让 reward 静默消失；训练 cfg 可通过启用 curriculum 覆盖该行为。
    global_lambda = getattr(env, lambda_attr_name, None)
    if global_lambda is None:
        global_lambda = torch.tensor(default_lambda, device=env.device, dtype=torch.float32)
    elif not isinstance(global_lambda, torch.Tensor):
        global_lambda = torch.tensor(float(global_lambda), device=env.device, dtype=torch.float32)
    else:
        global_lambda = global_lambda.to(device=env.device, dtype=torch.float32)

    global_lambda = torch.clamp(global_lambda, 0.0, 1.0)  # $\lambda_{global}\in[0,1]$
    gain = float(lambda_floor) + (float(lambda_max) - float(lambda_floor)) * global_lambda  # $\lambda_i$
    if gain.ndim == 0:
        gain = gain.expand(env.num_envs)  # `[B]`，所有 env 共用同一 release 系数
    return gain


__all__ = [
    "curriculum_gain",
    "full_pose_keypoint_distances",
    "impulse_to_rate",
    "normalized_keypoint_kernel",
    "orientation_keypoint_distance",
    "resolve_axis_e",
    "resolve_goal_quat_w",
    "six_axis_keypoints_o",
]
