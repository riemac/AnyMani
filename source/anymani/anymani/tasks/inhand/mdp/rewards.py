r"""Official LEAP reward family for AnyMani in-hand tasks.

本文件收敛当前 inhand 官方主线真正仍在使用的 reward 语义，只处理 official LEAP
reorientation / ADR 分支，不再混入历史 task reward、动作正则 reward 或 tactile reward。

设计目标：

1. **保留一个 combined official reward**，用于 N010/N020/N030 以及 N031 dt-ablation：
   $$
   r_t^{official}=
   -10\lVert p_o^e-p_g^e\rVert_2
   +\frac{1}{|\theta_t|+0.1}
   -0.0002\lVert a_t^{exec}\rVert_2^2
   -0.3\lVert q_t^{cmd}-q^{pregrasp}\rVert_2^2
   +250\mathbf{1}_{success}
   -10\mathbf{1}_{fall}
   +\mathbf{1}_{0.25<\omega_z<1.5}.
   $$
2. **同时提供一事一议的拆分项**，使 reward cfg 可以既写成一个 combined term，也能按子项重组；
3. **只在 combined reward 中提供 `divide_by_step_dt` 切换**：
   - `True`：返回 $r_t^{official}/\Delta t$，再由 ManagerBased `RewardManager` 乘回 $\Delta t$，
     恢复 DirectRLEnv 的单步 reward 数值；
   - `False`：直接返回 $r_t^{official}$，因此真正进入 PPO 的是 $\Delta t\cdot r_t^{official}$。

拆分项的默认约定：
    每个拆分项都返回 **direct-step semantic** 的值，即内部固定除以 `env.step_dt`。这样如果用
    与 official 相同的权重把各项相加，数值语义与 combined `divide_by_step_dt=True` 一致。

NOTE:
    本文件当前不保留 torque 正则项。官方 reward 的 `torque_penalty_scale` 在主线配置中就是 0，
    且用户明确要求本轮只保留七个真正要讨论和重组的 official 子项。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_goal_pose_from_command_term(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""解析当前 reward 语义所需的 goal pose `(pos_e, quat_w)`。

    当前 official reward 主线既要兼容 `OfficialContinuousRotationCommand` 这种内部维护
    `pos_command_e` / `quat_command_w` buffer 的命令项，也要兼容历史上直接返回 7D pose-like
    command tensor 的旧接口。因此统一采用：

    1. 优先读取 command term 的内部 buffer；
    2. 否则回退到 `(pos_e, quat_w)` 的 legacy command tensor。

    Args:
        env (ManagerBasedRLEnv): 运行时环境。
        command_name (str): CommandManager 中的命令项名称，官方主线默认为 `goal_pose`。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - `goal_pos_e`：环境坐标系 `{e}` 下的目标位置，形状 `[N, 3]`；
            - `goal_quat_w`：世界坐标系 `{w}` 下的目标四元数，形状 `[N, 4]`。
    """

    term = env.command_manager.get_term(command_name)  # 命令项对象，本体缓存目标位置/姿态。
    goal_pos_e = getattr(term, "pos_command_e", None)  # 期望形状 `[N,3]`。
    goal_quat_w = getattr(term, "quat_command_w", None)  # 期望形状 `[N,4]`，wxyz。

    if isinstance(goal_pos_e, torch.Tensor) and isinstance(goal_quat_w, torch.Tensor):
        return goal_pos_e, goal_quat_w  # 当前 official command 主线的标准接口。

    cmd = env.command_manager.get_command(command_name)  # legacy pose-like command fallback。
    if not (isinstance(cmd, torch.Tensor) and cmd.shape[-1] >= 7):
        raise RuntimeError(
            f"Cannot resolve goal pose from command '{command_name}'. Expected term buffers or a pose-like tensor, "
            f"got: {type(cmd)} {getattr(cmd, 'shape', None)}"
        )
    return cmd[:, :3], cmd[:, -4:]  # 前三维位置，最后四维四元数。


def _to_manager_term_scale(env: ManagerBasedRLEnv, value: torch.Tensor) -> torch.Tensor:
    r"""把 DirectRLEnv step reward 量缩放成 ManagerBased reward term 返回值。

    IsaacLab `RewardManager` 会自动乘上 `env.step_dt`。因此若我们希望最终进入 PPO 的值等于
    DirectRLEnv 的 **每个 policy step reward**，当前 term 必须返回：

    $$
    \frac{r_t^{direct}}{\Delta t}.
    $$

    这里把该变换封装成 helper，避免每个拆分项都重复写。
    """

    return value / float(env.step_dt)  # 抵消 RewardManager 后续自动乘上的 $\Delta t$。


def official_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    r"""Official 位置项：$r_{pos}=\lVert p_o^e-p_g^e\rVert_2$。

    注意该函数只返回 **未带系数的正量距离**。在 reward cfg 中应显式使用官方权重 `-10.0`：

    $$
    -10\cdot r_{pos}.
    $$
    """

    object_asset: RigidObject = env.scene[object_cfg.name]  # 当前物体刚体。
    goal_pos_e, _ = _resolve_goal_pose_from_command_term(env, command_name)  # reward 所需目标位置。
    object_pos_e = object_asset.data.root_pos_w - env.scene.env_origins  # 变到环境系 `{e}`。
    goal_dist = torch.norm(object_pos_e - goal_pos_e, p=2, dim=-1)  # $\lVert p_o^e-p_g^e\rVert_2$。
    return _to_manager_term_scale(env, goal_dist)  # direct-step semantic 的位置距离。


def official_orientation(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 0.1,
) -> torch.Tensor:
    r"""Official 姿态项：$r_{rot}=1/(|\theta|+\varepsilon)$。"""

    object_asset: RigidObject = env.scene[object_cfg.name]  # 当前物体刚体。
    _, goal_quat_w = _resolve_goal_pose_from_command_term(env, command_name)  # reward 所需目标姿态。
    rot_dist = math_utils.quat_error_magnitude(goal_quat_w, object_asset.data.root_quat_w)  # $d_{SO(3)}$。
    reward = 1.0 / (torch.abs(rot_dist) + float(rot_eps))  # $1/(|\theta|+0.1)$。
    return _to_manager_term_scale(env, reward)  # 返回 direct-step semantic 的姿态奖励。


def official_action_l2(
    env: ManagerBasedRLEnv,
    action_term_name: str = "hand_joint_pos",
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    r"""Official 动作正则项：$r_{act}=\lVert a_t^{exec}\rVert_2^2$。"""

    action_term = env.action_manager.get_term(action_term_name)  # 动作项；需暴露 `executed_actions`。
    penalty = torch.sum(action_term.executed_actions**2, dim=-1)  # $\|a_t^{exec}\|_2^2$，形状 `[N]`。
    return _to_manager_term_scale(env, penalty)  # 交给 cfg 用官方负权重缩放。


def official_pregrasp_l2(
    env: ManagerBasedRLEnv,
    action_term_name: str = "hand_joint_pos",
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    r"""Official pregrasp 偏离项：$r_{pre}=\lVert q_t^{cmd}-q^{pregrasp}\rVert_2^2$。"""

    action_term = env.action_manager.get_term(action_term_name)  # 动作项；需暴露 current/pregrasp targets。
    pregrasp_l2 = torch.sum((action_term.current_targets - action_term.pregrasp_targets) ** 2, dim=-1)  # $\|q^{cmd}-q^{pre}\|_2^2$。
    return _to_manager_term_scale(env, pregrasp_l2)  # direct-step semantic 的 pregrasp 偏离项。


def official_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    success_tolerance: float = 0.2,
    position_success_threshold: float = 0.025,
) -> torch.Tensor:
    r"""Official 成功稀疏项：$r_{succ}=\mathbf{1}_{success}$。"""

    object_asset: RigidObject = env.scene[object_cfg.name]  # 当前物体刚体。
    goal_pos_e, goal_quat_w = _resolve_goal_pose_from_command_term(env, command_name)  # 同时需要目标位置和姿态。
    object_pos_e = object_asset.data.root_pos_w - env.scene.env_origins  # 当前物体环境系位置。
    goal_dist = torch.norm(object_pos_e - goal_pos_e, p=2, dim=-1)  # 位置误差。
    rot_dist = math_utils.quat_error_magnitude(goal_quat_w, object_asset.data.root_quat_w)  # 姿态误差。
    success = (rot_dist <= float(success_tolerance)) & (goal_dist <= float(position_success_threshold))  # 小目标成功判据。
    return _to_manager_term_scale(env, success.float())  # 布尔成功 mask 转成 0/1 再对齐 direct-step 量纲。


def official_fall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    fall_dist: float = 0.07,
) -> torch.Tensor:
    r"""Official 掉落项：$r_{fall}=\mathbf{1}_{\lVert p_o^e-p_g^e\rVert_2\ge d_{fall}}$。"""

    object_asset: RigidObject = env.scene[object_cfg.name]  # 当前物体刚体。
    goal_pos_e, _ = _resolve_goal_pose_from_command_term(env, command_name)  # 掉落项只依赖目标位置。
    object_pos_e = object_asset.data.root_pos_w - env.scene.env_origins  # 当前物体环境系位置。
    goal_dist = torch.norm(object_pos_e - goal_pos_e, p=2, dim=-1)  # 与目标位置的环境系距离。
    fall = goal_dist >= float(fall_dist)  # 大于 7 cm 时触发一次掉落惩罚事件。
    return _to_manager_term_scale(env, fall.float())  # 返回 0/1 事件，由 cfg 施加 -10 权重。


def official_z_spin_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lower: float = 0.25,
    upper: float = 1.5,
) -> torch.Tensor:
    r"""Official z-spin 项：$r_{spin}=\mathbf{1}_{0.25<\omega_z<1.5}$。"""

    object_asset: RigidObject = env.scene[object_cfg.name]  # 当前物体刚体。
    object_angvel_z = object_asset.data.root_ang_vel_w[:, 2]  # 世界系 z 轴角速度 $\omega_z$。
    bonus = (object_angvel_z > float(lower)) & (object_angvel_z < float(upper))  # 处于官方奖励窗口内。
    return _to_manager_term_scale(env, bonus.float())  # 返回 0/1，cfg 中通常用 +1 权重。


class OfficialLeapReward(ManagerTermBase):
    r"""Combined official LEAP reward with optional `dt` alignment switch.

    本类服务两条用途：

    1. **N010/N020/N030 官方主线**：`divide_by_step_dt=True`，使 ManagerBased reward 数值与
       DirectRLEnv 单步 reward 对齐；
    2. **N031 dt-ablation**：`divide_by_step_dt=False`，只改变 combined reward 的数值尺度，
       不改变子项构成。

    注意：拆分子项始终固定为 direct-step semantic；只有 combined reward 在这里暴露 dt 切换。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._action_term_name = cfg.params.get("action_term_name", "hand_joint_pos")  # official action term 名称。
        self._command_name = cfg.params.get("command_name", "goal_pose")  # official command term 名称。
        self._object_cfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))  # object selector。
        self._dist_reward_scale = float(cfg.params.get("dist_reward_scale", -10.0))  # 位置项系数。
        self._rot_reward_scale = float(cfg.params.get("rot_reward_scale", 1.0))  # 姿态项系数。
        self._rot_eps = float(cfg.params.get("rot_eps", 0.1))  # 姿态项防止除零的小常数。
        self._action_penalty_scale = float(cfg.params.get("action_penalty_scale", -0.0002))  # 动作正则系数。
        self._pose_diff_penalty_scale = float(cfg.params.get("pose_diff_penalty_scale", -0.3))  # pregrasp 偏离系数。
        self._success_tolerance = float(cfg.params.get("success_tolerance", 0.2))  # 小目标姿态成功阈值。
        self._position_success_threshold = float(
            cfg.params.get("position_success_threshold", 0.025)
        )  # 小目标位置成功阈值。
        self._reach_goal_bonus = float(cfg.params.get("reach_goal_bonus", 250.0))  # 成功稀疏奖励。
        self._fall_dist = float(cfg.params.get("fall_dist", 0.07))  # 掉落阈值，单位 m。
        self._fall_penalty = float(cfg.params.get("fall_penalty", -10.0))  # 掉落惩罚系数。
        self._z_rotation_steps = int(cfg.params.get("z_rotation_steps", 16))  # 16 个小目标为一整圈。
        self._divide_by_step_dt = bool(cfg.params.get("divide_by_step_dt", True))  # N030/N031 的唯一实验开关。

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_term_name: str = "hand_joint_pos",
        command_name: str = "goal_pose",
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        dist_reward_scale: float = -10.0,
        rot_reward_scale: float = 1.0,
        rot_eps: float = 0.1,
        action_penalty_scale: float = -0.0002,
        pose_diff_penalty_scale: float = -0.3,
        success_tolerance: float = 0.2,
        position_success_threshold: float = 0.025,
        reach_goal_bonus: float = 250.0,
        fall_dist: float = 0.07,
        fall_penalty: float = -10.0,
        z_rotation_steps: int = 16,
        divide_by_step_dt: bool = True,
    ) -> torch.Tensor:
        r"""计算当前一步的 combined official reward。

        Returns:
            torch.Tensor: 当 `divide_by_step_dt=True` 时返回 $r_t^{official}/\Delta t$；否则返回
            $r_t^{official}$，后续仍会被 `RewardManager` 乘上 `\Delta t`。
        """

        object_asset: RigidObject = env.scene[object_cfg.name]  # 当前物体刚体。
        action_term = env.action_manager.get_term(action_term_name)  # 动作项；需暴露 executed/current/pregrasp。
        goal_pos_e, goal_quat_w = _resolve_goal_pose_from_command_term(env, command_name)  # 当前 reward 所需目标位姿。

        object_pos_e = object_asset.data.root_pos_w - env.scene.env_origins  # 把世界系位置变到环境系 `{e}`。
        goal_dist = torch.norm(object_pos_e - goal_pos_e, p=2, dim=-1)  # $\|p_o^e-p_g^e\|_2$。
        rot_dist = math_utils.quat_error_magnitude(goal_quat_w, object_asset.data.root_quat_w)  # $d_{SO(3)}$。
        pregrasp_l2 = torch.sum((action_term.current_targets - action_term.pregrasp_targets) ** 2, dim=-1)  # $\|q^{cmd}-q^{pre}\|_2^2$。
        object_angvel_z = object_asset.data.root_ang_vel_w[:, 2]  # 世界系 z 轴角速度 $\omega_z$。

        dist_term = goal_dist * float(dist_reward_scale)  # $-10\|p_o^e-p_g^e\|_2$。
        rot_term = float(rot_reward_scale) / (torch.abs(rot_dist) + float(rot_eps))  # $1/(|\theta|+0.1)$。
        action_term_penalty = torch.sum(action_term.executed_actions**2, dim=-1) * float(action_penalty_scale)  # $-0.0002\|a\|_2^2$。
        pregrasp_term = pregrasp_l2 * float(pose_diff_penalty_scale)  # $-0.3\|q^{cmd}-q^{pre}\|_2^2$。

        reward = dist_term + rot_term + action_term_penalty + pregrasp_term  # 稠密基础项之和。

        success_mask = (rot_dist <= float(success_tolerance)) & (
            goal_dist <= float(position_success_threshold)
        )  # 当前小目标是否完成。
        reward = torch.where(success_mask, reward + float(reach_goal_bonus), reward)  # 成功后加 250。

        z_spin_bonus = (object_angvel_z > 0.25) & (object_angvel_z < 1.5)  # 官方 z-spin 窗口。
        reward = torch.where(z_spin_bonus, reward + 1.0, reward)  # 进入窗口时加 1。

        fall_mask = goal_dist >= float(fall_dist)  # 与 goal position 漂离超过 7 cm。
        reward = torch.where(fall_mask, reward + float(fall_penalty), reward)  # 触发一次掉落惩罚。

        if bool(divide_by_step_dt):
            return reward / float(self._env.step_dt)  # N030 / official parity：恢复 DirectRLEnv 单步 reward 数值。
        return reward  # N031 ablation：直接让 RewardManager 后续乘上 $\Delta t$。

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""写回 official 训练时依赖的 reset-time 诊断日志。"""

        if env_ids is None:
            env_ids = slice(None)  # 全部环境 reset 时的 IsaacLab 约定。

        object_asset: RigidObject = self._env.scene[self._object_cfg.name]  # 当前 object 刚体，用于线速度/角速度日志。
        command_term = self._env.command_manager.get_term(self._command_name)  # 读取连续小目标成功数。
        action_term = self._env.action_manager.get_term(self._action_term_name)  # 当前动作项，用于读取 current/pregrasp target。
        pregrasp_l2 = torch.sum((action_term.current_targets - action_term.pregrasp_targets) ** 2, dim=-1)  # reset 前最后一步的 pregrasp 偏离量。

        log = getattr(self._env, "extras", {}).get("log")  # rl_games observer 消费的 reset-time 标量字典。
        if isinstance(log, dict):
            log["consecutive_successes"] = torch.mean(command_term.success_counter[env_ids]).item() / float(
                self._z_rotation_steps
            )  # 以 rotations 而非小目标数记录成功统计。
            log["pose_diff_penalty"] = torch.mean(pregrasp_l2[env_ids]).item()  # 记录 pregrasp 偏离量。
            log["object_linvel"] = torch.norm(object_asset.data.root_lin_vel_w[env_ids], p=1, dim=-1).mean().item()
            log["roll"] = object_asset.data.root_ang_vel_w[env_ids, 0].mean().item()  # 世界系 x 轴角速度均值。
            log["pitch"] = object_asset.data.root_ang_vel_w[env_ids, 1].mean().item()  # 世界系 y 轴角速度均值。
            log["yaw"] = object_asset.data.root_ang_vel_w[env_ids, 2].mean().item()  # 世界系 z 轴角速度均值。
            log["num_adr_increases"] = float(getattr(self._env, "leap_adr_increment", 0))  # 当前 ADR 档位增量计数。
            log["adr_criteria"] = float(getattr(self._env, "leap_adr_criteria", 0.0))  # 当前 rotations/sec 升级判据。
            if hasattr(self._env, "leap_adr_episode_lengths"):
                lengths_s = self._env.leap_adr_episode_lengths.float() * float(self._env.step_dt)  # per-env horizon 转秒数。
                log["avg_episode_length_s"] = lengths_s.mean().item()  # ADR 随机 horizon 的平均秒数。
                log["min_episode_length_s"] = lengths_s.min().item()  # 最短 horizon。
                log["max_episode_length_s"] = lengths_s.max().item()  # 最长 horizon。


__all__ = [
    "OfficialLeapReward",
    "official_action_l2",
    "official_fall_penalty",
    "official_goal_distance",
    "official_orientation",
    "official_pregrasp_l2",
    "official_success_bonus",
    "official_z_spin_bonus",
]
