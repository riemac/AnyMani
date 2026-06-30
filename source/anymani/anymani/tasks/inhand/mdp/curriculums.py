# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的课程学习函数
里面的步数指的是全局步数，所有环境累计交互的次数
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def leap_adr_interpolate(initial_value, final_value, frac: float):
    r"""按 LEAP ADR 档位比例线性插值标量或嵌套结构。

    LEAP 官方 ADR 的核心不是按 wall-clock time 硬拉满随机化，而是在全局档位
    $k\in\{0,\dots,N\}$ 上对每个随机化上限做线性课程：
    $$
    x_k = x_0 + \frac{k}{N}(x_N-x_0).
    $$

    Args:
        initial_value: ADR 第 0 档的值，可以是标量、tuple/list 或嵌套 dict。
        final_value: ADR 第 $N$ 档的值，结构必须与 ``initial_value`` 一致。
        frac: 当前档位比例 $k/N$，理论范围 $[0,1]$。

    Returns:
        与 ``initial_value`` 同结构的插值结果。
    """

    # dict 递归用于 material range 这类结构化参数，例如 {"static": (...), "dynamic": (...)}。
    if isinstance(initial_value, dict):
        return {key: leap_adr_interpolate(initial_value[key], final_value[key], frac) for key in initial_value}

    # tuple/list 递归保留容器类型，使 EventTerm 参数仍保持 IsaacLab 原本期望的 tuple/list 语义。
    if isinstance(initial_value, (tuple, list)):
        return type(initial_value)(
            leap_adr_interpolate(init_item, final_item, frac)
            for init_item, final_item in zip(initial_value, final_value, strict=True)
        )

    # 标量叶子节点执行公式 $x_k=x_0+\lambda(x_N-x_0)$。
    return float(initial_value) + float(frac) * (float(final_value) - float(initial_value))


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice) -> torch.Tensor:
    r"""把 CurriculumManager 传入的 env ids 统一成 GPU tensor。"""

    # ManagerBasedRLEnv 在全环境 reset 时可能传入 slice(None)，这里转成显式索引便于 gather。
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)

    # 普通 reset 分支通常传入 tensor；as_tensor 保证 list/tuple 也能安全处理。
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


class LeapADRGlobalScheduler(ManagerTermBase):
    r"""LEAP 官方风格的全局 ADR 档位调度器。

    本类用于 ManagerBasedRLEnv 的 ``CurriculumManager``，但保留官方 DirectRLEnv
    的科研语义：所有并行环境共享一个全局 ADR 档位 $k$，不是 DexSuite 那种 per-env
    difficulty。ADR 更新只发生在 reset hook 中，因此下面用 $\mathcal{E}_t$ 表示
    “第 $t$ 次 ADR 更新时刚结束 episode 的环境集合”。

    符号解释：
        - $t$：ADR scheduler 的 reset-hook 更新编号，不是 physics step，也不是 PPO iteration。
        - $i\in\mathcal{E}_t$：本次进入 reset 流程的第 $i$ 个并行环境。
        - $S_i$：环境 $i$ 刚结束 episode 内完成的连续小目标数；一个小目标对应
          $\Delta\theta=\pi/8$ 的 z 轴旋转命令推进。
        - $C_t$：全局 EMA success counter，不是 reward；它把不同 reset batch 的 $S_i$ 平滑成
          一个共享的课程进度信号。
        - $\alpha=0.1$：官方 EMA 更新率，约等价于保留最近若干 reset batch 的低通统计。

    每次 reset hook 先读取刚结束 episode 的 $S_i$，再更新全局 EMA 指标：
    $$
    C_{t+1}=\alpha\frac{1}{|\mathcal{E}_t|}\sum_{i\in\mathcal{E}_t}S_i+(1-\alpha)C_t.
    $$

    随后把“连续小目标数”换算成“整圈旋转速度”作为升级判据：
    $$
    \mathrm{ADRScore}=\frac{C_t/z_\mathrm{steps}}{\bar T}\ge 0.15.
    $$

    其中：
        - $z_\mathrm{steps}=16$：16 个 $\pi/8$ 小目标等于一整圈 $2\pi$。
        - $\bar T$：$\mathcal{E}_t$ 中这些 episode 的 randomized horizon 平均秒数，用于把
          success count 归一化为 rotations/sec。
        - $0.15$：官方 ADR 升级阈值，量纲是 rotations/sec。
        - $k\in\{0,\dots,25\}$：全局 ADR 档位；升级后执行 $k\leftarrow k+1$，并将
          $C_t\leftarrow0$，让策略必须在新难度下重新证明自己。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        r"""初始化全局 ADR 运行态。"""

        super().__init__(cfg, env)
        self.increment = int(cfg.params.get("starting_increment", 0))  # 全局档位 $k$。
        self.ema_success = torch.tensor(0.0, device=env.device)  # $C_t$，全局 episode success EMA。
        self.adr_criteria = torch.tensor(0.0, device=env.device)  # rotations/sec，日志与升级判据共用。
        self.reset_checks_since_increase = 0  # 官方 reset-hook cooldown 计数，不是严格 policy step。
        self._publish_state(env)  # 初始化 env.leap_adr_*，供 events/action term 读取。

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "goal_pose",
        metric_key: str = "consecutive_success",
        num_increments: int = 25,
        min_rot_adr_coeff: float = 0.15,
        min_steps_for_dr_change: int = 240 * 4,
        z_rotation_steps: int = 16,
        ema_alpha: float = 0.1,
        min_episode_length_s: float = 20.0,
        episode_length_s: float = 120.0,
    ) -> dict[str, torch.Tensor | int | float]:
        r"""在 reset hook 中更新 LEAP-style ADR 档位。"""

        env_id_tensor = _resolve_env_ids(env, env_ids)  # 本次 reset 的环境集合 $\mathcal{E}_t$，不是 reward。
        if env_id_tensor.numel() == 0:
            return self._state_dict(num_increments)

        # 读取 command term 在 reset 前保留的 episode 小目标成功数 $S_i$。
        command_term = env.command_manager.get_term(command_name)
        successes = command_term.metrics.get(metric_key, torch.zeros(env.num_envs, device=env.device))[env_id_tensor]

        episode_success_mean = successes.float().mean()  # $\bar S_t=|\mathcal{E}_t|^{-1}\sum_i S_i$。

        # 官方用 randomized horizon 均值归一化，而不是用已经提前掉落的真实 episode 长度。
        if hasattr(env, "leap_adr_episode_lengths"):
            horizon_steps = env.leap_adr_episode_lengths[env_id_tensor].float().mean()
        else:
            # 初始 reset 前尚无 horizon buffer 时，用官方均匀分布期望值作为 bootstrap。
            horizon_steps = torch.tensor((min_episode_length_s + episode_length_s) * 0.5 / env.step_dt, device=env.device)
        horizon_s = torch.clamp(horizon_steps * env.step_dt, min=1.0e-6)  # $\bar T$，单位秒。

        # 更新 $C_{t+1}=\alpha\bar S_t+(1-\alpha)C_t$，该值既是日志也是升级判据输入。
        self.ema_success = ema_alpha * episode_success_mean + (1.0 - ema_alpha) * self.ema_success
        self.adr_criteria = (self.ema_success / float(z_rotation_steps)) / horizon_s

        # 官方实现是在 reset hook 中做 cooldown 计数；这里保留该行为，并用注释明确语义。
        can_increase = self.reset_checks_since_increase >= min_steps_for_dr_change
        strong_enough = self.adr_criteria >= min_rot_adr_coeff
        if can_increase and strong_enough and self.increment < num_increments:
            self.increment += 1  # $k\leftarrow k+1$，全局生效。
            self.ema_success.zero_()  # $C_t\leftarrow0$，避免旧难度高 EMA 连续触发升级。
            self.reset_checks_since_increase = 0  # 新难度重新累计 reset-hook cooldown。
        else:
            self.reset_checks_since_increase += 1  # 官方式 reset-hook check counter。

        self._publish_state(env, num_increments=num_increments)  # 将插值后的 ADR 参数写到 env。
        self._update_event_ranges(env)  # 让 reset EventTerm 在本次 reset 中使用新范围。
        return self._state_dict(num_increments)

    def _state_dict(self, num_increments: int = 25) -> dict[str, torch.Tensor | int | float]:
        r"""返回 CurriculumManager 可日志化的 ADR 状态。"""

        frac = float(self.increment) / float(max(num_increments, 1))  # $k/N$，用于 TensorBoard 观察课程进度。
        return {
            "increment": self.increment,
            "fraction": frac,
            "ema_success": self.ema_success.detach(),
            "adr_criteria": self.adr_criteria.detach(),
            "reset_checks_since_increase": self.reset_checks_since_increase,
        }

    def _publish_state(self, env: ManagerBasedRLEnv, num_increments: int = 25) -> None:
        r"""把当前 ADR 档位插值成 events/action term 可直接读取的标量。"""

        frac = float(self.increment) / float(max(num_increments, 1))  # 当前课程比例 $\lambda=k/N$。
        env.leap_adr_increment = self.increment  # 全局 ADR 档位 $k$。
        env.leap_adr_fraction = frac  # 全局 ADR 比例 $k/N$。
        env.leap_adr_ema_success = self.ema_success  # 全局成功 EMA $C_t$。
        env.leap_adr_criteria = self.adr_criteria  # 当前 rotations/sec 判据。
        env.leap_adr_joint_pos_noise = leap_adr_interpolate(0.0, 0.05, frac)  # reset $q$ 噪声半宽，rad。
        env.leap_adr_joint_vel_noise = leap_adr_interpolate(0.0, 0.01, frac)  # reset $\dot q$ 噪声半宽，rad/s。
        env.leap_adr_object_x_width = leap_adr_interpolate(0.0, 0.01, frac)  # object spawn x 半宽，m。
        env.leap_adr_object_y_width = leap_adr_interpolate(0.0, 0.01, frac)  # object spawn y 半宽，m。
        env.leap_adr_object_x_rot = leap_adr_interpolate(0.0, 0.1, frac)  # object spawn roll 半宽，rad。
        env.leap_adr_object_y_rot = leap_adr_interpolate(0.0, 0.1, frac)  # object spawn pitch 半宽，rad。
        env.leap_adr_object_z_rot = 0.0  # 官方 z spawn rotation 固定为 0，避免改变主旋转任务相位。
        env.leap_adr_action_noise = leap_adr_interpolate(0.1, 0.2, frac)  # action 高斯噪声 std，严格官方。
        env.leap_adr_action_latency = leap_adr_interpolate(0.0, 3.0, frac)  # action delay 上限，policy steps。
        env.leap_adr_max_linear_accel = leap_adr_interpolate(0.5, 5.0, frac)  # wrench 最大线加速度，m/s^2。
        env.leap_adr_object_mass_range = leap_adr_interpolate((1.0, 1.0), (0.9, 1.3), frac)  # object mass scale。
        env.leap_adr_stiffness_range = leap_adr_interpolate((3.0, 3.0), (2.5, 3.1), frac)  # actuator Kp。
        env.leap_adr_damping_range = leap_adr_interpolate((0.1, 0.1), (0.05, 0.15), frac)  # actuator Kd。
        env.leap_adr_robot_material_ranges = leap_adr_interpolate(
            {"static": (1.0, 1.0), "dynamic": (1.0, 1.0), "restitution": (0.0, 0.0)},
            {"static": (1.0, 1.0), "dynamic": (1.0, 1.0), "restitution": (0.0, 0.5)},
            frac,
        )
        env.leap_adr_object_material_ranges = leap_adr_interpolate(
            {"static": (1.0, 1.0), "dynamic": (1.0, 1.0), "restitution": (0.0, 0.0)},
            {"static": (0.3, 1.5), "dynamic": (0.3, 1.5), "restitution": (0.0, 0.5)},
            frac,
        )

    @staticmethod
    def _set_event_param(env: ManagerBasedRLEnv, term_name: str, param_name: str, value) -> None:
        r"""安全更新 EventTerm 参数；缺失项表示当前 env 未启用该 ADR 子项。"""

        try:
            term_cfg = env.event_manager.get_term_cfg(term_name)
        except ValueError:
            return
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)

    def _update_event_ranges(self, env: ManagerBasedRLEnv) -> None:
        r"""把 scheduler 插值得到的物理参数范围写入 reset EventTerm。"""

        self._set_event_param(env, "randomized_object_mass", "mass_distribution_params", env.leap_adr_object_mass_range)
        self._set_event_param(env, "randomized_actuator_gains", "stiffness_distribution_params", env.leap_adr_stiffness_range)
        self._set_event_param(env, "randomized_actuator_gains", "damping_distribution_params", env.leap_adr_damping_range)


# ============================================================================
# 奖励权重调整课程学习函数
# ============================================================================

def modify_rotation_velocity_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str = "rotation_velocity_reward",
    early_weight: float = 10.0,
    mid_weight: float = 15.0,
    late_weight: float = 20.0,
    mid_step: int = 300_000,
    late_step: int = 800_000
) -> float:
    """
    旋转速度奖励权重调整 - 训练初期低权重，后期逐步提高

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        term_name: 奖励项名称
        early_weight: 初期权重
        mid_weight: 中期权重
        late_weight: 后期权重
        mid_step: 中期开始步数
        late_step: 后期开始步数

    Returns:
        新的奖励权重值
    """
    current_step = env.common_step_counter

    # 确定当前应该使用的权重
    if current_step >= late_step:
        new_weight = late_weight
    elif current_step >= mid_step:
        new_weight = mid_weight
    else:
        new_weight = early_weight

    # 获取当前奖励项配置并更新权重
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if term_cfg.weight != new_weight:
        term_cfg.weight = new_weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return new_weight


def modify_rotation_axis_alignment_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str = "rotation_axis_alignment_reward",
    early_weight: float = 1.0,
    mid_weight: float = 0.5,
    late_weight: float = 0.1,
    mid_step: int = 300_000,
    late_step: int = 800_000
) -> float:
    """
    旋转轴对齐奖励权重调整 - 训练初期高权重，后期逐步降低

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        term_name: 奖励项名称
        early_weight: 初期权重
        mid_weight: 中期权重
        late_weight: 后期权重
        mid_step: 中期开始步数
        late_step: 后期开始步数

    Returns:
        新的奖励权重值
    """
    current_step = env.common_step_counter

    # 确定当前应该使用的权重
    if current_step >= late_step:
        new_weight = late_weight
    elif current_step >= mid_step:
        new_weight = mid_weight
    else:
        new_weight = early_weight

    # 获取当前奖励项配置并更新权重
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if term_cfg.weight != new_weight:
        term_cfg.weight = new_weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return new_weight


# ============================================================================
# 自适应域随机化课程学习函数
# ============================================================================

def object_mass_adr(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: tuple[float, float],
    enable_step: int = 600_000,
    max_strength_step: int = 1_200_000,
    max_variation: float = 0.5
) -> tuple[float, float]:
    """
    物体质量自适应域随机化 - 修改EventCfg中的mass_distribution_params

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        old_value: 当前的mass_distribution_params值 (min_scale, max_scale)
        enable_step: 启用步数
        max_strength_step: 达到最大强度的步数
        max_variation: 最大变化幅度（相对于1.0的偏差）

    Returns:
        新的mass_distribution_params值 (min_scale, max_scale)
    """
    current_step = env.common_step_counter

    if current_step < enable_step:
        return mdp.modify_env_param.NO_CHANGE

    if current_step >= max_strength_step:
        strength = max_variation
    else:
        progress = (current_step - enable_step) / (max_strength_step - enable_step)
        strength = progress * max_variation

    # 计算新的随机化范围：1.0 ± strength
    min_scale = 1.0 - strength
    max_scale = 1.0 + strength

    return (min_scale, max_scale)


def friction_adr(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: tuple[float, float],
    enable_step: int = 800_000,
    max_strength_step: int = 1_500_000,
    max_variation: float = 0.3
) -> tuple[float, float]:
    """
    摩擦系数自适应域随机化 - 修改EventCfg中的static_friction_range

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        old_value: 当前的static_friction_range值 (min_friction, max_friction)
        enable_step: 启用步数
        max_strength_step: 达到最大强度的步数
        max_variation: 最大变化幅度（相对于1.0的偏差）

    Returns:
        新的static_friction_range值 (min_friction, max_friction)
    """
    current_step = env.common_step_counter

    if current_step < enable_step:
        return mdp.modify_env_param.NO_CHANGE

    if current_step >= max_strength_step:
        strength = max_variation
    else:
        progress = (current_step - enable_step) / (max_strength_step - enable_step)
        strength = progress * max_variation

    # 计算新的随机化范围：1.0 ± strength，确保最小值不小于0.1
    min_friction = max(0.1, 1.0 - strength)
    max_friction = 1.0 + strength

    return (min_friction, max_friction)


def object_scale_adr(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: dict[str, tuple[float, float]],
    enable_step: int = 1_000_000,
    max_strength_step: int = 1_800_000,
    max_variation: float = 0.2
) -> dict[str, tuple[float, float]]:
    """
    物体尺寸自适应域随机化 - 修改EventCfg中的scale_range

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        old_value: 当前的scale_range值 {"x": (min_scale, max_scale), "y": ..., "z": ...}
        enable_step: 启用步数
        max_strength_step: 达到最大强度的步数
        max_variation: 最大变化幅度（相对于1.0的偏差）

    Returns:
        新的scale_range值 {"x": (min_scale, max_scale), "y": ..., "z": ...}
    """
    current_step = env.common_step_counter

    if current_step < enable_step:
        return mdp.modify_env_param.NO_CHANGE

    if current_step >= max_strength_step:
        strength = max_variation
    else:
        progress = (current_step - enable_step) / (max_strength_step - enable_step)
        strength = progress * max_variation

    # 计算新的随机化范围：1.0 ± strength
    min_scale = 1.0 - strength
    max_scale = 1.0 + strength

    # 返回所有轴的随机化范围
    return {
        "x": (min_scale, max_scale),
        "y": (min_scale, max_scale),
        "z": (min_scale, max_scale)
    }


# ============================================================================
# 动作缩放因子调整
# ============================================================================


# ============================================================================
# 旋转轴复杂度课程学习函数
# ============================================================================

def simple_rotation_axis(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: str,
    z_axis_step: int = 0,
    random_axis_step: int = 1_200_000
) -> str:
    """
    简化旋转轴复杂度调整：Z轴 → 任意轴

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        old_value: 当前旋转轴模式
        z_axis_step: Z轴阶段开始步数
        random_axis_step: 任意轴阶段开始步数

    Returns:
        新的旋转轴模式
    """
    current_step = env.common_step_counter

    if current_step >= random_axis_step:
        return "random"
    else:
        return "z_axis"
