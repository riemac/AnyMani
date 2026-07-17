r"""Curriculum terms for `tasks.gm`.

本模块只承载 `gm` 任务语义内部的 curriculum 状态更新，不处理 asset-bank
采样策略，也不处理训练算法超参。当前唯一落子的 curriculum 是 AnyRotate 风格的
adaptive reward curriculum：先让策略学会基本重定向，再逐步释放 contact / stable
正则项。

核心思想来自 AnyRotate Appendix B.3：

$$
\lambda_{rew} = \operatorname{clip}\left(
\frac{g_{eval}-g_{min}}{g_{max}-g_{min}},\ 0,\ 1
\right)
$$

但本项目把 $g_{eval}$ 明确命名为 `goal_success_count` 的全局 EMA，避免沿用
IsaacLab 官方 inhand 中语义偏含糊的 `consecutive_success`。这里的
`goal_success_count` 指“一个 episode 内完成了多少个重定向子目标”，不是在
阈值内停留了多少帧。

TODO(net-rotation reward curriculum):
    tactile rotation baseline 新增独立 curriculum，不修改现有 goal-success 类的语义。
    新 term 从 command 读取刚结束 episode 的未裁剪有向净旋转：

    $$
    n_i^{turn}
    =
    \max
    \left(
    \frac{\Psi_i}{2\pi},0
    \right).
    $$

    全局 EMA 与 release 为：

    $$
    G_{k+1}
    =
    (1-\beta)G_k
    +
    \beta\operatorname{mean}_{i\in\mathcal E_k}(n_i^{turn}),
    $$

    $$
    \lambda_{rew}
    =
    \operatorname{clip}(G_k-1,0,1).
    $$

    也就是平均真实净旋转从 1 圈到 2 圈时，contact 与 stable 两组连续释放。往返旋转
    在 signed `Psi` 中相消；反向净转只提供 0 competence，不写入负 curriculum state。
    新 tactile rotation env 只配置该 net-rotation curriculum；不得与现有 goal-success
    curriculum 同时乘到同一 reward group。

TODO(LEAP ADR by actual turns per second):
    新增明确的 net-rotation-rate scheduler；它可以复用 inhand LEAP ADR 的 range 发布原件，
    但不得改变旧 `LeapADRGlobalScheduler` 或继续沿用 `min_rot_adr_coeff` 命名。promotion 为：

    $$
    R_k^{turn}
    =
    \frac{\operatorname{EMA}(n_i^{turn})}{\bar T_{sampled}}
    \ge
    0.08\ \mathrm{turns/s}.
    $$

    分母使用 `sampled_horizon_steps * env.step_dt` 得到的 sampled full horizon seconds，
    使提前掉落自然降低能力分。配置和日志必须直接写
    `net_rotation_rate_turns_per_s` 与 threshold 的物理单位。CurriculumManager 在 command
    reset 前执行，因此应读取尚未清零的 episode metric；该 manager 顺序是实现契约的一部分。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from .adr_state import get_gm_adr_state
from .commands.tactile_rotation_command import ensure_post_physics_progress_updated


def net_rotation_reward_release(net_turns_ema: torch.Tensor, release_start: float, release_end: float) -> torch.Tensor:
    r"""把策略正向净旋转 competence 映射到 `[0,1]` reward release。"""

    if float(release_end) <= float(release_start):
        raise ValueError(f"release_end must exceed release_start, got {release_start}, {release_end}.")
    return torch.clamp(
        (net_turns_ema - float(release_start)) / (float(release_end) - float(release_start)), 0.0, 1.0
    )


def net_rotation_rate_turns_per_s(net_turns_ema: torch.Tensor, sampled_horizon_s: torch.Tensor) -> torch.Tensor:
    r"""用 sampled full horizon 归一化净圈数；提前掉落不会缩短分母而虚增能力。"""

    return net_turns_ema / torch.clamp(sampled_horizon_s, min=1.0e-6)


def leap_adr_interpolate(initial_value, final_value, fraction: float):
    r"""按 GM-owned LEAP ADR fraction 递归线性插值标量、tuple/list 或 dict。"""

    if isinstance(initial_value, dict):
        return {key: leap_adr_interpolate(initial_value[key], final_value[key], fraction) for key in initial_value}
    if isinstance(initial_value, (tuple, list)):
        return type(initial_value)(
            leap_adr_interpolate(initial, final, fraction)
            for initial, final in zip(initial_value, final_value, strict=True)
        )
    return float(initial_value) + float(fraction) * (float(final_value) - float(initial_value))


class RewardCurriculumByGoalSuccess(ManagerTermBase):
    r"""根据平均子目标成功数释放 reward curriculum 系数。

    该 term 维护一个全局标量 $\lambda_{global}\in[0,1]$，并写入 env 属性，
    供 `rewards.py` 中的 contact / stable / action 正则项读取。之所以采用
    全局标量，而不是 per-env 系数，是因为它更接近 AnyRotate 的实验语义：
    “策略整体已经学会基本重定向后，再要求接触质量和动作稳定性”。

    计算流程：

    1. command term 在每个 env 中维护 `metrics[metric_key]`，默认
       `metric_key="goal_success_count"`；该值表示当前 episode 已完成的
       重定向子目标数量。
    2. 本 curriculum term 对当前参与更新的 env ids 求平均：
       $g_{batch}=\operatorname{mean}(g_i)$。
    3. 用 EMA 得到平滑全局指标：
       $g_{ema}\leftarrow (1-\alpha)g_{ema}+\alpha g_{batch}$。
    4. 线性映射为 release 系数：
       $\lambda=\operatorname{clip}((g_{ema}-g_{min})/(g_{max}-g_{min}),0,1)$。

    preset:
        - `g_min=1.0, g_max=2.0`：对齐 AnyRotate 的释放区间直觉，即平均每
          个 episode 至少完成约 1 个子目标后才开始释放 contact/stable；到约
          2 个子目标时完全释放。
        - `ema_alpha=0.05`：约 $1/\alpha=20$ 次 curriculum update 的平滑窗口，
          作为第一版保守默认；后续可按 rollout 频率调参。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        r"""初始化 EMA 状态。

        Args:
            cfg (CurriculumTermCfg): Isaac Lab curriculum term 配置。
            env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        """

        # 父类保存 cfg/env，保持与 Isaac Lab manager term 生命周期一致
        super().__init__(cfg, env)

        # EMA 从 0 开始，意味着训练最初默认处在 curriculum 未释放状态
        self._ema_goal_success = torch.tensor(0.0, device=env.device)  # 标量，$g_{ema}$
        self._lambda_global = torch.tensor(0.0, device=env.device)  # 标量，$\lambda_{global}$

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""Reset hook：不清空全局 curriculum 进度。

        Curriculum 表示训练整体阶段，不是单个 env episode 状态；因此 env reset
        时不应把 EMA 清零。若用户要重新开始训练，应重新创建 env / manager。

        Args:
            env_ids (Sequence[int] | None): Isaac Lab manager 传入的 reset env ids。
        """

        _ = env_ids  # curriculum 是全局训练状态，单个 env reset 不影响它

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        command_name: str,
        metric_key: str = "goal_success_count",
        g_min: float = 1.0,
        g_max: float = 2.0,
        ema_alpha: float = 0.05,
        lambda_attr_name: str = "_gm_reward_curriculum_lambda",
        progress_attr_name: str = "_gm_reward_curriculum_goal_success_ema",
    ) -> dict[str, torch.Tensor]:
        r"""更新并写出全局 reward curriculum 系数。

        Args:
            env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
            env_ids (Sequence[int] | slice): 本次 curriculum update 涉及的 env ids。
            command_name (str): command manager 中的重定向 command term 名称。
            metric_key (str): command term `metrics` 中的 per-env 成功计数字段名。
            g_min (float): release 起点；$g_{ema}\le g_{min}$ 时 $\lambda=0$。
            g_max (float): release 终点；$g_{ema}\ge g_{max}$ 时 $\lambda=1$。
            ema_alpha (float): EMA 更新率 $\alpha$。
            lambda_attr_name (str): 写入 env 的全局 release 系数属性名。
            progress_attr_name (str): 写入 env 的 EMA progress 属性名。

        Returns:
            dict[str, torch.Tensor]: 供 Isaac Lab logging 使用的 curriculum 状态。

        Raises:
            RuntimeError: 当 command term 没有暴露 `metric_key` 时抛出。
            ValueError: 当 `g_max <= g_min` 或 `ema_alpha` 越界时抛出。
        """

        # 检查 release 区间，避免除以 0 或负斜率导致 curriculum 语义反转
        if float(g_max) <= float(g_min):
            raise ValueError(f"g_max must be larger than g_min, got g_min={g_min}, g_max={g_max}.")
        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}.")

        # 从 command term 读取 per-env 子目标成功数；该 metric 是 ReorientCommand 必须兑现的契约
        command_term = env.command_manager.get_term(command_name)  # 重定向 command term
        goal_success_count = command_term.metrics.get(metric_key, None)  # `[B]`，每个 env 当前 episode 完成的子目标数
        if goal_success_count is None:
            raise RuntimeError(
                f"Command term '{command_name}' must expose metrics['{metric_key}'] for reward curriculum. "
                "Use `goal_success_count` to count completed subgoals, not threshold-satisfied frames."
            )

        # 按 Isaac Lab 传入的 env_ids 取子集；slice(None) 表示全部 env
        if isinstance(env_ids, slice):
            batch_success = goal_success_count[env_ids].float()  # `[B_update]`，本次更新子集
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=goal_success_count.device, dtype=torch.long)  # env id 索引
            batch_success = goal_success_count[env_ids_tensor].float()  # `[B_update]`，本次更新子集

        # 计算 batch 平均成功数，并做 EMA 平滑，避免单个 rollout 抖动导致 reward 突然释放/关闭
        batch_mean = batch_success.mean()  # 标量，$g_{batch}$
        alpha = float(ema_alpha)  # EMA 更新率 $\alpha$
        self._ema_goal_success = (1.0 - alpha) * self._ema_goal_success + alpha * batch_mean.detach()  # $g_{ema}$

        # 线性 release：$g_{min}$ 前为 0，$g_{max}$ 后为 1，中间线性过渡
        raw_lambda = (self._ema_goal_success - float(g_min)) / (float(g_max) - float(g_min))  # 未裁剪 release
        self._lambda_global = torch.clamp(raw_lambda, 0.0, 1.0)  # $\lambda_{global}\in[0,1]$

        # 写入 env 属性，供 rewards.py 的 `_curriculum_gain` 读取；detach 表示这是训练调度状态而非梯度图
        setattr(env, lambda_attr_name, self._lambda_global.detach())  # 全局 reward release 系数
        setattr(env, progress_attr_name, self._ema_goal_success.detach())  # 平滑后的平均子目标成功数

        # 返回 dict 让 Isaac Lab 的 curriculum logging 能记录关键状态
        return {
            "lambda": self._lambda_global.detach(),  # 当前 release 系数
            "goal_success_ema": self._ema_goal_success.detach(),  # 平滑 progress
            "goal_success_batch": batch_mean.detach(),  # 当前 batch 原始平均成功数
        }


class RewardCurriculumByNetRotation(ManagerTermBase):
    r"""根据刚结束 episodes 的真实 signed net rotation 释放 contact/stable reward。

    Command 已把反向与往返运动保留在 signed accumulation 中，最后只把正 competence 暴露为
    `net_rotation_turns=max(Psi,0)/(2*pi)`。本 term 对 reset batch 均值做全局 EMA，并用
    1→2 turns 线性 release；它不与旧 goal-success curriculum 叠乘。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        r"""初始化训练全局 EMA 与 release coefficient。"""

        super().__init__(cfg, env)
        self.net_turns_ema = torch.tensor(0.0, device=env.device)
        self.lambda_global = torch.tensor(0.0, device=env.device)
        setattr(env, "_gm_reward_curriculum_lambda", self.lambda_global)
        setattr(env, "_gm_reward_curriculum_net_turns_ema", self.net_turns_ema)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""单个 env reset 不清训练全局 curriculum state。"""

        _ = env_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        command_name: str,
        release_start_turns: float = 1.0,
        release_end_turns: float = 2.0,
        ema_alpha: float = 0.05,
        lambda_attr_name: str = "_gm_reward_curriculum_lambda",
        progress_attr_name: str = "_gm_reward_curriculum_net_turns_ema",
    ) -> dict[str, torch.Tensor]:
        r"""在 command partial reset 前读取 episode net turns，并更新全局 reward release。"""

        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError(f"ema_alpha must be in (0,1], got {ema_alpha}.")
        command = ensure_post_physics_progress_updated(env, command_name)
        ids = _resolve_env_ids(env, env_ids)
        if ids.numel() == 0:
            return {"lambda": self.lambda_global.detach(), "net_turns_ema": self.net_turns_ema.detach()}
        batch_mean = command.net_rotation_turns[ids].float().mean()
        alpha = float(ema_alpha)
        self.net_turns_ema = (1.0 - alpha) * self.net_turns_ema + alpha * batch_mean.detach()
        self.lambda_global = net_rotation_reward_release(
            self.net_turns_ema, release_start_turns, release_end_turns
        )
        setattr(env, lambda_attr_name, self.lambda_global.detach())
        setattr(env, progress_attr_name, self.net_turns_ema.detach())
        return {
            "lambda": self.lambda_global.detach(),
            "net_turns_ema": self.net_turns_ema.detach(),
            "net_turns_batch": batch_mean.detach(),
        }


class LeapADRByNetRotationRate(ManagerTermBase):
    r"""以 actual net turns/s 推进 GM-owned 25-level LEAP ADR ranges。

    第 0 档在首次有效 reset check 自动进入第 1 档，避免所有随机化宽度为零时 curriculum
    长期无法获得探索扰动。之后升级要求：reset-check cooldown 至少 960，且
    `EMA(net_turns) / mean(sampled_full_horizon_s) >= 0.08 turns/s`。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        r"""初始化 ADR increment、EMA、物理单位 criterion 与 env runtime ranges。"""

        super().__init__(cfg, env)
        starting_increment = cfg.params.get("starting_increment", 0)
        if not isinstance(starting_increment, (int, float)):
            raise TypeError(f"starting_increment must be numeric, got {starting_increment!r}.")
        self.increment = int(starting_increment)
        self.net_turns_ema = torch.tensor(0.0, device=env.device)
        self.net_rotation_rate = torch.tensor(0.0, device=env.device)
        self.reset_checks_since_increase = 0
        self._publish_state(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""ADR 是训练全局状态，不随单 env episode reset 清零。"""

        _ = env_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        command_name: str,
        num_increments: int = 25,
        threshold_turns_per_s: float = 0.08,
        min_reset_checks_for_increase: int = 960,
        ema_alpha: float = 0.1,
        min_episode_length_s: float = 20.0,
        episode_length_s: float = 120.0,
    ) -> dict[str, torch.Tensor | int | float]:
        r"""读取 episode net turns/full horizon，更新 criterion 并按规则至多升级一档。"""

        if num_increments <= 0:
            raise ValueError(f"num_increments must be positive, got {num_increments}.")
        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError(f"ema_alpha must be in (0,1], got {ema_alpha}.")
        ids = _resolve_env_ids(env, env_ids)
        if ids.numel() == 0:
            return self._state_dict(num_increments)
        command = ensure_post_physics_progress_updated(env, command_name)
        batch_mean_turns = command.net_rotation_turns[ids].float().mean()
        alpha = float(ema_alpha)
        self.net_turns_ema = (1.0 - alpha) * self.net_turns_ema + alpha * batch_mean_turns.detach()

        sampled_horizon_steps = getattr(env, "leap_adr_episode_lengths", None)
        if isinstance(sampled_horizon_steps, torch.Tensor):
            horizon_s = sampled_horizon_steps[ids].float().mean() * float(env.step_dt)
        else:
            horizon_s = torch.tensor((min_episode_length_s + episode_length_s) * 0.5, device=env.device)
        self.net_rotation_rate = net_rotation_rate_turns_per_s(self.net_turns_ema, horizon_s)

        auto_bootstrap = self.increment == 0
        cooldown_ready = self.reset_checks_since_increase >= int(min_reset_checks_for_increase)
        criterion_ready = self.net_rotation_rate >= float(threshold_turns_per_s)
        if self.increment < num_increments and (auto_bootstrap or (cooldown_ready and criterion_ready)):
            self.increment += 1
            self.net_turns_ema.zero_()  # 新难度必须重新积累 competence evidence
            self.reset_checks_since_increase = 0
        else:
            self.reset_checks_since_increase += 1

        self._publish_state(env, num_increments)
        self._update_event_ranges(env)
        return self._state_dict(num_increments)

    def _publish_state(self, env: ManagerBasedRLEnv, num_increments: int = 25) -> None:
        r"""发布 action/events 消费的 LEAP ranges 与 AnyRotate COM half-width。"""

        fraction = float(self.increment) / float(max(num_increments, 1))
        published = {
            "leap_adr_increment": self.increment,
            "leap_adr_fraction": fraction,
            "leap_adr_net_turns_ema": self.net_turns_ema,
            "leap_adr_net_rotation_rate_turns_per_s": self.net_rotation_rate,
            "leap_adr_joint_pos_noise": leap_adr_interpolate(0.0, 0.05, fraction),
            "leap_adr_joint_vel_noise": leap_adr_interpolate(0.0, 0.01, fraction),
            "leap_adr_object_x_width": leap_adr_interpolate(0.0, 0.01, fraction),
            "leap_adr_object_y_width": leap_adr_interpolate(0.0, 0.01, fraction),
            "leap_adr_object_x_rot": leap_adr_interpolate(0.0, 0.1, fraction),
            "leap_adr_object_y_rot": leap_adr_interpolate(0.0, 0.1, fraction),
            "leap_adr_object_z_rot": 0.0,
            "leap_adr_action_noise": leap_adr_interpolate(0.1, 0.2, fraction),
            "leap_adr_action_latency": leap_adr_interpolate(0.0, 3.0, fraction),
            "leap_adr_max_linear_accel": leap_adr_interpolate(0.5, 5.0, fraction),
            "gm_adr_com_half_width": leap_adr_interpolate(0.0, 0.01, fraction),
            "leap_adr_object_mass_range": leap_adr_interpolate((1.0, 1.0), (0.9, 1.3), fraction),
            "leap_adr_stiffness_range": leap_adr_interpolate((3.0, 3.0), (2.5, 3.1), fraction),
            "leap_adr_damping_range": leap_adr_interpolate((0.1, 0.1), (0.05, 0.15), fraction),
            "leap_adr_robot_material_ranges": leap_adr_interpolate(
                {"static": (1.0, 1.0), "dynamic": (1.0, 1.0), "restitution": (0.0, 0.0)},
                {"static": (1.0, 1.0), "dynamic": (1.0, 1.0), "restitution": (0.0, 0.5)},
                fraction,
            ),
            "leap_adr_object_material_ranges": leap_adr_interpolate(
                {"static": (1.0, 1.0), "dynamic": (1.0, 1.0), "restitution": (0.0, 0.0)},
                {"static": (0.3, 1.5), "dynamic": (0.3, 1.5), "restitution": (0.0, 0.5)},
                fraction,
            ),
        }
        for name, value in published.items():
            setattr(env, name, value)
        state = get_gm_adr_state(env)
        state.set(env, "action_noise", float(published["leap_adr_action_noise"]))
        state.set(env, "max_acceleration", float(published["leap_adr_max_linear_accel"]))
        state.set(env, "fraction", fraction)

    def _state_dict(self, num_increments: int) -> dict[str, torch.Tensor | int | float]:
        r"""返回带明确 turns/s 单位名的 CurriculumManager 日志。"""

        return {
            "increment": self.increment,
            "fraction": float(self.increment) / float(max(num_increments, 1)),
            "net_turns_ema": self.net_turns_ema.detach(),
            "net_rotation_rate_turns_per_s": self.net_rotation_rate.detach(),
            "reset_checks_since_increase": self.reset_checks_since_increase,
        }

    @staticmethod
    def _set_event_param(env: ManagerBasedRLEnv, term_name: str, param_name: str, value) -> None:
        r"""若 env 启用了目标 EventTerm，则安全更新其当前 ADR range。"""

        try:
            term_cfg = env.event_manager.get_term_cfg(term_name)
        except ValueError:
            return
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)

    def _update_event_ranges(self, env: ManagerBasedRLEnv) -> None:
        r"""让同一次 reset 后续 events 使用刚发布的新 mass/Kp/Kd ranges。"""

        self._set_event_param(
            env, "randomized_object_mass", "mass_distribution_params", getattr(env, "leap_adr_object_mass_range")
        )
        self._set_event_param(
            env, "randomized_actuator_gains", "stiffness_distribution_params", getattr(env, "leap_adr_stiffness_range")
        )
        self._set_event_param(
            env, "randomized_actuator_gains", "damping_distribution_params", getattr(env, "leap_adr_damping_range")
        )


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice) -> torch.Tensor:
    r"""把 CurriculumManager reset subset 统一成 env-device LongTensor。"""

    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


__all__ = [
    "LeapADRByNetRotationRate",
    "RewardCurriculumByGoalSuccess",
    "RewardCurriculumByNetRotation",
    "leap_adr_interpolate",
    "net_rotation_rate_turns_per_s",
    "net_rotation_reward_release",
]
