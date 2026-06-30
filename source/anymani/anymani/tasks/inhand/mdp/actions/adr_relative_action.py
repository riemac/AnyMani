# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""LEAP-style ADR action term for AnyMani in-hand tasks.

官方 LEAP DirectRLEnv 在 action 入口处加入两类 sim2real 相关扰动：

$$
\tilde a_t=a_t+\eta_t,\qquad \eta_t\sim\mathcal N(0,\sigma_a(k)^2I),
$$

以及离散 action latency：

$$
a_t^{exec}=\tilde a_{t-\ell},\qquad
\ell=\max(0,\lfloor h(k)-r\rfloor),\quad r\in\{0,1\}.
$$

本文件把这两个机制迁移到 IsaacLab ManagerBased action term 中。注意 actor 观测中的
``last_action`` 仍保持 IsaacLab 语义：它记录 policy 输入 action $a_t$，而不是加噪/延迟后的
执行 action $a_t^{exec}$。这样保持 `Tactile-ADR-v0` 的 observation dimension 与 N000 baseline
一致，便于前 100 epoch 做公平对比。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction, RelativeJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


def compute_leap_adr_latency_steps(
    latency: float | torch.Tensor,
    random_subtraction: torch.Tensor,
    max_latency: int = 3,
) -> torch.Tensor:
    r"""计算 LEAP 官方离散 action latency index。

    官方实现不是直接把连续 curriculum 值 $h(k)\in[0,3]$ 当成 fractional delay，而是
    先减去 episode 内采样的随机整数 $r\in\{0,1\}$，再取 floor 并裁剪到合法历史窗口：
    $$
    \ell=\operatorname{clip}\bigl(\max(0,\lfloor h(k)-r\rfloor),\ 0,\ L_{\max}\bigr).
    $$

    Args:
        latency: 当前 ADR 档位插值得到的 $h(k)$，单位是 policy step。
        random_subtraction: episode 内采样的 $r$，形状通常为 $[N_{env},1]$。
        max_latency: 历史 buffer 最大可索引延迟 $L_{\max}$。

    Returns:
        torch.Tensor: 每个 env/action 使用的离散历史索引 $\ell$，dtype 为 ``long``。
    """

    latency_tensor = torch.as_tensor(latency, device=random_subtraction.device, dtype=torch.float32)  # $h(k)$。
    latency_steps = torch.floor(latency_tensor - random_subtraction.float())  # $\lfloor h(k)-r\rfloor$。
    latency_steps = torch.clamp(latency_steps, min=0, max=max_latency)  # 投影到 $[0,L_{\max}]$。
    return latency_steps.long()  # 离散 history index，形状跟 ``random_subtraction`` 一致。


def compute_official_target_update(
    prev_targets: torch.Tensor,
    executed_actions: torch.Tensor,
    scale: float | torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    r"""计算官方 LEAP 的 target-buffer 关节目标更新。

    官方 DirectRLEnv 的 relative 控制不是 IsaacLab `RelativeJointPositionAction` 的
    ``q_t + \Delta q`` 语义，而是维护一个内部 target buffer：
    $$
    q_t^{target}=
    \operatorname{clip}
    \left(
        q_{t-1}^{target}+\alpha a_t^{exec},
        q_{\min}, q_{\max}
    \right),
    \qquad \alpha=\frac{1}{24}.
    $$

    Args:
        prev_targets: 上一时刻的目标关节位置 $q_{t-1}^{target}$，形状 ``[N, J]``。
        executed_actions: 已经过 action noise 与 latency 的规范化动作 $a_t^{exec}$，形状 ``[N, J]``。
        scale: 动作到关节目标增量的缩放系数 $\alpha$。
        lower_limits: 关节下限 $q_{\min}$，形状 ``[N, J]`` 或可广播到该形状。
        upper_limits: 关节上限 $q_{\max}$，形状 ``[N, J]`` 或可广播到该形状。

    Returns:
        torch.Tensor: 裁剪后的新目标关节位置 $q_t^{target}$，形状 ``[N, J]``。
    """

    target_delta = executed_actions * scale
    return torch.clamp(prev_targets + target_delta, min=lower_limits, max=upper_limits)


class ADRRelativeJointPositionAction(RelativeJointPositionAction):
    r"""带 LEAP ADR action noise/latency 的相对关节位置动作项。"""

    cfg: ADRRelativeJointPositionActionCfg

    def __init__(self, cfg: ADRRelativeJointPositionActionCfg, env):
        r"""初始化动作历史 buffer。"""

        super().__init__(cfg, env)
        self._max_latency = int(cfg.max_latency)  # 最大离散延迟步数，官方上限为 3。
        self._latency_rand = int(cfg.latency_rand)  # 官方每 episode 从 {0,1} 中减去一个随机整数。
        self._action_history = torch.zeros(
            self.num_envs, self.action_dim, self._max_latency + 1, device=self.device
        )  # $[N_{env},N_a,L+1]$，第 0 层是最新 noisy action。
        self._latency_steps = torch.zeros(
            self.num_envs, self.action_dim, dtype=torch.long, device=self.device
        )  # 每个 env/action 的 $\ell$，单位 policy step。

    def process_actions(self, actions: torch.Tensor):
        r"""记录 policy action，并生成加噪/延迟后的执行 action。"""

        self._raw_actions[:] = actions  # 保持 IsaacLab last_action 语义：raw action 是 policy 输出 $a_t$。
        action_noise = float(getattr(self._env, "leap_adr_action_noise", 0.0))
        noisy_actions = actions + torch.randn_like(actions) * action_noise
        noisy_actions = torch.clamp(noisy_actions, -1.0, 1.0)

        # 历史 buffer 第 0 层保存最新 $\tilde a_t$，更旧动作向右移动。
        self._action_history = torch.roll(self._action_history, shifts=1, dims=2)
        self._action_history[:, :, 0] = noisy_actions

        # 按 reset 时采样的 $\ell$ 取 $\tilde a_{t-\ell}$。
        env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.action_dim)
        act_ids = torch.arange(self.action_dim, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        delayed_actions = self._action_history[env_ids, act_ids, self._latency_steps]

        # 以下 affine transform 与 IsaacLab JointAction 保持一致：processed = delayed * scale + offset。
        self._processed_actions = delayed_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""在 episode reset 时重置 action 历史并重采样 latency。"""

        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._raw_actions[env_ids] = 0.0
        self._action_history[env_ids] = 0.0
        latency_float = float(getattr(self._env, "leap_adr_action_latency", 0.0))  # $h(k)$，单位 policy step。
        random_sub = torch.randint(0, self._latency_rand + 1, (len(env_ids), 1), device=self.device)  # $r\in\{0,1\}$。
        latency = compute_leap_adr_latency_steps(latency_float, random_sub, self._max_latency)  # $\ell$。
        self._latency_steps[env_ids] = latency.expand(-1, self.action_dim)  # 每个关节共享同一 env-level delay。


@configclass
class ADRRelativeJointPositionActionCfg(actions_cfg.RelativeJointPositionActionCfg):
    r"""ADR action term 配置。"""

    class_type: type[ActionTerm] = ADRRelativeJointPositionAction
    max_latency: int = 3
    latency_rand: int = 1


class OfficialADRTargetJointPositionAction(JointAction):
    r"""官方 LEAP 风格的 target-buffer relative action。

    该动作项复刻官方 DirectRLEnv 的两级语义：

    1. 先对 policy 输出的规范化动作加入 ADR action noise，并施加 episode-level latency；
    2. 再把执行动作解释为对内部 target buffer 的增量，而不是对当前关节位置的增量：
       $$
       q_t^{target}=\operatorname{clip}(q_{t-1}^{target}+\alpha a_t^{exec}, q_{\min}, q_{\max}).
       $$

    这与 IsaacLab 默认 `RelativeJointPositionAction` 的
    $$
    q_t^{cmd}=q_t+\alpha a_t
    $$
    完全不同；官方 actor 观测中的 `cur_targets` 历史正是围绕这个内部 target buffer 构建的。
    """

    cfg: OfficialADRTargetJointPositionActionCfg

    def __init__(self, cfg: OfficialADRTargetJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._max_latency = int(cfg.max_latency)
        self._latency_rand = int(cfg.latency_rand)
        self._action_history = torch.zeros(self.num_envs, self.action_dim, self._max_latency + 1, device=self.device)
        self._latency_steps = torch.zeros(self.num_envs, self.action_dim, dtype=torch.long, device=self.device)
        self._executed_actions = torch.zeros_like(self._raw_actions)
        self._previous_targets = torch.zeros_like(self._raw_actions)
        self._current_targets = torch.zeros_like(self._raw_actions)
        self._joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        self._joint_lower = self._joint_limits[..., 0]
        self._joint_upper = self._joint_limits[..., 1]
        self._pregrasp_targets = torch.tensor(
            cfg.pregrasp_joint_pos, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        self._pregrasp_targets = self._pregrasp_targets.expand(self.num_envs, -1).clone()

    @property
    def previous_targets(self) -> torch.Tensor:
        return self._previous_targets

    @property
    def current_targets(self) -> torch.Tensor:
        return self._current_targets

    @property
    def executed_actions(self) -> torch.Tensor:
        return self._executed_actions

    @property
    def pregrasp_targets(self) -> torch.Tensor:
        return self._pregrasp_targets

    def process_actions(self, actions: torch.Tensor):
        r"""记录 raw policy action，并生成官方语义下的执行动作 $a_t^{exec}$。"""

        self._raw_actions[:] = actions
        action_noise = float(getattr(self._env, "leap_adr_action_noise", 0.0))
        noisy_actions = actions + torch.randn_like(actions) * action_noise

        self._action_history = torch.roll(self._action_history, shifts=1, dims=2)
        self._action_history[:, :, 0] = noisy_actions

        env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.action_dim)
        act_ids = torch.arange(self.action_dim, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        delayed_actions = self._action_history[env_ids, act_ids, self._latency_steps]
        self._executed_actions = torch.clamp(delayed_actions, -1.0, 1.0)
        self._processed_actions = self._executed_actions

    def apply_actions(self):
        r"""把执行动作解释为内部 target buffer 的增量，并下发关节位置目标。"""

        self._current_targets = compute_official_target_update(
            prev_targets=self._previous_targets,
            executed_actions=self._executed_actions,
            scale=self._scale,
            lower_limits=self._joint_lower,
            upper_limits=self._joint_upper,
        )
        self._asset.set_joint_position_target(self._current_targets, joint_ids=self._joint_ids)
        self._previous_targets[:] = self._current_targets

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""重置 target buffer 与 action history，并按官方方式重采样 latency。"""

        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._executed_actions[env_ids] = 0.0
        self._action_history[env_ids] = 0.0

        if hasattr(self._env, "leap_official_reset_joint_pos"):
            reset_targets = self._env.leap_official_reset_joint_pos[env_ids]
        else:
            reset_targets = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids].clone()

        self._previous_targets[env_ids] = reset_targets
        self._current_targets[env_ids] = reset_targets

        latency_float = float(getattr(self._env, "leap_adr_action_latency", 0.0))
        random_sub = torch.randint(0, self._latency_rand + 1, (len(env_ids), 1), device=self.device)
        latency = compute_leap_adr_latency_steps(latency_float, random_sub, self._max_latency)
        self._latency_steps[env_ids] = latency.expand(-1, self.action_dim)


@configclass
class OfficialADRTargetJointPositionActionCfg(actions_cfg.RelativeJointPositionActionCfg):
    r"""官方 LEAP target-buffer relative action 配置。"""

    class_type: type[ActionTerm] = OfficialADRTargetJointPositionAction
    max_latency: int = 3
    latency_rand: int = 1
    pregrasp_joint_pos: tuple[float, ...] = ()
