r"""ADR-aware declarative joint-position action terms for GM in-hand tasks.

本模块把 LEAP official action corruption 与两类关节位置命令语义拆成可声明配置的动作项。
核心目标不是增加更多 Python class，而是把动作空间的科研变量显式收敛到少量字段：

- `reference="target"`：参考上一帧 command target buffer $u_{t-1}$；
- `reference="current"`：参考当前真实关节角 $q_t$；
- `ADRRelativeJointPositionAction`：policy 输出解释为 raw-rad delta；
- `ADREMAJointPositionToLimitsAction`：policy 输出先映射为 joint-limit absolute target，再做 EMA / blend。

统一符号：

$$
a_t\in[-1,1]^J
$$

表示 policy-facing normalized action，$J$ 是动作关节数；ADR 后的执行动作为：

$$
a_t^{exec}=\operatorname{clip}(\tilde a_{t-\ell},-1,1).
$$

official latency 采用 LEAP 当前实现：

$$
\ell=\operatorname{clip}\left(\max(0,\lfloor h(k)-r\rfloor),0,3\right),\quad r\in\{0,1\}.
$$

所有派生 action term 都暴露同一个 runtime contract：

- `current_targets`：本步实际下发给 PD controller 的 command target $u_t$，单位 rad；
- `executed_actions`：加 ADR noise / latency 后的 normalized action $a_t^{exec}$；
- `pregrasp_targets`：official reward 的 pregrasp anchor $q^{pre}$，单位 rad。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


ActionReference = Literal["current", "target"]
r"""动作参考点枚举。

`current` 表示参考真实关节角 $q_t$；`target` 表示参考上一帧 command target buffer $u_{t-1}$。
"""

OFFICIAL_ADR_MAX_LATENCY = 3
r"""LEAP official action latency 最大历史索引 $L_{max}=3$。"""

OFFICIAL_ADR_LATENCY_RAND = 1
r"""LEAP official reset-time 随机减项 $r\in\{0,1\}$ 的上界。"""

LEAP_ACTION_SCALE = 1.0 / 24.0
r"""LEAP reorientation 当前配置中的动作尺度锚点 $1/24$。"""


def compute_leap_adr_latency_steps(
    latency: float | torch.Tensor,
    random_subtraction: torch.Tensor,
    max_latency: int = OFFICIAL_ADR_MAX_LATENCY,
) -> torch.Tensor:
    r"""把连续 ADR latency 强度 $h(k)$ 投影成离散 action history 索引。

    LEAP official action delay 不是 fractional delay，而是在每个 episode reset 时采样一个
    $r\in\{0,1\}$，随后计算：

    $$
    \ell=\operatorname{clip}\left(\max(0,\lfloor h(k)-r\rfloor),0,L_{max}\right).
    $$

    Args:
        latency (float | torch.Tensor): ADR curriculum 给出的连续延迟强度 $h(k)$，单位为 policy step。
        random_subtraction (torch.Tensor): episode-level 随机减项 $r$，通常形状为 $[N,1]$。
        max_latency (int): action history 可索引的最大延迟 $L_{max}$。

    Returns:
        torch.Tensor: 离散延迟索引 $\ell$，dtype 为 `torch.long`，形状跟 `random_subtraction` 一致。
    """

    # 将标量或张量形式的 $h(k)$ 放到与随机减项相同的 device，保证 reset 时不触发 CPU/GPU 混用。
    latency_tensor = torch.as_tensor(latency, device=random_subtraction.device, dtype=torch.float32)  # $h(k)$。

    # 复刻 official 公式中的 $\lfloor h(k)-r\rfloor$；`random_subtraction` 是 reset-time 采样的整数张量。
    latency_steps = torch.floor(latency_tensor - random_subtraction.float())  # 未裁剪的离散延迟索引。

    # `torch.clamp(min=0)` 已经实现公式里的 `max(0,·)`，再限制到 history buffer 的最大索引。
    latency_steps = torch.clamp(latency_steps, min=0, max=max_latency)  # 合法区间 $[0,L_{max}]$。
    return latency_steps.long()  # 作为 action history 的 gather index，必须是整数类型。


def compute_relative_joint_command(
    reference_positions: torch.Tensor,
    processed_deltas: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    r"""计算 relative raw-rad delta 对应的 joint command target。

    统一公式为：

    $$
    u_t=\operatorname{clip}(r_t+\Delta q_t,q_{min},q_{max}).
    $$

    `reference="target"` 时 $r_t=u_{t-1}$，对应 LEAP official relative；
    `reference="current"` 时 $r_t=q_t$，对应 IsaacLab current-relative。

    Args:
        reference_positions (torch.Tensor): 参考位置 $r_t$，形状 $[N,J]$，单位 rad。
        processed_deltas (torch.Tensor): raw-rad delta $\Delta q_t$，形状 $[N,J]$，单位 rad。
        lower_limits (torch.Tensor): soft joint lower limits $q_{min}$，形状 $[N,J]$，单位 rad。
        upper_limits (torch.Tensor): soft joint upper limits $q_{max}$，形状 $[N,J]$，单位 rad。

    Returns:
        torch.Tensor: 裁剪后的 command target $u_t$，形状 $[N,J]$，单位 rad。
    """

    # 先在物理量纲空间加 raw-rad delta，得到尚未投影到关节限位的目标。
    target_uncapped = reference_positions + processed_deltas  # $r_t+\Delta q_t$，单位 rad。

    # command target 必须显式投影到 soft limits；不要依赖 PhysX actuator 底层隐式兜底。
    return torch.clamp(target_uncapped, min=lower_limits, max=upper_limits)  # $u_t\in[q_{min},q_{max}]$。


def compute_ema_joint_command(
    absolute_targets: torch.Tensor,
    reference_positions: torch.Tensor,
    alpha: float | torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    r"""计算 joint-limit absolute target 的 EMA / blend command。

    absolute 动作先把 $a_t^{exec}$ 映射到 joint-limit 绝对目标 $v_t=S(a_t^{exec})$，再与参考点
    做一阶 blend：

    $$
    u_t=\operatorname{clip}\left(\alpha v_t+(1-\alpha)r_t,q_{min},q_{max}\right).
    $$

    Args:
        absolute_targets (torch.Tensor): joint-limit absolute target $v_t$，形状 $[N,J]$，单位 rad。
        reference_positions (torch.Tensor): blend 参考点 $r_t$，形状 $[N,J]$，单位 rad。
        alpha (float | torch.Tensor): EMA / blend 系数 $\alpha\in[0,1]$，可为标量或 $[N,J]$ 张量。
        lower_limits (torch.Tensor): soft joint lower limits $q_{min}$，形状 $[N,J]$，单位 rad。
        upper_limits (torch.Tensor): soft joint upper limits $q_{max}$，形状 $[N,J]$，单位 rad。

    Returns:
        torch.Tensor: 裁剪后的 command target $u_t$，形状 $[N,J]$，单位 rad。
    """

    # 一阶 EMA / blend：`reference_positions` 可以是上一帧 target buffer，也可以是当前真实关节角。
    blended_targets = alpha * absolute_targets + (1.0 - alpha) * reference_positions  # $\alpha v_t+(1-\alpha)r_t$。

    # 即使 $v_t$ 和 $r_t$ 理论上都在限位内，浮点和配置误差仍需最终投影到 soft limits。
    return torch.clamp(blended_targets, min=lower_limits, max=upper_limits)  # $u_t\in[q_{min},q_{max}]$。


class ADRJointAction(JointAction):
    r"""ADR-aware joint action 基类。

    该类只负责三件跨动作语义共享的事情：

    1. 解析 action joint order、scale、offset、clip 等 IsaacLab `JointAction` 公共字段；
    2. 复刻 LEAP official action noise / latency，得到 $a_t^{exec}$；
    3. 维护 `current_targets`、`previous_targets`、`pregrasp_targets` 这些 obs / reward 共享 contract。

    派生类负责把 $a_t^{exec}$ 解释为 raw-rad delta 或 joint-limit absolute target。
    """

    cfg: actions_cfg.JointActionCfg

    def __init__(self, cfg: actions_cfg.JointActionCfg, env) -> None:
        r"""初始化 ADR runtime buffer 与 action/target 共享状态。

        Args:
            cfg (actions_cfg.JointActionCfg): 派生 action cfg，必须含 `use_adr` 与 `pregrasp_joint_pos` 字段。
            env: IsaacLab ManagerBased env；action noise / latency 从其 runtime ADR 属性读取。
        """

        # 委托 IsaacLab 解析 joint_names / scale / offset / clip，保持与官方 action manager 兼容。
        super().__init__(cfg, env)

        # ADR 是 N4x train env 的默认语义；contract / smoke 可以通过 `use_adr=False` 获得确定性动作。
        self._use_adr = bool(getattr(cfg, "use_adr", True))  # 是否启用 official action noise / latency。

        # action history 第 0 层保存最新 noisy action，更大索引保存更旧 action，用于离散 latency gather。
        self._action_history = torch.zeros(
            self.num_envs,
            self.action_dim,
            OFFICIAL_ADR_MAX_LATENCY + 1,
            device=self.device,
        )  # 形状 $[N,J,L_{max}+1]$。

        # 每个 env 在 reset 时采样一个 latency index，然后扩展到所有 action joint。
        self._latency_steps = torch.zeros(
            self.num_envs,
            self.action_dim,
            dtype=torch.long,
            device=self.device,
        )  # 形状 $[N,J]$，元素为 $\ell$。

        # `executed_actions` 是 official reward 的 action penalty 所需量，也是 action law 的共同输入。
        self._executed_actions = torch.zeros_like(self._raw_actions)  # $a_t^{exec}$，形状 $[N,J]$，无量纲。

        # target buffer 支撑 `reference="target"`，同时作为 official obs 的 current target 读出源。
        self._previous_targets = torch.zeros_like(self._raw_actions)  # $u_{t-1}$，形状 $[N,J]$，单位 rad。
        self._current_targets = torch.zeros_like(self._raw_actions)  # $u_t$，形状 $[N,J]$，单位 rad。

        # soft joint limits 是所有 position command 的最终投影边界，单位为 rad。
        self._joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]  # $[N,J,2]$。
        self._joint_lower = self._joint_limits[..., 0]  # $q_{min}$，形状 $[N,J]$。
        self._joint_upper = self._joint_limits[..., 1]  # $q_{max}$，形状 $[N,J]$。

        # pregrasp anchor 供 official reward 的 pose-diff penalty 复用；正式 inhand env 应显式配置该向量。
        pregrasp_joint_pos = tuple(float(v) for v in getattr(cfg, "pregrasp_joint_pos", ()))  # 配置中的 $q^{pre}$。
        if pregrasp_joint_pos:
            if len(pregrasp_joint_pos) != self.action_dim:
                raise ValueError(
                    f"pregrasp_joint_pos length {len(pregrasp_joint_pos)} must match action_dim {self.action_dim}."
                )
            pregrasp = torch.tensor(pregrasp_joint_pos, device=self.device, dtype=torch.float32).unsqueeze(0)  # $[1,J]$。
            self._pregrasp_targets = pregrasp.expand(self.num_envs, -1).clone()  # $[N,J]$，单位 rad。
        else:
            self._pregrasp_targets = self._asset.data.default_joint_pos[:, self._joint_ids].clone()  # fallback $[N,J]$。

    @property
    def previous_targets(self) -> torch.Tensor:
        r"""上一帧 command target buffer $u_{t-1}$，形状 `[N,J]`，单位 rad。"""

        return self._previous_targets

    @property
    def current_targets(self) -> torch.Tensor:
        r"""当前帧实际下发的 command target $u_t$，形状 `[N,J]`，单位 rad。"""

        return self._current_targets

    @property
    def executed_actions(self) -> torch.Tensor:
        r"""ADR corruption 后的 normalized executed action $a_t^{exec}$，形状 `[N,J]`。"""

        return self._executed_actions

    @property
    def pregrasp_targets(self) -> torch.Tensor:
        r"""official pregrasp anchor $q^{pre}$，形状 `[N,J]`，单位 rad。"""

        return self._pregrasp_targets

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        r"""把 IsaacLab reset 传入的 env ids 统一成一维 `torch.long` 张量。"""

        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)  # 全环境 reset。
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()  # 局部 reset。

    def _reset_targets_from_env(self, env_ids: torch.Tensor) -> torch.Tensor:
        r"""读取 reset 后应写入 target buffer 的 joint pose。

        `reset_adr_robot_joints` 会把真实 reset joint pose 写到 `env.leap_official_reset_joint_pos`；优先使用
        该值可确保第一帧 $u_0=q_0$，避免 target buffer 与仿真状态错位。
        """

        if hasattr(self._env, "leap_official_reset_joint_pos"):
            return self._env.leap_official_reset_joint_pos[env_ids].clone()  # reset event 记录的 $q_0$，形状 $[N,J]$。
        return self._asset.data.default_joint_pos[env_ids][:, self._joint_ids].clone()  # fallback 资产默认位姿。

    def _sample_latency_steps(self, env_ids: torch.Tensor) -> None:
        r"""按 official 公式为本批 reset env 采样 episode-level latency index。"""

        if not self._use_adr:
            self._latency_steps[env_ids] = 0  # 非 ADR smoke / contract 下使用零延迟。
            return

        # 当前 ADR 档位的连续 latency 强度由 curriculum 写到 env runtime 属性上。
        latency_float = float(getattr(self._env, "leap_adr_action_latency", 0.0))  # $h(k)$，单位 policy step。

        # official 每个 episode 采样一个 $r\in\{0,1\}$，同一 env 的所有 joint 共享该随机减项。
        random_sub = torch.randint(
            0,
            OFFICIAL_ADR_LATENCY_RAND + 1,
            (len(env_ids), 1),
            device=self.device,
        )  # $r$，形状 $[N_{reset},1]$。

        # 将标量 $h(k)$ 与随机减项投影到离散 history index，并扩展到所有 action joint。
        latency = compute_leap_adr_latency_steps(latency_float, random_sub, OFFICIAL_ADR_MAX_LATENCY)  # $[N,1]$。
        self._latency_steps[env_ids] = latency.expand(-1, self.action_dim)  # $[N,J]$。

    def _update_executed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        r"""从 policy action 生成 official ADR 后的 executed action。

        Args:
            actions (torch.Tensor): policy raw action，形状 `[N,J]`，通常由 rl_games wrapper 裁剪到 $[-1,1]$。

        Returns:
            torch.Tensor: executed action $a_t^{exec}$，形状 `[N,J]`，范围 $[-1,1]$。
        """

        # `raw_actions` 保留 policy-facing 输入，供 IsaacLab / rl_games 的 last_action 语义和调试使用。
        self._raw_actions[:] = actions  # $a_t$，无量纲。

        if self._use_adr:
            # action noise std 随 ADR 档位变化，由 curriculum 写入 env runtime 属性。
            action_noise = float(getattr(self._env, "leap_adr_action_noise", 0.0))  # $\sigma_a(k)$。
            noisy_actions = actions + torch.randn_like(actions) * action_noise  # $\tilde a_t=a_t+\eta_t$。

            # 历史窗口整体右移，index 0 始终保存当前 policy step 的 noisy action。
            self._action_history = torch.roll(self._action_history, shifts=1, dims=2)  # 时间维右移一格。
            self._action_history[:, :, 0] = noisy_actions  # 写入 $\tilde a_t$。

            # 按每个 env/joint 的 latency index 从三维 history 中 gather $\tilde a_{t-\ell}$。
            env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.action_dim)  # $[N,J]$。
            act_ids = torch.arange(self.action_dim, device=self.device).unsqueeze(0).expand(self.num_envs, -1)  # $[N,J]$。
            delayed_actions = self._action_history[env_ids, act_ids, self._latency_steps]  # $\tilde a_{t-\ell}$。
        else:
            delayed_actions = actions  # 非 ADR 路径只关闭 noise / latency，不改变 action law 本身。

        # LEAP official 在进入 action law 前把执行动作裁剪回 normalized action domain。
        self._executed_actions[:] = torch.clamp(delayed_actions, -1.0, 1.0)  # $a_t^{exec}\in[-1,1]^J$。
        return self._executed_actions

    def _reference_positions(self, reference: ActionReference) -> torch.Tensor:
        r"""按 `reference` 枚举读取动作更新律的参考点 $r_t$。"""

        if reference == "current":
            return self._asset.data.joint_pos[:, self._joint_ids]  # $q_t$，真实当前关节角，单位 rad。
        if reference == "target":
            return self._previous_targets  # $u_{t-1}$，上一帧 command target，单位 rad。
        raise ValueError(f"Unsupported action reference '{reference}'. Expected 'current' or 'target'.")

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        r"""重置 action history、target buffer 与 latency index。"""

        # IsaacLab 可能传 `None`、`slice(None)`、list 或 tensor；统一成可索引张量。
        env_ids_tensor = self._resolve_env_ids(env_ids)  # reset env ids，形状 $[N_{reset}]$。

        # reset 起点的 policy action、processed action 和 executed action 都归零，避免跨 episode 泄漏。
        self._raw_actions[env_ids_tensor] = 0.0  # $a_0=0$。
        self._processed_actions[env_ids_tensor] = 0.0  # 派生类会在下一次 process/apply 中重写。
        self._executed_actions[env_ids_tensor] = 0.0  # $a_0^{exec}=0$。
        self._action_history[env_ids_tensor] = 0.0  # 清空 action latency history。

        # target buffer 初始化为 reset 后真实关节角，保证 official obs 中的 target 与仿真状态对齐。
        reset_targets = self._reset_targets_from_env(env_ids_tensor)  # $q_0$，形状 $[N_{reset},J]$。
        self._previous_targets[env_ids_tensor] = reset_targets  # $u_{-1}=q_0$。
        self._current_targets[env_ids_tensor] = reset_targets  # $u_0=q_0$。

        # 每个 episode 重采样 latency；非 ADR 路径会把索引显式设为 0。
        self._sample_latency_steps(env_ids_tensor)


class ADRRelativeJointPositionAction(ADRJointAction):
    r"""ADR-aware relative raw-rad delta action。

    policy 输出经 ADR 后仍是 normalized executed action $a_t^{exec}$，随后 affine 变换成 raw-rad delta：

    $$
    \Delta q_t=s a_t^{exec}+b.
    $$

    最终 command target 由 `reference` 决定：

    $$
    u_t=\operatorname{clip}(r_t+\Delta q_t,q_{min},q_{max}).
    $$
    """

    cfg: ADRRelativeJointPositionActionCfg

    def __init__(self, cfg: ADRRelativeJointPositionActionCfg, env) -> None:
        r"""初始化 relative action，并确保 relative 默认无 offset。"""

        super().__init__(cfg, env)
        if cfg.use_zero_offset:
            self._offset = 0.0  # relative raw delta 的物理零点必须是 $\Delta q=0$。

    def process_actions(self, actions: torch.Tensor) -> None:
        r"""把 policy action 变成带量纲 raw-rad delta。"""

        # 先复刻 official action noise / latency，得到 normalized executed action $a_t^{exec}$。
        executed_actions = self._update_executed_actions(actions)  # $[N,J]$，无量纲。

        # 与 IsaacLab JointAction 一致：scale / offset 在 executed action 上执行，结果单位为 rad delta。
        self._processed_actions = executed_actions * self._scale + self._offset  # $\Delta q_t$，单位 rad。

        # 可选 `clip` 是 raw-rad delta 的安全网，不改变默认 $s=1/24$ 的主尺度语义。
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )  # $\Delta q_t\in[c_{lo},c_{hi}]$。

    def apply_actions(self) -> None:
        r"""按 `reference` 声明计算并下发 relative command target。"""

        # 读取 $r_t$：current 分支为真实关节角，target 分支为上一帧 command target buffer。
        reference_positions = self._reference_positions(self.cfg.reference)  # $r_t$，形状 $[N,J]$。

        # 计算 $u_t=\operatorname{clip}(r_t+\Delta q_t,q_{min},q_{max})$。
        self._current_targets = compute_relative_joint_command(
            reference_positions=reference_positions,
            processed_deltas=self._processed_actions,
            lower_limits=self._joint_lower,
            upper_limits=self._joint_upper,
        )  # $u_t$，形状 $[N,J]$，单位 rad。

        # 下发给 articulation PD controller，并把本帧 command 写回 target buffer。
        self._asset.set_joint_position_target(self._current_targets, joint_ids=self._joint_ids)  # PD setpoint $u_t$。
        self._previous_targets[:] = self._current_targets  # $u_{t-1}\leftarrow u_t$，供下一帧 target reference 使用。


class ADREMAJointPositionToLimitsAction(ADRJointAction):
    r"""ADR-aware joint-limit absolute target EMA action。

    该动作项先把 normalized executed action 映射为 joint-limit absolute target：

    $$
    v_t=S(a_t^{exec})=q_{min}+\frac{a_t^{exec}+1}{2}(q_{max}-q_{min}),
    $$

    再根据 `reference` 做 EMA / blend：

    $$
    u_t=\operatorname{clip}(\alpha v_t+(1-\alpha)r_t,q_{min},q_{max}).
    $$
    """

    cfg: ADREMAJointPositionToLimitsActionCfg

    def __init__(self, cfg: ADREMAJointPositionToLimitsActionCfg, env) -> None:
        r"""初始化 absolute EMA action，并解析 $\alpha$。"""

        super().__init__(cfg, env)
        self._alpha = self._parse_alpha(cfg.alpha)  # EMA / blend 系数 $\alpha$，标量或 $[N,J]$。
        self._absolute_targets = torch.zeros_like(self._raw_actions)  # $v_t=S(a_t^{exec})$，单位 rad。

    def _parse_alpha(self, alpha: float | dict[str, float]) -> float | torch.Tensor:
        r"""解析 EMA / blend 系数 $\alpha$，支持全局标量或按 joint regex 配置。"""

        if isinstance(alpha, (float, int)):
            alpha_float = float(alpha)  # 全局 EMA 系数。
            if not 0.0 <= alpha_float <= 1.0:
                raise ValueError(f"EMA alpha must be in [0, 1], got {alpha_float}.")
            return alpha_float

        if isinstance(alpha, dict):
            alpha_tensor = torch.ones((self.num_envs, self.action_dim), device=self.device)  # 默认 $\alpha=1$。
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(alpha, self._joint_names)
            for name, value in zip(names_list, value_list, strict=True):
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"EMA alpha for joint '{name}' must be in [0, 1], got {value}.")
            alpha_tensor[:, index_list] = torch.tensor(value_list, device=self.device)  # 写入 per-joint $\alpha_i$。
            return alpha_tensor

        raise ValueError(f"Unsupported alpha type: {type(alpha)}. Expected float or dict[str, float].")

    def process_actions(self, actions: torch.Tensor) -> None:
        r"""把 policy action 映射为 joint-limit absolute target $v_t$。"""

        # 先复刻 official action noise / latency，得到 normalized executed action $a_t^{exec}$。
        executed_actions = self._update_executed_actions(actions)  # $[N,J]$，无量纲。

        # scale / offset 在 normalized action 空间执行；默认 scale=1, offset=0，等价于直接使用 $a_t^{exec}$。
        normalized_targets = executed_actions * self._scale + self._offset  # 仍是无量纲动作坐标。

        # 可选 `clip` 作用在 unscale 前的 normalized target 上，常用于限制 absolute target 搜索区间。
        if self.cfg.clip is not None:
            normalized_targets = torch.clamp(
                normalized_targets,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )  # normalized target 安全裁剪。

        # joint-limit absolute action 的定义域固定为 $[-1,1]$，再 unscale 到 soft limits。
        normalized_targets = torch.clamp(normalized_targets, -1.0, 1.0)  # $a\in[-1,1]$。
        self._absolute_targets = math_utils.unscale_transform(
            normalized_targets,
            self._joint_lower,
            self._joint_upper,
        )  # $v_t=S(a_t^{exec})$，单位 rad。

        # 在 process 阶段先把 processed_actions 记录为 $v_t$；apply 后会更新为实际 command target $u_t$。
        self._processed_actions = self._absolute_targets.clone()  # 便于调试 absolute target 映射。

    def apply_actions(self) -> None:
        r"""按 `reference` 声明计算并下发 EMA absolute command target。"""

        # 读取 $r_t$：target 分支是真正 EMA，current 分支是用户提出的 current-blended absolute。
        reference_positions = self._reference_positions(self.cfg.reference)  # $r_t$，形状 $[N,J]$。

        # 计算 $u_t=\operatorname{clip}(\alpha v_t+(1-\alpha)r_t,q_{min},q_{max})$。
        self._current_targets = compute_ema_joint_command(
            absolute_targets=self._absolute_targets,
            reference_positions=reference_positions,
            alpha=self._alpha,
            lower_limits=self._joint_lower,
            upper_limits=self._joint_upper,
        )  # $u_t$，形状 $[N,J]$，单位 rad。

        # 下发给 articulation PD controller，并把本帧 command 写回 target buffer。
        self._asset.set_joint_position_target(self._current_targets, joint_ids=self._joint_ids)  # PD setpoint $u_t$。
        self._previous_targets[:] = self._current_targets  # $u_{t-1}\leftarrow u_t$。
        self._processed_actions = self._current_targets.clone()  # processed action 对 absolute EMA 表示实际 command target。


@configclass
class ADRRelativeJointPositionActionCfg(actions_cfg.RelativeJointPositionActionCfg):
    r"""ADR-aware relative raw-rad delta action cfg。

    默认数值锚点沿用 LEAP reorientation current config：

    $$
    s=\frac{1}{24}\ \mathrm{rad}.
    $$
    """

    class_type: type[ActionTerm] = ADRRelativeJointPositionAction
    reference: ActionReference = "current"
    use_adr: bool = True
    scale: float | dict[str, float] = LEAP_ACTION_SCALE
    pregrasp_joint_pos: tuple[float, ...] = ()


@configclass
class ADREMAJointPositionToLimitsActionCfg(actions_cfg.JointActionCfg):
    r"""ADR-aware joint-limit absolute EMA action cfg。

    默认 $\alpha=1/24$ 贴近 LEAP reorientation `act_moving_average` 的数值语境；若要复刻
    IsaacLab Allegro inhand 的快速 EMA，可在实验配置中显式改成 $\alpha=0.95$。
    """

    class_type: type[ActionTerm] = ADREMAJointPositionToLimitsAction
    reference: ActionReference = "target"
    use_adr: bool = True
    alpha: float | dict[str, float] = LEAP_ACTION_SCALE
    scale: float | dict[str, float] = 1.0
    pregrasp_joint_pos: tuple[float, ...] = ()


__all__ = [
    "ADREMAJointPositionToLimitsAction",
    "ADREMAJointPositionToLimitsActionCfg",
    "ADRJointAction",
    "ADRRelativeJointPositionAction",
    "ADRRelativeJointPositionActionCfg",
    "ActionReference",
    "LEAP_ACTION_SCALE",
    "OFFICIAL_ADR_LATENCY_RAND",
    "OFFICIAL_ADR_MAX_LATENCY",
    "compute_ema_joint_command",
    "compute_leap_adr_latency_steps",
    "compute_relative_joint_command",
]
