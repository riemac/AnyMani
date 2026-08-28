r"""heterogeneous canonical hand 的 policy-step masked relative joint action。

本动作只服务异构基础设施阶段，不承载 ADR、latency、noise、EMA 或动作 curriculum。策略仍输出
固定 16 槽向量；每只手的 active mask 把它投影到真实关节子空间。一次 20 Hz policy step 执行：

$$
\Delta q_t = m\odot\operatorname{clip}(s a_t,-0.1,0.1),
\qquad
u_{t+1}=m\odot\operatorname{clip}(u_t+\Delta q_t,q_{min},q_{max}),
$$

随后六个 120 Hz physics substeps 只重复下发同一个 $u_{t+1}$。因此 decimation 改变 target 的
保持时长，不会让同一个 action 被累计六次。ghost 槽的 raw action、processed delta、target 与
last-action observation 均为零；PPO 概率侧由 ``distill.rl.masked_ppo`` 使用同一 mask 排除。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def compute_policy_step_masked_relative_target(
    previous_target: torch.Tensor,
    processed_delta: torch.Tensor,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    r"""计算一次 policy-step target 状态转移。

    Args:
        previous_target (torch.Tensor): 上一 policy step 的 PD target $u_t$，形状 `[B,16]`，单位 rad。
        processed_delta (torch.Tensor): scale/clip 后的增量 $\Delta q_t$，形状 `[B,16]`，单位 rad。
        lower_limit (torch.Tensor): soft lower limits $q_{min}$，形状 `[B,16]`，单位 rad。
        upper_limit (torch.Tensor): soft upper limits $q_{max}$，形状 `[B,16]`，单位 rad。
        active_mask (torch.Tensor): 真实关节 mask $m$，bool `[B,16]`。

    Returns:
        torch.Tensor: 下一 target $u_{t+1}$，形状 `[B,16]`，ghost 恒为 0 rad。
    """

    expected = previous_target.shape  # 五个输入必须共享 `[batch,joint]` 轴
    tensors = (processed_delta, lower_limit, upper_limit, active_mask)
    if previous_target.ndim != 2 or any(tensor.shape != expected for tensor in tensors):
        raise ValueError("target, delta, limits and active_mask must share rank-2 [B,J] shape")
    if active_mask.dtype != torch.bool:
        raise TypeError("active_mask must be bool")
    active_delta = processed_delta * active_mask.to(dtype=processed_delta.dtype)  # $m\odot\Delta q_t$
    unclamped = previous_target + active_delta  # target-reference accumulator $u_t+\Delta q_t$
    bounded = torch.clamp(unclamped, min=lower_limit, max=upper_limit)  # 投影到 soft joint interval
    return torch.where(active_mask, bounded, torch.zeros_like(bounded))  # ghost target 固定 $0$ rad


def _select_joint_rows(
    joint_values: torch.Tensor,
    env_ids: Sequence[int] | torch.Tensor | None,
    joint_ids: Sequence[int] | torch.Tensor | slice,
) -> torch.Tensor:
    r"""按 outer-product 语义选择 ``[env,joint]`` 子矩阵。

    PyTorch ``x[env_ids, joint_ids]`` 把两个 tensor 都视为 advanced indices，并要求它们可广播；
    ``[K]`` 与 ``[16]`` 在 full reset 时会直接 shape mismatch。分两步索引明确得到 `[K,16]`。
    """

    env_rows = joint_values if env_ids is None else joint_values[env_ids]  # `[K,J_all]`
    return env_rows[:, joint_ids]  # `[K,J_action]`，不是 pairwise advanced indexing


class PolicyStepMaskedRelativeJointPositionAction(RelativeJointPositionAction):
    r"""每个 policy step 推进一次、physics substeps 幂等 hold 的 16 槽动作。

    active mask 必须由 canonical startup event 安装到
    ``env._anymani_canonical_active_joint_mask``。缺失或 shape 不匹配时 fail closed，避免把 ghost
    静默当作真实关节。该类沿用 IsaacLab ``JointAction`` 对 scale/clip 的解析，只改变 target
    状态转移发生在 ``process_actions`` 而不是 ``apply_actions`` 的时间语义。
    """

    cfg: PolicyStepMaskedRelativeJointPositionActionCfg
    """heterogeneous infra-stage action 配置。"""

    def __init__(self, cfg: PolicyStepMaskedRelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        r"""初始化固定 target buffers；首次 environment reset 会写入真实 canonical home pose。"""

        super().__init__(cfg, env)  # 解析 16 个 joint、scale、offset 与 per-step clip
        self._env = env  # canonical active mask 的 runtime owner
        self._current_targets = torch.zeros_like(self.raw_actions)  # $u_t$，`[B,16]`，单位 rad
        self._previous_targets = torch.zeros_like(self.raw_actions)  # 上一提交 target，供 diagnostics/checkpoint
        self._executed_actions = torch.zeros_like(self.raw_actions)  # $a_t^{exec}$，无 ADR/noise/latency 的 masked raw action

    @property
    def current_targets(self) -> torch.Tensor:
        r"""返回当前 policy-step PD target $u_t$，形状 `[B,16]`，单位 rad。"""

        return self._current_targets

    @property
    def previous_targets(self) -> torch.Tensor:
        r"""返回上一次 target buffer；process 后与 current target 同值。"""

        return self._previous_targets

    @property
    def executed_actions(self) -> torch.Tensor:
        r"""返回实际执行的无量纲 policy action，形状 `[B,16]`，ghost 恒为零。

        infra action 不含 ADR noise、latency 或 clipping；因此 $a_t^{exec}=m\odot a_t^{policy}$。
        该接口供 N000 diagnostics 计算实际 action rate，不改变 PPO probability contract。
        """

        return self._executed_actions

    def _active_mask(self) -> torch.Tensor:
        r"""读取并裁剪当前 action term 的 env-level canonical active mask。"""

        full_mask = getattr(self._env, "_anymani_canonical_active_joint_mask", None)
        if not isinstance(full_mask, torch.Tensor):
            raise RuntimeError("canonical active joint mask must be installed before action processing")
        if full_mask.ndim != 2 or full_mask.shape[0] != self.num_envs:
            raise RuntimeError("canonical active joint mask must have shape [num_envs,num_joints]")
        mask = full_mask[:, self._joint_ids]  # `[B,J_action]`，与 preserve_order joint axis 对齐
        if mask.shape != self.raw_actions.shape or mask.dtype != torch.bool:
            raise RuntimeError("canonical active joint mask disagrees with the action joint axis")
        return mask

    def process_actions(self, actions: torch.Tensor) -> None:
        r"""把一次 policy sample 提交为一次 target-buffer 状态转移。

        ``ActionManager`` 每个 environment step 只调用一次本方法；因此 $u_t\to u_{t+1}$ 与
        20 Hz policy rate 对齐，不随六次 physics ``apply_actions`` 重复累计。
        """

        active_mask = self._active_mask()  # `[B,16]`，真实 source joint 子空间
        masked_actions = actions * active_mask.to(dtype=actions.dtype)  # ghost raw action 进入 term 前即清零
        self._executed_actions[:] = masked_actions  # 无 ADR route：实际执行 action 即 masked policy sample
        super().process_actions(masked_actions)  # processed delta $\Delta q_t=s a_t$，单位 rad
        self._processed_actions *= active_mask.to(dtype=self._processed_actions.dtype)  # 数值防御：ghost delta=0
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]  # `[B,16,2]`，单位 rad
        next_targets = compute_policy_step_masked_relative_target(
            self._current_targets,
            self._processed_actions,
            limits[..., 0],
            limits[..., 1],
            active_mask,
        )  # $u_{t+1}$，只在本 policy step 计算一次
        self._previous_targets[:] = self._current_targets  # 提交前 target，供 action-rate/diagnostics 读取
        self._current_targets[:] = next_targets  # 新 target 成为六个 substeps 的固定 setpoint

    def apply_actions(self) -> None:
        r"""幂等下发当前 target；调用次数只决定物理保持时长。"""

        self._asset.set_joint_position_target(self._current_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""把被 reset 环境的 action 和 target 对齐到 event 已写入的 canonical joint pose。

        Args:
            env_ids (Sequence[int] | None): 需要 reset 的环境；``None`` 表示全部环境。
        """

        selection = slice(None) if env_ids is None else env_ids  # partial reset 只修改被选 rows
        active_mask = self._active_mask()[selection]  # reset event 已先安装/更新 canonical mask
        self._raw_actions[selection] = 0.0  # 上一 policy sample 清零
        self._executed_actions[selection] = 0.0  # diagnostics 的实际 action snapshot 清零
        self._processed_actions[selection] = 0.0  # last executed rad delta 清零
        joint_pos = _select_joint_rows(
            self._asset.data.joint_pos,
            env_ids,
            self._joint_ids,
        )  # event 后真实 $q_0$，形状 `[K,16]`，单位 rad
        reset_target = torch.where(active_mask, joint_pos, torch.zeros_like(joint_pos))  # ghost $u_0=0$
        self._current_targets[selection] = reset_target
        self._previous_targets[selection] = reset_target


@configclass
class PolicyStepMaskedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    r"""heterogeneous infra-stage 的固定 16 槽相对 target 配置。

    preset：``scale=0.1 rad``，``clip=±0.1 rad``，``preserve_order=True``，
    ``use_zero_offset=True``。这是当前容量/基础设施实验的动作锚点，不冻结未来正式 PPO action 研究。
    """

    class_type: type[ActionTerm] = PolicyStepMaskedRelativeJointPositionAction
    """ActionManager 实例化的真实 term 类。"""

    scale: float = 0.1
    r"""policy raw action 到每策略步关节增量的尺度，单位 rad。"""

    clip: dict[str, tuple[float, float]] | None = None
    r"""可选 processed delta 安全限幅；环境 preset 显式设置 ``{'.*':(-0.1,0.1)}``。"""

    preserve_order: bool = True
    """保持 canonical importer 的 depth-major 16 槽顺序。"""

    use_zero_offset: bool = True
    """相对增量的物理零点固定为 0 rad。"""


__all__ = [
    "PolicyStepMaskedRelativeJointPositionAction",
    "PolicyStepMaskedRelativeJointPositionActionCfg",
    "compute_policy_step_masked_relative_target",
]
