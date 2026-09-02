r"""Preload-aware、mask-aware的heterogeneous policy-step relative joint action。

策略每20 Hz step提交一次$[-1,1]^{16}$动作，得到最多$1/24$ rad的target增量；随后的六个120 Hz
physics substeps只重复下发同一target。Reset不从actual joint position推断target，而从pregrasp sidecar恢复
独立$\mathbf q_t$，从而保留认证contact basin所需的隐式PD preload。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .runtime_state import (
    CANONICAL_JOINT_COUNT,
    HETERO_PREGRASP_STATE_ATTR,
    HeterogeneousPregraspState,
    compute_policy_step_masked_relative_target,
    normalize_env_ids,
    synchronize_action_reset,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

POLICY_STEP_AUTHORITY_RAD = 1.0 / 24.0  # sealed baseline：每policy step最多$2.387^\circ$


class PreloadAwareMaskedRelativeJointPositionAction(RelativeJointPositionAction):
    r"""一次policy-step推进一次、六个physics substeps幂等hold的canonical action term。

    Runtime sidecar必须已由pregrasp event安装并对所有活动环境置``valid``。Ghost actions在raw、processed、
    executed和target四个层级均为零；soft joint limits只约束真实joint。首版不包含ADR、latency、noise或EMA，
    避免这些状态在partial reset中引入另一组历史buffer。
    """

    cfg: PreloadAwareMaskedRelativeJointPositionActionCfg  # task-specific action配置

    def __init__(self, cfg: PreloadAwareMaskedRelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        r"""解析canonical joint axis并分配controller target buffers。"""

        super().__init__(cfg, env)  # IsaacLab解析joint names、scale、offset和soft clip
        self._env = env  # pregrasp sidecar由同一ManagerBased env持有
        if self.action_dim != CANONICAL_JOINT_COUNT:
            raise ValueError("heterogeneous canonical action must resolve exactly 16 joints")
        if not isinstance(cfg.scale, (float, int)) or abs(float(cfg.scale) - POLICY_STEP_AUTHORITY_RAD) > 1.0e-12:
            raise ValueError("heterogeneous baseline action scale must remain exactly 1/24 rad")
        self._current_targets = torch.zeros_like(self.raw_actions)  # $u_t\in\mathbb R^{N\times16}$，单位rad
        self._previous_targets = torch.zeros_like(self.raw_actions)  # 上一policy-step setpoint
        self._pregrasp_targets = torch.zeros_like(self.raw_actions)  # 最近reset认证的$\mathbf q_t$
        self._executed_actions = torch.zeros_like(self.raw_actions)  # clip/mask后的无量纲动作$[-1,1]$

    @property
    def current_targets(self) -> torch.Tensor:
        r"""返回当前PD target$u_t$，形状$[N,16]$，单位rad。"""

        return self._current_targets

    @property
    def previous_targets(self) -> torch.Tensor:
        r"""返回本policy transition之前的target，形状$[N,16]$。"""

        return self._previous_targets

    @property
    def pregrasp_targets(self) -> torch.Tensor:
        r"""返回最近一次reset安装的PD preload target$\mathbf q_t$。"""

        return self._pregrasp_targets

    @property
    def executed_actions(self) -> torch.Tensor:
        r"""返回经过$[-1,1]$裁剪和ghost mask后的实际无量纲policy action。"""

        return self._executed_actions

    def _sidecar(self) -> HeterogeneousPregraspState:
        r"""读取event发布的full-size pregrasp state；缺失时fail closed。"""

        state = getattr(self._env, HETERO_PREGRASP_STATE_ATTR, None)
        if not isinstance(state, HeterogeneousPregraspState):
            raise RuntimeError("heterogeneous pregrasp reset state must be installed before action use")
        if state.num_envs != self.num_envs or state.device != self.raw_actions.device:
            raise RuntimeError("pregrasp sidecar disagrees with action environment/device")
        return state

    def _active_mask(self) -> torch.Tensor:
        r"""返回与action joint order对齐的bool$[N,16]$有效关节mask。"""

        state = self._sidecar()
        if not bool(state.valid.all().item()):
            raise RuntimeError("all environments must resolve pregrasp before policy action processing")
        mask = state.active_joint_mask[:, self._joint_ids]  # 两阶段语义在这里只需全env+joint slice
        if mask.shape != self.raw_actions.shape or mask.dtype != torch.bool:
            raise RuntimeError("pregrasp active mask disagrees with canonical action axis")
        return mask

    def process_actions(self, actions: torch.Tensor) -> None:
        r"""把一个policy sample转换成恰好一次target accumulator transition。

        $$
        \Delta q_t=m\odot\frac{\operatorname{clip}(a_t,-1,1)}{24},\qquad
        u_{t+1}=m\odot\operatorname{clip}(u_t+\Delta q_t,q_{\min},q_{\max}).
        $$
        """

        if actions.shape != self.raw_actions.shape or not bool(torch.isfinite(actions).all().item()):
            raise ValueError("policy actions must be finite and have shape [num_envs,16]")
        active_mask = self._active_mask()  # 真实action子空间$m$
        bounded_actions = torch.clamp(actions, min=-1.0, max=1.0)  # 任务动作空间$[-1,1]$
        masked_actions = bounded_actions * active_mask.to(dtype=actions.dtype)  # ghost raw action=0
        self._executed_actions[:] = masked_actions
        super().process_actions(masked_actions)  # $\Delta q_t=a_t/24$，单位rad
        self._processed_actions *= active_mask.to(dtype=self._processed_actions.dtype)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]  # $[N,16,2]$，单位rad
        next_targets = compute_policy_step_masked_relative_target(
            self._current_targets,
            self._processed_actions,
            limits[..., 0],
            limits[..., 1],
            active_mask,
        )
        self._previous_targets[:] = self._current_targets  # transition左端$u_t$
        self._current_targets[:] = next_targets  # transition右端$u_{t+1}$

    def apply_actions(self) -> None:
        r"""幂等下发当前target；重复六次不会再次累加policy delta。"""

        self._asset.set_joint_position_target(self._current_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        r"""从event sidecar恢复selected rows的preload target并清除stale action。

        ``ManagerBasedRLEnv``在reset event之后调用本方法。因此这里重新写入$\mathbf q_t$，确保Isaac action
        lifecycle不会把controller target退化为actual$\mathbf q_s$。
        """

        ids = normalize_env_ids(env_ids, num_envs=self.num_envs, device=self.raw_actions.device)
        mask = synchronize_action_reset(
            env_ids=ids,
            sidecar=self._sidecar(),
            joint_ids=self._joint_ids,
            raw_actions=self._raw_actions,
            processed_actions=self._processed_actions,
            executed_actions=self._executed_actions,
            current_targets=self._current_targets,
            previous_targets=self._previous_targets,
            pregrasp_targets=self._pregrasp_targets,
        )
        reset_target = self._current_targets[ids]  # sidecar$\mathbf q_t$，形状$[K,16]$
        if not bool(torch.equal(reset_target[~mask], torch.zeros_like(reset_target[~mask]))):
            raise RuntimeError("ghost pregrasp targets must remain exactly zero")
        # IsaacLab注解写Sequence，但实现执行``env_ids[:,None]``；device Tensor才是partial outer-indexing实参。
        self._asset.set_joint_position_target(
            reset_target, joint_ids=self._joint_ids, env_ids=ids  # type: ignore[arg-type]
        )
        self._asset.set_joint_velocity_target(
            torch.zeros_like(reset_target), joint_ids=self._joint_ids, env_ids=ids  # type: ignore[arg-type]
        )


@configclass
class PreloadAwareMaskedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    r"""Canonical-v1 action配置：16 slots、每policy step$1/24$ rad、六substep hold。"""

    class_type: type[ActionTerm] = PreloadAwareMaskedRelativeJointPositionAction  # ActionManager真实term
    scale: float = POLICY_STEP_AUTHORITY_RAD  # raw$[-1,1]$到rad增量的固定比例
    clip: dict[str, tuple[float, float]] | None = None  # term内部已先裁剪raw$[-1,1]$
    preserve_order: bool = True  # 保持canonical depth-major 16-slot joint order
    use_zero_offset: bool = True  # relative delta没有default-q offset


__all__ = [
    "POLICY_STEP_AUTHORITY_RAD",
    "PreloadAwareMaskedRelativeJointPositionAction",
    "PreloadAwareMaskedRelativeJointPositionActionCfg",
]
