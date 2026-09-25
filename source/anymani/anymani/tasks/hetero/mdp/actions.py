(
    'Preload-aware masked relative joint action, updated once per policy step. '
    'Each 20 Hz action in [-1,1]^16 changes the target by at most 1/24 rad; six '
    '120 Hz physics steps hold that target. Reset restores the independent '
    'preload target q_t from the pregrasp sidecar instead of inferring it from '
    'joint position, preserving the certified contact basin.'
)

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

POLICY_STEP_AUTHORITY_RAD = 1.0 / 24.0  # Sealed baseline: at most 2.387 deg per policy step.


class PreloadAwareMaskedRelativeJointPositionAction(RelativeJointPositionAction):
    (
        'Canonical action term: one target update per policy step, then an idempotent '
        'hold for six physics substeps. The pregrasp event must install a valid '
        'full-size sidecar for every active env. Ghost actions stay zero at every '
        'stage, and soft limits apply only to real joints. No ADR, latency, noise, or '
        'EMA state is added to partial-reset history.'
    )

    cfg: PreloadAwareMaskedRelativeJointPositionActionCfg  # Task-specific action configuration.

    def __init__(self, cfg: PreloadAwareMaskedRelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        'Resolve the canonical joint axis and allocate controller target buffers.'

        super().__init__(cfg, env)  # Isaac Lab resolves joint names, scale, offset, and soft clip.
        self._env = env  # The pregrasp sidecar is owned by the same ManagerBased env.
        if self.action_dim != CANONICAL_JOINT_COUNT:
            raise ValueError("heterogeneous canonical action must resolve exactly 16 joints")
        if not isinstance(cfg.scale, (float, int)) or abs(float(cfg.scale) - POLICY_STEP_AUTHORITY_RAD) > 1.0e-12:
            raise ValueError("heterogeneous baseline action scale must remain exactly 1/24 rad")
        self._current_targets = torch.zeros_like(self.raw_actions)  # Controller target u_t: [N,16], rad.
        self._previous_targets = torch.zeros_like(self.raw_actions)  # Previous policy-step setpoint.
        self._pregrasp_targets = torch.zeros_like(self.raw_actions)  # Latest reset-authenticated preload target q_t.
        self._executed_actions = torch.zeros_like(self.raw_actions)  # Clipped, masked, dimensionless action in [-1,1].

    @property
    def current_targets(self) -> torch.Tensor:
        'Return the current PD target u_t, shape [N,16], in rad.'

        return self._current_targets

    @property
    def previous_targets(self) -> torch.Tensor:
        'Return the target before this policy transition, shape [N,16].'

        return self._previous_targets

    @property
    def pregrasp_targets(self) -> torch.Tensor:
        'Return the PD preload target q_t installed by the latest reset.'

        return self._pregrasp_targets

    @property
    def executed_actions(self) -> torch.Tensor:
        'Return the clipped, ghost-masked dimensionless policy action.'

        return self._executed_actions

    def _sidecar(self) -> HeterogeneousPregraspState:
        'Read the pregrasp state published by the event; fail closed if missing.'

        state = getattr(self._env, HETERO_PREGRASP_STATE_ATTR, None)
        if not isinstance(state, HeterogeneousPregraspState):
            raise RuntimeError("heterogeneous pregrasp reset state must be installed before action use")
        if state.num_envs != self.num_envs or state.device != self.raw_actions.device:
            raise RuntimeError("pregrasp sidecar disagrees with action environment/device")
        return state

    def _active_mask(self) -> torch.Tensor:
        'Return the valid-joint mask aligned with action order, shape bool [N,16].'

        state = self._sidecar()
        if not bool(state.valid.all().item()):
            raise RuntimeError("all environments must resolve pregrasp before policy action processing")
        mask = state.active_joint_mask[:, self._joint_ids]  # This two-stage operation needs only full-env and joint slices.
        if mask.shape != self.raw_actions.shape or mask.dtype != torch.bool:
            raise RuntimeError("pregrasp active mask disagrees with canonical action axis")
        return mask

    def process_actions(self, actions: torch.Tensor) -> None:
        (
            'Apply one policy sample as exactly one target-accumulator transition: delta '
            'q = m * clip(a,-1,1) / 24 rad; then clip u + delta q to the active soft '
            'limits.'
        )

        if actions.shape != self.raw_actions.shape or not bool(torch.isfinite(actions).all().item()):
            raise ValueError("policy actions must be finite and have shape [num_envs,16]")
        active_mask = self._active_mask()  # Active action subspace m.
        bounded_actions = torch.clamp(actions, min=-1.0, max=1.0)  # Task action space [-1,1].
        masked_actions = bounded_actions * active_mask.to(dtype=actions.dtype)  # ghost raw action=0
        self._executed_actions[:] = masked_actions
        super().process_actions(masked_actions)  # Delta q_t = a_t / 24 rad.
        self._processed_actions *= active_mask.to(dtype=self._processed_actions.dtype)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]  # Joint limits [N,16,2], rad.
        next_targets = compute_policy_step_masked_relative_target(
            self._current_targets,
            self._processed_actions,
            limits[..., 0],
            limits[..., 1],
            active_mask,
        )
        self._previous_targets[:] = self._current_targets  # Transition input target u_t.
        self._current_targets[:] = next_targets  # Transition output target u_(t+1).

    def apply_actions(self) -> None:
        'Idempotently write the current target; repeated substeps do not reapply the policy delta.'

        self._asset.set_joint_position_target(self._current_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        (
            "Restore the reset rows' preload target from the event sidecar and clear "
            'stale actions. ManagerBasedRLEnv calls this after reset events, so the '
            'controller target remains q_t instead of reverting to actual joint state '
            'q_s.'
        )

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
        reset_target = self._current_targets[ids]  # Sidecar target q_t: [K,16].
        if not bool(torch.equal(reset_target[~mask], torch.zeros_like(reset_target[~mask]))):
            raise RuntimeError("ghost pregrasp targets must remain exactly zero")
        # Isaac Lab annotates Sequence, but partial outer indexing with env_ids[:, None] requires a device tensor.
        self._asset.set_joint_position_target(
            reset_target, joint_ids=self._joint_ids, env_ids=ids  # type: ignore[arg-type]
        )
        self._asset.set_joint_velocity_target(
            torch.zeros_like(reset_target), joint_ids=self._joint_ids, env_ids=ids  # type: ignore[arg-type]
        )


@configclass
class PreloadAwareMaskedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    'Canonical v1: 16 slots, at most 1/24 rad per policy step, held for six physics substeps.'

    class_type: type[ActionTerm] = PreloadAwareMaskedRelativeJointPositionAction  # Actual ActionManager term.
    scale: float = POLICY_STEP_AUTHORITY_RAD  # Fixed scale from raw [-1,1] to rad increments.
    clip: dict[str, tuple[float, float]] | None = None  # The term first clips raw actions to [-1,1].
    preserve_order: bool = True  # Preserve canonical depth-major 16-slot joint order.
    use_zero_offset: bool = True  # Relative delta has no default-q offset.


__all__ = [
    "POLICY_STEP_AUTHORITY_RAD",
    "PreloadAwareMaskedRelativeJointPositionAction",
    "PreloadAwareMaskedRelativeJointPositionActionCfg",
]
