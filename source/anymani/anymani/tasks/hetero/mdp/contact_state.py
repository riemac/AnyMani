r"""Heterogeneous task唯一的20 Hz object-contact EMA与owner reduction状态。

所有actor、critic、reward和diagnostics读取同一snapshot。State sensor轴固定TIP4+finger-non-tip19+PALM；
owner轴固定PALM1+JOINT16+TIP4。每个sensor先取body/filter pair最大力，再做EMA与strict$>0.25$ N判定。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..contact_layout import HeterogeneousContactLayout, active_contact_sensor_mask
from ..contact_sensors import sensor_contact_magnitude
from .runtime_state import CANONICAL_OWNER_COUNT, HETERO_PREGRASP_STATE_ATTR, HeterogeneousPregraspState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

HETERO_CONTACT_STATE_ATTR = "_anymani_hetero_contact_state"


class HeterogeneousContactState:
    r"""Per-env shared contact snapshot，支持幂等policy update与partial reset。"""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        *,
        layout: HeterogeneousContactLayout,
        active_joint_mask_by_env: Sequence[Sequence[bool]],
        ema_alpha: float = 0.5,
        force_threshold_N: float = 0.25,
    ) -> None:
        r"""验证固定layout/routing并分配$[N,24]$ state buffers。"""

        if not 0.0 < ema_alpha <= 1.0 or force_threshold_N < 0.0:
            raise ValueError("contact EMA alpha/threshold must be valid")
        joint_mask = torch.tensor(active_joint_mask_by_env, dtype=torch.bool, device=env.device)
        if joint_mask.shape != (env.num_envs, 16):
            raise ValueError("contact state requires one canonical joint mask per env")
        self.layout = layout
        self.ema_alpha = float(ema_alpha)
        self.force_threshold_N = float(force_threshold_N)
        self.active_sensor_mask = active_contact_sensor_mask(joint_mask, layout)
        self.active_owner_mask = torch.cat(
            (
                torch.ones(env.num_envs, 1, dtype=torch.bool, device=env.device),
                joint_mask,
                joint_mask.reshape(env.num_envs, 4, 4).any(dim=1),
            ),
            dim=-1,
        )
        self.force_ema_N = torch.zeros(env.num_envs, len(layout.state_sensor_names), device=env.device)
        self.contact_bits = torch.zeros_like(self.force_ema_N, dtype=torch.bool)
        self.last_update_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._owner_indices = torch.tensor(layout.sensor_owner_indices, dtype=torch.long, device=env.device)

    @property
    def tip_force_ema_N(self) -> torch.Tensor:
        r"""返回$[N,4]$ TIP force EMA。"""

        return self.force_ema_N[:, :4]

    @property
    def tip_bits(self) -> torch.Tensor:
        r"""返回$[N,4]$ active TIP contact bits。"""

        return self.contact_bits[:, :4]

    @property
    def finger_non_tip_force_ema_N(self) -> torch.Tensor:
        r"""返回$[N,19]$ finger non-tip force EMA，不含PALM。"""

        return self.force_ema_N[:, 4:23]

    @property
    def finger_non_tip_bits(self) -> torch.Tensor:
        r"""返回$[N,19]$ finger non-tip bits。"""

        return self.contact_bits[:, 4:23]

    @property
    def palm_force_ema_N(self) -> torch.Tensor:
        r"""返回$[N,1]$合法PALM support force EMA。"""

        return self.force_ema_N[:, 23:24]

    @property
    def palm_bits(self) -> torch.Tensor:
        r"""返回$[N,1]$合法PALM support bits。"""

        return self.contact_bits[:, 23:24]

    def ensure_updated(self, env: ManagerBasedRLEnv) -> None:
        r"""当前common step中每env最多更新一次EMA。"""

        step = int(env.common_step_counter)
        update_mask = self.last_update_step != step
        if not bool(update_mask.any().item()):
            return
        raw_force = torch.stack(
            [sensor_contact_magnitude(env, sensor_name) for sensor_name in self.layout.state_sensor_names], dim=-1
        )
        raw_force *= self.active_sensor_mask.to(dtype=raw_force.dtype)
        updated = (1.0 - self.ema_alpha) * self.force_ema_N[update_mask] + self.ema_alpha * raw_force[update_mask]
        self.force_ema_N[update_mask] = updated * self.active_sensor_mask[update_mask].to(dtype=updated.dtype)
        self.contact_bits[update_mask] = (
            self.force_ema_N[update_mask] > self.force_threshold_N
        ) & self.active_sensor_mask[update_mask]
        self.last_update_step[update_mask] = step
        self._validate_pregrasp_routing(env)

    def owner_force_and_bits(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""把24 sensors以amax/OR规约到21 owners，并清ghost owners。"""

        indices = self._owner_indices.reshape(1, -1).expand(self.force_ema_N.shape[0], -1)
        owner_force = torch.zeros(
            self.force_ema_N.shape[0], CANONICAL_OWNER_COUNT, device=self.force_ema_N.device
        )
        owner_force.scatter_reduce_(1, indices, self.force_ema_N, reduce="amax", include_self=True)
        owner_bits_int = torch.zeros_like(owner_force, dtype=torch.int64)
        owner_bits_int.scatter_reduce_(
            1, indices, self.contact_bits.to(dtype=torch.int64), reduce="amax", include_self=True
        )
        owner_force *= self.active_owner_mask.to(dtype=owner_force.dtype)
        owner_bits = owner_bits_int.to(dtype=torch.bool) & self.active_owner_mask
        return owner_force, owner_bits

    def reset(self, env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        r"""只清selected rows，并阻止同一stamp读取stale pre-reset sensor data。"""

        ids = _env_ids(env, env_ids)
        self.force_ema_N[ids] = 0.0
        self.contact_bits[ids] = False
        self.last_update_step[ids] = int(env.common_step_counter)

    def _validate_pregrasp_routing(self, env: ManagerBasedRLEnv) -> None:
        r"""Pregrasp sidecar存在时交叉核对joint/owner masks。"""

        sidecar = getattr(env, HETERO_PREGRASP_STATE_ATTR, None)
        if sidecar is None:
            return  # ObservationManager shape inference可能早于首次reset
        if not isinstance(sidecar, HeterogeneousPregraspState):
            raise RuntimeError("heterogeneous pregrasp sidecar has incompatible type")
        valid = sidecar.valid
        if bool(valid.any().item()) and not torch.equal(
            sidecar.active_owner_mask[valid], self.active_owner_mask[valid]
        ):
            raise RuntimeError("contact and pregrasp owner routing disagree")


def get_contact_state(
    env: ManagerBasedRLEnv,
    *,
    layout: HeterogeneousContactLayout,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
    update: bool = True,
) -> HeterogeneousContactState:
    r"""取得env singleton并按需刷新；不同配置请求fail closed。"""

    signature = (
        layout,
        tuple(tuple(bool(value) for value in row) for row in active_joint_mask_by_env),
        float(ema_alpha),
        float(force_threshold_N),
    )
    state = getattr(env, HETERO_CONTACT_STATE_ATTR, None)
    if state is None:
        state = HeterogeneousContactState(
            env,
            layout=layout,
            active_joint_mask_by_env=active_joint_mask_by_env,
            ema_alpha=ema_alpha,
            force_threshold_N=force_threshold_N,
        )
        setattr(state, "_signature", signature)
        setattr(env, HETERO_CONTACT_STATE_ATTR, state)
    elif not isinstance(state, HeterogeneousContactState) or getattr(state, "_signature", None) != signature:
        raise RuntimeError("environment already owns a different heterogeneous contact contract")
    if update:
        state.ensure_updated(env)
    return state


def reset_contact_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor,
    *,
    layout: HeterogeneousContactLayout,
    active_joint_mask_by_env: Sequence[Sequence[bool]],
    ema_alpha: float = 0.5,
    force_threshold_N: float = 0.25,
) -> None:
    r"""Event adapter：创建singleton并partial-clear selected rows。"""

    state = get_contact_state(
        env,
        layout=layout,
        active_joint_mask_by_env=active_joint_mask_by_env,
        ema_alpha=ema_alpha,
        force_threshold_N=force_threshold_N,
        update=False,
    )
    state.reset(env, env_ids)


def _env_ids(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
) -> torch.Tensor:
    r"""把full/partial ids规约为env-device LongTensor。"""

    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


__all__ = [
    "HETERO_CONTACT_STATE_ATTR",
    "HeterogeneousContactState",
    "get_contact_state",
    "reset_contact_state",
]
