r"""GM tactile rotation privileged critic 的 48D actual ADR state。

该 buffer 由随机化 event/action reset 在采样时写入，policy step observation 只做 GPU tensor 读取，
不反复调用 CPU PhysX views。固定 schema：

`[scale1, mass1, COM3, object_material3, hand_contact_material3, Kp16, Kd16,
action_noise1, actual_latency1, wrench_gate1, max_acceleration1, ADR_fraction1]`。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

ADR_STATE_DIM = 48
ADR_STATE_SLICES: dict[str, slice] = {
    "scale": slice(0, 1),
    "mass": slice(1, 2),
    "com": slice(2, 5),
    "object_material": slice(5, 8),
    "hand_contact_material": slice(8, 11),
    "stiffness": slice(11, 27),
    "damping": slice(27, 43),
    "action_noise": slice(43, 44),
    "latency_steps": slice(44, 45),
    "wrench_gate": slice(45, 46),
    "max_acceleration": slice(46, 47),
    "fraction": slice(47, 48),
}


class GmADRState:
    r"""Vectorized actual ADR values 的唯一 GPU owner。"""

    def __init__(self, env: Any, action_dim: int = 16) -> None:
        r"""分配 `[num_envs,48]` buffer，并锁定 16-DOF baseline schema。"""

        if int(action_dim) != 16:
            raise ValueError(f"GM tactile ADR v1 fixes action_dim=16, got {action_dim}.")
        self.action_dim = int(action_dim)
        self.values = torch.zeros(env.num_envs, ADR_STATE_DIM, dtype=torch.float32, device=env.device)
        self.values[:, ADR_STATE_SLICES["scale"]] = 1.0  # prestartup scale event 前的中性 multiplier
        self.values[:, ADR_STATE_SLICES["fraction"]] = float(getattr(env, "leap_adr_fraction", 0.0))

    def set(
        self,
        env: Any,
        field: str,
        value: float | bool | torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        r"""按命名 slice 写实际采样值，并验证 feature width 可广播。"""

        if field not in ADR_STATE_SLICES:
            raise KeyError(f"Unknown GM ADR state field {field!r}; valid fields={tuple(ADR_STATE_SLICES)}.")
        ids = _resolve_env_ids(env, env_ids)
        target = self.values[ids, ADR_STATE_SLICES[field]]
        source = torch.as_tensor(value, dtype=torch.float32, device=env.device)
        if source.ndim == 0:
            source = source.expand_as(target)
        elif source.ndim == 1:
            if source.numel() == ids.numel() and target.shape[1] == 1:
                source = source[:, None]  # per-env scalar
            elif source.numel() == target.shape[1]:
                source = source[None, :].expand(ids.numel(), -1)  # one shared feature vector
        if source.shape != target.shape:
            raise ValueError(
                f"ADR field {field!r} expects shape {tuple(target.shape)} for env subset, got {tuple(source.shape)}."
            )
        self.values[ids, ADR_STATE_SLICES[field]] = source


def get_gm_adr_state(env: Any, action_dim: int = 16) -> GmADRState:
    r"""取得 env-level actual ADR singleton；不同 action schema 请求会 fail fast。"""

    state = getattr(env, "_gm_adr_state", None)
    if state is None:
        state = GmADRState(env, action_dim=action_dim)
        setattr(env, "_gm_adr_state", state)
    elif not isinstance(state, GmADRState) or state.action_dim != int(action_dim):
        raise RuntimeError(
            f"Existing GM ADR state has incompatible schema: {type(state).__name__}, "
            f"action_dim={getattr(state, 'action_dim', None)}; requested action_dim={action_dim}."
        )
    return state


def _resolve_env_ids(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
) -> torch.Tensor:
    r"""统一 event/action partial env ids。"""

    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


__all__ = ["ADR_STATE_DIM", "ADR_STATE_SLICES", "GmADRState", "get_gm_adr_state"]
