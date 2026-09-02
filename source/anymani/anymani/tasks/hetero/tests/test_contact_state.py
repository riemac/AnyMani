r"""Shared contact pair-max、EMA、owner reduction与partial-reset合同。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch

import anymani.tasks.hetero.mdp.contact_state as contact_module
from anymani.tasks.hetero.contact_layout import build_canonical_contact_layout
from anymani.tasks.hetero.contact_sensors import sensor_contact_magnitude
from anymani.tasks.hetero.mdp.contact_state import HeterogeneousContactState


def _joint_mask(batch_size: int = 2) -> tuple[tuple[bool, ...], ...]:
    r"""返回两行10-DoF compact mask。"""

    row = (True, True, True, True, True, True, True, True, True, False, False, True, False, False, False, False)
    return (row,) * batch_size


def test_sensor_contact_magnitude_uses_pair_max_not_vector_sum() -> None:
    r"""相反方向的两个2 N pairs仍报告2 N，而不是向量和0。"""

    force = torch.tensor(((((2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),),))  # [B,body,filter,3]
    sensor = SimpleNamespace(data=SimpleNamespace(force_matrix_w=force, friction_forces_w=None))
    env = SimpleNamespace(scene={"contact": sensor})
    magnitude = sensor_contact_magnitude(env, "contact")
    assert torch.equal(magnitude, torch.tensor((2.0,)))


def test_contact_state_is_policy_step_idempotent_and_partial_reset_safe(monkeypatch: Any) -> None:
    r"""同stamp重复读取不更新；reset row保持零直到下一policy step，另一row不变。"""

    layout = build_canonical_contact_layout()
    env = cast(Any, SimpleNamespace(num_envs=2, device="cpu", common_step_counter=1))
    raw = {name: torch.zeros(2) for name in layout.state_sensor_names}
    raw[layout.fingertip_sensor_names[0]][:] = 1.0
    raw[layout.fingertip_sensor_names[1]][:] = 2.0
    monkeypatch.setattr(contact_module, "sensor_contact_magnitude", lambda _env, name: raw[name])
    state = HeterogeneousContactState(env, layout=layout, active_joint_mask_by_env=_joint_mask())
    state.ensure_updated(env)
    assert torch.equal(state.tip_force_ema_N[:, :2], torch.tensor(((0.5, 1.0), (0.5, 1.0))))
    first = state.force_ema_N.clone()
    raw[layout.fingertip_sensor_names[0]][:] = 10.0
    state.ensure_updated(env)
    assert torch.equal(state.force_ema_N, first)  # 同common step幂等

    state.reset(env, torch.tensor((1,)))
    assert torch.equal(state.force_ema_N[1], torch.zeros(24))
    assert torch.equal(state.force_ema_N[0], first[0])
    state.ensure_updated(env)
    assert torch.equal(state.force_ema_N[1], torch.zeros(24))  # reset当前stamp不读stale sensor
    env.common_step_counter = 2
    state.ensure_updated(env)
    assert float(state.tip_force_ema_N[1, 0].item()) == 5.0


def test_owner_reduction_maps_root_to_palm_and_clears_ghost_owners() -> None:
    r"""Root与PALM sensors做amax归owner0，ghost JOINT owners保持零。"""

    layout = build_canonical_contact_layout()
    env = cast(Any, SimpleNamespace(num_envs=2, device="cpu", common_step_counter=0))
    state = HeterogeneousContactState(env, layout=layout, active_joint_mask_by_env=_joint_mask())
    state.force_ema_N[:, 4] = 3.0  # index_root -> PALM owner0
    state.force_ema_N[:, 23] = 2.0  # palm sensor -> PALM owner0
    state.contact_bits = state.force_ema_N > 0.25
    owner_force, owner_bits = state.owner_force_and_bits()
    assert torch.equal(owner_force[:, 0], torch.tensor((3.0, 3.0)))
    assert bool(owner_bits[:, 0].all())
    assert torch.equal(owner_force[:, 10], torch.zeros(2))  # joint index9为ghost，owner index10
    assert not bool(owner_bits[:, 10].any())
