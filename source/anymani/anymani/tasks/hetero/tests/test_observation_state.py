r"""Structured JOINT/TIP observation、mask与depth-major routing合同。"""

from __future__ import annotations

import math

import pytest
import torch

from anymani.tasks.hetero.mdp.observation_state import (
    actor_joint_current,
    actor_joint_history_frame,
    actor_joint_limits,
    actor_tip_contact,
    broadcast_tip_contact_to_joints,
    critic_joint_state,
)
from anymani.tasks.hetero.mdp.runtime_state import derive_tip_and_owner_masks


def _prefix_mask() -> torch.Tensor:
    r"""返回finger lengths index/middle/ring/thumb=`4/3/2/1`的depth-major mask。"""

    return torch.tensor(
        (
            (
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
            ),
        )
    )


def test_joint_mask_derives_tip_and_owner_axes_in_canonical_order() -> None:
    r"""Owner顺序固定PALM、16 JOINT、4 TIP，TIP顺序index/middle/ring/thumb。"""

    joint_mask = _prefix_mask()
    tip_mask, owner_mask = derive_tip_and_owner_masks(joint_mask)
    assert tip_mask.shape == (1, 4)
    assert tip_mask.tolist() == [[True, True, True, True]]
    assert owner_mask.shape == (1, 21)
    assert bool(owner_mask[0, 0])  # PALM恒有效
    assert torch.equal(owner_mask[:, 1:17], joint_mask)
    assert torch.equal(owner_mask[:, 17:21], tip_mask)


def test_joint_mask_rejects_nonprefix_finger_topology() -> None:
    r"""某指depth0缺失而depth1存在时fail closed。"""

    mask = _prefix_mask()
    mask[0, 0] = False  # index depth0 inactive，depth1 index仍active
    with pytest.raises(ValueError, match="compact prefix"):
        derive_tip_and_owner_masks(mask)


def test_tip_contact_broadcast_follows_depth_major_joint_axis() -> None:
    r"""每个depth block重复index/middle/ring/thumb contact，而非finger-major展开。"""

    mask = _prefix_mask()
    tip_bits = torch.tensor(((True, False, True, False),))
    joint_bits = broadcast_tip_contact_to_joints(tip_bits, mask)
    expected_unmasked = tip_bits.unsqueeze(1).expand(-1, 4, -1).reshape(1, 16)
    assert torch.equal(joint_bits, expected_unmasked & mask)


def test_actor_current_and_history_mask_every_ghost_channel() -> None:
    r"""Poisoned ghost q/target/action/contact在structured frame中全部归零。"""

    mask = _prefix_mask()
    q = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    target = q + 0.25  # 明确保留actual$q_s$与preload$q_t$差异
    action = torch.linspace(-1.0, 1.0, 16).reshape(1, 16)
    tip_bits = torch.ones(1, 4, dtype=torch.bool)
    current = actor_joint_current(q, target, action, mask)
    history_frame = actor_joint_history_frame(q, target, action, tip_bits, mask)
    assert current.shape == (1, 16, 3)
    assert history_frame.shape == (1, 16, 4)
    assert torch.equal(current[~mask], torch.zeros_like(current[~mask]))
    assert torch.equal(history_frame[~mask], torch.zeros_like(history_frame[~mask]))
    assert torch.allclose(current[0, 0, :2], torch.tensor((0.0, 0.25 / math.pi)))


def test_limits_tip_and_critic_keep_named_axes_and_units() -> None:
    r"""Limits保留`[joint,bound]`，TIP保留role轴，critic velocity不做角度归一化。"""

    mask = _prefix_mask()
    limits = torch.stack((torch.full((1, 16), -math.pi), torch.full((1, 16), math.pi)), dim=-1)
    normalized_limits = actor_joint_limits(limits, mask)
    assert normalized_limits.shape == (1, 16, 2)
    assert torch.equal(normalized_limits[0, 0], torch.tensor((-1.0, 1.0)))
    assert torch.equal(normalized_limits[~mask], torch.zeros_like(normalized_limits[~mask]))

    tip = actor_tip_contact(torch.ones(1, 4, dtype=torch.bool), mask)
    assert tip.shape == (1, 4, 1)
    assert torch.equal(tip, torch.ones_like(tip))

    q = torch.zeros(1, 16)
    velocity = torch.full((1, 16), 2.5)
    target = torch.full((1, 16), 0.1)
    action = torch.zeros(1, 16)
    critic = critic_joint_state(q, velocity, target, action, mask)
    assert critic.shape == (1, 16, 4)
    assert float(critic[0, 0, 1].item()) == 2.5  # rad/s保持物理单位
    assert torch.equal(critic[~mask], torch.zeros_like(critic[~mask]))
