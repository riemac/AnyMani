"""Recovery exploration mask and fixed sigma-floor contracts for the palm-rotation Actor."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationDirectActor,
    PalmRotationGeometry,
    PalmRotationResidualActor,
)

RAD_MARGIN = 0.02 / math.pi
"""The fixed gate is 0.02 rad, while target and limits are stored in rad/pi."""


def _owner_valid(joint_valid: torch.Tensor, tip_valid: torch.Tensor) -> torch.Tensor:
    """Build the canonical PALM/JOINT/TIP validity axis."""

    return torch.cat((torch.ones(joint_valid.shape[0], 1, dtype=torch.bool), joint_valid, tip_valid), dim=-1)


def _observation(
    *,
    joint_valid: torch.Tensor | None = None,
    tip_valid: torch.Tensor | None = None,
    q: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    previous_action: torch.Tensor | None = None,
    limits: torch.Tensor | None = None,
    tip_contact: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
) -> PalmRotationActorObservation:
    """Create a controlled actor packet without simulator or physical state."""

    batch = 1
    joint_valid = torch.ones(batch, 16, dtype=torch.bool) if joint_valid is None else joint_valid
    tip_valid = torch.ones(batch, 4, dtype=torch.bool) if tip_valid is None else tip_valid
    q = torch.zeros(batch, 16) if q is None else q
    target = torch.zeros(batch, 16) if target is None else target
    previous_action = torch.zeros(batch, 16) if previous_action is None else previous_action
    limits = (
        torch.stack((torch.full((batch, 16), -1.0), torch.full((batch, 16), 1.0)), dim=-1)
        if limits is None
        else limits
    )
    tip_contact = torch.zeros(batch, 4, dtype=torch.bool) if tip_contact is None else tip_contact
    history = torch.zeros(batch, 30, 16, 5) if history is None else history

    frame = torch.stack((q, target, previous_action), dim=-1)
    frame = torch.cat(
        (
            frame,
            torch.zeros(batch, 16, 1),
            tip_contact.unsqueeze(1).expand(-1, 4, -1).reshape(batch, 16, 1).to(torch.float32),
        ),
        dim=-1,
    )
    owner_contact = torch.zeros(batch, 21, 1)
    owner_contact[:, 17:21, 0] = tip_contact.to(torch.float32)
    return PalmRotationActorObservation(
        jnt_current=frame,
        jnt_history=history,
        jnt_limits=limits,
        owner_contact=owner_contact,
        jnt_valid=joint_valid,
        tip_valid=tip_valid,
        owner_valid=_owner_valid(joint_valid, tip_valid),
    )


def _geometry(observation: PalmRotationActorObservation) -> PalmRotationGeometry:
    """Build zero graph evidence with the observation's canonical owner mask."""

    batch = observation.jnt_current.shape[0]
    graph = torch.zeros(batch, 21, 21, dtype=torch.long)
    return PalmRotationGeometry(
        tokens=torch.zeros(batch, 21, 128),
        owner_valid=observation.owner_valid,
        shortest_path=graph,
        parent_direction=graph.clone(),
        child_direction=graph.clone(),
    )


def _gate_observation() -> PalmRotationActorObservation:
    """Create upper/lower outward gates plus deliberately invalid controls."""

    q = torch.zeros(1, 16)
    target = torch.zeros(1, 16)
    previous_action = torch.zeros(1, 16)
    target[0, 0] = 1.0 - 0.019 / math.pi
    previous_action[0, 0] = 0.051
    q[0, 0] = -1.0  # far from upper limit: the gate must use target u, not physical q.
    target[0, 1] = -1.0 + 0.019 / math.pi
    previous_action[0, 1] = -0.051
    q[0, 1] = 1.0  # far from lower limit.
    target[0, 2] = 1.0 - 0.019 / math.pi
    previous_action[0, 2] = -0.051  # wrong direction at the upper limit.
    target[0, 3] = -1.0 + 0.019 / math.pi
    previous_action[0, 3] = 0.051  # wrong direction at the lower limit.
    target[0, 4] = 1.0 - 0.021 / math.pi  # 0.021 rad: outside the fixed 0.02-rad gate.
    previous_action[0, 4] = 0.051
    target[0, 5] = 1.0 - 0.019 / math.pi
    previous_action[0, 5] = 0.049  # strict previous-action magnitude gate.
    target[0, 6] = 1.0 - 0.019 / math.pi
    previous_action[0, 6] = 0.051
    joint_valid = torch.ones(1, 16, dtype=torch.bool)
    joint_valid[0, 7] = False  # ghost has a gate-looking target/action but must never recover.
    target[0, 7] = 1.0 - 0.019 / math.pi
    previous_action[0, 7] = 0.051
    return _observation(joint_valid=joint_valid, q=q, target=target, previous_action=previous_action)


def test_recovery_mask_is_disabled_without_floor() -> None:
    """The disabled route returns a correctly shaped all-false mask."""

    actor = PalmRotationDirectActor(recovery_sigma_floor=None)
    mask = actor.recovery_exploration_mask(_gate_observation())
    assert mask.shape == (1, 16)
    assert mask.dtype == torch.bool
    assert not bool(mask.any())


def test_recovery_mask_uses_target_rad_pi_and_outward_sign() -> None:
    """Upper/lower signs, target-vs-q and the strict 0.02-rad/0.05-action gates are explicit."""

    actor = PalmRotationDirectActor(recovery_sigma_floor=0.64)
    mask = actor.recovery_exploration_mask(_gate_observation())
    expected = torch.zeros(1, 16, dtype=torch.bool)
    expected[0, :2] = True
    expected[0, 6] = True
    torch.testing.assert_close(mask, expected)

    # A normalized distance of 0.01 is 0.01*pi rad and must be outside the 0.02-rad gate.
    target = _gate_observation().jnt_current[..., 1].clone()
    target[0, 6] = 1.0 - 0.01
    outside = replace(_gate_observation(), jnt_current=torch.stack(
        (
            _gate_observation().jnt_current[..., 0],
            target,
            _gate_observation().jnt_current[..., 2],
            _gate_observation().jnt_current[..., 3],
            _gate_observation().jnt_current[..., 4],
        ),
        dim=-1,
    ))
    assert not bool(actor.recovery_exploration_mask(outside)[0, 6])


def test_recovery_mask_is_hand_wide_and_requires_valid_tip() -> None:
    """One valid TIP contact disables every joint; invalid TIP contacts are ignored."""

    actor = PalmRotationResidualActor(recovery_sigma_floor=0.64)
    baseline = _gate_observation()
    assert bool(actor.recovery_exploration_mask(baseline)[0, 0])

    one_contact = torch.tensor([[True, False, False, False]])
    contacted_current = baseline.jnt_current.clone()
    contacted_current[..., 4] = one_contact.unsqueeze(1).expand(-1, 4, -1).reshape(1, 16)
    contacted_owner = baseline.owner_contact.clone()
    contacted_owner[:, 17:21, 0] = one_contact.to(torch.float32)
    contacted = replace(baseline, jnt_current=contacted_current, owner_contact=contacted_owner)
    assert not bool(actor.recovery_exploration_mask(contacted).any())

    invalid_tip = torch.tensor([[False, True, True, True]])
    invalid_contact = torch.tensor([[True, False, False, False]])
    invalid_tip_obs = _observation(
        tip_valid=invalid_tip,
        tip_contact=invalid_contact,
        q=baseline.jnt_current[..., 0],
        target=baseline.jnt_current[..., 1],
        previous_action=baseline.jnt_current[..., 2],
        limits=baseline.jnt_limits,
    )
    assert bool(actor.recovery_exploration_mask(invalid_tip_obs)[0, 0])

    no_valid_tip = _observation(
        tip_valid=torch.zeros(1, 4, dtype=torch.bool),
        q=baseline.jnt_current[..., 0],
        target=baseline.jnt_current[..., 1],
        previous_action=baseline.jnt_current[..., 2],
        limits=baseline.jnt_limits,
    )
    assert not bool(actor.recovery_exploration_mask(no_valid_tip).any())


def test_recovery_mask_does_not_read_q_or_history_padding() -> None:
    """Changing physical q or History30 cannot change this current-frame recovery gate."""

    actor = PalmRotationDirectActor(recovery_sigma_floor=0.64)
    baseline = _gate_observation()
    changed_q = baseline.jnt_current.clone()
    changed_q[..., 0] = torch.randn_like(changed_q[..., 0]) * 100.0
    changed_history = torch.randn_like(baseline.jnt_history) * 100.0
    changed = replace(baseline, jnt_current=changed_q, jnt_history=changed_history)
    torch.testing.assert_close(actor.recovery_exploration_mask(changed), actor.recovery_exploration_mask(baseline))


def test_recovery_floor_requires_global_sigma_and_valid_bounded_floor() -> None:
    """The floor is fixed, positive, finite, and cannot exceed the base exploration ceiling."""

    for constructor in (PalmRotationResidualActor, PalmRotationDirectActor):
        actor = constructor(recovery_sigma_floor=0.64, sigma_mode="global")
        assert actor.recovery_sigma_floor == pytest.approx(0.64)
    package = PalmRotationActorCritic(arm="direct_token", recovery_sigma_floor=0.64, sigma_mode="global")
    assert package.actor.recovery_sigma_floor == pytest.approx(0.64)

    for invalid in (0.0, -0.1, float("nan"), float("inf"), 1.0):
        with pytest.raises(ValueError, match="recovery.*sigma|floor|ceiling"):
            PalmRotationActorCritic(arm="direct_token", recovery_sigma_floor=invalid)
    with pytest.raises(ValueError, match="global"):
        PalmRotationActorCritic(arm="direct_token", sigma_mode="conditional", recovery_sigma_floor=0.64)


def test_recovery_floor_changes_logstd_only_and_adds_no_learned_keys() -> None:
    """A zero-parameter floor preserves every Actor mean and the complete learned key set."""

    observation = _gate_observation()
    geometry = _geometry(observation)
    base = PalmRotationActorCritic(arm="direct_token", sigma_mode="global")
    recovery = PalmRotationActorCritic(arm="direct_token", sigma_mode="global", recovery_sigma_floor=0.64)
    recovery.load_state_dict(base.state_dict(), strict=True)
    base_output = base.actor(observation, geometry)
    recovery_output = recovery.actor(observation, geometry)

    torch.testing.assert_close(recovery_output.mean, base_output.mean, rtol=0.0, atol=0.0)
    assert set(recovery.actor.state_dict()) == set(base.actor.state_dict())
    assert set(dict(recovery.actor.named_parameters())) == set(dict(base.actor.named_parameters()))
    assert recovery_output.log_std.shape == (1, 16)
    expected_logstd = torch.full((1, 16), -0.5)
    expected_logstd[recovery.actor.recovery_exploration_mask(observation)] = math.log(0.64)
    expected_logstd[~observation.jnt_valid] = 0.0
    torch.testing.assert_close(recovery_output.log_std, expected_logstd, rtol=0.0, atol=1.0e-7)


def test_recovery_floor_gradient_is_zero_on_floor_branch_and_base_remains_learnable() -> None:
    """The fixed floor stops global-base sigma gradients only where it is active."""

    observation = _gate_observation()
    geometry = _geometry(observation)
    actor = PalmRotationDirectActor(recovery_sigma_floor=0.64, sigma_mode="global")
    output = actor(observation, geometry)
    gate = actor.recovery_exploration_mask(observation)
    assert bool(gate[0, 0]) and bool(gate[0, 6])
    output.log_std[gate].sum().backward()
    assert actor.global_log_std.grad is not None
    assert actor.global_log_std.grad.item() == pytest.approx(0.0, abs=0.0)

    actor.zero_grad(set_to_none=True)
    unflagged = observation.jnt_current.clone()
    unflagged[0, 6, 1] = 0.0
    unflagged_obs = replace(observation, jnt_current=unflagged)
    unflagged_output = actor(unflagged_obs, _geometry(unflagged_obs))
    unflagged_gate = actor.recovery_exploration_mask(unflagged_obs)
    assert not bool(unflagged_gate[0, 6])
    unflagged_output.log_std[unflagged_obs.jnt_valid & ~unflagged_gate].sum().backward()
    assert actor.global_log_std.grad is not None
    assert actor.global_log_std.grad.item() > 0.0
