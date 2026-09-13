r"""三组离线 family student policy 的纯 PyTorch 模型合同。

测试只验证 actor 的张量边界与参数身份，不启动 Isaac Sim，也不宣称任何物理能力提升。
"""

from __future__ import annotations

import torch
from anymani.distill.models.family_rotation_policy import (
    FAMILY_ROTATION_VARIANTS,
    FamilyRotationActorOutput,
    build_family_rotation_policy,
)
from anymani.distill.models.palm_rotation_policy import PalmRotationActorObservation, PalmRotationGeometry


def _packet(batch: int = 2) -> tuple[PalmRotationActorObservation, PalmRotationGeometry, torch.Tensor]:
    r"""构造 16-slot/21-owner/History30 的最小结构化 actor 输入。"""

    generator = torch.Generator().manual_seed(17)
    valid = torch.ones(batch, 16, dtype=torch.bool)
    valid[0, 12:] = False
    tips = torch.ones(batch, 4, dtype=torch.bool)
    owner = torch.cat((torch.ones(batch, 1, dtype=torch.bool), valid, tips), dim=-1)
    current = torch.randn(batch, 16, 5, generator=generator)
    history = torch.randn(batch, 30, 16, 5, generator=generator)
    limits = torch.stack((-torch.ones(batch, 16), torch.ones(batch, 16)), dim=-1)
    contact = torch.zeros(batch, 21, 1)
    observation = PalmRotationActorObservation(current, history, limits, contact, valid, tips, owner)
    tokens = torch.randn(batch, 21, 128, generator=generator)
    graph = torch.zeros(batch, 21, 21, dtype=torch.long)
    geometry = PalmRotationGeometry(tokens, owner, graph, graph.clone(), graph.clone())
    kinematics = torch.randn(batch, 16, 15, generator=generator)
    return observation, geometry, kinematics


def test_all_variants_share_fixed_actor_abi_and_mask_ghost_actions() -> None:
    r"""三组均为 direct-token/TCN/phase-off，且无效 joint action 严格为零。"""

    observation, geometry, kinematics = _packet()
    assert FAMILY_ROTATION_VARIANTS == ("n040", "no_z", "fk")
    for variant in FAMILY_ROTATION_VARIANTS:
        torch.manual_seed(29)
        actor = build_family_rotation_policy(variant, device="cpu")
        output = actor(observation, geometry, joint_kinematics=kinematics)
        assert isinstance(output, FamilyRotationActorOutput)
        assert output.mean.shape == (2, 16)
        assert torch.equal(output.mean[0, 12:], torch.zeros(4))
        assert output.fk_prediction is not None if variant == "fk" else output.fk_prediction is None
        assert actor.global_log_std.requires_grad is False


def test_shared_parameters_are_identical_under_same_seed_and_fk_head_is_extra() -> None:
    r"""同 seed 下三组共享 trunk/kinematic adapter 逐值一致，FK head 仅属于 FK variant。"""

    actors = {}
    for variant in FAMILY_ROTATION_VARIANTS:
        torch.manual_seed(41)
        actors[variant] = build_family_rotation_policy(variant, device="cpu")
    fk_keys = set(actors["fk"].state_dict()) - set(actors["n040"].state_dict())
    assert fk_keys and all(key.startswith("fk_head.") for key in fk_keys)
    for key in actors["n040"].state_dict():
        assert torch.equal(actors["n040"].state_dict()[key], actors["no_z"].state_dict()[key])
        assert torch.equal(actors["n040"].state_dict()[key], actors["fk"].state_dict()[key])


def test_no_z_and_fk_ignore_geometry_tokens_but_keep_graph() -> None:
    r"""No-Z/FK 清零 N040 token，graph/mask/history/kinematics 输入仍在 forward 合同中。"""

    observation, geometry, kinematics = _packet(batch=1)
    changed = PalmRotationGeometry(
        geometry.tokens + 13.0,
        geometry.owner_valid,
        geometry.shortest_path,
        geometry.parent_direction,
        geometry.child_direction,
    )
    for variant in ("no_z", "fk"):
        torch.manual_seed(53)
        actor = build_family_rotation_policy(variant, device="cpu")
        first = actor(observation, geometry, joint_kinematics=kinematics).mean
        second = actor(observation, changed, joint_kinematics=kinematics).mean
        assert torch.equal(first, second)
    assert geometry.shortest_path.shape == (1, 21, 21)


def test_tip_only_actor_does_not_read_non_tip_contact_channels() -> None:
    r"""改变 PALM/JOINT 与每帧 non-tip contact 不得改变三组动作均值。"""

    observation, geometry, kinematics = _packet(batch=1)
    changed = PalmRotationActorObservation(
        jnt_current=observation.jnt_current.clone(),
        jnt_history=observation.jnt_history.clone(),
        jnt_limits=observation.jnt_limits,
        owner_contact=observation.owner_contact.clone(),
        jnt_valid=observation.jnt_valid,
        tip_valid=observation.tip_valid,
        owner_valid=observation.owner_valid,
    )
    changed.jnt_current[..., 3] = 1.0
    changed.jnt_history[..., 3] = 1.0
    changed.owner_contact[:, :17] = 1.0
    for variant in FAMILY_ROTATION_VARIANTS:
        torch.manual_seed(61)
        actor = build_family_rotation_policy(variant, device="cpu")
        first = actor(observation, geometry, joint_kinematics=kinematics).mean
        second = actor(changed, geometry, joint_kinematics=kinematics).mean
        assert torch.equal(first, second)


def test_fk_head_is_auxiliary_and_does_not_change_action_interface() -> None:
    r"""扰动 FK head 参数只能改变 FK prediction，不能改变 direct action mean。"""

    observation, geometry, kinematics = _packet(batch=1)
    torch.manual_seed(67)
    actor = build_family_rotation_policy("fk", device="cpu")
    before = actor(observation, geometry, joint_kinematics=kinematics)
    assert before.fk_prediction is not None
    with torch.no_grad():
        for parameter in actor.fk_head.parameters():  # type: ignore[union-attr]
            parameter.add_(0.1)
    after = actor(observation, geometry, joint_kinematics=kinematics)
    assert torch.equal(before.mean, after.mean)
