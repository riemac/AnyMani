r"""MVP80 base-action、zero-init residual与independent critic纯Torch合同。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.models.palm_rotation_policy import (
    BASE_ACTION_LIMIT,
    RESIDUAL_LIMIT,
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationCriticObservation,
    PalmRotationGeometry,
    masked_max,
)


def _masks(batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""构造16/12/9-DoF depth-major prefix masks并截到请求batch。"""

    full = torch.ones(16, dtype=torch.bool)
    twelve = torch.tensor((True,) * 8 + (True, True, False, True) + (True, False, False, True))
    nine = torch.tensor((True, False, True, True, True, False, True, True, True, False, False, True) + (False,) * 4)
    joint = torch.stack((full, twelve, nine))[:batch]
    tip = joint.reshape(batch, 4, 4).any(dim=1)
    owner = torch.cat((torch.ones(batch, 1, dtype=torch.bool), joint, tip), dim=-1)
    return joint, tip, owner


def _fixture(batch: int = 3):
    r"""返回actor/critic/geometry同batch fixture。"""

    torch.manual_seed(7)
    joint, tip, owner = _masks(batch)
    actor = PalmRotationActorObservation(
        jnt_current=torch.randn(batch, 16, 5),
        jnt_history=torch.randn(batch, 30, 16, 5),
        jnt_limits=torch.stack((torch.full((batch, 16), -1.0), torch.full((batch, 16), 1.0)), dim=-1),
        owner_contact=torch.randint(0, 2, (batch, 21, 1), dtype=torch.float32),
        jnt_valid=joint,
        tip_valid=tip,
        owner_valid=owner,
    )
    critic = PalmRotationCriticObservation(
        jnt_state=torch.randn(batch, 16, 4),
        owner_contact=torch.randn(batch, 21, 2),
        obj=torch.randn(batch, 1, 15),
        task=torch.randn(batch, 1, 8),
        reward_release=torch.zeros(batch, 1),
        jnt_valid=joint,
        tip_valid=tip,
        owner_valid=owner,
    )
    graph = torch.zeros(batch, 21, 21, dtype=torch.long)
    geometry = PalmRotationGeometry(
        tokens=torch.randn(batch, 21, 128),
        owner_valid=owner,
        shortest_path=graph,
        parent_direction=graph.clone(),
        child_direction=graph.clone(),
    )
    return actor, critic, geometry


def test_zero_initialized_action_residual_is_exact_base_policy() -> None:
    r"""初始化时global branch不得对任一active/ghost action产生数值影响。"""

    actor_observation, _, geometry = _fixture()
    package = PalmRotationActorCritic(arm="residual")
    output = package.actor(actor_observation, geometry)
    assert output.base_mean is not None and output.residual_mean is not None
    assert torch.equal(output.mean, output.base_mean)
    assert torch.equal(output.residual_mean, torch.zeros_like(output.residual_mean))
    assert torch.equal(output.mean[~actor_observation.jnt_valid], torch.zeros_like(output.mean[~actor_observation.jnt_valid]))
    assert output.log_std.numel() == 1


def test_global_exploration_projection_enforces_n000_early_budget_ceiling() -> None:
    r"""Trainable $\log\sigma$可下降，但optimizer上推必须投影回$-0.43$。"""

    package = PalmRotationActorCritic(initial_log_std=-0.5, max_log_std=-0.43)
    with torch.no_grad():
        package.actor.global_log_std.fill_(1.0)
    package.actor.project_exploration_parameters()
    torch.testing.assert_close(package.actor.global_log_std, torch.tensor(-0.43))


def test_raw_history_actor_reads_all_150_lagged_scalars_without_temporal_bottleneck() -> None:
    r"""Raw-stack arm应直接扩大local MLP输入，同时保持动作mask、bound与完整backward。"""

    actor_observation, _, geometry = _fixture()
    package = PalmRotationActorCritic(arm="residual", history_encoder="raw_stack")
    first_local_linear = package.actor.local_encoder[1]
    assert isinstance(first_local_linear, torch.nn.Linear)
    assert first_local_linear.in_features == 5 + 2 + 2 + 30 * 5  # current/limits/lag/q/raw History30
    output = package.actor(actor_observation, geometry)
    output.mean.square().mean().backward()

    assert output.mean.shape == (3, 16)
    assert torch.equal(output.mean[~actor_observation.jnt_valid], torch.zeros_like(output.mean[~actor_observation.jnt_valid]))
    assert any(parameter.grad is not None for parameter in package.actor.local_encoder.parameters())


def test_masked_max_empty_group_has_exact_zero_output_and_finite_zero_gradient() -> None:
    r"""全padding finger不得让dtype-min哨兵进入compiled backward；值与梯度都应为0。"""

    tokens = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[True, False, True, False], [False, False, False, False]])
    output = masked_max(tokens, mask, dim=1)
    output.square().sum().backward()

    torch.testing.assert_close(output[1], torch.zeros(3), rtol=0.0, atol=0.0)
    assert tokens.grad is not None
    torch.testing.assert_close(tokens.grad[1], torch.zeros(4, 3), rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(tokens.grad).all())


def test_residual_head_is_bounded_and_cannot_write_ghost_actions() -> None:
    r"""即使residual raw logit饱和，物理修正也严格位于$[-0.2,0.2]$。"""

    actor_observation, _, geometry = _fixture()
    package = PalmRotationActorCritic(arm="residual")
    final = package.actor.residual_head[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias.fill_(20.0)
    output = package.actor(actor_observation, geometry)
    assert output.base_mean is not None and output.residual_mean is not None
    active = actor_observation.jnt_valid
    assert bool((output.residual_mean[active].abs() <= RESIDUAL_LIMIT).all())
    assert bool((output.residual_mean[active] > 0.19).all())
    assert bool((output.base_mean[active].abs() <= BASE_ACTION_LIMIT).all())
    assert bool((output.mean[active].abs() <= 1.0).all())
    assert torch.equal(output.residual_mean[~active], torch.zeros_like(output.residual_mean[~active]))


def test_actor_and_two_layer_critic_are_disjoint_and_backward_finite() -> None:
    r"""Actor/critic共享输入tensor但不共享可训练参数或optimizer梯度。"""

    actor_observation, critic_observation, geometry = _fixture()
    package = PalmRotationActorCritic(arm="residual")
    actor_ids, critic_ids = package.trainable_parameter_sets()
    assert actor_ids.isdisjoint(critic_ids)
    actor_output = package.actor(actor_observation, geometry)
    value = package.critic(critic_observation, geometry)
    assert value.shape == (3,)
    loss = actor_output.mean.square().mean() + value.square().mean()
    loss.backward()
    assert all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in package.parameters())
    residual_final = package.actor.residual_head[-1]
    assert isinstance(residual_final, torch.nn.Linear)
    assert residual_final.weight.grad is not None


def test_palm_rotation_graph_bias_uses_exact_indexed_embedding_semantics() -> None:
    r"""PPO专用图偏置应等于三表直接查值，并向每个被访问关系桶传播梯度。"""

    package = PalmRotationActorCritic(arm="residual")
    backbone = package.actor.global_backbone  # PPO专用indexed lookup；参数namespace与共享backbone一致
    shortest = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.long)
    parent = shortest.roll(1, dims=-1)
    child = shortest.roll(1, dims=-2)
    actual = backbone._graph_bias(shortest, parent, child)
    expected = (
        backbone.shortest_path_bias(shortest)
        + backbone.parent_direction_bias(parent)
        + backbone.child_direction_bias(child)
    ).permute(0, 3, 1, 2)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    parameters = (
        backbone.shortest_path_bias.weight,
        backbone.parent_direction_bias.weight,
        backbone.child_direction_bias.weight,
    )
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("arm", ("residual", "direct", "direct_token"))
def test_consistent_finger_permutation_is_actor_equivariant_and_critic_invariant(arm: str) -> None:
    r"""同步重排JOINT/TIP/owner/graph后，动作随关节变换而hand-level value不变。

    Canonical JOINT轴是depth-major，故finger permutation必须在每个depth block内同步作用；只重排连续
    4-joint块会错误地把depth当finger。该测试直接证伪finger pooling/routing轴解释错误。
    """

    actor_observation, critic_observation, geometry = _fixture(batch=1)
    package = PalmRotationActorCritic(arm=arm).eval()  # type: ignore[arg-type]
    original_action = package.actor(actor_observation, geometry).mean
    original_value = package.critic(critic_observation, geometry)

    # 新finger顺序依次读取旧ring/index/thumb/middle；每个depth的4 slots使用同一置换。
    finger_permutation = torch.tensor((2, 0, 3, 1), dtype=torch.long)
    joint_permutation = torch.cat(
        tuple(depth * 4 + finger_permutation for depth in range(4)), dim=0
    )  # `[16]` depth-major
    owner_permutation = torch.cat(
        (torch.tensor((0,), dtype=torch.long), 1 + joint_permutation, 17 + finger_permutation), dim=0
    )  # `[21]` PALM固定

    permuted_actor = PalmRotationActorObservation(
        jnt_current=actor_observation.jnt_current[:, joint_permutation],
        jnt_history=actor_observation.jnt_history[:, :, joint_permutation],
        jnt_limits=actor_observation.jnt_limits[:, joint_permutation],
        owner_contact=actor_observation.owner_contact[:, owner_permutation],
        jnt_valid=actor_observation.jnt_valid[:, joint_permutation],
        tip_valid=actor_observation.tip_valid[:, finger_permutation],
        owner_valid=actor_observation.owner_valid[:, owner_permutation],
    )
    permuted_critic = PalmRotationCriticObservation(
        jnt_state=critic_observation.jnt_state[:, joint_permutation],
        owner_contact=critic_observation.owner_contact[:, owner_permutation],
        obj=critic_observation.obj,
        task=critic_observation.task,
        reward_release=critic_observation.reward_release,
        jnt_valid=critic_observation.jnt_valid[:, joint_permutation],
        tip_valid=critic_observation.tip_valid[:, finger_permutation],
        owner_valid=critic_observation.owner_valid[:, owner_permutation],
    )
    permuted_geometry = PalmRotationGeometry(
        tokens=geometry.tokens[:, owner_permutation],
        owner_valid=geometry.owner_valid[:, owner_permutation],
        shortest_path=geometry.shortest_path[:, owner_permutation][:, :, owner_permutation],
        parent_direction=geometry.parent_direction[:, owner_permutation][:, :, owner_permutation],
        child_direction=geometry.child_direction[:, owner_permutation][:, :, owner_permutation],
    )
    permuted_action = package.actor(permuted_actor, permuted_geometry).mean
    permuted_value = package.critic(permuted_critic, permuted_geometry)
    torch.testing.assert_close(permuted_action, original_action[:, joint_permutation], rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(permuted_value, original_value, rtol=1.0e-5, atol=1.0e-6)


def test_direct_actor_has_full_authority_null_residual_diagnostics_and_matched_capacity() -> None:
    r"""Direct由contextual JOINT直接输出，ghost为0，且参数量与residual相差不超过5%。"""

    actor_observation, _, geometry = _fixture()
    residual = PalmRotationActorCritic(arm="residual")
    direct = PalmRotationActorCritic(arm="direct")
    output = direct.actor(actor_observation, geometry)

    assert output.base_mean is None and output.residual_mean is None
    assert output.direct_mean is not None and torch.equal(output.mean, output.direct_mean)
    assert bool((output.mean[actor_observation.jnt_valid].abs() <= 1.0).all())
    assert torch.equal(output.mean[~actor_observation.jnt_valid], torch.zeros_like(output.mean[~actor_observation.jnt_valid]))
    assert not hasattr(direct.actor, "base_head") and not hasattr(direct.actor, "residual_head")
    residual_parameters = sum(parameter.numel() for parameter in residual.actor.parameters())
    direct_parameters = sum(parameter.numel() for parameter in direct.actor.parameters())
    assert abs(direct_parameters / residual_parameters - 1.0) <= 0.05

    output.mean.square().mean().backward()
    assert any(parameter.grad is not None for parameter in direct.actor.direct_head.parameters())  # type: ignore[union-attr]


def test_token_direct_head_reads_only_contextual_joint_token_at_matched_capacity() -> None:
    r"""Canonical Direct必须由$H^a_{t,j}$单独读出动作，不得把局部latent再次旁路到policy head。"""

    actor_observation, _, geometry = _fixture()
    residual = PalmRotationActorCritic(arm="residual")
    local_skip = PalmRotationActorCritic(arm="direct")
    token_direct = PalmRotationActorCritic(arm="direct_token")
    output = token_direct.actor(actor_observation, geometry)

    token_norm = token_direct.actor.direct_head[0]  # type: ignore[union-attr]  # policy head的唯一输入边界
    token_projection = token_direct.actor.direct_head[1]  # type: ignore[union-attr]
    local_skip_projection = local_skip.actor.direct_head[1]  # type: ignore[union-attr]
    assert isinstance(token_norm, torch.nn.LayerNorm) and token_norm.normalized_shape == (128,)
    assert isinstance(token_projection, torch.nn.Linear) and token_projection.in_features == 128
    assert token_projection.out_features == 96  # $128\times96=192\times64$，匹配两种Direct首层权重数
    assert isinstance(local_skip_projection, torch.nn.Linear) and local_skip_projection.in_features == 192
    assert output.direct_mean is not None and torch.equal(output.mean, output.direct_mean)
    assert output.base_mean is None and output.residual_mean is None

    residual_parameters = sum(parameter.numel() for parameter in residual.actor.parameters())
    token_parameters = sum(parameter.numel() for parameter in token_direct.actor.parameters())
    assert abs(token_parameters / residual_parameters - 1.0) <= 0.05
