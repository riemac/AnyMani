"""Palm rotation episode-phase adapter contracts.

这些合同只构造纯 Torch 模型与结构化 tensor，不启动 Isaac Sim、Kit、GPU 或训练入口；
它们锁定 phase clock 的输入边界、零初始化迁移语义、mask 语义和 functional/vmap 可组合性。
"""

from __future__ import annotations

from dataclasses import fields
from typing import cast

import pytest
import torch
from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationCriticObservation,
    PalmRotationDirectActor,
    PalmRotationGeometry,
    PalmRotationStructuredCritic,
)
from torch.func import functional_call, grad, vmap


def _masks(batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""构造多种合法 JOINT/TIP/owner mask，保留全手、稀疏手和 ghost 行。"""

    full = torch.ones(16, dtype=torch.bool)  # 全部16个JOINT有效，作为无padding参照。
    twelve = torch.tensor((True,) * 8 + (True, True, False, True) + (True, False, False, True))
    nine = torch.tensor(
        (True, False, True, True, True, False, True, True, True, False, False, True) + (False,) * 4
    )
    sparse = torch.tensor((True, False, False, False) * 4)  # 每个depth只保留一个JOINT，检验TIP映射。
    joint = torch.stack((full, twelve, nine, sparse))[:batch]
    tip = joint.reshape(batch, 4, 4).any(dim=1)  # Canonical JOINT轴为depth-major。
    owner = torch.cat((torch.ones(batch, 1, dtype=torch.bool), joint, tip), dim=-1)
    return joint, tip, owner


def _fixture(batch: int = 3) -> tuple[
    PalmRotationActorObservation, PalmRotationCriticObservation, PalmRotationGeometry
]:
    r"""返回同一批次的 actor、privileged critic 与 N040 geometry fixture。"""

    generator = torch.Generator().manual_seed(230912)  # 固定随机源，便于逐值复验迁移合同。
    joint, tip, owner = _masks(batch)
    actor = PalmRotationActorObservation(
        jnt_current=torch.randn(batch, 16, 5, generator=generator),
        jnt_history=torch.randn(batch, 30, 16, 5, generator=generator),
        jnt_limits=torch.stack(
            (torch.full((batch, 16), -1.0), torch.full((batch, 16), 1.0)), dim=-1
        ),
        owner_contact=torch.randint(0, 2, (batch, 21, 1), generator=generator, dtype=torch.float32),
        jnt_valid=joint,
        tip_valid=tip,
        owner_valid=owner,
    )
    critic = PalmRotationCriticObservation(
        jnt_state=torch.randn(batch, 16, 4, generator=generator),
        owner_contact=torch.randn(batch, 21, 2, generator=generator),
        obj=torch.randn(batch, 1, 15, generator=generator),
        task=torch.randn(batch, 1, 8, generator=generator),
        reward_release=torch.randn(batch, 1, generator=generator),
        jnt_valid=joint,
        tip_valid=tip,
        owner_valid=owner,
    )
    graph = torch.zeros(batch, 21, 21, dtype=torch.long)  # 合法的同结构关系桶0，避免引入额外变量。
    geometry = PalmRotationGeometry(
        tokens=torch.randn(batch, 21, 128, generator=generator),
        owner_valid=owner,
        shortest_path=graph,
        parent_direction=graph.clone(),
        child_direction=graph.clone(),
    )
    return actor, critic, geometry


def _phase(batch: int) -> torch.Tensor:
    r"""返回有限 sin/cos 编码，dtype/device与默认 fixture 的 float32/CPU一致。"""

    values = ((0.0, 1.0), (1.0, 0.0), (0.70710677, 0.70710677), (-1.0, 0.0))
    return torch.tensor(values[:batch], dtype=torch.float32)


def test_phase_enabled_direct_registers_the_two_zero_adapters_and_restricts_arms() -> None:
    r"""启用 phase clock 时，Actor/Critic 仅暴露精确的两个无bias零权重适配器。"""

    package = PalmRotationActorCritic(arm="direct_token", phase_clock_enabled=True)
    direct = PalmRotationDirectActor(phase_clock_enabled=True)
    token_direct = PalmRotationDirectActor(local_skip=False, phase_clock_enabled=True)
    critic = PalmRotationStructuredCritic(phase_clock_enabled=True)

    assert package.phase_clock_enabled is True
    assert direct.phase_clock_enabled is True
    assert critic.phase_clock_enabled is True
    assert direct.phase_contextual_adapter.weight.shape == (128, 2)
    assert critic.phase_readout_adapter.weight.shape == (896, 2)
    assert direct.phase_contextual_adapter.bias is None
    assert critic.phase_readout_adapter.bias is None
    assert torch.equal(direct.phase_contextual_adapter.weight, torch.zeros(128, 2))
    assert torch.equal(critic.phase_readout_adapter.weight, torch.zeros(896, 2))
    assert direct.direct_head[0].normalized_shape == (192,)
    assert token_direct.direct_head[0].normalized_shape == (128,)
    assert critic.value_head[0].normalized_shape == (896,)
    assert {field.name for field in fields(PalmRotationActorObservation)} == {
        "jnt_current",
        "jnt_history",
        "jnt_limits",
        "owner_contact",
        "jnt_valid",
        "tip_valid",
        "owner_valid",
    }
    assert {field.name for field in fields(PalmRotationCriticObservation)} == {
        "jnt_state",
        "owner_contact",
        "obj",
        "task",
        "reward_release",
        "jnt_valid",
        "tip_valid",
        "owner_valid",
    }

    for arm in ("base", "residual"):
        with pytest.raises(ValueError, match="direct"):
            PalmRotationActorCritic(arm=arm, phase_clock_enabled=True)  # type: ignore[arg-type]
        disabled_arm = PalmRotationActorCritic(arm=arm)  # type: ignore[arg-type]
        assert disabled_arm.actor.phase_clock_enabled is False
        assert disabled_arm.critic.phase_clock_enabled is False

    disabled = PalmRotationActorCritic(arm="direct_token", phase_clock_enabled=False)
    assert not hasattr(disabled.actor, "phase_contextual_adapter")
    assert not hasattr(disabled.critic, "phase_readout_adapter")
    assert not any("phase_" in key for key in disabled.state_dict())


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_enabled_phase_requires_batch_two_finite_matching_input(role: str) -> None:
    r"""启用分支对缺失、shape、dtype、device与非有限 phase 给出明确错误。"""

    actor_observation, critic_observation, geometry = _fixture()
    package = PalmRotationActorCritic(arm="direct_token", phase_clock_enabled=True)
    forward = package.actor if role == "actor" else package.critic
    observation = actor_observation if role == "actor" else critic_observation
    batch = 3

    with pytest.raises(ValueError, match="phase_clock_enabled=True.*phase_clock"):
        forward(observation, geometry)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="phase_clock.*shape"):
        forward(observation, geometry, phase_clock=torch.zeros(batch, 1))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="phase_clock.*dtype"):
        forward(observation, geometry, phase_clock=_phase(batch).double())  # type: ignore[call-arg]
    with pytest.raises((RuntimeError, ValueError), match="finite sin/cos"):
        invalid = _phase(batch)
        invalid[0, 0] = float("nan")
        forward(observation, geometry, phase_clock=invalid)  # type: ignore[call-arg]


def test_disabled_phase_argument_is_ignored_and_keeps_the_old_path() -> None:
    r"""禁用模型即使收到 phase keyword 也逐值走旧路径，且没有新增参数。"""

    actor_observation, critic_observation, geometry = _fixture()
    package = PalmRotationActorCritic(arm="direct_token", phase_clock_enabled=False)
    actor_without = package.actor(actor_observation, geometry)
    actor_with = package.actor(actor_observation, geometry, phase_clock=_phase(3))
    value_without = package.critic(critic_observation, geometry)
    value_with = package.critic(critic_observation, geometry, phase_clock=_phase(3))

    assert torch.equal(actor_without.mean, actor_with.mean)
    assert torch.equal(actor_without.log_std, actor_with.log_std)
    assert torch.equal(value_without, value_with)
    assert not any("phase_" in key for key in package.state_dict())


@pytest.mark.parametrize("arm", ["direct", "direct_token"])
def test_zero_adapters_copy_all_old_weights_and_preserve_outputs_for_masks_and_phase(arm: str) -> None:
    r"""复制旧参数并保持新权重为零时，多种 mask/phase 下 mean、log_std、value 必须逐bit相等。"""

    actor_observation, critic_observation, geometry = _fixture()
    old = PalmRotationActorCritic(arm=arm, phase_clock_enabled=False)  # type: ignore[arg-type]
    enabled = PalmRotationActorCritic(arm=arm, phase_clock_enabled=True)  # type: ignore[arg-type]
    load_result = enabled.load_state_dict(old.state_dict(), strict=False)
    assert set(load_result.missing_keys) == {
        "actor.phase_contextual_adapter.weight",
        "critic.phase_readout_adapter.weight",
    }
    assert not load_result.unexpected_keys
    for name, value in old.state_dict().items():
        assert torch.equal(enabled.state_dict()[name], value), name

    phase_a = _phase(3)
    phase_b = phase_a.roll(1, dims=0)
    old_actor = old.actor(actor_observation, geometry)
    enabled_actor_a = enabled.actor(actor_observation, geometry, phase_clock=phase_a)
    enabled_actor_b = enabled.actor(actor_observation, geometry, phase_clock=phase_b)
    old_value = old.critic(critic_observation, geometry)
    enabled_value_a = enabled.critic(critic_observation, geometry, phase_clock=phase_a)
    enabled_value_b = enabled.critic(critic_observation, geometry, phase_clock=phase_b)

    assert torch.equal(old_actor.mean, enabled_actor_a.mean)
    assert torch.equal(old_actor.log_std, enabled_actor_a.log_std)
    assert torch.equal(enabled_actor_a.mean, enabled_actor_b.mean)
    assert torch.equal(enabled_actor_a.log_std, enabled_actor_b.log_std)
    assert torch.equal(old_value, enabled_value_a)
    assert torch.equal(enabled_value_a, enabled_value_b)


def test_enabled_state_dict_changes_only_by_the_two_named_weight_keys() -> None:
    r"""phase enabled 的旧键/shape全部保持不变，新增恰为 Actor/Critic 各一个 weight。"""

    old = PalmRotationActorCritic(arm="direct_token", phase_clock_enabled=False)
    enabled = PalmRotationActorCritic(arm="direct_token", phase_clock_enabled=True)
    old_state = old.state_dict()
    enabled_state = enabled.state_dict()
    new_keys = set(enabled_state) - set(old_state)
    assert new_keys == {"actor.phase_contextual_adapter.weight", "critic.phase_readout_adapter.weight"}
    assert set(old_state) - set(enabled_state) == set()
    for key, value in old_state.items():
        assert enabled_state[key].shape == value.shape, key
    assert set(dict(enabled.actor.named_parameters())) - set(dict(old.actor.named_parameters())) == {
        "phase_contextual_adapter.weight"
    }
    assert set(dict(enabled.critic.named_parameters())) - set(dict(old.critic.named_parameters())) == {
        "phase_readout_adapter.weight"
    }


def test_nonzero_phase_adapters_receive_finite_gradients_and_preserve_ghost_zero() -> None:
    r"""零初始化适配器可获非零有限梯度；学习到非零权重后 phase 才改变 active mean/value。"""

    actor_observation, critic_observation, geometry = _fixture()
    package = PalmRotationActorCritic(arm="direct_token", phase_clock_enabled=True)
    direct_actor = cast(PalmRotationDirectActor, package.actor)
    phase_a = _phase(3)
    phase_b = phase_a.roll(1, dims=0)
    actor_output = package.actor(actor_observation, geometry, phase_clock=phase_a)
    value = package.critic(critic_observation, geometry, phase_clock=phase_a)
    actor_gradient, critic_gradient = torch.autograd.grad(
        actor_output.mean.square().sum() + value.square().sum(),
        (direct_actor.phase_contextual_adapter.weight, package.critic.phase_readout_adapter.weight),
    )
    assert torch.isfinite(actor_gradient).all() and bool((actor_gradient != 0).any())
    assert torch.isfinite(critic_gradient).all() and bool((critic_gradient != 0).any())

    with torch.no_grad():
        direct_actor.phase_contextual_adapter.weight[:, 0].copy_(torch.linspace(-0.03, 0.03, 128))
        direct_actor.phase_contextual_adapter.weight[:, 1].copy_(torch.linspace(0.02, -0.02, 128))
        package.critic.phase_readout_adapter.weight[:, 0].copy_(torch.linspace(-0.01, 0.01, 896))
        package.critic.phase_readout_adapter.weight[:, 1].copy_(torch.linspace(0.01, -0.01, 896))
    changed_actor = package.actor(actor_observation, geometry, phase_clock=phase_b)
    changed_value = package.critic(critic_observation, geometry, phase_clock=phase_b)
    actor_delta = (changed_actor.mean - actor_output.mean).abs()
    value_delta = (changed_value - value).abs()
    assert bool((actor_delta[actor_observation.jnt_valid] > 1.0e-8).any())
    assert bool((value_delta > 1.0e-8).any())
    assert torch.equal(
        changed_actor.mean[~actor_observation.jnt_valid],
        torch.zeros_like(changed_actor.mean[~actor_observation.jnt_valid]),
    )
    assert torch.equal(
        actor_output.mean[~actor_observation.jnt_valid],
        torch.zeros_like(actor_output.mean[~actor_observation.jnt_valid]),
    )


@pytest.mark.parametrize("sigma_mode", ["global", "conditional"])
def test_phase_adapter_does_not_change_global_or_conditional_sigma(sigma_mode: str) -> None:
    r"""phase 只进入 direct mean；global scalar 与 conditional joint sigma 均保持旧语义。"""

    actor_observation, _, geometry = _fixture()
    package = PalmRotationActorCritic(
        arm="direct_token", sigma_mode=sigma_mode, phase_clock_enabled=True  # type: ignore[arg-type]
    )
    direct_actor = cast(PalmRotationDirectActor, package.actor)
    with torch.no_grad():
        direct_actor.phase_contextual_adapter.weight.normal_(std=0.02)
    output_a = package.actor(actor_observation, geometry, phase_clock=_phase(3))
    output_b = package.actor(actor_observation, geometry, phase_clock=_phase(3).roll(1, dims=0))

    assert torch.equal(output_a.log_std, output_b.log_std)
    if sigma_mode == "global":
        assert output_a.log_std.numel() == 1
    else:
        assert output_a.log_std.shape == output_a.mean.shape


def test_validated_functional_vmap_grad_path_carries_phase_to_both_adapters() -> None:
    r"""``functional_call``、``vmap``、``grad``组合时，_validated路径仍返回两类phase梯度。"""

    actor_observation, critic_observation, geometry = _fixture(batch=2)
    package = PalmRotationActorCritic(arm="direct_token", phase_clock_enabled=True)
    actor_parameters = dict(package.actor.named_parameters())
    critic_parameters = dict(package.critic.named_parameters())

    # 外层资产轴模拟 task-gradient 的分组视图；内层 batch 轴保留合法 `[S,2]` phase shape。
    def stack_assets(value: torch.Tensor) -> torch.Tensor:
        copies = [value, value + 0.01, value - 0.02] if value.dtype.is_floating_point else [value] * 3
        return torch.stack(copies, dim=0)

    actor_data = {field.name: stack_assets(getattr(actor_observation, field.name)) for field in fields(actor_observation)}
    critic_data = {field.name: stack_assets(getattr(critic_observation, field.name)) for field in fields(critic_observation)}
    geometry_data = {field.name: stack_assets(getattr(geometry, field.name)) for field in fields(geometry)}
    phases = torch.tensor(
        [
            ((0.0, 1.0), (1.0, 0.0)),
            ((0.70710677, 0.70710677), (-1.0, 0.0)),
            ((0.0, -1.0), (-0.70710677, 0.70710677)),
        ],
        dtype=torch.float32,
    )

    def raw_packet(cls: type, data: dict[str, torch.Tensor]):
        r"""在 vmap 内建立已由外层验证的 dataclass view，不触发 data-dependent 检查。"""

        packet = object.__new__(cls)
        for field in fields(cls):
            object.__setattr__(packet, field.name, data[field.name])
        return packet

    def objective(
        actor_params: dict[str, torch.Tensor],
        critic_params: dict[str, torch.Tensor],
        actor_values: dict[str, torch.Tensor],
        critic_values: dict[str, torch.Tensor],
        geometry_values: dict[str, torch.Tensor],
        phase_values: torch.Tensor,
    ) -> torch.Tensor:
        r"""在单个资产的 `[S,...]` 视图上调用两个 functional module。"""

        actor_view = raw_packet(PalmRotationActorObservation, actor_values)
        critic_view = raw_packet(PalmRotationCriticObservation, critic_values)
        geometry_view = raw_packet(PalmRotationGeometry, geometry_values)
        actor_output = functional_call(
            package.actor,
            actor_params,
            (actor_view, geometry_view),
            {"phase_clock": phase_values, "_validated": True},
        )
        value = functional_call(
            package.critic,
            critic_params,
            (critic_view, geometry_view),
            {"phase_clock": phase_values, "_validated": True},
        )
        return actor_output.mean.square().mean() + value.square().mean()

    gradient_function = grad(objective, argnums=(0, 1))
    actor_gradients, critic_gradients = vmap(
        gradient_function,
        in_dims=(None, None, 0, 0, 0, 0),
    )(actor_parameters, critic_parameters, actor_data, critic_data, geometry_data, phases)
    assert set(actor_gradients) == set(actor_parameters)
    assert set(critic_gradients) == set(critic_parameters)
    actor_phase_gradient = actor_gradients["phase_contextual_adapter.weight"]
    critic_phase_gradient = critic_gradients["phase_readout_adapter.weight"]
    assert actor_phase_gradient.shape[:2] == (3, 128)
    assert critic_phase_gradient.shape[:2] == (3, 896)
    assert torch.isfinite(actor_phase_gradient).all() and bool((actor_phase_gradient != 0).any())
    assert torch.isfinite(critic_phase_gradient).all() and bool((critic_phase_gradient != 0).any())
