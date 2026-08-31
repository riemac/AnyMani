r"""canonical policy 的置换等变、mask 隔离与 global log_std 合同。"""

from __future__ import annotations

from dataclasses import fields

import torch
from anymani.distill.models.policy import (
    CANONICAL_JOINT_COUNT,
    CANONICAL_OWNER_COUNT,
    CanonicalPolicyCfg,
    EmbodimentPolicy,
    EmbodimentPolicyInput,
)


def _inputs(
    *,
    owner_features: torch.Tensor | None = None,
    joint_features: torch.Tensor | None = None,
    geometry_entities: torch.Tensor | None = None,
):
    r"""构造一个包含 7 个 active joints 的最小 canonical owner batch。"""

    batch_size = 2
    owner = owner_features if owner_features is not None else torch.randn(batch_size, CANONICAL_OWNER_COUNT, 5)
    joint = joint_features if joint_features is not None else torch.randn(batch_size, CANONICAL_JOINT_COUNT, 3)
    joint_mask = torch.zeros(batch_size, CANONICAL_JOINT_COUNT, dtype=torch.bool)
    joint_mask[:, :7] = True
    owner_mask = torch.ones(batch_size, CANONICAL_OWNER_COUNT, dtype=torch.bool)
    owner_mask[:, 8:17] = False  # 9 个 inactive JOINT owners；TIP mask 仍由测试输入显式控制
    owner_mask[:, 17:] = True
    relation = torch.zeros(CANONICAL_OWNER_COUNT, CANONICAL_OWNER_COUNT, dtype=torch.long)
    return EmbodimentPolicyInput(
        owner_features=owner,
        joint_features=joint,
        owner_valid_mask=owner_mask,
        joint_valid_mask=joint_mask,
        shortest_path=relation,
        parent_direction=relation,
        child_direction=relation,
        asset_row=torch.tensor([3, 11], dtype=torch.long),
        geometry_entities=geometry_entities,
    )


def test_canonical_policy_is_equivariant_under_synchronous_joint_owner_permutation() -> None:
    r"""同步置换 joint/owner features、graph 两轴、mask 后，action 应按同一置换变化而 value 不变。"""

    torch.manual_seed(7)
    policy = EmbodimentPolicy(CanonicalPolicyCfg(owner_feature_dim=5, joint_feature_dim=3))
    policy.eval()
    original = _inputs()
    output = policy(original)

    permutation = torch.tensor([0, 4, 2, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 17, 19, 18])
    joint_permutation = permutation[1 : 1 + CANONICAL_JOINT_COUNT] - 1
    permuted = EmbodimentPolicyInput(
        owner_features=original.owner_features[:, permutation],
        joint_features=original.joint_features[:, joint_permutation],
        owner_valid_mask=original.owner_valid_mask[:, permutation],
        joint_valid_mask=original.joint_valid_mask[:, joint_permutation],
        shortest_path=original.shortest_path[permutation][:, permutation],
        parent_direction=original.parent_direction[permutation][:, permutation],
        child_direction=original.child_direction[permutation][:, permutation],
        asset_row=original.asset_row,
    )
    permuted_output = policy(permuted)

    torch.testing.assert_close(
        permuted_output.action_mean,
        output.action_mean[:, joint_permutation],
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(permuted_output.value, output.value, atol=1.0e-5, rtol=1.0e-5)


def test_inactive_joint_features_cannot_change_active_outputs_or_global_std() -> None:
    r"""ghost 输入可为任意大数，但有效 action/value 与 global log_std 必须保持有限且不变。"""

    torch.manual_seed(11)
    policy = EmbodimentPolicy(CanonicalPolicyCfg(owner_feature_dim=5, joint_feature_dim=3))
    policy.eval()
    original = _inputs()
    baseline = policy(original)

    poisoned_owner = original.owner_features.clone()
    poisoned_joint = original.joint_features.clone()
    poisoned_owner[:, 8:17] = 1.0e12
    poisoned_joint[:, 7:] = -1.0e12
    poisoned = policy(_inputs(owner_features=poisoned_owner, joint_features=poisoned_joint))
    active = original.joint_valid_mask

    torch.testing.assert_close(poisoned.action_mean[active], baseline.action_mean[active])
    torch.testing.assert_close(poisoned.value, baseline.value)
    torch.testing.assert_close(poisoned.action_log_std, baseline.action_log_std)
    assert sum(parameter.numel() for name, parameter in policy.named_parameters() if "log_std" in name) == 1


def test_policy_rejects_noncanonical_owner_or_joint_axes() -> None:
    r"""不允许通过隐式 padding 把 20-DOF/5-finger 新 schema 混入 v1。"""

    with torch.no_grad():
        bad_owner = torch.zeros(1, 20, 5)
        bad_joint = torch.zeros(1, 16, 3)
    try:
        _ = EmbodimentPolicyInput(
            owner_features=bad_owner,
            joint_features=bad_joint,
            owner_valid_mask=torch.ones(1, 20, dtype=torch.bool),
            joint_valid_mask=torch.ones(1, 16, dtype=torch.bool),
            shortest_path=torch.zeros(20, 20, dtype=torch.long),
            parent_direction=torch.zeros(20, 20, dtype=torch.long),
            child_direction=torch.zeros(20, 20, dtype=torch.long),
            asset_row=torch.zeros(1, dtype=torch.long),
        )
    except ValueError as exc:
        assert "owner_features" in str(exc)
    else:
        raise AssertionError("non-canonical owner axis must fail closed")


def test_policy_consumes_one_unified_geometry_entity_tensor() -> None:
    r"""PPO feature contract 只允许 `[B,21,D]` unified $Z$，不得恢复独立 JOINT geometry latent。"""

    torch.manual_seed(17)
    policy = EmbodimentPolicy(
        CanonicalPolicyCfg(owner_feature_dim=5, joint_feature_dim=3, geometry_entity_width=6)
    )
    geometry = torch.randn(2, CANONICAL_OWNER_COUNT, 6, requires_grad=True)
    inputs = _inputs(geometry_entities=geometry)
    output = policy(inputs)
    (output.action_mean.square().sum() + output.value.square().sum()).backward()

    assert geometry.grad is not None and torch.count_nonzero(geometry.grad) > 0
    field_names = {field.name for field in fields(EmbodimentPolicyInput)}
    assert "geometry_entities" in field_names
    assert "geometry_owner_latent" not in field_names
    assert "geometry_joint_latent" not in field_names
