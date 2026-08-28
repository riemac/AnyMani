r"""masked rl_games model 的 probability、entropy、bounds 与 backward 合同。"""

from __future__ import annotations

import torch
from anymani.distill.models.input_adapters.geometry import StaticGeometryEvidence
from anymani.distill.models.policy import CanonicalEvidenceBank
from anymani.distill.rl.masked_ppo import (
    ANYMANI_MASKED_PPO_ALGO_KEY,
    CANONICAL_MASKED_OBS_DIM,
    AnyManiMaskedContinuousModel,
    AnyManiMaskedPpoAgent,
    CanonicalMaskedPpoBuilder,
)


def _model() -> AnyManiMaskedContinuousModel.Network:
    r"""构造不依赖 Isaac Sim 的 rl_games continuous model。"""

    builder = CanonicalMaskedPpoBuilder()
    builder.load({"canonical_policy": {"owner_feature_dim": 32, "joint_feature_dim": 16}})
    model_factory = AnyManiMaskedContinuousModel(builder)
    return model_factory.build(
        {
            "actions_num": 16,
            "input_shape": (CANONICAL_MASKED_OBS_DIM,),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    )


def _evidence_bank() -> CanonicalEvidenceBank:
    r"""构造两行完整 canonical raw evidence，验证 rl_games 边界确实执行 geometry encoder。"""

    rows, owners, joints = 2, 21, 16
    anchors = torch.tensor([[[0.0, 0.0, 0.0]], [[0.01, 0.0, 0.0]]])
    home_points = torch.zeros(rows, owners, 1, 3)
    home_points[1, :, 0, 0] = 0.01
    screws = torch.zeros(rows, joints, 6)
    screws[:, :, 0] = 1.0
    entity_role = torch.tensor([[0, *([1] * joints), *([2] * 4)]]).expand(rows, -1).clone()
    entity_joint_index = torch.tensor([[-1, *range(joints), *([-1] * 4)]]).expand(rows, -1).clone()
    joint_entity_index = torch.arange(1, 1 + joints).expand(rows, -1).clone()
    graph = torch.zeros(rows, owners, owners, dtype=torch.long)
    graph[1, 0, 1:] = 1
    valid_owners = torch.ones(rows, owners, dtype=torch.bool)
    valid_joints = torch.ones(rows, joints, dtype=torch.bool)
    evidence = StaticGeometryEvidence(
        anchors=anchors,
        home_surface_points=home_points,
        home_surface_mask=torch.ones(rows, owners, 1, dtype=torch.bool),
        palm_normal=torch.tensor([[0.0, 0.0, 1.0]]).expand(rows, -1).clone(),
        space_screws=screws,
        q_home=torch.zeros(rows, joints),
        entity_role=entity_role,
        entity_joint_index=entity_joint_index,
        joint_entity_index=joint_entity_index,
        shortest_path=graph,
        parent_direction=graph,
        child_direction=graph,
        entity_valid_mask=valid_owners,
        joint_valid_mask=valid_joints,
        anchor_valid_mask=torch.ones(rows, 1, dtype=torch.bool),
    )
    return CanonicalEvidenceBank(
        evidence=evidence,
        asset_ids=("asset-0", "asset-1"),
        physical_geometry_hashes=("hash-0", "hash-1"),
    )


def _obs() -> torch.Tensor:
    r"""构造两种 active DOF 的 canonical flat observations。"""

    obs = torch.randn(2, CANONICAL_MASKED_OBS_DIM)
    obs[:, -16:] = 0.0
    obs[0, -16:-9] = 1.0  # 7 active joints
    obs[1, -16:] = 1.0  # 16 active joints
    obs[:, 99] = torch.tensor([3.0, 11.0])  # asset_row，位于 joint/command/object/contact fields 之后
    return obs


def test_masked_model_excludes_ghost_from_logprob_and_entropy() -> None:
    r"""inactive action 任意变化不能改变 per-env log-prob，entropy 按 active 数取均值。"""

    model = _model()
    obs = _obs()
    actions = torch.randn(2, 16)
    poisoned = actions.clone()
    poisoned[0, 7:] = 1.0e12
    active = torch.tensor([[True] * 7 + [False] * 9, [True] * 16])
    baseline = model({"obs": obs, "prev_actions": actions, "is_train": True})
    changed = model({"obs": obs, "prev_actions": poisoned, "is_train": True})

    torch.testing.assert_close(baseline["prev_neglogp"], changed["prev_neglogp"])
    torch.testing.assert_close(baseline["entropy"], changed["entropy"])
    assert model.a2c_network.last_active_joint_mask is not None
    torch.testing.assert_close(model.a2c_network.last_active_joint_mask, active)


def test_masked_model_sampling_backward_and_checkpoint_contract() -> None:
    r"""sampling 后 action 清零 ghost，global log_std 只有一个参数且 backward 有限。"""

    model = _model()
    output = model({"obs": _obs(), "is_train": False})
    active = model.a2c_network.last_active_joint_mask
    assert active is not None
    assert torch.all(output["actions"][~active] == 0.0)
    assert sum(parameter.numel() for name, parameter in model.named_parameters() if "global_log_std" in name) == 1

    train_output = model({"obs": _obs(), "prev_actions": torch.zeros(2, 16), "is_train": True})
    loss = train_output["prev_neglogp"].mean() + train_output["entropy"].mean() + train_output["values"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert ANYMANI_MASKED_PPO_ALGO_KEY == "anymani_masked_ppo"


def test_flat_adapter_restores_per_joint_features_and_corresponding_tip_contacts() -> None:
    r"""block-flat observation 必须还原逐关节特征，四个 contact bit 不能复制给所有 TIP。"""

    model = _model()
    obs = torch.zeros(1, CANONICAL_MASKED_OBS_DIM)
    expected_q = torch.arange(16, dtype=torch.float32)
    expected_qd = expected_q + 100.0
    expected_delta = expected_q + 200.0
    expected_limits = torch.stack((expected_q - 1.0, expected_q + 1.0), dim=-1)
    obs[0, :16] = expected_q
    obs[0, 16:32] = expected_qd
    obs[0, 32:48] = expected_delta
    obs[0, 48:80] = expected_limits.flatten()
    obs[0, 95:99] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    obs[0, -16:] = 1.0

    policy_input = model.a2c_network._build_policy_input(obs)

    torch.testing.assert_close(
        policy_input.joint_features[0, :, :5],
        torch.cat(
            (
                expected_q[:, None],
                expected_qd[:, None],
                expected_delta[:, None],
                expected_limits,
            ),
            dim=-1,
        ),
    )
    torch.testing.assert_close(policy_input.owner_features[0, 17:21, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.count_nonzero(policy_input.owner_features[0, 17:21, 1:]) == 0


def test_tip_owner_mask_uses_manifest_physx_finger_order() -> None:
    r"""TIP owner 必须使用 manifest 的 index/middle/ring/thumb 轴；contact 在输入边界单独换轴。"""

    model = _model()
    obs = torch.zeros(1, CANONICAL_MASKED_OBS_DIM)
    obs[0, -16] = 1.0  # index_j0
    obs[0, -13] = 1.0  # thumb_j0

    policy_input = model.a2c_network._build_policy_input(obs)

    torch.testing.assert_close(
        policy_input.owner_valid_mask[0, 17:21],
        torch.tensor([True, False, False, True]),  # index/middle/ring/thumb
    )


def test_model_gathers_real_graph_and_retains_geometry_encoder_in_checkpoint() -> None:
    r"""asset_row 必须选择真实 graph/evidence，retained encoder 参数必须进入模型 state_dict。"""

    builder = CanonicalMaskedPpoBuilder()
    builder.load(
        {
            "canonical_policy": {"owner_feature_dim": 32, "joint_feature_dim": 16},
            "canonical_evidence_bank": _evidence_bank(),
        }
    )
    model = AnyManiMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": (CANONICAL_MASKED_OBS_DIM,),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    )
    obs = torch.zeros(2, CANONICAL_MASKED_OBS_DIM)
    obs[:, -16:] = 1.0
    obs[:, 99] = torch.tensor([0.0, 1.0])

    policy_input = model.a2c_network._build_policy_input(obs)
    output = model({"obs": obs, "is_train": False})

    assert policy_input.shortest_path[0, 0, 1] == 0
    assert policy_input.shortest_path[1, 0, 1] == 1
    assert torch.isfinite(output["actions"]).all()
    assert any(key.startswith("a2c_network.geometry_encoder.") for key in model.state_dict())


def test_scheduler_kl_is_invariant_to_active_dof_count_and_ghost_values() -> None:
    r"""相同逐关节策略变化在 1/2 个 active DOF 下应给同一 KL，ghost 任意值不参与。"""

    current_mu = torch.tensor([[0.2, 1.0e6], [0.2, 0.2]])
    current_sigma = torch.tensor([[0.8, 5.0], [0.8, 0.8]])
    old_mu = torch.zeros_like(current_mu)
    old_sigma = torch.ones_like(current_sigma)
    mask = torch.tensor([[True, False], [True, True]])

    kl = AnyManiMaskedPpoAgent.masked_policy_kl(current_mu, current_sigma, old_mu, old_sigma, mask)

    torch.testing.assert_close(kl[0], kl[1])
