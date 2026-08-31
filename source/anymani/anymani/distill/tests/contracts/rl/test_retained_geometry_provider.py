"""冻结 N040 provider 的 q-dependent Z、routing、identity 与梯度隔离合同。"""

from __future__ import annotations

from pathlib import Path

import torch
from anymani.distill.methods.density_material_jacobian.artifact import SE3RetainedEncoderArtifact
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.artifact import RetainedLoadReport
from anymani.distill.models.input_adapters.geometry import StaticGeometryEvidence
from anymani.distill.models.input_adapters.se3_invariant_encoder import SE3InvariantGeometryEncoder
from anymani.distill.models.policy import CanonicalEvidenceBank
from anymani.distill.rl.heterogeneous_masked_ppo import (
    HETEROGENEOUS_N040_HISTORY_OBS_DIM,
    HeterogeneousN040HistoryPpoBuilder,
)
from anymani.distill.rl.masked_ppo import AnyManiMaskedContinuousModel
from anymani.distill.rl.runtime.retained_geometry import RetainedGeometryProvider


def _evidence_bank() -> CanonicalEvidenceBank:
    r"""构造两资产 canonical bank；第二行静态几何/图与第一行不同。"""

    rows, owners, joints = 2, 21, 16  # 真实 heterogeneous canonical ABI
    anchors = torch.tensor([[[0.0, 0.0, 0.0]], [[0.01, 0.0, 0.0]]])  # `[A,1,3]`，m
    home_points = torch.zeros(rows, owners, 1, 3)  # 每 owner 一个最小 home point，m
    home_points[1, :, 0, 0] = 0.01  # 第二资产沿 hand x 轴平移静态表面
    screws = torch.zeros(rows, joints, 6)  # `[A,16,6]` spatial screws
    screws[:, :, 2] = 1.0  # revolute axis $\omega=e_z$
    screws[1, :, 4] = -0.01  # 第二资产 $v=-\omega\times p_0$ 的 y 分量，m
    roles = torch.tensor([[0, *([1] * joints), *([2] * 4)]]).expand(rows, -1).clone()
    entity_joint_index = torch.tensor([[-1, *range(joints), *([-1] * 4)]]).expand(rows, -1).clone()
    joint_entity_index = torch.arange(1, 1 + joints).expand(rows, -1).clone()
    graph = torch.zeros(rows, owners, owners, dtype=torch.long)  # 第一资产最小关系桶
    graph[1, 0, 1:] = 1  # 第二资产的 graph identity 与第一资产不同
    evidence = StaticGeometryEvidence(
        anchors=anchors,
        home_surface_points=home_points,
        home_surface_mask=torch.ones(rows, owners, 1, dtype=torch.bool),
        palm_normal=torch.tensor([[0.0, 0.0, 1.0]]).expand(rows, -1).clone(),
        space_screws=screws,
        q_home=torch.zeros(rows, joints),
        entity_role=roles,
        entity_joint_index=entity_joint_index,
        joint_entity_index=joint_entity_index,
        shortest_path=graph,
        parent_direction=graph,
        child_direction=graph,
        entity_valid_mask=torch.ones(rows, owners, dtype=torch.bool),
        joint_valid_mask=torch.ones(rows, joints, dtype=torch.bool),
        anchor_valid_mask=torch.ones(rows, 1, dtype=torch.bool),
    )
    return CanonicalEvidenceBank(
        evidence=evidence,
        asset_ids=("asset-0", "asset-1"),
        physical_geometry_hashes=("physical-0", "physical-1"),
    )


def _provider() -> RetainedGeometryProvider:
    r"""构造不依赖本地正式 artifact 文件的随机初始化 N040 provider。"""

    loaded = SE3RetainedEncoderArtifact(
        encoder=SE3InvariantGeometryEncoder(),
        load_report=RetainedLoadReport((), ()),
        artifact_sha256="a" * 64,
        path=Path("/tmp/synthetic-retained-encoder.pt"),
        feature_spec={"entity_width": 128},
        input_contract={"retained_inputs": "physical q + static geometry evidence"},
        lineage={"code_revision": "synthetic"},
    )
    return RetainedGeometryProvider(
        artifact=loaded,
        evidence_bank=_evidence_bank(),
        dataset_digest="dataset",
        manifest_digest="manifest",
        canonical_schema_digest="canonical-schema",
        evidence_source_config={"static_sampling_seed": 0},
    )


def test_retained_provider_is_frozen_q_dependent_and_routes_manifest_axes() -> None:
    r"""Z 必须随当前物理 q 改变，而 masks/graph 严格跟随 asset row。"""

    torch.manual_seed(7)
    provider = _provider()
    rows = torch.tensor([1, 0, 1], dtype=torch.long)  # runtime env→asset routing
    q_zero = torch.zeros(3, 16)  # rad
    q_changed = q_zero.clone()
    q_changed[:, 0] = 0.2  # 第一个有效 JOINT 改变 $0.2\,\mathrm{rad}$

    with torch.no_grad():
        reference = provider.encoder(q_zero, provider._evidence(), evidence_row_index=rows).entities
    baseline = provider.resolve(rows, q_zero)
    changed = provider.resolve(rows, q_changed)

    assert baseline.geometry_entities.shape == (3, 21, 128)
    torch.testing.assert_close(baseline.geometry_entities, reference, atol=1.0e-6, rtol=1.0e-6)
    assert not torch.equal(baseline.geometry_entities, changed.geometry_entities)
    assert torch.equal(baseline.shortest_path[:, 0, 1], torch.tensor([1, 0, 1]))
    assert baseline.geometry_entities.requires_grad is False
    assert provider.encoder.training is False
    assert all(parameter.requires_grad is False and parameter.grad is None for parameter in provider.encoder.parameters())


def test_retained_provider_identity_uses_artifact_and_evidence_not_static_z_table() -> None:
    r"""Checkpoint identity 锚定 artifact/evidence，不得保留 hash-Z table 的伪语义。"""

    provider = _provider()
    identity = provider.identity

    assert identity["provider_type"] == "retained_se3_geometry_encoder"
    assert identity["retained_artifact"]["sha256"] == "a" * 64
    assert identity["dataset_digest"] == "dataset"
    assert identity["manifest_digest"] == "manifest"
    assert identity["evidence_tensor_digest"]
    assert identity["identity_digest"]
    assert "z_table_sha256" not in identity

    provider.train()
    assert provider.encoder.training is False  # PPO `.train()` 不得打开 frozen encoder dropout/training state


def test_history30_builder_routes_n040_temporal_limits_and_shared_action_head() -> None:
    r"""1969D flat ABI必须恢复History30 axes，并只训练TCN/policy adapter。"""

    torch.manual_seed(11)
    provider = _provider()
    builder = HeterogeneousN040HistoryPpoBuilder()
    builder.load(
        {
            "heterogeneous_policy": {
                "owner_feature_dim": 1,
                "joint_feature_dim": 6,
                "temporal_feature_dim": 64,
                "layers": 1,
            },
            "retained_geometry_provider": provider,
        }
    )
    model = AnyManiMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": (HETEROGENEOUS_N040_HISTORY_OBS_DIM,),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    )
    obs = torch.zeros(2, HETEROGENEOUS_N040_HISTORY_OBS_DIM)
    history = obs[:, : 30 * 16 * 4].reshape(2, 30, 16, 4)
    history[:, :, :, 0] = 0.1  # $q/\pi$
    history[:, :, :, 1] = 0.2  # target/$\pi$
    history[:, :, :, 2] = -0.3  # previous action
    history[:, :, :, 3] = torch.tensor([0.1, 0.2, 0.3, 0.4]).repeat(4)  # depth-major TIP copies
    limits = obs[:, 30 * 16 * 4 : 30 * 16 * 4 + 32].reshape(2, 16, 2)
    limits[:, :, 0] = -0.5  # $q_{min}/\pi$
    limits[:, :, 1] = 0.5  # $q_{max}/\pi$
    obs[:, -17] = torch.tensor([0.0, 1.0])  # asset rows
    obs[:, -16:] = 1.0  # synthetic evidence两行均为16 active DOF

    policy_input = model.a2c_network._build_policy_input(obs)
    assert policy_input.temporal_features is not None
    assert policy_input.temporal_features.shape == (2, 16, 64)
    assert policy_input.joint_features.shape == (2, 16, 6)
    torch.testing.assert_close(policy_input.owner_features[0, 17:21, 0], torch.tensor([0.1, 0.2, 0.3, 0.4]))

    output = model({"obs": obs, "prev_actions": torch.zeros(2, 16), "is_train": True})
    loss = output["prev_neglogp"].mean() + output["values"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.a2c_network.retained_geometry_provider.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.a2c_network.temporal_encoder.parameters())
    assert isinstance(model.a2c_network.policy.action_head[0], torch.nn.LayerNorm)
    assert isinstance(model.a2c_network.policy.action_head[1], torch.nn.Linear)
