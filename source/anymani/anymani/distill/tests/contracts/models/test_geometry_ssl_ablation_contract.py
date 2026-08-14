"""query-only 与 latent-shuffle 不改变 query path 的合同。"""

from __future__ import annotations

import torch
from anymani.distill.diagnostics.evaluation.geometry_ssl import (
    cross_asset_permutation,
    geometry_ssl_ablation_forward,
    same_asset_q_permutation,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelConfig
from anymani.distill.models.input_adapters.geometry import GeometryEncoderConfig, StaticGeometryEvidence


def _evidence() -> StaticGeometryEvidence:
    """构造 1-JOINT 三实体证据。"""

    return StaticGeometryEvidence(
        anchors=torch.tensor([[-0.03, -0.02, 0.0], [0.03, 0.02, 0.0]], dtype=torch.float64),
        home_surface_points=torch.tensor(
            [
                [[-0.03, -0.02, 0.0], [0.03, 0.02, 0.0]],
                [[0.04, -0.01, 0.0], [0.05, 0.01, 0.0]],
                [[0.07, -0.01, 0.0], [0.08, 0.01, 0.0]],
            ],
            dtype=torch.float64,
        ),
        home_surface_mask=torch.ones(3, 2, dtype=torch.bool),
        palm_normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        space_screws=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        q_home=torch.zeros(1, dtype=torch.float64),
        entity_role=torch.tensor([0, 1, 2]),
        entity_joint_index=torch.tensor([-1, 0, -1]),
        joint_entity_index=torch.tensor([1]),
        shortest_path=torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
        parent_direction=torch.tensor([[0, 4, 4], [1, 0, 4], [2, 1, 0]]),
        child_direction=torch.tensor([[0, 1, 2], [4, 0, 1], [4, 4, 0]]),
    )


def _model() -> GeometrySSLModel:
    """返回无 dropout 的确定性小模型。"""

    return GeometrySSLModel(
        GeometrySSLModelConfig(
            encoder=GeometryEncoderConfig(
                relation_width=8,
                home_width=8,
                screw_width=8,
                hidden_width=16,
                zero_order_width=12,
                first_order_width=8,
                transformer_layers=1,
                attention_heads=4,
                feedforward_width=24,
                dropout=0.0,
            ),
            decoder_hidden_width=16,
            decoder_residual_blocks=1,
            bandwidth_count=2,
        )
    ).to(dtype=torch.float64)


def _two_joint_evidence() -> StaticGeometryEvidence:
    """构造 PALM–JOINT–JOINT 三实体证据，专门验证 JOINT 轴干预。"""

    base = _evidence()
    return StaticGeometryEvidence(
        anchors=base.anchors,
        home_surface_points=base.home_surface_points,
        home_surface_mask=base.home_surface_mask,
        palm_normal=base.palm_normal,
        space_screws=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, -0.04, 0.0]],
            dtype=torch.float64,
        ),
        q_home=torch.zeros(2, dtype=torch.float64),
        entity_role=torch.tensor([0, 1, 1]),
        entity_joint_index=torch.tensor([-1, 0, 1]),
        joint_entity_index=torch.tensor([1, 2]),
        shortest_path=base.shortest_path,
        parent_direction=base.parent_direction,
        child_direction=base.child_direction,
    )


def test_query_only_zeros_latents_but_preserves_query_features() -> None:
    """query-only baseline 只能删除 morphology latent，不能换 query encoder。"""

    model = _model().eval()
    evidence = _evidence()
    q = torch.tensor([[0.2], [-0.3]], dtype=torch.float64)
    queries = torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.02
    expected_query = model.encoder.encode_points(queries, evidence)
    result = geometry_ssl_ablation_forward(
        model,
        q,
        evidence,
        queries,
        owner_index=torch.tensor([1]),
        query_index=torch.tensor([0]),
        joint_index=torch.tensor([0]),
        ablation="query_only",
    )

    assert torch.count_nonzero(result.latents.zero_order) == 0
    assert torch.count_nonzero(result.latents.first_order) == 0
    torch.testing.assert_close(result.query_features, expected_query)


def test_latent_shuffle_reorders_only_batch_latents() -> None:
    """shuffle 必须错配 conditioning，不得同步打乱 query 或 selector。"""

    model = _model().eval()
    evidence = _evidence()
    q = torch.tensor([[0.2], [-0.3]], dtype=torch.float64)
    queries = torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.02
    original = model.encoder(q, evidence)
    result = geometry_ssl_ablation_forward(
        model,
        q,
        evidence,
        queries,
        owner_index=torch.tensor([1]),
        query_index=torch.tensor([0]),
        joint_index=torch.tensor([0]),
        ablation="latent_shuffle",
        batch_permutation=torch.tensor([1, 0]),
    )

    torch.testing.assert_close(result.latents.zero_order, original.zero_order.flip(0))
    torch.testing.assert_close(result.latents.first_order, original.first_order.flip(0))
    torch.testing.assert_close(result.query_features, model.encoder.encode_points(queries, evidence))


def test_fixed_asset_permutations_preserve_declared_shuffle_semantics() -> None:
    """同手 shuffle 不跨资产，跨手 shuffle 的每个 source 必须来自不同资产。"""

    asset_ids = ("a", "a", "b", "b")
    same_asset = same_asset_q_permutation(asset_ids, device=torch.device("cpu"))
    cross_asset = cross_asset_permutation(asset_ids, device=torch.device("cpu"))

    assert same_asset.tolist() == [1, 0, 3, 2]
    assert all(asset_ids[source] == asset_ids[target] for target, source in enumerate(same_asset.tolist()))
    assert all(asset_ids[source] != asset_ids[target] for target, source in enumerate(cross_asset.tolist()))


def test_first_order_ablations_leave_zero_order_and_query_path_unchanged() -> None:
    """z1 zero/sign/JOINT shuffle 只能干预一阶包，不得暗改零阶 morphology 或 query evidence。"""

    model = _model().eval()
    evidence = _two_joint_evidence()
    q = torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float64)
    queries = torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.02
    original = model.encoder(q, evidence)
    common = {
        "owner_index": torch.tensor([1, 2]),
        "query_index": torch.tensor([0, 1]),
        "joint_index": torch.tensor([0, 1]),
    }

    zero = geometry_ssl_ablation_forward(model, q, evidence, queries, ablation="first_order_zero", **common)
    sign = geometry_ssl_ablation_forward(model, q, evidence, queries, ablation="first_order_sign_flip", **common)
    shuffled = geometry_ssl_ablation_forward(
        model, q, evidence, queries, ablation="first_order_joint_shuffle", **common
    )

    for result in (zero, sign, shuffled):
        torch.testing.assert_close(result.latents.zero_order, original.zero_order)
        torch.testing.assert_close(result.query_features, model.encoder.encode_points(queries, evidence))
    assert torch.count_nonzero(zero.latents.first_order) == 0
    torch.testing.assert_close(sign.latents.first_order, -original.first_order)
    torch.testing.assert_close(shuffled.latents.first_order, original.first_order.roll(1, dims=1))
