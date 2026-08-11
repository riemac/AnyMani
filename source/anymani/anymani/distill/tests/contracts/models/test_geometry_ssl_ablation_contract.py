"""query-only 与 latent-shuffle 不改变 query path 的合同。"""

from __future__ import annotations

import torch
from anymani.distill.diagnostics.evaluation.geometry_ssl import geometry_ssl_ablation_forward
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
