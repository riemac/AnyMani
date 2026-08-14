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
from anymani.distill.objectives.representations.field_reconstruction import selected_density_coordinate_derivative


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
        torch.tensor([0.004, 0.016], dtype=torch.float64),
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
        torch.tensor([0.004, 0.016], dtype=torch.float64),
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
        "bandwidths": torch.tensor([0.004, 0.016], dtype=torch.float64),
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


def test_density_decoder_treats_sigma_as_a_variable_data_axis() -> None:
    """同一 scalar decoder 应接受任意 sigma 数量，重复 sigma 必须产生相同逐点读取。"""

    model = _model().eval()
    evidence = _evidence()
    q = torch.tensor([[0.2], [-0.3]], dtype=torch.float64)
    queries = torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.02
    latents = model.encoder(q, evidence)
    query_features = model.encoder.encode_points(queries, evidence)

    repeated_sigma = torch.tensor([0.004, 0.004], dtype=torch.float64, requires_grad=True)
    repeated = model.density_decoder(latents.zero_order, query_features, repeated_sigma)
    five_sigma = model.density_decoder(
        latents.zero_order,
        query_features,
        torch.tensor([0.004, 0.008, 0.016, 0.032, 0.064], dtype=torch.float64),
    )

    assert repeated.shape == (2, 3, 4, 2)
    assert five_sigma.shape == (2, 3, 4, 5)
    torch.testing.assert_close(repeated[..., 0], repeated[..., 1], atol=1.0e-15, rtol=1.0e-15)
    assert torch.max(torch.abs(five_sigma[..., 0] - five_sigma[..., -1])) > 1.0e-8
    repeated.sum().backward()
    assert repeated_sigma.grad is None


def test_density_q_jvp_holds_explicit_sigma_fixed() -> None:
    """Sobolev 导数只沿 physical q 图传播，外生 sigma 即使标记梯度也必须被截断。"""

    model = _model().eval()
    q = torch.tensor([[0.2], [-0.3]], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([[0.004, 0.016], [0.004, 0.016]], dtype=torch.float64, requires_grad=True)
    owner_index = torch.tensor([1], dtype=torch.long)
    query_index = torch.tensor([0], dtype=torch.long)
    joint_index = torch.tensor([0], dtype=torch.long)
    prediction = model(
        q,
        _evidence(),
        torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.02,
        sigma,
        owner_index,
        query_index,
        joint_index,
    )
    derivative = selected_density_coordinate_derivative(
        prediction.density,
        q,
        owner_index,
        query_index,
        joint_index,
        create_graph=True,
    )
    (prediction.density.sum() + derivative.square().sum()).backward()

    assert derivative.shape == (2, 1, 2)
    assert torch.isfinite(derivative).all()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert sigma.grad is None
