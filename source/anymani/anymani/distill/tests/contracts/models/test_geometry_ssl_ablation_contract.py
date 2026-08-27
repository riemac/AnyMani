"""query-only 与 latent-shuffle 不改变 query path 的合同。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from anymani.distill.diagnostics.evaluation.geometry_ssl import (
    cross_asset_permutation,
    density_configuration_jvp,
    geometry_ssl_ablation_forward,
    joint_sign_observable_metrics,
    same_asset_q_permutation,
    task_gradient_gram,
)
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import (
    ConditionalDensityDecoder,
    DistanceSensitivityDecoder,
    DistanceSensitivityDecoderCfg,
    GeometrySSLDecoderCfg,
    ScalarSigmaFiLMDensityDecoderCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel, GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryLatents,
    GeometryPaddingCfg,
    SO2AnchorFrontendCfg,
    StaticGeometryEvidence,
    pad_static_geometry_evidence,
)


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
        GeometrySSLModelCfg(
            encoder=GeometryEncoderCfg(
                frontend=SO2AnchorFrontendCfg(relation_width=8, home_width=8, screw_width=8),
                backbone=GraphBiasedTransformerCfg(
                    hidden_width=16,
                    layers=1,
                    attention_heads=4,
                    feedforward_width=24,
                    dropout=0.0,
                ),
            ),
            ssl_decoders=GeometrySSLDecoderCfg(
                density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=16, residual_blocks=1),
                sensitivity=DistanceSensitivityDecoderCfg(
                    hidden_width=16,
                    residual_blocks=2,
                    readout_rank=8,
                    physical_scale_m=0.1,
                ),
            ),
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

    assert torch.count_nonzero(result.latents.entities) == 0
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

    torch.testing.assert_close(result.latents.entities, original.entities.flip(0))
    torch.testing.assert_close(result.query_features, model.encoder.encode_points(queries, evidence))


def test_fixed_asset_permutations_preserve_declared_shuffle_semantics() -> None:
    """同手 shuffle 不跨资产，跨手 shuffle 的每个 source 必须来自不同资产。"""

    asset_ids = ("a", "a", "b", "b")
    same_asset = same_asset_q_permutation(asset_ids, device=torch.device("cpu"))
    cross_asset = cross_asset_permutation(asset_ids, device=torch.device("cpu"))

    assert same_asset.tolist() == [1, 0, 3, 2]
    assert all(asset_ids[source] == asset_ids[target] for target, source in enumerate(same_asset.tolist()))
    assert all(asset_ids[source] != asset_ids[target] for target, source in enumerate(cross_asset.tolist()))


def test_joint_token_shuffle_changes_only_valid_joint_entities() -> None:
    """JOINT shuffle 只错配统一 Z 中的有效 JOINT tokens，不得暗改 PALM/TIP 或 query evidence。"""

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

    shuffled = geometry_ssl_ablation_forward(
        model, q, evidence, queries, ablation="joint_token_shuffle", **common
    )

    torch.testing.assert_close(shuffled.latents.entities[:, :1], original.entities[:, :1])
    torch.testing.assert_close(shuffled.latents.entities[:, 1:3], original.entities[:, 1:3].roll(1, dims=1))
    torch.testing.assert_close(shuffled.query_features, model.encoder.encode_points(queries, evidence))


def test_density_decoder_treats_sigma_as_a_variable_data_axis() -> None:
    """同一 scalar decoder 应接受任意 sigma 数量，重复 sigma 必须产生相同逐点读取。"""

    model = _model().eval()
    evidence = _evidence()
    q = torch.tensor([[0.2], [-0.3]], dtype=torch.float64)
    queries = torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.02
    latents = model.encoder(q, evidence)
    query_features = model.encoder.encode_points(queries, evidence)

    repeated_sigma = torch.tensor([0.004, 0.004], dtype=torch.float64, requires_grad=True)
    repeated = model.density_decoder(latents.entities, query_features, repeated_sigma)
    five_sigma = model.density_decoder(
        latents.entities,
        query_features,
        torch.tensor([0.004, 0.008, 0.016, 0.032, 0.064], dtype=torch.float64),
    )

    assert repeated.shape == (2, 3, 4, 2)
    assert five_sigma.shape == (2, 3, 4, 5)
    torch.testing.assert_close(repeated[..., 0], repeated[..., 1], atol=1.0e-15, rtol=1.0e-15)
    assert torch.max(torch.abs(five_sigma[..., 0] - five_sigma[..., -1])) > 1.0e-8
    repeated.sum().backward()
    assert repeated_sigma.grad is None


def test_model_unique_evidence_rows_match_fully_expanded_reference() -> None:
    r"""q-row routing必须同时覆盖 encoder、query anchors、mask 与 JOINT entity selector。"""

    model = _model().eval()
    first = _evidence()
    second = replace(first, anchors=first.anchors + torch.tensor([0.004, -0.002, 0.001], dtype=torch.float64))
    padding = GeometryPaddingCfg(max_joint_count=1, max_tip_count=1, max_graph_distance=8)
    unique = pad_static_geometry_evidence((first, second), config=padding)
    expanded = pad_static_geometry_evidence((first, first, second, second), config=padding)
    row_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    q = torch.tensor([[0.2], [-0.3], [0.1], [0.4]], dtype=torch.float64)
    queries = torch.randn(4, 3, 4, 3, dtype=torch.float64) * 0.02
    kwargs = {
        "bandwidths": torch.tensor([0.004, 0.016], dtype=torch.float64),
        "owner_index": torch.tensor([1]),
        "query_index": torch.tensor([0]),
        "joint_index": torch.tensor([0]),
    }

    routed = model(q, unique, queries, evidence_row_index=row_index, **kwargs)
    reference = model(q, expanded, queries, **kwargs)

    torch.testing.assert_close(routed.latents.entities, reference.latents.entities, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(routed.query_features, reference.query_features, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(routed.density, reference.density, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(routed.kappa, reference.kappa, atol=1.0e-12, rtol=1.0e-12)


def test_ablation_unique_evidence_rows_match_expanded_reference() -> None:
    r"""默认 fixed-bank 的 B q-rows/A unique-assets routing 必须贯穿 ablation 全路径。"""

    model = _model().eval()
    first = _evidence()
    second = replace(first, anchors=first.anchors + torch.tensor([0.004, -0.002, 0.001], dtype=torch.float64))
    padding = GeometryPaddingCfg(max_joint_count=1, max_tip_count=1, max_graph_distance=8)
    unique = pad_static_geometry_evidence((first, second), config=padding)
    expanded = pad_static_geometry_evidence((first, first, second, second), config=padding)
    row_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    q = torch.tensor([[0.2], [-0.3], [0.1], [0.4]], dtype=torch.float64)
    queries = torch.randn(4, 3, 4, 3, dtype=torch.float64) * 0.02
    kwargs = {
        "bandwidths": torch.tensor([0.004, 0.016], dtype=torch.float64),
        "owner_index": torch.tensor([1]),
        "query_index": torch.tensor([0]),
        "joint_index": torch.tensor([0]),
        "ablation": "query_only",
    }

    routed = geometry_ssl_ablation_forward(
        model,
        q,
        unique,
        queries,
        evidence_row_index=row_index,
        **kwargs,
    )
    reference = geometry_ssl_ablation_forward(model, q, expanded, queries, **kwargs)

    torch.testing.assert_close(routed.query_features, reference.query_features, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(routed.density, reference.density, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(routed.kappa, reference.kappa, atol=1.0e-12, rtol=1.0e-12)


def test_every_density_residual_block_reads_owner_latent_through_film() -> None:
    r"""两层 canonical FiLM 路径必须各自拥有 $z_g\to(\gamma,\beta)$ 投影。"""

    decoder = ConditionalDensityDecoder(
        ScalarSigmaFiLMDensityDecoderCfg(hidden_width=16, residual_blocks=2),
        entity_width=12,
        query_width=8,
    )

    assert len(decoder.blocks) == 2
    assert all(block.modulation.in_features == 12 for block in decoder.blocks)
    assert all(block.modulation.out_features == 32 for block in decoder.blocks)
    assert {
        name for name in decoder.state_dict() if name.endswith("modulation.weight")
    } == {f"blocks.{index}.modulation.weight" for index in range(2)}


def test_kappa_decoder_uses_owner_query_row_and_joint_column_without_label_leakage() -> None:
    r"""两层 owner-FiLM 与低秩 JOINT 列读取必须共同决定有物理单位的 signed scalar。"""

    model = _model()
    decoder = model.sensitivity_decoder
    assert len(decoder.blocks) == 2
    assert all(block.modulation.in_features == 16 for block in decoder.blocks)
    assert decoder.row_projection.out_features == 8
    assert decoder.joint_projection.out_features == 8
    assert decoder.row_projection.bias is None
    assert decoder.joint_projection.bias is None
    assert not hasattr(decoder, "output")
    assert not any("sigma" in name or "screw" in name for name, _module in decoder.named_modules())

    owner = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
    joint = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
    query = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    prediction = decoder(owner, joint, query)
    prediction.square().mean().backward()

    assert prediction.shape == (2, 4)
    assert owner.grad is not None and torch.count_nonzero(owner.grad) > 0
    assert joint.grad is not None and torch.count_nonzero(joint.grad) > 0
    assert query.grad is not None and torch.count_nonzero(query.grad) > 0


@pytest.mark.parametrize("seed", [3, 17, 101])
def test_kappa_decoder_initial_output_has_declared_physical_rms(seed: int) -> None:
    r"""低秩双侧小非零初始化应把初始 $\hat\kappa$ RMS 锚定在 0.0125 m/rad 附近。"""

    torch.manual_seed(seed)
    decoder = DistanceSensitivityDecoder(
        DistanceSensitivityDecoderCfg(
            hidden_width=128,
            residual_blocks=2,
            readout_rank=64,
            physical_scale_m=0.1,
        ),
        entity_width=128,
        query_width=64,
    )
    owner = torch.randn(32, 96, 128)
    joint = torch.randn(32, 96, 128)
    query = torch.randn(32, 96, 64)

    prediction = decoder(owner, joint, query)
    rms = float(prediction.square().mean().sqrt())

    assert 0.00625 <= rms <= 0.025, f"initial kappa RMS={rms:.6g} m/rad is outside declared range"


def test_manual_observable_rewrite_jvp_and_parameter_gradient_probes() -> None:
    r"""post-hoc API 只检查 observable/JVP/所选参数梯度，不引入 latent parity objective。"""

    density = torch.tensor([[[[0.2], [0.4]]]], dtype=torch.float64)
    kappa = torch.tensor([[0.3, -0.5]], dtype=torch.float64)
    latents = GeometryLatents(torch.zeros(1, 1, 2, dtype=torch.float64))
    query_features = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    reference = GeometrySSLForward(latents, query_features, density, kappa)
    rewritten = GeometrySSLForward(
        latents,
        query_features,
        density.clone(),
        torch.tensor([[-0.3, -0.5]], dtype=torch.float64),
    )
    parity = joint_sign_observable_metrics(
        reference,
        rewritten,
        joint_sign=torch.tensor([-1.0, 1.0]),
        joint_index=torch.tensor([0, 1]),
        density_valid_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        edge_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert parity == {"density_invariance_mse": 0.0, "kappa_sign_equivariance_mse": 0.0}

    model = _model().eval()
    evidence = _evidence()
    q = torch.tensor([[0.2], [-0.3]], dtype=torch.float64)
    queries = torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.02
    primal, tangent = density_configuration_jvp(
        model,
        q,
        evidence,
        queries,
        torch.tensor([0.004, 0.016], dtype=torch.float64),
        owner_index=torch.tensor([1]),
        query_index=torch.tensor([0]),
        joint_index=torch.tensor([0]),
        direction=torch.ones_like(q),
    )
    assert primal.shape == tangent.shape == (2, 3, 4, 2)
    assert torch.isfinite(tangent).all()

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    gram = task_gradient_gram(
        {"density": parameter.square().sum(), "kappa": (parameter - 3.0).square().sum()},
        (parameter,),
        baselines={"density": 1.0, "kappa": 2.0},
    )
    assert gram["rho_norm"] > 0.0 and gram["kappa_norm"] > 0.0
    assert -1.0 <= gram["cosine"] <= 1.0
