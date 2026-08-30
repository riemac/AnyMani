r"""Density + relational Material-point Jacobian 联合模型合同。"""

from __future__ import annotations

import torch
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import ScalarSigmaFiLMDensityDecoderCfg
from anymani.distill.models.decoders.representations.material_point_jacobian import (
    AnchorRelationalJacobianDecoderCfg,
)
from anymani.distill.models.density_material_jacobian_ssl import (
    DensityMaterialJacobianModelCfg,
    DensityMaterialJacobianSSLModel,
)
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    SO2AnchorFrontendCfg,
    StaticGeometryEvidence,
)


def _evidence() -> StaticGeometryEvidence:
    r"""构造 PALM–JOINT–TIP 与非对称 anchors/home material identities。"""

    return StaticGeometryEvidence(
        anchors=torch.tensor(
            ((-0.04, -0.03, 0.0), (-0.02, 0.04, 0.01), (0.03, 0.02, -0.01), (0.05, -0.02, 0.0)),
            dtype=torch.float64,
        ),
        home_surface_points=torch.tensor(
            (
                ((-0.03, -0.02, 0.0), (0.03, 0.02, 0.0)),
                ((0.04, -0.01, 0.01), (0.06, 0.01, 0.01)),
                ((0.07, -0.01, 0.015), (0.08, 0.01, 0.015)),
            ),
            dtype=torch.float64,
        ),
        home_surface_mask=torch.ones(3, 2, dtype=torch.bool),
        palm_normal=torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64),
        space_screws=torch.tensor(((0.0, 0.0, 1.0, 0.0, 0.0, 0.0),), dtype=torch.float64),
        q_home=torch.zeros(1, dtype=torch.float64),
        entity_role=torch.tensor((0, 1, 2), dtype=torch.long),
        entity_joint_index=torch.tensor((-1, 0, -1), dtype=torch.long),
        joint_entity_index=torch.tensor((1,), dtype=torch.long),
        shortest_path=torch.tensor(((0, 1, 2), (1, 0, 1), (2, 1, 0)), dtype=torch.long),
        parent_direction=torch.tensor(((0, 1, 2), (0, 0, 1), (0, 0, 0)), dtype=torch.long),
        child_direction=torch.tensor(((0, 0, 0), (1, 0, 0), (2, 1, 0)), dtype=torch.long),
    )


def _model() -> DensityMaterialJacobianSSLModel:
    r"""使用小容量但完整的联合模型。"""

    encoder = GeometryEncoderCfg(
        frontend=SO2AnchorFrontendCfg(
            relation_width=16,
            home_width=16,
            screw_width=16,
            role_width=4,
            length_scale_m=0.1,
        ),
        backbone=GraphBiasedTransformerCfg(
            hidden_width=32,
            layers=1,
            attention_heads=4,
            feedforward_width=64,
            dropout=0.0,
            max_graph_distance=4,
        ),
    )
    return DensityMaterialJacobianSSLModel(
        DensityMaterialJacobianModelCfg(
            encoder=encoder,
            density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=32, residual_blocks=1),
            material_jacobian=AnchorRelationalJacobianDecoderCfg(
                latent_width=32,
                relation_width=16,
                hidden_width=32,
            ),
        )
    ).double()


def test_joint_model_outputs_density_gamma_and_retains_only_encoder() -> None:
    r"""两个 readers 共享 unified Z，retained state 不包含任何 reader 参数。"""

    torch.manual_seed(13)
    model = _model()
    evidence = _evidence()
    q = torch.tensor(((0.2,), (-0.3,)), dtype=torch.float64)
    queries = torch.randn(2, 3, 5, 3, dtype=torch.float64) * 0.03
    bandwidths = torch.tensor(((0.004, 0.016, 0.064), (0.004, 0.016, 0.064)), dtype=torch.float64)
    owner = torch.tensor(((1, 2, 0), (1, 2, 0)), dtype=torch.long)
    joint = torch.zeros_like(owner)
    material = torch.tensor(((0, 1, 0), (0, 1, 0)), dtype=torch.long)
    output = model(q, evidence, queries, bandwidths, owner, joint, material)

    assert output.latents.entities.shape == (2, 3, 32)
    assert output.density.shape == (2, 3, 5, 3)
    assert output.material_jacobian.shape == (2, 3, 4, 4)
    assert output.material_pair_features.shape == (2, 3, 4, 16)
    retained = model.retained_state_dict()
    assert retained
    assert all(name.startswith("encoder.") for name in retained)
    assert not any("decoder" in name for name in retained)


def test_joint_model_backpropagates_both_tasks_into_shared_encoder() -> None:
    r"""Density 与 Gamma losses 都必须单独连接到同一个 retained encoder 参数集合。"""

    torch.manual_seed(17)
    model = _model()
    evidence = _evidence()
    q = torch.tensor(((0.15,), (-0.25,)), dtype=torch.float64)
    queries = torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.02
    bandwidths = torch.tensor((0.004, 0.016, 0.064), dtype=torch.float64)
    owner = torch.tensor(((1, 2), (1, 2)), dtype=torch.long)
    joint = torch.zeros_like(owner)
    material = torch.tensor(((0, 1), (0, 1)), dtype=torch.long)
    output = model(q, evidence, queries, bandwidths, owner, joint, material)
    shared = tuple(model.encoder.parameters())
    density_grad = torch.autograd.grad(output.density.square().mean(), shared, retain_graph=True, allow_unused=True)
    gamma_grad = torch.autograd.grad(output.material_jacobian.square().mean(), shared, allow_unused=True)

    assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in density_grad)
    assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gamma_grad)


def test_explicit_latent_replay_supports_query_only_intervention() -> None:
    r"""固定 query/material features 时，zero-Z intervention 应只改变 readers 的 latent condition。"""

    torch.manual_seed(23)
    model = _model()
    evidence = _evidence()
    q = torch.tensor(((0.1,), (-0.2,)), dtype=torch.float64)
    queries = torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.02
    bandwidths = torch.tensor((0.004, 0.016, 0.064), dtype=torch.float64)
    owner = torch.tensor(((1, 2), (1, 2)), dtype=torch.long)
    joint = torch.zeros_like(owner)
    material = torch.tensor(((0, 1), (0, 1)), dtype=torch.long)
    full = model(q, evidence, queries, bandwidths, owner, joint, material)
    zero_latents = type(full.latents)(entities=torch.zeros_like(full.latents.entities))
    query_only = model.decode_features(
        zero_latents,
        full.query_features,
        full.material_pair_features,
        bandwidths,
        evidence,
        owner,
        joint,
        evidence_row_index=None,
        entity_valid_mask=None,
    )

    assert query_only.density.shape == full.density.shape
    assert query_only.material_jacobian.shape == full.material_jacobian.shape
    assert not torch.equal(query_only.density, full.density)
    assert not torch.equal(query_only.material_jacobian, full.material_jacobian)
