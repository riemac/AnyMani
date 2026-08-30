r"""N040 联合模型的 Z、density 与 Gamma proper-SE(3) parity。"""

from __future__ import annotations

import torch
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.decoders.representations.implicit_field import ScalarSigmaFiLMDensityDecoderCfg
from anymani.distill.models.decoders.representations.material_point_jacobian import (
    AnchorRelationalJacobianDecoderCfg,
)
from anymani.distill.models.input_adapters.evidence import StaticGeometryEvidence
from anymani.distill.models.input_adapters.se3_gauge import rewrite_static_geometry_evidence_se3
from anymani.distill.models.input_adapters.se3_invariant_encoder import (
    SE3InvariantAnchorFrontendCfg,
    SE3InvariantGeometryEncoderCfg,
)
from anymani.distill.models.se3_density_material_jacobian_ssl import (
    SE3DensityMaterialJacobianModelCfg,
    SE3DensityMaterialJacobianSSLModel,
)


def _evidence() -> StaticGeometryEvidence:
    r"""构造单 JOINT、三 owner、四 anchors 的非对称 physical evidence。"""

    omega = torch.tensor((0.37, -0.51, 0.776), dtype=torch.float64)
    omega = omega / torch.linalg.vector_norm(omega)
    point = torch.tensor((0.034, -0.019, 0.027), dtype=torch.float64)
    linear = -torch.cross(omega, point, dim=-1)
    return StaticGeometryEvidence(
        anchors=torch.tensor(
            ((-0.047, -0.031, 0.004), (-0.021, 0.038, -0.003), (0.029, 0.027, 0.006), (0.052, -0.018, -0.002)),
            dtype=torch.float64,
        ),
        home_surface_points=torch.tensor(
            (
                ((-0.04, -0.03, 0.0), (0.04, 0.03, 0.0)),
                ((0.03, -0.01, 0.01), (0.06, 0.01, 0.02)),
                ((0.07, -0.01, 0.015), (0.09, 0.01, 0.025)),
            ),
            dtype=torch.float64,
        ),
        home_surface_mask=torch.ones(3, 2, dtype=torch.bool),
        palm_normal=torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64),
        space_screws=torch.cat((omega, linear)).unsqueeze(0),
        q_home=torch.tensor((0.13,), dtype=torch.float64),
        entity_role=torch.tensor((0, 1, 2), dtype=torch.long),
        entity_joint_index=torch.tensor((-1, 0, -1), dtype=torch.long),
        joint_entity_index=torch.tensor((1,), dtype=torch.long),
        shortest_path=torch.tensor(((0, 1, 2), (1, 0, 1), (2, 1, 0)), dtype=torch.long),
        parent_direction=torch.tensor(((0, 1, 2), (0, 0, 1), (0, 0, 0)), dtype=torch.long),
        child_direction=torch.tensor(((0, 0, 0), (1, 0, 0), (2, 1, 0)), dtype=torch.long),
    )


def _model() -> SE3DensityMaterialJacobianSSLModel:
    r"""构造小容量完整 invariant model。"""

    encoder = SE3InvariantGeometryEncoderCfg(
        frontend=SE3InvariantAnchorFrontendCfg(
            relation_width=24,
            home_width=24,
            screw_width=16,
            role_width=8,
            length_scale_m=0.1,
        ),
        backbone=GraphBiasedTransformerCfg(
            hidden_width=48,
            layers=2,
            attention_heads=4,
            feedforward_width=96,
            dropout=0.0,
            max_graph_distance=4,
        ),
    )
    return SE3DensityMaterialJacobianSSLModel(
        SE3DensityMaterialJacobianModelCfg(
            encoder=encoder,
            density=ScalarSigmaFiLMDensityDecoderCfg(hidden_width=48, residual_blocks=1),
            material_jacobian=AnchorRelationalJacobianDecoderCfg(
                latent_width=48,
                relation_width=24,
                hidden_width=48,
            ),
        )
    ).double()


def test_full_joint_model_is_invariant_to_proper_se3_coordinate_rewrite() -> None:
    r"""同一物理 q/evidence/query 的任意 proper-SE3 表达必须给出相同 Z、density 与 Gamma。"""

    torch.manual_seed(47)
    model = _model().eval()
    evidence = _evidence()
    q = torch.tensor(((0.31,), (-0.27,)), dtype=torch.float64)
    queries = torch.randn(2, 3, 5, 3, dtype=torch.float64) * 0.04
    bandwidths = torch.tensor((0.004, 0.016, 0.064), dtype=torch.float64)
    owner = torch.tensor(((1, 2), (1, 2)), dtype=torch.long)
    joint = torch.zeros_like(owner)
    material = torch.tensor(((0, 1), (0, 1)), dtype=torch.long)
    baseline = model(q, evidence, queries, bandwidths, owner, joint, material)

    axis = torch.tensor((0.31, -0.72, 0.62), dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis
    skew = torch.tensor(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=torch.float64)
    theta = torch.tensor(0.83, dtype=torch.float64)
    rotation = torch.eye(3, dtype=torch.float64) + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)
    translation = torch.tensor((0.041, -0.027, 0.019), dtype=torch.float64)
    rewritten_evidence = rewrite_static_geometry_evidence_se3(
        evidence,
        rotation=rotation,
        translation=translation,
    )
    rewritten_queries = queries @ rotation.T + translation
    actual = model(q, rewritten_evidence, rewritten_queries, bandwidths, owner, joint, material)

    torch.testing.assert_close(actual.latents.entities, baseline.latents.entities, atol=3.0e-10, rtol=3.0e-10)
    torch.testing.assert_close(actual.density, baseline.density, atol=3.0e-10, rtol=3.0e-10)
    torch.testing.assert_close(actual.material_jacobian, baseline.material_jacobian, atol=3.0e-10, rtol=3.0e-10)
