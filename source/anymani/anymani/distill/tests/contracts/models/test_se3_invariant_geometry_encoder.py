r"""N040 proper-SE(3)-invariant frontend/encoder 与 legacy origin counterexample。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.input_adapters.encoder import (
    GeometryEncoderCfg,
    ImplicitGeometryEncoder,
    SO2AnchorFrontendCfg,
)
from anymani.distill.models.input_adapters.evidence import StaticGeometryEvidence, stack_static_geometry_evidence
from anymani.distill.models.input_adapters.se3_gauge import rewrite_static_geometry_evidence_se3
from anymani.distill.models.input_adapters.se3_invariant_encoder import (
    SE3InvariantAnchorFrontendCfg,
    SE3InvariantGeometryEncoder,
    SE3InvariantGeometryEncoderCfg,
)

pytestmark = pytest.mark.contract


def _evidence() -> StaticGeometryEvidence:
    r"""构造非轴对齐 screw 和非对称 anchor constellation，使 origin 泄漏可观测。"""

    omega = torch.tensor((0.37, -0.51, 0.776), dtype=torch.float64)
    omega = omega / torch.linalg.vector_norm(omega)
    axis_point = torch.tensor((0.034, -0.019, 0.027), dtype=torch.float64)
    linear = -torch.cross(omega, axis_point, dim=-1)
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


def _rotation() -> torch.Tensor:
    r"""返回任意 det=+1 的 proper rotation。"""

    axis = torch.tensor((0.31, -0.72, 0.62), dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis
    skew = torch.tensor(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=torch.float64)
    theta = torch.tensor(0.83, dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)


def _se3_encoder() -> SE3InvariantGeometryEncoder:
    r"""使用小容量完整 N040 frontend/backbone。"""

    return SE3InvariantGeometryEncoder(
        SE3InvariantGeometryEncoderCfg(
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
    ).double()


def _legacy_encoder() -> ImplicitGeometryEncoder:
    r"""构造相同容量的 N031 origin-dependent control。"""

    return ImplicitGeometryEncoder(
        GeometryEncoderCfg(
            frontend=SO2AnchorFrontendCfg(
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
    ).double()


def test_se3_encoder_screw_features_and_z_are_invariant_to_translation_and_full_se3() -> None:
    r"""正确 co-transform 后，f_screw 与 Z 只能留下 float64 舍入误差。"""

    torch.manual_seed(41)
    evidence = _evidence()
    encoder = _se3_encoder().eval()
    q = torch.tensor(((0.37,), (-0.22,)), dtype=torch.float64)
    reference_screw = encoder.screw_features(evidence)
    reference_z = encoder(q, evidence).entities
    transforms = (
        (torch.eye(3, dtype=torch.float64), torch.tensor((0.041, -0.027, 0.019), dtype=torch.float64)),
        (_rotation(), torch.tensor((0.041, -0.027, 0.019), dtype=torch.float64)),
    )
    for rotation, translation in transforms:
        rewritten = rewrite_static_geometry_evidence_se3(evidence, rotation=rotation, translation=translation)
        torch.testing.assert_close(encoder.screw_features(rewritten), reference_screw, atol=2.0e-10, rtol=2.0e-10)
        torch.testing.assert_close(encoder(q, rewritten).entities, reference_z, atol=2.0e-10, rtol=2.0e-10)


def test_legacy_encoder_exposes_origin_translation_counterexample() -> None:
    r"""同一 physical screw line 改写 origin 后，legacy Z 必须显著变化，证明测试有判别力。"""

    torch.manual_seed(43)
    evidence = _evidence()
    encoder = _legacy_encoder().eval()
    q = torch.tensor(((0.31,),), dtype=torch.float64)
    rewritten = rewrite_static_geometry_evidence_se3(
        evidence,
        rotation=torch.eye(3, dtype=torch.float64),
        translation=torch.tensor((0.041, -0.027, 0.019), dtype=torch.float64),
    )
    difference = torch.linalg.vector_norm(encoder(q, evidence).entities - encoder(q, rewritten).entities)
    assert float(difference) > 1.0e-3


def test_se3_rewrite_rejects_reflection() -> None:
    r"""Reflection 不属于 proper SE(3)，不得混入 coordinate gauge augmentation。"""

    reflection = torch.diag(torch.tensor((-1.0, 1.0, 1.0), dtype=torch.float64))
    with pytest.raises(ValueError, match="proper"):
        rewrite_static_geometry_evidence_se3(
            _evidence(),
            rotation=reflection,
            translation=torch.zeros(3, dtype=torch.float64),
        )


def test_batched_rewrite_applies_one_transform_per_unique_evidence_row() -> None:
    r"""训练 augmentation 的 asset-level transform 不得在 q rows 或 assets 间串扰。"""

    evidence = _evidence()
    batched = stack_static_geometry_evidence((evidence, evidence))
    rotations = torch.stack((torch.eye(3, dtype=torch.float64), _rotation()))
    translations = torch.tensor(((0.03, -0.01, 0.02), (-0.02, 0.04, -0.03)), dtype=torch.float64)
    actual = rewrite_static_geometry_evidence_se3(
        batched,
        rotation=rotations,
        translation=translations,
    )
    for row in range(2):
        expected = rewrite_static_geometry_evidence_se3(
            evidence,
            rotation=rotations[row],
            translation=translations[row],
        )
        torch.testing.assert_close(actual.anchors[row], expected.anchors)
        torch.testing.assert_close(actual.home_surface_points[row], expected.home_surface_points)
        torch.testing.assert_close(actual.palm_normal[row], expected.palm_normal)
        torch.testing.assert_close(actual.space_screws[row], expected.space_screws)
