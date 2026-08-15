from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch
from anymani.distill.models.backbones.geometry_transformer import GraphBiasedTransformerCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderCfg,
    GeometryLatentHeadsCfg,
    GeometryPaddingCfg,
    ImplicitGeometryEncoder,
    SO2AnchorFrontendCfg,
    StaticGeometryEvidence,
    pad_static_geometry_evidence,
    stack_static_geometry_evidence,
)
from anymani.distill.objectives.representations.gauge_consistency import (
    deterministic_partial_joint_sign,
    joint_sign_paired_loss,
    rewrite_joint_sign_coordinates,
)

pytestmark = pytest.mark.contract


def _static_evidence(*, dtype: torch.dtype = torch.float64) -> StaticGeometryEvidence:
    r"""构造一个 PALM–JOINT–TIP 三实体结构及非对称 anchor 星座。"""

    anchors = torch.tensor(
        [
            [-0.045, -0.031, 0.002],
            [-0.018, 0.037, -0.001],
            [0.026, 0.029, 0.004],
            [0.049, -0.022, -0.003],
        ],
        dtype=dtype,
    )
    home_surface_points = torch.tensor(
        [
            [[-0.04, -0.03, 0.0], [-0.04, 0.03, 0.0], [0.04, -0.03, 0.0], [0.04, 0.03, 0.0]],
            [[0.03, -0.01, 0.01], [0.05, -0.01, 0.01], [0.03, 0.01, 0.01], [0.05, 0.01, 0.01]],
            [[0.065, -0.009, 0.012], [0.075, -0.009, 0.012], [0.065, 0.009, 0.012], [0.075, 0.009, 0.012]],
        ],
        dtype=dtype,
    )
    shortest_path = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.long)
    parent_direction = torch.tensor([[0, 1, 2], [0, 0, 1], [0, 0, 0]], dtype=torch.long)
    child_direction = parent_direction.transpose(0, 1).contiguous()
    return StaticGeometryEvidence(
        anchors=anchors,
        home_surface_points=home_surface_points,
        home_surface_mask=torch.ones(3, 4, dtype=torch.bool),
        palm_normal=torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        space_screws=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        q_home=torch.zeros(1, dtype=dtype),
        entity_role=torch.tensor([0, 1, 2], dtype=torch.long),
        entity_joint_index=torch.tensor([-1, 0, -1], dtype=torch.long),
        joint_entity_index=torch.tensor([1], dtype=torch.long),
        shortest_path=shortest_path,
        parent_direction=parent_direction,
        child_direction=child_direction,
    )


def _encoder() -> ImplicitGeometryEncoder:
    r"""使用小宽度但完整结构的 deterministic contract encoder。"""

    config = GeometryEncoderCfg(
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
        heads=GeometryLatentHeadsCfg(zero_order_width=32, first_order_width=16),
    )
    return ImplicitGeometryEncoder(config).to(dtype=torch.float64)


def _two_joint_static_evidence(*, dtype: torch.dtype = torch.float64) -> StaticGeometryEvidence:
    """构造 PALM–JOINT–JOINT–TIP 四实体结构，验证跨长度 padding。"""

    base = _static_evidence(dtype=dtype)
    extra_surface = torch.tensor(
        [[[0.09, -0.01, 0.014], [0.11, -0.01, 0.014], [0.09, 0.01, 0.014], [0.11, 0.01, 0.014]]],
        dtype=dtype,
    )
    return StaticGeometryEvidence(
        anchors=base.anchors,
        home_surface_points=torch.cat((base.home_surface_points[:2], extra_surface, base.home_surface_points[2:]), dim=0),
        home_surface_mask=torch.ones(4, 4, dtype=torch.bool),
        palm_normal=base.palm_normal,
        space_screws=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, -0.05, 0.0]],
            dtype=dtype,
        ),
        q_home=torch.tensor([0.0, 0.08], dtype=dtype),
        entity_role=torch.tensor([0, 1, 1, 2], dtype=torch.long),
        entity_joint_index=torch.tensor([-1, 0, 1, -1], dtype=torch.long),
        joint_entity_index=torch.tensor([1, 2], dtype=torch.long),
        shortest_path=torch.tensor(
            [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=torch.long
        ),
        parent_direction=torch.tensor(
            [[0, 4, 4, 4], [1, 0, 4, 4], [2, 1, 0, 4], [3, 2, 1, 0]], dtype=torch.long
        ),
        child_direction=torch.tensor(
            [[0, 1, 2, 3], [4, 0, 1, 2], [4, 4, 0, 1], [4, 4, 4, 0]], dtype=torch.long
        ),
    )


def _rotate_about_palm_normal(evidence: StaticGeometryEvidence, angle: float) -> StaticGeometryEvidence:
    r"""对所有三维物理证据执行同一个绕 palm normal 的被动坐标重写。"""

    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=evidence.anchors.dtype,
    )
    screws = evidence.space_screws.clone()
    screws[:, :3] = screws[:, :3] @ rotation.T
    screws[:, 3:] = screws[:, 3:] @ rotation.T
    return StaticGeometryEvidence(
        anchors=evidence.anchors @ rotation.T,
        home_surface_points=evidence.home_surface_points @ rotation.T,
        home_surface_mask=evidence.home_surface_mask,
        palm_normal=evidence.palm_normal @ rotation.T,
        space_screws=screws,
        q_home=evidence.q_home,
        entity_role=evidence.entity_role,
        entity_joint_index=evidence.entity_joint_index,
        joint_entity_index=evidence.joint_entity_index,
        shortest_path=evidence.shortest_path,
        parent_direction=evidence.parent_direction,
        child_direction=evidence.child_direction,
    )


def test_encoder_has_no_joint_limit_input_and_returns_typed_shapes() -> None:
    r"""Geometry encoder 只读取当前 q 与静态证据，limits 不属于其输入类型。"""

    evidence = _static_evidence()
    encoder = _encoder()
    q = torch.tensor([[0.2], [-0.4]], dtype=torch.float64, requires_grad=True)

    latents = encoder(q, evidence)
    assert latents.zero_order.shape == (2, 3, 32)
    assert latents.first_order.shape == (2, 1, 16)
    assert "joint_limits" not in {item.name for item in fields(StaticGeometryEvidence)}

    (latents.zero_order.square().mean() + latents.first_order.square().mean()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_anchor_permutation_and_common_so2_rotation_do_not_change_latents() -> None:
    r"""所有 anchors 等地位，且公共 palm-plane 旋转只改写坐标、不改写表示。"""

    torch.manual_seed(7)
    evidence = _static_evidence()
    encoder = _encoder().eval()
    q = torch.tensor([[0.37], [-0.18]], dtype=torch.float64)
    reference = encoder(q, evidence)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = StaticGeometryEvidence(
        anchors=evidence.anchors[permutation],
        home_surface_points=evidence.home_surface_points,
        home_surface_mask=evidence.home_surface_mask,
        palm_normal=evidence.palm_normal,
        space_screws=evidence.space_screws,
        q_home=evidence.q_home,
        entity_role=evidence.entity_role,
        entity_joint_index=evidence.entity_joint_index,
        joint_entity_index=evidence.joint_entity_index,
        shortest_path=evidence.shortest_path,
        parent_direction=evidence.parent_direction,
        child_direction=evidence.child_direction,
    )
    permuted_latents = encoder(q, permuted)
    rotated_latents = encoder(q, _rotate_about_palm_normal(evidence, 1.137))

    torch.testing.assert_close(permuted_latents.zero_order, reference.zero_order, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(permuted_latents.first_order, reference.first_order, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(rotated_latents.zero_order, reference.zero_order, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(rotated_latents.first_order, reference.first_order, atol=1.0e-10, rtol=1.0e-10)


def test_encoder_uses_single_screw_feature_and_shared_residual_head() -> None:
    r"""canonical encoder 不得保留 even/odd 双投影与 coefficient/carrier 乘法结构。"""

    encoder = _encoder()
    module_names = {name for name, _module in encoder.named_modules()}

    assert "screw_projection" in module_names
    assert "first_order_head" in module_names
    assert "screw_even_projection" not in module_names
    assert "screw_odd_projection" not in module_names
    assert "first_order_coefficient" not in module_names
    assert "first_order_carrier" not in module_names


def test_partial_joint_sign_rewrite_and_paired_loss_encode_even_odd_contract() -> None:
    r"""只改写 JOINT 1 时，paired loss 要求全部 Z0 偶且只有对应 z1 分量为奇。"""

    torch.manual_seed(13)
    evidence = _two_joint_static_evidence()
    encoder = _encoder().eval()
    q = torch.tensor([[0.37, -0.18], [-0.21, 0.42]], dtype=torch.float64)
    reference = encoder(q, evidence)
    rewritten_q, rewritten_evidence, sign = rewrite_joint_sign_coordinates(q, evidence, joint_index=1)

    torch.testing.assert_close(rewritten_q[:, 0], q[:, 0])
    torch.testing.assert_close(rewritten_q[:, 1], -q[:, 1])
    torch.testing.assert_close(rewritten_evidence.space_screws[0], evidence.space_screws[0])
    torch.testing.assert_close(rewritten_evidence.space_screws[1], -evidence.space_screws[1])
    torch.testing.assert_close(rewritten_evidence.q_home[0], evidence.q_home[0])
    torch.testing.assert_close(rewritten_evidence.q_home[1], -evidence.q_home[1])
    torch.testing.assert_close(sign, torch.tensor([1.0, -1.0], dtype=torch.float64))

    ideal_paired = reference.__class__(
        zero_order=reference.zero_order,
        first_order=reference.first_order * sign.view(1, -1, 1),
    )
    ideal_loss = joint_sign_paired_loss(reference, ideal_paired, joint_sign=sign)
    torch.testing.assert_close(ideal_loss, torch.zeros_like(ideal_loss))
    shifted_paired = reference.__class__(
        zero_order=reference.zero_order + 1.0,
        first_order=reference.first_order * sign.view(1, -1, 1) + 2.0,
    )
    shifted_loss = joint_sign_paired_loss(reference, shifted_paired, joint_sign=sign)
    torch.testing.assert_close(shifted_loss, torch.tensor(5.0, dtype=torch.float64))  # $1^2+2^2$

    actual_paired = encoder(rewritten_q, rewritten_evidence)
    actual_loss = joint_sign_paired_loss(reference, actual_paired, joint_sign=sign)
    actual_loss.backward()
    assert torch.isfinite(actual_loss)
    assert any(parameter.grad is not None for parameter in encoder.parameters())

    valid = torch.tensor([[True, True, False], [True, False, False]])
    first_pattern = deterministic_partial_joint_sign(valid, step=0, dtype=torch.float64)
    repeated_pattern = deterministic_partial_joint_sign(valid, step=0, dtype=torch.float64)
    torch.testing.assert_close(first_pattern, repeated_pattern)
    torch.testing.assert_close(first_pattern[:, 2], torch.ones(2, dtype=torch.float64))


def test_so2_and_partial_joint_sign_rewrites_commute() -> None:
    r"""面内 frame rewrite 与单 JOINT 坐标反向的先后顺序不得改变 paired 物理输入。"""

    evidence = _two_joint_static_evidence()
    q = torch.tensor([[0.31, -0.27]], dtype=torch.float64)
    angle = 0.913

    sign_q, sign_evidence, sign = rewrite_joint_sign_coordinates(q, evidence, joint_index=1)
    sign_then_so2 = _rotate_about_palm_normal(sign_evidence, angle)

    so2_evidence = _rotate_about_palm_normal(evidence, angle)
    so2_then_sign_q, so2_then_sign, second_sign = rewrite_joint_sign_coordinates(
        q,
        so2_evidence,
        joint_index=1,
    )

    torch.testing.assert_close(sign_q, so2_then_sign_q, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(sign_then_so2.anchors, so2_then_sign.anchors, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        sign_then_so2.home_surface_points,
        so2_then_sign.home_surface_points,
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(sign_then_so2.palm_normal, so2_then_sign.palm_normal, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(sign_then_so2.space_screws, so2_then_sign.space_screws, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(sign_then_so2.q_home, so2_then_sign.q_home, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(sign, second_sign)


def test_same_structure_assets_share_one_forward_with_per_sample_static_evidence() -> None:
    """同 topology 的不同形态可堆成一批，结果等于逐资产独立前向。"""

    torch.manual_seed(19)
    first = _static_evidence()
    second = StaticGeometryEvidence(
        anchors=first.anchors + torch.tensor([0.003, -0.002, 0.001], dtype=first.anchors.dtype),
        home_surface_points=first.home_surface_points * 1.08,
        home_surface_mask=first.home_surface_mask,
        palm_normal=first.palm_normal,
        space_screws=first.space_screws,
        q_home=torch.tensor([0.11], dtype=first.q_home.dtype),
        entity_role=first.entity_role,
        entity_joint_index=first.entity_joint_index,
        joint_entity_index=first.joint_entity_index,
        shortest_path=first.shortest_path,
        parent_direction=first.parent_direction,
        child_direction=first.child_direction,
    )
    batched = stack_static_geometry_evidence((first, second))
    q = torch.tensor([[0.27], [-0.31]], dtype=torch.float64)
    encoder = _encoder().eval()

    together = encoder(q, batched)
    first_alone = encoder(q[:1], first)
    second_alone = encoder(q[1:], second)

    torch.testing.assert_close(together.zero_order[:1], first_alone.zero_order, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(together.zero_order[1:], second_alone.zero_order, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(together.first_order[:1], first_alone.first_order, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(together.first_order[1:], second_alone.first_order, atol=1.0e-10, rtol=1.0e-10)


def test_cross_structure_padding_matches_independent_variable_length_forwards() -> None:
    """20-JOINT/26-entity padding 的有效输出必须等于网络原生可变长前向。"""

    torch.manual_seed(29)
    one_joint = _static_evidence()
    two_joint = _two_joint_static_evidence()
    padding = GeometryPaddingCfg(max_joint_count=20, max_tip_count=5, max_graph_distance=4)
    evidence = pad_static_geometry_evidence((one_joint, two_joint), config=padding)
    q = torch.zeros(2, padding.max_joint_count, dtype=torch.float64)
    q[0, 0] = 0.21
    q[1, :2] = torch.tensor([-0.17, 0.31], dtype=torch.float64)
    encoder = _encoder().eval()

    padded = encoder(q, evidence)
    one_alone = encoder(q[:1, :1], one_joint)
    two_alone = encoder(q[1:2, :2], two_joint)

    torch.testing.assert_close(padded.zero_order[0, :3], one_alone.zero_order[0], atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(padded.zero_order[1, :4], two_alone.zero_order[0], atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(padded.first_order[0, :1], one_alone.first_order[0], atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(padded.first_order[1, :2], two_alone.first_order[0], atol=1.0e-10, rtol=1.0e-10)
    assert torch.count_nonzero(padded.zero_order[0, 3:]) == 0
    assert torch.count_nonzero(padded.first_order[0, 1:]) == 0
    assert torch.count_nonzero(padded.zero_order[1, 4:]) == 0
    assert torch.count_nonzero(padded.first_order[1, 2:]) == 0

    parameters = tuple(encoder.parameters())
    padded_valid_loss = (
        padded.zero_order[0, :3].square().sum()
        + padded.first_order[0, :1].square().sum()
        + padded.zero_order[1, :4].square().sum()
        + padded.first_order[1, :2].square().sum()
    )
    independent_valid_loss = (
        one_alone.zero_order.square().sum()
        + one_alone.first_order.square().sum()
        + two_alone.zero_order.square().sum()
        + two_alone.first_order.square().sum()
    )
    padded_gradients = torch.autograd.grad(padded_valid_loss, parameters, retain_graph=True)
    independent_gradients = torch.autograd.grad(independent_valid_loss, parameters)
    for padded_gradient, independent_gradient in zip(padded_gradients, independent_gradients):
        torch.testing.assert_close(padded_gradient, independent_gradient, atol=1.0e-9, rtol=1.0e-9)
