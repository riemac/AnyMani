from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch
from anymani.distill.models.input_adapters.geometry import (
    GeometryEncoderConfig,
    ImplicitGeometryEncoder,
    StaticGeometryEvidence,
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

    config = GeometryEncoderConfig(
        relation_width=24,
        home_width=24,
        screw_width=16,
        hidden_width=48,
        zero_order_width=32,
        first_order_width=16,
        transformer_layers=2,
        attention_heads=4,
        feedforward_width=96,
        dropout=0.0,
        length_scale_m=0.1,
        max_graph_distance=4,
    )
    return ImplicitGeometryEncoder(config).to(dtype=torch.float64)


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


def test_paired_joint_sign_rewrite_makes_zero_order_even_and_first_order_odd() -> None:
    r"""同步翻转 screw、q 与 q_home 时，零阶严格为偶、一阶严格为奇。"""

    torch.manual_seed(13)
    evidence = _static_evidence()
    encoder = _encoder().eval()
    q = torch.tensor([[0.37], [-0.18]], dtype=torch.float64)
    reference = encoder(q, evidence)

    rewritten = StaticGeometryEvidence(
        anchors=evidence.anchors,
        home_surface_points=evidence.home_surface_points,
        home_surface_mask=evidence.home_surface_mask,
        palm_normal=evidence.palm_normal,
        space_screws=-evidence.space_screws,
        q_home=-evidence.q_home,
        entity_role=evidence.entity_role,
        entity_joint_index=evidence.entity_joint_index,
        joint_entity_index=evidence.joint_entity_index,
        shortest_path=evidence.shortest_path,
        parent_direction=evidence.parent_direction,
        child_direction=evidence.child_direction,
    )
    paired = encoder(-q, rewritten)

    torch.testing.assert_close(paired.zero_order, reference.zero_order, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(paired.first_order, -reference.first_order, atol=1.0e-10, rtol=1.0e-10)
