from __future__ import annotations

import math

import pytest
import torch
from anymani.distill.representations.fields.density import (
    field_sensitivity_from_distance,
    gaussian_density_from_distance,
)
from anymani.distill.representations.sources.kinematics import (
    KinematicTreeSpec,
    forward_owner_transforms,
    selected_point_jacobian,
    transform_owner_points,
)
from anymani.distill.representations.targets.field_samples import FieldTargetBatch, QueryStratum, SensitivityTargetBatch

pytestmark = pytest.mark.contract


def _planar_two_joint_spec(*, dtype: torch.dtype = torch.float64) -> KinematicTreeSpec:
    r"""构造两个平面转动关节和三个 PALM/JOINT owner 的解析测试机构。"""

    space_screws = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        ],
        dtype=dtype,
    )
    owner_home_transforms = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    owner_home_transforms[1, 0, 3] = 1.0
    owner_home_transforms[2, 0, 3] = 2.0
    owner_ancestor_mask = torch.tensor(
        [[False, False], [True, False], [True, True]],
        dtype=torch.bool,
    )
    joint_ancestor_mask = torch.tensor(
        [[False, False], [True, False]],
        dtype=torch.bool,
    )
    return KinematicTreeSpec(
        space_screws=space_screws,
        q_home=torch.zeros(2, dtype=dtype),
        owner_home_transforms=owner_home_transforms,
        owner_ancestor_mask=owner_ancestor_mask,
        joint_ancestor_mask=joint_ancestor_mask,
    )


def test_poe_fk_and_selected_point_jacobian_match_finite_difference() -> None:
    r"""验证 POE 放置与任意 owner material point Jacobian 的中心有限差分。"""

    spec = _planar_two_joint_spec()
    q = torch.tensor([[0.37, -0.52]], dtype=torch.float64)
    owner_index = torch.tensor([2, 2, 0], dtype=torch.long)
    joint_index = torch.tensor([0, 1, 1], dtype=torch.long)
    local_points = torch.zeros(3, 3, dtype=torch.float64)

    transforms = forward_owner_transforms(spec, q)
    points = transform_owner_points(transforms, owner_index, local_points)
    jacobian = selected_point_jacobian(spec, q, owner_index, joint_index, local_points)

    epsilon = 1.0e-6
    for edge in range(2):
        q_plus = q.clone()
        q_minus = q.clone()
        q_plus[:, joint_index[edge]] += epsilon
        q_minus[:, joint_index[edge]] -= epsilon
        point_plus = transform_owner_points(
            forward_owner_transforms(spec, q_plus), owner_index[edge : edge + 1], local_points[edge : edge + 1]
        )
        point_minus = transform_owner_points(
            forward_owner_transforms(spec, q_minus), owner_index[edge : edge + 1], local_points[edge : edge + 1]
        )
        finite_difference = (point_plus[:, 0] - point_minus[:, 0]) / (2.0 * epsilon)
        torch.testing.assert_close(jacobian[:, edge], finite_difference, atol=2.0e-8, rtol=2.0e-8)

    assert torch.count_nonzero(jacobian[:, 2]) == 0, "PALM owner 对任意 finger JOINT 必须是结构零"
    assert points.shape == (1, 3, 3)


def test_nonzero_home_configuration_is_subtracted_before_poe() -> None:
    r"""资产定义的非零基准构型必须满足在 q=q_home 时恢复 owner home transforms。"""

    base = _planar_two_joint_spec()
    q_home = torch.tensor([0.2, -0.4], dtype=torch.float64)
    spec = KinematicTreeSpec(
        space_screws=base.space_screws,
        q_home=q_home,
        owner_home_transforms=base.owner_home_transforms,
        owner_ancestor_mask=base.owner_ancestor_mask,
        joint_ancestor_mask=base.joint_ancestor_mask,
    )

    transforms = forward_owner_transforms(spec, q_home.unsqueeze(0))
    torch.testing.assert_close(transforms[0], base.owner_home_transforms)


def test_multiband_density_and_field_sensitivity_obey_chain_and_scale_laws() -> None:
    r"""验证多带宽 Gaussian 邻近场、链式灵敏度和共同尺度变换律。"""

    distance = torch.tensor([[[0.0, 0.012, 0.031]]], dtype=torch.float64)
    bandwidths = torch.tensor([0.004, 0.012, 0.032, 0.064], dtype=torch.float64)
    kappa = torch.tensor([[[[0.0, 0.0], [0.03, -0.02], [0.07, 0.01]]]], dtype=torch.float64)

    density = gaussian_density_from_distance(distance, bandwidths)
    sensitivity = field_sensitivity_from_distance(distance, density, bandwidths, kappa)

    expected = torch.exp(-distance.unsqueeze(-1).square() / (2.0 * bandwidths.square()))
    torch.testing.assert_close(density, expected)
    expected_sensitivity = (
        -distance.unsqueeze(-1).unsqueeze(-1)
        / bandwidths.square().view(1, 1, 1, -1, 1)
        * density.unsqueeze(-1)
        * kappa.unsqueeze(-2)
    )
    torch.testing.assert_close(sensitivity, expected_sensitivity)

    scale = 1.73
    scaled_density = gaussian_density_from_distance(scale * distance, scale * bandwidths)
    scaled_sensitivity = field_sensitivity_from_distance(
        scale * distance,
        scaled_density,
        scale * bandwidths,
        scale * kappa,
    )
    torch.testing.assert_close(scaled_density, density, atol=1.0e-14, rtol=1.0e-14)
    torch.testing.assert_close(scaled_sensitivity, sensitivity, atol=1.0e-14, rtol=1.0e-14)

    assert density.shape == (1, 1, 3, 4)
    assert sensitivity.shape == (1, 1, 3, 4, 2)
    assert math.isclose(float(density[0, 0, 0, 0]), 1.0)


def test_density_rejects_nonpositive_bandwidth() -> None:
    r"""物理带宽必须严格为正，零值不能通过 clamp 静默改变场定义。"""

    with pytest.raises(ValueError, match="strictly positive"):
        gaussian_density_from_distance(torch.zeros(1, 1, 1), torch.tensor([0.0]))


def test_field_target_batch_keeps_query_and_sampled_edge_axes_distinct() -> None:
    r"""零阶 target 保留完整 query 轴，一阶 target 只 materialize sampled edges。"""

    batch_size, owner_count, query_count, bandwidth_count = 2, 3, 4, 2
    query_points = torch.zeros(batch_size, owner_count, query_count, 3)
    distance = torch.full((batch_size, owner_count, query_count), 0.01)
    bandwidths = torch.tensor([0.004, 0.012])
    density = gaussian_density_from_distance(distance, bandwidths)
    targets = FieldTargetBatch(
        query_points=query_points,
        query_stratum=torch.tensor(
            [
                [QueryStratum.WORKSPACE, QueryStratum.WORKSPACE, QueryStratum.OWNER_SHELL, QueryStratum.ADJACENT]
            ]
        )
        .expand(batch_size, owner_count, query_count)
        .clone(),
        distance=distance,
        density=density,
        valid_mask=torch.ones(batch_size, owner_count, query_count, dtype=torch.bool),
        owner_role=torch.tensor([0, 1, 2], dtype=torch.long),
        bandwidths=bandwidths,
        provenance={"frame": "h", "length_unit": "m"},
    )
    sensitivity = SensitivityTargetBatch(
        owner_index=torch.tensor([1, 2], dtype=torch.long),
        query_index=torch.tensor([2, 3], dtype=torch.long),
        joint_index=torch.tensor([0, 1], dtype=torch.long),
        ancestor_mask=torch.tensor([True, False]),
        closest_point=torch.zeros(batch_size, 2, 3),
        closest_source=torch.tensor([[4, 9], [4, 9]], dtype=torch.long),
        uniqueness_margin=torch.full((batch_size, 2), 0.003),
        kappa=torch.tensor([[0.02, 0.0], [0.03, 0.0]]),
        field_sensitivity=torch.zeros(batch_size, 2, bandwidth_count),
        valid_mask=torch.ones(batch_size, 2, dtype=torch.bool),
    )

    assert targets.density.shape == (batch_size, owner_count, query_count, bandwidth_count)
    assert sensitivity.kappa.shape == (batch_size, 2)
    assert sensitivity.field_sensitivity.shape == (batch_size, 2, bandwidth_count)
