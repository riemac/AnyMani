"""representation source 纯张量 POE、当前轴线与物质点 Jacobian 合同。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.representations.sources.kinematics import (
    EmbodimentGeometrySpec,
    forward_owner_transforms,
    selected_point_jacobian,
    transform_owner_points,
)

pytestmark = pytest.mark.contract


def _planar_two_joint_spec(*, dtype: torch.dtype = torch.float64) -> EmbodimentGeometrySpec:
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
    return EmbodimentGeometrySpec(
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
    spec = EmbodimentGeometrySpec(
        space_screws=base.space_screws,
        q_home=q_home,
        owner_home_transforms=base.owner_home_transforms,
        owner_ancestor_mask=base.owner_ancestor_mask,
        joint_ancestor_mask=base.joint_ancestor_mask,
    )

    transforms = forward_owner_transforms(spec, q_home.unsqueeze(0))
    torch.testing.assert_close(transforms[0], base.owner_home_transforms)
