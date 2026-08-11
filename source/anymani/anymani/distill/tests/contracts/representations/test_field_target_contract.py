"""distill Gaussian field 与 query/edge target 轴合同。"""

from __future__ import annotations

import math

import pytest
import torch
from anymani.distill.representations.fields.density import (
    field_sensitivity_from_distance,
    gaussian_density_from_distance,
)
from anymani.distill.representations.targets.field_samples import (
    FieldTargetBatch,
    QueryStratum,
    SensitivityTargetBatch,
)

pytestmark = pytest.mark.contract


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
            [[QueryStratum.WORKSPACE, QueryStratum.WORKSPACE, QueryStratum.OWNER_SHELL, QueryStratum.ADJACENT]]
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
