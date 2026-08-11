"""跨结构 `[B,E]` selectors 的 Sobolev 坐标导数合同。"""

from __future__ import annotations

import torch
from anymani.distill.objectives.representations.field_reconstruction import selected_density_coordinate_derivative


def test_batched_edge_selectors_read_each_samples_own_owner_query_and_joint() -> None:
    """padding 后每个样本可选择不同 owner/query/JOINT，且物理 q 导数不串样本。"""

    q = torch.tensor([[0.2, -0.4], [0.3, 0.5]], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=torch.float64,
    )  # `[G=2,N_Q=2,N_J=2]`
    density = torch.einsum("grj,bj->bgr", coefficient, q).unsqueeze(-1)  # `[B,G,N_Q,L=1]`
    owner_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    query_index = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
    joint_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    derivative = selected_density_coordinate_derivative(
        density,
        q,
        owner_index,
        query_index,
        joint_index,
        create_graph=True,
    )

    expected = torch.tensor([[[3.0], [6.0]], [[8.0], [1.0]]], dtype=torch.float64)
    torch.testing.assert_close(derivative, expected)
