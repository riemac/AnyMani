r"""Anchor-relational Material-point Jacobian disposable reader 合同。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.models.decoders.representations.material_point_jacobian import (
    AnchorRelationalJacobianDecoder,
    AnchorRelationalJacobianDecoderCfg,
    BilinearAnchorRelationalJacobianDecoder,
    BilinearAnchorRelationalJacobianDecoderCfg,
)

pytestmark = pytest.mark.contract


def test_decoder_preserves_variable_anchor_axis_and_permutation_equivariance() -> None:
    r"""共享 per-anchor reader 只能同步置换输出 $K$ 轴，不能依赖 anchor 存储下标。"""

    torch.manual_seed(7)
    decoder = AnchorRelationalJacobianDecoder(
        AnchorRelationalJacobianDecoderCfg(latent_width=16, relation_width=8, hidden_width=12)
    ).double()
    owner = torch.randn(2, 5, 16, dtype=torch.float64)  # `[B,E,D]` owner Z
    joint = torch.randn(2, 5, 16, dtype=torch.float64)  # `[B,E,D]` selected JOINT Z
    pair = torch.randn(2, 5, 7, 8, dtype=torch.float64)  # `[B,E,K,D_r]` static material-anchor query
    baseline = decoder(owner, joint, pair)
    permutation = torch.tensor((4, 0, 6, 2, 1, 5, 3), dtype=torch.long)
    permuted = decoder(owner, joint, pair[:, :, permutation])

    assert baseline.shape == (2, 5, 7, 4)
    torch.testing.assert_close(permuted, baseline[:, :, permutation], atol=2.0e-15, rtol=2.0e-15)


def test_decoder_backpropagates_to_owner_joint_and_pair_features() -> None:
    r"""四通道 objective 必须同时向 owner Z、JOINT Z 与 static relation frontend 传播梯度。"""

    decoder = AnchorRelationalJacobianDecoder(
        AnchorRelationalJacobianDecoderCfg(latent_width=12, relation_width=6, hidden_width=10)
    )
    owner = torch.randn(3, 4, 12, requires_grad=True)
    joint = torch.randn(3, 4, 12, requires_grad=True)
    pair = torch.randn(3, 4, 5, 6, requires_grad=True)
    decoder(owner, joint, pair).square().mean().backward()

    for value in (owner, joint, pair):
        assert value.grad is not None
        assert torch.count_nonzero(value.grad) > 0


def test_decoder_rejects_misaligned_edge_or_relation_width() -> None:
    r"""Reader 在 source/method padding 之前 fail closed，不允许 silent broadcast 错配 edge。"""

    decoder = AnchorRelationalJacobianDecoder(
        AnchorRelationalJacobianDecoderCfg(latent_width=8, relation_width=4, hidden_width=8)
    )
    owner = torch.zeros(2, 3, 8)
    joint = torch.zeros(2, 4, 8)
    pair = torch.zeros(2, 3, 5, 4)
    with pytest.raises(ValueError, match="owner_latent and joint_latent"):
        decoder(owner, joint, pair)
    with pytest.raises(ValueError, match="static_pair_feature"):
        decoder(owner, owner, torch.zeros(2, 3, 5, 5))


def test_bilinear_decoder_preserves_anchor_permutation_and_all_gradient_paths() -> None:
    r"""低秩 row/column reader 必须保持 K 等变并连接 owner、JOINT 与 static query。"""

    torch.manual_seed(19)
    decoder = BilinearAnchorRelationalJacobianDecoder(
        BilinearAnchorRelationalJacobianDecoderCfg(
            latent_width=12,
            relation_width=6,
            hidden_width=10,
            readout_rank=5,
        )
    ).double()
    owner = torch.randn(2, 4, 12, dtype=torch.float64, requires_grad=True)
    joint = torch.randn(2, 4, 12, dtype=torch.float64, requires_grad=True)
    pair = torch.randn(2, 4, 7, 6, dtype=torch.float64, requires_grad=True)
    baseline = decoder(owner, joint, pair)
    permutation = torch.tensor((6, 2, 0, 5, 1, 4, 3), dtype=torch.long)
    permuted = decoder(owner, joint, pair[:, :, permutation])

    assert baseline.shape == (2, 4, 7, 4)
    torch.testing.assert_close(permuted, baseline[:, :, permutation], atol=2.0e-15, rtol=2.0e-15)
    baseline.square().mean().backward()
    for value in (owner, joint, pair):
        assert value.grad is not None
        assert torch.count_nonzero(value.grad) > 0
