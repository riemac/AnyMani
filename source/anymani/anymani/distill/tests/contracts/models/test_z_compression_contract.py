"""统一 Z 的流式 PCA、padding mask 与原 reader 重放合同。"""

from __future__ import annotations

import torch
from anymani.distill.diagnostics.evaluation.z_compression import (
    UnifiedPCAAccumulator,
    reconstruct_unified_entities,
    unified_pca_basis_digest,
)


def test_streaming_unified_pca_matches_direct_covariance_and_masks_padding() -> None:
    torch.manual_seed(37)
    entities = torch.randn(5, 4, 8, dtype=torch.float64)
    valid = torch.tensor(
        [[True, True, True, True], [True, True, False, False], [True, True, True, False], [True] * 4, [True] * 4]
    )
    accumulator = UnifiedPCAAccumulator(8)
    accumulator.update(entities[:2], valid[:2])
    accumulator.update(entities[2:], valid[2:])
    basis = accumulator.finalize()

    selected = entities[valid]
    direct_mean = selected.mean(dim=0)
    direct_covariance = torch.cov(selected.transpose(0, 1))
    direct_values = torch.linalg.eigvalsh(direct_covariance).flip(0)
    torch.testing.assert_close(basis.mean, direct_mean, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(basis.eigenvalues, direct_values, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(
        basis.components @ basis.components.transpose(0, 1),
        torch.eye(8, dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    reconstructed = reconstruct_unified_entities(entities, valid, basis, rank=4)
    assert reconstructed.shape == entities.shape
    assert torch.count_nonzero(reconstructed[~valid]) == 0
    assert torch.isfinite(reconstructed[valid]).all()
    assert len(unified_pca_basis_digest(basis)) == 64
