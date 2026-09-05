r"""异构Actor完整梯度按资产/replica-half分解的纯Torch合同。"""

from __future__ import annotations

import torch
from anymani.distill.rl.algorithms.gradient_audit import (
    compute_actor_gradient_scope_audit,
    per_asset_replica_half_gradients,
)


def test_replica_half_gradients_preserve_asset_means_and_self_consistency() -> None:
    r"""每资产两个replica halves独立求mean gradient，完整资产方向按样本数加权恢复。"""

    parameter = torch.nn.Parameter(torch.tensor((0.5, -0.25)))
    features = torch.tensor(
        (
            (1.0, 0.0),
            (3.0, 0.0),
            (5.0, 0.0),
            (7.0, 0.0),
            (0.0, 2.0),
            (0.0, 4.0),
            (0.0, 6.0),
            (0.0, 8.0),
        )
    )
    objective = features @ parameter  # 线性sample objective使期望gradient可逐值手算
    labels = torch.tensor((0, 0, 0, 0, 1, 1, 1, 1))
    replica_halves = torch.tensor((0, 0, 1, 1, 0, 0, 1, 1))

    gradients, counts = per_asset_replica_half_gradients(
        objective,
        labels,
        replica_halves,
        (parameter,),
        asset_count=2,
    )

    torch.testing.assert_close(counts, torch.full((2, 2), 2, dtype=torch.long))
    torch.testing.assert_close(gradients[0], torch.tensor(((2.0, 0.0), (6.0, 0.0))))
    torch.testing.assert_close(gradients[1], torch.tensor(((0.0, 3.0), (0.0, 7.0))))
    audit = compute_actor_gradient_scope_audit(gradients, counts)
    torch.testing.assert_close(audit["asset_gradients"], torch.tensor(((4.0, 0.0), (0.0, 5.0))))
    torch.testing.assert_close(audit["half_self_cosine"], torch.ones(2))
    torch.testing.assert_close(audit["asset_cosine"], torch.eye(2))


def test_gradient_scope_audit_exposes_unreliable_replica_halves() -> None:
    r"""同资产两个half方向相反时self-cosine必须为负，不能被完整batch平均掩盖。"""

    half_gradients = torch.tensor(
        (
            ((1.0, 0.0), (-1.0, 0.0)),
            ((0.0, 2.0), (0.0, 4.0)),
        )
    )
    counts = torch.full((2, 2), 3, dtype=torch.long)
    audit = compute_actor_gradient_scope_audit(half_gradients, counts)

    torch.testing.assert_close(audit["half_self_cosine"], torch.tensor((-1.0, 1.0)))
    assert float(audit["reliable_asset_fraction"].item()) == 0.5
    assert float(audit["aggregate_to_individual_norm_ratio"].item()) == 1.0
