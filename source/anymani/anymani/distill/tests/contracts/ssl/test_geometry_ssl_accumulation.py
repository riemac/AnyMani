r"""不等大小 minibatch 的 Geometry SSL denominator-aware accumulation 合同。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.objectives.representations.field_reconstruction import (
    GeometryFieldObjectiveCfg,
    GeometryFieldObjectiveTerms,
)
from anymani.distill.ssl.runtime.objective import accumulated_objective

pytestmark = pytest.mark.contract


def _terms(
    field_numerator: float,
    field_denominator: float,
    zero_numerator: float,
    zero_denominator: float,
    first_numerator: float,
    first_denominator: float,
) -> GeometryFieldObjectiveTerms:
    """构造六项中只激活 density 与 paired 的可微标量包。"""

    density = torch.tensor(field_numerator, dtype=torch.float64, requires_grad=True)
    zero = torch.tensor(zero_numerator, dtype=torch.float64, requires_grad=True)
    first = torch.tensor(first_numerator, dtype=torch.float64, requires_grad=True)
    field_zero = density * 0.0
    pair_loss = zero / zero_denominator + first / first_denominator
    return GeometryFieldObjectiveTerms(
        total=density / field_denominator + pair_loss,
        density=density / field_denominator,
        kappa=field_zero,
        derived_field=field_zero,
        sobolev=field_zero,
        chain=field_zero,
        paired=pair_loss,
        derived_field_sensitivity=torch.empty(0),
        auto_field_sensitivity=torch.empty(0),
        numerators=(density, field_zero, field_zero, field_zero, field_zero, pair_loss),
        denominators=tuple(torch.tensor(value, dtype=torch.float64) for value in (field_denominator, 1, 1, 1, 1, 1)),
        paired_additive_numerators=(zero, first),
        paired_additive_denominators=(
            torch.tensor(zero_denominator, dtype=torch.float64),
            torch.tensor(first_denominator, dtype=torch.float64),
        ),
    )


def test_accumulation_uses_global_field_and_separate_paired_denominators() -> None:
    r"""尾 batch 不得与满 batch 等权；paired 必须保持两支 MSE 的和。"""

    first = _terms(8.0, 4.0, 4.0, 4.0, 18.0, 6.0)
    second = _terms(10.0, 2.0, 8.0, 2.0, 8.0, 2.0)
    weights = GeometryFieldObjectiveCfg(
        density=1.0,
        kappa=0.0,
        derived_field=0.0,
        sobolev=0.0,
        chain=0.0,
        paired=1.0,
    )
    field_totals = tuple(torch.tensor(value, dtype=torch.float64) for value in (6.0, 2.0, 2.0, 2.0, 2.0))
    paired_totals = (torch.tensor(6.0, dtype=torch.float64), torch.tensor(8.0, dtype=torch.float64))

    accumulated = accumulated_objective(first, field_totals, paired_totals, weights) + accumulated_objective(
        second, field_totals, paired_totals, weights
    )

    # density=(8+10)/(4+2)=3；zero=(4+8)/(4+2)=2；first=(18+8)/(6+2)=3.25。
    assert float(accumulated.detach()) == pytest.approx(8.25)
    accumulated.backward()
    assert first.numerators[0].grad == pytest.approx(torch.tensor(1.0 / 6.0, dtype=torch.float64))
    assert second.paired_additive_numerators[0].grad == pytest.approx(torch.tensor(1.0 / 6.0, dtype=torch.float64))
