r"""$(asset,q)$ 等权 accumulation：尾 minibatch 不得与满 minibatch 按标量个数争权重。"""

from __future__ import annotations

import torch
from anymani.distill.methods.contracts import AdditiveStatistic, MethodStep, ObjectiveTermResult
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.config import (
    DensityObjectiveCfg,
    MultiAnchorGaussianObjectivesCfg,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives import reduce_method_steps


def test_update_reduction_weights_asset_q_equally() -> None:
    r"""两个 step 分别含 4 个和 1 个 $(asset,q)$ 时，均值按样本数加权而不是按 minibatch 等权。"""

    dtype = torch.float64
    first = MethodStep(
        objectives={
            "density": ObjectiveTermResult(
                "density",
                (AdditiveStatistic("density", torch.tensor(8.0, dtype=dtype, requires_grad=True), torch.tensor(4.0, dtype=dtype)),),
                {},
            )
        },
        sample_count=4,
    )
    second = MethodStep(
        objectives={
            "density": ObjectiveTermResult(
                "density",
                (AdditiveStatistic("density", torch.tensor(1.0, dtype=dtype, requires_grad=True), torch.tensor(1.0, dtype=dtype)),),
                {},
            )
        },
        sample_count=1,
    )
    update = reduce_method_steps(
        (first, second),
        MultiAnchorGaussianObjectivesCfg(density=DensityObjectiveCfg(weight=1.0), kappa=None, derived_field=None, sobolev=None, chain=None),
    )
    # $(8+1)/(4+1)=1.8$，不是两个 minibatch 均值 $(2+1)/2=1.5$。
    torch.testing.assert_close(update.loss, torch.tensor(1.8, dtype=dtype), atol=0.0, rtol=0.0)
    update.loss.backward()
    assert first.objectives["density"].components[0].numerator.grad is not None
