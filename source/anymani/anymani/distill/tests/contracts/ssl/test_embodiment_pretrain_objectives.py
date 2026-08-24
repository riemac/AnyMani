r"""三项独立 objective 的 $(asset,q)$ 等权与 active/zero 1:1 合同。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from anymani.distill.methods.contracts import MethodStep
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.config import (
    DensityObjectiveCfg,
    KappaObjectiveCfg,
    MultiAnchorGaussianObjectivesCfg,
)
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.objectives import (
    density_objective,
    derived_field_objective,
    kappa_objective,
    reduce_method_steps,
)


@dataclass
class _Context:
    """把三项 objective 需要的预测与真值暴露给无状态 callable。"""

    density_prediction: torch.Tensor
    density_target: torch.Tensor
    density_valid_mask: torch.Tensor
    kappa_prediction: torch.Tensor
    kappa_target: torch.Tensor
    edge_valid_mask: torch.Tensor
    field_sensitivity_target: torch.Tensor
    derived_field_sensitivity: torch.Tensor
    active_mask: torch.Tensor


def test_three_terms_reduce_by_asset_q_and_split_active_zero() -> None:
    r"""两个异构样本等权；active 全被 mask 时 kappa 退化为 zero 均值。"""

    dtype = torch.float64
    density_prediction = torch.tensor([[[[0.2], [0.4]], [[0.6], [0.8]]]], dtype=dtype)
    density_target = torch.zeros_like(density_prediction)
    density_valid = torch.tensor(
        [
            [[True, True], [True, False]],  # 样本 0：三个有效标量，MSE=(0.04+0.16+0.36)/3
            [[False, False], [False, True]],  # 样本 1：一个有效标量，MSE=0.64
        ]
    )
    # 样本 0：1 active 有效 + 1 zero 有效；样本 1：active 全无效，zero 有效。
    kappa_prediction = torch.tensor([[0.4, 0.0], [9.0, 0.2]], dtype=dtype)
    kappa_target = torch.zeros_like(kappa_prediction)
    edge_valid = torch.tensor([[True, True], [False, True]])
    active_mask = torch.tensor([[True, False], [True, False]])
    field_sensitivity = torch.zeros(2, 2, 1, dtype=dtype)
    context = _Context(
        density_prediction=density_prediction.expand(2, -1, -1, -1).clone(),
        density_target=density_target.expand(2, -1, -1, -1).clone(),
        density_valid_mask=density_valid,
        kappa_prediction=kappa_prediction,
        kappa_target=kappa_target,
        edge_valid_mask=edge_valid,
        field_sensitivity_target=field_sensitivity,
        derived_field_sensitivity=field_sensitivity.clone(),
        active_mask=active_mask,
    )
    density = density_objective(context)
    kappa = kappa_objective(context)
    derived = derived_field_objective(context)
    assert density.name == "density"
    # density: 样本内先按有效元素取 MSE，再让两个 $(asset,q)$ 等权；不是把四个标量全局平均。
    assert float(density.components[0].denominator.detach()) == 2.0
    torch.testing.assert_close(
        density.metrics["loss"],
        torch.tensor(0.5 * ((0.04 + 0.16 + 0.36) / 3.0 + 0.64), dtype=dtype),
        atol=1.0e-12,
        rtol=0.0,
    )
    # kappa 样本 0：0.5*(0.16+0)=0.08；样本 1：只有 zero=0.04；等权 0.06。
    torch.testing.assert_close(kappa.metrics["loss"], torch.tensor(0.06, dtype=dtype), atol=1.0e-12, rtol=0.0)
    update = reduce_method_steps(
        (
            MethodStep(
                objectives={
                    "density": density,
                    "kappa": kappa,
                    "derived_field": derived,
                },
                sample_count=2,
            ),
        ),
        MultiAnchorGaussianObjectivesCfg(
            density=DensityObjectiveCfg(weight=1.0),
            kappa=KappaObjectiveCfg(weight=1.0),
        ),
    )
    assert update.sample_count == 2
    assert set(update.terms) == {"density", "kappa", "derived_field"}
