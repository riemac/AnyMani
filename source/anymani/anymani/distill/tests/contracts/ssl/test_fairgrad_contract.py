"""两任务精确 FairGrad 的解析方向、异常分支与参数分区合同。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.methods.multi_anchor_gaussian_implicit_field import MultiAnchorGaussianMethod
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.config import MultiAnchorGaussianMethodCfg
from anymani.distill.methods.multi_anchor_gaussian_implicit_field.training import combine_fairgrad

pytestmark = pytest.mark.contract


def _flatten(result: object) -> torch.Tensor:
    """把 FairGrad 返回的逐参数梯度拼为一维向量，便于核对解析几何。"""

    return torch.cat(tuple(gradient.reshape(-1) for gradient in result.combined if gradient is not None))


def test_fairgrad_is_invariant_to_positive_task_rescaling() -> None:
    r"""任意正缩放 $g_j\mapsto s_jg_j$ 不得改变 $\alpha=1$ 的组合方向。"""

    density = (torch.tensor([3.0, 0.0], dtype=torch.float64),)
    kappa = (torch.tensor([1.0, 2.0], dtype=torch.float64),)
    reference = combine_fairgrad(density, kappa)
    scaled = combine_fairgrad((17.0 * density[0],), (0.03 * kappa[0],))

    torch.testing.assert_close(_flatten(reference), _flatten(scaled), atol=1.0e-12, rtol=1.0e-12)
    assert reference.evidence.combined_norm == pytest.approx(2.0**0.5)


def test_fairgrad_handles_zero_and_near_opposite_shared_gradients() -> None:
    """单任务为零时使用另一任务单位方向；近乎完全反向时阻塞 shared 更新。"""

    one_zero = combine_fairgrad(
        (torch.zeros(2, dtype=torch.float64),),
        (torch.tensor([0.0, -4.0], dtype=torch.float64),),
    )
    torch.testing.assert_close(_flatten(one_zero), torch.tensor([0.0, -1.0], dtype=torch.float64))
    assert not one_zero.evidence.shared_conflict_blocked

    blocked = combine_fairgrad(
        (torch.tensor([1.0, 0.0], dtype=torch.float64),),
        (torch.tensor([-1.0, 0.0], dtype=torch.float64),),
    )
    assert blocked.evidence.shared_conflict_blocked
    assert blocked.combined == (None,)


def test_fairgrad_rejects_non_finite_input() -> None:
    """非有限 task gradient 必须在 optimizer step 前 fail closed。"""

    with pytest.raises(FloatingPointError, match="non-finite"):
        combine_fairgrad(
            (torch.tensor([1.0, float("nan")]),),
            (torch.ones(2),),
        )


def test_method_parameter_groups_are_disjoint_and_cover_model() -> None:
    """Method 必须显式给出 shared、density-private、kappa-private 三个完整参数组。"""

    method = MultiAnchorGaussianMethod(MultiAnchorGaussianMethodCfg())
    method.initialize_model(device=torch.device("cpu"), dtype=torch.float32)
    groups = method.optimizer_parameter_groups()
    grouped = [parameter for group in groups for parameter in group.parameters]

    assert tuple(group.name for group in groups) == ("shared_encoder", "density_reader", "kappa_reader")
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in method.parameters()}
