r"""CAGrad全部任务Gram求解的纯Torch合同，验证平均锚点与可复核的最差局部投影。"""

from __future__ import annotations

import torch
from anymani.distill.rl.algorithms.cagrad import cagrad_coefficients, combine_task_gradients


def test_zero_c_recovers_average_and_zero_gram_is_finite() -> None:
    r"""c=0必须逐值恢复均值；零梯度不能造成除零或伪更新。"""
    gradient = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float64)
    coefficients, _ = cagrad_coefficients(gradient @ gradient.T, c=0)
    torch.testing.assert_close(coefficients, torch.full((2,), 0.5, dtype=torch.float64))
    coefficients, diagnostics = cagrad_coefficients(torch.zeros(4, 4, dtype=torch.float64))
    assert torch.isfinite(coefficients).all()
    assert diagnostics["mean_gradient_norm_sq"] == 0


def test_two_task_dual_matches_dense_simplex_reference() -> None:
    r"""两任务simplex是一条线，可用密集网格独立核对最优目标和primal/dual gap。"""
    gradient = torch.tensor([[3.0, 0.0], [-1.0, 2.0]], dtype=torch.float64)
    gram = gradient @ gradient.T
    c = 0.4
    coefficients, diagnostics = cagrad_coefficients(gram, c=c, iterations=128, tolerance=1e-7)
    mean = gradient.mean(0)
    direction = coefficients @ gradient * (1 + c * c)
    assert torch.linalg.vector_norm(direction - mean) <= c * torch.linalg.vector_norm(mean) + 1e-8
    t = torch.linspace(0, 1, 100001, dtype=torch.float64)
    mixed = t[:, None] * gradient[0] + (1 - t[:, None]) * gradient[1]
    optimum = (mixed @ mean + c * mean.norm() * mixed.norm(dim=1)).min()
    worst = (gradient @ direction).min()
    torch.testing.assert_close(worst, optimum, rtol=1e-5, atol=1e-6)
    assert diagnostics["relative_gap"] < 1e-5


def test_common_scale_and_parameter_partition_do_not_change_direction() -> None:
    r"""共同缩放与参数分块不能改变任务组合语义；所有任务行都保留。"""
    torch.manual_seed(42)
    gradient = torch.randn(8, 12, dtype=torch.float64)
    first, _ = combine_task_gradients({"a": gradient[:, :4], "b": gradient[:, 4:]}, iterations=128)
    second, _ = combine_task_gradients({"all": gradient}, iterations=128)
    torch.testing.assert_close(torch.cat((first["a"], first["b"])), second["all"], rtol=1e-6, atol=1e-7)
    scaled, _ = combine_task_gradients({"all": gradient * 100}, iterations=128)
    torch.testing.assert_close(scaled["all"], second["all"] * 100, rtol=1e-6, atol=1e-6)


def test_opposed_tasks_with_zero_mean_keep_zero_direction() -> None:
    r"""平均梯度为零时局部约束球退化，不能凭空声称同时改善两个相反任务。"""
    gradient = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    coefficients, _ = cagrad_coefficients(gradient @ gradient.T)
    torch.testing.assert_close(coefficients @ gradient, torch.zeros(2, dtype=torch.float64))

