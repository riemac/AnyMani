r"""真实tanh策略KL的CPU回归：同分布为零、潜空间中心正确、ghost不进入统计。"""

from __future__ import annotations

import pytest
import torch
from anymani.distill.rl.algorithms.policy_statistics import mean_preserving_squashed_kl


@pytest.mark.parametrize("sigma", [0.60653066, 0.15, 0.035087805, 0.001])
def test_identical_distributions_have_zero_kl(sigma: float) -> None:
    r"""小探索尺度也必须满足KL恒等式，不能因分母加epsilon得到负数。"""
    mean = torch.tensor([[0.1, -0.4], [0.8, -0.9]], dtype=torch.float64)
    std = torch.full_like(mean, sigma)
    actual = mean_preserving_squashed_kl(mean, std, mean, std, torch.ones_like(mean, dtype=torch.bool))
    torch.testing.assert_close(actual, torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_latent_normal_kl_and_current_to_reference_direction(dtype: torch.dtype) -> None:
    r"""tanh是共同的可逆变换，因此KL等于atanh均值对应的潜Normal KL。"""
    current = torch.tensor([[0.98, -0.4, 0.2], [0.1, 0.2, -0.3]], dtype=dtype)
    reference = torch.tensor([[0.8, -0.2, 0.1], [-0.1, 0.3, 0.0]], dtype=dtype)
    sigma = torch.tensor([[0.1, 0.2, 0.04], [0.2, 0.3, 0.1]], dtype=dtype)
    old_sigma = torch.full_like(sigma, 0.15)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(torch.atanh(current), sigma),
        torch.distributions.Normal(torch.atanh(reference), old_sigma),
    )
    expected = (expected * mask).sum(-1) / mask.sum(-1)
    actual = mean_preserving_squashed_kl(current, sigma, reference, old_sigma, mask)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    assert (actual >= 0).all()


def test_ghost_values_do_not_contaminate_kl() -> None:
    r"""无效槽位即使有零sigma或NaN，也不进入分子或有效关节分母。"""
    current = torch.tensor([[0.2, float("nan")]])
    reference = torch.tensor([[0.0, float("nan")]])
    sigma = torch.tensor([[0.1, 0.0]])
    mask = torch.tensor([[True, False]])
    actual = mean_preserving_squashed_kl(current, sigma, reference, sigma, mask)
    expected = torch.atanh(current[:, 0]).square() / (2 * sigma[:, 0].square())
    torch.testing.assert_close(actual, expected)


def test_invalid_active_sigma_is_rejected() -> None:
    r"""真实有效关节的退化分布应报错，而不是用epsilon掩盖。"""
    with pytest.raises(RuntimeError, match="sigma"):
        mean_preserving_squashed_kl(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
        )
