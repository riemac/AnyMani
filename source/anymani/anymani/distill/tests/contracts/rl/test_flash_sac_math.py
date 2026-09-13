# Adapted source (公式参照): flash_rl/agents/flashSAC/{update,layer}.py,
# flash_rl/agents/utils/reward_normalization.py.
# Upstream copyright (c) 2026 Holiday Robotics; MIT，见 rl/flash_sac/LICENSE.upstream。
# SPDX-License-Identifier: MIT
r"""Flash SAC 的 CPU 数学合同：概率换元、有效关节熵、分布式 TD 与奖励统计。

动作与 Gaussian 潜变量均采用无量纲归一坐标；概率使用自然对数，熵单位为 nat。
测试直接载入纯数学文件，避免执行 anymani 顶层的环境注册和 Isaac 导入。
预期值来自 PyTorch 概率分布、C51 三角基函数和独立手算，不依赖上游安装。
"""

from __future__ import annotations

import importlib.util
import io
import math
from pathlib import Path

import pytest
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

# 仅执行被测数学模块；不触发父包的任务注册，也不修改全局 sys.modules 中的包。
_PATH = Path(__file__).resolve().parents[3] / "rl" / "flash_sac" / "math.py"  # distill 内的唯一被测文件
_SPEC = importlib.util.spec_from_file_location("_flash_sac_math_contract", _PATH)  # 独立模块名
assert _SPEC is not None and _SPEC.loader is not None, f"无法加载纯数学文件：{_PATH}"
_MATH = importlib.util.module_from_spec(_SPEC)  # 载入不依赖 Isaac 的 torch 数学定义
_SPEC.loader.exec_module(_MATH)  # 只执行纯数学函数与统计模块定义
masked_tanh_normal = _MATH.masked_tanh_normal  # [B,J] 掩码概率
masked_target_entropy = _MATH.masked_target_entropy  # [B] Gaussian 等效熵
select_min_q_log_probs = _MATH.select_min_q_log_probs  # [Q,B,K] → [B,K]
project_categorical = _MATH.project_categorical  # C51 投影
DiscountedRewardNormalizer = _MATH.DiscountedRewardNormalizer  # 逐环境步统计模块


@pytest.mark.parametrize("scalar_kind", ["full", "python", "tensor"])
def test_tanh_normal_matches_distribution_and_analytic_density(scalar_kind: str) -> None:
    r"""用 Normal、TanhTransform 和手写换元密度同时核对标准潜 Gaussian。"""
    mean = torch.tensor([[0.2, -0.7, 0.4], [1.2, 0.0, -0.3]])  # [2,3]，允许潜均值超出动作区间
    valid = torch.tensor([[True, True, False], [True, True, True]])  # 活跃数分别为 2、3
    noise = torch.tensor([[0.3, -0.4, 0.8], [-0.5, 0.9, 0.1]])  # 固定标准正态样本，无采样误差
    log_std = {"full": torch.full_like(mean, -0.8), "python": -0.8, "tensor": torch.tensor(-0.8)}[
        scalar_kind
    ]  # scalar 与 [B,J] 三种输入必须定义同一 Gaussian

    # 以未饱和动作核对 inverse-tanh 路径，避免数值逆变换的边界歧义。
    actions, joint, average = masked_tanh_normal(mean, log_std, valid, noise=noise)  # FP32 输出
    normal = Normal(mean, torch.full_like(mean, math.exp(-0.8)))  # 无量纲潜尺度 sigma
    latent = mean + normal.scale * noise  # z = mu + sigma * epsilon
    transform = TanhTransform(cache_size=0)  # 由 torch 实现独立的换元 Jacobian
    expected = normal.log_prob(latent) - transform.log_abs_det_jacobian(latent, latent.tanh())  # [B,J]
    transformed = TransformedDistribution(normal, [transform]).log_prob(latent.tanh())  # 动作密度
    analytic = -0.5 * noise.square() + 0.8 - 0.5 * math.log(2 * math.pi)  # Gaussian log p(z)
    analytic -= torch.log1p(-latent.tanh().square())  # 减去 log(1-a^2)
    torch.testing.assert_close(expected, transformed, rtol=2e-6, atol=2e-6)  # 两条分布计算路径
    torch.testing.assert_close(expected, analytic, rtol=2e-6, atol=2e-6)  # 显式换元公式
    torch.testing.assert_close(actions, latent.tanh().masked_fill(~valid, 0))  # ghost 动作精确零
    torch.testing.assert_close(joint, expected.masked_fill(~valid, 0).sum(-1), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(average * valid.sum(-1), joint)  # n_active * mean = sum


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf"), 1e30, -1e30])
def test_ghost_poison_has_zero_influence_and_zero_gradient(poison: float) -> None:
    r"""污染 ghost 的 mu、log sigma、noise 后，前向和所有参数梯度都保持不变。"""
    valid = torch.tensor([[True, False, True], [False, True, False]])  # [2,3]，含多个 ghost
    seeds = (torch.full((2, 3), 0.4), torch.full((2, 3), -1.1), torch.full((2, 3), -0.3))  # 潜参数
    clean = tuple(value.clone().requires_grad_() for value in seeds)  # 干净参照叶节点
    dirty = tuple(value.masked_fill(~valid, poison).requires_grad_() for value in seeds)  # 污染仅在 ghost

    # 同时覆盖动作梯度、联合 logp 和 per-active logp，避免只验证被约掉的一条路径。
    reference = masked_tanh_normal(clean[0], clean[1], valid, noise=clean[2])  # 三个可微输出
    actual = masked_tanh_normal(dirty[0], dirty[1], valid, noise=dirty[2])  # 非有限值须先屏蔽
    for expected, observed in zip(reference, actual, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)  # ghost 不改变任何有效输出
    clean_loss = torch.stack([output.sum() for output in reference]).sum()  # 动作、联合及平均 logp 的总导数
    dirty_loss = torch.stack([output.sum() for output in actual]).sum()  # 污染后的同一标量目标
    clean_grads = torch.autograd.grad(clean_loss, clean)  # dL/d(mu,logσ,ε)
    dirty_grads = torch.autograd.grad(dirty_loss, dirty)  # 污染后的同一导数
    for expected, observed in zip(clean_grads, dirty_grads, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)  # 有效维导数同样不受污染
        assert torch.isfinite(observed).all(), f"ghost={poison} 使梯度出现非有限值：{observed}"
        assert torch.count_nonzero(observed[~valid]) == 0, f"ghost 参数梯度必须精确零：{observed}"
    assert torch.count_nonzero(actual[0][~valid]) == 0, "ghost 动作必须逐元素精确为 0"


@pytest.mark.parametrize("deterministic", [False, True])
def test_explicit_noise_and_deterministic_do_not_consume_rng(deterministic: bool) -> None:
    r"""固定 noise 或确定性中心动作均不应推进 RNG；确定性中心是 tanh(mu)。"""
    mean = torch.tensor([[1.3, -0.6, float("nan")]])  # 潜均值不是归一动作均值
    valid = torch.tensor([[True, True, False]])  # 屏蔽非有限 ghost
    kwargs = {"deterministic": True} if deterministic else {"noise": torch.tensor([[0.2, -0.3, 0.0]])}
    before = torch.random.get_rng_state().clone()  # 比较完整 CPU RNG 状态而不是抽样近似
    actions, joint, _ = masked_tanh_normal(mean, -0.4, valid, **kwargs)  # 两种无需新随机数的模式
    assert torch.equal(before, torch.random.get_rng_state()), "显式 noise / deterministic 消耗了 RNG"
    if deterministic:
        torch.testing.assert_close(actions, mean.masked_fill(~valid, 0).tanh())  # 标准 latent Gaussian
        # 中心处 Gaussian 二次项为零，剩余 -logσ 与 tanh 的换元项。
        latent = mean[valid]  # [2]，仅真实关节
        log_det = TanhTransform().log_abs_det_jacobian(latent, latent.tanh())  # nat / joint
        expected = (0.4 - 0.5 * math.log(2 * math.pi) - log_det).sum().reshape(1)  # nat / hand
        torch.testing.assert_close(joint, expected, rtol=2e-6, atol=2e-6)  # 中心动作仍返回策略密度


def test_random_sampling_is_reproducible_with_a_seed() -> None:
    r"""无显式噪声时确实采样标准 Gaussian，并可由 CPU seed 精确重放。"""
    mean = torch.zeros(2, 3)  # [B,J]，单位潜标准差
    valid = torch.ones_like(mean, dtype=torch.bool)  # 全部真实关节
    with torch.random.fork_rng(devices=[]):  # 测试结束恢复外部 RNG，不触碰 CUDA
        torch.manual_seed(719)  # 固定抽样锚点
        expected_noise = torch.randn_like(mean)  # 独立生成标准正态参照
        torch.manual_seed(719)  # 让被测路径取得相同样本
        actual = masked_tanh_normal(mean, 0.0, valid)  # 随机动作
    torch.testing.assert_close(actual[0], expected_noise.tanh(), rtol=0, atol=0)  # 逐位可复现


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_extreme_latent_and_small_scale_remain_finite(dtype: torch.dtype) -> None:
    r"""z=±1000 时动作饱和，但 log Jacobian、logp 与梯度仍应有限。"""
    mean = torch.tensor([[1000.0, -1000.0, 0.0]], dtype=dtype, requires_grad=True)  # 极端潜中心
    log_std = torch.tensor([[-20.0, 2.0, -50.0]], dtype=dtype, requires_grad=True)  # 小 sigma 不应平方下溢
    noise = torch.tensor([[0.5, -0.5, 0.2]], dtype=dtype)  # 有限标准化残差
    valid = torch.ones_like(mean, dtype=torch.bool)  # 三个均为有效关节
    actions, joint, average = masked_tanh_normal(mean, log_std, valid, noise=noise)  # 内部升为 FP32
    assert all(x.dtype == torch.float32 for x in (actions, joint, average)), "概率计算必须保持 FP32"
    assert all(torch.isfinite(x).all() for x in (actions, joint, average)), "饱和 tanh 不应产生无穷 logp"

    # 双精度仅作证据参照；直接用 log(1-tanh(z)^2) 会在此用例给出 -inf。
    latent = mean.detach().double() + log_std.detach().double().exp() * noise.double()  # 高精度 z
    jacobian = TanhTransform().log_abs_det_jacobian(latent, latent.tanh())  # 稳定变换公式
    expected = -0.5 * noise.double().square() - log_std.detach().double() - 0.5 * math.log(2 * math.pi)
    expected = (expected - jacobian).sum(-1).float()  # [B]，nat / hand
    torch.testing.assert_close(joint, expected, rtol=1e-6, atol=1e-4)  # FP32 千量级误差界
    gradients = torch.autograd.grad(actions.sum() + joint.sum() + average.sum(), (mean, log_std))
    assert all(torch.isfinite(x).all() for x in gradients), f"极端潜变量梯度不有限：{gradients}"


@pytest.mark.parametrize("active_counts", [[1, 4, 16], [3, 7, 11]])
def test_entropy_and_logp_use_actual_degrees_of_freedom(active_counts: list[int]) -> None:
    r"""canonical 16 槽位只负责存储；目标熵与概率分母只统计真实自由度。"""
    counts = torch.tensor(active_counts)  # [B]，每只手的物理关节数
    valid = torch.arange(16)[None, :] < counts[:, None]  # [B,16]，不同 DoF 共用存储 ABI
    mean = torch.full(valid.shape, 0.2)  # 每个有效关节具有相同分布与噪声
    noise = torch.full_like(mean, 0.3)  # 所以 per-active logp 应与手型无关
    _, joint, average = masked_tanh_normal(mean, math.log(0.15), valid, noise=noise)  # sigma=0.15
    torch.testing.assert_close(joint, counts * average)  # 联合密度的可加性
    torch.testing.assert_close(average, average[0].expand_as(average))  # 均值不含 ghost 分母
    per_joint = 0.5 * math.log(2 * math.pi * math.e * 0.15**2)  # Gaussian 等效熵，nat / joint
    expected = torch.full((len(active_counts),), per_joint)  # 与 tanh 后真实熵明确区分
    torch.testing.assert_close(masked_target_entropy(valid), expected)  # 默认 reduction='mean'
    torch.testing.assert_close(masked_target_entropy(valid, reduction="sum"), counts * expected)  # nat / hand


@pytest.mark.parametrize("bad_sigma", [0.0, -0.1, float("nan"), float("inf")])
def test_target_entropy_rejects_invalid_scale(bad_sigma: float) -> None:
    r"""Gaussian 等效熵要求有限正潜尺度，不能靠 epsilon 把退化分布变合法。"""
    with pytest.raises(ValueError, match="target_sigma"):
        masked_target_entropy(torch.ones(1, 2, dtype=torch.bool), bad_sigma)  # sigma > 0


@pytest.mark.parametrize("field", ["mean", "log_std", "noise"])
def test_tanh_normal_rejects_nonfinite_active_inputs(field: str) -> None:
    r"""非有限值只允许出现在 ghost，真实关节必须给出可定义的潜分布。"""
    inputs = {"mean": torch.zeros(1, 2), "log_std": torch.zeros(1, 2), "noise": torch.zeros(1, 2)}
    inputs[field][0, 0] = float("nan")  # 污染有效关节 0
    with pytest.raises(ValueError, match=field):
        masked_tanh_normal(**inputs, joint_valid=torch.ones(1, 2, dtype=torch.bool))  # 不隐藏真实错误


def test_mask_shape_and_empty_hand_are_rejected() -> None:
    r"""拒绝广播错位、非布尔掩码与全 ghost 行；空物理手不能定义平均熵。"""
    mean = torch.zeros(2, 3)  # [B,J]
    valid = torch.ones_like(mean, dtype=torch.bool)  # 基准合法掩码
    for wrong in (torch.ones(2, 2, dtype=torch.bool), valid.float(), torch.tensor([[True] * 3, [False] * 3])):
        with pytest.raises(ValueError, match="joint_valid"):
            masked_tanh_normal(mean, 0.0, wrong)  # 尺寸、类型或有效数不满足合同
    with pytest.raises(ValueError, match="log_std"):
        masked_tanh_normal(mean, torch.zeros(3), valid)  # [J] 广播可能掩盖上游形状错误
    with pytest.raises(ValueError, match="noise"):
        masked_tanh_normal(mean, 0.0, valid, noise=torch.zeros(2, 1))  # 噪声须严格 [B,J]
    with pytest.raises(ValueError, match="joint_valid"):
        masked_target_entropy(torch.zeros(1, 3, dtype=torch.bool))  # n_active=0
    with pytest.raises(ValueError, match="reduction"):
        masked_target_entropy(valid, reduction="canonical")  # 只有 sum、mean 两种物理归约


def test_min_q_selects_one_whole_distribution_per_sample() -> None:
    r"""按 E[Q] 选 critic，再取整条分布；逐 atom 取 min 会破坏归一化。"""
    probabilities = torch.tensor([[[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]], [[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]]])
    support = torch.tensor([-1.0, 0.0, 1.0])  # 单位：与奖励一致的折扣价值
    q_values = (probabilities * support).sum(-1)  # [Q=2,B=2]，最小 critic 分别为 0、1
    logs = probabilities.log().requires_grad_()  # [2,2,3]，可检验选择后的梯度路由
    selected = select_min_q_log_probs(q_values, logs)  # 整分布选择
    expected = torch.stack((logs[0, 0], logs[1, 1]))  # 各样本只来自一个 critic
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)  # 精确 gather
    torch.testing.assert_close(selected.exp().sum(-1), torch.ones(2))  # 不丢失总质量
    assert not torch.allclose(selected, logs.min(0).values), "错误地采用了逐 atom 的最小 logp"
    gradient = torch.autograd.grad(selected.sum(), logs)[0]  # 未被选 critic 应无梯度
    torch.testing.assert_close(gradient.sum(-1), torch.tensor([[3.0, 0.0], [0.0, 3.0]]))


def test_min_q_ties_and_singleton_batch() -> None:
    r"""Q 个 critic 的并列最小值选首个；B=1 不能被无条件 squeeze 丢轴。"""
    values = torch.tensor([[2.0], [1.0], [1.0]])  # Q=3，后两者并列
    logs = torch.tensor([[[0.2, 0.8]], [[0.7, 0.3]], [[0.5, 0.5]]]).log()  # [3,1,2]
    torch.testing.assert_close(select_min_q_log_probs(values, logs), logs[1])  # [1,2]
    with pytest.raises(ValueError, match="shape"):
        select_min_q_log_probs(values, logs[:, :, :0])  # K=0 不能表示概率分布
    with pytest.raises(ValueError, match="shape"):
        select_min_q_log_probs(values, logs.expand(-1, 2, -1))  # 不允许广播 sample 轴


def test_c51_exact_bins_and_support_boundaries_preserve_mass() -> None:
    r"""恒等投影保持每个 exact-bin 质量，越界目标全压到最近端点。"""
    support = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])  # K=5，均匀间隔 1
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25]])  # 全支撑分布
    logs = probabilities.log().expand(3, -1)  # B=3，分别恒等、超上界、超下界
    actual = project_categorical(logs, torch.tensor([0.0, 100.0, -100.0]), torch.ones(3), torch.zeros(3), support)
    expected = torch.cat((probabilities, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]])))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)  # exact-bin 两个 scatter 不丢质量
    torch.testing.assert_close(actual.sum(-1), torch.ones(3))  # 各行总质量 1


def test_c51_terminal_collapses_all_atoms_to_reward() -> None:
    r"""discount=0 时，全部未来原子与熵项消失，reward 的邻近支撑点分摊单位质量。"""
    support = torch.arange(-2.0, 3.0)  # K=5
    logs = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25]]).log().expand(2, -1)  # 全支撑非零
    rewards = torch.tensor([0.25, -1.0])  # 第一行插值，第二行恰落在支撑点
    actual = project_categorical(logs, rewards, torch.zeros(2), torch.tensor([-100.0, 100.0]), support)
    expected = torch.tensor([[0.0, 0.0, 0.75, 0.25, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])  # 终止 TD
    torch.testing.assert_close(actual, expected)  # 与下一状态的分布形状和熵无关


def test_c51_discounted_entropy_sign_and_zero_probability_atoms() -> None:
    r"""alpha*logp 为负时增加 soft value；折扣同时作用于价值与熵成本。"""
    support = torch.arange(-2.0, 3.0)  # 支撑含 0 与 ±1
    logs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]).log().expand(2, -1)  # -inf 表示合法零质量
    discounts = torch.full((2,), 0.5**2)  # 已含 gamma**n，函数不得再乘一次 gamma
    costs = torch.tensor([-2.0, 2.0])  # alpha * per-active logp，单位同 Q
    actual = project_categorical(logs, torch.zeros(2), discounts, costs, support)  # 目标分别为 ±0.5
    expected = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0], [0.0, 0.5, 0.5, 0.0, 0.0]])
    torch.testing.assert_close(actual, expected)  # r+d*(z-cost) 的符号与括号位置
    torch.testing.assert_close((actual * support).sum(-1), torch.tensor([0.5, -0.5]))  # 一阶矩锚点


def test_c51_matches_independent_triangular_basis_and_detaches_target() -> None:
    r"""用双精度三角基函数代替 scatter 验证批量投影，TD 目标不建立反向图。"""
    generator = torch.Generator(device="cpu").manual_seed(318)  # 随机参照可复算
    support = torch.linspace(-3.0, 4.0, 29, requires_grad=True)  # delta=0.25
    logs = torch.randn(7, 29, generator=generator).log_softmax(-1).requires_grad_()  # [B,K]
    rewards = torch.linspace(-4.0, 4.0, 7, requires_grad=True)  # 包含边界裁剪
    discounts = torch.linspace(0.0, 1.0, 7, requires_grad=True)  # 从终止到不折扣
    costs = torch.linspace(-0.9, 0.9, 7, requires_grad=True)  # 熵成本正负均覆盖
    actual = project_categorical(logs, rewards, discounts, costs, support)  # FP32 no_grad 目标

    # phi_i(t)=max(0,1-|t-z_i|/delta)，直接对全部 source atoms 求和作独立参照。
    grid = support.detach().double()  # [K]，奖励单位
    atoms = rewards.detach().double()[:, None] + discounts.detach().double()[:, None] * (
        grid[None, :] - costs.detach().double()[:, None]
    )  # [B,K_source]，soft Bellman 原子
    atoms = atoms.clamp(grid[0], grid[-1])  # C51 先裁剪再投影
    weights = (1 - (atoms[:, :, None] - grid[None, None, :]).abs() / (grid[1] - grid[0])).clamp_min(0)
    expected = (logs.detach().double().exp()[:, :, None] * weights).sum(1).float()  # source → target
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-7)  # FP32 投影误差
    torch.testing.assert_close(actual.sum(-1), torch.ones(7), rtol=1e-6, atol=1e-7)  # 全批质量守恒
    assert not actual.requires_grad and actual.grad_fn is None, "Bellman 目标保留了反向计算图"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_c51_accumulates_low_precision_input_in_float32(dtype: torch.dtype) -> None:
    r"""低精度输入也以 FP32 scatter 累积，避免大量小质量在半精度下丢失。"""
    support = torch.linspace(-1.0, 1.0, 17, dtype=dtype)  # 间隔 1/8 可精确表示
    logs = torch.full((2, 17), -math.log(17), dtype=dtype)  # 量化后的 logp 总质量可能略偏离 1
    rewards = torch.tensor([0.03125, -0.0625], dtype=dtype)  # 非整 bin 目标
    discounts = torch.tensor([0.75, 0.5], dtype=dtype)  # 半精度精确数值
    costs = torch.zeros(2, dtype=dtype)  # 无熵项
    actual = project_categorical(logs, rewards, discounts, costs, support)  # 内部统一 FP32
    expected = project_categorical(logs.float(), rewards.float(), discounts.float(), costs.float(), support.float())
    assert actual.dtype == torch.float32, f"投影输出应为 FP32，实际为 {actual.dtype}"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)  # dtype 边界可直接复算
    torch.testing.assert_close(actual.sum(-1), logs.float().exp().sum(-1))  # 保留输入质量，无隐式 softmax


@pytest.mark.parametrize("support", [[0.0], [0.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, 1.0, 3.0], [0.0, float("nan")]])
def test_c51_rejects_invalid_support(support: list[float]) -> None:
    r"""投影的相邻-bin 公式仅对有限、至少两点、均匀严格递增支撑成立。"""
    grid = torch.tensor(support)  # 候选非法价值支撑
    with pytest.raises(ValueError, match="support"):
        project_categorical(
            torch.full((1, len(support)), -math.log(len(support))), torch.zeros(1), torch.ones(1), torch.zeros(1), grid
        )


def test_c51_rejects_sample_axis_broadcast_and_empty_distribution() -> None:
    r"""reward [B,1] 不能静默广播成 [B,B,K]；全 -inf 行没有概率质量。"""
    support = torch.tensor([-1.0, 0.0, 1.0])  # 合法支撑
    logs = torch.full((2, 3), -math.log(3))  # 合法 logp
    with pytest.raises(ValueError, match="rewards"):
        project_categorical(logs, torch.zeros(2, 1), torch.ones(2), torch.zeros(2), support)  # 错误 sample 轴
    with pytest.raises(ValueError, match="log_probs"):
        project_categorical(
            torch.full_like(logs, -float("inf")), torch.zeros(2), torch.ones(2), torch.zeros(2), support
        )


def test_reward_normalizer_matches_stepwise_hand_calculation() -> None:
    r"""两环境三步手算覆盖正常延续、terminated、truncated 与 RMS epsilon 递推。"""
    normalizer = DiscountedRewardNormalizer(2, gamma=0.5, max_return=5.0)  # 上游 count 初值不是 epsilon
    torch.testing.assert_close(normalizer.count, torch.tensor(0.0))  # 初始无真实样本
    torch.testing.assert_close(normalizer.mean, torch.tensor(0.0))  # 一阶矩初始化
    torch.testing.assert_close(normalizer.var, torch.tensor(1.0))  # 上游方差初始化为 1
    # G: [1,3] → [4,0.5] → [1,-2]；done 清掉旧 trace 后仍纳入本步 reward。
    steps = [
        ([1.0, 3.0], [False, False], [False, False], [1.0, 3.0]),  # t=1，均值 2
        ([4.0, -1.0], [True, False], [False, False], [4.0, 0.5]),  # t=2，环境 0 terminated
        ([-1.0, -2.0], [False, False], [False, True], [1.0, -2.0]),  # t=3，环境 1 truncated
    ]  # 所有数值单位均与原 reward 相同
    variance1 = (2.0 + 1e-4) / 2.0  # 初始 var*(0+epsilon) + 两环境的 M2
    variance2 = (variance1 * (2.0 + 1e-4) + 6.125 + 0.0625) / 4.0  # delta=0.25
    variance3 = (variance2 * (4.0 + 1e-4) + 4.5 + 2.625**2 * 4.0 / 3.0) / 6.0  # delta=-2.625
    means = [2.0, 2.125, 1.25]  # 对全部累计 trace 样本求 global mean
    variances = [variance1, variance2, variance3]  # 每步都加入上游 RMS epsilon，而非只加一次
    maxima = [3.0, 4.0, 4.0]  # 历史 max_abs 不因 episode reset 降低

    # 每个真实 env step 调一次 observe，count 每次增加 num_envs=2。
    for index, (reward, terminated, truncated, trace) in enumerate(steps):
        normalizer.observe(torch.tensor(reward), torch.tensor(terminated), torch.tensor(truncated))  # t → t+1
        torch.testing.assert_close(normalizer.returns, torch.tensor(trace), rtol=0, atol=0)  # 环境级 trace
        torch.testing.assert_close(normalizer.mean, torch.tensor(means[index]))  # 全局均值
        torch.testing.assert_close(normalizer.var, torch.tensor(variances[index]), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(normalizer.count, torch.tensor(float(2 * (index + 1))))  # 真样本计数
        torch.testing.assert_close(normalizer.max_abs, torch.tensor(maxima[index]))  # 单调历史界


@pytest.mark.parametrize("max_return", [0.5, 100.0])
def test_reward_scale_is_readonly_uncentered_and_differentiable(max_return: float) -> None:
    r"""分别让历史界与方差主导分母；scale 不减均值，不计入 replay 的重复读取。"""
    normalizer = DiscountedRewardNormalizer(2, gamma=0.5, max_return=max_return)  # 两种归一化主导机制
    normalizer.observe(
        torch.tensor([1.0, 3.0], requires_grad=True), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    )
    saved = {name: value.clone() for name, value in normalizer.state_dict().items()}  # 包含 trace 与计数
    rewards = torch.tensor([[0.0, 2.0], [-2.0, 1.0]], requires_grad=True)  # replay 形状可独立于 num_envs
    denominator = max(math.sqrt((2.0 + 1e-4) / 2.0 + 1e-8), 3.0 / max_return)  # 手算尺度
    actual = normalizer.scale(rewards)  # 只读，不把 batch 再 observe 一次
    torch.testing.assert_close(actual, rewards / denominator)  # 不减去 mean=2
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), rewards)[0], torch.full_like(rewards, 1 / denominator))
    normalizer.scale(rewards.detach())  # 同一 replay 再次 scale 不改变统计
    for name, value in normalizer.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)  # 全部 buffer 逐位不变
        assert not value.requires_grad, f"统计 buffer {name} 不应持有环境 reward 的梯度"


def test_reward_normalizer_state_dict_roundtrip_and_continuation() -> None:
    r"""仅用标准 Module state_dict 在内存恢复；恢复后的下一步应逐位一致。"""
    original = DiscountedRewardNormalizer(2, gamma=0.7, max_return=3.0, epsilon=1e-6)  # 非默认配置锚点
    done = torch.zeros(2, dtype=torch.bool)  # 首步不结束
    original.observe(torch.tensor([2.0, -1.0]), done, done)  # 建立非空 trace 与统计
    stream = io.BytesIO()  # 不创建 checkpoint 文件
    torch.save(original.state_dict(), stream)  # 只使用 torch 标准序列化
    stream.seek(0)  # 重放同一内存 checkpoint
    restored = DiscountedRewardNormalizer(2)  # checkpoint 同时恢复尺度配置与统计状态
    restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))  # 无自定义 load
    assert not list(restored.parameters()), "奖励统计是 buffer，不应出现可训练 Parameter"
    assert set(restored.state_dict()) == set(dict(restored.named_buffers())), "统计与配置必须可随 Module 迁移"

    # 继续推进有 timeout 的真实环境步，检验 G_old、gamma、RMS count 是否都已恢复。
    reward = torch.tensor([-0.5, 4.0])  # 下一环境步
    timeout = torch.tensor([False, True])  # 第二环境截断
    original.observe(reward, done, timeout)  # 原状态继续
    restored.observe(reward, done, timeout)  # 恢复状态继续
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)  # 完整数值状态
    torch.testing.assert_close(restored.scale(reward), original.scale(reward), rtol=0, atol=0)  # 尺度同样一致


def test_single_env_zero_rewards_and_failed_observation_leave_valid_state() -> None:
    r"""单环境采用总体方差；零奖励与非法 observe 均不能污染已形成的统计。"""
    normalizer = DiscountedRewardNormalizer(1)  # N=1 时 unbiased 方差会产生 NaN
    done = torch.zeros(1, dtype=torch.bool)  # [N]
    normalizer.observe(torch.zeros(1), done, done)  # 方差应为初始 M2 中的 1e-4
    torch.testing.assert_close(normalizer.var, torch.tensor(1e-4))  # population variance=0
    torch.testing.assert_close(normalizer.scale(torch.zeros(3)), torch.zeros(3))  # epsilon 防止 0/0
    before = {name: value.clone() for name, value in normalizer.state_dict().items()}  # 非法步前状态
    with pytest.raises(ValueError, match="reward"):
        normalizer.observe(torch.tensor([float("nan")]), done, done)  # 真实 reward 必须有限
    with pytest.raises(ValueError, match="shape"):
        normalizer.observe(torch.zeros(1, 1), done, done)  # 禁止环境轴的静默广播
    with pytest.raises(ValueError, match="terminated"):
        normalizer.observe(torch.zeros(1), torch.tensor([0.5]), done)  # done 必须是布尔物理事实
    for name, value in normalizer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)  # 校验失败不推进统计
