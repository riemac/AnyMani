r"""TIP-silent rejected-action cost 的纯 Torch 科学合同。

这些测试只证伪动作空间代数、有效性掩码、TIP gate、ghost 隔离和可微 ABI；它们不启动 Isaac Sim，
也不对策略学习效果、接触质量或恢复探索收益作任何声明。
"""

from __future__ import annotations

import math

import pytest
import torch
from anymani.distill.rl.algorithms.action_regularization import tip_silent_rejected_action_cost
from torch.func import grad, vmap


def _all_silent_inputs(
    *, batch: int = 1, joints: int = 3, tips: int = 4, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, ...]:
    r"""构造所有 TIP 有效且无接触的确定性输入，target/limits 均为 rad 除以 pi。

    返回顺序与公开 helper 的六个参数一致；此 fixture 只提供数学边界，不携带任何 task/force/history 信息。
    """

    target = torch.zeros(batch, joints, dtype=dtype)  # $u$，当前接受目标，单位为 rad/$\pi$
    limits = torch.stack(
        (
            torch.full_like(target, -1.0),  # $l_j$，归一化下限，单位为 rad/$\pi$
            torch.full_like(target, 1.0),  # $h_j$，归一化上限，单位为 rad/$\pi$
        ),
        dim=-1,
    )  # $[B,J,2]$，合法目标区间的两个端点
    joint_valid = torch.ones(batch, joints, dtype=torch.bool)  # 有效 JOINT 轴，ghost 为 False
    tip_contact = torch.zeros(batch, tips, dtype=torch.bool)  # 当前 TIP 无接触
    tip_valid = torch.ones(batch, tips, dtype=torch.bool)  # 至少一个有效 TIP，且此处全部有效
    return target, limits, joint_valid, tip_contact, tip_valid


def test_no_truncation_has_exact_zero_cost_and_gradient() -> None:
    r"""动作均在可接受区间内时，cost 与对 mean 的一阶导数必须逐值精确为零。"""

    target, limits, joint_valid, tip_contact, tip_valid = _all_silent_inputs()  # 所有有效 TIP 静默
    mean = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64, requires_grad=True)  # 归一化 policy mean

    cost, rejected_fraction = tip_silent_rejected_action_cost(
        mean=mean,
        target_normalized=target,
        limits_normalized=limits,
        joint_valid=joint_valid,
        tip_contact=tip_contact,
        tip_valid=tip_valid,
    )
    gradient = torch.autograd.grad(cost.sum(), mean)[0]  # 仅检查 action mean 的可微路径

    assert torch.equal(cost, torch.zeros_like(cost))  # 没有被截断的 JOINT 不产生惩罚
    assert torch.equal(rejected_fraction, torch.zeros_like(rejected_fraction))  # 没有拒绝事件
    assert torch.equal(gradient, torch.zeros_like(gradient))  # interior clamp 的一阶导数为零


def test_lower_and_upper_truncation_match_action_space_bounds_and_gradient() -> None:
    r"""上下限同时截断时，返回值和 mean 梯度核对 action-space bounds 代数。"""

    target, _, joint_valid, tip_contact, tip_valid = _all_silent_inputs(joints=3)  # 保持 gate=1
    limits = torch.tensor(
        [[[-0.01, 0.01], [-0.5, 0.5], [-0.01, 0.01]]], dtype=torch.float64
    )  # 前后两维只有 $0.01\pi$ 的归一化余量
    mean = torch.tensor([[-1.0, 0.25, 2.0]], dtype=torch.float64, requires_grad=True)  # 下截断、内部、上截断
    action_bounds = 24.0 * math.pi * (limits - target.unsqueeze(-1))  # $24\pi(l-u),24\pi(h-u)$
    accepted_action = torch.clamp(mean, min=action_bounds[..., 0], max=action_bounds[..., 1])  # action-space clamp
    expected_rejected = mean - accepted_action  # $r=\mu-\operatorname{clamp}(\mu,\mathrm{bounds})$
    expected_cost = expected_rejected.square().sum(dim=-1) / 3.0  # 按三个有效 JOINT 的数量取均值
    expected_fraction = (expected_rejected.abs() > 1.0e-6).to(torch.float64).sum(dim=-1) / 3.0  # 两个事件/三维

    cost, rejected_fraction = tip_silent_rejected_action_cost(
        mean=mean,
        target_normalized=target,
        limits_normalized=limits,
        joint_valid=joint_valid,
        tip_contact=tip_contact,
        tip_valid=tip_valid,
    )
    gradient = torch.autograd.grad(cost.sum(), mean)[0]  # outside clamp 的导数来自 rejected squared
    expected_gradient = 2.0 * expected_rejected / 3.0  # $\partial(r^2/3)/\partial\mu=2r/3$

    torch.testing.assert_close(cost, expected_cost, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(rejected_fraction, expected_fraction, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)


def test_five_milliradian_room_maps_to_012_action_units() -> None:
    r"""验证 0.005 rad 可用余量经 $24\pi$ 换算后恰为 0.12 无量纲动作单位。"""

    target, _, joint_valid, tip_contact, tip_valid = _all_silent_inputs(joints=1)  # 单一有效 JOINT
    target = torch.tensor([[0.2]], dtype=torch.float64)  # $u=0.2$，单位 rad/$\pi$
    room_normalized = torch.tensor(0.005 / math.pi, dtype=torch.float64)  # 0.005 rad 除以 pi
    limits = torch.stack((target - 1.0, target + room_normalized), dim=-1)  # 上限距目标 0.005 rad
    mean = torch.tensor([[0.2]], dtype=torch.float64)  # $\mu=0.2$，超出上限 action-space余量 0.08

    cost, rejected_fraction = tip_silent_rejected_action_cost(
        mean=mean,
        target_normalized=target,
        limits_normalized=limits,
        joint_valid=joint_valid,
        tip_contact=tip_contact,
        tip_valid=tip_valid,
    )
    expected_room = 24.0 * math.pi * room_normalized  # $24\pi(0.005/\pi)=0.12$
    expected_rejected = mean - expected_room  # $0.2-0.12=0.08$

    torch.testing.assert_close(cost, expected_rejected.square().squeeze(-1), rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(rejected_fraction, torch.ones_like(rejected_fraction), rtol=0.0, atol=0.0)


def test_hand_wide_gate_uses_only_valid_tip_contacts() -> None:
    r"""gate 只由 valid TIP 的 contact 决定，并区分无 valid TIP、有效接触和 invalid ghost 接触。"""

    target, _, joint_valid, _, _ = _all_silent_inputs(batch=3, joints=2)  # 三个样本共用截断动作
    limits = torch.full((3, 2, 2), 0.01, dtype=torch.float64)  # 正 mean 会被上限拒绝
    limits[..., 0] = -0.01  # 下限与上限对称，保持每个 valid JOINT 的同一 action bounds
    mean = torch.full((3, 2), 1.0, dtype=torch.float64)  # 每个样本本身都有正的候选 cost
    tip_valid = torch.tensor(
        [[True, False, False, False], [False, False, False, False], [True, True, False, False]], dtype=torch.bool
    )  # 样本 0 仅一个 valid TIP；样本 1 没有；样本 2 有两个
    tip_contact = torch.tensor(
        [[False, True, True, True], [False, False, False, False], [False, True, False, False]], dtype=torch.bool
    )  # 样本 0 的 invalid TIP 虽接触仍不能关闭 gate

    cost, rejected_fraction = tip_silent_rejected_action_cost(
        mean=mean,
        target_normalized=target,
        limits_normalized=limits,
        joint_valid=joint_valid,
        tip_contact=tip_contact,
        tip_valid=tip_valid,
    )
    active = torch.tensor([True, False, False])  # 只有样本 0 满足“至少一个 valid TIP 且全部无接触”

    assert torch.all(cost[active] > 0.0)  # gate 开启时保留整手 cost
    assert torch.equal(cost[~active], torch.zeros_like(cost[~active]))  # gate 关闭时逐 sample 清零
    assert torch.all(rejected_fraction[active] > 0.0)  # gate 同样控制诊断比例
    assert torch.equal(rejected_fraction[~active], torch.zeros_like(rejected_fraction[~active]))


def test_nan_ghost_mean_target_and_limits_are_cleaned_before_formula_and_backward() -> None:
    r"""ghost 的 NaN 不得污染 cost、fraction 或 mean/target/limits 梯度。"""

    torch.manual_seed(17)  # 固定随机源，保证该 ghost 隔离合同可复现
    target, limits, joint_valid, tip_contact, tip_valid = _all_silent_inputs(batch=2, joints=3)  # 两个样本
    joint_valid[:, 1] = False  # 第 1 个 JOINT 设为 ghost，三类连续输入的 NaN 才应被隔离
    raw_mean = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)  # 可微 policy mean
    raw_target = target.detach().clone().requires_grad_()  # 可微目标输入
    raw_limits = limits.detach().clone().requires_grad_()  # 可微 limits 输入
    with torch.no_grad():
        raw_mean[:, 1] = float("nan")  # 第 1 个 JOINT 是 ghost，mean 以 NaN 模拟上游 poison
        raw_target[:, 1] = float("nan")  # ghost target 同样不应进入 u
        raw_limits[:, 1, :] = float("nan")  # ghost 两端 limits 同样不应进入 clamp
    clean_cost, clean_fraction = tip_silent_rejected_action_cost(
        mean=torch.nan_to_num(raw_mean.detach(), nan=0.0),
        target_normalized=torch.nan_to_num(raw_target.detach(), nan=0.0),
        limits_normalized=torch.nan_to_num(raw_limits.detach(), nan=0.0),
        joint_valid=joint_valid,
        tip_contact=tip_contact,
        tip_valid=tip_valid,
    )  # 对照值只用于确认 ghost 行被完全排除
    cost, rejected_fraction = tip_silent_rejected_action_cost(
        mean=raw_mean,
        target_normalized=raw_target,
        limits_normalized=raw_limits,
        joint_valid=joint_valid,
        tip_contact=tip_contact,
        tip_valid=tip_valid,
    )
    gradients = torch.autograd.grad(cost.sum(), (raw_mean, raw_target, raw_limits))  # 检查三条输入梯度路径

    torch.testing.assert_close(cost, clean_cost, rtol=0.0, atol=0.0)
    torch.testing.assert_close(rejected_fraction, clean_fraction, rtol=0.0, atol=0.0)
    assert torch.isfinite(cost).all() and torch.isfinite(rejected_fraction).all()  # forward 不得出现 NaN
    for gradient in gradients:
        assert torch.isfinite(gradient).all()  # backward 不得从 ghost poison 生成 NaN
    assert torch.equal(gradients[0][:, 1], torch.zeros_like(gradients[0][:, 1]))  # ghost mean 无梯度
    assert torch.equal(gradients[1][:, 1], torch.zeros_like(gradients[1][:, 1]))  # ghost target 无梯度
    assert torch.equal(gradients[2][:, 1, :], torch.zeros_like(gradients[2][:, 1, :]))  # ghost limits 无梯度


def test_joint_sign_reflection_preserves_cost_and_rejected_fraction() -> None:
    r"""同时反射 $\mu,u,[l,h]$ 时 rejected 变号但平方 cost/事件比例不变。"""

    target, limits, joint_valid, tip_contact, tip_valid = _all_silent_inputs(joints=4)  # 原坐标合同
    torch.manual_seed(23)  # 固定反射样本
    mean = torch.randn(1, 4, dtype=torch.float64) * 2.0  # 同时包含内点与被截断点
    target = torch.randn(1, 4, dtype=torch.float64) * 0.1  # 合法目标中心
    half_width = torch.full_like(target, 0.03)  # 对称归一化限位半宽
    limits = torch.stack((target - half_width, target + half_width), dim=-1)  # $u$ 位于 limits 内
    reflected_limits = torch.stack((-limits[..., 1], -limits[..., 0]), dim=-1)  # $[-h,-l]$

    original = tip_silent_rejected_action_cost(
        mean=mean,
        target_normalized=target,
        limits_normalized=limits,
        joint_valid=joint_valid,
        tip_contact=tip_contact,
        tip_valid=tip_valid,
    )
    reflected = tip_silent_rejected_action_cost(
        mean=-mean,
        target_normalized=-target,
        limits_normalized=reflected_limits,
        joint_valid=joint_valid,
        tip_contact=tip_contact,
        tip_valid=tip_valid,
    )

    torch.testing.assert_close(reflected[0], original[0], rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(reflected[1], original[1], rtol=0.0, atol=0.0)


def test_float64_cost_matches_explicit_target_space_prediction() -> None:
    r"""float64 下与 $\operatorname{clip}(u+\mu/(24\pi),l,h)$ 的 target-space 公式逐 sample 对照。"""

    target, _, joint_valid, tip_contact, tip_valid = _all_silent_inputs(batch=2, joints=4)  # gate=1 的双样本
    torch.manual_seed(29)  # 固定可复核浮点输入
    target = torch.randn(2, 4, dtype=torch.float64) * 0.1  # 合法归一化目标 $u$
    half_width = torch.rand(2, 4, dtype=torch.float64) * 0.1 + 0.01  # 每维正的归一化 limits 半宽
    limits = torch.stack((target - half_width, target + half_width), dim=-1)  # $u$ 严格位于 limits 内
    mean = torch.randn(2, 4, dtype=torch.float64) * 2.0  # 产生内点、上下截断和大幅拒绝
    predicted_target = torch.clamp(
        target + mean / (24.0 * math.pi), min=limits[..., 0], max=limits[..., 1]
    )  # $\hat q=\operatorname{clip}(u+\mu/(24\pi),l,h)$
    explicit_rejected = mean - 24.0 * math.pi * (predicted_target - target)  # 参考公式的接受动作相减
    expected_cost = explicit_rejected.square().sum(dim=-1) / joint_valid.sum(dim=-1).to(torch.float64)  # 未加权 cost
    expected_fraction = (explicit_rejected.abs() > 1.0e-6).to(torch.float64).sum(dim=-1) / 4.0  # 诊断阈值 1e-6

    actual_cost, actual_fraction = tip_silent_rejected_action_cost(
        mean=mean,
        target_normalized=target,
        limits_normalized=limits,
        joint_valid=joint_valid,
        tip_contact=tip_contact,
        tip_valid=tip_valid,
    )

    torch.testing.assert_close(actual_cost, expected_cost, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(actual_fraction, expected_fraction, rtol=0.0, atol=0.0)


def test_clamp_threshold_has_zero_boundary_gradient_and_continuous_first_derivative() -> None:
    r"""平方 cost 的一阶导数在 clamp 边界为零且连续；不把二阶曲率变化误判为梯度跳变。"""

    dtype = torch.float64  # 用高精度观察边界的一阶导数极限
    target = torch.zeros(1, 1, dtype=dtype)  # $u=0$
    limits = torch.tensor([[[-0.02, 0.03]]], dtype=dtype)  # target-space 两端
    joint_valid = torch.ones(1, 1, dtype=torch.bool)  # 唯一 JOINT 有效
    tip_contact = torch.zeros(1, 4, dtype=torch.bool)  # 全部 valid TIP 无接触
    tip_valid = torch.ones(1, 4, dtype=torch.bool)  # gate=1
    bounds = 24.0 * math.pi * limits  # 直接换算 action-space 的 lower/upper

    def cost_and_gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""返回一个边界测试点的 scalar cost 与 mean 一阶导。"""

        mean = value.reshape(1, 1).clone().requires_grad_()  # 当前测试点的可微 policy mean
        cost, _ = tip_silent_rejected_action_cost(
            mean=mean,
            target_normalized=target,
            limits_normalized=limits,
            joint_valid=joint_valid,
            tip_contact=tip_contact,
            tip_valid=tip_valid,
        )
        gradient = torch.autograd.grad(cost.sum(), mean)[0]  # $∂cost/∂\mu$
        return cost.squeeze(0), gradient.squeeze(0)

    epsilon = torch.tensor(1.0e-7, dtype=dtype)  # 两侧靠近边界的 action-space 距离
    for boundary in bounds.reshape(-1):  # 分别验证 lower 与 upper
        at_cost, at_gradient = cost_and_gradient(boundary)  # 精确边界点
        left_cost, left_gradient = cost_and_gradient(boundary - epsilon)  # 边界左侧
        right_cost, right_gradient = cost_and_gradient(boundary + epsilon)  # 边界右侧
        assert torch.equal(at_cost, torch.zeros_like(at_cost))  # $r=0\Rightarrow cost=0$
        assert torch.equal(at_gradient, torch.zeros_like(at_gradient))  # clamp 边界的一阶导为零
        assert torch.abs(left_gradient).item() < 1.0e-6  # 左极限趋向零
        assert torch.abs(right_gradient).item() < 1.0e-6  # 右极限趋向零
        assert torch.abs(left_cost).item() <= 2.0e-14 and torch.abs(right_cost).item() <= 2.0e-14


def test_helper_supports_vmap_and_functional_grad_without_tensor_data_asserts() -> None:
    r"""rank-1 单样本视图可被 vmap/grad 组合，证明 helper 没有 data-dependent 控制流。"""

    target, limits, joint_valid, tip_contact, tip_valid = _all_silent_inputs(batch=3, joints=2)  # vmap 的三行
    torch.manual_seed(31)  # 固定函数式变换输入
    means = torch.randn(3, 2, dtype=torch.float64)  # [B,J]，vmap 后每次传入 [J]

    def per_sample(
        sample_mean: torch.Tensor,
        sample_target: torch.Tensor,
        sample_limits: torch.Tensor,
        sample_joint_valid: torch.Tensor,
        sample_tip_contact: torch.Tensor,
        sample_tip_valid: torch.Tensor,
    ) -> torch.Tensor:
        r"""将公开 batch helper 作为 rank-1 单样本纯函数调用。"""

        return tip_silent_rejected_action_cost(
            mean=sample_mean,
            target_normalized=sample_target,
            limits_normalized=sample_limits,
            joint_valid=sample_joint_valid,
            tip_contact=sample_tip_contact,
            tip_valid=sample_tip_valid,
        )[0]

    vmapped_cost = vmap(per_sample)(means, target, limits, joint_valid, tip_contact, tip_valid)  # [B]
    direct_cost = per_sample(means, target, limits, joint_valid, tip_contact, tip_valid)  # 同一批量调用 [B]
    first_gradient = grad(per_sample)(means[0], target[0], limits[0], joint_valid[0], tip_contact[0], tip_valid[0])
    first_mean = means[0].detach().clone().requires_grad_()  # 独立 eager 路径，用于核对函数式 grad 数值
    first_cost = per_sample(
        first_mean, target[0], limits[0], joint_valid[0], tip_contact[0], tip_valid[0]
    )  # scalar cost
    eager_gradient = torch.autograd.grad(first_cost, first_mean)[0]  # 直接 autograd 的 $∂cost/∂\mu$

    torch.testing.assert_close(vmapped_cost, direct_cost, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(first_gradient, eager_gradient, rtol=1.0e-12, atol=1.0e-12)
    assert first_gradient.shape == (2,)  # $\nabla_\mu cost$ 保留 JOINT 轴
    assert torch.isfinite(first_gradient).all()  # vmap/grad 路径不产生 NaN


def test_shape_checks_are_static_and_report_expected_axes() -> None:
    r"""rank/axis 错误在进入张量数据运算前被拒绝，检查本身不依赖 tensor 内容。"""

    target, limits, joint_valid, tip_contact, tip_valid = _all_silent_inputs(joints=2)  # 合法参照
    with pytest.raises(ValueError, match=r"mean.*\[B,J\]"):
        tip_silent_rejected_action_cost(
            mean=torch.zeros(1, 2, 1, dtype=torch.float64),  # 非法 rank-3 mean
            target_normalized=target,
            limits_normalized=limits,
            joint_valid=joint_valid,
            tip_contact=tip_contact,
            tip_valid=tip_valid,
        )
    with pytest.raises(ValueError, match=r"limits_normalized.*\[B,J,2\]"):
        tip_silent_rejected_action_cost(
            mean=torch.zeros(1, 2, dtype=torch.float64),
            target_normalized=target,
            limits_normalized=torch.zeros(1, 2, 3, dtype=torch.float64),  # 最后一轴不是 [lo,hi]
            joint_valid=joint_valid,
            tip_contact=tip_contact,
            tip_valid=tip_valid,
        )
