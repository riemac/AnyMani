r"""FlashSAC 基础层的 CPU 数学合同；所有输入均为小尺寸合成张量。

通过文件位置加载纯 PyTorch 模块，避免包级环境注册；测试不依赖上游检出目录。
双网络参考使用逐网络运算，分别核对输出、输入梯度、每组参数梯度和 BN 统计量。
"""

from __future__ import annotations

import importlib.util
import math
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

# 只加载被测数学模块；父目录 distill 的位置由本测试文件定位。
_PATH = Path(__file__).resolve().parents[3] / "rl" / "flash_sac" / "normalized.py"  # 项目内稳定路径
_SPEC = importlib.util.spec_from_file_location("_flash_sac_normalized_contract", _PATH)  # 独立模块身份
assert _SPEC is not None and _SPEC.loader is not None, f"无法加载数学模块：{_PATH}"
n = importlib.util.module_from_spec(_SPEC)  # 不执行 anymani 的环境注册
_SPEC.loader.exec_module(n)  # 被测文件只依赖标准库与 PyTorch


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize(("input_dim", "output_dim"), [(3, 7), (7, 3), (1, 2)])
def test_linear_orthogonal_initialization_and_safe_row_projection(ensemble, input_dim, output_dim) -> None:
    r"""正交初始化与逐输出行单位范数是两步；投影不得重绑参数或改写已算梯度。"""
    torch.manual_seed(104)  # 固定正交矩阵的随机方向
    layer = n.EnsembleUnitLinear(2, input_dim, output_dim) if ensemble else n.UnitLinear(input_dim, output_dim)  # Q=1/2
    weight = layer.weight if ensemble else layer.w.weight  # 上游公开参数路径
    matrices = weight if ensemble else weight.unsqueeze(0)  # 统一成 [Q, out, in]

    # 高矩阵正交的是列，宽矩阵正交的是行；不能把初始化误当成已完成行投影。
    for matrix in matrices:
        gram = matrix.T @ matrix if output_dim >= input_dim else matrix @ matrix.T  # 较小空间的 Gram 矩阵
        torch.testing.assert_close(gram, torch.eye(min(input_dim, output_dim)), atol=5e-7, rtol=5e-7)  # FP32 正交误差
    weight.square().sum().backward()  # 生成非空梯度，验证投影保持优化器持有的 Parameter
    gradient = weight.grad.clone()  # 已计算的梯度应保持逐元素不变
    with torch.no_grad():
        weight.mul_(3.0)  # 模拟一次离开约束球面的参数更新
    layer.normalize_parameters()  # 默认启用梯度时也必须安全
    norms = torch.linalg.vector_norm(weight, dim=-1)  # 每个输出特征的输入方向范数
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=2e-7, rtol=2e-7)  # 逐输出行半径为 1
    torch.testing.assert_close(weight.grad, gradient, atol=0, rtol=0)  # 投影不充当一次梯度更新
    assert weight is (layer.weight if ensemble else layer.w.weight), "投影不得替换 Parameter 身份"


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("batch_norm", [False, True])
def test_affine_projection_uses_joint_radius_and_keeps_zero_finite(ensemble, batch_norm) -> None:
    r"""BN 的 $(\gamma,\beta)$ 联合半径与 RMS 的 $\gamma$ 半径均为 $\sqrt d$。"""
    torch.manual_seed(105)  # 非均匀 scale/bias 避免只验证初始化特例
    name = ("Ensemble" if ensemble else "") + ("UnitBatchNorm" if batch_norm else "UnitRMSNorm")  # 四类仿射约束
    layer = getattr(n, name)(2, 5) if ensemble else getattr(n, name)(5)  # d=5，Q=2
    with torch.no_grad():
        layer.weight.normal_(mean=0.4, std=1.7)  # scale 允许正负，不按每特征独立约束
        if batch_norm:  # RMS 只有 scale，BN 的 bias 也参与球面约束
            layer.bias.normal_(mean=-0.2, std=0.9)  # bias 与 scale 共享一个归一化因子
    before = layer.weight.detach().clone()  # 保存投影前方向
    bias_before = layer.bias.detach().clone() if batch_norm else torch.zeros_like(before)  # RMS 的 bias 项为零
    radius = (before.square() + bias_before.square()).sum(dim=-1, keepdim=True).sqrt()  # 联合欧氏范数
    layer.normalize_parameters()  # 方法自己进入 no_grad
    torch.testing.assert_close(layer.weight, before * math.sqrt(5) / radius, atol=3e-7, rtol=3e-7)  # 检查方向保持
    if batch_norm:  # 联合球面投影不能独立改变 bias 相对 scale 的比例
        torch.testing.assert_close(layer.bias, bias_before * math.sqrt(5) / radius, atol=3e-7, rtol=3e-7)  # 同一因子
    square_norm = layer.weight.square().sum(-1)  # RMS 的约束对象只有 scale
    if batch_norm:  # 拼接向量的平方范数等于两部分平方范数之和
        square_norm = square_norm + layer.bias.square().sum(-1)  # BN 必须包含 bias
    torch.testing.assert_close(square_norm.sqrt(), torch.full_like(square_norm, math.sqrt(5)))  # 目标半径 sqrt(d)

    # 上游稳定项使全零向量成为有限的固定点；没有定义从零方向重建单位向量。
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()  # 覆盖 scale 和可能存在的 bias
    layer.normalize_parameters()  # 不能因零范数产生 NaN
    for parameter in layer.parameters():
        torch.testing.assert_close(parameter, torch.zeros_like(parameter), atol=0, rtol=0)  # 零方向是精确固定点


@pytest.mark.parametrize("ensemble", [False, True])
def test_rms_activation_norm_and_constant_input(ensemble) -> None:
    r"""$\gamma=1$ 时 $\|y\|^2=d\|x\|^2/(\|x\|^2+d\epsilon)$；常量输入不被减均值。"""
    layer = n.EnsembleUnitRMSNorm(2, 4).double() if ensemble else n.UnitRMSNorm(4).double()  # d=4
    x = torch.tensor([[0.0] * 4, [1.0] * 4, [-3.0, -1.0, 2.0, 4.0]], dtype=torch.float64)  # 三类输入
    if ensemble:  # 每个网络可有不同特征幅度，归一化不能跨 Q 归约
        x = torch.stack((x, 2 * x))  # 第二组改变幅度以检查逐网络归一化
    actual = layer(x)  # [B, d] 或 [Q, B, d]
    expected = x / (x.square().mean(dim=-1, keepdim=True) + layer.eps).sqrt()  # 独立的 RMS 定义
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)  # 逐分量检查 RMS 分母
    square_norm = x.square().sum(dim=-1)  # 输入平方范数，逐样本计算
    expected_radius = (4 * square_norm / (square_norm + 4 * layer.eps)).sqrt()  # 稳定项引起的范数收缩
    torch.testing.assert_close(actual.norm(dim=-1), expected_radius, atol=1e-12, rtol=1e-12)  # 检查 epsilon 的收缩量
    assert torch.all(actual[..., 1, :] > 0.99), "常量非零输入应保持方向，RMSNorm 不减均值"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize(
    ("name", "arguments", "input_dim", "has_training"),
    [
        ("UnitLinear", (3, 4), 3, False),  # 线性参数布局转换
        ("UnitBatchNorm", (3,), 3, True),  # 独立批统计与仿射梯度
        ("UnitRMSNorm", (3,), 3, False),  # 样本内均方根与 scale 梯度
        ("FlashSACEmbedder", (3, 4), 3, True),  # BN -> Linear 的梯度链
        ("FlashSACBlock", (3, 2), 3, True),  # 两次 BN/ReLU 与恒等残差的梯度链
    ],
)
def test_double_q_matches_independent_network_outputs_gradients_and_buffers(
    dtype, training, name, arguments, input_dim, has_training
) -> None:
    r"""双 Q 批运算必须等于两次单网络运算，包含输入及每个参数的梯度。

    $L=\sum_{qbo}c_{qbo}y_{qbo}$ 使用非均匀余切 $c$；单纯对 BN 输出求和会掩盖输入梯度错误。
    FP32 容差 $2\times10^{-5}$ 覆盖原生 BN 与显式归约的累加次序差异；FP64 使用 $10^{-10}$。
    """
    torch.manual_seed(206)  # 所有组件、两种精度使用同一可复现采样序列
    ensemble = getattr(n, f"Ensemble{name}")(2, *arguments).to(dtype=dtype)  # Q=2
    references = [getattr(n, name)(*arguments).to(dtype=dtype) for _ in range(2)]  # 逐网络参考
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10  # 根据归约精度设定可审计容差

    # 使用非平凡的正负参数和非初始运行统计，防止单一初始化掩盖网络轴错误。
    with torch.no_grad():
        for parameter in ensemble.parameters():
            parameter.normal_(mean=0.2, std=0.8)  # Q 组参数相互独立
        for key, buffer in ensemble.named_buffers():
            buffer.uniform_(0.4, 1.5) if key.endswith("running_var") else buffer.normal_()  # 方差保持正值
    state = ensemble.state_dict()  # ensemble 键为 weight/w1.weight，单网络多一层 w
    key_map = {
        key: "weight" if key == "w.weight" else key.replace(".w.weight", ".weight")
        for key in references[0].state_dict()
    }  # 其余 BN/RMS 的参数和 buffer 名称相同
    for index, reference in enumerate(references):
        reference.load_state_dict({key: state[key_map[key]][index] for key in key_map}, strict=True)
        reference.train(not training)  # 故意使 Module.training 与显式参数冲突
    ensemble.train(not training)  # 组合层也必须逐层转发显式模式

    # 转置构造非连续输入，并让两组输入均值明显不同，暴露跨网络归约错误。
    x = torch.randn(5, 2, input_dim, dtype=dtype).transpose(0, 1)  # 非连续 [Q=2, B=5, in]
    x[1].add_(2.3)  # 不同网络的数据分布偏移
    x.requires_grad_()  # 输入是叶子张量，直接比较其梯度
    reference_x = x.detach().clone().requires_grad_()  # 独立的参考计算图
    actual = ensemble(x, training) if has_training else ensemble(x)  # 并行计算
    expected = torch.stack(
        [
            reference(reference_x[q], training) if has_training else reference(reference_x[q])
            for q, reference in enumerate(references)
        ]
    )  # 两个网络分别前向，再仅在输出端堆叠
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance, msg=f"{name} 输出不一致")  # 逐网络前向
    cotangent = torch.randn_like(actual)  # 不同网络/样本/特征使用不同损失方向
    (actual * cotangent).sum().backward()  # 向量-雅可比积的并行实现
    (expected * cotangent).sum().backward()  # 逐网络参考梯度
    torch.testing.assert_close(  # 非均匀余切对应的输入向量-雅可比积
        x.grad, reference_x.grad, atol=tolerance, rtol=tolerance, msg=f"{name} 输入梯度"
    )  # dL/dx

    # 按公开 state_dict 路径逐组核对参数梯度及训练产生的运行统计。
    parameters = dict(ensemble.named_parameters())  # Q 轴仍保留在每个参数的第一维
    buffers = dict(ensemble.named_buffers())  # 每个网络独立的均值/方差
    for index, reference in enumerate(references):
        for key, parameter in reference.named_parameters():
            actual_gradient = parameters[key_map[key]].grad[index]  # 指定网络 q 的梯度切片
            torch.testing.assert_close(
                actual_gradient, parameter.grad, atol=tolerance, rtol=tolerance, msg=f"{name}: q={index}, 参数={key}"
            )
        for key, buffer in reference.named_buffers():
            torch.testing.assert_close(
                buffers[key_map[key]][index],  # 集成网络 q 的运行统计
                buffer,  # 独立单网络的参考统计
                atol=tolerance,
                rtol=tolerance,
                msg=f"{name}: q={index}, 统计={key}",
            )


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("momentum", [0.0, 0.01, 1.0])
def test_bn_training_updates_unbiased_statistics_and_eval_is_read_only(ensemble, dtype, momentum) -> None:
    r"""训练标准化用 correction=0，运行方差用 correction=1；显式 eval 单样本只读。

    连续使用 $B=5$、$B=2$ 两批，第二批无偏修正因子为 2，能区分有偏/无偏统计。
    $m=0,.01,1$ 分别检查冻结、上游默认和完全替换的指数更新边界。
    """
    layer = (  # 对比共享的单网络统计公式与双 Q 逐组统计公式
        n.EnsembleUnitBatchNorm(2, 3, momentum=momentum) if ensemble else n.UnitBatchNorm(3, momentum=momentum)
    )  # d=3
    layer = layer.to(dtype=dtype)  # CPU 的单精度与双精度均可更新统计
    axis = 1 if ensemble else 0  # 唯一参与统计归约的批轴
    expected_mean = torch.zeros_like(layer.running_mean)  # 初始运行均值
    expected_var = torch.ones_like(layer.running_var)  # 初始运行方差
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12  # 原生与显式 BN 归约误差
    layer.eval()  # 显式 True 必须覆盖这一状态

    # 两批数据均含特征间不同均值，且 Q=2 时两个网络的均值不同。
    for batch_size in (5, 2):
        shape = (2, batch_size, 3) if ensemble else (batch_size, 3)  # 当前批 shape
        x = (torch.arange(math.prod(shape), dtype=dtype).reshape(shape) / 5 - batch_size).requires_grad_()
        mean = x.mean(dim=axis, keepdim=True)  # 当前批均值
        biased_var = x.var(dim=axis, correction=0, keepdim=True)  # 前向应使用的方差
        expected_output = (x - mean) / (biased_var + layer.eps).sqrt()  # gamma=1,beta=0 的解析前向
        expected_mean = torch.lerp(expected_mean, mean.squeeze(axis).detach(), momentum)  # 运行均值 EMA
        expected_var = torch.lerp(expected_var, x.detach().var(dim=axis, correction=1), momentum)  # 无偏方差 EMA
        actual = layer(x, training=True)  # .eval() 状态下仍执行训练语义
        torch.testing.assert_close(actual, expected_output, atol=tolerance, rtol=tolerance)  # 训练前向用有偏方差
        torch.testing.assert_close(layer.running_mean, expected_mean, atol=tolerance, rtol=tolerance)  # EMA 均值
        torch.testing.assert_close(layer.running_var, expected_var, atol=tolerance, rtol=tolerance)  # EMA 无偏方差
        assert layer.running_mean.grad_fn is None and layer.running_var.grad_fn is None, "运行统计不得接入历史梯度图"

    # 再切换为 Module.train()，以显式 False 检验单样本评估及全部 state_dict 内容不变。
    layer.train()  # 与本次传入的 training=False 冲突
    saved = {key: value.clone() for key, value in layer.state_dict().items()}  # 包括仿射参数与 buffer
    x = torch.full((2, 1, 3) if ensemble else (1, 3), 4.2, dtype=dtype)  # B=1 的推理样本
    expected = (x - expected_mean.unsqueeze(axis)) / (expected_var.unsqueeze(axis) + layer.eps).sqrt()  # 固定运行统计
    torch.testing.assert_close(layer(x, training=False), expected, atol=tolerance, rtol=tolerance)  # B=1 合法推理
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, saved[key], atol=0, rtol=0, msg=f"评估改写了 {key}")  # eval 不产生状态漂移


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("batch_size", [0, 1])
def test_bn_rejects_undefined_training_batch_before_mutating_buffers(ensemble, batch_size) -> None:
    r"""$B\le1$ 的训练批在统计更新前拒绝，不允许除以零或写入 NaN。"""
    layer = n.EnsembleUnitBatchNorm(2, 3) if ensemble else n.UnitBatchNorm(3)  # 同一每通道样本数边界
    saved = {key: value.clone() for key, value in layer.state_dict().items()}  # 失败前完整状态
    x = torch.zeros((2, batch_size, 3) if ensemble else (batch_size, 3))  # 包括空批和单样本
    with pytest.raises(ValueError, match="more than 1 sample"):  # 统计样本不足必须在写入前失败
        layer(x, training=True)  # 必须先判断 B，再更新 running_var
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, saved[key], atol=0, rtol=0, msg=f"无效批改写了 {key}")  # 失败也不得污染统计


def test_single_bn_preserves_upstream_spatial_sample_count() -> None:
    r"""单网络 F.batch_norm 的 $[B,d,L]$ 输入统计样本数为 $BL$，不只是 B。"""
    layer = n.UnitBatchNorm(2).double()  # 两个通道，各有三个空间/序列样本
    x = torch.tensor([[[1.0, 3.0, 2.0], [-1.0, -3.0, -2.0]]], dtype=torch.float64)  # [1, 2, 3]
    actual = layer(x, training=True)  # B=1 但每通道 M=3，训练统计有定义
    expected = (x - x.mean(-1, keepdim=True)) / (  # 每个通道先减去序列均值
        x.var(-1, correction=0, keepdim=True) + layer.eps
    ).sqrt()  # 沿 L 轴统计，分母使用有偏方差
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)  # 与上游额外样本轴行为一致


def test_categorical_known_probabilities_expectation_and_mode_independence() -> None:
    r"""已知三原子概率产生手算期望 $Q_0=1.9,Q_1=0.1$，两个模式相同。"""
    torch.manual_seed(307)  # 小尺寸读出初始化可复现
    head = n.EnsembleCategoricalValue(2, 4, 3, -2.0, 4.0)  # 支持集 z=(-2,1,4)
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])  # 各网络独立的概率
    with torch.no_grad():
        head.w.weight.zero_()  # 用偏置直接指定分布，使期望可手算
        head.bias.copy_(probabilities.log())  # softmax(log p)=p
    x = torch.zeros(2, 5, 4)  # [Q=2, B=5, hidden=4]
    q_values, info = head(x, training=True)  # 训练模式的分布读出
    eval_q, eval_info = head(x, training=False)  # 评估不改变这一公式
    assert q_values.shape == (2, 5), f"期望 Q shape=[2,5]，实际 {q_values.shape}"
    assert set(info) == {"log_prob"} and info["log_prob"].shape == (2, 5, 3), "离散分布需保留 [Q,B,K]"
    torch.testing.assert_close(info["log_prob"].exp(), probabilities[:, None, :].expand(-1, 5, -1))  # 概率质量为给定 p
    torch.testing.assert_close(q_values, torch.tensor([[1.9], [0.1]]).expand(-1, 5), atol=2e-7, rtol=2e-6)  # 手算期望
    torch.testing.assert_close(eval_q, q_values, atol=0, rtol=0)  # 模式不改变期望值
    torch.testing.assert_close(eval_info["log_prob"], info["log_prob"], atol=0, rtol=0)  # 模式不改变完整分布
    assert head.bin_values.shape == (1, 1, 3) and head.bin_values.dtype == torch.float32, "上游原子 buffer ABI"
    assert set(head.state_dict()) == {"w.weight", "bias", "bin_values"}, "价值头参数路径须兼容上游"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_categorical_gradients_match_analytic_expectation_and_cross_entropy(dtype) -> None:
    r"""核对 $\partial Q/\partial\ell_k=p_k(z_k-Q)$ 与负对数概率的 $p_k-1_{k=t}$。

    损失 $L=\sum c_{qb}Q_{qb}-0.2\sum\log p_{qb,1}$ 同时使用价值与离散分布监督。
    再由线性层链式法则独立核对每个网络的输入、权重和偏置梯度。
    """
    torch.manual_seed(308)  # 非零权重、非均匀概率与损失余切
    head = n.EnsembleCategoricalValue(2, 3, 4, -3.0, 6.0).to(dtype=dtype)  # 四原子整数支持
    x = torch.randn(2, 4, 3, dtype=dtype, requires_grad=True)  # Q=2，B=4，hidden=3
    actual_q, info = head(x, training=True)  # 两条可微输出
    logits = torch.stack(  # 参考路径逐个网络执行普通线性层
        [F.linear(x[q].detach(), head.w.weight[q].detach(), head.bias[q].detach()) for q in range(2)]
    )  # [Q,B,K]
    probabilities = logits.softmax(-1)  # 逐网络参考概率，不借助被测头的输出
    atoms = head.bin_values.detach()  # z_k，广播形状 [1,1,4]
    expected_q = (probabilities * atoms).sum(-1)  # 独立期望参考
    tolerance = 3e-6 if dtype == torch.float32 else 1e-12  # 小尺寸线性归约的精度范围
    torch.testing.assert_close(actual_q, expected_q, atol=tolerance, rtol=tolerance)  # E[Z] 的逐网络参考
    torch.testing.assert_close(  # log(sum_k p_k)=0，按每个网络、每个样本验证
        info["log_prob"].logsumexp(-1), torch.zeros_like(actual_q), atol=tolerance, rtol=0
    )  # 概率归一
    assert torch.all((actual_q >= -3) & (actual_q <= 6)), "概率加权期望必须落在价值支持凸包内"

    # 使用解析 logits 梯度，而非再次调用自动微分实现来生成期望答案。
    cotangent = torch.randn_like(actual_q)  # 各个 Q 与样本的损失系数
    loss = (actual_q * cotangent).sum() - 0.2 * info["log_prob"][..., 1].sum()  # 期望 + 分类损失
    loss.backward()  # 被测计算图的实际梯度
    target = torch.zeros_like(probabilities)  # 离散监督的 one-hot
    target[..., 1] = 1  # 所有样本目标原子 t=1
    logits_gradient = cotangent[..., None] * probabilities * (atoms - expected_q[..., None])  # c*p_k*(z_k-Q)
    logits_gradient = logits_gradient + 0.2 * (probabilities - target)  # 分类监督的解析梯度
    expected_dx = torch.stack([logits_gradient[q] @ head.w.weight[q].detach() for q in range(2)])  # dL/dx
    expected_dw = torch.stack([logits_gradient[q].T @ x[q].detach() for q in range(2)])  # dL/dW
    torch.testing.assert_close(x.grad, expected_dx, atol=tolerance, rtol=tolerance)  # dL/dx=(dL/dlogits)*W
    torch.testing.assert_close(head.w.weight.grad, expected_dw, atol=tolerance, rtol=tolerance)  # dL/dW
    torch.testing.assert_close(  # 同组样本共享一个离散读出偏置
        head.bias.grad, logits_gradient.sum(1), atol=tolerance, rtol=tolerance
    )  # bias 梯度沿 B 累加


def test_categorical_extreme_logits_remain_finite() -> None:
    r"""大幅度 logits 的 log-softmax 应稳定，不能先求 exp 再取 log。"""
    torch.manual_seed(309)  # 读出初始化可复现
    head = n.EnsembleCategoricalValue(2, 2, 3, -1.0, 1.0)  # 有界价值支持
    with torch.no_grad():
        head.w.weight.zero_()  # 仅测试概率数值稳定性
        head.bias.copy_(torch.tensor([[1000.0, 0.0, -1000.0], [-1000.0, 0.0, 1000.0]]))  # 极端分布
    q_values, info = head(torch.zeros(2, 1, 2), training=False)  # 单样本推理
    assert torch.isfinite(info["log_prob"]).all(), "log-softmax 对有限极端 logits 应输出有限对数概率"
    torch.testing.assert_close(q_values, torch.tensor([[-1.0], [1.0]]), atol=0, rtol=0)  # 集中分布趋于支持端点


def test_projection_visits_semantic_leaves_once_and_preserves_ordinary_layers() -> None:
    r"""混合模型仅投影指定六类基础层；共享引用不重复，普通 Actor 层与同名方法不触发。"""
    torch.manual_seed(410)  # 每个组件与普通 Linear 的初值可复现
    single_embedder = n.FlashSACEmbedder(3, 4)  # 两个有投影语义的基础层
    single_block = n.FlashSACBlock(4, 2)  # 四个有投影语义的基础层
    single_rms = n.UnitRMSNorm(4)  # 一个独立 RMS 叶子
    ensemble_embedder = n.EnsembleFlashSACEmbedder(2, 3, 4)  # 独立网络的两个叶子
    ensemble_block = n.EnsembleFlashSACBlock(2, 4, 2)  # 独立网络的四个叶子
    ensemble_rms = n.EnsembleUnitRMSNorm(2, 4)  # 一个集成 RMS 叶子
    head = n.EnsembleCategoricalValue(2, 4, 3, -1.0, 2.0)  # 只投影其中的 w，bias 保持自由
    ordinary = torch.nn.Linear(3, 4)  # 代表现有 Actor 的普通仿射层
    root = torch.nn.ModuleList(
        [single_embedder, single_block, single_rms, ensemble_embedder, ensemble_block, ensemble_rms, head, ordinary]
    )  # 混合类型模型
    root.append(single_block.w1)  # 同一叶子重复注册到另一个路径
    root.normalize_parameters = Mock(side_effect=AssertionError("不得触发组合层的同名方法"))  # 非白名单方法
    ordinary.normalize_parameters = Mock(side_effect=AssertionError("不得触发现有普通 Actor 层"))  # 同名不代表语义
    leaves = [
        single_embedder.norm,  # 输入空间 BN
        single_embedder.w,  # 单网络嵌入矩阵
        single_block.w1,  # 单网络扩张矩阵
        single_block.w2,  # 单网络收缩矩阵
        single_block.norm1,  # 单网络扩张空间 BN
        single_block.norm2,  # 单网络隐空间 BN
        single_rms,  # 单网络最终缩放
        ensemble_embedder.norm,  # 逐网络输入空间 BN
        ensemble_embedder.w,  # 集成嵌入矩阵
        ensemble_block.w1,  # 集成扩张矩阵
        ensemble_block.w2,  # 集成收缩矩阵
        ensemble_block.norm1,  # 逐网络扩张空间 BN
        ensemble_block.norm2,  # 逐网络隐空间 BN
        ensemble_rms,  # 逐网络最终缩放
        head.w,  # 离散 logits 的线性映射
    ]  # 15 个不同对象；显式枚举不借助被测遍历逻辑生成期望

    # 保存不属于投影对象的参数和全部运行统计，并扰动需要投影的参数以确保操作可观察。
    with torch.no_grad():
        for parameter in root.parameters():
            parameter.add_(0.3)  # 包括 BN bias，让联合范数约束真正生效
    protected = [ordinary.weight, ordinary.bias, head.bias, *root.buffers()]  # 绝不应被投影改写
    saved = [tensor.clone() for tensor in protected]  # 精确逐元素比较
    with ExitStack() as stack:
        calls = [
            stack.enter_context(patch.object(layer, "normalize_parameters", wraps=layer.normalize_parameters))
            for layer in leaves
        ]  # 记录实际方法调用，并执行真实投影公式
        result = n.project_unit_parameters(root)  # 默认梯度模式下执行一次完整遍历
        assert result is None, "投影 API 只返回 None"
        for index, call in enumerate(calls):
            assert call.call_count == 1, f"投影叶子 index={index} 调用 {call.call_count} 次，期望 1 次"
    root.normalize_parameters.assert_not_called()  # 组合层不执行递归投影
    ordinary.normalize_parameters.assert_not_called()  # 不能仅根据 hasattr 选择普通层
    for actual, expected in zip(protected, saved, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)  # 普通 Actor 参数、价值偏置和统计逐元素不变
    row_norms = single_block.w1.w.weight.norm(dim=-1)  # 共享层确实完成行投影
    torch.testing.assert_close(row_norms, torch.ones_like(row_norms), atol=2e-7, rtol=2e-7)  # 共享层仍实际完成一次投影


@pytest.mark.parametrize(
    ("name", "arguments", "dimension_indices"),
    [
        ("UnitLinear", (3, 4), (0, 1)),  # input_dim/output_dim
        ("UnitBatchNorm", (3,), (0,)),  # input_dim
        ("UnitRMSNorm", (3,), (0,)),  # input_dim
        ("FlashSACEmbedder", (3, 4), (0, 1)),  # input_dim/hidden_dim
        ("FlashSACBlock", (3, 2), (0, 1)),  # hidden_dim/expansion
        ("EnsembleUnitLinear", (2, 3, 4), (0, 1, 2)),  # num_ensemble/input_dim/output_dim
        ("EnsembleUnitBatchNorm", (2, 3), (0, 1)),  # num_ensemble/input_dim
        ("EnsembleUnitRMSNorm", (2, 3), (0, 1)),  # num_ensemble/input_dim
        ("EnsembleFlashSACEmbedder", (2, 3, 4), (0, 1, 2)),  # num_ensemble/input_dim/hidden_dim
        ("EnsembleFlashSACBlock", (2, 3, 2), (0, 1, 2)),  # num_ensemble/hidden_dim/expansion
        ("EnsembleCategoricalValue", (2, 3, 4, -1.0, 1.0), (0, 1, 2)),  # num_ensemble/hidden_dim/num_bins
    ],
)
@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_invalid_constructor_dimensions_are_rejected(name, arguments, dimension_indices, invalid) -> None:
    r"""离散维度与网络数不接受零、负数、小数或布尔量。"""
    torch.manual_seed(411)  # 某些组合在后续维度拒绝前会初始化较早的层
    for index in dimension_indices:
        invalid_arguments = list(arguments)  # 每次只扰动一个维度
        invalid_arguments[index] = invalid  # 其余配置维持有效
        with pytest.raises(ValueError, match="integer"):  # 每个公开离散维度都必须满足整数定义域
            getattr(n, name)(*invalid_arguments)  # 必须明确拒绝，而非静默生成空层


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("UnitBatchNorm", (3,)),  # 单网络批归一化
        ("UnitRMSNorm", (3,)),  # 单网络均方根归一化
        ("EnsembleUnitBatchNorm", (2, 3)),  # 双网络批归一化
        ("EnsembleUnitRMSNorm", (2, 3)),  # 双网络均方根归一化
    ],
)
@pytest.mark.parametrize("eps", [0.0, -1e-6, float("nan"), float("inf")])
def test_invalid_normalization_epsilon_is_rejected(name, arguments, eps) -> None:
    r"""方差分母必须具有有限正稳定项，常量/零输入才有定义。"""
    with pytest.raises(ValueError, match="eps"):  # 防止零方差输入分母失效
        getattr(n, name)(*arguments, eps=eps)  # 非有限或非正的方差稳定项


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("momentum", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_bn_momentum_is_rejected(ensemble, momentum) -> None:
    r"""EMA 权重只能在 $[0,1]$ 内，不允许外插或非有限统计。"""
    constructor = n.EnsembleUnitBatchNorm if ensemble else n.UnitBatchNorm  # 两类 BN 共用统计约束
    with pytest.raises(ValueError, match="momentum"):  # 防止运行方差的非凸更新
        constructor(*((2, 3) if ensemble else (3,)), momentum=momentum)  # 参数解析期拒绝


@pytest.mark.parametrize(
    ("min_v", "max_v"),
    [(1.0, 1.0), (2.0, -1.0), (float("nan"), 1.0), (-1.0, float("inf")), (-1e100, 1.0), (-3e38, 3e38)],
)
def test_invalid_categorical_support_is_rejected(min_v, max_v) -> None:
    r"""价值支持需要有限有序端点，FP32 的端点及跨度不得溢出。"""
    with pytest.raises(ValueError, match="min_v|max_v"):  # 端点顺序与 FP32 数值域均需检查
        n.EnsembleCategoricalValue(2, 3, 4, min_v, max_v)  # 校验发生在 linspace 构造前


def test_categorical_requires_at_least_two_bins() -> None:
    r"""$K=1$ 没有覆盖非零长度价值区间的原子间距，应明确拒绝。"""
    with pytest.raises(ValueError, match="num_bins"):  # K-1 必须非零，支持间距才有定义
        n.EnsembleCategoricalValue(2, 3, 1, -1.0, 1.0)  # num_bins 的下界为 2


@pytest.mark.parametrize(
    ("name", "arguments", "shape", "training"),
    [
        ("UnitLinear", (3, 4), (2, 5), None),  # 最后一维错误
        ("UnitRMSNorm", (3,), (), None),  # 缺少特征轴
        ("UnitBatchNorm", (3,), (3,), False),  # 缺少批轴
        ("UnitBatchNorm", (3,), (2, 4), True),  # 通道数错误
        ("EnsembleUnitLinear", (2, 3, 4), (1, 5, 3), None),  # 不得广播 Q=1
        ("EnsembleUnitLinear", (2, 3, 4), (2, 5, 4), None),  # 输入维错误
        ("EnsembleUnitRMSNorm", (2, 3), (5, 3), None),  # 缺少网络轴
        ("EnsembleUnitBatchNorm", (2, 3), (1, 5, 3), True),  # 统计网络轴错误
        ("EnsembleUnitBatchNorm", (2, 3), (2, 5, 4), False),  # 评估也检查通道数
        ("EnsembleCategoricalValue", (2, 3, 4, -1.0, 1.0), (5, 3), False),  # 价值头需要显式 Q 轴
    ],
)
def test_invalid_forward_shapes_are_rejected(name, arguments, shape, training) -> None:
    r"""形状检查必须在计算前阻止网络轴混合与特征轴错配。"""
    torch.manual_seed(412)  # 初始化可复现
    layer = getattr(n, name)(*arguments)  # 合法构造，单独检验输入布局
    x = torch.zeros(shape)  # 全零数值使失败只能来自 shape 合同
    with pytest.raises(ValueError, match="shape"):  # 不允许把布局错误解释为广播
        layer(x) if training is None else layer(x, training=training)  # 根据上游签名传递模式


@pytest.mark.parametrize("ensemble", [False, True])
def test_bn_requires_host_bool_training_flag(ensemble) -> None:
    r"""拒绝张量模式标志，避免正常设备热路径出现隐式标量同步。"""
    layer = n.EnsembleUnitBatchNorm(2, 3) if ensemble else n.UnitBatchNorm(3)  # 两类显式模式 BN
    x = torch.zeros((2, 3, 3) if ensemble else (3, 3))  # 有效训练批，排除样本数错误
    with pytest.raises(TypeError, match="training must be bool"):  # 不执行 Tensor.__bool__
        layer(x, training=torch.tensor(True))  # 即使 CPU bool Tensor 也不转换为 Python bool
