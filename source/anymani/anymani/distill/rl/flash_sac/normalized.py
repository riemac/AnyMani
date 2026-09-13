# Copyright (c) 2026 Holiday Robotics
# SPDX-License-Identifier: MIT
# 来源：FlashRL 的 flash_rl/agents/flashSAC/layer.py；许可全文见同目录 LICENSE.upstream。
r"""FlashSAC 的归一化基础层、并行集成层与离散价值读出。

数学与属性名移植自上游 ``flash_rl/agents/flashSAC/layer.py``；组合顺序参照
``flash_rl/agents/flashSAC/network.py``，投影调用时机参照 ``agents/utils/network.py``。
记 $B$ 为样本数、$Q$ 为独立网络数（双 Q 时 $Q=2$）、$d$ 为特征维度、$K$ 为价值原子数。
单网络特征通常为 $[B,d]$，集成特征严格为 $[Q,B,d]$；网络轴不参与 BN 统计归约。

参数投影与前向归一化是不同操作：线性层每个输出行满足 $\|w_o\|_2=1$，
BN 的拼接仿射向量满足 $\|(\gamma,\beta)\|_2\simeq\sqrt d$，
RMSNorm 的缩放向量满足 $\|\gamma\|_2\simeq\sqrt d$。
仿射投影保留上游分母中的 $10^{-8}$；零向量保持为零，极小向量的范数小于目标半径。
构造函数只做上游初始化；调用方应在初始化后和 ``optimizer.step()`` 后调用
``project_unit_parameters``。前向不投影参数，编译与优化器生命周期由调用方控制。

BN 显式接收 ``training``，其值覆盖 ``module.training`` 的通常含义；训练使用批统计，
评估只读运行统计。参数/形状检查只读取 Python 标量与张量元数据，前向不读取设备上的元素值。
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_dimension(name: str, value: int, minimum: int = 1) -> None:
    r"""要求离散维度为至少 ``minimum`` 的整数；布尔量不表示维度。"""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:  # d/Q/e >= 1，K >= 2
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _finite_scalar(name: str, value: float) -> float:
    r"""构造期只接受有限实标量；不将设备张量转换为 Python 数值。"""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):  # 超参数属于有限实数域
        raise ValueError(f"{name} must be a finite real scalar, got {value!r}")
    return float(value)  # 配置值留在主机；前向无需读取设备元素


def _positive_eps(eps: float) -> float:
    r"""要求方差分母的稳定项 $\epsilon>0$，保证常量输入仍有有限输出。"""
    eps = _finite_scalar("eps", eps)  # 排除 NaN/Inf，避免污染运行统计和反向传播
    if eps <= 0:  # 零方差时仍要求正分母
        raise ValueError(f"eps must be positive, got {eps!r}")
    return eps  # BN 默认 1e-5，RMSNorm 默认 1e-6


def _batch_momentum(momentum: float) -> float:
    r"""运行统计的插值系数 $m\in[0,1]$；$m=0$ 冻结，$m=1$ 完全替换。"""
    momentum = _finite_scalar("momentum", momentum)  # 只检查构造期的标量超参数
    if not 0 <= momentum <= 1:  # EMA 必须是新旧统计的凸组合
        raise ValueError(f"momentum must be in [0, 1], got {momentum!r}")
    return momentum  # 新统计权重为 m，旧统计权重为 1-m


def _check_training(training: bool) -> None:
    r"""要求显式布尔模式，避免设备标量的隐式布尔转换导致同步。"""
    if not isinstance(training, bool):  # 模式属于主机控制流，不是可微张量
        raise TypeError(f"training must be bool, got {type(training).__name__}")


def _check_features(x: torch.Tensor, input_dim: int) -> None:
    r"""单网络线性/RMS 层在最后一维操作，允许任意前导批维。"""
    if x.ndim < 1 or x.shape[-1] != input_dim:  # 线性内积与 RMS 归约共享同一个输入特征轴
        raise ValueError(f"expected input shape [..., {input_dim}], got {tuple(x.shape)}")


def _check_ensemble_features(x: torch.Tensor, num_ensemble: int, input_dim: int) -> None:
    r"""固定 $[Q,B,d]$ 布局，拒绝网络轴广播造成的参数/样本归属歧义。"""
    if x.ndim != 3 or x.shape[0] != num_ensemble or x.shape[2] != input_dim:  # Q 与 d 固定，B 可变
        raise ValueError(f"expected input shape [{num_ensemble}, B, {input_dim}], got {tuple(x.shape)}")


class UnitLinear(nn.Module):
    r"""无偏置线性映射 $y_o=\sum_i w_{oi}x_i$，投影后每个输出行具有单位范数。

    ``w`` 保留为 ``nn.Linear``，权重路径为 ``w.weight``，形状 $[d_{out},d_{in}]$。
    正交初始化的 gain 为 1；当 $d_{out}>d_{in}$ 时，初始化只保证列正交，仍需调用行投影。
    """

    def __init__(self, input_dim: int, output_dim: int):
        r"""构造正整数输入/输出维度的映射，不自动执行参数投影。

        Args:
            input_dim (int): 每个样本的输入特征数 $d_{in}>0$。
            output_dim (int): 输出方向数 $d_{out}>0$，不要求小于输入维度。
        """
        super().__init__()
        _check_dimension("input_dim", input_dim)  # 特征数 d_in > 0
        _check_dimension("output_dim", output_dim)  # 输出数 d_out > 0
        self.w = nn.Linear(input_dim, output_dim, bias=False)  # w.weight: [out, in]，不含平移
        nn.init.orthogonal_(self.w.weight, gain=1)  # 保留上游正交初始化与随机数消耗顺序

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""计算 $xW^\top$，仅在最后一个特征轴上做内积。

        Args:
            x (torch.Tensor): 特征张量 $[...,d_{in}]$，前导维按样本独立处理。

        Returns:
            torch.Tensor: 相同前导维的输出 $[...,d_{out}]$。
        """
        _check_features(x, self.w.in_features)  # 只检查最后一维的元数据
        return self.w(x)  # 仿射偏置由需要它的独立读出层持有

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        r"""原位投影 $w_o\leftarrow w_o/\max(\|w_o\|_2,10^{-8})$，不创建梯度图。

        保持 Parameter 身份及现有 ``.grad``；应在初始化后或优化器更新后调用。
        零行保持为零；范数不小于 $10^{-8}$ 的非零行投影到单位球面。
        """
        self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))  # 输入轴归约，输出行独立


class UnitBatchNorm(nn.Module):
    r"""显式模式的批归一化，仿射参数联合约束为半径 $\sqrt d$。

    对每个特征，训练前向为 $y=\gamma(x-\mu_B)/\sqrt{v_B+\epsilon}+\beta$，
    其中 $v_B=M^{-1}\sum(x-\mu_B)^2$，$M$ 为每通道的样本数。
    运行统计满足 $\bar\mu\leftarrow(1-m)\bar\mu+m\mu_B$、
    $\bar v\leftarrow(1-m)\bar v+m\frac{M}{M-1}v_B$；评估以 $\bar\mu,\bar v$ 替代批统计。
    默认 $m=0.01,\epsilon=10^{-5}$；``weight``、``bias`` 与两项运行统计均为 $[d]$。
    """

    running_mean: torch.Tensor  # 各特征的指数移动均值 [d]
    running_var: torch.Tensor  # 无偏样本方差的指数移动平均 [d]

    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        r"""初始化 $\gamma=1,\beta=0,\bar\mu=0,\bar v=1$。

        Args:
            input_dim (int): 通道数 $d>0$；BN 沿除通道轴外的所有样本轴归约。
            momentum (float): 新批统计权重 $m\in[0,1]$，默认 .01。
            eps (float): 加到前向方差上的正稳定项，默认 $10^{-5}$。
        """
        super().__init__()
        _check_dimension("input_dim", input_dim)  # BN 通道数
        self.momentum = _batch_momentum(momentum)  # m：新批统计的权重
        self.eps = _positive_eps(eps)  # 方差稳定项，不参与运行统计
        self.weight = nn.Parameter(torch.ones(input_dim))  # gamma: [d]
        self.bias = nn.Parameter(torch.zeros(input_dim))  # beta: [d]
        self.register_buffer("running_mean", torch.zeros(input_dim))  # mu_bar: [d]，随 state_dict 保存
        self.register_buffer("running_var", torch.ones(input_dim))  # v_bar: [d]，没有 num_batches_tracked

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        r"""归一化 $[B,d,*S]$，输出同形状；常用 RL 输入为 $[B,d]$。

        ``training=True`` 要求 $M=B\prod S>1$，否则在更新统计前抛出 ValueError；
        ``training=False`` 允许单样本且不更新 buffer。模式完全由本参数指定。

        Args:
            x (torch.Tensor): 通道位于第二轴的浮点特征 $[B,d,*S]$。
            training (bool): True 使用当前批统计，False 使用运行统计。

        Returns:
            torch.Tensor: 应用通道仿射变换后的同形状张量。
        """
        _check_training(training)  # 不读取 self.training
        if x.ndim < 2 or x.shape[1] != self.weight.shape[0]:
            raise ValueError(f"expected input shape [B, {self.weight.shape[0]}, ...], got {tuple(x.shape)}")
        sample_count = x.shape[0] * math.prod(x.shape[2:])  # M：沿所有非通道轴计数
        if training and sample_count <= 1:  # 无偏方差 M/(M-1) 的有效样本数边界
            raise ValueError(f"training batch norm requires more than 1 sample per channel, got {sample_count}")

        # F.batch_norm 负责上游相同的有偏前向方差和无偏运行方差；显式模式覆盖 Module.train/eval。
        return F.batch_norm(
            x,  # [B, d, *S]
            self.running_mean,  # 均值 buffer；只在 training=True 时更新
            self.running_var,  # 无偏方差 buffer
            self.weight,  # gamma: [d]
            self.bias,  # beta: [d]
            training=training,  # 本次前向的统计语义
            momentum=self.momentum,  # 指数更新系数 m
            eps=self.eps,  # 开平方前加入 epsilon
        )

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        r"""联合投影 $(\gamma,\beta)\leftarrow\sqrt{d}(\gamma,\beta)/\sqrt{s+10^{-8}}$。

        $s=\sum_i(\gamma_i^2+\beta_i^2)$，故投影后联合平方范数为 $d\,s/(s+10^{-8})$。
        这不是分别将 scale 与 bias 归一化；不修改运行均值/方差。
        """
        scale, bias = self.weight, self.bias  # 两个 Parameter 共享同一个径向缩放
        ndim = scale.shape[-1]  # d：特征数，不是拼接后的 2d
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)  # s，形状 [1]
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)  # sqrt(d)/sqrt(s+epsilon)
        self.weight.copy_(scale * norm_factor)  # gamma 的方向与符号保留
        self.bias.copy_(bias * norm_factor)  # beta 与 gamma 使用相同因子


class UnitRMSNorm(nn.Module):
    r"""最后一维的均方根归一化 $y_i=\gamma_i x_i/\sqrt{d^{-1}\sum_j x_j^2+\epsilon}$。

    不减均值、不维护运行统计；``weight`` 为 $[d]$，参数投影半径为 $\sqrt d$。
    $\gamma=1$ 时输出范数趋近 $\sqrt d$，有稳定项时略小；非均匀的可学习 $\gamma$ 不保证输出定范数。
    """

    def __init__(self, input_dim: int, eps: float = 1e-6):
        r"""初始化 $\gamma=1$，将特征的共同幅度与可学习方向缩放分开。

        Args:
            input_dim (int): 最后一个轴的特征数 $d>0$。
            eps (float): 加在均方值上的正稳定项，默认 $10^{-6}$。
        """
        super().__init__()
        _check_dimension("input_dim", input_dim)  # 最后一维特征数 d
        self.weight = nn.Parameter(torch.ones(input_dim))  # gamma: [d]
        self.eps = _positive_eps(eps)  # epsilon > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""训练和评估使用相同逐样本公式，不估计批间统计。

        Args:
            x (torch.Tensor): 特征张量 $[...,d]$，前导样本相互独立。

        Returns:
            torch.Tensor: 同形状的 RMS 归一化并经 $\gamma$ 缩放的特征。
        """
        _check_features(x, self.weight.shape[0])  # 保留任意前导样本维
        return F.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)  # 只沿最后的 d 维求均方根

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        r"""原位投影 $\gamma\leftarrow\sqrt d\,\gamma/\sqrt{\sum_i\gamma_i^2+10^{-8}}$。"""
        scale = self.weight  # gamma: [d]，不借助 .data 绕开版本计数
        ndim = scale.shape[-1]  # 目标半径 sqrt(d)
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)  # scale 的平方范数 [1]
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)  # 上游的有限径向缩放
        self.weight.copy_(scale * norm_factor)  # 不重绑 Parameter，不改已有梯度


class FlashSACEmbedder(nn.Module):
    r"""输入映射 $h=W\operatorname{BN}(x)$；先规范输入统计，再投影到隐空间。"""

    def __init__(self, input_dim: int, hidden_dim: int):
        r"""构造 ``norm`` 与 ``w``，映射 $d_{in}\rightarrow d_h$。

        Args:
            input_dim (int): 原始输入特征数 $d_{in}>0$，也是输入 BN 的维度。
            hidden_dim (int): 隐空间特征数 $d_h>0$，也是线性权重的行数。
        """
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)  # 输入空间 BN，参数/buffer 为 [input_dim]
        self.w = UnitLinear(input_dim, hidden_dim)  # w.w.weight: [hidden_dim, input_dim]

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        r"""嵌入前先按当前指定模式规范原始输入的通道统计。

        Args:
            x (torch.Tensor): 原始特征 $[B,d_{in}]$。
            training (bool): 输入 BN 的模式，训练时要求 $B>1$。

        Returns:
            torch.Tensor: 隐空间特征 $[B,d_h]$。
        """
        x = self.norm(x, training=training)  # 同一批内逐特征标准化 [B, input_dim]
        x = self.w(x)  # 无偏置嵌入 [B, hidden_dim]
        return x  # 上游此处没有激活函数


class FlashSACBlock(nn.Module):
    r"""双层残差块 $y=x+\operatorname{ReLU}(\operatorname{BN}_2(W_2\operatorname{ReLU}(\operatorname{BN}_1(W_1x))))$。

    ``expansion=4`` 是上游扩张倍率；两次 ReLU 都在相应 BN 后，第二次 ReLU 在残差相加前。
    """

    def __init__(self, hidden_dim: int, expansion: int = 4):
        r"""通过中间扩张空间构造保留隐维的残差支路。

        Args:
            hidden_dim (int): 残差两端的共享维度 $d_h>0$。
            expansion (int): 整数扩张率 $e>0$，中间宽度 $e d_h$，默认 4。
        """
        super().__init__()
        _check_dimension("hidden_dim", hidden_dim)  # 残差两端共享的隐维
        _check_dimension("expansion", expansion)  # 中间层宽度的整数倍率
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)  # W1: [e*h, h]
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)  # W2: [h, e*h]
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)  # 扩张空间统计
        self.norm2 = UnitBatchNorm(hidden_dim)  # 残差输出空间统计

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        r"""保留非负支路与恒等残差的逐元素相加。

        Args:
            x (torch.Tensor): 残差输入 $[B,d_h]$。
            training (bool): 两个 BN 的共同模式；训练时要求 $B>1$。

        Returns:
            torch.Tensor: 残差输出 $[B,d_h]$，恒等项可保留负值。
        """
        residual = x  # 恒等支路，保留输入及其梯度
        x = self.w1(x)  # [B, h] -> [B, e*h]
        x = self.norm1(x, training=training)  # 扩张空间 BN
        x = F.relu(x)  # 第一次非负截断
        x = self.w2(x)  # [B, e*h] -> [B, h]
        x = self.norm2(x, training=training)  # 收缩空间 BN
        x = F.relu(x)  # 第二次非负截断发生在残差相加之前
        x = x + residual  # y=x_input+f(x_input)，形状 [B, h]
        return x  # 最后没有第三次激活


class EnsembleUnitLinear(nn.Module):
    r"""独立线性映射 $y_{qbo}=\sum_i W_{qoi}x_{qbi}$，权重 ``weight`` 为 $[Q,d_{out},d_{in}]$。

    不在 $Q$ 轴共享或混合参数；每个网络独立正交初始化，投影沿最后的输入特征轴。
    """

    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int):
        r"""按网络索引依次进行 gain=1 的独立正交初始化。

        Args:
            num_ensemble (int): 独立网络数 $Q>0$，双 Q 使用 2。
            input_dim (int): 每组输入特征数 $d_{in}>0$。
            output_dim (int): 每组输出方向数 $d_{out}>0$。
        """
        super().__init__()
        _check_dimension("num_ensemble", num_ensemble)  # 独立网络数 Q
        _check_dimension("input_dim", input_dim)  # 输入维度 d_in
        _check_dimension("output_dim", output_dim)  # 输出维度 d_out
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))  # [Q, out, in]
        for index in range(num_ensemble):
            nn.init.orthogonal_(self.weight[index], gain=1)  # init 本身无梯度，保留上游抽样顺序

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""逐网络进行线性内积，不广播网络轴。

        Args:
            x (torch.Tensor): 具有显式网络轴的特征 $[Q,B,d_{in}]$。

        Returns:
            torch.Tensor: 各网络的独立输出 $[Q,B,d_{out}]$。
        """
        _check_ensemble_features(x, self.weight.shape[0], self.weight.shape[2])  # 布局检查不读取元素
        return torch.einsum("nbi,noi->nbo", x, self.weight)  # Q 组独立的 x_q W_q^T

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        r"""逐 $(q,o)$ 投影 $W_{qo:}/\max(\|W_{qo:}\|_2,10^{-8})$；零行保持为零。"""
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))  # [Q, out, in]，仅输入轴归约


class EnsembleUnitBatchNorm(nn.Module):
    r"""每个网络独立统计的 BN，输入 $[Q,B,d]$，仿射参数和运行统计均为 $[Q,d]$。

    $\mu_{qi}=B^{-1}\sum_bx_{qbi}$，$v_{qi}=B^{-1}\sum_b(x_{qbi}-\mu_{qi})^2$。
    训练前向使用有偏方差 $v$，运行方差更新使用无偏修正 $Bv/(B-1)$。
    $B>1$ 是无偏修正的定义域；评估支持 $B=1$，且不改变运行统计。
    """

    running_mean: torch.Tensor  # 每个网络独立的运行均值 [Q, d]
    running_var: torch.Tensor  # 每个网络独立的无偏运行方差 [Q, d]

    def __init__(self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        r"""初始化每组 $\gamma=1,\beta=0,\bar\mu=0,\bar v=1$。

        Args:
            num_ensemble (int): 独立网络/统计组数 $Q>0$。
            input_dim (int): 每组通道数 $d>0$。
            momentum (float): 运行统计的新批权重 $m\in[0,1]$，默认 .01。
            eps (float): 前向方差的正稳定项，默认 $10^{-5}$。
        """
        super().__init__()
        _check_dimension("num_ensemble", num_ensemble)  # 独立统计组数 Q
        _check_dimension("input_dim", input_dim)  # 每组通道数 d
        self.momentum = _batch_momentum(momentum)  # 新统计权重 m
        self.eps = _positive_eps(eps)  # 方差稳定项 epsilon
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))  # gamma: [Q, d]
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))  # beta: [Q, d]
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))  # mu_bar: [Q, d]
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))  # v_bar: [Q, d]

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        r"""返回同形状 $[Q,B,d]$；显式 ``training`` 控制批统计/运行统计的选择。

        训练更新先将批统计转换为 buffer dtype：默认 FP32 时与上游 ``.float()`` 一致，
        同时支持整个模块转为 FP64 的 CPU 精度验证。批均值/方差仍按输入 dtype 计算。

        Args:
            x (torch.Tensor): 显式网络轴、批轴、特征轴的 $[Q,B,d]$ 输入。
            training (bool): True 沿 B 轴统计并更新 buffer，False 只读运行统计。

        Returns:
            torch.Tensor: 标准化并仿射变换后的同形状特征。
        """
        _check_training(training)  # 避免设备布尔张量引起隐式同步
        _check_ensemble_features(x, self.weight.shape[0], self.weight.shape[1])  # 严格 [Q, B, d]
        if training:  # 样本耦合只存在于训练统计分支
            batch_size = x.shape[1]  # B：只在样本轴上估计方差
            if batch_size <= 1:  # 在任何 EMA 写入前检查 B/(B-1) 的定义域
                raise ValueError(f"training batch norm requires more than 1 sample per channel, got {batch_size}")
            mean = x.mean(dim=1, keepdim=True)  # mu: [Q, 1, d]
            var = x.var(dim=1, correction=0, keepdim=True)  # 有偏 v: [Q, 1, d]

            # EMA 只记录统计事实，不沿历史 batch 反向传播；无偏修正位于 dtype 转换之前。
            with torch.no_grad():
                batch_mean = mean.squeeze(1).to(dtype=self.running_mean.dtype)  # 默认 FP32，[Q, d]
                batch_var = (var.squeeze(1) * (batch_size / (batch_size - 1))).to(  # s²=B*v/(B-1)，形状 [Q,d]
                    dtype=self.running_var.dtype  # 与运行统计同精度，默认 FP32
                )  # 无偏修正先于 dtype 转换，保持上游浮点运算顺序
                self.running_mean.lerp_(batch_mean, self.momentum)  # mu_bar=(1-m)*mu_bar+m*mu
                self.running_var.lerp_(batch_var, self.momentum)  # v_bar=(1-m)*v_bar+m*B*v/(B-1)
            x = (x - mean) * torch.rsqrt(var + self.eps)  # 标准化用有偏方差，形状 [Q, B, d]
        else:  # 评估使用固定运行统计，样本间解耦
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)  # [Q,B,d]
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)  # gamma*x_hat+beta，[Q, B, d]

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        r"""每个网络独立投影 $[\gamma_q,\beta_q]$ 到近似半径 $\sqrt d$。

        $s_q=\sum_i(\gamma_{qi}^2+\beta_{qi}^2)$，缩放因子为 $\sqrt d/\sqrt{s_q+10^{-8}}$。
        网络间不共享范数，运行统计不参与投影。
        """
        scale, bias = self.weight, self.bias  # gamma/beta: [Q, d]
        ndim = scale.shape[-1]  # 每个网络的特征维 d
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)  # s_q: [Q, 1]
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)  # 每组独立径向因子 [Q, 1]
        self.weight.copy_(scale * norm_factor)  # gamma'_q
        self.bias.copy_(bias * norm_factor)  # beta'_q


class EnsembleUnitRMSNorm(nn.Module):
    r"""逐网络、逐样本的 RMSNorm：$y_{qbi}=\gamma_{qi}x_{qbi}/\sqrt{d^{-1}\sum_jx_{qbj}^2+\epsilon}$。

    ``weight`` 为 $[Q,d]$；只沿最后一维归约，不跨网络或样本估计统计。
    """

    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6):
        r"""初始化每组 $\gamma_q=1$，每组参数的初始范数为 $\sqrt d$。

        Args:
            num_ensemble (int): 独立网络数 $Q>0$。
            input_dim (int): 每组特征数 $d>0$，也是 RMS 的归约维度。
            eps (float): 均方根分母中的正稳定项，默认 $10^{-6}$。
        """
        super().__init__()
        _check_dimension("num_ensemble", num_ensemble)  # 网络数 Q
        _check_dimension("input_dim", input_dim)  # 特征数 d
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))  # gamma: [Q, d]
        self.eps = _positive_eps(eps)  # 零输入的有限分母

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""无运行统计，因此不接收 ``training``；样本间与网络间均不耦合。

        Args:
            x (torch.Tensor): 显式网络轴的特征 $[Q,B,d]$。

        Returns:
            torch.Tensor: 同形状的 RMS 归一化并经各组 $\gamma_q$ 缩放的特征。
        """
        _check_ensemble_features(x, self.weight.shape[0], self.weight.shape[1])  # 元数据检查
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)  # RMS: [Q, B, 1]
        return (x / rms) * self.weight.unsqueeze(1)  # gamma 沿样本轴广播，不沿网络轴广播

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        r"""各组 $\gamma_q\leftarrow\sqrt d\,\gamma_q/\sqrt{\sum_i\gamma_{qi}^2+10^{-8}}$。"""
        scale = self.weight  # gamma: [Q, d]
        ndim = scale.shape[-1]  # 每组目标半径 sqrt(d)
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)  # 平方范数 [Q, 1]
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)  # 每个网络独立缩放
        self.weight.copy_(scale * norm_factor)  # 原位修改，保留 Parameter 身份


class EnsembleFlashSACEmbedder(nn.Module):
    r"""并行输入映射 $h_q=W_q\operatorname{BN}_q(x_q)$，不混合双 Q 的统计或参数。"""

    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int):
        r"""创建 ``norm`` 和 ``w``，映射 $[Q,B,d_{in}]\rightarrow[Q,B,d_h]$。

        Args:
            num_ensemble (int): 独立网络数 $Q>0$。
            input_dim (int): 每组原始输入维度 $d_{in}>0$。
            hidden_dim (int): 每组嵌入后的隐维度 $d_h>0$。
        """
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)  # 输入统计 [Q, input_dim]
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)  # w.weight: [Q, hidden, input]

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        r"""先逐网络 BN，再做无偏置线性映射；输出不添加激活。

        Args:
            x (torch.Tensor): 各网络的原始输入 $[Q,B,d_{in}]$。
            training (bool): 输入 BN 的模式，训练时要求 $B>1$。

        Returns:
            torch.Tensor: 嵌入特征 $[Q,B,d_h]$。
        """
        x = self.norm(x, training=training)  # [Q, B, input_dim]
        x = self.w(x)  # [Q, B, hidden_dim]
        return x  # 保留各网络独立的特征通道


class EnsembleFlashSACBlock(nn.Module):
    r"""逐网络执行 FlashSAC 双层残差式，布局始终为 $[Q,B,d]$。

    $y_q=x_q+\operatorname{ReLU}(\operatorname{BN}_{q2}(W_{q2}\operatorname{ReLU}(\operatorname{BN}_{q1}(W_{q1}x_q))))$。
    默认 ``expansion=4``；权重、BN 仿射参数与运行统计都不跨 $Q$ 共享。
    """

    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4):
        r"""构造 $d_h\rightarrow e d_h\rightarrow d_h$ 的独立网络组。

        Args:
            num_ensemble (int): 独立残差网络数 $Q>0$。
            hidden_dim (int): 每组残差两端的维度 $d_h>0$。
            expansion (int): 中间空间的整数扩张率 $e>0$，默认 4。
        """
        super().__init__()
        _check_dimension("hidden_dim", hidden_dim)  # 残差输入输出共享 h
        _check_dimension("expansion", expansion)  # 中间宽度倍率 e
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)  # [Q, e*h, h]
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)  # [Q, h, e*h]
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)  # 扩张空间 BN
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)  # 收缩空间 BN

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        r"""第二次 ReLU 后再加残差，逐网络执行相同的运算顺序。

        Args:
            x (torch.Tensor): 逐网络残差输入 $[Q,B,d_h]$。
            training (bool): 两个 BN 的共同模式，训练时要求 $B>1$。

        Returns:
            torch.Tensor: 同形状的逐网络残差输出。
        """
        residual = x  # 恒等支路，逐网络保留梯度
        x = self.w1(x)  # [Q, B, h] -> [Q, B, e*h]
        x = self.norm1(x, training=training)  # 各网络仅沿 B 归约
        x = F.relu(x)  # 扩张空间的非负激活
        x = self.w2(x)  # [Q, B, e*h] -> [Q, B, h]
        x = self.norm2(x, training=training)  # 隐空间 BN
        x = F.relu(x)  # 残差相加前的第二次激活
        x = x + residual  # 恒等项与非负分支相加
        return x  # [Q, B, h]


class EnsembleCategoricalValue(nn.Module):
    r"""离散价值分布的并行读出；不是策略分布，也不在此构造 Bellman 目标。

    原子 $z_k=v_{min}+k(v_{max}-v_{min})/(K-1)$，$k=0,\ldots,K-1$。
    logits 为 $\ell_{qbk}=\sum_iW_{qki}x_{qbi}+\beta_{qk}$，
    $\log p=\operatorname{logsoftmax}(\ell)$，期望价值为 $q_{qb}=\sum_kp_{qbk}z_k$。
    ``bias`` 为 $[Q,K]$ 且不参与单位投影；``bin_values`` 是 FP32 初始化的 $[1,1,K]$ buffer。
    """

    bin_values: torch.Tensor  # 价值原子支持集 [1, 1, K]，与回报使用相同量纲

    def __init__(self, num_ensemble: int, hidden_dim: int, num_bins: int, min_v: float, max_v: float):
        r"""构造有限端点 $v_{min}<v_{max}$ 的等间距 FP32 支持集。

        Args:
            num_ensemble (int): 价值网络数 $Q>0$。
            hidden_dim (int): 每组输入隐维 $d_h>0$。
            num_bins (int): 价值原子数 $K\ge2$。
            min_v (float): 有限支持下界，与回报同量纲。
            max_v (float): 有限支持上界；端点和跨度必须能用 FP32 表示。
        """
        super().__init__()
        _check_dimension("num_bins", num_bins, minimum=2)  # 至少两个不同位置的价值原子
        min_v = _finite_scalar("min_v", min_v)  # 回报支持下界
        max_v = _finite_scalar("max_v", max_v)  # 回报支持上界
        fp32_max = torch.finfo(torch.float32).max  # 上游以 FP32 构造 linspace
        if not min_v < max_v or max(abs(min_v), abs(max_v), max_v - min_v) > fp32_max:
            raise ValueError("min_v must be < max_v; endpoints and span must be finite and representable in float32")
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)  # w.weight: [Q, K, hidden]
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))  # beta: [Q, K]，无范数约束
        self.register_buffer(
            "bin_values",  # 与上游 state_dict 键一致
            torch.linspace(start=min_v, end=max_v, steps=num_bins, dtype=torch.float32).reshape(1, 1, -1),
        )  # 支持原子 [1, 1, K]，跨网络与样本广播

    def forward(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        r"""输入 $[Q,B,d_h]$，返回 ``(q_values[Q,B], {'log_prob': [Q,B,K]})``。

        ``training`` 保留上游签名；本读出无模式相关统计，两个模式使用相同公式。
        稳定的 log-softmax 同时提供离散分布损失所需的对数概率。

        Args:
            x (torch.Tensor): 价值特征 $[Q,B,d_h]$。
            training (bool): 接口模式标志，本层不据此改变分布计算。

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: 期望 $[Q,B]$ 与 ``log_prob`` 的 $[Q,B,K]$。
        """
        _check_training(training)  # 接口接受主机 bool，本层数值不依赖其值
        value = self.w(x) + self.bias.unsqueeze(1)  # logits: [Q, B, K]
        log_prob = F.log_softmax(value, dim=-1)  # 每个 (q,b) 在 K 个原子上归一化
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)  # E[Z]: [Q, B]
        info: dict[str, torch.Tensor] = {"log_prob": log_prob}  # 完整离散对数概率 [Q, B, K]
        return value, info  # 不在此执行双 Q 最小值或跨网络聚合


@torch.no_grad()
def project_unit_parameters(module: nn.Module) -> None:
    r"""每次调用仅对六类有单位投影语义的基础层各执行一次原位投影。

    ``module.modules()`` 按模块身份去重，因此同一层被多个容器引用仍只投影一次。
    白名单只包含本文件的 UnitLinear/BatchNorm/RMSNorm 及其 ensemble 版本；
    Embedder/Block/价值头作为组合层不递归调用投影，普通 Linear 与自定义同名方法均不触发。
    UnitLinear 虽持有 ``w`` 子模块，但它本身是投影语义上的叶子：内层普通 Linear 不再投影。

    Args:
        module (nn.Module): 待遍历的模型、容器或单个基础层。

    Returns:
        None: 保持参数身份、已有梯度和全部 BN 运行统计；应在初始化/优化器更新后调用。
    """
    unit_layers = (
        UnitLinear,  # 线性输出行单位范数
        UnitBatchNorm,  # scale/bias 联合半径 sqrt(d)
        UnitRMSNorm,  # scale 半径 sqrt(d)
        EnsembleUnitLinear,  # 每个网络独立的输出行单位范数
        EnsembleUnitBatchNorm,  # 每个网络独立的联合仿射半径
        EnsembleUnitRMSNorm,  # 每个网络独立的 scale 半径
    )
    for layer in module.modules():
        if isinstance(layer, unit_layers):
            layer.normalize_parameters()  # 类型白名单，不使用 hasattr 探测任意 Actor 的同名方法
