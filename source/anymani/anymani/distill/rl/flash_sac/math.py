# Adapted source: flash_rl/agents/flashSAC/{update,layer}.py,
# flash_rl/agents/utils/reward_normalization.py.
# Copyright (c) 2026 Holiday Robotics.
# SPDX-License-Identifier: MIT
# 上游许可全文随本包保留于 LICENSE.upstream。
r"""异构手型 Flash SAC 的纯 torch 数学：掩码策略、分布式 TD 与奖励尺度。

符号：$B$ 为样本数，$J$ 为关节槽位数，$Q$ 为 critic 数，$K$ 为价值原子数。
joint_valid=True 表示真实可控关节，False 表示 ghost；槽位数不定义物理自由度。
归一动作 $a\in[-1,1]$ 与 tanh 前潜变量 $z$ 均无量纲；logp 与熵以 nat 计。
reward、support、entropy_cost 的单位必须一致，discount 是无量纲的完整 TD 折扣。

策略保留上游标准 $z=\mu+\sigma\epsilon,\ a=\tanh z$，并显式给出联合 logp
和每有效关节平均 logp。C51 保留上游插值权重；奖励统计保留其 trace/RMS 递推。
所有概率非线性与投影采用 FP32，函数独立于环境注册、Isaac 与训练网络。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _check_tensor(
    value: Tensor, name: str, shape: tuple[int, ...], device: torch.device, *, boolean: bool = False
) -> None:
    r"""检查物理轴与设备一致性，避免广播悄悄改变样本或关节对应关系。"""
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:  # 物理轴必须逐维一致
        raise ValueError(f"{name} shape must be {shape}, got {getattr(value, 'shape', None)}")
    if value.device != device:  # 输入应已由调用方放在相同设备，数学函数不隐式搬运
        raise ValueError(f"{name} device must be {device}, got {value.device}")
    if (boolean and value.dtype != torch.bool) or (not boolean and not value.is_floating_point()):
        raise ValueError(f"{name} must have {'bool' if boolean else 'floating'} dtype, got {value.dtype}")


def _check_finite(value: Tensor, name: str) -> None:
    r"""有限性检查用于已屏蔽的潜参数或具有物理意义的完整张量。"""
    if not bool(torch.isfinite(value).all()):  # ghost 应在调用此检查前替换为有限常量
        raise ValueError(f"{name} must be finite in the computation dtype on all active entries")


def _active_counts(joint_valid: Tensor) -> Tensor:
    r"""返回 $n_b=\sum_j m_{bj}$，形状 [B]；每个样本至少包含一个有效关节。"""
    if not isinstance(joint_valid, Tensor) or joint_valid.ndim != 2 or 0 in joint_valid.shape:
        raise ValueError("joint_valid shape must be nonempty [B,J]")  # 空 batch 或空关节轴不定义本合同
    if joint_valid.dtype != torch.bool:  # 不把任意非零浮点值解释为物理有效性
        raise ValueError("joint_valid must have bool dtype")
    counts = joint_valid.sum(-1)  # [B]，整数物理自由度，与 canonical J 无关
    if bool((counts == 0).any()):  # n_active=0 时联合策略与 per-active 熵均无定义
        raise ValueError("joint_valid must contain at least one active joint per row; all-ghost rows are invalid")
    return counts  # [B]，归约分母由此唯一确定


def _check_log_probs(log_probs: Tensor, name: str) -> None:
    r"""允许 -inf 表示零质量原子，但拒绝 NaN、+inf 和没有任何有限原子的行。"""
    if bool((torch.isnan(log_probs) | torch.isposinf(log_probs)).any()):  # -inf 对 exp 的贡献恰好为零
        raise ValueError(f"{name} may contain -inf for zero mass, but not NaN or +inf")
    if not bool(torch.isfinite(log_probs).any(-1).all()):  # 最后一维是一整条 categorical 分布
        raise ValueError(f"{name} must contain positive probability mass in every row")


def masked_tanh_normal(
    mean: Tensor,
    log_std: Tensor | float,
    joint_valid: Tensor,
    *,
    noise: Tensor | None = None,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""采样有效关节上的标准潜 Gaussian，并返回 tanh 动作与两种 logp 归约。

    $$z_{bj}=\mu_{bj}+e^{\ell_{bj}}\epsilon_{bj},\qquad a_{bj}=\tanh z_{bj}.$$
    $$\log\pi_j(a_j)=-\tfrac12\epsilon_j^2-\ell_j-\tfrac12\log(2\pi)-\log(1-\tanh^2z_j).$$
    $$\log(1-\tanh^2z)=2\{\log2-z-\operatorname{softplus}(-2z)\}.$$
    $$\log\pi_{\rm joint}=\sum_{j:m_j=1}\log\pi_j,\qquad
    \overline{\log\pi}=\log\pi_{\rm joint}/n_{\rm active}.$$

    使用已知标准化噪声计算 Gaussian 二次项，避免小 sigma 的方差平方下溢和
    $(z-\mu)/\sigma$ 的相消。对重参数化样本及其路径梯度，这与 Normal.log_prob 等价。
    先将 ghost 的 mu、log sigma、noise 替换为常量，再做 exp、乘法、平方与 tanh，
    因而污染 ghost 的 NaN/Inf/极值既不改变输出，也不改变有效梯度；ghost 梯度严格为零。

    Args:
        mean: 浮点 [B,J]，无量纲潜均值，可超出 [-1,1]；不是 tanh 后动作均值。
        log_std: Python 标量、零维浮点 Tensor 或严格 [B,J] 的 log sigma；不接受 [J] 广播。
            调用方负责尺度参数化；本函数不裁剪 log sigma，exp(log sigma) 须在 FP32 中有限且正。
        joint_valid: bool [B,J]，每行至少一个真实关节；与 mean 同设备。
        noise: 可选浮点 [B,J] 标准正态噪声。给定时不采新随机数；ghost 值可污染。
        deterministic: True 时取 z=mean、标准化残差为零，不消耗 RNG；若给 noise，仍校验但不使用。

    Returns:
        (actions [B,J], log_prob_joint [B], log_prob_mean [B])，均 FP32、同输入设备。
        ghost 动作精确为 0；确定性模式的 logp 仍是原策略在中心动作处的连续密度。

    Raises:
        ValueError: shape/device/dtype 错误、全 ghost 行、非有限有效输入或 FP32 不可表示的尺度/输出。
    """
    if not isinstance(mean, Tensor) or mean.ndim != 2 or 0 in mean.shape:  # 明确 [B,J] 的两个物理轴
        raise ValueError("mean shape must be nonempty [B,J]")
    shape = tuple(mean.shape)  # (B,J)，所有关节张量使用同一对轴
    _check_tensor(mean, "mean", shape, mean.device)  # 实数浮点潜均值
    _check_tensor(joint_valid, "joint_valid", shape, mean.device, boolean=True)  # bool 物理掩码
    counts = _active_counts(joint_valid)  # [B]，per-active 归约的唯一分母

    # 标量尺度只广播到关节轴，不允许其他形状靠 PyTorch 隐式拼出一个分布。
    if not isinstance(log_std, Tensor):  # Python 标量转同设备 FP32；Tensor 输入保留原有梯度连接
        log_std = torch.as_tensor(log_std, dtype=torch.float32, device=mean.device)  # 标量 log sigma
    if log_std.ndim == 0:  # 零维 Tensor 是全关节共享的可微尺度参数
        _check_tensor(log_std, "log_std", (), mean.device)  # 拒绝跨设备或整数 Tensor
        log_std = log_std.expand(shape)  # [B,J]，共享参数的梯度只累加有效关节项
    _check_tensor(log_std, "log_std", shape, mean.device)  # [J]、[B,1] 等非合同广播均报错
    safe_mean = mean.masked_fill(~joint_valid, 0).float()  # [B,J]，先屏蔽后进入概率非线性
    safe_log_std = log_std.masked_fill(~joint_valid, 0).float()  # ghost 以 log sigma=0 作为有限占位值
    _check_finite(safe_mean, "mean")  # ghost 原值不参与合法性判断
    _check_finite(safe_log_std, "log_std")  # FP64 转 FP32 后的有效值也须可表示

    # 显式噪声先校验和屏蔽；确定性分支绝不调用任何随机数算子。
    supplied_noise = None  # 可选 [B,J] FP32 已净化噪声
    if noise is not None:  # 给定噪声是精确重放路径，包括含污染 ghost 的样本
        _check_tensor(noise, "noise", shape, mean.device)  # 噪声不进行隐式广播
        supplied_noise = noise.masked_fill(~joint_valid, 0).float()  # exp 乘噪声前先隔离 ghost
        _check_finite(supplied_noise, "noise")  # 真实关节的 epsilon 必须有限
    if deterministic:  # epsilon=0 对应 Gaussian 潜中心，不是 tanh 后分布的期望
        standardized = torch.zeros_like(safe_mean)  # [B,J]，无 RNG 消耗
    elif supplied_noise is not None:  # 显式 noise 优先，保持调用方的逐样本重放语义
        standardized = supplied_noise  # [B,J]，保留噪声自身的可微路径
    else:  # 标准随机重参数化路径，每个槽位产生独立 N(0,1) 样本
        standardized = torch.randn_like(safe_mean).masked_fill(~joint_valid, 0)  # ghost 抽样不参与概率

    # 所有非线性都作用于已净化的 FP32 参数；不把零 mask 留到乘出 NaN 之后。
    std = safe_log_std.exp()  # [B,J]，无量纲潜标准差 sigma
    if not bool((torch.isfinite(std) & (std > 0)).all()):  # 不静默裁剪用户给出的真实 Gaussian
        raise ValueError("log_std must define a finite positive exp(log_std) in float32 on active entries")
    latent = safe_mean + std * standardized  # [B,J]，z = mu + sigma * epsilon
    log_gaussian = -0.5 * standardized.square() - safe_log_std - 0.5 * math.log(2 * math.pi)  # nat / joint
    log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))  # 稳定 log(1-tanh(z)^2)
    log_prob = (log_gaussian - log_jacobian).masked_fill(~joint_valid, 0)  # [B,J]，ghost 无概率贡献
    actions = latent.tanh().masked_fill(~joint_valid, 0)  # [B,J]，归一动作，ghost 精确零
    joint = log_prob.sum(-1)  # [B]，nat / hand，独立有效关节的联合 log 密度
    average = joint / counts  # [B]，nat / active joint，SAC 温度与目标熵须选同一归约
    _check_finite(actions, "actions")  # 不返回 FP32 溢出造成的非法动作
    _check_finite(joint, "log_prob_joint")  # 包括潜值极端到 log Jacobian 无法表示的情形
    return actions, joint, average  # 三者均为 FP32；保留 actor 重参数化梯度


def masked_target_entropy(joint_valid: Tensor, target_sigma: float = 0.15, *, reduction: str = "mean") -> Tensor:
    r"""由归一动作潜尺度定义 Gaussian 等效目标熵，单位为 nat。

    $$h(\sigma_*)=\tfrac12\log(2\pi e\sigma_*^2),\quad
    H_b^{\rm sum}=n_bh,\quad H_b^{\rm mean}=h.$$
    preset: target_sigma=0.15。此数值指定 tanh 前无量纲潜尺度，并不等于真实
    tanh 动作的标准差或熵；后者还依赖潜均值和 Jacobian 的分布期望。

    Args:
        joint_valid: bool [B,J]，有效关节掩码，禁止全 ghost 行。
        target_sigma: 有限正标量；归一动作潜尺度，不是弧度或物理关节位移。
        reduction: 'mean' 对应每有效关节平均熵，'sum' 对应整手联合熵。

    Returns:
        FP32 [B]，同 joint_valid 设备；ghost 不进入任何熵系数或分母。

    Raises:
        ValueError: 掩码不合法、sigma 非有限/非正或归约名称不在上述两者中。
    """
    counts = _active_counts(joint_valid).float()  # [B]，实际 DoF，而非 canonical 存储槽位数
    if not math.isfinite(target_sigma) or target_sigma <= 0:  # Gaussian 微分熵仅对正尺度有定义
        raise ValueError("target_sigma must be finite and positive")
    if reduction not in ("mean", "sum"):  # 概率和目标熵必须显式选用同一种归约
        raise ValueError("reduction must be 'mean' or 'sum'")
    entropy = math.log(target_sigma) + 0.5 * math.log(2 * math.pi * math.e)  # 避免 sigma**2 上/下溢
    per_joint = torch.full_like(counts, entropy)  # [B]，nat / active joint，不依赖 mu
    return counts * per_joint if reduction == "sum" else per_joint  # 整手熵仅随真实关节数线性变化


def select_min_q_log_probs(q_values: Tensor, log_probs: Tensor) -> Tensor:
    r"""按每个样本的最小期望 Q 选择一个 critic 的完整离散分布。

    $$q_b^*=\arg\min_q\mathbb E[Z_{qb}],\qquad L_{bk}=L_{q_b^*,b,k}.$$
    并列时选择索引最小的 critic，与 torch.argmin 和上游 gather 约定一致。
    选择索引不可微；被选 log_probs 的整条分布仍保留梯度，TD 目标的断梯度由投影负责。

    Args:
        q_values: 浮点 [Q,B]，同一价值单位下各 critic 的期望 Q。
        log_probs: 浮点 [Q,B,K]，与 q_values 对应的归一化 log 概率，允许 -inf 零原子。

    Returns:
        [B,K]，保持 log_probs 的 dtype/device；不是逐 atom 求最小值。

    Raises:
        ValueError: 空轴、shape/device/dtype 错误或非法概率/非有限 Q。
    """
    if not isinstance(q_values, Tensor) or q_values.ndim != 2 or 0 in q_values.shape:  # critic 与样本双轴
        raise ValueError("q_values shape must be nonempty [Q,B]")
    if not isinstance(log_probs, Tensor) or log_probs.ndim != 3 or log_probs.shape[-1] == 0:
        raise ValueError("log_probs shape must be nonempty [Q,B,K]")  # 一整条 K 原子分布
    q_count, batch = q_values.shape  # Q 任意正整数，B=1 仍保留样本轴
    bins = log_probs.shape[-1]  # K 个价值原子
    _check_tensor(q_values, "q_values", (q_count, batch), q_values.device)  # 实数期望价值
    _check_tensor(log_probs, "log_probs", (q_count, batch, bins), q_values.device)  # critic/sample 严格对齐
    _check_finite(q_values, "q_values")  # 非有限 Q 无法给出可信的最小值排序
    _check_log_probs(log_probs, "log_probs")  # 零质量原子可使用 -inf
    minimum = q_values.argmin(dim=0)  # [B]，每个样本仅选一个 critic
    indices = minimum[None, :, None].expand(1, batch, bins)  # [1,B,K]，同一样本全部 atom 使用同一索引
    return log_probs.gather(0, indices)[0]  # [B,K]，只去掉 critic 轴，保留 singleton batch


@torch.no_grad()
def project_categorical(
    next_log_probs: Tensor,
    rewards: Tensor,
    discounts: Tensor,
    entropy_cost: Tensor,
    support: Tensor,
) -> Tensor:
    r"""将 soft Bellman 目标原子线性投影到均匀递增价值支撑，保持输入概率质量。

    $$t_{bk}=\operatorname{clip}\{r_b+d_b(z_k-c_b),z_0,z_{K-1}\},\qquad
    c_b=\alpha\overline{\log\pi(a'_b|s'_b)}.$$
    $$u=(t-z_0)/\Delta z,\quad l=\lfloor u\rfloor,\quad h=\min(l+1,K-1),\quad f=u-l.$$
    每个 source atom 的质量 p 分配为 $m_l\mathrel{+}=p(1-f)$、$m_h\mathrel{+}=pf$。
    exact-bin 的 f=0 会完整写入 l；最右端即使 l=h 也保持总质量。
    负 entropy_cost 对应正 soft-value 奖励；discount 同时作用于价值与熵成本。

    Args:
        next_log_probs: 浮点 [B,K]，输入为归一化 log 概率，允许 -inf 表示零质量。
            按上游使用 exp 而非再次 softmax：投影线性保留输入质量，低精度量化误差也保留。
        rewards: 浮点 [B]，已形成的单步或 n 步奖励；单位同 support。
        discounts: 浮点 [B]，已含 gamma**n 与正确的学习终止 mask，范围 [0,1]。
        entropy_cost: 浮点 [B]，alpha * 有效关节归一 logp，单位同价值；不是熵的正值。
        support: 浮点 [K]，K>=2、有限均匀严格递增。FP32 间隔误差容限为支撑幅度的 4 ulp。

    Returns:
        FP32 [B,K]，同输入设备、无反向图的目标概率；超界原子压到端点。

    Raises:
        ValueError: shape/device/dtype、概率、有限性、折扣或均匀支撑条件不满足。
    """
    if not isinstance(next_log_probs, Tensor) or next_log_probs.ndim != 2 or 0 in next_log_probs.shape:
        raise ValueError("next_log_probs shape must be nonempty [B,K]")  # 禁止任何 sample 轴广播
    batch, bins = next_log_probs.shape  # B 个独立 TD 目标，每条含 K 个原子
    device = next_log_probs.device  # 数学函数只在输入设备上运算
    _check_tensor(next_log_probs, "next_log_probs", (batch, bins), device)  # 实数 log 密度
    if bins < 2:  # 单原子支撑不具有 C51 插值间隔
        raise ValueError("support requires at least two atoms")
    _check_tensor(support, "support", (bins,), device)  # 支撑原子和 critic 的 K 轴严格一致
    for name, value in (("rewards", rewards), ("discounts", discounts), ("entropy_cost", entropy_cost)):
        _check_tensor(value, name, (batch,), device)  # 每个样本一个物理标量
        _check_finite(value.float(), name)  # 进入 FP32 后仍需有限
    grid = support.float()  # [K]，价值支撑升为 FP32
    _check_finite(grid, "support")  # 非有限支撑无法定位 bin
    width = (grid[-1] - grid[0]) / (bins - 1)  # 标量 Delta z，价值单位
    if not bool(torch.isfinite(width) & (width > 0)) or not bool((grid[1:] > grid[:-1]).all()):
        raise ValueError("support must be finite and strictly increasing in float32")  # 也拒绝量化后的重复 bin

    # 用端点构造均匀参考网格；容限只吸收 FP32 表示误差，不允许非均匀价值量化。
    uniform = torch.linspace(grid[0], grid[-1], bins, device=device, dtype=torch.float32)  # [K] 理想支撑
    tolerance = 4 * torch.finfo(torch.float32).eps * torch.maximum(grid.abs().max(), width)  # 约 4 ulp
    if not bool(((grid - uniform).abs() <= tolerance).all()):  # 绝对误差随支撑物理尺度缩放
        raise ValueError("support must be uniformly spaced in float32")
    discount = discounts.float()  # [B]，不再乘 gamma 或额外 done
    if bool(((discount < 0) | (discount > 1)).any()):  # gamma**n * learning_mask 的物理范围
        raise ValueError("discounts must lie in [0,1]")
    logs = next_log_probs.float()  # 小质量先升精度再 exp/scatter，避免半精度累积损失
    _check_log_probs(logs, "next_log_probs")  # -inf 合法，全部 -inf 的行不合法
    probabilities = logs.exp()  # [B,K]，严格保留上游 exp(logp) 的质量定义
    _check_finite(probabilities, "next_log_probs probabilities")  # 防止把非法大正 log 值变成无穷质量
    if bool((probabilities.sum(-1) <= 0).any()):  # FP32 下全部质量下溢也无法表示一个分布
        raise ValueError("next_log_probs must retain positive mass in float32")

    # 终止行的熵先清零，确保 d=0 时不因中间 support-cost 的溢出出现 0*inf。
    cost = entropy_cost.float().masked_fill(discount == 0, 0)  # [B]，终止状态不使用未来熵
    atoms = rewards.float()[:, None] + discount[:, None] * (grid[None, :] - cost[:, None])  # [B,K]
    atoms = atoms.clamp(grid[0], grid[-1])  # 价值越界质量移至最近支撑端点
    positions = ((atoms - grid[0]) / width).clamp(0, bins - 1)  # [B,K]，无量纲 bin 坐标
    lower = positions.floor().long()  # [B,K]，左端索引
    upper = (lower + 1).clamp_max(bins - 1)  # exact-bin 也有左端权重 1，避免 floor=ceil 丢质量
    fraction = positions - lower.float()  # [B,K]，右端权重 f∈[0,1)
    projected = torch.zeros_like(probabilities)  # [B,K]，FP32 目标质量缓冲
    projected.scatter_add_(1, lower, probabilities * (1 - fraction))  # 左端质量 p(1-f)
    projected.scatter_add_(1, upper, probabilities * fraction)  # 右端质量 pf，两者之和等于 p
    return projected  # no_grad：不保留 next critic、reward、discount 或 alpha 的目标梯度


class DiscountedRewardNormalizer(nn.Module):
    r"""使用上游折扣统计 trace 的全局方差和历史绝对界，缩放即时或回放奖励。

    $$G_{t,e}=\gamma(1-(\mathrm{terminated}_{t,e}\lor\mathrm{truncated}_{t,e}))G_{t-1,e}+r_{t,e}.$$
    $$s=\max\{\sqrt{v+\epsilon},\ M/\mathrm{max\_return}\},\quad \widetilde r=r/s.$$
    G 是上游的**统计 trace 约定**：done 当步先舍弃旧 trace，仍加入当步 reward，
    下一非 done 步会继续衰减这一 trace。它不是供 TD 使用的跨 episode 返回值。
    学习目标是否 bootstrap、timeout 如何处理，应由外部正确形成 discounts 的 mask。

    每个真实向量环境步恰调一次 observe；每次纳入 N=num_envs 个 trace 样本，
    global mean/var/count 跨环境与历史累积。scale 可任意次读取 replay，不改变统计。
    preset: gamma=0.99、max_return=5、尺度 epsilon=1e-8；上游 RMS epsilon 独立固定为 1e-4。
    上游 count 从 0、var 从 1 开始，每次合并均使用 var*(count+1e-4)，不是 epsilon 伪样本计数。

    状态以 buffers 注册：returns [N]；mean、var、count、max_abs 和三个尺度配置均为标量。
    默认 FP32，随 Module.to(device) 搬运；无可训练参数。标准 state_dict/load_state_dict
    可恢复全部数值状态与配置，要求 num_envs 一致并维持 returns 的环境索引对应关系。
    """

    returns: Tensor  # [N]，每个环境的上游统计 trace，单位同奖励
    mean: Tensor  # 标量 global mean，奖励单位；仅作统计，scale 不减此值
    var: Tensor  # 标量 global population variance，奖励单位平方，含上游 epsilon 修正
    count: Tensor  # 标量样本数，每次 observe 增加 N；默认 FP32，与上游一致
    max_abs: Tensor  # 标量历史 max |G|，奖励单位，单调不减
    gamma: Tensor  # 标量无量纲统计折扣，checkpoint 保存
    max_return: Tensor  # 标量归一 trace 的历史幅值尺度，默认 5
    epsilon: Tensor  # 标量方差分母稳定项，数值单位同 var

    def __init__(
        self,
        num_envs: int,
        gamma: float = 0.99,
        max_return: float = 5.0,
        device: torch.device | str = "cpu",
        epsilon: float = 1e-8,
    ) -> None:
        r"""按固定环境数建立 trace 与全局统计，配置和统计共同进入 state_dict。

        Args:
            num_envs: 正整数，真实并行环境数，决定 returns [N] 的长度。
            gamma: [0,1] 的有限统计折扣；不隐式参与外部 TD target。
            max_return: 有限正标量，分母的历史界项为 max_abs/max_return。
            device: 全部 buffers 的初始设备，默认 CPU。
            epsilon: 有限正标量，只加在 sqrt(var+epsilon) 内，默认 1e-8。

        Raises:
            ValueError: 环境数或尺度配置不满足定义域。
        """
        super().__init__()  # 标准 nn.Module 负责 buffer 搬运与状态恢复
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")  # 固定 N 条独立环境 trace
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:  # 折扣 trace 的定义域
            raise ValueError("gamma must be finite and in [0,1]")
        for name, value in (("max_return", max_return), ("epsilon", epsilon)):
            if not math.isfinite(value) or value <= 0:  # 正分母与稳定项是缩放定义的一部分
                raise ValueError(f"{name} must be finite and positive")
        self.num_envs = num_envs  # N 是环境索引合同，load 时由 returns shape 强制一致
        self.register_buffer("returns", torch.zeros(num_envs, dtype=torch.float32, device=device))  # [N]，G 初值 0

        # 初始 mean=0、var=1、count=0 完整保留上游 RMS；配置也纳入标准 checkpoint。
        for name, value in (
            ("mean", 0.0),  # global mean 初值
            ("var", 1.0),  # global var 初值，对首批 M2 提供 1e-4
            ("count", 0.0),  # 真实样本计数，不把 RMS epsilon 当伪样本
            ("max_abs", 0.0),  # 历史绝对 trace 上界
            ("gamma", gamma),  # 统计折扣配置
            ("max_return", max_return),  # 历史界尺度配置
            ("epsilon", epsilon),  # sqrt 内方差稳定项
        ):
            tensor = torch.tensor(value, dtype=torch.float32, device=device)  # 同设备标量 buffer
            if not bool(torch.isfinite(tensor)) or (name in ("max_return", "epsilon") and not bool(tensor > 0)):
                raise ValueError(f"{name} must remain finite and valid in float32")  # 拒绝转换后的溢出/下溢
            self.register_buffer(name, tensor)  # 不需要自定义 save/load，也不注册 Parameter

    @torch.no_grad()
    def observe(self, reward: Tensor, terminated: Tensor, truncated: Tensor) -> None:
        r"""在一个真实环境步之后，纳入 N 个新 trace 样本并更新 global moments。

        $$\delta=\mu_b-\mu,\quad C'=C+N,\quad w=N/C',\quad \mu'=\mu+\delta w.$$
        $$M_2=v(C+10^{-4})+Nv_b+\delta^2 Cw,\qquad v'=M_2/C'.$$
        使用总体方差 v_b（correction=0），故 N=1 的批方差为 0 而不是 NaN。
        所有新量先由同一旧状态计算，然后写回；梯度不穿过环境观测或统计时间轴。

        Args:
            reward: 有限浮点 [N]，本步原始 reward，单位同 G；与 buffers 同设备。
            terminated: bool [N]，本步环境终止事实。
            truncated: bool [N]，本步环境截断事实；统计 reset 使用两者逻辑或。
        """
        shape = (self.num_envs,)  # N 固定，不能把 replay batch 当作真实环境轴
        _check_tensor(reward, "reward", shape, self.returns.device)  # 每环境恰有一个本步奖励
        _check_tensor(terminated, "terminated", shape, self.returns.device, boolean=True)  # 终止事实
        _check_tensor(truncated, "truncated", shape, self.returns.device, boolean=True)  # 截断事实
        reward = reward.to(dtype=self.returns.dtype)  # 默认 FP32；遵循 Module.to(dtype) 的显式迁移
        _check_finite(reward, "reward")  # 校验完成前不修改任何统计
        done = terminated | truncated  # [N]，只用于上游统计 trace reset
        trace = self.gamma * self.returns.masked_fill(done, 0) + reward  # G_new=gamma*(1-done)*G_old+r
        _check_finite(trace, "reward trace")  # 拒绝溢出，以免污染全部后续 global moments

        # 平行方差合并：每步将 N 条新 trace 作为一批，累积所有环境和全部历史。
        batch_mean = trace.mean()  # 标量 mu_b，奖励单位
        batch_var = trace.var(correction=0)  # 标量 v_b，奖励单位平方；N=1 时为零
        delta = batch_mean - self.mean  # 新旧均值差，奖励单位
        total = self.count + self.num_envs  # 标量 C'=C+N，真实样本累计数
        ratio = self.num_envs / total  # 无量纲新批权重 w
        new_mean = self.mean + delta * ratio  # 合并均值 mu'
        old_m2 = self.var * (self.count + 1e-4)  # 上游 RMS epsilon 每次加在旧 M2 的计数因子中
        batch_m2 = batch_var * self.num_envs  # 新批总体中心平方和 Nv_b
        new_var = (old_m2 + batch_m2 + delta.square() * self.count * ratio) / total  # 合并总体方差 v'
        maximum = torch.maximum(self.max_abs, trace.abs().max())  # 历史 max |G|，reset 不降低上界
        _check_finite(new_mean, "reward mean")  # 有效输入仍可能超出矩统计可表示范围
        _check_finite(new_var, "reward variance")  # 方差失效时不提交部分状态

        # 一次真实步的全部统计同时提交；copy_ 保留注册 buffer 对象及标准 state_dict 路径。
        self.returns.copy_(trace)  # [N] 下一步 G_old
        self.mean.copy_(new_mean)  # 标量 global mean
        self.var.copy_(new_var)  # 标量 global variance
        self.count.copy_(total)  # 标量 count，恰增加 N
        self.max_abs.copy_(maximum)  # 标量历史绝对界

    def scale(self, rewards: Tensor) -> Tensor:
        r"""只读缩放任意形状的 reward，不减 global mean，也不推进统计。

        $$\widetilde r=r/\max\{\sqrt{v+\epsilon},\ M/\mathrm{max\_return}\}.$$
        分母是全局标量，避免对不同手型或 replay minibatch 隐式重新估计奖励单位。

        Args:
            rewards: 任意形状的有限浮点奖励，同 buffers 设备；可来自当前步或 replay。

        Returns:
            同 shape、统计 buffer dtype（默认 FP32）的缩放奖励；保留 rewards 自身的梯度。
        """
        if not isinstance(rewards, Tensor):  # 形状任意，但设备和数值合同仍与统计相同
            raise ValueError("rewards must be a floating Tensor")
        _check_tensor(rewards, "rewards", tuple(rewards.shape), self.returns.device)  # 无环境数限制
        values = rewards.to(dtype=self.var.dtype)  # 使用统计精度计算尺度
        _check_finite(values, "rewards")  # 非有限 reward 不是可缩放的训练样本
        standard_deviation = torch.sqrt(self.var + self.epsilon)  # sqrt(v+eps)，奖励单位
        historical_bound = self.max_abs / self.max_return  # 历史幅值约束给出的最小分母
        denominator = torch.maximum(standard_deviation, historical_bound)  # 一个 global 奖励尺度
        return values / denominator  # 不中心化、不改 trace/moments/count/max_abs
