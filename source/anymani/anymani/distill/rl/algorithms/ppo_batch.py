r"""异构PPO的逐资产估计与分层采样。

本模块只变换已经采集的rollout张量，不读取仿真器、不更新模型，也不持有跨回合状态。资产标签只定义
统计总体和minibatch配额；它不成为actor的数值输入。完整rollout的归一化结果在各mini-epoch间保持固定。
"""

from __future__ import annotations

import torch


def bounded_adaptive_learning_rate(requested_lr: float, reference_lr: float) -> float:
    r"""把rl_games adaptive scheduler限制为只从方法锚点向下调节。

    rl_games会在KL低于阈值一半时每次乘1.5，且默认上限为$10^{-2}$。MVP每update执行16×5个optimizer
    steps，早期zero-init residual产生很小KL，若无此门会在两个updates内把$3\times10^{-4}$推到$10^{-2}$。
    这里保留高KL时降低LR、低KL时恢复LR的机制，但恢复不能越过预先声明的方法锚点。
    """

    if not (requested_lr > 0.0 and reference_lr > 0.0):
        raise ValueError("adaptive and reference learning rates must be positive")
    return min(float(requested_lr), float(reference_lr))


def stratified_asset_permutation(
    prototype_index: torch.Tensor,
    *,
    asset_count: int,
    minibatch_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""构造每个minibatch逐资产等量的sample permutation。

    对资产$i$的样本集合$\mathcal I_i$独立随机排列并均分成$M$份，再令第$m$个minibatch为：

    $$
    \mathcal M_m=\bigcup_{i=0}^{A-1}\mathcal I_i^{(m)}.
    $$

    Args:
        prototype_index (torch.Tensor): selection-local asset index，形状`[B]`或`[B,1]`。
        asset_count (int): 训练支持域资产数$A$；正式MVP固定80。
        minibatch_count (int): 每update minibatch数$M$；正式MVP固定16。
        generator (torch.Generator | None): 可选确定性随机生成器，仅控制组内顺序。

    Returns:
        torch.Tensor: long `[B]`，连续切成$M$段后每段资产计数完全相同。

    Raises:
        ValueError: asset标签缺失、计数不平衡，或单资产计数不能被$M$整除。
    """

    labels = prototype_index.reshape(-1).long()  # `[B]`，asset row只作sampling certificate
    if asset_count < 1 or minibatch_count < 1:
        raise ValueError("asset_count and minibatch_count must be positive")
    if labels.numel() < asset_count or bool(((labels < 0) | (labels >= asset_count)).any().item()):
        raise ValueError("prototype_index contains an invalid or incomplete asset axis")

    # 每个asset必须具有相同rollout cardinality；否则global advantage仍可算，但不满足matched预算。
    counts = torch.bincount(labels, minlength=asset_count)  # `[A]`
    if bool((counts != counts[0]).any().item()) or int(counts[0].item()) % minibatch_count != 0:
        raise ValueError("stratified PPO requires equal per-asset counts divisible by minibatch_count")
    per_asset_per_minibatch = int(counts[0].item()) // minibatch_count  # 2560-env正式$960/16=60$

    # `parts[m][i]`保存第m个minibatch的第i个asset samples；最后按minibatch主序拼接。
    parts: list[list[torch.Tensor]] = [[] for _ in range(minibatch_count)]
    for asset_index in range(asset_count):
        members = torch.nonzero(labels == asset_index, as_tuple=False).squeeze(-1)  # $\mathcal I_i$
        order = torch.randperm(members.numel(), device=members.device, generator=generator)  # 组内随机
        members = members[order]
        for minibatch_index in range(minibatch_count):
            start = minibatch_index * per_asset_per_minibatch  # 当前asset在第m份的起点
            stop = start + per_asset_per_minibatch
            parts[minibatch_index].append(members[start:stop])

    # 每个minibatch内部再随机化asset拼接顺序，避免网络连续看到同一手型样本。
    minibatches: list[torch.Tensor] = []
    for asset_parts in parts:
        indices = torch.cat(asset_parts, dim=0)  # `[B/M]`，每asset严格相同数量
        random_order = torch.randperm(indices.numel(), device=indices.device, generator=generator)
        minibatches.append(indices[random_order])
    return torch.cat(minibatches, dim=0)  # `[B]`，rl_games contiguous slicing直接消费


def normalize_advantages_per_asset(
    advantages: torch.Tensor,
    prototype_index: torch.Tensor,
    *,
    asset_count: int,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""用完整rollout内逐资产moments标准化Actor advantage。

    对资产$i$的rollout样本集合$\mathcal I_i$计算无偏标准差：

    $$
    \hat A_t^{(i)}=\frac{A_t^{(i)}-\mu_i}{\sigma_i+\epsilon},\qquad
    \sigma_i^2=\frac{1}{|\mathcal I_i|-1}\sum_{t\in\mathcal I_i}(A_t^{(i)}-\mu_i)^2.
    $$

    Moments在minibatch permutation前由每资产完整rollout形成，随后五轮PPO复用同一标准化结果。这样不会让每个
    60-sample minibatch独立改变优化目标，也不把asset identity暴露给部署Actor。

    Args:
        advantages (torch.Tensor): 物理GAE或其任意共同正仿射变换，形状``[B]``或``[B,1]``。
        prototype_index (torch.Tensor): selection-local asset标签，形状``[B]``或``[B,1]``。
        asset_count (int): 当前支持资产数$A$。
        epsilon (float): 仅防止零方差除零的正数值项；不作为未经声明的variance floor。

    Returns:
        tuple: 逐样本normalized advantage（原shape）、逐资产mean``[A]``、逐资产sample std``[A]``。

    Raises:
        ValueError: shape/标签非法、任一资产少于两个样本、输入非有限或epsilon非正。
    """

    flat = advantages.reshape(-1)  # `[B]`，保持原device/dtype与sample顺序
    labels = prototype_index.reshape(-1).long()  # `[B]`，只服务训练归约
    if flat.numel() != labels.numel():
        raise ValueError("advantage and prototype labels must align sample-by-sample")
    if asset_count < 1 or epsilon <= 0.0:
        raise ValueError("asset_count and normalization epsilon must be positive")
    if not torch.is_floating_point(flat) or not bool(torch.isfinite(flat).all().item()):
        raise ValueError("advantages must be finite floating-point values")
    if bool(((labels < 0) | (labels >= asset_count)).any().item()):
        raise ValueError("prototype labels lie outside the declared asset axis")

    counts_int = torch.bincount(labels, minlength=asset_count)  # `[A]`，每资产full-rollout样本数
    if bool((counts_int < 2).any().item()):
        raise ValueError("per-asset advantage normalization requires at least two samples per asset")
    counts = counts_int.to(dtype=flat.dtype)
    sums = torch.zeros(asset_count, dtype=flat.dtype, device=flat.device)
    sums.scatter_add_(0, labels, flat)
    means = sums / counts  # `[A]`，每资产rollout GAE均值
    centered = flat - means[labels]  # `[B]`，保持每个sample所属资产对应
    squared_sums = torch.zeros_like(sums)
    squared_sums.scatter_add_(0, labels, centered.square())
    variances = squared_sums / (counts - 1.0)  # 与``torch.std()``默认无偏分母一致
    standard_deviations = torch.sqrt(torch.clamp_min(variances, 0.0))  # 浮点舍入下方差不得出现负值
    normalized = centered / (standard_deviations[labels] + float(epsilon))
    return normalized.reshape_as(advantages), means, standard_deviations
