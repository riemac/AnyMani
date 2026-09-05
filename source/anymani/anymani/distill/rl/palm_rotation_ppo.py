r"""MVP80 structured actor/privileged critic的rl_games custom PPO边界。

本模块保持三项训练语义：

1. actor与critic读取同一份rollout-cached FP32 $Z^e$，但参数与optimizers完全分离；
2. canonical ghost joints不进入Normal log-prob、entropy、KL、bounds或动作执行；
3. 每个minibatch对80个assets严格等量，正式$B=76800,M=16$时每个minibatch含每资产60项。

N040不属于本network module，因此5个mini-epochs不会触发encoder forward，也不会把冻结encoder写入
actor/critic optimizer。Actor optimizer内部保留base与global-residual两个参数组，其学习率始终维持
$3\!:\!1$；critic使用独立optimizer和$5\times10^{-4}$初始学习率。
"""

from __future__ import annotations

import json
import math
import os
import random
import resource
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.common import common_losses
from rl_games.common.diagnostics import PpoDiagnostics
from torch import nn
from torch.nn.utils import clip_grad_norm_

from anymani.distill.diagnostics.recording.rl.palm_rotation import PalmRotationMetricsRecorder
from anymani.distill.models.palm_rotation_policy import (
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationCriticObservation,
    PalmRotationGeometry,
)
from anymani.distill.rl.algorithms.gradient_audit import (
    compute_actor_gradient_scope_audit,
    per_asset_replica_half_gradients,
)
from anymani.tasks.hetero.mdp.curriculum_state import (
    HETERO_REWARD_RELEASE_STATE_ATTR,
    HeterogeneousRewardReleaseState,
)

from .masked_ppo import (
    AnyManiMaskedContinuousModel,
    AnyManiMaskedPpoAgent,
    AnyManiMaskedPpoPlayer,
    AnyManiMaskedRunner,
    register_anymani_masked_ppo,
)
from .runtime.palm_rotation_vecenv import (
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_FLOAT_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
)
from .runtime.palm_rotation_warm_start import load_actor_init_checkpoint, should_load_actor_init_checkpoint

PALM_ROTATION_PPO_ALGO = "anymani_palm_rotation_ppo"
PALM_ROTATION_NETWORK = "anymani_palm_rotation"
CRITIC_OPTIMIZER_KEY = "anymani_critic_optimizer"
DIAGNOSTICS_RECORDER_KEY = "anymani_metrics_recorder"
TRAINING_CONTINUATION_KEY = "anymani_training_continuation"


def bounded_adaptive_learning_rate(requested_lr: float, reference_lr: float) -> float:
    r"""把rl_games adaptive scheduler限制为只从方法锚点向下调节。

    rl_games会在KL低于阈值一半时每次乘1.5，且默认上限为$10^{-2}$。MVP每update执行16×5个optimizer
    steps，早期zero-init residual产生很小KL，若无此门会在两个updates内把$3\times10^{-4}$推到$10^{-2}$。
    这里保留高KL时降低LR、低KL时恢复LR的机制，但恢复不能越过预先声明的方法锚点。
    """

    if not (requested_lr > 0.0 and reference_lr > 0.0):
        raise ValueError("adaptive and reference learning rates must be positive")
    return min(float(requested_lr), float(reference_lr))


def validate_gradient_probe_compile_compatibility(
    compile_mode: str | None,
    probe_frequency: int,
    full_gradient_shadow_frequency: int = 0,
) -> None:
    r"""拒绝Inductor donated-buffer与多次head-gradient反向的不兼容组合。

    当前gradient probe对同一forward graph按资产执行多次``autograd.grad(retain_graph=True)``；PyTorch
    AOTAutograd在compiled backward启用non-empty donated buffers时要求单次``retain_graph=False``。Eager路径已通过
    真实16资产probe；compile仍可用于probe关闭的纯性能/训练run。该门在scene创建前由launcher调用。
    """

    if probe_frequency < 0 or full_gradient_shadow_frequency < 0:
        raise ValueError("gradient probe frequencies must be non-negative")
    if compile_mode is not None and (probe_frequency > 0 or full_gradient_shadow_frequency > 0):
        raise ValueError("head-gradient probe requires eager actor/critic forward; disable --torch_compile or the probe")


def denormalize_value_readonly(model: Any, value: torch.Tensor) -> torch.Tensor:
    r"""在不更新value running moments的条件下恢复物理尺度预测。

    ``rl_games.BaseModelNetwork.denorm_value``内部调用同一个``RunningMeanStd.forward``；该module若处于
    train mode，会在执行``denorm=True``前先把输入写入running mean/variance/count。PPO诊断输入是normalized
    Critic output，不属于value-target统计总体，因此这里仅临时切换value normalizer本身，调用后恢复原生命周期。

    Args:
        model (Any): rl_games model facade，需暴露``normalize_value``、``value_mean_std``与``denorm_value``。
        value (torch.Tensor): normalized Critic prediction，形状``[M,1]``。

    Returns:
        torch.Tensor: 采用调用前冻结moments恢复的物理value，形状与输入一致。
    """

    if not bool(getattr(model, "normalize_value", False)):
        return value  # 未启用value normalization时物理空间与model输出空间相同
    normalizer = getattr(model, "value_mean_std", None)
    if not isinstance(normalizer, nn.Module):
        raise TypeError("normalized value model must expose an nn.Module value_mean_std")
    was_training = bool(normalizer.training)  # 只保存目标normalizer模式，不改变Actor/Critic train状态
    normalizer.eval()  # ``RunningMeanStd.forward``在eval mode只读已冻结moments
    try:
        return model.denorm_value(value)  # upstream保留clamp与epsilon的唯一反归一化实现
    finally:
        normalizer.train(was_training)  # 异常路径也恢复下一次合法return/value统计更新所需模式


def _linux_memory_snapshot() -> dict[str, int]:
    r"""读取当前训练进程RSS/swap、峰值RSS和系统可用内存，单位byte。

    Isaac Sim、PhysX、N040与PPO位于同一Python进程，因此``/proc/self``覆盖本run主要host分配。GPU峰值由
    CUDA allocator独立记录；两者不能互相替代。
    """

    status: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmSwap:")):
            name, value, _unit = line.split()
            status[name.rstrip(":")] = int(value) * 1024  # Linux status以KiB报告
    available = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024  # 系统可回收后available bytes
            break
    return {
        "process_rss_bytes": status.get("VmRSS", 0),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "process_swap_bytes": status.get("VmSwap", 0),
        "system_available_memory_bytes": available,
    }


class PalmRotationPpoDiagnostics(PpoDiagnostics):
    r"""在GPU上累计rl_games PPO诊断，只在epoch发布边界传回CPU。

    Upstream :class:`PpoDiagnostics` 在每个activation microbatch把explained variance与clip fraction
    ``.cpu()``，正式MVP每update共有$16\times5=80$个microbatches，因而产生160次CUDA stream
    synchronization。这里保持完全相同的统计公式与mini-epoch/epoch归约，只延迟device transfer；该改动不
    改变loss、gradient或optimizer step。
    """

    def mini_batch(self, agent: Any, batch: Mapping[str, Any], e_clip: float, minibatch: int) -> None:
        r"""记录一个microbatch的device-resident explained variance与clip fraction。"""

        _ = (agent, minibatch)  # 统计只由当前batch与PPO clip阈值决定
        with torch.no_grad():
            values = batch["values"].detach()  # rollout value，形状`[M,1]`
            returns = batch["returns"].detach()  # GAE return target，形状`[M,1]`
            new_neglogp = batch["new_neglogp"].detach()  # 当前策略negative log-prob，形状`[M]`
            old_neglogp = batch["old_neglogp"].detach()  # rollout策略negative log-prob，形状`[M]`
            masks = batch["masks"]  # feed-forward MVP固定为None
            exp_var = torch_ext.explained_variance(values, returns, masks)  # scalar，保留当前CUDA device
            clip_frac = torch_ext.policy_clip_fraction(new_neglogp, old_neglogp, e_clip, masks)  # scalar
            self.exp_vars.append(exp_var.detach())  # epoch边界一次性归约/传输，不在hot loop同步
            self.clip_fracs.append(clip_frac.detach())  # mini-epoch边界仍使用upstream相同mean


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


def rollout_policy_mechanism_metrics(
    rollout_mean: torch.Tensor,
    mechanism: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    actor_arm: Literal["base", "residual", "direct", "direct_token"],
) -> dict[str, torch.Tensor]:
    r"""从不可变rollout策略均值计算动作分解诊断。

    ``rl_games.PPODataset.update_mu_sigma``会在每个minibatch后原地更新dataset中的``mu/sigma``，使其成为
    后续mini-epoch的KL参考分布。该可变参考不再等于采样轨迹时的策略均值，因此Direct恒等式
    $\mu^{direct}=\mu^{rollout}$以及Residual分解
    $\mu^{base}=\mu^{rollout}-\Delta\mu^{global}$必须使用独立存储的``rollout_mean``。

    Args:
        rollout_mean (torch.Tensor): 采样当前rollout时冻结的有界策略均值，形状``[M,J]``。
        mechanism (torch.Tensor): Direct完整均值或Residual/BASE的global residual，形状``[M,J]``。
        active_mask (torch.Tensor): active-joint布尔掩码，形状``[M,J]``。
        actor_arm (Literal): 两种Direct、``residual``或仅保留base的``base``机制。

    Returns:
        dict[str, torch.Tensor]: 每样本动作均值及arm-specific分解统计，所有值形状均为``[M]``。

    Raises:
        RuntimeError: 三个joint-axis张量shape不一致，或Direct side-channel不等于冻结rollout mean。
        ValueError: Actor arm不属于声明集合。
    """

    if rollout_mean.ndim != 2 or mechanism.shape != rollout_mean.shape or active_mask.shape != rollout_mean.shape:
        raise RuntimeError(
            "rollout mechanism tensors must share [M,J] shape: "
            f"mean={tuple(rollout_mean.shape)}, mechanism={tuple(mechanism.shape)}, mask={tuple(active_mask.shape)}"
        )
    if actor_arm not in {"base", "residual", "direct", "direct_token"}:
        raise ValueError(f"unsupported rollout mechanism arm: {actor_arm!r}")

    active_float = active_mask.to(dtype=rollout_mean.dtype)  # `[M,J]`，ghost不进入动作幅度统计
    active_count = active_float.sum(dim=-1).clamp_min(1.0)  # `[M]`，每样本active DoF分母
    policy_mean_rms = torch.sqrt((rollout_mean.square() * active_float).sum(dim=-1) / active_count)
    near_bound = ((rollout_mean.abs() >= 0.95).to(dtype=rollout_mean.dtype) * active_float).sum(dim=-1) / active_count
    metrics = {
        "policy_mean_rms": policy_mean_rms,  # $\sqrt{J^{-1}\sum_j(\mu_j^{rollout})^2}$
        "policy_mean_near_bound_fraction": near_bound,  # $|\mu_j^{rollout}|\ge0.95$的active-joint比例
    }

    if actor_arm in {"direct", "direct_token"}:
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(mechanism == rollout_mean),
            "direct rollout side-channel disagrees with immutable rollout policy mean",
        )
        metrics.update(
            {
                "direct_mean_rms": policy_mean_rms,
                "direct_mean_near_bound_fraction": near_bound,
                "direct_pre_tanh_derivative_mean": (
                    ((1.0 - mechanism.square()) * active_float).sum(dim=-1) / active_count
                ),  # $J^{-1}\sum_j(1-(\mu_j^{direct})^2)$，tanh局部导数
            }
        )
        return metrics

    residual_rms = torch.sqrt((mechanism.square() * active_float).sum(dim=-1) / active_count)
    base_mean = rollout_mean - mechanism  # $\mu^{base}=\mu^{rollout}-\Delta\mu^{global}$
    base_mean_rms = torch.sqrt((base_mean.square() * active_float).sum(dim=-1) / active_count)
    metrics.update(
        {
            "base_mean_rms": base_mean_rms,
            "residual_rms": residual_rms,
            "residual_fraction": residual_rms / (base_mean_rms + residual_rms).clamp_min(1.0e-8),
        }
    )
    return metrics


class PalmRotationRlGamesBuilder:
    r"""rl_games builder：按Dict observation ABI构造正式actor/critic package。"""

    def __init__(self, **kwargs: Any) -> None:
        r"""初始化空配置；实际参数由``load``在Runner build阶段注入。"""

        _ = kwargs
        self.params: dict[str, Any] = {}  # YAML network mapping

    def load(self, params: dict[str, Any]) -> None:
        r"""保存residual开关与checkpoint identity。"""

        self.params = params

    def build(self, name: str, **kwargs: Any) -> PalmRotationRlGamesNetwork:
        r"""构造满足continuous-logstd contract的structured network。"""

        _ = name
        return PalmRotationRlGamesNetwork(self.params, **kwargs)


class PalmRotationRlGamesNetwork(nn.Module):
    r"""将单份experience Dict严格分流到actor、critic和共享geometry。

    Actor构造函数只接收``actor_*``、masks和geometry；``critic_*``从未出现在其dataclass中。
    Critic读取privileged tensors，但不读取``prototype_index``。该index只由agent的分层sampler使用。
    """

    def __init__(self, params: Mapping[str, Any], **kwargs: Any) -> None:
        r"""验证16-action与完整named-shape ABI后实例化独立actor/critic。"""

        super().__init__()
        actions_num = int(kwargs.pop("actions_num"))  # canonical action slots，必须为16
        input_shape = kwargs.pop("input_shape")  # Dict[str, sample shape]
        self.value_size = int(kwargs.pop("value_size", 1))  # hand-level scalar value
        self.num_seqs = int(kwargs.pop("num_seqs", 1))  # 非RNN，仅保留rl_games接口字段
        if actions_num != 16 or self.value_size != 1:
            raise ValueError("palm-rotation PPO requires 16 canonical actions and scalar value")
        if not isinstance(input_shape, Mapping):
            raise TypeError("palm-rotation PPO requires a Dict observation space")
        expected_shapes = {
            **PALM_ROTATION_FLOAT_SHAPES,
            **PALM_ROTATION_BOOL_SHAPES,
            **PALM_ROTATION_INT16_SHAPES,
        }
        normalized_shapes = {key: tuple(int(dim) for dim in shape) for key, shape in input_shape.items()}
        if normalized_shapes != expected_shapes:
            missing = sorted(set(expected_shapes) - set(normalized_shapes))
            extra = sorted(set(normalized_shapes) - set(expected_shapes))
            wrong = sorted(
                key
                for key in set(expected_shapes) & set(normalized_shapes)
                if expected_shapes[key] != normalized_shapes[key]
            )
            raise ValueError(f"palm-rotation observation ABI mismatch: missing={missing}, extra={extra}, wrong={wrong}")

        network_cfg = params.get("palm_rotation", {})  # method-specific YAML block
        arm_raw = str(network_cfg.get("arm", "residual"))
        if arm_raw not in {"base", "residual", "direct", "direct_token"}:
            raise ValueError(f"unsupported palm-rotation actor arm: {arm_raw!r}")
        self.arm = cast(Literal["base", "residual", "direct", "direct_token"], arm_raw)
        initial_log_std = float(network_cfg.get("initial_log_std", -0.5))  # shared scalar$\log\sigma$
        max_log_std = float(network_cfg.get("max_log_std", -0.43))  # N000 early-budget exploration ceiling
        base_action_limit = float(network_cfg.get("base_action_limit", 0.8))  # 与0.2 residual构成exact action bound
        history_encoder_raw = str(network_cfg.get("history_encoder", "tcn"))  # History30归纳偏置run identity
        if history_encoder_raw not in {"tcn", "raw_stack"}:
            raise ValueError(f"unsupported palm-rotation history encoder: {history_encoder_raw!r}")
        history_encoder = cast(Literal["tcn", "raw_stack"], history_encoder_raw)
        self.package = PalmRotationActorCritic(
            arm=self.arm,
            initial_log_std=initial_log_std,
            max_log_std=max_log_std,
            base_action_limit=base_action_limit,
            history_encoder=history_encoder,
        )  # actor/critic完全分参；N040不属于此module
        compile_mode_raw = network_cfg.get("compile_mode")  # 只允许编译纯actor/critic forward，不包装rl_games model
        if compile_mode_raw not in {None, "default", "reduce-overhead"}:
            raise ValueError(f"unsupported palm-rotation compile mode: {compile_mode_raw!r}")
        self.compile_mode = None if compile_mode_raw is None else str(compile_mode_raw)
        self._actor_forward = self.package.actor.forward
        self._critic_forward = self.package.critic.forward
        if self.compile_mode is not None:
            self._actor_forward = torch.compile(self._actor_forward, mode=self.compile_mode)
            self._critic_forward = torch.compile(self._critic_forward, mode=self.compile_mode)
        # Compiled bound functions不是nn.Module children；checkpoint keys与optimizers继续锚定`package`原始参数。
        identity = params.get("anymani_identity")
        if not isinstance(identity, dict):
            raise ValueError("palm-rotation network requires a JSON-safe AnyMani runtime identity")
        self.anymani_identity = identity  # checkpoint pre-load identity gate
        self.last_active_joint_mask: torch.Tensor | None = None  # masked Normal读取的当前batch$[B,16]$
        self.last_residual_mean: torch.Tensor | None = None  # scalar diagnostics的detached action residual
        self.last_direct_mean: torch.Tensor | None = None  # direct arm的detached完整authority mean
        self.last_film_modulation_rms: torch.Tensor | None = None  # `[B,16]` geometry FiLM贡献

    def is_rnn(self) -> bool:
        r"""History30由environment observation显式交付，模型不是rl_games recurrent network。"""

        return False

    def get_default_rnn_state(self) -> None:
        r"""非RNN模型无隐状态。"""

    def get_aux_loss(self) -> None:
        r"""MVP不添加distillation/auxiliary loss。"""

    def get_value_layer(self) -> nn.Module:
        r"""返回critic scalar head，供rl_games introspection。"""

        return self.package.critic.value_head

    def actor_parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        r"""将actor参数拆成base-$3e{-4}$与global-residual-$1e{-4}$两组。

        ``geometry_adapter``及owner/dynamic projections只服务global contextual branch，因此与一层graph
        backbone及residual head共同使用较小LR。TCN、local/finger/hand/base和shared log-std属于base组。
        """

        actor = self.package.actor  # 正式actor module
        contextual_modules: tuple[nn.Module, ...] = (
            actor.geometry_adapter,
            actor.owner_contact_projection,
            actor.palm_dynamic_projection,
            actor.joint_dynamic_projection,
            actor.tip_dynamic_projection,
            actor.global_backbone,
            actor.direct_head if self.arm in {"direct", "direct_token"} else actor.residual_head,  # type: ignore[union-attr]
        )
        contextual_ids = {id(parameter) for module in contextual_modules for parameter in module.parameters()}
        base = [parameter for parameter in actor.parameters() if id(parameter) not in contextual_ids]  # temporal/local trunk
        contextual = [parameter for parameter in actor.parameters() if id(parameter) in contextual_ids]  # graph action path
        if {id(parameter) for parameter in base} & {id(parameter) for parameter in contextual}:
            raise RuntimeError("actor local/contextual optimizer groups overlap")
        if len(base) + len(contextual) != len(list(actor.parameters())):
            raise RuntimeError("actor optimizer groups do not cover all parameters")
        return base, contextual

    def forward(
        self, input_dict: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        r"""执行actor mean/logstd与privileged scalar value前向。

        Args:
            input_dict (Mapping[str, Any]): rl_games mapping，``obs``为named tensor Dict。

        Returns:
            tuple: ``(mu, logstd, value, None)``，shape为`[B,16]`、`[B,16]`、`[B,1]`。
        """

        observation = input_dict.get("obs")
        if not isinstance(observation, Mapping):
            raise TypeError("palm-rotation network expects a named observation mapping")

        # Shared geometry只由rollout-cached tensors重建；本forward绝不持有或调用N040 encoder。
        geometry = PalmRotationGeometry(
            tokens=observation["geometry_tokens"].float(),  # FP32 `[B,21,128]`
            owner_valid=observation["owner_valid"].bool(),
            shortest_path=observation["shortest_path"].long(),  # int16 storage -> exact embedding indices
            parent_direction=observation["parent_direction"].long(),
            child_direction=observation["child_direction"].long(),
        )
        actor_observation = PalmRotationActorObservation(
            jnt_current=observation["actor_jnt_current"].float(),
            jnt_history=observation["actor_jnt_history"].float(),
            jnt_limits=observation["actor_jnt_limits"].float(),
            owner_contact=observation["actor_owner_contact"].float(),
            jnt_valid=observation["jnt_valid"].bool(),
            tip_valid=observation["tip_valid"].bool(),
            owner_valid=observation["owner_valid"].bool(),
        )  # actor无法引用任何`critic_*` key
        critic_observation = PalmRotationCriticObservation(
            jnt_state=observation["critic_jnt_state"].float(),
            owner_contact=observation["critic_owner_contact"].float(),
            obj=observation["critic_obj"].float(),
            task=observation["critic_task"].float(),
            reward_release=observation["critic_reward_release"].float(),
            jnt_valid=observation["jnt_valid"].bool(),
            tip_valid=observation["tip_valid"].bool(),
            owner_valid=observation["owner_valid"].bool(),
        )  # privileged critic不读取prototype/cell one-hot

        actor_output = self._actor_forward(actor_observation, geometry)  # base + bounded global residual
        value = self._critic_forward(critic_observation, geometry).unsqueeze(-1)  # `[B,1]`
        self.last_active_joint_mask = actor_observation.jnt_valid  # probability/entropy/KL ghost mask
        self.last_residual_mean = (
            actor_output.residual_mean.detach() if actor_output.residual_mean is not None else None
        )  # residual/base diagnostics；direct保持None
        self.last_direct_mean = (
            actor_output.direct_mean.detach() if actor_output.direct_mean is not None else None
        )  # direct diagnostics；residual/base保持None
        self.last_film_modulation_rms = actor_output.film_modulation_rms.detach()  # local hidden调制幅度
        logstd = actor_output.log_std.expand_as(actor_output.mean)  # 一个共享scalar$\log\sigma$
        return actor_output.mean, logstd, value, None


class PalmRotationMaskedContinuousModel(AnyManiMaskedContinuousModel):
    r"""动作级mean-preserving tanh-squashed masked Normal与机制side-channels。

    Actor直接输出有界动作均值$\bar a=0.8\tanh b+0.2\tanh r\in[-1,1]$。分布先将其映射为
    latent location$m=\operatorname{atanh}(\bar a)$，再采样$z\sim\mathcal N(m,\sigma^2)$并执行
    $a=\tanh z$。因此deterministic action仍严格等于base+residual分解，而随机动作、likelihood与物理
    action space使用同一个变量。
    """

    class Network(AnyManiMaskedContinuousModel.Network):
        r"""计算squashed likelihood/Jacobian并交付residual/FiLM side-channels。"""

        _ACTION_EPS = 1.0e-6  # float32 atanh/log-Jacobian边界，不改变常规open-interval samples

        @classmethod
        def _action_to_latent(cls, action: torch.Tensor) -> torch.Tensor:
            r"""把物理动作$a\in[-1,1]$稳定映射为$z=\operatorname{atanh}(a)$。"""

            bounded = action.clamp(min=-1.0 + cls._ACTION_EPS, max=1.0 - cls._ACTION_EPS)
            return torch.atanh(bounded)

        @classmethod
        def _squashed_per_joint_neglogp(
            cls,
            actions: torch.Tensor,
            action_mean: torch.Tensor,
            sigma: torch.Tensor,
            logstd: torch.Tensor,
        ) -> torch.Tensor:
            r"""返回tanh push-forward在每个joint上的exact negative log-density。

            $$
            -\log\pi_A(a)
            =-\log\mathcal N\!\left(\operatorname{atanh}a;\operatorname{atanh}\bar a,\sigma^2\right)
             +\log(1-a^2).
            $$
            """

            latent_action = cls._action_to_latent(actions)  # $z=\operatorname{atanh}a$
            latent_mean = cls._action_to_latent(action_mean)  # $m=\operatorname{atanh}\bar a$
            normal_neglogp = (
                0.5 * ((latent_action - latent_mean) / sigma).square()
                + logstd
                + 0.5 * math.log(2.0 * math.pi)
            )
            log_jacobian = torch.log((1.0 - actions.square()).clamp_min(cls._ACTION_EPS))
            return normal_neglogp + log_jacobian

        def forward(self, input_dict: dict[str, Any]) -> dict[str, torch.Tensor | None]:
            r"""返回rl_games兼容的bounded action、likelihood、KL parameters与机制诊断。"""

            is_train = bool(input_dict.get("is_train", True))
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            action_mean, logstd, value, states = self.a2c_network(input_dict)
            active_mask = self.a2c_network.last_active_joint_mask
            if not isinstance(active_mask, torch.Tensor):
                raise RuntimeError("palm-rotation squashed policy did not expose active-joint mask")
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.all(action_mean.abs() <= 1.0 + 1.0e-6),
                "palm-rotation deterministic action mean escaped [-1,1]",
            )  # analytic bound仍逐forward fail closed，但不把CUDA stream同步回host
            active_float = active_mask.to(dtype=action_mean.dtype)
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            sigma = torch.exp(logstd)  # latent Normal standard deviation，ghost随后由mask排除
            latent_mean = self._action_to_latent(action_mean)
            distribution = torch.distributions.Normal(latent_mean, sigma, validate_args=False)

            if is_train:
                previous_actions = input_dict.get("prev_actions")
                if not isinstance(previous_actions, torch.Tensor):
                    raise RuntimeError("squashed PPO update requires bounded previous actions")
                per_joint_neglogp = self._squashed_per_joint_neglogp(
                    previous_actions,
                    action_mean,
                    sigma,
                    logstd,
                )
                prev_neglogp = (per_joint_neglogp * active_float).sum(dim=-1)
                entropy_latent = distribution.rsample()  # current-policy Monte Carlo differential entropy sample
                entropy_action = torch.tanh(entropy_latent) * active_float
                entropy_per_joint = self._squashed_per_joint_neglogp(
                    entropy_action,
                    action_mean,
                    sigma,
                    logstd,
                )
                entropy = (entropy_per_joint * active_float).sum(dim=-1) / active_count
                result: dict[str, torch.Tensor | None] = {
                    "prev_neglogp": prev_neglogp,
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": action_mean,  # PPO buffer/player保存deterministic物理动作均值
                    "sigmas": sigma,  # KL所需latent标准差
                }
            else:
                latent_action = distribution.sample()
                selected_action = torch.tanh(latent_action) * active_float  # 物理动作严格位于open interval
                per_joint_neglogp = self._squashed_per_joint_neglogp(
                    selected_action,
                    action_mean,
                    sigma,
                    logstd,
                )
                result = {
                    "neglogpacs": (per_joint_neglogp * active_float).sum(dim=-1),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": action_mean,
                    "sigmas": sigma,
                }
            if self.a2c_network.arm in {"direct", "direct_token"}:
                direct = getattr(self.a2c_network, "last_direct_mean", None)
                if not isinstance(direct, torch.Tensor):
                    raise RuntimeError("palm-rotation direct actor did not expose its bounded mean")
                result["direct_means"] = direct  # 已detach，与rollout action mean逐值一致
            else:
                residual = getattr(self.a2c_network, "last_residual_mean", None)
                if not isinstance(residual, torch.Tensor):
                    raise RuntimeError("palm-rotation residual/base actor did not expose bounded residual")
                result["residuals"] = residual  # 已detach，不延长rollout autograd graph
            film = getattr(self.a2c_network, "last_film_modulation_rms", None)
            if not isinstance(film, torch.Tensor):
                raise RuntimeError("palm-rotation actor did not expose geometry FiLM diagnostics")
            result["film_modulations"] = film  # `[B,16]`，与joint active mask同轴
            return result


class PalmRotationPpoAgent(AnyManiMaskedPpoAgent):
    r"""双optimizer、严格分层minibatch与cached-N040 identity的PPO agent。"""

    def __init__(self, base_name: str, params: dict[str, Any]) -> None:
        r"""让upstream完成Runner状态构造，再替换为actor/critic独立Adam optimizers。"""

        super().__init__(base_name, params)
        if self.has_central_value:
            raise ValueError("palm-rotation custom package already owns the privileged critic; duplicate CV is forbidden")
        if self.mixed_precision:
            raise ValueError("actor, critic, PPO losses and optimizers must remain FP32")
        if self.multi_gpu:
            raise ValueError("MVP80 dual-optimizer agent currently supports one GPU only")
        self.diagnostics = PalmRotationPpoDiagnostics()  # 公式不变，只移除upstream逐microbatch`.cpu()`同步
        network = self.model.a2c_network  # AnyManiMaskedContinuousModel facade下的正式network
        if not isinstance(network, PalmRotationRlGamesNetwork):
            raise TypeError("palm-rotation PPO agent received an incompatible network")

        # Actor-only迁移发生在fresh optimizers构造前；checkpoint其余state从未交给rl_games restore。
        actor_init_path = str(self.config.get("actor_init_checkpoint", "")).strip()
        runtime_identity = network.anymani_identity
        training_identity = runtime_identity.get("training") if isinstance(runtime_identity, Mapping) else None
        warm_start = training_identity.get("actor_warm_start") if isinstance(training_identity, Mapping) else None
        load_actor_init = should_load_actor_init_checkpoint(
            actor_init_path=actor_init_path,
            warm_start=warm_start,
            full_checkpoint_resume=bool(self.config.get("full_checkpoint_resume", False)),
        )
        if load_actor_init:
            loaded_keys = load_actor_init_checkpoint(
                network.package.actor,
                actor_init_path,
                expected_checkpoint_sha256=str(warm_start["checkpoint_sha256"]),  # type: ignore[index]
            )
            if len(loaded_keys) != int(warm_start["loaded_tensor_count"]):  # type: ignore[index]
                raise RuntimeError("actor-init loaded tensor count disagrees with inspected identity")

        base_parameters, contextual_parameters = network.actor_parameter_groups()  # disjoint actor groups
        critic_parameters = list(network.package.critic.parameters())  # completely separate$\theta^c$
        actor_ids = {id(parameter) for parameter in (*base_parameters, *contextual_parameters)}
        critic_ids = {id(parameter) for parameter in critic_parameters}
        if actor_ids & critic_ids:
            raise RuntimeError("actor and critic optimizer parameters overlap")

        # 三个LR锚点来自MVP计划；adaptive scheduler只更新base LR，update_lr保持固定比例。
        self._base_lr_reference = float(self.config["learning_rate"])  # 默认$3e-4$
        self._base_lr_ceiling = float(self.config.get("adaptive_lr_max", self._base_lr_reference))
        if self._base_lr_ceiling != self._base_lr_reference:
            raise ValueError("MVP adaptive_lr_max must equal the declared actor base learning-rate anchor")
        self.actor_arm = network.arm  # 四种arm决定side-channel、机制指标与第二参数组名称
        secondary_lr_key = (
            "contextual_learning_rate" if self.actor_arm in {"direct", "direct_token"} else "residual_learning_rate"
        )
        self._secondary_lr_ratio = float(self.config.get(secondary_lr_key, 1.0e-4)) / self._base_lr_reference
        self._secondary_group_name = (
            "actor_contextual_direct" if self.actor_arm in {"direct", "direct_token"} else "actor_global_residual"
        )
        self._critic_lr_ratio = float(self.config.get("critic_learning_rate", 5.0e-4)) / self._base_lr_reference
        self._gradient_accumulation_steps = int(self.config.get("gradient_accumulation_steps", 1))
        if self._gradient_accumulation_steps < 1 or self.num_minibatches % self._gradient_accumulation_steps != 0:
            raise ValueError("gradient accumulation must divide the number of stratified activation minibatches")
        fused = torch.device(self.ppo_device).type == "cuda"  # CUDA正式训练使用fused Adam
        self.optimizer = torch.optim.Adam(
            [
                {"params": base_parameters, "lr": self.last_lr, "name": "actor_base"},
                {
                    "params": contextual_parameters,
                    "lr": self.last_lr * self._secondary_lr_ratio,
                    "name": self._secondary_group_name,
                },
            ],
            eps=1.0e-8,
            weight_decay=self.weight_decay,
            fused=fused,
        )  # actor checkpoint optimizer；覆盖upstream临时single optimizer
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=self.last_lr * self._critic_lr_ratio,
            eps=1.0e-8,
            weight_decay=self.weight_decay,
            fused=fused,
        )  # independent critic optimizer
        self.asset_count = int(self.config.get("asset_count", 80))  # 正式支持域$A=80$
        self.advantage_normalization_scope = str(self.config.get("advantage_normalization_scope", "global"))
        if self.advantage_normalization_scope not in {"global", "per_asset_rollout"}:
            raise ValueError("advantage_normalization_scope must be global or per_asset_rollout")
        if not self.normalize_advantage:
            raise ValueError("palm-rotation PPO requires normalize_advantage for a declared normalization scope")
        self.last_stratified_permutation: torch.Tensor | None = None  # diagnostics/test evidence
        self.last_advantage_asset_means: torch.Tensor | None = None  # full-rollout raw GAE mean `[A]`
        self.last_advantage_asset_stds: torch.Tensor | None = None  # full-rollout raw GAE sample std `[A]`
        identity = self._runtime_identity()
        if not isinstance(identity, dict) or not isinstance(identity.get("identity_digest"), str):
            raise RuntimeError("palm-rotation diagnostics require the exact runtime identity")
        self.metrics_recorder = PalmRotationMetricsRecorder(
            self.experiment_dir,
            identity_digest=identity["identity_digest"],
            flush_every_updates=int(self.config.get("diagnostics_flush_updates", 50)),
        )  # run-owned Parquet shard lifecycle
        self._optimization_count = torch.zeros(self.asset_count, device=self.ppo_device)  # mini-epoch samples$[A]$
        common_optimization_fields = (
            "advantage",
            "advantage_square",
            "value_error",
            "return_target",
            "return_target_square",
            "value_prediction",
            "value_prediction_square",
            "value_residual_square",
            "value_error_physical",
            "value_clip_fraction",
            "kl",
            "clip_fraction",
            "action_rms",
            "policy_mean_rms",
            "policy_mean_near_bound_fraction",
            "film_modulation_rms",
        )
        self._mechanism_metric_fields = (
            ("direct_mean_rms", "direct_mean_near_bound_fraction", "direct_pre_tanh_derivative_mean")
            if self.actor_arm in {"direct", "direct_token"}
            else ("base_mean_rms", "residual_rms", "residual_fraction")
        )
        self._optimization_sums = {
            name: torch.zeros(self.asset_count, device=self.ppo_device)
            for name in (*common_optimization_fields, *self._mechanism_metric_fields)
        }  # 当前update跨全部minibatches×mini-epochs之和
        self._gradient_probe_per_asset: dict[str, torch.Tensor] | None = None
        self._gradient_probe_global: dict[str, float] | None = None
        self._optimizer_step_count = 0  # 当前update真实optimizer step次数
        self._optimizer_microbatch_count = 0  # 当前update真实forward/backward microbatch次数
        self._gradient_microbatch_index = 0  # 必须在每个update边界回到0
        self._optimizer_scalar_sums = {
            name: torch.zeros((), dtype=torch.float32, device=self.ppo_device)
            for name in (
                "actor_loss",
                "critic_loss",
                "entropy",
                "policy_sigma",
                "actor_grad_norm",
                "critic_grad_norm",
            )
        }  # 无per-asset归属的标量留在GPU累计，update边界才执行六次host transfer

    def init_tensors(self) -> None:
        r"""在upstream experience buffer增加detached action-residual side-channel。"""

        super().init_tensors()
        batch = self.num_agents * self.num_actors  # rollout并行样本数$N$
        mechanism_key = "direct_means" if self.actor_arm in {"direct", "direct_token"} else "residuals"
        self.experience_buffer.tensor_dict[mechanism_key] = torch.zeros(
            self.horizon_length,
            batch,
            16,
            dtype=torch.float32,
            device=self.ppo_device,
        )  # `[H,N,16]`，与actions/mus同axis
        self.update_list.append(mechanism_key)  # play_steps从custom model输出写入buffer
        self.tensor_list.append(mechanism_key)  # rollout结束后swap env/time并flatten
        self.experience_buffer.tensor_dict["film_modulations"] = torch.zeros(
            self.horizon_length,
            batch,
            16,
            dtype=torch.float32,
            device=self.ppo_device,
        )  # `[H,N,16]`，逐joint local-hidden FiLM RMS
        self.update_list.append("film_modulations")
        self.tensor_list.append("film_modulations")

    def update_lr(self, lr: float) -> None:
        r"""保持base/residual/critic学习率比例随adaptive schedule同步缩放。"""

        current = bounded_adaptive_learning_rate(float(lr), self._base_lr_ceiling)  # 禁止低KL指数越过锚点
        self.last_lr = current  # upstream先写入未限幅值，必须同步恢复scheduler state
        for group in self.optimizer.param_groups:
            group["lr"] = current if group.get("name") == "actor_base" else current * self._secondary_lr_ratio
        for group in self.critic_optimizer.param_groups:
            group["lr"] = current * self._critic_lr_ratio

    def _assert_actor_learning_rate_ratio(self) -> None:
        r"""在真实step前验证base/residual LR仍保持声明的$3:1$比例。

        rl_games ``A2CAgent.train_actor_critic``会在每个microbatch后把所有actor groups无条件写成
        ``last_lr``。梯度累积使该副作用恰好发生在下一次逻辑step之前，导致residual实际使用base LR。
        本agent覆盖该wrapper，并在此以step-time值fail closed，update尾日志不再替代真实执行证据。
        """

        groups = {str(group.get("name")): float(group["lr"]) for group in self.optimizer.param_groups}
        expected_base = float(self.last_lr)
        expected_secondary = expected_base * self._secondary_lr_ratio
        if abs(groups.get("actor_base", -1.0) - expected_base) > 1.0e-12:
            raise RuntimeError(f"actor base LR drifted before optimizer step: {groups}")
        if abs(groups.get(self._secondary_group_name, -1.0) - expected_secondary) > 1.0e-12:
            raise RuntimeError(f"actor contextual LR ratio drifted before optimizer step: {groups}")

    def train_actor_critic(self, input_dict: dict[str, Any]):
        r"""执行custom gradient step且禁止upstream把两个actor LR groups合并。

        Returns:
            tuple[Any, ...]: rl_games训练循环消费的标准loss/KL/LR/mu/sigma结果。
        """

        self.set_train()
        self.calc_gradients(input_dict)
        return self.train_result

    @staticmethod
    def masked_policy_kl(
        current_mu: torch.Tensor,
        current_sigma: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""利用tanh双射，在latent Normal中计算物理squashed policy的精确KL。"""

        current_latent = PalmRotationMaskedContinuousModel.Network._action_to_latent(current_mu)
        old_latent = PalmRotationMaskedContinuousModel.Network._action_to_latent(old_mu)
        c1 = torch.log(old_sigma / current_sigma + 1.0e-5)
        c2 = (current_sigma.square() + (old_latent - current_latent).square()) / (
            2.0 * (old_sigma.square() + 1.0e-5)
        )
        weights = active_mask.to(dtype=current_mu.dtype)
        return ((c1 + c2 - 0.5) * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def _gradient_probe_parameters(self) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
        r"""返回动作末层与value scalar末层参数，定义低成本task-gradient proxy。

        Full actor/critic参数Gram对A128需每次执行256次完整反向，成本不可接受。这里固定读取最接近输出的
        可训练层：residual/base arm取base与residual两个末层，direct取direct末层；Critic取scalar value末层。
        该proxy只用于判断是否值得启动更昂贵的offline/full-parameter probe，不冒充完整梯度。
        """

        network = self.model.a2c_network
        actor = network.package.actor
        if self.actor_arm in {"direct", "direct_token"}:
            actor_modules = (actor.direct_head[-1],)  # type: ignore[union-attr]
        else:
            actor_modules = (actor.base_head[-1], actor.residual_head[-1])  # type: ignore[union-attr]
        critic_modules = (network.package.critic.value_head[-1],)
        actor_parameters = tuple(parameter for module in actor_modules for parameter in module.parameters())
        critic_parameters = tuple(parameter for module in critic_modules for parameter in module.parameters())
        return actor_parameters, critic_parameters

    @staticmethod
    def _per_asset_gradient_matrix(
        objective: torch.Tensor,
        labels: torch.Tensor,
        parameters: tuple[nn.Parameter, ...],
        *,
        asset_count: int,
    ) -> torch.Tensor:
        r"""计算每资产mean objective对指定末层参数的flattened gradient matrix。

        Args:
            objective (torch.Tensor): 当前stratified minibatch的per-sample scalar objective，形状$[M]$。
            labels (torch.Tensor): selection-local asset index，形状$[M]$且每资产样本数相同。
            parameters (tuple[nn.Parameter, ...]): 固定proxy参数集合。
            asset_count (int): 支持域资产数$A$。

        Returns:
            torch.Tensor: $G\in\mathbb R^{A\times P}$；unused branch参数以0填充。
        """

        scalar_objective = objective.reshape(-1)
        if scalar_objective.shape != labels.shape or not parameters:
            raise ValueError("gradient probe objective/labels/parameters are malformed")
        rows = []
        for asset_index in range(asset_count):
            member = labels == asset_index
            if not bool(member.any().item()):
                raise RuntimeError(f"gradient probe minibatch lacks asset {asset_index}")
            gradients = torch.autograd.grad(
                scalar_objective[member].mean(),
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            rows.append(
                torch.cat(
                    tuple(
                        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
                        for parameter, gradient in zip(parameters, gradients, strict=True)
                    )
                )
            )
        return torch.stack(rows, dim=0)  # `[A,P]`，不写入parameter.grad

    def _run_gradient_probe(
        self,
        *,
        actor_objective: torch.Tensor,
        critic_objective: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        r"""形成head-level per-asset Actor/Critic Gram并原子保存dense NPZ。

        Probe只在配置cadence命中的首个stratified minibatch运行一次。Parquet保存每资产norm/mean cosine与global
        span/negative-pair/top-1 norm fraction；完整$A\times A$ Gram保存在run-local NPZ，便于事后重算阈值。
        """

        start = time.perf_counter()
        actor_parameters, critic_parameters = self._gradient_probe_parameters()
        actor_gradients = self._per_asset_gradient_matrix(
            actor_objective,
            labels,
            actor_parameters,
            asset_count=self.asset_count,
        )
        critic_gradients = self._per_asset_gradient_matrix(
            critic_objective,
            labels,
            critic_parameters,
            asset_count=self.asset_count,
        )
        per_asset: dict[str, torch.Tensor] = {}
        global_metrics: dict[str, float] = {}
        dense: dict[str, np.ndarray] = {}
        for prefix, gradients in (("actor", actor_gradients), ("critic", critic_gradients)):
            gram = gradients @ gradients.T  # $\Gamma=GG^T\in\mathbb R^{A\times A}$
            norms = torch.linalg.vector_norm(gradients, dim=-1)  # $\|g_i\|_2$
            denominator = norms[:, None] * norms[None, :]
            cosine = torch.where(denominator > 1.0e-20, gram / denominator.clamp_min(1.0e-20), torch.zeros_like(gram))
            off_diagonal = ~torch.eye(self.asset_count, dtype=torch.bool, device=gram.device)
            pair_values = cosine[torch.triu(off_diagonal, diagonal=1)]
            mean_cosine = cosine.masked_fill(~off_diagonal, 0.0).sum(dim=1) / max(self.asset_count - 1, 1)
            positive_min = norms[norms > 1.0e-12].min() if bool((norms > 1.0e-12).any().item()) else norms.new_tensor(1.0e-12)
            per_asset[f"{prefix}_probe_grad_norm"] = norms.detach().cpu()
            per_asset[f"{prefix}_probe_mean_cosine"] = mean_cosine.detach().cpu()
            global_metrics[f"{prefix}_probe_negative_pair_fraction"] = (
                float((pair_values < 0.0).float().mean().item()) if pair_values.numel() else 0.0
            )
            global_metrics[f"{prefix}_probe_top1_norm_fraction"] = float(
                (norms.max() / norms.sum().clamp_min(1.0e-12)).item()
            )
            global_metrics[f"{prefix}_probe_grad_norm_span"] = float((norms.max() / positive_min).item())
            dense[f"{prefix}_gram"] = gram.detach().cpu().numpy()
            dense[f"{prefix}_cosine"] = cosine.detach().cpu().numpy()
            dense[f"{prefix}_norm"] = norms.detach().cpu().numpy()
        global_metrics["gradient_probe_seconds"] = time.perf_counter() - start
        self._gradient_probe_per_asset = per_asset
        self._gradient_probe_global = global_metrics

        root = Path(self.experiment_dir) / "gradient_probes"
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"update_{int(self.epoch_num):06d}.npz"
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            schema_version=np.asarray("head-gradient-gram-v1"),
            update=np.asarray(int(self.epoch_num), dtype=np.int64),
            asset_index=np.arange(self.asset_count, dtype=np.int64),
            **dense,
        )
        temporary.replace(destination)  # dense artifact只在完整写入后可见

    def _run_full_actor_gradient_shadow(
        self,
        *,
        global_objective: torch.Tensor,
        per_asset_objective: torch.Tensor,
        labels: torch.Tensor,
        replica_halves: torch.Tensor,
    ) -> None:
        r"""在同一首个stratified minibatch比较两种advantage scope的完整Actor梯度。

        ``full``表示覆盖Actor全部可训练参数，而非使用完整rollout作为一次activation batch。每种scope按真实
        replica parity形成两个half gradients；同一update共享forward、old policy、actions与PPO clip状态，唯一变化
        是Actor advantage。该函数只调用``autograd.grad``并写probe artifact，不触碰``parameter.grad``或optimizer。
        """

        start = time.perf_counter()
        actor_parameters = tuple(self.model.a2c_network.package.actor.parameters())  # 完整共享Actor坐标，含global logstd
        if not actor_parameters:
            raise RuntimeError("full Actor gradient shadow received an empty parameter set")
        scope_audits: dict[str, dict[str, torch.Tensor]] = {}
        dense: dict[str, np.ndarray] = {}
        summary: dict[str, Any] = {
            "schema_version": "full-actor-gradient-shadow-v1",
            "update": int(self.epoch_num),
            "asset_count": self.asset_count,
            "actor_parameter_count": sum(parameter.numel() for parameter in actor_parameters),
            "sample_population": "first-stratified-minibatch-with-full-rollout-advantage-moments",
            "replica_split": "even-versus-odd-runtime-replica-index",
            "scopes": {},
        }
        for scope, objective in (("global", global_objective), ("per_asset_rollout", per_asset_objective)):
            half_gradients, half_counts = per_asset_replica_half_gradients(
                objective,
                labels,
                replica_halves,
                actor_parameters,
                asset_count=self.asset_count,
            )  # `[A,2,P]`，两个scope逐样本只改变advantage尺度
            audit = compute_actor_gradient_scope_audit(half_gradients, half_counts)
            scope_audits[scope] = audit
            for name in (
                "half_counts",
                "half_norms",
                "half_self_cosine",
                "asset_gradients",
                "asset_norms",
                "asset_gram",
                "asset_cosine",
            ):
                dense[f"{scope}_{name}"] = audit[name].detach().cpu().numpy()
            self_cosine = audit["half_self_cosine"]
            summary["scopes"][scope] = {
                "half_self_cosine_min": float(self_cosine.min().item()),
                "half_self_cosine_q25": float(torch.quantile(self_cosine, 0.25).item()),
                "half_self_cosine_median": float(self_cosine.median().item()),
                "reliable_asset_fraction_positive_self_cosine": float(audit["reliable_asset_fraction"].item()),
                "negative_pair_fraction": float(audit["negative_pair_fraction"].item()),
                "top1_norm_fraction": float(audit["top1_norm_fraction"].item()),
                "norm_span": float(audit["norm_span"].item()),
                "aggregate_to_individual_norm_ratio": float(audit["aggregate_to_individual_norm_ratio"].item()),
            }

        # 同一资产与合成update在两种scope间的余弦，直接量化尺度校准是否实质改变Actor方向。
        global_gradients = scope_audits["global"]["asset_gradients"]
        per_asset_gradients = scope_audits["per_asset_rollout"]["asset_gradients"]
        scope_cross_cosine = torch.sum(global_gradients * per_asset_gradients, dim=-1) / (
            torch.linalg.vector_norm(global_gradients, dim=-1)
            * torch.linalg.vector_norm(per_asset_gradients, dim=-1)
        ).clamp_min(1.0e-20)
        global_aggregate = global_gradients.sum(dim=0)
        per_asset_aggregate = per_asset_gradients.sum(dim=0)
        aggregate_scope_cosine = torch.dot(global_aggregate, per_asset_aggregate) / (
            torch.linalg.vector_norm(global_aggregate) * torch.linalg.vector_norm(per_asset_aggregate)
        ).clamp_min(1.0e-20)
        dense["scope_cross_cosine_per_asset"] = scope_cross_cosine.detach().cpu().numpy()
        summary["scope_comparison"] = {
            "per_asset_cross_scope_cosine_min": float(scope_cross_cosine.min().item()),
            "per_asset_cross_scope_cosine_median": float(scope_cross_cosine.median().item()),
            "aggregate_cross_scope_cosine": float(aggregate_scope_cosine.item()),
        }
        summary["wall_seconds"] = time.perf_counter() - start

        root = Path(self.experiment_dir) / "full_gradient_shadows"
        root.mkdir(parents=True, exist_ok=True)
        array_destination = root / f"update_{int(self.epoch_num):06d}.npz"
        array_temporary = array_destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            array_temporary,
            update=np.asarray(int(self.epoch_num), dtype=np.int64),
            asset_index=np.arange(self.asset_count, dtype=np.int64),
            **dense,
        )
        array_temporary.replace(array_destination)  # dense gradients/Gram仅完整压缩后原子可见
        summary_destination = root / f"update_{int(self.epoch_num):06d}.json"
        summary_temporary = summary_destination.with_suffix(".tmp.json")
        summary_temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_temporary.replace(summary_destination)

    @staticmethod
    def _index_dataset_value(value: Any, indices: torch.Tensor) -> Any:
        r"""对dataset tensor或一层Dict统一应用batch permutation。"""

        if isinstance(value, dict):
            return {key: tensor[indices] for key, tensor in value.items()}  # named experience tensors
        return value[indices] if isinstance(value, torch.Tensor) else value

    def prepare_dataset(self, batch_dict: dict[str, Any]) -> None:
        r"""形成global/per-asset两份GAE，按显式scope选择Actor输入，再分层排列minibatches。"""

        raw_returns = batch_dict.get("returns")
        raw_values = batch_dict.get("values")
        if not isinstance(raw_returns, torch.Tensor) or not isinstance(raw_values, torch.Tensor):
            raise RuntimeError("palm-rotation rollout lacks raw return/value tensors")
        rollout_observation = batch_dict.get("obses")
        if not isinstance(rollout_observation, Mapping) or "prototype_index" not in rollout_observation:
            raise RuntimeError("palm-rotation rollout lacks prototype labels for advantage normalization")
        rollout_labels = rollout_observation["prototype_index"].reshape(-1).long()  # permutation前的`[B]` asset axis
        raw_advantages = (raw_returns - raw_values).sum(dim=1)  # upstream PPO定义的物理GAE `[B]`
        if raw_advantages.numel() % self.horizon_length != 0:
            raise RuntimeError("flattened rollout does not divide into complete environment trajectories")
        flat_index = torch.arange(raw_advantages.numel(), device=raw_advantages.device)
        environment_index = torch.div(flat_index, self.horizon_length, rounding_mode="floor")  # env-major flatten axis
        expected_labels = environment_index.remainder(self.asset_count)  # runtime route定义$k_e=e\bmod A$
        if not bool(torch.equal(rollout_labels, expected_labels)):
            raise RuntimeError("rollout prototype labels disagree with env-major round-robin routing")
        replica_index = torch.div(environment_index, self.asset_count, rounding_mode="floor")
        replica_halves = replica_index.remainder(2)  # even/odd replica IDs形成确定性、近等量的独立轨迹halves
        mechanism_key = "direct_means" if self.actor_arm in {"direct", "direct_token"} else "residuals"
        mechanism = batch_dict.get(mechanism_key)  # detached arm-specific rollout diagnostic`[B,16]`
        if not isinstance(mechanism, torch.Tensor):
            raise RuntimeError(f"palm-rotation rollout is missing {mechanism_key} side-channel")
        rollout_mean = batch_dict.get("mus")  # rollout策略生成动作时的有界均值`[B,16]`
        if not isinstance(rollout_mean, torch.Tensor) or rollout_mean.shape != mechanism.shape:
            raise RuntimeError("palm-rotation rollout mean and mechanism side-channel shapes disagree")
        frozen_rollout_mean = rollout_mean.detach().clone()  # 与dataset可变KL reference断开storage alias
        super().prepare_dataset(batch_dict)  # 保持upstream GAE/value normalization与PPO fields
        global_advantages = self.dataset.values_dict.get("advantages")
        if not isinstance(global_advantages, torch.Tensor) or global_advantages.shape != raw_advantages.shape:
            raise RuntimeError("upstream global advantage tensor disagrees with rollout GAE shape")
        per_asset_advantages, asset_means, asset_stds = normalize_advantages_per_asset(
            raw_advantages,
            rollout_labels,
            asset_count=self.asset_count,
        )  # `[B]`与`[A]`；在任何minibatch permutation前固定完整rollout moments
        self.dataset.values_dict["global_advantages"] = global_advantages.detach()
        self.dataset.values_dict["per_asset_advantages"] = per_asset_advantages.detach()
        self.dataset.values_dict["replica_halves"] = replica_halves.detach()
        self.dataset.values_dict["advantages"] = (
            per_asset_advantages if self.advantage_normalization_scope == "per_asset_rollout" else global_advantages
        )  # 只有该具名字段进入主Actor surrogate
        self.last_advantage_asset_means = asset_means.detach()
        self.last_advantage_asset_stds = asset_stds.detach()
        self.dataset.values_dict["raw_returns"] = raw_returns.detach()  # normalization前物理return target`[B,1]`
        self.dataset.values_dict["raw_values"] = raw_values.detach()  # rollout时denormalized value`[B,1]`
        self.dataset.values_dict["rollout_mu"] = frozen_rollout_mean  # 五轮PPO均只读的$\mu^{rollout}$
        self.dataset.values_dict[mechanism_key] = mechanism  # optimizer diagnostics，不进入loss
        film_modulations = batch_dict.get("film_modulations")
        if not isinstance(film_modulations, torch.Tensor) or film_modulations.shape != mechanism.shape:
            raise RuntimeError("palm-rotation rollout is missing geometry FiLM side-channel")
        self.dataset.values_dict["film_modulations"] = film_modulations  # detached mechanism diagnostic
        observation = self.dataset.values_dict.get("obs")
        if not isinstance(observation, dict) or "prototype_index" not in observation:
            raise RuntimeError("stratified PPO requires prototype_index in cached observations")
        permutation = stratified_asset_permutation(
            observation["prototype_index"],
            asset_count=self.asset_count,
            minibatch_count=self.num_minibatches,
        )  # `[B]`，每连续`minibatch_size`严格平衡
        if permutation.numel() != self.batch_size or self.minibatch_size * self.num_minibatches != self.batch_size:
            raise RuntimeError("stratified permutation disagrees with rl_games batch geometry")
        self.dataset.values_dict = {
            key: self._index_dataset_value(value, permutation)
            for key, value in self.dataset.values_dict.items()
        }  # 所有old policy/value/action/obs字段保持同一sample correspondence
        self.last_stratified_permutation = permutation.detach()  # scalar/table diagnostics可审计本update顺序

    def calc_gradients(self, input_dict: dict[str, Any]) -> None:
        r"""同一前向图分别对$\theta^a$与$\theta^c$执行FP32 PPO/value更新。"""

        value_predictions = input_dict["old_values"]  # rollout normalized values`[M,1]`
        old_neglogp = input_dict["old_logp_actions"]  # masked action negative log probability`[M]`
        advantage = input_dict["advantages"]  # identity-selected global/per-asset GAE`[M]`
        global_advantages = input_dict.get("global_advantages")  # 同一rollout的upstream global标准化GAE
        per_asset_advantages = input_dict.get("per_asset_advantages")  # full-rollout逐资产标准化GAE
        replica_halves = input_dict.get("replica_halves")  # even/odd runtime replica split `[M]`
        if not all(isinstance(value, torch.Tensor) for value in (global_advantages, per_asset_advantages, replica_halves)):
            raise RuntimeError("PPO minibatch lacks advantage-scope or replica-half shadow tensors")
        kl_reference_mu = input_dict["mu"]  # rl_games逐minibatch更新的KL参考means`[M,16]`
        kl_reference_sigma = input_dict["sigma"]  # 与means同生命周期的KL参考stds`[M,16]`
        rollout_mean = input_dict.get("rollout_mu")  # 当前rollout采样时冻结的策略means`[M,16]`
        if not isinstance(rollout_mean, torch.Tensor):
            raise RuntimeError("PPO minibatch lacks immutable rollout policy means")
        returns = input_dict["returns"]  # normalized return targets`[M,1]`
        raw_returns = input_dict.get("raw_returns")  # value normalization前物理return targets`[M,1]`
        raw_values = input_dict.get("raw_values")  # rollout时denormalized value predictions`[M,1]`
        if not isinstance(raw_returns, torch.Tensor) or not isinstance(raw_values, torch.Tensor):
            raise RuntimeError("PPO minibatch lacks raw return/value diagnostics")
        actions = input_dict["actions"]  # sampled canonical actions`[M,16]`
        observation = self._preproc_obs(input_dict["obs"])  # named Dict；normalize_input=False
        labels = observation["prototype_index"].reshape(-1).long()  # sampler certificate，不进模型
        result = self.model({"is_train": True, "prev_actions": actions, "obs": observation})
        new_neglogp = result["prev_neglogp"]  # masked active-joint likelihood
        values = result["values"]  # privileged critic prediction`[M,1]`
        entropy = result["entropy"]  # mean entropy per active DoF`[M]`
        mu = result["mus"]  # current actor means`[M,16]`
        sigma = result["sigmas"]  # shared scalar expanded to`[M,16]`

        # Actor objective只含clipped surrogate、active-DoF entropy与masked bounds；无critic gradient path。
        actor_loss_vector = self.actor_loss_func(old_neglogp, new_neglogp, advantage, self.ppo, self.e_clip)
        bounds_loss_vector = self.bound_loss(mu)  # active-DoF mean bounds penalty
        actor_objective_vector = (
            actor_loss_vector - entropy * self.entropy_coef + bounds_loss_vector * self.bounds_loss_coef
        )  # `[M]`，仅供per-asset gradient probe；主loss归约仍由apply_masks定义
        actor_terms, _ = torch_ext.apply_masks(
            [actor_loss_vector.unsqueeze(1), entropy.unsqueeze(1), bounds_loss_vector.unsqueeze(1)],
            None,
        )
        actor_loss, entropy_loss, bounds_loss = actor_terms
        actor_objective = actor_loss - entropy_loss * self.entropy_coef + bounds_loss * self.bounds_loss_coef

        # Critic objective使用独立structured critic和optimizer；0.5保持upstream PPO value-loss convention。
        critic_vector = common_losses.critic_loss(
            self.model,
            value_predictions,
            values,
            self.e_clip,
            returns,
            self.clip_value,
        )
        critic_terms, _ = torch_ext.apply_masks([critic_vector], None)
        critic_loss = critic_terms[0]
        critic_objective = 0.5 * self.critic_coef * critic_loss
        critic_objective_vector = 0.5 * self.critic_coef * critic_vector.reshape(-1)  # `[M]`

        # 非有限objective会污染Adam moments与后续checkpoint；设备异步断言保留逐项故障名但不阻塞host。
        finite_forward = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "bounds_loss": bounds_loss,
            "actor_objective": actor_objective,
            "critic_objective": critic_objective,
            "mu": mu,
            "sigma": sigma,
            "value": values,
        }
        for name, value in finite_forward.items():
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.isfinite(value).all(),
                f"palm-rotation PPO produced non-finite {name}",
            )

        # 可选完整Actor shadow在同一forward/minibatch上只改变advantage scope，不写主gradient或optimizer。
        full_gradient_shadow_frequency = int(self.config.get("full_gradient_shadow_frequency", 0))
        if (
            full_gradient_shadow_frequency > 0
            and int(self.epoch_num) % full_gradient_shadow_frequency == 0
            and self._gradient_microbatch_index == 0
        ):
            common_actor_regularizer = -entropy * self.entropy_coef + bounds_loss_vector * self.bounds_loss_coef
            global_actor_objective = self.actor_loss_func(
                old_neglogp,
                new_neglogp,
                cast(torch.Tensor, global_advantages),
                self.ppo,
                self.e_clip,
            ) + common_actor_regularizer
            per_asset_actor_objective = self.actor_loss_func(
                old_neglogp,
                new_neglogp,
                cast(torch.Tensor, per_asset_advantages),
                self.ppo,
                self.e_clip,
            ) + common_actor_regularizer
            self._run_full_actor_gradient_shadow(
                global_objective=global_actor_objective,
                per_asset_objective=per_asset_actor_objective,
                labels=labels,
                replica_halves=cast(torch.Tensor, replica_halves),
            )

        # 固定cadence只在本update首个stratified minibatch形成一次head-gradient proxy。
        gradient_probe_frequency = int(self.config.get("gradient_probe_frequency", 0))
        if (
            gradient_probe_frequency > 0
            and int(self.epoch_num) % gradient_probe_frequency == 0
            and self._gradient_microbatch_index == 0
        ):
            self._run_gradient_probe(
                actor_objective=actor_objective_vector,
                critic_objective=critic_objective_vector,
                labels=labels,
            )

        # 四个activation microbatches组成一个原76,800/4逻辑minibatch；参数只在组末更新一次。
        accumulation_offset = self._gradient_microbatch_index % self._gradient_accumulation_steps
        if accumulation_offset == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
        (actor_objective / self._gradient_accumulation_steps).backward()
        (critic_objective / self._gradient_accumulation_steps).backward()
        network = self.model.a2c_network  # validated PalmRotationRlGamesNetwork
        if not self.truncate_grads:
            raise RuntimeError("palm-rotation PPO requires independent actor/critic gradient clipping")
        accumulation_boundary = accumulation_offset + 1 == self._gradient_accumulation_steps
        if accumulation_boundary:
            self._assert_actor_learning_rate_ratio()  # 必须读取step-time optimizer groups
            actor_grad_norm = clip_grad_norm_(network.package.actor.parameters(), self.grad_norm)  # 逻辑batch$\|g_a\|_2$
            critic_grad_norm = clip_grad_norm_(network.package.critic.parameters(), self.grad_norm)  # 逻辑batch$\|g_c\|_2$
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.isfinite(actor_grad_norm) & torch.isfinite(critic_grad_norm),
                "palm-rotation PPO produced non-finite actor or critic gradient norm",
            )
            self.optimizer.step()
            self.critic_optimizer.step()
            network.package.actor.project_exploration_parameters()  # optimizer后立即恢复$\log\sigma\le-0.43$
            self._optimizer_step_count += 1
            self._optimizer_scalar_sums["actor_grad_norm"].add_(actor_grad_norm.detach().float())
            self._optimizer_scalar_sums["critic_grad_norm"].add_(critic_grad_norm.detach().float())
        self._gradient_microbatch_index += 1

        # Loss/entropy/sigma按每个等大microbatch累计；梯度范数只在逻辑optimizer boundary累计。
        self._optimizer_microbatch_count += 1
        microbatch_scalars = {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": entropy_loss.detach(),
            "policy_sigma": sigma.detach().mean(),
        }
        for name, value in microbatch_scalars.items():
            self._optimizer_scalar_sums[name].add_(value.float())  # scalar detach已阻断autograd graph

        # Adaptive scheduler消费active-DoF-normalized KL，ghost sigma/mean不影响统计。
        active_mask = network.last_active_joint_mask
        if not isinstance(active_mask, torch.Tensor) or active_mask.shape != mu.shape:
            raise RuntimeError("palm-rotation network did not expose active-joint mask")
        with torch.no_grad():
            kl_per_sample = self.masked_policy_kl(
                mu.detach(), sigma.detach(), kl_reference_mu, kl_reference_sigma, active_mask
            )
            kl = kl_per_sample.mean()
            ratio = torch.exp(old_neglogp - new_neglogp.detach())  # PPO importance ratio`[M]`
            clip_fraction = (torch.abs(ratio - 1.0) > self.e_clip).float()  # clipped sample indicator
            active_float = active_mask.float()
            active_count = active_float.sum(dim=-1).clamp_min(1.0)
            action_rms = torch.sqrt((actions.square() * active_float).sum(dim=-1) / active_count)
            value_prediction_physical = denormalize_value_readonly(self.model, values.detach()).reshape(-1)
            return_target_physical = raw_returns.reshape(-1)
            value_residual_physical = value_prediction_physical - return_target_physical
            value_clip_fraction = (
                (values.detach().reshape(-1) - value_predictions.reshape(-1)).abs() > self.e_clip
            ).float()  # normalized value space中的PPO clip激活
            mechanism_key = "direct_means" if self.actor_arm in {"direct", "direct_token"} else "residuals"
            mechanism = input_dict.get(mechanism_key)  # 与$\mu^{rollout}$同一次forward保存的arm-specific量
            if not isinstance(mechanism, torch.Tensor):
                raise RuntimeError(f"PPO minibatch lacks {mechanism_key} diagnostics")
            mechanism_metrics = rollout_policy_mechanism_metrics(
                rollout_mean,
                mechanism,
                active_mask,
                actor_arm=cast(Literal["base", "residual", "direct", "direct_token"], self.actor_arm),
            )
            film_modulations = input_dict.get("film_modulations")
            if not isinstance(film_modulations, torch.Tensor) or film_modulations.shape != actions.shape:
                raise RuntimeError("PPO minibatch geometry FiLM diagnostics disagree with action shape")
            film_modulation_rms = (film_modulations * active_float).sum(dim=-1) / active_count
            optimization_values = {
                "advantage": advantage.detach(),
                "advantage_square": advantage.detach().square(),
                "value_error": torch.abs(values.detach().squeeze(-1) - returns.squeeze(-1)),
                "return_target": return_target_physical,
                "return_target_square": return_target_physical.square(),
                "value_prediction": value_prediction_physical,
                "value_prediction_square": value_prediction_physical.square(),
                "value_residual_square": value_residual_physical.square(),
                "value_error_physical": value_residual_physical.abs(),
                "value_clip_fraction": value_clip_fraction,
                "kl": kl_per_sample,
                "clip_fraction": clip_fraction,
                "action_rms": action_rms,
                "policy_mean_rms": mechanism_metrics["policy_mean_rms"],
                "policy_mean_near_bound_fraction": mechanism_metrics["policy_mean_near_bound_fraction"],
                "film_modulation_rms": film_modulation_rms,
            }
            optimization_values.update(
                {name: mechanism_metrics[name] for name in self._mechanism_metric_fields}
            )  # arm-specific字段均来自同一冻结rollout分解，不混入后续KL reference
            self._optimization_count.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
            for name, per_sample in optimization_values.items():
                self._optimization_sums[name].scatter_add_(0, labels, per_sample.float())
        self.diagnostics.mini_batch(
            self,
            {
                "values": value_predictions,
                "returns": returns,
                "new_neglogp": new_neglogp,
                "old_neglogp": old_neglogp,
                "masks": None,
            },
            self.e_clip,
            0,
        )
        self.train_result = (
            actor_loss.detach(),
            critic_loss.detach(),
            entropy_loss.detach(),
            kl.detach(),
            self.last_lr,
            1.0,
            mu.detach(),
            sigma.detach(),
            bounds_loss.detach(),
        )  # 与rl_games ContinuousA2CBase.train_epoch tuple contract一致

    def _reset_optimization_metrics(self) -> None:
        r"""在每次rollout/update前清零mini-epoch optimization统计。"""

        self._optimization_count.zero_()
        for total in self._optimization_sums.values():
            total.zero_()
        self._optimizer_step_count = 0
        self._optimizer_microbatch_count = 0
        self._gradient_microbatch_index = 0
        self.optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        for name in self._optimizer_scalar_sums:
            self._optimizer_scalar_sums[name].zero_()  # 保留device scalar storage，避免每update重新分配
        self._gradient_probe_per_asset = None
        self._gradient_probe_global = None

    def _drain_optimizer_scalars(self) -> dict[str, float]:
        r"""返回当前update跨全部minibatches/mini-epochs的global优化统计。"""

        if self._optimizer_step_count < 1 or self._optimizer_microbatch_count < 1:
            raise RuntimeError("optimizer scalar diagnostics observed no update steps")
        if self._gradient_microbatch_index % self._gradient_accumulation_steps != 0:
            raise RuntimeError("PPO update ended inside a logical accumulated minibatch")
        microbatch_denominator = float(self._optimizer_microbatch_count)  # 正式$16\times5=80$
        step_denominator = float(self._optimizer_step_count)  # 正式$(16/4)\times5=20$
        return {
            **{
                name: float((self._optimizer_scalar_sums[name] / microbatch_denominator).item())
                for name in ("actor_loss", "critic_loss", "entropy", "policy_sigma")
            },
            **{
                name: float((self._optimizer_scalar_sums[name] / step_denominator).item())
                for name in ("actor_grad_norm", "critic_grad_norm")
            },
            "optimizer_microbatches": microbatch_denominator,
            "optimizer_steps": step_denominator,
        }

    def _drain_optimization_metrics(self) -> dict[str, torch.Tensor]:
        r"""返回跨全部mini-epochs的per-asset优化均值与advantage标准差。"""

        if bool((self._optimization_count <= 0).any().item()):
            raise RuntimeError("optimization diagnostics did not cover every asset")
        count = self._optimization_count
        advantage_mean = self._optimization_sums["advantage"] / count
        advantage_variance = self._optimization_sums["advantage_square"] / count - advantage_mean.square()
        return_target_mean = self._optimization_sums["return_target"] / count
        return_target_variance = (
            self._optimization_sums["return_target_square"] / count - return_target_mean.square()
        ).clamp_min(0.0)
        value_prediction_mean = self._optimization_sums["value_prediction"] / count
        value_prediction_variance = (
            self._optimization_sums["value_prediction_square"] / count - value_prediction_mean.square()
        ).clamp_min(0.0)
        residual_mse = self._optimization_sums["value_residual_square"] / count
        explained_variance = torch.where(
            return_target_variance > 1.0e-12,
            1.0 - residual_mse / return_target_variance.clamp_min(1.0e-12),
            torch.zeros_like(return_target_variance),
        )  # $1-\operatorname{MSE}(V,R)/\operatorname{Var}(R)$，逐资产物理return口径
        result = {
            "advantage_mean": advantage_mean.detach().cpu(),
            "advantage_std": advantage_variance.clamp_min(0.0).sqrt().detach().cpu(),
            "value_error_mean": (self._optimization_sums["value_error"] / count).detach().cpu(),
            "return_target_mean": return_target_mean.detach().cpu(),
            "return_target_std": return_target_variance.sqrt().detach().cpu(),
            "value_prediction_mean": value_prediction_mean.detach().cpu(),
            "value_prediction_std": value_prediction_variance.sqrt().detach().cpu(),
            "value_error_physical_mean": (
                self._optimization_sums["value_error_physical"] / count
            ).detach().cpu(),
            "value_explained_variance": explained_variance.detach().cpu(),
            "value_clip_fraction": (self._optimization_sums["value_clip_fraction"] / count).detach().cpu(),
            "kl_per_active_dof": (self._optimization_sums["kl"] / count).detach().cpu(),
            "clip_fraction": (self._optimization_sums["clip_fraction"] / count).detach().cpu(),
            "action_rms": (self._optimization_sums["action_rms"] / count).detach().cpu(),
            "policy_mean_rms": (self._optimization_sums["policy_mean_rms"] / count).detach().cpu(),
            "policy_mean_near_bound_fraction": (
                self._optimization_sums["policy_mean_near_bound_fraction"] / count
            ).detach().cpu(),
            "film_modulation_rms": (self._optimization_sums["film_modulation_rms"] / count).detach().cpu(),
            **{
                name: (self._optimization_sums[name] / count).detach().cpu()
                for name in self._mechanism_metric_fields
            },
        }
        if self._gradient_probe_per_asset is not None:
            if any(value.shape != (self.asset_count,) for value in self._gradient_probe_per_asset.values()):
                raise RuntimeError("gradient probe per-asset metrics disagree with support cardinality")
            result.update(self._gradient_probe_per_asset)
        return result

    @staticmethod
    def _mean_fields(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, float]:
        r"""对同一cell或global的per-asset标量作资产等权均值。"""

        return {
            field: float(sum(float(row[field]) for row in rows) / len(rows))
            for field in fields
        }

    def _record_update_metrics(self, epoch_result: tuple[Any, ...]) -> None:
        r"""合并task/optimization/curriculum/resource事实并写global/cell/asset宽表。

        完整MVP80仍固定$1+8+80=89$行；single/few-embodiment closure只写当前支持集实际出现的cells，
        不伪造空cell统计。
        """

        vec_env: Any = self.vec_env  # Runner创建的PalmRotationRlGamesGpuEnv
        if vec_env is None or not hasattr(vec_env, "drain_rollout_metrics"):
            raise RuntimeError("palm-rotation vec env does not expose rollout diagnostics")
        rollout = vec_env.drain_rollout_metrics()  # per-asset post-physics facts`[A]`
        optimization = self._drain_optimization_metrics()  # per-asset optimizer facts`[A]`
        optimizer_scalars = self._drain_optimizer_scalars()  # global loss/sigma/gradient facts
        wrapper = getattr(vec_env, "env", None)  # PalmRotationRlGamesGpuEnv -> structured wrapper
        runtime = getattr(wrapper, "unwrapped", None)
        curriculum = getattr(runtime, HETERO_REWARD_RELEASE_STATE_ATTR, None)
        if not isinstance(curriculum, HeterogeneousRewardReleaseState):
            raise RuntimeError("reward-release curriculum state is unavailable for diagnostics")
        fields = tuple(rollout) + tuple(optimization)
        fields = tuple(field for field in fields if field != "sample_count")  # schema只记录均值，不复制count
        transitions = int(self.frame + self.curr_frames)  # 当前update完成后的nominal transition坐标

        # Asset rows保留formal dataset row与8-cell label；counterfactual ADR在ADR-0阶段恒为0。
        asset_rows: list[dict[str, Any]] = []
        cell_ids = curriculum.cell_ids_by_asset.detach().cpu()
        for asset_index in range(self.asset_count):
            row = {
                "update": int(self.epoch_num),
                "transitions": transitions,
                "scope": "asset",
                "scope_index": asset_index,
                "dataset_row": int(curriculum.dataset_rows_by_asset[asset_index]),
                "cell_id": int(cell_ids[asset_index].item()),
                "candidate_lambda": float(curriculum.asset_candidate_lambda[asset_index].item()),
                "actual_lambda": float(curriculum.cell_lambda[cell_ids[asset_index]].item()),
                "counterfactual_adr_level": 0.0,
            }
            row.update({field: float((rollout | optimization)[field][asset_index].item()) for field in fields})
            asset_rows.append(row)

        # Cell/global都由per-asset rows等权聚合，replica数量不能改变某只手的统计权重。
        aggregate_fields = fields + ("candidate_lambda", "actual_lambda", "counterfactual_adr_level")
        cell_rows: list[dict[str, Any]] = []
        active_cell_ids = sorted({int(row["cell_id"]) for row in asset_rows})  # subset实际覆盖的诊断cells
        for cell_id in active_cell_ids:
            members = [row for row in asset_rows if row["cell_id"] == cell_id]
            if self.asset_count == 80 and len(members) != 10:
                raise RuntimeError(f"MVP80 diagnostics expected 10 assets in cell {cell_id}, got {len(members)}")
            cell_rows.append(
                {
                    "update": int(self.epoch_num),
                    "transitions": transitions,
                    "scope": "cell",
                    "scope_index": cell_id,
                    "cell_id": cell_id,
                    **self._mean_fields(members, aggregate_fields),
                }
            )
        global_row = {
            "update": int(self.epoch_num),
            "transitions": transitions,
            "scope": "global",
            "scope_index": 0,
            **self._mean_fields(asset_rows, aggregate_fields),
            **optimizer_scalars,
        }
        if self._gradient_probe_global is not None:
            global_row.update(self._gradient_probe_global)  # pairwise/top-1/span及probe自身墙钟只属于global scope
        learning_rates = {str(group.get("name")): float(group["lr"]) for group in self.optimizer.param_groups}
        global_row["actor_base_lr"] = learning_rates["actor_base"]
        global_row["actor_residual_lr"] = (
            learning_rates["actor_global_residual"]
            if self.actor_arm not in {"direct", "direct_token"}
            else None
        )
        global_row["actor_contextual_lr"] = (
            learning_rates["actor_contextual_direct"]
            if self.actor_arm in {"direct", "direct_token"}
            else None
        )
        global_row["critic_lr"] = float(self.critic_optimizer.param_groups[0]["lr"])
        global_row.update(_linux_memory_snapshot())  # 同一update的host memory/swap evidence
        # rl_games tuple把纯environment step、完整rollout和PPO update分开，避免仅由总fps猜测Amdahl占比。
        global_row["environment_step_seconds"] = float(epoch_result[0])  # 30个`env.step()`累计墙钟
        global_row["rollout_policy_seconds"] = float(epoch_result[1])  # observation/policy/physics完整collection
        global_row["ppo_update_seconds"] = float(epoch_result[2])  # dataset prepare + 5 mini-epochs
        total_time = float(epoch_result[3])
        global_row["epoch_total_seconds"] = total_time  # collection + update，不含当前Parquet/TensorBoard发布
        global_row["steps_per_second"] = float(self.curr_frames / max(total_time, 1.0e-9))
        resource_violation: str | None = None  # 先记录unsafe水位，再在同一update边界停止
        if torch.cuda.is_available() and torch.device(self.ppo_device).type == "cuda":
            device = torch.device(self.ppo_device)  # PyTorch与PhysX共享的正式CUDA device
            peak_allocated = int(torch.cuda.max_memory_allocated(device))  # allocator历史active tensor峰值
            current_allocated = int(torch.cuda.memory_allocated(device))  # update结束仍被active tensors占用
            current_reserved = int(torch.cuda.memory_reserved(device))  # PyTorch caching allocator持有的总segment
            peak_reserved = int(torch.cuda.max_memory_reserved(device))  # allocator历史segment峰值
            driver_free, driver_total = (int(value) for value in torch.cuda.mem_get_info(device))  # 全device口径
            global_row.update(
                {
                    "gpu_memory_bytes": peak_allocated,  # schema-2兼容字段，语义固定为peak allocated
                    "gpu_memory_allocated_bytes": current_allocated,
                    "gpu_memory_reserved_bytes": current_reserved,
                    "gpu_peak_reserved_bytes": peak_reserved,
                    "gpu_driver_free_bytes": driver_free,
                    "gpu_driver_total_bytes": driver_total,
                }
            )
            if peak_allocated / driver_total >= float(self.config.get("gpu_memory_fraction_limit", 0.85)):
                resource_violation = (
                    "palm-rotation training exceeded PyTorch allocated safety fraction: "
                    f"peak={peak_allocated}, total={driver_total}"
                )
            minimum_driver_free = int(self.config.get("gpu_driver_free_memory_bytes_min", 0))
            if driver_free < minimum_driver_free:
                resource_violation = (
                    "palm-rotation training exhausted CUDA driver headroom: "
                    f"free={driver_free}, required={minimum_driver_free}, total={driver_total}, "
                    f"torch_reserved={current_reserved}"
                )
        self.metrics_recorder.record([global_row, *cell_rows, *asset_rows])  # 89 rows/update
        if resource_violation is not None:
            self.metrics_recorder.flush(reason="resource-safety")  # 保留故障前完整update但不生成新checkpoint
            raise RuntimeError(resource_violation)

        # TensorBoard只保存global与8-cell在线曲线；80-asset详情仅进入Parquet。
        if self.writer is not None:
            tensorboard_fields = [
                "reward_mean",
                "goal_count_mean",
                "frontier_count_mean",
                "net_turns_mean",
                "drop_rate",
                "kl_per_active_dof",
                "action_clamp_fraction",
                "film_modulation_rms",
            ]
            tensorboard_fields.append(
                "direct_mean_rms" if self.actor_arm in {"direct", "direct_token"} else "residual_rms"
            )
            for field in tensorboard_fields:
                self.writer.add_scalar(f"mvp80/global/{field}", global_row[field], transitions)
                for row in cell_rows:
                    self.writer.add_scalar(f"mvp80/cell_{row['cell_id']}/{field}", row[field], transitions)

    def train_epoch(self):
        r"""运行标准rollout/update，并在dataset释放前记录紧凑per-asset证据。"""

        if hasattr(self, "_resume_last_mean_rewards"):
            self.last_mean_rewards = float(self._resume_last_mean_rewards)  # 恢复被upstream train()重置的best门
            del self._resume_last_mean_rewards
        self._reset_optimization_metrics()
        result = super().train_epoch()
        self._record_update_metrics(result)
        return result

    def train(self):
        r"""执行rl_games训练循环，并在正常预算结束后发布单个metrics.parquet。"""

        result = super().train()
        self.metrics_recorder.finalize()
        return result

    def write_stats(
        self,
        total_time,
        epoch_num,
        step_time,
        play_time,
        update_time,
        actor_losses,
        critic_losses,
        entropies,
        kls,
        last_lr,
        lr_mul,
        frame,
        scaled_time,
        scaled_play_time,
        curr_frames,
    ) -> None:
        r"""沿用rl_games统计，并在每320 updates保存固定评估锚点checkpoint。

        ``ContinuousA2CBase.train``在调用本方法前已经把``self.frame``增加当前batch，因此这里保存的frame、
        两套optimizers、课程和Parquet cursor都对应完整update，而不是collection前状态。
        """

        super().write_stats(
            total_time,
            epoch_num,
            step_time,
            play_time,
            update_time,
            actor_losses,
            critic_losses,
            entropies,
            kls,
            last_lr,
            lr_mul,
            frame,
            scaled_time,
            scaled_play_time,
            curr_frames,
        )
        cadence = int(self.config.get("evaluation_frequency", 320))
        if cadence > 0 and int(epoch_num) % cadence == 0:
            path = f"{self.nn_dir}/evaluation_{self.config['name']}_ep_{int(epoch_num):05d}"
            self.save(path)  # full identity/model/dual-optimizer/curriculum/diagnostic state

    def get_full_state_weights(self) -> dict[str, Any]:
        r"""保存模型、两套optimizer、课程、诊断与可精确续接的随机/调度状态。"""

        self.metrics_recorder.flush(reason="checkpoint")  # checkpoint不得领先于durable metric rows
        state = super().get_full_state_weights()  # model、actor optimizer、normalizer、env state、identity
        state[CRITIC_OPTIMIZER_KEY] = self.critic_optimizer.state_dict()  # 独立critic Adam moments
        state[DIAGNOSTICS_RECORDER_KEY] = self.metrics_recorder.state_dict()  # shard inventory/append cursor
        state[TRAINING_CONTINUATION_KEY] = {
            "schema_version": "1.0.0",
            "last_lr": float(self.last_lr),  # adaptive scheduler下一update的base LR
            "entropy_coef": float(self.entropy_coef),  # scheduler可能共同修改的exploration权重
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }  # pickle-safe完整随机状态；rank-0 reset无噪声，但Normal action sampling仍依赖Torch RNG
        return state

    def save(self, filename: str) -> None:
        r"""在同一文件系统以temporary→replace原子发布完整checkpoint。

        Args:
            filename (str): rl_games传入的不含``.pth``目标路径。
        """

        destination = Path(filename if filename.endswith(".pth") else f"{filename}.pth")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")  # `<name>.pth.tmp`
        if temporary.exists():
            temporary.unlink()  # 只清理当前目标上次未发布的run-owned temporary
        state = self.get_full_state_weights()  # 先flush Parquet，再冻结同一update checkpoint state
        torch.save(state, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())  # rename前确保checkpoint bytes已提交到底层文件系统
        temporary.replace(destination)

    def set_full_state_weights(self, weights: dict[str, Any], set_epoch: bool = True) -> None:
        r"""在identity gate通过后恢复model、actor optimizer、课程及critic optimizer。"""

        if CRITIC_OPTIMIZER_KEY not in weights:
            raise RuntimeError("palm-rotation checkpoint is missing independent critic optimizer state")
        if DIAGNOSTICS_RECORDER_KEY not in weights:
            raise RuntimeError("palm-rotation checkpoint is missing metrics recorder state")
        continuation = weights.get(TRAINING_CONTINUATION_KEY)
        if not isinstance(continuation, Mapping) or continuation.get("schema_version") != "1.0.0":
            raise RuntimeError("palm-rotation checkpoint is missing exact training continuation state")
        super().set_full_state_weights(weights, set_epoch=set_epoch)  # 先执行AnyMani identity验证
        self.critic_optimizer.load_state_dict(weights[CRITIC_OPTIMIZER_KEY])  # 精确恢复critic Adam moments
        self.metrics_recorder.load_state_dict(weights[DIAGNOSTICS_RECORDER_KEY])  # 核对durable Parquet shards
        self.last_lr = float(continuation["last_lr"])  # scheduler scalar不能只依赖optimizer param-group LR
        self.entropy_coef = float(continuation["entropy_coef"])
        self.update_lr(self.last_lr)  # 三参数组恢复与adaptive ratio一致的当前LR
        random.setstate(continuation["python_random_state"])
        np.random.set_state(continuation["numpy_random_state"])
        torch.set_rng_state(continuation["torch_cpu_rng_state"])
        cuda_states = continuation.get("torch_cuda_rng_states", [])
        if torch.cuda.is_available() and cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        self._resume_last_mean_rewards = float(weights.get("last_mean_rewards", -1.0e9))


class PalmRotationPpoRunner(AnyManiMaskedRunner):
    r"""在进程局部Runner factories中注册MVP80 custom PPO。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        r"""保留现有masked model/player并增加双optimizer algorithm factory。"""

        super().__init__(*args, **kwargs)
        self.algo_factory.register_builder(
            PALM_ROTATION_PPO_ALGO,
            lambda **factory_kwargs: PalmRotationPpoAgent(**factory_kwargs),
        )
        self.player_factory.register_builder(
            PALM_ROTATION_PPO_ALGO,
            lambda **factory_kwargs: AnyManiMaskedPpoPlayer(**factory_kwargs),
        )


def register_palm_rotation_ppo() -> None:
    r"""注册custom masked model与MVP structured network builder。"""

    register_anymani_masked_ppo()  # `anymani_masked_continuous`及shared player contract
    model_builder.register_model("anymani_palm_rotation_masked_continuous", PalmRotationMaskedContinuousModel)
    model_builder.register_network(PALM_ROTATION_NETWORK, PalmRotationRlGamesBuilder)


__all__ = [
    "CRITIC_OPTIMIZER_KEY",
    "DIAGNOSTICS_RECORDER_KEY",
    "PALM_ROTATION_NETWORK",
    "PALM_ROTATION_PPO_ALGO",
    "TRAINING_CONTINUATION_KEY",
    "bounded_adaptive_learning_rate",
    "denormalize_value_readonly",
    "normalize_advantages_per_asset",
    "validate_gradient_probe_compile_compatibility",
    "PalmRotationPpoAgent",
    "PalmRotationPpoRunner",
    "PalmRotationMaskedContinuousModel",
    "PalmRotationRlGamesBuilder",
    "PalmRotationRlGamesNetwork",
    "register_palm_rotation_ppo",
    "stratified_asset_permutation",
]
