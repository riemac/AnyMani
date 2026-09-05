r"""掌旋PPO的只读梯度探针与证据发布。

探针从调用方给出的同一minibatch计算逐资产目标梯度；autograd.grad不写主parameter.grad。完整参数坐标
不等于完整rollout样本总体，后者由每份产物显式记录。探针是否启用由agent的cadence控制，本模块不选择
优化方法、不执行optimizer，也不把负余弦自动解释为可修复的任务冲突。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from ..algorithms.gradient_audit import compute_actor_gradient_scope_audit, per_asset_replica_half_gradients

if TYPE_CHECKING:
    from ..palm_rotation_ppo import PalmRotationPpoAgent


def gradient_probe_parameters(agent: PalmRotationPpoAgent) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
    r"""返回动作末层与value scalar末层参数，定义低成本task-gradient proxy。

    Full actor/critic参数Gram对A128需每次执行256次完整反向，成本不可接受。这里固定读取最接近输出的
    可训练层：residual/base arm取base与residual两个末层，direct取direct末层；Critic取scalar value末层。
    该proxy只用于判断是否值得启动更昂贵的offline/full-parameter probe，不冒充完整梯度。
    """

    network = agent.model.a2c_network
    actor = network.package.actor
    if agent.actor_arm in {"direct", "direct_token"}:
        actor_modules = (actor.direct_head[-1],)  # type: ignore[union-attr]
    else:
        actor_modules = (actor.base_head[-1], actor.residual_head[-1])  # type: ignore[union-attr]
    critic_modules = (network.package.critic.value_head[-1],)
    actor_parameters = tuple(parameter for module in actor_modules for parameter in module.parameters())
    critic_parameters = tuple(parameter for module in critic_modules for parameter in module.parameters())
    return actor_parameters, critic_parameters


def per_asset_gradient_matrix(
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


def run_gradient_probe(
    agent: PalmRotationPpoAgent,
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
    actor_parameters, critic_parameters = agent._gradient_probe_parameters()
    actor_gradients = agent._per_asset_gradient_matrix(
        actor_objective,
        labels,
        actor_parameters,
        asset_count=agent.asset_count,
    )
    critic_gradients = agent._per_asset_gradient_matrix(
        critic_objective,
        labels,
        critic_parameters,
        asset_count=agent.asset_count,
    )
    per_asset: dict[str, torch.Tensor] = {}
    global_metrics: dict[str, float] = {}
    dense: dict[str, np.ndarray] = {}
    for prefix, gradients in (("actor", actor_gradients), ("critic", critic_gradients)):
        gram = gradients @ gradients.T  # $\Gamma=GG^T\in\mathbb R^{A\times A}$
        norms = torch.linalg.vector_norm(gradients, dim=-1)  # $\|g_i\|_2$
        denominator = norms[:, None] * norms[None, :]
        cosine = torch.where(denominator > 1.0e-20, gram / denominator.clamp_min(1.0e-20), torch.zeros_like(gram))
        off_diagonal = ~torch.eye(agent.asset_count, dtype=torch.bool, device=gram.device)
        pair_values = cosine[torch.triu(off_diagonal, diagonal=1)]
        mean_cosine = cosine.masked_fill(~off_diagonal, 0.0).sum(dim=1) / max(agent.asset_count - 1, 1)
        positive_min = (
            norms[norms > 1.0e-12].min() if bool((norms > 1.0e-12).any().item()) else norms.new_tensor(1.0e-12)
        )
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
    agent._gradient_probe_per_asset = per_asset
    agent._gradient_probe_global = global_metrics

    root = Path(agent.experiment_dir) / "gradient_probes"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"update_{int(agent.epoch_num):06d}.npz"
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray("head-gradient-gram-v1"),
        update=np.asarray(int(agent.epoch_num), dtype=np.int64),
        asset_index=np.arange(agent.asset_count, dtype=np.int64),
        **dense,
    )
    temporary.replace(destination)  # dense artifact只在完整写入后可见


def run_full_actor_gradient_shadow(
    agent: PalmRotationPpoAgent,
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
    actor_parameters = tuple(agent.model.a2c_network.package.actor.parameters())  # 完整共享Actor坐标，含global logstd
    if not actor_parameters:
        raise RuntimeError("full Actor gradient shadow received an empty parameter set")
    scope_audits: dict[str, dict[str, torch.Tensor]] = {}
    dense: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "schema_version": "full-actor-gradient-shadow-v1",
        "update": int(agent.epoch_num),
        "asset_count": agent.asset_count,
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
            asset_count=agent.asset_count,
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
        torch.linalg.vector_norm(global_gradients, dim=-1) * torch.linalg.vector_norm(per_asset_gradients, dim=-1)
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

    root = Path(agent.experiment_dir) / "full_gradient_shadows"
    root.mkdir(parents=True, exist_ok=True)
    array_destination = root / f"update_{int(agent.epoch_num):06d}.npz"
    array_temporary = array_destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        array_temporary,
        update=np.asarray(int(agent.epoch_num), dtype=np.int64),
        asset_index=np.arange(agent.asset_count, dtype=np.int64),
        **dense,
    )
    array_temporary.replace(array_destination)  # dense gradients/Gram仅完整压缩后原子可见
    summary_destination = root / f"update_{int(agent.epoch_num):06d}.json"
    summary_temporary = summary_destination.with_suffix(".tmp.json")
    summary_temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_temporary.replace(summary_destination)
