r"""掌旋PPO运行事实的归约与发布适配。

环境侧已经冻结的post-physics/pre-reset事实、优化器侧累计量和资源水位在update边界汇合。函数读取agent
拥有的张量和计数器，将资产等权统计交给diagnostics.recording；不重算reward，不改变梯度或optimizer更新。
梯度探针的额外求导与文件发布独立位于palm_rotation_probes，主训练路径可将其完全关闭。
"""

from __future__ import annotations

import resource
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from rl_games.algos_torch import torch_ext
from rl_games.common.diagnostics import PpoDiagnostics

from anymani.tasks.hetero.mdp.curriculum_state import (
    HETERO_REWARD_RELEASE_STATE_ATTR,
    HeterogeneousRewardReleaseState,
)

if TYPE_CHECKING:
    from ..palm_rotation_ppo import PalmRotationPpoAgent


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


def drain_optimizer_scalars(agent: PalmRotationPpoAgent) -> dict[str, float]:
    r"""返回当前update跨全部minibatches/mini-epochs的global优化统计。"""

    if agent._optimizer_step_count < 1 or agent._optimizer_microbatch_count < 1:
        raise RuntimeError("optimizer scalar diagnostics observed no update steps")
    if agent._gradient_microbatch_index % agent._gradient_accumulation_steps != 0:
        raise RuntimeError("PPO update ended inside a logical accumulated minibatch")
    microbatch_denominator = float(agent._optimizer_microbatch_count)  # 正式$16\times5=80$
    step_denominator = float(agent._optimizer_step_count)  # 正式$(16/4)\times5=20$
    return {
        **{
            name: float((agent._optimizer_scalar_sums[name] / microbatch_denominator).item())
            for name in ("actor_loss", "critic_loss", "entropy", "policy_sigma")
        },
        **{
            name: float((agent._optimizer_scalar_sums[name] / step_denominator).item())
            for name in ("actor_grad_norm", "critic_grad_norm")
        },
        "optimizer_microbatches": microbatch_denominator,
        "optimizer_steps": step_denominator,
    }


def drain_optimization_metrics(agent: PalmRotationPpoAgent) -> dict[str, torch.Tensor]:
    r"""返回跨全部mini-epochs的per-asset优化均值与advantage标准差。"""

    if bool((agent._optimization_count <= 0).any().item()):
        raise RuntimeError("optimization diagnostics did not cover every asset")
    count = agent._optimization_count
    advantage_mean = agent._optimization_sums["advantage"] / count
    advantage_variance = agent._optimization_sums["advantage_square"] / count - advantage_mean.square()
    return_target_mean = agent._optimization_sums["return_target"] / count
    return_target_variance = (
        agent._optimization_sums["return_target_square"] / count - return_target_mean.square()
    ).clamp_min(0.0)
    value_prediction_mean = agent._optimization_sums["value_prediction"] / count
    value_prediction_variance = (
        agent._optimization_sums["value_prediction_square"] / count - value_prediction_mean.square()
    ).clamp_min(0.0)
    residual_mse = agent._optimization_sums["value_residual_square"] / count
    explained_variance = torch.where(
        return_target_variance > 1.0e-12,
        1.0 - residual_mse / return_target_variance.clamp_min(1.0e-12),
        torch.zeros_like(return_target_variance),
    )  # $1-\operatorname{MSE}(V,R)/\operatorname{Var}(R)$，逐资产物理return口径
    result = {
        "advantage_mean": advantage_mean.detach().cpu(),
        "advantage_std": advantage_variance.clamp_min(0.0).sqrt().detach().cpu(),
        "value_error_mean": (agent._optimization_sums["value_error"] / count).detach().cpu(),
        "return_target_mean": return_target_mean.detach().cpu(),
        "return_target_std": return_target_variance.sqrt().detach().cpu(),
        "value_prediction_mean": value_prediction_mean.detach().cpu(),
        "value_prediction_std": value_prediction_variance.sqrt().detach().cpu(),
        "value_error_physical_mean": (agent._optimization_sums["value_error_physical"] / count).detach().cpu(),
        "value_explained_variance": explained_variance.detach().cpu(),
        "value_clip_fraction": (agent._optimization_sums["value_clip_fraction"] / count).detach().cpu(),
        "kl_per_active_dof": (agent._optimization_sums["kl"] / count).detach().cpu(),
        "clip_fraction": (agent._optimization_sums["clip_fraction"] / count).detach().cpu(),
        "action_rms": (agent._optimization_sums["action_rms"] / count).detach().cpu(),
        "policy_mean_rms": (agent._optimization_sums["policy_mean_rms"] / count).detach().cpu(),
        "policy_mean_near_bound_fraction": (agent._optimization_sums["policy_mean_near_bound_fraction"] / count)
        .detach()
        .cpu(),
        "film_modulation_rms": (agent._optimization_sums["film_modulation_rms"] / count).detach().cpu(),
        **{name: (agent._optimization_sums[name] / count).detach().cpu() for name in agent._mechanism_metric_fields},
    }
    if agent._gradient_probe_per_asset is not None:
        if any(value.shape != (agent.asset_count,) for value in agent._gradient_probe_per_asset.values()):
            raise RuntimeError("gradient probe per-asset metrics disagree with support cardinality")
        result.update(agent._gradient_probe_per_asset)
    return result


def mean_fields(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, float]:
    r"""对同一cell或global的per-asset标量作资产等权均值。"""

    return {field: float(sum(float(row[field]) for row in rows) / len(rows)) for field in fields}


def record_update_metrics(agent: PalmRotationPpoAgent, epoch_result: tuple[Any, ...]) -> None:
    r"""合并task/optimization/curriculum/resource事实并写global/cell/asset宽表。

    完整MVP80仍固定$1+8+80=89$行；single/few-embodiment closure只写当前支持集实际出现的cells，
    不伪造空cell统计。
    """

    vec_env: Any = agent.vec_env  # Runner创建的PalmRotationRlGamesGpuEnv
    if vec_env is None or not hasattr(vec_env, "drain_rollout_metrics"):
        raise RuntimeError("palm-rotation vec env does not expose rollout diagnostics")
    rollout = vec_env.drain_rollout_metrics()  # per-asset post-physics facts`[A]`
    optimization = agent._drain_optimization_metrics()  # per-asset optimizer facts`[A]`
    optimizer_scalars = agent._drain_optimizer_scalars()  # global loss/sigma/gradient facts
    wrapper = getattr(vec_env, "env", None)  # PalmRotationRlGamesGpuEnv -> structured wrapper
    runtime = getattr(wrapper, "unwrapped", None)
    curriculum = getattr(runtime, HETERO_REWARD_RELEASE_STATE_ATTR, None)
    if not isinstance(curriculum, HeterogeneousRewardReleaseState):
        raise RuntimeError("reward-release curriculum state is unavailable for diagnostics")
    fields = tuple(rollout) + tuple(optimization)
    fields = tuple(field for field in fields if field != "sample_count")  # schema只记录均值，不复制count
    transitions = int(agent.frame + agent.curr_frames)  # 当前update完成后的nominal transition坐标

    # Asset rows保留formal dataset row与8-cell label；counterfactual ADR在ADR-0阶段恒为0。
    asset_rows: list[dict[str, Any]] = []
    cell_ids = curriculum.cell_ids_by_asset.detach().cpu()
    for asset_index in range(agent.asset_count):
        row = {
            "update": int(agent.epoch_num),
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
        if agent.asset_count == 80 and len(members) != 10:
            raise RuntimeError(f"MVP80 diagnostics expected 10 assets in cell {cell_id}, got {len(members)}")
        cell_rows.append(
            {
                "update": int(agent.epoch_num),
                "transitions": transitions,
                "scope": "cell",
                "scope_index": cell_id,
                "cell_id": cell_id,
                **agent._mean_fields(members, aggregate_fields),
            }
        )
    global_row = {
        "update": int(agent.epoch_num),
        "transitions": transitions,
        "scope": "global",
        "scope_index": 0,
        **agent._mean_fields(asset_rows, aggregate_fields),
        **optimizer_scalars,
    }
    if agent._gradient_probe_global is not None:
        global_row.update(agent._gradient_probe_global)  # pairwise/top-1/span及probe自身墙钟只属于global scope
    learning_rates = {str(group.get("name")): float(group["lr"]) for group in agent.optimizer.param_groups}
    global_row["actor_base_lr"] = learning_rates["actor_base"]
    global_row["actor_residual_lr"] = (
        learning_rates["actor_global_residual"] if agent.actor_arm not in {"direct", "direct_token"} else None
    )
    global_row["actor_contextual_lr"] = (
        learning_rates["actor_contextual_direct"] if agent.actor_arm in {"direct", "direct_token"} else None
    )
    global_row["critic_lr"] = float(agent.critic_optimizer.param_groups[0]["lr"])
    global_row.update(_linux_memory_snapshot())  # 同一update的host memory/swap evidence
    # rl_games tuple把纯environment step、完整rollout和PPO update分开，避免仅由总fps猜测Amdahl占比。
    global_row["environment_step_seconds"] = float(epoch_result[0])  # 30个`env.step()`累计墙钟
    global_row["rollout_policy_seconds"] = float(epoch_result[1])  # observation/policy/physics完整collection
    global_row["ppo_update_seconds"] = float(epoch_result[2])  # dataset prepare + 5 mini-epochs
    total_time = float(epoch_result[3])
    global_row["epoch_total_seconds"] = total_time  # collection + update，不含当前Parquet/TensorBoard发布
    global_row["steps_per_second"] = float(agent.curr_frames / max(total_time, 1.0e-9))
    resource_violation: str | None = None  # 先记录unsafe水位，再在同一update边界停止
    if torch.cuda.is_available() and torch.device(agent.ppo_device).type == "cuda":
        device = torch.device(agent.ppo_device)  # PyTorch与PhysX共享的正式CUDA device
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
        if peak_allocated / driver_total >= float(agent.config.get("gpu_memory_fraction_limit", 0.85)):
            resource_violation = (
                "palm-rotation training exceeded PyTorch allocated safety fraction: "
                f"peak={peak_allocated}, total={driver_total}"
            )
        minimum_driver_free = int(agent.config.get("gpu_driver_free_memory_bytes_min", 0))
        if driver_free < minimum_driver_free:
            resource_violation = (
                "palm-rotation training exhausted CUDA driver headroom: "
                f"free={driver_free}, required={minimum_driver_free}, total={driver_total}, "
                f"torch_reserved={current_reserved}"
            )
    agent.metrics_recorder.record([global_row, *cell_rows, *asset_rows])  # 89 rows/update
    if resource_violation is not None:
        agent.metrics_recorder.flush(reason="resource-safety")  # 保留故障前完整update但不生成新checkpoint
        raise RuntimeError(resource_violation)

    # TensorBoard只保存global与8-cell在线曲线；80-asset详情仅进入Parquet。
    if agent.writer is not None:
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
            "direct_mean_rms" if agent.actor_arm in {"direct", "direct_token"} else "residual_rms"
        )
        for field in tensorboard_fields:
            agent.writer.add_scalar(f"mvp80/global/{field}", global_row[field], transitions)
            for row in cell_rows:
                agent.writer.add_scalar(f"mvp80/cell_{row['cell_id']}/{field}", row[field], transitions)
