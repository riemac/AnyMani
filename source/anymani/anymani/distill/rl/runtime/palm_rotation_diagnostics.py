'Reduce Actor, Critic, and task measurements that are already present in the rollout.'

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
    'Handle linux memory snapshot.'

    status: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmSwap:")):
            name, value, _unit = line.split()
            status[name.rstrip(":")] = int(value) * 1024
    available = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    return {
        "process_rss_bytes": status.get("VmRSS", 0),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "process_swap_bytes": status.get("VmSwap", 0),
        "system_available_memory_bytes": available,
    }


class PalmRotationPpoDiagnostics(PpoDiagnostics):
    'Contract for PALM rotation PPO diagnostics.'

    def mini_batch(self, agent: Any, batch: Mapping[str, Any], e_clip: float, minibatch: int) -> None:
        'Handle mini batch.'

        _ = (agent, minibatch)
        with torch.no_grad():
            values = batch["values"].detach()  # shapes [M,1]; units M
            returns = batch["returns"].detach()  # shapes [M,1]; units M
            new_neglogp = batch["new_neglogp"].detach()  # shapes [M]; units M
            old_neglogp = batch["old_neglogp"].detach()  # shapes [M]; units M
            masks = batch["masks"]
            exp_var = torch_ext.explained_variance(values, returns, masks)
            clip_frac = torch_ext.policy_clip_fraction(new_neglogp, old_neglogp, e_clip, masks)  # scalar
            self.exp_vars.append(exp_var.detach())
            self.clip_fracs.append(clip_frac.detach())


def rollout_policy_mechanism_metrics(
    rollout_mean: torch.Tensor,
    mechanism: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    actor_arm: Literal["base", "residual", "direct", "direct_token"],
) -> dict[str, torch.Tensor]:
    'Handle rollout policy mechanism metrics; shapes [M,J], [str,torch.Tensor], [M]; units M.'

    if rollout_mean.ndim != 2 or mechanism.shape != rollout_mean.shape or active_mask.shape != rollout_mean.shape:
        raise RuntimeError(
            "rollout mechanism tensors must share [M,J] shape: "
            f"mean={tuple(rollout_mean.shape)}, mechanism={tuple(mechanism.shape)}, mask={tuple(active_mask.shape)}"
        )
    if actor_arm not in {"base", "residual", "direct", "direct_token"}:
        raise ValueError(f"unsupported rollout mechanism arm: {actor_arm!r}")

    active_float = active_mask.to(dtype=rollout_mean.dtype)  # shapes [M,J]; units M
    active_count = active_float.sum(dim=-1).clamp_min(1.0)  # shapes [M]; units M
    policy_mean_rms = torch.sqrt((rollout_mean.square() * active_float).sum(dim=-1) / active_count)
    near_bound = ((rollout_mean.abs() >= 0.95).to(dtype=rollout_mean.dtype) * active_float).sum(dim=-1) / active_count
    metrics = {
        "policy_mean_rms": policy_mean_rms,
        "policy_mean_near_bound_fraction": near_bound,
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
                ),
            }
        )
        return metrics

    residual_rms = torch.sqrt((mechanism.square() * active_float).sum(dim=-1) / active_count)
    base_mean = rollout_mean - mechanism
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
    'Handle drain optimizer scalars.'

    if agent._optimizer_step_count < 1 or agent._optimizer_microbatch_count < 1:
        raise RuntimeError("optimizer scalar diagnostics observed no update steps")
    if agent._gradient_microbatch_index % agent._gradient_accumulation_steps != 0:
        raise RuntimeError("PPO update ended inside a logical accumulated minibatch")
    microbatch_denominator = float(agent._optimizer_microbatch_count)
    step_denominator = float(agent._optimizer_step_count)
    result = {
        **{
            name: float((agent._optimizer_scalar_sums[name] / microbatch_denominator).item())
            for name in (
                "actor_loss", "critic_loss", "entropy", "policy_sigma", "policy_base_sigma", "recovery_floor_fraction",
                "actor_rejected_action_cost", "actor_rejected_action_fraction",
            )
        },
        **{
            name: float((agent._optimizer_scalar_sums[name] / step_denominator).item())
            for name in ("actor_grad_norm", "critic_grad_norm")
        },
        "optimizer_microbatches": microbatch_denominator,
        "optimizer_steps": step_denominator,
    }
    if getattr(agent, "gradient_aggregation", "mean") == "cagrad":
        for side in ("actor", "critic"):
            for name in ("relative_gap", "worst_projection", "iterations"):
                key = f"{side}_cagrad_{name}"
                result[key] = float((agent._optimizer_scalar_sums[key] / step_denominator).item())
    if getattr(agent, "student_mode", False):
        stats = getattr(agent, "student_stats", None)
        if stats is None:
            raise RuntimeError("student diagnostics require student stats")
        result.update(
            {
                "student_rollout_updates": float(stats.rollout_updates),
                "student_critic_optimizer_steps": float(stats.critic_optimizer_steps),
                "student_actor_optimizer_steps": float(stats.actor_optimizer_steps),
                "student_warmup_rollouts": float(stats.warmup_rollouts),
                "student_warmup_critic_optimizer_steps": float(stats.warmup_critic_optimizer_steps),
                "student_anchor_loss": float(
                    (agent._optimizer_scalar_sums["student_anchor_loss"] / step_denominator).item()
                ),
                "student_anchor_weighted_loss": float(
                    (agent._optimizer_scalar_sums["student_anchor_weighted_loss"] / step_denominator).item()
                ),
                "student_fk_loss": float(
                    (agent._optimizer_scalar_sums["student_fk_loss"] / microbatch_denominator).item()
                ),
                "student_fk_weighted_loss": float(
                    (agent._optimizer_scalar_sums["student_fk_weighted_loss"] / microbatch_denominator).item()
                ),
            }
        )
    normalizer = getattr(agent, "value_mean_std", None)
    compensation = getattr(agent, "_last_popart_compensation", None)
    if compensation is not None:
        assert normalizer is not None
        result["popart_weight_error"] = float(compensation["weight_error"].item())
        result["popart_bias_error"] = float(compensation["bias_error"].item())
        result["popart_count"] = float(normalizer.count.item())
    return result


def drain_optimization_metrics(agent: PalmRotationPpoAgent) -> dict[str, torch.Tensor]:
    'Handle drain optimization metrics.'

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
    )
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
    'Handle mean fields.'

    return {field: float(sum(float(row[field]) for row in rows) / len(rows)) for field in fields}


def _print_live_metrics(agent: PalmRotationPpoAgent, global_row: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    'Handle print live metrics.'
    if int(agent.epoch_num) == 1:
        print("[METRICS] live_net=visited training-state turns; end_net=completed episode turns; not fixed30/R16 evaluation.", flush=True)
        print("[METRICS] drop/end and axis/end are separate terminal-event fractions; checkpoints contain completed updates only.", flush=True)
        print("[FIRST30] per-asset recent windows: median turns/goals/direction; asset-equal safety; training proxies, not frozen R16.", flush=True)
    free_gib = float(global_row.get("gpu_driver_free_bytes", 0)) / 2**30
    print(
        f"\n[UPDATE {agent.epoch_num:04d}/{agent.max_epochs}] frames={global_row['transitions']/1e6:.3f}M "
        f"wall={global_row['epoch_total_seconds']:.2f}s "
        f"rollout={global_row['rollout_policy_seconds']:.2f}s opt={global_row['ppo_update_seconds']:.2f}s "
        f"sigma={global_row['policy_sigma']:.3f} GPU-free={free_gib:.2f}GiB",
        flush=True,
    )
    split = int(agent.config.get("console_family_split", 0))
    if split and len(rows) != 2 * split:
        raise ValueError("console family split must exactly partition the declared frozen asset axis")
    groups = [("LEAP", rows[:split]), ("Allegro", rows[split:])] if split else [("ALL", rows)]
    for label, group in groups:
        current = mean_fields(group, ("reward_mean", "net_turns_mean", "goal_count_mean"))
        count = sum(row["completed_episode_count"] for row in group)
        if count:
            terminal = {
                key: sum(row["completed_episode_count"] * row[key] for row in group) / count
                for key in ("terminal_net_turns_mean", "terminal_drop_rate", "terminal_axis_failure_rate", "terminal_timeout_rate")
            }
            ending = (f"end_net={terminal['terminal_net_turns_mean']:.3f} "
                      f"drop/end={terminal['terminal_drop_rate']:.1%} axis/end={terminal['terminal_axis_failure_rate']:.1%} "
                      f"timeout/end={terminal['terminal_timeout_rate']:.1%}")
        else:
            ending = "end_net=-- (no completed episodes this update)"
        print(f"  {label:7s} reward/step={current['reward_mean']:+.4f} live_net={current['net_turns_mean']:.3f} "
               f"live_goal={current['goal_count_mean']:.2f} ended={int(count)} {ending}", flush=True)
        evidence = getattr(getattr(agent.vec_env, "env", None), "training_evidence", None)
        tracker = getattr(evidence, "first30_statistics", None)
        if tracker is not None:
            window = tracker.summary([int(row["scope_index"]) for row in group])
            formatted = {
                name: "--" if window[name] is None else f"{window[name]:.3f}"
                for name in ("first30_net_median", "first30_goal_median", "first30_direction_median", "first30_safe_fraction")
            }
            print(
                f"  {label:7s} FIRST30 net_med={formatted['first30_net_median']} turns "
                f"goals_med={formatted['first30_goal_median']} direction={formatted['first30_direction_median']} "
                f"safe={formatted['first30_safe_fraction']} assets={window['first30_observed_assets']}/{window['first30_asset_count']} "
                f"windows={window['first30_window_count']} proxy_1turn={window['first30_one_turn_proxy_assets']} "
                f"proxy_2turn={window['first30_two_turn_proxy_assets']}", flush=True,
            )


def record_update_metrics(agent: PalmRotationPpoAgent, epoch_result: tuple[Any, ...]) -> None:
    'Record update metrics.'

    vec_env: Any = agent.vec_env
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
    fields = tuple(field for field in fields if field != "sample_count")
    transitions = int(agent.frame + agent.curr_frames)


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


    aggregate_fields = fields + ("candidate_lambda", "actual_lambda", "counterfactual_adr_level")
    cell_rows: list[dict[str, Any]] = []
    active_cell_ids = sorted({int(row["cell_id"]) for row in asset_rows})
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


    evidence = getattr(wrapper, "training_evidence", None)
    first30 = getattr(evidence, "first30_statistics", None)
    if first30 is not None:
        per_asset = first30.per_asset()
        for row in asset_rows:
            row.update(per_asset[int(row["scope_index"])])
        for row in cell_rows:
            indices = [int(asset["scope_index"]) for asset in asset_rows if asset["cell_id"] == row["cell_id"]]
            row.update(first30.summary(indices))
        global_row.update(first30.summary())
    if agent._gradient_probe_global is not None:
        global_row.update(agent._gradient_probe_global)
    learning_rates = {str(group.get("name")): float(group["lr"]) for group in agent.optimizer.param_groups}
    global_row["actor_base_lr"] = learning_rates["actor_base"]
    global_row["actor_residual_lr"] = (
        learning_rates["actor_global_residual"] if agent.actor_arm not in {"direct", "direct_token"} else None
    )
    global_row["actor_contextual_lr"] = (
        learning_rates.get("actor_contextual_direct", learning_rates["actor_base"])
        if agent.actor_arm in {"direct", "direct_token"}
        else None
    )
    global_row["critic_lr"] = float(agent.critic_optimizer.param_groups[0]["lr"])
    global_row.update(_linux_memory_snapshot())

    global_row["environment_step_seconds"] = float(epoch_result[0])
    global_row["rollout_policy_seconds"] = float(epoch_result[1])
    global_row["ppo_update_seconds"] = float(epoch_result[2])  # dataset prepare + 5 mini-epochs
    total_time = float(epoch_result[3])
    global_row["epoch_total_seconds"] = total_time
    global_row["steps_per_second"] = float(agent.curr_frames / max(total_time, 1.0e-9))
    resource_violation: str | None = None
    if torch.cuda.is_available() and torch.device(agent.ppo_device).type == "cuda":
        device = torch.device(agent.ppo_device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        current_allocated = int(torch.cuda.memory_allocated(device))
        current_reserved = int(torch.cuda.memory_reserved(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        driver_free, driver_total = (int(value) for value in torch.cuda.mem_get_info(device))
        global_row.update(
            {
                "gpu_memory_bytes": peak_allocated,
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
    cadence = int(agent.config.get("console_metrics_frequency", 0))
    if cadence > 0 and int(agent.epoch_num) % cadence == 0:
        _print_live_metrics(agent, global_row, asset_rows)
    if resource_violation is not None:
        agent.metrics_recorder.flush(reason="resource-safety")
        raise RuntimeError(resource_violation)


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
        if first30 is not None:
            tensorboard_fields.extend(("first30_net_median", "first30_goal_median", "first30_direction_median", "first30_safe_fraction"))
        for field in tensorboard_fields:
            if global_row[field] is not None:
                agent.writer.add_scalar(f"mvp80/global/{field}", global_row[field], transitions)
            for row in cell_rows:
                if row[field] is not None:
                    agent.writer.add_scalar(f"mvp80/cell_{row['cell_id']}/{field}", row[field], transitions)
