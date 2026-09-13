#!/usr/bin/env python3
r"""冻结检查点下的短时配对动作响应，不改变任务时域或执行训练。

沿用checkpoint原有256资产、4副本、rank0和120s任务合同，每条件只观察20个策略步后主动停止采样。
零动作、真实反馈策略、固定初始mean正负方向和4个Rademacher正负方向分别从记录初态开始。只有记录初态
相符且两条first trajectory均完整存活的正负对，才形成局部符号contrast；阴性不证明30s旋转循环不可行。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audit_palm_rotation_checkpoint_gradients import _atomic_json, _sha256, run_checkpoint_audit


def _action_conditions(
    initial_mean: torch.Tensor, mask: torch.Tensor, *, amplitude: float, seed: int
) -> dict[str, torch.Tensor | None]:
    r"""构造12条有界条件；None只表示每步重新计算的真实反馈策略。

    固定mean使用原始动作单位，随机方向使用$\alpha v$，$v_j\in\{-1,1\}$；ghost恒为0。随机方向在
    canonical joint轴上共享，避免每个家族采用不同方向集合。局部CPU generator不改变环境随机流。
    """

    if initial_mean.ndim != 2 or mask.shape != initial_mean.shape or mask.dtype != torch.bool:
        raise ValueError("initial mean and bool mask must share [environment,joint] shape")
    if (
        not 0.0 < amplitude <= 1.0
        or not bool(torch.isfinite(initial_mean).all())
        or bool((initial_mean.abs() > 1).any())
    ):
        raise ValueError("response amplitude and initial mean must be finite bounded actions")
    mean = initial_mean.detach() * mask  # 固定reset时刻的方向，不在正负条件中重算反馈。
    conditions: dict[str, torch.Tensor | None] = {
        "zero": torch.zeros_like(mean),
        "feedback": None,
        "frozenmean_positive": mean,
        "frozenmean_negative": -mean,
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (4, mean.shape[1]), generator=generator).to(mean.device, mean.dtype) * 2 - 1
    for index, sign in enumerate(signs):
        action = amplitude * sign[None, :] * mask  # 每个有效joint相同幅值；维数不改变单joint权威。
        conditions[f"rademacher_{index}_positive"] = action
        conditions[f"rademacher_{index}_negative"] = -action
    return conditions


def _initial_matches(
    reference: Mapping[str, torch.Tensor], current: Mapping[str, torch.Tensor], *, atol: float
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""逐环境检查记录初态，返回匹配mask和每字段最大差；不声称复制PhysX隐藏状态。"""

    if reference.keys() != current.keys() or atol < 0:
        raise ValueError("initial fingerprint keys/tolerance are invalid")
    first = next(iter(reference.values()))
    matched = torch.ones(first.shape[0], dtype=torch.bool, device=first.device)
    errors: dict[str, float] = {}
    for name, left in reference.items():
        right = current[name]
        if left.shape != right.shape or left.shape[0] != matched.numel():
            raise ValueError(f"initial fingerprint axis changed: {name}")
        difference = (left.double() - right.double()).abs().reshape(matched.numel(), -1).amax(dim=1)
        matched &= torch.isfinite(difference) & (difference <= (atol if left.is_floating_point() else 0.0))
        errors[name] = float(difference.max())
    return matched, errors


def _sign_contrast(
    positive: Mapping[str, np.ndarray], negative: Mapping[str, np.ndarray], *, amplitude: float, horizon_seconds: float
) -> dict[str, np.ndarray]:
    r"""记录$D=\Psi_+-\Psi_-$与$g=D/(2\alpha T)$；valid仅含初态匹配且完整存活的配对。

    净转角单位rad，$T$为观察时长s。无效对也保留原始D，但不进入带单位的条件统计，不以零值伪造响应。
    """

    if amplitude <= 0 or horizon_seconds <= 0:
        raise ValueError("contrast amplitude and duration must be positive")
    valid = (
        positive["full_survival"]
        & negative["full_survival"]
        & positive["initial_matched"]
        & negative["initial_matched"]
    )
    difference = positive["net_rotation_rad"] - negative["net_rotation_rad"]
    return {
        "valid": valid,
        "difference_rad": difference,
        "response_rad_s_per_action_unit": difference / (2 * amplitude * horizon_seconds),
    }


def _actor_mean(network: Any, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
    r"""使用现有公开Actor输入合同，只执行mean forward，不采样动作或修改RNG。"""

    from anymani.distill.models.palm_rotation_policy import (  # noqa: PLC0415
        PalmRotationActorObservation,
        PalmRotationGeometry,
    )

    geometry = PalmRotationGeometry(
        tokens=observation["geometry_tokens"].float(),
        owner_valid=observation["owner_valid"].bool(),
        shortest_path=observation["shortest_path"].long(),
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
    )
    return network.package.actor(actor_observation, geometry).mean


def _run_response_probe(agent: Any, args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> None:
    r"""复用已恢复的Agent与真实任务，采集12条无更新条件并保留pre-reset事实。"""

    from anymani.tasks.hetero.mdp.contact_state import HETERO_CONTACT_STATE_ATTR  # noqa: PLC0415
    from anymani.tasks.hetero.mdp.runtime_state import (  # noqa: PLC0415
        HETERO_PREGRASP_STATE_ATTR,
        compute_policy_step_masked_relative_target,
    )

    root = args.output.expanduser().resolve()
    runtime = agent.vec_env.env.unwrapped
    action_term = runtime.action_manager.get_term("hand_joint_pos")
    robot, cube = runtime.scene["robot"], runtime.scene["object"]
    joint_ids = action_term._joint_ids  # 与生产action显式选择的canonical轴完全相同。
    dt = float(runtime.step_dt)
    if abs(dt - 0.05) > 1.0e-9 or abs(float(action_term.cfg.scale) - 1.0 / 24.0) > 1.0e-12:
        raise RuntimeError("action response requires the frozen20Hz,1/24rad contract")
    nenv, assets = int(agent.num_actors), int(agent.asset_count)
    replicas = nenv // assets
    cohort = json.loads(args.cohort_lock.read_text(encoding="utf-8"))
    group_names = [member["provenance"]["group_name"] for member in cohort["members"]]
    if len(group_names) != assets or nenv != assets * replicas:
        raise RuntimeError("response cohort and environment axes disagree")
    groups = np.asarray([group_names[index % assets] for index in range(nenv)])
    frozen_model = {name: value.detach().clone() for name, value in agent.model.state_dict().items()}
    frozen_env = deepcopy(agent.vec_env.get_env_state())
    seed = int(checkpoint["anymani_identity"]["training"]["seed"]) + int(args.seed_offset)
    agent.model.eval()
    reference: dict[str, torch.Tensor] | None = None
    reference_records: tuple[str | None, ...] | None = None
    conditions: dict[str, torch.Tensor | None] = {}
    results: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, Any] = {}
    minimum_free: int | None = None
    optimizer_steps_before = {
        label: sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})
        for label, optimizer in (("actor", agent.optimizer), ("critic", agent.critic_optimizer))
    }

    try:
        with torch.no_grad():
            condition_count = len(args.conditions) if args.conditions is not None else 12
            for condition_index in range(condition_count):
                agent.model.load_state_dict(frozen_model, strict=True)
                agent.vec_env.set_env_state(deepcopy(frozen_env))
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                observation = agent.env_reset()["obs"]
                mask = observation["jnt_valid"].bool()
                if not bool(mask.any(-1).all()):
                    raise RuntimeError("response requires at least one active joint per environment")
                sidecar = getattr(runtime, HETERO_PREGRASP_STATE_ATTR)
                contact = getattr(runtime, HETERO_CONTACT_STATE_ATTR)
                object_state = cube.data.root_state_w.detach().clone()
                object_state[:, :3] -= runtime.scene.env_origins  # 环境局部位置，不比较grid平移。
                initial = {
                    "q_rad": robot.data.joint_pos[:, joint_ids].detach().clone(),
                    "target_rad": action_term.current_targets.detach().clone(),
                    "joint_velocity_rad_s": robot.data.joint_vel[:, joint_ids].detach().clone(),
                    "object_state_local": object_state,
                    "active_joint_mask": mask.clone(),
                    "owner_valid": observation["owner_valid"].clone(),
                    "tip_valid": observation["tip_valid"].clone(),
                    "contact_bits": contact.contact_bits.detach().clone(),
                    "force_ema_N": contact.force_ema_N.detach().clone(),
                    "goal_command": runtime.command_manager.get_command("goal_pose").detach().clone(),
                }
                records = tuple(sidecar.record_digests)
                if any(record is None for record in records) or any(
                    not bool(torch.isfinite(value).all()) for value in initial.values() if value.is_floating_point()
                ):
                    raise RuntimeError("response initial state is non-finite or lacks strict record identity")
                q_from_observation = observation["actor_jnt_current"][..., 0] * torch.pi
                if float((q_from_observation - initial["q_rad"]).abs()[mask].max()) > 1.0e-6:
                    raise RuntimeError("response joint observation and physical action axes disagree")
                if reference is None:
                    reference, reference_records = initial, records
                    conditions = _action_conditions(
                        _actor_mean(agent.model.a2c_network, observation), mask, amplitude=args.amplitude, seed=seed + 1
                    )
                initial_matched, initial_errors = _initial_matches(reference, initial, atol=1.0e-6)
                if reference_records is None:
                    raise RuntimeError("response has no reference reset identity")
                initial_matched &= torch.tensor(
                    [left == right for left, right in zip(reference_records, records, strict=True)], device=mask.device
                )
                name = (
                    args.conditions[condition_index]
                    if args.conditions is not None
                    else list(conditions)[condition_index]
                )
                fixed_action = conditions[name]
                print(f"[action-response] condition {condition_index}: {name}", flush=True)
                destination = root / name
                destination.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(
                    destination / "initial.npz",
                    record_digests=np.asarray(records, dtype=str),
                    **{key: value.cpu().numpy() for key, value in initial.items()},
                )
                active = torch.ones(nenv, dtype=torch.bool, device=mask.device)
                counts = torch.zeros(nenv, dtype=torch.long, device=mask.device)
                final_net = torch.zeros(nenv, device=mask.device)
                trace: dict[str, torch.Tensor] = {}
                target_error_max, reward_error_max = 0.0, 0.0
                for step in range(int(args.steps)):
                    actions = (
                        _actor_mean(agent.model.a2c_network, observation) if fixed_action is None else fixed_action
                    )
                    actions = actions * mask  # ghost绝不进入物理动作。
                    if step in args.snapshot_steps:
                        snapshot_dir = destination / "actor_observations"
                        snapshot_dir.mkdir(exist_ok=True)
                        actor_keys = (
                            "actor_jnt_current",
                            "actor_jnt_history",
                            "actor_jnt_limits",
                            "actor_owner_contact",
                            "jnt_valid",
                            "tip_valid",
                            "owner_valid",
                            "geometry_tokens",
                            "shortest_path",
                            "parent_direction",
                            "child_direction",
                            "prototype_index",
                        )
                        np.savez_compressed(
                            snapshot_dir / f"step_{step:04d}.npz",
                            active=active.cpu().numpy(),
                            applied_action=actions.cpu().numpy(),
                            **{key: observation[key].detach().cpu().numpy() for key in actor_keys},
                        )  # 同一动作前观察；后续counterfactual只读这些packet，不改变物理轨迹。
                    q_before = robot.data.joint_pos[:, joint_ids].detach().clone()
                    target_before = action_term.current_targets.detach().clone()
                    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
                    delta = actions.clamp(-1, 1) * mask / 24.0
                    predicted = compute_policy_step_masked_relative_target(
                        target_before, delta, limits[..., 0], limits[..., 1], mask
                    )
                    clipped_target = ((predicted - target_before - delta).abs() > 1.0e-7) & mask
                    next_observation, reward, done, _ = agent.vec_env.step(actions)
                    done = done.reshape(-1).bool()
                    snapshot = runtime.command_manager.get_term("goal_pose").post_physics_evaluation_snapshot
                    if not bool(snapshot["valid"].all()) or bool((snapshot["termination_time_out"] & active).any()):
                        raise RuntimeError("response observed invalid snapshot or an artificial short-horizon timeout")
                    post_valid = active & ~done
                    post_target = action_term.current_targets.detach().clone()
                    post_q = next_observation["obs"]["actor_jnt_current"][..., 0] * torch.pi
                    if bool(post_valid.any()):
                        error = float((post_target - predicted).abs()[post_valid & mask.any(-1)].max())
                        target_error_max = max(target_error_max, error)
                        if error > 1.0e-6:
                            raise RuntimeError(f"action target disagrees with production formula: {error}")
                    reward_terms = runtime.reward_manager._step_reward * dt  # 已有producer的真实每步贡献。
                    reward_flat = reward.reshape(-1)
                    if bool(active.any()):
                        reward_error_max = max(
                            reward_error_max, float((reward_terms.sum(-1) - reward_flat).abs()[active].max())
                        )
                        if reward_error_max > 1.0e-4:
                            raise RuntimeError(f"reward components do not reproduce actual reward: {reward_error_max}")
                    values = {
                        key: snapshot[key]
                        for key in (
                            "net_rotation_rad",
                            "absolute_path_rotation_rad",
                            "axis_speed_rad_s",
                            "episode_duration_s",
                            "tip_active_count",
                            "palm_contact",
                            "finger_non_tip_contact",
                            "position_error_m",
                            "orientation_keypoint_error_m",
                            "termination_object_out_of_anchor",
                            "termination_goal_axis_misaligned",
                            "termination_time_out",
                            "goal_success_pulse",
                        )
                    }
                    values.update(
                        {
                            "active": active,
                            "post_state_valid": post_valid,
                            "action": actions,
                            "pre_q_rad": q_before,
                            "pre_target_rad": target_before,
                            "predicted_target_rad": predicted,
                            "post_target_rad": post_target,
                            "post_q_rad": post_q,
                            "target_limit_clip": clipped_target,
                            "raw_action_clip": (actions.abs() > 1) & mask,
                            "reward_terms_step": reward_terms,
                            "reward_step": reward_flat,
                        }
                    )
                    if not trace:
                        trace = {
                            key: torch.empty((args.steps, *value.shape), dtype=value.dtype, device=value.device)
                            for key, value in values.items()
                        }
                    for key, value in values.items():
                        trace[key][step].copy_(value)  # terminal snapshot也记录一次；post状态另由mask约束。
                    final_net[active] = snapshot["net_rotation_rad"][active]
                    counts += active.long()
                    active = active & ~done
                    observation = next_observation["obs"]
                    if mask.device.type == "cuda":
                        free, total = torch.cuda.mem_get_info(mask.device)
                        minimum_free = free if minimum_free is None else min(minimum_free, free)
                        required = int(agent.config.get("gpu_driver_free_memory_bytes_min", 0))
                        if free < required:
                            raise RuntimeError(
                                f"palm-rotation training exhausted CUDA driver headroom: free={free}, required={required}, total={total}, scope=action-response"
                            )

                arrays = {key: value.cpu().numpy() for key, value in trace.items()}
                for key, value in arrays.items():
                    validity = arrays["post_state_valid"] if key.startswith("post_") else arrays["active"]
                    if value.dtype.kind == "f" and not np.isfinite(value[validity]).all():
                        raise RuntimeError(f"non-finite response trace under its declared mask: {key}")
                np.savez_compressed(destination / "trace.npz", **arrays)
                active_np, post_np, mask_np = arrays["active"], arrays["post_state_valid"], mask.cpu().numpy()
                joint_count = mask_np.sum(-1)
                sample_count = active_np.sum(0)
                post_count = post_np.sum(0)
                result = {
                    "net_rotation_rad": final_net.cpu().numpy(),
                    "full_survival": active.cpu().numpy(),
                    "initial_matched": initial_matched.cpu().numpy(),
                    "observed_steps": counts.cpu().numpy(),
                    "target_clip_fraction": (arrays["target_limit_clip"] * active_np[..., None]).sum((0, 2))
                    / np.maximum(sample_count * joint_count, 1),
                    "achieved_joint_path_rad": (
                        np.abs(arrays["post_q_rad"] - arrays["pre_q_rad"]) * post_np[..., None] * mask_np
                    ).sum((0, 2))
                    / joint_count,
                    "commanded_target_path_rad": (
                        np.abs(arrays["predicted_target_rad"] - arrays["pre_target_rad"])
                        * active_np[..., None]
                        * mask_np
                    ).sum((0, 2))
                    / joint_count,
                    "tracking_error_rad": (
                        np.abs(arrays["post_q_rad"] - arrays["predicted_target_rad"]) * post_np[..., None] * mask_np
                    ).sum((0, 2))
                    / np.maximum(post_count * joint_count, 1),
                    "valid_post_steps": post_count,
                    "reward_sum": (arrays["reward_step"] * active_np).sum(0),
                }
                np.savez_compressed(destination / "outcomes.npz", **result)
                results[name] = result
                summaries[name] = {
                    "initial_matched_count": int(initial_matched.sum()),
                    "initial_max_abs_error": initial_errors,
                    "target_formula_max_error_rad": target_error_max,
                    "reward_recount_max_error": reward_error_max,
                    "families": {
                        group: {key: float(value[groups == group].mean()) for key, value in result.items()}
                        for group in sorted(set(group_names))
                    },
                }
                if any(not torch.equal(value, agent.model.state_dict()[key]) for key, value in frozen_model.items()):
                    raise RuntimeError("action response changed model parameters or normalization buffers")
                if any(parameter.grad is not None for parameter in agent.model.parameters()):
                    raise RuntimeError("action response wrote a model gradient")

            contrasts = {}
            for prefix in ("frozenmean", "rademacher_0", "rademacher_1", "rademacher_2", "rademacher_3"):
                if f"{prefix}_positive" not in results or f"{prefix}_negative" not in results:
                    continue  # 单条件时序诊断不伪造不存在的正负配对。
                contrast = _sign_contrast(
                    results[f"{prefix}_positive"],
                    results[f"{prefix}_negative"],
                    amplitude=1.0 if prefix == "frozenmean" else args.amplitude,
                    horizon_seconds=args.steps * dt,
                )
                np.savez_compressed(root / f"{prefix}_contrast.npz", **contrast)
                contrasts[prefix] = {}
                for group in sorted(set(group_names)):
                    selected = (groups == group) & contrast["valid"]
                    contrasts[prefix][group] = {
                        "valid_pairs": int(selected.sum()),
                        "response_median_rad_s_per_action_unit": float(
                            np.median(contrast["response_rad_s_per_action_unit"][selected])
                        )
                        if selected.any()
                        else None,
                    }
            optimizer_steps_after = {
                label: sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})
                for label, optimizer in (("actor", agent.optimizer), ("critic", agent.critic_optimizer))
            }
            if optimizer_steps_before != optimizer_steps_after:
                raise RuntimeError("action response changed optimizer counters")
            _atomic_json(
                root / "summary.json",
                {
                    "schema_version": "frozen-checkpoint-action-response-v1",
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": _sha256(args.checkpoint),
                    "method_identity_digest": checkpoint["anymani_identity"]["identity_digest"],
                    "driver_source_sha256": _sha256(Path(__file__)),
                    "harness_source_sha256": _sha256(Path(run_checkpoint_audit.__code__.co_filename)),
                    "cohort_lock": str(args.cohort_lock),
                    "asset_count": assets,
                    "replicas": replicas,
                    "group_names_by_asset": group_names,
                    "seed": seed,
                    "policy_dt_s": dt,
                    "observation_steps": args.steps,
                    "selected_conditions": list(results),
                    "actor_snapshot_steps": args.snapshot_steps,
                    "random_action_amplitude": args.amplitude,
                    "task_horizon_contract_unchanged": True,
                    "optimizer_steps": 0,
                    "optimizer_counters_before": optimizer_steps_before,
                    "optimizer_counters_after": optimizer_steps_after,
                    "model_state_frozen_exact": True,
                    "parameter_gradients_untouched": True,
                    "reward_term_names": list(runtime.reward_manager.active_terms),
                    "minimum_driver_free_bytes": minimum_free,
                    "conditions": summaries,
                    "paired_sign_contrasts": contrasts,
                    "limitations": [
                        "recorded initial equality does not copy hidden PhysX state",
                        "negative one-second response does not prove a full rotation cycle infeasible",
                        "post-q/target values are only valid under post_state_valid",
                        "family means have equal replicas per asset; topology clustering is a separate analysis",
                    ],
                },
            )
    finally:
        agent.model.load_state_dict(frozen_model, strict=True)
        agent.vec_env.set_env_state(deepcopy(frozen_env))


def main() -> None:
    r"""解析仅作用于诊断的动作条件与观察长度，再复用原checkpoint恢复门。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cohort_lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--seed_offset", type=int, default=20000)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=(
            "zero",
            "feedback",
            "frozenmean_positive",
            "frozenmean_negative",
            *[f"rademacher_{index}_{sign}" for index in range(4) for sign in ("positive", "negative")],
        ),
        help="显式条件子集；省略时执行12条。",
    )
    parser.add_argument(
        "--snapshot_steps", nargs="*", type=int, default=[], help="保存完整动作前Actor观察的零起始步号。"
    )
    args = parser.parse_args()
    condition_count = len(args.conditions) if args.conditions is not None else 12
    if (
        args.output.exists()
        or not 1 <= args.steps <= 600
        or args.steps * condition_count > 600
        or not 0 < args.amplitude <= 1
    ):
        parser.error("output must be new; total vectorized steps<=600 and amplitude in(0,1]")
    if args.conditions is not None and len(set(args.conditions)) != len(args.conditions):
        parser.error("conditions must not repeat")
    if any(step < 0 or step >= args.steps for step in args.snapshot_steps):
        parser.error("snapshot steps must lie inside the observed window")
    run_checkpoint_audit(args, _run_response_probe)


if __name__ == "__main__":
    main()
