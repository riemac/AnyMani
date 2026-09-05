r"""MVP80 residual policy的fixed-scale、ADR-0正式能力评估入口。

主评估固定16 replicas/asset、deterministic actor mean、600个20 Hz policy steps（30 s），2400步用于耐久复查。每个replica只消费第一次
trajectory；若底层ManagerBased环境在drop/axis/timeout后自动reset，后续state不再进入该replica统计。
所有终局量来自RewardManager最后一项冻结的post-physics/pre-reset snapshot，避免读取rank-0新回合零值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from isaaclab.app import AppLauncher

ANYMANI_ROOT = Path(__file__).resolve().parents[5]  # `<repo>/source/anymani/anymani/distill/rl/file.py`
DEFAULT_MANIFEST = ANYMANI_ROOT / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml"
DEFAULT_REFERENCE = ANYMANI_ROOT / "outputs/hetero/evaluation/n000-fixed-s1p1-adr0-reference.json"


def _load_rows(path: Path) -> tuple[int, ...]:
    r"""在Isaac/task import前读取并验证固定80-row支持轴。"""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("MVP80 evaluation manifest must contain a mapping")
    rows = tuple(int(row) for row in document.get("selected_rows", ()))
    if len(rows) != 80 or len(set(rows)) != 80:
        raise ValueError("MVP80 evaluation requires exactly 80 unique selected_rows")
    return rows


def _select_support_rows(mvp80_rows: tuple[int, ...], raw_support_rows: str | None) -> tuple[int, ...]:
    r"""返回完整MVP80或显式有序子集，语义与训练入口一致。"""

    if raw_support_rows is None:
        return mvp80_rows
    selected = tuple(int(item.strip()) for item in raw_support_rows.split(",") if item.strip())
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("--support_rows must contain unique formal rows")
    outside = tuple(row for row in selected if row not in set(mvp80_rows))
    if outside:
        raise ValueError(f"evaluation support rows lie outside the frozen MVP80 manifest: {outside}")
    return selected


parser = argparse.ArgumentParser(description="Evaluate one MVP80 residual PPO checkpoint on fixed first trajectories.")
parser.add_argument("--asset_manifest", type=Path, default=DEFAULT_MANIFEST, help="Exact 80-row manifest.")
parser.add_argument(
    "--support_rows", type=str, default=None, help="Ordered MVP80 subset used by the checkpoint closure run."
)
parser.add_argument(
    "--cohort_lock",
    type=Path,
    default=None,
    help="训练checkpoint使用的member-level cohort lock；与legacy support rows互斥。",
)
parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Accepted N000 fixed reference JSON.")
parser.add_argument("--num_replicas", type=int, default=16, help="Fixed replicas per asset; formal protocol uses 16.")
parser.add_argument("--steps", type=int, default=2400, help="Fixed episode policy steps: 600=30 s, 2400=120 s.")
parser.add_argument(
    "--trace_stride", type=int, default=0, help="Per-step diagnostic sampling stride; 0 disables trace."
)
parser.add_argument("--output", type=Path, default=None, help="Cohort JSON; sibling .h5 stores trajectory arrays.")
parser.add_argument(
    "--implementation_certificate",
    type=Path,
    default=None,
    help="跨源码实现只读回放所需的精确重构等价证书；不用于完整训练续接。",
)
parser.add_argument(
    "--residual_off", action="store_true", help="Evaluate the same checkpoint with global residual set to zero."
)
parser.add_argument("--real-time", action="store_true", help="Pace GUI replay at the 20 Hz policy period.")
parser.add_argument(
    "--viewer_asset_index",
    type=int,
    default=0,
    help="GUI follows this selection-local asset index; each replica block preserves the same asset order.",
)
parser.add_argument("--rl_games_strict", action="store_true", help="Require pinned local rl_games commit.")
parser.add_argument(
    "--tip_only_intervention",
    action="store_true",
    help="Frozen-policy ablation: mask non-TIP contact at every actor input route.",
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=Path, required=True, help="Full schema-3/4 palm-rotation checkpoint.")
args_cli, launcher_unknown_args = parser.parse_known_args()

if args_cli.num_replicas < 1 or args_cli.steps < 1:
    raise ValueError("evaluation replicas and steps must be positive")
if args_cli.trace_stride < 0:
    raise ValueError("trace stride must be non-negative")
if args_cli.cohort_lock is not None:
    if args_cli.support_rows is not None:
        raise ValueError("--cohort_lock and --support_rows are mutually exclusive")
    cohort_lock_path = (
        args_cli.cohort_lock
        if args_cli.cohort_lock.is_absolute()
        else (ANYMANI_ROOT / args_cli.cohort_lock).resolve(strict=True)
    )
    cohort_document = yaml.safe_load(cohort_lock_path.read_text(encoding="utf-8"))
    if not isinstance(cohort_document, dict) or cohort_document.get("schema_version") != "1.2.0":
        raise ValueError("cohort evaluation requires a schema-1.2 canonical-final lock")
    cohort_members = cohort_document.get("members")
    if not isinstance(cohort_members, list) or not cohort_members:
        raise ValueError("--cohort_lock must contain a non-empty members list")
    selected_rows = tuple(range(len(cohort_members)))  # selection-local runtime/evaluation axis
    mother_ids = tuple(str(member["provenance"]["mother_name"]) for member in cohort_members)
    source_member_keys = tuple(f"{member['source_alias']}#{int(member['source_row'])}" for member in cohort_members)
    support_manifest_path = cohort_lock_path  # exact lock bytes必须与训练method identity一致
    os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
    os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(cohort_lock_path)
else:
    mvp80_rows = _load_rows(args_cli.asset_manifest)
    selected_rows = _select_support_rows(mvp80_rows, args_cli.support_rows)
    mother_ids = tuple(f"legacy-row-{row}" for row in selected_rows)  # A80历史评估不应用mother晋级门
    source_member_keys = tuple(f"ppo#{row}" for row in selected_rows)
    support_manifest_path = args_cli.asset_manifest
    os.environ.pop("ANYMANI_HETERO_COHORT_LOCK", None)
    os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in selected_rows)
asset_count = len(selected_rows)  # $A=1$ closure或$A=80$正式cohort
if not 0 <= int(args_cli.viewer_asset_index) < asset_count:
    raise ValueError(f"--viewer_asset_index must lie in [0,{asset_count}), got {args_cli.viewer_asset_index}")
num_envs = asset_count * int(args_cli.num_replicas)  # round-robin$A\times R$ environments
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(num_envs)
sys.argv = [sys.argv[0], *launcher_unknown_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from anymani.distill.rl.rl_games_backend import prefer_local_rl_games  # noqa: E402

backend_info = prefer_local_rl_games(strict=bool(args_cli.rl_games_strict))  # pin before any `rl_games.*` import

import anymani.distill.rl  # noqa: F401, E402
import anymani.tasks.hetero  # noqa: F401, E402
from anymani.distill.diagnostics.evaluation.rl.palm_rotation import (  # noqa: E402
    PalmRotationReference,
    evaluate_cohort,
    evaluate_pairs,
    evaluate_physical_support_trajectory_medians,
    evaluate_scale_ladder_cohort,
    evaluate_support_trajectory_medians,
)
from anymani.distill.diagnostics.recording.rl.palm_rotation import (  # noqa: E402
    write_selected_trajectories_hdf5,
)
from anymani.distill.models.palm_rotation_policy import (  # noqa: E402
    PalmRotationActorObservation,
    PalmRotationGeometry,
)
from anymani.distill.rl.palm_rotation_ppo import PalmRotationRlGamesBuilder  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_geometry import (  # noqa: E402
    build_palm_rotation_bf16_geometry_provider,
)
from anymani.distill.rl.runtime.palm_rotation_identity import (  # noqa: E402
    build_palm_rotation_method_identity,
    palm_rotation_code_provenance,
    validate_palm_rotation_evaluation_identity,
)
from anymani.distill.rl.runtime.palm_rotation_precision import enforce_palm_rotation_precision  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_vecenv import (  # noqa: E402
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_FLOAT_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
    PalmRotationRlGamesVecEnv,
)
from anymani.tasks.hetero.config.generated.palm_rotation_mvp_env_cfg import (  # noqa: E402
    GOOD_PREGRASP_RESET_CFG,
    GeneratedPalmRotationMvpEnvCfg,
)
from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING  # noqa: E402
from anymani.tasks.hetero.mdp.contact_state import HETERO_CONTACT_STATE_ATTR  # noqa: E402

SINGLE_CLOSURE_NET_TURNS_MIN = 1.0  # RL tiny-overfit最低持续有向旋转能力
SINGLE_CLOSURE_DIRECTIONAL_CONSISTENCY_MIN = 0.7  # 排除absolute path由往返jitter构成
SINGLE_CLOSURE_SAFE_REPLICA_FRACTION_MIN_EXCLUSIVE = 0.5  # 严格多数replicas同时无drop/axis


def _sha256(path: Path) -> str:
    r"""流式计算checkpoint/reference/manifest证据摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actor_mean(network: Any, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
    r"""只执行deterministic actor，不为正式能力评估额外计算privileged critic。"""

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
    return network.package.actor(actor_observation, geometry).mean  # deterministic$\mu_t\in\mathbb R^{N\times16}$


def _group_by_asset(values: torch.Tensor, replicas: int) -> np.ndarray:
    r"""把round-robin env轴$e\bmod A$恢复为`[A,R]`资产矩阵。"""

    if values.shape != (asset_count * replicas,):
        raise ValueError("evaluation trajectory tensor disagrees with assets×replicas environment axis")
    return values.detach().cpu().numpy().reshape(replicas, asset_count).T


def _manifest_pairs(path: Path) -> tuple[tuple[int, int], ...]:
    r"""按最终manifest顺序返回当前支持集中完整出现的left/right pairs。"""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    pairs: list[tuple[int, int]] = []
    for cell in document.get("cells", ()):  # 4 handedness-neutral cells
        for pair in cell.get("pairs", ()):  # 每cell 10组
            candidate = (int(pair["left"]["row"]), int(pair["right"]["row"]))
            if candidate[0] in selected_rows and candidate[1] in selected_rows:
                pairs.append(candidate)  # single closure可自然没有完整pair
    if asset_count == 80 and (len(pairs) != 40 or {row for pair in pairs for row in pair} != set(selected_rows)):
        raise ValueError("MVP80 manifest must define exactly 40 complete left/right pairs")
    return tuple(pairs)


def main() -> None:
    r"""运行fixed first-trajectory evaluation并原子发布JSON/HDF5能力证据。"""

    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_identity = checkpoint.get("anymani_identity")
    if not isinstance(checkpoint_identity, dict) or checkpoint_identity.get("identity_schema_version") not in {
        "3.0.0",
        "4.0.0",
    }:
        raise RuntimeError("evaluation requires a schema-3/4 palm-rotation checkpoint identity")
    run_contract = checkpoint_identity.get("training")
    if not isinstance(run_contract, dict):
        raise RuntimeError("checkpoint identity is missing the exact training contract")
    enforce_palm_rotation_precision(allow_tf32=bool(run_contract.get("allow_tf32", False)))

    reference_doc = json.loads(args_cli.reference.read_text(encoding="utf-8"))
    reference_values = reference_doc.get("reference", {})
    reference = PalmRotationReference(
        goal_count_median=float(
            reference_values.get("G0_goal_count_median", reference_values.get("goal_count_median"))
        ),
        net_turns_median=float(
            reference_values.get("N0_signed_net_turns_median", reference_values.get("signed_net_turns_median"))
        ),
    )

    env_cfg = GeneratedPalmRotationMvpEnvCfg()
    # 同时约束当前帧、History30和owner token三条入口；冻结遮蔽是独立干预，不改checkpoint的训练身份。
    actor_tip_only = run_contract.get("actor_contact", "all") == "tip" or bool(args_cli.tip_only_intervention)
    for name in ("jnt_current", "jnt_history", "owner_contact"):
        getattr(env_cfg.observations.policy, name).params["tip_only"] = actor_tip_only
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = int(run_contract["seed"])
    env_cfg.viewer.env_index = int(args_cli.viewer_asset_index)  # round-robin首个replica中env index等于asset index
    device = str(run_contract.get("device", "cuda:0"))
    env_cfg.sim.device = device
    policy_dt_s = float(env_cfg.sim.dt) * int(env_cfg.decimation)  # $6/120=0.05$ s，GUI只按真实policy周期节流
    evaluation_horizon_s = int(args_cli.steps) * policy_dt_s  # 固定评价窗口，不修改checkpoint训练协议
    env_cfg.episode_length_s = evaluation_horizon_s
    env_cfg.commands.goal_pose.horizon_s = evaluation_horizon_s
    env = gym.make("AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0", cfg=env_cfg)
    provider = build_palm_rotation_bf16_geometry_provider(ASSET_BINDING, device=device)
    prototype_index = torch.tensor(ASSET_BINDING.asset_index_by_env(num_envs), dtype=torch.long, device=device)
    transport = PalmRotationRlGamesVecEnv(
        env,
        geometry_provider=provider,
        prototype_index=prototype_index,
        rl_device=device,
        clip_observations=100.0,
        clip_actions=1.0,
    )

    try:
        current_identity = build_palm_rotation_method_identity(
            provider_identity=provider.identity,
            manifest_path=support_manifest_path,
            selected_rows=selected_rows,
            pregrasp=GOOD_PREGRASP_RESET_CFG,
            arm=str(checkpoint_identity["policy"]["arm"]),
            run_contract=run_contract,
        )
        implementation_certificate = (
            json.loads(args_cli.implementation_certificate.read_text(encoding="utf-8"))
            if args_cli.implementation_certificate is not None
            else None
        )
        validate_palm_rotation_evaluation_identity(
            runtime_identity=current_identity,
            checkpoint_identity=checkpoint_identity,
            implementation_certificate=implementation_certificate,
        )
        if run_contract.get("rl_games_backend_commit") != backend_info.git_commit:
            raise RuntimeError("evaluation rl_games backend disagrees with checkpoint training contract")

        builder = PalmRotationRlGamesBuilder()
        builder.load(
            {
                "palm_rotation": {
                    "arm": checkpoint_identity["policy"]["arm"],
                    "initial_log_std": float(run_contract["initial_log_std"]),
                    "max_log_std": float(run_contract["max_log_std"]),
                    "base_action_limit": float(run_contract["base_action_limit"]),
                    "history_encoder": str(run_contract.get("history_encoder", "tcn")),
                    "compile_mode": None,  # fixed evaluation不承担训练compile cold-start或私有pool
                },
                "anymani_identity": checkpoint_identity,
            }
        )
        input_shape = {**PALM_ROTATION_FLOAT_SHAPES, **PALM_ROTATION_BOOL_SHAPES, **PALM_ROTATION_INT16_SHAPES}
        network = builder.build("a2c", actions_num=16, input_shape=input_shape, value_size=1, num_seqs=1).to(device)
        model_state = checkpoint.get("model")
        if not isinstance(model_state, dict):
            raise RuntimeError("checkpoint is missing rl_games model state")
        prefix = "a2c_network."
        network_state = {key[len(prefix) :]: value for key, value in model_state.items() if key.startswith(prefix)}
        network.load_state_dict(network_state, strict=True)
        if args_cli.residual_off and checkpoint_identity["policy"]["arm"] != "residual":
            raise ValueError("--residual_off is defined only for a residual-arm checkpoint")
        if args_cli.residual_off:
            network.package.actor.residual_enabled = False  # 同checkpoint反事实，不改base/FiLM参数
        network.eval()

        # 每env持续保存其first trajectory最新充分统计；done后即冻结并忽略自动reset的新episode。
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        goal_count = torch.zeros(num_envs, dtype=torch.float32, device=device)
        frontier_count = torch.zeros_like(goal_count)  # $K_T$，历史正向30°物理前沿总数
        frontier_delta_recount = torch.zeros_like(goal_count)  # $\sum_t\Delta K_t$生命周期重计
        max_positive_rotation_rad = torch.zeros_like(goal_count)  # $M_T$，rad
        net_turns = torch.zeros_like(goal_count)
        path_turns = torch.zeros_like(goal_count)
        duration_s = torch.zeros_like(goal_count)
        termination_drop = torch.zeros(num_envs, dtype=torch.bool, device=device)
        termination_axis = torch.zeros_like(termination_drop)
        termination_timeout = torch.zeros_like(termination_drop)

        # Goal-count audit沿每条first trajectory累计双门充分统计。它只观察command已经冻结的pre-reset事实，
        # 不重新定义reward或success；目标是区分orientation门、25 mm anchor门与记录链错误。
        diagnostic_steps = torch.zeros(num_envs, dtype=torch.float32, device=device)  # 每条轨迹已观察policy steps
        orientation_error_sum_m = torch.zeros_like(goal_count)  # $\sum_t e_R(t)$，单位m
        orientation_error_max_m = torch.zeros_like(goal_count)  # $\max_t e_R(t)$，单位m
        orientation_error_final_m = torch.zeros_like(goal_count)  # terminal/window末端$e_R$，单位m
        position_error_sum_m = torch.zeros_like(goal_count)  # $\sum_t e_p(t)$，单位m
        position_error_max_m = torch.zeros_like(goal_count)  # $\max_t e_p(t)$，单位m
        position_error_final_m = torch.zeros_like(goal_count)  # terminal/window末端$e_p$，单位m
        orientation_gate_steps = torch.zeros_like(goal_count)  # $e_R<5\,\mathrm{mm}$的step数
        position_gate_steps = torch.zeros_like(goal_count)  # $e_p<25\,\mathrm{mm}$的step数
        goal_success_pulse_recount = torch.zeros_like(goal_count)  # 从逐step snapshot重计的pulse总数
        first_position_gate_failure_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )  # 首次不满足25 mm anchor门；全程满足保持-1
        last_position_gate_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )  # 最后一次满足anchor门的1-based step
        last_goal_success_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )  # 最后一次真实goal pulse的1-based step
        net_turns_at_last_goal_success = torch.zeros_like(goal_count)  # 最后pulse时的signed累计净圈
        position_error_at_last_goal_success_m = torch.zeros_like(goal_count)  # 最后pulse时的anchor误差，单位m
        observation = transport.reset()["obs"]
        trace_buffers: dict[str, torch.Tensor] = {}
        trace_count = 0
        trace_capacity = (
            (int(args_cli.steps) + int(args_cli.trace_stride) - 1) // int(args_cli.trace_stride)
            if args_cli.trace_stride
            else 0
        )
        two_pi = 2.0 * torch.pi
        with torch.no_grad():
            for _step in range(int(args_cli.steps)):
                step_started_at = time.perf_counter()  # GUI replay墙钟节流起点；不进入任何物理状态或指标
                actions = _actor_mean(network, observation)
                next_observation, _reward, done, _extras = transport.step(actions)
                command = transport.unwrapped.command_manager.get_term("goal_pose")
                snapshot = command.post_physics_evaluation_snapshot
                if not bool(snapshot["valid"].all().item()):
                    raise RuntimeError("fixed evaluator observed an invalid pre-reset snapshot")
                if trace_capacity and _step % int(args_cli.trace_stride) == 0:
                    # 聚合量来自真正pre-reset snapshot；逐sensor量只在未reset的step有效，单独保存mask。
                    contact = getattr(transport.unwrapped, HETERO_CONTACT_STATE_ATTR)
                    values = {
                        name: snapshot[name]
                        for name in (
                            "episode_duration_s",
                            "net_rotation_rad",
                            "absolute_path_rotation_rad",
                            "axis_speed_rad_s",
                            "tip_active_count",
                            "palm_contact",
                            "finger_non_tip_contact",
                            "goal_success_pulse",
                            "position_error_m",
                            "orientation_keypoint_error_m",
                            "termination_object_out_of_anchor",
                            "termination_goal_axis_misaligned",
                            "termination_time_out",
                        )
                    }
                    values.update(
                        {
                            "active": active,
                            "post_state_valid": active & ~done.bool(),
                            "sensor_contact_bits": contact.contact_bits,
                            "sensor_force_ema_N": contact.force_ema_N,
                            "action": actions,
                            "pre_owner_contact": observation["actor_owner_contact"].squeeze(-1),
                            "post_joint_position_rad": next_observation["obs"]["actor_jnt_current"][..., 0] * torch.pi,
                        }
                    )
                    if not trace_buffers:
                        trace_buffers = {
                            name: torch.empty((trace_capacity, *value.shape), dtype=value.dtype, device=value.device)
                            for name, value in values.items()
                        }  # 一次分配，采样循环不做GPU→CPU搬运
                    for name, value in values.items():
                        trace_buffers[name][trace_count].copy_(value)
                    trace_count += 1
                goal_count[active] = snapshot["completed_subgoals"][active]
                frontier_count[active] = snapshot["rotation_frontier_count"][active]
                max_positive_rotation_rad[active] = snapshot["max_positive_net_rotation_rad"][active]
                net_turns[active] = snapshot["net_rotation_rad"][active] / two_pi
                path_turns[active] = snapshot["absolute_path_rotation_rad"][active] / two_pi
                duration_s[active] = snapshot["episode_duration_s"][active]

                # Snapshot与active mask共同保证terminal step被计入一次，automatic reset后的新episode不进入旧轨迹。
                orientation_error = snapshot["orientation_keypoint_error_m"]  # `[N]`，六轴keypoint平均距离m
                position_error = snapshot["position_error_m"]  # `[N]`，相对reset anchor的中心距离m
                orientation_gate = orientation_error < float(command.cfg.orientation_success_threshold_m)
                position_gate = position_error < float(command.cfg.position_success_threshold_m)
                success_pulse = snapshot["goal_success_pulse"].bool()
                active_float = active.to(dtype=goal_count.dtype)  # bool轨迹membership转为可累计的0/1权重
                frontier_delta_recount += snapshot["rotation_frontier_delta"] * active_float
                diagnostic_steps += active_float
                orientation_error_sum_m += orientation_error * active_float
                position_error_sum_m += position_error * active_float
                orientation_error_max_m = torch.where(
                    active, torch.maximum(orientation_error_max_m, orientation_error), orientation_error_max_m
                )
                position_error_max_m = torch.where(
                    active, torch.maximum(position_error_max_m, position_error), position_error_max_m
                )
                orientation_error_final_m[active] = orientation_error[active]
                position_error_final_m[active] = position_error[active]
                orientation_gate_steps += (active & orientation_gate).to(dtype=goal_count.dtype)
                position_gate_steps += (active & position_gate).to(dtype=goal_count.dtype)
                goal_success_pulse_recount += (active & success_pulse).to(dtype=goal_count.dtype)

                # 首次/最后gate时刻解释“物体继续转但goal count停止”的候选机制，不把短时越界当成termination。
                first_position_failure = active & ~position_gate & (first_position_gate_failure_step < 0)
                first_position_gate_failure_step[first_position_failure] = _step + 1
                last_position_gate_step[active & position_gate] = _step + 1
                active_success = active & success_pulse
                last_goal_success_step[active_success] = _step + 1
                net_turns_at_last_goal_success[active_success] = snapshot["net_rotation_rad"][active_success] / two_pi
                position_error_at_last_goal_success_m[active_success] = position_error[active_success]
                newly_done = active & done.bool()
                termination_drop[newly_done] = snapshot["termination_object_out_of_anchor"][newly_done]
                termination_axis[newly_done] = snapshot["termination_goal_axis_misaligned"][newly_done]
                termination_timeout[newly_done] = snapshot["termination_time_out"][newly_done]
                active &= ~newly_done
                observation = next_observation["obs"]
                if args_cli.real_time:
                    sleep_s = policy_dt_s - (time.perf_counter() - step_started_at)  # 目标20 Hz policy周期
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)  # 仿真较快时补足墙钟；较慢时不跳过任何physics step
                if not bool(active.any().item()):
                    break  # 全部first trajectories已冻结，禁止继续消费automatic-reset episodes
        if bool(active.any().item()):
            raise RuntimeError(
                f"fixed evaluation ended before {int(active.sum().item())}/{num_envs} first trajectories terminated"
            )

        safe_diagnostic_steps = diagnostic_steps.clamp_min(1.0)  # 每条first trajectory理论上至少含terminal/window一步
        frontier_from_maximum = torch.floor(
            max_positive_rotation_rad / float(command.cfg.rotation_frontier_interval_rad)
        )  # 按定义从$M_T$独立重算$K_T$
        expected_goals_from_positive_net = 12.0 * torch.clamp(net_turns, min=0.0)  # 理想30°计数$12N^+$
        goal_turn_ratio = torch.where(
            expected_goals_from_positive_net > torch.finfo(goal_count.dtype).eps,
            goal_count / expected_goals_from_positive_net.clamp_min(torch.finfo(goal_count.dtype).eps),
            torch.zeros_like(goal_count),
        )  # 每trajectory配对ratio，避免ratio-of-independent-medians掩盖多峰分布
        arrays = {
            "goal_count": _group_by_asset(goal_count, args_cli.num_replicas).astype(np.float32),
            "rotation_frontier_count": _group_by_asset(frontier_count, args_cli.num_replicas).astype(np.float32),
            "rotation_frontier_delta_recount": _group_by_asset(frontier_delta_recount, args_cli.num_replicas).astype(
                np.float32
            ),
            "rotation_frontier_recount_error": _group_by_asset(
                frontier_delta_recount - frontier_count, args_cli.num_replicas
            ).astype(np.float32),
            "rotation_frontier_formula_error": _group_by_asset(
                frontier_from_maximum - frontier_count, args_cli.num_replicas
            ).astype(np.float32),
            "max_positive_net_turns": _group_by_asset(max_positive_rotation_rad / two_pi, args_cli.num_replicas).astype(
                np.float32
            ),
            "signed_net_turns": _group_by_asset(net_turns, args_cli.num_replicas).astype(np.float32),
            "absolute_path_turns": _group_by_asset(path_turns, args_cli.num_replicas).astype(np.float32),
            "duration_s": _group_by_asset(duration_s, args_cli.num_replicas).astype(np.float32),
            "termination_drop": _group_by_asset(termination_drop, args_cli.num_replicas).astype(np.bool_),
            "termination_axis": _group_by_asset(termination_axis, args_cli.num_replicas).astype(np.bool_),
            "termination_timeout": _group_by_asset(termination_timeout, args_cli.num_replicas).astype(np.bool_),
            "goal_success_pulse_recount": _group_by_asset(goal_success_pulse_recount, args_cli.num_replicas).astype(
                np.float32
            ),
            "goal_count_recount_error": _group_by_asset(
                goal_success_pulse_recount - goal_count, args_cli.num_replicas
            ).astype(np.float32),
            "expected_goal_count_from_positive_net": _group_by_asset(
                expected_goals_from_positive_net, args_cli.num_replicas
            ).astype(np.float32),
            "goal_turn_ratio_per_trajectory": _group_by_asset(goal_turn_ratio, args_cli.num_replicas).astype(
                np.float32
            ),
            "diagnostic_step_count": _group_by_asset(diagnostic_steps, args_cli.num_replicas).astype(np.float32),
            "orientation_keypoint_error_mean_m": _group_by_asset(
                orientation_error_sum_m / safe_diagnostic_steps, args_cli.num_replicas
            ).astype(np.float32),
            "orientation_keypoint_error_max_m": _group_by_asset(orientation_error_max_m, args_cli.num_replicas).astype(
                np.float32
            ),
            "orientation_keypoint_error_final_m": _group_by_asset(
                orientation_error_final_m, args_cli.num_replicas
            ).astype(np.float32),
            "position_error_mean_m": _group_by_asset(
                position_error_sum_m / safe_diagnostic_steps, args_cli.num_replicas
            ).astype(np.float32),
            "position_error_max_m": _group_by_asset(position_error_max_m, args_cli.num_replicas).astype(np.float32),
            "position_error_final_m": _group_by_asset(position_error_final_m, args_cli.num_replicas).astype(np.float32),
            "orientation_gate_fraction": _group_by_asset(
                orientation_gate_steps / safe_diagnostic_steps, args_cli.num_replicas
            ).astype(np.float32),
            "position_pass_given_orientation_gate": _group_by_asset(
                goal_success_pulse_recount / orientation_gate_steps.clamp_min(1.0), args_cli.num_replicas
            ).astype(np.float32),
            "position_gate_fraction": _group_by_asset(
                position_gate_steps / safe_diagnostic_steps, args_cli.num_replicas
            ).astype(np.float32),
            "first_position_gate_failure_step": _group_by_asset(
                first_position_gate_failure_step, args_cli.num_replicas
            ).astype(np.int64),
            "last_position_gate_step": _group_by_asset(last_position_gate_step, args_cli.num_replicas).astype(np.int64),
            "last_goal_success_step": _group_by_asset(last_goal_success_step, args_cli.num_replicas).astype(np.int64),
            "net_turns_at_last_goal_success": _group_by_asset(
                net_turns_at_last_goal_success, args_cli.num_replicas
            ).astype(np.float32),
            "position_error_at_last_goal_success_m": _group_by_asset(
                position_error_at_last_goal_success_m, args_cli.num_replicas
            ).astype(np.float32),
        }
        asset_results, finite_and_identity_valid = evaluate_support_trajectory_medians(
            dataset_rows=ASSET_BINDING.dataset_rows,
            cell_ids=ASSET_BINDING.morphology_cell_ids,
            goal_counts=arrays["goal_count"].tolist(),
            net_turns=arrays["signed_net_turns"].tolist(),
            absolute_path_turns=arrays["absolute_path_turns"].tolist(),
            termination_drop=arrays["termination_drop"].tolist(),
            termination_axis=arrays["termination_axis"].tolist(),
            termination_timeout=arrays["termination_timeout"].tolist(),
            reference=reference,
            command_turn_ratio_relative_tolerance=0.10,
        )
        physical_asset_results, physical_finite = evaluate_physical_support_trajectory_medians(
            dataset_rows=ASSET_BINDING.dataset_rows,
            cell_ids=ASSET_BINDING.morphology_cell_ids,
            mother_ids=mother_ids,
            frontier_counts=arrays["rotation_frontier_count"].tolist(),
            max_positive_net_turns=arrays["max_positive_net_turns"].tolist(),
            net_turns=arrays["signed_net_turns"].tolist(),
            absolute_path_turns=arrays["absolute_path_turns"].tolist(),
            termination_drop=arrays["termination_drop"].tolist(),
            termination_axis=arrays["termination_axis"].tolist(),
        )
        scale_protocol_matched = (
            int(args_cli.steps) == 600
            and int(args_cli.num_replicas) == 16
            and actor_tip_only
            and not args_cli.residual_off
            and not args_cli.tip_only_intervention
        )  # 耐久/R1/冻结遮蔽仅诊断，不冒充30秒R16的正式扩张资格
        scale_ladder_result = (
            evaluate_scale_ladder_cohort(
                physical_asset_results,
                finite_and_identity_valid=physical_finite,
            )
            if asset_count in (16, 64, 128) and scale_protocol_matched
            else None
        )  # 其它cardinality仍保存逐资产物理证据，但不伪造scale-ladder结论
        cohort = (
            evaluate_cohort(
                seed=int(run_contract["seed"]),
                asset_results=asset_results,
                finite_and_identity_valid=finite_and_identity_valid,
            )
            if asset_count == 80
            else None
        )  # 54/80 cohort gate只对完整冻结支持集有定义
        closure_passed_assets = sum(result.viability_passed for result in physical_asset_results)
        closure_passed = bool(physical_finite and closure_passed_assets == asset_count)
        pair_results = (
            evaluate_pairs(asset_results, _manifest_pairs(args_cli.asset_manifest))
            if args_cli.cohort_lock is None
            else ()
        )
        pair_counts = Counter(result.outcome for result in pair_results)
        closure_thresholds = {
            "net_turns_median_min": SINGLE_CLOSURE_NET_TURNS_MIN,
            "directional_consistency_min": SINGLE_CLOSURE_DIRECTIONAL_CONSISTENCY_MIN,
            "safe_replica_fraction_min_exclusive": SINGLE_CLOSURE_SAFE_REPLICA_FRACTION_MIN_EXCLUSIVE,
        }  # 阈值同时进入evaluation identity与人类可读support结果

        output = args_cli.output
        if output is None:
            intervention = "-residual-off" if args_cli.residual_off else ""
            intervention += "-tip-mask" if args_cli.tip_only_intervention else ""
            output = (
                checkpoint_path.parent.parent
                / "evaluation"
                / f"{checkpoint_path.stem}-fixed{evaluation_horizon_s:g}s-r{args_cli.num_replicas}{intervention}.json"
            )
        output = output.expanduser().resolve()
        if any(path.exists() for path in (output, output.with_suffix(".h5"), output.with_suffix(".trace.h5"))):
            raise FileExistsError(f"evaluation output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        hdf5_path = output.with_suffix(".h5")
        checkpoint_sha = _sha256(checkpoint_path)
        evaluation_identity = {
            "schema_version": "1.4.0",
            "method_identity_digest": checkpoint_identity["identity_digest"],
            "execution_identity_digest": current_identity["identity_digest"],
            "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
            "execution_implementation": current_identity["implementation"],
            "code_provenance": palm_rotation_code_provenance(),
            "implementation_certificate_sha256": (
                _sha256(args_cli.implementation_certificate)
                if args_cli.implementation_certificate is not None
                else None
            ),
            "checkpoint_sha256": checkpoint_sha,
            "reference_sha256": _sha256(args_cli.reference),
            "manifest_sha256": _sha256(support_manifest_path),
            "protocol": {
                "num_assets": asset_count,
                "replicas_per_asset": int(args_cli.num_replicas),
                "policy_steps": int(args_cli.steps),
                "policy_dt_s": 0.05,
                "horizon_s": evaluation_horizon_s,
                "trace_stride": int(args_cli.trace_stride),
                "actor_contact_intervention": "tip-only-mask" if args_cli.tip_only_intervention else "none",
                "actor_contact": "tip-only-binary" if actor_tip_only else "all-owner-binary-no-force",
                "scale_ready_protocol_matched": scale_protocol_matched,
                "reference_horizon_s": reference_doc.get("protocol", {}).get("horizon_s"),
                "reference_horizon_matched": reference_doc.get("protocol", {}).get("horizon_s") == evaluation_horizon_s,
                "deterministic_actor_mean": True,
                "first_trajectory_only": True,
                "pregrasp_rank": 0,
                "adr_enabled": False,
                "command_turn_ratio_relative_tolerance": 0.10,
                "asset_failure_replica_fraction": 0.5,
                "support_closure_thresholds": closure_thresholds,
                "residual_off_intervention": bool(args_cli.residual_off),
                "goal_tracking_diagnostics": "paired-per-trajectory-gate-audit-v1",
                "rotation_frontier_diagnostics": "historical-positive-net-30deg-frontier-v1",
                "scale_ready_thresholds": {
                    "net_turns_median_min": 2.0,
                    "directional_consistency_min": 0.85,
                    "safe_replica_fraction_min": 0.75,
                },
            },
        }
        evaluation_identity["identity_digest"] = hashlib.sha256(
            json.dumps(evaluation_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        write_selected_trajectories_hdf5(
            hdf5_path,
            arrays=arrays,
            metadata={
                **evaluation_identity,
                "dataset_rows": list(ASSET_BINDING.dataset_rows),
                "source_member_keys": list(source_member_keys),
                "mother_ids": list(mother_ids),
            },
        )

        trace_result = None
        if trace_buffers:
            trace_path = output.with_suffix(".trace.h5")
            trace_arrays = {
                name: value[:trace_count]
                .reshape(trace_count, args_cli.num_replicas, asset_count, *value.shape[2:])
                .transpose(1, 2)
                .cpu()
                .numpy()
                for name, value in trace_buffers.items()
            }  # 所有逐时数组为[T,A,R,...]，active含最后terminal step，后续自动reset样本无效
            trace_arrays["policy_step"] = np.arange(trace_count, dtype=np.int64) * int(args_cli.trace_stride) + 1
            write_selected_trajectories_hdf5(
                trace_path,
                arrays=trace_arrays,
                metadata={
                    **evaluation_identity,
                    "trace_stride": int(args_cli.trace_stride),
                    "axes": "time,asset,replica,feature",
                    "sensor_names": list(ASSET_BINDING.contact_layout.state_sensor_names),
                    "post_state_valid_semantics": "sensor forces/bits and post q exclude automatic-reset terminal rows",
                    "pre_owner_contact_semantics": "actor input before the applied action; aggregates are post-physics/pre-reset",
                },
            )
            trace_result = {"path": str(trace_path), "sha256": _sha256(trace_path), "samples": trace_count}

        # JSON保留逐资产可读的tracking分布摘要；逐replica原值仍以HDF5为唯一dense事实源。
        goal_tracking_assets: list[dict[str, Any]] = []
        frontier_assets: list[dict[str, Any]] = []
        reference_ratio = float(reference.command_turn_ratio)  # accepted N000的提前触发校准，仅作参照
        for asset_index, dataset_row in enumerate(ASSET_BINDING.dataset_rows):
            asset_goals = arrays["goal_count"][asset_index].astype(np.float64)
            asset_turns = arrays["signed_net_turns"][asset_index].astype(np.float64)
            asset_ratios = arrays["goal_turn_ratio_per_trajectory"][asset_index].astype(np.float64)
            positive = asset_turns > np.finfo(np.float32).eps
            ratio_inlier = positive & (np.abs(asset_ratios / reference_ratio - 1.0) <= 0.10)
            goal_turn_correlation = None
            if args_cli.num_replicas > 1 and float(np.std(asset_goals)) > 0.0 and float(np.std(asset_turns)) > 0.0:
                goal_turn_correlation = float(np.corrcoef(asset_goals, asset_turns)[0, 1])
            frontier_assets.append(
                {
                    "dataset_row": int(dataset_row),
                    "source_member_key": source_member_keys[asset_index],
                    "mother_id": mother_ids[asset_index],
                    "frontier_count_min_median_max": [
                        float(np.min(arrays["rotation_frontier_count"][asset_index])),
                        float(np.median(arrays["rotation_frontier_count"][asset_index])),
                        float(np.max(arrays["rotation_frontier_count"][asset_index])),
                    ],
                    "max_positive_net_turns_min_median_max": [
                        float(np.min(arrays["max_positive_net_turns"][asset_index])),
                        float(np.median(arrays["max_positive_net_turns"][asset_index])),
                        float(np.max(arrays["max_positive_net_turns"][asset_index])),
                    ],
                    "delta_recount_max_abs_error": float(
                        np.max(np.abs(arrays["rotation_frontier_recount_error"][asset_index]))
                    ),
                    "formula_recount_max_abs_error": float(
                        np.max(np.abs(arrays["rotation_frontier_formula_error"][asset_index]))
                    ),
                }
            )
            goal_tracking_assets.append(
                {
                    "dataset_row": int(dataset_row),
                    "replica_count": int(args_cli.num_replicas),
                    "strict_goal_count_min_median_max": [
                        float(np.min(asset_goals)),
                        float(np.median(asset_goals)),
                        float(np.max(asset_goals)),
                    ],
                    "signed_net_turns_min_median_max": [
                        float(np.min(asset_turns)),
                        float(np.median(asset_turns)),
                        float(np.max(asset_turns)),
                    ],
                    "paired_goal_turn_ratio_q10_median_q90": [
                        float(np.quantile(asset_ratios, 0.10)),
                        float(np.median(asset_ratios)),
                        float(np.quantile(asset_ratios, 0.90)),
                    ],
                    "n000_reference_goal_turn_ratio": reference_ratio,
                    "n000_ratio_inlier_replica_fraction": float(np.mean(ratio_inlier)),
                    "goal_count_net_turns_pearson": goal_turn_correlation,
                    "pulse_recount_max_abs_error": float(
                        np.max(np.abs(arrays["goal_count_recount_error"][asset_index]))
                    ),
                    "position_gate_fraction_median": float(np.median(arrays["position_gate_fraction"][asset_index])),
                    "position_pass_given_orientation_gate_q10_median_q90": [
                        float(np.quantile(arrays["position_pass_given_orientation_gate"][asset_index], 0.10)),
                        float(np.median(arrays["position_pass_given_orientation_gate"][asset_index])),
                        float(np.quantile(arrays["position_pass_given_orientation_gate"][asset_index], 0.90)),
                    ],
                    "last_goal_success_step_min_median_max": [
                        int(np.min(arrays["last_goal_success_step"][asset_index])),
                        float(np.median(arrays["last_goal_success_step"][asset_index])),
                        int(np.max(arrays["last_goal_success_step"][asset_index])),
                    ],
                    "net_turns_at_last_goal_success_min_median_max": [
                        float(np.min(arrays["net_turns_at_last_goal_success"][asset_index])),
                        float(np.median(arrays["net_turns_at_last_goal_success"][asset_index])),
                        float(np.max(arrays["net_turns_at_last_goal_success"][asset_index])),
                    ],
                }
            )
        document = {
            "artifact_type": (
                "anymani.palm_rotation_mvp80_fixed_evaluation"
                if asset_count == 80
                else "anymani.palm_rotation_support_fixed_evaluation"
            ),
            "schema_version": "1.3.0",
            "evaluation_identity": evaluation_identity,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_frame": int(checkpoint["frame"]),
            "reference": asdict(reference),
            "support": {
                "asset_count": asset_count,
                "semantics": "legacy N000-relative strict-tracking conjunction; not the scale-ladder gate",
                "closure_thresholds": closure_thresholds,
                "finite_and_identity_valid": finite_and_identity_valid,
                "closure_passed_assets": closure_passed_assets,
                "closure_passed": closure_passed,
                "asset_results": [asdict(result) for result in asset_results],
            },
            "physical_rotation": {
                "semantics": "fixed-ADR-0 net turns, path directionality, joint drop/axis survival, and 30deg frontier",
                "finite_and_identity_valid": physical_finite,
                "viability_passed_assets": closure_passed_assets,
                "all_assets_viable": closure_passed,
                "asset_results": [asdict(result) for result in physical_asset_results],
                "frontier_recount": frontier_assets,
            },
            "scale_ladder": asdict(scale_ladder_result) if scale_ladder_result is not None else None,
            "cohort": asdict(cohort) if cohort is not None else None,
            "pair_diagnostics": {
                "counts": dict(sorted(pair_counts.items())),
                "pairs": [asdict(result) for result in pair_results],
            },
            "goal_tracking_diagnostics": {
                "semantics": (
                    "strict full-SO(3) orientation-keypoint and fixed-position-anchor goal hits; "
                    "not a direct 30-degree unwrapped-axis counter"
                ),
                "asset_results": goal_tracking_assets,
            },
            "trajectory_hdf5": str(hdf5_path),
            "trajectory_hdf5_sha256": _sha256(hdf5_path),
            "step_trace": trace_result,
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "formal_cohort_applicable": cohort is not None,
                    "formal_cohort_passed": cohort.passed if cohort is not None else None,
                    "formal_asset_passed_count": sum(result.passed for result in asset_results),
                    "scale_ready_applicable": scale_ladder_result is not None,
                    "scale_ready_passed": scale_ladder_result.passed if scale_ladder_result is not None else None,
                    "scale_ready_passed_assets": (
                        scale_ladder_result.scale_ready_assets if scale_ladder_result is not None else None
                    ),
                    "sustained_rotation_closure_passed": closure_passed,
                    "sustained_rotation_closure_passed_assets": closure_passed_assets,
                    "passed_by_cell": cohort.passed_by_cell if cohort is not None else None,
                    "pair_counts": dict(pair_counts),
                },
                sort_keys=True,
            )
        )
    finally:
        transport.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
