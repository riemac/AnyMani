r"""在scale-1.1、ADR-0、固定120 s协议下评估accepted N000 best checkpoint。

该脚本不改变N000源码配置类，而在实例化后的typed cfg上显式关闭scale/material/COM/wrench/action ADR，
把timeout固定为120 s。Evaluation只取每个replica的第一条trajectory；若drop/axis提前结束，reset hook在
CommandTerm清零前冻结terminal goals、signed net turns与$\sum_t|\Delta\psi_t|$。

输出中的$G_0$与$N_0$分别为fixed replicas的连续30°目标数和实际signed净圈数中位数，供MVP80能力分数：

$$
S_i=\min(G_i/G_0,N_i/N_0).
$$
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import yaml
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / (
    "logs/distill/rl_games/gm_tactile_rotation_tcn/gm_tactile_rotation_tcn_v050_s42/nn/"
    "gm_tactile_rotation_tcn.pth"
)

parser = argparse.ArgumentParser(description="Evaluate the N000 best checkpoint under the fixed MVP protocol.")
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument("--num_envs", type=int, default=64, help="Fixed evaluation replicas for the single N000 asset.")
parser.add_argument("--steps", type=int, default=2400, help="2400 policy steps equal the formal 120-second horizon.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("outputs/hetero/evaluation/n000-fixed-s1p1-adr0-reference.json"),
)
parser.add_argument("--rl_games_strict", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
if args.num_envs < 1 or args.steps < 1:
    raise ValueError("N000 fixed evaluation requires positive num_envs and steps")
sys.argv = [sys.argv[0], *unknown]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import isaaclab.envs.mdp as isaac_mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games  # noqa: E402

backend = prefer_local_rl_games(strict=bool(args.rl_games_strict))  # 任何`rl_games.*`间接import前固定backend

import anymani.distill.rl  # noqa: F401, E402
import anymani.tasks.gm  # noqa: F401, E402
from anymani.distill.diagnostics.recording.rl.palm_rotation import (  # noqa: E402
    write_selected_trajectories_hdf5,
)
from anymani.distill.rl.rl_games_networks import register_anymani_rl_games_networks  # noqa: E402
from anymani.tasks.gm.config.single_asset.tactile_rotation_env_cfg import (  # noqa: E402
    GmTactileRotationHistory30EnvCfg,
)
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import TerminationTermCfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402


def _sha256(path: Path) -> str:
    r"""流式计算N000 checkpoint identity。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_protocol_cfg() -> GmTactileRotationHistory30EnvCfg:
    r"""从N000 typed cfg构造scale-1.1、无随机化、固定120 s evaluation变体。"""

    cfg = GmTactileRotationHistory30EnvCfg()
    cfg.scene.num_envs = int(args.num_envs)
    cfg.seed = int(args.seed)
    cfg.episode_length_s = 120.0
    cfg.actions.hand_joint_pos.use_adr = False  # 保留$u_{t+1}=u_t+a_t/24$，关闭noise/latency
    object_spawn = cast(sim_utils.UsdFileCfg, cfg.scene.object.spawn)
    object_spawn.scale = (1.1, 1.1, 1.1)  # exact absolute DexCube scale

    # Prestartup scale和reset-time COM/material/wrench scheduler全部退出，初态只保留N000固定preset。
    disabled_events = (
        "randomized_object_scale",
        "resample_object_material_from_adr",
        "resample_hand_contact_material_from_adr",
        "randomized_object_com",
        "reset_episode_length",
        "reset_wrench_gate",
        "object_wrench",
    )
    for event_name in disabled_events:
        setattr(cfg.events, event_name, None)  # ManagerBased cfg以None显式删除该event term

    # Startup材质仍建立PhysX material，但上下界相等；不存在bucket随机性。
    for term in (cfg.events.initialize_object_material, cfg.events.initialize_hand_contact_material):
        term.params["static_friction_range"] = (1.0, 1.0)
        term.params["dynamic_friction_range"] = (1.0, 1.0)
        term.params["restitution_range"] = (0.0, 0.0)
        term.params["num_buckets"] = 1
    cfg.events.randomized_object_mass.params["mass_distribution_params"] = (1.0, 1.0)  # nominal mass
    cfg.events.randomized_actuator_gains.params["stiffness_distribution_params"] = (3.0, 3.0)
    cfg.events.randomized_actuator_gains.params["damping_distribution_params"] = (0.1, 0.1)
    setattr(cfg.curriculum, "adr", None)  # no range progression or yaw/noise publication
    cfg.terminations.time_out = TerminationTermCfg(func=isaac_mdp.time_out, time_out=True)  # fixed 2400 steps
    return cfg


def _agent_cfg() -> dict[str, Any]:
    r"""读取accepted TCN的versioned rl_games network/YAML合同。"""

    path = ROOT / "source/anymani/anymani/distill/rl/agents/gm_tactile_rotation_tcn_ppo.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("N000 agent YAML must contain a mapping")
    return document


def main() -> dict[str, Any]:
    r"""恢复N000 actor，采集first trajectories并发布reference与dense per-replica HDF5。"""

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    checkpoint = checkpoint.resolve(strict=True)
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint_state.get("epoch", -1))
    checkpoint_frame = int(checkpoint_state.get("frame", -1))
    if checkpoint_epoch != 3831 or checkpoint_frame != 470753280:
        raise RuntimeError(
            "N000 fixed reference requires the accepted best checkpoint at epoch/frame 3831/470753280"
        )
    cfg = _fixed_protocol_cfg()
    agent_cfg = _agent_cfg()
    rl_device = str(agent_cfg["params"]["config"].get("device", args.device))
    cfg.sim.device = args.device if args.device is not None else cfg.sim.device
    agent_cfg["params"]["seed"] = int(args.seed)
    agent_cfg["params"]["config"]["num_actors"] = int(args.num_envs)
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)

    env = gym.make("AnyMani-GM-SingleAsset-TactileRotation-History30Obs-v0", cfg=cfg)
    runtime = cast(ManagerBasedRLEnv, env.unwrapped)
    clip_obs = float(agent_cfg["params"]["env"]["clip_observations"])
    clip_actions = float(agent_cfg["params"]["env"]["clip_actions"])
    wrapped = RlGamesVecEnvWrapper(cast(Any, env), rl_device, clip_obs, clip_actions)
    vecenv.register(
        "N000FixedReferenceWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "N000FixedReferenceWrapper", "env_creator": lambda **_: wrapped})
    register_anymani_rl_games_networks()
    runner = Runner()
    runner.load(agent_cfg)
    player: BasePlayer = runner.create_player()
    player.restore(str(checkpoint))
    player.reset()

    # First-trajectory recorder在command reset清零前截获terminal state；initial explicit reset期间关闭capture。
    command: Any = runtime.command_manager.get_term("goal_pose")
    original_reset = command.reset
    device = runtime.device
    finished = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    absolute_path_rad = torch.zeros(args.num_envs, device=device)  # $\sum_t|\Delta\psi_t|$
    goals = torch.full((args.num_envs,), torch.nan, device=device)
    net_rotation_rad = torch.full((args.num_envs,), torch.nan, device=device)
    duration_s = torch.full((args.num_envs,), torch.nan, device=device)
    drop = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    axis_failure = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    timed_out = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    capture_enabled = False

    def capture_then_reset(env_ids=None):
        r"""在automatic reset前冻结当前first-trajectory terminal sufficient statistics。"""

        all_ids = torch.arange(args.num_envs, device=device)
        ids = (
            all_ids
            if env_ids is None
            else all_ids[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=device).flatten()
        )
        if capture_enabled and ids.numel() > 0:
            selected = ids[~finished[ids]]  # 每个replica只保留第一条trajectory
            if selected.numel() > 0:
                command.ensure_post_physics_progress_updated(runtime)
                absolute_path_rad[selected] += command.delta_psi[selected].abs()
                goals[selected] = command.goal_success_count[selected] + command.goal_success_pulse[selected].float()
                net_rotation_rad[selected] = command.net_rotation_rad[selected]
                duration_s[selected] = runtime.episode_length_buf[selected].float() * float(runtime.step_dt)
                drop[selected] = runtime.termination_manager.get_term("object_out_of_anchor")[selected]
                axis_failure[selected] = runtime.termination_manager.get_term("goal_axis_misaligned")[selected]
                timed_out[selected] = runtime.termination_manager.get_term("time_out")[selected]
                finished[selected] = True
        return original_reset(env_ids)

    command.reset = capture_then_reset  # evaluation-only instance hook；不修改task class或checkpoint
    observation = wrapped.reset()
    capture_enabled = True
    policy_observation = observation["obs"] if isinstance(observation, dict) else observation
    _ = player.get_batch_size(policy_observation, 1)
    completed_steps = 0
    try:
        while completed_steps < int(args.steps) and not bool(finished.all().item()):
            active_before = ~finished
            with torch.no_grad():  # Isaac persistent buffers不能由inference tensor污染
                policy_observation = player.obs_to_torch(policy_observation)
                actions = player.get_action(policy_observation, is_deterministic=True)
                next_observation, _, _, _ = wrapped.step(actions)
                policy_observation = next_observation["obs"] if isinstance(next_observation, dict) else next_observation
            command.ensure_post_physics_progress_updated(runtime)
            surviving = active_before & ~finished  # terminal rows已由hook计入最后$|\Delta\psi|$
            absolute_path_rad[surviving] += command.delta_psi[surviving].abs()
            completed_steps += 1

        # Evaluation window结束但未natural-terminal的replicas以当前post-physics state右删失finalize。
        active = ~finished
        if bool(active.any().item()):
            command.ensure_post_physics_progress_updated(runtime)
            goals[active] = command.goal_success_count[active] + command.goal_success_pulse[active].float()
            net_rotation_rad[active] = command.net_rotation_rad[active]
            duration_s[active] = completed_steps * float(runtime.step_dt)
            finished[active] = True
    finally:
        wrapped.close()

    if not bool(torch.isfinite(goals).all().item() and torch.isfinite(net_rotation_rad).all().item()):
        raise RuntimeError("N000 fixed evaluation did not finalize every first trajectory")
    net_turns = net_rotation_rad / (2.0 * math.pi)
    absolute_path_turns = absolute_path_rad / (2.0 * math.pi)
    consistency = torch.clamp(net_turns, min=0.0) / absolute_path_turns.clamp_min(torch.finfo(torch.float32).eps)
    g0 = float(torch.quantile(goals, 0.5).item())
    n0 = float(torch.quantile(net_turns, 0.5).item())
    command_turn_ratio = g0 / (12.0 * n0) if n0 > 0.0 else float("inf")
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    hdf5_path = output_path.with_suffix(".h5")
    hdf5_sha = _sha256(
        write_selected_trajectories_hdf5(
            hdf5_path,
            arrays={
                "goal_count": goals.detach().cpu().numpy(),
                "signed_net_turns": net_turns.detach().cpu().numpy(),
                "absolute_path_turns": absolute_path_turns.detach().cpu().numpy(),
                "directional_consistency": consistency.detach().cpu().numpy(),
                "duration_s": duration_s.detach().cpu().numpy(),
                "termination_drop": drop.detach().cpu().numpy(),
                "termination_axis": axis_failure.detach().cpu().numpy(),
                "termination_timeout": timed_out.detach().cpu().numpy(),
            },
            metadata={
                "checkpoint_sha256": _sha256(checkpoint),
                "seed": int(args.seed),
                "deterministic_actor_mean": True,
                "first_trajectory_only": True,
                "action_authority_rad_per_policy_step": 1.0 / 24.0,
            },
        )
    )
    result: dict[str, Any] = {
        "artifact_type": "anymani.n000_fixed_mvp_reference",
        "schema_version": "2.0.0",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_frame": checkpoint_frame,
        "seed": int(args.seed),
        "num_replicas": int(args.num_envs),
        "requested_steps": int(args.steps),
        "completed_steps": completed_steps,
        "protocol": {
            "object_scale": 1.1,
            "adr": 0,
            "action_noise": 0.0,
            "action_latency_steps": 0,
            "initial_noise": 0.0,
            "yaw": 0.0,
            "wrench": False,
            "horizon_s": 120.0,
            "policy_dt_s": float(runtime.step_dt),
            "deterministic_actor_mean": True,
            "first_trajectory_only": True,
            "action_authority_rad_per_policy_step": 1.0 / 24.0,
            "subgoal_degrees": 30.0,
            "orientation_success_threshold_m": 0.005,
            "position_success_threshold_m": 0.025,
            "drop_distance_m": 0.07,
            "axis_failure_degrees": 45.0,
            "reward_identity": "accepted-n000-3b94c8e-tcn30-composite-reward",
            "task_source_commit": "3b94c8e8911dffadc503e812a84a30f062c19d5c",
        },
        "reference": {
            "G0_goal_count_median": g0,
            "N0_signed_net_turns_median": n0,
            "command_turn_ratio": command_turn_ratio,
            "interpretation": "(G0/12)/N0 calibrates moving-goal tolerance against physical net turns",
        },
        "distribution": {
            "goal_count_mean": float(goals.mean().item()),
            "signed_net_turns_mean": float(net_turns.mean().item()),
            "directional_consistency_median": float(torch.quantile(consistency, 0.5).item()),
            "drop_fraction": float(drop.float().mean().item()),
            "axis_failure_fraction": float(axis_failure.float().mean().item()),
            "timeout_fraction": float(timed_out.float().mean().item()),
        },
        "trajectory_hdf5": str(hdf5_path),
        "trajectory_hdf5_sha256": hdf5_sha,
        "rl_games_backend": {"file": str(backend.package_file), "commit": backend.git_commit},
        "formal_120s_complete": int(args.steps) == 2400,
    }
    if result["formal_120s_complete"] and (g0 <= 0.0 or n0 <= 0.0):
        raise RuntimeError(f"accepted N000 produced non-positive fixed-protocol reference: G0={g0}, N0={n0}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
