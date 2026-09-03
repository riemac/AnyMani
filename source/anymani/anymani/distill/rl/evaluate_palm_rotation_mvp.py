r"""MVP80 residual policy的fixed-scale、ADR-0正式能力评估入口。

评估固定16 replicas/asset、deterministic actor mean、2400个20 Hz policy steps。每个replica只消费第一次
trajectory；若底层ManagerBased环境在drop/axis/timeout后自动reset，后续state不再进入该replica统计。
所有终局量来自RewardManager最后一项冻结的post-physics/pre-reset snapshot，避免读取rank-0新回合零值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from isaaclab.app import AppLauncher

ANYMANI_ROOT = Path(__file__).resolve().parents[5]  # `<repo>/source/anymani/anymani/distill/rl/file.py`
DEFAULT_MANIFEST = (
    ANYMANI_ROOT
    / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml"
)
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


parser = argparse.ArgumentParser(description="Evaluate one MVP80 residual PPO checkpoint on fixed first trajectories.")
parser.add_argument("--checkpoint", type=Path, required=True, help="Full schema-3 MVP80 checkpoint.")
parser.add_argument("--asset_manifest", type=Path, default=DEFAULT_MANIFEST, help="Exact 80-row manifest.")
parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Accepted N000 fixed reference JSON.")
parser.add_argument("--num_replicas", type=int, default=16, help="Fixed replicas per asset; formal protocol uses 16.")
parser.add_argument("--steps", type=int, default=2400, help="20 Hz policy steps; formal protocol uses 2400=120 s.")
parser.add_argument("--output", type=Path, default=None, help="Cohort JSON; sibling .h5 stores trajectory arrays.")
parser.add_argument("--residual_off", action="store_true", help="Evaluate the same checkpoint with global residual set to zero.")
parser.add_argument("--rl_games_strict", action="store_true", help="Require pinned local rl_games commit.")
AppLauncher.add_app_launcher_args(parser)
args_cli, launcher_unknown_args = parser.parse_known_args()

if args_cli.num_replicas < 1 or args_cli.steps < 1:
    raise ValueError("evaluation replicas and steps must be positive")
selected_rows = _load_rows(args_cli.asset_manifest)
num_envs = 80 * int(args_cli.num_replicas)  # formal$80\times16=1280$ environments
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in selected_rows)
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
    evaluate_pairs,
    evaluate_trajectory_medians,
)
from anymani.distill.diagnostics.recording.rl.palm_rotation import (  # noqa: E402
    write_selected_trajectories_hdf5,
)
from anymani.distill.models.palm_rotation_policy import (  # noqa: E402
    PalmRotationActorObservation,
    PalmRotationGeometry,
)
from anymani.distill.rl.masked_ppo import validate_anymani_checkpoint_identity  # noqa: E402
from anymani.distill.rl.palm_rotation_ppo import PalmRotationRlGamesBuilder  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_geometry import (  # noqa: E402
    build_palm_rotation_bf16_geometry_provider,
)
from anymani.distill.rl.runtime.palm_rotation_identity import (  # noqa: E402
    build_palm_rotation_method_identity,
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
    r"""把round-robin env轴$e\bmod80$恢复为`[80,R]`正式资产矩阵。"""

    if values.shape != (80 * replicas,):
        raise ValueError("evaluation trajectory tensor disagrees with 80×replicas environment axis")
    return values.detach().cpu().numpy().reshape(replicas, 80).T


def _manifest_pairs(path: Path) -> tuple[tuple[int, int], ...]:
    r"""按最终manifest cell/pair顺序读取40组left/right formal rows。"""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    pairs: list[tuple[int, int]] = []
    for cell in document.get("cells", ()):  # 4 handedness-neutral cells
        for pair in cell.get("pairs", ()):  # 每cell 10组
            pairs.append((int(pair["left"]["row"]), int(pair["right"]["row"])))
    if len(pairs) != 40 or {row for pair in pairs for row in pair} != set(selected_rows):
        raise ValueError("MVP80 manifest must define exactly 40 complete left/right pairs")
    return tuple(pairs)


def main() -> None:
    r"""运行fixed first-trajectory evaluation并原子发布JSON/HDF5能力证据。"""

    enforce_palm_rotation_precision()
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_identity = checkpoint.get("anymani_identity")
    if not isinstance(checkpoint_identity, dict) or checkpoint_identity.get("identity_schema_version") != "3.0.0":
        raise RuntimeError("evaluation requires a schema-3 MVP80 checkpoint identity")
    run_contract = checkpoint_identity.get("training")
    if not isinstance(run_contract, dict):
        raise RuntimeError("checkpoint identity is missing the exact training contract")

    reference_doc = json.loads(args_cli.reference.read_text(encoding="utf-8"))
    reference_values = reference_doc.get("reference", {})
    reference = PalmRotationReference(
        goal_count_median=float(reference_values["G0_goal_count_median"]),
        net_turns_median=float(reference_values["N0_signed_net_turns_median"]),
    )

    env_cfg = GeneratedPalmRotationMvpEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = int(run_contract["seed"])
    device = str(run_contract.get("device", "cuda:0"))
    env_cfg.sim.device = device
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
            manifest_path=args_cli.asset_manifest,
            selected_rows=selected_rows,
            pregrasp=GOOD_PREGRASP_RESET_CFG,
            arm=str(checkpoint_identity["policy"]["arm"]),
            run_contract=run_contract,
        )
        validate_anymani_checkpoint_identity(
            runtime_identity=current_identity,
            checkpoint_identity=checkpoint_identity,
        )
        if run_contract.get("rl_games_backend_commit") != backend_info.git_commit:
            raise RuntimeError("evaluation rl_games backend disagrees with checkpoint training contract")

        builder = PalmRotationRlGamesBuilder()
        builder.load(
            {
                "palm_rotation": {
                    "residual_enabled": checkpoint_identity["policy"]["arm"] == "residual",
                    "initial_log_std": float(run_contract["initial_log_std"]),
                    "max_log_std": float(run_contract["max_log_std"]),
                    "base_action_limit": float(run_contract["base_action_limit"]),
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
        if args_cli.residual_off:
            network.package.actor.residual_enabled = False  # 同checkpoint反事实，不改base/FiLM参数
        network.eval()

        # 每env持续保存其first trajectory最新充分统计；done后即冻结并忽略自动reset的新episode。
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        goal_count = torch.zeros(num_envs, dtype=torch.float32, device=device)
        net_turns = torch.zeros_like(goal_count)
        path_turns = torch.zeros_like(goal_count)
        duration_s = torch.zeros_like(goal_count)
        termination_drop = torch.zeros(num_envs, dtype=torch.bool, device=device)
        termination_axis = torch.zeros_like(termination_drop)
        termination_timeout = torch.zeros_like(termination_drop)
        observation = transport.reset()["obs"]
        two_pi = 2.0 * torch.pi
        with torch.no_grad():
            for _step in range(int(args_cli.steps)):
                actions = _actor_mean(network, observation)
                next_observation, _reward, done, _extras = transport.step(actions)
                command = transport.unwrapped.command_manager.get_term("goal_pose")
                snapshot = command.post_physics_evaluation_snapshot
                if not bool(snapshot["valid"].all().item()):
                    raise RuntimeError("fixed evaluator observed an invalid pre-reset snapshot")
                goal_count[active] = snapshot["completed_subgoals"][active]
                net_turns[active] = snapshot["net_rotation_rad"][active] / two_pi
                path_turns[active] = snapshot["absolute_path_rotation_rad"][active] / two_pi
                duration_s[active] = snapshot["episode_duration_s"][active]
                newly_done = active & done.bool()
                termination_drop[newly_done] = snapshot["termination_object_out_of_anchor"][newly_done]
                termination_axis[newly_done] = snapshot["termination_goal_axis_misaligned"][newly_done]
                termination_timeout[newly_done] = snapshot["termination_time_out"][newly_done]
                active &= ~newly_done
                observation = next_observation["obs"]
                if not bool(active.any().item()):
                    break  # 全部first trajectories已冻结，禁止继续消费automatic-reset episodes
        if bool(active.any().item()):
            raise RuntimeError(
                f"fixed evaluation ended before {int(active.sum().item())}/{num_envs} first trajectories terminated"
            )

        arrays = {
            "goal_count": _group_by_asset(goal_count, args_cli.num_replicas).astype(np.float32),
            "signed_net_turns": _group_by_asset(net_turns, args_cli.num_replicas).astype(np.float32),
            "absolute_path_turns": _group_by_asset(path_turns, args_cli.num_replicas).astype(np.float32),
            "duration_s": _group_by_asset(duration_s, args_cli.num_replicas).astype(np.float32),
            "termination_drop": _group_by_asset(termination_drop, args_cli.num_replicas).astype(np.bool_),
            "termination_axis": _group_by_asset(termination_axis, args_cli.num_replicas).astype(np.bool_),
            "termination_timeout": _group_by_asset(termination_timeout, args_cli.num_replicas).astype(np.bool_),
        }
        cohort = evaluate_trajectory_medians(
            seed=int(run_contract["seed"]),
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
        pair_results = evaluate_pairs(cohort.asset_results, _manifest_pairs(args_cli.asset_manifest))
        pair_counts = Counter(result.outcome for result in pair_results)

        output = args_cli.output
        if output is None:
            intervention = "-residual-off" if args_cli.residual_off else ""
            output = checkpoint_path.parent.parent / "evaluation" / f"{checkpoint_path.stem}-fixed-r16{intervention}.json"
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        hdf5_path = output.with_suffix(".h5")
        checkpoint_sha = _sha256(checkpoint_path)
        evaluation_identity = {
            "schema_version": "1.0.0",
            "method_identity_digest": checkpoint_identity["identity_digest"],
            "checkpoint_sha256": checkpoint_sha,
            "reference_sha256": _sha256(args_cli.reference),
            "manifest_sha256": _sha256(args_cli.asset_manifest),
            "protocol": {
                "num_assets": 80,
                "replicas_per_asset": int(args_cli.num_replicas),
                "policy_steps": int(args_cli.steps),
                "policy_dt_s": 0.05,
                "deterministic_actor_mean": True,
                "first_trajectory_only": True,
                "pregrasp_rank": 0,
                "adr_enabled": False,
                "command_turn_ratio_relative_tolerance": 0.10,
                "asset_failure_replica_fraction": 0.5,
                "residual_off_intervention": bool(args_cli.residual_off),
            },
        }
        evaluation_identity["identity_digest"] = hashlib.sha256(
            json.dumps(evaluation_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        write_selected_trajectories_hdf5(
            hdf5_path,
            arrays=arrays,
            metadata={**evaluation_identity, "dataset_rows": list(ASSET_BINDING.dataset_rows)},
        )
        document = {
            "artifact_type": "anymani.palm_rotation_mvp80_fixed_evaluation",
            "schema_version": "1.0.0",
            "evaluation_identity": evaluation_identity,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_frame": int(checkpoint["frame"]),
            "reference": asdict(reference),
            "cohort": asdict(cohort),
            "pair_diagnostics": {
                "counts": dict(sorted(pair_counts.items())),
                "pairs": [asdict(result) for result in pair_results],
            },
            "trajectory_hdf5": str(hdf5_path),
            "trajectory_hdf5_sha256": _sha256(hdf5_path),
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "passed": cohort.passed,
                    "passed_assets": cohort.passed_assets,
                    "passed_by_cell": cohort.passed_by_cell,
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
