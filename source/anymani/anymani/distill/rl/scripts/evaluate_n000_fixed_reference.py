r"""在scale-1.1、ADR-0、固定120 s协议下评估accepted N000 best checkpoint。

该脚本不改变N000源码配置类，而在实例化后的typed cfg上显式关闭scale/material/COM/wrench/action ADR，
把timeout固定为120 s。Evaluation只取每个replica的第一条trajectory；若drop/axis提前结束，reset hook在
CommandTerm清零前冻结terminal goals、signed net turns与$\sum_t|\Delta\psi_t|$。

输出中的$G_0$与$N_0$分别为fixed replicas的连续30°目标数和实际signed净圈数中位数，供MVP80能力分数：

$$
S_i=\min(G_i/G_0,N_i/N_0).
$$

显式给出`--actor_dataset_output`时，脚本还按固定stride保存teacher真正收到的History30、raw deterministic
mean、wrapper裁剪后的环境动作，以及由同一步contact/limits构造的current student actor packet。该artifact不含
object/goal/force privilege，也不伪造N040；只有补齐旧source的selection-local row0 N040身份后才能进入student拟合。

显式给出`--student_checkpoint`时，脚本在同一GM环境中逐步构造current student packet与row0 N040，
把student canonical deterministic mean按name-based inverse permutation回写native action轴，并发布独立student
first-trajectory artifact。该模式要求显式output，避免覆盖accepted teacher reference。

`--collect_teacher_labels_on_student_rollout`把两条支路显式组合成DAgger correction采集：student决定
environment action，accepted teacher只在student访问的History30状态上提供监督标签。Artifact分别绑定behavior
checkpoint与teacher source，不能与teacher-on-policy reference bank混称。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from anymani.assets.bank.path_utils import resolve_anymani_root
from isaaclab.app import AppLauncher

ROOT = resolve_anymani_root()
DEFAULT_CHECKPOINT = ROOT / (
    "logs/distill/rl_games/gm_tactile_rotation_tcn/gm_tactile_rotation_tcn_v050_s42/nn/"
    "gm_tactile_rotation_tcn.pth"
)
DEFAULT_OUTPUT = Path("outputs/hetero/evaluation/n000-fixed-s1p1-adr0-reference.json")
FORMAL_CANONICAL_MANIFEST = ROOT / (
    "outputs/canonical_runtime/v1/groups/"
    "22320b8e00c8df73699d20cd0f56e68fbab720e8d63b36bba898ee964334a8dc/train-2048.json"
)
N000_FORMAL_ROW = 848
N000_DISTILL_PHYSICAL_GEOMETRY_HASH = "c2c581fb7acb541f706884881f3600e375bbf61ebd77a4a8bce947d4824b1fc3"
N000_CONFIGURATION_DOMAIN_HASH = "48d7f7b0c58c4cfa8171689ce56d452245285c2e51682e1182a7dd86af4fe140"

parser = argparse.ArgumentParser(description="Evaluate the N000 best checkpoint under the fixed MVP protocol.")
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument(
    "--student_checkpoint",
    type=Path,
    default=None,
    help="Optional current-actor mean-imitation checkpoint; when set, its deterministic mean controls the GM task.",
)
parser.add_argument(
    "--allow_unpassed_student_probe",
    action="store_true",
    help="Explicitly evaluate a structurally valid but offline-unpassed student as diagnostic-only evidence.",
)
parser.add_argument("--num_envs", type=int, default=64, help="Fixed evaluation replicas for the single N000 asset.")
parser.add_argument("--steps", type=int, default=2400, help="2400 policy steps equal the formal 120-second horizon.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--actor_dataset_output",
    type=Path,
    default=None,
    help="Optional HDF5 with sampled teacher means and current-student actor packets; N040 is attached later.",
)
parser.add_argument(
    "--actor_dataset_stride",
    type=int,
    default=4,
    help="Record one actor sample every N policy steps while still evaluating every step.",
)
parser.add_argument(
    "--collect_teacher_labels_on_student_rollout",
    action="store_true",
    help="Query accepted-teacher actions while the supplied student controls the environment (DAgger correction data).",
)
parser.add_argument(
    "--output",
    type=Path,
    default=DEFAULT_OUTPUT,
)
parser.add_argument("--rl_games_strict", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
if args.num_envs < 1 or args.steps < 1 or args.actor_dataset_stride < 1:
    raise ValueError("N000 fixed evaluation requires positive num_envs, steps, and actor dataset stride")
dual_policy_collection = args.student_checkpoint is not None and args.actor_dataset_output is not None
if dual_policy_collection != bool(args.collect_teacher_labels_on_student_rollout):
    raise ValueError(
        "--collect_teacher_labels_on_student_rollout must be supplied exactly when student control and teacher labels coexist"
    )
if args.student_checkpoint is not None and args.output == DEFAULT_OUTPUT:
    raise ValueError("student closed-loop evaluation requires explicit --output and cannot overwrite N000 reference")
if args.allow_unpassed_student_probe and args.student_checkpoint is None:
    raise ValueError("--allow_unpassed_student_probe requires --student_checkpoint")
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
from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1  # noqa: E402
from anymani.distill.diagnostics.recording.rl.n000_teacher import (  # noqa: E402
    build_n000_student_actor_packet,
    canonical_from_native_indices,
    contact_bits_to_owner,
    native_from_canonical_indices,
    sensor_owner_indices_from_sidecar,
)
from anymani.distill.diagnostics.recording.rl.palm_rotation import (  # noqa: E402
    write_selected_trajectories_hdf5,
)
from anymani.distill.models.palm_rotation_policy import (  # noqa: E402
    PalmRotationActorObservation,
    PalmRotationResidualActor,
)
from anymani.distill.rl.rl_games_networks import register_anymani_rl_games_networks  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_geometry import (  # noqa: E402
    build_palm_rotation_bf16_geometry_provider,
)
from anymani.robots.hand_spawn import HandSpawnAdapter  # noqa: E402
from anymani.tasks.gm.config.single_asset.single_asset_env_cfg import (  # noqa: E402
    GM_SINGLE_ASSET_CONTACT_LAYOUT,
    GM_SINGLE_ASSET_HAND_SPAWN_CFG,
)
from anymani.tasks.gm.config.single_asset.tactile_rotation_env_cfg import (  # noqa: E402
    TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
    TACTILE_PALM_SENSOR_NAME,
    TACTILE_TIP_SENSOR_NAMES,
    GmTactileRotationHistory30EnvCfg,
)
from anymani.tasks.gm.mdp.tactile_contact_state import get_tactile_contact_state  # noqa: E402
from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding  # noqa: E402
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


def _formal_geometry_provenance(source_asset_id: str) -> dict[str, Any]:
    r"""绑定formal row848 canonical artifact，但不把其global row误用作selection-local routing。"""

    path = FORMAL_CANONICAL_MANIFEST.resolve(strict=True)
    document = json.loads(path.read_text(encoding="utf-8"))
    artifacts = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(artifacts, list) or len(artifacts) <= N000_FORMAL_ROW:
        raise RuntimeError("formal train-2048 canonical manifest does not expose N000 row 848")
    artifact = artifacts[N000_FORMAL_ROW]
    if not isinstance(artifact, dict) or artifact.get("asset_id") != source_asset_id:
        raise RuntimeError("formal row 848 no longer resolves to the N000 physical asset ID")
    routing = artifact.get("routing")
    if not isinstance(routing, dict) or routing.get("asset_row") != N000_FORMAL_ROW:
        raise RuntimeError("formal N000 artifact routing row is inconsistent")
    return {
        "row": N000_FORMAL_ROW,
        "manifest": str(path),
        "manifest_sha256": _sha256(path),
        "canonical_urdf_hash": artifact.get("canonical_urdf_hash"),
        "canonical_artifact_physical_hash": artifact.get("physical_geometry_hash"),
        "source_content_hash": artifact.get("source_content_hash"),
        "source_urdf_hash": artifact.get("source_urdf_hash"),
        "routing_is_selection_local": False,
    }


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
    r"""恢复accepted teacher或current student，采集first trajectories并发布dense HDF5。"""

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

    # Optional bridge dataset保留teacher真正收到的History30，同时按name/sidecar语义构造current student packet。
    actor_buffers: dict[str, list[torch.Tensor]] = {}
    actor_sample_steps: list[int] = []
    actor_static: dict[str, torch.Tensor] = {}
    actor_metadata: dict[str, Any] | None = None
    own_contact_history: torch.Tensor | None = None
    soft_limits_native: torch.Tensor | None = None
    student_static_recorded = False
    sensor_owner_indices: tuple[int, ...] = ()
    canonical_from_native: tuple[int, ...] = ()
    student_actor: PalmRotationResidualActor | None = None
    student_provider: Any = None
    student_state: dict[str, Any] | None = None
    student_checkpoint_path: Path | None = None
    student_checkpoint_sha256: str | None = None
    student_identity_digest: str | None = None
    student_offline_gate_schema_version: str | None = None
    student_offline_passed: bool | None = None
    native_from_canonical: tuple[int, ...] = ()
    bridge_enabled = args.actor_dataset_output is not None or args.student_checkpoint is not None
    if bridge_enabled:
        robot: Any = runtime.scene["robot"]
        native_joint_names = tuple(str(name) for name in robot.joint_names)
        canonical_joint_names = CANONICAL_HAND_SCHEMA_V1.joint_names
        canonical_from_native = canonical_from_native_indices(native_joint_names, canonical_joint_names)
        source_assets = HandSpawnAdapter(GM_SINGLE_ASSET_HAND_SPAWN_CFG).selection.assets
        if len(source_assets) != 1:
            raise RuntimeError("N000 teacher bridge requires exactly one resolved source asset")
        source_asset = source_assets[0]
        state_sensor_links = (
            *GM_SINGLE_ASSET_CONTACT_LAYOUT.fingertip_link_names,
            *GM_SINGLE_ASSET_CONTACT_LAYOUT.finger_non_tip_link_names,
            GM_SINGLE_ASSET_CONTACT_LAYOUT.palm_link_name,
        )  # 与GmTactileContactState的TIP/non-tip/PALM state轴逐值一致
        sensor_owner_indices = sensor_owner_indices_from_sidecar(
            source_asset.sidecar,
            state_sensor_links=state_sensor_links,
            canonical_joint_names=canonical_joint_names,
            canonical_finger_names=CANONICAL_HAND_SCHEMA_V1.physx_finger_order,
        )
        contact_state = get_tactile_contact_state(
            runtime,
            fingertip_sensor_names=TACTILE_TIP_SENSOR_NAMES,
            finger_non_tip_sensor_names=TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
            palm_sensor_name=TACTILE_PALM_SENSOR_NAME,
        )
        expected_sensor_names = (
            *TACTILE_TIP_SENSOR_NAMES,
            *TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
            TACTILE_PALM_SENSOR_NAME,
        )
        if contact_state.sensor_names != expected_sensor_names:
            raise RuntimeError("N000 contact singleton order disagrees with the sidecar-derived bridge mapping")
        owner_contact = contact_bits_to_owner(contact_state.contact_bits, sensor_owner_indices)
        own_contact_history = owner_contact[:, 1:17].unsqueeze(1).expand(-1, 30, -1).clone()
        resolved_soft_limits = robot.data.soft_joint_pos_limits.clone()
        if resolved_soft_limits.shape != (args.num_envs, 16, 2):
            raise RuntimeError("N000 articulation soft limits do not expose the expected [B,16,2] native axis")
        soft_limits_native = resolved_soft_limits
        actor_metadata = {
            "artifact_type": "anymani.n000_teacher_mean_bridge_trajectory",
            "schema_version": "1.2.0",
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_frame": checkpoint_frame,
            "task_source_commit": "3b94c8e8911dffadc503e812a84a30f062c19d5c",
            "source_asset_id": source_asset.asset_id,
            "source_urdf": str(source_asset.urdf_path),
            "source_urdf_sha256": _sha256(source_asset.urdf_path),
            "source_sidecar": str(source_asset.sidecar_path),
            "source_sidecar_sha256": _sha256(source_asset.sidecar_path),
            "source_mesh_sha256": {
                str(mesh.virtual_path): _sha256(mesh.real_path) for mesh in source_asset.mesh_refs
            },
            "formal_geometry_provenance": _formal_geometry_provenance(source_asset.asset_id),
            "audited_distill_physical_geometry_hash": N000_DISTILL_PHYSICAL_GEOMETRY_HASH,
            "audited_configuration_domain_hash": N000_CONFIGURATION_DOMAIN_HASH,
            "native_joint_names": list(native_joint_names),
            "canonical_joint_names": list(canonical_joint_names),
            "canonical_from_native_indices": list(canonical_from_native),
            "contact_state_sensor_names": list(expected_sensor_names),
            "contact_state_sensor_links": list(state_sensor_links),
            "contact_state_owner_indices": list(sensor_owner_indices),
            "teacher_policy_input": "wrapper-clipped [q/pi,u/pi,previous-env-action,TIP-EMA] History30",
            "teacher_supervision_target": "action_env_canonical=clamp(mu_raw_native,-1,1) reordered by joint name",
            "student_actor_privileged_inputs": False,
            "n040_attached": False,
            "n040_requirement": "materialize this source as selection-local row 0 before student training",
            "sample_stride_policy_steps": int(args.actor_dataset_stride),
            "policy_dt_s": float(runtime.step_dt),
            "valid_semantics": "only samples before each replica's first terminal event are valid",
        }
    if args.student_checkpoint is not None:
        resolved_student_checkpoint = (
            args.student_checkpoint if args.student_checkpoint.is_absolute() else ROOT / args.student_checkpoint
        ).resolve(strict=True)
        student_checkpoint_path = resolved_student_checkpoint
        student_checkpoint_sha256 = _sha256(resolved_student_checkpoint)
        loaded_student_state = torch.load(resolved_student_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(loaded_student_state, dict):
            raise RuntimeError("student checkpoint root must be a mapping")
        student_state = loaded_student_state
        if student_state.get("artifact_type") != "anymani.n000_teacher_mean_student":
            raise RuntimeError("student closed-loop evaluation requires a teacher-mean student checkpoint")
        student_offline_passed = bool(student_state.get("passed"))
        student_identity = student_state.get("identity")
        if not isinstance(student_identity, dict):
            raise RuntimeError("student checkpoint lacks its supervised method identity")
        student_offline_gate_schema_version = str(student_identity.get("schema_version", ""))
        if (
            not student_offline_passed or student_offline_gate_schema_version not in {"1.1.0", "1.2.0"}
        ) and not args.allow_unpassed_student_probe:
            raise RuntimeError(
                "student closed-loop evaluation requires a corrected-gate offline-passed checkpoint unless the diagnostic override is explicit"
            )
        student_identity_digest = str(student_identity.get("identity_digest", ""))
        if not student_identity_digest:
            raise RuntimeError("student checkpoint method identity lacks a digest")
        raw_history_encoder = str(student_identity.get("history_encoder", ""))
        if raw_history_encoder not in {"tcn", "raw_stack"}:
            raise RuntimeError("student checkpoint has an unknown History30 encoder")
        history_encoder = cast(Literal["tcn", "raw_stack"], raw_history_encoder)
        binding = build_generated_asset_binding((N000_FORMAL_ROW,))
        if binding.canonical_artifacts[0].routing.asset_row != 0:
            raise RuntimeError("student N040 binding did not materialize selection-local row 0")
        student_provider = build_palm_rotation_bf16_geometry_provider(binding, device=device).eval()
        checkpoint_provider = student_identity.get("geometry_provider")
        if not isinstance(checkpoint_provider, dict) or student_provider.identity != checkpoint_provider:
            raise RuntimeError("student checkpoint and closed-loop N040 provider identities disagree")
        student_actor = PalmRotationResidualActor(history_encoder=history_encoder).to(device).eval()
        actor_state = student_state.get("actor")
        if not isinstance(actor_state, dict):
            raise RuntimeError("student checkpoint lacks actor state")
        student_actor.load_state_dict(actor_state, strict=True)
        native_from_canonical = native_from_canonical_indices(canonical_from_native)
    if actor_metadata is not None:
        actor_metadata.update(
            {
                "collection_behavior_policy": (
                    "current_teacher_mean_student" if student_actor is not None else "accepted_teacher"
                ),
                "behavior_checkpoint": str(student_checkpoint_path or checkpoint),
                "behavior_checkpoint_sha256": student_checkpoint_sha256 or _sha256(checkpoint),
                "behavior_student_method_identity_digest": student_identity_digest,
                "behavior_student_offline_passed": student_offline_passed,
                "behavior_student_offline_gate_schema_version": student_offline_gate_schema_version,
                "dataset_role": (
                    "dagger_teacher_correction" if student_actor is not None else "teacher_on_policy_reference"
                ),
                "behavior_action_semantics": "wrapper-clipped action actually passed to environment step",
            }
        )
    completed_steps = 0
    try:
        while completed_steps < int(args.steps) and not bool(finished.all().item()):
            active_before = ~finished
            with torch.no_grad():  # Isaac persistent buffers不能由inference tensor污染
                converted_observation = player.obs_to_torch(policy_observation)
                if not isinstance(converted_observation, torch.Tensor):
                    raise TypeError("N000 History30 player observation must resolve to one tensor")
                policy_observation = converted_observation
                teacher_actions = (
                    player.get_action(policy_observation, is_deterministic=True)
                    if student_actor is None or args.actor_dataset_output is not None
                    else None
                )
                capture_actor_sample = (
                    args.actor_dataset_output is not None
                    and completed_steps % int(args.actor_dataset_stride) == 0
                )
                packet = None
                teacher_history = None
                if student_actor is not None or capture_actor_sample:
                    if own_contact_history is None or soft_limits_native is None:
                        raise AssertionError("N000 bridge contact history and limits were not initialized")
                    teacher_history = policy_observation.reshape(args.num_envs, 30, 52)
                    contact_state = get_tactile_contact_state(
                        runtime,
                        fingertip_sensor_names=TACTILE_TIP_SENSOR_NAMES,
                        finger_non_tip_sensor_names=TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
                        palm_sensor_name=TACTILE_PALM_SENSOR_NAME,
                    )
                    owner_contact = contact_bits_to_owner(contact_state.contact_bits, sensor_owner_indices)
                    if not torch.equal(owner_contact[:, 1:17], own_contact_history[:, -1]):
                        raise RuntimeError("N000 own-contact history is one policy step out of phase")
                    if not torch.equal(teacher_history[:, -1, 48:52].to(dtype=torch.bool), contact_state.tip_bits):
                        raise RuntimeError("N000 teacher TIP history disagrees with the shared current contact state")
                    packet = build_n000_student_actor_packet(
                        teacher_history_native=teacher_history,
                        own_joint_contact_history=own_contact_history,
                        owner_contact=owner_contact,
                        soft_joint_limits_native_rad=soft_limits_native,
                        canonical_from_native=canonical_from_native,
                    )
                if student_actor is not None:
                    if student_provider is None or packet is None:
                        raise AssertionError("student actor requires initialized packet and N040 provider")
                    student_observation = PalmRotationActorObservation(
                        jnt_current=packet.jnt_current,
                        jnt_history=packet.jnt_history,
                        jnt_limits=packet.jnt_limits,
                        owner_contact=packet.owner_contact,
                        jnt_valid=packet.jnt_valid,
                        tip_valid=packet.tip_valid,
                        owner_valid=packet.owner_valid,
                    )
                    geometry = student_provider.resolve(
                        torch.zeros(args.num_envs, dtype=torch.long, device=device),
                        student_observation,
                    )
                    canonical_actions = student_actor(student_observation, geometry).mean
                    inverse_index = torch.tensor(native_from_canonical, dtype=torch.long, device=device)
                    actions = canonical_actions.index_select(-1, inverse_index)
                else:
                    if teacher_actions is None:
                        raise AssertionError("teacher evaluation did not produce actions")
                    actions = teacher_actions
                if capture_actor_sample:
                    if packet is None or teacher_history is None or teacher_actions is None:
                        raise AssertionError("teacher actor sample lacks packet, observation, or action")
                    canonical_index = torch.tensor(canonical_from_native, dtype=torch.long, device=actions.device)
                    action_env_native = teacher_actions.clamp(min=-clip_actions, max=clip_actions)
                    behavior_action_env_native = actions.clamp(min=-clip_actions, max=clip_actions)
                    sample = {
                        "valid": active_before,
                        "teacher_policy_history_native": teacher_history,
                        "teacher_mu_raw_native": teacher_actions,
                        "teacher_action_env_native": action_env_native,
                        "teacher_mu_raw_canonical": teacher_actions.index_select(-1, canonical_index),
                        "teacher_action_env_canonical": action_env_native.index_select(-1, canonical_index),
                        "behavior_action_env_native": behavior_action_env_native,
                        "behavior_action_env_canonical": behavior_action_env_native.index_select(-1, canonical_index),
                        "student_jnt_current": packet.jnt_current,
                        "student_jnt_history": packet.jnt_history,
                        "student_owner_contact": packet.owner_contact,
                    }
                    for name, value in sample.items():
                        actor_buffers.setdefault(name, []).append(value.detach().cpu())
                    if not student_static_recorded:
                        actor_static.update(
                            {
                                "student_jnt_limits_canonical": packet.jnt_limits.detach().cpu(),
                                "student_jnt_valid": packet.jnt_valid.detach().cpu(),
                                "student_tip_valid": packet.tip_valid.detach().cpu(),
                                "student_owner_valid": packet.owner_valid.detach().cpu(),
                            }
                        )
                        student_static_recorded = True
                    actor_sample_steps.append(completed_steps)
                next_observation, _, _, _ = wrapped.step(actions)
                policy_observation = next_observation["obs"] if isinstance(next_observation, dict) else next_observation
            command.ensure_post_physics_progress_updated(runtime)
            surviving = active_before & ~finished  # terminal rows已由hook计入最后$|\Delta\psi|$
            absolute_path_rad[surviving] += command.delta_psi[surviving].abs()
            if own_contact_history is not None:
                next_contact_state = get_tactile_contact_state(
                    runtime,
                    fingertip_sensor_names=TACTILE_TIP_SENSOR_NAMES,
                    finger_non_tip_sensor_names=TACTILE_FINGER_NON_TIP_SENSOR_NAMES,
                    palm_sensor_name=TACTILE_PALM_SENSOR_NAME,
                )
                next_owner_contact = contact_bits_to_owner(next_contact_state.contact_bits, sensor_owner_indices)
                own_contact_history = torch.roll(own_contact_history, shifts=-1, dims=1)
                own_contact_history[:, -1] = next_owner_contact[:, 1:17]
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
    actor_dataset_result: dict[str, Any] | None = None
    if args.actor_dataset_output is not None:
        if actor_metadata is None or not actor_sample_steps or not actor_buffers or not student_static_recorded:
            raise RuntimeError("N000 actor dataset was requested but no complete bridge samples were captured")
        actor_dataset_path = (
            args.actor_dataset_output
            if args.actor_dataset_output.is_absolute()
            else ROOT / args.actor_dataset_output
        ).resolve()
        actor_arrays = {
            name: torch.stack(values, dim=1).numpy()  # 每项由`[B,...]`堆成`[B,T_sample,...]`
            for name, values in actor_buffers.items()
        }
        actor_arrays["sample_policy_step"] = torch.tensor(actor_sample_steps, dtype=torch.int64).numpy()
        actor_arrays.update({name: value.numpy() for name, value in actor_static.items()})
        actor_metadata = {
            **actor_metadata,
            "replica_count": int(args.num_envs),
            "sample_count_per_replica": len(actor_sample_steps),
            "requested_policy_steps": int(args.steps),
            "completed_policy_steps": completed_steps,
        }
        actor_dataset_sha = _sha256(
            write_selected_trajectories_hdf5(
                actor_dataset_path,
                arrays=actor_arrays,
                metadata=actor_metadata,
            )
        )
        actor_dataset_result = {
            "path": str(actor_dataset_path),
            "sha256": actor_dataset_sha,
            "schema_version": actor_metadata["schema_version"],
            "sample_count_per_replica": len(actor_sample_steps),
            "sample_stride_policy_steps": int(args.actor_dataset_stride),
            "n040_attached": False,
            "dataset_role": actor_metadata["dataset_role"],
            "collection_behavior_policy": actor_metadata["collection_behavior_policy"],
        }
    net_turns = net_rotation_rad / (2.0 * math.pi)
    absolute_path_turns = absolute_path_rad / (2.0 * math.pi)
    consistency = torch.clamp(net_turns, min=0.0) / absolute_path_turns.clamp_min(torch.finfo(torch.float32).eps)
    g0 = float(torch.quantile(goals, 0.5).item())
    n0 = float(torch.quantile(net_turns, 0.5).item())
    command_turn_ratio = g0 / (12.0 * n0) if n0 > 0.0 else 0.0
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    hdf5_path = output_path.with_suffix(".h5")
    evaluated_checkpoint_path = student_checkpoint_path or checkpoint
    evaluated_checkpoint_sha256 = student_checkpoint_sha256 or _sha256(checkpoint)
    evaluated_policy_type = "current_teacher_mean_student" if student_actor is not None else "accepted_teacher"
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
                "evaluated_policy_type": evaluated_policy_type,
                "evaluated_checkpoint_sha256": evaluated_checkpoint_sha256,
                "teacher_source_checkpoint_sha256": _sha256(checkpoint),
                "student_method_identity_digest": student_identity_digest,
                "student_offline_passed": student_offline_passed,
                "student_offline_gate_schema_version": student_offline_gate_schema_version,
                "seed": int(args.seed),
                "deterministic_actor_mean": True,
                "first_trajectory_only": True,
                "action_authority_rad_per_policy_step": 1.0 / 24.0,
            },
        )
    )
    result: dict[str, Any] = {
        "artifact_type": (
            "anymani.n000_teacher_mean_student_fixed_evaluation"
            if student_actor is not None
            else "anymani.n000_fixed_mvp_reference"
        ),
        "schema_version": "2.2.0",
        "checkpoint": str(evaluated_checkpoint_path),
        "checkpoint_sha256": evaluated_checkpoint_sha256,
        "checkpoint_epoch": checkpoint_epoch if student_actor is None else None,
        "checkpoint_frame": checkpoint_frame if student_actor is None else None,
        "checkpoint_update": int(student_state["update"]) if student_state is not None else None,
        "teacher_source": (
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "epoch": checkpoint_epoch,
                "frame": checkpoint_frame,
            }
            if student_actor is not None
            else None
        ),
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
            "evaluated_policy_type": evaluated_policy_type,
            "student_method_identity_digest": student_identity_digest,
            "student_offline_passed": student_offline_passed,
            "student_offline_gate_schema_version": student_offline_gate_schema_version,
            "allow_unpassed_student_probe": bool(args.allow_unpassed_student_probe),
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
        "actor_dataset": actor_dataset_result,
        "rl_games_backend": {"file": str(backend.package_file), "commit": backend.git_commit},
        "formal_120s_complete": int(args.steps) == 2400,
    }
    capability_summary: dict[str, Any] = {
        "goal_count_median": g0,
        "signed_net_turns_median": n0,
        "command_turn_ratio": command_turn_ratio,
        "interpretation": "(goals/12)/net_turns calibrates moving-goal tracking against physical net turns",
    }
    if student_actor is not None:
        capability_summary["closure_passed"] = bool(
            n0 >= 1.0
            and float(torch.quantile(consistency, 0.5).item()) >= 0.7
            and float(drop.float().mean().item()) < 0.5
            and float(axis_failure.float().mean().item()) < 0.5
        )
    result["student_result" if student_actor is not None else "reference"] = capability_summary
    if student_actor is None and result["formal_120s_complete"] and (g0 <= 0.0 or n0 <= 0.0):
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
