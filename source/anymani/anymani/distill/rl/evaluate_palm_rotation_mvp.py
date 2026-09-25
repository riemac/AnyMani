"Evaluate the frozen palm rotation policy on a fixed cohort and retain first-trajectory failures."

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from anymani.assets.bank.cohort import parse_hand_asset_cohort_document
from anymani.distill.rl.runtime.video_settings import (
    VideoSettings,
    add_video_settings_arguments,
)
from isaaclab.app import AppLauncher

ANYMANI_ROOT = Path(__file__).resolve().parents[5]  # `<repo>/source/anymani/anymani/distill/rl/file.py`
DEFAULT_REFERENCE = None
DEFAULT_N040_ARTIFACT = ANYMANI_ROOT / "data/models/encoder.pt"
DEFAULT_N040_SHA256 = "cda44cc9eae5ca28a1a735176ef4764805559d13e235c52477b6ac438b20ddea"


parser = argparse.ArgumentParser(description="Evaluate one MVP80 residual PPO checkpoint on fixed first trajectories.")
parser.add_argument(
    "--cohort_lock",
    type=Path,
    default=None,
    help="Member-level cohort lock; use --cohort_transfer when evaluating a different training support set.",
)
parser.add_argument(
    "--cohort_transfer",
    action="store_true",
    help="Explicit frozen-policy evaluation on a different canonical cohort; shared assets keep exact pregrasps.",
)
parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Frozen N000 fixed reference JSON.")
parser.add_argument(
    "--n040_artifact",
    type=Path,
    default=DEFAULT_N040_ARTIFACT,
    help="Standalone retained N040 encoder artifact.",
)
parser.add_argument(
    "--n040_sha256",
    type=str,
    default=DEFAULT_N040_SHA256,
    help="Expected SHA-256 of the standalone retained N040 encoder.",
)
parser.add_argument("--num_replicas", type=int, default=16, help="Fixed replicas per asset; formal protocol uses 16.")
parser.add_argument("--steps", type=int, default=2400, help="Fixed episode policy steps: 600=30 s, 2400=120 s.")
parser.add_argument(
    "--action_mode",
    choices=("mean", "sample"),
    default="mean",
    help="Mean is formal evaluation; sample is a frozen-policy diagnostic.",
)
parser.add_argument(
    "--action_seed", type=int, default=None, help="Explicit seed for the independent sample-mode action generator."
)
parser.add_argument(
    "--diagnostic_only", action="store_true", help="Publish diagnostic trajectories without formal capability verdicts."
)
parser.add_argument(
    "--diagnostic_boundary_step",
    type=int,
    default=None,
    help="Capture the state after this many actions; optionally hand off the Actor before the next action.",
)
parser.add_argument(
    "--actor_switch_checkpoint",
    type=Path,
    default=None,
    help="Same-method Actor taking over at the diagnostic boundary; requires --diagnostic_only.",
)
parser.add_argument(
    "--trace_stride", type=int, default=0, help="Per-step diagnostic sampling stride; 0 disables trace."
)
parser.add_argument(
    "--trace_rewards", action="store_true", help="Include actual weighted per-step reward terms in trace."
)
parser.add_argument("--output", type=Path, default=None, help="Cohort JSON; sibling .h5 stores trajectory arrays.")
parser.add_argument(
    "--video", type=Path, default=None, help="Optional MP4 of the selected viewer asset's first trajectory."
)
add_video_settings_arguments(parser)
parser.add_argument(
    "--implementation_certificate",
    type=Path,
    default=None,
    help="Exact implementation-compatibility record for frozen evaluation; it does not certify full training resume.",
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
parser.add_argument(
    "--direct_logit_gain",
    type=float,
    default=1.0,
    help="Frozen Direct-policy diagnostic: mu=tanh(gain*logit); physical action bounds remain unchanged.",
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=Path, required=True, help="Full schema-3/4 palm-rotation checkpoint.")
parser.add_argument(
    "--student_checkpoint",
    type=Path,
    default=None,
    help="Complete student PPO .pth; its deterministic Actor replaces the reference Actor for this fixed evaluation.",
)
parser.add_argument(
    "--student_ppo_checkpoint",
    dest="student_checkpoint",
    type=Path,
    help="Explicit alias for --student_checkpoint when invoking the PPO student route directly.",
)
parser.add_argument(
    "--student_variant",
    choices=("n040", "no_z", "fk"),
    default=None,
    help="Optional student variant assertion; the checkpoint method identity remains authoritative.",
)
parser.add_argument(
    "--student_export",
    type=Path,
    default=None,
    help="Optional new TorchScript path for the loaded student Actor; sibling JSON stores deployment identity.",
)
parser.add_argument(
    "--student_export_metadata",
    type=Path,
    default=None,
    help="Optional JSON sidecar path paired with --student_export (defaults to <student_export>.json).",
)
args_cli, launcher_unknown_args = parser.parse_known_args()
if args_cli.cohort_lock is None:
    raise ValueError("paper runtime evaluation requires an explicit packaged --cohort_lock")
args_cli.n040_artifact = (
    args_cli.n040_artifact.expanduser()
    if args_cli.n040_artifact.is_absolute()
    else (ANYMANI_ROOT / args_cli.n040_artifact).expanduser()
).resolve(strict=True)
if len(args_cli.n040_sha256) != 64 or any(
    character not in "0123456789abcdef" for character in args_cli.n040_sha256.lower()
):
    raise ValueError("--n040_sha256 must contain a hexadecimal SHA-256 digest")
with args_cli.n040_artifact.open("rb") as artifact_stream:
    n040_actual_sha256 = hashlib.file_digest(artifact_stream, "sha256").hexdigest()
if n040_actual_sha256 != args_cli.n040_sha256.lower():
    raise ValueError(
        f"N040 encoder checksum mismatch: expected {args_cli.n040_sha256.lower()}, got {n040_actual_sha256}"
    )
args_cli.n040_sha256 = args_cli.n040_sha256.lower()
video_settings = VideoSettings.from_arguments(args_cli)
# ``anymani.distill.il.evaluate_family`` also uses the historical ``--student_checkpoint`` name, paired with
# ``--student_torchscript``.  Detect that wrapper before wiring the new direct-PPO option so existing IL calls
# continue to pass their own actor_override unchanged.
student_wrapper_invocation = any(
    value == "--student_torchscript" or value.startswith("--student_torchscript=") for value in sys.argv[1:]
)
direct_student_checkpoint = args_cli.student_checkpoint is not None and not student_wrapper_invocation

if args_cli.num_replicas < 1 or args_cli.steps < 1:
    raise ValueError("evaluation replicas and steps must be positive")
if args_cli.trace_stride < 0:
    raise ValueError("trace stride must be non-negative")
if args_cli.trace_rewards and args_cli.trace_stride != 1:
    raise ValueError("--trace_rewards requires --trace_stride 1 for complete reward attribution")
if args_cli.student_export_metadata is not None and args_cli.student_export is None:
    raise ValueError("--student_export_metadata requires --student_export")
if direct_student_checkpoint:
    if args_cli.cohort_lock is None:
        raise ValueError("student PPO fixed evaluation requires an explicit --cohort_lock")
    if args_cli.action_mode != "mean":
        raise ValueError("student PPO fixed evaluation requires deterministic mean actions")

    if args_cli.diagnostic_boundary_step is not None or args_cli.actor_switch_checkpoint is not None:
        raise ValueError("student PPO fixed evaluation cannot combine diagnostic Actor relay options")
    if args_cli.residual_off or args_cli.tip_only_intervention or args_cli.direct_logit_gain != 1:
        raise ValueError("student PPO fixed evaluation requires the unmodified deterministic control route")
if not math.isfinite(args_cli.direct_logit_gain) or args_cli.direct_logit_gain <= 0.0:
    raise ValueError("direct logit gain must be finite and positive")
if args_cli.cohort_transfer and args_cli.cohort_lock is None:
    raise ValueError("--cohort_transfer requires an explicit --cohort_lock")
if args_cli.actor_switch_checkpoint is not None and args_cli.diagnostic_boundary_step is None:
    raise ValueError("Actor handoff requires an explicit completed-step boundary")
if args_cli.diagnostic_boundary_step is not None:
    if not args_cli.diagnostic_only or not 0 < args_cli.diagnostic_boundary_step < args_cli.steps:
        raise ValueError("a diagnostic boundary must lie strictly inside a diagnostic trajectory")
    if (
        args_cli.action_mode != "mean"
        or args_cli.residual_off
        or args_cli.tip_only_intervention
        or args_cli.direct_logit_gain != 1
    ):
        raise ValueError("Actor relay uses unmodified deterministic checkpoint means")
    if args_cli.trace_stride != 1 or not args_cli.trace_rewards:
        raise ValueError("Actor relay requires complete stride-1 reward traces")
if args_cli.video is not None:
    if args_cli.video.suffix.lower() != ".mp4" or args_cli.video.exists():
        raise ValueError("--video requires a new .mp4 output path")
    args_cli.enable_cameras = True
if args_cli.reference is None:
    raise ValueError("pass the frozen 30-second reference JSON with --reference (paper wrappers provide it)")
args_cli.reference = (
    args_cli.reference.expanduser()
    if args_cli.reference.is_absolute()
    else (ANYMANI_ROOT / args_cli.reference).expanduser()
).resolve(strict=True)
cohort_lock_path = (
    args_cli.cohort_lock
    if args_cli.cohort_lock.is_absolute()
    else (ANYMANI_ROOT / args_cli.cohort_lock).resolve(strict=True)
)
cohort_document = parse_hand_asset_cohort_document(cohort_lock_path.read_bytes())
if not isinstance(cohort_document, dict) or cohort_document.get("schema_version") != "1.2.0":
    raise ValueError("cohort evaluation requires a schema-1.2 canonical-final lock")
cohort_members = cohort_document.get("members")
if not isinstance(cohort_members, list) or not cohort_members:
    raise ValueError("--cohort_lock must contain a non-empty members list")
selected_rows = tuple(range(len(cohort_members)))
mother_ids = tuple(str(member["provenance"]["mother_name"]) for member in cohort_members)
topology_ids = tuple(
    f"{member['provenance']['group_name']}/{member['provenance']['mother_name']}" for member in cohort_members
)
source_member_keys = tuple(f"{member['source_alias']}#{int(member['source_row'])}" for member in cohort_members)
support_manifest_path = cohort_lock_path
os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(cohort_lock_path)
asset_count = len(selected_rows)
if not 0 <= int(args_cli.viewer_asset_index) < asset_count:
    raise ValueError(f"--viewer_asset_index must lie in [0,{asset_count}), got {args_cli.viewer_asset_index}")
num_envs = asset_count * int(args_cli.num_replicas)
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
from anymani.distill.diagnostics.evaluation.rl import (  # noqa: E402
    palm_rotation as palm_rotation_metrics,
)
from anymani.distill.diagnostics.evaluation.rl import (  # noqa: E402
    palm_rotation_transfer,
)
from anymani.distill.diagnostics.evaluation.rl.palm_rotation import (  # noqa: E402
    PalmRotationReference,
    evaluate_cohort,
    evaluate_physical_support_trajectory_medians,
    evaluate_reliable_topology_coverage,
    evaluate_scale_ladder_cohort,
    evaluate_support_trajectory_medians,
)
from anymani.distill.diagnostics.recording.rl.palm_rotation import (  # noqa: E402
    write_selected_trajectories_hdf5,
)
from anymani.distill.models.palm_rotation_policy import (  # noqa: E402
    PalmRotationActorObservation,
    PalmRotationActorOutput,
    PalmRotationGeometry,
)
from anymani.distill.rl.palm_rotation_ppo import (  # noqa: E402
    PalmRotationRlGamesBuilder,
)
from anymani.distill.rl.runtime import (  # noqa: E402
    frozen_action_selection,
    frozen_actor_switch,
)


from anymani.distill.rl.runtime.frozen_action_selection import (  # noqa: E402
    select_frozen_actor_actions,
    validate_frozen_action_mode,
)
from anymani.distill.rl.runtime.frozen_actor_switch import (  # noqa: E402
    FrozenActorSwitch,
)
from anymani.distill.rl.runtime.palm_rotation_geometry import (  # noqa: E402
    build_palm_rotation_bf16_geometry_provider,
)
from anymani.distill.rl.runtime.palm_rotation_identity import (  # noqa: E402
    build_palm_rotation_method_identity,
    palm_rotation_code_provenance,
    validate_palm_rotation_evaluation_identity,
)
from anymani.distill.rl.runtime.palm_rotation_precision import (  # noqa: E402
    enforce_palm_rotation_precision,
)
from anymani.distill.rl.runtime.palm_rotation_vecenv import (  # noqa: E402
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
    PalmRotationRlGamesVecEnv,
    palm_rotation_float_shapes,
)
from anymani.distill.rl.runtime.student_evaluation import (  # noqa: E402
    FrozenPpoStudent,
    export_frozen_student_actor,
)
from anymani.tasks.hetero.config.generated.palm_rotation_mvp_env_cfg import (  # noqa: E402
    GOOD_PREGRASP_RESET_CFG,
    GeneratedPalmRotationMvpEnvCfg,
)
from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING  # noqa: E402
from anymani.tasks.hetero.mdp.actions import (  # noqa: E402
    PreloadAwareMaskedRelativeJointPositionAction,
)
from anymani.tasks.hetero.mdp.adr import (  # noqa: E402
    HeterogeneousAdrCfg,
    ObjectPositionAdrCfg,
)
from anymani.tasks.hetero.mdp.contact_state import (  # noqa: E402
    HETERO_CONTACT_STATE_ATTR,
)
from anymani.tasks.hetero.mdp.orientation_goal import (  # noqa: E402
    OrientationGoalCfg,
    configure_orientation_goal,
)

SINGLE_CLOSURE_NET_TURNS_MIN = 1.0
SINGLE_CLOSURE_DIRECTIONAL_CONSISTENCY_MIN = 0.7
SINGLE_CLOSURE_SAFE_REPLICA_FRACTION_MIN_EXCLUSIVE = 0.5


def _sha256(path: Path) -> str:
    "Handle sha256."

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actor_output(network: Any, observation: Mapping[str, torch.Tensor]) -> PalmRotationActorOutput:
    "Handle Actor output."

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
    if network.phase_period_steps is not None:
        return network.package.actor(actor_observation, geometry, phase_clock=observation["phase_clock"].float())
    return network.package.actor(actor_observation, geometry)


def _group_by_asset(values: torch.Tensor, replicas: int) -> np.ndarray:
    "Handle group by asset; shapes [A,R]."

    if values.shape != (asset_count * replicas,):
        raise ValueError("evaluation trajectory tensor disagrees with the asset-by-replica environment axis")
    return values.detach().cpu().numpy().reshape(replicas, asset_count).T


def main(*, collector: Any = None, actor_override: Any = None) -> None:
    "Handle main."

    if student_wrapper_invocation and args_cli.student_checkpoint is not None and actor_override is None:
        raise ValueError("--student_torchscript compatibility flags require the external IL actor_override")

    # Direct CLI student evaluation owns one callback instance.  The reference ``--checkpoint`` remains
    # the environment/MDP source, while every deterministic action is produced by this separate PPO Actor.
    if direct_student_checkpoint:
        if actor_override is not None:
            raise ValueError("--student_checkpoint cannot be combined with an external actor_override")
        student_checkpoint = cast(Path, args_cli.student_checkpoint)
        student_override = FrozenPpoStudent(
            student_checkpoint,
            expected_variant=args_cli.student_variant,
        )
        if args_cli.student_export is not None:
            student_sidecar = export_frozen_student_actor(
                student_checkpoint,
                args_cli.student_export,
                metadata_path=args_cli.student_export_metadata,
                expected_variant=args_cli.student_variant,
            )
            student_override.attach_export_metadata(student_sidecar)
        actor_override = student_override

    validate_frozen_action_mode(args_cli.action_mode, args_cli.action_seed)
    if collector is not None and (
        args_cli.cohort_lock is None
        or args_cli.action_mode != "mean"
        or args_cli.diagnostic_boundary_step is not None
        or args_cli.residual_off
        or args_cli.tip_only_intervention
        or args_cli.direct_logit_gain != 1
    ):
        raise ValueError("collection callback requires an explicit cohort and one unmodified mean Actor entry")
    if actor_override is not None and (
        collector is not None
        or args_cli.cohort_lock is None
        or args_cli.action_mode != "mean"
        or args_cli.diagnostic_boundary_step is not None
        or args_cli.residual_off
        or args_cli.tip_only_intervention
        or args_cli.direct_logit_gain != 1
    ):
        raise ValueError(
            "student evaluation requires one explicit cohort and an unmodified deterministic control route"
        )
    if args_cli.action_mode == "sample" and (
        args_cli.residual_off or args_cli.tip_only_intervention or args_cli.direct_logit_gain != 1
    ):
        raise ValueError("sample-mode control requires the unmodified checkpoint action/input routes")
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
    replacement_checkpoint_path = None
    replacement_checkpoint_sha = None
    replacement_actor_state = None
    if args_cli.actor_switch_checkpoint is not None:
        replacement_checkpoint_path = args_cli.actor_switch_checkpoint.expanduser().resolve()
        replacement_checkpoint = torch.load(replacement_checkpoint_path, map_location="cpu", weights_only=False)
        if replacement_checkpoint.get("anymani_identity") != checkpoint_identity:
            raise ValueError("Actor relay requires two checkpoints of the exact same learned method")
        actor_prefix = "a2c_network.package.actor."
        replacement_actor_state = {
            name[len(actor_prefix) :]: value
            for name, value in replacement_checkpoint["model"].items()
            if name.startswith(actor_prefix)
        }
        replacement_checkpoint_sha = _sha256(replacement_checkpoint_path)
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

    actor_tip_only = run_contract.get("actor_contact", "all") == "tip" or bool(args_cli.tip_only_intervention)
    for name in ("jnt_current", "jnt_history", "owner_contact"):
        getattr(env_cfg.observations.policy, name).params["tip_only"] = actor_tip_only
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = int(run_contract["seed"])
    env_cfg.viewer.env_index = int(args_cli.viewer_asset_index)
    device = str(run_contract.get("device", "cuda:0"))
    env_cfg.sim.device = device
    policy_dt_s = float(env_cfg.sim.dt) * int(env_cfg.decimation)  # units s
    evaluation_horizon_s = int(args_cli.steps) * policy_dt_s
    env_cfg.episode_length_s = evaluation_horizon_s
    env_cfg.commands.goal_pose.horizon_s = evaluation_horizon_s
    env_cfg.rewards.pose_keypoint.weight = float(run_contract.get("pose_keypoint_reward_weight", 1.0))  # units s
    if run_contract.get("pose_keypoint_mode", "full_pose") not in ("full_pose", "position_only"):
        raise ValueError("pose keypoint mode must be full_pose or position_only")
    env_cfg.rewards.pose_keypoint.params["position_only"] = (
        run_contract.get("pose_keypoint_mode", "full_pose") == "position_only"
    )
    env_cfg.rewards.rotation_progress.params["clip_rad_per_step"] = float(
        run_contract.get("rotation_progress_clip_rad_per_step", 0.025)
    )
    env_cfg.rewards.rotation_progress.weight = float(
        run_contract.get("rotation_progress_reward_weight", 5.0)
    )  # units rad
    env_cfg.rewards.goal_success.weight = float(run_contract.get("strict_goal_reward_weight", 10.0))
    env_cfg.rewards.joint_pose_anchor.weight = float(run_contract.get("joint_pose_anchor_weight", -0.5))

    env_cfg.curriculum.reward_release.params.update(
        release_start_turns=float(run_contract.get("reward_release_start_turns", 1.0)),
        release_end_turns=float(run_contract.get("reward_release_end_turns", 2.0)),
        ema_alpha=float(run_contract.get("reward_release_ema_alpha", 0.05)),
        release_floor=float(run_contract.get("reward_release_floor", 0.0)),
        reference_seconds=float(run_contract.get("reward_release_reference_seconds", 120.0)),
    )
    if "orientation_goal" in run_contract:
        configure_orientation_goal(
            env_cfg,
            OrientationGoalCfg(**run_contract["orientation_goal"]),
            training=False,
            adr=HeterogeneousAdrCfg(object_position=ObjectPositionAdrCfg(**run_contract["adr"]["object_position"])),
        )
    if args_cli.video is not None:
        env_cfg.viewer.resolution = (960, 720)
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (0.23, -0.22, 0.28)
        env_cfg.viewer.lookat = (0.0, 0.075, 0.05)
    video_settings.apply_viewer(env_cfg)
    env = gym.make(
        "AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video is not None else None,
    )
    provider = build_palm_rotation_bf16_geometry_provider(
        ASSET_BINDING,
        artifact_path=args_cli.n040_artifact,
        artifact_sha256=args_cli.n040_sha256,
        device=device,
    )
    prototype_index = torch.tensor(ASSET_BINDING.asset_index_by_env(num_envs), dtype=torch.long, device=device)
    transport = PalmRotationRlGamesVecEnv(
        env,
        geometry_provider=provider,
        prototype_index=prototype_index,
        rl_device=device,
        clip_observations=100.0,
        clip_actions=1.0,
        phase_period_steps=run_contract.get("phase_period_steps"),
    )

    video_writer: Any = None
    video_frames = 0
    video_render_report: dict[str, object] = {}
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
        transfer_validation = None
        from anymani.publication.compatibility import manifest_semantics, pregrasp_semantics

        catalog_revision_changed = pregrasp_semantics(current_identity["pregrasp"]) != pregrasp_semantics(
            checkpoint_identity["pregrasp"]
        )
        if actor_override is not None:
            # Frozen IL/PPO Actor evaluation has an explicit bounded certificate.  It proves only
            # old/new frozen interface and pure task-math compatibility; full PPO training/refactor AST
            # equivalence remains owned by the regular transfer/resume gates below.
            transfer_validation = palm_rotation_transfer.validate_frozen_evaluation(
                current_identity,
                checkpoint_identity,
                root=ANYMANI_ROOT,
                implementation_certificate=implementation_certificate,
                actor_override=actor_override,
            )
        elif args_cli.cohort_transfer or catalog_revision_changed:
            if not args_cli.cohort_transfer and manifest_semantics(current_identity["manifest"]) != manifest_semantics(
                checkpoint_identity["manifest"]
            ):
                raise RuntimeError("new evaluation support requires explicit --cohort_transfer")
            transfer_validation = palm_rotation_transfer.validate_transfer_evaluation(
                current_identity,
                checkpoint_identity,
                root=ANYMANI_ROOT,
                implementation_certificate=implementation_certificate,
            )
        else:
            validate_palm_rotation_evaluation_identity(
                runtime_identity=current_identity,
                checkpoint_identity=checkpoint_identity,
                implementation_certificate=implementation_certificate,
            )
        checkpoint_backend_commit = run_contract.get("rl_games_backend_commit")
        if backend_info.git_commit is not None:
            backend_matches = checkpoint_backend_commit == backend_info.git_commit
        else:
            saved_version = run_contract.get("rl_games_backend_version")
            saved_identity_source = run_contract.get("rl_games_backend_identity_source")
            legacy_pinned_commit = checkpoint_backend_commit == backend_info.expected_commit
            explicit_version_identity = (
                saved_identity_source == "distribution_version" and saved_version == backend_info.expected_version
            )
            backend_matches = (
                backend_info.identity_source == "distribution_version"
                and backend_info.package_version == backend_info.expected_version
                and (legacy_pinned_commit or explicit_version_identity)
            )
        if not backend_matches:
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
                    "sigma_mode": str(run_contract.get("sigma_mode", "global")),
                    "recovery_sigma_floor": run_contract.get("recovery_sigma_floor"),
                    "phase_period_steps": run_contract.get("phase_period_steps"),
                    "compile_mode": None,
                },
                "anymani_identity": checkpoint_identity,
            }
        )
        input_shape = {
            **palm_rotation_float_shapes(run_contract.get("phase_period_steps")),
            **PALM_ROTATION_BOOL_SHAPES,
            **PALM_ROTATION_INT16_SHAPES,
        }
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
            network.package.actor.residual_enabled = False
        network.eval()
        frozen_actor_state = (
            {name: value.detach().clone() for name, value in network.package.actor.state_dict().items()}
            if args_cli.action_mode == "sample"
            else None
        )
        direct_gain_hook = None
        if args_cli.direct_logit_gain != 1.0:
            if checkpoint_identity["policy"]["arm"] not in {"direct", "direct_token"}:
                raise ValueError("--direct_logit_gain is defined only for a Direct-arm checkpoint")
            # values 24 rad; units rad

            frozen_actor_state = {
                name: value.detach().clone() for name, value in network.package.actor.state_dict().items()
            }
            direct_head = network.package.actor.direct_head
            if not isinstance(direct_head, torch.nn.Module):
                raise TypeError("Direct logit-gain diagnostic requires an action readout module")
            direct_gain_hook = direct_head.register_forward_hook(
                lambda _module, _inputs, logits: cast(torch.Tensor, logits) * float(args_cli.direct_logit_gain)
            )

        actor_relay = (
            FrozenActorSwitch(
                network.package.actor, replacement_actor_state, boundary_step=args_cli.diagnostic_boundary_step
            )
            if args_cli.diagnostic_boundary_step is not None
            else None
        )
        boundary_arrays: dict[str, np.ndarray] | None = None
        relay_report = None
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        goal_count = torch.zeros(num_envs, dtype=torch.float32, device=device)
        orientation_goal_count = torch.zeros_like(goal_count)
        frontier_count = torch.zeros_like(goal_count)
        frontier_delta_recount = torch.zeros_like(goal_count)
        max_positive_rotation_rad = torch.zeros_like(goal_count)  # units M, rad
        net_turns = torch.zeros_like(goal_count)
        path_turns = torch.zeros_like(goal_count)
        duration_s = torch.zeros_like(goal_count)
        termination_drop = torch.zeros(num_envs, dtype=torch.bool, device=device)
        termination_axis = torch.zeros_like(termination_drop)
        termination_timeout = torch.zeros_like(termination_drop)

        # values 25 mm; units mm
        diagnostic_steps = torch.zeros(num_envs, dtype=torch.float32, device=device)
        orientation_error_sum_m = torch.zeros_like(goal_count)  # units m
        orientation_error_max_m = torch.zeros_like(goal_count)  # units m
        orientation_error_final_m = torch.zeros_like(goal_count)  # units m
        position_error_sum_m = torch.zeros_like(goal_count)  # units m
        position_error_max_m = torch.zeros_like(goal_count)  # units m
        position_error_final_m = torch.zeros_like(goal_count)  # units m
        orientation_gate_steps = torch.zeros_like(goal_count)  # units mm
        position_gate_steps = torch.zeros_like(goal_count)  # units mm
        goal_success_pulse_recount = torch.zeros_like(goal_count)
        first_position_gate_failure_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)  # units mm
        last_position_gate_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        last_goal_success_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        net_turns_at_last_goal_success = torch.zeros_like(goal_count)
        position_error_at_last_goal_success_m = torch.zeros_like(goal_count)  # units m
        observation = transport.reset()["obs"]
        for policy_callback in (collector, actor_override):
            if policy_callback is None:
                continue
            policy_callback.start(
                checkpoint_path=checkpoint_path,
                checkpoint_identity=checkpoint_identity,
                runtime_identity=current_identity,
                binding=ASSET_BINDING,
                observation=observation,
                actor=network.package.actor,
                cohort_path=support_manifest_path,
                cohort_members=cohort_members,
                steps=int(args_cli.steps),
                replicas=int(args_cli.num_replicas),
            )
        action_generator = (
            torch.Generator(device=device).manual_seed(args_cli.action_seed)
            if args_cli.action_mode == "sample"
            else None
        )
        if args_cli.video is not None or not args_cli.headless:
            import omni.usd

            x, y, z = (
                transport.unwrapped.scene["object"].data.root_pos_w[int(args_cli.viewer_asset_index)].cpu().tolist()
            )
            camera_target = video_settings.lookat or (float(x), float(y), float(z) - 0.015)
            camera_eye = video_settings.eye or (float(x) + 0.23, float(y) - 0.25, float(z) + 0.23)
            env_cfg.viewer.eye, env_cfg.viewer.lookat = camera_eye, camera_target
            env_cfg.viewer.origin_type = "world"
            camera_controller = transport.unwrapped.viewport_camera_controller
            if camera_controller is not None:
                camera_controller.cfg.eye, camera_controller.cfg.lookat = camera_eye, camera_target
                camera_controller.set_view_env_index(int(args_cli.viewer_asset_index))
                camera_controller.update_view_to_world()
                camera_controller.update_view_location(eye=camera_eye, lookat=camera_target)
            else:
                transport.unwrapped.sim.set_camera_view(camera_eye, camera_target, env_cfg.viewer.cam_prim_path)

            video_render_report = video_settings.apply_renderer(
                omni.usd.get_context().get_stage(),
                visible_environment_ids=(
                    (int(args_cli.viewer_asset_index),) if video_settings.preset == "paper_white" else None
                ),
            )
        if args_cli.video is not None:
            import imageio.v2 as imageio

            args_cli.video.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(
                str(args_cli.video), **cast(dict[str, Any], video_settings.writer_kwargs(policy_dt_s=policy_dt_s))
            )
            for _ in range(8):
                transport.unwrapped.render()
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
                step_started_at = time.perf_counter()
                if video_writer is not None and bool(active[int(args_cli.viewer_asset_index)].item()):
                    video_writer.append_data(transport.unwrapped.render())
                    video_frames += 1
                at_boundary = actor_relay is not None and _step == actor_relay.boundary_step
                if at_boundary:
                    assert actor_relay is not None

                    hand = transport.unwrapped.scene["robot"]
                    obj = transport.unwrapped.scene["object"]
                    action_name = getattr(env_cfg.observations.policy, "jnt_current").params["action_name"]
                    action_term = transport.unwrapped.action_manager.get_term(action_name)
                    if not isinstance(action_term, PreloadAwareMaskedRelativeJointPositionAction):
                        raise TypeError("Actor relay requires the original target-buffer controller")
                    continuity = {"obs__" + key: value for key, value in observation.items()}
                    continuity.update(
                        {
                            "controller__current_targets": action_term.current_targets,
                            "controller__previous_targets": action_term.previous_targets,
                            "controller__pregrasp_targets": action_term.pregrasp_targets,
                            "physics__joint_position": hand.data.joint_pos,
                            "physics__joint_velocity": hand.data.joint_vel,
                            "physics__object_root_state": obj.data.root_state_w,
                            "active_first_trajectory": active,
                        }
                    )
                    old_boundary_mean = _actor_output(network, observation).mean.detach().clone()
                    boundary_arrays = {key: value.detach().cpu().numpy().copy() for key, value in continuity.items()}
                    boundary_arrays["actor_mean_before"] = old_boundary_mean.cpu().numpy()
                    actor_relay.apply(_step, continuity)
                    if torch.cuda.mem_get_info(device)[0] < 2 * 1024**3:
                        raise RuntimeError("Actor relay crossed the 2GiB driver-free boundary")
                if actor_override is not None:
                    actions = actor_override.act(_step, observation)
                    action_trace = {}
                else:
                    actor_output = _actor_output(network, observation)
                    if at_boundary:
                        assert boundary_arrays is not None
                        boundary_arrays["actor_mean_after"] = actor_output.mean.detach().cpu().numpy().copy()
                    actions, action_trace = select_frozen_actor_actions(
                        actor_output.mean,
                        actor_output.log_std,
                        observation["jnt_valid"].bool(),
                        mode=args_cli.action_mode,
                        generator=action_generator,
                    )
                    if collector is not None:
                        actions = collector.act(_step, observation, actor_output.mean, actor_output.log_std)
                next_observation, _reward, done, _extras = transport.step(actions)
                if collector is not None:
                    collector.after_step(done)
                if actor_override is not None:
                    actor_override.after_step()
                command = transport.unwrapped.command_manager.get_term("goal_pose")
                snapshot = command.post_physics_evaluation_snapshot
                if not bool(snapshot["valid"].all().item()):
                    raise RuntimeError("fixed evaluator observed an invalid pre-reset snapshot")
                if trace_capacity and _step % int(args_cli.trace_stride) == 0:
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
                            "orientation_error_rad",
                            "goal_advance_pulse",
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
                            "post_goal_axis_alignment": (command.goal_normal_alignment),
                        }
                    )
                    values.update(action_trace)
                    if "phase_clock" in observation:
                        values["pre_phase_clock"] = observation["phase_clock"]  # shapes [N,2]
                    if actor_relay is not None:
                        values["actor_checkpoint_phase"] = torch.full(
                            (num_envs,), actor_relay.phase, dtype=torch.int8, device=device
                        )
                    if args_cli.trace_rewards:
                        values["reward_terms_step"] = transport.unwrapped.reward_manager._step_reward * policy_dt_s
                        values["reward_step"] = _reward
                        values["reward_release_gain"] = observation["critic_reward_release"].squeeze(-1)
                    if not trace_buffers:
                        trace_buffers = {
                            name: torch.empty((trace_capacity, *value.shape), dtype=value.dtype, device=value.device)
                            for name, value in values.items()
                        }
                    for name, value in values.items():
                        trace_buffers[name][trace_count].copy_(value)
                    trace_count += 1
                goal_count[active] = snapshot["completed_subgoals"][active]
                orientation_goal_count[active] = snapshot["completed_orientation_subgoals"][active]
                frontier_count[active] = snapshot["rotation_frontier_count"][active]
                max_positive_rotation_rad[active] = snapshot["max_positive_net_rotation_rad"][active]
                net_turns[active] = snapshot["net_rotation_rad"][active] / two_pi
                path_turns[active] = snapshot["absolute_path_rotation_rad"][active] / two_pi
                duration_s[active] = snapshot["episode_duration_s"][active]

                orientation_error = snapshot["orientation_keypoint_error_m"]  # shapes [N]; units m
                position_error = snapshot["position_error_m"]  # shapes [N]; units m
                orientation_gate = (
                    snapshot["orientation_error_rad"] <= float(command.cfg.orientation_success_threshold_rad)
                    if command.cfg.orientation_only_advance
                    else orientation_error < float(command.cfg.orientation_success_threshold_m)
                )
                position_gate = position_error < float(command.cfg.position_success_threshold_m)
                success_pulse = snapshot["goal_success_pulse"].bool()
                active_float = active.to(dtype=goal_count.dtype)
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
                    sleep_s = policy_dt_s - (time.perf_counter() - step_started_at)  # units Hz
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                if not bool(active.any().item()):
                    break
        if bool(active.any().item()):
            raise RuntimeError(
                f"fixed evaluation ended before {int(active.sum().item())}/{num_envs} first trajectories terminated"
            )
        if actor_relay is not None:
            relay_report = actor_relay.finish()
        if collector is not None:
            collector.finish(
                net_turns=net_turns,
                path_turns=path_turns,
                duration_s=duration_s,
                termination_drop=termination_drop,
                termination_axis=termination_axis,
                termination_timeout=termination_timeout,
                policy_step_count=diagnostic_steps.long(),
            )
        if actor_override is not None:
            actor_override.finish()
        if video_writer is not None:
            video_writer.close()
            video_writer = None
        if frozen_actor_state is not None:
            if any(
                not torch.equal(value, network.package.actor.state_dict()[name])
                for name, value in frozen_actor_state.items()
            ):
                raise RuntimeError("frozen action diagnostic changed Actor parameters or buffers")
            if direct_gain_hook is not None:
                direct_gain_hook.remove()

        safe_diagnostic_steps = diagnostic_steps.clamp_min(1.0)
        frontier_from_maximum = torch.floor(
            max_positive_rotation_rad / float(command.cfg.rotation_frontier_interval_rad)
        )  # units M
        expected_goals_from_positive_net = 12.0 * torch.clamp(net_turns, min=0.0)
        goal_turn_ratio = torch.where(
            expected_goals_from_positive_net > torch.finfo(goal_count.dtype).eps,
            goal_count / expected_goals_from_positive_net.clamp_min(torch.finfo(goal_count.dtype).eps),
            torch.zeros_like(goal_count),
        )
        arrays = {
            "goal_count": _group_by_asset(goal_count, args_cli.num_replicas).astype(np.float32),
            "orientation_goal_count": _group_by_asset(orientation_goal_count, args_cli.num_replicas).astype(np.float32),
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
        reliable_protocol_matched = (
            int(args_cli.steps) == 600
            and int(args_cli.num_replicas) == 16
            and args_cli.action_mode == "mean"
            and not args_cli.diagnostic_only
            and actor_tip_only
            and not args_cli.residual_off
            and not args_cli.tip_only_intervention
            and args_cli.direct_logit_gain == 1.0
        )
        reliable_coverage = (
            evaluate_reliable_topology_coverage(
                physical_asset_results, topology_ids=topology_ids, horizon_s=evaluation_horizon_s
            )
            if topology_ids and reliable_protocol_matched
            else None
        )
        scale_protocol_matched = reliable_protocol_matched and not args_cli.cohort_transfer
        scale_ladder_result = (
            evaluate_scale_ladder_cohort(
                physical_asset_results,
                finite_and_identity_valid=physical_finite,
            )
            if asset_count in (16, 64, 128) and scale_protocol_matched
            else None
        )
        cohort = (
            evaluate_cohort(
                seed=int(run_contract["seed"]),
                asset_results=asset_results,
                finite_and_identity_valid=finite_and_identity_valid,
            )
            if asset_count == 80 and args_cli.action_mode == "mean" and not args_cli.diagnostic_only
            else None
        )
        closure_passed_assets = sum(result.viability_passed for result in physical_asset_results)
        closure_passed = bool(physical_finite and closure_passed_assets == asset_count)
        pair_results = ()
        pair_counts = Counter(result.outcome for result in pair_results)
        closure_thresholds = {
            "net_turns_median_min": SINGLE_CLOSURE_NET_TURNS_MIN,
            "directional_consistency_min": SINGLE_CLOSURE_DIRECTIONAL_CONSISTENCY_MIN,
            "safe_replica_fraction_min_exclusive": SINGLE_CLOSURE_SAFE_REPLICA_FRACTION_MIN_EXCLUSIVE,
        }

        output = args_cli.output
        if output is None:
            intervention = "-residual-off" if args_cli.residual_off else ""
            intervention += "-tip-mask" if args_cli.tip_only_intervention else ""
            intervention += "-cohort-transfer" if args_cli.cohort_transfer else ""
            if args_cli.direct_logit_gain != 1.0:
                intervention += f"-logit-gain-{args_cli.direct_logit_gain:g}"
            if args_cli.action_mode == "sample":
                intervention += f"-sample-s{args_cli.action_seed}"
            if args_cli.diagnostic_only:
                intervention += "-diagnostic"
            if args_cli.diagnostic_boundary_step is not None:
                intervention += f"-boundary{args_cli.diagnostic_boundary_step}"
            if replacement_checkpoint_path is not None:
                intervention += f"-to-{replacement_checkpoint_path.stem}"
            output = (
                checkpoint_path.parent.parent
                / "evaluation"
                / f"{checkpoint_path.stem}-fixed{evaluation_horizon_s:g}s-r{args_cli.num_replicas}{intervention}.json"
            )
        output = output.expanduser().resolve()
        if any(
            path.exists()
            for path in (
                output,
                output.with_suffix(".h5"),
                output.with_suffix(".trace.h5"),
                output.with_suffix(".boundary.h5"),
            )
        ):
            raise FileExistsError(f"evaluation output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        hdf5_path = output.with_suffix(".h5")
        checkpoint_sha = _sha256(checkpoint_path)
        boundary_result = None
        if boundary_arrays is not None:
            boundary_path = output.with_suffix(".boundary.h5")
            write_selected_trajectories_hdf5(
                boundary_path,
                arrays=boundary_arrays,
                metadata={
                    "artifact_type": "anymani.palm_rotation_actor_boundary",
                    "method_identity_digest": checkpoint_identity["identity_digest"],
                    "initial_checkpoint_sha256": checkpoint_sha,
                    "replacement_checkpoint_sha256": replacement_checkpoint_sha,
                    "boundary_step": args_cli.diagnostic_boundary_step,
                    "axes": "environment=replica*num_assets+asset,features",
                    "source_member_keys": list(source_member_keys),
                    "num_assets": asset_count,
                    "replicas_per_asset": args_cli.num_replicas,
                },
            )
            boundary_result = {"path": str(boundary_path), "sha256": _sha256(boundary_path)}
        relay_metadata = None
        if actor_relay is not None:
            assert relay_report is not None
            relay_metadata = {
                **relay_report,
                "initial_checkpoint_sha256": checkpoint_sha,
                "replacement_checkpoint": str(replacement_checkpoint_path) if replacement_checkpoint_path else None,
                "replacement_checkpoint_sha256": replacement_checkpoint_sha,
                "boundary_time_s": actor_relay.boundary_step * policy_dt_s,
                "first_replacement_action_step": actor_relay.boundary_step + 1 if replacement_checkpoint_path else None,
                "boundary_snapshot": boundary_result,
            }
        evaluation_identity = {
            "schema_version": "1.7.0",
            "method_identity_digest": checkpoint_identity["identity_digest"],
            "execution_identity_digest": current_identity["identity_digest"],
            "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
            "action_selection_source_sha256": _sha256(Path(frozen_action_selection.__file__)),
            "actor_switch_source_sha256": (
                _sha256(Path(frozen_actor_switch.__file__)) if actor_relay is not None else None
            ),
            "physical_metrics_source_sha256": _sha256(Path(palm_rotation_metrics.__file__)),
            "transfer_validator_source_sha256": _sha256(Path(palm_rotation_transfer.__file__)),
            "transfer_validation": transfer_validation,
            "gui_viewer": (
                {
                    "asset_index": int(args_cli.viewer_asset_index),
                    "replica_index": 0,
                    "simulation_num_envs": num_envs,
                    "render_settings": video_settings.metadata(policy_dt_s=policy_dt_s, viewer=env_cfg.viewer),
                    "render_changes": video_render_report,
                    "trajectory_source": "live_policy",
                }
                if not args_cli.headless
                else None
            ),
            "video": (
                {
                    "path": str(args_cli.video.resolve()),
                    "asset_index": int(args_cli.viewer_asset_index),
                    "replica_index": 0,
                    "frames": video_frames,
                    "fps": round(1.0 / policy_dt_s),
                    "render_settings": video_settings.metadata(policy_dt_s=policy_dt_s, viewer=env_cfg.viewer),
                    "render_changes": video_render_report,
                    "trajectory_source": "live_policy",
                    "simulation_num_envs": num_envs,
                    "frame_semantics": "pre-action first-trajectory states; automatic-reset frames excluded",
                }
                if args_cli.video is not None
                else None
            ),
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
                "trace_rewards": bool(args_cli.trace_rewards),
                "evaluation_role": (
                    "diagnostic" if args_cli.diagnostic_only or args_cli.action_mode == "sample" else "capability"
                ),
                "actor_relay": relay_metadata,
                "actor_contact_intervention": "tip-only-mask" if args_cli.tip_only_intervention else "none",
                "cohort_transfer": bool(args_cli.cohort_transfer),
                "direct_logit_gain_intervention": float(args_cli.direct_logit_gain),
                "direct_logit_gain_parameter_check": (
                    "bitwise-equal" if args_cli.direct_logit_gain != 1.0 else "not-applicable"
                ),
                "actor_contact": "tip-only-binary" if actor_tip_only else "all-owner-binary-no-force",
                "scale_ready_protocol_matched": scale_protocol_matched,
                "reliable_topology_coverage_protocol_matched": reliable_coverage is not None,
                "reliable_topology_coverage_thresholds": reliable_coverage["thresholds"] if reliable_coverage else None,
                "reference_horizon_s": reference_doc.get("protocol", {}).get("horizon_s"),
                "reference_horizon_matched": reference_doc.get("protocol", {}).get("horizon_s") == evaluation_horizon_s,
                "deterministic_actor_mean": args_cli.action_mode == "mean",
                "action_selection": {
                    "mode": args_cli.action_mode,
                    "seed": args_cli.action_seed,
                    "generator_device": str(action_generator.device) if action_generator is not None else None,
                    "distribution": "masked-tanh-normal" if args_cli.action_mode == "sample" else None,
                    "latent_action_epsilon": (
                        frozen_action_selection.LATENT_ACTION_EPSILON if args_cli.action_mode == "sample" else None
                    ),
                    "actor_parameter_check": "bitwise-equal" if args_cli.action_mode == "sample" else "not-applicable",
                },
                "first_trajectory_only": True,
                "pregrasp_rank": 0,
                "goal_advance": (
                    "angle-only" if env_cfg.commands.goal_pose.orientation_only_advance else "qualified-pose"
                ),
                "goal_reference": env_cfg.commands.goal_pose.goal_reference,
                "orientation_tolerance_rad": (
                    env_cfg.commands.goal_pose.orientation_success_threshold_rad
                    if env_cfg.commands.goal_pose.orientation_only_advance
                    else None
                ),
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
        if collector is not None:
            evaluation_identity["teacher_collection"] = collector.metadata
            evaluation_identity["protocol"].update(
                evaluation_role="teacher-data-collection",
                action_mode=collector.mode,
                action_seed=collector.seed,
                deterministic_actor_mean=collector.mode == "mean",
                action_selection={
                    "mode": collector.mode,
                    "seed": collector.seed,
                    "distribution": "masked-tanh-normal" if collector.mode == "sample" else None,
                    "latent_location": "frozen backend _action_to_latent",
                    "actor_parameter_check": "bitwise-equal",
                },
            )
        student_document_updates = None
        if actor_override is not None:
            try:
                student_updates = actor_override.identity_updates()
            except (TypeError, ValueError) as error:
                write_selected_trajectories_hdf5(
                    output.with_suffix(".unreported.h5"),
                    arrays=arrays,
                    metadata={
                        "artifact_type": "anymani.student_unreported_physical_results",
                        "metadata_error": f"{type(error).__name__}: {error}",
                        "reference_checkpoint": str(checkpoint_path),
                        "reference_checkpoint_sha256": checkpoint_sha,
                        "student_checkpoint": str(args_cli.student_checkpoint),
                        "student_checkpoint_sha256": (
                            _sha256(args_cli.student_checkpoint) if args_cli.student_checkpoint is not None else None
                        ),
                        "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
                        "cohort_lock_sha256": _sha256(support_manifest_path),
                        "dataset_rows": list(ASSET_BINDING.dataset_rows),
                        "source_member_keys": list(source_member_keys),
                        "mother_ids": list(mother_ids),
                        "requested_policy_steps": int(args_cli.steps),
                        "executed_policy_steps": _step + 1,
                        "replicas_per_asset": int(args_cli.num_replicas),
                        "policy_dt_s": policy_dt_s,
                        "first_trajectory_only": True,
                        "student_freeze_check_completed": True,
                        "report_complete": False,
                    },
                )
                raise
            student_document_updates = student_updates.pop("document_updates")
            evaluation_identity.update(student_updates)
            evaluation_identity["checkpoint_sha256"] = student_updates["student_checkpoint_sha256"]
            student_action_source = (
                "frozen-posttrain-ppo-student-actor"
                if isinstance(actor_override, FrozenPpoStudent)
                else "frozen-offline-student-torchscript"
            )
            evaluation_identity["protocol"].update(
                evaluation_role="student-diagnostic" if args_cli.diagnostic_only else "student-capability",
                action_mode="mean",
                action_seed=None,
                deterministic_actor_mean=True,
                action_selection={
                    "mode": "mean",
                    "source": student_action_source,
                    "actor_parameter_check": "bitwise-equal",
                },
            )
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
                name: (
                    value[:trace_count]
                    .reshape(trace_count, args_cli.num_replicas, asset_count, *value.shape[2:])
                    .transpose(1, 2)
                    .cpu()
                    .numpy()
                )
                for name, value in trace_buffers.items()
            }
            trace_arrays["policy_step"] = np.arange(trace_count, dtype=np.int64) * int(args_cli.trace_stride) + 1
            reward_recount_error = None
            if args_cli.trace_rewards:
                error = np.abs(trace_arrays["reward_terms_step"].sum(axis=-1) - trace_arrays["reward_step"])
                reward_recount_error = float(error[trace_arrays["active"]].max())
                if not np.isfinite(reward_recount_error) or reward_recount_error > 1.0e-4:
                    raise RuntimeError("reward term trace does not reconstruct the actual step reward")
            write_selected_trajectories_hdf5(
                trace_path,
                arrays=trace_arrays,
                metadata={
                    **evaluation_identity,
                    "trace_stride": int(args_cli.trace_stride),
                    "axes": "time,asset,replica,feature",
                    "sensor_names": list(ASSET_BINDING.contact_layout.state_sensor_names),
                    "post_state_valid_semantics": (
                        "sensor forces/bits, post q and post axis alignment exclude automatic-reset terminal rows"
                    ),
                    "pre_owner_contact_semantics": (
                        "actor input before the applied action; aggregates are post-physics/pre-reset"
                    ),
                    **(
                        {
                            "pre_phase_clock_semantics": "sin/cos from physical episode counter before applied action",
                            "phase_clock": current_identity["policy"]["phase_clock"],
                        }
                        if "phase_clock" in current_identity["policy"]
                        else {}
                    ),
                    "reward_term_names": (
                        list(transport.unwrapped.reward_manager.active_terms) if args_cli.trace_rewards else None
                    ),
                    "reward_recount_max_abs_error": reward_recount_error,
                },
            )
            trace_result = {"path": str(trace_path), "sha256": _sha256(trace_path), "samples": trace_count}

        goal_tracking_assets: list[dict[str, Any]] = []
        frontier_assets: list[dict[str, Any]] = []
        reference_ratio = float(reference.command_turn_ratio)
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
                "anymani.palm_rotation_stochastic_diagnostic"
                if args_cli.action_mode == "sample"
                else (
                    "anymani.palm_rotation_actor_relay_diagnostic"
                    if actor_relay is not None
                    else (
                        "anymani.palm_rotation_frozen_diagnostic"
                        if args_cli.diagnostic_only
                        else (
                            "anymani.palm_rotation_mvp80_fixed_evaluation"
                            if asset_count == 80
                            else "anymani.palm_rotation_support_fixed_evaluation"
                        )
                    )
                )
            ),
            "schema_version": "1.4.0",
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
                "closure_passed_assets": closure_passed_assets if not args_cli.diagnostic_only else None,
                "closure_passed": closure_passed if not args_cli.diagnostic_only else None,
                "asset_results": [asdict(result) for result in asset_results],
            },
            "physical_rotation": {
                "semantics": "fixed-ADR-0 net turns, path directionality, joint drop/axis survival, and 30deg frontier",
                "finite_and_identity_valid": physical_finite,
                "viability_passed_assets": closure_passed_assets if not args_cli.diagnostic_only else None,
                "all_assets_viable": closure_passed if not args_cli.diagnostic_only else None,
                "asset_results": [asdict(result) for result in physical_asset_results],
                "frontier_recount": frontier_assets,
            },
            "scale_ladder": asdict(scale_ladder_result) if scale_ladder_result is not None else None,
            "reliable_topology_coverage": reliable_coverage,
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
        if collector is not None:
            document["artifact_type"] = "anymani.family_teacher_collection_evaluation"
            document["teacher_collection"] = collector.metadata
            document["support"]["closure_passed"] = None
            document["support"]["closure_passed_assets"] = None
            document["cohort"] = None
            document["scale_ladder"] = None
        if actor_override is not None:
            assert student_document_updates is not None
            document.update(student_document_updates)
            document["artifact_type"] = (
                "anymani.family_student_frozen_diagnostic"
                if args_cli.diagnostic_only
                else "anymani.family_student_fixed_evaluation"
            )
            document["checkpoint"] = student_document_updates["student_checkpoint_path"]
            document["checkpoint_epoch"] = student_document_updates["training_state"]["epoch"]
            if "frame" in student_document_updates["training_state"]:
                document["checkpoint_frame"] = student_document_updates["training_state"]["frame"]
            else:
                document.pop("checkpoint_frame")
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
                    "formal_asset_passed_count": (
                        sum(result.passed for result in asset_results)
                        if args_cli.action_mode == "mean" and not args_cli.diagnostic_only
                        else None
                    ),
                    "scale_ready_applicable": scale_ladder_result is not None,
                    "scale_ready_passed": scale_ladder_result.passed if scale_ladder_result is not None else None,
                    "scale_ready_passed_assets": (
                        scale_ladder_result.scale_ready_assets if scale_ladder_result is not None else None
                    ),
                    "sustained_rotation_closure_passed": closure_passed if not args_cli.diagnostic_only else None,
                    "sustained_rotation_closure_passed_assets": (
                        closure_passed_assets if not args_cli.diagnostic_only else None
                    ),
                    "reliable_passed_assets": reliable_coverage["passed_asset_count"] if reliable_coverage else None,
                    "reliable_passed_topologies": (
                        reliable_coverage["passed_topology_count"] if reliable_coverage else None
                    ),
                    "reliable_topology_count": reliable_coverage["topology_count"] if reliable_coverage else None,
                    "passed_by_cell": cohort.passed_by_cell if cohort is not None else None,
                    "pair_counts": dict(pair_counts),
                },
                sort_keys=True,
            )
        )
    finally:
        if collector is not None:
            collector.close()
        if video_writer is not None:
            video_writer.close()
        transport.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        failure_base = args_cli.output or args_cli.checkpoint.with_name(f"{args_cli.checkpoint.stem}-evaluation.json")
        failure_path = failure_base.with_suffix(".failure.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "reference_checkpoint": str(args_cli.checkpoint),
                    "student_checkpoint": str(args_cli.student_checkpoint) if args_cli.student_checkpoint else None,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
