'Train the paper palm-rotation policy with the versioned PPO configuration and explicit paper asset cohorts.'

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml
from isaaclab.app import AppLauncher

from anymani.assets.bank.cohort import parse_hand_asset_cohort_document
from anymani.assets.bank.path_utils import resolve_anymani_root

ANYMANI_ROOT = resolve_anymani_root()
TASK_ID = "AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0"
DEFAULT_MANIFEST = Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml")


def _load_mvp80_rows(path: Path) -> tuple[int, ...]:
    'Load mvp80 rows.'

    resolved = path if path.is_absolute() else ANYMANI_ROOT / path
    payload = resolved.read_bytes()  # exact checkpoint provenance bytes
    document = yaml.safe_load(payload)
    if not isinstance(document, dict):
        raise ValueError("MVP80 manifest must contain a YAML mapping")
    rows = tuple(int(row) for row in document.get("selected_rows", ()))  # formal ppo.yaml rows
    if len(rows) != 80 or len(set(rows)) != 80:
        raise ValueError(f"MVP80 training requires exactly 80 unique selected_rows, got {len(rows)}")
    return rows


def _select_support_rows(mvp80_rows: tuple[int, ...], raw_support_rows: str | None) -> tuple[int, ...]:
    'Select support rows.'

    if raw_support_rows is None:
        return mvp80_rows
    selected = tuple(int(item.strip()) for item in raw_support_rows.split(",") if item.strip())
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("--support_rows must contain unique formal rows")
    outside = tuple(row for row in selected if row not in set(mvp80_rows))
    if outside:
        raise ValueError(f"--support_rows must be a subset of the frozen MVP80 manifest, got outside rows={outside}")
    return selected


parser = argparse.ArgumentParser(description="Train the 80-hand palm-rotation MVP with structured rl_games PPO.")
parser.add_argument("--data-dir", type=Path, default=ANYMANI_ROOT / "data", help="Verified publication data directory.")
parser.add_argument(
    "--paper-cohort",
    choices=("leap_training", "allegro_training"),
    default=None,
    help="Use the portable ready training cohort and its packaged pregrasp catalog.",
)
parser.add_argument(
    "--asset_manifest", type=Path, default=DEFAULT_MANIFEST, help="Versioned 80-row selection manifest."
)
parser.add_argument(
    "--support_rows",
    type=str,
    default=None,
    help="Explicit ordered subset of frozen MVP80 rows for single/few-embodiment closure.",
)
parser.add_argument(
    "--cohort_lock",
    type=Path,
    default=None,
    help="Member-level cohort lock; mutually exclusive with --support_rows and the legacy MVP80 axis.",
)
parser.add_argument("--num_envs", type=int, default=None, help="2560 formal, 1280 fallback, or 80 with --smoke.")
parser.add_argument("--orientation_goal", action="store_true", help="Orientation goal sequence preset; replaces keypoint shaping, uses qualified +250 goal impulses.")
parser.add_argument("--orientation_kernel", choices=("inverse", "exponential"), default="inverse", help="Dense angle kernel; exponential uses 1/(exp(4*theta)+0.1) per policy step.")
parser.add_argument("--position_adr", action="store_true", help="Enable nested per-environment object-position ADR (25-level scale, maximum level5).")
parser.add_argument("--sigma_mode", choices=("global", "conditional"), default="global", help="Shared scalar or contextual per-joint diagonal exploration scale.")
parser.add_argument(
    "--recovery_sigma_floor", type=float, default=None,
    help="Optional latent sigma floor only at TIP-silent outward target-limit states; global sigma only.",
)
parser.add_argument(
    "--adapt_recovery_exploration", action="store_true",
    help="Explicitly adapt the parameter-free recovery exploration rule when initializing a new method branch.",
)
parser.add_argument(
    "--arm",
    choices=("base", "residual", "direct", "direct_token"),
    default="residual",
    help="Actor arm; direct preserves the historical local skip, while direct_token reads only contextual tokens.",
)
parser.add_argument(
    "--history_encoder",
    choices=("tcn", "raw_stack"),
    default="tcn",
    help="Per-joint History30 route; raw_stack feeds all 150 lagged scalars to the local FiLM MLP.",
)
parser.add_argument("--seed", type=int, default=42, help="Formal protocol uses 42, then 43/44 after seed42 passes.")
parser.add_argument(
    "--max_updates", type=int, default=None, help="Override PPO updates; default preserves 30M pulse budget."
)
parser.add_argument(
    "--rollout_steps", type=int, default=30,
    help="Consecutive policy steps per PPO rollout; default30=1.5s at20Hz. Non-default lengths require --max_updates.",
)
parser.add_argument("--gae_lambda", type=float, default=None, help="GAE trace parameter; None preserves the YAML value. Gamma and rollout length are independent.")
parser.add_argument(
    "--reward_release_start_turns",
    type=float,
    default=1.0,
    help="Cell-median positive-turn EMA where stability/contact shaping begins; baseline is 1.0 turn.",
)
parser.add_argument(
    "--reward_release_end_turns",
    type=float,
    default=2.0,
    help="Cell-median positive-turn EMA where stability/contact shaping reaches full weight.",
)
parser.add_argument(
    "--minibatches", type=int, default=None, help="Formal default 16; every minibatch remains asset-balanced."
)
parser.add_argument(
    "--gradient_probe_frequency",
    type=int,
    default=None,
    help="Head-level per-asset Gram probe cadence; 0 disables it, and the formal default is 500.",
)
parser.add_argument(
    "--full_gradient_shadow_frequency",
    type=int,
    default=None,
    help="Full Actor global/per-asset replica-half gradient-shadow cadence; 0 disables it without changing the main optimizer.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=None,
    help="Activation slices per logical optimizer step; use 8 slices/2 accumulation to preserve 4 logical steps.",
)
parser.add_argument(
    "--advantage_normalization_scope",
    choices=("global", "per_asset_rollout"),
    default=None,
    help="Actor GAE normalization population; per_asset_rollout uses each asset's complete rollout moments.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Full actor/critic/optimizers/curriculum checkpoint.")
parser.add_argument("--init_optimizers", action="store_true", help="New research branch: inherit both Adam states and sampling RNG; requires --actor_init_checkpoint --init_critic.")
parser.add_argument("--phase_period_steps", type=int, default=None, help="Optional internal sin/cos clock period in policy steps; no object information.")
parser.add_argument("--adapt_phase_clock", action="store_true", help="Explicit old-policy initialization with zero phase adapters.")
parser.add_argument("--actor_init_optimizer_names", type=Path, default=None, help="SHA-bound source optimizer parameter-name ledger for architecture adaptation.")
parser.add_argument(
    "--actor_init_checkpoint",
    type=str,
    default=None,
    help="Initialize a new run from Actor weights only; reset Critic, optimizers, normalizer, and curricula.",
)
parser.add_argument(
    "--student_checkpoint",
    "--family_student_checkpoint",
    dest="student_checkpoint",
    type=str,
    default=None,
    help="Explicit offline family student policy.pt; loads actor_state_dict through load_family_student for RL.",
)
parser.add_argument(
    "--student_variant",
    choices=("n040", "no_z", "fk"),
    default=None,
    help="Expected offline student variant; required when --student_checkpoint is supplied.",
)
parser.add_argument(
    "--student_critic_seed",
    type=int,
    default=0,
    help="Dedicated Critic initialization seed; independent of Actor/FK construction RNG consumption.",
)
parser.add_argument(
    "--student_critic_warmup_updates",
    type=int,
    default=20,
    help="Complete Critic-only PPO rollout/update count before Actor optimizer updates are enabled (default 20).",
)
parser.add_argument(
    "--student_anchor_weight",
    type=float,
    default=0.1,
    help="Deferred offline-mean anchor loss weight; consumed only by the explicit student auxiliary hook.",
)
parser.add_argument(
    "--student_fk_weight",
    type=float,
    default=0.1,
    help="Deferred FK origin loss weight; target must come from raw q/parsed kinematics.",
)
parser.add_argument(
    "--student_anchor_dataset",
    action="append",
    default=None,
    help="Optional train-only HDF5 anchor source; repeat for multiple family collections.",
)
parser.add_argument(
    "--student_anchor_lock",
    type=Path,
    default=None,
    help="Explicit dataset-lock.json for train-only anchor sources; mutually exclusive with --student_anchor_dataset.",
)
parser.add_argument(
    "--student_anchor_microbatch_size",
    type=int,
    default=256,
    help="CPU/GPU microbatch used to realize each 2048-state offline anchor loss.",
)
parser.add_argument(
    "--actor_init_sigma",
    type=float,
    default=None,
    help="Set shared latent Gaussian standard deviation after loading Actor weights; None inherits and cannot be used for full resume.",
)
parser.add_argument("--experiment_name", type=str, default=None, help="Run name under logs/distill/rl_games.")
parser.add_argument("--sigma", type=float, default=None, help="Optional rl_games play-time sigma override.")
parser.add_argument(
    "--tf32", action="store_true", help="Explicit TF32 numeric-speed candidate; tensors and Adam remain FP32."
)
parser.add_argument(
    "--torch_compile",
    choices=("default", "reduce-overhead"),
    default=None,
    help="Compile the PPO model after checkpoint restore; default YAML remains eager.",
)
parser.add_argument(
    "--smoke", action="store_true", help="Use 80 env, horizon 4, one mini-epoch/update integration mode."
)
parser.add_argument("--rl_games_strict", action="store_true", help="Require pinned local rl_games v1.6.5 commit.")
parser.add_argument(
    "--actor_contact",
    choices=("all", "tip"),
    default="tip",
    help="Actor contact information; critic/reward keep full contact.",
)
parser.add_argument("--episode_seconds_min", type=float, default=20.0)
parser.add_argument("--episode_seconds_max", type=float, default=60.0)
parser.add_argument(
    "--reward_release_floor",
    type=float,
    default=0.0,
    help="Explicit minimum shaping gain; continuation controls may use 1.",
)
parser.add_argument("--reward_release_reference_seconds", type=float, default=120.0)
parser.add_argument(
    "--rotation_progress_clip_rad",
    type=float,
    default=0.025,
    help="Symmetric per-policy-step progress clip: 0.025/0.04 rad correspond to 0.5/0.8 rad/s at 20 Hz.",
)
parser.add_argument(
    "--rotation_progress_reward_weight",
    type=float,
    default=5.0,
    help="Signed progress coefficient in reward/rad; per-step reward uses the clipped angle increment and does not change action authority.",
)
parser.add_argument(
    "--pose_keypoint_reward_weight",
    type=float,
    default=1.0,
    help="Object position/full-pose kernel coefficient in reward/s; 0 disables shaping selected by pose_keypoint_mode.",
)
parser.add_argument(
    "--pose_keypoint_mode",
    choices=("full_pose", "position_only"),
    default="full_pose",
    help="Position-only mode reduces the six reward points to the object center; strict goals and physical termination keep their original definitions.",
)
parser.add_argument(
    "--learning_rate", type=float, default=None, help="Override actor base LR and scale other groups by the same ratio."
)
parser.add_argument(
    "--gamma", type=float, default=None, help="Dimensionless discount per policy step in (0,1); None keeps the YAML value."
)
parser.add_argument(
    "--value_normalization",
    choices=("rms", "popart"),
    default="rms",
    help="Value coordinate strategy: existing RMS or returns-only output-preserving global PopArt.",
)
parser.add_argument("--gradient_aggregation", choices=("mean", "cagrad"), default="mean")
parser.add_argument(
    "--rejected_action_weight", type=float, default=0.0,
    help="Actor-only cost for TIP-silent mean-action components rejected by target limits; 0 disables.",
)
parser.add_argument("--cagrad_c", type=float, default=0.4)
parser.add_argument("--cagrad_task_chunk", type=int, default=128)
parser.add_argument("--optimization_audit_frequency", type=int, default=512)
parser.add_argument("--console_metrics_frequency", type=int, default=0, help="Print research metrics every N updates; 0 disables.")
parser.add_argument("--console_family_split", type=int, default=0, help="Explicit LEAP/Allegro boundary on the frozen asset axis; 0 prints global only.")
parser.add_argument("--evaluation_frequency", type=int, default=None, help="Save a completed-update checkpoint every N updates.")
parser.add_argument("--release_rollout_batch", action="store_true", help="Release stale flattened rollout after stratified dataset materialization.")
parser.add_argument("--env_major_rollout_storage", action="store_true", help="Store rollout env-major beneath time-major views to avoid flatten copies.")
parser.add_argument("--gpu_driver_free_gib", type=float, default=None, help="Explicit CUDA driver headroom at completed-update boundaries, in GiB.")
parser.add_argument(
    "--strict_goal_reward_weight",
    type=float,
    default=10.0,
    help="Strict pose-and-position goal pulse weight; frontier reward remains zero.",
)
parser.add_argument(
    "--joint_pose_anchor_weight",
    type=float,
    default=-0.5,
    help="Soft joint displacement penalty relative to pregrasp; 0 disables only this reward term.",
)
parser.add_argument(
    "--init_critic",
    action="store_true",
    help="Also initialize compatible critic/value RMS from actor_init_checkpoint, not optimizer state.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, launcher_unknown_args = parser.parse_known_args()
if args_cli.paper_cohort is not None:
    if args_cli.cohort_lock is not None or args_cli.support_rows is not None:
        raise ValueError("--paper-cohort is mutually exclusive with --cohort_lock and --support_rows")
    from anymani.publication.paper_data import PaperAssetBundle

    paper_data_dir = args_cli.data_dir.expanduser().resolve(strict=True)
    paper_bundle = PaperAssetBundle(paper_data_dir / "assets")
    paper_cohort = paper_bundle.cohort(args_cli.paper_cohort)
    args_cli.cohort_lock = paper_bundle.lock_path(args_cli.paper_cohort)
    args_cli.data_dir = paper_data_dir
    os.environ["ANYMANI_DATA_DIR"] = str(paper_data_dir)
    os.environ["ANYMANI_HETERO_PAPER_COHORT"] = args_cli.paper_cohort
    os.environ["ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT"] = str(
        paper_bundle.catalog_root(args_cli.paper_cohort)
    )
    if int(paper_cohort["ready_count"]) != int(paper_cohort["nominal_count"]):
        raise ValueError("formal teacher training requires every nominal cohort member to be ready")
if args_cli.checkpoint is not None and args_cli.actor_init_checkpoint is not None:
    raise ValueError("--checkpoint and --actor_init_checkpoint are mutually exclusive")
if args_cli.student_checkpoint is not None:
    if args_cli.actor_init_checkpoint is not None:
        raise ValueError("--student_checkpoint cannot mix with old --actor_init_checkpoint")
    if args_cli.student_variant is None:
        raise ValueError("--student_variant is required with --student_checkpoint for fresh or full student resume")
    if Path(args_cli.student_checkpoint).suffix.lower() in {".ts", ".torchscript"}:
        raise ValueError("--student_checkpoint must be family student policy.pt, not FrozenFamilyStudent TorchScript")
    if args_cli.history_encoder != "tcn":
        raise ValueError("family student Actor ABI fixes history_encoder=tcn")
    if args_cli.sigma_mode != "global":
        raise ValueError("family student Actor ABI fixes sigma_mode=global")
    if args_cli.phase_period_steps is not None:
        raise ValueError("family student Actor ABI fixes phase_clock_enabled=False")
    if args_cli.recovery_sigma_floor is not None or args_cli.adapt_recovery_exploration:
        raise ValueError("family student Actor ABI does not expose recovery sigma-floor adaptation")
    if args_cli.actor_contact != "tip":
        raise ValueError("family student Actor ABI fixes actor_contact=tip")
    if args_cli.gamma is not None or args_cli.gae_lambda is not None:
        raise ValueError("family student PPO fixes gamma=.995 and GAE lambda=.95")
    if args_cli.rollout_steps != 30 or args_cli.minibatches is not None or args_cli.gradient_accumulation_steps is not None:
        raise ValueError("family student PPO candidate fixes H30, M8 and accumulation=2")
    if args_cli.student_anchor_dataset and args_cli.student_anchor_lock is not None:
        raise ValueError("--student_anchor_lock and --student_anchor_dataset are mutually exclusive")
else:
    if args_cli.student_variant is not None:
        raise ValueError("--student_variant requires --student_checkpoint")
    if args_cli.student_critic_seed != 0:
        raise ValueError("--student_critic_seed requires --student_checkpoint")
    if args_cli.student_critic_warmup_updates != 20:
        raise ValueError("--student_critic_warmup_updates requires --student_checkpoint")
    if args_cli.student_anchor_weight != 0.1 or args_cli.student_fk_weight != 0.1:
        raise ValueError("student auxiliary weights require --student_checkpoint")
    if args_cli.student_anchor_dataset or args_cli.student_anchor_lock is not None:
        raise ValueError("student anchor source options require --student_checkpoint")
if args_cli.student_critic_warmup_updates < 1:
    raise ValueError("--student_critic_warmup_updates must be positive")
if args_cli.student_critic_seed < 0:
    raise ValueError("--student_critic_seed must be non-negative")
if not math.isfinite(args_cli.student_anchor_weight) or args_cli.student_anchor_weight < 0.0:
    raise ValueError("--student_anchor_weight must be finite and non-negative")
if not math.isfinite(args_cli.student_fk_weight) or args_cli.student_fk_weight < 0.0:
    raise ValueError("--student_fk_weight must be finite and non-negative")
if args_cli.student_anchor_microbatch_size < 1:
    raise ValueError("--student_anchor_microbatch_size must be positive")
if args_cli.student_checkpoint is not None and args_cli.learning_rate is not None:
    raise ValueError("student RL fixes actor LR=1e-5 and critic LR=3e-4; do not use --learning_rate scaling")
if args_cli.student_checkpoint is not None and args_cli.gradient_aggregation != "mean":
    raise ValueError("student RL currently exposes mean PPO only; CAGrad hook requires student kinematics support")
if args_cli.student_checkpoint is not None and any(
    value is not None and value > 0
    for value in (args_cli.gradient_probe_frequency, args_cli.full_gradient_shadow_frequency)
):
    raise ValueError("student RL disables legacy gradient probes until the student kinematics hook is integrated")
if args_cli.actor_init_sigma is not None:
    if args_cli.actor_init_checkpoint is None:
        raise ValueError("--actor_init_sigma requires a fresh --actor_init_checkpoint")
    if not math.isfinite(args_cli.actor_init_sigma) or args_cli.actor_init_sigma <= 0.0:
        raise ValueError("--actor_init_sigma must be finite and positive")
if args_cli.init_critic and args_cli.actor_init_checkpoint is None:
    raise ValueError("--init_critic requires --actor_init_checkpoint")
if args_cli.init_optimizers and (args_cli.actor_init_checkpoint is None or not args_cli.init_critic):
    raise ValueError("--init_optimizers requires Actor and Critic initialization from the same checkpoint")
if args_cli.phase_period_steps is not None and (
    args_cli.phase_period_steps < 2 or args_cli.arm not in {"direct", "direct_token"}
):
    raise ValueError("--phase_period_steps requires a direct actor and an integer period >=2")
if args_cli.adapt_phase_clock and args_cli.actor_init_checkpoint is None:
    raise ValueError("--adapt_phase_clock requires a fresh --actor_init_checkpoint branch")
if args_cli.actor_init_optimizer_names is not None and not args_cli.init_optimizers:
    raise ValueError("--actor_init_optimizer_names requires --init_optimizers")
if args_cli.recovery_sigma_floor is not None and (
    args_cli.sigma_mode != "global"
    or not math.isfinite(args_cli.recovery_sigma_floor)
    or args_cli.recovery_sigma_floor <= 0.0
):
    raise ValueError("--recovery_sigma_floor requires global sigma and a finite positive value")
if args_cli.adapt_recovery_exploration and args_cli.actor_init_checkpoint is None:
    raise ValueError("--adapt_recovery_exploration requires a fresh --actor_init_checkpoint branch")
if args_cli.gae_lambda is not None and (not math.isfinite(args_cli.gae_lambda) or not 0 <= args_cli.gae_lambda <= 1):
    raise ValueError("--gae_lambda must lie in [0,1]")
if not 0 < args_cli.episode_seconds_min <= args_cli.episode_seconds_max or not math.isfinite(
    args_cli.episode_seconds_max
):
    raise ValueError("episode duration bounds must satisfy 0 < minimum <= maximum")
if not 0 <= args_cli.reward_release_floor <= 1 or not 0 < args_cli.reward_release_reference_seconds < math.inf:
    raise ValueError("release floor/reference duration are invalid")
if args_cli.learning_rate is not None and not 0 < args_cli.learning_rate < math.inf:
    raise ValueError("learning rate must be finite and positive")
if args_cli.gamma is not None and (not math.isfinite(args_cli.gamma) or not 0 < args_cli.gamma < 1):
    raise ValueError("--gamma must be finite and lie in (0,1)")
if not 0 <= args_cli.cagrad_c < 1 or args_cli.cagrad_task_chunk < 1:
    raise ValueError("CAGrad requires 0<=c<1 and positive task chunk size")
if args_cli.optimization_audit_frequency < 0:
    raise ValueError("optimization audit frequency must be nonnegative")
if not math.isfinite(args_cli.rejected_action_weight) or args_cli.rejected_action_weight < 0.0:
    raise ValueError("--rejected_action_weight must be finite and nonnegative")
if args_cli.gradient_aggregation == "cagrad" and args_cli.arm not in {"direct", "direct_token"}:
    raise ValueError("functional CAGrad requires direct or direct_token actor")
if not 0 < args_cli.rotation_progress_clip_rad < math.inf:
    raise ValueError("rotation progress clip must be finite and positive")
if not 0 <= args_cli.rotation_progress_reward_weight < math.inf:
    raise ValueError("rotation progress reward weight must be finite and non-negative")
if not 0 <= args_cli.pose_keypoint_reward_weight < math.inf:
    raise ValueError("pose keypoint reward weight must be finite and non-negative")
if not 0 <= args_cli.strict_goal_reward_weight < math.inf:
    raise ValueError("strict goal reward weight must be finite and non-negative")
if not math.isfinite(args_cli.joint_pose_anchor_weight) or args_cli.joint_pose_anchor_weight > 0.0:
    raise ValueError("joint pose anchor weight must be finite and non-positive")
if args_cli.reward_release_start_turns < 0.0:
    raise ValueError("--reward_release_start_turns must be non-negative")
if args_cli.reward_release_end_turns <= args_cli.reward_release_start_turns:
    raise ValueError("--reward_release_end_turns must exceed --reward_release_start_turns")


if args_cli.cohort_lock is not None:
    if args_cli.support_rows is not None:
        raise ValueError("--cohort_lock and --support_rows are mutually exclusive")
    cohort_lock_path = (
        args_cli.cohort_lock
        if args_cli.cohort_lock.is_absolute()
        else (ANYMANI_ROOT / args_cli.cohort_lock).resolve(strict=True)
    )
    cohort_document = parse_hand_asset_cohort_document(cohort_lock_path.read_bytes())
    if not isinstance(cohort_document, dict) or cohort_document.get("schema_version") not in {
        "1.0.0",
        "1.1.0",
        "1.2.0",
    }:
        raise ValueError("--cohort_lock must contain a supported schema-1.x member-level cohort")
    cohort_members = cohort_document.get("members")
    if not isinstance(cohort_members, list) or not cohort_members:
        raise ValueError("--cohort_lock must contain a non-empty members list")
    if not args_cli.smoke and cohort_document.get("schema_version") != "1.2.0":
        raise ValueError("formal cohort PPO requires a schema-1.2 canonical-final lock")
    selected_rows = tuple(range(len(cohort_members)))  # selection-local diagnostic/prototype axis
    asset_count = len(selected_rows)
    support_manifest_path = cohort_lock_path  # exact membership/order bytes enter method identity
    os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(cohort_lock_path)
    os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
else:
    mvp80_rows = _load_mvp80_rows(args_cli.asset_manifest)
    selected_rows = _select_support_rows(mvp80_rows, args_cli.support_rows)
    asset_count = len(selected_rows)
    support_manifest_path = args_cli.asset_manifest
    os.environ.pop("ANYMANI_HETERO_COHORT_LOCK", None)
    os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in selected_rows)
num_envs = int(args_cli.num_envs if args_cli.num_envs is not None else (80 if args_cli.smoke else 2560))
if args_cli.smoke:
    if num_envs < asset_count or num_envs % asset_count != 0:
        raise ValueError("smoke num_envs must be a positive multiple of the selected support assets")
elif args_cli.cohort_lock is not None and (args_cli.num_envs is None or num_envs % asset_count != 0):
    raise ValueError("cohort training requires explicit --num_envs divisible by cohort asset count")
elif asset_count == 80 and num_envs not in (1280, 2560):
    raise ValueError("full MVP80 permits only 2560 envs or the 1280-env memory fallback")
elif asset_count < 80 and (args_cli.num_envs is None or num_envs % asset_count != 0):
    raise ValueError("subset closure requires explicit --num_envs divisible by its selected asset count")
os.environ["ANYMANI_HETERO_NUM_ENVS"] = str(num_envs)  # static scene/mask/reset/command routing axis


sys.argv = [sys.argv[0], *launcher_unknown_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

import anymani.distill.rl  # noqa: E402
import anymani.tasks.hetero  # noqa: E402,F401 - Register the task after AppLauncher.
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games  # noqa: E402


backend_info = prefer_local_rl_games(strict=bool(args_cli.rl_games_strict))
torch.backends.cuda.matmul.allow_tf32 = bool(args_cli.tf32)
torch.backends.cudnn.allow_tf32 = bool(args_cli.tf32)

from isaaclab.managers import EventTermCfg, TerminationTermCfg  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402

from anymani.distill.rl.observers import OneShotIsaacAlgoObserver  # noqa: E402
from anymani.distill.rl.palm_rotation_ppo import (  # noqa: E402
    PalmRotationPpoRunner,
    register_palm_rotation_ppo,
    validate_gradient_probe_compile_compatibility,
)
from anymani.distill.rl.runtime.palm_rotation_geometry import (  # noqa: E402
    build_palm_rotation_bf16_geometry_provider,
)
from anymani.distill.rl.runtime.palm_rotation_identity import (  # noqa: E402
    build_palm_rotation_method_identity,
    palm_rotation_code_provenance,
)
from anymani.distill.rl.runtime.palm_rotation_optimizer_init import load_optimizer_parameter_names  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_precision import enforce_palm_rotation_precision  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_student import (  # noqa: E402
    JointOriginTargetProvider,
    binding_joint_kinematics_bank,
    build_joint_origin_target,
    resolve_family_student_anchor_paths,
    resolve_family_student_anchor_source_hashes,
    student_candidate_config,
)
from anymani.distill.rl.runtime.palm_rotation_vecenv import (  # noqa: E402
    PalmRotationRlGamesGpuEnv,
    PalmRotationRlGamesVecEnv,
)
from anymani.distill.rl.runtime.palm_rotation_warm_start import (  # noqa: E402
    inspect_actor_init_checkpoint,
    inspect_resumed_actor_warm_start,
)
from anymani.tasks.hetero.config.generated.palm_rotation_mvp_env_cfg import (  # noqa: E402
    GOOD_PREGRASP_RESET_CFG,
    GeneratedPalmRotationMvpEnvCfg,
)
from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING  # noqa: E402
from anymani.tasks.hetero.mdp.adr import HeterogeneousAdrCfg, ObjectPositionAdrCfg  # noqa: E402
from anymani.tasks.hetero.mdp.episode_horizon import (  # noqa: E402
    EPISODE_HORIZON_STEPS_ATTR,
    planned_time_out,
    reset_episode_horizon,
)
from anymani.tasks.hetero.mdp.orientation_goal import OrientationGoalCfg, configure_orientation_goal  # noqa: E402


def _resolve_seed(agent_cfg: dict[str, Any]) -> int:
    'Resolve seed.'

    seed = random.randint(0, 10000) if int(args_cli.seed) == -1 else int(args_cli.seed)
    agent_cfg["params"]["seed"] = seed  # rl_games model/rollout RNG
    return seed


def _configure_budget(agent_cfg: dict[str, Any]) -> tuple[int, int, int]:
    'Handle configure budget; shapes [int,int,int]; units M.'

    config = agent_cfg["params"]["config"]  # rl_games PPO config
    if getattr(args_cli, "gae_lambda", None) is not None:
        config["tau"] = float(args_cli.gae_lambda)
    rollout_steps = int(args_cli.rollout_steps)
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if args_cli.smoke and rollout_steps != 30:
        raise ValueError("smoke uses its fixed H4; custom rollout_steps requires a non-smoke run")
    if not args_cli.smoke and rollout_steps != 30 and args_cli.max_updates is None:
        raise ValueError("non-default rollout_steps requires an explicit --max_updates budget")
    horizon = 4 if args_cli.smoke else rollout_steps
    batch_size = num_envs * horizon
    student_mode = getattr(args_cli, "student_checkpoint", None) is not None
    default_minibatches = 4 if args_cli.smoke else (8 if student_mode else 16)
    minibatch_count = int(args_cli.minibatches if args_cli.minibatches is not None else default_minibatches)
    if minibatch_count < 1 or (batch_size // asset_count) % minibatch_count != 0:
        raise ValueError("per-asset rollout samples must be divisible into all stratified minibatches")
    minibatch_size = batch_size // minibatch_count
    default_accumulation_steps = 2 if student_mode else agent_cfg["params"]["config"]["gradient_accumulation_steps"]
    accumulation_steps = int(
        args_cli.gradient_accumulation_steps
        if args_cli.gradient_accumulation_steps is not None
        else default_accumulation_steps
    )
    if accumulation_steps < 1 or minibatch_count % accumulation_steps != 0:
        raise ValueError("gradient accumulation steps must be positive and divide activation minibatches")
    if not args_cli.smoke and asset_count < 80 and args_cli.max_updates is None:
        raise ValueError("subset closure requires an explicit --max_updates scientific budget")
    if args_cli.smoke:
        default_updates = 1
    elif student_mode:

        default_updates = int(agent_cfg["params"]["config"].get("student_candidate", {}).get("max_updates", 180))
    else:
        default_updates = 391 * (2560 // num_envs)  # units M
    max_updates = int(args_cli.max_updates if args_cli.max_updates is not None else default_updates)
    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    if student_mode and max_updates < 20:
        raise ValueError("student PPO max_updates must include all 20 Critic-only warmup rollouts")
    config["horizon_length"] = horizon
    config["minibatch_size"] = minibatch_size
    config["gradient_accumulation_steps"] = accumulation_steps
    config["mini_epochs"] = 1 if args_cli.smoke else 5
    config["max_epochs"] = max_updates
    config["num_actors"] = num_envs
    config["asset_count"] = asset_count
    if args_cli.gradient_probe_frequency is not None:
        if args_cli.gradient_probe_frequency < 0:
            raise ValueError("--gradient_probe_frequency must be non-negative")
        config["gradient_probe_frequency"] = int(args_cli.gradient_probe_frequency)
    if args_cli.full_gradient_shadow_frequency is not None:
        if args_cli.full_gradient_shadow_frequency < 0:
            raise ValueError("--full_gradient_shadow_frequency must be non-negative")
        config["full_gradient_shadow_frequency"] = int(args_cli.full_gradient_shadow_frequency)
    if args_cli.advantage_normalization_scope is not None:
        config["advantage_normalization_scope"] = str(args_cli.advantage_normalization_scope)
    effective_arm = "direct_token" if student_mode else args_cli.arm
    config["name"] = f"heterogeneous_palm_rotation_mvp_{effective_arm}"
    config["torch_compile"] = False
    agent_cfg["params"]["network"]["palm_rotation"]["arm"] = effective_arm
    agent_cfg["params"]["network"]["palm_rotation"]["history_encoder"] = args_cli.history_encoder
    agent_cfg["params"]["network"]["palm_rotation"]["sigma_mode"] = getattr(args_cli, "sigma_mode", "global")
    phase_period = getattr(args_cli, "phase_period_steps", None)
    if phase_period is not None:
        phase_period = int(phase_period)
    if phase_period is not None:
        agent_cfg["params"]["network"]["palm_rotation"]["phase_period_steps"] = phase_period
    recovery_sigma_floor = getattr(args_cli, "recovery_sigma_floor", None)
    if recovery_sigma_floor is not None:
        max_log_std = float(agent_cfg["params"]["network"]["palm_rotation"]["max_log_std"])
        if math.log(recovery_sigma_floor) > max_log_std:
            raise ValueError("recovery sigma floor exceeds the existing exploration ceiling")
        agent_cfg["params"]["network"]["palm_rotation"]["recovery_sigma_floor"] = float(recovery_sigma_floor)
    agent_cfg["params"]["network"]["palm_rotation"]["compile_mode"] = args_cli.torch_compile
    return horizon, minibatch_size, max_updates


def _configure_student_candidate(agent_cfg: dict[str, Any]) -> dict[str, object] | None:
    'Handle configure student candidate.'

    if args_cli.student_checkpoint is None:
        return None
    if args_cli.student_variant is None:
        raise ValueError("student candidate requires an explicit student_variant")
    student_path = Path(args_cli.student_checkpoint).expanduser().resolve()
    if not student_path.is_file():
        raise FileNotFoundError(f"student checkpoint does not exist: {student_path}")
    checkpoint_digest = hashlib.sha256()
    with student_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checkpoint_digest.update(block)

    candidate = student_candidate_config(
        num_envs=1024,
        horizon_length=30,
        minibatches=8,
        gradient_accumulation_steps=2,
        mini_epochs=5,
        critic_warmup_updates=int(args_cli.student_critic_warmup_updates),
        anchor_weight=float(args_cli.student_anchor_weight),
        fk_weight=float(args_cli.student_fk_weight),
    )
    params = agent_cfg["params"]
    config = params["config"]
    network_cfg = params["network"]["palm_rotation"]
    if not isinstance(config, dict) or not isinstance(network_cfg, dict):
        raise TypeError("student candidate requires mutable agent config/network mappings")
    anchor_dataset_paths_metadata: list[str] | None = None

    config.update(
        {
            "learning_rate": candidate.actor_learning_rate,
            "adaptive_lr_max": candidate.actor_learning_rate,
            "residual_learning_rate": candidate.actor_learning_rate,
            "contextual_learning_rate": candidate.actor_learning_rate,
            "critic_learning_rate": candidate.critic_learning_rate,
            "gamma": candidate.gamma,
            "tau": candidate.gae_lambda,
            "e_clip": candidate.clip_epsilon,
            "grad_norm": candidate.grad_norm,
            "entropy_coef": candidate.entropy_coef,
            "lr_schedule": "identity",
            "gradient_probe_frequency": 0,
            "full_gradient_shadow_frequency": 0,
            "student_mode": True,
            "student_candidate": candidate.as_dict(),
        }
    )
    network_cfg.update(
        {
            "student_enabled": True,
            "student_checkpoint": str(student_path),
            "student_variant": args_cli.student_variant,
            "student_rl": {
                **candidate.as_dict(),
                "critic_seed": int(args_cli.student_critic_seed),
            },
            "student_include_fk_target": args_cli.student_variant == "fk",
            "student_anchor_seed": int(params.get("seed", 42)),
            "student_anchor_batch_size": 2048,
            "student_anchor_microbatch_size": int(args_cli.student_anchor_microbatch_size),
        }
    )
    anchor_lock_metadata: str | None = None
    anchor_lock_hashes: dict[str, str] | None = None
    if args_cli.student_anchor_lock is not None:
        lock_path = args_cli.student_anchor_lock.expanduser().resolve()
        if not lock_path.is_file():
            raise FileNotFoundError(f"student anchor dataset lock does not exist: {lock_path}")
        locked_paths = resolve_family_student_anchor_paths(lock_path)
        anchor_lock_hashes = resolve_family_student_anchor_source_hashes(lock_path)
        network_cfg["student_anchor_dataset_paths"] = [str(path) for path in locked_paths]
        network_cfg["student_anchor_source_hashes"] = dict(anchor_lock_hashes)
        anchor_dataset_paths_metadata = [str(path) for path in locked_paths]
        anchor_lock_metadata = str(lock_path)
    elif args_cli.student_anchor_dataset:
        paths = tuple(Path(path).expanduser().resolve() for path in args_cli.student_anchor_dataset)
        if len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
            raise ValueError("--student_anchor_dataset must list unique existing HDF5 sources")
        network_cfg["student_anchor_dataset_paths"] = [str(path) for path in paths]
        # Custom anchor sources are part of the training identity; a resume that omits them must fail at the
        # semantic identity gate instead of silently falling back to the published four-source lock.
        anchor_dataset_paths_metadata = [str(path) for path in paths]
    return {
        "variant": args_cli.student_variant,
        "checkpoint_path": str(student_path),
        "checkpoint_sha256": checkpoint_digest.hexdigest(),
        "candidate": candidate.as_dict(),
        "critic_seed": int(args_cli.student_critic_seed),
        "anchor_microbatch_size": int(args_cli.student_anchor_microbatch_size),
        "warmup_hook": "family_student_critic_warmup_v2",
        "auxiliary_hook": {
            "anchor_weight": float(args_cli.student_anchor_weight),
            "fk_weight": float(args_cli.student_fk_weight),
            "fk_target_source": "raw_q_and_parsed_joint_kinematics",
        },
        **(
            {"anchor_dataset_paths": anchor_dataset_paths_metadata}
            if anchor_dataset_paths_metadata is not None
            else {}
        ),
        **({"anchor_dataset_lock": anchor_lock_metadata} if anchor_lock_metadata is not None else {}),
        **({"anchor_source_hashes": anchor_lock_hashes} if anchor_lock_hashes is not None else {}),
    }


def _configure_log_dir(agent_cfg: dict[str, Any], *, checkpoint: str | None) -> tuple[Path, Path]:
    'Handle configure log dir.'

    config = agent_cfg["params"]["config"]
    root = ANYMANI_ROOT / "logs" / "distill" / "rl_games" / str(config["name"])
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()  # exact checkpoint artifact
        run_dir = checkpoint_path.parent.parent  # `<run>/nn/file.pth` -> `<run>`
        if checkpoint_path.parent.name != "nn" or run_dir.parent.resolve() != root.resolve():
            raise ValueError(f"resume checkpoint must belong to the expected arm run root: {root}")
        if args_cli.experiment_name is not None and args_cli.experiment_name != run_dir.name:
            raise ValueError("--experiment_name must match the checkpoint-owned run directory on resume")
        run_name = run_dir.name
    else:
        run_name = args_cli.experiment_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = root / run_name
    config["train_dir"] = str(root)  # rl_games checkpoint/TensorBoard parent
    config["full_experiment_name"] = run_name
    config["rl_games_backend_file"] = str(backend_info.package_file)
    config["rl_games_backend_commit"] = backend_info.git_commit
    config["rl_games_backend_version"] = backend_info.package_version
    config["rl_games_backend_identity_source"] = backend_info.identity_source
    run_dir.joinpath("params").mkdir(parents=True, exist_ok=True)
    return root, run_dir


def main() -> None:
    'Handle main.'

    env_cfg: ManagerBasedRLEnvCfg = GeneratedPalmRotationMvpEnvCfg()
    for name in ("jnt_current", "jnt_history", "owner_contact"):
        getattr(env_cfg.observations.policy, name).params["tip_only"] = args_cli.actor_contact == "tip"
    env_cfg.episode_length_s = float(args_cli.episode_seconds_max)
    env_cfg.commands.goal_pose.horizon_s = float(args_cli.episode_seconds_max)
    env_cfg.events.episode_horizon = EventTermCfg(
        func=reset_episode_horizon,
        mode="reset",
        params={"minimum_seconds": args_cli.episode_seconds_min, "maximum_seconds": args_cli.episode_seconds_max},
    )
    env_cfg.terminations.time_out = TerminationTermCfg(func=planned_time_out, time_out=True)

    env_cfg.rewards.pose_keypoint.weight = float(args_cli.pose_keypoint_reward_weight)  # reward/s
    env_cfg.rewards.pose_keypoint.params["position_only"] = args_cli.pose_keypoint_mode == "position_only"


    env_cfg.rewards.rotation_progress.params["clip_rad_per_step"] = float(args_cli.rotation_progress_clip_rad)
    env_cfg.rewards.rotation_progress.weight = float(args_cli.rotation_progress_reward_weight)  # reward/rad
    env_cfg.rewards.goal_success.weight = float(args_cli.strict_goal_reward_weight)
    env_cfg.rewards.joint_pose_anchor.weight = float(
        args_cli.joint_pose_anchor_weight
    )
    reward_release_params = env_cfg.curriculum.reward_release.params  # shapes [union-attr]
    reward_release_params["release_start_turns"] = float(args_cli.reward_release_start_turns)
    reward_release_params["release_end_turns"] = float(args_cli.reward_release_end_turns)
    reward_release_params["release_floor"] = float(args_cli.reward_release_floor)
    reward_release_params["reference_seconds"] = float(args_cli.reward_release_reference_seconds)
    reward_release_ema_alpha = float(
        cast(Any, reward_release_params["ema_alpha"])
    )
    if args_cli.orientation_kernel != "inverse" and not args_cli.orientation_goal:
        raise ValueError("--orientation_kernel requires --orientation_goal")
    orientation_goal_cfg = OrientationGoalCfg(kernel=args_cli.orientation_kernel) if args_cli.orientation_goal else None
    adr_cfg = HeterogeneousAdrCfg(object_position=ObjectPositionAdrCfg(enabled=bool(args_cli.position_adr)))
    if args_cli.position_adr and not args_cli.orientation_goal:
        raise ValueError("position ADR is enabled through the explicit orientation-goal preset")
    if orientation_goal_cfg is not None:
        configure_orientation_goal(env_cfg, orientation_goal_cfg, training=True, adr=adr_cfg)
    agent_path = ANYMANI_ROOT / "source/anymani/anymani/distill/rl/agents/heterogeneous_palm_rotation_mvp_ppo.yaml"
    agent_cfg = yaml.safe_load(agent_path.read_text(encoding="utf-8"))  # versioned rl_games config
    if not isinstance(agent_cfg, dict):
        raise TypeError("palm-rotation rl_games YAML must contain a mapping")
    seed = _resolve_seed(agent_cfg)
    student_initialization = _configure_student_candidate(agent_cfg)
    runtime_arm = "direct_token" if student_initialization is not None else str(args_cli.arm)
    horizon, minibatch_size, max_updates = _configure_budget(agent_cfg)
    if args_cli.gamma is not None:
        agent_cfg["params"]["config"]["gamma"] = float(args_cli.gamma)
    agent_cfg["params"]["config"]["value_normalization"] = args_cli.value_normalization
    agent_cfg["params"]["config"]["gradient_aggregation"] = args_cli.gradient_aggregation
    agent_cfg["params"]["config"]["rejected_action_weight"] = float(args_cli.rejected_action_weight)
    agent_cfg["params"]["config"]["cagrad_c"] = args_cli.cagrad_c
    agent_cfg["params"]["config"]["cagrad_task_chunk"] = args_cli.cagrad_task_chunk
    agent_cfg["params"]["config"]["optimization_audit_frequency"] = args_cli.optimization_audit_frequency
    agent_cfg["params"]["config"]["console_metrics_frequency"] = args_cli.console_metrics_frequency
    agent_cfg["params"]["config"]["console_family_split"] = args_cli.console_family_split
    agent_cfg["params"]["config"]["release_rollout_batch"] = args_cli.release_rollout_batch
    agent_cfg["params"]["config"]["env_major_rollout_storage"] = args_cli.env_major_rollout_storage
    if args_cli.gpu_driver_free_gib is not None:
        if args_cli.gpu_driver_free_gib <= 0:
            raise ValueError("--gpu_driver_free_gib must be positive")
        agent_cfg["params"]["config"]["gpu_driver_free_memory_bytes_min"] = int(args_cli.gpu_driver_free_gib * 2**30)
    if args_cli.evaluation_frequency is not None:
        agent_cfg["params"]["config"]["evaluation_frequency"] = args_cli.evaluation_frequency
    if args_cli.learning_rate is not None:
        config = agent_cfg["params"]["config"]
        factor = float(args_cli.learning_rate) / float(config["learning_rate"])
        for name in (
            "learning_rate",
            "adaptive_lr_max",
            "residual_learning_rate",
            "contextual_learning_rate",
            "critic_learning_rate",
        ):
            config[name] = float(config[name]) * factor
    validate_gradient_probe_compile_compatibility(
        args_cli.torch_compile,
        int(agent_cfg["params"]["config"]["gradient_probe_frequency"]),
        int(agent_cfg["params"]["config"]["full_gradient_shadow_frequency"]),
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = seed
    rl_device = str(agent_cfg["params"]["config"].get("device", env_cfg.sim.device))
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    checkpoint = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint is not None else None
    actor_init_checkpoint = (
        retrieve_file_path(args_cli.actor_init_checkpoint) if args_cli.actor_init_checkpoint is not None else None
    )
    _, run_dir = _configure_log_dir(agent_cfg, checkpoint=checkpoint)
    if checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["full_checkpoint_resume"] = checkpoint is not None
    env_cfg.log_dir = str(run_dir)


    env = gym.make(TASK_ID, cfg=env_cfg)
    provider = build_palm_rotation_bf16_geometry_provider(ASSET_BINDING, device=rl_device)
    student_joint_kinematics = None
    student_fk_target_provider: JointOriginTargetProvider | None = None
    if student_initialization is not None:
        retained_sha = provider.identity.get("retained_artifact", {}).get("sha256")
        if not isinstance(retained_sha, str):
            raise RuntimeError("student RL requires a concrete N040 retained artifact SHA")
        agent_cfg["params"]["network"]["palm_rotation"]["expected_n040_sha256"] = retained_sha

        parsed_joint_bank = binding_joint_kinematics_bank(ASSET_BINDING, device=rl_device)
        student_joint_kinematics = parsed_joint_bank.features.float()
        student_initialization["joint_kinematics_shape"] = list(student_joint_kinematics.shape)
        student_initialization["joint_kinematics_dtype"] = str(student_joint_kinematics.dtype).replace("torch.", "")
        if args_cli.student_variant == "fk":

            def _student_fk_target_provider_impl(
                q_rad: torch.Tensor,
                asset_index: torch.Tensor,
                joint_kinematics: torch.Tensor,
            ) -> torch.Tensor:
                _ = joint_kinematics
                return build_joint_origin_target(parsed_joint_bank, q_rad, asset_index)

            student_fk_target_provider = _student_fk_target_provider_impl

    actor_warm_start = (
        inspect_actor_init_checkpoint(
            actor_init_checkpoint,
            target_arm=str(args_cli.arm),
            target_history_encoder=str(args_cli.history_encoder),
            target_provider_identity=provider.identity,
            initialize_critic=bool(args_cli.init_critic),
            actor_init_sigma=args_cli.actor_init_sigma,
            target_sigma_mode=args_cli.sigma_mode,
            target_recovery_sigma_floor=args_cli.recovery_sigma_floor,
            allow_recovery_exploration_adaptation=bool(args_cli.adapt_recovery_exploration),
            target_phase_period_steps=args_cli.phase_period_steps,
            allow_phase_clock_adaptation=bool(args_cli.adapt_phase_clock),
        )
        if actor_init_checkpoint is not None
        else (inspect_resumed_actor_warm_start(checkpoint) if checkpoint is not None else None)
    )
    if actor_init_checkpoint is not None:
        if args_cli.init_optimizers:
            assert actor_warm_start is not None
            if actor_warm_start.get("phase_clock_adaptation") and args_cli.actor_init_optimizer_names is None:
                raise ValueError("phase adapter Adam inheritance requires --actor_init_optimizer_names")
            if args_cli.actor_init_optimizer_names is not None:
                names_path = args_cli.actor_init_optimizer_names.expanduser().resolve(strict=True)
                names_sha = hashlib.sha256(names_path.read_bytes()).hexdigest()
                load_optimizer_parameter_names(
                    names_path,
                    expected_checkpoint_sha256=str(actor_warm_start["checkpoint_sha256"]),
                    expected_ledger_sha256=names_sha,
                )
                actor_warm_start["optimizer_name_ledger_sha256"] = names_sha
                agent_cfg["params"]["config"]["actor_init_optimizer_names"] = str(names_path)
            actor_warm_start["initialize_optimizers"] = True
            actor_warm_start["reset_components"] = [
                name for name in actor_warm_start["reset_components"]
                if name not in {"actor_optimizer", "critic_optimizer", "random_states"}
            ]
        agent_cfg["params"]["config"]["actor_init_checkpoint"] = actor_init_checkpoint
    prototype_index = torch.tensor(
        ASSET_BINDING.asset_index_by_env(num_envs),
        dtype=torch.long,
        device=rl_device,
    )
    transport = PalmRotationRlGamesVecEnv(
        env,
        geometry_provider=provider,
        prototype_index=prototype_index,
        rl_device=rl_device,
        clip_observations=float(agent_cfg["params"]["env"]["clip_observations"]),
        clip_actions=float(agent_cfg["params"]["env"]["clip_actions"]),
        phase_period_steps=args_cli.phase_period_steps,
        student=student_initialization is not None,
        student_variant=args_cli.student_variant,
        joint_kinematics=student_joint_kinematics,
        joint_origin_target_provider=student_fk_target_provider,
    )


    ppo_cfg = agent_cfg["params"]["config"]  # resolved optimizer/sampling contract
    identity = build_palm_rotation_method_identity(
        provider_identity=provider.identity,
        manifest_path=support_manifest_path,
        selected_rows=selected_rows,
        pregrasp=GOOD_PREGRASP_RESET_CFG,
        arm=runtime_arm,
        run_contract={
            "seed": seed,
            "num_envs": num_envs,
            "asset_count": asset_count,
            "cohort_id": ASSET_BINDING.cohort_id,
            "cohort_lock_sha256": ASSET_BINDING.cohort_lock_sha256,
            "source_member_keys": list(ASSET_BINDING.source_member_keys),
            "actor_warm_start": actor_warm_start,
            "actor_contact": str(args_cli.actor_contact),
            **({"student_initialization": student_initialization} if student_initialization is not None else {}),
            **({"phase_period_steps": args_cli.phase_period_steps} if args_cli.phase_period_steps is not None else {}),
            **({"sigma_mode": args_cli.sigma_mode} if args_cli.sigma_mode != "global" else {}),
            **({"recovery_sigma_floor": float(args_cli.recovery_sigma_floor)} if args_cli.recovery_sigma_floor is not None else {}),
            **({"orientation_goal": orientation_goal_cfg.to_dict(),
                "adr": {"object_position": vars(adr_cfg.object_position)}} if orientation_goal_cfg is not None else {}),
            "episode_seconds_min": float(args_cli.episode_seconds_min),
            "episode_seconds_max": float(args_cli.episode_seconds_max),
            "reward_release_floor": float(args_cli.reward_release_floor),
            "reward_release_reference_seconds": float(args_cli.reward_release_reference_seconds),
            "rotation_progress_clip_rad_per_step": float(args_cli.rotation_progress_clip_rad),
            **(
                {"pose_keypoint_reward_weight": float(args_cli.pose_keypoint_reward_weight)}
                if args_cli.pose_keypoint_reward_weight != 1.0
                else {}
            ),
            **(
                {"pose_keypoint_mode": str(args_cli.pose_keypoint_mode)}
                if args_cli.pose_keypoint_mode != "full_pose"
                else {}
            ),
            **(
                {"rotation_progress_reward_weight": float(args_cli.rotation_progress_reward_weight)}
                if args_cli.rotation_progress_reward_weight != 5.0
                else {}
            ),
            "strict_goal_reward_weight": float(env_cfg.rewards.goal_success.weight),
            **(
                {"joint_pose_anchor_weight": float(args_cli.joint_pose_anchor_weight)}
                if args_cli.joint_pose_anchor_weight != -0.5
                else {}
            ),
            "horizon_length": horizon,
            "minibatch_size": minibatch_size,
            "minibatch_count": (num_envs * horizon) // minibatch_size,
            "gradient_accumulation_steps": int(ppo_cfg["gradient_accumulation_steps"]),
            "mini_epochs": int(ppo_cfg["mini_epochs"]),
            "gradient_probe_frequency": int(ppo_cfg["gradient_probe_frequency"]),
            "full_gradient_shadow_frequency": int(ppo_cfg["full_gradient_shadow_frequency"]),
            "gamma": float(ppo_cfg["gamma"]),
            "gae_lambda": float(ppo_cfg["tau"]),
            "ppo_clip": float(ppo_cfg["e_clip"]),
            "entropy_coef": float(ppo_cfg["entropy_coef"]),
            "grad_norm": float(ppo_cfg["grad_norm"]),
            "actor_base_lr": float(ppo_cfg["learning_rate"]),
            "adaptive_lr_max": float(ppo_cfg["adaptive_lr_max"]),
            "actor_residual_lr": float(ppo_cfg["residual_learning_rate"]),
            "actor_contextual_lr": float(ppo_cfg["contextual_learning_rate"]),
            "critic_lr": float(ppo_cfg["critic_learning_rate"]),
            "lr_schedule": str(ppo_cfg["lr_schedule"]),
            "normalize_advantage": bool(ppo_cfg["normalize_advantage"]),
            "advantage_normalization_scope": str(ppo_cfg["advantage_normalization_scope"]),
            "reward_release_start_turns": float(args_cli.reward_release_start_turns),
            "reward_release_end_turns": float(args_cli.reward_release_end_turns),
            "reward_release_ema_alpha": reward_release_ema_alpha,
            "normalize_value": bool(ppo_cfg["normalize_value"]),
            "value_normalization": str(ppo_cfg["value_normalization"]),
            "gradient_aggregation": str(ppo_cfg["gradient_aggregation"]),
            **({"rejected_action_weight": float(args_cli.rejected_action_weight)} if args_cli.rejected_action_weight else {}),
            "cagrad_c": float(ppo_cfg["cagrad_c"]),
            "cagrad_task_chunk": int(ppo_cfg["cagrad_task_chunk"]),
            "optimization_audit_frequency": int(ppo_cfg["optimization_audit_frequency"]),
            "initial_log_std": float(agent_cfg["params"]["network"]["palm_rotation"]["initial_log_std"]),
            "max_log_std": float(agent_cfg["params"]["network"]["palm_rotation"]["max_log_std"]),
            "base_action_limit": float(agent_cfg["params"]["network"]["palm_rotation"]["base_action_limit"]),
            "history_encoder": str(agent_cfg["params"]["network"]["palm_rotation"]["history_encoder"]),
            "allow_tf32": bool(args_cli.tf32),
            "torch_compile": agent_cfg["params"]["network"]["palm_rotation"]["compile_mode"],
            "rl_games_backend_commit": backend_info.git_commit,
            "rl_games_backend_version": backend_info.package_version,
            "rl_games_backend_identity_source": backend_info.identity_source,
            "device": rl_device,
        },
    )
    agent_cfg["params"]["network"]["anymani_identity"] = identity
    transport.configure_training_evidence(run_dir, str(identity["identity_digest"]))
    if transport.training_evidence is not None:
        agent_cfg["params"]["config"]["evidence_segment_id"] = transport.training_evidence.segment_id
    agent_cfg["params"]["config"]["num_actors"] = transport.num_envs
    agent_cfg["params"]["config"]["code_provenance"] = palm_rotation_code_provenance()
    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)
    (run_dir / "params" / "runtime_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


    vecenv.register(
        "AnyManiPalmRotationWrapper",
        lambda config_name, num_actors, **kwargs: PalmRotationRlGamesGpuEnv(
            config_name,
            num_actors,
            env=transport,
        ),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "AnyManiPalmRotationWrapper", "env_creator": lambda **kwargs: transport},
    )
    register_palm_rotation_ppo()
    runner = PalmRotationPpoRunner(OneShotIsaacAlgoObserver())
    precision_flags = enforce_palm_rotation_precision(
        allow_tf32=bool(args_cli.tf32)
    )
    runner.load(agent_cfg)
    runner.reset()
    if enforce_palm_rotation_precision(allow_tf32=bool(args_cli.tf32)) != precision_flags:
        raise RuntimeError("palm-rotation precision flags changed during Runner model construction")


    runner_args: dict[str, Any] = {"train": True, "play": False, "sigma": args_cli.sigma}
    if checkpoint is not None:
        runner_args["checkpoint"] = checkpoint
    print(
        json.dumps(
            {
                "task": TASK_ID,
                "arm": runtime_arm,
                "history_encoder": args_cli.history_encoder,
                "tf32": bool(args_cli.tf32),
                "torch_compile": agent_cfg["params"]["network"]["palm_rotation"]["compile_mode"],
                "seed": seed,
                "num_envs": num_envs,
                "asset_count": asset_count,
                "cohort_id": ASSET_BINDING.cohort_id,
                "horizon": horizon,
                "batch_size": num_envs * horizon,
                "minibatch_size": minibatch_size,
                "gradient_accumulation_steps": agent_cfg["params"]["config"]["gradient_accumulation_steps"],
                "advantage_normalization_scope": agent_cfg["params"]["config"]["advantage_normalization_scope"],
                "reward_release_start_turns": float(args_cli.reward_release_start_turns),
                "reward_release_end_turns": float(args_cli.reward_release_end_turns),
                "reward_release_ema_alpha": reward_release_ema_alpha,
                "rotation_progress_reward_weight": float(args_cli.rotation_progress_reward_weight),
                "pose_keypoint_reward_weight": 0.0 if orientation_goal_cfg else float(args_cli.pose_keypoint_reward_weight),
                "orientation_goal": orientation_goal_cfg.to_dict() if orientation_goal_cfg else None,
                "sigma_mode": args_cli.sigma_mode,
                "recovery_sigma_floor": args_cli.recovery_sigma_floor,
                "rejected_action_weight": float(args_cli.rejected_action_weight),
                "position_adr": vars(adr_cfg.object_position),
                "full_gradient_shadow_frequency": agent_cfg["params"]["config"]["full_gradient_shadow_frequency"],
                "mini_epochs": agent_cfg["params"]["config"]["mini_epochs"],
                "max_updates": max_updates,
                "identity_digest": identity["identity_digest"],
                "run_dir": str(run_dir),
                "actor_init_checkpoint_sha256": (
                    actor_warm_start["checkpoint_sha256"] if actor_warm_start is not None else None
                ),
            },
            sort_keys=True,
        )
    )
    try:
        runner.run(runner_args)  # PPO rollout/update/checkpoint lifecycle
        metrics_path = run_dir / "metrics.parquet"
        if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
            raise RuntimeError("palm-rotation Runner returned without any finalized PPO update metrics")

        task_env = cast(Any, env.unwrapped)
        policy = task_env.obs_buf["policy"]
        own_max = max(
            float(policy["jnt_current"][..., 3].abs().max().item()),
            float(policy["jnt_history"][..., 3].abs().max().item()),
            float(policy["owner_contact"][:, :17].abs().max().item()),
        )
        if args_cli.actor_contact == "tip" and own_max != 0.0:
            raise RuntimeError("TIP-only actor observation leaks non-tip contact")
        plans = getattr(task_env, EPISODE_HORIZON_STEPS_ATTR)
        planned_seconds = plans * float(task_env.step_dt)
        if not bool(
            (
                (planned_seconds >= args_cli.episode_seconds_min - 1e-5)
                & (planned_seconds <= args_cli.episode_seconds_max + 1e-5)
            ).all()
        ):
            raise RuntimeError("runtime episode horizons lie outside the declared interval")
        if bool((task_env.episode_length_buf >= plans).any()):
            raise RuntimeError("an expired episode survived automatic reset")
        (run_dir / "params" / "protocol_runtime_check.json").write_text(
            json.dumps(
                {
                    "identity_digest": identity["identity_digest"],
                    "actor_contact": args_cli.actor_contact,
                    "actor_non_tip_input_max_abs": own_max,
                    "planned_duration_min_s": float(planned_seconds.min().item()),
                    "planned_duration_max_s": float(planned_seconds.max().item()),
                    "planned_duration_unique_count": int(plans.unique().numel()),
                    "expired_unreset_env_count": 0,
                    "scope": "final live observation and plans; episode timing is audited separately from recorded terminations",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        transport.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        exit_code = 1
        evidence_dir = os.environ.get("ANYMANI_RL_EVIDENCE_DIR")
        if evidence_dir:
            (Path(evidence_dir) / "python_failure.json").write_text(
                json.dumps({"exception_type": type(error).__name__, "message": str(error)}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    finally:
        try:
            simulation_app.close()
        except SystemExit as shutdown:
            if exit_code == 0:
                exit_code = int(shutdown.code or 0)
    raise SystemExit(exit_code)
