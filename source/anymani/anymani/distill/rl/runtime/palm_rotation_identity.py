"""Build method identities for palm-rotation checkpoints and evaluations.

The identity binds data, pregrasp catalogs, N040 precision, structured ABI, actor arm, task rewards, and PPO settings. Training resume requires the full run identity; fixed evaluation records a separate evaluation identity and may use a different replica count.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.distill.diagnostics.recording.rl.palm_rotation import PALM_ROTATION_METRICS_SCHEMA_VERSION
from anymani.distill.models.palm_rotation_policy import recovery_exploration_contract

from .palm_rotation_phase import normalize_phase_period_steps, phase_clock_contract
from .palm_rotation_vecenv import (
    PALM_ROTATION_BOOL_SHAPES,
    PALM_ROTATION_INT16_SHAPES,
    palm_rotation_float_shapes,
)

TASK_ID = "AnyMani-Hetero-Generated-PalmRotation-MVP-RLGames-v0"
PALM_ROTATION_IDENTITY_SCHEMA_VERSION = "4.0.0"
"""Source bytes define implementation identity; Git provenance is stored separately."""

_IMPLEMENTATION_PATHS = (
    "source/anymani/anymani/distill/rl/train_palm_rotation_mvp.py",
    "source/anymani/anymani/distill/rl/agents/heterogeneous_palm_rotation_mvp_ppo.yaml",
    "source/anymani/anymani/distill/rl/canonical_evidence.py",
    "source/anymani/anymani/distill/rl/runtime/source_config.py",
    "source/anymani/anymani/distill/representations/sources/joint_frames.py",
    "source/anymani/anymani/publication/paper_data.py",
    "source/anymani/anymani/publication/compatibility.py",
    "source/anymani/anymani/assets/bank/hand_container.py",
    "source/anymani/anymani/assets/canonical_runtime.py",
    "source/anymani/anymani/assets/exporter/urdf_writer.py",
    "source/anymani/anymani/distill/representations/sources/collision_geometry.py",
    "source/anymani/anymani/distill/rl/rl_games_backend.py",
    "source/anymani/anymani/distill/diagnostics/evaluation/rl/palm_rotation_transfer.py",
    "source/anymani/anymani/distill/models/palm_rotation_policy.py",
    "source/anymani/anymani/distill/rl/palm_rotation_ppo.py",
    "source/anymani/anymani/distill/rl/algorithms/ppo_batch.py",
    "source/anymani/anymani/distill/rl/algorithms/policy_statistics.py",
    "source/anymani/anymani/distill/rl/algorithms/action_regularization.py",
    "source/anymani/anymani/distill/rl/algorithms/popart.py",
    "source/anymani/anymani/distill/rl/algorithms/cagrad.py",
    "source/anymani/anymani/distill/rl/algorithms/task_gradients.py",
    "source/anymani/anymani/distill/diagnostics/recording/rl/training_evidence.py",
    "source/anymani/anymani/distill/diagnostics/recording/rl/first_window.py",
    "source/anymani/anymani/distill/diagnostics/recording/rl/episode_evidence.py",
    "source/anymani/anymani/distill/diagnostics/recording/rl/optimization_evidence.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_network.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_experience.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_diagnostics.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_probes.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_warm_start.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_optimizer_init.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_phase.py",
    "source/anymani/anymani/distill/models/temporal_encoder.py",
    "source/anymani/anymani/distill/models/backbones/geometry_transformer.py",
    "source/anymani/anymani/distill/models/input_adapters/encoder.py",
    "source/anymani/anymani/distill/models/input_adapters/se3_invariant_encoder.py",
    "source/anymani/anymani/distill/rl/algorithms/gradient_audit.py",
    "source/anymani/anymani/distill/rl/masked_ppo.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_geometry.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_identity.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_precision.py",
    "source/anymani/anymani/distill/rl/runtime/retained_geometry.py",
    "source/anymani/anymani/distill/rl/runtime/structured_geometry.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_vecenv.py",
    "source/anymani/anymani/distill/rl/runtime/palm_rotation_student.py",
    "source/anymani/anymani/distill/rl/runtime/student_anchor.py",
    "source/anymani/anymani/distill/models/family_rotation_student_actor_critic.py",
    "source/anymani/anymani/distill/models/family_rotation_policy.py",
    "source/anymani/anymani/distill/il/family_student.py",
    "source/anymani/anymani/tasks/hetero/config/generated/palm_rotation_mvp_env_cfg.py",
    "source/anymani/anymani/tasks/hetero/config/generated/asset_binding.py",
    "source/anymani/anymani/tasks/hetero/config/generated/scene.py",
    "source/anymani/anymani/pregrasp/good_catalog.py",
    "source/anymani/anymani/pregrasp/revalidation.py",
    "source/anymani/anymani/assets/bank/cohort.py",
    "source/anymani/anymani/tasks/hetero/mdp/actions.py",
    "source/anymani/anymani/tasks/hetero/mdp/adr.py",
    "source/anymani/anymani/tasks/hetero/mdp/orientation_goal.py",
    "source/anymani/anymani/tasks/hetero/mdp/commands.py",
    "source/anymani/anymani/tasks/hetero/mdp/contact_state.py",
    "source/anymani/anymani/tasks/hetero/mdp/curriculum_state.py",
    "source/anymani/anymani/tasks/hetero/mdp/events.py",
    "source/anymani/anymani/tasks/hetero/mdp/episode_horizon.py",
    "source/anymani/anymani/tasks/hetero/mdp/curriculums.py",
    "source/anymani/anymani/tasks/hetero/mdp/object_state.py",
    "source/anymani/anymani/tasks/hetero/mdp/observation_state.py",
    "source/anymani/anymani/tasks/hetero/mdp/observations.py",
    "source/anymani/anymani/tasks/hetero/mdp/rewards.py",
    "source/anymani/anymani/tasks/hetero/mdp/task_math.py",
    "source/anymani/anymani/tasks/hetero/contact_layout.py",
    "source/anymani/anymani/tasks/hetero/contact_sensors.py",
    "source/anymani/anymani/robots/hand_spawn.py",
)


class _Binding(Protocol):
    """Minimal schema-3 pregrasp binding used by identity construction."""

    @property
    def key_json(self) -> str: ...


class PalmRotationPregraspIdentityCfg(Protocol):
    """Structural pregrasp contract that avoids importing Isaac event classes."""

    @property
    def catalog_root(self) -> str: ...

    @property
    def bindings(self) -> tuple[_Binding, ...]: ...

    @property
    def rank(self) -> int: ...

    @property
    def require_strict(self) -> bool: ...


def _sha256(path: Path) -> str:
    """Stream a file into a SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(payload: dict[str, Any]) -> str:
    """Hash a JSON-safe identity using canonical JSON encoding."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> str:
    """Store repository paths relatively and external paths absolutely."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _git_head(root: Path) -> str:
    """Read the Git HEAD used to start a run."""

    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if len(revision) != 40:
        raise RuntimeError(f"unexpected AnyMani Git revision: {revision!r}")
    return revision


def palm_rotation_code_provenance() -> dict[str, str]:
    """Return code provenance separately from method semantics."""

    return {"git_head": _git_head(resolve_anymani_root())}


def palm_rotation_implementation_files() -> dict[str, str]:
    """Hash each source file that defines the executed method."""

    root = resolve_anymani_root()
    return {path: _sha256(root / path) for path in _IMPLEMENTATION_PATHS}


def validate_palm_rotation_evaluation_identity(
    *,
    runtime_identity: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    implementation_certificate: Mapping[str, Any] | None = None,
) -> None:
    """Validate read-only evaluation identity and any exact implementation mapping certificate.

    Full resume uses the stricter masked-PPO gate. This path requires assets, pregrasps, N040, actor, task, and training contracts to match. If implementation file maps differ, a certificate must bind both maps; a Boolean override cannot bypass the check.
    """

    for label, identity in (("runtime", runtime_identity), ("checkpoint", checkpoint_identity)):
        if identity.get("identity_schema_version") not in {"3.0.0", "4.0.0"}:
            raise RuntimeError(f"{label} evaluation identity has unsupported schema")
        payload = {key: value for key, value in identity.items() if key != "identity_digest"}
        if identity.get("identity_digest") != _stable_digest(payload):
            raise RuntimeError(f"{label} evaluation identity digest is inconsistent with its payload")
    fields = (
        "task_id",
        "task_contract",
        "policy",
        "manifest",
        "pregrasp",
        "geometry_provider",
        "transport_abi",
        "training",
    )
    from anymani.publication.compatibility import geometry_semantics, manifest_semantics, pregrasp_semantics

    def semantics(name: str, identity: Mapping[str, Any]) -> Any:
        value = identity.get(name)
        if name == "manifest" and isinstance(value, Mapping):
            return manifest_semantics(value)
        if name == "pregrasp" and isinstance(value, Mapping):
            return pregrasp_semantics(value)
        if name == "geometry_provider" and isinstance(value, Mapping):
            return geometry_semantics(value, keep_population=True)
        return value

    mismatched = [
        name
        for name in fields
        if name not in runtime_identity or semantics(name, runtime_identity) != semantics(name, checkpoint_identity)
    ]
    if mismatched:
        raise RuntimeError(f"evaluation semantic identity mismatch: {mismatched}")
    runtime_files = runtime_identity.get("implementation", {}).get("files")
    checkpoint_files = checkpoint_identity.get("implementation", {}).get("files")
    if (
        not isinstance(runtime_files, dict)
        or not runtime_files
        or not isinstance(checkpoint_files, dict)
        or not checkpoint_files
    ):
        raise RuntimeError("evaluation requires non-empty implementation file identities")
    if runtime_files == checkpoint_files:
        return
    certificate = implementation_certificate
    if (
        isinstance(certificate, Mapping)
        and certificate.get("artifact_type") == "anymani.publication.static_compatibility"
    ):
        from anymani.publication.compatibility import validate_source_compatibility

        validate_source_compatibility(certificate, checkpoint=checkpoint_identity, current_files=runtime_files)
        return
    if not isinstance(certificate, Mapping):
        raise RuntimeError("evaluation implementation changed; an exact refactor certificate is required")
    if (
        certificate.get("artifact_type") != "anymani.palm_rotation.refactor_equivalence"
        or certificate.get("schema_version") != "1.0.0"
        or certificate.get("passed") is not True
        or certificate.get("reference_implementation_files") != checkpoint_files
        or certificate.get("current_implementation_files") != runtime_files
    ):
        raise RuntimeError("evaluation refactor certificate does not cover these exact implementations")


def build_palm_rotation_method_identity(
    *,
    provider_identity: dict[str, Any],
    manifest_path: Path,
    selected_rows: tuple[int, ...],
    pregrasp: PalmRotationPregraspIdentityCfg,
    arm: str,
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the method identity shared by resume and fixed evaluation."""

    if arm not in {"base", "residual", "direct", "direct_token"}:
        raise ValueError("palm-rotation arm must be base, residual, direct or direct_token")
    if not selected_rows or len(set(selected_rows)) != len(selected_rows):
        raise ValueError("palm-rotation identity requires non-empty unique selected rows")
    if not run_contract:
        raise ValueError("palm-rotation identity requires a non-empty PPO run contract")
    progress_weight = float(
        run_contract.get("rotation_progress_reward_weight", 5.0)
    )  # Reward per radian; legacy default 5.
    if not math.isfinite(progress_weight) or progress_weight < 0.0:
        raise ValueError("rotation progress reward weight must be finite and non-negative")
    pose_weight = float(run_contract.get("pose_keypoint_reward_weight", 1.0))  # Selected pose-kernel rate, reward/s.
    if not math.isfinite(pose_weight) or pose_weight < 0.0:
        raise ValueError("pose keypoint reward weight must be finite and non-negative")
    pose_mode = run_contract.get("pose_keypoint_mode", "full_pose")  # Missing legacy fields mean full-pose mode.
    if pose_mode not in ("full_pose", "position_only"):
        raise ValueError("pose keypoint mode must be full_pose or position_only")
    canonical_run_contract = dict(run_contract)  # Do not mutate the caller's run record.
    phase_period_steps = normalize_phase_period_steps(run_contract.get("phase_period_steps"))
    phase_contract = phase_clock_contract(phase_period_steps)
    if phase_contract is not None:
        if arm not in {"direct", "direct_token"}:
            raise ValueError("phase clock requires a direct actor")
        canonical_run_contract["phase_period_steps"] = phase_period_steps
    else:
        canonical_run_contract.pop("phase_period_steps", None)
    rejected_action_weight = float(run_contract.get("rejected_action_weight", 0.0))
    if not math.isfinite(rejected_action_weight) or rejected_action_weight < 0.0:
        raise ValueError("rejected action weight must be finite and nonnegative")
    if rejected_action_weight > 0.0:
        canonical_run_contract["rejected_action_regularization"] = {
            "formula": "tip-silent-squared-rejected-target-action-v1",
            "weight": rejected_action_weight,
            "applied_to": "actor-objective-only-not-environment-reward",
            "mean_source": "current-ppo-forward-not-rollout-or-kl-reference",
            "target_and_limits_units": "rad-divided-by-pi",
            "action_authority_rad_per_policy_step": 1.0 / 24.0,
            "contact_scope": "all-valid-tips-zero-with-at-least-one-valid-tip",
            "reduction": "mean-active-joints-then-mean-samples",
            "diagnostic_fraction_threshold": 1.0e-6,
        }
    else:
        canonical_run_contract.pop("rejected_action_weight", None)
        canonical_run_contract.pop("rejected_action_regularization", None)
    if pose_mode == "full_pose":
        canonical_run_contract.pop("pose_keypoint_mode", None)  # Keep the legacy default field layout.
    joint_anchor_weight = float(run_contract.get("joint_pose_anchor_weight", -0.5))
    if not math.isfinite(joint_anchor_weight) or joint_anchor_weight > 0.0:
        raise ValueError("joint pose anchor weight must be finite and non-positive")
    if not pregrasp.require_strict or int(pregrasp.rank) != 0 or len(pregrasp.bindings) != len(selected_rows):
        raise ValueError("palm-rotation method requires one strict rank-0 binding per selected asset")
    root = resolve_anymani_root()
    resolved_manifest = manifest_path if manifest_path.is_absolute() else root / manifest_path
    catalog_root = Path(pregrasp.catalog_root)
    catalog_root = catalog_root if catalog_root.is_absolute() else root / catalog_root
    catalog_index = catalog_root / "index.json"
    if not resolved_manifest.is_file() or not catalog_index.is_file():
        raise FileNotFoundError("palm-rotation manifest or strict catalog index is missing")
    implementation_files = palm_rotation_implementation_files()
    orientation_goal = run_contract.get("orientation_goal")  # Keep this task mode separate from legacy KD names.
    adr_config = run_contract.get("adr", {})
    sigma_mode = run_contract.get("sigma_mode", "global")
    if sigma_mode not in {"global", "conditional"}:
        raise ValueError("unknown sigma parameterization")
    recovery_exploration = recovery_exploration_contract(
        run_contract.get("recovery_sigma_floor"),
        max_log_std=float(run_contract.get("max_log_std", -0.43)),
        sigma_mode=sigma_mode,
    )  # Use the actor's rule to keep metadata and execution thresholds aligned.
    key_digests = [hashlib.sha256(binding.key_json.encode("utf-8")).hexdigest() for binding in pregrasp.bindings]
    payload = {
        "identity_schema_version": PALM_ROTATION_IDENTITY_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_contract": {
            "object": "DexCube",
            "object_scale": 1.1,
            "rotation_axis_h": [0.0, 0.0, 1.0],
            "subgoal_degrees": 30.0,
            "training_mdp_anchor": "orientation-goal-cold" if orientation_goal else "N000-gm-tactile-rotation-v0.5.0",
            "training_goal_bonus": "so3-angle-and-position-qualified"
            if orientation_goal
            else "strict-full-pose-and-position-2p5cm",
            **(
                {"orientation_goal": orientation_goal, "goal_advance": "angle-only", "goal_reference": "previous-goal"}
                if orientation_goal
                else {}
            ),
            "evaluation_primary": "physical-frontier-net-turns-directionality-and-survival",
            "rotation_frontier_degrees": 30.0,
            "rotation_frontier_reward_weight": 0.0,
            "rotation_progress_clip_rad_per_step": float(
                run_contract.get("rotation_progress_clip_rad_per_step", 0.025)
            ),
            # Non-default weights are part of the MDP identity; retain the legacy default layout.
            **({"rotation_progress_reward_weight": progress_weight} if progress_weight != 5.0 else {}),
            **({"pose_keypoint_reward_weight": pose_weight} if pose_weight != 1.0 else {}),
            **(
                {"pose_keypoint_mode": pose_mode} if pose_mode != "full_pose" else {}
            ),  # Geometry is independent of its weight.
            "strict_tracking_reward_weight": float(run_contract.get("strict_goal_reward_weight", 10.0)),
            **({"joint_pose_anchor_weight": joint_anchor_weight} if joint_anchor_weight != -0.5 else {}),
            "critic_task_state": "axis-goal-error-max-positive-net-and-current-net",
            "episode_seconds": float(run_contract.get("episode_seconds_max", 120.0)),
            "episode_seconds_min": float(run_contract.get("episode_seconds_min", 120.0)),
            "episode_horizon_sampling": "uniform-policy-step-interval",
            "adr_enabled": bool(adr_config.get("object_position", {}).get("enabled", False)),
            **(
                {"adr": adr_config, "strict_pregrasp_scope": "nominal-anchor-before-declared-reset-perturbation"}
                if adr_config
                else {}
            ),
            "pregrasp_rank": 0,
            "pregrasp_strict": True,
            "stable_joint_reduction": "reference-dof-16",
            "linear_velocity_penalty": "world-l2-squared",
            "reward_release": {
                "aggregation": "per-asset-ema-to-handedness-inclusive-cell-median",
                "start_turns": float(run_contract.get("reward_release_start_turns", 1.0)),
                "end_turns": float(run_contract.get("reward_release_end_turns", 2.0)),
                "ema_alpha": float(run_contract.get("reward_release_ema_alpha", 0.05)),
                "floor": float(run_contract.get("reward_release_floor", 0.0)),
                "reference_seconds": float(run_contract.get("reward_release_reference_seconds", 120.0)),
            },
        },
        "policy": {
            "arm": arm,
            **({"phase_clock": phase_contract} if phase_contract is not None else {}),
            "actor_contact": "tip-only-binary"
            if run_contract.get("actor_contact", "all") == "tip"
            else "all-owner-binary-no-force",
            "distribution": "mean-preserving-tanh-squashed-active-joint-diagonal-normal",
            **({"recovery_exploration": recovery_exploration} if recovery_exploration is not None else {}),
            **(
                {
                    "sigma_mode": sigma_mode,
                    "sigma_parameterization": "global-baseline-plus-log2-tanh-contextual-joint-head",
                    "sigma_min": 0.05,
                    "sigma_max_log": float(run_contract.get("max_log_std", -0.43)),
                }
                if sigma_mode != "global"
                else {}
            ),
            "action_authority_rad_per_policy_step": 1.0 / 24.0,
            "residual_decomposition": (
                "bounded-0p8-dynamic-film-base-plus-bounded-0p2-global-action-residual"
                if arm in {"base", "residual"}
                else None
            ),
            "direct_decomposition": {
                "direct": "full-authority-contextual-plus-local-skip",
                "direct_token": "full-authority-contextual-token-only",
            }.get(arm),
            **(
                {"student_initialization": canonical_run_contract["student_initialization"]}
                if isinstance(canonical_run_contract.get("student_initialization"), Mapping)
                else {}
            ),
        },
        "manifest": {
            "path": _relative_or_absolute(resolved_manifest, root),
            "sha256": _sha256(resolved_manifest),
            "support_asset_count": len(selected_rows),
            "selected_rows": list(selected_rows),
        },
        "pregrasp": {
            "catalog_root": _relative_or_absolute(catalog_root, root),
            "index_sha256": _sha256(catalog_index),
            "ordered_key_digests": key_digests,
        },
        "geometry_provider": provider_identity,
        "implementation": {
            "files": implementation_files,
        },
        "transport_abi": {
            "float_shapes": {
                key: list(shape)
                for key, shape in palm_rotation_float_shapes(
                    phase_period_steps,
                    student=isinstance(canonical_run_contract.get("student_initialization"), Mapping),
                    include_fk_target=(
                        isinstance(canonical_run_contract.get("student_initialization"), Mapping)
                        and canonical_run_contract["student_initialization"].get("variant") == "fk"  # type: ignore[union-attr]
                    ),
                ).items()
            },
            "bool_shapes": {key: list(shape) for key, shape in PALM_ROTATION_BOOL_SHAPES.items()},
            "int16_shapes": {key: list(shape) for key, shape in PALM_ROTATION_INT16_SHAPES.items()},
        },
        "diagnostics": {
            "metrics_schema_version": PALM_ROTATION_METRICS_SCHEMA_VERSION,
            "parquet_writer": "polars-1.32.3-zstd",
            "trajectory_writer": "hdf5-gzip-v1",
            "first_window_seconds": 30.0,  # Fixed evidence window; independent of curriculum reference time.
            "first_window_capacity_per_asset": 32,
            "first_window_proxy_minimum_count": 16,
            "first_window_reduction": "median-of-asset-medians-and-asset-equal-safety",
        },
        "training": json.loads(
            json.dumps(canonical_run_contract, sort_keys=True)
        ),  # Preserve non-default modes for evaluation.
    }
    return {**payload, "identity_digest": _stable_digest(payload)}


__all__ = [
    "TASK_ID",
    "PALM_ROTATION_IDENTITY_SCHEMA_VERSION",
    "PalmRotationPregraspIdentityCfg",
    "build_palm_rotation_method_identity",
    "palm_rotation_code_provenance",
    "palm_rotation_implementation_files",
    "validate_palm_rotation_evaluation_identity",
]
