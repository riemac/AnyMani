r"""Validate frozen-policy evaluation on a different morphology population.

Only the selected asset population may change. Policy/task/observation semantics,
retained geometry encoder, precision and implementation remain checked. Shared
physical assets retain their exact ranked pregrasps. Catalog growth is allowed
only after checking the archived source index and current selected payloads.
These rules authorize evaluation, not full PPO-state resume or a capability pass.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from anymani.publication.compatibility import (
    ARTIFACT_TYPE,
    geometry_semantics,
    manifest_semantics,
    validate_source_compatibility,
)

FROZEN_EVALUATION_CERTIFICATE_TYPE = "anymani.palm_rotation.frozen_evaluation_equivalence"
FROZEN_EVALUATION_CERTIFICATE_SCHEMA_VERSION = "1.0.0"
FROZEN_EVALUATION_SEMANTIC_FIELDS = (
    "task_id",
    "task_contract",
    "policy",
    "transport_abi",
    "training",
)


def _catalog_entries(pregrasp: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    r"""Read the exact catalog revision referenced by an evaluation or checkpoint."""
    catalog = root / pregrasp["catalog_root"]
    expected = pregrasp["index_sha256"]
    candidates = [catalog / "index.json", catalog / "index_history" / f"{expected}.json"]
    data_dir = os.environ.get("ANYMANI_DATA_DIR")
    if data_dir:
        assets = (Path(data_dir) / "assets").resolve(strict=True)
        bundle = json.loads((assets / "paper_bundle.json").read_text(encoding="utf-8"))
        for cohort in bundle["cohorts"].values():
            metadata = cohort["pregrasp_catalog"]
            for key in ("index_path", "original_index_path"):
                candidate = (assets / metadata[key]).resolve(strict=True)
                if not candidate.is_relative_to(assets):
                    raise RuntimeError("Pregrasp index path escapes the asset bundle")
                candidates.append(candidate)
    payload = None
    for candidate in candidates:
        if candidate.is_file():
            contents = candidate.read_bytes()
            if hashlib.sha256(contents).hexdigest() == expected:
                payload = contents
                break
    if payload is None:
        raise RuntimeError("pregrasp catalog revision does not match its recorded contents")
    entries = json.loads(payload)["entries"]
    by_key = {entry["key_digest"]: entry for entry in entries}
    if len(by_key) != len(entries):
        raise RuntimeError("pregrasp catalog contains duplicate keys")
    return by_key


def _identity_sha(identity: Mapping[str, Any], name: str) -> str:
    """Return the retained N040 artifact SHA from an identity record."""

    geometry = identity.get("geometry_provider")
    if not isinstance(geometry, Mapping):
        raise RuntimeError(f"{name} identity misses geometry_provider")
    retained = geometry.get("retained_artifact")
    if not isinstance(retained, Mapping) or not isinstance(retained.get("sha256"), str):
        raise RuntimeError(f"{name} identity misses retained N040 artifact SHA")
    return str(retained["sha256"])


def _semantic_digest(identity: Mapping[str, Any]) -> str:
    """Hash task, control, observation, and shared N040 semantics.

    Transfer evaluation may change the asset population and pregrasp records. The validator compares the other fields separately; this digest binds the certificate to its reference identity without claiming PPO training equivalence.
    """

    geometry = identity.get("geometry_provider")
    if not isinstance(geometry, Mapping):
        raise RuntimeError("frozen evaluation identity misses geometry_provider")
    geometry_shared = {
        key: value
        for key, value in geometry.items()
        if key not in {"asset_ids", "physical_geometry_hashes", "base_provider_identity_digest", "identity_digest"}
    }
    payload = {key: identity.get(key) for key in FROZEN_EVALUATION_SEMANTIC_FIELDS}
    payload["geometry_provider"] = geometry_shared
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_hex(value: Any, name: str) -> str:
    """Validate a certificate field as a 64-character hexadecimal SHA-256."""

    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"frozen evaluation certificate {name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(f"frozen evaluation certificate {name} must be hexadecimal") from error
    return value.lower()


def _finite_nonnegative(value: Any, name: str) -> float:
    """Parse a finite, nonnegative parity value."""

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"frozen evaluation certificate {name} must be numeric") from error
    if result < 0.0 or not result == result or result == float("inf") or result == -float("inf"):
        raise RuntimeError(f"frozen evaluation certificate {name} must be finite and non-negative")
    return result


def _validate_frozen_certificate(
    certificate: Mapping[str, Any],
    *,
    source_files: Mapping[str, Any],
    target_files: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a bounded certificate for deterministic frozen-actor evaluation.

    The certificate binds both source maps, reference identity, N040 bytes, input ABI, actor parity, and excluded training scopes. The outer validator still checks task, reset, population, and pregrasp semantics. This function does not certify PPO training or physics equivalence.
    """

    if certificate.get("artifact_type") != FROZEN_EVALUATION_CERTIFICATE_TYPE:
        raise RuntimeError("frozen evaluation requires a frozen_evaluation certificate artifact")
    if certificate.get("schema_version") != FROZEN_EVALUATION_CERTIFICATE_SCHEMA_VERSION:
        raise RuntimeError("unsupported frozen evaluation certificate schema")
    if certificate.get("scope") != "frozen_evaluation":
        raise RuntimeError("certificate scope must be frozen_evaluation")
    if certificate.get("passed") is not True:
        raise RuntimeError("frozen evaluation certificate is not passed")
    if certificate.get("reference_implementation_files") != dict(source_files):
        raise RuntimeError("frozen evaluation certificate does not cover the reference implementation map")
    if certificate.get("current_implementation_files") != dict(target_files):
        raise RuntimeError("frozen evaluation certificate does not cover the current implementation map")
    changed_paths = sorted(
        path for path in set(source_files) | set(target_files) if source_files.get(path) != target_files.get(path)
    )
    equal_paths = sorted(
        path for path in set(source_files) & set(target_files) if source_files[path] == target_files[path]
    )
    if certificate.get("changed_paths") != changed_paths:
        raise RuntimeError("frozen evaluation certificate changed_paths disagrees with source maps")
    if certificate.get("byte_equal_paths") != equal_paths:
        raise RuntimeError("frozen evaluation certificate byte_equal_paths disagrees with source maps")
    map_digest_payload = {
        "reference": dict(source_files),
        "current": dict(target_files),
    }
    encoded_maps = json.dumps(map_digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    expected_maps_digest = hashlib.sha256(encoded_maps).hexdigest()
    if certificate.get("source_maps_sha256") != expected_maps_digest:
        raise RuntimeError("frozen evaluation certificate source map digest disagrees")
    _sha256_hex(certificate.get("reference_source_sha256"), "reference_source_sha256")
    _sha256_hex(certificate.get("current_source_sha256"), "current_source_sha256")
    ppo_source_path = "source/anymani/anymani/distill/rl/palm_rotation_ppo.py"
    if certificate.get("reference_source_sha256") != source_files.get(ppo_source_path) or certificate.get(
        "current_source_sha256"
    ) != target_files.get(ppo_source_path):
        raise RuntimeError("frozen evaluation certificate source SHA does not bind palm_rotation_ppo.py map entries")

    reference_identity_digest = certificate.get("reference_identity_digest")
    if reference_identity_digest != checkpoint.get("identity_digest"):
        raise RuntimeError("frozen evaluation certificate reference identity disagrees with checkpoint")
    if certificate.get("reference_semantic_digest") != _semantic_digest(checkpoint):
        raise RuntimeError("frozen evaluation certificate MDP/control semantic digest disagrees")
    reference_n040 = _sha256_hex(certificate.get("reference_n040_sha256"), "reference_n040_sha256")
    if reference_n040 != _identity_sha(checkpoint, "reference checkpoint").lower():
        raise RuntimeError("frozen evaluation certificate retained N040 SHA disagrees with checkpoint")
    if certificate.get("actor_input_abi") != checkpoint.get("transport_abi"):
        raise RuntimeError("frozen evaluation certificate Actor input ABI disagrees with checkpoint")
    checkpoint_training = checkpoint.get("training")
    checkpoint_policy = checkpoint.get("policy")
    if not isinstance(checkpoint_training, Mapping) or not isinstance(checkpoint_policy, Mapping):
        raise RuntimeError("frozen evaluation checkpoint lacks training/policy branch guards")
    if (
        checkpoint_training.get("orientation_goal") is not None
        or checkpoint_training.get("phase_period_steps") is not None
        or checkpoint_policy.get("phase_clock") not in (None, False)
    ):
        raise RuntimeError("frozen evaluation cannot certify an enabled orientation-goal or phase branch")

    parity = certificate.get("inference_parity")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise RuntimeError("frozen evaluation certificate lacks passed default Actor inference parity")
    if parity.get("execution_scope") != "default_actor_deterministic_mean_only":
        raise RuntimeError("frozen evaluation certificate inference scope is not deterministic Actor mean")
    tolerance = _finite_nonnegative(parity.get("tolerance"), "inference_parity.tolerance")
    if tolerance <= 0.0:
        raise RuntimeError("frozen evaluation certificate parity tolerance must be positive")
    for key in ("max_abs", "mask_ghost_max_abs", "range_excess_max_abs"):
        if _finite_nonnegative(parity.get(key), f"inference_parity.{key}") > tolerance:
            raise RuntimeError("frozen evaluation certificate default Actor parity did not pass")
    if parity.get("output_shape") != ["B", 16] or parity.get("output_dtype") != "float32":
        raise RuntimeError("frozen evaluation certificate Actor output ABI disagrees")
    old_root = parity.get("old_source_root")
    current_root = parity.get("current_source_root")
    if not isinstance(old_root, str) or not isinstance(current_root, str) or old_root == current_root:
        raise RuntimeError("frozen evaluation certificate lacks isolated old/current source roots")
    required_modules = {
        "anymani.distill.rl.palm_rotation_ppo",
        "anymani.distill.rl.runtime.palm_rotation_network",
        "anymani.distill.models.palm_rotation_policy",
        "anymani.distill.rl.runtime.palm_rotation_vecenv",
        "anymani.distill.rl.masked_ppo",
    }
    for label, imports in (("old", parity.get("old_imports")), ("current", parity.get("current_imports"))):
        if not isinstance(imports, Mapping) or not required_modules.issubset(set(imports)):
            raise RuntimeError(f"frozen evaluation certificate lacks isolated {label} import provenance")
        for module_name in required_modules:
            info = imports.get(module_name)
            if not isinstance(info, Mapping) or not isinstance(info.get("path"), str):
                raise RuntimeError(f"frozen evaluation certificate {label} import path is malformed")
            _sha256_hex(info.get("sha256"), f"inference_parity.{label}_imports.{module_name}.sha256")
    ppo_source_path = "source/anymani/anymani/distill/rl/palm_rotation_ppo.py"
    old_ppo_import = parity["old_imports"]["anymani.distill.rl.palm_rotation_ppo"]
    if old_ppo_import.get("sha256") != source_files.get(ppo_source_path):
        raise RuntimeError("frozen evaluation old Actor import SHA disagrees with reference source map")
    current_ppo_import = parity["current_imports"]["anymani.distill.rl.palm_rotation_ppo"]
    if current_ppo_import.get("sha256") != target_files.get(ppo_source_path):
        raise RuntimeError("frozen evaluation current Actor import SHA disagrees with current source map")

    scope_guards = certificate.get("scope_guards")
    if not isinstance(scope_guards, Mapping):
        raise RuntimeError("frozen evaluation certificate misses scope_guards")
    if scope_guards.get("actor_override_required") is not True:
        raise RuntimeError("frozen evaluation certificate must require an explicit actor_override")
    for key in ("training_ast_checked", "gradient_checked", "optimizer_checked", "resume_checked"):
        if scope_guards.get(key) is not False:
            raise RuntimeError(f"frozen evaluation certificate {key} must be false outside frozen scope")
    expected_semantic_fields = list(FROZEN_EVALUATION_SEMANTIC_FIELDS)
    if scope_guards.get("identity_fields") != expected_semantic_fields:
        raise RuntimeError("frozen evaluation certificate identity field guard is incomplete")

    mdp_evidence = certificate.get("mdp_evidence")
    if not isinstance(mdp_evidence, Mapping):
        raise RuntimeError("frozen evaluation certificate misses MDP evidence declaration")
    for key in ("identity_checked", "shared_pregrasp_checked", "population_checked"):
        if mdp_evidence.get(key) is not True:
            raise RuntimeError(f"frozen evaluation certificate {key} is not asserted")
    if mdp_evidence.get("runtime_physics_checked") is not False:
        raise RuntimeError("frozen evaluation certificate must leave runtime physics to the evaluator")
    if mdp_evidence.get("runtime_physics_required") is not True:
        raise RuntimeError("frozen evaluation certificate must require runtime physics evidence")
    branch_guards = certificate.get("branch_guards")
    if branch_guards != {
        "orientation_goal": "disabled",
        "phase_clock": "disabled",
        "verification": "reference_identity_and_runtime_contract",
    }:
        raise RuntimeError("frozen evaluation certificate must prove orientation/phase optional branches are disabled")
    pure_math = certificate.get("task_math_parity")
    if not isinstance(pure_math, Mapping) or pure_math.get("passed") is not True:
        raise RuntimeError("frozen evaluation certificate lacks task_math pure-function parity")
    if pure_math.get("execution_scope") != "default_control_progress_termination_functions":
        raise RuntimeError("frozen evaluation task_math parity scope is invalid")
    math_tolerance = _finite_nonnegative(pure_math.get("tolerance"), "task_math_parity.tolerance")
    if (
        math_tolerance <= 0.0
        or _finite_nonnegative(pure_math.get("max_abs"), "task_math_parity.max_abs") > math_tolerance
    ):
        raise RuntimeError("frozen evaluation task_math parity did not pass")
    required_math_functions = {
        "projected_space_rotation_delta",
        "rotation_frontier_update",
        "task_termination_flags",
    }
    if (
        not isinstance(pure_math.get("functions"), list)
        or not pure_math["functions"]
        or not required_math_functions.issubset(set(pure_math["functions"]))
    ):
        raise RuntimeError("frozen evaluation certificate task_math function coverage is empty")
    pending = mdp_evidence.get("pending_runtime_checks")
    required_pending = {
        "scene construction and imported articulation/fixed-joint lowering",
        "reset q/limits/pregrasp target buffers",
        "contact sensor bits and all-owner contact reduction",
        "action target buffer and physical step terminal snapshot",
        "R16 first-trajectory replay with all failure denominators retained",
    }
    if not isinstance(pending, list) or not required_pending.issubset(set(pending)):
        raise RuntimeError("frozen evaluation certificate must retain pending reset/contact/action runtime checks")
    byte_equal_input_files = mdp_evidence.get("byte_equal_input_files")
    if (
        not isinstance(byte_equal_input_files, list)
        or not byte_equal_input_files
        or not set(byte_equal_input_files).issubset(set(equal_paths))
        or not any("observation" in path for path in byte_equal_input_files)
        or not any("contact" in path for path in byte_equal_input_files)
    ):
        raise RuntimeError("frozen evaluation certificate lacks byte-equal observation/contact input evidence")
    return dict(certificate)


def validate_transfer_evaluation(
    runtime: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    root: Path,
    implementation_certificate: Mapping[str, Any] | None = None,
    _frozen_evaluation: bool = False,
    _actor_override: Any = None,
) -> dict[str, Any]:
    r"""Validate shared semantics and report the changed evaluation population.

    The `training` block still describes the source checkpoint. New assets are
    declared by the runtime manifest/provider/pregrasp fields, never represented
    as newly trained assets. Geometry population digests may vary; the retained
    artifact, feature definitions and precision may not.
    """
    for identity in (runtime, checkpoint):
        if identity.get("identity_schema_version") not in {"3.0.0", "4.0.0"}:
            raise RuntimeError("unsupported transfer evaluation identity schema")
        payload = {key: value for key, value in identity.items() if key != "identity_digest"}
        actual = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        if actual != identity.get("identity_digest"):
            raise RuntimeError("transfer evaluation identity does not match its payload")

    for field in ("task_id", "task_contract", "policy", "transport_abi", "training"):
        if field not in runtime or runtime[field] != checkpoint.get(field):
            raise RuntimeError(f"transfer evaluation changed shared semantics: {field}")
    geometry = [identity["geometry_provider"] for identity in (checkpoint, runtime)]
    shared_geometry = [geometry_semantics(item) for item in geometry]
    if shared_geometry[0] != shared_geometry[1] or not shared_geometry[0].get("retained_artifact"):
        raise RuntimeError("transfer evaluation changed geometry encoder or precision semantics")

    source_files = checkpoint.get("implementation", {}).get("files")
    target_files = runtime.get("implementation", {}).get("files")
    if not source_files or not target_files:
        raise RuntimeError("transfer evaluation requires implementation identities")
    if (
        isinstance(implementation_certificate, Mapping)
        and implementation_certificate.get("artifact_type") == ARTIFACT_TYPE
    ):
        validate_source_compatibility(implementation_certificate, checkpoint=checkpoint, current_files=target_files)
    elif _frozen_evaluation:
        if _actor_override is None:
            raise RuntimeError("frozen evaluation validation requires an explicit actor_override")
        _validate_frozen_certificate(
            implementation_certificate or {},
            source_files=source_files,
            target_files=target_files,
            checkpoint=checkpoint,
        )
    elif source_files != target_files:
        certificate = implementation_certificate or {}
        if not (
            certificate.get("artifact_type") == "anymani.palm_rotation.refactor_equivalence"
            and certificate.get("schema_version") == "1.0.0"
            and certificate.get("passed") is True
            and certificate.get("reference_implementation_files") == source_files
            and certificate.get("current_implementation_files") == target_files
        ):
            raise RuntimeError("transfer evaluation implementation changed without an exact equivalence certificate")

    # Pair reset identities by physical geometry, not by selection-local row number.
    physical_keys = []
    assets = []
    for identity, provider in zip((checkpoint, runtime), geometry, strict=True):
        physical = provider["physical_geometry_hashes"]
        ids = provider["asset_ids"]
        keys = identity["pregrasp"]["ordered_key_digests"]
        count = identity["manifest"]["support_asset_count"]
        if not (len(physical) == len(ids) == len(keys) == count) or len(set(physical)) != count:
            raise RuntimeError("transfer population/pregrasp axes disagree")
        physical_keys.append(dict(zip(physical, keys, strict=True)))
        assets.append(dict(zip(ids, physical, strict=True)))
    for asset in assets[0].keys() & assets[1].keys():
        if assets[0][asset] != assets[1][asset]:
            raise RuntimeError("shared asset physical geometry changed")

    target_entries = _catalog_entries(runtime["pregrasp"], root)
    if any(key not in target_entries for key in physical_keys[1].values()):
        raise RuntimeError("target pregrasp catalog misses selected assets")
    shared = physical_keys[0].keys() & physical_keys[1].keys()
    if shared:
        source_entries = _catalog_entries(checkpoint["pregrasp"], root)
        for physical in shared:
            key = physical_keys[0][physical]
            if key != physical_keys[1][physical] or key not in source_entries:
                raise RuntimeError("shared asset pregrasp protocol changed")
            if source_entries[key]["entry_digest"] != target_entries[key]["entry_digest"]:
                raise RuntimeError("shared asset pregrasp payload changed")
    return {
        "source_asset_count": len(physical_keys[0]),
        "target_asset_count": len(physical_keys[1]),
        "shared_asset_count": len(shared),
        "new_asset_count": len(physical_keys[1]) - len(shared),
        "support_changed": manifest_semantics(runtime["manifest"]) != manifest_semantics(checkpoint["manifest"]),
        "shared_pregrasp_records_equal": True,
        "source_cohort_id": checkpoint["training"].get("cohort_id"),
    }


def validate_frozen_evaluation(
    runtime: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    root: Path,
    implementation_certificate: Mapping[str, Any] | None = None,
    actor_override: Any = None,
) -> dict[str, Any]:
    """Validate an explicit frozen actor override for evaluation only.

    Transfer checks still compare task, control, transport, geometry, population, and pregrasp semantics. The certificate scope is limited to default actor inference and task-math parity; PPO agents, gradients, optimizers, resume, and physics trajectories are not certified. An explicit callback is required.
    """

    if actor_override is None:
        raise RuntimeError("validate_frozen_evaluation requires an explicit actor_override")
    result = validate_transfer_evaluation(
        runtime,
        checkpoint,
        root=root,
        implementation_certificate=implementation_certificate,
        _frozen_evaluation=True,
        _actor_override=actor_override,
    )
    result.update(
        {
            "scope": "frozen_evaluation",
            "passed_meaning": "frozen_actor_interface_compatibility_only",
            "training_refactor_equivalence": False,
            "physics_trajectory_equivalence": False,
        }
    )
    return result
