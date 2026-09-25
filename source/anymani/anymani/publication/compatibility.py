"""Validate the released source review for frozen-policy inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

ARTIFACT_TYPE = "anymani.publication.static_compatibility"


def _bounded_error(record: Mapping[str, Any], key: str, ceiling: float) -> None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RuntimeError(f"Publication evidence has no numeric {key}")
    if not math.isfinite(value) or value < 0 or value > ceiling:
        raise RuntimeError(f"Publication evidence failed {key}: {value}")


def validate_source_compatibility(
    certificate: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    current_files: Mapping[str, str],
) -> None:
    """Check exact source maps and measured inference/math evidence, excluding physics."""
    if (
        certificate.get("artifact_type") != ARTIFACT_TYPE
        or certificate.get("schema_version") != "1.0.0"
        or certificate.get("status") != "static_scope_passed"
    ):
        raise RuntimeError("Frozen replay requires the completed publication source review")
    scope = certificate.get("scope", {})
    for key in ("source_ast_after_removing_standalone_strings", "deterministic_actor_mean", "pure_task_math"):
        if scope.get(key) is not True:
            raise RuntimeError(f"Publication source review lacks {key}")
    for key in ("physics_trajectory_equivalence", "gradients_checked", "optimizer_checked", "training_resume_checked"):
        if scope.get(key) is not False:
            raise RuntimeError("The publication review is limited to frozen inference")

    cases = [
        item
        for item in certificate.get("family_cases", ())
        if item.get("identity_digest") == checkpoint.get("identity_digest")
    ]
    if len(cases) != 1:
        raise RuntimeError("Publication review does not identify this teacher checkpoint")
    case = cases[0]
    reference_files = checkpoint.get("implementation", {}).get("files")
    if case.get("reference_implementation_files") != reference_files or case.get(
        "current_implementation_files"
    ) != dict(current_files):
        raise RuntimeError("Publication review source hashes differ from the running implementation")
    comparison = case.get("source_comparison", {})
    if comparison.get("reference_map_matches_checkpoint_identity") is not True:
        raise RuntimeError("Publication reference source does not match the checkpoint")
    if case.get("n040_sha256") != checkpoint.get("geometry_provider", {}).get("retained_artifact", {}).get("sha256"):
        raise RuntimeError("Publication review uses a different geometry encoder")
    if case.get("n040_artifact_matches_checkpoint_identity") is not True:
        raise RuntimeError("Publication encoder bytes were not verified")

    review = certificate.get("source_change_review", {})
    if review.get("status") not in {"reviewed_for_frozen_inference", "no_substantive_ast_delta"}:
        raise RuntimeError("Publication source differences remain unreviewed")
    changes = [item for item in review.get("files", ()) if item.get("family") == case["family"]]
    approved = review.get("approved_changes", ())
    for change in changes:
        if not any(item.get("change") == change and item.get("reason") for item in approved):
            raise RuntimeError("Publication source review omits an executable change")

    actor = case.get("actor_forward", {})
    if actor.get("status") != "recorded_pass" or actor.get("passed") is not True:
        raise RuntimeError("Publication teacher forward comparison has not passed")
    if actor.get("output_shape") != ["B", 16] or actor.get("output_dtype") != "float32":
        raise RuntimeError("Publication actor output ABI differs")
    for key in ("max_abs", "mask_ghost_max_abs", "range_excess_max_abs"):
        _bounded_error(actor, key, 1e-5)

    math_checks = [item for item in certificate.get("task_math", ()) if item.get("family") == case["family"]]
    if (
        len(math_checks) != 1
        or math_checks[0].get("status") != "recorded_pass"
        or math_checks[0].get("passed") is not True
    ):
        raise RuntimeError("Publication task-math comparison has not passed")
    _bounded_error(math_checks[0], "max_abs", 1e-7)
    required = {"projected_space_rotation_delta", "rotation_frontier_update", "task_termination_flags"}
    if not required.issubset(set(math_checks[0].get("functions", ()))):
        raise RuntimeError("Publication task-math comparison is incomplete")
    evidence = certificate.get("existing_parity_reports", {})
    for key in ("teacher_cuda_labels", "student_real_input_loader"):
        if evidence.get(key, {}).get("status") != "recorded_pass":
            raise RuntimeError(f"Publication model evidence is missing: {key}")


def geometry_semantics(identity: Mapping[str, Any], *, keep_population: bool = False) -> dict[str, Any]:
    """Compare encoder content and precision independently of its disk location."""
    excluded = {"base_provider_identity_digest", "identity_digest"}
    if not keep_population:
        excluded.update(("asset_ids", "physical_geometry_hashes"))
    result = {key: value for key, value in identity.items() if key not in excluded}
    retained = dict(result.get("retained_artifact", {}))
    retained.pop("path", None)
    result["retained_artifact"] = retained
    return result


def manifest_semantics(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Keep manifest bytes and the ordered member axis independently of its path."""
    return {key: value for key, value in identity.items() if key != "path"}


def pregrasp_semantics(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Keep catalog revision and ordered keys independently of its directory."""
    return {key: value for key, value in identity.items() if key != "catalog_root"}
