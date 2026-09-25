(
    'Bounded pure-CPU input contract for Allegro128 local joint-limit revision. '
    "This artifact starts from each asset's ranked parent Top-8 and projects only "
    'expanded non-thumb j1 slots upward; it is not a Sobol/CEM search. Identity '
    'reuses the original strict 1 s physics gate, but these candidates remain '
    'uncertified and cannot enter the certified catalog. Store canonical 16-slot '
    'radians with zero ghosts and upright hand-frame object quaternion (1,0,0,0). '
    'Import no tasks, robots, Isaac Lab, Torch, or CUDA. The physical runner must '
    'bind candidates to revised assets and run the real PhysX gate; this module '
    'makes file identity/source/order checks repeatable and fail-closed.'
)

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .schema import canonical_json_bytes

# Independent artifact/kind/schema tuple; do not confuse this with historical c82 catalog.
REVALIDATION_GENERATION_ARTIFACT_TYPE = "anymani.pregrasp.revalidation_generation_identity"
REVALIDATION_GENERATION_KIND = "allegro128.local_joint_limit_revalidation"
REVALIDATION_GENERATION_SCHEMA_VERSION = "1.0.0"
REVALIDATION_GENERATION_PROTOCOL = "allegro128-local-joint-limit-revalidation-v1"
REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE = "anymani.pregrasp.revalidation_refinement_generation_identity"
REVALIDATION_REFINEMENT_GENERATION_KIND = "allegro128.local_joint_limit_revalidation_with_cem"
REVALIDATION_REFINEMENT_GENERATION_SCHEMA_VERSION = "2.0.0"
REVALIDATION_REFINEMENT_GENERATION_PROTOCOL = "allegro128-local-joint-limit-revalidation-cem-v2"

# Candidate source and publication count are fixed; preserve parent ranks 0..7 per asset.
REVALIDATION_CANDIDATE_SOURCE = "original-top8-with-projection"
REVALIDATION_CANDIDATE_COUNT = 8
REVALIDATION_RANKING = "preserve-parent-rank"
REVALIDATION_MAX_ASSET_COUNT = 128
REVALIDATION_SCENE_SEED = 20260902
REVALIDATION_REFINED_RANKING = "preserve-parent-rank-when-all-initial-pass-else-physical-quality"

# Projected state must retain 10.1% normalized margin under the new upper limit; the original strict gate
# remains 10%. Maximum delta is an auditable value from this round's fixture, in rad.
REVALIDATION_PROJECTION_MARGIN_FRACTION = 0.101
REVALIDATION_STRICT_MARGIN_FRACTION = 0.10
REVALIDATION_MAX_PROJECTION_DELTA_RAD = 0.06453248217813295

# Original strict cold-reset window; identity records the protocol but does not simulate.
REVALIDATION_STRICT_PHYSICS_WINDOW_SECONDS = 1.0
REVALIDATION_UPRIGHT_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)

# Only these canonical non-thumb j1 slots allow upward projection.
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_NPZ_KEYS = frozenset(
    {
        "original_q_rad",
        "projected_q_rad",
        "object_position_h_m",
        "object_orientation_h_wxyz",
        "active_joint_mask",
        "parent_asset_id",
        "new_asset_id",
        "parent_entry_digest",
        "canonical_joint_names",
    }
)
_TARGET_KEY_MARKERS = ("target", "qtarget")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    (
        'Compute canonical JSON SHA-256 with sorted fields and NaN rejection. Use for '
        'JSON generation identity only. Hash NPZ identity directly from '
        'path.read_bytes(); do not convert compressed contents to JSON and reuse that '
        'digest. Require finite JSON-native mappings; raise ValueError for non-finite '
        'or unserializable values.'
    )

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()  # Stable content digest for identity.


def stable_digest(payload: Mapping[str, Any]) -> str:
    'Canonical JSON digest alias matching pregrasp schema semantics.'

    return canonical_json_sha256(payload)  # One public pure-CPU digest entry point.


def _validate_sha256(value: Any, field_name: str) -> str:
    'Accept lowercase 64-character SHA-256 only; never fix caller casing.'

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256")
    return value


def _fixed_float(value: Any, *, expected: float, field_name: str) -> float:
    'Validate fixed protocol floats; reject bool, NaN, infinity, or changed limits.'

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must equal fixed protocol value {expected}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must equal fixed protocol value {expected}") from error
    if not math.isfinite(parsed) or not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{field_name} must equal fixed protocol value {expected}")
    return expected


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    'Require nested identity fields to be mappings; do not let strings masquerade as iterables.'

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


_REFINEMENT_REQUIRED_KEYS = frozenset(
    {
        "elite_physical_candidates",
        "elite_proposal_counts_descending",
        "full_physics_candidates_per_round",
        "joint_pca_dimensions",
        "object_position_dimensions",
        "per_asset_manual_parameters",
        "position_center_feedback",
        "proposals_per_round_per_failed_asset",
        "random_stream_key",
        "rounds_max",
        "settle_height_feedback",
        "strict_mode_exploitation",
        "type",
    }
)
_REFINEMENT_CENTER_COUNTS = (24, 20, 16, 12, 8, 8, 6, 6, 4, 4, 4, 4, 3, 3, 3, 3)


def _json_mapping_copy(value: Any, field_name: str) -> dict[str, Any]:
    'Freeze nested profile through canonical JSON round-trip so caller mutation cannot change identity.'

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    try:
        normalized = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{field_name} must contain finite JSON values") from error
    if not isinstance(normalized, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return normalized


def _fixed_int(value: Any, *, expected: int, field_name: str) -> int:
    'Validate fixed integer budgets; distinguish bool, float, and actual counts.'

    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) != expected:
        raise ValueError(f"{field_name} must equal fixed protocol value {expected}")
    return expected


def _validate_refinement_identity(value: Any, field_name: str = "refinement") -> dict[str, Any]:
    'Validate every field of the existing strict low-rank CEM refinement identity.'

    refinement = _json_mapping_copy(value, field_name)
    if set(refinement) != _REFINEMENT_REQUIRED_KEYS:
        missing = sorted(_REFINEMENT_REQUIRED_KEYS - set(refinement))
        extra = sorted(set(refinement) - _REFINEMENT_REQUIRED_KEYS)
        raise ValueError(f"{field_name} keys drifted; missing={missing}, extra={extra}")
    _fixed_int(
        refinement["elite_physical_candidates"], expected=16, field_name=f"{field_name}.elite_physical_candidates"
    )
    counts = refinement["elite_proposal_counts_descending"]
    if counts != list(_REFINEMENT_CENTER_COUNTS):
        raise ValueError(f"{field_name}.elite_proposal_counts_descending must preserve the legacy allocation")
    _fixed_int(
        refinement["full_physics_candidates_per_round"],
        expected=128,
        field_name=f"{field_name}.full_physics_candidates_per_round",
    )
    _fixed_int(refinement["joint_pca_dimensions"], expected=4, field_name=f"{field_name}.joint_pca_dimensions")
    _fixed_int(
        refinement["object_position_dimensions"],
        expected=3,
        field_name=f"{field_name}.object_position_dimensions",
    )
    if refinement["per_asset_manual_parameters"] is not False:
        raise ValueError(f"{field_name}.per_asset_manual_parameters must be false")
    if refinement["position_center_feedback"] != "minus_physx_contact_normal_times_1p10_depth_plus_0p25mm":
        raise ValueError(f"{field_name}.position_center_feedback drifted")
    _fixed_int(
        refinement["proposals_per_round_per_failed_asset"],
        expected=128,
        field_name=f"{field_name}.proposals_per_round_per_failed_asset",
    )
    if refinement["random_stream_key"] != "source_content_and_physical_geometry_sha256":
        raise ValueError(f"{field_name}.random_stream_key drifted")
    _fixed_int(refinement["rounds_max"], expected=3, field_name=f"{field_name}.rounds_max")
    if refinement["settle_height_feedback"] != (
        "if_initial_penetration_le_0p5mm_lower_by_clamped_displacement_minus_4p5mm"
    ):
        raise ValueError(f"{field_name}.settle_height_feedback drifted")
    if refinement["type"] != "per_asset_elite_mixture_low_rank_gaussian_cem":
        raise ValueError(f"{field_name}.type must identify the existing low-rank CEM")
    exploitation = _json_mapping_copy(refinement["strict_mode_exploitation"], f"{field_name}.strict_mode_exploitation")
    if set(exploitation) != {"activation", "joint_std_rad", "max_proposals_per_round", "position_std_m"}:
        raise ValueError(f"{field_name}.strict_mode_exploitation keys drifted")
    if exploitation["activation"] != "one_to_seven_strict_or_normalized_gate_violation_le_0p35":
        raise ValueError(f"{field_name}.strict_mode_exploitation.activation drifted")
    _fixed_float(
        exploitation["joint_std_rad"],
        expected=0.0005,
        field_name=f"{field_name}.strict_mode_exploitation.joint_std_rad",
    )
    _fixed_int(
        exploitation["max_proposals_per_round"],
        expected=96,
        field_name=f"{field_name}.strict_mode_exploitation.max_proposals_per_round",
    )
    position_std = exploitation["position_std_m"]
    if position_std != [5e-05, 5e-05, 2.5e-05]:
        raise ValueError(f"{field_name}.strict_mode_exploitation.position_std_m drifted")
    return refinement


def build_revalidation_generation_identity(
    *,
    candidate_npz_sha256: str,
    source_catalog_index_sha256: str,
    strict_gate_digest: str,
    physics_identity_digest: str,
    candidate_count: int = REVALIDATION_CANDIDATE_COUNT,
    projection_margin_fraction: float = REVALIDATION_PROJECTION_MARGIN_FRACTION,
    refinement_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    (
        'Build a separate generation identity for Allegro128 local joint-limit '
        "revision. Start from each asset's parent-ranked Top-8 and project only "
        'non-thumb j1 values into [lower_new+0.101*range, upper_new-0.101*range]. '
        'This input contract records the 10.1% projection margin; the upper layer '
        'supplies revised limits and real contact physics. Bind the original strict 1 '
        's gate digest and its 10% margin unchanged. Add no Sobol/CEM search, random '
        'seed, or candidate generation. Preserve v1 identity exactly when '
        'refinement_identity is None; a supplied mapping enables a separate '
        'three-round resume protocol. Return JSON-safe identity; reject invalid '
        'digests, candidate count other than 8, or margin other than 0.101.'
    )

    candidate_sha = _validate_sha256(candidate_npz_sha256, "candidate_npz_sha256")
    source_sha = _validate_sha256(source_catalog_index_sha256, "source_catalog_index_sha256")
    strict_sha = _validate_sha256(strict_gate_digest, "strict_gate_digest")
    physics_sha = _validate_sha256(physics_identity_digest, "physics_identity_digest")
    try:
        parsed_count = int(candidate_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("candidate_count must be fixed at 8 for revalidation") from error
    if (
        not isinstance(candidate_count, Integral)
        or isinstance(candidate_count, bool)
        or parsed_count != candidate_count
    ):
        raise ValueError("candidate_count must be fixed at 8 for revalidation")
    if parsed_count != REVALIDATION_CANDIDATE_COUNT:
        raise ValueError("candidate_count must be fixed at 8 for revalidation")
    projection_margin = _fixed_float(
        projection_margin_fraction,
        expected=REVALIDATION_PROJECTION_MARGIN_FRACTION,
        field_name="projection_margin_fraction",
    )

    # Top-level digest supports direct runner/task checks; nested fields preserve scientific meaning
    # so identity does not hide which physical gate it reuses.
    base_identity = {
        "artifact_type": REVALIDATION_GENERATION_ARTIFACT_TYPE,
        "kind": REVALIDATION_GENERATION_KIND,
        "schema_version": REVALIDATION_GENERATION_SCHEMA_VERSION,
        "protocol": REVALIDATION_GENERATION_PROTOCOL,
        "candidate_source": REVALIDATION_CANDIDATE_SOURCE,
        "candidate_count": REVALIDATION_CANDIDATE_COUNT,
        "candidate_count_per_asset": REVALIDATION_CANDIDATE_COUNT,
        "projection_margin_fraction": projection_margin,
        "strict_gate_margin_fraction": REVALIDATION_STRICT_MARGIN_FRACTION,
        "candidate_npz_sha256": candidate_sha,
        "source_catalog_index_sha256": source_sha,
        "strict_gate_digest": strict_sha,
        "physics_identity_digest": physics_sha,
        "strict_physics_gate": {
            "name": "original-strict-1s-physics-gate",
            "window_seconds": REVALIDATION_STRICT_PHYSICS_WINDOW_SECONDS,
            "margin_fraction": REVALIDATION_STRICT_MARGIN_FRACTION,
            "gate_digest": strict_sha,
        },
        "projection": {
            "operation": "clip-to-new-limit-with-margin",
            "allowed_joint_class": "non-thumb-j1-upper",
            "max_delta_rad": REVALIDATION_MAX_PROJECTION_DELTA_RAD,
        },
        "search": {
            "source": REVALIDATION_CANDIDATE_SOURCE,
            "uses_sobol": False,
            "uses_cem": False,
            "sobol": False,
            "cem": False,
            "randomized": False,
        },
        "ranking": REVALIDATION_RANKING,
        "publication": {
            "candidate_count_per_asset": REVALIDATION_CANDIDATE_COUNT,
            "ranking": REVALIDATION_RANKING,
            "physical_certification_required": True,
        },
        "candidate_status": "candidates-only-not-physically-certified",
    }
    if refinement_identity is None:
        return base_identity  # Keep v1 dictionary and canonical digest value-for-value unchanged.

    refinement = _validate_refinement_identity(refinement_identity)
    # New identity keeps the same Top-8/source/physics/gate and records “resume only when fewer than 8”
    # plus legacy CEM center/elite allocation; these fields enter stable_digest.
    return {
        **base_identity,
        "artifact_type": REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE,
        "kind": REVALIDATION_REFINEMENT_GENERATION_KIND,
        "schema_version": REVALIDATION_REFINEMENT_GENERATION_SCHEMA_VERSION,
        "protocol": REVALIDATION_REFINEMENT_GENERATION_PROTOCOL,
        "seed": REVALIDATION_SCENE_SEED,
        "initial": {
            "candidate_source": REVALIDATION_CANDIDATE_SOURCE,
            "candidate_count": REVALIDATION_CANDIDATE_COUNT,
            "uses_sobol": False,
            "seed": REVALIDATION_SCENE_SEED,
        },
        "search": {
            "source": REVALIDATION_CANDIDATE_SOURCE,
            "uses_sobol": False,
            "uses_cem": True,
            "sobol": False,
            "cem": True,
            "randomized": True,
        },
        "refinement_rounds": 3,
        "refinement": refinement,
        "refinement_scope": "assets-with-fewer-than-8-initial-strict-passes",
        "cem_continuation": {
            "initial_elite_count": 8,
            "initial_center_assignment": "round_robin",
            "later_elite_count": 16,
            "later_center_assignment": "legacy_cem_proposal_counts_descending",
        },
        "ranking": REVALIDATION_REFINED_RANKING,
        "publication": {
            "candidate_count_per_asset": REVALIDATION_CANDIDATE_COUNT,
            "ranking": REVALIDATION_REFINED_RANKING,
            "physical_certification_required": True,
        },
        "source_hashes": {
            "candidate_npz_sha256": candidate_sha,
            "source_catalog_index_sha256": source_sha,
            "strict_gate_digest": strict_sha,
            "physics_identity_digest": physics_sha,
        },
    }


def _reject_c82_markers(document: Mapping[str, Any]) -> None:
    'Reject historical c82 protocol/algorithm fields in the revised identity.'

    for field_name in ("protocol", "algorithm", "generation_protocol", "kind"):
        value = document.get(field_name)
        if isinstance(value, str) and "c82" in value.lower():
            raise ValueError("revalidation identity must not masquerade as legacy c82 protocol")


def validate_revalidation_generation_identity(document: Mapping[str, Any]) -> dict[str, Any]:
    (
        'Validate and return the original revised-generation identity mapping. '
        'Enforce separate artifact/kind/schema, parent source, 8 candidates per '
        'asset, 10.1% projection, original strict 1 s/10% gate, no Sobol/CEM, and '
        'four persistent digests. Permit unrelated provenance additions but do not '
        'allow overrides of fixed protocol fields.'
    )

    if not isinstance(document, Mapping):
        raise ValueError("revalidation generation identity must be a mapping")
    _reject_c82_markers(document)

    refined = document.get("artifact_type") == REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE
    if not refined and ("refinement" in document or "refinement_rounds" in document):
        raise ValueError("legacy v1 identity cannot be upgraded by attaching refinement fields")
    expected_artifact = (
        REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE if refined else REVALIDATION_GENERATION_ARTIFACT_TYPE
    )
    expected_kind = REVALIDATION_REFINEMENT_GENERATION_KIND if refined else REVALIDATION_GENERATION_KIND
    expected_schema = (
        REVALIDATION_REFINEMENT_GENERATION_SCHEMA_VERSION if refined else REVALIDATION_GENERATION_SCHEMA_VERSION
    )
    expected_protocol = REVALIDATION_REFINEMENT_GENERATION_PROTOCOL if refined else REVALIDATION_GENERATION_PROTOCOL
    expected_ranking = REVALIDATION_REFINED_RANKING if refined else REVALIDATION_RANKING
    expected_strings = {
        "artifact_type": expected_artifact,
        "kind": expected_kind,
        "schema_version": expected_schema,
        "protocol": expected_protocol,
        "candidate_source": REVALIDATION_CANDIDATE_SOURCE,
        "ranking": expected_ranking,
        "candidate_status": "candidates-only-not-physically-certified",
    }
    for field_name, expected in expected_strings.items():
        if document.get(field_name) != expected:
            raise ValueError(f"{field_name} must equal {expected!r} for revalidation protocol")

    candidate_count = document.get("candidate_count")
    if (
        not isinstance(candidate_count, Integral)
        or isinstance(candidate_count, bool)
        or (candidate_count != REVALIDATION_CANDIDATE_COUNT)
    ):
        raise ValueError("candidate_count must be fixed at 8 for revalidation")
    if refined and document.get("candidate_count_per_asset") not in (None, REVALIDATION_CANDIDATE_COUNT):
        raise ValueError("candidate_count_per_asset must be fixed at 8 when present")
    _fixed_float(
        document.get("projection_margin_fraction"),
        expected=REVALIDATION_PROJECTION_MARGIN_FRACTION,
        field_name="projection_margin_fraction",
    )
    _fixed_float(
        document.get("strict_gate_margin_fraction"),
        expected=REVALIDATION_STRICT_MARGIN_FRACTION,
        field_name="strict_gate_margin_fraction",
    )
    for field_name in (
        "candidate_npz_sha256",
        "source_catalog_index_sha256",
        "strict_gate_digest",
        "physics_identity_digest",
    ):
        _validate_sha256(document.get(field_name), field_name)

    strict_gate = _require_mapping(document.get("strict_physics_gate"), "strict_physics_gate")
    if strict_gate.get("name") != "original-strict-1s-physics-gate":
        raise ValueError("strict_physics_gate must identify the original strict 1 s gate")
    _fixed_float(
        strict_gate.get("window_seconds"),
        expected=REVALIDATION_STRICT_PHYSICS_WINDOW_SECONDS,
        field_name="strict_physics_gate.window_seconds",
    )
    _fixed_float(
        strict_gate.get("margin_fraction"),
        expected=REVALIDATION_STRICT_MARGIN_FRACTION,
        field_name="strict_physics_gate.margin_fraction",
    )
    if strict_gate.get("gate_digest") != document.get("strict_gate_digest"):
        raise ValueError("strict_physics_gate.gate_digest must match strict_gate_digest")

    projection = _require_mapping(document.get("projection"), "projection")
    if projection.get("operation") != "clip-to-new-limit-with-margin":
        raise ValueError("projection operation must be clip-to-new-limit-with-margin")
    if projection.get("allowed_joint_class") != "non-thumb-j1-upper":
        raise ValueError("projection is restricted to non-thumb j1 upper joints")
    _fixed_float(
        projection.get("max_delta_rad"),
        expected=REVALIDATION_MAX_PROJECTION_DELTA_RAD,
        field_name="projection.max_delta_rad",
    )

    search = _require_mapping(document.get("search"), "search")
    if search.get("source") != REVALIDATION_CANDIDATE_SOURCE:
        raise ValueError("search source must be original-top8-with-projection")
    if search.get("uses_sobol") is not False:
        raise ValueError("revalidation protocol cannot enable Sobol search")
    if "sobol" in search and search.get("sobol") is not False:
        raise ValueError("revalidation protocol cannot enable Sobol search")
    expected_cem = refined
    if search.get("uses_cem") is not expected_cem:
        raise ValueError(f"revalidation search uses_cem must be {expected_cem}")
    if "cem" in search and search.get("cem") is not expected_cem:
        raise ValueError(f"revalidation search cem must be {expected_cem}")
    if search.get("randomized") is not refined:
        raise ValueError(f"revalidation search randomized must be {refined}")

    publication = _require_mapping(document.get("publication"), "publication")
    if publication.get("candidate_count_per_asset") != REVALIDATION_CANDIDATE_COUNT:
        raise ValueError("publication candidate count must be fixed at 8")
    if publication.get("ranking") != expected_ranking:
        raise ValueError("publication ranking does not match generation protocol")
    if publication.get("physical_certification_required") is not True:
        raise ValueError("revalidation publication must require physical certification")

    if refined:
        _fixed_int(document.get("seed"), expected=REVALIDATION_SCENE_SEED, field_name="seed")
        _fixed_int(
            document.get("refinement_rounds"),
            expected=3,
            field_name="refinement_rounds",
        )
        initial = _require_mapping(document.get("initial"), "initial")
        if initial.get("candidate_source") != REVALIDATION_CANDIDATE_SOURCE:
            raise ValueError("initial candidate source must be original-top8-with-projection")
        _fixed_int(initial.get("candidate_count"), expected=8, field_name="initial.candidate_count")
        _fixed_int(initial.get("seed"), expected=REVALIDATION_SCENE_SEED, field_name="initial.seed")
        if initial.get("uses_sobol") is not False:
            raise ValueError("revalidation initial proposal cannot use Sobol")
        if document.get("refinement_scope") != "assets-with-fewer-than-8-initial-strict-passes":
            raise ValueError("refinement_scope must target only initial strict failures")
        _validate_refinement_identity(document.get("refinement"))
        continuation = _require_mapping(document.get("cem_continuation"), "cem_continuation")
        if continuation.get("initial_elite_count") != 8:
            raise ValueError("initial CEM continuation must use eight parent elites")
        if continuation.get("initial_center_assignment") != "round_robin":
            raise ValueError("initial CEM continuation must use round-robin center assignment")
        if continuation.get("later_elite_count") != 16:
            raise ValueError("later CEM continuation must use sixteen elites")
        if continuation.get("later_center_assignment") != "legacy_cem_proposal_counts_descending":
            raise ValueError("later CEM continuation must preserve legacy center allocation")
        source_hashes = _require_mapping(document.get("source_hashes"), "source_hashes")
        for field_name in (
            "candidate_npz_sha256",
            "source_catalog_index_sha256",
            "strict_gate_digest",
            "physics_identity_digest",
        ):
            _validate_sha256(source_hashes.get(field_name), f"source_hashes.{field_name}")
            if source_hashes.get(field_name) != document.get(field_name):
                raise ValueError(f"source_hashes.{field_name} must match top-level digest")

    return dict(document) if not isinstance(document, dict) else document


def _string_array(array: np.ndarray, *, field_name: str) -> np.ndarray:
    'Normalize fixed-width Unicode/byte arrays to Unicode.'

    if array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field_name} must be a string array")
    if array.dtype.kind == "S":
        values = np.asarray([value.decode("utf-8") for value in array.reshape(-1)], dtype=str)
        return values.reshape(array.shape)
    return np.asarray(array, dtype=str)


def _numeric_array(array: np.ndarray, *, field_name: str) -> np.ndarray:
    'Normalize real numeric arrays and reject NaN/Inf before shape checks.'

    if array.dtype.kind not in {"f", "i", "u"} or array.dtype.kind == "b":
        raise ValueError(f"{field_name} must be a real numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return np.asarray(array, dtype=np.float64)


def _validate_unique_ids(array: np.ndarray, *, field_name: str) -> np.ndarray:
    'Require non-empty string IDs and reject duplicate row keys.'

    if array.ndim != 1:
        raise ValueError(f"{field_name} must have shape [asset_count]")
    values = _string_array(array, field_name=field_name).reshape(-1)
    text_values = np.asarray([str(value) for value in values], dtype=str)
    if any(not value for value in text_values):
        raise ValueError(f"{field_name} must contain non-empty IDs")
    if len(set(text_values.tolist())) != len(text_values):
        raise ValueError(f"{field_name} must not contain duplicate IDs")
    return text_values


def _validate_entry_digests(array: np.ndarray) -> np.ndarray:
    'Validate persistent parent-entry digest format and row provenance.'

    if array.ndim != 1:
        raise ValueError("parent_entry_digest must have shape [asset_count]")
    values = _string_array(array, field_name="parent_entry_digest").reshape(-1)
    for value in values:
        _validate_sha256(str(value), "parent_entry_digest")
    return np.asarray(values, dtype=str)


def _validate_joint_names(array: np.ndarray) -> np.ndarray:
    'Require unique non-empty canonical 16-slot joint names.'

    if array.ndim != 1:
        raise ValueError("canonical_joint_names must have shape [16]")
    values = _string_array(array, field_name="canonical_joint_names").reshape(-1)
    if len(values) != 16 or any(not str(value) for value in values):
        raise ValueError("canonical_joint_names must contain 16 non-empty names")
    if len(set(str(value) for value in values)) != 16:
        raise ValueError("canonical_joint_names must be unique")
    return np.asarray(values, dtype=str)


def _validate_projection(
    original_q: np.ndarray,
    projected_q: np.ndarray,
    joint_names: np.ndarray,
) -> None:
    "Allow upward projection only on non-thumb j1 and enforce this round's maximum-delta anchor."

    delta = projected_q - original_q  # Delta q = projected q - original q, rad.
    changed = delta != 0.0  # Unchanged slots must exactly preserve the parent state.
    allowed_slots = np.asarray(
        [str(name).endswith("_j1") and not str(name).startswith("thumb_") for name in joint_names],
        dtype=np.bool_,
    )
    if np.any(changed[..., ~allowed_slots]):
        raise ValueError("projection may change only non-thumb j1 upper joints")
    allowed_delta = delta[..., allowed_slots]
    if np.any(allowed_delta < 0.0):
        raise ValueError("projection must be upward for expanded upper limits")
    if allowed_delta.size and float(np.max(allowed_delta)) > REVALIDATION_MAX_PROJECTION_DELTA_RAD + 1.0e-12:
        raise ValueError("projection delta exceeds fixed 0.06453248217813295 rad bound")


def _validate_external_masks(
    active_joint_masks: Any,
    *,
    requested_ids: Sequence[str],
    file_ids: np.ndarray,
    file_mask: np.ndarray,
) -> np.ndarray:
    'Align caller active masks to requested new_asset_id order.'

    file_index = {str(asset_id): index for index, asset_id in enumerate(file_ids.tolist())}
    if isinstance(active_joint_masks, Mapping):
        selected: list[np.ndarray] = []
        for asset_id in requested_ids:
            if asset_id not in active_joint_masks:
                raise ValueError(f"active_joint_masks missing asset ID {asset_id!r}")
            candidate = np.asarray(active_joint_masks[asset_id])
            if candidate.shape != (16,) or candidate.dtype.kind != "b":
                raise ValueError(f"active_joint_masks[{asset_id!r}] must be bool shape (16,)")
            selected.append(candidate)
        expected = np.stack(selected, axis=0)
    else:
        expected = np.asarray(active_joint_masks)
        if expected.dtype.kind != "b" or expected.ndim != 2 or expected.shape[1:] != (16,):
            raise ValueError("active_joint_masks must be bool shape [asset_count,16]")
        if expected.shape[0] == len(requested_ids):
            pass  # Same order as caller asset_ids.
        elif expected.shape[0] == file_ids.shape[0]:
            expected = expected[[file_index[asset_id] for asset_id in requested_ids]]
        else:
            raise ValueError("active_joint_masks row count must match requested or file assets")
    actual = file_mask[[file_index[asset_id] for asset_id in requested_ids]]
    if not np.array_equal(expected, actual):
        raise ValueError("active_joint_masks disagree with candidate file")
    return np.asarray(actual, dtype=np.bool_)


def _validate_unique_candidate_pairs(projected_q: np.ndarray, positions: np.ndarray) -> None:
    'Reject padding/repeated candidates; keep all eight projected (q,position) pairs unique.'

    for asset_index in range(projected_q.shape[0]):
        pairs = np.concatenate(
            (
                projected_q[asset_index].reshape(REVALIDATION_CANDIDATE_COUNT, -1),
                positions[asset_index].reshape(REVALIDATION_CANDIDATE_COUNT, -1),
            ),
            axis=1,
        )
        if np.unique(pairs, axis=0).shape[0] != REVALIDATION_CANDIDATE_COUNT:
            raise ValueError(f"asset row {asset_index} contains repeated/padded projected candidates")


@dataclass(frozen=True)
class RevalidationCandidates:
    (
        'Read-only candidates ordered by new_asset_id; not yet PhysX-certified. '
        'projected_q_rad and original_q_rad have shape [A,8,16]; position/orientation '
        'have [A,8,3] and [A,8,4]. A is any non-empty caller subset. There is '
        'deliberately no q_target_rad: this input describes state candidates only and '
        'cannot infer PD preload target.'
    )

    projected_q_rad: np.ndarray
    original_q_rad: np.ndarray
    object_position_h_m: np.ndarray
    object_orientation_h_wxyz: np.ndarray
    parent_asset_id: np.ndarray
    parent_entry_digest: np.ndarray
    new_asset_id: np.ndarray
    active_joint_mask: np.ndarray
    canonical_joint_names: np.ndarray
    candidate_source: str = REVALIDATION_CANDIDATE_SOURCE
    physically_certified: bool = False
    ranking: str = REVALIDATION_RANKING

    def __post_init__(self) -> None:
        'Copy and freeze ndarray so callers cannot mutate validated input views.'

        array_fields = (
            "projected_q_rad",
            "original_q_rad",
            "object_position_h_m",
            "object_orientation_h_wxyz",
            "parent_asset_id",
            "parent_entry_digest",
            "new_asset_id",
            "active_joint_mask",
            "canonical_joint_names",
        )
        for field_name in array_fields:
            value = np.asarray(getattr(self, field_name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)
        if self.candidate_source != REVALIDATION_CANDIDATE_SOURCE:
            raise ValueError("RevalidationCandidates candidate_source is fixed")
        if self.physically_certified is not False:
            raise ValueError("revalidation candidates cannot be marked physically certified")
        if self.ranking != REVALIDATION_RANKING:
            raise ValueError("revalidation candidates must preserve parent rank")

    @property
    def asset_count(self) -> int:
        'Return asset count A in this caller subset.'

        return int(self.new_asset_id.shape[0])

    @property
    def candidate_count(self) -> int:
        'Return fixed candidate count C=8 per asset.'

        return int(self.projected_q_rad.shape[1])


def load_revalidation_candidates(
    path: str | Path,
    *,
    generation_identity: Mapping[str, Any],
    asset_ids: Sequence[str],
    active_joint_masks: Any,
    joint_names: Sequence[str],
) -> RevalidationCandidates:
    (
        'Load and validate the revalidation NPZ, then reorder any non-empty asset '
        'subset by new_asset_id. Require nine arrays: original/projected state, '
        'object position/orientation, active mask, parent/new IDs, parent-entry '
        'digest, and canonical joint names. Hash raw NPZ bytes and validate '
        'generation identity before loading arrays; then check finite values, shapes, '
        'ghosts, projection, upright quaternion, unique candidates, and caller '
        'masks/names. Reject any q_target_rad/target/qtarget array because state and '
        'PD preload semantics must remain separate. Return read-only arrays with '
        'physically_certified=False; Isaac runner still performs the real strict 1 s '
        'gate.'
    )

    identity = validate_revalidation_generation_identity(generation_identity)
    candidate_path = Path(path).expanduser()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)

    actual_file_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()  # Identity of raw NPZ bytes.
    if actual_file_sha != identity["candidate_npz_sha256"]:
        raise ValueError("candidate NPZ SHA-256 does not match generation identity")

    requested_ids = tuple(str(asset_id) for asset_id in asset_ids)
    if not requested_ids:
        raise ValueError("asset_ids must be a non-empty subset")
    if any(not asset_id for asset_id in requested_ids):
        raise ValueError("asset_ids must contain non-empty IDs")
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("asset_ids must not contain duplicate IDs")

    caller_names = np.asarray([str(name) for name in joint_names], dtype=str)
    if caller_names.shape != (16,) or any(not name for name in caller_names):
        raise ValueError("joint_names must contain 16 non-empty names")
    if len(set(caller_names.tolist())) != 16:
        raise ValueError("joint_names must be unique")

    try:
        with np.load(candidate_path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            extra_keys = keys - _REQUIRED_NPZ_KEYS
            if extra_keys:
                target_keys = [
                    key for key in sorted(extra_keys) if any(marker in key.lower() for marker in _TARGET_KEY_MARKERS)
                ]
                if target_keys:
                    raise ValueError(f"candidate NPZ contains ambiguous qtarget arrays: {target_keys}")
                raise ValueError(f"candidate NPZ contains unexpected arrays: {sorted(extra_keys)}")
            missing_keys = _REQUIRED_NPZ_KEYS - keys
            if missing_keys:
                raise ValueError(f"candidate NPZ is missing required arrays: {sorted(missing_keys)}")
            arrays = {key: archive[key].copy() for key in _REQUIRED_NPZ_KEYS}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"cannot read candidate NPZ: {error}") from error

    original_q = _numeric_array(arrays["original_q_rad"], field_name="original_q_rad")
    projected_q = _numeric_array(arrays["projected_q_rad"], field_name="projected_q_rad")
    positions = _numeric_array(arrays["object_position_h_m"], field_name="object_position_h_m")
    orientations = _numeric_array(
        arrays["object_orientation_h_wxyz"],
        field_name="object_orientation_h_wxyz",
    )
    file_mask = arrays["active_joint_mask"]
    if file_mask.dtype.kind != "b":
        raise ValueError("active_joint_mask must be a bool array")
    file_mask = np.asarray(file_mask, dtype=np.bool_)
    parent_ids = _validate_unique_ids(arrays["parent_asset_id"], field_name="parent_asset_id")
    new_ids = _validate_unique_ids(arrays["new_asset_id"], field_name="new_asset_id")
    parent_digests = _validate_entry_digests(arrays["parent_entry_digest"])
    file_joint_names = _validate_joint_names(arrays["canonical_joint_names"])

    # Lock all row/candidate axes here to prevent silent broadcast of wrong shapes later.
    asset_count = new_ids.shape[0]
    expected_count = REVALIDATION_CANDIDATE_COUNT
    if not 1 <= asset_count <= REVALIDATION_MAX_ASSET_COUNT:
        raise ValueError("candidate NPZ asset count must lie in [1,128]")
    if original_q.shape != (asset_count, expected_count, 16):
        raise ValueError("original_q_rad must have shape [asset_count,8,16]")
    if projected_q.shape != original_q.shape:
        raise ValueError("projected_q_rad must have shape [asset_count,8,16]")
    if positions.shape != (asset_count, expected_count, 3):
        raise ValueError("object_position_h_m must have shape [asset_count,8,3]")
    if orientations.shape != (asset_count, expected_count, 4):
        raise ValueError("object_orientation_h_wxyz must have shape [asset_count,8,4]")
    if file_mask.shape != (asset_count, 16):
        raise ValueError("active_joint_mask must have shape [asset_count,16]")
    if parent_ids.shape != (asset_count,) or parent_digests.shape != (asset_count,):
        raise ValueError("asset IDs and parent_entry_digest must have one value per asset")
    if not np.array_equal(file_joint_names, caller_names):
        raise ValueError("canonical joint names disagree with caller binding")
    if not np.all(orientations == np.asarray(REVALIDATION_UPRIGHT_QUATERNION_WXYZ)):
        raise ValueError("object orientation must be exact upright quaternion wxyz=(1,0,0,0)")

    # Inactive canonical slots are ghosts; original and projected state must both be exact zero.
    inactive = np.broadcast_to(~file_mask[:, None, :], original_q.shape)
    if np.any(original_q[inactive]) or np.any(projected_q[inactive]):
        raise ValueError("inactive canonical joint states must be exactly zero ghost")
    _validate_projection(original_q, projected_q, file_joint_names)
    _validate_unique_candidate_pairs(projected_q, positions)

    file_index = {asset_id: index for index, asset_id in enumerate(new_ids.tolist())}
    missing_ids = [asset_id for asset_id in requested_ids if asset_id not in file_index]
    if missing_ids:
        raise ValueError(f"asset_ids missing from candidate NPZ new_asset_id: {missing_ids}")
    selected_indices = np.asarray([file_index[asset_id] for asset_id in requested_ids], dtype=np.int64)
    selected_masks = _validate_external_masks(
        active_joint_masks,
        requested_ids=requested_ids,
        file_ids=new_ids,
        file_mask=file_mask,
    )

    return RevalidationCandidates(
        projected_q_rad=projected_q[selected_indices],
        original_q_rad=original_q[selected_indices],
        object_position_h_m=positions[selected_indices],
        object_orientation_h_wxyz=orientations[selected_indices],
        parent_asset_id=parent_ids[selected_indices],
        parent_entry_digest=parent_digests[selected_indices],
        new_asset_id=new_ids[selected_indices],
        active_joint_mask=selected_masks,
        canonical_joint_names=file_joint_names,
    )


__all__ = [
    "REVALIDATION_CANDIDATE_COUNT",
    "REVALIDATION_CANDIDATE_SOURCE",
    "REVALIDATION_GENERATION_ARTIFACT_TYPE",
    "REVALIDATION_GENERATION_KIND",
    "REVALIDATION_GENERATION_PROTOCOL",
    "REVALIDATION_GENERATION_SCHEMA_VERSION",
    "REVALIDATION_RANKING",
    "REVALIDATION_REFINED_RANKING",
    "REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE",
    "REVALIDATION_REFINEMENT_GENERATION_KIND",
    "REVALIDATION_REFINEMENT_GENERATION_PROTOCOL",
    "REVALIDATION_REFINEMENT_GENERATION_SCHEMA_VERSION",
    "REVALIDATION_SCENE_SEED",
    "RevalidationCandidates",
    "build_revalidation_generation_identity",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "load_revalidation_candidates",
    "stable_digest",
    "validate_revalidation_generation_identity",
]
