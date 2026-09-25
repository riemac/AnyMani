r"""Official pregrasp source/runtime identity and certificate consumer contract.

This module is deliberately independent of Isaac/Kit.  It constructs the
replica-independent identity from a native session audit and verifies that a
certificate is still compatible with the current source URDF, mesh bytes,
calibrated frame, importer lowering, controller, and actual body properties.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from anymani.assets.bank.official_hand import OfficialHandSemanticsCfg
from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE

if TYPE_CHECKING:
    from .official_pregrasp import OfficialPregraspCertificate

OFFICIAL_PREGRASP_SCHEMA_VERSION = "1.0.0"
"""Shared official pregrasp certificate schema version."""
OFFICIAL_PREGRASP_ARTIFACT_TYPE = "anymani.official_native_pregrasp"
"""Shared official pregrasp artifact type."""
OFFICIAL_FINGER_ORDER = ("index", "middle", "ring", "thumb")
"""Canonical TIP role order used by both proposal and identity modules."""
_JOINT_COUNT = 16
_SHA256_LENGTH = 64
_OFFICIAL_PREGRASP_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_PREGRASP_REPO_ROOT = _OFFICIAL_PREGRASP_PACKAGE_ROOT.parents[1]


def _stable_digest(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-safe identity mapping deterministically."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a source or mesh file without writing to it."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    """Validate a persisted lowercase SHA-256 digest."""

    parsed = str(value)
    if len(parsed) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in parsed):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256")
    return parsed


@dataclass(frozen=True)
class OfficialPregraspIdentity:
    'Source/physics/controller/generation identity for official pregrasp certificates.'

    family: Literal["allegro", "leap"]
    asset_id: str
    source_urdf_sha256: str
    source_content_hash: str
    mesh_digests: tuple[Mapping[str, Any], ...]
    object_asset_id: str
    object_asset_sha256: str
    object_scale: float
    physics_identity: Mapping[str, Any]
    controller_identity: Mapping[str, Any]
    generation_identity: Mapping[str, Any]
    source_implementation_identity: Mapping[str, Any]
    identity_digest: str

    def __post_init__(self) -> None:
        'Recompute identity digest and reject mismatched source/cube/protocol.'

        _validate_sha256(self.source_urdf_sha256, "source_urdf_sha256")
        _validate_sha256(self.source_content_hash, "source_content_hash")
        _validate_sha256(self.object_asset_sha256, "object_asset_sha256")
        _validate_sha256(self.identity_digest, "identity_digest")
        if self.family not in {"allegro", "leap"}:
            raise ValueError(f"unsupported official pregrasp family: {self.family!r}")
        if (
            not self.asset_id
            or not self.object_asset_id
            or not math.isfinite(self.object_scale)
            or self.object_scale <= 0
        ):
            raise ValueError("official pregrasp identity requires non-empty IDs and positive scale")
        expected = _stable_digest(self._payload())
        if expected != self.identity_digest:
            raise ValueError("official pregrasp identity digest mismatch")

    def _payload(self) -> dict[str, Any]:
        'Return the JSON-safe payload covered by the identity digest.'

        return {
            "schema_version": OFFICIAL_PREGRASP_SCHEMA_VERSION,
            "family": self.family,
            "asset_id": self.asset_id,
            "source_urdf_sha256": self.source_urdf_sha256,
            "source_content_hash": self.source_content_hash,
            "mesh_digests": [dict(item) for item in self.mesh_digests],
            "object_asset_id": self.object_asset_id,
            "object_asset_sha256": self.object_asset_sha256,
            "object_scale": self.object_scale,
            "physics_identity": dict(self.physics_identity),
            "controller_identity": dict(self.controller_identity),
            "generation_identity": dict(self.generation_identity),
            "source_implementation_identity": dict(self.source_implementation_identity),
        }

    def to_dict(self) -> dict[str, Any]:
        'Return JSON-safe identity.'

        return self._payload() | {"identity_digest": self.identity_digest}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OfficialPregraspIdentity:
        'Restore identity from JSON and recompute its digest.'

        return cls(
            family=str(payload["family"]),  # type: ignore[arg-type]
            asset_id=str(payload["asset_id"]),
            source_urdf_sha256=str(payload["source_urdf_sha256"]),
            source_content_hash=str(payload["source_content_hash"]),
            mesh_digests=tuple(dict(item) for item in payload["mesh_digests"]),
            object_asset_id=str(payload["object_asset_id"]),
            object_asset_sha256=str(payload["object_asset_sha256"]),
            object_scale=float(payload["object_scale"]),
            physics_identity=dict(payload["physics_identity"]),
            controller_identity=dict(payload["controller_identity"]),
            generation_identity=dict(payload["generation_identity"]),
            source_implementation_identity=dict(payload["source_implementation_identity"]),
            identity_digest=str(payload["identity_digest"]),
        )


def build_official_pregrasp_identity(
    description: OfficialHandSemanticsCfg,
    *,
    object_asset_id: str,
    object_asset_sha256: str,
    object_scale: float,
    physics_identity: Mapping[str, Any],
    controller_identity: Mapping[str, Any],
    generation_identity: Mapping[str, Any],
    source_implementation_identity: Mapping[str, Any],
) -> OfficialPregraspIdentity:
    'Bind official native source, cube, physics/controller, and generation identity.'

    payload = {
        "family": description.family,
        "asset_id": description.asset_id,
        "source_urdf_sha256": description.source_digest,
        "source_content_hash": description.content_hash,
        "mesh_digests": _mesh_digest_records(description),
        "object_asset_id": object_asset_id,
        "object_asset_sha256": _validate_sha256(object_asset_sha256, "object_asset_sha256"),
        "object_scale": float(object_scale),
        "physics_identity": dict(physics_identity),
        "controller_identity": dict(controller_identity),
        "generation_identity": dict(generation_identity),
        "source_implementation_identity": dict(source_implementation_identity),
    }
    digest = _stable_digest({"schema_version": OFFICIAL_PREGRASP_SCHEMA_VERSION, **payload})
    return OfficialPregraspIdentity(**payload, identity_digest=digest)


def _required_audit_mapping(session_audit: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    """Return one required JSON-safe audit mapping or fail closed."""

    value = session_audit.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"official session audit requires mapping {field_name!r}")
    return value


def _mesh_digest_records(description: OfficialHandSemanticsCfg) -> list[dict[str, Any]]:
    """Return mesh provenance keyed by source URI/content, independent of checkout paths."""

    return [
        {
            "source_uri": str(mesh.source_uri),
            "sha256": str(mesh.sha256),
            "size_bytes": int(mesh.size_bytes),
        }
        for mesh in description.mesh_digests
    ]


def _normalise_mesh_digest_payload(value: Any) -> list[dict[str, Any]]:
    """Drop resolved filesystem paths before comparing source mesh provenance."""

    if not isinstance(value, (list, tuple)):
        raise ValueError("official mesh digest payload must be a sequence")
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("official mesh digest row must be a mapping")
        result.append(
            {
                "source_uri": str(item["source_uri"]),
                "sha256": str(item["sha256"]),
                "size_bytes": int(item["size_bytes"]),
            }
        )
    return result


def _identity_values_equal(left: Any, right: Any, *, tolerance: float = 1.0e-8) -> bool:
    """Compare nested identity values, allowing only measured floating-point noise."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_identity_values_equal(left[key], right[key], tolerance=tolerance) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_identity_values_equal(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True))
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def _first_identity_difference(left: Any, right: Any, path: str = "$") -> str | None:
    """Return a concise nested identity mismatch location."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                return f"{path}.{key}"
            difference = _first_identity_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return f"{path}.length"
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            difference = _first_identity_difference(a, b, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if _identity_values_equal(left, right) else path


def _first_replica_numeric(value: Any, field_name: str) -> list[Any]:
    """Validate equal replica rows and return only the first row for identity."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or not np.isfinite(array).all():
        raise ValueError(f"official session audit {field_name!r} must be finite numeric data")
    if array.ndim == 1:
        return array.tolist()
    first = array[0]
    for replica in array[1:]:
        if replica.shape != first.shape or not np.allclose(replica, first, rtol=1.0e-8, atol=1.0e-8):
            raise ValueError(f"official session audit {field_name!r} differs across replicas")
    return first.tolist()


def _normalise_import_source_audit(import_source: Mapping[str, Any]) -> dict[str, Any]:
    """Keep import-source identity while excluding cache paths and generated filenames."""

    required = ("adapter", "identity", "frame_lowering", "root_transform")
    for field_name in required:
        if field_name not in import_source:
            raise ValueError(f"official import-source audit lacks {field_name!r}")
    frame_lowering = import_source["frame_lowering"]
    root_transform = import_source["root_transform"]
    if not isinstance(frame_lowering, Mapping) or not isinstance(root_transform, Mapping):
        raise ValueError("official import-source frame/root audit must be mappings")
    return {
        "adapter": str(import_source["adapter"]),
        "identity": str(import_source["identity"]),
        "frame_lowering": {
            "version": str(frame_lowering["version"]),
            "original_root_link": str(frame_lowering["original_root_link"]),
            "import_root_link": str(frame_lowering["import_root_link"]),
            "root_promoted": bool(frame_lowering["root_promoted"]),
            "removed_reference_frames": [str(value) for value in frame_lowering["removed_reference_frames"]],
            "removed_reference_joints": [str(value) for value in frame_lowering["removed_reference_joints"]],
        },
        "root_transform": {
            "original_root_link": str(root_transform["original_root_link"]),
            "import_root_link": str(root_transform["import_root_link"]),
            "root_promoted": bool(root_transform["root_promoted"]),
            "T_original_root_from_import_root": [
                float(value) for value in root_transform["T_original_root_from_import_root"]
            ],
        },
        "physical_subtree_unchanged": bool(import_source["physical_subtree_unchanged"]),
        "geometry_and_material_assignment_unchanged": bool(import_source["geometry_and_material_assignment_unchanged"]),
    }


def _source_frame_identity(description: OfficialHandSemanticsCfg, spawn: Mapping[str, Any]) -> dict[str, Any]:
    """Extract calibrated frame evidence from the actual native spawn audit."""

    root_frame = spawn.get("root_frame")
    if not isinstance(root_frame, Mapping):
        raise ValueError("official spawn audit requires root_frame")
    expected = {
        "R_wh": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "semantic_R_ha": [float(value) for value in description.semantic_R_ha],
        "semantic_p_ha_m": [float(value) for value in description.semantic_p_ha],
    }
    for field_name, values in expected.items():
        actual = root_frame.get(field_name)
        if not _identity_values_equal(actual, values):
            raise ValueError(f"official native root frame disagrees with description: {field_name}")
    return {
        "R_wh": list(expected["R_wh"]),
        "semantic_R_ha": list(expected["semantic_R_ha"]),
        "semantic_p_ha_m": list(expected["semantic_p_ha_m"]),
        "root_position_h_m": [float(value) for value in root_frame["root_position_h_m"]],
        "root_position_a_m": [float(value) for value in root_frame["root_position_a_m"]],
        "root_quaternion_wxyz": [float(value) for value in root_frame["root_quaternion_wxyz"]],
    }


def build_official_physics_identity(
    description: OfficialHandSemanticsCfg,
    session_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the replica-independent native physics identity from actual runtime audit.

    The identity deliberately omits replica count, USD cache paths and render settings. It
    retains the calibrated source frame, actual imported body names, first-replica body
    mass/inertia rows (after checking every replica is equal), importer collision semantics,
    frame-lowering adapter evidence, and observed DexCube properties.
    """

    spawn = _required_audit_mapping(session_audit, "spawn")
    if str(spawn.get("family")) != description.family or str(spawn.get("asset_id")) != description.asset_id:
        raise ValueError("official spawn audit family/asset_id disagrees with native description")
    source_urdf = _required_audit_mapping(spawn, "source_urdf")
    if str(source_urdf.get("sha256")) != description.source_digest:
        raise ValueError("official spawn audit URDF hash disagrees with native description")
    if str(spawn.get("description_content_hash")) != description.content_hash:
        raise ValueError("official spawn audit description hash disagrees with native description")
    mesh_digests = spawn.get("mesh_digests")
    expected_mesh_digests = _mesh_digest_records(description)
    if not _identity_values_equal(_normalise_mesh_digest_payload(mesh_digests), expected_mesh_digests):
        raise ValueError("official spawn audit mesh digests disagree with native description")
    source_links = spawn.get("source_link_to_usd_link")
    if not isinstance(source_links, Mapping):
        raise ValueError("official spawn audit requires source_link_to_usd_link")
    source_links_json = {str(key): str(value) for key, value in source_links.items()}
    importer = _required_audit_mapping(spawn, "importer")
    importer_json = {}
    for field_name in (
        "fix_base",
        "merge_fixed_joints",
        "force_usd_conversion",
        "collision_from_visuals",
        "collider_type",
        "self_collision",
        "activate_contact_sensors",
    ):
        if field_name not in importer:
            raise ValueError(f"official importer audit lacks {field_name!r}")
        importer_json[field_name] = importer[field_name]
    import_source = _required_audit_mapping(spawn, "import_source")
    import_source_json = _normalise_import_source_audit(import_source)
    body_names_raw = session_audit.get("body_names")
    if not isinstance(body_names_raw, (list, tuple)) or not body_names_raw:
        raise ValueError("official session audit requires nonempty actual body_names")
    body_names = [str(value) for value in body_names_raw]
    if len(body_names) != len(set(body_names)):
        raise ValueError("official actual body_names must be unique")
    removed_source_links = set(import_source_json["frame_lowering"]["removed_reference_frames"])
    expected_body_names = {
        usd_name for source_name, usd_name in source_links_json.items() if source_name not in removed_source_links
    }
    if set(body_names) != expected_body_names:
        raise ValueError("official actual body_names disagree with source frame-lowering mapping")
    body_mass = _first_replica_numeric(session_audit.get("hand_body_mass_kg"), "hand_body_mass_kg")
    body_inertia = _first_replica_numeric(session_audit.get("hand_body_inertia_kg_m2"), "hand_body_inertia_kg_m2")
    if len(body_mass) != len(body_names) or len(body_inertia) != len(body_names):
        raise ValueError("official body names/mass/inertia rows have inconsistent lengths")
    object_mass = _first_replica_numeric(session_audit.get("object_mass_kg"), "object_mass_kg")
    physics_dt = float(session_audit["physics_dt_s"])
    policy_dt = float(session_audit["policy_dt_s"])
    if not math.isclose(physics_dt, 1.0 / 120.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("official physics dt must be exactly 1/120 s")
    if not math.isclose(policy_dt, 0.05, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("official policy dt must be exactly 0.05 s")
    tip_approximations_raw = session_audit.get("tip_collision_approximations")
    if not isinstance(tip_approximations_raw, (list, tuple)):
        raise ValueError("official session audit requires tip_collision_approximations")
    tip_approximations = [
        {"finger": str(item["finger"]), "approximation": str(item["approximation"])}
        for item in tip_approximations_raw
        if isinstance(item, Mapping)
    ]
    if len(tip_approximations) != 4 or {item["finger"] for item in tip_approximations} != set(OFFICIAL_FINGER_ORDER):
        raise ValueError("official session audit must verify all four TIP approximations")
    if any(item["approximation"] != "convexHull" for item in tip_approximations):
        raise ValueError("official TIP collider approximation is not convexHull")
    cube_sha256 = str(session_audit.get("cube_sha256"))
    _validate_sha256(cube_sha256, "session_audit.cube_sha256")
    cube_scale_raw = session_audit.get("cube_scale")
    if not isinstance(cube_scale_raw, (float, int)) or isinstance(cube_scale_raw, bool):
        raise ValueError("official session audit cube_scale must be numeric")
    cube_scale = float(cube_scale_raw)
    if not math.isfinite(cube_scale) or cube_scale <= 0.0:
        raise ValueError("official session audit cube_scale must be positive")
    return {
        "schema": "official-native-physics-v2",
        "source_frame": _source_frame_identity(description, spawn),
        "source_link_to_usd_link": source_links_json,
        "importer": importer_json,
        "import_source": import_source_json,
        "physics_dt_s": physics_dt,
        "policy_dt_s": policy_dt,
        "physics_substeps_per_policy": 6,
        "friction": {"static": 1.0, "dynamic": 1.0, "restitution": 0.0, "combine_mode": "average"},
        "solver": {"position_iterations": 8, "velocity_iterations": 0, "bounce_threshold_velocity_m_s": 0.2},
        "object": {
            "asset_id": "DexCube",
            "sha256": cube_sha256,
            "scale": cube_scale,
            "mass_kg_first_replica": object_mass,
        },
        "actual_body_names": body_names,
        "actual_body_mass_kg_first_replica": body_mass,
        "actual_body_inertia_kg_m2_first_replica": body_inertia,
        "tip_collision_approximations": sorted(tip_approximations, key=lambda item: item["finger"]),
        "structural_collision_filter": "bidirectional_palm_and_same_finger_filter; cross_finger_kept",
    }


def build_official_controller_identity(
    description: OfficialHandSemanticsCfg,
    session_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the controller/source actuator identity from actual native audit."""

    from anymani.robots.official_hand_spawn import source_joint_to_usd_joint_name

    spawn = _required_audit_mapping(session_audit, "spawn")
    raw_actuator = spawn.get("actuator")
    if not isinstance(raw_actuator, Mapping) or not raw_actuator:
        raise ValueError("official spawn audit requires nonempty actuator data")
    raw_joints = spawn.get("joints")
    if not isinstance(raw_joints, (list, tuple)) or len(raw_joints) != _JOINT_COUNT:
        raise ValueError("official spawn audit requires exactly 16 source actuator joint rows")
    actual_joint_names = session_audit.get("joint_names")
    canonical_indices = session_audit.get("canonical_joint_indices")
    if not isinstance(actual_joint_names, (list, tuple)) or not isinstance(canonical_indices, (list, tuple)):
        raise ValueError("official session audit requires actual joint names and canonical indices")
    if len(actual_joint_names) != _JOINT_COUNT or len(canonical_indices) != _JOINT_COUNT:
        raise ValueError("official actual joint/index mapping must have length 16")
    expected_rows: list[dict[str, Any]] = []
    for joint, raw in zip(description.joints, raw_joints, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("official actuator joint row must be a mapping")
        expected_usd_name = source_joint_to_usd_joint_name(description, joint.source_name)
        checks = {
            "canonical_slot": joint.canonical_name,
            "source_joint_name": joint.source_name,
            "usd_joint_name": expected_usd_name,
            "source_lower_rad": float(joint.lower_rad),
            "source_upper_rad": float(joint.upper_rad),
            "source_effort_cap_nm": float(joint.effort_nm),
            "source_velocity_cap_rad_s": float(joint.velocity_rad_s),
        }
        for field_name, expected in checks.items():
            if not _identity_values_equal(raw.get(field_name), expected):
                raise ValueError(f"official source actuator row disagrees with description: {field_name}")
        expected_rows.append({str(key): raw[key] for key in raw})
    actuator = {str(key): raw_actuator[key] for key in raw_actuator}
    if str(actuator.get("type")) != "implicit_pd":
        raise ValueError("official source actuator type must be implicit_pd")
    for field_name, expected in (("stiffness", 3.0), ("damping", 0.1), ("armature", 0.001)):
        if not _identity_values_equal(actuator.get(field_name), expected):
            raise ValueError(f"official actuator {field_name} disagrees with native controller contract")
    actual_names = [str(value) for value in actual_joint_names]
    indices = [int(value) for value in canonical_indices]
    if sorted(indices) != list(range(_JOINT_COUNT)):
        raise ValueError("official canonical joint indices must be a permutation of 0..15")
    expected_names = [source_joint_to_usd_joint_name(description, joint.source_name) for joint in description.joints]
    if set(actual_names) != set(expected_names) or any(
        actual_names[index] != expected_names[canonical_index] for canonical_index, index in enumerate(indices)
    ):
        raise ValueError("official actual joint names do not close over canonical source mapping")
    return {
        "schema": "official-native-controller-v2",
        "type": str(actuator["type"]),
        "stiffness": float(actuator["stiffness"]),
        "damping": float(actuator["damping"]),
        "armature": float(actuator["armature"]),
        "q0_equals_u0": True,
        "q_home_rad": [float(value) for value in description.q_home_rad],
        "initial_joint_velocity_rad_s": 0.0,
        "zero_action_hold": True,
        "action_scale_per_policy_step": 1.0 / 24.0,
        "joint_mapping": expected_rows,
        "actual_joint_names": actual_names,
        "canonical_joint_indices": indices,
        "source_actuator": actuator,
    }


def build_official_source_implementation_identity() -> dict[str, Any]:
    """Hash the native pregrasp/runtime implementation, including frame lowering."""

    package_root = _OFFICIAL_PREGRASP_PACKAGE_ROOT
    repo_root = _OFFICIAL_PREGRASP_REPO_ROOT
    paths: dict[str, Path] = {
        "official_pregrasp": package_root / "anymani" / "pregrasp" / "official_pregrasp.py",
        "official_runtime": package_root / "anymani" / "tasks" / "hetero" / "official_runtime.py",
        "official_contact": package_root / "anymani" / "tasks" / "hetero" / "official_contact.py",
        "official_state": package_root / "anymani" / "tasks" / "hetero" / "official_state.py",
        "official_hand_spawn": package_root / "anymani" / "robots" / "official_hand_spawn.py",
        "official_proxy": package_root / "anymani" / "assets" / "official_proxy.py",
        "prepare_script": package_root / "anymani" / "pregrasp" / "scripts" / "prepare_official_pregrasp.py",
    }
    import_adapter = package_root / "anymani" / "robots" / "_official_import_source.py"
    if not import_adapter.is_file():
        raise FileNotFoundError("official import-source lowering module is required for native certificates")
    paths["official_import_source"] = import_adapter
    result: dict[str, Any] = {}
    for name, path in paths.items():
        resolved = path.resolve(strict=True)
        result[name] = {
            "path": str(resolved.relative_to(repo_root)),
            "sha256": _sha256_file(resolved),
        }
    return result


def validate_official_pregrasp_for_session(
    certificate_path: Path,
    description: OfficialHandSemanticsCfg,
    session_audit: Mapping[str, Any],
) -> OfficialPregraspCertificate:
    """Validate a certificate against the current native source and runtime audit.

    This is the public fail-closed consumer contract for evaluation. It re-runs the
    certificate loader, then compares source URDF/mesh/frame identity, replica-independent
    physics/controller identities, actual body mass/inertia/name rows, cube identity and
    implementation hashes. Generation identity is read from the certificate and is never
    rebuilt from an evaluation seed.
    """

    from .official_pregrasp import load_official_pregrasp_certificate

    certificate = load_official_pregrasp_certificate(certificate_path)
    identity = certificate.identity
    if identity.family != description.family or identity.asset_id != description.asset_id:
        raise ValueError("official pregrasp certificate family/asset_id disagrees with current description")
    source_path = Path(description.source_urdf_path).expanduser().resolve(strict=True)
    if _sha256_file(source_path) != description.source_digest:
        raise ValueError("current native URDF bytes disagree with description")
    if identity.source_urdf_sha256 != description.source_digest:
        raise ValueError("current native URDF hash disagrees with certificate")
    if identity.source_content_hash != description.content_hash:
        raise ValueError("current native description content hash disagrees with certificate")
    current_mesh_digests = _mesh_digest_records(description)
    if not _identity_values_equal(
        _normalise_mesh_digest_payload(identity.mesh_digests),
        current_mesh_digests,
    ):
        raise ValueError("current native mesh digests disagree with certificate")
    for mesh in description.mesh_digests:
        mesh_path = Path(mesh.resolved_path).expanduser().resolve(strict=True)
        if _sha256_file(mesh_path) != mesh.sha256:
            raise ValueError(f"current native mesh bytes disagree with description: {mesh_path}")
    audit_cube_sha256 = str(session_audit.get("cube_sha256"))
    if identity.object_asset_id != "DexCube" or identity.object_asset_sha256 != audit_cube_sha256:
        raise ValueError("current DexCube identity disagrees with certificate")
    if not _identity_values_equal(identity.object_scale, session_audit.get("cube_scale")):
        raise ValueError("current DexCube scale disagrees with certificate")
    current_physics = build_official_physics_identity(description, session_audit)
    difference = _first_identity_difference(identity.physics_identity, current_physics)
    if difference is not None:
        raise ValueError(f"official physics identity mismatch at {difference}")
    current_controller = build_official_controller_identity(description, session_audit)
    difference = _first_identity_difference(identity.controller_identity, current_controller)
    if difference is not None:
        raise ValueError(f"official controller identity mismatch at {difference}")
    current_source = build_official_source_implementation_identity()
    difference = _first_identity_difference(identity.source_implementation_identity, current_source)
    if difference is not None:
        raise ValueError(f"official source implementation identity mismatch at {difference}")
    generation = identity.generation_identity
    if generation.get("algorithm") != "official-native-strict-v1":
        raise ValueError("official certificate uses an unknown generation algorithm")
    if generation.get("strict_gate_digest") != MVP80_STRICT_GOOD_PREGRASP_GATE.digest:
        raise ValueError("official certificate strict gate digest disagrees with current gate")
    return certificate


__all__ = [
    "OFFICIAL_FINGER_ORDER",
    "OFFICIAL_PREGRASP_ARTIFACT_TYPE",
    "OFFICIAL_PREGRASP_SCHEMA_VERSION",
    "OfficialPregraspIdentity",
    "build_official_controller_identity",
    "build_official_physics_identity",
    "build_official_pregrasp_identity",
    "build_official_source_implementation_identity",
    "validate_official_pregrasp_for_session",
]
