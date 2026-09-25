"Parses the exact official right-hand URDFs. It preserves source joint order separately from canonical depth-major order and uses q_home equal to zero."

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ..asset_schema_geometry import HandGeometrySemanticsCfg, geometry_semantics_to_dict
from ..canonical_runtime import CANONICAL_HAND_SCHEMA_V1

Vector3 = tuple[float, float, float]
"Three-value vector; the owning field declares whether it stores meters, radians, or a unit axis."

Matrix4Flat = tuple[float, ...]
"Sixteen row-major values for a 4x4 homogeneous transform."

FrameRole = Literal["base", "palm", "terminal_body", "marker"]
"Semantic role of a base, palm, terminal body, or fixed marker frame."

OFFICIAL_HAND_SEMANTICS_SCHEMA_VERSION = "official-hand-1.0.0"
"Version of the exact official-hand semantics wrapper."

OFFICIAL_MIGRATION_VERSION = "official-urdf-exact-v1"
"Version of the exact official-URDF parser and topology contract."




CANONICAL_FINGER_SLOTS: tuple[str, ...] = tuple(CANONICAL_HAND_SCHEMA_V1.physx_finger_order)
"Canonical finger-slot order used when lowering active joints."

CANONICAL_JOINT_SLOTS: tuple[str, ...] = tuple(CANONICAL_HAND_SCHEMA_V1.joint_names)
"Canonical active-joint slots in depth-major order."


@dataclass(frozen=True)
class OfficialMeshDigestCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    source_uri: str
    resolved_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class OfficialJointSemanticsCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    canonical_name: str
    finger_name: str  # canonical finger slot
    depth: int
    source_name: str
    parent_link: str
    child_link: str
    origin_pos_m: Vector3
    origin_rpy_rad: Vector3
    axis_local: Vector3
    lower_rad: float
    upper_rad: float
    effort_nm: float
    velocity_rad_s: float


@dataclass(frozen=True)
class OfficialFrameSemanticsCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    frame_name: str
    role: FrameRole  # base / palm / terminal_body / marker
    source_link: str
    parent_link: str | None
    source_joint_name: str | None
    origin_pos_m: Vector3
    origin_rpy_rad: Vector3
    pose_root: Matrix4Flat
    finger_name: str | None
    has_collision: bool


@dataclass(frozen=True)
class OfficialHandSemanticsCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    schema_version: str
    migration_version: str  # exact parser identity
    family: Literal["allegro", "leap"]  # source family
    asset_id: str
    source_urdf_path: str
    source_digest: str
    root_link: str
    palm_link: str
    tip_collision_body_names: dict[str, str]  # finger -> terminal body carrying TIP collision
    marker_link_by_finger: dict[str, str]  # finger -> fixed collision-free marker link
    source_active_joint_names: tuple[str, ...]
    canonical_active_joint_names: tuple[str, ...]
    joint_name_by_slot: dict[str, str]  # canonical slot -> official source joint
    joints: tuple[OfficialJointSemanticsCfg, ...]
    frames: tuple[OfficialFrameSemanticsCfg, ...]  # base/palm/terminal/marker frame records
    mesh_digests: tuple[OfficialMeshDigestCfg, ...]
    semantic_R_ha: tuple[float, ...]
    semantic_p_ha: Vector3
    geometry_semantics: HandGeometrySemanticsCfg
    content_hash: str

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.schema_version != OFFICIAL_HAND_SEMANTICS_SCHEMA_VERSION:
            raise ValueError(f"unsupported official schema version: {self.schema_version!r}")
        if self.migration_version != OFFICIAL_MIGRATION_VERSION:
            raise ValueError(f"unsupported official migration version: {self.migration_version!r}")
        if len(self.semantic_R_ha) != 9 or len(self.semantic_p_ha) != 3:
            raise ValueError("semantic {a}->{h} transform must be 3x3 plus 3-vector")
        if tuple(self.geometry_semantics.active_joint_names) != self.canonical_active_joint_names:
            raise ValueError("nested geometry semantics active names must equal canonical source order")
        if self.canonical_active_joint_names != tuple(self.joint_name_by_slot[name] for name in CANONICAL_JOINT_SLOTS):
            raise ValueError("joint_name_by_slot does not close over canonical slot order")
        if tuple(joint.canonical_name for joint in self.joints) != CANONICAL_JOINT_SLOTS:
            raise ValueError("official active joint records must use canonical depth-major order")
        if any(joint.source_name != self.joint_name_by_slot[joint.canonical_name] for joint in self.joints):
            raise ValueError("official active joint records disagree with joint_name_by_slot")
        if tuple(self.geometry_semantics.asset_to_hand_rotation) != tuple(self.semantic_R_ha):
            raise ValueError("nested geometry semantics R_ha disagrees with semantic_R_ha")
        if tuple(self.geometry_semantics.asset_to_hand_translation_m) != tuple(self.semantic_p_ha):
            raise ValueError("nested geometry semantics p_ha disagrees with semantic_p_ha")
        if len(self.content_hash) != 64:
            raise ValueError("official content_hash must be a SHA-256 hexadecimal digest")
        payload = _official_payload(self)
        if _stable_hash(payload) != self.content_hash:
            raise ValueError("official content_hash does not match its payload")

    @property
    def q_home_rad(self) -> tuple[float, ...]:
        "Returns the zero mathematical reference pose in canonical active-joint order."

        return self.geometry_semantics.q_home_rad

    @property
    def canonical_joint_slots(self) -> tuple[str, ...]:
        "Returns the fixed canonical depth-major joint-slot sequence."

        return CANONICAL_JOINT_SLOTS

    def to_dict(self) -> dict[str, Any]:
        'Serializes the typed object as a dictionary.'

        return cast(dict[str, Any], _jsonable(_official_payload(self) | {"content_hash": self.content_hash}))


def load_official_hand_semantics(
    urdf_path: str | Path,
    *,
    asset_id: str | None = None,
    semantic_R_ha: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    semantic_p_ha: Sequence[float] = (0.0, 0.0, 0.0),
) -> OfficialHandSemanticsCfg:
    "Reads the exact right-hand URDF, preserves its source joint order, builds the canonical depth-major mapping, and verifies every referenced mesh hash."


    from ._official_hand_source import (
        _build_active_joint_records,
        _build_frames,
        _build_geometry_semantics,
        _canonical_source_maps,
        _detect_family,
        _find_root_link,
        _palm_collision_link,
        _parse_source_urdf,
        _tip_and_marker_maps,
        _topological_joints,
        _validate_expected_topology,
    )

    resolved_urdf = Path(urdf_path).expanduser().resolve(strict=False)
    if not resolved_urdf.is_file():
        raise FileNotFoundError(f"official URDF does not exist: {resolved_urdf}")
    family = _detect_family(resolved_urdf)
    source = _parse_source_urdf(resolved_urdf)
    _validate_expected_topology(source, family=family)
    canonical_maps = _canonical_source_maps(family)
    ordered_joints = _topological_joints(source, canonical_maps["source_to_slot"])
    semantic_rotation = _float_tuple(semantic_R_ha, length=9, field_name="semantic_R_ha")
    semantic_translation = _vector3(semantic_p_ha, field_name="semantic_p_ha")
    canonical_joint_records = _build_active_joint_records(source, family=family, canonical_maps=canonical_maps)
    geometry = _build_geometry_semantics(
        source,
        family=family,
        ordered_joints=ordered_joints,
        canonical_maps=canonical_maps,
        semantic_rotation=semantic_rotation,
        semantic_translation=semantic_translation,
        asset_id=str(asset_id or source.root_name),
    )
    root_link = _find_root_link(source)
    link_transforms = _forward_link_transforms(geometry, q_canonical_rad=(0.0,) * len(CANONICAL_JOINT_SLOTS))
    frames = _build_frames(
        source,
        family=family,
        root_link=root_link,
        link_transforms=link_transforms,
        canonical_maps=canonical_maps,
    )
    tip_collision_body_names, marker_link_by_finger = _tip_and_marker_maps(family)
    payload = {
        "schema_version": OFFICIAL_HAND_SEMANTICS_SCHEMA_VERSION,
        "migration_version": OFFICIAL_MIGRATION_VERSION,
        "family": family,
        "asset_id": str(asset_id or source.root_name),
        "source_urdf_path": str(resolved_urdf),
        "source_digest": _sha256_file(resolved_urdf),
        "root_link": root_link,
        "palm_link": _palm_collision_link(family),
        "tip_collision_body_names": tip_collision_body_names,
        "marker_link_by_finger": marker_link_by_finger,
        "source_active_joint_names": tuple(joint.name for joint in source.joints if joint.joint_type == "revolute"),
        "canonical_active_joint_names": tuple(canonical_maps["slot_to_source"][slot] for slot in CANONICAL_JOINT_SLOTS),
        "joint_name_by_slot": {slot: canonical_maps["slot_to_source"][slot] for slot in CANONICAL_JOINT_SLOTS},
        "joints": tuple(canonical_joint_records),
        "frames": tuple(frames),
        "mesh_digests": source.mesh_digests,
        "semantic_R_ha": semantic_rotation,
        "semantic_p_ha": semantic_translation,
        "geometry_semantics": geometry,
    }
    content_hash = _stable_hash(payload)
    return OfficialHandSemanticsCfg(**payload, content_hash=content_hash)


def load_official_geometry_semantics(
    urdf_path: str | Path,
    *,
    asset_id: str | None = None,
    semantic_R_ha: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    semantic_p_ha: Sequence[float] = (0.0, 0.0, 0.0),
) -> HandGeometrySemanticsCfg:
    'Loads and validates official geometry semantics.'

    return load_official_hand_semantics(
        urdf_path,
        asset_id=asset_id,
        semantic_R_ha=semantic_R_ha,
        semantic_p_ha=semantic_p_ha,
    ).geometry_semantics


def forward_kinematics(
    description: OfficialHandSemanticsCfg,
    q_canonical_rad: Sequence[float] | None = None,
) -> dict[str, Matrix4Flat]:
    "Computes link transforms from the source joint tree and canonical joint values in radians."

    q = _canonical_q_tuple(q_canonical_rad)
    return _forward_link_transforms(description.geometry_semantics, q_canonical_rad=q)


def frame_transform(
    description: OfficialHandSemanticsCfg,
    frame_name: str,
    q_canonical_rad: Sequence[float] | None = None,
) -> Matrix4Flat:
    "Returns the transform between two named source frames at the mathematical reference pose."

    frame = next((item for item in description.frames if item.frame_name == frame_name), None)
    if frame is None:
        raise KeyError(f"unknown official frame {frame_name!r}")
    return forward_kinematics(description, q_canonical_rad).get(frame.source_link, frame.pose_root)


def canonical_joint_frame_transform(
    description: OfficialHandSemanticsCfg,
    canonical_slot: str,
    q_canonical_rad: Sequence[float] | None = None,
) -> Matrix4Flat:
    "Returns a canonical joint-frame transform without treating source XML order as action order."

    if canonical_slot not in CANONICAL_JOINT_SLOTS:
        raise KeyError(f"unknown canonical joint slot {canonical_slot!r}")
    q = _canonical_q_tuple(q_canonical_rad)
    geometry = description.geometry_semantics
    link_transforms = _forward_link_transforms(geometry, q_canonical_rad=q)
    source_joint_name = description.joint_name_by_slot[canonical_slot]
    joint = next(item for item in geometry.kinematic_joints if item.joint_name == source_joint_name)
    parent = _unflatten4(link_transforms[joint.parent_link])
    origin = _transform_from_rpy(joint.origin_rpy_rad, joint.origin_pos_m)
    return _flatten4(_matmul(parent, origin))


def official_hand_semantics_to_dict(description: OfficialHandSemanticsCfg) -> dict[str, Any]:
    "Serializes official source joints, frames, limits, mesh hashes, and nested geometry semantics."

    return description.to_dict()


def _forward_link_transforms(
    geometry: HandGeometrySemanticsCfg,
    *,
    q_canonical_rad: Sequence[float],
) -> dict[str, Matrix4Flat]:

    q = _canonical_q_tuple(q_canonical_rad)
    index_by_joint = {name: index for index, name in enumerate(geometry.active_joint_names)}


    transforms: dict[str, tuple[tuple[float, ...], ...]] = {geometry.palm_link: _identity4()}
    for joint in geometry.kinematic_joints:
        try:
            parent_transform = transforms[joint.parent_link]
        except KeyError as exc:
            raise ValueError(
                f"geometry tree parent {joint.parent_link!r} not available for {joint.joint_name!r}"
            ) from exc
        origin = _transform_from_rpy(joint.origin_rpy_rad, joint.origin_pos_m)
        if joint.joint_type == "revolute":
            active_index = joint.active_joint_index
            if active_index is None or joint.joint_name not in index_by_joint:
                raise ValueError(f"revolute joint {joint.joint_name!r} lacks canonical active index")
            child_delta = _axis_transform(joint.axis_local, q[active_index])
            child_transform = _matmul(_matmul(parent_transform, origin), child_delta)
        else:
            child_transform = _matmul(parent_transform, origin)
        transforms[joint.child_link] = child_transform
    return {link_name: _flatten4(transform) for link_name, transform in transforms.items()}


def _canonical_q_tuple(q: Sequence[float] | None) -> tuple[float, ...]:

    if q is None:
        return (0.0,) * len(CANONICAL_JOINT_SLOTS)
    values = tuple(float(value) for value in q)
    if len(values) != len(CANONICAL_JOINT_SLOTS):
        raise ValueError(f"canonical q must have length {len(CANONICAL_JOINT_SLOTS)}, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("canonical q must contain finite values")
    return values


def _transform_from_rpy(rpy: Vector3, translation: Vector3) -> tuple[tuple[float, ...], ...]:

    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rotation_x = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    rotation_y = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rotation_z = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    rotation = _matmul3(_matmul3(rotation_z, rotation_y), rotation_x)
    return (
        (*rotation[0], translation[0]),
        (*rotation[1], translation[1]),
        (*rotation[2], translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _axis_transform(axis: Vector3, angle: float) -> tuple[tuple[float, ...], ...]:

    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return (
        (c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s, 0.0),
        (y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s, 0.0),
        (z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _identity4() -> tuple[tuple[float, ...], ...]:

    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _matmul(a: tuple[tuple[float, ...], ...], b: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:

    return tuple(tuple(sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)) for row in range(4))


def _matmul3(a: tuple[tuple[float, ...], ...], b: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:

    return tuple(tuple(sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)) for row in range(3))


def _flatten4(matrix: tuple[tuple[float, ...], ...]) -> Matrix4Flat:

    return tuple(value for row in matrix for value in row)


def _unflatten4(values: Matrix4Flat) -> tuple[tuple[float, ...], ...]:

    if len(values) != 16:
        raise ValueError(f"homogeneous transform must contain 16 values, got {len(values)}")
    return tuple(tuple(values[4 * row + col] for col in range(4)) for row in range(4))


def _vector3(values: Sequence[float], *, field_name: str) -> Vector3:

    if len(values) != 3:
        raise ValueError(f"{field_name} must contain 3 values, got {len(values)}")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain finite values")
    return cast(Vector3, result)


def _float_tuple(values: Sequence[float], *, length: int, field_name: str) -> tuple[float, ...]:

    if len(values) != length:
        raise ValueError(f"{field_name} must contain {length} values, got {len(values)}")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain finite values")
    return result


def _sha256_file(path: Path) -> str:

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:

    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:

    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _official_payload(description: OfficialHandSemanticsCfg) -> dict[str, Any]:

    return {
        "schema_version": description.schema_version,
        "migration_version": description.migration_version,
        "family": description.family,
        "asset_id": description.asset_id,
        "source_urdf_path": description.source_urdf_path,
        "source_digest": description.source_digest,
        "root_link": description.root_link,
        "palm_link": description.palm_link,
        "tip_collision_body_names": description.tip_collision_body_names,
        "marker_link_by_finger": description.marker_link_by_finger,
        "source_active_joint_names": description.source_active_joint_names,
        "canonical_active_joint_names": description.canonical_active_joint_names,
        "joint_name_by_slot": description.joint_name_by_slot,
        "joints": description.joints,
        "frames": description.frames,
        "mesh_digests": description.mesh_digests,
        "semantic_R_ha": description.semantic_R_ha,
        "semantic_p_ha": description.semantic_p_ha,
        "geometry_semantics": geometry_semantics_to_dict(description.geometry_semantics),
    }


__all__ = [
    "CANONICAL_FINGER_SLOTS",
    "CANONICAL_JOINT_SLOTS",
    "OFFICIAL_HAND_SEMANTICS_SCHEMA_VERSION",
    "OFFICIAL_MIGRATION_VERSION",
    "OfficialFrameSemanticsCfg",
    "OfficialHandSemanticsCfg",
    "OfficialJointSemanticsCfg",
    "OfficialMeshDigestCfg",
    "canonical_joint_frame_transform",
    "forward_kinematics",
    "frame_transform",
    "load_official_geometry_semantics",
    "load_official_hand_semantics",
    "official_hand_semantics_to_dict",
]
