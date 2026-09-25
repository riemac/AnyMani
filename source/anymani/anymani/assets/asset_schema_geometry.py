"Defines versioned static geometry semantics, the complete joint tree, reference pose, collision owners, and anchors. Frame positions are in meters and angles in radians."

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias, cast

from .asset_schema_core import CollisionGeometryCfg, JointLimitCfg, PoseCfg
from .asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg

SEMANTICS_SCHEMA_VERSION = "1.0.0"
"Version of the serialized typed-geometry contract."

GENERATED_MIGRATION_VERSION = "generated-handcfg-v1"
"Version of the generated-sidecar geometry migration."

OwnerRole: TypeAlias = Literal["palm", "joint", "tip"]
"Allowed collision-owner class: palm, joint, or tip."

Vector3: TypeAlias = tuple[float, float, float]
Matrix3Flat: TypeAlias = tuple[float, float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class CollisionComponentSemanticsCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    component_id: str
    owner_id: str
    carrier_link: str
    collision_index: int
    collision_name: str | None
    geometry_kind: str  # box/cylinder/elliptic_cylinder/sphere/mesh
    geometry_payload: dict[str, Any]
    origin_pos_m: Vector3
    origin_rpy_rad: Vector3
    source_joint_name: str | None


@dataclass(frozen=True)
class KinematicJointSemanticsCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    joint_name: str
    joint_type: Literal["fixed", "revolute"]
    parent_link: str
    child_link: str
    origin_pos_m: Vector3
    origin_rpy_rad: Vector3
    axis_local: Vector3
    active_joint_index: int | None


@dataclass(frozen=True)
class GeometryOwnerSemanticsCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    owner_id: str
    owner_index: int
    role: OwnerRole
    parent_owner_id: str | None
    finger_name: str | None
    joint_name: str | None
    reference_link: str
    component_ids: tuple[str, ...]


@dataclass(frozen=True)
class AnchorSeedSemanticsCfg:
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    seed_id: str
    finger_name: str
    first_active_joint_name: str
    support_owner_id: str
    position_a_m: Vector3
    rotation_a: Matrix3Flat


@dataclass(frozen=True)
class HandGeometrySemanticsCfg:
    "Typed static geometry contract with an asset-to-hand transform, a complete joint tree, q_home, limits, collision owners, and semantic anchors."

    schema_version: str
    migration_version: str
    source_kind: Literal["generated", "official"]
    asset_id: str
    asset_name: str
    topology_key: str | None
    family: str
    handedness: Literal["left", "right", "unknown"]
    units: dict[str, str]
    asset_to_hand_rotation: Matrix3Flat
    asset_to_hand_translation_m: Vector3
    palm_link: str
    palm_origin_pos_m: Vector3
    palm_origin_rpy_rad: Vector3
    kinematic_joints: tuple[KinematicJointSemanticsCfg, ...]
    active_joint_names: tuple[str, ...]
    q_home_rad: tuple[float, ...]
    joint_limits_rad: tuple[tuple[float, float], ...]
    owners: tuple[GeometryOwnerSemanticsCfg, ...]
    components: tuple[CollisionComponentSemanticsCfg, ...]
    anchor_seeds: tuple[AnchorSeedSemanticsCfg, ...]
    content_hash: str

    def __post_init__(self) -> None:
        "Normalizes typed inputs and enforces the class invariants."

        if self.schema_version != SEMANTICS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported geometry semantics schema_version={self.schema_version!r}; "
                f"expected {SEMANTICS_SCHEMA_VERSION!r}"
            )
        if self.source_kind not in {"generated", "official"}:
            raise ValueError(f"invalid geometry semantics source_kind={self.source_kind!r}")
        if self.handedness not in {"left", "right", "unknown"}:
            raise ValueError(f"invalid handedness={self.handedness!r}")

        joint_count = len(self.active_joint_names)
        if len(set(self.active_joint_names)) != joint_count:
            raise ValueError("active_joint_names must be unique")
        if len(self.q_home_rad) != joint_count or len(self.joint_limits_rad) != joint_count:
            raise ValueError("q_home_rad and joint_limits_rad must align with active_joint_names")
        if self.units != {"length": "m", "angle": "rad"}:
            raise ValueError(f"geometry semantics require SI units, got {self.units}")
        if len(self.asset_to_hand_rotation) != 9 or len(self.asset_to_hand_translation_m) != 3:
            raise ValueError("asset-to-hand transform must contain a 3x3 rotation and 3D translation")

        kinematic_names = [joint.joint_name for joint in self.kinematic_joints]
        if len(kinematic_names) != len(set(kinematic_names)):
            raise ValueError("kinematic joint names must be unique")
        active_kinematic = tuple(
            joint for joint in self.kinematic_joints if joint.joint_type == "revolute"
        )
        if tuple(joint.joint_name for joint in active_kinematic) != self.active_joint_names:
            raise ValueError("revolute kinematic joint order must equal active_joint_names")
        if tuple(joint.active_joint_index for joint in active_kinematic) != tuple(range(joint_count)):
            raise ValueError("revolute active_joint_index must be contiguous and canonical")
        if any(joint.active_joint_index is not None for joint in self.kinematic_joints if joint.joint_type == "fixed"):
            raise ValueError("fixed kinematic joints must not have active_joint_index")

        known_links = {self.palm_link, *(joint.child_link for joint in self.kinematic_joints)}
        available_parents = {self.palm_link}
        for joint in self.kinematic_joints:
            if joint.parent_link not in available_parents:
                raise ValueError(
                    f"kinematic joint '{joint.joint_name}' parent '{joint.parent_link}' is unavailable"
                )
            available_parents.add(joint.child_link)

        for joint_name, limits in zip(self.active_joint_names, self.joint_limits_rad):
            lower, upper = limits
            if not lower < upper:
                raise ValueError(f"joint '{joint_name}' has invalid limits {limits}")



        owner_ids = [owner.owner_id for owner in self.owners]
        component_ids = [component.component_id for component in self.components]
        if len(owner_ids) != len(set(owner_ids)):
            raise ValueError("owner IDs must be unique")
        if [owner.owner_index for owner in self.owners] != list(range(len(self.owners))):
            raise ValueError("owner_index must be contiguous and match owners order")
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("collision component IDs must be unique")

        known_owners = set(owner_ids)
        assigned_components: list[str] = []
        for owner in self.owners:
            if owner.parent_owner_id is not None and owner.parent_owner_id not in known_owners:
                raise ValueError(f"owner '{owner.owner_id}' references unknown parent '{owner.parent_owner_id}'")
            assigned_components.extend(owner.component_ids)
        if len(assigned_components) != len(set(assigned_components)):
            raise ValueError("a collision component is assigned to more than one owner")
        if set(assigned_components) != set(component_ids):
            raise ValueError("owner component coverage must equal the complete collision component set")
        for component in self.components:
            if component.owner_id not in known_owners:
                raise ValueError(f"component '{component.component_id}' references unknown owner")
            if component.component_id not in self.owners[owner_ids.index(component.owner_id)].component_ids:
                raise ValueError(f"component '{component.component_id}' owner back-reference is inconsistent")
            if component.carrier_link not in known_links:
                raise ValueError(f"component '{component.component_id}' references unknown carrier link")
        for owner in self.owners:
            if owner.reference_link not in known_links:
                raise ValueError(f"owner '{owner.owner_id}' references unknown reference link")

        for seed in self.anchor_seeds:
            if seed.support_owner_id not in known_owners:
                raise ValueError(f"anchor seed '{seed.seed_id}' references unknown support owner")
            support = self.owners[owner_ids.index(seed.support_owner_id)]
            if support.role != "palm":
                raise ValueError(f"anchor seed '{seed.seed_id}' support owner must be PALM")
        if len(self.content_hash) != 64:
            raise ValueError("content_hash must be a SHA-256 hexadecimal digest")
        hash_payload = asdict(self)
        declared_hash = hash_payload.pop("content_hash")
        if _content_hash(hash_payload) != declared_hash:
            raise ValueError("geometry semantics content_hash does not match its payload")


@dataclass
class _OwnerBuilder:
    "Private intermediate record used by the asset_schema_geometry.py implementation."

    owner_id: str
    role: OwnerRole
    parent_owner_id: str | None
    finger_name: str | None
    joint_name: str | None
    reference_link: str
    component_ids: list[str]


def derive_generated_geometry_semantics(
    hand: HandCfg,
    *,
    asset_id: str,
    topology_key: str | None = None,
    q_home_rad: Mapping[str, float] | None = None,
    asset_to_hand_rotation: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    asset_to_hand_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> HandGeometrySemanticsCfg:
    'Derives generated geometry semantics.'

    palm = cast(PalmCfg, hand.palm)
    active_joints = tuple(joint for joint in hand.iter_joints() if joint.joint_type == "revolute")
    active_joint_names = tuple(joint.name for joint in active_joints)
    home_by_joint = _resolve_generated_q_home(active_joint_names, q_home_rad)
    home_values = tuple(home_by_joint[name] for name in active_joint_names)
    joint_limits = tuple(_joint_limits(joint) for joint in active_joints)
    active_index_by_name = {name: index for index, name in enumerate(active_joint_names)}
    kinematic_joints = tuple(
        KinematicJointSemanticsCfg(
            joint_name=joint.name,
            joint_type=joint.joint_type,
            parent_link=joint.parent,
            child_link=str(joint.child),
            origin_pos_m=_vector3(
                _exported_joint_origin(finger, joint, is_first=joint_index == 0).pos,
                f"joint[{joint.name}].origin.pos",
            ),
            origin_rpy_rad=_vector3(
                _exported_joint_origin(finger, joint, is_first=joint_index == 0).rpy,
                f"joint[{joint.name}].origin.rpy",
            ),
            axis_local=_vector3(joint.axis, f"joint[{joint.name}].axis"),
            active_joint_index=active_index_by_name.get(joint.name),
        )
        for finger in hand.fingers
        for joint_index, joint in enumerate(finger.joints)
    )

    palm_owner = _OwnerBuilder(
        owner_id="palm",
        role="palm",
        parent_owner_id=None,
        finger_name=None,
        joint_name=None,
        reference_link=palm.name,
        component_ids=[],
    )
    joint_owners: list[_OwnerBuilder] = []
    tip_owners: list[_OwnerBuilder] = []
    owner_by_id: dict[str, _OwnerBuilder] = {palm_owner.owner_id: palm_owner}
    components: list[CollisionComponentSemanticsCfg] = []

    for collision_index, collision in enumerate(palm.collisions):
        component = _make_component(
            collision,
            component_id=f"palm/{palm.name}/collision/{collision_index}",
            owner_id=palm_owner.owner_id,
            carrier_link=palm.name,
            collision_index=collision_index,
            source_joint_name=None,
        )
        components.append(component)
        palm_owner.component_ids.append(component.component_id)

    anchor_seeds: list[AnchorSeedSemanticsCfg] = []
    for finger in hand.fingers:
        _derive_finger_owners(
            hand,
            finger,
            palm_owner=palm_owner,
            joint_owners=joint_owners,
            tip_owners=tip_owners,
            owner_by_id=owner_by_id,
            components=components,
        )
        anchor_seeds.append(_derive_anchor_seed(hand, finger))

    owner_builders = [palm_owner, *joint_owners, *tip_owners]
    owners = tuple(
        GeometryOwnerSemanticsCfg(
            owner_id=owner.owner_id,
            owner_index=owner_index,
            role=owner.role,
            parent_owner_id=owner.parent_owner_id,
            finger_name=owner.finger_name,
            joint_name=owner.joint_name,
            reference_link=owner.reference_link,
            component_ids=tuple(owner.component_ids),
        )
        for owner_index, owner in enumerate(owner_builders)
    )

    for owner in owners:
        if not owner.component_ids:
            raise ValueError(f"generated owner '{owner.owner_id}' has no collision geometry")

    payload = {
        "schema_version": SEMANTICS_SCHEMA_VERSION,
        "migration_version": GENERATED_MIGRATION_VERSION,
        "source_kind": "generated",
        "asset_id": str(asset_id),
        "asset_name": hand.name,
        "topology_key": topology_key,
        "family": hand.family,
        "handedness": hand.handedness,
        "units": {"length": "m", "angle": "rad"},
        "asset_to_hand_rotation": _float_tuple(asset_to_hand_rotation, length=9, field_name="asset_to_hand_rotation"),
        "asset_to_hand_translation_m": _vector3(asset_to_hand_translation_m, "asset_to_hand_translation_m"),
        "palm_link": palm.name,
        "palm_origin_pos_m": _vector3(cast(PoseCfg, palm.origin).pos, "palm.origin.pos"),
        "palm_origin_rpy_rad": _vector3(cast(PoseCfg, palm.origin).rpy, "palm.origin.rpy"),
        "kinematic_joints": kinematic_joints,
        "active_joint_names": active_joint_names,
        "q_home_rad": home_values,
        "joint_limits_rad": joint_limits,
        "owners": owners,
        "components": tuple(components),
        "anchor_seeds": tuple(anchor_seeds),
    }
    content_hash = _content_hash(payload)
    return HandGeometrySemanticsCfg(**payload, content_hash=content_hash)


def geometry_semantics_to_dict(semantics: HandGeometrySemanticsCfg) -> dict[str, Any]:
    "Serializes typed geometry semantics without changing source units or joint order."

    return asdict(semantics)


def geometry_semantics_from_dict(document: Mapping[str, Any]) -> HandGeometrySemanticsCfg:
    "Validates and reconstructs typed geometry semantics from a serialized sidecar mapping."

    if not isinstance(document, Mapping):
        raise TypeError(f"geometry semantics must be a mapping, got {type(document).__name__}")

    owners = tuple(_owner_from_dict(item) for item in _mapping_sequence(document["owners"], "owners"))
    kinematic_joints = tuple(
        _kinematic_joint_from_dict(item)
        for item in _mapping_sequence(document["kinematic_joints"], "kinematic_joints")
    )
    components = tuple(
        _component_from_dict(item) for item in _mapping_sequence(document["components"], "components")
    )
    anchor_seeds = tuple(
        _anchor_seed_from_dict(item) for item in _mapping_sequence(document["anchor_seeds"], "anchor_seeds")
    )
    source_kind = str(document["source_kind"])
    handedness = str(document["handedness"])
    return HandGeometrySemanticsCfg(
        schema_version=str(document["schema_version"]),
        migration_version=str(document["migration_version"]),
        source_kind=cast(Literal["generated", "official"], source_kind),
        asset_id=str(document["asset_id"]),
        asset_name=str(document["asset_name"]),
        topology_key=None if document.get("topology_key") is None else str(document["topology_key"]),
        family=str(document["family"]),
        handedness=cast(Literal["left", "right", "unknown"], handedness),
        units={str(key): str(value) for key, value in _mapping(document["units"], "units").items()},
        asset_to_hand_rotation=cast(
            Matrix3Flat,
            _float_tuple(document["asset_to_hand_rotation"], length=9, field_name="asset_to_hand_rotation"),
        ),
        asset_to_hand_translation_m=_vector3(
            document["asset_to_hand_translation_m"], "asset_to_hand_translation_m"
        ),
        palm_link=str(document["palm_link"]),
        palm_origin_pos_m=_vector3(document["palm_origin_pos_m"], "palm_origin_pos_m"),
        palm_origin_rpy_rad=_vector3(document["palm_origin_rpy_rad"], "palm_origin_rpy_rad"),
        kinematic_joints=kinematic_joints,
        active_joint_names=tuple(str(name) for name in document["active_joint_names"]),
        q_home_rad=tuple(float(value) for value in document["q_home_rad"]),
        joint_limits_rad=tuple(
            cast(tuple[float, float], _float_tuple(limits, length=2, field_name="joint_limits_rad[]"))
            for limits in document["joint_limits_rad"]
        ),
        owners=owners,
        components=components,
        anchor_seeds=anchor_seeds,
        content_hash=str(document["content_hash"]),
    )


def _owner_from_dict(document: Mapping[str, Any]) -> GeometryOwnerSemanticsCfg:

    role = str(document["role"])
    return GeometryOwnerSemanticsCfg(
        owner_id=str(document["owner_id"]),
        owner_index=int(document["owner_index"]),
        role=cast(OwnerRole, role),
        parent_owner_id=None if document.get("parent_owner_id") is None else str(document["parent_owner_id"]),
        finger_name=None if document.get("finger_name") is None else str(document["finger_name"]),
        joint_name=None if document.get("joint_name") is None else str(document["joint_name"]),
        reference_link=str(document["reference_link"]),
        component_ids=tuple(str(component_id) for component_id in document["component_ids"]),
    )


def _kinematic_joint_from_dict(document: Mapping[str, Any]) -> KinematicJointSemanticsCfg:

    joint_type = str(document["joint_type"])
    return KinematicJointSemanticsCfg(
        joint_name=str(document["joint_name"]),
        joint_type=cast(Literal["fixed", "revolute"], joint_type),
        parent_link=str(document["parent_link"]),
        child_link=str(document["child_link"]),
        origin_pos_m=_vector3(document["origin_pos_m"], "kinematic_joint.origin_pos_m"),
        origin_rpy_rad=_vector3(document["origin_rpy_rad"], "kinematic_joint.origin_rpy_rad"),
        axis_local=_vector3(document["axis_local"], "kinematic_joint.axis_local"),
        active_joint_index=(
            None if document.get("active_joint_index") is None else int(document["active_joint_index"])
        ),
    )


def _component_from_dict(document: Mapping[str, Any]) -> CollisionComponentSemanticsCfg:

    return CollisionComponentSemanticsCfg(
        component_id=str(document["component_id"]),
        owner_id=str(document["owner_id"]),
        carrier_link=str(document["carrier_link"]),
        collision_index=int(document["collision_index"]),
        collision_name=None if document.get("collision_name") is None else str(document["collision_name"]),
        geometry_kind=str(document["geometry_kind"]),
        geometry_payload=dict(_mapping(document["geometry_payload"], "geometry_payload")),
        origin_pos_m=_vector3(document["origin_pos_m"], "origin_pos_m"),
        origin_rpy_rad=_vector3(document["origin_rpy_rad"], "origin_rpy_rad"),
        source_joint_name=(
            None if document.get("source_joint_name") is None else str(document["source_joint_name"])
        ),
    )


def _anchor_seed_from_dict(document: Mapping[str, Any]) -> AnchorSeedSemanticsCfg:

    return AnchorSeedSemanticsCfg(
        seed_id=str(document["seed_id"]),
        finger_name=str(document["finger_name"]),
        first_active_joint_name=str(document["first_active_joint_name"]),
        support_owner_id=str(document["support_owner_id"]),
        position_a_m=_vector3(document["position_a_m"], "position_a_m"),
        rotation_a=cast(Matrix3Flat, _float_tuple(document["rotation_a"], length=9, field_name="rotation_a")),
    )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return value


def _mapping_sequence(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of mappings")
    return tuple(_mapping(item, f"{field_name}[]") for item in value)


def _derive_finger_owners(
    hand: HandCfg,
    finger: FingerCfg,
    *,
    palm_owner: _OwnerBuilder,
    joint_owners: list[_OwnerBuilder],
    tip_owners: list[_OwnerBuilder],
    owner_by_id: dict[str, _OwnerBuilder],
    components: list[CollisionComponentSemanticsCfg],
) -> None:

    active_indices = [index for index, joint in enumerate(finger.joints) if joint.joint_type == "revolute"]
    if not active_indices:
        raise ValueError(f"generated finger '{finger.name}' has no active joint")
    first_active = active_indices[0]
    last_active = active_indices[-1]
    previous_active_owner_id = palm_owner.owner_id

    for joint_index, joint in enumerate(finger.joints):
        if joint.joint_type == "revolute":
            owner_id = f"joint/{joint.name}"
            owner = _OwnerBuilder(
                owner_id=owner_id,
                role="joint",
                parent_owner_id=previous_active_owner_id,
                finger_name=finger.name,
                joint_name=joint.name,
                reference_link=str(joint.child),
                component_ids=[],
            )
            joint_owners.append(owner)
            owner_by_id[owner_id] = owner
            previous_active_owner_id = owner_id
        elif joint_index < first_active:
            if joint.is_tip:
                raise ValueError(f"fixed root '{joint.name}' before first active joint cannot be marked is_tip")
            owner = palm_owner
        elif joint_index > last_active:
            if joint.collisions and not joint.is_tip:
                raise ValueError(
                    f"fixed descendant '{joint.name}' carries collision geometry but is_tip is false; "
                    "generated owner assignment is ambiguous"
                )
            if not joint.collisions and not joint.is_tip:
                continue
            owner_id = f"tip/{finger.name}"
            owner = owner_by_id.get(owner_id)
            if owner is None:
                owner = _OwnerBuilder(
                    owner_id=owner_id,
                    role="tip",
                    parent_owner_id=previous_active_owner_id,
                    finger_name=finger.name,
                    joint_name=None,
                    reference_link=str(finger.joints[last_active].child),
                    component_ids=[],
                )
                tip_owners.append(owner)
                owner_by_id[owner_id] = owner
        else:
            if joint.collisions:
                raise ValueError(
                    f"fixed joint '{joint.name}' lies between active joints and carries collision geometry; "
                    "generated owner assignment requires an explicit sidecar"
                )
            continue

        for collision_index, collision in enumerate(joint.collisions):
            component = _make_component(
                collision,
                component_id=f"finger/{finger.name}/joint/{joint.name}/collision/{collision_index}",
                owner_id=owner.owner_id,
                carrier_link=str(joint.child),
                collision_index=collision_index,
                source_joint_name=joint.name,
            )
            components.append(component)
            owner.component_ids.append(component.component_id)


def _derive_anchor_seed(hand: HandCfg, finger: FingerCfg) -> AnchorSeedSemanticsCfg:

    palm = cast(PalmCfg, hand.palm)
    rotation, translation = _pose_transform(cast(PoseCfg, palm.origin))
    for joint_index, joint in enumerate(finger.joints):
        origin = _exported_joint_origin(finger, joint, is_first=joint_index == 0)
        local_rotation, local_translation = _pose_transform(origin)
        rotation, translation = _compose_transform(rotation, translation, local_rotation, local_translation)
        if joint.joint_type == "revolute":
            return AnchorSeedSemanticsCfg(
                seed_id=f"finger/{finger.name}/first-active",
                finger_name=finger.name,
                first_active_joint_name=joint.name,
                support_owner_id="palm",
                position_a_m=translation,
                rotation_a=_flatten_rotation(rotation),
            )

    raise ValueError(f"generated finger '{finger.name}' has no active joint")


def _exported_joint_origin(finger: FingerCfg, joint: JointCfg, *, is_first: bool) -> PoseCfg:

    joint_origin = cast(PoseCfg, joint.origin)
    if not is_first:
        return joint_origin.copy()
    mount = cast(PoseCfg, finger.mount)
    return PoseCfg(
        pos=cast(Vector3, tuple(mount.pos[axis] + joint_origin.pos[axis] for axis in range(3))),
        rpy=cast(Vector3, tuple(mount.rpy[axis] + joint_origin.rpy[axis] for axis in range(3))),
    )


def _make_component(
    collision: CollisionGeometryCfg,
    *,
    component_id: str,
    owner_id: str,
    carrier_link: str,
    collision_index: int,
    source_joint_name: str | None,
) -> CollisionComponentSemanticsCfg:

    origin = cast(PoseCfg, collision.origin)
    geometry_payload = collision.geometry.to_dict()
    geometry_payload = {"type": collision.geometry.kind, **geometry_payload}
    return CollisionComponentSemanticsCfg(
        component_id=component_id,
        owner_id=owner_id,
        carrier_link=carrier_link,
        collision_index=collision_index,
        collision_name=collision.name,
        geometry_kind=collision.geometry.kind,
        geometry_payload=geometry_payload,
        origin_pos_m=_vector3(origin.pos, "collision.origin.pos"),
        origin_rpy_rad=_vector3(origin.rpy, "collision.origin.rpy"),
        source_joint_name=source_joint_name,
    )


def _resolve_generated_q_home(
    active_joint_names: tuple[str, ...],
    q_home_rad: Mapping[str, float] | None,
) -> dict[str, float]:

    if q_home_rad is None:
        return {name: 0.0 for name in active_joint_names}
    provided = set(q_home_rad)
    expected = set(active_joint_names)
    if provided != expected:
        missing = sorted(expected - provided)
        unknown = sorted(provided - expected)
        raise ValueError(f"q_home_rad must exactly cover active joints; missing={missing}, unknown={unknown}")
    return {name: float(q_home_rad[name]) for name in active_joint_names}


def _joint_limits(joint: JointCfg) -> tuple[float, float]:

    if joint.limit is None:
        raise ValueError(f"active joint '{joint.name}' is missing limits")
    limit = cast(JointLimitCfg, joint.limit)
    return (float(limit.lower), float(limit.upper))


def _pose_transform(pose: PoseCfg) -> tuple[tuple[Vector3, Vector3, Vector3], Vector3]:

    roll, pitch, yaw = pose.rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return rotation, _vector3(pose.pos, "pose.pos")


def _compose_transform(
    parent_rotation: tuple[Vector3, Vector3, Vector3],
    parent_translation: Vector3,
    local_rotation: tuple[Vector3, Vector3, Vector3],
    local_translation: Vector3,
) -> tuple[tuple[Vector3, Vector3, Vector3], Vector3]:

    rotation = tuple(
        tuple(
            sum(parent_rotation[row][inner] * local_rotation[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    rotated_translation = tuple(
        sum(parent_rotation[row][inner] * local_translation[inner] for inner in range(3)) for row in range(3)
    )
    translation = tuple(rotated_translation[axis] + parent_translation[axis] for axis in range(3))
    return rotation, translation  # type: ignore[return-value]


def _flatten_rotation(rotation: tuple[Vector3, Vector3, Vector3]) -> Matrix3Flat:

    values = tuple(value for row in rotation for value in row)
    return values  # type: ignore[return-value]


def _vector3(values: Sequence[float], field_name: str) -> Vector3:

    return _float_tuple(values, length=3, field_name=field_name)  # type: ignore[return-value]


def _float_tuple(values: Sequence[float], *, length: int, field_name: str) -> tuple[float, ...]:

    packed = tuple(float(value) for value in values)
    if len(packed) != length:
        raise ValueError(f"{field_name} must have length {length}, got {len(packed)}")
    if not all(math.isfinite(value) for value in packed):
        raise ValueError(f"{field_name} must contain finite values")
    return packed


def _content_hash(payload: Mapping[str, Any]) -> str:

    serializable = _to_serializable(payload)
    encoded = json.dumps(serializable, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _to_serializable(value: Any) -> Any:

    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_serializable(item) for item in value]
    return value


__all__ = [
    "AnchorSeedSemanticsCfg",
    "CollisionComponentSemanticsCfg",
    "GENERATED_MIGRATION_VERSION",
    "GeometryOwnerSemanticsCfg",
    "HandGeometrySemanticsCfg",
    "KinematicJointSemanticsCfg",
    "SEMANTICS_SCHEMA_VERSION",
    "derive_generated_geometry_semantics",
    "geometry_semantics_from_dict",
    "geometry_semantics_to_dict",
]
