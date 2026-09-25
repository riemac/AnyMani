"Extracts collision geometry and owner-local transforms for physical checks."

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

from ..asset_base import HandCfg
from ..asset_schema_core import (
    BoxGeometryCfg,
    CylinderGeometryCfg,
    EllipticCylinderGeometryCfg,
    MeshGeometryCfg,
    PoseCfg,
    SphereGeometryCfg,
    Vector3,
)


UnsupportedGeometryPolicy = Literal["fail", "warn_skip"]
"Action for collision geometry unsupported by the selected SDF backend."


SUPPORTED_PRIMITIVE_KINDS = ("box", "cylinder", "elliptic_cylinder", "sphere")
"Primitive geometry kinds accepted by the canonical lowering."


@dataclass(frozen=True)
class CollisionBodyRecord:
    "Collision geometry with its owning link, local pose, and typed dimensions."

    finger_name: str
    joint_name: str
    link_name: str
    body_name: str
    body_path: str
    geometry_kind: str
    geometry: BoxGeometryCfg | CylinderGeometryCfg | EllipticCylinderGeometryCfg | SphereGeometryCfg | MeshGeometryCfg
    world_pose: PoseCfg


@dataclass(frozen=True)
class SkippedCollisionBody:
    "Reason a source collision shape was excluded from a geometric check."

    finger_name: str
    joint_name: str
    link_name: str
    body_name: str
    body_path: str
    geometry_kind: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        'Serializes the typed object as a dictionary.'

        return {
            "finger_name": self.finger_name,
            "joint_name": self.joint_name,
            "link_name": self.link_name,
            "body_name": self.body_name,
            "body_path": self.body_path,
            "geometry_kind": self.geometry_kind,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CollisionExtractionResult:
    "Result record with ordered geometry, provenance, and validation status."

    bodies_by_finger: dict[str, list[CollisionBodyRecord]]
    skipped_bodies: list[SkippedCollisionBody]

    @property
    def complete(self) -> bool:
        "Reports whether collision geometry is fully supported for the requested check."

        return not self.skipped_bodies


def extract_finger_collision_bodies(
    hand: HandCfg,
    *,
    unsupported_policy: UnsupportedGeometryPolicy = "fail",
) -> CollisionExtractionResult:
    "Returns collision bodies grouped by semantic finger owner."

    if unsupported_policy not in {"fail", "warn_skip"}:
        raise ValueError(f"unsupported collision geometry policy: {unsupported_policy!r}")

    bodies_by_finger: dict[str, list[CollisionBodyRecord]] = {}
    skipped_bodies: list[SkippedCollisionBody] = []

    link_poses_by_finger = extract_finger_link_poses(hand)

    for finger in hand.fingers:
        bodies_by_finger.setdefault(finger.name, [])
        finger_link_poses = link_poses_by_finger.get(finger.name, ())

        for joint_index, joint in enumerate(finger.joints):
            link_pose = finger_link_poses[joint_index]

            for collision_index, collision in enumerate(joint.collisions):
                body_name = collision.name or f"{joint.child}_collision_{collision_index}"
                body_path = f"{finger.name}/{joint.name}/{joint.child}/{body_name}"
                geometry = collision.geometry
                if not isinstance(
                    geometry,
                    (BoxGeometryCfg, CylinderGeometryCfg, EllipticCylinderGeometryCfg, SphereGeometryCfg, MeshGeometryCfg),
                ):
                    skipped = _make_skipped_body(
                        finger_name=finger.name,
                        joint_name=joint.name,
                        link_name=str(joint.child),
                        body_name=body_name,
                        body_path=body_path,
                        geometry=geometry,
                    )
                    if unsupported_policy == "fail":
                        raise ValueError(
                            f"unsupported collision geometry for SDF clearance: {body_path} kind={skipped.geometry_kind!r}"
                        )
                    skipped_bodies.append(skipped)
                    continue



                world_pose = _compose_pose(link_pose, collision.origin)
                bodies_by_finger[finger.name].append(
                    CollisionBodyRecord(
                        finger_name=finger.name,
                        joint_name=joint.name,
                        link_name=str(joint.child),
                        body_name=body_name,
                        body_path=body_path,
                        geometry_kind=geometry.kind,
                        geometry=geometry,
                        world_pose=world_pose,
                    )
                )
    return CollisionExtractionResult(bodies_by_finger=bodies_by_finger, skipped_bodies=skipped_bodies)


def extract_finger_link_poses(hand: HandCfg) -> dict[str, list[PoseCfg]]:
    """Composes source joint origins from each finger mount at the home pose.

    This consumes the lowered joint tree after root-mount placement; it does not use the older mount-origin spacing approximation.
    """

    link_poses_by_finger: dict[str, list[PoseCfg]] = {}

    for finger in hand.fingers:
        finger_link_poses: list[PoseCfg] = []
        parent_link_pose = PoseCfg()

        for joint_index, joint in enumerate(finger.joints):
            if joint_index == 0:
                link_pose = _pose_add(finger.mount, joint.origin)
            else:
                link_pose = _compose_pose(parent_link_pose, joint.origin)
            finger_link_poses.append(link_pose)
            parent_link_pose = link_pose

        link_poses_by_finger[finger.name] = finger_link_poses

    return link_poses_by_finger
def _make_skipped_body(
    *,
    finger_name: str,
    joint_name: str,
    link_name: str,
    body_name: str,
    body_path: str,
    geometry: Any,
) -> SkippedCollisionBody:

    geometry_kind = getattr(geometry, "kind", type(geometry).__name__)
    if isinstance(geometry, MeshGeometryCfg):
        reason = "mesh geometry is not certified by sampled primitive SDF v1"
    else:
        reason = "geometry kind is not supported by sampled primitive SDF v1"
    return SkippedCollisionBody(
        finger_name=finger_name,
        joint_name=joint_name,
        link_name=link_name,
        body_name=body_name,
        body_path=body_path,
        geometry_kind=str(geometry_kind),
        reason=reason,
    )


def _pose_add(lhs: PoseCfg, rhs: PoseCfg) -> PoseCfg:

    return PoseCfg(
        pos=(
            lhs.pos[0] + rhs.pos[0],
            lhs.pos[1] + rhs.pos[1],
            lhs.pos[2] + rhs.pos[2],
        ),
        rpy=(
            lhs.rpy[0] + rhs.rpy[0],
            lhs.rpy[1] + rhs.rpy[1],
            lhs.rpy[2] + rhs.rpy[2],
        ),
    )


def _compose_pose(parent: PoseCfg, local: PoseCfg) -> PoseCfg:

    parent_rotation = rpy_rotation_matrix(parent.rpy)
    local_rotation = rpy_rotation_matrix(local.rpy)
    local_pos_in_world = apply_rotation(parent_rotation, local.pos)
    world_rotation = _matrix_multiply(parent_rotation, local_rotation)
    return PoseCfg(
        pos=(
            parent.pos[0] + local_pos_in_world[0],
            parent.pos[1] + local_pos_in_world[1],
            parent.pos[2] + local_pos_in_world[2],
        ),
        rpy=_matrix_to_rpy(world_rotation),
    )


def rpy_rotation_matrix(rpy: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    "Converts URDF roll-pitch-yaw angles in radians to a rotation matrix."

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def apply_rotation(matrix: tuple[Vector3, Vector3, Vector3], point: Vector3) -> Vector3:
    'Applies rotation.'

    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2],
    )


def apply_inverse_pose(pose: PoseCfg, point_world: Vector3) -> Vector3:
    'Applies inverse pose.'

    dx = point_world[0] - pose.pos[0]
    dy = point_world[1] - pose.pos[1]
    dz = point_world[2] - pose.pos[2]
    rotation = rpy_rotation_matrix(pose.rpy)
    return (
        rotation[0][0] * dx + rotation[1][0] * dy + rotation[2][0] * dz,
        rotation[0][1] * dx + rotation[1][1] * dy + rotation[2][1] * dz,
        rotation[0][2] * dx + rotation[1][2] * dy + rotation[2][2] * dz,
    )


def apply_pose(pose: PoseCfg, point_local: Vector3) -> Vector3:
    'Applies pose.'

    rotated = apply_rotation(rpy_rotation_matrix(pose.rpy), point_local)
    return (
        pose.pos[0] + rotated[0],
        pose.pos[1] + rotated[1],
        pose.pos[2] + rotated[2],
    )


def _matrix_multiply(
    lhs: tuple[Vector3, Vector3, Vector3],
    rhs: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:

    rhs_cols = (
        (rhs[0][0], rhs[1][0], rhs[2][0]),
        (rhs[0][1], rhs[1][1], rhs[2][1]),
        (rhs[0][2], rhs[1][2], rhs[2][2]),
    )
    rows: list[Vector3] = []
    for row in lhs:
        rows.append(
            (
                row[0] * rhs_cols[0][0] + row[1] * rhs_cols[0][1] + row[2] * rhs_cols[0][2],
                row[0] * rhs_cols[1][0] + row[1] * rhs_cols[1][1] + row[2] * rhs_cols[1][2],
                row[0] * rhs_cols[2][0] + row[1] * rhs_cols[2][1] + row[2] * rhs_cols[2][2],
            )
        )
    return (rows[0], rows[1], rows[2])


def _matrix_to_rpy(matrix: tuple[Vector3, Vector3, Vector3]) -> Vector3:

    pitch = math.asin(-max(-1.0, min(1.0, matrix[2][0])))
    cp = math.cos(pitch)

    if abs(cp) > 1e-12:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[0][1], matrix[1][1])
        yaw = 0.0

    return (roll, pitch, yaw)


__all__ = [
    "CollisionBodyRecord",
    "CollisionExtractionResult",
    "SkippedCollisionBody",
    "UnsupportedGeometryPolicy",
    "extract_finger_collision_bodies",
    "extract_finger_link_poses",
    "apply_inverse_pose",
    "apply_pose",
]
