"Mirrors generated hand geometry across the palm plane and validates the complete right-to-left contract."

from __future__ import annotations

import math
from typing import Literal

from .asset_schema_core import (
    CollisionGeometryCfg,
    InertialCfg,
    InertiaTensorCfg,
    MeshGeometryCfg,
    PoseCfg,
    Vector3,
    VisualGeometryCfg,
)
from .asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg

HandTarget = Literal["left", "right"]
"Type-level target for values emitted by the hand builder."


HANDEDNESS_CONTRACT_VERSION = "1.0"
"Version of the strict generated right-to-left mirror contract."


def rpy_rotation_matrix(rpy: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    "Converts roll, pitch, and yaw angles in radians to a rotation matrix."

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def matrix_to_rpy(matrix: tuple[Vector3, Vector3, Vector3]) -> Vector3:
    "Converts a rotation matrix to roll, pitch, and yaw angles in radians."

    horizontal_norm = math.hypot(matrix[0][0], matrix[1][0])
    pitch = math.atan2(-matrix[2][0], horizontal_norm)
    if horizontal_norm > 1e-12:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    return (roll, pitch, yaw)


def _matrix_multiply(
    lhs: tuple[Vector3, Vector3, Vector3],
    rhs: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:

    return tuple(
        tuple(sum(lhs[row][inner] * rhs[inner][column] for inner in range(3)) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _apply_rotation(rotation: tuple[Vector3, Vector3, Vector3], point: Vector3) -> Vector3:

    return tuple(
        sum(rotation[row][column] * point[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def compose_poses(parent: PoseCfg, local: PoseCfg) -> PoseCfg:
    "Composes parent and child poses in the declared transform order."

    parent_rotation = rpy_rotation_matrix(parent.rpy)
    local_rotation = rpy_rotation_matrix(local.rpy)
    rotated_local_position = _apply_rotation(parent_rotation, local.pos)
    return PoseCfg(
        pos=tuple(
            parent.pos[index] + rotated_local_position[index]
            for index in range(3)
        ),
        rpy=matrix_to_rpy(_matrix_multiply(parent_rotation, local_rotation)),
    )


def mirror_pose_about_yz(pose: PoseCfg) -> PoseCfg:
    'Reflects pose about yz.'

    return PoseCfg(
        pos=(-pose.pos[0], pose.pos[1], pose.pos[2]),
        rpy=(pose.rpy[0], -pose.rpy[1], -pose.rpy[2]),
    )


def mirror_revolute_axis_about_yz(axis: Vector3) -> Vector3:
    'Reflects revolute axis about yz.'

    return (axis[0], -axis[1], -axis[2])


def mirror_inertia_tensor_about_yz(inertia: InertiaTensorCfg) -> InertiaTensorCfg:
    'Reflects inertia tensor about yz.'

    return InertiaTensorCfg(
        ixx=inertia.ixx,
        iyy=inertia.iyy,
        izz=inertia.izz,
        ixy=-inertia.ixy,
        ixz=-inertia.ixz,
        iyz=inertia.iyz,
    )


def _mirror_inertial(inertial: InertialCfg | None) -> InertialCfg | None:

    if inertial is None:
        return None
    mirrored = inertial.copy()
    mirrored.origin = mirror_pose_about_yz(inertial.origin)
    mirrored.inertia = mirror_inertia_tensor_about_yz(inertial.inertia)
    return mirrored


def _mirror_mesh_geometry(geometry):

    if not isinstance(geometry, MeshGeometryCfg):
        return geometry.copy()
    return geometry.replace(reflected_about_yz=not geometry.reflected_about_yz)


def _mirror_collision(element: CollisionGeometryCfg) -> CollisionGeometryCfg:

    return element.replace(
        geometry=_mirror_mesh_geometry(element.geometry),
        origin=mirror_pose_about_yz(element.origin),
    )


def _mirror_visual(element: VisualGeometryCfg) -> VisualGeometryCfg:

    return element.replace(
        geometry=_mirror_mesh_geometry(element.geometry),
        origin=mirror_pose_about_yz(element.origin),
    )


def _mirror_joint(joint: JointCfg) -> JointCfg:

    axis = (
        mirror_revolute_axis_about_yz(joint.axis)
        if joint.joint_type == "revolute"
        else joint.axis
    )
    return joint.replace(
        origin=mirror_pose_about_yz(joint.origin),
        axis=axis,
        inertial=_mirror_inertial(joint.inertial),
        collisions=[_mirror_collision(element) for element in joint.collisions],
        visuals=[_mirror_visual(element) for element in joint.visuals],
    )


def _mirror_finger(finger: FingerCfg) -> FingerCfg:

    return finger.replace(
        mount=mirror_pose_about_yz(finger.mount),  # palm->finger root frame
        joints=[_mirror_joint(joint) for joint in finger.joints],
    )


def _mirror_palm(palm: PalmCfg) -> PalmCfg:

    metadata = dict(palm.metadata)
    raw_mounts = metadata.get("finger_mounts")
    if isinstance(raw_mounts, dict):
        metadata["finger_mounts"] = {
            name: mirror_pose_about_yz(PoseCfg.from_value(pose))
            for name, pose in raw_mounts.items()
        }
    return palm.replace(
        origin=mirror_pose_about_yz(palm.origin),  # hand root->palm frame
        inertial=_mirror_inertial(palm.inertial),
        collisions=[_mirror_collision(element) for element in palm.collisions],
        visuals=[_mirror_visual(element) for element in palm.visuals],
        metadata=metadata,
    )


def handedness_contract(*, target: HandTarget) -> dict[str, object]:
    "Returns the strict mirror-plane and same-joint-value contract."

    return {
        "version": HANDEDNESS_CONTRACT_VERSION,
        "canonical_handedness": "right",
        "target_handedness": target,
        "reflection_plane": "palm_yz",
        "same_q": True,
        "physical_lowering_complete": True,
    }


def validate_generated_handedness_contract(
    sidecar: dict[str, object],
    *,
    allow_legacy_left_handedness: bool = False,
) -> None:
    'Validates generated handedness contract.'

    if str(sidecar.get("handedness", "")).lower() != "left":
        return
    if allow_legacy_left_handedness:
        return

    contract = sidecar.get("handedness_contract")
    expected = handedness_contract(target="left")
    if not isinstance(contract, dict) or any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "legacy generated left hand lacks a valid strict handedness_contract; "
            "regenerate it with the current asset pipeline or set "
            "allow_legacy_left_handedness=True only for historical audit"
        )


def lower_hand_to_handedness(hand: HandCfg, target: HandTarget) -> HandCfg:
    'Applies hand to handedness.'

    if target not in {"left", "right"}:
        raise ValueError(f"unsupported handedness target: {target!r}")
    if hand.handedness not in {"left", "right"}:
        raise ValueError(f"strict handedness lowering requires known source handedness, got {hand.handedness!r}")

    if hand.handedness == target:
        lowered = hand.copy()
    else:
        lowered = hand.replace(
            palm=_mirror_palm(hand.palm),
            fingers=[_mirror_finger(finger) for finger in hand.fingers],
            handedness=target,
        )

    metadata = dict(lowered.metadata)
    metadata["handedness_contract"] = handedness_contract(target=target)
    return lowered.replace(handedness=target, metadata=metadata)


__all__ = [
    "HANDEDNESS_CONTRACT_VERSION",
    "compose_poses",
    "handedness_contract",
    "lower_hand_to_handedness",
    "matrix_to_rpy",
    "mirror_inertia_tensor_about_yz",
    "mirror_pose_about_yz",
    "mirror_revolute_axis_about_yz",
    "rpy_rotation_matrix",
    "validate_generated_handedness_contract",
]
