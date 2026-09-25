"Lowers surviving and removed finger slots without reordering source anatomy."

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Literal

from ...asset_base import AssetCfgBase, HandCfg
from ...asset_schema_core import CollisionGeometryCfg, InertialCfg, PoseCfg, VisualGeometryCfg


# ============================================================================

# ============================================================================


@dataclass
class JointDeleteCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["JointDeleteMutator"] | None = None
    "Associated runtime implementation for this configuration class."

    target_finger: str | None = None
    "Semantic finger slot affected by this proposal."

    deleted_joints: tuple[str, ...] = ()
    "Source joints removed by the declared connectivity recipe."

    regroup_strategy: Literal["merge", "drop", "keep"] = "merge"
    "Grouping rule used to assign source geometry owners."

    respect_preset: bool = True
    "Whether a named preset remains authoritative for the selected parameter."

    keep_terminal_joint: bool = True
    "Whether the final fixed tip joint remains in the lowered chain."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointDeleteMutator


# ============================================================================

# ============================================================================


class JointDeleteMutator:
    "Lowers a missing-joint topology while retaining surviving source slot identity."

    cfg: JointDeleteCfg

    def __init__(self, cfg: JointDeleteCfg):
        self.cfg = cfg

    def mutate(self, target: HandCfg) -> HandCfg | None:
        "Removes the declared joints and links while preserving surviving anatomy slots."

        if self.cfg.regroup_strategy == "keep":
            raise NotImplementedError(
                "regroup_strategy='keep' requires orphan sub-link support, which current schema does not provide."
            )

        mutated = target.copy()
        if not mutated.fingers:
            return None

        if self.cfg.target_finger is None:
            finger_index = random.randrange(len(mutated.fingers))
        else:
            finger_index = next((index for index, finger in enumerate(mutated.fingers) if finger.name == self.cfg.target_finger), -1)
            if finger_index < 0:
                return None

        finger = mutated.fingers[finger_index]
        deletable = [
            joint.name
            for joint in finger.joints
            if not (self.cfg.keep_terminal_joint and joint.is_tip)
        ]
        if not deletable:
            return None

        requested = list(self.cfg.deleted_joints) or [random.choice(deletable)]
        delete_set = {name for name in requested if name in deletable}
        if not delete_set:
            return None



        remaining_revolute = sum(
            1 for joint in finger.joints if joint.name not in delete_set and joint.joint_type == "revolute"
        )
        if self.cfg.respect_preset and remaining_revolute < 1:
            return None

        rebuilt = self._delete_from_finger(mutated, finger, delete_set)
        if rebuilt is None:
            return None

        mutated.fingers[finger_index] = rebuilt
        try:
            return mutated.replace(fingers=mutated.fingers)
        except Exception:
            return None

    def _delete_from_finger(self, hand: HandCfg, finger, delete_set: set[str]):

        new_joints = []
        last_kept_parent = finger.parent_link
        last_kept_container = hand.palm




        pending_cumulative_origin = PoseCfg()



        #


        pending_drop_relink_origin: PoseCfg | None = None

        for joint in finger.joints:


            geometry_origin_in_container = _compose_pose(pending_cumulative_origin, joint.origin)

            if joint.name in delete_set:


                if self.cfg.regroup_strategy == "merge":
                    _merge_deleted_joint_into_container(last_kept_container, joint, geometry_origin_in_container)




                if pending_drop_relink_origin is None:
                    pending_drop_relink_origin = joint.origin.copy()





                pending_cumulative_origin = geometry_origin_in_container
                continue


            if pending_drop_relink_origin is None:
                relink_origin = joint.origin.copy()
            elif self.cfg.regroup_strategy == "drop":



                relink_origin = pending_drop_relink_origin.copy()
            else:


                relink_origin = _compose_pose(pending_cumulative_origin, joint.origin)

            kept_joint = joint.replace(parent=last_kept_parent, origin=relink_origin)
            new_joints.append(kept_joint)


            last_kept_parent = kept_joint.child
            last_kept_container = kept_joint


            pending_cumulative_origin = PoseCfg()
            pending_drop_relink_origin = None

        if not new_joints:
            return None


        #


        #

        #
        # - `index_j0 -> index_mcp2`
        #


        renumbered_joints, surviving_joint_name_map = _renumber_surviving_joints(
            finger_name=finger.name,
            joints=new_joints,
        )
        finger_metadata = dict(finger.metadata)
        joint_delete_metadata = dict(finger_metadata.get("joint_delete", {}))
        joint_delete_metadata["deleted_joints"] = sorted(delete_set)
        joint_delete_metadata["surviving_joint_name_map"] = surviving_joint_name_map
        finger_metadata["joint_delete"] = joint_delete_metadata
        return finger.replace(joints=renumbered_joints, metadata=finger_metadata)


def _compose_pose(lhs: PoseCfg, rhs: PoseCfg) -> PoseCfg:

    return PoseCfg(
        pos=(lhs.pos[0] + rhs.pos[0], lhs.pos[1] + rhs.pos[1], lhs.pos[2] + rhs.pos[2]),
        rpy=(lhs.rpy[0] + rhs.rpy[0], lhs.rpy[1] + rhs.rpy[1], lhs.rpy[2] + rhs.rpy[2]),
    )


def _renumber_surviving_joints(*, finger_name: str, joints: list) -> tuple[list, list[dict[str, Any]]]:

    renumbered_joints = []
    surviving_joint_name_map: list[dict[str, Any]] = []
    compact_revolute_index = 0

    for joint in joints:
        original_name = joint.metadata.get("original_joint_name", joint.name)
        previous_name = joint.name

        if joint.joint_type == "revolute":
            new_name = f"{finger_name}_j{compact_revolute_index}"
            compact_revolute_index += 1
        elif joint.is_tip:
            new_name = f"{finger_name}_tip"
        else:


            new_name = joint.name

        new_metadata = dict(joint.metadata)
        new_metadata["original_joint_name"] = original_name
        new_metadata["previous_joint_name"] = previous_name
        new_metadata["current_joint_name"] = new_name
        if joint.joint_type == "revolute":
            new_metadata["joint_index"] = compact_revolute_index - 1

        renumbered_joints.append(joint.replace(name=new_name, metadata=new_metadata))
        surviving_joint_name_map.append(
            {
                "previous_name": previous_name,
                "current_name": new_name,
                "original_name": original_name,
                "child_link": str(joint.child),
                "joint_type": joint.joint_type,
                "is_tip": bool(joint.is_tip),
            }
        )

    return renumbered_joints, surviving_joint_name_map


def _merge_deleted_joint_into_container(container, joint, joint_pose_in_container: PoseCfg) -> None:

    container.collisions.extend(
        [
            CollisionGeometryCfg(
                name=collision.name,
                geometry=collision.geometry.copy(),
                origin=_compose_pose(joint_pose_in_container, collision.origin),
            )
            for collision in joint.collisions
        ]
    )
    container.visuals.extend(
        [
            VisualGeometryCfg(
                name=visual.name,
                geometry=visual.geometry.copy(),
                origin=_compose_pose(joint_pose_in_container, visual.origin),
            )
            for visual in joint.visuals
        ]
    )

    if getattr(container, "inertial", None) is not None and joint.inertial is not None:
        container.inertial = _merge_inertials(
            container.inertial,
            joint.inertial.replace(origin=_compose_pose(joint_pose_in_container, joint.inertial.origin)),
        )


def _merge_inertials(lhs: InertialCfg, rhs: InertialCfg) -> InertialCfg:

    m1 = lhs.mass
    m2 = rhs.mass
    total_mass = m1 + m2
    com = (
        (m1 * lhs.origin.pos[0] + m2 * rhs.origin.pos[0]) / total_mass,
        (m1 * lhs.origin.pos[1] + m2 * rhs.origin.pos[1]) / total_mass,
        (m1 * lhs.origin.pos[2] + m2 * rhs.origin.pos[2]) / total_mass,
    )

    dx1 = lhs.origin.pos[0] - com[0]
    dy1 = lhs.origin.pos[1] - com[1]
    dz1 = lhs.origin.pos[2] - com[2]
    dx2 = rhs.origin.pos[0] - com[0]
    dy2 = rhs.origin.pos[1] - com[1]
    dz2 = rhs.origin.pos[2] - com[2]

    inertia = {
        "ixx": lhs.inertia.ixx + m1 * (dy1 * dy1 + dz1 * dz1) + rhs.inertia.ixx + m2 * (dy2 * dy2 + dz2 * dz2),
        "iyy": lhs.inertia.iyy + m1 * (dx1 * dx1 + dz1 * dz1) + rhs.inertia.iyy + m2 * (dx2 * dx2 + dz2 * dz2),
        "izz": lhs.inertia.izz + m1 * (dx1 * dx1 + dy1 * dy1) + rhs.inertia.izz + m2 * (dx2 * dx2 + dy2 * dy2),
        "ixy": lhs.inertia.ixy + rhs.inertia.ixy,
        "ixz": lhs.inertia.ixz + rhs.inertia.ixz,
        "iyz": lhs.inertia.iyz + rhs.inertia.iyz,
    }
    return InertialCfg(mass=total_mass, origin=PoseCfg(pos=com), inertia=inertia)


__all__ = ["JointDeleteCfg", "JointDeleteMutator"]
