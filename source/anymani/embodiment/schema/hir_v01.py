from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

GeomType = Literal["box", "sphere", "capsule", "cylinder", "mesh", "unknown"]
JointType = Literal["revolute", "prismatic", "fixed", "continuous", "planar", "floating", "unknown"]


def is_movable_joint_type(joint_type: str) -> bool:
    """Return whether a joint contributes DoF in the current MVP."""
    return joint_type in {"revolute", "prismatic", "continuous"}


@dataclass
class Pose:
    """Rigid transform represented by xyz + rpy in parent frame."""

    xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class CollisionItem:
    """Collision geometry item under one link."""

    geom_type: GeomType = "unknown"
    pose: Pose = field(default_factory=Pose)
    # primitive parameters
    size: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    length: Optional[float] = None
    # mesh parameters
    mesh_file: Optional[str] = None
    mesh_scale: Optional[Tuple[float, float, float]] = None


@dataclass
class VisualItem:
    """Visual geometry item under one link."""

    geom_type: GeomType = "unknown"
    pose: Pose = field(default_factory=Pose)
    size: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    length: Optional[float] = None
    mesh_file: Optional[str] = None
    mesh_scale: Optional[Tuple[float, float, float]] = None


@dataclass
class InertialItem:
    """Inertial parameters of one link."""

    mass: Optional[float] = None
    pose: Pose = field(default_factory=Pose)
    # (ixx, ixy, ixz, iyy, iyz, izz)
    inertia: Optional[Tuple[float, float, float, float, float, float]] = None


@dataclass
class LinkSpec:
    """Link-level data in HIR."""

    link_id: str
    collisions: List[CollisionItem] = field(default_factory=list)
    visuals: List[VisualItem] = field(default_factory=list)
    inertial: Optional[InertialItem] = None


@dataclass
class JointSpec:
    """Joint-level data in HIR."""

    joint_id: str
    joint_type: JointType = "unknown"
    parent_link: str = ""
    child_link: str = ""
    pose_parent_to_joint: Pose = field(default_factory=Pose)
    axis_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    limit_lower: Optional[float] = None
    limit_upper: Optional[float] = None
    effort_limit: Optional[float] = None
    velocity_limit: Optional[float] = None


@dataclass
class FingerSpec:
    """One finger chain inferred from the link graph."""

    finger_id: str
    base_link: str
    chain_joint_ids: List[str] = field(default_factory=list)
    chain_link_ids: List[str] = field(default_factory=list)


@dataclass
class TipSpec:
    """Tip semantic object. Tip can be kinematic or fixed-tip child."""

    tip_link: str
    parent_link: Optional[str] = None
    tip_role: Optional[str] = None


@dataclass
class GraphDerived:
    """Derived graph-level structures for quick downstream use."""

    root_link: Optional[str] = None
    dof_count: int = 0
    joint_name_to_joint_i: Dict[str, int] = field(default_factory=dict)
    parent_map: Dict[str, Optional[str]] = field(default_factory=dict)
    child_map: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class HandHIR:
    """Hand Intermediate Representation (HIR) v0.1."""

    hir_version: str = "0.1"
    hand_id: str = "unknown_hand"
    family: str = "unknown_family"
    handedness: Literal["left", "right", "unknown"] = "unknown"
    root_link: Optional[str] = None

    links: Dict[str, LinkSpec] = field(default_factory=dict)
    joints: Dict[str, JointSpec] = field(default_factory=dict)
    fingers: List[FingerSpec] = field(default_factory=list)
    tips: List[TipSpec] = field(default_factory=list)
    graph: GraphDerived = field(default_factory=GraphDerived)

    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
