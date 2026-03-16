from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from embodiment.schema.hir_v01 import (
    CollisionItem,
    FingerSpec,
    GraphDerived,
    HandHIR,
    InertialItem,
    JointSpec,
    LinkSpec,
    Pose,
    TipSpec,
    VisualItem,
    is_movable_joint_type,
)


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _child(elem: ET.Element, tag: str) -> Optional[ET.Element]:
    for c in list(elem):
        if _strip_ns(c.tag) == tag:
            return c
    return None


def _children(elem: ET.Element, tag: str) -> List[ET.Element]:
    out = []
    for c in list(elem):
        if _strip_ns(c.tag) == tag:
            out.append(c)
    return out


def _parse_vec3(text: Optional[str], default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    if not text:
        return default
    vals = text.strip().split()
    if len(vals) != 3:
        return default
    try:
        return float(vals[0]), float(vals[1]), float(vals[2])
    except ValueError:
        return default


def _parse_origin(origin_elem: Optional[ET.Element]) -> Pose:
    if origin_elem is None:
        return Pose()
    return Pose(
        xyz=_parse_vec3(origin_elem.attrib.get("xyz"), (0.0, 0.0, 0.0)),
        rpy=_parse_vec3(origin_elem.attrib.get("rpy"), (0.0, 0.0, 0.0)),
    )


def _parse_mesh_scale(text: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not text:
        return None
    vals = _parse_vec3(text, (1.0, 1.0, 1.0))
    return vals


def _parse_geometry_item(geom_elem: Optional[ET.Element], pose: Pose, is_visual: bool):
    if geom_elem is None:
        return VisualItem(geom_type="unknown", pose=pose) if is_visual else CollisionItem(geom_type="unknown", pose=pose)

    box = _child(geom_elem, "box")
    if box is not None:
        size = _parse_vec3(box.attrib.get("size"))
        return VisualItem(geom_type="box", pose=pose, size=size) if is_visual else CollisionItem(geom_type="box", pose=pose, size=size)

    sphere = _child(geom_elem, "sphere")
    if sphere is not None:
        radius = float(sphere.attrib.get("radius", "0"))
        return VisualItem(geom_type="sphere", pose=pose, radius=radius) if is_visual else CollisionItem(geom_type="sphere", pose=pose, radius=radius)

    capsule = _child(geom_elem, "capsule")
    if capsule is not None:
        radius = float(capsule.attrib.get("radius", "0"))
        length = float(capsule.attrib.get("length", "0"))
        if is_visual:
            return VisualItem(geom_type="capsule", pose=pose, radius=radius, length=length)
        return CollisionItem(geom_type="capsule", pose=pose, radius=radius, length=length)

    cylinder = _child(geom_elem, "cylinder")
    if cylinder is not None:
        radius = float(cylinder.attrib.get("radius", "0"))
        length = float(cylinder.attrib.get("length", "0"))
        if is_visual:
            return VisualItem(geom_type="cylinder", pose=pose, radius=radius, length=length)
        return CollisionItem(geom_type="cylinder", pose=pose, radius=radius, length=length)

    mesh = _child(geom_elem, "mesh")
    if mesh is not None:
        if is_visual:
            return VisualItem(
                geom_type="mesh",
                pose=pose,
                mesh_file=mesh.attrib.get("filename"),
                mesh_scale=_parse_mesh_scale(mesh.attrib.get("scale")),
            )
        return CollisionItem(
            geom_type="mesh",
            pose=pose,
            mesh_file=mesh.attrib.get("filename"),
            mesh_scale=_parse_mesh_scale(mesh.attrib.get("scale")),
        )

    return VisualItem(geom_type="unknown", pose=pose) if is_visual else CollisionItem(geom_type="unknown", pose=pose)


def _parse_inertial(inertial_elem: Optional[ET.Element]) -> Optional[InertialItem]:
    if inertial_elem is None:
        return None

    pose = _parse_origin(_child(inertial_elem, "origin"))

    mass_elem = _child(inertial_elem, "mass")
    mass = None
    if mass_elem is not None and "value" in mass_elem.attrib:
        try:
            mass = float(mass_elem.attrib["value"])
        except ValueError:
            mass = None

    inertia_elem = _child(inertial_elem, "inertia")
    inertia = None
    if inertia_elem is not None:
        try:
            inertia = (
                float(inertia_elem.attrib.get("ixx", "0")),
                float(inertia_elem.attrib.get("ixy", "0")),
                float(inertia_elem.attrib.get("ixz", "0")),
                float(inertia_elem.attrib.get("iyy", "0")),
                float(inertia_elem.attrib.get("iyz", "0")),
                float(inertia_elem.attrib.get("izz", "0")),
            )
        except ValueError:
            inertia = None

    return InertialItem(mass=mass, pose=pose, inertia=inertia)


def _infer_root_link(parent_map: Dict[str, Optional[str]], child_map: Dict[str, List[str]]) -> Optional[str]:
    roots = [k for k, v in parent_map.items() if v is None]
    if not roots:
        return None
    if len(roots) == 1:
        return roots[0]

    # Pick the root with the largest subtree to improve robustness.
    def subtree_size(root: str) -> int:
        total = 0
        stack = [root]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            total += 1
            stack.extend(child_map.get(cur, []))
        return total

    roots_sorted = sorted(roots, key=lambda r: (-subtree_size(r), r))
    return roots_sorted[0]


def _infer_fingers(root_link: Optional[str], child_map: Dict[str, List[str]], joints: Dict[str, JointSpec]) -> List[FingerSpec]:
    if root_link is None:
        return []

    parent_child_to_joint = {(j.parent_link, j.child_link): j.joint_id for j in joints.values()}
    fingers: List[FingerSpec] = []
    for idx, base in enumerate(child_map.get(root_link, [])):
        chain_links = [base]
        chain_joints: List[str] = []
        visited = {root_link, base}
        cur = base
        while True:
            children = child_map.get(cur, [])
            if len(children) != 1:
                break
            nxt = children[0]
            if nxt in visited:
                break
            visited.add(nxt)
            jname = parent_child_to_joint.get((cur, nxt))
            if jname is not None:
                chain_joints.append(jname)
            chain_links.append(nxt)
            cur = nxt

        fingers.append(
            FingerSpec(
                finger_id=f"finger_{idx}",
                base_link=base,
                chain_joint_ids=chain_joints,
                chain_link_ids=chain_links,
            )
        )
    return fingers


def _infer_tips(child_map: Dict[str, List[str]], parent_map: Dict[str, Optional[str]], joints: Dict[str, JointSpec]) -> List[TipSpec]:
    child_to_joint = {j.child_link: j for j in joints.values()}
    tips: List[TipSpec] = []
    for link_name, children in child_map.items():
        if children:
            continue
        j = child_to_joint.get(link_name)
        role = "unknown_tip"
        if j is not None:
            role = "fixed_tip" if j.joint_type == "fixed" else "kinematic_tip"
        tips.append(TipSpec(tip_link=link_name, parent_link=parent_map.get(link_name), tip_role=role))
    return tips


def parse_urdf_to_hir(
    urdf_path: str,
    hand_id: Optional[str] = None,
    family: str = "unknown_family",
    handedness: str = "unknown",
    source_urdf: Optional[str] = None,
) -> HandHIR:
    """Parse one URDF file into HIR v0.1.

    Parsing strategy is topology-first and name-agnostic.
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    handedness_final = handedness if handedness in {"left", "right", "unknown"} else "unknown"
    hir = HandHIR(
        hand_id=hand_id or Path(urdf_path).stem,
        family=family,
        handedness=handedness_final,
    )
    hir.metadata["source_urdf"] = source_urdf or str(urdf_path)
    hir.metadata["embodiment_name"] = f"{family}_{hir.hand_id}"

    for elem in list(root):
        if _strip_ns(elem.tag) != "link":
            continue
        link_name = elem.attrib.get("name", "")
        if not link_name:
            continue
        link_spec = LinkSpec(link_id=link_name)
        for c in _children(elem, "collision"):
            pose = _parse_origin(_child(c, "origin"))
            link_spec.collisions.append(_parse_geometry_item(_child(c, "geometry"), pose=pose, is_visual=False))
        for v in _children(elem, "visual"):
            pose = _parse_origin(_child(v, "origin"))
            link_spec.visuals.append(_parse_geometry_item(_child(v, "geometry"), pose=pose, is_visual=True))
        link_spec.inertial = _parse_inertial(_child(elem, "inertial"))
        hir.links[link_name] = link_spec

    parent_map: Dict[str, Optional[str]] = {k: None for k in hir.links.keys()}
    child_map: Dict[str, List[str]] = {k: [] for k in hir.links.keys()}

    for elem in list(root):
        if _strip_ns(elem.tag) != "joint":
            continue
        joint_name = elem.attrib.get("name", "unknown_joint")
        joint_type = elem.attrib.get("type", "unknown")
        parent_elem = _child(elem, "parent")
        child_elem = _child(elem, "child")
        if parent_elem is None or child_elem is None:
            continue

        parent_link = parent_elem.attrib.get("link", "")
        child_link = child_elem.attrib.get("link", "")
        if parent_link not in hir.links or child_link not in hir.links:
            # Skip broken references in parser stage; validator will catch remaining issues.
            continue

        axis_elem = _child(elem, "axis")
        axis_local = _parse_vec3(axis_elem.attrib.get("xyz") if axis_elem is not None else None)

        limit_elem = _child(elem, "limit")
        lower = float(limit_elem.attrib["lower"]) if limit_elem is not None and "lower" in limit_elem.attrib else None
        upper = float(limit_elem.attrib["upper"]) if limit_elem is not None and "upper" in limit_elem.attrib else None
        effort = float(limit_elem.attrib["effort"]) if limit_elem is not None and "effort" in limit_elem.attrib else None
        velocity = float(limit_elem.attrib["velocity"]) if limit_elem is not None and "velocity" in limit_elem.attrib else None

        js = JointSpec(
            joint_id=joint_name,
            joint_type=joint_type if joint_type in {"revolute", "prismatic", "fixed", "continuous", "planar", "floating"} else "unknown",
            parent_link=parent_link,
            child_link=child_link,
            pose_parent_to_joint=_parse_origin(_child(elem, "origin")),
            axis_local=axis_local,
            limit_lower=lower,
            limit_upper=upper,
            effort_limit=effort,
            velocity_limit=velocity,
        )
        hir.joints[joint_name] = js

        parent_map[child_link] = parent_link
        child_map[parent_link].append(child_link)

    hir.root_link = _infer_root_link(parent_map=parent_map, child_map=child_map)
    if hir.root_link is None:
        hir.metadata["warning_no_root"] = "1"
    else:
        roots = [k for k, v in parent_map.items() if v is None]
        if len(roots) > 1:
            hir.metadata["warning_multi_root"] = ",".join(sorted(roots))

    hir.fingers = _infer_fingers(root_link=hir.root_link, child_map=child_map, joints=hir.joints)
    hir.tips = _infer_tips(child_map=child_map, parent_map=parent_map, joints=hir.joints)

    movable = sorted([j.joint_id for j in hir.joints.values() if is_movable_joint_type(j.joint_type)])
    joint_name_to_joint_i = {name: i for i, name in enumerate(movable)}

    hir.graph = GraphDerived(
        root_link=hir.root_link,
        dof_count=len(joint_name_to_joint_i),
        joint_name_to_joint_i=joint_name_to_joint_i,
        parent_map=parent_map,
        child_map=child_map,
    )

    return hir
