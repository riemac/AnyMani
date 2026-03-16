from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from embodiment.schema.hir_v01 import HandHIR, Pose


def _fmt_vec3(v) -> str:
    return f"{v[0]} {v[1]} {v[2]}"


def _append_origin(parent: ET.Element, pose: Pose):
    ET.SubElement(parent, "origin", attrib={"xyz": _fmt_vec3(pose.xyz), "rpy": _fmt_vec3(pose.rpy)})


def _append_geometry(geom_parent: ET.Element, geom_type: str, size, radius, length, mesh_file, mesh_scale):
    if geom_type == "box" and size is not None:
        ET.SubElement(geom_parent, "box", attrib={"size": _fmt_vec3(size)})
        return
    if geom_type == "sphere" and radius is not None:
        ET.SubElement(geom_parent, "sphere", attrib={"radius": str(radius)})
        return
    if geom_type in {"capsule", "cylinder"} and radius is not None and length is not None:
        ET.SubElement(geom_parent, geom_type, attrib={"radius": str(radius), "length": str(length)})
        return
    if geom_type == "mesh" and mesh_file:
        attr = {"filename": mesh_file}
        if mesh_scale is not None:
            attr["scale"] = _fmt_vec3(mesh_scale)
        ET.SubElement(geom_parent, "mesh", attrib=attr)


def _indent(elem: ET.Element, level: int = 0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def emit_hir_to_urdf(hir: HandHIR, out_path: str):
    """Serialize HIR into a deterministic URDF file."""
    robot = ET.Element("robot", attrib={"name": hir.hand_id})

    for link_name in sorted(hir.links.keys()):
        lspec = hir.links[link_name]
        link_elem = ET.SubElement(robot, "link", attrib={"name": link_name})

        if lspec.inertial is not None:
            inertial = ET.SubElement(link_elem, "inertial")
            _append_origin(inertial, lspec.inertial.pose)
            if lspec.inertial.mass is not None:
                ET.SubElement(inertial, "mass", attrib={"value": str(lspec.inertial.mass)})
            if lspec.inertial.inertia is not None:
                ixx, ixy, ixz, iyy, iyz, izz = lspec.inertial.inertia
                ET.SubElement(
                    inertial,
                    "inertia",
                    attrib={
                        "ixx": str(ixx),
                        "ixy": str(ixy),
                        "ixz": str(ixz),
                        "iyy": str(iyy),
                        "iyz": str(iyz),
                        "izz": str(izz),
                    },
                )

        for v in lspec.visuals:
            vis = ET.SubElement(link_elem, "visual")
            _append_origin(vis, v.pose)
            geom = ET.SubElement(vis, "geometry")
            _append_geometry(
                geom_parent=geom,
                geom_type=v.geom_type,
                size=v.size,
                radius=v.radius,
                length=v.length,
                mesh_file=v.mesh_file,
                mesh_scale=v.mesh_scale,
            )

        for c in lspec.collisions:
            col = ET.SubElement(link_elem, "collision")
            _append_origin(col, c.pose)
            geom = ET.SubElement(col, "geometry")
            _append_geometry(
                geom_parent=geom,
                geom_type=c.geom_type,
                size=c.size,
                radius=c.radius,
                length=c.length,
                mesh_file=c.mesh_file,
                mesh_scale=c.mesh_scale,
            )

    for joint_name in sorted(hir.joints.keys()):
        js = hir.joints[joint_name]
        joint = ET.SubElement(robot, "joint", attrib={"name": joint_name, "type": js.joint_type})
        ET.SubElement(joint, "parent", attrib={"link": js.parent_link})
        ET.SubElement(joint, "child", attrib={"link": js.child_link})
        _append_origin(joint, js.pose_parent_to_joint)
        ET.SubElement(joint, "axis", attrib={"xyz": _fmt_vec3(js.axis_local)})

        limit_attr = {}
        if js.limit_lower is not None:
            limit_attr["lower"] = str(js.limit_lower)
        if js.limit_upper is not None:
            limit_attr["upper"] = str(js.limit_upper)
        if js.effort_limit is not None:
            limit_attr["effort"] = str(js.effort_limit)
        if js.velocity_limit is not None:
            limit_attr["velocity"] = str(js.velocity_limit)
        if limit_attr:
            ET.SubElement(joint, "limit", attrib=limit_attr)

    _indent(robot)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(out, encoding="utf-8", xml_declaration=True)
