"Parses official URDF links, joints, collision ownership, and mesh hashes without a simulator."

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from ..asset_schema_geometry import (
    AnchorSeedSemanticsCfg,
    CollisionComponentSemanticsCfg,
    GeometryOwnerSemanticsCfg,
    HandGeometrySemanticsCfg,
    KinematicJointSemanticsCfg,
    Matrix3Flat,
)
from .official_hand import (
    CANONICAL_FINGER_SLOTS,
    CANONICAL_JOINT_SLOTS,
    OFFICIAL_MIGRATION_VERSION,
    FrameRole,
    Matrix4Flat,
    OfficialFrameSemanticsCfg,
    OfficialJointSemanticsCfg,
    OfficialMeshDigestCfg,
    Vector3,
    _identity4,
    _matmul,
    _sha256_file,
    _transform_from_rpy,
    _vector3,
)


@dataclass(frozen=True)
class _ParsedJoint:
    "Source URDF joint fields with original XML order, limits, axes, and parent-child links."

    name: str
    joint_type: Literal["fixed", "revolute"]
    parent_link: str
    child_link: str
    origin_pos_m: Vector3
    origin_rpy_rad: Vector3
    axis_local: Vector3
    lower_rad: float | None
    upper_rad: float | None
    effort_nm: float | None
    velocity_rad_s: float | None
    source_index: int


@dataclass(frozen=True)
class _ParsedCollision:
    "Source collision geometry, local transform, and carrier link before owner lowering."

    carrier_link: str
    collision_index: int
    collision_name: str | None
    geometry_kind: str
    geometry_payload: dict[str, Any]
    origin_pos_m: Vector3
    origin_rpy_rad: Vector3


@dataclass(frozen=True)
class _SourceBundle:
    "Parsed official-hand links, joints, collisions, and mesh-byte identities."

    root_name: str
    links: tuple[str, ...]
    joints: tuple[_ParsedJoint, ...]
    collisions_by_link: dict[str, tuple[_ParsedCollision, ...]]
    mesh_digests: tuple[OfficialMeshDigestCfg, ...]


def _parse_origin(element: ET.Element | None, *, context: str) -> tuple[Vector3, Vector3]:

    if element is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    pos = _vector3(_parse_float_list(element.attrib.get("xyz", "0 0 0")), field_name=f"{context}.xyz")
    rpy = _vector3(_parse_float_list(element.attrib.get("rpy", "0 0 0")), field_name=f"{context}.rpy")
    return pos, rpy


def _parse_float_list(value: str) -> tuple[float, ...]:

    try:
        return tuple(float(item) for item in value.split())
    except ValueError as exc:
        raise ValueError(f"invalid URDF float sequence {value!r}") from exc


def _required_attr(element: ET.Element | None, name: str, *, context: str) -> str:

    if element is None or not element.attrib.get(name):
        raise ValueError(f"{context} lacks required attribute {name!r}")
    return str(element.attrib[name])


def _required_float_attr(element: ET.Element, name: str, *, context: str) -> float:

    raw = _required_attr(element, name, context=context)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{context}.{name} is not numeric: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context}.{name} must be finite")
    return value


def _normalize_axis(axis: Vector3) -> Vector3:

    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1.0e-15:
        raise ValueError("revolute axis cannot be zero")
    return cast(Vector3, tuple(value / norm for value in axis))


def _detect_family(urdf_path: Path) -> Literal["allegro", "leap"]:

    if urdf_path.name == "allegro_hand_right.urdf":
        return "allegro"
    if urdf_path.name == "leap_hand_right.urdf":
        return "leap"
    raise ValueError(
        f"official bridge only accepts allegro_hand_right.urdf or leap_hand_right.urdf; got {urdf_path.name!r}"
    )


def _parse_source_urdf(urdf_path: Path) -> _SourceBundle:

    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"cannot parse official URDF {urdf_path}: {exc}") from exc
    if root.tag != "robot" or not root.attrib.get("name"):
        raise ValueError("official URDF root must be <robot name=...>")

    links = tuple(_required_attr(link, "name", context="link") for link in root.findall("link"))
    if len(links) != len(set(links)):
        raise ValueError("official URDF contains duplicate link names")
    link_set = set(links)
    parsed_joints: list[_ParsedJoint] = []
    for source_index, joint_node in enumerate(root.findall("joint")):
        name = _required_attr(joint_node, "name", context="joint")
        joint_type = joint_node.attrib.get("type")
        if joint_type not in {"fixed", "revolute"}:
            raise ValueError(f"official URDF joint {name!r} has unsupported type {joint_type!r}")
        parent_node = joint_node.find("parent")
        child_node = joint_node.find("child")
        parent_link = _required_attr(parent_node, "link", context=f"joint {name}.parent")
        child_link = _required_attr(child_node, "link", context=f"joint {name}.child")
        if parent_link not in link_set or child_link not in link_set:
            raise ValueError(f"joint {name!r} references an undeclared link")
        origin_pos, origin_rpy = _parse_origin(joint_node.find("origin"), context=f"joint {name}.origin")
        axis_node = joint_node.find("axis")
        axis = _vector3(
            _parse_float_list(axis_node.attrib.get("xyz", "0 0 1")) if axis_node is not None else (0.0, 0.0, 1.0),
            field_name=f"joint {name}.axis",
        )
        axis = _normalize_axis(axis) if joint_type == "revolute" else (0.0, 0.0, 1.0)
        limit_node = joint_node.find("limit")
        if joint_type == "revolute" and limit_node is None:
            raise ValueError(f"revolute joint {name!r} lacks <limit>")
        lower = upper = effort = velocity = None
        if limit_node is not None:
            lower = _required_float_attr(limit_node, "lower", context=f"joint {name}.limit")
            upper = _required_float_attr(limit_node, "upper", context=f"joint {name}.limit")
            effort = _required_float_attr(limit_node, "effort", context=f"joint {name}.limit")
            velocity = _required_float_attr(limit_node, "velocity", context=f"joint {name}.limit")
            if not lower < upper:
                raise ValueError(f"joint {name!r} has invalid limits ({lower}, {upper})")
        parsed_joints.append(
            _ParsedJoint(
                name=name,
                joint_type=cast(Literal["fixed", "revolute"], joint_type),
                parent_link=parent_link,
                child_link=child_link,
                origin_pos_m=origin_pos,
                origin_rpy_rad=origin_rpy,
                axis_local=axis,
                lower_rad=lower,
                upper_rad=upper,
                effort_nm=effort,
                velocity_rad_s=velocity,
                source_index=source_index,
            )
        )

    collisions_by_link: dict[str, tuple[_ParsedCollision, ...]] = {}
    mesh_uri_order: list[str] = []
    mesh_path_by_uri: dict[str, Path] = {}
    for link_node in root.findall("link"):
        link_name = _required_attr(link_node, "name", context="link")
        collisions: list[_ParsedCollision] = []
        for collision_index, collision_node in enumerate(link_node.findall("collision")):
            origin_pos, origin_rpy = _parse_origin(
                collision_node.find("origin"), context=f"link {link_name}.collision[{collision_index}].origin"
            )
            geometry_node = collision_node.find("geometry")
            if geometry_node is None:
                raise ValueError(f"link {link_name!r} collision[{collision_index}] lacks <geometry>")
            kind, payload, mesh_uri = _parse_geometry(
                geometry_node,
                context=f"link {link_name}.collision[{collision_index}].geometry",
            )
            if mesh_uri is not None:
                _record_mesh_uri(mesh_uri, urdf_path=urdf_path, order=mesh_uri_order, paths=mesh_path_by_uri)
                payload["resolved_path"] = str(mesh_path_by_uri[mesh_uri])
            collisions.append(
                _ParsedCollision(
                    carrier_link=link_name,
                    collision_index=collision_index,
                    collision_name=collision_node.attrib.get("name"),
                    geometry_kind=kind,
                    geometry_payload=payload,
                    origin_pos_m=origin_pos,
                    origin_rpy_rad=origin_rpy,
                )
            )
        collisions_by_link[link_name] = tuple(collisions)

        # visual meshes are part of source identity even when only collision geometry is consumed.
        for visual_node in link_node.findall("visual"):
            geometry_node = visual_node.find("geometry")
            if geometry_node is None:
                continue
            _, _, mesh_uri = _parse_geometry(
                geometry_node,
                context=f"link {link_name}.visual.geometry",
                allow_non_mesh=True,
            )
            if mesh_uri is not None:
                _record_mesh_uri(mesh_uri, urdf_path=urdf_path, order=mesh_uri_order, paths=mesh_path_by_uri)

    mesh_digests = tuple(_mesh_digest(uri, mesh_path_by_uri[uri]) for uri in mesh_uri_order)
    return _SourceBundle(
        root_name=str(root.attrib["name"]),
        links=links,
        joints=tuple(parsed_joints),
        collisions_by_link=collisions_by_link,
        mesh_digests=mesh_digests,
    )


def _parse_geometry(
    geometry_node: ET.Element,
    *,
    context: str,
    allow_non_mesh: bool = False,
) -> tuple[str, dict[str, Any], str | None]:

    children = list(geometry_node)
    if len(children) != 1:
        raise ValueError(f"{context} must contain exactly one primitive/mesh")
    geometry = children[0]
    if geometry.tag == "box":
        size = _vector3(
            _parse_float_list(_required_attr(geometry, "size", context=context)), field_name=f"{context}.size"
        )
        return "box", {"type": "box", "size": size}, None
    if geometry.tag == "cylinder":
        radius = _required_float_attr(geometry, "radius", context=context)
        length = _required_float_attr(geometry, "length", context=context)
        return "cylinder", {"type": "cylinder", "radius": radius, "length": length}, None
    if geometry.tag == "sphere":
        radius = _required_float_attr(geometry, "radius", context=context)
        return "sphere", {"type": "sphere", "radius": radius}, None
    if geometry.tag == "mesh":
        uri = _required_attr(geometry, "filename", context=context)
        scale = _vector3(
            _parse_float_list(geometry.attrib.get("scale", "1 1 1")),
            field_name=f"{context}.scale",
        )
        return "mesh", {"type": "mesh", "file_path": uri, "source_uri": uri, "scale": scale}, uri
    if allow_non_mesh:
        raise ValueError(f"{context} visual geometry kind {geometry.tag!r} is unsupported")
    raise ValueError(f"{context} collision geometry kind {geometry.tag!r} is unsupported")


def _record_mesh_uri(uri: str, *, urdf_path: Path, order: list[str], paths: dict[str, Path]) -> None:

    if uri.startswith("package://") or uri.startswith("model://") or uri.startswith("file://"):
        raise ValueError(f"official mesh URI must be local relative path, got {uri!r}")
    candidate = (urdf_path.parent / PurePosixPath(uri)).resolve(strict=False)
    if not candidate.is_file():
        raise FileNotFoundError(f"mesh referenced by official URDF does not exist: {uri!r} -> {candidate}")
    if uri not in paths:
        order.append(uri)
        paths[uri] = candidate


def _mesh_digest(uri: str, path: Path) -> OfficialMeshDigestCfg:

    return OfficialMeshDigestCfg(
        source_uri=uri,
        resolved_path=str(path),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _validate_expected_topology(source: _SourceBundle, *, family: Literal["allegro", "leap"]) -> None:

    expected_robot_name = "allegro_right" if family == "allegro" else "leap_right"
    if source.root_name != expected_robot_name:
        raise ValueError(
            f"{family} official robot name mismatch: expected {expected_robot_name!r}, got {source.root_name!r}"
        )
    expected = _expected_topology(family)
    if set(source.links) != set(expected["links"]):
        raise ValueError(f"{family} official link set mismatch")
    actual_edges = {(joint.name, joint.parent_link, joint.child_link, joint.joint_type) for joint in source.joints}
    expected_edges = set(expected["edges"])
    if actual_edges != expected_edges:
        raise ValueError(f"{family} official joint parent-child topology mismatch")
    if len(source.joints) != len(expected_edges):
        raise ValueError(f"{family} official joint count mismatch")


def _expected_topology(family: Literal["allegro", "leap"]) -> dict[str, Any]:

    if family == "allegro":
        links = (
            "base_link",
            "palm",
            "wrist",
            *(f"link_{index}.0" for index in range(16)),
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_15.0_tip",
        )
        edges: list[tuple[str, str, str, str]] = [
            ("palm_joint", "base_link", "palm", "fixed"),
            ("wrist_joint", "palm", "wrist", "fixed"),
        ]
        for finger_start in (0, 4, 8):
            edges.extend(
                (
                    (f"joint_{finger_start}.0", "base_link", f"link_{finger_start}.0", "revolute"),
                    (f"joint_{finger_start + 1}.0", f"link_{finger_start}.0", f"link_{finger_start + 1}.0", "revolute"),
                    (
                        f"joint_{finger_start + 2}.0",
                        f"link_{finger_start + 1}.0",
                        f"link_{finger_start + 2}.0",
                        "revolute",
                    ),
                    (
                        f"joint_{finger_start + 3}.0",
                        f"link_{finger_start + 2}.0",
                        f"link_{finger_start + 3}.0",
                        "revolute",
                    ),
                    (
                        f"joint_{finger_start + 3}.0_tip",
                        f"link_{finger_start + 3}.0",
                        f"link_{finger_start + 3}.0_tip",
                        "fixed",
                    ),
                )
            )
        edges.extend(
            (
                ("joint_12.0", "base_link", "link_12.0", "revolute"),
                ("joint_13.0", "link_12.0", "link_13.0", "revolute"),
                ("joint_14.0", "link_13.0", "link_14.0", "revolute"),
                ("joint_15.0", "link_14.0", "link_15.0", "revolute"),
                ("joint_15.0_tip", "link_15.0", "link_15.0_tip", "fixed"),
            )
        )
        return {"links": links, "edges": tuple(edges)}
    links = (
        "base",
        "palm_lower",
        "mcp_joint",
        "pip",
        "dip",
        "fingertip",
        "mcp_joint_2",
        "pip_2",
        "dip_2",
        "fingertip_2",
        "mcp_joint_3",
        "pip_3",
        "dip_3",
        "fingertip_3",
        "thumb_temp_base",
        "thumb_pip",
        "thumb_dip",
        "thumb_fingertip",
        "thumb_tip_head",
        "index_tip_head",
        "middle_tip_head",
        "ring_tip_head",
    )
    edges: list[tuple[str, str, str, str]] = [
        ("base_joint", "base", "palm_lower", "fixed"),
        ("a_0", "mcp_joint", "pip", "revolute"),
        ("a_1", "palm_lower", "mcp_joint", "revolute"),
        ("a_2", "pip", "dip", "revolute"),
        ("a_3", "dip", "fingertip", "revolute"),
        ("a_4", "mcp_joint_2", "pip_2", "revolute"),
        ("a_5", "palm_lower", "mcp_joint_2", "revolute"),
        ("a_6", "pip_2", "dip_2", "revolute"),
        ("a_7", "dip_2", "fingertip_2", "revolute"),
        ("a_8", "mcp_joint_3", "pip_3", "revolute"),
        ("a_9", "palm_lower", "mcp_joint_3", "revolute"),
        ("a_10", "pip_3", "dip_3", "revolute"),
        ("a_11", "dip_3", "fingertip_3", "revolute"),
        ("a_12", "palm_lower", "thumb_temp_base", "revolute"),
        ("a_13", "thumb_temp_base", "thumb_pip", "revolute"),
        ("a_14", "thumb_pip", "thumb_dip", "revolute"),
        ("a_15", "thumb_dip", "thumb_fingertip", "revolute"),
        ("thumb_tip", "thumb_fingertip", "thumb_tip_head", "fixed"),
        ("index_tip", "fingertip", "index_tip_head", "fixed"),
        ("middle_tip", "fingertip_2", "middle_tip_head", "fixed"),
        ("ring_tip", "fingertip_3", "ring_tip_head", "fixed"),
    ]
    return {"links": links, "edges": edges}


def _canonical_source_maps(family: Literal["allegro", "leap"]) -> dict[str, dict[str, str]]:

    if family == "allegro":
        source_by_finger = {
            "index": ("joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"),
            "middle": ("joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"),
            "ring": ("joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"),
            "thumb": ("joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"),
        }
        child_by_finger = {
            "index": ("link_0.0", "link_1.0", "link_2.0", "link_3.0"),
            "middle": ("link_4.0", "link_5.0", "link_6.0", "link_7.0"),
            "ring": ("link_8.0", "link_9.0", "link_10.0", "link_11.0"),
            "thumb": ("link_12.0", "link_13.0", "link_14.0", "link_15.0"),
        }
    else:
        source_by_finger = {
            "index": ("a_1", "a_0", "a_2", "a_3"),
            "middle": ("a_5", "a_4", "a_6", "a_7"),
            "ring": ("a_9", "a_8", "a_10", "a_11"),
            "thumb": ("a_12", "a_13", "a_14", "a_15"),
        }
        child_by_finger = {
            "index": ("mcp_joint", "pip", "dip", "fingertip"),
            "middle": ("mcp_joint_2", "pip_2", "dip_2", "fingertip_2"),
            "ring": ("mcp_joint_3", "pip_3", "dip_3", "fingertip_3"),
            "thumb": ("thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip"),
        }
    slot_to_source = {
        f"{finger}_j{depth}": source_by_finger[finger][depth] for depth in range(4) for finger in CANONICAL_FINGER_SLOTS
    }
    source_to_slot = {source: slot for slot, source in slot_to_source.items()}
    slot_to_child = {
        f"{finger}_j{depth}": child_by_finger[finger][depth] for depth in range(4) for finger in CANONICAL_FINGER_SLOTS
    }
    return {"slot_to_source": slot_to_source, "source_to_slot": source_to_slot, "slot_to_child": slot_to_child}


def _topological_joints(source: _SourceBundle, source_to_slot: Mapping[str, str]) -> tuple[_ParsedJoint, ...]:

    root_link = _find_root_link(source)
    remaining = list(source.joints)
    ordered: list[_ParsedJoint] = []
    available = {root_link}
    while remaining:
        ready = [joint for joint in remaining if joint.parent_link in available]
        if not ready:
            names = [joint.name for joint in remaining]
            raise ValueError(f"official native tree cannot be topologically ordered; pending joints={names}")
        ready.sort(key=lambda joint: _joint_sort_key(joint, source_to_slot))
        next_joint = ready[0]
        remaining.remove(next_joint)
        ordered.append(next_joint)
        available.add(next_joint.child_link)
    if set(available) != set(source.links):
        raise ValueError("official native tree does not reach every declared link")
    return tuple(ordered)


def _joint_sort_key(joint: _ParsedJoint, source_to_slot: Mapping[str, str]) -> tuple[int, int]:

    if joint.joint_type == "revolute":
        return (CANONICAL_JOINT_SLOTS.index(source_to_slot[joint.name]), joint.source_index)
    if joint.name in {"base_joint", "palm_joint", "wrist_joint"}:
        return (-100, joint.source_index)
    return (1000, joint.source_index)


def _build_active_joint_records(
    source: _SourceBundle,
    *,
    family: Literal["allegro", "leap"],
    canonical_maps: Mapping[str, Mapping[str, str]],
) -> tuple[OfficialJointSemanticsCfg, ...]:

    by_name = {joint.name: joint for joint in source.joints}
    result: list[OfficialJointSemanticsCfg] = []
    for slot in CANONICAL_JOINT_SLOTS:
        source_name = canonical_maps["slot_to_source"][slot]
        joint = by_name.get(source_name)
        if joint is None or joint.joint_type != "revolute":
            raise ValueError(f"{family} official missing canonical revolute joint {slot!r} -> {source_name!r}")
        finger, depth_text = slot.rsplit("_j", 1)
        depth = int(depth_text)
        if (
            joint.lower_rad is None
            or joint.upper_rad is None
            or joint.effort_nm is None
            or joint.velocity_rad_s is None
        ):
            raise ValueError(f"official revolute joint {source_name!r} lacks complete limit attributes")
        result.append(
            OfficialJointSemanticsCfg(
                canonical_name=slot,
                finger_name=finger,
                depth=depth,
                source_name=source_name,
                parent_link=joint.parent_link,
                child_link=joint.child_link,
                origin_pos_m=joint.origin_pos_m,
                origin_rpy_rad=joint.origin_rpy_rad,
                axis_local=joint.axis_local,
                lower_rad=joint.lower_rad,
                upper_rad=joint.upper_rad,
                effort_nm=joint.effort_nm,
                velocity_rad_s=joint.velocity_rad_s,
            )
        )
    return tuple(result)


def _build_geometry_semantics(
    source: _SourceBundle,
    *,
    family: Literal["allegro", "leap"],
    ordered_joints: Sequence[_ParsedJoint],
    canonical_maps: Mapping[str, Mapping[str, str]],
    semantic_rotation: tuple[float, ...],
    semantic_translation: Vector3,
    asset_id: str,
) -> HandGeometrySemanticsCfg:

    root_link = _find_root_link(source)
    active_names = tuple(canonical_maps["slot_to_source"][slot] for slot in CANONICAL_JOINT_SLOTS)
    active_index_by_source = {source_name: index for index, source_name in enumerate(active_names)}
    kinematic = tuple(
        KinematicJointSemanticsCfg(
            joint_name=joint.name,
            joint_type=joint.joint_type,
            parent_link=joint.parent_link,
            child_link=joint.child_link,
            origin_pos_m=joint.origin_pos_m,
            origin_rpy_rad=joint.origin_rpy_rad,
            axis_local=joint.axis_local,
            active_joint_index=active_index_by_source.get(joint.name),
        )
        for joint in ordered_joints
    )

    tip_bodies, _ = _tip_and_marker_maps(family)


    palm_link = _palm_collision_link(family)
    owner_id_by_slot = {slot: f"joint/{slot}" for slot in CANONICAL_JOINT_SLOTS}
    tip_owner_id_by_finger = {finger: f"tip/{finger}" for finger in CANONICAL_FINGER_SLOTS}
    owners: list[GeometryOwnerSemanticsCfg] = []
    components: list[CollisionComponentSemanticsCfg] = []


    palm_component_ids: list[str] = []
    for collision in source.collisions_by_link[palm_link]:
        component = _collision_component(
            collision,
            owner_id="palm",
            source_joint_name=None,
            component_id=f"palm/{palm_link}/collision/{collision.collision_index}",
        )
        components.append(component)
        palm_component_ids.append(component.component_id)
    owners.append(
        GeometryOwnerSemanticsCfg(
            owner_id="palm",
            owner_index=0,
            role="palm",
            parent_owner_id=None,
            finger_name=None,
            joint_name=None,
            reference_link=palm_link,
            component_ids=tuple(palm_component_ids),
        )
    )


    owner_index_by_id = {"palm": 0}
    for slot_index, slot in enumerate(CANONICAL_JOINT_SLOTS, start=1):
        source_name = canonical_maps["slot_to_source"][slot]
        source_joint = next(joint for joint in source.joints if joint.name == source_name)
        owner_id = owner_id_by_slot[slot]
        finger = slot.split("_j", 1)[0]
        depth = int(slot.rsplit("_j", 1)[1])
        parent_owner = "palm" if depth == 0 else owner_id_by_slot[f"{finger}_j{depth - 1}"]
        component_ids: list[str] = []
        source_collisions = source.collisions_by_link[source_joint.child_link]
        if family == "leap" and depth == 3:


            if len(source_collisions) != 4 or any(item.geometry_kind != "box" for item in source_collisions[:3]):
                raise ValueError("LEAP terminal body must expose exactly 3 box JOINT collisions before its mesh TIP")
            source_collisions = source_collisions[:3]
        for collision in source_collisions:
            component = _collision_component(
                collision,
                owner_id=owner_id,
                source_joint_name=source_name,
                component_id=f"joint/{slot}/{source_joint.child_link}/collision/{collision.collision_index}",
            )
            components.append(component)
            component_ids.append(component.component_id)
        if not component_ids:
            raise ValueError(f"official JOINT owner {slot!r} has no collision geometry")
        owners.append(
            GeometryOwnerSemanticsCfg(
                owner_id=owner_id,
                owner_index=slot_index,
                role="joint",
                parent_owner_id=parent_owner,
                finger_name=finger,
                joint_name=source_name,
                reference_link=source_joint.child_link,
                component_ids=tuple(component_ids),
            )
        )
        owner_index_by_id[owner_id] = slot_index


    tip_start_index = 1 + len(CANONICAL_JOINT_SLOTS)
    for tip_offset, finger in enumerate(CANONICAL_FINGER_SLOTS):
        tip_link = tip_bodies[finger]
        last_slot = f"{finger}_j3"
        tip_joint = _incoming_joint(source, tip_link)
        if tip_joint is None:
            raise ValueError(f"official {family} tip body {tip_link!r} lacks incoming joint")


        if family == "allegro" and tip_joint.joint_type != "fixed":
            raise ValueError(f"official Allegro tip body {tip_link!r} must be a fixed child")
        if family == "leap" and tip_joint.joint_type != "revolute":
            raise ValueError(f"official LEAP fingertip body {tip_link!r} must be the final active child")
        component_ids = []
        owner_id = tip_owner_id_by_finger[finger]
        tip_collisions = source.collisions_by_link[tip_link]
        if family == "leap":

            if len(tip_collisions) != 4 or tip_collisions[-1].geometry_kind != "mesh":
                raise ValueError("LEAP terminal body must end with exactly one mesh TIP collision")
            tip_collisions = tip_collisions[-1:]
        for collision in tip_collisions:
            component = _collision_component(
                collision,
                owner_id=owner_id,
                source_joint_name=tip_joint.name,
                component_id=f"tip/{finger}/{tip_link}/collision/{collision.collision_index}",
            )
            components.append(component)
            component_ids.append(component.component_id)
        if not component_ids:
            raise ValueError(f"official TIP owner {finger!r} has no collision geometry")
        owners.append(
            GeometryOwnerSemanticsCfg(
                owner_id=owner_id,
                owner_index=tip_start_index + tip_offset,
                role="tip",
                parent_owner_id=owner_id_by_slot[last_slot],
                finger_name=finger,
                joint_name=None,
                reference_link=tip_link if family == "leap" else canonical_maps["slot_to_child"][last_slot],
                component_ids=tuple(component_ids),
            )
        )

    active_limits = tuple(
        (float(joint.lower_rad), float(joint.upper_rad))
        for joint in _build_active_joint_records(source, family=family, canonical_maps=canonical_maps)
    )
    anchor_seeds = _build_anchor_seeds(
        source,
        root_link=root_link,
        kinematic_joints=kinematic,
        canonical_maps=canonical_maps,
    )
    payload = {
        "schema_version": "1.0.0",
        "migration_version": OFFICIAL_MIGRATION_VERSION,
        "source_kind": "official",
        "asset_id": asset_id,
        "asset_name": source.root_name,
        "topology_key": f"official_{family}_right",
        "family": family,
        "handedness": "right",
        "units": {"length": "m", "angle": "rad"},
        "asset_to_hand_rotation": semantic_rotation,
        "asset_to_hand_translation_m": semantic_translation,
        "palm_link": root_link,
        "palm_origin_pos_m": (0.0, 0.0, 0.0),
        "palm_origin_rpy_rad": (0.0, 0.0, 0.0),
        "kinematic_joints": kinematic,
        "active_joint_names": active_names,
        "q_home_rad": (0.0,) * len(active_names),
        "joint_limits_rad": active_limits,
        "owners": tuple(owners),
        "components": tuple(components),
        "anchor_seeds": anchor_seeds,
    }

    from ..asset_schema_geometry import _content_hash

    return HandGeometrySemanticsCfg(**payload, content_hash=_content_hash(payload))


def _build_anchor_seeds(
    source: _SourceBundle,
    *,
    root_link: str,
    kinematic_joints: Sequence[KinematicJointSemanticsCfg],
    canonical_maps: Mapping[str, Mapping[str, str]],
) -> tuple[AnchorSeedSemanticsCfg, ...]:

    link_home: dict[str, tuple[tuple[float, ...], ...]] = {root_link: _identity4()}
    for joint in kinematic_joints:
        parent_home = link_home.get(joint.parent_link)
        if parent_home is None:
            raise ValueError(f"cannot derive official anchor: parent link {joint.parent_link!r} is unavailable")
        link_home[joint.child_link] = _matmul(
            parent_home,
            _transform_from_rpy(joint.origin_rpy_rad, joint.origin_pos_m),
        )
    by_source_name = {joint.joint_name: joint for joint in kinematic_joints if joint.joint_type == "revolute"}
    seeds: list[AnchorSeedSemanticsCfg] = []
    for finger in CANONICAL_FINGER_SLOTS:
        slot = f"{finger}_j0"
        source_name = canonical_maps["slot_to_source"][slot]
        joint = by_source_name[source_name]
        parent_home = link_home[joint.parent_link]
        joint_frame = _matmul(
            parent_home,
            _transform_from_rpy(joint.origin_rpy_rad, joint.origin_pos_m),
        )
        seeds.append(
            AnchorSeedSemanticsCfg(
                seed_id=f"finger/{finger}/first-active",
                finger_name=finger,
                first_active_joint_name=source_name,
                support_owner_id="palm",
                position_a_m=(joint_frame[0][3], joint_frame[1][3], joint_frame[2][3]),
                rotation_a=cast(Matrix3Flat, tuple(value for row in joint_frame[:3] for value in row[:3])),
            )
        )
    return tuple(seeds)


def _collision_component(
    collision: _ParsedCollision,
    *,
    owner_id: str,
    source_joint_name: str | None,
    component_id: str,
) -> CollisionComponentSemanticsCfg:

    payload = dict(collision.geometry_payload)
    if collision.geometry_kind == "mesh":


        payload["file_path"] = payload["resolved_path"]
    return CollisionComponentSemanticsCfg(
        component_id=component_id,
        owner_id=owner_id,
        carrier_link=collision.carrier_link,
        collision_index=collision.collision_index,
        collision_name=collision.collision_name,
        geometry_kind=collision.geometry_kind,
        geometry_payload=payload,
        origin_pos_m=collision.origin_pos_m,
        origin_rpy_rad=collision.origin_rpy_rad,
        source_joint_name=source_joint_name,
    )


def _build_frames(
    source: _SourceBundle,
    *,
    family: Literal["allegro", "leap"],
    root_link: str,
    link_transforms: Mapping[str, Matrix4Flat],
    canonical_maps: Mapping[str, Mapping[str, str]],
) -> tuple[OfficialFrameSemanticsCfg, ...]:

    frames: list[OfficialFrameSemanticsCfg] = []
    frames.append(
        OfficialFrameSemanticsCfg(
            frame_name="base",
            role="base",
            source_link=root_link,
            parent_link=None,
            source_joint_name=None,
            origin_pos_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            pose_root=link_transforms[root_link],
            finger_name=None,
            has_collision=bool(source.collisions_by_link[root_link]),
        )
    )
    palm_link = _palm_frame_link(family)
    palm_joint = _incoming_joint(source, palm_link)
    if palm_joint is None and palm_link != root_link:
        raise ValueError(f"official palm link {palm_link!r} lacks incoming joint")
    frames.append(
        _frame_from_link(
            source,
            link_transforms,
            frame_name="palm",
            role="palm",
            source_link=palm_link,
            finger_name=None,
        )
    )
    tips, markers = _tip_and_marker_maps(family)
    for finger in CANONICAL_FINGER_SLOTS:
        frames.append(
            _frame_from_link(
                source,
                link_transforms,
                frame_name=f"{finger}_terminal_body",
                role="terminal_body",
                source_link=tips[finger],
                finger_name=finger,
            )
        )


    for finger, marker_link in markers.items():
        frames.append(
            _frame_from_link(
                source,
                link_transforms,
                frame_name=f"{finger}_marker",
                role="marker",
                source_link=marker_link,
                finger_name=finger,
            )
        )
    return tuple(frames)


def _frame_from_link(
    source: _SourceBundle,
    link_transforms: Mapping[str, Matrix4Flat],
    *,
    frame_name: str,
    role: FrameRole,
    source_link: str,
    finger_name: str | None,
) -> OfficialFrameSemanticsCfg:

    incoming = _incoming_joint(source, source_link)
    if incoming is None:
        raise ValueError(f"official frame source link {source_link!r} lacks incoming joint")
    has_collision = bool(source.collisions_by_link[source_link])
    if role == "marker" and has_collision:
        raise ValueError(f"official marker frame {frame_name!r} unexpectedly carries collision geometry")
    return OfficialFrameSemanticsCfg(
        frame_name=frame_name,
        role=role,
        source_link=source_link,
        parent_link=incoming.parent_link,
        source_joint_name=incoming.name,
        origin_pos_m=incoming.origin_pos_m,
        origin_rpy_rad=incoming.origin_rpy_rad,
        pose_root=link_transforms[source_link],
        finger_name=finger_name,
        has_collision=has_collision,
    )


def _tip_and_marker_maps(family: Literal["allegro", "leap"]) -> tuple[dict[str, str], dict[str, str]]:

    if family == "allegro":
        return (
            {
                "index": "link_3.0_tip",
                "middle": "link_7.0_tip",
                "ring": "link_11.0_tip",
                "thumb": "link_15.0_tip",
            },
            {},
        )
    return (
        {
            "index": "fingertip",
            "middle": "fingertip_2",
            "ring": "fingertip_3",
            "thumb": "thumb_fingertip",
        },
        {
            "index": "index_tip_head",
            "middle": "middle_tip_head",
            "ring": "ring_tip_head",
            "thumb": "thumb_tip_head",
        },
    )


def _palm_collision_link(family: Literal["allegro", "leap"]) -> str:

    return "base_link" if family == "allegro" else "palm_lower"


def _palm_frame_link(family: Literal["allegro", "leap"]) -> str:

    return "palm" if family == "allegro" else "palm_lower"


def _incoming_joint(source: _SourceBundle, child_link: str) -> _ParsedJoint | None:

    matches = [joint for joint in source.joints if joint.child_link == child_link]
    if len(matches) > 1:
        raise ValueError(f"official link {child_link!r} has multiple incoming joints")
    return matches[0] if matches else None


def _find_root_link(source: _SourceBundle) -> str:

    children = {joint.child_link for joint in source.joints}
    roots = [link for link in source.links if link not in children]
    if len(roots) != 1:
        raise ValueError(f"official native tree requires one root, got {roots!r}")
    return roots[0]
