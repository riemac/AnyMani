(
    'Build a mechanically equivalent import view for official URDFs. Isaac URDF '
    'import normalizes link names but concatenates OBJ labels such as o link_0.0 '
    'into an invalid SdfPath. Write a separate copy that adapts OBJ labels and '
    'selectively removes empty fixed reference frames with no inertial, visual, '
    'or collision content. Preserve original mesh lines (v/vt/vn/f/usemtl) and '
    'all retained joint/inertial/collision parameters. Keep source URDF/OBJ/MTL '
    'read-only and store the copy plus equivalence evidence in a separate cache.'
)

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from anymani.assets.bank.official_hand import OfficialHandSemanticsCfg

IMPORT_NAME_ADAPTER = "official-usd-safe-obj-labels-frame-lowering-v4"
"""OBJ label adapter plus selective empty-reference-frame lowering identity."""

FRAME_LOWERING_VERSION = "official-empty-reference-frame-lowering-v1"
"""Mechanical-equivalence contract for removing empty fixed reference links."""

_EMPTY_REFERENCE_FRAMES: dict[str, tuple[str, ...]] = {
    # LEAP tip_head links are four empty fixed leaves.  ``base`` is handled by
    # the root-promotion rule below because it has one retained fixed child.
    "leap": ("thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"),
    # The Allegro empty fixed chain must be removed leaf-first.
    "allegro": ("wrist", "palm"),
}
_EXPECTED_TOTAL_MASS_KG = {"leap": 0.746, "allegro": 0.9735}


def _sha(data: bytes) -> str:
    'Record original-byte digest separately from geometric equivalence.'

    return hashlib.sha256(data).hexdigest()


def _identifier(name: str) -> str:
    'Create a USD-valid identifier by changing label characters only; it does not affect geometry or physics.'

    result = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return result if result and not result[0].isdigit() else f"_{result}"


def _obj_geometry_bytes(data: str) -> bytes:
    'Ignore display group names and library paths; preserve vertex, topology, normal, UV, and material-assignment bytes.'

    return "".join(
        line for line in data.splitlines(keepends=True) if not line.startswith(("o ", "g ", "mtllib "))
    ).encode()


def _identity4() -> tuple[float, ...]:
    'Return a row-major 4x4 identity matrix.'

    return (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _matmul4(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> tuple[float, ...]:
    'Compose two row-major 4x4 rigid transforms.'

    return tuple(
        sum(lhs[row * 4 + inner] * rhs[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    )


def _origin_transform(origin: ET.Element | None) -> tuple[float, ...]:
    'Parse one URDF origin using Rz(yaw) @ Ry(pitch) @ Rx(roll).'

    if origin is None:
        xyz = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
    else:
        xyz = tuple(float(value) for value in origin.attrib.get("xyz", "0 0 0").split())
        rpy = tuple(float(value) for value in origin.attrib.get("rpy", "0 0 0").split())
    if len(xyz) != 3 or len(rpy) != 3 or not all(math.isfinite(value) for value in (*xyz, *rpy)):
        raise ValueError("official fixed-frame origin must contain finite xyz/rpy triples")
    roll, pitch, yaw = rpy
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)
    rx = (1.0, 0.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, 0.0, sx, cx, 0.0, 0.0, 0.0, 0.0, 1.0)
    ry = (cy, 0.0, sy, 0.0, 0.0, 1.0, 0.0, 0.0, -sy, 0.0, cy, 0.0, 0.0, 0.0, 0.0, 1.0)
    rz = (cz, -sz, 0.0, 0.0, sz, cz, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    rotation = _matmul4(_matmul4(rz, ry), rx)
    return rotation[:3] + (xyz[0],) + rotation[4:7] + (xyz[1],) + rotation[8:11] + (xyz[2],) + (0.0, 0.0, 0.0, 1.0)


def _link_zero_transforms(root: ET.Element, root_link: str) -> dict[str, tuple[float, ...]]:
    "Compute each link's static transform from the original root at q=0."

    children: dict[str, list[ET.Element]] = {}
    links = {str(node.attrib.get("name")) for node in root.findall("link")}
    for joint in root.findall("joint"):
        parent = _joint_link(joint, "parent")
        child = _joint_link(joint, "child")
        if parent is None or child is None:
            raise ValueError("official joint lacks parent/child link")
        children.setdefault(parent, []).append(joint)
    if root_link not in links:
        raise ValueError(f"official source root link is absent: {root_link!r}")
    transforms = {root_link: _identity4()}
    pending = [root_link]
    while pending:
        parent = pending.pop()
        for joint in children.get(parent, []):
            child = _joint_link(joint, "child")
            if child is None:
                raise ValueError("official joint child link is absent")
            if child in transforms:
                raise ValueError(f"official URDF tree repeats child link {child!r}")
            transforms[child] = _matmul4(transforms[parent], _origin_transform(joint.find("origin")))
            pending.append(child)
    if set(transforms) != links:
        raise ValueError("official URDF kinematic tree is disconnected")
    return transforms


def _has_physical_payload(link: ET.Element) -> bool:
    'Check whether a link has inertial, visual, or collision content.'

    return any(link.find(tag) is not None or bool(link.findall(tag)) for tag in ("inertial", "visual", "collision"))


def _inertial_audit(root: ET.Element, *, family: str) -> tuple[dict[str, Any], float]:
    'Extract per-link mass/COM/inertia from source and check both official total-mass anchors.'

    if family not in _EXPECTED_TOTAL_MASS_KG:
        raise ValueError(f"unsupported official inertial family: {family!r}")
    audit: dict[str, Any] = {}
    total_mass = 0.0
    for link in root.findall("link"):
        name = str(link.attrib.get("name", ""))
        inertial = link.find("inertial")
        if not name:
            raise ValueError("official URDF link lacks a name")
        if inertial is None:
            audit[name] = None
            continue
        mass_node = inertial.find("mass")
        inertia_node = inertial.find("inertia")
        if mass_node is None or inertia_node is None or mass_node.attrib.get("value") is None:
            raise ValueError(f"official inertial for {name!r} lacks mass or inertia")
        origin = inertial.find("origin")
        xyz = tuple(
            float(value)
            for value in (origin.attrib.get("xyz", "0 0 0").split() if origin is not None else (0.0, 0.0, 0.0))
        )
        rpy = tuple(
            float(value)
            for value in (origin.attrib.get("rpy", "0 0 0").split() if origin is not None else (0.0, 0.0, 0.0))
        )
        mass = float(mass_node.attrib["value"])
        inertia = {key: float(inertia_node.attrib[key]) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        if (
            len(xyz) != 3
            or len(rpy) != 3
            or not all(math.isfinite(value) for value in (*xyz, *rpy, mass, *inertia.values()))
        ):
            raise ValueError(f"official inertial for {name!r} contains non-finite values")
        audit[name] = {
            "mass_kg": mass,
            "origin_xyz_m": list(xyz),
            "origin_rpy_rad": list(rpy),
            "inertia_kg_m2": inertia,
        }
        total_mass += mass
    expected = _EXPECTED_TOTAL_MASS_KG[family]
    if abs(total_mass - expected) > 1.0e-12:
        raise ValueError(f"official {family} inertial total changed: expected={expected}, actual={total_mass}")
    return audit, expected


def _remove_named_nodes(root: ET.Element, *, link_names: tuple[str, ...], joint_names: tuple[str, ...]) -> None:
    'Remove validated link/joint nodes from the XML copy without touching the source file.'

    for link_name in link_names:
        link = next((node for node in root.findall("link") if node.attrib.get("name") == link_name), None)
        if link is None:
            raise ValueError(f"official frame-lowering link disappeared: {link_name!r}")
        root.remove(link)
    for joint_name in joint_names:
        joint = next((node for node in root.findall("joint") if node.attrib.get("name") == joint_name), None)
        if joint is None:
            raise ValueError(f"official frame-lowering joint disappeared: {joint_name!r}")
        root.remove(joint)


def _joint_link(joint: ET.Element, side: str) -> str | None:
    "Read a URDF joint's parent/child link attributes."

    node = joint.find(side)
    return None if node is None else node.attrib.get("link")


def _lower_empty_reference_frames(
    root: ET.Element,
    *,
    family: str,
    original_root_link: str,
) -> tuple[ET.Element, dict[str, Any]]:
    (
        'Remove only explicit-family empty fixed frames and record root compensation. '
        'Keep every fixed TIP body with mass or geometry. For LEAP, promote the empty '
        "base's only fixed child to import root and return "
        'T_original_root_from_import_root.'
    )

    if family not in _EMPTY_REFERENCE_FRAMES:
        raise ValueError(f"unsupported official frame-lowering family: {family!r}")
    links = {str(node.attrib.get("name")): node for node in root.findall("link")}
    joints = list(root.findall("joint"))
    zero_transforms = _link_zero_transforms(root, original_root_link)
    removed_frames: list[str] = []
    removed_joints: list[str] = []
    removed_frame_transforms: list[dict[str, Any]] = []
    for frame_name in _EMPTY_REFERENCE_FRAMES[family]:
        link = links.get(frame_name)
        if link is None or _has_physical_payload(link):
            raise ValueError(f"expected empty official fixed reference frame is not empty: {frame_name!r}")
        incoming = [joint for joint in joints if _joint_link(joint, "child") == frame_name]
        outgoing = [joint for joint in joints if _joint_link(joint, "parent") == frame_name]
        if len(incoming) != 1 or incoming[0].attrib.get("type") != "fixed" or outgoing:
            raise ValueError(f"official reference frame {frame_name!r} is not a fixed leaf at removal time")
        joint = incoming[0]
        parent_link = _joint_link(joint, "parent")
        if parent_link is None:
            raise ValueError(f"official reference frame {frame_name!r} incoming joint lacks parent")
        removed_frame_transforms.append(
            {
                "frame_link": frame_name,
                "incoming_joint": str(joint.attrib.get("name")),
                "parent_link": parent_link,
                "T_original_root_from_frame": list(zero_transforms[frame_name]),
                "T_parent_from_frame": list(_origin_transform(joint.find("origin"))),
            }
        )
        root.remove(link)
        root.remove(joint)
        joints.remove(joint)
        removed_frames.append(frame_name)
        removed_joints.append(str(joint.attrib.get("name")))

    import_root_link = original_root_link
    root_transform = _identity4()
    original_root = links.get(original_root_link)
    if original_root is None:
        raise ValueError(f"official source root link is absent: {original_root_link!r}")
    if not _has_physical_payload(original_root):
        remaining_joints = tuple(joints)
        fixed_children = [
            joint
            for joint in remaining_joints
            if joint.attrib.get("type") == "fixed" and _joint_link(joint, "parent") == original_root_link
        ]
        if len(fixed_children) != 1:
            raise ValueError(
                f"empty official root {original_root_link!r} must have exactly one fixed child for promotion"
            )
        root_joint = fixed_children[0]
        child_link = _joint_link(root_joint, "child")
        if not child_link or child_link not in links:
            raise ValueError("official root-promotion fixed child is missing from link tree")
        root_transform = _origin_transform(root_joint.find("origin"))
        removed_frame_transforms.append(
            {
                "frame_link": original_root_link,
                "incoming_joint": None,
                "parent_link": None,
                "T_original_root_from_frame": list(zero_transforms[original_root_link]),
                "T_parent_from_frame": list(_identity4()),
            }
        )
        root.remove(original_root)
        root.remove(root_joint)
        joints.remove(root_joint)
        import_root_link = child_link
        removed_frames.append(original_root_link)
        removed_joints.append(str(root_joint.attrib.get("name")))

    return root, {
        "version": FRAME_LOWERING_VERSION,
        "original_root_link": original_root_link,
        "import_root_link": import_root_link,
        "root_promoted": import_root_link != original_root_link,
        "removed_reference_frames": removed_frames,
        "removed_reference_joints": removed_joints,
        "removed_reference_frame_transforms": removed_frame_transforms,
        "T_original_root_from_import_root": list(root_transform),
    }


def prepare_official_import_source(
    description: OfficialHandSemanticsCfg,
    *,
    cache_dir: Path,
    write: bool = True,
) -> tuple[Path, dict[str, Any]]:
    (
        'Return an importable URDF and source report; write=False preview creates no '
        'files. Frame-only lowering always creates a separate import copy; add '
        'OBJ-label adaptation only when needed. Verify source mesh bytes and retained '
        'physical subtrees before/after URI and frame edits, including joint '
        'axes/limits, inertia, primitive collisions, and retained visual poses.'
    )

    # Use absolute paths for renamed OBJ references so caller path spelling cannot create duplicate cache bytes.
    cache_dir = Path(cache_dir).expanduser().resolve(strict=False)
    source = Path(description.source_urdf_path)
    source_data = source.read_bytes()
    if _sha(source_data) != description.source_digest:
        raise ValueError("official import source changed after metadata extraction")
    source_root = ET.fromstring(source_data)
    inertial_by_link, expected_total_mass = _inertial_audit(source_root, family=description.family)
    lowered_root, frame_plan = _lower_empty_reference_frames(
        source_root,
        family=description.family,
        original_root_link=description.root_link,
    )
    modified: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for mesh in description.mesh_digests:
        path = Path(mesh.resolved_path)
        mesh_bytes = path.read_bytes()
        if _sha(mesh_bytes) != mesh.sha256:
            raise ValueError(f"official mesh changed after metadata extraction: {path}")
        if path.suffix.lower() != ".obj":
            continue
        original = mesh_bytes.decode("utf-8")
        labels = [word for line in original.splitlines() if line.startswith(("o ", "g ")) for word in line.split()[1:]]
        if not any(word != _identifier(word) for word in labels) and path.stem == _identifier(path.stem):
            continue
        remapped: dict[str, str] = {}
        materials = []
        rewritten = []
        for line in original.splitlines(keepends=True):
            if line.startswith(("o ", "g ")):
                words = line.split()
                for word in words[1:]:
                    remapped[word] = _identifier(word)
                line = words[0] + " " + " ".join(_identifier(word) for word in words[1:]) + "\n"
            elif line.startswith("mtllib "):
                paths = [(path.parent / name).resolve(strict=True) for name in line.split()[1:]]
                materials.extend({"path": str(item), "sha256": _sha(item.read_bytes())} for item in paths)
                line = "mtllib " + " ".join(str(item) for item in paths) + "\n"  # Preserve the source library and relative-texture resolution base.
            rewritten.append(line)
        if len(set(remapped.values())) != len(remapped):
            raise ValueError("OBJ label sanitization would merge distinct source groups")
        adapted = "".join(rewritten)
        if _obj_geometry_bytes(original) != _obj_geometry_bytes(adapted):
            raise RuntimeError("OBJ name adapter changed mesh geometry or material assignment")
        modified[mesh.source_uri] = (
            adapted.encode(),
            {
                "source_path": str(path),
                "source_sha256": mesh.sha256,
                "label_mapping": remapped,
                "geometry_payload_sha256": _sha(_obj_geometry_bytes(original)),
                "materials": materials,
            },
        )
    identity = _sha(
        json.dumps(
            {
                "adapter": IMPORT_NAME_ADAPTER,
                "frame_lowering": frame_plan,
                "source": description.source_digest,
                "meshes": {uri: _sha(data) for uri, (data, _) in modified.items()},
            },
            sort_keys=True,
        ).encode()
    )
    directory = cache_dir / "import_sources" / identity
    imported_urdf = directory / source.name  # Keep the official entry basename while recording the source as the description origin.
    declared = {mesh.source_uri: mesh for mesh in description.mesh_digests}
    records = {}
    planned_files: dict[Path, bytes] = {}
    for index, (uri, (data, record)) in enumerate(modified.items()):
        path = directory / "meshes" / f"mesh_{index:02d}_{_identifier(Path(uri).stem)}.obj"
        planned_files[path] = data
        records[uri] = {**record, "import_path": str(path), "import_sha256": _sha(data)}
    expected_tree = copy.deepcopy(lowered_root)
    for original_node, target_node in zip(expected_tree.iter("mesh"), lowered_root.iter("mesh"), strict=True):
        uri = original_node.attrib["filename"]
        mesh = declared[uri]
        target = Path(records[uri]["import_path"]) if uri in records else Path(mesh.resolved_path)
        target_node.set("filename", str(target))  # Change references only; preserve all other XML nodes and values.
        original_node.set("filename", uri)
    # After removing the same reference frames from both files, retained physical XML for every link/joint must match byte for byte.
    source_pruned = ET.fromstring(source_data)
    _remove_named_nodes(
        source_pruned,
        link_names=tuple(frame_plan["removed_reference_frames"]),
        joint_names=tuple(frame_plan["removed_reference_joints"]),
    )
    if ET.tostring(source_pruned) != ET.tostring(expected_tree):
        raise RuntimeError("official frame lowering changed a retained physical URDF subtree")
    restored = copy.deepcopy(lowered_root)
    for node, original_node in zip(restored.iter("mesh"), expected_tree.iter("mesh"), strict=True):
        node.set("filename", original_node.attrib["filename"])
    if ET.tostring(restored) != ET.tostring(expected_tree):
        raise RuntimeError("official import adapter changed lowered non-URI URDF semantics")
    urdf_data = ET.tostring(lowered_root, encoding="utf-8", xml_declaration=True)
    planned_files[imported_urdf] = urdf_data
    report = {
        "adapted": True,
        "adapter": IMPORT_NAME_ADAPTER,
        "identity": identity,
        "source_urdf": str(source),
        "source_urdf_sha256": description.source_digest,
        "import_urdf": str(imported_urdf),
        "import_urdf_sha256": _sha(urdf_data),
        "meshes": records,
        "non_uri_urdf_unchanged": False,
        "physical_subtree_unchanged": True,
        "geometry_and_material_assignment_unchanged": True,
        "frame_lowering": frame_plan,
        "removed_reference_frames": list(frame_plan["removed_reference_frames"]),
        "removed_reference_joints": list(frame_plan["removed_reference_joints"]),
        "removed_reference_frame_transforms": list(frame_plan["removed_reference_frame_transforms"]),
        "inertial_by_link": inertial_by_link,
        "source_inertial_by_link": inertial_by_link,
        "expected_total_mass_kg": expected_total_mass,
        "root_transform": {
            "original_root_link": frame_plan["original_root_link"],
            "import_root_link": frame_plan["import_root_link"],
            "T_original_root_from_import_root": list(frame_plan["T_original_root_from_import_root"]),
            "root_promoted": bool(frame_plan["root_promoted"]),
        },
    }
    if write:
        for path, data in planned_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"conflicting immutable official import cache: {path}")
            if not path.exists():
                path.write_bytes(data)
        (directory / "import-equivalence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return imported_urdf, report
