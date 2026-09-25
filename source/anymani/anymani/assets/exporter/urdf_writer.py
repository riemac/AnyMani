"Serializes the typed hand tree and mesh dependencies. Each palm-local finger mount is folded into its first joint origin; no auxiliary mount link is emitted."

from __future__ import annotations

import hashlib
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_schema_core import (
    CollisionGeometryCfg,
    EllipticCylinderGeometryCfg,
    InertialCfg,
    MaterialCfg,
    MeshGeometryCfg,
    PoseCfg,
    VisualGeometryCfg,
)
from ..handedness import compose_poses
from ..procedural_meshes import (
    is_procedural_cs_tip_uri,
    materialize_procedural_cs_tip_mesh,
    parse_procedural_cs_tip_uri,
)
from ._base import ExporterBase, ExportResult

# ============================================================================

# ============================================================================


@dataclass
class UrdfWriterCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type[UrdfWriter] | None = None
    "Associated runtime implementation for this configuration class."

    filename: str = "hand.urdf"
    "Stable semantic identifier preserved in the output metadata."

    include_inertial: bool = True
    "Whether to serialize closed mass and inertia values into the URDF."

    default_effort: float = 10.0
    "Default joint effort limit in newton meters."

    default_velocity: float = 3.14
    "Default joint speed limit in radians per second."

    mesh_package_prefix: str | None = None
    "Package-root prefix used to resolve local mesh URIs."

    canonical_mesh_dirname: str = "meshes"
    "Filesystem location resolved relative to the declared asset root when not absolute."

    overwrite: bool = True
    "Whether an existing output path may be replaced."

    recolored_materials: dict[str, MaterialCfg] = field(default_factory=dict)
    "Visual-only material overrides, separate from physical geometry."

    portable_mesh_bindings: Mapping[str, tuple[Path, str]] | None = None
    "Verified source-locator bindings enabled only for a portable hand bundle."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = UrdfWriter
        normalized_materials: dict[str, MaterialCfg] = {}
        for link_name, material in self.recolored_materials.items():
            if isinstance(material, MaterialCfg):
                material_cfg = material.copy()
            elif isinstance(material, Mapping):
                material_cfg = MaterialCfg(**material)
            else:
                raise TypeError(
                    f"recolored_materials[{link_name!r}] must be MaterialCfg or mapping, got {material!r}"
                )
            normalized_materials[str(link_name)] = material_cfg
        self.recolored_materials = normalized_materials


# ============================================================================

# ============================================================================


class UrdfWriter(ExporterBase):
    "Writes links, joints, collision geometry, and relative mesh URIs into one URDF."

    cfg: UrdfWriterCfg

    def __init__(self, cfg: UrdfWriterCfg):
        self.cfg = cfg

    def export(
        self,
        target: HandCfg,
        output_dir: Path,
        *,
        mesh_root_dir: Path | None = None,
    ) -> ExportResult:  # type: ignore[override]
        "Writes the declared hand component and its metadata to the output bundle."

        out_path = output_dir / self.cfg.filename
        if out_path.exists() and not self.cfg.overwrite:
            return ExportResult(skipped=[out_path])

        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_state = _MeshExportState(
            output_dir=output_dir,
            mesh_dirname=self.cfg.canonical_mesh_dirname,
            mesh_root_dir=mesh_root_dir,
            portable_mesh_bindings=self.cfg.portable_mesh_bindings,
        )
        robot = _build_robot_elem(target, self.cfg, mesh_state=mesh_state)
        ET.indent(robot)
        tree = ET.ElementTree(robot)
        tree.write(out_path, encoding="unicode", xml_declaration=True)
        return ExportResult(written=[out_path, *mesh_state.written])

    def to_urdf_string(self, target: HandCfg) -> str:
        "Serializes the hand links, joints, limits, and geometry as URDF XML."

        mesh_state = _MeshExportState(
            output_dir=Path("."),
            mesh_dirname=self.cfg.canonical_mesh_dirname,
            write_enabled=False,
            portable_mesh_bindings=self.cfg.portable_mesh_bindings,
        )
        robot = _build_robot_elem(target, self.cfg, mesh_state=mesh_state)
        ET.indent(robot)
        return ET.tostring(robot, encoding="unicode", xml_declaration=True)


# ============================================================================

# ============================================================================


@dataclass
class _MeshExportState:
    "Tracks mesh files and relative URIs already written for one hand export."

    output_dir: Path
    mesh_dirname: str
    write_enabled: bool = True
    written: list[Path] = field(default_factory=list)
    mesh_root_dir: Path | None = None
    portable_mesh_bindings: Mapping[str, tuple[Path, str]] | None = None
    _unit_cylinder_relpath: str | None = None
    _materialized_mesh_relpaths: dict[Path, str] = field(default_factory=dict)
    _materialized_mesh_names: dict[str, Path] = field(default_factory=dict)


def _build_robot_elem(target: HandCfg, cfg: UrdfWriterCfg, *, mesh_state: _MeshExportState) -> ET.Element:

    robot = ET.Element("robot", attrib={"name": target.name})
    robot.append(
        _build_link_elem(
            target.palm.name,
            target.palm.inertial,
            target.palm.collisions,
            target.palm.visuals,
            cfg,
            mesh_state=mesh_state,
        )
    )

    for finger in target.fingers:
        parent_name = target.palm.name
        joints = [
            _copy_joint_with_mount(finger, joint) if index == 0 else joint
            for index, joint in enumerate(finger.joints)
        ]

        for joint in joints:
            robot.append(_build_joint_elem(joint, parent_name, cfg))
            robot.append(
                _build_link_elem(
                    joint.child,
                    joint.inertial,
                    joint.collisions,
                    joint.visuals,
                    cfg,
                    mesh_state=mesh_state,
                )
            )
            parent_name = joint.child
    return robot


def _copy_joint_with_mount(finger, joint):

    mount = finger.mount or PoseCfg()
    origin = compose_poses(mount, joint.origin)
    return joint.replace(origin=origin)


def _build_link_elem(
    name: str,
    inertial: InertialCfg | None,
    collisions: list[CollisionGeometryCfg],
    visuals: list[VisualGeometryCfg],
    cfg: UrdfWriterCfg,
    *,
    mesh_state: _MeshExportState,
) -> ET.Element:

    link = ET.Element("link", attrib={"name": name})
    if cfg.include_inertial:
        inertial_cfg = inertial or InertialCfg(
            mass=1e-6,
            origin=PoseCfg(),
            inertia={"ixx": 1e-9, "iyy": 1e-9, "izz": 1e-9},
        )
        inertial_elem = ET.SubElement(link, "inertial")
        ET.SubElement(inertial_elem, "origin", attrib=_pose_attrib(inertial_cfg.origin))
        ET.SubElement(inertial_elem, "mass", attrib={"value": _fmt_scalar(inertial_cfg.mass)})
        ET.SubElement(
            inertial_elem,
            "inertia",
            attrib={
                "ixx": _fmt_scalar(inertial_cfg.inertia.ixx),
                "ixy": _fmt_scalar(inertial_cfg.inertia.ixy),
                "ixz": _fmt_scalar(inertial_cfg.inertia.ixz),
                "iyy": _fmt_scalar(inertial_cfg.inertia.iyy),
                "iyz": _fmt_scalar(inertial_cfg.inertia.iyz),
                "izz": _fmt_scalar(inertial_cfg.inertia.izz),
            },
        )

    for visual in visuals:
        visual_elem = ET.SubElement(link, "visual")
        if visual.name:
            visual_elem.attrib["name"] = visual.name
        ET.SubElement(visual_elem, "origin", attrib=_pose_attrib(visual.origin))
        visual_elem.append(_build_geometry_elem(visual.geometry, cfg, mesh_state=mesh_state))
        material = cfg.recolored_materials.get(name) or visual.material
        if material is not None:
            visual_elem.append(_build_material_elem(material))

    for collision in collisions:
        collision_elem = ET.SubElement(link, "collision")
        if collision.name:
            collision_elem.attrib["name"] = collision.name
        ET.SubElement(collision_elem, "origin", attrib=_pose_attrib(collision.origin))
        collision_elem.append(_build_geometry_elem(collision.geometry, cfg, mesh_state=mesh_state))

    return link


def _build_joint_elem(joint, parent_override: str, cfg: UrdfWriterCfg) -> ET.Element:

    joint_elem = ET.Element("joint", attrib={"name": joint.name, "type": joint.joint_type})
    ET.SubElement(joint_elem, "parent", attrib={"link": parent_override})
    ET.SubElement(joint_elem, "child", attrib={"link": joint.child})
    ET.SubElement(joint_elem, "origin", attrib=_pose_attrib(joint.origin))

    if joint.joint_type != "fixed":
        ET.SubElement(joint_elem, "axis", attrib={"xyz": _fmt_triplet(joint.axis)})
        if joint.limit is not None:
            ET.SubElement(
                joint_elem,
                "limit",
                attrib={
                    "lower": _fmt_scalar(joint.limit.lower),
                    "upper": _fmt_scalar(joint.limit.upper),
                    "effort": _fmt_scalar(
                        joint.limit.effort if joint.limit.effort is not None else cfg.default_effort
                    ),
                    "velocity": _fmt_scalar(
                        joint.limit.velocity if joint.limit.velocity is not None else cfg.default_velocity
                    ),
                },
            )
        if joint.joint_properties is not None and joint.joint_properties.friction is not None:
            ET.SubElement(
                joint_elem,
                "joint_properties",
                attrib={"friction": _fmt_scalar(joint.joint_properties.friction)},
            )

    return joint_elem


def _build_fixed_joint(name: str, parent: str, child: str, origin: PoseCfg) -> ET.Element:

    joint_elem = ET.Element("joint", attrib={"name": name, "type": "fixed"})
    ET.SubElement(joint_elem, "parent", attrib={"link": parent})
    ET.SubElement(joint_elem, "child", attrib={"link": child})
    ET.SubElement(joint_elem, "origin", attrib=_pose_attrib(origin))
    return joint_elem


def _build_geometry_elem(geom, cfg: UrdfWriterCfg, *, mesh_state: _MeshExportState) -> ET.Element:

    geometry_elem = ET.Element("geometry")
    kind = geom.kind
    if kind == "box":
        ET.SubElement(geometry_elem, "box", attrib={"size": _fmt_triplet(geom.size)})
    elif kind == "cylinder":
        ET.SubElement(
            geometry_elem,
            "cylinder",
            attrib={"radius": _fmt_scalar(geom.radius), "length": _fmt_scalar(geom.length)},
        )
    elif kind == "elliptic_cylinder":
        mesh_geom = _lower_elliptic_cylinder_to_mesh(geom, mesh_state=mesh_state)
        filename = mesh_geom.file_path
        if cfg.mesh_package_prefix and not filename.startswith(("package://", "/")):
            filename = f"{cfg.mesh_package_prefix.rstrip('/')}/{filename.lstrip('./')}"
        mesh_attrib = {"filename": filename}
        if mesh_geom.scale != (1.0, 1.0, 1.0):
            mesh_attrib["scale"] = _fmt_triplet(mesh_geom.scale)
        ET.SubElement(geometry_elem, "mesh", attrib=mesh_attrib)
    elif kind == "sphere":
        ET.SubElement(geometry_elem, "sphere", attrib={"radius": _fmt_scalar(geom.radius)})
    elif kind == "mesh":
        if isinstance(geom, MeshGeometryCfg) and geom.reflected_about_yz:
            raise ValueError(
                "URDF export received reflected_about_yz=True; materialize strict handedness meshes "
                "before physics/validator/exporter"
            )
        filename = _materialize_mesh_geometry(geom, mesh_state=mesh_state)
        if cfg.mesh_package_prefix and not filename.startswith(("package://", "/")):
            filename = f"{cfg.mesh_package_prefix.rstrip('/')}/{filename.lstrip('./')}"
        mesh_attrib = {"filename": filename}
        if isinstance(geom, MeshGeometryCfg) and geom.scale != (1.0, 1.0, 1.0):
            mesh_attrib["scale"] = _fmt_triplet(geom.scale)
        ET.SubElement(geometry_elem, "mesh", attrib=mesh_attrib)
    else:
        raise ValueError(f"Unsupported URDF geometry kind: {kind}")
    return geometry_elem


def _lower_elliptic_cylinder_to_mesh(
    geom: EllipticCylinderGeometryCfg,
    *,
    mesh_state: _MeshExportState,
) -> MeshGeometryCfg:

    rel_path = _ensure_unit_cylinder_mesh(mesh_state)
    return MeshGeometryCfg(
        file_path=rel_path,
        scale=(2.0 * geom.radius_x, geom.length, 2.0 * geom.radius_z),
    )


def _materialize_mesh_geometry(geom: MeshGeometryCfg, *, mesh_state: _MeshExportState) -> str:

    if is_procedural_cs_tip_uri(geom.file_path):
        return _materialize_procedural_cs_tip_geometry(geom, mesh_state=mesh_state)

    source_locator = str(geom.file_path)
    source_path = Path(source_locator).expanduser()
    if not source_path.is_absolute():
        return geom.file_path

    copy_source_path = source_path
    expected_sha256: str | None = None
    if mesh_state.portable_mesh_bindings is not None:
        binding = mesh_state.portable_mesh_bindings.get(source_locator)
        if binding is None:
            raise FileNotFoundError(f"portable bundle has no verified mesh binding for {source_locator!r}")
        copy_source_path, expected_sha256 = binding
        copy_source_path = Path(copy_source_path).expanduser().resolve(strict=True)
        _verify_portable_mesh_source(copy_source_path, expected_sha256, locator=source_locator)

    cached = mesh_state._materialized_mesh_relpaths.get(source_path)
    if cached is not None:
        return cached

    mesh_root = mesh_state.mesh_root_dir or (mesh_state.output_dir / mesh_state.mesh_dirname)
    target_name = _resolve_materialized_mesh_name(source_path, mesh_state=mesh_state)
    target_path = mesh_root / target_name

    if mesh_state.write_enabled:
        mesh_root.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and expected_sha256 is not None:
            _verify_portable_mesh_source(target_path, expected_sha256, locator=source_locator)
        elif not target_path.exists():
            shutil.copy2(copy_source_path, target_path)
            if expected_sha256 is not None:
                _verify_portable_mesh_source(target_path, expected_sha256, locator=source_locator)
            mesh_state.written.append(target_path)

    rel_path = os.path.relpath(target_path, start=mesh_state.output_dir)
    mesh_state._materialized_mesh_relpaths[source_path] = rel_path
    return rel_path


def _verify_portable_mesh_source(path: Path, expected_sha256: str, *, locator: str) -> None:
    """Verify portable mesh bytes before exporting a frozen geometry locator."""

    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError(f"portable mesh locator {locator!r} has an invalid SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"portable mesh locator {locator!r} resolved to missing file: {path}")
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"portable mesh locator {locator!r} SHA-256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )


def _materialize_procedural_cs_tip_geometry(geom: MeshGeometryCfg, *, mesh_state: _MeshExportState) -> str:

    spec = parse_procedural_cs_tip_uri(geom.file_path)
    mesh_root = mesh_state.mesh_root_dir or (mesh_state.output_dir / mesh_state.mesh_dirname)
    mesh_path, written = materialize_procedural_cs_tip_mesh(
        spec,
        mesh_root,
        write_enabled=mesh_state.write_enabled,
    )
    if written:
        mesh_state.written.append(mesh_path)
    return os.path.relpath(mesh_path, start=mesh_state.output_dir)


def _resolve_materialized_mesh_name(source_path: Path, *, mesh_state: _MeshExportState) -> str:

    basename = source_path.name
    occupied = mesh_state._materialized_mesh_names.get(basename)
    if occupied is None or occupied == source_path:
        mesh_state._materialized_mesh_names[basename] = source_path
        return basename

    stem = source_path.stem
    suffix = source_path.suffix
    digest = hashlib.md5(str(source_path).encode("utf-8")).hexdigest()[:8]
    candidate = f"{stem}_{digest}{suffix}"
    mesh_state._materialized_mesh_names[candidate] = source_path
    return candidate


def _ensure_unit_cylinder_mesh(mesh_state: _MeshExportState) -> str:

    if mesh_state._unit_cylinder_relpath is not None:
        return mesh_state._unit_cylinder_relpath

    mesh_dir = mesh_state.mesh_root_dir or (mesh_state.output_dir / mesh_state.mesh_dirname)
    mesh_path = mesh_dir / "unit_cylinder_y.obj"
    if mesh_state.write_enabled:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        if not mesh_path.exists():
            mesh_path.write_text(_unit_cylinder_y_obj_text(), encoding="utf-8")
            mesh_state.written.append(mesh_path)
    mesh_state._unit_cylinder_relpath = os.path.relpath(mesh_path, start=mesh_state.output_dir)
    return mesh_state._unit_cylinder_relpath


def _unit_cylinder_y_obj_text(*, segments: int = 24) -> str:

    segments = max(int(segments), 3)
    lines: list[str] = ["# canonical unit cylinder aligned with +y"]
    lines.append("v 0 -0.5 0")
    lines.append("v 0 0.5 0")


    for ring_y in (-0.5, 0.5):
        for index in range(segments):
            theta = 2.0 * math.pi * index / segments
            x = math.cos(theta)
            z = math.sin(theta)
            lines.append(f"v {_fmt_scalar(x)} {_fmt_scalar(ring_y)} {_fmt_scalar(z)}")

    bottom_center_index = 1
    top_center_index = 2
    bottom_start = 3
    top_start = 3 + segments


    for index in range(segments):
        current = bottom_start + index
        nxt = bottom_start + ((index + 1) % segments)
        lines.append(f"f {bottom_center_index} {nxt} {current}")


    for index in range(segments):
        current = top_start + index
        nxt = top_start + ((index + 1) % segments)
        lines.append(f"f {top_center_index} {current} {nxt}")


    for index in range(segments):
        bottom_current = bottom_start + index
        bottom_next = bottom_start + ((index + 1) % segments)
        top_current = top_start + index
        top_next = top_start + ((index + 1) % segments)
        lines.append(f"f {bottom_current} {bottom_next} {top_next}")
        lines.append(f"f {bottom_current} {top_next} {top_current}")

    return "\n".join(lines) + "\n"


def _build_material_elem(material: MaterialCfg) -> ET.Element:

    material_elem = ET.Element("material")
    if material.name:
        material_elem.attrib["name"] = material.name
    ET.SubElement(material_elem, "color", attrib={"rgba": _fmt_triplet(material.rgba)})
    return material_elem


def _pose_attrib(pose: PoseCfg) -> dict[str, str]:
    return {"xyz": _fmt_triplet(pose.pos), "rpy": _fmt_triplet(pose.rpy)}


def _fmt_triplet(values) -> str:
    return " ".join(_fmt_scalar(value) for value in values)


def _fmt_scalar(value: float) -> str:
    return f"{float(value):.9g}"


__all__ = ["UrdfWriterCfg", "UrdfWriter"]
