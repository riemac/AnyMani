(
    'Native articulation spawn adapter for official Allegro/LEAP URDFs. Consume '
    'the hash-validated OfficialHandSemanticsCfg and pass the official source to '
    'Isaac Lab. Never proxy-fit official collisions. A separate mechanically '
    'equivalent import copy may remove empty reference frames, but retained body '
    'mass/inertia/limits and original assets remain unchanged. Root pose uses '
    'T_wa=T_wh*T_ha, with R_wh=I, R_wa=R_ha, and p_wa=p_wh+p_ha. Import Isaac Lab '
    'only inside build_official_hand_cfg so source mappings, audits, and cache '
    'keys remain testable in pure Python. If named materials are absent, keep '
    'importer appearance and record the fallback; use the render plan only when '
    'source material evidence exists.'
)

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from anymani.assets.bank.official_hand import OfficialHandSemanticsCfg

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

OfficialFamily = Literal["allegro", "leap"]
'Official native source family.'

_USD_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENTITY_R = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_IDENTITY_T = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
_DEFAULT_ROOT_POSITION_H = (0.0, 0.0, 0.5)
_OFFICIAL_SPAWN_SCHEMA = "anymani.official_hand_spawn.v2"
_OFFICIAL_ARMATURE = 0.001  # kg m^2; matches the current generated/native hand actuator contract.
_OFFICIAL_STIFFNESS = 3.0  # N m/rad; matches the existing implicit-PD hand route.
_OFFICIAL_DAMPING = 0.1  # N m/(rad s^-1); matches the existing implicit-PD hand route.


def source_joint_to_usd_joint_name(description: OfficialHandSemanticsCfg, source_joint_name: str) -> str:
    (
        'Map an exact validated source joint name to the importer USD name. Allegro '
        'dots become underscores; LEAP names stay unchanged. Reject undeclared names '
        'rather than using a permissive regex.'
    )

    known = tuple(joint.source_name for joint in description.joints)
    return _resolve_declared_usd_name(
        source_name=source_joint_name,
        known_names=known,
        family=description.family,
        kind="joint",
    )


def source_link_to_usd_link_name(description: OfficialHandSemanticsCfg, source_link_name: str) -> str:
    (
        'Map a source link to its exact USD name. '
        'Root/palm/frame/kinematic/component/owner fields close the native link set; '
        'reject unknown links, sanitize collisions, and undeclared characters.'
    )

    return _resolve_declared_usd_name(
        source_name=source_link_name,
        known_names=_known_source_links(description),
        family=description.family,
        kind="link",
    )


def resolve_official_link_name(description: OfficialHandSemanticsCfg, source_link_name: str) -> str:
    'Readable alias for source_link_to_usd_link_name.'

    return source_link_to_usd_link_name(description, source_link_name)


def resolve_official_joint_indices(
    description: OfficialHandSemanticsCfg,
    actual_joint_names: Sequence[str],
) -> tuple[int, ...]:
    (
        'Map runtime joint order to the official canonical depth-major slots. Return '
        '16 indices in description.joints order; reject duplicate, missing, or extra '
        'runtime names.'
    )

    actual = tuple(str(name) for name in actual_joint_names)
    if len(actual) != len(description.joints):
        raise ValueError(
            f"official articulation must expose exactly {len(description.joints)} active joints, got {len(actual)}"
        )
    if len(set(actual)) != len(actual):
        raise ValueError("official articulation joint names must be unique")
    expected = tuple(source_joint_to_usd_joint_name(description, joint.source_name) for joint in description.joints)
    if len(set(expected)) != len(expected):
        raise ValueError("declared source→USD joint mapping is not unique")
    expected_set = set(expected)
    actual_set = set(actual)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(f"official articulation joint names mismatch: missing={missing}, extra={extra}")
    actual_index = {name: index for index, name in enumerate(actual)}
    return tuple(actual_index[name] for name in expected)


def build_official_hand_cfg(
    description: OfficialHandSemanticsCfg,
    *,
    root_position_h: Sequence[float] = _DEFAULT_ROOT_POSITION_H,
    cache_dir: Path,
    render: bool = False,
) -> ArticulationCfg:
    (
        'Build an Isaac Lab ArticulationCfg from the official source URDF. The import '
        'copy may remove empty reference frames only; compensate root pose so '
        'retained body FK is unchanged. Keep fixed TIP bodies and source collisions. '
        'Lower root pose as '
        'T_world_importroot=T_world_hand*T_hand_originalroot*T_originalroot_importroot. '
        'Units: m, rad, N*m, rad/s. Use a hash-validated native description. Cache '
        'URDF/USD outputs separately from read-only source URDF/mesh; key by '
        'source/mesh, frame, render, and importer inputs. If render is requested '
        'without named material evidence, preserve importer appearance.'
    )

    root_h = _finite_vector3(root_position_h, field_name="root_position_h")
    source_urdf = Path(description.source_urdf_path).expanduser().resolve(strict=True)
    _verify_description_source_digest(description, source_urdf)
    from ._official_import_source import prepare_official_import_source

    import_urdf, import_source_audit = prepare_official_import_source(description, cache_dir=cache_dir)
    root_pose = _compose_import_root_pose(
        description,
        root_position_h=root_h,
        import_root_link=str(import_source_audit["root_transform"]["import_root_link"]),
        T_original_root_from_import_root=import_source_audit["root_transform"]["T_original_root_from_import_root"],
    )
    usd_joint_names = tuple(
        source_joint_to_usd_joint_name(description, joint.source_name) for joint in description.joints
    )
    source_friction_by_name = _read_source_joint_friction(description, source_urdf)
    usd_friction = (
        {source_joint_to_usd_joint_name(description, name): value for name, value in source_friction_by_name.items()}
        if source_friction_by_name
        else None
    )
    root_position_a = cast(tuple[float, float, float], root_pose["import_root_position_a"])
    root_quat_wxyz = cast(tuple[float, float, float, float], root_pose["import_root_quat_wxyz"])
    runtime_boot_joint_positions = {
        usd_name: _project_zero_to_limit(joint.lower_rad, joint.upper_rad)
        for usd_name, joint in zip(usd_joint_names, description.joints, strict=True)
    }  # Spawn pose must satisfy importer limits; retain mathematical q_home=0 in official metadata.

    render_plan, render_audit = _build_official_render_plan(description, source_urdf, render=render)
    bindable_visual_names = (
        set(render_plan.visual_rgba_by_name) - set(render_plan.unresolved_visual_names)
        if render_plan is not None
        else set()
    )
    render_wrapper_enabled = bool(render and bindable_visual_names)
    make_instanceable = not render_wrapper_enabled
    importer_config = {
        "fix_base": True,
        "merge_fixed_joints": False,
        "force_usd_conversion": False,
        "make_instanceable": make_instanceable,
        "collision_from_visuals": False,
        "collider_type": "convex_hull",
        "self_collision": True,
        "activate_contact_sensors": True,
        "joint_drive": {
            "target_type": "position",
            "drive_type": "force",
            "stiffness": _OFFICIAL_STIFFNESS,
            "damping": _OFFICIAL_DAMPING,
        },
    }
    if import_source_audit["adapted"]:
        importer_config["source_name_adapter_identity"] = import_source_audit["identity"]
    cache_path = _build_official_cache_dir(
        description,
        root_position_h=root_h,
        root_position_a=root_position_a,
        root_quat_wxyz=root_quat_wxyz,
        import_root_link=str(root_pose["import_root_link"]),
        T_original_root_from_import_root=cast(tuple[float, ...], root_pose["T_original_root_from_import_root"]),
        cache_dir=cache_dir,
        render=render,
        importer_config=importer_config,
    )

    # IsaacLab imports are deliberately local: pure name/cache/audit tests never load Kit/pxr.
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.converters import UrdfConverterCfg

    spawn_cfg = sim_utils.UrdfFileCfg(
        asset_path=str(import_urdf),  # Preserve the original mechanical description; adapt OBJ display labels only when required.
        usd_dir=str(cache_path),
        usd_file_name="hand.usd",
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=False,
        make_instanceable=make_instanceable,
        collider_type="convex_hull",
        self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=_OFFICIAL_STIFFNESS,
                damping=_OFFICIAL_DAMPING,
            ),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64.0 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
    )
    cast(Any, spawn_cfg).collision_from_visuals = False
    if render_wrapper_enabled and render_plan is not None:
        from anymani.robots._visual_materials import serialize_visual_material_restore_plan
        from anymani.robots.hand_spawn import _spawn_urdf_with_restored_visual_materials

        spawn_cfg.func = _spawn_urdf_with_restored_visual_materials
        cast(Any, spawn_cfg)._anymani_visual_material_plan = serialize_visual_material_restore_plan(render_plan)

    actuators = {
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=list(usd_joint_names),
            effort_limit={
                name: float(joint.effort_nm) for name, joint in zip(usd_joint_names, description.joints, strict=True)
            },
            velocity_limit={
                name: float(joint.velocity_rad_s)
                for name, joint in zip(usd_joint_names, description.joints, strict=True)
            },
            effort_limit_sim={
                name: float(joint.effort_nm) for name, joint in zip(usd_joint_names, description.joints, strict=True)
            },
            velocity_limit_sim={
                name: float(joint.velocity_rad_s)
                for name, joint in zip(usd_joint_names, description.joints, strict=True)
            },
            stiffness=_OFFICIAL_STIFFNESS,
            damping=_OFFICIAL_DAMPING,
            friction=usd_friction,
            armature=_OFFICIAL_ARMATURE,
        )
    }
    cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=root_position_a,
            rot=root_quat_wxyz,
            joint_pos=runtime_boot_joint_positions,
            joint_vel={name: 0.0 for name in usd_joint_names},
        ),
        actuators=cast(Any, actuators),
        soft_joint_pos_limit_factor=1.0,
    )
    audit = _build_official_spawn_audit(
        description,
        source_urdf=source_urdf,
        root_position_h=root_h,
        root_pose=root_pose,
        usd_joint_names=usd_joint_names,
        source_friction_by_name=source_friction_by_name,
        usd_cache_path=cache_path,
        render=render,
        render_wrapper_enabled=render_wrapper_enabled,
        render_audit=render_audit,
        importer_config=importer_config,
        import_source_audit=import_source_audit,
    )
    cast(Any, cfg)._anymani_official_hand_audit = audit  # JSON-safe render/runtime audit only; does not affect cfg hash.
    cast(Any, spawn_cfg)._anymani_official_hand_audit = audit
    return cfg


def audit_official_hand_cfg(cfg: ArticulationCfg) -> dict[str, object]:
    'Read the read-only JSON-safe audit written by build_official_hand_cfg.'

    raw = getattr(cfg, "_anymani_official_hand_audit", None)
    if not isinstance(raw, Mapping):
        raise ValueError("cfg does not contain official hand spawn audit")
    return cast(dict[str, object], json.loads(json.dumps(raw, sort_keys=True)))


def build_official_hand_spawn_audit(
    description: OfficialHandSemanticsCfg,
    *,
    root_position_h: Sequence[float] = _DEFAULT_ROOT_POSITION_H,
    cache_dir: Path,
    render: bool = False,
) -> dict[str, object]:
    (
        'Build the official spawn-contract audit without loading Isaac Lab. Share '
        'source/link/joint/cache logic with build_official_hand_cfg and record '
        'URDF/mesh hashes, frame, motor caps, and render fallback. Do not create '
        'cache directories, read USD stage, or modify source URDF.'
    )

    root_h = _finite_vector3(root_position_h, field_name="root_position_h")
    source_urdf = Path(description.source_urdf_path).expanduser().resolve(strict=True)
    _verify_description_source_digest(description, source_urdf)
    from ._official_import_source import prepare_official_import_source

    _import_urdf, import_source_audit = prepare_official_import_source(
        description,
        cache_dir=cache_dir,
        write=False,
    )
    root_pose = _compose_import_root_pose(
        description,
        root_position_h=root_h,
        import_root_link=str(import_source_audit["root_transform"]["import_root_link"]),
        T_original_root_from_import_root=import_source_audit["root_transform"]["T_original_root_from_import_root"],
    )
    root_position_a = cast(tuple[float, float, float], root_pose["import_root_position_a"])
    root_quat_wxyz = cast(tuple[float, float, float, float], root_pose["import_root_quat_wxyz"])
    usd_joint_names = tuple(
        source_joint_to_usd_joint_name(description, joint.source_name) for joint in description.joints
    )
    source_friction_by_name = _read_source_joint_friction(description, source_urdf)
    importer_config = {
        "fix_base": True,
        "merge_fixed_joints": False,
        "force_usd_conversion": False,
        "make_instanceable": True,
        "collision_from_visuals": False,
        "collider_type": "convex_hull",
        "self_collision": True,
        "activate_contact_sensors": True,
        "joint_drive": {
            "target_type": "position",
            "drive_type": "force",
            "stiffness": _OFFICIAL_STIFFNESS,
            "damping": _OFFICIAL_DAMPING,
        },
    }
    if import_source_audit["adapted"]:
        importer_config["source_name_adapter_identity"] = import_source_audit["identity"]
    cache_path = _build_official_cache_dir(
        description,
        root_position_h=root_h,
        root_position_a=root_position_a,
        root_quat_wxyz=root_quat_wxyz,
        import_root_link=str(root_pose["import_root_link"]),
        T_original_root_from_import_root=cast(tuple[float, ...], root_pose["T_original_root_from_import_root"]),
        cache_dir=cache_dir,
        render=render,
        importer_config=importer_config,
    )
    render_plan, render_audit = _build_official_render_plan(description, source_urdf, render=render)
    bindable = (
        set(render_plan.visual_rgba_by_name) - set(render_plan.unresolved_visual_names)
        if render_plan is not None
        else set()
    )
    return _build_official_spawn_audit(
        description,
        source_urdf=source_urdf,
        root_position_h=root_h,
        root_pose=root_pose,
        usd_joint_names=usd_joint_names,
        source_friction_by_name=source_friction_by_name,
        usd_cache_path=cache_path,
        render=render,
        render_wrapper_enabled=bool(render and bindable),
        render_audit=render_audit,
        importer_config={**importer_config, "make_instanceable": not bool(render and bindable)},
        import_source_audit=import_source_audit,
    )


def _resolve_declared_usd_name(
    *,
    source_name: str,
    known_names: Sequence[str],
    family: OfficialFamily,
    kind: str,
) -> str:
    'Check exact source-name membership, explicit sanitization, and collisions.'

    if source_name not in known_names:
        raise KeyError(f"unknown official source {kind} name {source_name!r}")
    mapped_names = tuple(_sanitize_official_name(name, family=family) for name in known_names)
    if len(mapped_names) != len(set(mapped_names)):
        raise ValueError(f"official {kind} source→USD sanitize mapping is not unique")
    mapped = _sanitize_official_name(source_name, family=family)
    if _USD_IDENTIFIER.fullmatch(mapped) is None:
        raise ValueError(f"official {kind} USD name is not a valid identifier: {mapped!r}")
    return mapped


def _sanitize_official_name(source_name: str, *, family: OfficialFamily) -> str:
    'Apply only declared family-specific sanitization; reject implicit regex or character replacement.'

    if family == "allegro":
        return source_name.replace(".", "_")
    return source_name


def _known_source_links(description: OfficialHandSemanticsCfg) -> tuple[str, ...]:
    'Close the source-link set from complete native description records.'

    links: set[str] = {description.root_link, description.palm_link}
    links.update(frame.source_link for frame in description.frames)
    for joint in description.geometry_semantics.kinematic_joints:
        links.add(joint.parent_link)
        links.add(joint.child_link)
    for owner in description.geometry_semantics.owners:
        links.add(owner.reference_link)
    for component in description.geometry_semantics.components:
        links.add(component.carrier_link)
    return tuple(sorted(links))


def _finite_vector3(values: Sequence[float], *, field_name: str) -> tuple[float, float, float]:
    'Validate a finite SI translation vector of length 3.'

    if len(values) != 3:
        raise ValueError(f"{field_name} must contain three finite values")
    result = (float(values[0]), float(values[1]), float(values[2]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain three finite values")
    return result


def _matrix4_from_pose(rotation: Sequence[float], translation: Sequence[float]) -> tuple[float, ...]:
    'Build a row-major 4x4 transform from a row-major 3x3 rotation and translation.'

    if len(rotation) != 9 or len(translation) != 3:
        raise ValueError("pose requires a 3x3 rotation and 3-vector translation")
    values = tuple(float(value) for value in (*rotation, *translation))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose contains non-finite values")
    return (
        values[0],
        values[1],
        values[2],
        values[9],
        values[3],
        values[4],
        values[5],
        values[10],
        values[6],
        values[7],
        values[8],
        values[11],
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _matrix4_multiply(lhs: Sequence[float], rhs: Sequence[float]) -> tuple[float, ...]:
    'Compose two row-major 4x4 homogeneous transforms.'

    if len(lhs) != 16 or len(rhs) != 16:
        raise ValueError("homogeneous transforms must contain 16 values")
    left = tuple(float(value) for value in lhs)
    right = tuple(float(value) for value in rhs)
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("homogeneous transform contains non-finite values")
    return tuple(
        sum(left[row * 4 + inner] * right[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    )


def _compose_import_root_pose(
    description: OfficialHandSemanticsCfg,
    *,
    root_position_h: tuple[float, float, float],
    import_root_link: str,
    T_original_root_from_import_root: Sequence[float],
) -> dict[str, object]:
    (
        'Compute import-root world pose after frame-only lowering. First build '
        'T_world_original_root, then right-multiply T_original_root_from_import_root. '
        'For LEAP this moves the pose from empty base to massive palm_lower; Allegro '
        'uses identity compensation and keeps base_link.'
    )

    original_position_a = (
        root_position_h[0] + float(description.semantic_p_ha[0]),
        root_position_h[1] + float(description.semantic_p_ha[1]),
        root_position_h[2] + float(description.semantic_p_ha[2]),
    )
    original_root = _matrix4_from_pose(description.semantic_R_ha, original_position_a)
    root_transform = tuple(float(value) for value in T_original_root_from_import_root)
    if len(root_transform) != 16:
        raise ValueError("T_original_root_from_import_root must contain 16 values")
    world_import_root = _matrix4_multiply(original_root, root_transform)
    import_rotation = tuple(world_import_root[index] for index in (0, 1, 2, 4, 5, 6, 8, 9, 10))
    original_quat = _quat_wxyz_from_matrix3(description.semantic_R_ha)
    import_quat = _quat_wxyz_from_matrix3(import_rotation)
    return {
        "original_root_link": description.root_link,
        "import_root_link": import_root_link,
        "T_original_root_from_import_root": root_transform,
        "T_world_original_root": original_root,
        "T_world_import_root": world_import_root,
        "original_root_position_a": original_position_a,
        "original_root_quat_wxyz": original_quat,
        "import_root_position_a": (
            world_import_root[3],
            world_import_root[7],
            world_import_root[11],
        ),
        "import_root_quat_wxyz": import_quat,
    }


def _project_zero_to_limit(lower_rad: float, upper_rad: float) -> float:
    'Project importer initial value 0 into official limits without changing mathematical q_home.'

    return min(max(0.0, float(lower_rad)), float(upper_rad))


def _verify_description_source_digest(description: OfficialHandSemanticsCfg, source_urdf: Path) -> None:
    'Confirm description matches current source URDF bytes; reject stale metadata before spawn.'

    actual_digest = _sha256_file(source_urdf)
    if actual_digest.lower() != description.source_digest.lower():
        raise ValueError(
            f"official source URDF digest mismatch: description={description.source_digest}, actual={actual_digest}"
        )


def _sha256_file(path: Path) -> str:
    'Stream source-byte hashing to avoid loading all official URDF/mesh files into memory.'

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quat_wxyz_from_matrix3(values: Sequence[float]) -> tuple[float, float, float, float]:
    'Lower row-major R_ha to an Isaac Lab (w,x,y,z) quaternion.'

    if len(values) != 9:
        raise ValueError("semantic_R_ha must contain 9 row-major values")
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = (float(value) for value in values)
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw, qx, qy, qz = 0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw, qx, qy, qz = (m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw, qx, qy, qz = (m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw, qx, qy, qz = (m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("semantic_R_ha produced an invalid quaternion")
    return (qw / norm, qx / norm, qy / norm, qz / norm)


def _read_source_joint_friction(
    description: OfficialHandSemanticsCfg,
    source_urdf: Path,
) -> dict[str, float]:
    'Read official URDF joint_properties friction; preserve importer default when absent.'

    root = ET.parse(source_urdf).getroot()
    expected = {joint.source_name for joint in description.joints}
    friction: dict[str, float] = {}
    for joint_elem in root.findall("./joint"):
        name = joint_elem.attrib.get("name")
        if name not in expected:
            continue
        node = joint_elem.find("./joint_properties")
        if node is None or node.attrib.get("friction") is None:
            continue
        value = float(node.attrib["friction"])
        if not math.isfinite(value):
            raise ValueError(f"official joint {name!r} friction must be finite")
        friction[name] = value
    return friction


def _build_official_cache_dir(
    description: OfficialHandSemanticsCfg,
    *,
    root_position_h: tuple[float, float, float],
    root_position_a: tuple[float, float, float],
    root_quat_wxyz: tuple[float, float, float, float],
    import_root_link: str,
    T_original_root_from_import_root: tuple[float, ...],
    cache_dir: Path,
    render: bool,
    importer_config: Mapping[str, object],
) -> Path:
    (
        'Return an official USD-cache child path without creating directories. Key by '
        'source/mesh hashes, frame/root pose, root compensation, render intent, and '
        'importer params so distinct families/calibrations/lowering choices cannot '
        'share a cache.'
    )

    payload = {
        "schema": _OFFICIAL_SPAWN_SCHEMA,
        "family": description.family,
        "asset_id": description.asset_id,
        "source_urdf_path": description.source_urdf_path,
        "source_urdf_sha256": description.source_digest,
        "mesh_digests": [asdict(item) for item in description.mesh_digests],
        "description_content_hash": description.content_hash,
        "semantic_R_ha": list(description.semantic_R_ha),
        "semantic_p_ha": list(description.semantic_p_ha),
        "root_position_h": list(root_position_h),
        "root_position_a": list(root_position_a),
        "root_quat_wxyz": list(root_quat_wxyz),
        "import_root_link": import_root_link,
        "T_original_root_from_import_root": list(T_original_root_from_import_root),
        "render": bool(render),
        "importer_config": dict(importer_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()
    return Path(cache_dir).expanduser().resolve(strict=False) / "official_hand" / description.family / key


def _build_official_render_plan(
    description: OfficialHandSemanticsCfg,
    source_urdf: Path,
    *,
    render: bool,
) -> tuple[Any | None, dict[str, object]]:
    'Build a native material plan from official source; never guess when named material evidence is absent.'

    from anymani.robots._visual_materials import audit_visual_material_restore_plan, build_visual_material_restore_plan

    plan = build_visual_material_restore_plan(source_urdf) if render else None
    if plan is not None:
        # The official importer sanitizes Allegro link dots; include that mapping in render targets;
        # keep source visual names/links unchanged in plan and audit.
        sanitized_links = {
            visual_name: source_link_to_usd_link_name(description, source_link)
            for visual_name, source_link in plan.source_visual_link_by_name.items()
            if visual_name in plan.visual_rgba_by_name
        }
        if sanitized_links != plan.visual_link_by_name:
            from dataclasses import replace

            plan = replace(plan, visual_link_by_name=sanitized_links)
        audit = audit_visual_material_restore_plan(plan)
        visual_nodes = ET.parse(source_urdf).getroot().findall(".//visual")
        if not plan.visual_rgba_by_name:
            audit = {
                **audit,
                "status": "importer_default",
                "reason": "official_visual_has_no_reliable_named_material",
                "visual_count": len(visual_nodes),
                "named_visual_count": sum(bool(node.attrib.get("name")) for node in visual_nodes),
                "colored_visual_count": sum(node.find("./material/color") is not None for node in visual_nodes),
            }
    else:
        visual_nodes = ET.parse(source_urdf).getroot().findall(".//visual")
        named_count = sum(bool(node.attrib.get("name")) for node in visual_nodes)
        colored_count = sum(node.find("./material/color") is not None for node in visual_nodes)
        audit = {
            "provenance": "none",
            "status": "importer_default",
            "reason": "render_disabled" if not render else "official_visual_has_no_reliable_named_material",
            "visual_count": len(visual_nodes),
            "named_visual_count": named_count,
            "colored_visual_count": colored_count,
        }
    return plan, audit


def _build_official_spawn_audit(
    description: OfficialHandSemanticsCfg,
    *,
    source_urdf: Path,
    root_position_h: tuple[float, float, float],
    root_pose: Mapping[str, object],
    usd_joint_names: tuple[str, ...],
    source_friction_by_name: Mapping[str, float],
    usd_cache_path: Path,
    render: bool,
    render_wrapper_enabled: bool,
    render_audit: Mapping[str, object],
    importer_config: Mapping[str, object],
    import_source_audit: Mapping[str, object],
) -> dict[str, object]:
    'Collect source/hash/frame/motor/render evidence into a JSON-safe spawn audit.'

    joints = [
        {
            "canonical_slot": joint.canonical_name,
            "source_joint_name": joint.source_name,
            "usd_joint_name": usd_name,
            "source_lower_rad": float(joint.lower_rad),
            "source_upper_rad": float(joint.upper_rad),
            "source_effort_cap_nm": float(joint.effort_nm),
            "source_velocity_cap_rad_s": float(joint.velocity_rad_s),
            "source_friction": source_friction_by_name.get(joint.source_name),
            "math_q_home_rad": 0.0,
            "runtime_boot_position_rad": _project_zero_to_limit(joint.lower_rad, joint.upper_rad),
        }
        for joint, usd_name in zip(description.joints, usd_joint_names, strict=True)
    ]
    source_links = {
        source_name: source_link_to_usd_link_name(description, source_name)
        for source_name in _known_source_links(description)
    }
    root_frame = {
        "formula": "T_world_importroot=T_world_hand*T_hand_originalroot*T_original_root_from_import_root",
        "R_wh": list(_IDENTITY_R),
        "root_position_h_m": list(root_position_h),
        "semantic_R_ha": list(description.semantic_R_ha),
        "semantic_p_ha_m": list(description.semantic_p_ha),
        "original_root_link": root_pose["original_root_link"],
        "import_root_link": root_pose["import_root_link"],
        "root_position_a_m": list(cast(tuple[float, float, float], root_pose["import_root_position_a"])),
        "root_quaternion_wxyz": list(cast(tuple[float, float, float, float], root_pose["import_root_quat_wxyz"])),
        "import_root_position_a_m": list(cast(tuple[float, float, float], root_pose["import_root_position_a"])),
        "import_root_quaternion_wxyz": list(
            cast(tuple[float, float, float, float], root_pose["import_root_quat_wxyz"])
        ),
        "original_root_position_a_m": list(cast(tuple[float, float, float], root_pose["original_root_position_a"])),
        "original_root_quaternion_wxyz": list(
            cast(tuple[float, float, float, float], root_pose["original_root_quat_wxyz"])
        ),
        "T_original_root_from_import_root": list(
            cast(tuple[float, ...], root_pose["T_original_root_from_import_root"])
        ),
        "T_world_original_root": list(cast(tuple[float, ...], root_pose["T_world_original_root"])),
        "T_world_import_root": list(cast(tuple[float, ...], root_pose["T_world_import_root"])),
        "root_pose_compensated": bool(
            cast(tuple[float, ...], root_pose["T_original_root_from_import_root"]) != _IDENTITY_T
        ),
    }
    return {
        "schema": _OFFICIAL_SPAWN_SCHEMA,
        "family": description.family,
        "asset_id": description.asset_id,
        "source_urdf": {
            "path": str(source_urdf),
            "sha256": description.source_digest,
        },
        "mesh_digests": [asdict(item) for item in description.mesh_digests],
        "description_content_hash": description.content_hash,
        "root_frame": root_frame,
        "source_link_to_usd_link": source_links,
        "joints": joints,
        "actuator": {
            "type": "implicit_pd",
            "stiffness": _OFFICIAL_STIFFNESS,
            "damping": _OFFICIAL_DAMPING,
            "armature": _OFFICIAL_ARMATURE,
            "friction_semantics": "source joint_properties friction; absent values remain importer default",
        },
        "importer": dict(importer_config),
        "inertial_by_link": import_source_audit.get("inertial_by_link"),
        "source_inertial_by_link": import_source_audit.get("source_inertial_by_link"),
        "expected_total_mass_kg": import_source_audit.get("expected_total_mass_kg"),
        "removed_reference_frames": import_source_audit.get("removed_reference_frames"),
        "removed_reference_joints": import_source_audit.get("removed_reference_joints"),
        "removed_reference_frame_transforms": import_source_audit.get("removed_reference_frame_transforms"),
        "import_source": dict(import_source_audit),
        "usd_cache_path": str(usd_cache_path),
        "render": {
            "requested": bool(render),
            "material_wrapper_enabled": bool(render_wrapper_enabled),
            "material_plan": dict(render_audit),
        },
    }


__all__ = [
    "audit_official_hand_cfg",
    "build_official_hand_cfg",
    "build_official_hand_spawn_audit",
    "resolve_official_joint_indices",
    "resolve_official_link_name",
    "source_joint_to_usd_joint_name",
    "source_link_to_usd_link_name",
]
