"""Non-UI helpers for ``single_asset_grasp_calibrator.py``.

The calibrator script is intentionally GUI-heavy. This helper keeps IsaacLab
asset construction, fallback preset parsing, scene cfg construction, and USD
collision-filter authoring out of the panel code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from anymani.assets.bank import HandBankCfg
from anymani.assets.bank.path_utils import resolve_bank_path
from anymani.robots.hand_spawn import (
    DEFAULT_HAND_ANCHOR_POS_E,
    HandFrameCfg,
    HandSpawnAdapter,
    HandSpawnCfg,
    HandUrdfSpawnCfg,
)

DEFAULT_HAND_BUNDLE_ID = (
    "source/anymani/anymani/assets/generated/2026-06-10_11-30-08/"
    "single_palm_leap/right_t4_i4_m4_r4"
)
"""Default generated mother bundle calibrated by the GUI."""

OFFICIAL_LEAP_URDF_PATH = Path(__file__).resolve().parents[2] / "assets/hands/leap_hand/leap_hand_right.urdf"
"""Project-local official LEAP raw URDF."""

OFFICIAL_LEAP_USD_PATH = Path(__file__).resolve().parents[2] / "assets/leap_hand_v1_right/leap_hand_right_edit.usd"
"""Project-local preconverted official LEAP USD."""

OFFICIAL_LEAP_ROOT_ROT_WXYZ = (0.5, 0.5, -0.5, 0.5)
"""Root orientation used by the existing LEAP scene seed."""

DEFAULT_OBJECT_SCALE = (1.2, 1.2, 1.2)
"""DexCube scale used when the GUI is launched with ``--object-source dex_cube_usd``."""

DEFAULT_LOCAL_CUBE_SIZE = (0.06, 0.06, 0.06)
"""Local cuboid fallback size in meters."""

DEFAULT_OBJECT_POS_CFG = (0.0, 0.075, 0.56)
"""Fallback object initial position in env frame."""

DEFAULT_OBJECT_RPY_XYZ = (0.0, 0.0, 0.0)
"""Fallback object orientation in XYZ Euler angles."""

DEFAULT_PREGRASP_JOINT_POS_DEG = {
    "index_j0": 0.0,
    "index_j1": -22.0,
    "index_j2": 82.0,
    "index_j3": 74.0,
    "middle_j0": 0.0,
    "middle_j1": 0.0,
    "middle_j2": 72.0,
    "middle_j3": 62.0,
    "ring_j0": 6.0,
    "ring_j1": 27.0,
    "ring_j2": 81.0,
    "ring_j3": 50.0,
    "thumb_j0": 7.0,
    "thumb_j1": 90.0,
    "thumb_j2": 33.0,
    "thumb_j3": 98.0,
}
"""Human-readable fallback pre-grasp seed in degrees."""

DEFAULT_PREGRASP_JOINT_POS_RAD = {
    joint_name: math.radians(float(value_deg)) for joint_name, value_deg in DEFAULT_PREGRASP_JOINT_POS_DEG.items()
}
"""Fallback pre-grasp seed in radians."""

GENERATED_FINGER_LINK_CHAINS = {
    "index": ("index_root_fixed_link", "index_mcp1", "index_mcp2", "index_pip", "index_dip", "index_tip"),
    "middle": ("middle_root_fixed_link", "middle_mcp1", "middle_mcp2", "middle_pip", "middle_dip", "middle_tip"),
    "ring": ("ring_root_fixed_link", "ring_mcp1", "ring_mcp2", "ring_pip", "ring_dip", "ring_tip"),
    "thumb": ("thumb_cmc1", "thumb_cmc2", "thumb_mcp", "thumb_dip", "thumb_tip"),
}
"""Generated LEAP-like hand link chains used by collision-filter ablations."""

GENERATED_FINGER_LINK_CHAINS_BY_NAME = tuple(GENERATED_FINGER_LINK_CHAINS.values())
"""Generated finger link chains without semantic finger labels."""

GENERATED_COLLISION_GROUP_ROOT = "/World/anymani_calibrator_generated_collision_filters"
"""USD scope for temporary generated-hand collision filter groups."""


def _round_float(value: float, ndigits: int = 8) -> float:
    """Round exported YAML floats for readable diffs."""

    return round(float(value), ndigits)


def _as_float_list(values: Any, *, expected_len: int, field_name: str) -> list[float]:
    """Parse a fixed-length float list from a YAML payload."""

    if not isinstance(values, (list, tuple)) or len(values) != expected_len:
        raise ValueError(f"Preset field {field_name!r} must be a list of length {expected_len}, got {values!r}.")
    return [float(value) for value in values]


def _quat_from_rpy_xyz(rpy_xyz: tuple[float, float, float], device: str) -> torch.Tensor:
    """Convert XYZ Euler angles to IsaacLab ``wxyz`` quaternion."""

    roll = torch.tensor([float(rpy_xyz[0])], dtype=torch.float32, device=device)
    pitch = torch.tensor([float(rpy_xyz[1])], dtype=torch.float32, device=device)
    yaw = torch.tensor([float(rpy_xyz[2])], dtype=torch.float32, device=device)
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)


def _rpy_xyz_from_quat(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    """Convert IsaacLab ``wxyz`` quaternion to XYZ Euler angles for sliders."""

    quat_batch = quat_wxyz.reshape(1, 4)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_batch)
    return (float(roll[0].item()), float(pitch[0].item()), float(yaw[0].item()))


def _resolve_hand_bundle_input(hand_bundle: str | None) -> str:
    """Resolve CLI hand-bundle input to an absolute generated bundle path."""

    if hand_bundle is None:
        return str(resolve_bank_path(DEFAULT_HAND_BUNDLE_ID))

    candidate = Path(hand_bundle).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(resolve_bank_path(hand_bundle))


def _build_hand_spawn_cfg(hand_bundle_path: str) -> HandSpawnCfg:
    """Build the generated hand spawn cfg used by the calibrator."""

    return HandSpawnCfg(
        bank=HandBankCfg(
            source_mode="post_mutate",
            selection_mode="explicit",
            containers=(hand_bundle_path,),
            validate_mesh_relpaths=True,
            parse_visual_rgba=True,
        ),
        frame=HandFrameCfg(
            semantic_R_ha=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            semantic_p_ha=(0.0, 0.0, 0.0),
            anchor_R_eh=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            anchor_p_eh=DEFAULT_HAND_ANCHOR_POS_E,
        ),
        urdf=HandUrdfSpawnCfg(activate_contact_sensors=False),
        asset_routing="round_robin",
        restore_visual_materials=True,
        validate_same_schema=True,
    )


def _resolve_official_leap_urdf_input(urdf_path_arg: str | None) -> Path:
    """Resolve the official LEAP raw URDF path."""

    urdf_path = Path(urdf_path_arg).expanduser() if urdf_path_arg is not None else OFFICIAL_LEAP_URDF_PATH
    urdf_path = urdf_path.resolve(strict=False)
    if not urdf_path.is_file():
        raise FileNotFoundError(f"official LEAP URDF does not exist: {urdf_path}")
    return urdf_path


def _resolve_official_leap_usd_input(usd_path_arg: str | None) -> Path:
    """Resolve the official LEAP preconverted USD path."""

    usd_path = Path(usd_path_arg).expanduser() if usd_path_arg is not None else OFFICIAL_LEAP_USD_PATH
    usd_path = usd_path.resolve(strict=False)
    if not usd_path.is_file():
        raise FileNotFoundError(f"official LEAP USD does not exist: {usd_path}")
    return usd_path


def _build_generated_hand_articulation_cfg(hand_spawn_cfg: HandSpawnCfg) -> ArticulationCfg:
    """Lower a generated hand bundle to an IsaacLab articulation cfg."""

    return HandSpawnAdapter(hand_spawn_cfg).build_articulation_cfg(prim_path="{ENV_REGEX_NS}/Robot")


def _build_official_leap_usd_articulation_cfg(usd_path: Path) -> ArticulationCfg:
    """Build the official LEAP USD articulation cfg for the GUI probe."""

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64.0 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DEFAULT_HAND_ANCHOR_POS_E,
            rot=OFFICIAL_LEAP_ROOT_ROT_WXYZ,
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.5,
                velocity_limit_sim=100.0,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
                armature=0.001,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def _build_official_leap_articulation_cfg(urdf_path: Path) -> ArticulationCfg:
    """Build the official LEAP raw URDF articulation cfg for fallback probing."""

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            fix_base=True,
            merge_fixed_joints=False,
            force_usd_conversion=False,
            make_instanceable=True,
            collision_from_visuals=False,
            self_collision=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=3.0, damping=0.1),
            ),
            activate_contact_sensors=False,
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
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DEFAULT_HAND_ANCHOR_POS_E,
            rot=OFFICIAL_LEAP_ROOT_ROT_WXYZ,
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.95,
                velocity_limit_sim=8.48,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
                armature=0.001,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def _build_object_spawn_cfg(object_source: str) -> Any:
    """Build the object spawn cfg for the GUI calibrator."""

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=True,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
        sleep_threshold=0.005,
        stabilization_threshold=0.0025,
        max_depenetration_velocity=1000.0,
    )
    mass_props = sim_utils.MassPropertiesCfg(density=400.0)

    if object_source == "local_cube":
        return sim_utils.CuboidCfg(
            size=DEFAULT_LOCAL_CUBE_SIZE,
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    if object_source == "dex_cube_usd":
        return sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=rigid_props,
            mass_props=mass_props,
            scale=DEFAULT_OBJECT_SCALE,
        )
    raise ValueError(f"unknown object source: {object_source!r}")


def _joint_pos_from_preset(payload: Mapping[str, Any] | None) -> dict[str, float]:
    """Read ``joint_pos_rad`` from a preset payload, or return the fallback seed."""

    if payload is None:
        return dict(DEFAULT_PREGRASP_JOINT_POS_RAD)
    joint_pos = payload.get("joint_pos_rad")
    if not isinstance(joint_pos, dict):
        return dict(DEFAULT_PREGRASP_JOINT_POS_RAD)
    return {str(joint_name): float(value) for joint_name, value in joint_pos.items()}


def _object_pose_from_preset(
    payload: Mapping[str, Any] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Read object cfg pose from a preset payload, or return the fallback seed."""

    if payload is None:
        return DEFAULT_OBJECT_POS_CFG, DEFAULT_OBJECT_RPY_XYZ

    object_pose_cfg = payload.get("object_pose_cfg")
    if not isinstance(object_pose_cfg, dict):
        return DEFAULT_OBJECT_POS_CFG, DEFAULT_OBJECT_RPY_XYZ

    pos_cfg = tuple(_as_float_list(object_pose_cfg.get("pos"), expected_len=3, field_name="object_pose_cfg.pos"))
    if "rpy_xyz_rad" in object_pose_cfg:
        rpy_xyz = tuple(
            _as_float_list(object_pose_cfg.get("rpy_xyz_rad"), expected_len=3, field_name="object_pose_cfg.rpy_xyz_rad")
        )
        return pos_cfg, rpy_xyz

    rot_wxyz = torch.tensor(
        _as_float_list(object_pose_cfg.get("rot_wxyz"), expected_len=4, field_name="object_pose_cfg.rot_wxyz"),
        dtype=torch.float32,
    )
    return pos_cfg, _rpy_xyz_from_quat(rot_wxyz)


def _hand_bundle_from_preset(payload: Mapping[str, Any] | None) -> str | None:
    """Read the generated hand bundle path stored in a preset payload."""

    if payload is None:
        return None
    asset = payload.get("asset")
    if not isinstance(asset, dict):
        return None
    hand_bundle = asset.get("hand_bundle")
    return str(hand_bundle) if hand_bundle is not None else None


def _make_scene_cfg(
    hand_cfg: ArticulationCfg,
    object_source: str,
    object_pos_cfg: tuple[float, float, float],
    object_rpy_xyz: tuple[float, float, float],
    device: str,
) -> InteractiveSceneCfg:
    """Construct the one-env calibration scene cfg."""

    object_quat = tuple(float(v) for v in _quat_from_rpy_xyz(object_rpy_xyz, device).cpu()[0].tolist())
    object_spawn_cfg = _build_object_spawn_cfg(object_source)

    @configclass
    class SingleAssetCalibrationSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = hand_cfg
        object: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object",
            spawn=object_spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos_cfg, rot=object_quat),
        )
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
        )
        light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=1000.0),
        )

    return SingleAssetCalibrationSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=True)


def _apply_generated_collision_filter(scene: InteractiveScene, filter_mode: str) -> None:
    """Author temporary generated-hand collision filtering before ``sim.reset()``."""

    if filter_mode == "none":
        return
    if filter_mode not in {"finger_palm", "finger_palm_same_finger"}:
        raise ValueError(f"unknown generated collision filter mode: {filter_mode!r}")
    if not hasattr(UsdPhysics, "CollisionGroup"):
        raise RuntimeError("Current USD build does not expose UsdPhysics.CollisionGroup")

    link_names = _generated_collision_filter_link_names(filter_mode)
    link_group_paths = _author_generated_link_collision_groups(scene, link_names)
    filtered_link_pairs = _generated_filtered_link_pairs(filter_mode)
    authored_group_edges = 0
    missing_link_names: set[str] = set()

    for link_a, link_b in filtered_link_pairs:
        group_a_path = link_group_paths.get(link_a)
        group_b_path = link_group_paths.get(link_b)
        if group_a_path is None or group_b_path is None:
            missing_link_names.update(
                link for link, group_path in ((link_a, group_a_path), (link_b, group_b_path)) if group_path is None
            )
            continue
        authored_group_edges += _author_generated_filtered_group_edge(scene.stage, group_a_path, group_b_path)
        authored_group_edges += _author_generated_filtered_group_edge(scene.stage, group_b_path, group_a_path)

    if missing_link_names:
        print(f"[WARN]: generated collision filter skipped missing hand links: {sorted(missing_link_names)}")
    print(
        "[INFO]: generated collision filter authored "
        f"mode={filter_mode!r}, groups={len(link_group_paths)}, "
        f"link_pairs={len(filtered_link_pairs)}, directed_edges={authored_group_edges}"
    )


def _generated_collision_filter_link_names(filter_mode: str) -> tuple[str, ...]:
    """Return generated links participating in the requested collision filter."""

    if filter_mode == "none":
        return tuple()
    link_names = {"palm"}
    for finger_link_chain in GENERATED_FINGER_LINK_CHAINS_BY_NAME:
        link_names.update(finger_link_chain)
    return tuple(sorted(link_names))


def _generated_filtered_link_pairs(filter_mode: str) -> tuple[tuple[str, str], ...]:
    """Build unordered generated-hand link pairs to filter."""

    filtered_pairs: set[tuple[str, str]] = set()
    if filter_mode == "none":
        return tuple()
    for finger_link_chain in GENERATED_FINGER_LINK_CHAINS_BY_NAME:
        filtered_pairs.update(tuple(sorted(("palm", link_name))) for link_name in finger_link_chain)
    if filter_mode == "finger_palm_same_finger":
        for finger_link_chain in GENERATED_FINGER_LINK_CHAINS_BY_NAME:
            filtered_pairs.update(tuple(sorted(pair)) for pair in combinations(finger_link_chain, 2))
    return tuple(sorted(filtered_pairs))


def _author_generated_link_collision_groups(
    scene: InteractiveScene,
    link_names: tuple[str, ...],
) -> dict[str, str]:
    """Create external USD ``PhysicsCollisionGroup`` prims per generated link."""

    stage = scene.stage
    root_layer = stage.GetRootLayer()
    link_group_paths: dict[str, str] = {}

    with Usd.EditContext(stage, Usd.EditTarget(root_layer)):
        UsdGeom.Scope.Define(stage, GENERATED_COLLISION_GROUP_ROOT)

    collision_group_root_spec = root_layer.GetPrimAtPath(GENERATED_COLLISION_GROUP_ROOT)
    if collision_group_root_spec is None:
        raise RuntimeError(f"Failed to define collision group scope at {GENERATED_COLLISION_GROUP_ROOT}")

    with Sdf.ChangeBlock():
        for link_name in link_names:
            first_env_link_path = _generated_link_prim_path(scene.env_prim_paths[0], link_name)
            if not stage.GetPrimAtPath(first_env_link_path).IsValid():
                continue

            collision_group = Sdf.PrimSpec(
                collision_group_root_spec,
                link_name,
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            collision_group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))

            expansion_rule = Sdf.AttributeSpec(
                collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            includes_rel = Sdf.RelationshipSpec(collision_group, "collection:colliders:includes", False)
            for env_prim_path in scene.env_prim_paths:
                includes_rel.targetPathList.Append(_generated_link_prim_path(env_prim_path, link_name))

            link_group_paths[link_name] = f"{GENERATED_COLLISION_GROUP_ROOT}/{link_name}"

    return link_group_paths


def _generated_link_prim_path(env_prim_path: str, link_name: str) -> str:
    """Return a generated hand link prim path inside one cloned env."""

    return f"{env_prim_path}/Robot/{link_name}"


def _author_generated_filtered_group_edge(stage, source_group_path: str, target_group_path: str) -> int:
    """Author one directed ``physics:filteredGroups`` edge."""

    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        source_group = UsdPhysics.CollisionGroup.Get(stage, source_group_path)
        if not source_group:
            raise RuntimeError(f"Missing generated collision group at {source_group_path}")

        filtered_groups_rel = source_group.GetFilteredGroupsRel()
        if not filtered_groups_rel:
            filtered_groups_rel = source_group.CreateFilteredGroupsRel()

        target_path = Sdf.Path(target_group_path)
        if target_path in set(filtered_groups_rel.GetTargets()):
            return 0
        filtered_groups_rel.AddTarget(target_path)
        return 1


__all__ = [
    "DEFAULT_HAND_BUNDLE_ID",
    "DEFAULT_LOCAL_CUBE_SIZE",
    "DEFAULT_OBJECT_SCALE",
    "_apply_generated_collision_filter",
    "_build_generated_hand_articulation_cfg",
    "_build_hand_spawn_cfg",
    "_build_official_leap_articulation_cfg",
    "_build_official_leap_usd_articulation_cfg",
    "_hand_bundle_from_preset",
    "_joint_pos_from_preset",
    "_make_scene_cfg",
    "_object_pose_from_preset",
    "_quat_from_rpy_xyz",
    "_resolve_hand_bundle_input",
    "_resolve_official_leap_urdf_input",
    "_resolve_official_leap_usd_input",
    "_round_float",
    "_rpy_xyz_from_quat",
]
