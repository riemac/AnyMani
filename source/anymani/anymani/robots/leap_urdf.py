(
    'Official LEAP hand URDF articulation config. This route is separate from '
    'robots.leap.LEAP_HAND_CFG, which uses historical USD/edited-USD. '
    'LEAP_HAND_URDF_CFG imports the official URDF directly through Isaac Lab. '
    'Keep both names explicit so training configs reveal which source is active. '
    'Current GM LEAP comparison uses URDF as source of truth.'
)

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

LEAP_HAND_URDF_PATH = Path(__file__).resolve().parents[2] / "assets" / "hands" / "leap_hand" / "leap_hand_right.urdf"
'Path to the official LEAP right-hand URDF, preserving original link/joint names and collision boxes.'


LEAP_HAND_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(LEAP_HAND_URDF_PATH),
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=False,
        make_instanceable=True,
        collision_from_visuals=False,
        self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=3.0,
                damping=0.1,
            ),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64.0 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1.0e32,
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
        pos=(0.0, 0.0, 0.5),
        rot=(0.5, 0.5, -0.5, 0.5),
        joint_pos={"a_.*": 0.0},
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
(
    'Official LEAP articulation config backed by the URDF importer. Set '
    'merge_fixed_joints=False to preserve palm_lower/fingertip/thumb_fingertip '
    'links for per-link contact sensors. Set activate_contact_sensors=True so '
    'imported rigid bodies receive PhysxContactReportAPI required by '
    'ContactSensorCfg. Keep fix_base and fix_root_link true for the hand-in-place '
    'reorientation baseline.'
)


__all__ = ["LEAP_HAND_URDF_CFG", "LEAP_HAND_URDF_PATH"]
