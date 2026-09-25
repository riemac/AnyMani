"""Isaac Lab articulation settings for the packaged LEAP hand asset."""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

LEAP_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{Path(__file__).parent.parent.parent}/assets/leap_hand_v1_right/leap_hand_right_edit.usd",
        activate_contact_sensors=True,  # Required for the task's contact observations.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True
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
            # Isaac Lab's URDF uses 0.5 N m; LEAP_Hand_Sim uses 0.95 N m.
            # The real hardware limit has not been verified.
            effort_limit=0.5,
            velocity_limit=100.0,  # rad/s
            stiffness=3.0,  # N/m; low stiffness approximates compliant tendon actuation.
            damping=0.1,  # N s/m
            friction=0.01,  # Dimensionless simulator coefficient.
            armature=0.001,  # kg m^2, matching LEAP_Hand_Sim.
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Body names follow the packaged LEAP right-hand asset. Palm: palm_lower.
# Index: mcp_joint -> pip -> dip -> fingertip -> index_tip_head.
# Thumb: thumb_temp_base -> thumb_pip -> thumb_dip -> thumb_fingertip -> thumb_tip_head.
# Middle: mcp_joint_2 -> pip_2 -> dip_2 -> fingertip_2 -> middle_tip_head.
# Ring: mcp_joint_3 -> pip_3 -> dip_3 -> fingertip_3 -> ring_tip_head.
