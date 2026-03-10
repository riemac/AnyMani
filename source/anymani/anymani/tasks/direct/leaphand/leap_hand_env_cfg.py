# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand direct reorientation environment configuration migrated from LEAP_Isaaclab."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from anymani.robots.leap import LEAP_HAND_CFG


@configclass
class EventCfg:
    """Configuration for domain randomization and ADR-backed events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # pyright: ignore[reportArgumentType]
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,  # pyright: ignore[reportArgumentType]
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (3.0, 3.0),
            "damping_distribution_params": (0.1, 0.1),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # pyright: ignore[reportArgumentType]
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # pyright: ignore[reportArgumentType]
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    object_scale_size = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": (1.1, 1.25),
        },
    )


@configclass
class LeapHandEnvCfg(DirectRLEnvCfg):
    """Configuration for the migrated LEAP direct reorientation task."""

    decimation = 4
    min_episode_length_s = 20.0
    episode_length_s = 120.0

    act_dim = 16
    hist_len = 3
    obs_frame_dim = 32
    action_space = 16
    observation_space = 96
    state_space = 0
    store_cur_actions = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**25,
            gpu_max_rigid_patch_count=2**25,
        ),
    )

    robot_cfg = LEAP_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # pyright: ignore[reportAttributeAccessIssue]
    actuated_joint_names = [
        "a_0",
        "a_1",
        "a_2",
        "a_3",
        "a_4",
        "a_5",
        "a_6",
        "a_7",
        "a_8",
        "a_9",
        "a_10",
        "a_11",
        "a_12",
        "a_13",
        "a_14",
        "a_15",
    ]
    fingertip_body_names = ["fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.00, -0.1, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
            )
        },
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=False)

    z_rotation_steps = 16
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    torque_penalty_scale = -0.0
    pose_diff_penalty_scale = -0.3
    reach_goal_bonus = 250
    fall_penalty = -10
    fall_dist = 0.07
    success_tolerance = 0.2
    av_factor = 0.1
    action_type = "relative"
    act_moving_average = 1.0 / 24.0

    enable_adr = True
    starting_adr_increments = 0
    min_rot_adr_coeff = 0.15
    min_steps_for_dr_change = 240 * 4
    obs_per_timestep = 32
    obs_timesteps = 3

    wrench_trigger_every = 90
    torsional_radius = 0.0
    wrench_prob_per_rollout = 0.5

    events: EventCfg = EventCfg()

    adr_cfg_dict = {
        "num_increments": 25,
        "robot_physics_material": {
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.5),
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (2.5, 3.1),
            "damping_distribution_params": (0.05, 0.15),
        },
        "object_physics_material": {
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.5),
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.9, 1.3),
        },
    }
    adr_custom_cfg_dict = {
        "object_wrench": {
            "max_linear_accel": (0.5, 5.0),
        },
        "object_spawn": {
            "x_width_spawn": (0.0, 0.01),
            "y_width_spawn": (0.0, 0.01),
            "x_rotation": (0.0, 0.1),
            "y_rotation": (0.0, 0.1),
            "z_rotation": (0.0, 0.0),
        },
        "object_state_noise": {
            "object_pos_noise": (0.0, 0.0),
            "object_pos_bias": (0.0, 0.0),
            "object_rot_noise": (0.0, 0.0),
            "object_rot_bias": (0.0, 0.0),
        },
        "robot_spawn": {
            "joint_pos_noise": (0.0, 0.05),
            "joint_vel_noise": (0.0, 0.01),
        },
        "robot_state_noise": {
            "robot_noise": (0.0, 0.05),
            "robot_bias": (0.0, 0.03),
        },
        "robot_action_noise": {
            "hand_noise": (0.1, 0.2),
        },
        "action_latency": {
            "hand_latency": (0.0, 3.0),
        },
        "obs_latency": {
            "latency": (0.0, 0.0),
        },
    }

    act_max_latency = int(adr_custom_cfg_dict["action_latency"]["hand_latency"][1])
    act_latency_rand = 1
    obs_max_latency = int(adr_custom_cfg_dict["obs_latency"]["latency"][1])
    obs_latency_rand = 1
