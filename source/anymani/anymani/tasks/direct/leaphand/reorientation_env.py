# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand direct reorientation environment migrated from LEAP_Isaaclab."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    saturate,
)

from .sim2real.adr import LeapHandADR
from .sim2real.noise import apply_object_wrench, init_adr_noise_buffers, update_adr_noise_buffers
from .utils.history import ActionLatencyBuffer, ObservationHistoryBuffer
from .utils.rewards import compute_rewards, scale, unscale

if TYPE_CHECKING:
    from .leap_hand_env_cfg import LeapHandEnvCfg


class ReorientationEnv(DirectRLEnv):
    cfg: LeapHandEnvCfg

    def __init__(self, cfg: LeapHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.hand = self.scene.articulations["robot"]
        self.object = self.scene.rigid_objects["object"]

        self.act_dim = self.cfg.act_dim
        self.obs_frame_dim = self.cfg.obs_frame_dim
        self.num_hand_dofs = self.hand.num_joints

        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        self.actuated_dof_indices = [self.hand.joint_names.index(j) for j in self.cfg.actuated_joint_names]
        self.actuated_dof_indices.sort()

        self.finger_bodies = [self.hand.body_names.index(name) for name in self.cfg.fingertip_body_names]
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] += 0.01

        self.target_z_angle = torch.full(
            (self.num_envs,),
            2 * math.pi / self.cfg.z_rotation_steps,
            dtype=torch.float,
            device=self.device,
        )

        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], dtype=torch.float, device=self.device)

        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.override_default_joint_pos = torch.tensor(
            [[
                0.000,
                0.500,
                0.000,
                0.000,
                -0.750,
                1.300,
                0.000,
                0.750,
                1.750,
                1.500,
                1.750,
                1.750,
                0.000,
                1.000,
                0.000,
                0.000,
            ]],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)

        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_rot[:, 0] = 1.0

        self.obs_history = ObservationHistoryBuffer(
            num_envs=self.num_envs,
            frame_dim=self.obs_frame_dim,
            history_length=self.cfg.hist_len,
            device=self.device,
            extra_length=self.cfg.obs_max_latency if self.cfg.enable_adr else 0,
        )

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.randomized_episode_lengths = torch.randint(
            int(self.cfg.min_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)),
            self.max_episode_length + 1,
            (self.num_envs,),
            dtype=torch.int32,
            device=self.device,
        )

        if self.cfg.enable_adr:
            self.leap_adr = LeapHandADR(self.event_manager, self.cfg.adr_cfg_dict, self.cfg.adr_custom_cfg_dict)
            self.step_since_last_dr_change = 0
            self.leap_adr.set_num_increments(self.cfg.starting_adr_increments)
            init_adr_noise_buffers(self)
            self.obs_latency = torch.zeros(
                (self.num_envs, self.cfg.obs_per_timestep),
                device=self.device,
                dtype=torch.long,
            )
            self.act_latency = torch.zeros((self.num_envs, self.act_dim), device=self.device, dtype=torch.long)
            self.action_history = ActionLatencyBuffer(
                num_envs=self.num_envs,
                action_dim=self.act_dim,
                max_latency=self.cfg.act_max_latency,
                device=self.device,
            )

            print("starting ranges:")
            self.leap_adr.print_params()

        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        if self.cfg.enable_adr:
            hand_noise = self.leap_adr.get_custom_param_value("robot_action_noise", "hand_noise")
            if hand_noise > 0:
                self.actions = self.actions + torch.randn_like(self.actions) * hand_noise
            self.actions = self.action_history.append_and_get(self.actions, self.act_latency)

        self.actions = torch.clamp(self.actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        if self.cfg.action_type == "relative":
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.cfg.act_moving_average * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                targets,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        elif self.cfg.action_type == "absolute":
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                self.cur_targets[:, self.actuated_dof_indices],
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        else:
            raise ValueError(f"Unsupported action type: {self.cfg.action_type}. Must be relative or absolute.")

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        if self.cfg.enable_adr:
            apply_object_wrench(self, self.object, "object")

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    def _update_continuous_z_rotation(self, goal_env_ids: torch.Tensor):
        add_rot = quat_from_angle_axis(self.target_z_angle, self.z_unit_tensor)
        self.goal_rot[goal_env_ids] = quat_mul(add_rot[goal_env_ids], self.goal_rot[goal_env_ids])
        self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        frame = unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        if self.cfg.store_cur_actions:
            frame = torch.cat((frame, self.cur_targets), dim=-1)

        self.obs_history.append(frame)

        if self.cfg.enable_adr:
            obs = self.obs_history.get_with_latency(self.obs_latency, self.cfg.obs_timesteps)
        else:
            obs = self.obs_history.get()
        return {"policy": obs.float()}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        pose_diff_penalty = ((self.cur_targets[:, self.actuated_dof_indices] - self.override_default_joint_pos) ** 2).sum(-1)
        torque_penalty = (self.hand.data.computed_torque ** 2).sum(-1)

        total_reward, self.reset_goal_buf, self.successes[:], self.consecutive_successes[:] = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.fingertip_pos,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.object_linvel,
            self.object_angvel,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            pose_diff_penalty,
            self.cfg.pose_diff_penalty_scale,
            torque_penalty,
            self.cfg.torque_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean() / self.cfg.z_rotation_steps
        self.extras["log"]["pose_diff_penalty"] = pose_diff_penalty.mean()
        self.extras["log"]["torque_info"] = torque_penalty.mean()
        self.extras["log"]["object_linvel"] = torch.norm(self.object_linvel, p=1, dim=-1).mean()
        self.extras["log"]["roll"] = self.object_angvel[:, 0].mean()
        self.extras["log"]["pitch"] = self.object_angvel[:, 1].mean()
        self.extras["log"]["yaw"] = self.object_angvel[:, 2].mean()
        self.extras["log"]["avg_episode_length_s"] = (
            self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation
        ).mean()
        self.extras["log"]["min_episode_length_s"] = (
            self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation
        ).min()
        self.extras["log"]["max_episode_length_s"] = (
            self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation
        ).max()

        if self.cfg.enable_adr:
            adr_criteria = (
                (self.consecutive_successes / self.cfg.z_rotation_steps)
                / (self.randomized_episode_lengths.float().mean() * self.cfg.sim.dt * self.cfg.decimation)
            ).float().mean()
            self.extras["log"]["adr_criteria"] = adr_criteria

        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if goal_env_ids.numel() > 0:
            self._update_continuous_z_rotation(goal_env_ids)
            self.reset_goal_buf[goal_env_ids] = False

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist
        time_out = self.episode_length_buf >= self.randomized_episode_lengths - 1

        obj_z = matrix_from_quat(self.object_rot)[:, :, 2]
        goal_z = matrix_from_quat(self.goal_rot)[:, :, 2]
        diff = torch.sum(obj_z * goal_z, dim=1)
        flipped = torch.abs(diff) < 0.5

        return out_of_reach | flipped, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        env_ids_tensor = self.hand._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        num_ids = int(env_ids_tensor.numel())

        if self.cfg.enable_adr:
            adr_criteria = (
                (self.consecutive_successes.float().mean() / self.cfg.z_rotation_steps)
                / (self.randomized_episode_lengths.float().mean() * self.cfg.sim.dt * self.cfg.decimation)
            ).float().mean()

        super()._reset_idx(env_ids_tensor)  # pyright: ignore[reportArgumentType]

        self.randomized_episode_lengths[env_ids_tensor] = torch.randint(
            int(self.cfg.min_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)),
            self.max_episode_length + 1,
            (num_ids,),
            dtype=torch.int32,
            device=self.device,
        )

        object_default_state = self.object.data.default_root_state.clone()[env_ids_tensor]
        dof_pos = self.override_default_joint_pos[env_ids_tensor].clone()
        dof_vel = self.hand.data.default_joint_vel[env_ids_tensor].clone()

        object_default_state[:, 0:3] += self.scene.env_origins[env_ids_tensor]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids_tensor, 7:])

        if self.cfg.enable_adr:
            x_width = self.leap_adr.get_custom_param_value("object_spawn", "x_width_spawn")
            y_width = self.leap_adr.get_custom_param_value("object_spawn", "y_width_spawn")
            x_rot = self.leap_adr.get_custom_param_value("object_spawn", "x_rotation")
            y_rot = self.leap_adr.get_custom_param_value("object_spawn", "y_rotation")
            z_rot = self.leap_adr.get_custom_param_value("object_spawn", "z_rotation")

            if x_width > 0 or y_width > 0:
                pos_noise = sample_uniform(-1.0, 1.0, (num_ids, 2), device=self.device)
                object_default_state[:, 0] += pos_noise[:, 0] * x_width
                object_default_state[:, 1] += pos_noise[:, 1] * y_width

            if x_rot > 0:
                x_rot_noise = sample_uniform(-1.0, 1.0, (num_ids,), device=self.device)
                x_rot_quat = quat_from_angle_axis(x_rot_noise * x_rot, self.x_unit_tensor[env_ids_tensor])
                object_default_state[:, 3:7] = quat_mul(x_rot_quat, object_default_state[:, 3:7])

            if y_rot > 0:
                y_rot_noise = sample_uniform(-1.0, 1.0, (num_ids,), device=self.device)
                y_rot_quat = quat_from_angle_axis(y_rot_noise * y_rot, self.y_unit_tensor[env_ids_tensor])
                object_default_state[:, 3:7] = quat_mul(y_rot_quat, object_default_state[:, 3:7])

            if z_rot > 0:
                z_rot_noise = sample_uniform(-1.0, 1.0, (num_ids,), device=self.device)
                z_rot_quat = quat_from_angle_axis(z_rot_noise * z_rot, self.z_unit_tensor[env_ids_tensor])
                object_default_state[:, 3:7] = quat_mul(z_rot_quat, object_default_state[:, 3:7])

            joint_pos_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_pos_noise")
            joint_vel_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_vel_noise")

            if joint_pos_noise_width > 0:
                joint_pos_noise = sample_uniform(-1.0, 1.0, (num_ids, self.num_hand_dofs), device=self.device)
                dof_pos += joint_pos_noise * joint_pos_noise_width

            if joint_vel_noise_width > 0:
                joint_vel_noise = sample_uniform(-1.0, 1.0, (num_ids, self.num_hand_dofs), device=self.device)
                dof_vel += joint_vel_noise * joint_vel_noise_width

        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids_tensor)  # pyright: ignore[reportArgumentType]
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids_tensor)  # pyright: ignore[reportArgumentType]

        self.prev_targets[env_ids_tensor] = dof_pos
        self.cur_targets[env_ids_tensor] = dof_pos
        self.hand_dof_targets[env_ids_tensor] = dof_pos
        self.successes[env_ids_tensor] = 0
        self.reset_goal_buf[env_ids_tensor] = False
        self.obs_history.reset(env_ids_tensor)

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids_tensor)  # pyright: ignore[reportArgumentType]
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids_tensor)  # pyright: ignore[reportArgumentType]

        if self.cfg.enable_adr and num_ids > 0:
            self.action_history.reset(env_ids_tensor)
            update_adr_noise_buffers(self, env_ids_tensor)

            obs_latency_resets = self.leap_adr.get_custom_param_value("obs_latency", "latency") - torch.randint(
                0, self.cfg.obs_latency_rand + 1, (num_ids, 1), device=self.device
            )
            obs_latency_resets = torch.clamp(obs_latency_resets, min=0).long()
            self.obs_latency[env_ids_tensor, :] = obs_latency_resets.expand(-1, self.cfg.obs_per_timestep)

            act_latency_resets = self.leap_adr.get_custom_param_value("action_latency", "hand_latency") - torch.randint(
                0, self.cfg.act_latency_rand + 1, (num_ids, 1), device=self.device
            )
            act_latency_resets = torch.clamp(act_latency_resets, min=0).long()
            self.act_latency[env_ids_tensor, :] = act_latency_resets.expand(-1, self.act_dim)

            self.extras["log"]["num_adr_increases"] = self.leap_adr.num_increments()
            if (
                self.step_since_last_dr_change >= self.cfg.min_steps_for_dr_change
                and adr_criteria >= self.cfg.min_rot_adr_coeff
            ):
                self.step_since_last_dr_change = 0
                self.leap_adr.increase_ranges()
                self.leap_adr.print_params()
                self.consecutive_successes.fill_(0.0)
            else:
                self.step_since_last_dr_change += 1

            self.object_mass = self.object.root_physx_view.get_masses().to(device=self.device)
            self.apply_wrench = torch.rand(self.num_envs, device=self.device) <= self.cfg.wrench_prob_per_rollout

        self._compute_intermediate_values()
        roll, pitch, yaw = euler_xyz_from_quat(self.object_rot[env_ids_tensor])
        roll[:] = 0.0
        pitch[:] = 0.0
        self.goal_rot[env_ids_tensor] = quat_from_euler_xyz(roll, pitch, yaw)
        self._update_continuous_z_rotation(env_ids_tensor)

    def _compute_intermediate_values(self):
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w
