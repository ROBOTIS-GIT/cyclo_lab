# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
# Reset adapted from Cyclo's ReferenceTrajectoryCommand (whole_body_tracking MIT
# attribution retained in THIRD_PARTY_LICENSES.md); link-velocity writes added.
"""SuperMimic reference targets and reset twists use world link-origin velocities."""

import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform
from cyclo_lab.manager_based.mimic.mdp.commands import ReferenceTrajectoryCommand
from .motion_physics import validate_motion_physics, URDF
from .fk import UrdfKinematics


class SuperMimicReferenceCommand(ReferenceTrajectoryCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.motion_contract, root_velocity = validate_motion_physics(
            cfg.trajectory_file
        )
        self.resample_counts = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        ref = self.reference
        root = self.robot.body_names.index("pelvis")
        fk = UrdfKinematics(
            URDF, self.robot.joint_names, self.robot.body_names, self.device
        )
        _, _, link_velocity, angular_velocity = fk(
            ref._body_position_w[:, root],
            ref._body_orientation_w[:, root, [3, 0, 1, 2]],
            ref.joint_position,
            root_velocity.to(self.device),
            ref._body_angular_velocity_w[:, root],
            ref.joint_velocity,
        )
        # Canonicalize only the in-memory view. Source NPZ stays byte-for-byte intact.
        ref._body_linear_velocity_w = link_velocity
        ref._body_angular_velocity_w = angular_velocity

    @property
    def robot_body_lin_vel_w(self):
        return self.robot.data.body_link_lin_vel_w.torch[:, self.tracked_body_ids]

    @property
    def robot_body_ang_vel_w(self):
        return self.robot.data.body_link_ang_vel_w.torch[:, self.tracked_body_ids]

    @property
    def robot_anchor_lin_vel_w(self):
        return self.robot.data.body_link_lin_vel_w.torch[:, self.robot_anchor_body_id]

    @property
    def robot_anchor_ang_vel_w(self):
        return self.robot.data.body_link_ang_vel_w.torch[:, self.robot_anchor_body_id]

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return
        self.resample_counts[env_ids] += 1
        self._sample_start_frames(env_ids)
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_vel = torch.cat(
            [self.body_lin_vel_w[env_ids, 0], self.body_ang_vel_w[env_ids, 0]], -1
        )
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        ranges = torch.tensor(
            [self.cfg.reset_pose_noise.get(k, (0.0, 0.0)) for k in keys],
            device=self.device,
        )
        noise = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos += noise[:, :3]
        root_ori = quat_mul(
            quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]), root_ori
        )
        ranges = torch.tensor(
            [self.cfg.reset_velocity_noise.get(k, (0.0, 0.0)) for k in keys],
            device=self.device,
        )
        root_vel += sample_uniform(
            ranges[:, 0], ranges[:, 1], root_vel.shape, device=self.device
        )
        joint_pos = self.joint_pos[env_ids].clone()
        joint_pos += sample_uniform(
            *self.cfg.reset_joint_position_noise, joint_pos.shape, joint_pos.device
        )
        limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = torch.clamp(joint_pos, limits[:, :, 0], limits[:, :, 1])
        self.robot.write_joint_state_to_sim_index(
            position=joint_pos, velocity=self.joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat([root_pos, root_ori], -1), env_ids=env_ids
        )
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=root_vel, env_ids=env_ids
        )
