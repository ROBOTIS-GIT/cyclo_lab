# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
# Reset adapted from Cyclo's ReferenceTrajectoryCommand (whole_body_tracking MIT
# attribution retained in THIRD_PARTY_LICENSES.md); link-velocity writes added.
"""Motion library targets and reset twists use world link-origin velocities."""

import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform
from cyclo_lab.manager_based.mimic.beyondmimic.mdp.commands import ReferenceTrajectoryCommand
from cyclo_lab.utils.k1_motion_physics import validate_motion_physics, URDF
from cyclo_lab.utils.k1_kinematics import UrdfKinematics


def at_motion_end(env, command_name):
    command = env.command_manager.get_term(command_name)
    # Super targets the next frame; Blink targets its current physical frame.
    ahead = 1 if command.cfg.targets_next_frame else 2
    return torch.isin(command.frame_ids + ahead, command.motion_ends)


class MotionLibraryCommand(ReferenceTrajectoryCommand):
    @property
    def termination_frame_ids(self):
        # Isaac tests for termination before advancing the command. Blink observes
        # frame f before the step and compares the resulting pose with frame f+1.
        if getattr(self.cfg, "clip_timeout", False) and not self.cfg.targets_next_frame:
            ends = self.motion_ends[torch.bucketize(self.frame_ids, self.motion_ends, right=True)]
            return (self.frame_ids + 1).minimum(ends - 1)
        return self.frame_ids

    def _load_reference(self, path):
        paths = [path, *getattr(self.cfg, "additional_motions", ())]
        clips = [super(MotionLibraryCommand, self)._load_reference(p) for p in paths]
        if any(float(torch.as_tensor(clip.fps).item()) != 50 for clip in clips):
            raise ValueError("Native tracking requires 50 Hz references")
        self.motion_ends = torch.tensor([clip.num_frames for clip in clips], device=self.device).cumsum(0)
        contracts, velocities = zip(*(validate_motion_physics(p) for p in paths))
        self.motion_contracts = contracts
        self.root_velocity = torch.cat(velocities).to(self.device)
        reference = clips[0]
        if len(clips) > 1:
            for key in (
                "joint_position",
                "joint_velocity",
                "_body_position_w",
                "_body_orientation_w",
                "_body_linear_velocity_w",
                "_body_angular_velocity_w",
            ):
                setattr(reference, key, torch.cat([getattr(clip, key) for clip in clips]))
            reference.num_frames = int(self.motion_ends[-1])
        return reference

    def _sample_start_frames(self, env_ids):
        if getattr(self, "playback_frames", None) is not None:
            self.frame_ids[env_ids] = self.playback_frames[env_ids]
            return
        if not self.cfg.start_from_zero and getattr(self, "sampler", None) is not None:
            self.frame_ids[env_ids] = self.sampler.sample(env_ids)
            return
        if not self.cfg.start_from_zero or len(self.motion_ends) == 1:
            super()._sample_start_frames(env_ids)
            if getattr(self.cfg, "clip_timeout", False):
                frames = self.frame_ids[env_ids]
                self.frame_ids[env_ids] = frames - torch.isin(frames, self.motion_ends - 1).long()
            return
        # Replay each chosen clip from its beginning; the next clip follows a wrap.
        index = torch.bucketize(self.frame_ids[env_ids], self.motion_ends, right=True)
        starts = torch.cat((self.motion_ends.new_zeros(1), self.motion_ends[:-1]))
        if getattr(self.cfg, "clip_timeout", False):
            offset = 1 if self.cfg.targets_next_frame else 2
            at_end = self.frame_ids[env_ids] + offset >= self.motion_ends[index.clamp(max=len(starts) - 1)]
            index = index + at_end.long()
        self.frame_ids[env_ids] = starts[index.remainder(len(starts))]

    def sampling_state(self):
        return dict(
            clips=[c["sha256"] for c in self.motion_contracts],
            ends=self.motion_ends.cpu(),
            scores=self.failure_score_by_bin.cpu(),
            sampler=self._sampler_snapshot(),
        )

    def _sampler_snapshot(self):
        sampler = getattr(self, "sampler", None)
        if sampler is None:
            return None
        frames = torch.arange(self.reference.num_frames, device=self.device)
        return dict(
            bins=sampler.bins_at(frames).cpu(),
            stats={key: value.cpu() for key, value in sampler.statistics().items()},
        )

    def restore_sampling(self, saved):
        """Keep exact bins on resume and map shared clips by local frame on transfer."""
        keys = [c["sha256"] for c in self.motion_contracts]
        sampler = getattr(self, "sampler", None)
        if sampler is not None and saved.get("sampler"):
            snapshot = saved["sampler"]
            statistics = sampler.statistics()
            known = {k: v.to(self.device) for k, v in snapshot["stats"].items() if k in statistics}
            if keys == saved["clips"] and all(statistics[k].shape == v.shape for k, v in known.items()):
                for name, value in known.items():
                    statistics[name].copy_(value)
            else:
                starts = dict(zip(saved["clips"], [0, *saved["ends"].tolist()[:-1]], strict=True))
                centers = sampler.bin_centers()
                frame_bins = snapshot["bins"].to(self.device)
                begin = 0
                for key, end in zip(keys, self.motion_ends.tolist(), strict=True):
                    if key in starts:
                        mask = (centers >= begin) & (centers < end)
                        old_bins = frame_bins[centers[mask] - begin + starts[key]]
                        for name, value in known.items():
                            statistics[name][mask] = value[old_bins]
                    begin = end
        old_scores = saved["scores"].to(self.device)
        if keys == saved["clips"] and len(old_scores) == self.start_bin_count:
            self.failure_score_by_bin.copy_(old_scores)
            return
        previous = {}
        begin = 0
        for key, end in zip(saved["clips"], saved["ends"].tolist(), strict=True):
            previous[key] = (begin, end)
            begin = end
        total = self.reference.num_frames
        centers = (
            (torch.arange(self.start_bin_count, device=self.device) + 0.5) * total / self.start_bin_count
        ).long()
        self.failure_score_by_bin.zero_()
        begin = 0
        for key, end in zip(keys, self.motion_ends.tolist(), strict=True):
            if key in previous:
                old_begin, old_end = previous[key]
                mask = (centers >= begin) & (centers < end)
                frame = (old_begin + centers[mask] - begin).clamp(max=old_end - 1)
                self.failure_score_by_bin[mask] = old_scores[
                    frame * len(old_scores) // int(saved["ends"][-1])
                ]
            begin = end

    def _advance_reference(self):
        clip = torch.bucketize(self.frame_ids, self.motion_ends, right=True)
        if getattr(self.cfg, "clip_timeout", False):
            self.frame_ids += (self.reset_step != self._env.common_step_counter).long()
        else:
            self.frame_ids += 1
        return torch.where(self.frame_ids >= self.motion_ends[clip])[0]

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reset_step = torch.full_like(self.frame_ids, -1)
        self.motion_contract = self.motion_contracts[0]
        root_velocity = self.root_velocity
        self.resample_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        ref = self.reference
        root = self.robot.body_names.index("pelvis")
        fk = UrdfKinematics(URDF, self.robot.joint_names, self.robot.body_names, self.device)
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
        if getattr(self.cfg, "clip_timeout", False):
            self.reset_step[env_ids] = self._env.common_step_counter
            ends = self.motion_ends[torch.bucketize(self.frame_ids[env_ids], self.motion_ends, right=True)]
            self.frame_ids[env_ids] = self.frame_ids[env_ids].minimum(ends - 2)
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_vel = torch.cat([self.body_lin_vel_w[env_ids, 0], self.body_ang_vel_w[env_ids, 0]], -1)
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        ranges = torch.tensor(
            [self.cfg.reset_pose_noise.get(k, (0.0, 0.0)) for k in keys],
            device=self.device,
        )
        noise = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += noise[:, :3]
        root_ori = quat_mul(quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]), root_ori)
        ranges = torch.tensor(
            [self.cfg.reset_velocity_noise.get(k, (0.0, 0.0)) for k in keys],
            device=self.device,
        )
        root_vel += sample_uniform(ranges[:, 0], ranges[:, 1], root_vel.shape, device=self.device)
        joint_pos = self.joint_pos[env_ids].clone()
        joint_pos += sample_uniform(*self.cfg.reset_joint_position_noise, joint_pos.shape, joint_pos.device)
        use_hard_limits = getattr(self.cfg, "clip_timeout", False) and not self.cfg.targets_next_frame
        limits = (
            self.robot.data.default_joint_pos_limits if use_hard_limits else self.robot.data.soft_joint_pos_limits
        ).torch[env_ids]
        joint_pos = torch.clamp(joint_pos, limits[:, :, 0], limits[:, :, 1])
        self.robot.write_joint_state_to_sim_index(
            position=joint_pos, velocity=self.joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat([root_pos, root_ori], -1), env_ids=env_ids
        )
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        if getattr(self.cfg, "targets_next_frame", False):
            self.frame_ids[env_ids] += 1
