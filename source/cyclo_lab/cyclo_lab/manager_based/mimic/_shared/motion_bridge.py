# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Kiwoong Park

"""Read live Isaac articulation state; adapt only the upstream learner contract."""

import math
import torch
from cyclo_lab.utils.k1_kinematics import UrdfKinematics
from cyclo_lab.utils.k1_motion_validation import validate_motion
from cyclo_lab.utils.k1_motion_physics import URDF, validate_motion_physics

BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
)


def wxyz(q):
    """Convert Isaac Lab xyzw quaternions to the motion bridge’s wxyz order."""
    return q[..., [3, 0, 1, 2]]


class IsaacMotion:
    def __init__(self, env, fk):
        # Existing Cyclo loader validates/reorders named arrays into live Isaac order.
        command = env.command_manager.get_term("reference_trajectory")
        reference = command.reference
        if not math.isclose(
            float(torch.as_tensor(reference.fps).flatten()[0]) * env.step_dt,
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Motion FPS must match the Isaac policy rate; resample before training")
        self.fk = fk
        self.joint_pos = reference.joint_position
        self.joint_vel = reference.joint_velocity
        root = env.scene["robot"].body_names.index("pelvis")
        self.root_pos_w = reference._body_position_w[:, root]
        self.root_quat_w = wxyz(reference._body_orientation_w[:, root])
        if hasattr(command, "motion_contracts"):
            self.contract = command.motion_contracts[0]
            self.root_lin_vel_w = command.root_velocity
            self.motion_ends = command.motion_ends
        else:
            self.contract, root_velocity = validate_motion_physics(
                env.cfg.commands.reference_trajectory.trajectory_file, env.cfg.scene.robot.spawn.asset_path
            )
            self.root_lin_vel_w = root_velocity.to(env.device)
            self.motion_ends = torch.tensor([reference.num_frames], device=env.device)
        self.root_ang_vel_w = reference._body_angular_velocity_w[:, root]
        self.time_step_total = reference.num_frames

    def body_state(self, times):
        return self.fk(
            self.root_pos_w[times],
            self.root_quat_w[times],
            self.joint_pos[times],
            self.root_lin_vel_w[times],
            self.root_ang_vel_w[times],
            self.joint_vel[times],
        )


def generalized_state(env):
    d = env.scene["robot"].data
    # Link origin velocities, not COM velocities, match URDF FK and root pose.
    return torch.cat(
        [
            d.root_link_pos_w.torch - env.scene.env_origins,
            wxyz(d.root_link_quat_w.torch),
            d.joint_pos.torch,
            d.root_link_lin_vel_w.torch,
            d.root_link_ang_vel_w.torch,
            d.joint_vel.torch,
        ],
        -1,
    ).clone()


class IsaacMotionBridge:
    def __init__(self, env):
        self.env = env.unwrapped
        self.device = self.env.device
        self.robot = self.env.scene["robot"]
        self.num_actions = self.robot.num_joints
        if self.num_actions != 23:
            raise ValueError("K1 motion tracking expects exactly 23 articulated joints")
        self.fk = UrdfKinematics(self.env.cfg.scene.robot.spawn.asset_path, self.robot.joint_names, BODIES, self.device)
        self.motion_contract = validate_motion(
            self.env.cfg.commands.reference_trajectory.trajectory_file,
            self.robot.joint_names,
            self.robot.body_names,
            fps=50,
        )
        self.motion = IsaacMotion(self.env, self.fk)
        self.scene = self.env.scene
        self.step_dt = self.env.step_dt
        self.anchor_index = 0
        self.anchor_pos_threshold = self.env.cfg.terminations.anchor_height.params["threshold"]
        self.anchor_ori_threshold = self.env.cfg.terminations.anchor_ori.params["threshold"]

    def generalized_state(self):
        return generalized_state(self.env)

    def frame_ids(self):
        return self.env.command_manager.get_term("reference_trajectory").frame_ids.clone()

    def reset_counts(self):
        return self.env.command_manager.get_term("reference_trajectory").resample_counts.clone()

    def action_transform(self):
        term = self.env.action_manager.get_term("joint_pos")
        scale, offset = term._scale, getattr(self.env, "nominal_action_offset", term._offset)

        def tensor(v):
            v = torch.as_tensor(v, device=self.device, dtype=torch.float32)
            return v.expand(self.env.num_envs, self.num_actions)[0].clone()

        return tensor(scale), tensor(offset)
