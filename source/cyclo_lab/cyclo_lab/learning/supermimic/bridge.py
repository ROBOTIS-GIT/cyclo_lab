# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Read live Isaac articulation state; adapt only the upstream learner contract."""

import torch
from . import ROOT
from .fk import UrdfKinematics
from .motion import validate_motion

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
URDF = ROOT / "third_party/ai_sapiens/ai_sapiens_description/urdf/k1_rev1/k1.urdf"


def wxyz(q):
    """Isaac Lab 3.0 xyzw -> upstream SuperMimic wxyz, exactly once."""
    return q[..., [3, 0, 1, 2]]


class IsaacMotion:
    def __init__(self, env, fk):
        # Existing Cyclo loader validates/reorders named arrays into live Isaac order.
        command = env.command_manager.get_term("reference_trajectory")
        reference = command.reference
        if float(torch.as_tensor(reference.fps).flatten()[0]) != round(1 / env.step_dt):
            raise ValueError(
                "Motion FPS must match the Isaac policy rate; resample before training"
            )
        self.fk = fk
        self.joint_pos = reference.joint_position
        self.joint_vel = reference.joint_velocity
        root = env.scene["robot"].body_names.index("pelvis")
        self.root_pos_w = reference._body_position_w[:, root]
        self.root_quat_w = wxyz(reference._body_orientation_w[:, root])
        self.root_lin_vel_w = reference._body_linear_velocity_w[:, root]
        self.root_ang_vel_w = reference._body_angular_velocity_w[:, root]
        self.motion_ends = torch.tensor(
            [reference.num_frames], device=env.device, dtype=torch.long
        )
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


class IsaacBridge:
    """View of the real Isaac env plus small metadata needed by original WM env.

    Simulator state always comes from Isaac tensors. The constant context is a
    documented no-DR configuration; no fake MuJoCo model or environment exists.
    """

    def __init__(self, env):
        self.env = env.unwrapped
        self.device = self.env.device
        self.robot = self.env.scene["robot"]
        self.num_actions = self.robot.num_joints
        if self.num_actions != 23:
            raise ValueError("K1 SuperMimic expects exactly 23 articulated joints")
        self.fk = UrdfKinematics(URDF, self.robot.joint_names, BODIES, self.device)
        self.motion_contract = validate_motion(
            self.env.cfg.commands.reference_trajectory.trajectory_file,
            self.robot.joint_names,
            self.robot.body_names,
            fps=round(1 / self.env.step_dt),
        )
        self.motion = IsaacMotion(self.env, self.fk)
        self.scene = self.env.scene
        self.step_dt = self.env.step_dt
        self.anchor_index = 0
        self.anchor_pos_threshold = self.env.cfg.terminations.anchor_height.params[
            "threshold"
        ]
        self.anchor_ori_threshold = self.env.cfg.terminations.anchor_ori.params[
            "threshold"
        ]
        self.max_step_reward = self.step_dt * sum(
            getattr(self.env.cfg.rewards, name).weight
            for name in (
                "reference_anchor_position",
                "reference_anchor_orientation",
                "reference_body_position",
                "reference_body_orientation",
                "reference_body_linear_velocity",
                "reference_body_angular_velocity",
            )
        )

    def state(self):
        d = self.robot.data
        # Link origin velocities, not COM velocities, match URDF FK and root pose.
        return torch.cat(
            [
                d.root_link_pos_w.torch - self.env.scene.env_origins,
                wxyz(d.root_link_quat_w.torch),
                d.joint_pos.torch,
                d.root_link_lin_vel_w.torch,
                d.root_link_ang_vel_w.torch,
                d.joint_vel.torch,
            ],
            -1,
        ).clone()

    def frame_ids(self):
        return self.env.command_manager.get_term(
            "reference_trajectory"
        ).frame_ids.clone()

    def context(self):
        return torch.zeros((self.env.num_envs, 1), device=self.device)

    def action_transform(self):
        term = self.env.action_manager.get_term("joint_pos")
        scale, offset = term._scale, term._offset

        def tensor(v):
            v = torch.as_tensor(v, device=self.device, dtype=torch.float32)
            return v.expand(self.env.num_envs, self.num_actions)[0].clone()

        return tensor(scale), tensor(offset)

    def check_fk(self, atol=2e-3):
        state = self.state()
        pos, quat, vel, ang = self.fk(
            state[:, :3],
            state[:, 3:7],
            state[:, 7:30],
            state[:, 30:33],
            state[:, 33:36],
            state[:, 36:],
        )
        ids = [self.robot.body_names.index(n) for n in BODIES]
        d = self.robot.data
        actual_pos = (
            d.body_link_pos_w.torch[:, ids] - self.env.scene.env_origins[:, None]
        )
        actual_q = wxyz(d.body_link_quat_w.torch[:, ids])
        errors = {
            "position": (pos - actual_pos).abs().max(),
            "orientation": (1 - (quat * actual_q).sum(-1).abs()).abs().max(),
            "linear_velocity": (vel - d.body_link_lin_vel_w.torch[:, ids]).abs().max(),
            "angular_velocity": (ang - d.body_link_ang_vel_w.torch[:, ids]).abs().max(),
        }
        if any(float(v) > atol for v in errors.values()):
            raise RuntimeError(f"Isaac/URDF FK mismatch: {errors}")
        return {k: float(v) for k, v in errors.items()}
