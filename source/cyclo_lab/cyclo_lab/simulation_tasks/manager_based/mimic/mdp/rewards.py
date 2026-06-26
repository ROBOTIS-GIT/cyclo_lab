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
# Author: Woojae Lee
#
# Additional notices:
# This module is adapted from HybridRobotics/whole_body_tracking, licensed under the MIT License.
# See THIRD_PARTY_LICENSES.md for details.

"""Reward terms for Mimic tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude

from .commands import ReferenceTrajectoryCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _tracked_body_ids(command: ReferenceTrajectoryCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _safe_clamp(x: torch.Tensor, clip_max: float) -> torch.Tensor:
    return torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=clip_max, neginf=0.0), max=clip_max)


def reference_anchor_position_tracking(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: ReferenceTrajectoryCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def reference_anchor_orientation_tracking(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: ReferenceTrajectoryCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def reference_body_position_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: ReferenceTrajectoryCommand = env.command_manager.get_term(command_name)
    tracked_body_ids = _tracked_body_ids(command, body_names)
    error = torch.sum(
        torch.square(
            command.aligned_body_position_w[:, tracked_body_ids] - command.robot_body_pos_w[:, tracked_body_ids]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def reference_body_orientation_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: ReferenceTrajectoryCommand = env.command_manager.get_term(command_name)
    tracked_body_ids = _tracked_body_ids(command, body_names)
    error = (
        quat_error_magnitude(
            command.aligned_body_orientation_w[:, tracked_body_ids],
            command.robot_body_quat_w[:, tracked_body_ids],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def reference_body_linear_velocity_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: ReferenceTrajectoryCommand = env.command_manager.get_term(command_name)
    tracked_body_ids = _tracked_body_ids(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, tracked_body_ids] - command.robot_body_lin_vel_w[:, tracked_body_ids]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def reference_body_angular_velocity_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: ReferenceTrajectoryCommand = env.command_manager.get_term(command_name)
    tracked_body_ids = _tracked_body_ids(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, tracked_body_ids] - command.robot_body_ang_vel_w[:, tracked_body_ids]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def bounded_action_delta_l2(env: ManagerBasedRLEnv, clip_max: float = 100.0) -> torch.Tensor:
    return _safe_clamp(torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1), clip_max)


def ankle_coupling_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_pitch_name: str = "left_ankle_pitch_joint",
    left_roll_name: str = "left_ankle_roll_joint",
    right_pitch_name: str = "right_ankle_pitch_joint",
    right_roll_name: str = "right_ankle_roll_joint",
    p_neg: float = 0.720,
    p_pos: float = 0.652,
    roll_limit: float = 0.392,
) -> torch.Tensor:
    """Penalize ankle pitch-roll combinations outside the configured linkage envelope."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    if left_pitch_name not in joint_names:
        return torch.zeros(env.num_envs, device=env.device)

    lp_idx = joint_names.index(left_pitch_name)
    lr_idx = joint_names.index(left_roll_name)
    rp_idx = joint_names.index(right_pitch_name)
    rr_idx = joint_names.index(right_roll_name)
    joint_pos = asset.data.joint_pos

    def ellipse_violation(pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
        pitch_norm = torch.where(pitch < 0, pitch / p_neg, pitch / p_pos)
        roll_norm = roll / roll_limit
        return torch.clamp(pitch_norm**2 + roll_norm**2 - 1.0, min=0.0)

    return ellipse_violation(joint_pos[:, lp_idx], joint_pos[:, lr_idx]) + ellipse_violation(
        joint_pos[:, rp_idx], joint_pos[:, rr_idx]
    )
