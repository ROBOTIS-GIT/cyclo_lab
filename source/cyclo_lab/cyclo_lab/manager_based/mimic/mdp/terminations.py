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

"""Termination terms for Mimic tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from .commands import ReferenceTrajectoryCommand
from .rewards import _tracked_body_ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reference_anchor_height_deviation(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: ReferenceTrajectoryCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def reference_anchor_gravity_deviation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: ReferenceTrajectoryCommand = env.command_manager.get_term(command_name)
    reference_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (reference_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def reference_body_height_deviation(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: ReferenceTrajectoryCommand = env.command_manager.get_term(command_name)
    tracked_body_ids = _tracked_body_ids(command, body_names)
    error = torch.abs(
        command.aligned_body_position_w[:, tracked_body_ids, -1] - command.robot_body_pos_w[:, tracked_body_ids, -1]
    )
    return torch.any(error > threshold, dim=-1)


def physx_nan_detected(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    return (
        torch.any(torch.isnan(data.joint_pos[:, asset_cfg.joint_ids]), dim=1)
        | torch.any(torch.isnan(data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
        | torch.any(torch.isnan(data.root_pos_w), dim=1)
        | torch.any(torch.isnan(data.root_quat_w), dim=1)
        | torch.any(torch.isnan(data.root_lin_vel_w), dim=1)
        | torch.any(torch.isnan(data.root_ang_vel_w), dim=1)
    )
