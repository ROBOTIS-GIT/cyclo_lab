# Copyright 2025 ROBOTIS CO., LTD.
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
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    distance_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward the agent for lifting the object above a minimal height.

    *Only if* it is within a certain distance from the end-effector.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Get object height (z position in world frame)
    obj_height = object.data.root_pos_w[:, 2]  # (num_envs,)

    # Get positions
    obj_pos = object.data.root_pos_w  # (num_envs, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    # Compute Euclidean distance between object and end-effector
    dist = torch.norm(obj_pos - ee_pos, dim=1)  # (num_envs,)

    # Reward is 1.0 if object is above minimal height AND within_reach to EE
    lifted = obj_height > minimal_height
    within_reach = dist < distance_threshold

    reward = torch.where(lifted & within_reach, 1.0, 0.0)

    # print(f"lifted: {lifted}")
    # print(f"within_reach_to_ee: {within_reach}")
    # print(f"lift reward: {reward}")

    return reward


def object_grasp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.03,
    gripper_close_threshold: float = 0.6,
) -> torch.Tensor:
    """
    Reward function for detecting if the object is being grasped.

    Combines end-effector proximity and gripper closure conditions.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Compute the distance between end-effector and object
    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    # Check if gripper joints are closed beyond threshold
    gripper_closed = torch.logical_and(
        robot.data.joint_pos[:, -1] >= gripper_close_threshold,
        robot.data.joint_pos[:, -2] >= gripper_close_threshold,
    )

    # Combine both conditions
    is_grasped = torch.logical_and(pose_diff < diff_threshold, gripper_closed)

    # print(f"object grasp reward: {is_grasped.float()}")

    return is_grasped.float()


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.3,  # standard deviation
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)

    reward = torch.exp(-0.5 * (distance / std) ** 2)

    return reward


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
