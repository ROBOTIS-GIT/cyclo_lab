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
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def eef_pose(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"), robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Return the state of the end effector frame in the robot coordinate system.
    """
    robot = env.scene[robot_cfg.name]
    robot_root_pos, robot_root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos, ee_frame_quat = ee_frame.data.target_pos_w[:, 0, :], ee_frame.data.target_quat_w[:, 0, :]
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    eef_pose = torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)

    return eef_pose


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


def joint_pos_name(env: ManagerBasedEnv, joint_names: list[str], asset_name: str = "robot") -> torch.Tensor:
    """
    Returns the relative joint positions for the specified joint names.

    Args:
        env: ManagerBasedEnv instance.
        joint_names: List of joint names to extract.
        asset_name: Name of the asset (default: "robot").

    Returns:
        torch.Tensor of shape [1, len(joint_names)]
    """
    asset: Articulation = env.scene[asset_name]

    joint_ids = [asset.joint_names.index(name) for name in joint_names]

    joint_pos = asset.data.joint_pos[:, joint_ids]

    return joint_pos


def joint_vel_name(env: ManagerBasedEnv, joint_names: list[str], asset_name: str = "robot") -> torch.Tensor:
    """
    Returns the relative joint velocities for the specified joint names.

    Args:
        env: ManagerBasedEnv instance.
        joint_names: List of joint names to extract.
        asset_name: Name of the asset (default: "robot").

    Returns:
        torch.Tensor of shape [1, len(joint_names)]
    """
    asset: Articulation = env.scene[asset_name]

    joint_ids = [asset.joint_names.index(name) for name in joint_names]

    return asset.data.joint_vel[:, joint_ids]


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.1,
    gripper_close_threshold: torch.tensor = torch.tensor([0.3]),
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        robot.data.joint_pos[:, -1] >= gripper_close_threshold.to(env.device),
    )
    grasped = torch.logical_and(
        grasped, robot.data.joint_pos[:, -2] >= gripper_close_threshold.to(env.device)
    )
    return grasped


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def joint_pos_target_name(env: ManagerBasedEnv, joint_names: list[str], asset_name: str = "robot") -> torch.Tensor:
    """The joint positions target of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_name]

    joint_ids = [asset.joint_names.index(name) for name in joint_names]

    joint_pos_target = asset.data.joint_pos_target[:, joint_ids]

    return joint_pos_target


def bottle_in_basket(
    env: ManagerBasedRLEnv,
    bottle_cfg: SceneEntityCfg,
    basket_cfg: SceneEntityCfg,
    distance_threshold: float = 0.1,
) -> torch.Tensor:
    """Check if the bottle is placed inside the basket."""
    bottle: RigidObject = env.scene[bottle_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]

    bottle_pos = bottle.data.root_pos_w
    basket_pos = basket.data.root_pos_w

    # Check if bottle is close to basket horizontally (x, y)
    horizontal_distance = torch.linalg.vector_norm(bottle_pos[:, :2] - basket_pos[:, :2], dim=1)
    horizontal_ok = horizontal_distance < distance_threshold

    # Check if bottle is above the basket bottom but below basket rim
    # Assuming basket bottom is at basket_pos z and rim is ~0.1m higher
    basket_bottom = basket_pos[:, 2]
    basket_rim = basket_pos[:, 2] + 0.1  # Adjust based on actual basket height

    vertical_ok = (bottle_pos[:, 2] > basket_bottom) & (bottle_pos[:, 2] < basket_rim)

    in_basket = horizontal_ok & vertical_ok

    return in_basket
