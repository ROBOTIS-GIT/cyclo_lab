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


def eef_pose(env: ManagerBasedRLEnv, eef_cfg: SceneEntityCfg = SceneEntityCfg("eef"), robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Return the state of the end effector frame in the robot coordinate system.
    """
    robot = env.scene[robot_cfg.name]
    robot_root_pos, robot_root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    eef: FrameTransformer = env.scene[eef_cfg.name]
    eef_pos, eef_quat = eef.data.target_pos_w[:, 0, :], eef.data.target_quat_w[:, 0, :]
    eef_pos_robot, eef_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, eef_pos, eef_quat
    )
    eef_pose = torch.cat([eef_pos_robot, eef_quat_robot], dim=1)

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
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.1,
    gripper_joint_name: str = "gripper_r_joint1",
    gripper_close_threshold: torch.tensor = torch.tensor([0.2]),
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    eef: FrameTransformer = env.scene[eef_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = eef.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    gripper_joint_pose = robot.data.joint_pos[:, robot.joint_names.index(gripper_joint_name)]
    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        gripper_joint_pose >= gripper_close_threshold.to(env.device),
    )
    return grasped


def eef_pos(env: ManagerBasedRLEnv, eef_cfg: SceneEntityCfg = SceneEntityCfg("eef")) -> torch.Tensor:
    eef: FrameTransformer = env.scene[eef_cfg.name]
    eef_pos = eef.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return eef_pos


def eef_quat(env: ManagerBasedRLEnv, eef_cfg: SceneEntityCfg = SceneEntityCfg("eef")) -> torch.Tensor:
    eef: FrameTransformer = env.scene[eef_cfg.name]
    eef_quat = eef.data.target_quat_w[:, 0, :]

    return eef_quat


def joint_pos_target_name(env: ManagerBasedEnv, joint_names: list[str], asset_name: str = "robot") -> torch.Tensor:
    """The joint positions target of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_name]

    joint_ids = [asset.joint_names.index(name) for name in joint_names]

    joint_pos_target = asset.data.joint_pos_target[:, joint_ids]

    return joint_pos_target


def object_in_basket(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    basket_cfg: SceneEntityCfg,
    distance_threshold: float = 0.1,
) -> torch.Tensor:
    """Check if the object is placed inside the basket."""
    object: RigidObject = env.scene[object_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]

    object_pos = object.data.root_pos_w
    basket_pos = basket.data.root_pos_w

    # Check 3D distance between object and basket
    distance_3d = torch.linalg.vector_norm(object_pos - basket_pos, dim=1)
    done = distance_3d < distance_threshold

    return done
