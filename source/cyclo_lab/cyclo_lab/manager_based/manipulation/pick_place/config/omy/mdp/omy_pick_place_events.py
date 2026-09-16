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

import math
import random
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.sensors.camera import Camera
from isaaclab.managers import SceneEntityCfg
from typing import Literal

from pxr import Gf

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1000.0, 3000.0),
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((0.5, 1.0), (0.5, 1.0), (0.5, 1.0)),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    # Random intensity
    new_intensity = random.uniform(intensity_range[0], intensity_range[1])
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(new_intensity)

    # Random color
    new_color = Gf.Vec3f(
        random.uniform(color_range[0][0], color_range[0][1]),
        random.uniform(color_range[1][0], color_range[1][1]),
        random.uniform(color_range[2][0], color_range[2][1]),
    )
    color_attr = light_prim.GetAttribute("inputs:color")
    color_attr.Set(new_color)


def randomize_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]] = None,
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation within the given ranges."""
    if pose_range is None:
        pose_range = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }

    asset: Camera = env.scene[asset_cfg.name]

    # Store initial positions and quaternions once
    if not hasattr(asset, "_initial_pos_w"):
        asset._initial_pos_w = asset.data.pos_w.clone()
        asset._initial_quat_w_ros = asset.data.quat_w_ros.clone()
        asset._initial_quat_w_opengl = asset.data.quat_w_opengl.clone()
        asset._initial_quat_w_world = asset.data.quat_w_world.clone()

    ori_pos_w = asset._initial_pos_w
    if convention == "ros":
        ori_quat_w = asset._initial_quat_w_ros
    elif convention == "opengl":
        ori_quat_w = asset._initial_quat_w_opengl
    elif convention == "world":
        ori_quat_w = asset._initial_quat_w_world

    # Get pose ranges
    range_list = [pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)

    # Sample random offsets for each environment independently
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    # Apply per-env randomization
    for i, env_id in enumerate(env_ids.tolist()):
        pos = ori_pos_w[env_id, 0:3] + rand_samples[i, 0:3]
        ori_delta = math_utils.quat_from_euler_xyz(
            rand_samples[i, 3], rand_samples[i, 4], rand_samples[i, 5]
        )
        ori = math_utils.quat_mul(ori_quat_w[env_id], ori_delta)
        asset.set_world_poses(
            pos.unsqueeze(0), ori.unsqueeze(0), env_ids=torch.tensor([env_id], device=asset.device), convention=convention
        )
