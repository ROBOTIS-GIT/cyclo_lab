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

from pxr import Gf, UsdShade

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def create_joint_position_mapping(joint_names: list[str], desired_values: dict[str, float]) -> torch.Tensor:
    """Create a tensor with joint positions in the correct order based on joint names."""
    joint_positions = []

    for joint_name in joint_names:
        if joint_name in desired_values:
            joint_positions.append(desired_values[joint_name])
        else:
            joint_positions.append(0.0)  # Default value

    return torch.tensor(joint_positions, dtype=torch.float32)


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_positions: dict[str, float],  # Change to dict instead of list
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the default joint positions for the robot using joint names."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Create properly ordered tensor from joint name mapping
    default_pose_tensor = create_joint_position_mapping(asset.joint_names, joint_positions)
    default_pose_tensor = default_pose_tensor.to(device=env.device)

    # Ensure correct shape for multiple environments
    if default_pose_tensor.dim() == 1:
        default_pose_tensor = default_pose_tensor.unsqueeze(0).repeat(len(env_ids), 1)

    # Set joint positions
    asset.set_joint_position_target(default_pose_tensor, env_ids=env_ids)
    asset.write_joint_state_to_sim(default_pose_tensor, torch.zeros_like(default_pose_tensor), env_ids=env_ids)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] = None,  # Add parameter to specify which joints to randomize
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Use current joint positions instead of default positions
    joint_pos = asset.data.joint_pos[env_ids].clone()
    joint_vel = asset.data.joint_vel[env_ids].clone()

    # If joint_names is specified, only randomize those joints
    if joint_names is not None:
        for joint_name in joint_names:
            if joint_name in asset.joint_names:
                joint_idx = asset.joint_names.index(joint_name)
                noise = math_utils.sample_gaussian(mean, std, (len(env_ids), 1), joint_pos.device)
                joint_pos[:, joint_idx:joint_idx + 1] += noise
    else:
        # Original behavior - randomize all joints
        joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

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


def randomize_robot_base_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] = None,
):
    """Randomize the robot's base position and orientation."""
    if pose_range is None:
        return

    if env_ids is None:
        return

    asset: Articulation = env.scene[asset_cfg.name]

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        # Sample random pose
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        sample = [random.uniform(range[0], range[1]) for range in range_list]

        # Convert to tensor and add environment origin
        pose_tensor = torch.tensor([sample], device=env.device)
        position = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
        orientation = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])

        # Write pose to simulation
        asset.write_root_pose_to_sim(
            torch.cat([position, orientation], dim=-1), 
            env_ids=torch.tensor([cur_env], device=env.device)
        )
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), 
            env_ids=torch.tensor([cur_env], device=env.device)
        )


def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: dict[str, float],
):
    """Set a fixed pose for an object."""
    if env_ids is None:
        return

    asset = env.scene[asset_cfg.name]

    # Extract pose values with defaults
    x = pose.get("x", 0.0)
    y = pose.get("y", 0.0)
    z = pose.get("z", 0.0)
    roll = pose.get("roll", 0.0)
    pitch = pose.get("pitch", 0.0)
    yaw = pose.get("yaw", 0.0)

    # Set poses for each environment
    for cur_env in env_ids.tolist():
        # Convert to tensor and add environment origin
        position = torch.tensor([[x, y, z]], device=env.device) + env.scene.env_origins[cur_env, 0:3]
        orientation = math_utils.quat_from_euler_xyz(
            torch.tensor([roll], device=env.device),
            torch.tensor([pitch], device=env.device),
            torch.tensor([yaw], device=env.device)
        )

        # Write pose to simulation
        asset.write_root_pose_to_sim(
            torch.cat([position, orientation], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device)
        )
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device)
        )


# Pre-defined slot positions for object placement
# Left side slots (y > 0)
LEFT_SLOTS = [
    (0.555, 0.26, 0.95),
    (0.555, 0.087, 0.95),
    (0.555, 0.26, 1.235),
    (0.555, 0.087, 1.235),
]

# Right side slots (y < 0)
RIGHT_SLOTS = [
    (0.555, -0.087, 0.95),
    (0.555, -0.26, 0.95),
    (0.555, -0.087, 1.235),
    (0.555, -0.26, 1.235),
]

ALL_SLOTS = LEFT_SLOTS + RIGHT_SLOTS


def randomize_table_with_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_cfg: SceneEntityCfg,
    object_cfgs: list[SceneEntityCfg],
    object_relative_poses: list[dict[str, float]],
    table_pose_range: dict[str, tuple[float, float]],
):
    """Randomize table pose and move all objects together maintaining relative positions.

    Essential for trajectory augmentation - EE-Object relative trajectory remains valid
    when objects move together with the table.
    """
    if env_ids is None:
        return

    table_asset = env.scene[table_cfg.name]

    for cur_env in env_ids.tolist():
        # 1. Sample random table pose
        range_list = [table_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        table_sample = [random.uniform(r[0], r[1]) for r in range_list]

        # Table position and orientation in world frame
        table_pos = torch.tensor([table_sample[0:3]], device=env.device) + env.scene.env_origins[cur_env, 0:3]
        table_quat = math_utils.quat_from_euler_xyz(
            torch.tensor([table_sample[3]], device=env.device),
            torch.tensor([table_sample[4]], device=env.device),
            torch.tensor([table_sample[5]], device=env.device)
        )

        # 2. Write table pose
        table_asset.write_root_pose_to_sim(
            torch.cat([table_pos, table_quat], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device)
        )
        table_asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device)
        )

        # 3. Move all objects relative to table
        for obj_cfg, rel_pose in zip(object_cfgs, object_relative_poses):
            obj_asset = env.scene[obj_cfg.name]

            # Get relative position and orientation
            rel_pos = torch.tensor([[
                rel_pose.get("x", 0.0),
                rel_pose.get("y", 0.0),
                rel_pose.get("z", 0.0)
            ]], device=env.device)
            rel_quat = math_utils.quat_from_euler_xyz(
                torch.tensor([rel_pose.get("roll", 0.0)], device=env.device),
                torch.tensor([rel_pose.get("pitch", 0.0)], device=env.device),
                torch.tensor([rel_pose.get("yaw", 0.0)], device=env.device)
            )

            # Transform relative pose to world frame using table pose
            # obj_pos_world = table_pos + rotate(rel_pos, table_quat)
            rotated_rel_pos = math_utils.quat_apply(table_quat, rel_pos)
            obj_pos_world = table_pos + rotated_rel_pos

            # obj_quat_world = table_quat * rel_quat
            obj_quat_world = math_utils.quat_mul(table_quat, rel_quat)

            # Write object pose
            obj_asset.write_root_pose_to_sim(
                torch.cat([obj_pos_world, obj_quat_world], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device)
            )
            obj_asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device)
            )


def randomize_table_with_objects_on_slots(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_cfg: SceneEntityCfg,
    basket_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    other_asset_cfgs: list[SceneEntityCfg],
    basket_relative_pose: dict[str, float],
    target_side: Literal["left", "right"] = "left",
    table_pose_range: dict[str, tuple[float, float]] = None,
):
    """Randomize table pose and place objects on slots, all moving together.

    This maintains relative positions between objects for proper trajectory augmentation.
    The target object is placed on a random slot on target_side, other objects on remaining slots.
    """
    if env_ids is None:
        return

    if table_pose_range is None:
        table_pose_range = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
        }

    table_asset = env.scene[table_cfg.name]
    basket_asset = env.scene[basket_cfg.name]

    # Determine which slots to use for target
    if target_side == "left":
        target_slots = LEFT_SLOTS.copy()
    else:
        target_slots = RIGHT_SLOTS.copy()

    for cur_env in env_ids.tolist():
        # 1. Sample random table pose
        range_list = [table_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        table_sample = [random.uniform(r[0], r[1]) for r in range_list]

        # Table position and orientation
        table_pos = torch.tensor([table_sample[0:3]], device=env.device) + env.scene.env_origins[cur_env, 0:3]
        table_quat = math_utils.quat_from_euler_xyz(
            torch.tensor([table_sample[3]], device=env.device),
            torch.tensor([table_sample[4]], device=env.device),
            torch.tensor([table_sample[5]], device=env.device)
        )

        # 2. Write table pose
        table_asset.write_root_pose_to_sim(
            torch.cat([table_pos, table_quat], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device)
        )
        table_asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device)
        )

        # 3. Place basket relative to table
        basket_rel_pos = torch.tensor([[
            basket_relative_pose.get("x", 0.41),
            basket_relative_pose.get("y", 0.0),
            basket_relative_pose.get("z", 0.72)
        ]], device=env.device)
        basket_rel_quat = math_utils.quat_from_euler_xyz(
            torch.tensor([basket_relative_pose.get("roll", 0.0)], device=env.device),
            torch.tensor([basket_relative_pose.get("pitch", 0.0)], device=env.device),
            torch.tensor([basket_relative_pose.get("yaw", 0.0)], device=env.device)
        )

        rotated_basket_pos = math_utils.quat_apply(table_quat, basket_rel_pos)
        basket_pos_world = table_pos + rotated_basket_pos
        basket_quat_world = math_utils.quat_mul(table_quat, basket_rel_quat)

        basket_asset.write_root_pose_to_sim(
            torch.cat([basket_pos_world, basket_quat_world], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device)
        )
        basket_asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device)
        )

        # 4. Place objects on slots (relative to table)
        available_target_slots = target_slots.copy()
        available_all_slots = ALL_SLOTS.copy()

        # Place target object
        target_asset = env.scene[target_asset_cfg.name]
        target_slot = random.choice(available_target_slots)
        available_all_slots.remove(target_slot)

        # Transform slot position by table pose
        slot_rel_pos = torch.tensor([[target_slot[0], target_slot[1], target_slot[2]]], device=env.device)
        rotated_slot_pos = math_utils.quat_apply(table_quat, slot_rel_pos)
        target_pos_world = table_pos + rotated_slot_pos
        target_quat_world = table_quat.clone()  # Same orientation as table

        target_asset.write_root_pose_to_sim(
            torch.cat([target_pos_world, target_quat_world], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device)
        )
        target_asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device)
        )

        # Place other objects
        random.shuffle(available_all_slots)
        for i, asset_cfg in enumerate(other_asset_cfgs):
            if i >= len(available_all_slots):
                break

            asset = env.scene[asset_cfg.name]
            slot = available_all_slots[i]

            # Transform slot position by table pose
            slot_rel_pos = torch.tensor([[slot[0], slot[1], slot[2]]], device=env.device)
            rotated_slot_pos = math_utils.quat_apply(table_quat, slot_rel_pos)
            obj_pos_world = table_pos + rotated_slot_pos
            obj_quat_world = table_quat.clone()

            asset.write_root_pose_to_sim(
                torch.cat([obj_pos_world, obj_quat_world], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device)
            )


def randomize_background_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
):
    """Randomize the color of a background cube for visual domain randomization."""
    if env_ids is None:
        return

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    for cur_env in env_ids.tolist():
        # Construct prim path for this environment
        prim_path = f"/World/envs/env_{cur_env}/BackgroundCube"
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            continue

        # Generate random color
        new_color = Gf.Vec3f(
            random.uniform(color_range[0][0], color_range[0][1]),
            random.uniform(color_range[1][0], color_range[1][1]),
            random.uniform(color_range[2][0], color_range[2][1]),
        )

        # Find and update material color
        _update_prim_material_color(stage, prim, new_color)


def _update_prim_material_color(stage, prim, color: Gf.Vec3f) -> bool:
    """Helper to find and update material color on a prim."""
    prim_path = prim.GetPath().pathString

    # Try common material paths
    for mat_suffix in ["/Looks/PreviewSurface", "/Looks/material_0", "/Material", "/Looks"]:
        mat_prim = stage.GetPrimAtPath(f"{prim_path}{mat_suffix}")
        if mat_prim.IsValid():
            # Try to get shader (might be at /Shader or directly on material)
            shader_prim = stage.GetPrimAtPath(f"{mat_prim.GetPath().pathString}/Shader")
            if not shader_prim.IsValid():
                shader_prim = mat_prim

            # Try different color attribute names
            for attr_name in ["inputs:diffuseColor", "inputs:diffuse_color_constant", "inputs:baseColor"]:
                color_attr = shader_prim.GetAttribute(attr_name)
                if color_attr.IsValid():
                    color_attr.Set(color)
                    return True

    # Fallback: search geometry children for material bindings
    return _search_geometry_for_material(prim, color)


def _search_geometry_for_material(search_prim, color: Gf.Vec3f) -> bool:
    """Recursively search for geometry and update its material color."""
    for child in search_prim.GetChildren():
        if child.GetTypeName() in ["Mesh", "Cube"] or "Geom" in child.GetTypeName():
            binding_api = UsdShade.MaterialBindingAPI(child)
            material, _ = binding_api.ComputeBoundMaterial()
            if material:
                for shader_child in material.GetPrim().GetChildren():
                    if shader_child.GetTypeName() == "Shader":
                        for attr_name in ["inputs:diffuseColor", "inputs:diffuse_color_constant", "inputs:baseColor"]:
                            color_attr = shader_child.GetAttribute(attr_name)
                            if color_attr.IsValid():
                                color_attr.Set(color)
                                return True
        if _search_geometry_for_material(child, color):
            return True
    return False
