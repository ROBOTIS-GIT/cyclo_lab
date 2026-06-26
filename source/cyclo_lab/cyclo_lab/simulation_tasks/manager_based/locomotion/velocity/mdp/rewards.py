# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Portions of this file are derived from IsaacLab:
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_airtime_touchdown(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_threshold: float = 0.1,
    target_air_time: float = 0.4,
    threshold: float | None = None,
) -> torch.Tensor:
    """Sparse touchdown airtime reward for regulating stepping frequency."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    walking_reward = ((last_air_time - target_air_time) * first_contact.float()).sum(dim=1)

    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_standing = cmd_norm < stand_threshold
    return torch.where(is_standing, torch.ones_like(walking_reward), walking_reward)


def feet_single_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_threshold: float = 0.1,
    window_s: float = 0.2,
) -> torch.Tensor:
    """Reward recent single-foot contact while allowing natural double-support overlap."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    num_contacts = torch.sum(contact_time > 0.0, dim=1)
    is_single_stance = num_contacts == 1
    is_recent_double_stance = (num_contacts == 2) & (torch.min(contact_time, dim=1).values < window_s)
    is_recent_flight = (num_contacts == 0) & (torch.min(air_time, dim=1).values < window_s)
    walking_reward = (is_single_stance | is_recent_double_stance | is_recent_flight).float()

    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :2], dim=1)
    yaw_norm = torch.abs(command[:, 2])
    is_standing = (cmd_norm < stand_threshold) & (yaw_norm < stand_threshold)
    return torch.where(is_standing, torch.ones_like(walking_reward), walking_reward)


def feet_touchdown_acc(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 50.0,
) -> torch.Tensor:
    """Penalize foot acceleration at touchdown events."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    asset: RigidObject = env.scene[asset_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    foot_acc = torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1)
    impact_acc = torch.clamp(foot_acc - threshold, min=0.0)
    return torch.sum(first_contact.float() * impact_acc, dim=1)


def feet_touchdown_xy_vel_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize horizontal foot velocity at touchdown events."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    asset: RigidObject = env.scene[asset_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    foot_xy_vel = torch.sum(torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=-1)
    return torch.sum(first_contact.float() * foot_xy_vel, dim=1)


def feet_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground."""
    asset: RigidObject = env.scene[asset_cfg.name]
    num_feet = len(asset_cfg.body_ids)
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity_w = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)
    feet_proj_g = quat_apply_inverse(feet_quat, gravity_w)
    return torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1).sum(dim=-1)
