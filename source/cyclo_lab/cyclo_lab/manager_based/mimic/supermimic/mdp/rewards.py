# Copyright 2025 The mjlab Developers
# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
# Adapted from cyclo_mjlab/source/tasks/supermimic/mdp/rewards.py for Isaac Lab.
#
# Author: Kiwoong Park

import torch
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat


def heading_velocity_tracking(env, command_name, std, angular=False, body_names=None):
    """Match SuperMimic's heading-relative velocity rewards in the Isaac simulator.

    Return-based reset sampling uses these simulator rewards to select motion frames.
    """
    command = env.command_manager.get_term(command_name)
    heading = yaw_quat(quat_mul(command.robot_anchor_quat_w, quat_inv(command.anchor_quat_w)))
    if angular:
        target, actual = command.body_ang_vel_w, command.robot_body_ang_vel_w
    else:
        target, actual = command.body_lin_vel_w, command.robot_body_lin_vel_w
    if body_names is not None:
        indices = [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
        target, actual = target[:, indices], actual[:, indices]
    aligned = quat_apply(heading[:, None, :].expand(-1, target.shape[1], -1), target)
    error = (actual - aligned).square().sum(dim=-1).mean(dim=-1)
    return torch.exp(-error / std**2)
