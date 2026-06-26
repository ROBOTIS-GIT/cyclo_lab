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

"""
Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, basket_cfg: SceneEntityCfg, distance_threshold: float = 0.10) -> torch.Tensor:
    """
    Success = object placed inside the basket.
    """

    object: RigidObject = env.scene[object_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]

    object_pos = object.data.root_pos_w
    basket_pos = basket.data.root_pos_w

    # Check 3D distance between object and basket
    distance_3d = torch.linalg.vector_norm(object_pos - basket_pos, dim=1)
    done = distance_3d < distance_threshold

    return done


def object_dropped(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, velocity_threshold: float = 1.0) -> torch.Tensor:
    """
    Failure = object is falling with velocity exceeding threshold.

    Args:
        env: The RL environment.
        object_cfg: Configuration for the object.
        velocity_threshold: Maximum allowed velocity magnitude (m/s).

    Returns:
        Boolean tensor indicating which environments have dropped the object.
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Get linear velocity of the object
    object_vel = object.data.root_lin_vel_w

    # Calculate velocity magnitude (3D vector norm)
    velocity_magnitude = torch.linalg.vector_norm(object_vel, dim=1)

    # Check if velocity exceeds threshold
    dropped = velocity_magnitude > velocity_threshold

    return dropped
