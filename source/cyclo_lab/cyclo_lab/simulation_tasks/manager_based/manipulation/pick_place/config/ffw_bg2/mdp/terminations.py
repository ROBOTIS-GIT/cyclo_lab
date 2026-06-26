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

"""Common functions that can be used to activate certain terminations for the lift task.

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


def task_done(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_wrist_max_x: float = 0.2,
    min_x: float = 0.4,
    max_x: float = 0.8,
    min_y: float = -1.05,
    max_y: float = -0.40,
    min_height: float = 1.2,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Determine if the object placement task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x pos threshold)

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        min_x: Minimum x position of the object for task completion.
        max_x: Maximum x position of the object for task completion.
        min_y: Minimum y position of the object for task completion.
        max_y: Maximum y position of the object for task completion.
        min_height: Minimum height (z position) of the object for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]

    # Extract wheel position relative to environment origin
    wheel_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    wheel_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    wheel_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    wheel_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index("arm_r_link7")
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

    # Check all success conditions and combine with logical AND
    done = wheel_x < max_x
    done = torch.logical_and(done, wheel_x > min_x)
    done = torch.logical_and(done, wheel_y < max_y)
    done = torch.logical_and(done, wheel_y > min_y)
    done = torch.logical_and(done, wheel_height < min_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, wheel_vel[:, 0] < min_vel)
    done = torch.logical_and(done, wheel_vel[:, 1] < min_vel)
    done = torch.logical_and(done, wheel_vel[:, 2] < min_vel)

    return done


def object_fallen_over(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_y: float = -0.40,
    min_height: float = 1.02,
) -> torch.Tensor:

    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]

    # Extract wheel position relative to environment origin
    wheel_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    wheel_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    # Check all success conditions and combine with logical AND
    fail = wheel_y > max_y
    fail = torch.logical_and(fail, wheel_height < min_height)

    return fail
