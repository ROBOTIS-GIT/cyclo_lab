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

import gymnasium as gym

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Cyclo-Real-Pick-Place-Bottle-OMY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OMYBottlePickPlaceEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Cyclo-Real-Mimic-Pick-Place-Bottle-OMY-v0",
    entry_point="cyclo_lab.real_world_tasks.manager_based.OMY.pick_place.pick_place_mimic_env:OMYPickPlaceMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_mimic_env_cfg:OMYPickPlaceMimicEnvCfg",
    },
    disable_env_checker=True,
)
