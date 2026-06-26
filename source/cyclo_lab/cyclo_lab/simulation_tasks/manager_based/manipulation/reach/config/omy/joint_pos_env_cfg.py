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


from isaaclab.utils import configclass

import cyclo_lab.simulation_tasks.manager_based.manipulation.reach.config.omy.mdp as mdp
from cyclo_lab.simulation_tasks.manager_based.manipulation.reach.config.omy.reach_env_cfg import ReachEnvCfg
from isaaclab.managers import SceneEntityCfg

##
# Pre-defined configs
##
from cyclo_lab.assets.robots.OMY import OMY_OFF_SELF_COLLISION_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class OMYReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to OMY robot
        self.scene.robot = OMY_OFF_SELF_COLLISION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link6"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link6"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link6"]

        arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=arm_joint_names,
            scale=0.5,
            use_default_offset=True
        )
        # override observations policy
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names
        )

        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "link6"


@configclass
class OMYReachEnvCfg_PLAY(OMYReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
