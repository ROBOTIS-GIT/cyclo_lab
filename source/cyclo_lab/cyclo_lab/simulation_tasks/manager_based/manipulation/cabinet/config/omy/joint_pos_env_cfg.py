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

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from cyclo_lab.simulation_tasks.manager_based.manipulation.cabinet.config.omy import mdp

from cyclo_lab.simulation_tasks.manager_based.manipulation.cabinet.config.omy.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    CabinetEnvCfg,
)

##
# Pre-defined configs
##
from cyclo_lab.assets.robots.OMY import OMY_CFG  # isort: skip


@configclass
class OMYCabinetEnvCfg(CabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set OMY as robot
        self.scene.robot = OMY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set Actions for the specific robot type (OMY)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["rh_r1_joint"],
            open_command_expr={"rh_r1_joint": 0.0},
            close_command_expr={"rh_r1_joint": 1.2},
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/world",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, -0.248, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/gripper/rh_p12_rn_l2",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/gripper/rh_p12_rn_r2",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

        # override rewards
        self.rewards.approach_gripper_handle.params["offset"] = 0.0567
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.0
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["rh_r1_joint"]


@configclass
class OMYCabinetEnvCfg_PLAY(OMYCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
