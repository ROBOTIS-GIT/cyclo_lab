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

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from cyclo_lab.simulation_tasks.manager_based.manipulation.pick_place.config.ffw_bg2 import mdp
from cyclo_lab.simulation_tasks.manager_based.manipulation.pick_place.config.ffw_bg2.mdp import ffw_bg2_pick_place_events
from cyclo_lab.simulation_tasks.manager_based.manipulation.pick_place.config.ffw_bg2.pick_place_env_cfg import PickPlaceEnvCfg

from cyclo_lab.assets.robots.FFW_BG2 import FFW_BG2_WITHOUT_MIMIC_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_ffw_bg2_pose = EventTerm(
        func=ffw_bg2_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": {
                "arm_r_joint2": -1.13,
                "arm_r_joint3": 0.03,
                "arm_r_joint4": -2.1,
                "arm_r_joint5": -1.44,
                "arm_r_joint6": 0.43,
                "arm_r_joint7": -0.65,
                "head_joint1": 0.695,
                "head_joint2": -0.35,
            },
        },
    )

    randomize_object_position = EventTerm(
        func=ffw_bg2_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.5, 0.7), "y": (-0.20, 0.0), "z": (1.1413, 1.1413)},
            "min_separation": 0.12,
            "asset_cfgs": [SceneEntityCfg("object")],
        },
    )


@configclass
class PickPlaceFFWBG2EnvCfg(PickPlaceEnvCfg):
    """Configuration for the FFW_BG2 environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set FFW_BG2 as robot
        self.scene.robot = FFW_BG2_WITHOUT_MIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        self.scene.object.spawn.semantic_tags = [("class", "object")]

        arm_joint_names = [
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
            "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"
        ]
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names
        )
        # Set actions for the specific robot type (FFW_BG2)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=arm_joint_names, scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_r_joint[1-4]"],
            open_command_expr={"gripper_r_joint.*": 0.0},
            close_command_expr={"gripper_r_joint.*": 1.0},
        )
        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/right_arm/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/right_wrist_cam",
            update_period=0.0,
            height=244,
            width=244,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.08, 0.0, 0.0), rot=(0.5, -0.5, -0.5, 0.5), convention="isaac"
            ),
        )
        self.scene.head_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/head/head_link2/head_cam",
            update_period=0.0,
            height=244,
            width=244,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.03, 0.04, 0.0), rot=(0.5, 0.5, -0.5, -0.5), convention="isaac"
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/right_arm/arm_r_link7",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )
