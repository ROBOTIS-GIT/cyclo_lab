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

from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.pick_place.mdp import ffw_sg2_pick_place_events
from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.pick_place.pick_place_env_cfg import PickPlaceEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from cyclo_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG  # isort: skip
from cyclo_lab.assets.object.robotis_net_table import NET_TABLE_CFG
from cyclo_lab.assets.object.plastic_basket2 import PLASTIC_BASKET2_CFG
from cyclo_lab.assets.object.brush_ring import BRUSH_RING_CFG
from cyclo_lab.assets.object.silicone_tube_ring import SILICONE_TUBE_RING_CFG
from cyclo_lab.assets.object.pliers_ring import PLIERS_RING_CFG
from cyclo_lab.assets.object.scissors_ring import SCISSORS_RING_CFG
from cyclo_lab.assets.object.screw_driver_ring import SCREW_DRIVER_RING_CFG
from cyclo_lab.assets.object.tooth_brush import TOOTH_BRUSH_CFG
from cyclo_lab.assets.object.background_cube import BACKGROUND_CUBE_CFG


@configclass
class EventCfg:
    """Configuration for reset events.

    Events are organized in the following order:
    1. Robot state initialization
    2. Scene object placement (table + objects moving together)
    3. Visual domain randomization (background color, lighting)
    """

    # ========== Robot State Initialization ==========
    set_robot_joint_pose = EventTerm(
        func=ffw_sg2_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": {
                "arm_l_joint1": 0.75, "arm_l_joint4": -2.30,
                "arm_r_joint1": 0.75, "arm_r_joint4": -2.30,
                "head_joint1": 0.549, "lift_joint": -0.0993,
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_ffw_sg2_joint_state = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.05,
            "joint_names": [
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
                "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
                "arm_r_joint5", "arm_r_joint6", "arm_r_joint7",
                "head_joint1", "head_joint2",
            ],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_head_camera = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_camera_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cam_head"),
            "pose_range": {
                "x": (0.0, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
            "convention": "ros",
        },
    )

    # ========== Scene Object Placement ==========

    randomize_table_with_objects: EventTerm | None = None

    # ========== Visual Domain Randomization ==========
    randomize_background_color = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_background_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("background_cube"),
            "color_range": ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        },
    )

    randomize_scene_light = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (500.0, 3000.0),
            "color_range": ((0.8, 1.0), (0.8, 1.0), (0.8, 1.0)),
            "asset_cfg": SceneEntityCfg("light"),
        },
    )


@configclass
class FFWSG2PickPlaceEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events = EventCfg()

        # Configure table randomization with all objects moving together
        # This maintains relative positions for correct trajectory augmentation
        other_objects = [obj for obj in self.all_objects if obj != self.target_object]
        self.events.randomize_table_with_objects = EventTerm(
            func=ffw_sg2_pick_place_events.randomize_table_with_objects_on_slots,
            mode="reset",
            params={
                "table_cfg": SceneEntityCfg("table"),
                "basket_cfg": SceneEntityCfg("basket"),
                "target_asset_cfg": SceneEntityCfg(self.target_object),
                "other_asset_cfgs": [SceneEntityCfg(obj) for obj in other_objects],
                "basket_relative_pose": {"x": 0.41, "y": 0.0, "z": 0.72},
                "target_side": self.target_side,
                "table_pose_range": {
                    "x": (-0.02, 0.02),
                    "y": (-0.02, 0.02),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.05, 0.05),  # ±~3 degrees
                },
            },
        )

        # ========== Scene Setup ==========
        # Robot
        self.scene.robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Table and Objects
        self.scene.table = NET_TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")
        self.scene.basket = PLASTIC_BASKET2_CFG.replace(prim_path="{ENV_REGEX_NS}/Basket")
        self.scene.brush = BRUSH_RING_CFG.replace(prim_path="{ENV_REGEX_NS}/Brush")
        self.scene.pliers = PLIERS_RING_CFG.replace(prim_path="{ENV_REGEX_NS}/Pliers")
        self.scene.silicone = SILICONE_TUBE_RING_CFG.replace(prim_path="{ENV_REGEX_NS}/SiliconeTube")
        self.scene.scissors = SCISSORS_RING_CFG.replace(prim_path="{ENV_REGEX_NS}/Scissors")
        self.scene.driver = SCREW_DRIVER_RING_CFG.replace(prim_path="{ENV_REGEX_NS}/ScrewDriver")
        self.scene.tooth_brush = TOOTH_BRUSH_CFG.replace(prim_path="{ENV_REGEX_NS}/ToothBrush")

        # Visual Domain Randomization
        self.scene.background_cube = BACKGROUND_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/BackgroundCube")
        self.scene.plane.semantic_tags = [("class", "ground")]

        self.scene.cam_head = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/head_link2/zed/cam_head",
            update_period=0.0,
            height=376,
            width=672,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.4, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.03, 0.0),
                rot=(0.5, 0.5, -0.5, -0.5),
                convention="isaac",
            )
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.right_eef = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,  
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_r_link7",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, -0.2],
                    ),
                ),
            ],
        )

        self.scene.left_eef = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_l_link7",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, -0.2],
                    ),
                ),
            ],
        )
