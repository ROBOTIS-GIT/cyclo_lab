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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from . import mdp


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    left_eef: FrameTransformerCfg = MISSING
    right_eef: FrameTransformerCfg = MISSING

    brush: AssetBaseCfg = MISSING
    basket: AssetBaseCfg = MISSING
    table: AssetBaseCfg = MISSING
    silicone: AssetBaseCfg = MISSING
    scissors: AssetBaseCfg = MISSING
    driver: AssetBaseCfg = MISSING

    cam_head: CameraCfg = MISSING

    # Background cube for color randomization
    background_cube: AssetBaseCfg = MISSING

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_l_action: mdp.ActionTermCfg = MISSING
    gripper_l_action: mdp.ActionTermCfg = MISSING
    arm_r_action: mdp.ActionTermCfg = MISSING
    gripper_r_action: mdp.ActionTermCfg = MISSING
    lift_action: mdp.ActionTermCfg = MISSING
    head_action: mdp.ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_name,
            params={"joint_names": ["arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
                                    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
                                    "head_joint1", "head_joint2", "lift_joint"],
                    "asset_name": "robot"},
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={"joint_names": ["arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
                                    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
                                    "head_joint1", "head_joint2", "lift_joint"],
                    "asset_name": "robot"},
        )
        left_eef_pose = ObsTerm(func=mdp.eef_pose, params={"eef_cfg": SceneEntityCfg("left_eef"), "robot_cfg": SceneEntityCfg("robot")})
        right_eef_pose = ObsTerm(func=mdp.eef_pose, params={"eef_cfg": SceneEntityCfg("right_eef"), "robot_cfg": SceneEntityCfg("robot")})

        cam_head = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_head"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # Note: object_cfg will be set dynamically based on target_object in __post_init__
        grasp_object = None
        object_in_basket = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Note: success and object_dropped will be set dynamically based on target_object in __post_init__
    success = None
    object_dropped = None


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick and place environment."""

    # All available objects for randomization
    all_objects: list = ["brush", "driver", "scissors", "pliers", "tooth_brush", "silicone"]
    # Target object configuration
    target_object: str = "brush"  # Options: "silicone", "brush", "scissors", "driver", "pliers", "tooth_brush"
    # Target side configuration: which side to place the target object
    target_side: str = "right"  # Options: "left", "right"

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Determine eef and gripper based on target_side
        if self.target_side == "left":
            eef_name = "left_eef"
            gripper_joint_name = "gripper_l_joint1"
        else:  # right
            eef_name = "right_eef"
            gripper_joint_name = "gripper_r_joint1"

        # Initialize dynamic observations and terminations based on target_object
        self.observations.subtask_terms.grasp_object = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "eef_cfg": SceneEntityCfg(eef_name),
                "object_cfg": SceneEntityCfg(self.target_object),
                "gripper_joint_name": gripper_joint_name,
            },
        )

        self.observations.subtask_terms.object_in_basket = ObsTerm(
            func=mdp.object_in_basket,
            params={
                "object_cfg": SceneEntityCfg(self.target_object),
                "basket_cfg": SceneEntityCfg("basket"),
                "distance_threshold": 0.15,
            },
        )

        self.terminations.success = DoneTerm(
            func=mdp.task_done,
            params={
                "object_cfg": SceneEntityCfg(self.target_object),
                "basket_cfg": SceneEntityCfg("basket"),
                "distance_threshold": 0.15,
            },
        )

        self.terminations.object_dropped = DoneTerm(
            func=mdp.object_dropped,
            params={
                "object_cfg": SceneEntityCfg(self.target_object),
                "velocity_threshold": 2.0,
            },
        )

    def init_action_cfg(self, mode: str):
        print(f"Initializing action configuration for device: {mode}")
        if mode in ['record', 'inference']:
            self.actions.arm_l_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["arm_l_joint[1-7]"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.gripper_l_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper_l_joint1"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.arm_r_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["arm_r_joint[1-7]"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.gripper_r_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper_r_joint1"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.head_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["head_joint1", "head_joint2"],
                scale=1.0,
            )
            self.actions.lift_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["lift_joint"],
                scale=1.0,
            )
        elif mode in ['mimic_ik']:
            self.actions.arm_l_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["arm_l_joint[1-7]"],
                body_name="arm_l_link7",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=False
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.2]),
            )
            self.actions.gripper_l_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper_l_joint1"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.arm_r_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["arm_r_joint[1-7]"],
                body_name="arm_r_link7",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=False
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.2]),
            )
            self.actions.gripper_r_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper_r_joint1"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.head_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["head_joint1", "head_joint2"],
                scale=1.0,
            )
            self.actions.lift_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["lift_joint"],
                scale=1.0,
            )
        else:
            raise ValueError(f"Unknown action mode: {mode}")
