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
    ee_frame: FrameTransformerCfg = MISSING

    table: AssetBaseCfg = MISSING

    cam_wrist: CameraCfg = MISSING
    cam_top: CameraCfg = MISSING

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
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
    print("ActionsCfg: arm_action and gripper_action will be set by agent env cfg")
    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_name,
            params={"joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"],
                    "asset_name": "robot"},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_name,
            params={"joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"],
                    "asset_name": "robot"},
        )
        cam_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist"), "data_type": "rgb", "normalize": False},
        )
        cam_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_top"), "data_type": "rgb", "normalize": False},
        )
        eef_pose = ObsTerm(func=mdp.eef_pose, params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")})
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target_name,
                                   params={"joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"],
                                           "asset_name": "robot"},)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_bottle = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("bottle"),
            },
        )

        bottle_in_basket = ObsTerm(
            func=mdp.bottle_in_basket,
            params={
                "bottle_cfg": SceneEntityCfg("bottle"),
                "basket_cfg": SceneEntityCfg("basket"),
            },
        )

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

    success = DoneTerm(
        func=mdp.task_done, params={"bottle_cfg": SceneEntityCfg("bottle"), "basket_cfg": SceneEntityCfg("basket"), "distance_threshold": 0.1}
    )

    bottle_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("bottle")}
    )


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick and place environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
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

    def init_action_cfg(self, mode: str):
        print(f"Initializing action configuration for device: {mode}")
        if mode in ['record', 'inference']:
            self.actions.arm_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["joint.*"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.gripper_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["rh_r1_joint"],
                scale=1.0,
                use_default_offset=False,
            )
        elif mode in ['mimic_ik']:
            self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["joint[1-6]"],
                body_name="link6",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=False
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, -0.248, 0.0]),
            )
            self.actions.gripper_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["rh_r1_joint"],
                scale=1.0,
                use_default_offset=False,
            )
        else:
            self.actions.arm_action = None
            self.actions.gripper_action = None
