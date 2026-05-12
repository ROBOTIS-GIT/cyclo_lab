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

from dataclasses import MISSING # aaa

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robotis_lab.simulation_tasks.manager_based.FFW_BG2.reach.mdp as mdp
import math


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the FFW_BG2 reach environment."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    ee_pose_l = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.65),
            pos_y=(0.1, 0.3),
            pos_z=(0.8, 1.6),
            roll=(- math.pi / 8, math.pi / 8),
            pitch=(math.pi / 2 - math.pi / 8, math.pi / 2 + math.pi / 8),
            yaw=(- math.pi / 8, math.pi / 8),
        ),
    )
    ee_pose_r = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.5, 0.7),
            pos_y=(- 0.3, - 0.1),
            pos_z=(0.8, 1.6),
            roll=(- math.pi / 8, math.pi / 8),
            pitch=(math.pi / 2 - math.pi / 8, math.pi / 2 + math.pi / 8),
            yaw=(- math.pi / 8, math.pi / 8),
        ),
    )


@configclass
class ActionsCfg:
    """Action configuration with base velocity and joint position actions."""

    lift_action: mdp.JointPositionActionCfg = MISSING
    arm_l_action: mdp.JointPositionActionCfg = MISSING
    arm_r_action: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation configuration."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command_l = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose_l"})
        pose_command_r = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose_r"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset event configuration."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward configuration."""

    # Reward left end-effector tracking
    end_effector_position_tracking_left = RewTerm(
        func=mdp.position_command_error,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose_l"},
    )
    end_effector_position_tracking_fine_grained_left = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.12,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose_l"},
    )
    end_effector_orientation_tracking_left = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.12,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose_r"},
    )

    # Reward right end-effector tracking
    end_effector_position_tracking_right = RewTerm(
        func=mdp.position_command_error,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose_r"},
    )
    end_effector_position_tracking_fine_grained_right = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.12,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose_r"},
    )
    end_effector_orientation_tracking_right = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.12,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose_r"},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum learning configuration."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 1000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 1000}
    )


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """FFW_BG2 reach environment configuration."""

    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
