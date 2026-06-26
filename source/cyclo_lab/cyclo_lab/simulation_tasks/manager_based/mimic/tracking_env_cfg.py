# Copyright 2026 ROBOTIS CO., LTD.
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
# Additional notices:
# This module is adapted from HybridRobotics/whole_body_tracking, licensed under the MIT License.
# See THIRD_PARTY_LICENSES.md for details.

"""Robot-agnostic mimic tracking environment configuration."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import cyclo_lab.simulation_tasks.manager_based.mimic.mdp as mdp

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    reference_trajectory = mdp.ReferenceTrajectoryCommandCfg(
        asset_name="robot",
        trajectory_file=MISSING,
        anchor_body_name=MISSING,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        reset_pose_noise={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        reset_velocity_noise=VELOCITY_RANGE,
        reset_joint_position_noise=(-0.1, 0.1),
        body_names=MISSING,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "reference_trajectory"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.reference_anchor_orientation_b,
            params={"command_name": "reference_trajectory"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "reference_trajectory"})
        reference_anchor_position_b = ObsTerm(
            func=mdp.reference_anchor_position_b, params={"command_name": "reference_trajectory"}
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.reference_anchor_orientation_b, params={"command_name": "reference_trajectory"}
        )
        measured_body_positions_b = ObsTerm(
            func=mdp.measured_body_positions_in_anchor_frame, params={"command_name": "reference_trajectory"}
        )
        measured_body_orientations_b = ObsTerm(
            func=mdp.measured_body_orientations_in_anchor_frame, params={"command_name": "reference_trajectory"}
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    joint_home_offset_noise = EventTerm(
        func=mdp.apply_home_joint_offset_noise,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    torso_com_offset_noise = EventTerm(
        func=mdp.apply_link_com_offset_noise,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reference_anchor_position = RewTerm(
        func=mdp.reference_anchor_position_tracking,
        weight=0.5,
        params={"command_name": "reference_trajectory", "std": 0.3},
    )
    reference_anchor_orientation = RewTerm(
        func=mdp.reference_anchor_orientation_tracking,
        weight=0.5,
        params={"command_name": "reference_trajectory", "std": 0.4},
    )
    reference_body_position = RewTerm(
        func=mdp.reference_body_position_tracking,
        weight=1.0,
        params={"command_name": "reference_trajectory", "std": 0.3},
    )
    reference_body_orientation = RewTerm(
        func=mdp.reference_body_orientation_tracking,
        weight=1.0,
        params={"command_name": "reference_trajectory", "std": 0.4},
    )
    reference_body_linear_velocity = RewTerm(
        func=mdp.reference_body_linear_velocity_tracking,
        weight=1.0,
        params={"command_name": "reference_trajectory", "std": 1.0},
    )
    reference_body_angular_velocity = RewTerm(
        func=mdp.reference_body_angular_velocity_tracking,
        weight=1.0,
        params={"command_name": "reference_trajectory", "std": 3.14},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_height = DoneTerm(
        func=mdp.reference_anchor_height_deviation, params={"command_name": "reference_trajectory", "threshold": 0.25}
    )
    anchor_ori = DoneTerm(
        func=mdp.reference_anchor_gravity_deviation,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "reference_trajectory", "threshold": 0.8},
    )
    end_body_pos = DoneTerm(
        func=mdp.reference_body_height_deviation,
        params={
            "command_name": "reference_trajectory",
            "threshold": 0.25,
            "body_names": MISSING,
        },
    )
    joint_vel_explosion = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    physx_nan = DoneTerm(func=mdp.physx_nan_detected, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    export_sim2real_cfg: bool = True

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


# Backwards-compatible names used by the K1 Rev.1 mimic configs.
TrackingSceneCfg = MySceneCfg
ReferenceCommandsCfg = CommandsCfg
TrackingActionsCfg = ActionsCfg
TrackingObservationsCfg = ObservationsCfg
TrackingEventsCfg = EventCfg
TrackingRewardsCfg = RewardsCfg
TrackingTerminationsCfg = TerminationsCfg
