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
# Author: Kiwoong Park

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import cyclo_lab.simulation_tasks.manager_based.locomotion.velocity.mdp as mdp
from cyclo_lab.simulation_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    LocomotionVelocityEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

from cyclo_lab.assets.robots.K1_rev1 import K1_REV1_CFG


@configclass
class K1Rev1CommandsCfg(CommandsCfg):
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class K1Rev1ActionsCfg(ActionsCfg):
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class K1Rev1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class K1Rev1EventCfg(EventCfg):
    pass


@configclass
class K1Rev1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.80, "sensor_cfg": None},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.10)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-3.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_airtime_touchdown,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
            "target_air_time": 0.4,
        },
    )
    feet_single_contact = RewTerm(
        func=mdp.feet_single_contact,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
            "window_s": 0.2,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_touchdown_acc = RewTerm(
        func=mdp.feet_touchdown_acc,
        weight=-0.002,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 50.0,
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.3,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.feet_orientation_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    feet_touchdown_xy_vel_l2 = RewTerm(
        func=mdp.feet_touchdown_xy_vel_l2,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_wrist_roll_joint",
                ],
            )
        },
    )
    joint_deviation_shoulder_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_pitch_joint")},
    )
    joint_deviation_elbow = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_elbow_joint")},
    )
    joint_deviation_arm_lateral = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_wrist_roll_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint")},
    )


@configclass
class K1Rev1TerminationsCfg(TerminationsCfg):
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]),
            "threshold": 1.0,
        },
    )


@configclass
class K1Rev1CurriculumCfg(CurriculumCfg):
    pass


@configclass
class K1Rev1FlatEnvCfg(LocomotionVelocityEnvCfg):
    export_sim2real_cfg: bool = True

    commands: K1Rev1CommandsCfg = K1Rev1CommandsCfg()
    actions: K1Rev1ActionsCfg = K1Rev1ActionsCfg()
    observations: K1Rev1ObservationsCfg = K1Rev1ObservationsCfg()
    events: K1Rev1EventCfg = K1Rev1EventCfg()
    rewards: K1Rev1Rewards = K1Rev1Rewards()
    terminations: K1Rev1TerminationsCfg = K1Rev1TerminationsCfg()
    curriculum: K1Rev1CurriculumCfg = K1Rev1CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = K1_REV1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # Randomization
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.base_external_force_torque.mode = "reset"
        self.events.base_external_force_torque.interval_range_s = None
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 0.0)

        self.rewards.joint_deviation_ankle_pitch = None


class K1Rev1FlatEnvCfg_PLAY(K1Rev1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        # remove random pushing
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
