# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Isaac environment for multi-clip motion learners."""

from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.robots.K1_rev1 import K1_REV1_INERTIA_TUNED_ACTION_SCALE, K1_REV1_INERTIA_TUNED_CFG
from cyclo_lab.utils.k1_physics import configure_k1_physics

from .beyondmimic.mdp import rewards, terminations
from .beyondmimic.mdp.commands import ReferenceTrajectoryCommandCfg
from .beyondmimic.tracking_env_cfg import ActionsCfg, MySceneCfg
from .motion_bridge import BODIES
from .motion_library_command import MotionLibraryCommand, at_motion_end
from .tracking_physics import initialize_dynamics, sample_encoder_bias


def tracking_reward(func, weight, std):
    return RewardTermCfg(func=func, weight=weight, params={"command_name": "reference_trajectory", "std": std})


@configclass
class TrackingRewardsCfg:
    reference_anchor_position = tracking_reward(rewards.reference_anchor_position_tracking, 0.5, 0.3)
    reference_anchor_orientation = tracking_reward(rewards.reference_anchor_orientation_tracking, 0.5, 0.4)
    reference_body_position = tracking_reward(rewards.reference_body_position_tracking, 1.0, 0.3)
    reference_body_orientation = tracking_reward(rewards.reference_body_orientation_tracking, 1.0, 0.4)
    reference_body_linear_velocity = tracking_reward(rewards.reference_body_linear_velocity_tracking, 1.0, 1.0)
    reference_body_angular_velocity = tracking_reward(rewards.reference_body_angular_velocity_tracking, 1.0, 3.14)
    joint_acc = None
    joint_torque = None
    undesired_contacts = None

    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewardTermCfg(
        func=mdp.joint_pos_limits, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class MotionLibraryEventsCfg:
    physics_material = EventTermCfg(
        func=initialize_dynamics, mode="startup", params={"feet_only": False, "evaluation": False}
    )
    joint_home_offset_noise = EventTermCfg(func=sample_encoder_bias, mode="reset")
    torso_com_offset_noise = None
    push_robot = None


@configclass
class MotionLibraryObservationsCfg:
    """The runners assemble actor inputs directly from the simulated state."""


@configclass
class MotionLibraryCommandCfg(ReferenceTrajectoryCommandCfg):
    class_type: type = MotionLibraryCommand
    additional_motions: tuple[str, ...] = ()
    clip_timeout: bool = True
    targets_next_frame: bool = False


@configclass
class MotionLibraryCommandsCfg:
    reference_trajectory = MotionLibraryCommandCfg(
        asset_name="robot",
        anchor_body_name="pelvis",
        body_names=list(BODIES),
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        reset_pose_noise={
            "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2),
        },
        reset_velocity_noise={
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
        },
        reset_joint_position_noise=(-0.1, 0.1),
    )


@configclass
class MotionLibraryTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    anchor_height = TerminationTermCfg(
        func=terminations.reference_anchor_height_deviation,
        params={"command_name": "reference_trajectory", "threshold": 0.25},
    )
    anchor_ori = TerminationTermCfg(
        func=terminations.reference_anchor_gravity_deviation,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "reference_trajectory", "threshold": 0.8},
    )
    joint_vel_explosion = TerminationTermCfg(
        func=mdp.joint_vel_out_of_manual_limit,
        params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    physx_nan = TerminationTermCfg(
        func=terminations.physx_nan_detected, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    motion_end = TerminationTermCfg(func=at_motion_end, time_out=True, params={"command_name": "reference_trajectory"})


@configclass
class K1MotionLibraryEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: MotionLibraryObservationsCfg = MotionLibraryObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: MotionLibraryCommandsCfg = MotionLibraryCommandsCfg()
    terminations: MotionLibraryTerminationsCfg = MotionLibraryTerminationsCfg()
    export_sim2real_cfg: bool = False
    events: MotionLibraryEventsCfg = MotionLibraryEventsCfg()
    curriculum = None
    rewards: TrackingRewardsCfg = TrackingRewardsCfg()

    def set_motion_paths(self, paths):
        first, *rest = paths
        command = self.commands.reference_trajectory
        command.trajectory_file = str(first)
        command.additional_motions = tuple(map(str, rest))

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.robot = K1_REV1_INERTIA_TUNED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_REV1_INERTIA_TUNED_ACTION_SCALE
        configure_k1_physics(self)

        self.episode_length_s = 10.0
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 300
        # Deep contacts during random reference resets exceed the walking preset.
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 128
        self.sim.physics.default = self.sim.physics.newton_mjwarp

    def play_mode(self) -> None:
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.commands.reference_trajectory.start_from_zero = True
        self.events.push_robot = None
