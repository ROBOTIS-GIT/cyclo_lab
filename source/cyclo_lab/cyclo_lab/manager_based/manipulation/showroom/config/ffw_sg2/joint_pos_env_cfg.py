"""Joint-position SG2 showroom environment configuration."""

from __future__ import annotations

import math
from copy import deepcopy

from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_BACKGROUND_TEXTURE_PATHS,
    make_robotis_showroom_environment_cfg,
)
from cyclo_lab.assets.robots import FFW_SG2_PHYSICS_CFG
from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    make_ffw_sg2_head_camera_cfg,
    make_ffw_sg2_overhead_camera_cfg,
    make_ffw_sg2_wrist_camera_cfg,
)
from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_SWERVE_DRIVE_SPEED_SCALE
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .mdp import ffw_sg2_showroom_events
from .showroom_env_cfg import ShowroomEnvCfg

SG2_SHOWROOM_ROBOT_POS = (-1.116, 1.681, 0.0)
SG2_SHOWROOM_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
SG2_SHOWROOM_HEAD_CAMERA_WIDTH = 640
SG2_SHOWROOM_HEAD_CAMERA_HEIGHT = 480
SG2_SHOWROOM_ROOT_POSITION_RANDOMIZATION_RADIUS = 0.1
SG2_SHOWROOM_ROOT_YAW_RANDOMIZATION = math.radians(10.0)
SG2_SHOWROOM_WALL_BACKGROUND_ZOOM_RANGE = (1.0, 1.3)
SG2_SHOWROOM_INITIAL_JOINT_POSITIONS = {
    "arm_l_joint1": 0.0005,
    "arm_l_joint2": 0.6040,
    "arm_l_joint3": -0.2963,
    "arm_l_joint4": -2.5052,
    "arm_l_joint5": 0.5672,
    "arm_l_joint6": 0.4926,
    "arm_l_joint7": 0.7391,
    "gripper_l_joint1": 0.0,
    "arm_r_joint1": 0.0005,
    "arm_r_joint2": -0.6040,
    "arm_r_joint3": 0.2963,
    "arm_r_joint4": -2.5052,
    "arm_r_joint5": -0.5672,
    "arm_r_joint6": 0.4926,
    "arm_r_joint7": -0.7391,
    "gripper_r_joint1": 0.0,
    "head_joint1": 0.5961,
    "head_joint2": 0.0,
    "lift_joint": 0.0,
}


def make_sg2_showroom_robot_cfg() -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_SG2_PHYSICS_CFG)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.init_state.pos = SG2_SHOWROOM_ROBOT_POS
    robot_cfg.init_state.rot = SG2_SHOWROOM_ROBOT_ROT
    robot_cfg.init_state.joint_pos.update(SG2_SHOWROOM_INITIAL_JOINT_POSITIONS)
    base_drive_actuator = robot_cfg.actuators.get("base_drive")
    if base_drive_actuator is not None:
        base_drive_actuator.velocity_limit_sim *= FFW_SG2_SWERVE_DRIVE_SPEED_SCALE
    return robot_cfg


@configclass
class EventCfg:
    """Reset events for the SG2 showroom joint-position task."""

    reset_scene_to_default = EventTerm(
        func=reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    randomize_robot_root_pose = EventTerm(
        func=ffw_sg2_showroom_events.randomize_root_pose_in_radius,
        mode="reset",
        params={
            "max_translation_radius": SG2_SHOWROOM_ROOT_POSITION_RANDOMIZATION_RADIUS,
            "max_yaw": SG2_SHOWROOM_ROOT_YAW_RANDOMIZATION,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_wall_background = EventTerm(
        func=ffw_sg2_showroom_events.randomize_wall_background,
        mode="reset",
        params={
            "texture_paths": ROBOTIS_SHOWROOM_BACKGROUND_TEXTURE_PATHS,
            "zoom_range": SG2_SHOWROOM_WALL_BACKGROUND_ZOOM_RANGE,
        },
    )

    set_robot_joint_pose = EventTerm(
        func=ffw_sg2_showroom_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": SG2_SHOWROOM_INITIAL_JOINT_POSITIONS,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FFWSG2ShowroomEnvCfg(ShowroomEnvCfg):
    """Canonical SG2 showroom environment used by the ROS2 topic runner."""

    def __post_init__(self):
        super().__post_init__()
        self.events = EventCfg()

        self.scene.robot = make_sg2_showroom_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.environment = make_robotis_showroom_environment_cfg()
        # Rendering cadence is owned by sim.render_interval. Sensors expose each
        # newly rendered frame directly to the topic bridge and operator viewer.
        self.scene.cam_head = make_ffw_sg2_head_camera_cfg(
            update_period=0.0,
            width=SG2_SHOWROOM_HEAD_CAMERA_WIDTH,
            height=SG2_SHOWROOM_HEAD_CAMERA_HEIGHT,
        )
        self.scene.cam_wrist_left = make_ffw_sg2_wrist_camera_cfg(
            "left",
            update_period=0.0,
        )
        self.scene.cam_wrist_right = make_ffw_sg2_wrist_camera_cfg(
            "right",
            update_period=0.0,
        )

    def enable_operator_preview_cameras(self) -> None:
        """Enable the robot-following cameras used only by the operator dashboard."""
        self.scene.cam_overhead_left = make_ffw_sg2_overhead_camera_cfg(
            "left",
            update_period=0.0,
        )
        self.scene.cam_overhead_center = make_ffw_sg2_overhead_camera_cfg(
            "center",
            update_period=0.0,
        )
        self.scene.cam_overhead_right = make_ffw_sg2_overhead_camera_cfg(
            "right",
            update_period=0.0,
        )
