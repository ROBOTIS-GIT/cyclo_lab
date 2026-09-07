"""Continuous SG2 showroom environment exposed through ROS2-compatible topics."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from cyclo_lab.manager_based.actions import FFWSG2MobileActionsCfg
from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_PUBLISHED_JOINT_NAMES
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp

SHOWROOM_CAMERA_NAMES = ("cam_head", "cam_wrist_left", "cam_wrist_right")


@configclass
class ShowroomSceneCfg(InteractiveSceneCfg):
    """Empty 5 m showroom with a free SG2 and robot cameras."""

    robot: ArticulationCfg = MISSING
    environment: AssetBaseCfg = MISSING
    cam_head: CameraCfg = MISSING
    cam_wrist_left: CameraCfg = MISSING
    cam_wrist_right: CameraCfg = MISSING
    cam_overhead_left: CameraCfg | None = None
    cam_overhead_center: CameraCfg | None = None
    cam_overhead_right: CameraCfg | None = None

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg(FFWSG2MobileActionsCfg):
    """Canonical 19 SG2 joint targets followed by 3D base velocity."""


@configclass
class ObservationsCfg:
    """Low-dimensional state kept available to manager-based consumers."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_name,
            params={"joint_names": FFW_SG2_PUBLISHED_JOINT_NAMES, "asset_name": "robot"},
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={"joint_names": FFW_SG2_PUBLISHED_JOINT_NAMES, "asset_name": "robot"},
        )
        base_twist = ObsTerm(func=mdp.base_twist, params={"asset_name": "robot"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    pass


@configclass
class ShowroomEnvCfg(ManagerBasedEnvCfg):
    """Continuous SG2 showroom environment without RL episode semantics."""

    env_name: str = "Cyclo-Real-Showroom-FFW-SG2-v0"
    control_hz: float = 15.0
    physics_hz: float = 30.0
    camera_hz: float = 15.0
    operator_camera_rows: tuple = (
        (
            ("cam_overhead_left", "External Left"),
            ("cam_overhead_center", "External Top"),
            ("cam_overhead_right", "External Right"),
        ),
        (
            ("cam_wrist_left", "Wrist Left"),
            ("cam_head", "Head"),
            ("cam_wrist_right", "Wrist Right"),
        ),
    )
    operator_camera_rotations: tuple = (
        ("cam_wrist_left", 1),
        ("cam_wrist_right", 1),
    )
    operator_camera_title: str = "SG2 Operator Dashboard"
    operator_camera_window_size: int = 1800
    scene: ShowroomSceneCfg = ShowroomSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        if min(self.control_hz, self.physics_hz, self.camera_hz) <= 0.0:
            raise ValueError("Showroom control, physics, and camera rates must be positive.")
        if self.physics_hz < self.control_hz:
            raise ValueError("Showroom physics rate must be greater than or equal to the control rate.")
        if self.physics_hz < self.camera_hz:
            raise ValueError("Showroom physics rate must be greater than or equal to the camera rate.")

        physics_steps_per_control = self.physics_hz / self.control_hz
        physics_steps_per_render = self.physics_hz / self.camera_hz
        for name, ratio in (
            ("physics/control", physics_steps_per_control),
            ("physics/render", physics_steps_per_render),
        ):
            if not ratio.is_integer():
                raise ValueError(f"Showroom rate ratio {name} must be an integer, got {ratio}.")

        self.decimation = int(physics_steps_per_control)
        self.sim.dt = 1.0 / self.physics_hz
        self.sim.render_interval = int(physics_steps_per_render)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

    def set_camera_set(self, camera_set: str):
        """Enable or disable all robot camera sensors."""
        if camera_set == "all":
            camera_names = SHOWROOM_CAMERA_NAMES
        elif camera_set == "none":
            camera_names = ()
        else:
            raise ValueError(f"Unsupported showroom camera set: {camera_set}")

        enabled = set(camera_names)
        if "cam_head" not in enabled:
            self.scene.cam_head = None
        if "cam_wrist_left" not in enabled:
            self.scene.cam_wrist_left = None
        if "cam_wrist_right" not in enabled:
            self.scene.cam_wrist_right = None
