"""Continuous FFW-SH5 bringup environments."""

from __future__ import annotations

from copy import deepcopy

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from cyclo_lab.assets.environments.simple_warehouse import (
    make_card_boxes_graspable,
    make_simple_warehouse_environment_cfg,
)
from cyclo_lab.assets.robots import FFW_SH5_CFG
from cyclo_lab.assets.sensors.ffw_sh5_cameras import (
    FFW_SH5_HEAD_CAMERA_NAME,
    FFW_SH5_WRIST_LEFT_CAMERA_NAME,
    FFW_SH5_WRIST_RIGHT_CAMERA_NAME,
    make_ffw_sh5_camera_cfg,
)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

FFW_SH5_CONTROL_HZ = 15.0
FFW_SH5_PHYSICS_HZ = 30.0
FFW_SH5_CAMERA_HZ = 15.0
FFW_SH5_SOLVER_POSITION_ITERATIONS = 16
FFW_SH5_ROBOT_POS = (0.0, 0.0, -0.18)


def _make_robot_cfg() -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_SH5_CFG)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = FFW_SH5_SOLVER_POSITION_ITERATIONS
    robot_cfg.init_state.pos = FFW_SH5_ROBOT_POS
    return robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _make_warehouse_boxes_graspable(_env, _env_ids=None) -> None:
    make_card_boxes_graspable()


@configclass
class SH5BringupSceneCfg(InteractiveSceneCfg):
    """SH5, lighting, and an optional authored environment."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    environment: AssetBaseCfg | None = None
    robot: ArticulationCfg = _make_robot_cfg()
    # Rendering cadence is owned by sim.render_interval. Keeping the sensors
    # current avoids a second time-based refresh when the topic bridge reads RGB.
    cam_head = make_ffw_sh5_camera_cfg(
        FFW_SH5_HEAD_CAMERA_NAME,
        update_period=0.0,
    )
    cam_wrist_left = make_ffw_sh5_camera_cfg(
        FFW_SH5_WRIST_LEFT_CAMERA_NAME,
        update_period=0.0,
    )
    cam_wrist_right = make_ffw_sh5_camera_cfg(
        FFW_SH5_WRIST_RIGHT_CAMERA_NAME,
        update_period=0.0,
    )


@configclass
class SH5BringupActionsCfg:
    """SH5 targets are applied by its Zenoh topic bridge."""

    pass


@configclass
class SH5BringupObservationsCfg:
    """Minimal state exposed by the continuous environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SH5BringupEventCfg:
    reset_scene_to_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class SH5WarehouseEventCfg(SH5BringupEventCfg):
    make_card_boxes_graspable = EventTerm(
        func=_make_warehouse_boxes_graspable,
        mode="startup",
    )


@configclass
class FFWSH5BringupEnvCfg(ManagerBasedEnvCfg):
    """FFW-SH5 on a ground plane for ROS2-compatible topic control."""

    control_hz: float = FFW_SH5_CONTROL_HZ
    physics_hz: float = FFW_SH5_PHYSICS_HZ
    camera_hz: float = FFW_SH5_CAMERA_HZ
    operator_camera_rows: tuple = (
        (
            ("cam_wrist_left", "Wrist Left"),
            ("cam_head", "Head"),
            ("cam_wrist_right", "Wrist Right"),
        ),
    )
    # SH5 USD camera tensors are upright; only the ROS2 wrist streams need rotation.
    operator_camera_rotations: tuple = ()
    operator_camera_title: str = "SH5 Operator Dashboard"
    operator_camera_window_size: int = 1800
    scene: SH5BringupSceneCfg = SH5BringupSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=False,
    )
    actions: SH5BringupActionsCfg = SH5BringupActionsCfg()
    observations: SH5BringupObservationsCfg = SH5BringupObservationsCfg()
    events: SH5BringupEventCfg = SH5BringupEventCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.8, -2.2, 1.8),
        lookat=(0.0, 0.0, 0.8),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if min(self.control_hz, self.physics_hz, self.camera_hz) <= 0.0:
            raise ValueError("SH5 control, physics, and camera rates must be positive.")

        physics_steps_per_control = self.physics_hz / self.control_hz
        physics_steps_per_render = self.physics_hz / self.camera_hz
        for name, ratio in (
            ("physics/control", physics_steps_per_control),
            ("physics/render", physics_steps_per_render),
        ):
            if not ratio.is_integer():
                raise ValueError(f"SH5 rate ratio {name} must be an integer, got {ratio}.")

        self.decimation = int(physics_steps_per_control)
        self.sim.dt = 1.0 / self.physics_hz
        self.sim.render_interval = int(physics_steps_per_render)
        self.wait_for_textures = False


@configclass
class FFWSH5SimpleWarehouseBringupEnvCfg(FFWSH5BringupEnvCfg):
    """FFW-SH5 in NVIDIA Simple Warehouse."""

    events: SH5WarehouseEventCfg = SH5WarehouseEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.environment = make_simple_warehouse_environment_cfg()
