# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bring up the FFW SH5 USD model and control it from DDS JointTrajectory topics.

* spawns the FFW_SH5 USD as an Isaac Lab articulation,
* subscribes to retargeted JointTrajectory topics from Cyclo Control,
* applies the latest received joint positions as simulation joint targets,
* publishes the simulated robot state on ``/joint_states` , ``/tf``.

Example:

.. code-block:: bash

    python scripts/sim2real/bringup/sh5_dds_bringup.py --enable_cameras

"""

import argparse
import os
import sys
import threading
import time
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

from isaaclab.app import AppLauncher


RIGHT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"
RIGHT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory"
LEFT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"
LEFT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_left_hand/joint_trajectory"
HEAD_TOPIC = "/leader/joystick_controller_left/joint_trajectory"
LIFT_TOPIC = "/leader/joystick_controller_right/joint_trajectory"
LIFT_JOINT_NAME = "lift_joint"
CMD_VEL_TOPIC = "/cmd_vel"
JOINT_STATES_TOPIC = "/joint_states"
TF_TOPIC = "/tf"
BASE_FRAME = "base_link"
PUBLISH_HZ = 30.0
STEP_HZ = 30.0
RENDER_INTERVAL = 1
ROBOT_POS = (0.0, 0.0, -0.18)
ARTICULATION_ROOT_PRIM_PATH = "/base_link/base_link"
SWERVE_STEERING_JOINTS = ("left_wheel_steer_joint", "right_wheel_steer_joint", "rear_wheel_steer_joint")
SWERVE_WHEEL_JOINTS = ("left_wheel_drive_joint", "right_wheel_drive_joint", "rear_wheel_drive_joint")
SWERVE_MODULE_X_OFFSETS = (0.18, 0.18, -0.18)
SWERVE_MODULE_Y_OFFSETS = (0.18, -0.18, 0.0)
SWERVE_MODULE_ANGLE_OFFSETS = (0.0, 0.0, 0.0)
SWERVE_WHEEL_RADIUS = 0.05
CMD_VEL_TIMEOUT = 0.1
BASE_LINEAR_DAMPING = 2.0
BASE_ANGULAR_DAMPING = 4.0
ENVIRONMENT_POS = (0.0, 0.0, 0.0)
ENVIRONMENT_ROT = (1.0, 0.0, 0.0, 0.0)
OVERVIEW_CAMERA_EYE = (2.8, -2.2, 1.8)
OVERVIEW_CAMERA_TARGET = (0.0, 0.0, 0.8)
CAMERA_CENTER_NAME = "Head_Camera"
CAMERA_LEFT_NAME = "Left_Camera"
CAMERA_RIGHT_NAME = "Right_Camera"
CAMERA_VIEW_WINDOWS = []

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


parser = argparse.ArgumentParser(description="FFW SH5 DDS bringup for Isaac Sim.")
parser.add_argument("--disable_head", action="store_true", help="Do not subscribe to the head topic.")
parser.add_argument("--disable_lift", action="store_true", help="Do not subscribe to the lift topic.")
parser.add_argument("--lift_topic", default=LIFT_TOPIC, help="DDS trajectory topic for the lift joint.")
parser.add_argument("--disable_cmd_vel", action="store_true", help="Do not subscribe to cmd_vel for the swerve base.")
parser.add_argument("--cmd_vel_topic", default=CMD_VEL_TOPIC, help="DDS geometry_msgs/Twist topic for the swerve base.")
parser.add_argument("--cmd_vel_timeout", type=float, default=CMD_VEL_TIMEOUT, help="Seconds before stale cmd_vel is treated as zero.")
parser.add_argument("--wheel_radius", type=float, default=SWERVE_WHEEL_RADIUS, help="Swerve wheel radius in meters.")
parser.add_argument("--base_linear_damping", type=float, default=BASE_LINEAR_DAMPING, help="Rigid-body linear damping for the SH5 USD bodies.")
parser.add_argument("--base_angular_damping", type=float, default=BASE_ANGULAR_DAMPING, help="Rigid-body angular damping for the SH5 USD bodies.")
parser.add_argument(
    "--swerve_module_x_offsets",
    default=",".join(str(value) for value in SWERVE_MODULE_X_OFFSETS),
    help="Comma-separated swerve module x offsets in meters.",
)
parser.add_argument(
    "--swerve_module_y_offsets",
    default=",".join(str(value) for value in SWERVE_MODULE_Y_OFFSETS),
    help="Comma-separated swerve module y offsets in meters.",
)
parser.add_argument(
    "--swerve_module_angle_offsets",
    default=",".join(str(value) for value in SWERVE_MODULE_ANGLE_OFFSETS),
    help="Comma-separated steering joint angle offsets in radians.",
)
parser.add_argument("--domain_id", type=int, default=None, help="DDS domain id. Defaults to ROS_DOMAIN_ID or 0.")
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity on the SH5 rigid bodies.")
parser.add_argument(
    "--environment_usd",
    default=None,
    help="USD file to spawn as the static environment. Defaults to source/robotis_lab/data/robots/table2.usd.",
)
parser.add_argument("--disable_environment", action="store_true", help="Do not spawn the environment USD.")
parser.add_argument(
    "--environment_pos",
    default=",".join(str(value) for value in ENVIRONMENT_POS),
    help="Comma-separated environment position in meters.",
)
parser.add_argument(
    "--environment_rot",
    default=",".join(str(value) for value in ENVIRONMENT_ROT),
    help="Comma-separated environment quaternion as w,x,y,z.",
)
parser.add_argument(
    "--enable_camera_views",
    action="store_true",
    help="Open Isaac Sim viewport windows for overview, Head_Camera, Left_Camera, and Right_Camera.",
)
parser.add_argument(
    "--camera_center_name",
    default=CAMERA_CENTER_NAME,
    help="USD camera prim name for the top-left center camera viewport.",
)
parser.add_argument(
    "--camera_left_name",
    default=CAMERA_LEFT_NAME,
    help="USD camera prim name for the bottom-left camera viewport.",
)
parser.add_argument(
    "--camera_right_name",
    default=CAMERA_RIGHT_NAME,
    help="USD camera prim name for the bottom-right camera viewport.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from cyclonedds.core import Qos, Policy
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import bind_physics_material, clone
from isaaclab.utils import configclass

from robotis_dds_python.idl.builtin_interfaces.msg import Time_
from robotis_dds_python.idl.geometry_msgs.msg import Quaternion_, Transform_, TransformStamped_, Twist_, Vector3_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.tf2_msgs.msg import TFMessage_
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.tools.topic_manager import TopicManager

from robotis_lab.assets.robots import FFW_SH5_CFG, ROBOTIS_LAB_ASSETS_DATA_DIR
from common.swerve_drive import SwerveModule, compute_swerve_commands


ENVIRONMENT_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="max",
    restitution_combine_mode="min",
    static_friction=2.0,
    dynamic_friction=1.8,
    restitution=0.0,
)


@clone
def spawn_environment_with_friction(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the environment USD and bind a high-friction material to its collision geometry."""
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    material_path = f"{prim_path}/environmentPhysicsMaterial"
    ENVIRONMENT_PHYSICS_MATERIAL.func(material_path, ENVIRONMENT_PHYSICS_MATERIAL)
    bind_physics_material(prim_path, material_path)

    return prim


def _default_sh5_usd_path() -> str:
    return FFW_SH5_CFG.spawn.usd_path


def _default_environment_usd_path() -> str:
    return f"{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/table2.usd"


def _trajectory_qos() -> Qos:
    return Qos(
        Policy.Reliability.BestEffort,
        Policy.Durability.Volatile,
        Policy.History.KeepLast(10),
    )


def _parse_float_list(value: str, expected_len: int, name: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated float list: {value}") from exc
    if len(parsed) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {len(parsed)}")
    return parsed


@configclass
class SH5BringupSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    environment: AssetBaseCfg = None
    robot: ArticulationCfg = None


class SH5DdsBridge:
    def __init__(
        self,
        robot,
        topic_manager: TopicManager,
        topic_names: dict[str, str],
        joint_states_topic: str,
        tf_topic: str,
        base_frame: str,
        trajectory_qos: Qos,
        cmd_vel_topic: str | None,
        swerve_modules: list[SwerveModule],
        wheel_radius: float,
        cmd_vel_timeout: float,
    ):
        self.robot = robot
        self.base_frame = base_frame
        self.swerve_modules = swerve_modules
        self.wheel_radius = wheel_radius
        self.cmd_vel_timeout = cmd_vel_timeout
        self.running = True
        self.lock = threading.Lock()
        self.pending_positions: dict[str, float] = {}
        self.latest_cmd_vel = (0.0, 0.0, 0.0)
        self.last_cmd_vel_time = 0.0
        self.unknown_joints: set[str] = set()
        self._warned_missing_base_frame = False
        self._warned_missing_swerve_joints: set[str] = set()
        self.readers = []
        self.threads = []
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name=joint_states_topic,
            topic_type=JointState_,
        )
        self.tf_writer = topic_manager.topic_writer(
            topic_name=tf_topic,
            topic_type=TFMessage_,
        )

        for label, topic_name in topic_names.items():
            if not topic_name:
                continue
            reader = topic_manager.topic_reader(topic_name=topic_name, topic_type=JointTrajectory_, qos=trajectory_qos)
            thread = threading.Thread(
                target=self._trajectory_loop,
                args=(label, reader),
                daemon=True,
            )
            self.readers.append(reader)
            self.threads.append(thread)
            thread.start()
            print(f"[DDS] Subscribing {label}: {topic_name}")

        if cmd_vel_topic:
            cmd_vel_reader = topic_manager.topic_reader(
                topic_name=cmd_vel_topic,
                topic_type=Twist_,
                qos=trajectory_qos,
            )
            cmd_vel_thread = threading.Thread(target=self._cmd_vel_loop, args=(cmd_vel_reader,), daemon=True)
            self.readers.append(cmd_vel_reader)
            self.threads.append(cmd_vel_thread)
            cmd_vel_thread.start()
            print(f"[DDS] Subscribing cmd_vel: {cmd_vel_topic}")

    def _trajectory_loop(self, label: str, reader):
        try:
            while self.running:
                for msg in reader.take_iter():
                    self._store_trajectory(label, msg)
                time.sleep(0.001)
        except Exception as exc:
            print(f"[DDS] {label} subscriber exception: {exc}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    def _cmd_vel_loop(self, reader):
        try:
            while self.running:
                for msg in reader.take_iter():
                    self._store_cmd_vel(msg)
                time.sleep(0.001)
        except Exception as exc:
            print(f"[DDS] cmd_vel subscriber exception: {exc}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    def _store_trajectory(self, label: str, msg):
        if msg is None or not msg.points:
            return

        point = msg.points[-1]
        joint_names = list(msg.joint_names)
        positions = list(point.positions)

        if label == "lift":
            lift_position = None
            if LIFT_JOINT_NAME in joint_names:
                lift_position = 0.5 * positions[joint_names.index(LIFT_JOINT_NAME)]  # for experiment sit
            elif len(positions) == 1:
                lift_position = positions[0]
            if lift_position is None:
                print(f"[DDS] Ignoring lift message: '{LIFT_JOINT_NAME}' not found in joint_names={joint_names}")
                return
            joint_names = [LIFT_JOINT_NAME]
            positions = [lift_position]

        if len(joint_names) != len(positions):
            print(
                f"[DDS] Ignoring {label} message: joint_names={len(joint_names)} "
                f"positions={len(positions)}"
            )
            return

        with self.lock:
            self.pending_positions.update(dict(zip(joint_names, positions)))

    def _store_cmd_vel(self, msg):
        if msg is None:
            return
        with self.lock:
            self.latest_cmd_vel = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            self.last_cmd_vel_time = time.time()

    def _current_cmd_vel(self) -> tuple[float, float, float]:
        with self.lock:
            command = self.latest_cmd_vel
            last_msg_time = self.last_cmd_vel_time

        if last_msg_time == 0.0:
            return 0.0, 0.0, 0.0
        if self.cmd_vel_timeout > 0.0 and time.time() - last_msg_time > self.cmd_vel_timeout:
            return 0.0, 0.0, 0.0
        return command

    def apply_latest_targets(self):
        with self.lock:
            commands = dict(self.pending_positions)

        joint_names = self.robot.data.joint_names
        position_target = self.robot.data.joint_pos_target.clone()
        velocity_target = self.robot.data.joint_vel_target.clone()

        for name, position in commands.items():
            if name not in joint_names:
                if name not in self.unknown_joints:
                    self.unknown_joints.add(name)
                    print(f"[DDS] Joint '{name}' is not in the SH5 USD articulation; ignoring it.")
                continue
            joint_id = joint_names.index(name)
            position_target[:, joint_id] = float(position)

        self._apply_swerve_targets(joint_names, position_target, velocity_target)

        self.robot.set_joint_position_target(position_target)
        self.robot.set_joint_velocity_target(velocity_target)

    def _apply_swerve_targets(self, joint_names: list[str], position_target, velocity_target):
        if not self.swerve_modules:
            return

        missing_joints = [
            joint_name
            for module in self.swerve_modules
            for joint_name in (module.steering_joint, module.wheel_joint)
            if joint_name not in joint_names
        ]
        for joint_name in missing_joints:
            if joint_name not in self._warned_missing_swerve_joints:
                self._warned_missing_swerve_joints.add(joint_name)
                print(f"[DDS] Swerve joint '{joint_name}' is not in the SH5 USD articulation; ignoring cmd_vel.")
        if missing_joints:
            return

        steering_joint_ids = [joint_names.index(module.steering_joint) for module in self.swerve_modules]
        current_steering = [
            float(value)
            for value in self.robot.data.joint_pos[0, steering_joint_ids].detach().cpu().tolist()
        ]
        linear_x, linear_y, angular_z = self._current_cmd_vel()
        module_commands = compute_swerve_commands(
            linear_x,
            linear_y,
            angular_z,
            self.swerve_modules,
            self.wheel_radius,
            current_steering_positions=current_steering,
            optimize_steering=True,
        )

        for module_command in module_commands:
            steering_id = joint_names.index(module_command.steering_joint)
            wheel_id = joint_names.index(module_command.wheel_joint)
            position_target[:, steering_id] = module_command.steering_position
            velocity_target[:, wheel_id] = module_command.wheel_velocity

    def publish_joint_states(self):
        now = time.time()
        stamp = Time_(sec=int(now), nanosec=int((now - int(now)) * 1_000_000_000))
        header = Header_(stamp=stamp, frame_id="base_link")

        joint_names = list(self.robot.data.joint_names)
        positions = self.robot.data.joint_pos.squeeze(0).detach().cpu().tolist()
        velocities = self.robot.data.joint_vel.squeeze(0).detach().cpu().tolist()
        efforts = [0.0] * len(joint_names)

        msg = JointState_(
            header=header,
            name=joint_names,
            position=positions,
            velocity=velocities,
            effort=efforts,
        )
        try:
            self.joint_state_writer.write(msg)
        except Exception as exc:
            print(f"[DDS] joint_states write error: {exc}")

    def publish_tf(self):
        body_names = list(self.robot.data.body_names)
        if self.base_frame not in body_names:
            if not self._warned_missing_base_frame:
                self._warned_missing_base_frame = True
                print(
                    f"[DDS] Cannot publish TF: base frame '{self.base_frame}' is not in SH5 body names. "
                    f"Available bodies: {body_names}"
                )
            return

        now = time.time()
        stamp = Time_(sec=int(now), nanosec=int((now - int(now)) * 1_000_000_000))
        base_id = body_names.index(self.base_frame)
        body_pose_w = self.robot.data.body_link_state_w[0, :, :7]
        base_pose_w = body_pose_w[base_id]
        base_pos_w = base_pose_w[:3].unsqueeze(0)
        base_quat_w = base_pose_w[3:7].unsqueeze(0)

        transforms = []
        for body_id, child_frame in enumerate(body_names):
            if child_frame == self.base_frame:
                continue

            child_pose_w = body_pose_w[body_id]
            child_pos_b, child_quat_b = math_utils.subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                child_pose_w[:3].unsqueeze(0),
                child_pose_w[3:7].unsqueeze(0),
            )
            pos = child_pos_b.squeeze(0).detach().cpu().tolist()
            quat_wxyz = child_quat_b.squeeze(0).detach().cpu().tolist()

            transforms.append(
                TransformStamped_(
                    header=Header_(stamp=stamp, frame_id=self.base_frame),
                    child_frame_id=child_frame,
                    transform=Transform_(
                        translation=Vector3_(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                        rotation=Quaternion_(
                            x=float(quat_wxyz[1]),
                            y=float(quat_wxyz[2]),
                            z=float(quat_wxyz[3]),
                            w=float(quat_wxyz[0]),
                        ),
                    ),
                )
            )

        try:
            self.tf_writer.write(TFMessage_(transforms=transforms))
        except Exception as exc:
            print(f"[DDS] tf write error: {exc}")

    def shutdown(self):
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
        for reader in self.readers:
            try:
                reader.Close()
            except Exception:
                pass
        try:
            self.joint_state_writer.Close()
        except Exception:
            pass
        try:
            self.tf_writer.Close()
        except Exception:
            pass


def _enabled_topics() -> dict[str, str]:
    topics = {
        "right_arm": RIGHT_ARM_TOPIC,
        "right_hand": RIGHT_HAND_TOPIC,
        "left_arm": LEFT_ARM_TOPIC,
        "left_hand": LEFT_HAND_TOPIC,
    }
    if not args_cli.disable_head:
        topics["head"] = HEAD_TOPIC
    if not args_cli.disable_lift:
        topics["lift"] = args_cli.lift_topic
    return topics


def _swerve_modules_from_args() -> list[SwerveModule]:
    module_count = len(SWERVE_STEERING_JOINTS)
    x_offsets = _parse_float_list(args_cli.swerve_module_x_offsets, module_count, "--swerve_module_x_offsets")
    y_offsets = _parse_float_list(args_cli.swerve_module_y_offsets, module_count, "--swerve_module_y_offsets")
    angle_offsets = _parse_float_list(
        args_cli.swerve_module_angle_offsets,
        module_count,
        "--swerve_module_angle_offsets",
    )
    return [
        SwerveModule(
            steering_joint=steering_joint,
            wheel_joint=wheel_joint,
            x_offset=x_offsets[index],
            y_offset=y_offsets[index],
            angle_offset=angle_offsets[index],
        )
        for index, (steering_joint, wheel_joint) in enumerate(zip(SWERVE_STEERING_JOINTS, SWERVE_WHEEL_JOINTS))
    ]


def _print_joint_groups(joint_names: Iterable[str]):
    names = list(joint_names)
    print("[INFO] SH5 articulation joints:")
    for name in names:
        print(f"  - {name}")


def _write_default_joint_state(robot):
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)


def _find_child_prim_by_name(stage, root_path: str, prim_name: str):
    root_path = root_path.rstrip("/")
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_path) and prim.GetName() == prim_name:
            return prim
    return None


def _find_camera_prim_by_name(stage, prim_name: str):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name and prim.GetTypeName() == "Camera":
            return prim
    return None


def _create_camera_prim(camera_path: str, translation=None, orientation=None):
    import isaacsim.core.utils.prims as prim_utils

    if not prim_utils.is_prim_path_valid(camera_path):
        prim = prim_utils.create_prim(camera_path, "Camera", translation=translation, orientation=orientation)
    else:
        prim = prim_utils.get_prim_at_path(camera_path)

    prim.GetAttribute("focalLength").Set(18.0)
    prim.GetAttribute("horizontalAperture").Set(20.955)
    clipping_attr = prim.GetAttribute("clippingRange")
    if clipping_attr and clipping_attr.IsValid():
        clipping_attr.Set((0.05, 100.0))

    _ensure_camera_viewport_attrs(prim)
    return prim


def _ensure_camera_viewport_attrs(camera_prim):
    from pxr import Gf, Sdf

    coi_attr = camera_prim.GetProperty("omni:kit:centerOfInterest")
    if not coi_attr or not coi_attr.IsValid():
        coi_attr = camera_prim.CreateAttribute(
            "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
        )
    if coi_attr.Get() is None:
        coi_attr.Set(Gf.Vec3d(0.0, 0.0, -10.0))


def _position_window(window, width: int, height: int, x: int | None = None, y: int | None = None):
    for attr_name, value in (("width", width), ("height", height), ("position_x", x), ("position_y", y)):
        if value is None:
            continue
        try:
            setattr(window, attr_name, value)
        except Exception:
            pass
        try:
            frame = getattr(window, "frame", None)
            if frame is not None:
                setattr(frame, attr_name, value)
        except Exception:
            pass


def _set_viewport_camera(
    window_name: str,
    camera_path: str,
    width: int = 640,
    height: int = 480,
    x: int | None = None,
    y: int | None = None,
):
    try:
        from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name
        from pxr import Sdf

        viewport = get_viewport_from_window_name(window_name)
        if viewport is None:
            window = create_viewport_window(
                window_name,
                width=width,
                height=height,
                position_x=0 if x is None else x,
                position_y=0 if y is None else y,
                camera_path=Sdf.Path(camera_path),
            )
            CAMERA_VIEW_WINDOWS.append(window)
            _position_window(window, width, height, x, y)
            viewport = get_viewport_from_window_name(window_name)
        if viewport is not None:
            viewport.set_active_camera(camera_path)
            return True
    except Exception as exc:
        print(f"[WARN] Could not create viewport '{window_name}': {exc}")
    return False


def _setup_camera_views(scene: InteractiveScene):
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()

    camera_specs = (
        ("Center Camera", args_cli.camera_center_name, 520, 330, 50, 20),
        ("Left Camera", args_cli.camera_left_name, 258, 200, 50, 350),
        ("Right Camera", args_cli.camera_right_name, 258, 200, 312, 350),
    )
    camera_paths: dict[str, str] = {}
    missing_camera_names: list[str] = []

    for window_name, camera_name, width, height, x, y in camera_specs:
        camera_prim = _find_camera_prim_by_name(stage, camera_name)
        if camera_prim is None:
            missing_camera_names.append(camera_name)
            continue
        _ensure_camera_viewport_attrs(camera_prim)
        camera_path = str(camera_prim.GetPath())
        camera_paths[camera_name] = camera_path
        _set_viewport_camera(window_name, camera_path, width=width, height=height, x=x, y=y)

    print("[INFO] Main Isaac Sim viewport left unchanged for overview/manual view.")
    for camera_name, camera_path in camera_paths.items():
        print(f"[INFO] {camera_name}: {camera_path}")
    if missing_camera_names:
        available_cameras = [
            str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "Camera"
        ]
        print(f"[WARN] Missing requested camera prims: {missing_camera_names}")
        print(f"[WARN] Available cameras: {available_cameras}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, bridge: SH5DdsBridge):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    step_period = 1.0 / STEP_HZ if STEP_HZ > 0 else 0.0
    publish_period = 1.0 / PUBLISH_HZ if PUBLISH_HZ > 0 else 0.0
    last_publish = 0.0
    last_step = time.time()

    while simulation_app.is_running():
        bridge.apply_latest_targets()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        now = time.time()
        if publish_period == 0.0 or now - last_publish >= publish_period:
            bridge.publish_joint_states()
            bridge.publish_tf()
            last_publish = now

        if step_period > 0.0:
            next_step = last_step + step_period
            sleep_time = next_step - time.time()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            last_step = next_step if sleep_time > 0.0 else time.time()


def main():
    usd_path = _default_sh5_usd_path()
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"SH5 USD not found: {usd_path}")

    environment_usd_path = args_cli.environment_usd or _default_environment_usd_path()
    if not args_cli.disable_environment and not os.path.exists(environment_usd_path):
        raise FileNotFoundError(f"Environment USD not found: {environment_usd_path}")

    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / STEP_HZ,
        render_interval=RENDER_INTERVAL,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, -2.2, 1.8], [0.0, 0.0, 0.8])

    scene_cfg = SH5BringupSceneCfg(num_envs=1, env_spacing=2.0)
    if not args_cli.disable_environment:
        scene_cfg.environment = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Environment",
            spawn=sim_utils.UsdFileCfg(
                func=spawn_environment_with_friction,
                usd_path=environment_usd_path,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.003,
                    rest_offset=0.0,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=_parse_float_list(args_cli.environment_pos, 3, "--environment_pos"),
                rot=_parse_float_list(args_cli.environment_rot, 4, "--environment_rot"),
            ),
        )
    robot_cfg = deepcopy(FFW_SH5_CFG)
    robot_cfg.spawn.usd_path = usd_path
    robot_cfg.spawn.rigid_props.disable_gravity = not args_cli.enable_gravity
    robot_cfg.spawn.rigid_props.linear_damping = args_cli.base_linear_damping
    robot_cfg.spawn.rigid_props.angular_damping = args_cli.base_angular_damping
    robot_cfg.articulation_root_prim_path = ARTICULATION_ROOT_PRIM_PATH
    robot_cfg.init_state.pos = ROBOT_POS
    scene_cfg.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    _write_default_joint_state(robot)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    _print_joint_groups(robot.data.joint_names)
    if args_cli.enable_camera_views:
        _setup_camera_views(scene)

    domain_id = args_cli.domain_id if args_cli.domain_id is not None else int(os.getenv("ROS_DOMAIN_ID", 0))
    topic_manager = TopicManager(domain_id=domain_id)
    bridge = SH5DdsBridge(
        robot=robot,
        topic_manager=topic_manager,
        topic_names=_enabled_topics(),
        joint_states_topic=JOINT_STATES_TOPIC,
        tf_topic=TF_TOPIC,
        base_frame=BASE_FRAME,
        trajectory_qos=_trajectory_qos(),
        cmd_vel_topic=None if args_cli.disable_cmd_vel else args_cli.cmd_vel_topic,
        swerve_modules=[] if args_cli.disable_cmd_vel else _swerve_modules_from_args(),
        wheel_radius=args_cli.wheel_radius,
        cmd_vel_timeout=args_cli.cmd_vel_timeout,
    )

    print(f"[INFO] FFW SH5 DDS bringup ready. ROS_DOMAIN_ID={domain_id}")
    if not args_cli.disable_environment:
        print(f"[INFO] Environment USD: {environment_usd_path}")
    print("[DDS] JointTrajectory subscriber reliability: best_effort")
    print(f"[DDS] Publishing joint states: {JOINT_STATES_TOPIC}")
    print(f"[DDS] Publishing TF: {TF_TOPIC} ({BASE_FRAME} -> robot links)")
    if not args_cli.disable_cmd_vel:
        print(f"[DDS] Applying swerve cmd_vel: {args_cli.cmd_vel_topic}")

    try:
        run_simulator(sim, scene, bridge)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
