# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import threading
import time

import torch
from cyclo_lab.assets.sensors.ffw_sh5_cameras import FFW_SH5_CAMERA_IMAGE_ROTATIONS
from cyclo_lab.robot_specs.ffw import sh5 as sh5_cfg
from cyclo_lab.robot_specs.ffw.mobile_base.odometry import SwerveOdometry, yaw_to_quaternion
from cyclo_lab.robot_specs.ffw.mobile_base.swerve_drive import SwerveDriveController, SwerveModule
from cyclo_lab.runtime.publishers.articulation_state_publisher import ArticulationStatePublisher
from cyclo_lab.runtime.publishers.camera_publishers import CompressedCameraPublishers
from cyclo_lab.runtime.transport.ros2_zenoh import (
    EMPTY,
    JOINT_TRAJECTORY,
    ODOMETRY,
    TWIST,
    best_effort_qos,
    close_endpoints,
    create_publisher,
    create_subscriber,
    make_odometry_kwargs,
    now_time_msg,
    ros_domain_id,
)

DEFAULT_CMD_VEL_TIMEOUT_SECONDS = 0.1


def _swerve_modules() -> list[SwerveModule]:
    return [
        SwerveModule(
            steering_joint=steering_joint,
            wheel_joint=wheel_joint,
            x_offset=sh5_cfg.SH5_SWERVE_MODULE_X_OFFSETS[index],
            y_offset=sh5_cfg.SH5_SWERVE_MODULE_Y_OFFSETS[index],
            angle_offset=sh5_cfg.SH5_SWERVE_MODULE_ANGLE_OFFSETS[index],
            steering_limit_lower=sh5_cfg.FFW_SH5_SWERVE_STEERING_LIMIT_LOWER,
            steering_limit_upper=sh5_cfg.FFW_SH5_SWERVE_STEERING_LIMIT_UPPER,
            wheel_speed_limit_lower=sh5_cfg.FFW_SH5_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
            wheel_speed_limit_upper=sh5_cfg.FFW_SH5_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
        )
        for index, (steering_joint, wheel_joint) in enumerate(
            zip(sh5_cfg.SH5_SWERVE_STEERING_JOINTS, sh5_cfg.SH5_SWERVE_WHEEL_JOINTS)
        )
    ]


class FFWSH5TopicBridge:
    """Apply SH5 Zenoh ROS2 commands and publish simulated robot state."""

    requires_activation = False

    def __init__(
        self,
        env,
        *,
        disable_head: bool = False,
        disable_lift: bool = False,
        disable_cmd_vel: bool = False,
        subscribe_reset: bool = True,
        camera_publish_hz: float | None = None,
        cmd_vel_timeout: float = DEFAULT_CMD_VEL_TIMEOUT_SECONDS,
    ) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.base_frame = sh5_cfg.BASE_FRAME
        self.odom_frame = sh5_cfg.ODOM_FRAME
        self._reset_requested = threading.Event()
        self._closed = False
        self._zero_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
            dtype=torch.float32,
        )

        topic_names = {
            label: sh5_cfg.FFW_SH5_ACTION_TOPICS[label]
            for label in ("right_arm", "right_hand", "left_arm", "left_hand")
        }
        if not disable_head:
            topic_names["head"] = sh5_cfg.FFW_SH5_ACTION_TOPICS["head"]
        if not disable_lift:
            topic_names["lift"] = sh5_cfg.FFW_SH5_ACTION_TOPICS["lift"]
        trajectory_qos = best_effort_qos(10)
        cmd_vel_topic = None if disable_cmd_vel else sh5_cfg.CMD_VEL_TOPIC
        swerve_modules = [] if disable_cmd_vel else _swerve_modules()
        wheel_radius = sh5_cfg.SH5_SWERVE_WHEEL_RADIUS

        self.swerve_modules = swerve_modules
        self.cmd_vel_timeout = cmd_vel_timeout
        self.swerve_controller = SwerveDriveController(swerve_modules, wheel_radius) if swerve_modules else None
        self.odometry = (
            SwerveOdometry(
                [module.x_offset for module in swerve_modules],
                [module.y_offset for module in swerve_modules],
                wheel_radius,
            )
            if swerve_modules
            else None
        )
        self._last_swerve_update_time = time.monotonic()
        self.lock = threading.Lock()
        self.pending_positions: dict[str, float] = {}
        self.latest_cmd_vel = (0.0, 0.0, 0.0)
        self.last_cmd_vel_time = 0.0
        self.unknown_joints: set[str] = set()
        self._warned_missing_swerve_joints: set[str] = set()
        self._joint_name_to_index = {name: index for index, name in enumerate(self.robot.data.joint_names)}
        self._missing_swerve_joints = [
            joint_name
            for module in self.swerve_modules
            for joint_name in (module.steering_joint, module.wheel_joint)
            if joint_name not in self._joint_name_to_index
        ]
        self._swerve_steering_joint_ids = [
            self._joint_name_to_index[module.steering_joint]
            for module in self.swerve_modules
            if module.steering_joint in self._joint_name_to_index
        ]
        self._swerve_wheel_joint_ids = [
            self._joint_name_to_index[module.wheel_joint]
            for module in self.swerve_modules
            if module.wheel_joint in self._joint_name_to_index
        ]
        self.subscribers = []
        self.publishers = []
        self.state_publisher = ArticulationStatePublisher(
            self.robot,
            joint_names=self.robot.data.joint_names,
            joint_states_topic=sh5_cfg.JOINT_STATES_TOPIC,
            base_frame=self.base_frame,
            base_body=self.base_frame,
            tf_topic=sh5_cfg.TF_TOPIC,
        )
        self.odom_writer = create_publisher(sh5_cfg.ODOM_TOPIC, ODOMETRY)
        self.publishers.extend([*self.state_publisher.publishers, self.odom_writer])
        self.camera_publishers = CompressedCameraPublishers(
            env.scene,
            sh5_cfg.FFW_SH5_CAMERA_TOPICS,
            camera_publish_hz,
            image_rotations=FFW_SH5_CAMERA_IMAGE_ROTATIONS,
        )
        self.publishers.extend(self.camera_publishers.endpoints)

        for label, topic_name in topic_names.items():
            if not topic_name:
                continue
            subscriber = create_subscriber(
                topic=topic_name,
                msg_type=JOINT_TRAJECTORY,
                callback=lambda msg, label=label: self._store_trajectory(label, msg),
                qos=trajectory_qos,
            )
            self.subscribers.append(subscriber)
            print(f"[Zenoh ROS2] Subscribing {label}: {topic_name}")

        if cmd_vel_topic:
            cmd_vel_subscriber = create_subscriber(
                topic=cmd_vel_topic,
                msg_type=TWIST,
                callback=self._store_cmd_vel,
                qos=trajectory_qos,
            )
            self.subscribers.append(cmd_vel_subscriber)
            print(f"[Zenoh ROS2] Subscribing cmd_vel: {cmd_vel_topic}")

        if subscribe_reset:
            self.subscribers.append(
                create_subscriber(
                    topic=sh5_cfg.SIMULATION_RESET_TOPIC,
                    msg_type=EMPTY,
                    callback=self._on_reset,
                )
            )

        print(f"[Zenoh ROS2] FFW-SH5 topic bridge ready. ROS_DOMAIN_ID={ros_domain_id()}")

    def _on_reset(self, _msg) -> None:
        self._reset_requested.set()

    def consume_reset_request(self) -> bool:
        if not self._reset_requested.is_set():
            return False
        self._reset_requested.clear()
        return True

    def get_action(self) -> torch.Tensor:
        self.apply_latest_targets()
        return self._zero_action

    def publish_observations(self) -> None:
        stamp = now_time_msg()
        self.update_odometry(float(self.env.step_dt))
        self.state_publisher.publish_joint_states(stamp=stamp)
        self.publish_odometry(stamp=stamp)
        self.state_publisher.publish_tf(stamp=stamp)
        self.camera_publishers.publish()

    # Parse trajectory topics and match joints
    def _store_trajectory(self, label: str, msg):
        if msg is None or not msg.points:
            return

        point = msg.points[-1]
        joint_names = list(msg.joint_names)
        positions = list(point.positions)

        if label == "lift":
            lift_position = None
            if sh5_cfg.FFW_SH5_LIFT_JOINT_NAME in joint_names:
                lift_position = (
                    sh5_cfg.FFW_SH5_LIFT_POSITION_SCALE * positions[joint_names.index(sh5_cfg.FFW_SH5_LIFT_JOINT_NAME)]
                )
            elif len(positions) == 1:
                lift_position = sh5_cfg.FFW_SH5_LIFT_POSITION_SCALE * positions[0]
            if lift_position is None:
                print(
                    f"[Zenoh ROS2] Ignoring lift message: '{sh5_cfg.FFW_SH5_LIFT_JOINT_NAME}' "
                    f"not found in joint_names={joint_names}"
                )
                return
            joint_names = [sh5_cfg.FFW_SH5_LIFT_JOINT_NAME]
            positions = [lift_position]

        if len(joint_names) != len(positions):
            print(f"[Zenoh ROS2] Ignoring {label} message: joint_names={len(joint_names)} positions={len(positions)}")
            return

        with self.lock:
            self.pending_positions.update(dict(zip(joint_names, positions)))

    # Apply swerve drive mobile base command
    def _store_cmd_vel(self, msg):
        if msg is None:
            return
        with self.lock:
            self.latest_cmd_vel = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            self.last_cmd_vel_time = time.monotonic()

    def _current_cmd_vel(self) -> tuple[float, float, float]:
        with self.lock:
            command = self.latest_cmd_vel
            last_msg_time = self.last_cmd_vel_time

        if last_msg_time == 0.0:
            return 0.0, 0.0, 0.0
        if self.cmd_vel_timeout > 0.0 and time.monotonic() - last_msg_time > self.cmd_vel_timeout:
            return 0.0, 0.0, 0.0
        return command

    def apply_latest_targets(self):
        with self.lock:
            commands = dict(self.pending_positions)

        position_target = self.robot.data.joint_pos_target.clone()
        velocity_target = self.robot.data.joint_vel_target.clone()

        for name, position in commands.items():
            joint_id = self._joint_name_to_index.get(name)
            if joint_id is None:
                if name not in self.unknown_joints:
                    self.unknown_joints.add(name)
                    print(f"[Zenoh ROS2] Joint '{name}' is not in the SH5 USD articulation; ignoring it.")
                continue
            position_target[:, joint_id] = float(position)

        self._apply_swerve_targets(position_target, velocity_target)

        self.robot.set_joint_position_target(position_target)
        self.robot.set_joint_velocity_target(velocity_target)

    def _apply_swerve_targets(self, position_target, velocity_target):
        if not self.swerve_modules:
            return

        for joint_name in self._missing_swerve_joints:
            if joint_name not in self._warned_missing_swerve_joints:
                self._warned_missing_swerve_joints.add(joint_name)
                print(f"[Zenoh ROS2] Swerve joint '{joint_name}' is not in the SH5 USD articulation; ignoring cmd_vel.")
        if self._missing_swerve_joints:
            return

        current_steering = [
            float(value)
            for value in self.robot.data.joint_pos[0, self._swerve_steering_joint_ids].detach().cpu().tolist()
        ]
        current_wheel_velocities = [
            float(value) for value in self.robot.data.joint_vel[0, self._swerve_wheel_joint_ids].detach().cpu().tolist()
        ]
        linear_x, linear_y, angular_z = self._current_cmd_vel()
        now = time.monotonic()
        dt = now - self._last_swerve_update_time
        self._last_swerve_update_time = now

        if self.swerve_controller is None:
            return
        module_commands = self.swerve_controller.compute_commands(
            linear_x,
            linear_y,
            angular_z,
            current_steering_positions=current_steering,
            current_wheel_velocities=current_wheel_velocities,
            dt=dt,
        )
        for module_command, steering_id, wheel_id in zip(
            module_commands,
            self._swerve_steering_joint_ids,
            self._swerve_wheel_joint_ids,
        ):
            position_target[:, steering_id] = module_command.steering_position
            velocity_target[:, wheel_id] = module_command.wheel_velocity

    def update_odometry(self, dt: float):
        if self.odometry is None or not self.swerve_modules or self._missing_swerve_joints:
            return

        steering_positions = [
            float(value) + module.angle_offset
            for value, module in zip(
                self.robot.data.joint_pos[0, self._swerve_steering_joint_ids].detach().cpu().tolist(),
                self.swerve_modules,
            )
        ]
        wheel_velocities = [
            float(value) for value in self.robot.data.joint_vel[0, self._swerve_wheel_joint_ids].detach().cpu().tolist()
        ]
        self.odometry.update(steering_positions, wheel_velocities, dt)

    # Publish wheel-integrated odometry.
    def publish_odometry(self, *, stamp=None):
        if self.odometry is None:
            return

        state = self.odometry.state()
        quat_x, quat_y, quat_z, quat_w = yaw_to_quaternion(state.yaw)
        covariance = [0.0] * 36
        for index in (0, 7, 14, 21, 28, 35):
            covariance[index] = 0.001

        try:
            self.odom_writer.publish(
                **make_odometry_kwargs(
                    frame_id=self.odom_frame,
                    child_frame_id=self.base_frame,
                    position_xyz=(state.x, state.y, 0.0),
                    orientation_xyzw=(quat_x, quat_y, quat_z, quat_w),
                    linear_xyz=(state.vx, state.vy, 0.0),
                    angular_xyz=(0.0, 0.0, state.wz),
                    covariance=covariance,
                    stamp=stamp,
                )
            )
        except Exception as exc:
            print(f"[Zenoh ROS2] odom publish error: {exc}")

    def clear_command_cache(self):
        """Discard received targets and reset base controller state."""
        with self.lock:
            self.pending_positions.clear()
            self.latest_cmd_vel = (0.0, 0.0, 0.0)
            self.last_cmd_vel_time = 0.0
        self._last_swerve_update_time = time.monotonic()
        if self.swerve_controller is not None:
            self.swerve_controller.reset()
        if self.odometry is not None:
            self.odometry.reset()

    def reset(self) -> None:
        self._reset_requested.clear()
        self.clear_command_cache()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.camera_publishers.close()
        close_endpoints(self.subscribers)
        close_endpoints(self.publishers)
