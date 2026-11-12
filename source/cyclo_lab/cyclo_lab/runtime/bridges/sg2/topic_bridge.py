"""Always-active FFW-SG2 bridge over ROS2-compatible Zenoh topics."""

from __future__ import annotations

import threading
import time

import torch
from cyclo_lab.robot_specs.ffw.sg2 import (
    BASE_BODY,
    BASE_FRAME,
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_ACTION_TOPICS,
    FFW_SG2_CAMERA_TOPICS,
    FFW_SG2_JOINT_POSITION_LIMITS,
    FFW_SG2_PUBLISHED_JOINT_NAMES,
    JOINT_STATES_TOPIC,
    ODOM_FRAME,
    ODOM_TOPIC,
    SIMULATION_RESET_TOPIC,
    TF_TOPIC,
)
from cyclo_lab.runtime.publishers.articulation_state_publisher import ArticulationStatePublisher
from cyclo_lab.runtime.publishers.camera_publishers import CompressedCameraPublishers
from cyclo_lab.runtime.transport.ros2_zenoh import (
    EMPTY,
    JOINT_TRAJECTORY,
    TWIST,
    close_endpoints,
    create_subscriber,
    ros_domain_id,
)

DEFAULT_CMD_VEL_TIMEOUT_SECONDS = 0.1


class FFWSG2TopicBridge:
    """Translate SG2 command topics into Isaac Lab actions and publish simulation state."""

    requires_activation = True

    def __init__(
        self,
        env,
        *,
        camera_publish_hz: float | None = None,
        publish_odometry_tf: bool = True,
        subscribe_reset: bool = False,
        cmd_vel_timeout: float = DEFAULT_CMD_VEL_TIMEOUT_SECONDS,
    ) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.domain_id = ros_domain_id()
        self.publish_odometry_tf = publish_odometry_tf
        self.cmd_vel_timeout = float(cmd_vel_timeout)
        self.joint_names = list(FFW_SG2_ACTION_JOINT_NAMES)
        self.published_joint_names = list(FFW_SG2_PUBLISHED_JOINT_NAMES)
        self.total_action_dim = env.action_manager.total_action_dim
        self.include_base_action = (
            "base_action" in env.action_manager.active_terms and self.total_action_dim == len(self.joint_names) + 3
        )
        if self.total_action_dim not in (len(self.joint_names), len(self.joint_names) + 3):
            raise ValueError(f"FFW-SG2 bridge expects 19D or 22D actions, got {self.total_action_dim}D.")

        self._lock = threading.Lock()
        self._reset_requested = threading.Event()
        self._target_joint_state: dict[str, float] | None = None
        self._trajectory_commands: dict[str, dict[str, float] | None] = {
            "left_arm": None,
            "right_arm": None,
            "head": None,
            "lift": None,
        }
        self._latest_cmd_vel = (0.0, 0.0, 0.0)
        self._last_cmd_vel_time = 0.0
        self._closed = False

        self.subscribers = [
            create_subscriber(
                topic=FFW_SG2_ACTION_TOPICS[label],
                msg_type=JOINT_TRAJECTORY,
                callback=lambda msg, command_label=label: self._on_joint_trajectory(command_label, msg),
            )
            for label in ("left_arm", "right_arm", "head", "lift")
        ]
        if self.include_base_action:
            self.subscribers.append(
                create_subscriber(
                    topic=FFW_SG2_ACTION_TOPICS["mobile"],
                    msg_type=TWIST,
                    callback=self._on_cmd_vel,
                )
            )
        if subscribe_reset:
            self.subscribers.append(
                create_subscriber(
                    topic=SIMULATION_RESET_TOPIC,
                    msg_type=EMPTY,
                    callback=self._on_reset,
                )
            )

        self.state_publisher = ArticulationStatePublisher(
            self.robot,
            joint_names=self.published_joint_names,
            joint_states_topic=JOINT_STATES_TOPIC,
            base_frame=BASE_FRAME,
            base_body=BASE_BODY,
            odom_topic=ODOM_TOPIC if publish_odometry_tf else None,
            tf_topic=TF_TOPIC if publish_odometry_tf else None,
            odom_frame=ODOM_FRAME,
        )

        self.camera_publishers = CompressedCameraPublishers(
            env.scene,
            FFW_SG2_CAMERA_TOPICS,
            camera_publish_hz,
        )
        self.publishers = [*self.state_publisher.publishers, *self.camera_publishers.endpoints]

        print(
            f"[Zenoh ROS2] FFW-SG2 topic bridge ready: {self.total_action_dim}D "
            f"({'joint+base' if self.include_base_action else 'joint-only'}), "
            f"ROS_DOMAIN_ID={self.domain_id}"
        )

    def _clamp_joint(self, name: str, value: float) -> float:
        lower, upper = FFW_SG2_JOINT_POSITION_LIMITS[name]
        return min(max(float(value), lower), upper)

    def _on_joint_trajectory(self, label: str, msg) -> None:
        if msg is None or not msg.points:
            return
        command = {
            name: self._clamp_joint(name, position)
            for name, position in zip(msg.joint_names, msg.points[-1].positions)
            if name in FFW_SG2_JOINT_POSITION_LIMITS
        }
        if not command:
            return
        with self._lock:
            cached = self._trajectory_commands[label] or {}
            cached.update(command)
            self._trajectory_commands[label] = cached

    def _on_cmd_vel(self, msg) -> None:
        if msg is None:
            return
        with self._lock:
            self._latest_cmd_vel = (
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.angular.z),
            )
            self._last_cmd_vel_time = time.monotonic()

    def _on_reset(self, _msg) -> None:
        self._reset_requested.set()

    def consume_reset_request(self) -> bool:
        """Return and clear the reset request without resetting from the callback thread."""
        if not self._reset_requested.is_set():
            return False
        self._reset_requested.clear()
        return True

    def _read_current_joint_state(self) -> dict[str, float]:
        positions = self.robot.data.joint_pos[0].detach().cpu().tolist()
        name_to_index = {name: index for index, name in enumerate(self.robot.data.joint_names)}
        return {
            name: self._clamp_joint(name, positions[name_to_index[name]])
            for name in self.joint_names
            if name in name_to_index
        }

    def _joint_targets(self) -> dict[str, float]:
        with self._lock:
            if self._target_joint_state is None:
                self._target_joint_state = self._read_current_joint_state()
            for command in self._trajectory_commands.values():
                if command:
                    self._target_joint_state.update(command)
            return dict(self._target_joint_state)

    def _current_cmd_vel(self) -> tuple[float, float, float]:
        with self._lock:
            command = self._latest_cmd_vel
            last_command_time = self._last_cmd_vel_time
        if last_command_time == 0.0:
            return 0.0, 0.0, 0.0
        if self.cmd_vel_timeout > 0.0 and time.monotonic() - last_command_time > self.cmd_vel_timeout:
            return 0.0, 0.0, 0.0
        return command

    def _make_action(
        self,
        targets: dict[str, float],
        base_command: tuple[float, float, float],
    ) -> torch.Tensor:
        values = [targets[name] for name in self.joint_names]
        if self.include_base_action:
            values.extend(base_command)
        return torch.tensor(values, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def get_action(self) -> torch.Tensor:
        """Return the latest absolute joint targets and optional base velocity."""
        return self._make_action(self._joint_targets(), self._current_cmd_vel())

    def get_hold_action(self) -> torch.Tensor:
        """Hold the current joint pose and command zero base velocity."""
        with self._lock:
            if self._target_joint_state is None:
                self._target_joint_state = self._read_current_joint_state()
            targets = dict(self._target_joint_state)
        return self._make_action(targets, (0.0, 0.0, 0.0))

    def publish_observations(self) -> None:
        self.state_publisher.publish_all()
        self.camera_publishers.publish()

    def clear_command_cache(self) -> None:
        with self._lock:
            self._target_joint_state = None
            for label in self._trajectory_commands:
                self._trajectory_commands[label] = None
            self._latest_cmd_vel = (0.0, 0.0, 0.0)
            self._last_cmd_vel_time = 0.0
        self.state_publisher.reset_odom_origin()

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
        print("[Zenoh ROS2] FFW-SG2 topic bridge closed.")
