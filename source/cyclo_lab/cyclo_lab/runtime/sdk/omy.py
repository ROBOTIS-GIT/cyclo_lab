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

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import torch
from cyclo_lab.robot_specs.omy import (
    OMY_CAMERA_TOPICS,
    OMY_JOINT_NAMES,
    OMY_JOINT_STATES_TOPIC,
    OMY_JOINT_TRAJECTORY_TOPIC,
)
from cyclo_lab.runtime.publishers.camera_publishers import publish_compressed_camera
from cyclo_lab.runtime.transport.ros2_zenoh import (
    COMPRESSED_IMAGE,
    JOINT_STATE,
    JOINT_TRAJECTORY,
    close_endpoints,
    create_publisher,
    create_subscriber,
    make_joint_state_kwargs,
    ros_domain_id,
)


class OMYSdk:
    """OMY teleoperation interface over Zenoh ROS2 topics."""

    def __init__(
        self,
        env,
        mode: str,
        camera_publish_hz: float | None = None,
        enable_keyboard_listener: bool = True,
    ):
        self.env = env
        self.mode = mode  # 'record' or 'inference'
        self.domain_id = ros_domain_id()
        self.camera_publish_hz = camera_publish_hz
        self.enable_keyboard_listener = enable_keyboard_listener
        self._last_camera_publish_time = 0.0
        self.joint_trajectory_cmd = None
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self._episode_phase = "idle"
        self.lock = threading.Lock()

        self.joint_names = list(OMY_JOINT_NAMES)
        self.exclude_joints = []

        self.subscribers = [
            create_subscriber(
                topic=OMY_JOINT_TRAJECTORY_TOPIC,
                msg_type=JOINT_TRAJECTORY,
                callback=self._on_joint_trajectory,
            )
        ]

        self.joint_state_writer = create_publisher(OMY_JOINT_STATES_TOPIC, JOINT_STATE)
        self.camera_writers = {}
        if self.camera_publish_hz is None or self.camera_publish_hz != 0.0:
            self.camera_writers = {
                cam_name: create_publisher(topic, COMPRESSED_IMAGE) for cam_name, topic in OMY_CAMERA_TOPICS.items()
            }
        self.publishers = [self.joint_state_writer, *self.camera_writers.values()]

        self.listener = None
        if self.enable_keyboard_listener:
            try:
                from pynput.keyboard import Listener

                self.listener = Listener(on_press=self._on_press)
                self.listener.start()
            except ImportError as exc:
                print(f"[Control] Keyboard listener unavailable: {exc}")

        print(f"[Zenoh ROS2] OMYSdk ready. ROS_DOMAIN_ID={self.domain_id}")
        if self.camera_publish_hz == 0.0:
            print("[Zenoh ROS2] Camera topic publishing disabled for this session.")
        elif self.camera_publish_hz is not None and self.camera_publish_hz > 0.0:
            print(f"[Zenoh ROS2] Camera topic publishing throttled to {self.camera_publish_hz:g} Hz.")
        if not self.enable_keyboard_listener:
            print("[Zenoh ROS2] Keyboard listener disabled for this session.")
        self._keyboard_controls()

    # ----------------------
    # Keyboard controls
    # ----------------------
    def _keyboard_controls(self):
        print("\n[Control] Press keys to control the robot:")
        if self.mode == "record":
            print("[N] Save successful episode and proceed to the next one")
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start recording the current episode")
            print("[Info] Robot control is always active in record mode.")
        elif self.mode == "inference":
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start/Resume robot control")

    def _on_press(self, key):
        try:
            if self.mode == "record":
                if key.char == "b":
                    self.start_recording()
                elif key.char == "r":
                    self._skip_episode()
                elif key.char == "n":
                    self._save_episode()
            elif self.mode == "inference":
                if key.char == "b":
                    self._started = True
                    self._reset_state = False
                elif key.char == "r":
                    self._started = False
                    self._reset_state = True
                    self._call_callback("R")
        except AttributeError:
            pass

    def _call_callback(self, key):
        if key in self._additional_callbacks:
            self._additional_callbacks[key]()

    def start_recording(self):
        if self._episode_phase == "recording":
            return
        print("[Control] Start recording requested.")
        self._started = True
        self._reset_state = False
        self._episode_phase = "recording"
        self._call_callback("B")

    def _save_episode(self):
        if self.mode == "record" and self._episode_phase != "recording":
            print("[Control] Save ignored because recording has not started.")
            return
        print("[Control] Save episode requested.")
        self._started = False
        self._reset_state = True
        self._episode_phase = "idle"
        self._call_callback("N")

    def _skip_episode(self):
        print("[Control] Reset/skip episode requested.")
        self._started = False
        self._reset_state = True
        self._episode_phase = "idle"
        self._call_callback("R")

    # ----------------------
    # Subscribers
    # ----------------------
    def _on_joint_trajectory(self, msg):
        if msg is None or not msg.points:
            return
        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
        with self.lock:
            self.joint_trajectory_cmd = [float(joint_dict.get(name, 0.0)) for name in self.joint_names]

    # ----------------------
    # Publishers
    # ----------------------
    def _publish_joint_states(self):
        obs_joint_name = self.env.scene["robot"].data.joint_names
        all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()
        all_velocities = self.env.scene["robot"].data.joint_vel.squeeze(0).tolist()
        all_efforts = [0.0] * len(all_positions)

        if isinstance(all_positions[0], list):
            all_positions = [p for sub in all_positions for p in sub]
        if isinstance(all_velocities[0], list):
            all_velocities = [v for sub in all_velocities for v in sub]

        indices = [obs_joint_name.index(name) for name in self.joint_names]
        positions = [all_positions[i] for i in indices]
        velocities = [all_velocities[i] for i in indices]
        efforts = [all_efforts[i] for i in indices]

        try:
            self.joint_state_writer.publish(
                **make_joint_state_kwargs(
                    names=self.joint_names,
                    positions=positions,
                    velocities=velocities,
                    efforts=efforts,
                    frame_id="base_link",
                )
            )
        except Exception as exc:
            print("[Zenoh ROS2] joint_states publish error:", exc)

    def _publish_camera(self, cam_name: str):
        writer = self.camera_writers.get(cam_name)
        if writer is None:
            return
        if cam_name not in self.env.scene.keys():
            return
        try:
            publish_compressed_camera(cam_name, self.env.scene[cam_name], writer, frame_id="camera_frame")
        except Exception as exc:
            print(f"[Zenoh ROS2] camera publish error for {cam_name}:", exc)

    # ----------------------
    # Action/state handling
    # ----------------------
    def _compute_action_state(self):
        state = {"reset": self._reset_state, "started": self._started}
        if state["reset"]:
            self._reset_state = False
            return state
        state["joint_state"] = self._get_device_state()
        return state

    def _get_device_state(self):
        with self.lock:
            obs_joint_name = self.env.scene["robot"].data.joint_names
            all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()

            if isinstance(all_positions[0], list):
                all_positions = [p for sub in all_positions for p in sub]

            joint_state = {}
            for name in self.joint_names:
                if name in obs_joint_name:
                    joint_state[name] = all_positions[obs_joint_name.index(name)]
                else:
                    joint_state[name] = 0.0

            if self.joint_trajectory_cmd:
                joint_state.update(dict(zip(self.joint_names, self.joint_trajectory_cmd)))

            return joint_state

    def get_action(self):
        action = self._compute_action_state()
        if action["reset"]:
            return {"reset": True}
        if self.mode != "record" and not action["started"]:
            return None

        joint_state = action["joint_state"]
        positions = [joint_state.get(name, 0.0) for name in self.joint_names]
        return torch.tensor(positions, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def publish_observations(self):
        self._publish_joint_states()
        if not self.camera_writers:
            return
        if self.camera_publish_hz is not None and self.camera_publish_hz > 0.0:
            now = time.monotonic()
            if now - self._last_camera_publish_time < 1.0 / self.camera_publish_hz:
                return
            self._last_camera_publish_time = now
        for cam_name in self.camera_writers:
            self._publish_camera(cam_name)

    # ----------------------
    # Utility
    # ----------------------
    def shutdown(self):
        close_endpoints(self.subscribers)
        close_endpoints(self.publishers)
        if self.listener is not None:
            try:
                self.listener.stop()
            except Exception:
                pass
        print("OMYSdk shutdown complete")

    def reset(self):
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func
