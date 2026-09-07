#!/usr/bin/env python3

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

import argparse
import threading
import time

import numpy as np
from cyclo_lab.runtime.policy_executor import PolicyExecutor
from cyclo_lab.runtime.transport.ros2_zenoh import (
    JOINT_STATE,
    JOINT_TRAJECTORY,
    TF_MESSAGE,
    close_endpoints,
    create_publisher,
    create_subscriber,
    make_joint_trajectory_kwargs,
    make_tf_message_kwargs,
    ros_domain_id,
    transform_stamped_msg,
)
from reach_env_cfg import ReachEnvConfig


class OMYReachPolicy(PolicyExecutor):
    """Zenoh ROS2 policy executor for executing a reach policy on the OMY robot."""

    def __init__(self, model_dir: str):
        super().__init__()

        self.cfg = ReachEnvConfig(model_dir=model_dir)
        self.load_policy_model(self.cfg.policy_model_path)
        self.load_policy_yaml(self.cfg.policy_env_path)

        self.domain_id = ros_domain_id()
        self.running = True
        self.iteration = 0
        self.has_joint_data = False
        self.lock = threading.Lock()

        self.action_scale = self.get_action_scale()
        self.joint_names = self.get_observation_joint_names()
        self.default_pos = self.get_default_joint_positions(self.joint_names)

        self.target_command = np.zeros(7)  # [x, y, z, qw, qx, qy, qz]
        self.num_joints = len(self.joint_names)
        self.previous_action = np.zeros(self.num_joints)
        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)

        self.joint_state_subscriber = create_subscriber(
            topic=self.cfg.joint_state_topic,
            msg_type=JOINT_STATE,
            callback=self._on_joint_state,
        )
        self.joint_trajectory_writer = create_publisher(
            topic=self.cfg.joint_trajectory_topic,
            msg_type=JOINT_TRAJECTORY,
        )
        self.tf_writer = create_publisher(topic="/tf", msg_type=TF_MESSAGE)

        self.subscribers = [self.joint_state_subscriber]
        self.publishers = [self.joint_trajectory_writer, self.tf_writer]

        print(f"OMYReachPolicy initialized with Zenoh ROS2. ROS_DOMAIN_ID={self.domain_id}")

    def _on_joint_state(self, msg):
        if msg is None:
            return
        with self.lock:
            name_to_index = {name: i for i, name in enumerate(msg.name)}
            for i, name in enumerate(self.joint_names):
                if name in name_to_index:
                    idx = name_to_index[name]
                    self.current_joint_positions[i] = msg.position[idx]
                    if idx < len(msg.velocity):
                        self.current_joint_velocities[i] = msg.velocity[idx]
                    else:
                        self.current_joint_velocities[i] = 0.0
                else:
                    print(f"Warning: Joint '{name}' not found in JointState message.")
            self.has_joint_data = True

    def run_control_loop(self):
        """Main control loop: sample target, compute action, and publish joint commands."""
        try:
            print("Waiting for joint state data...")
            while self.running:
                if not self.has_joint_data:
                    time.sleep(self.cfg.step_size)
                    continue

                command_interval = int(self.cfg.send_command_interval / self.cfg.step_size)
                phase = self.iteration % (2 * command_interval)

                if phase == 0:
                    with self.lock:
                        self.target_command = self.cfg.sample_random_pose()
                        self.broadcast_target_pose_tf()
                        print(f"New target command: {np.round(self.target_command, 4)}")

                if phase < command_interval:
                    joint_positions = self.default_pos
                else:
                    joint_positions = self.run_policy_step(self.target_command)
                    if len(joint_positions) != self.num_joints:
                        raise ValueError(f"Expected {self.num_joints} joint positions, got {len(joint_positions)}")

                self.joint_trajectory_writer.publish(**self.create_trajectory_command(joint_positions))

                self.iteration += 1
                time.sleep(self.cfg.step_size)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.shutdown()

    def create_trajectory_command(self, joint_positions: np.ndarray) -> dict:
        """Create JointTrajectory publish kwargs from joint positions."""
        return make_joint_trajectory_kwargs(
            joint_names=self.joint_names,
            positions=joint_positions,
            time_from_start_sec=self.cfg.trajectory_time_from_start,
        )

    def broadcast_target_pose_tf(self):
        """Publish a TF transform for the target pose."""
        transform = transform_stamped_msg(
            parent_frame="world",
            child_frame="target_pose",
            translation=(
                self.target_command[0],
                self.target_command[1],
                self.target_command[2],
            ),
            rotation_xyzw=(
                self.target_command[4],
                self.target_command[5],
                self.target_command[6],
                self.target_command[3],
            ),
        )
        self.tf_writer.publish(**make_tf_message_kwargs([transform]))

    def update_observation(self, command: np.ndarray) -> np.ndarray:
        """Build the observation vector for the policy."""
        with self.lock:
            obs = np.concatenate([
                self.current_joint_positions - self.default_pos,
                self.current_joint_velocities,
                command,
                self.previous_action,
            ]).astype(np.float32)

        return obs

    def run_policy_step(self, command: np.ndarray) -> np.ndarray:
        """Run a single policy step and return joint positions to command."""
        observation = self.update_observation(command)
        self.action = self.update_action(observation)
        self.previous_action = self.action.copy()
        joint_positions = self.default_pos + (self.action * self.action_scale)

        return joint_positions

    def shutdown(self):
        """Stop the policy executor and close Zenoh ROS2 endpoints."""
        if not self.running:
            return
        self.running = False
        close_endpoints(self.subscribers)
        close_endpoints(self.publishers)
        print("Zenoh ROS2 connections closed.")


def main(args=None):
    """Entry point to run the reach policy node over Zenoh ROS2."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Relative path to the trained policy directory under logs/rsl_rl/reach_omy/",
    )

    parsed_args = parser.parse_args(args)

    policy = OMYReachPolicy(model_dir=parsed_args.model_dir)
    policy.run_control_loop()


if __name__ == "__main__":
    main()
