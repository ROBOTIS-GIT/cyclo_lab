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
#
# Author: Taehyeong Kim

import argparse
import os
import threading
import time
from datetime import datetime

import numpy as np

from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_, JointTrajectoryPoint_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.geometry_msgs.msg import TransformStamped_, Transform_, Vector3_, Quaternion_
from robotis_dds_python.idl.tf2_msgs.msg import TFMessage_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.builtin_interfaces.msg import Time_, Duration_

from robotis_dds_python.tools.topic_manager import TopicManager

from reach_env_cfg import ReachEnvConfig

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from inference.utils.policy_executor import PolicyExecutor


class OMYReachPolicy(PolicyExecutor):
    """DDS-based policy executor for executing a reach policy on the OMY robot."""

    def __init__(self, model_dir: str):
        super().__init__()

        self.cfg = ReachEnvConfig(model_dir=model_dir)
        self.load_policy_model(self.cfg.policy_model_path)
        self.load_policy_yaml(self.cfg.policy_env_path)

        self.domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
        self.running = True
        self.iteration = 0
        self.has_joint_data = False
        self.lock = threading.Lock()  # Protect shared state

        self.action_scale = self.get_action_scale()
        self.joint_names = self.get_observation_joint_names()
        self.default_pos = self.get_default_joint_positions(self.joint_names)

        self.target_command = np.zeros(7)  # [x, y, z, qx, qy, qz, qw]
        self.num_joints = len(self.joint_names)
        self.previous_action = np.zeros(self.num_joints)
        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)

        # DDS Topic Manager
        self.topic_manager = TopicManager(domain_id=self.domain_id)

        # Subscriber for joint states
        self.joint_state_reader = self.topic_manager.topic_reader(
            topic_name=self.cfg.joint_state_topic,
            topic_type=JointState_
        )

        # Publisher for joint trajectory
        self.joint_trajectory_writer = self.topic_manager.topic_writer(
            topic_name=self.cfg.joint_trajectory_topic,
            topic_type=JointTrajectory_
        )

        # Publisher for TF (target pose visualization)
        self.tf_writer = self.topic_manager.topic_writer(
            topic_name="/tf",
            topic_type=TFMessage_
        )

        # Start subscriber thread
        self.thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self.thread.start()

        print("OMYReachPolicy initialized with DDS.")

    def _subscriber_loop(self):
        """Continuously read joint state messages from DDS in a separate thread."""
        try:
            while self.running:
                has_message = False
                for msg in self.joint_state_reader.take_iter():
                    if msg:
                        has_message = True
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
                
                # Prevent CPU spinning when no messages are available
                if not has_message:
                    time.sleep(0.001)  # 1ms delay to reduce CPU usage
        except Exception as e:
            print("Subscriber thread exception:", e)
        finally:
            try:
                self.joint_state_reader.Close()
            except Exception:
                pass
            print("Joint state subscriber closed")

    def run_control_loop(self):
        """Main control loop: samples target, computes action, and publishes joint commands."""
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
                    joint_trajectory_msg = self.create_trajectory_command(self.default_pos)
                    self.joint_trajectory_writer.write(joint_trajectory_msg)
                else:
                    joint_positions = self.run_policy_step(self.target_command)
                    if len(joint_positions) != self.num_joints:
                        raise ValueError(f"Expected {self.num_joints} joint positions, got {len(joint_positions)}")
                    joint_trajectory_msg = self.create_trajectory_command(joint_positions)
                    self.joint_trajectory_writer.write(joint_trajectory_msg)

                self.iteration += 1
                time.sleep(self.cfg.step_size)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.shutdown()

    def create_trajectory_command(self, joint_positions: np.ndarray) -> JointTrajectory_:
        """Creates a JointTrajectory_ message from joint positions."""
        point = JointTrajectoryPoint_(
            positions=joint_positions.tolist(),
            velocities=[],
            accelerations=[],
            effort=[],
            time_from_start=Duration_(
                sec=0,
                nanosec=int(self.cfg.trajectory_time_from_start * 1e9)
            )
        )

        header = Header_(
            stamp=Time_(sec=0, nanosec=0),
            frame_id=""
        )

        msg = JointTrajectory_(
            header=header,
            joint_names=self.joint_names,
            points=[point]
        )
        return msg

    def broadcast_target_pose_tf(self):
        """Publishes a TF transform for the target pose."""
        now = datetime.now()
        stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
        header = Header_(stamp=stamp, frame_id="world")

        transform = TransformStamped_(
            header=header,
            child_frame_id="target_pose",
            transform=Transform_(
                translation=Vector3_(
                    x=self.target_command[0],
                    y=self.target_command[1],
                    z=self.target_command[2]
                ),
                rotation=Quaternion_(
                    x=self.target_command[3],
                    y=self.target_command[4],
                    z=self.target_command[5],
                    w=self.target_command[6]
                )
            )
        )

        tf_message = TFMessage_(transforms=[transform])
        self.tf_writer.write(tf_message)

    def update_observation(self, command: np.ndarray) -> np.ndarray:
        """Builds the observation vector for the policy."""
        with self.lock:
            obs = np.concatenate([
                self.current_joint_positions - self.default_pos,
                self.current_joint_velocities,
                command,
                self.previous_action,
            ]).astype(np.float32)

        return obs

    def run_policy_step(self, command: np.ndarray) -> np.ndarray:
        """Runs a single step of the policy and returns the joint positions to command."""
        observation = self.update_observation(command)
        self.action = self.update_action(observation)
        self.previous_action = self.action.copy()
        joint_positions = self.default_pos + (self.action * self.action_scale)

        return joint_positions

    def shutdown(self):
        """Gracefully shut down the policy executor by stopping threads and closing DDS connections."""
        self.running = False
        try:
            self.joint_state_reader.Close()
            self.joint_trajectory_writer.Close()
            self.tf_writer.Close()
        except:
            pass
        print("DDS connections closed.")

def main(args=None):
    """Entry point to run the reach policy node with DDS."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Relative path to the trained policy directory under logs/rsl_rl/reach_omy/"
    )

    parsed_args = parser.parse_args(args)

    policy = OMYReachPolicy(model_dir=parsed_args.model_dir)
    policy.run_control_loop()


if __name__ == '__main__':
    main()
