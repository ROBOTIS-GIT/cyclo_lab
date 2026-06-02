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
# Author: Howon Kim

# AI Worker arm JointTrajectory input topics.
AI_WORKER_RIGHT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"
AI_WORKER_LEFT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"

# SH5 hand JointTrajectory input topics.
SH5_RIGHT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory"
SH5_LEFT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_left_hand/joint_trajectory"

# Head and lift JointTrajectory input topics.
HEAD_TOPIC = "/leader/joystick_controller_left/joint_trajectory"
LIFT_TOPIC = "/leader/joystick_controller_right/joint_trajectory"
LIFT_JOINT_NAME = "lift_joint"
LIFT_POSITION_SCALE = 0.5

# DDS output topics and base frame.
CMD_VEL_TOPIC = "/cmd_vel"
ODOM_TOPIC = "/odom"
JOINT_STATES_TOPIC = "/joint_states"
TF_TOPIC = "/tf"
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"

# Simulation timing and robot spawn pose.
PUBLISH_HZ = 30.0
STEP_HZ = 30.0
RENDER_INTERVAL = 1
ROBOT_POS = (0.0, 0.0, -0.18)

# Swerve drive joint limits and cmd_vel timeout.
AI_WORKER_SWERVE_STEERING_LIMIT_LOWER = -1.570796
AI_WORKER_SWERVE_STEERING_LIMIT_UPPER = 1.570796
AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_LOWER = -50.0
AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_UPPER = 50.0
CMD_VEL_TIMEOUT = 0.1

# Isaac Sim overview viewport placement.
ISAAC_SIM_OVERVIEW_CAMERA_EYE = (2.8, -2.2, 1.8)
ISAAC_SIM_OVERVIEW_CAMERA_TARGET = (0.0, 0.0, 0.8)

# AI Worker camera prim names.
AI_WORKER_CAMERA_CENTER_NAME = "Head_Camera"
AI_WORKER_CAMERA_LEFT_NAME = "Left_Camera"
AI_WORKER_CAMERA_RIGHT_NAME = "Right_Camera"
AI_WORKER_CAMERA_VIEW_WINDOWS = []
