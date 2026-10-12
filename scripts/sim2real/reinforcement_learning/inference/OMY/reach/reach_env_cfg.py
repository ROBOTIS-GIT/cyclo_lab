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

import math
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path


class ReachEnvConfig:
    def __init__(self, model_dir: str):
        self.step_size = 1.0 / 1000  # 1000Hz
        self.trajectory_time_from_start = 1.0 / 20.0  # seconds
        self.send_command_interval = 3.0  # seconds

        self.joint_state_topic = "/joint_states"
        self.joint_trajectory_topic = "/leader/joint_trajectory"

        repo_root = Path(__file__).resolve().parents[6]
        self.policy_model_path = repo_root / "logs/rsl_rl/reach_omy" / model_dir / "exported/policy.pt"
        self.policy_env_path = repo_root / "logs/rsl_rl/reach_omy" / model_dir / "params/env.yaml"

    def sample_random_pose(self) -> np.ndarray:
        """Return a random target pose: [x, y, z, qx, qy, qz, qw]."""
        pos = np.random.uniform([0.25, -0.2, 0.3], [0.45, 0.2, 0.45])
        roll = np.random.uniform(-math.pi / 4, math.pi / 4)
        pitch = 0.0 
        yaw = np.random.uniform(math.pi / 4, math.pi * 3 / 4)
        quat = Rotation.from_euler("zyx", [yaw, pitch, roll]).as_quat()  # [x, y, z, w]
        return np.concatenate([pos, quat])  # [x, y, z, qx, qy, qz, qw]
