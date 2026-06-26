# Copyright 2026 ROBOTIS CO., LTD.
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
# Author: Woojae Lee
#
# Additional notices:
# This module is adapted from HybridRobotics/whole_body_tracking, licensed under the MIT License.
# See THIRD_PARTY_LICENSES.md for details.

"""Reference trajectory storage for Mimic tasks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


class ReferenceTrajectory:
    """Tensor view over a reference trajectory NPZ file."""

    REQUIRED_KEYS = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )

    def __init__(self, trajectory_file: str, tracked_body_ids: Sequence[int], device: str = "cpu"):
        path = Path(trajectory_file)
        if not path.is_file():
            raise FileNotFoundError(f"Reference trajectory file does not exist: {path}")

        data = np.load(path)
        missing = [key for key in self.REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"Reference trajectory file is missing keys: {missing}")

        self.fps = data["fps"]
        self.joint_position = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_velocity = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_position_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_orientation_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_linear_velocity_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_angular_velocity_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._tracked_body_ids = tracked_body_ids
        self.num_frames = self.joint_position.shape[0]

    @property
    def body_position_w(self) -> torch.Tensor:
        return self._body_position_w[:, self._tracked_body_ids]

    @property
    def body_orientation_w(self) -> torch.Tensor:
        return self._body_orientation_w[:, self._tracked_body_ids]

    @property
    def body_linear_velocity_w(self) -> torch.Tensor:
        return self._body_linear_velocity_w[:, self._tracked_body_ids]

    @property
    def body_angular_velocity_w(self) -> torch.Tensor:
        return self._body_angular_velocity_w[:, self._tracked_body_ids]
