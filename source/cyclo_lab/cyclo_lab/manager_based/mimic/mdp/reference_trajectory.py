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

    def __init__(
        self,
        trajectory_file: str,
        runtime_joint_names: Sequence[str],
        tracked_body_names: Sequence[str],
        device: str = "cpu",
        source_joint_names: Sequence[str] | None = None,
        source_body_names: Sequence[str] | None = None,
        source_quat_order: str | None = None,
        expected_fps: float | None = None,
    ):
        path = Path(trajectory_file)
        if not path.is_file():
            raise FileNotFoundError(f"Reference trajectory file does not exist: {path}")

        data = np.load(path)
        missing = [key for key in self.REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"Reference trajectory file is missing keys: {missing}")

        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if expected_fps is not None and not np.isclose(self.fps, expected_fps):
            raise ValueError(
                f"Reference trajectory is {self.fps:g} Hz, but the policy runs at {expected_fps:g} Hz: {path}"
            )

        file_joint_names = self._resolve_names(data, "joint_names", source_joint_names, "joint")
        file_body_names = self._resolve_names(data, "body_names", source_body_names, "body")
        quat_order = self._resolve_quat_order(data, source_quat_order)

        joint_pos = np.asarray(data["joint_pos"])
        joint_vel = np.asarray(data["joint_vel"])
        body_pos = np.asarray(data["body_pos_w"])
        body_quat = np.asarray(data["body_quat_w"])
        body_lin_vel = np.asarray(data["body_lin_vel_w"])
        body_ang_vel = np.asarray(data["body_ang_vel_w"])
        self._validate_shapes(
            path,
            joint_pos,
            joint_vel,
            body_pos,
            body_quat,
            body_lin_vel,
            body_ang_vel,
            file_joint_names,
            file_body_names,
        )

        joint_ids = self._indices_by_name(file_joint_names, runtime_joint_names, "joint")
        body_ids = self._indices_by_name(file_body_names, tracked_body_names, "body")
        if quat_order == "wxyz":
            body_quat = body_quat[..., [1, 2, 3, 0]]

        self.joint_position = torch.tensor(joint_pos[:, joint_ids], dtype=torch.float32, device=device)
        self.joint_velocity = torch.tensor(joint_vel[:, joint_ids], dtype=torch.float32, device=device)
        self._body_position_w = torch.tensor(body_pos[:, body_ids], dtype=torch.float32, device=device)
        self._body_orientation_w = torch.tensor(body_quat[:, body_ids], dtype=torch.float32, device=device)
        self._body_linear_velocity_w = torch.tensor(body_lin_vel[:, body_ids], dtype=torch.float32, device=device)
        self._body_angular_velocity_w = torch.tensor(body_ang_vel[:, body_ids], dtype=torch.float32, device=device)
        self.num_frames = self.joint_position.shape[0]

    @staticmethod
    def _string_value(value: np.ndarray) -> str:
        item = np.asarray(value).reshape(-1)[0]
        return item.decode() if isinstance(item, bytes) else str(item)

    @classmethod
    def _resolve_names(
        cls,
        data: np.lib.npyio.NpzFile,
        key: str,
        fallback: Sequence[str] | None,
        kind: str,
    ) -> list[str]:
        if key in data:
            names = [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in np.asarray(data[key]).reshape(-1)
            ]
        elif fallback is not None:
            names = list(fallback)
        else:
            raise KeyError(
                f"Reference trajectory has no '{key}' metadata. Configure source_{kind}_names for this legacy NPZ."
            )
        if len(names) != len(set(names)):
            raise ValueError(f"Reference trajectory contains duplicate {kind} names: {names}")
        return names

    @classmethod
    def _resolve_quat_order(cls, data: np.lib.npyio.NpzFile, fallback: str | None) -> str:
        quat_order = cls._string_value(data["quat_order"]) if "quat_order" in data else fallback
        if quat_order not in {"xyzw", "wxyz"}:
            raise ValueError(
                "Reference trajectory quaternion order must be 'xyzw' or 'wxyz'. "
                "Add quat_order metadata or configure source_quat_order for a legacy NPZ."
            )
        return quat_order

    @staticmethod
    def _indices_by_name(source_names: Sequence[str], target_names: Sequence[str], kind: str) -> list[int]:
        source_index = {name: index for index, name in enumerate(source_names)}
        missing = [name for name in target_names if name not in source_index]
        if missing:
            raise ValueError(f"Reference trajectory is missing {kind} names: {missing}")
        return [source_index[name] for name in target_names]

    @staticmethod
    def _validate_shapes(
        path: Path,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        body_lin_vel: np.ndarray,
        body_ang_vel: np.ndarray,
        joint_names: Sequence[str],
        body_names: Sequence[str],
    ) -> None:
        if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
            raise ValueError(f"Invalid joint_pos/joint_vel shapes in {path}: {joint_pos.shape}, {joint_vel.shape}")
        if joint_pos.shape[1] != len(joint_names):
            raise ValueError(f"{path} has {joint_pos.shape[1]} joint columns but {len(joint_names)} joint names.")
        frame_count = joint_pos.shape[0]
        expected_body_shapes = {
            "body_pos_w": (frame_count, len(body_names), 3),
            "body_quat_w": (frame_count, len(body_names), 4),
            "body_lin_vel_w": (frame_count, len(body_names), 3),
            "body_ang_vel_w": (frame_count, len(body_names), 3),
        }
        body_arrays = {
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
            "body_lin_vel_w": body_lin_vel,
            "body_ang_vel_w": body_ang_vel,
        }
        for name, expected_shape in expected_body_shapes.items():
            if body_arrays[name].shape != expected_shape:
                raise ValueError(
                    f"Invalid {name} shape in {path}: {body_arrays[name].shape}, expected {expected_shape}"
                )

    @property
    def body_position_w(self) -> torch.Tensor:
        return self._body_position_w

    @property
    def body_orientation_w(self) -> torch.Tensor:
        return self._body_orientation_w

    @property
    def body_linear_velocity_w(self) -> torch.Tensor:
        return self._body_linear_velocity_w

    @property
    def body_angular_velocity_w(self) -> torch.Tensor:
        return self._body_angular_velocity_w
