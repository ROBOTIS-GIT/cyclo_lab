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

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence


_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class OdometryState:
    """Current odometry pose and base-frame twist."""

    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float


class RollingMean:
    """Small rolling mean accumulator matching the C++ odometry behavior."""

    def __init__(self, window_size: int):
        self.window_size = max(1, int(window_size))
        self.values: deque[float] = deque(maxlen=self.window_size)

    def reset(self):
        self.values.clear()

    def add(self, value: float) -> float:
        self.values.append(float(value))
        return sum(self.values) / len(self.values)


class SwerveOdometry:
    """Python port of the C++ swerve odometry update path."""

    def __init__(
        self,
        module_x_offsets: Sequence[float],
        module_y_offsets: Sequence[float],
        wheel_radius: float,
        *,
        velocity_rolling_window_size: int = 1,
    ):
        if len(module_x_offsets) != len(module_y_offsets):
            raise ValueError("module_x_offsets and module_y_offsets must have the same length")
        if len(module_x_offsets) not in (3, 4):
            raise ValueError("number of modules must be 3 or 4")
        if wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")

        self.module_x_offsets = [float(value) for value in module_x_offsets]
        self.module_y_offsets = [float(value) for value in module_y_offsets]
        self.wheel_radius = float(wheel_radius)
        self.velocity_rolling_window_size = max(1, int(velocity_rolling_window_size))

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._vx_acc = RollingMean(self.velocity_rolling_window_size)
        self._vy_acc = RollingMean(self.velocity_rolling_window_size)
        self._wz_acc = RollingMean(self.velocity_rolling_window_size)

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.reset_accumulators()

    def reset_accumulators(self):
        self._vx_acc = RollingMean(self.velocity_rolling_window_size)
        self._vy_acc = RollingMean(self.velocity_rolling_window_size)
        self._wz_acc = RollingMean(self.velocity_rolling_window_size)

    def set_velocity_rolling_window_size(self, velocity_rolling_window_size: int):
        self.velocity_rolling_window_size = max(1, int(velocity_rolling_window_size))
        self.reset_accumulators()

    def update(
        self,
        steering_positions: Sequence[float],
        wheel_velocities: Sequence[float],
        dt: float,
    ) -> bool:
        module_count = len(self.module_x_offsets)
        if len(steering_positions) != module_count or len(wheel_velocities) != module_count:
            return False
        if dt < 0.00001:
            return False

        rows: list[list[float]] = []
        values: list[float] = []
        for theta_s, omega_w, lx, ly in zip(
            steering_positions,
            wheel_velocities,
            self.module_x_offsets,
            self.module_y_offsets,
        ):
            wheel_linear_velocity = float(omega_w) * self.wheel_radius
            vx_module = wheel_linear_velocity * math.cos(theta_s)
            vy_module = wheel_linear_velocity * math.sin(theta_s)

            rows.append([1.0, 0.0, -ly])
            values.append(vx_module)
            rows.append([0.0, 1.0, lx])
            values.append(vy_module)

        twist = self._solve_least_squares_3d(rows, values)
        self._integrate(twist[0], twist[1], twist[2], dt, use_rolling_mean=True)
        return True

    def update_from_command(self, target_vx: float, target_vy: float, target_wz: float, dt: float) -> bool:
        if dt < 0.00001:
            return False
        self._integrate(target_vx, target_vy, target_wz, dt, use_rolling_mean=False)
        return True

    def state(self) -> OdometryState:
        return OdometryState(self.x, self.y, self.yaw, self.vx, self.vy, self.wz)

    def _integrate(self, vx: float, vy: float, wz: float, dt: float, *, use_rolling_mean: bool):
        if use_rolling_mean and self.velocity_rolling_window_size > 1:
            self.vx = self._vx_acc.add(vx)
            self.vy = self._vy_acc.add(vy)
            self.wz = self._wz_acc.add(wz)
        else:
            self.vx = float(vx)
            self.vy = float(vy)
            self.wz = float(wz)

        yaw_prev = self.yaw
        yaw_next = normalize_angle(yaw_prev + self.wz * dt)
        yaw_mid = normalize_angle(yaw_prev + 0.5 * self.wz * dt)
        cos_yaw = math.cos(yaw_mid)
        sin_yaw = math.sin(yaw_mid)
        odom_vx = cos_yaw * self.vx - sin_yaw * self.vy
        odom_vy = sin_yaw * self.vx + cos_yaw * self.vy
        self.x += odom_vx * dt
        self.y += odom_vy * dt
        self.yaw = yaw_next

    def _solve_least_squares_3d(self, rows: Sequence[Sequence[float]], values: Sequence[float]) -> tuple[float, float, float]:
        ata = [[0.0 for _ in range(3)] for _ in range(3)]
        atb = [0.0 for _ in range(3)]
        for row, value in zip(rows, values):
            for i in range(3):
                atb[i] += row[i] * value
                for j in range(3):
                    ata[i][j] += row[i] * row[j]

        solution = _solve_3x3(ata, atb)
        if solution is None:
            return 0.0, 0.0, 0.0
        return solution


def normalize_angle(angle_rad: float) -> float:
    remainder = math.fmod(angle_rad + math.pi, _TWO_PI)
    if remainder < 0.0:
        remainder += _TWO_PI
    return remainder - math.pi


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def _solve_3x3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float] | None:
    a00, a01, a02 = matrix[0]
    a10, a11, a12 = matrix[1]
    a20, a21, a22 = matrix[2]

    det = (
        a00 * (a11 * a22 - a21 * a12)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    if abs(det) < 1.0e-9:
        return None

    inv_det = 1.0 / det
    inv = [
        [
            (a11 * a22 - a21 * a12) * inv_det,
            (a02 * a21 - a01 * a22) * inv_det,
            (a01 * a12 - a02 * a11) * inv_det,
        ],
        [
            (a12 * a20 - a10 * a22) * inv_det,
            (a00 * a22 - a02 * a20) * inv_det,
            (a10 * a02 - a00 * a12) * inv_det,
        ],
        [
            (a10 * a21 - a20 * a11) * inv_det,
            (a20 * a01 - a00 * a21) * inv_det,
            (a00 * a11 - a10 * a01) * inv_det,
        ],
    ]
    return (
        inv[0][0] * vector[0] + inv[0][1] * vector[1] + inv[0][2] * vector[2],
        inv[1][0] * vector[0] + inv[1][1] * vector[1] + inv[1][2] * vector[2],
        inv[2][0] * vector[0] + inv[2][1] * vector[1] + inv[2][2] * vector[2],
    )
