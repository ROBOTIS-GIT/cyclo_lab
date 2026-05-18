# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Swerve drive inverse kinematics helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


_TWO_PI = 2.0 * math.pi
_EPSILON = 1.0e-9


@dataclass(frozen=True)
class SwerveModule:
    """Geometry and joint mapping for one swerve module."""

    steering_joint: str
    wheel_joint: str
    x_offset: float
    y_offset: float
    angle_offset: float = 0.0


@dataclass(frozen=True)
class SwerveModuleCommand:
    """Command for one swerve module."""

    steering_joint: str
    wheel_joint: str
    steering_position: float
    wheel_velocity: float
    wheel_linear_speed: float


def normalize_angle(angle_rad: float) -> float:
    """Normalize an angle to [-pi, pi)."""

    remainder = math.fmod(angle_rad + math.pi, _TWO_PI)
    if remainder < 0.0:
        remainder += _TWO_PI
    return remainder - math.pi


def shortest_angular_distance(from_angle: float, to_angle: float) -> float:
    """Return the shortest signed angular distance from ``from_angle`` to ``to_angle``."""

    return normalize_angle(to_angle - from_angle)


def compute_swerve_commands(
    linear_x: float,
    linear_y: float,
    angular_z: float,
    modules: Sequence[SwerveModule],
    wheel_radius: float,
    *,
    current_steering_positions: Sequence[float] | None = None,
    linear_deadband: float = 0.0,
    angular_deadband: float = 0.0,
    optimize_steering: bool = True,
) -> list[SwerveModuleCommand]:
    """Convert a body-frame cmd_vel into per-module steering and wheel commands.
    """

    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive")

    if current_steering_positions is not None and len(current_steering_positions) != len(modules):
        raise ValueError("current_steering_positions length must match modules length")

    vx = 0.0 if abs(linear_x) < linear_deadband else float(linear_x)
    vy = 0.0 if abs(linear_y) < linear_deadband else float(linear_y)
    wz = 0.0 if abs(angular_z) < angular_deadband else float(angular_z)

    command_is_zero = vx == 0.0 and vy == 0.0 and wz == 0.0
    commands: list[SwerveModuleCommand] = []

    for index, module in enumerate(modules):
        current_steering = None
        if current_steering_positions is not None:
            current_steering = float(current_steering_positions[index])

        if command_is_zero:
            steering_position = current_steering if current_steering is not None else 0.0
            commands.append(
                SwerveModuleCommand(
                    steering_joint=module.steering_joint,
                    wheel_joint=module.wheel_joint,
                    steering_position=normalize_angle(steering_position),
                    wheel_velocity=0.0,
                    wheel_linear_speed=0.0,
                )
            )
            continue

        wheel_vel_x = vx - wz * module.y_offset
        wheel_vel_y = vy + wz * module.x_offset
        steering_robot_frame = math.atan2(wheel_vel_y, wheel_vel_x + _EPSILON)
        steering_joint_frame = normalize_angle(steering_robot_frame - module.angle_offset)
        wheel_linear_speed = math.hypot(wheel_vel_x, wheel_vel_y)
        wheel_direction = 1.0

        if optimize_steering and current_steering is not None:
            angle_diff = shortest_angular_distance(current_steering, steering_joint_frame)
            if abs(angle_diff) > (math.pi / 2.0):
                steering_joint_frame = normalize_angle(steering_joint_frame + math.pi)
                wheel_direction = -1.0

        commands.append(
            SwerveModuleCommand(
                steering_joint=module.steering_joint,
                wheel_joint=module.wheel_joint,
                steering_position=steering_joint_frame,
                wheel_velocity=wheel_direction * wheel_linear_speed / wheel_radius,
                wheel_linear_speed=wheel_direction * wheel_linear_speed,
            )
        )

    return commands
