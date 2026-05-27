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
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


_TWO_PI = 2.0 * math.pi
_EPSILON = 1.0e-9
_PI_HALF = math.pi / 2.0
_LIMIT_EPSILON = 1.0e-6

DEFAULT_REVERSAL_DECEL_RATE = 4.0
DEFAULT_REVERSAL_ACCEL_RATE = 4.0
DEFAULT_REVERSAL_THRESHOLD = 0.05
DEFAULT_STEERING_TOLERANCE = 0.03


@dataclass(frozen=True)
class SwerveModule:
    """Geometry and joint mapping for one swerve module."""

    steering_joint: str
    wheel_joint: str
    x_offset: float
    y_offset: float
    angle_offset: float = 0.0
    steering_limit_lower: float = -math.pi
    steering_limit_upper: float = math.pi
    wheel_speed_limit_lower: float = -math.inf
    wheel_speed_limit_upper: float = math.inf


@dataclass(frozen=True)
class SwerveModuleCommand:
    """Command for one swerve module."""

    steering_joint: str
    wheel_joint: str
    steering_position: float
    wheel_velocity: float
    wheel_linear_speed: float


class ReversalPhase(Enum):
    """Direction-reversal phase for one swerve module."""

    NORMAL = "normal"
    DECELERATING = "decelerating"
    STEERING = "steering"
    ACCELERATING = "accelerating"


@dataclass
class SpeedLimiter:
    """Python port of the ROS controller SpeedLimiter."""

    has_velocity_limits: bool = False
    has_acceleration_limits: bool = False
    has_jerk_limits: bool = False
    min_velocity: float = math.nan
    max_velocity: float = math.nan
    min_acceleration: float = math.nan
    max_acceleration: float = math.nan
    min_jerk: float = math.nan
    max_jerk: float = math.nan

    def __post_init__(self):
        if self.has_velocity_limits:
            if math.isnan(self.max_velocity):
                raise ValueError("Cannot apply velocity limits if max_velocity is not specified")
            if math.isnan(self.min_velocity):
                self.min_velocity = -self.max_velocity
        if self.has_acceleration_limits:
            if math.isnan(self.max_acceleration):
                raise ValueError("Cannot apply acceleration limits if max_acceleration is not specified")
            if math.isnan(self.min_acceleration):
                self.min_acceleration = -self.max_acceleration
        if self.has_jerk_limits:
            if math.isnan(self.max_jerk):
                raise ValueError("Cannot apply jerk limits if max_jerk is not specified")
            if math.isnan(self.min_jerk):
                self.min_jerk = -self.max_jerk

    def limit(self, value: float, previous: float, previous_previous: float, dt: float) -> float:
        value = self.limit_jerk(value, previous, previous_previous, dt)
        value = self.limit_acceleration(value, previous, dt)
        return self.limit_velocity(value)

    def limit_velocity(self, value: float) -> float:
        if self.has_velocity_limits:
            return max(self.min_velocity, min(value, self.max_velocity))
        return value

    def limit_acceleration(self, value: float, previous: float, dt: float) -> float:
        if self.has_acceleration_limits:
            dv_min = self.min_acceleration * dt
            dv_max = self.max_acceleration * dt
            dv = max(dv_min, min(value - previous, dv_max))
            return previous + dv
        return value

    def limit_jerk(self, value: float, previous: float, previous_previous: float, dt: float) -> float:
        if self.has_jerk_limits:
            dv = value - previous
            dv0 = previous - previous_previous
            dt2 = 2.0 * dt * dt
            da_min = self.min_jerk * dt2
            da_max = self.max_jerk * dt2
            da = max(da_min, min(dv - dv0, da_max))
            return previous + dv0 + da
        return value


@dataclass(frozen=True)
class SwerveControllerConfig:
    """Controller options matching the C++ command-generation path."""

    linear_deadband: float = 0.1
    angular_deadband: float = 0.1
    enabled_speed_limits: bool = False
    linear_x_limiter: SpeedLimiter = field(default_factory=SpeedLimiter)
    linear_y_limiter: SpeedLimiter = field(default_factory=SpeedLimiter)
    angular_z_limiter: SpeedLimiter = field(default_factory=SpeedLimiter)
    enabled_steering_angular_velocity_limit: bool = True
    steering_angular_velocity_limit: float = 1.0
    steering_alignment_angle_error_threshold: float = 0.2
    steering_alignment_start_angle_error_threshold: float = 0.2
    steering_alignment_start_speed_error_threshold: float = 0.1
    enabled_wheel_saturation_scaling: bool = False
    reversal_decel_rate: float = DEFAULT_REVERSAL_DECEL_RATE
    reversal_accel_rate: float = DEFAULT_REVERSAL_ACCEL_RATE
    reversal_threshold: float = DEFAULT_REVERSAL_THRESHOLD
    steering_tolerance: float = DEFAULT_STEERING_TOLERANCE


def normalize_angle(angle_rad: float) -> float:
    """Normalize an angle to [-pi, pi)."""

    remainder = math.fmod(angle_rad + math.pi, _TWO_PI)
    if remainder < 0.0:
        remainder += _TWO_PI
    return remainder - math.pi


def normalize_angle_positive(angle_rad: float) -> float:
    """Normalize an angle to [0, 2*pi)."""

    result = math.fmod(angle_rad, _TWO_PI)
    return result + _TWO_PI if result < 0.0 else result


def shortest_angular_distance(from_angle: float, to_angle: float) -> float:
    """Return the shortest signed angular distance from ``from_angle`` to ``to_angle``."""

    result = normalize_angle_positive(to_angle) - normalize_angle_positive(from_angle)
    if result > math.pi:
        return result - _TWO_PI
    if result < -math.pi:
        return result + _TWO_PI
    return result


class SwerveDriveController:
    """Stateful Python port of the C++ swerve command logic.

    This class intentionally keeps only the command-generation pieces from the
    ROS controller: speed limits, 180-degree steering optimization, smooth
    reversal phases, steering rate limiting, alignment gating, and wheel
    saturation handling.
    """

    def __init__(
        self,
        modules: Sequence[SwerveModule],
        wheel_radius: float,
        config: SwerveControllerConfig | None = None,
    ):
        if wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")

        self.modules = list(modules)
        self.wheel_radius = float(wheel_radius)
        self.config = config if config is not None else SwerveControllerConfig()

        self.reversal_phase = [ReversalPhase.NORMAL for _ in self.modules]
        self.previous_wheel_rotation_direction = [1.0 for _ in self.modules]
        self.wheel_speed_scale = [1.0 for _ in self.modules]
        self.reversal_target_steering_angle = [0.0 for _ in self.modules]
        self.previous_steering_commands = [0.0 for _ in self.modules]
        self.cmd_velocity_history = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

    def reset(self):
        """Reset stateful reversal and limiter history."""

        for index in range(len(self.modules)):
            self.reversal_phase[index] = ReversalPhase.NORMAL
            self.previous_wheel_rotation_direction[index] = 1.0
            self.wheel_speed_scale[index] = 1.0
            self.reversal_target_steering_angle[index] = 0.0
            self.previous_steering_commands[index] = 0.0
        self.cmd_velocity_history = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

    def compute_commands(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
        *,
        current_steering_positions: Sequence[float] | None = None,
        current_wheel_velocities: Sequence[float] | None = None,
        dt: float = 1.0 / 60.0,
        enabled_open_loop: bool = False,
    ) -> list[SwerveModuleCommand]:
        """Convert a body-frame cmd_vel into per-module commands."""

        module_count = len(self.modules)
        if current_steering_positions is not None and len(current_steering_positions) != module_count:
            raise ValueError("current_steering_positions length must match modules length")
        if current_wheel_velocities is not None and len(current_wheel_velocities) != module_count:
            raise ValueError("current_wheel_velocities length must match modules length")

        time_gap = max(0.001, float(dt))
        vx = 0.0 if abs(linear_x) < self.config.linear_deadband else float(linear_x)
        vy = 0.0 if abs(linear_y) < self.config.linear_deadband else float(linear_y)
        wz = 0.0 if abs(angular_z) < self.config.angular_deadband else float(angular_z)

        if any(math.isnan(value) for value in (vx, vy, wz)):
            vx = vy = wz = 0.0

        if self.config.enabled_speed_limits:
            previous_previous = self.cmd_velocity_history[0]
            previous = self.cmd_velocity_history[1]
            vx = self.config.linear_x_limiter.limit(vx, previous[0], previous_previous[0], time_gap)
            vy = self.config.linear_y_limiter.limit(vy, previous[1], previous_previous[1], time_gap)
            wz = self.config.angular_z_limiter.limit(wz, previous[2], previous_previous[2], time_gap)
            self.cmd_velocity_history = [previous, (vx, vy, wz)]

        command_is_zero = vx == 0.0 and vy == 0.0 and wz == 0.0
        if command_is_zero:
            for index in range(module_count):
                self.reversal_phase[index] = ReversalPhase.NORMAL
                self.wheel_speed_scale[index] = 1.0

        final_steering_commands = [0.0 for _ in self.modules]
        final_wheel_velocity_commands = [0.0 for _ in self.modules]
        all_steering_aligned = True
        wheel_saturation_scale_factor = 1.0

        for index, module in enumerate(self.modules):
            current_steering_angle = self._current_steering(
                index, current_steering_positions, enabled_open_loop
            )
            current_wheel_velocity = (
                float(current_wheel_velocities[index]) if current_wheel_velocities is not None else 0.0
            )

            wheel_vel_x = vx - wz * module.y_offset
            wheel_vel_y = vy + wz * module.x_offset
            target_steering_angle_robot = math.atan2(wheel_vel_y, wheel_vel_x + _EPSILON)
            target_wheel_speed = math.hypot(wheel_vel_x, wheel_vel_y)
            target_steering_joint_angle = normalize_angle(
                target_steering_angle_robot - module.angle_offset
            )

            optimized_steering_angle, wheel_rotation_direction = self._select_steering_solution(
                current_steering_angle,
                target_steering_joint_angle,
                module,
            )

            angle_diff_after_opt = shortest_angular_distance(
                current_steering_angle, optimized_steering_angle
            )
            crosses_boundary = (
                (current_steering_angle > 0.0 and optimized_steering_angle < 0.0 and angle_diff_after_opt > 0.0)
                or (current_steering_angle < 0.0 and optimized_steering_angle > 0.0 and angle_diff_after_opt < 0.0)
            )
            if crosses_boundary:
                optimized_steering_angle = normalize_angle(optimized_steering_angle + math.pi)
                wheel_rotation_direction *= -1.0

            if not command_is_zero:
                direction_changed = (
                    wheel_rotation_direction != self.previous_wheel_rotation_direction[index]
                )
                if direction_changed and self.reversal_phase[index] == ReversalPhase.NORMAL:
                    self.reversal_phase[index] = ReversalPhase.DECELERATING
                    self.reversal_target_steering_angle[index] = optimized_steering_angle

            limited_steering_cmd = max(
                module.steering_limit_lower,
                min(normalize_angle(optimized_steering_angle), module.steering_limit_upper),
            )
            steering_target_for_this_cycle = limited_steering_cmd

            if not command_is_zero:
                phase = self.reversal_phase[index]
                if phase == ReversalPhase.DECELERATING:
                    steering_target_for_this_cycle = current_steering_angle
                    self.wheel_speed_scale[index] -= self.config.reversal_decel_rate * time_gap
                    if self.wheel_speed_scale[index] <= self.config.reversal_threshold:
                        self.wheel_speed_scale[index] = 0.0
                        self.reversal_target_steering_angle[index] = limited_steering_cmd
                        self.reversal_phase[index] = ReversalPhase.STEERING
                elif phase == ReversalPhase.STEERING:
                    self.reversal_target_steering_angle[index] = limited_steering_cmd
                    steering_target_for_this_cycle = limited_steering_cmd
                    self.wheel_speed_scale[index] = 0.0
                    steering_error = abs(
                        shortest_angular_distance(current_steering_angle, limited_steering_cmd)
                    )
                    if steering_error < self.config.steering_tolerance:
                        self.previous_wheel_rotation_direction[index] = wheel_rotation_direction
                        self.reversal_phase[index] = ReversalPhase.ACCELERATING
                elif phase == ReversalPhase.ACCELERATING:
                    self.wheel_speed_scale[index] += self.config.reversal_accel_rate * time_gap
                    if self.wheel_speed_scale[index] >= 1.0:
                        self.wheel_speed_scale[index] = 1.0
                        self.reversal_phase[index] = ReversalPhase.NORMAL
                else:
                    self.wheel_speed_scale[index] = 1.0

            self.wheel_speed_scale[index] = max(0.0, min(self.wheel_speed_scale[index], 1.0))

            if self.config.enabled_steering_angular_velocity_limit:
                optimized_steering_angle = self._apply_steering_velocity_limit(
                    current_steering_angle, steering_target_for_this_cycle, time_gap
                )
            else:
                optimized_steering_angle = steering_target_for_this_cycle

            optimized_steering_angle = max(
                module.steering_limit_lower,
                min(optimized_steering_angle, module.steering_limit_upper),
            )

            effective_direction = (
                self.previous_wheel_rotation_direction[index]
                if self.reversal_phase[index] == ReversalPhase.DECELERATING
                else wheel_rotation_direction
            )
            final_wheel_vel_cmd = (
                effective_direction * target_wheel_speed * self.wheel_speed_scale[index] / self.wheel_radius
            )

            align_err = abs(shortest_angular_distance(current_steering_angle, limited_steering_cmd))
            module_steering_aligned = True
            if abs(current_wheel_velocity) >= self.config.steering_alignment_start_speed_error_threshold:
                if self.config.steering_alignment_angle_error_threshold <= align_err:
                    final_wheel_vel_cmd = 0.0
                    module_steering_aligned = False
            elif self.config.steering_alignment_start_angle_error_threshold <= align_err:
                final_wheel_vel_cmd = 0.0
                module_steering_aligned = False
            all_steering_aligned = all_steering_aligned and module_steering_aligned

            if (
                final_wheel_vel_cmd < module.wheel_speed_limit_lower
                or final_wheel_vel_cmd > module.wheel_speed_limit_upper
            ):
                clipped = max(module.wheel_speed_limit_lower, min(final_wheel_vel_cmd, module.wheel_speed_limit_upper))
                if self.config.enabled_wheel_saturation_scaling and final_wheel_vel_cmd != 0.0:
                    wheel_saturation_scale_factor = min(
                        wheel_saturation_scale_factor,
                        abs(clipped / final_wheel_vel_cmd),
                    )
                else:
                    final_wheel_vel_cmd = clipped

            final_steering_commands[index] = optimized_steering_angle
            final_wheel_velocity_commands[index] = final_wheel_vel_cmd

        commands: list[SwerveModuleCommand] = []
        if command_is_zero:
            for index, module in enumerate(self.modules):
                hold_angle = self._current_steering(index, current_steering_positions, enabled_open_loop)
                hold_angle = max(module.steering_limit_lower, min(hold_angle, module.steering_limit_upper))
                self.previous_steering_commands[index] = hold_angle
                commands.append(
                    SwerveModuleCommand(
                        steering_joint=module.steering_joint,
                        wheel_joint=module.wheel_joint,
                        steering_position=hold_angle,
                        wheel_velocity=0.0,
                        wheel_linear_speed=0.0,
                    )
                )
            return commands

        for index, module in enumerate(self.modules):
            wheel_velocity = (
                final_wheel_velocity_commands[index] * wheel_saturation_scale_factor
                if all_steering_aligned
                else 0.0
            )
            self.previous_steering_commands[index] = final_steering_commands[index]
            commands.append(
                SwerveModuleCommand(
                    steering_joint=module.steering_joint,
                    wheel_joint=module.wheel_joint,
                    steering_position=final_steering_commands[index],
                    wheel_velocity=wheel_velocity,
                    wheel_linear_speed=wheel_velocity * self.wheel_radius,
                )
            )

        return commands

    def _current_steering(
        self,
        index: int,
        current_steering_positions: Sequence[float] | None,
        enabled_open_loop: bool,
    ) -> float:
        if enabled_open_loop:
            return self.previous_steering_commands[index]
        if current_steering_positions is None:
            return self.previous_steering_commands[index]
        return float(current_steering_positions[index])

    def _select_steering_solution(
        self,
        current_steering_angle: float,
        target_steering_joint_angle: float,
        module: SwerveModule,
    ) -> tuple[float, float]:
        direct_angle = normalize_angle(target_steering_joint_angle)
        flipped_angle = normalize_angle(target_steering_joint_angle + math.pi)
        candidates = [
            (direct_angle, 1.0),
            (flipped_angle, -1.0),
        ]

        valid_candidates = []
        for angle, direction in candidates:
            if (
                module.steering_limit_lower - _LIMIT_EPSILON
                <= angle
                <= module.steering_limit_upper + _LIMIT_EPSILON
            ):
                clamped_angle = max(
                    module.steering_limit_lower,
                    min(angle, module.steering_limit_upper),
                )
                valid_candidates.append((clamped_angle, direction))
        if not valid_candidates:
            clipped_angle = max(module.steering_limit_lower, min(direct_angle, module.steering_limit_upper))
            return clipped_angle, 1.0

        return min(
            valid_candidates,
            key=lambda candidate: abs(shortest_angular_distance(current_steering_angle, candidate[0])),
        )

    def _apply_steering_velocity_limit(
        self,
        current_steering_angle: float,
        steering_target_for_this_cycle: float,
        dt: float,
    ) -> float:
        effective_limit = self.config.steering_angular_velocity_limit
        if effective_limit <= 0.0 or effective_limit >= float("inf"):
            effective_limit = 1.0

        max_change = effective_limit * dt
        desired_change = shortest_angular_distance(current_steering_angle, steering_target_for_this_cycle)
        if abs(desired_change) <= max_change:
            return steering_target_for_this_cycle

        direction = 1.0 if desired_change > 0.0 else -1.0
        new_angle = current_steering_angle + direction * max_change
        normalized = normalize_angle(new_angle)
        if abs(normalized - current_steering_angle) > math.pi:
            return math.pi if new_angle > 0.0 else -math.pi
        return normalized


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

    Stateless compatibility wrapper. Use ``SwerveDriveController`` when smooth
    reversal phases and speed-limiter history must persist across frames.
    """

    return _compute_direct_swerve_commands(
        linear_x,
        linear_y,
        angular_z,
        modules,
        wheel_radius,
        current_steering_positions=current_steering_positions,
        linear_deadband=linear_deadband,
        angular_deadband=angular_deadband,
        optimize_steering=optimize_steering,
    )


def _compute_direct_swerve_commands(
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
    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive")

    vx = 0.0 if abs(linear_x) < linear_deadband else float(linear_x)
    vy = 0.0 if abs(linear_y) < linear_deadband else float(linear_y)
    wz = 0.0 if abs(angular_z) < angular_deadband else float(angular_z)

    command_is_zero = vx == 0.0 and vy == 0.0 and wz == 0.0
    commands: list[SwerveModuleCommand] = []
    for index, module in enumerate(modules):
        current_steering = (
            float(current_steering_positions[index]) if current_steering_positions is not None else 0.0
        )
        if command_is_zero:
            commands.append(
                SwerveModuleCommand(module.steering_joint, module.wheel_joint, current_steering, 0.0, 0.0)
            )
            continue

        wheel_vel_x = vx - wz * module.y_offset
        wheel_vel_y = vy + wz * module.x_offset
        steering_robot_frame = math.atan2(wheel_vel_y, wheel_vel_x + _EPSILON)
        steering_joint_frame = normalize_angle(steering_robot_frame - module.angle_offset)
        wheel_linear_speed = math.hypot(wheel_vel_x, wheel_vel_y)
        wheel_direction = 1.0

        if optimize_steering and current_steering_positions is not None:
            angle_diff = shortest_angular_distance(current_steering, steering_joint_frame)
            if abs(angle_diff) > _PI_HALF:
                steering_joint_frame = normalize_angle(steering_joint_frame + math.pi)
                wheel_direction = -1.0

        commands.append(
            SwerveModuleCommand(
                module.steering_joint,
                module.wheel_joint,
                steering_joint_frame,
                wheel_direction * wheel_linear_speed / wheel_radius,
                wheel_direction * wheel_linear_speed,
            )
        )
    return commands
