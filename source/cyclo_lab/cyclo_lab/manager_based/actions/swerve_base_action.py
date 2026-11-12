"""Isaac Lab action term for SG2/SH5 style swerve base velocity commands."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from cyclo_lab.robot_specs.ffw.mobile_base.swerve_drive import (
    SpeedLimiter,
    SwerveControllerConfig,
    SwerveDriveController,
    SwerveModule,
)
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class SwerveBaseVelocityAction(ActionTerm):
    """Apply ``[vx, vy, wz]`` body-frame velocity commands to a swerve base."""

    cfg: SwerveBaseVelocityActionCfg
    _asset: Articulation

    def __init__(self, cfg: SwerveBaseVelocityActionCfg, env):
        super().__init__(cfg, env)
        self._validate_cfg()

        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(
            list(self.cfg.steering_joint_names), preserve_order=True
        )
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(
            list(self.cfg.wheel_joint_names), preserve_order=True
        )
        if len(self._steering_joint_ids) != len(self.cfg.steering_joint_names):
            raise ValueError(
                "Failed to resolve all steering joints: "
                f"{self.cfg.steering_joint_names} -> {self._steering_joint_names}"
            )
        if len(self._wheel_joint_ids) != len(self.cfg.wheel_joint_names):
            raise ValueError(
                f"Failed to resolve all wheel joints: {self.cfg.wheel_joint_names} -> {self._wheel_joint_names}"
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._modules = self._make_modules()
        self._module_count = len(self._modules)
        self._controllers = [self._make_controller() for _ in range(self.num_envs)]
        self._steering_targets = torch.zeros(self.num_envs, len(self._steering_joint_ids), device=self.device)
        self._wheel_velocity_targets = torch.zeros(self.num_envs, len(self._wheel_joint_ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self):
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "SwerveBaseVelocityAction"
        self._IO_descriptor.joint_names = [*self._steering_joint_names, *self._wheel_joint_names]
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.nan_to_num(actions.to(self.device), nan=0.0, posinf=0.0, neginf=0.0)

    def apply_actions(self):
        state_rows = (
            torch.cat(
                (
                    self._processed_actions,
                    self._asset.data.joint_pos[:, self._steering_joint_ids],
                    self._asset.data.joint_vel[:, self._wheel_joint_ids],
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        dt = float(getattr(self._env, "physics_dt", 1.0 / 60.0))

        steering_targets = []
        wheel_velocity_targets = []
        for env_id, state_row in enumerate(state_rows):
            base_command = state_row[: self.action_dim]
            steering_state = state_row[self.action_dim : self.action_dim + self._module_count]
            wheel_state = state_row[self.action_dim + self._module_count :]
            module_commands = self._controllers[env_id].compute_commands(
                float(base_command[0]),
                float(base_command[1]),
                float(base_command[2]),
                current_steering_positions=steering_state,
                current_wheel_velocities=wheel_state,
                dt=dt,
            )
            steering_targets.append([module_command.steering_position for module_command in module_commands])
            wheel_velocity_targets.append([module_command.wheel_velocity for module_command in module_commands])

        self._steering_targets.copy_(
            torch.as_tensor(steering_targets, device=self.device, dtype=self._steering_targets.dtype)
        )
        self._wheel_velocity_targets.copy_(
            torch.as_tensor(wheel_velocity_targets, device=self.device, dtype=self._wheel_velocity_targets.dtype)
        )

        self._asset.set_joint_position_target(self._steering_targets, joint_ids=self._steering_joint_ids)
        self._asset.set_joint_velocity_target(self._wheel_velocity_targets, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = range(self.num_envs)
        if isinstance(env_ids, slice):
            env_ids = range(self.num_envs)
        if torch.is_tensor(env_ids):
            env_ids = env_ids.detach().cpu().tolist()
        for env_id in env_ids:
            self._raw_actions[env_id] = 0.0
            self._processed_actions[env_id] = 0.0
            self._controllers[int(env_id)].reset()

    def _validate_cfg(self) -> None:
        lengths = {
            len(self.cfg.steering_joint_names),
            len(self.cfg.wheel_joint_names),
            len(self.cfg.module_x_offsets),
            len(self.cfg.module_y_offsets),
            len(self.cfg.module_angle_offsets),
        }
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("SwerveBaseVelocityActionCfg module joint and geometry lengths must match.")
        if self.cfg.wheel_radius <= 0.0:
            raise ValueError("SwerveBaseVelocityActionCfg wheel_radius must be positive.")

    def _make_modules(self) -> list[SwerveModule]:
        return [
            SwerveModule(
                steering_joint=steering_joint,
                wheel_joint=wheel_joint,
                x_offset=self.cfg.module_x_offsets[index],
                y_offset=self.cfg.module_y_offsets[index],
                angle_offset=self.cfg.module_angle_offsets[index],
                steering_limit_lower=self.cfg.steering_limit_lower,
                steering_limit_upper=self.cfg.steering_limit_upper,
                wheel_speed_limit_lower=self.cfg.wheel_speed_limit_lower * self.cfg.drive_speed_scale,
                wheel_speed_limit_upper=self.cfg.wheel_speed_limit_upper * self.cfg.drive_speed_scale,
            )
            for index, (steering_joint, wheel_joint) in enumerate(
                zip(self.cfg.steering_joint_names, self.cfg.wheel_joint_names)
            )
        ]

    def _make_controller(self) -> SwerveDriveController:
        return SwerveDriveController(
            self._modules,
            self.cfg.wheel_radius,
            config=SwerveControllerConfig(
                enabled_speed_limits=self.cfg.enabled_speed_limits,
                linear_x_limiter=SpeedLimiter(
                    has_acceleration_limits=True,
                    max_acceleration=self.cfg.linear_acceleration_limit * self.cfg.drive_speed_scale,
                ),
                linear_y_limiter=SpeedLimiter(
                    has_acceleration_limits=True,
                    max_acceleration=self.cfg.linear_acceleration_limit * self.cfg.drive_speed_scale,
                ),
                angular_z_limiter=SpeedLimiter(
                    has_acceleration_limits=True,
                    max_acceleration=self.cfg.angular_acceleration_limit * self.cfg.drive_speed_scale,
                ),
                steering_angular_velocity_limit=self.cfg.steering_angular_velocity_limit,
                steering_alignment_angle_error_threshold=self.cfg.steering_alignment_angle_error_threshold,
                steering_alignment_start_angle_error_threshold=self.cfg.steering_alignment_start_angle_error_threshold,
                steering_alignment_start_speed_error_threshold=self.cfg.steering_alignment_start_speed_error_threshold,
                enabled_wheel_saturation_scaling=self.cfg.enabled_wheel_saturation_scaling,
            ),
        )


@configclass
class SwerveBaseVelocityActionCfg(ActionTermCfg):
    """Configuration for body-frame swerve base velocity control."""

    class_type: type[ActionTerm] = SwerveBaseVelocityAction

    steering_joint_names: tuple[str, ...] = ()
    wheel_joint_names: tuple[str, ...] = ()
    module_x_offsets: tuple[float, ...] = ()
    module_y_offsets: tuple[float, ...] = ()
    module_angle_offsets: tuple[float, ...] = ()
    wheel_radius: float = 0.0
    steering_limit_lower: float = -3.141592653589793
    steering_limit_upper: float = 3.141592653589793
    wheel_speed_limit_lower: float = -50.0
    wheel_speed_limit_upper: float = 50.0
    steering_angular_velocity_limit: float = 4.0
    enabled_speed_limits: bool = True
    linear_acceleration_limit: float = 0.6
    angular_acceleration_limit: float = 1.2
    steering_alignment_angle_error_threshold: float = 0.2
    steering_alignment_start_angle_error_threshold: float = 0.2
    steering_alignment_start_speed_error_threshold: float = 0.1
    enabled_wheel_saturation_scaling: bool = False
    drive_speed_scale: float = 1.0
