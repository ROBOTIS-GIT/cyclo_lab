"""Reusable Isaac Lab action configurations for FFW SG2."""

from __future__ import annotations

from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_LIFT_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    FFW_SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT,
    FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
    FFW_SG2_SWERVE_ENABLED_SPEED_LIMITS,
    FFW_SG2_SWERVE_ENABLED_WHEEL_SATURATION_SCALING,
    FFW_SG2_SWERVE_LINEAR_ACCELERATION_LIMIT,
    FFW_SG2_SWERVE_STEERING_ALIGNMENT_ANGLE_ERROR_THRESHOLD,
    FFW_SG2_SWERVE_STEERING_ALIGNMENT_START_ANGLE_ERROR_THRESHOLD,
    FFW_SG2_SWERVE_STEERING_ALIGNMENT_START_SPEED_ERROR_THRESHOLD,
    FFW_SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
    FFW_SG2_SWERVE_STEERING_LIMIT_LOWER,
    FFW_SG2_SWERVE_STEERING_LIMIT_UPPER,
    FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
    FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
    SG2_SWERVE_MODULE_ANGLE_OFFSETS,
    SG2_SWERVE_MODULE_X_OFFSETS,
    SG2_SWERVE_MODULE_Y_OFFSETS,
    SG2_SWERVE_STEERING_JOINTS,
    SG2_SWERVE_WHEEL_JOINTS,
    SG2_SWERVE_WHEEL_RADIUS,
)
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from .swerve_base_action import SwerveBaseVelocityActionCfg


def make_ffw_sg2_joint_position_action_cfg(
    joint_names: tuple[str, ...],
    asset_name: str = "robot",
) -> JointPositionActionCfg:
    """Build an ordered absolute joint-position action for FFW-SG2."""
    return JointPositionActionCfg(
        asset_name=asset_name,
        joint_names=list(joint_names),
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class FFWSG2JointPositionActionsCfg:
    """Ordered 19D absolute joint-position action contract for FFW-SG2."""

    arm_l_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(FFW_SG2_LEFT_ARM_JOINT_NAMES)
    gripper_l_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(FFW_SG2_LEFT_GRIPPER_JOINT_NAMES)
    arm_r_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(FFW_SG2_RIGHT_ARM_JOINT_NAMES)
    gripper_r_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES)
    lift_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(FFW_SG2_LIFT_JOINT_NAMES)
    head_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(FFW_SG2_HEAD_JOINT_NAMES)


def make_ffw_sg2_swerve_base_action_cfg(asset_name: str = "robot") -> SwerveBaseVelocityActionCfg:
    """Build the shared SG2 body-velocity action configuration."""
    return SwerveBaseVelocityActionCfg(
        asset_name=asset_name,
        steering_joint_names=tuple(SG2_SWERVE_STEERING_JOINTS),
        wheel_joint_names=tuple(SG2_SWERVE_WHEEL_JOINTS),
        module_x_offsets=tuple(SG2_SWERVE_MODULE_X_OFFSETS),
        module_y_offsets=tuple(SG2_SWERVE_MODULE_Y_OFFSETS),
        module_angle_offsets=tuple(SG2_SWERVE_MODULE_ANGLE_OFFSETS),
        wheel_radius=SG2_SWERVE_WHEEL_RADIUS,
        steering_limit_lower=FFW_SG2_SWERVE_STEERING_LIMIT_LOWER,
        steering_limit_upper=FFW_SG2_SWERVE_STEERING_LIMIT_UPPER,
        wheel_speed_limit_lower=FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
        wheel_speed_limit_upper=FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
        steering_angular_velocity_limit=FFW_SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
        enabled_speed_limits=FFW_SG2_SWERVE_ENABLED_SPEED_LIMITS,
        linear_acceleration_limit=FFW_SG2_SWERVE_LINEAR_ACCELERATION_LIMIT,
        angular_acceleration_limit=FFW_SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT,
        steering_alignment_angle_error_threshold=FFW_SG2_SWERVE_STEERING_ALIGNMENT_ANGLE_ERROR_THRESHOLD,
        steering_alignment_start_angle_error_threshold=(FFW_SG2_SWERVE_STEERING_ALIGNMENT_START_ANGLE_ERROR_THRESHOLD),
        steering_alignment_start_speed_error_threshold=(FFW_SG2_SWERVE_STEERING_ALIGNMENT_START_SPEED_ERROR_THRESHOLD),
        enabled_wheel_saturation_scaling=FFW_SG2_SWERVE_ENABLED_WHEEL_SATURATION_SCALING,
        drive_speed_scale=FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
    )


@configclass
class FFWSG2MobileActionsCfg(FFWSG2JointPositionActionsCfg):
    """FFW-SG2 joint actions followed by the 3D swerve velocity action."""

    base_action: SwerveBaseVelocityActionCfg = make_ffw_sg2_swerve_base_action_cfg()
