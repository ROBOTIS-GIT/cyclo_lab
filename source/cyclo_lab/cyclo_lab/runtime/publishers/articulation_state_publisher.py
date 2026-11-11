"""ROS2-compatible state publishers for Isaac Lab articulations."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.utils.math as math_utils
import torch
from cyclo_lab.runtime.transport.ros2_zenoh import (
    JOINT_STATE,
    ODOMETRY,
    TF_MESSAGE,
    close_endpoints,
    create_publisher,
    make_joint_state_kwargs,
    make_odometry_kwargs,
    make_tf_message_kwargs,
    now_time_msg,
    transform_stamped_msg,
)


class ArticulationStatePublisher:
    """Publish joint_states, odometry, and TF for a single Isaac Lab articulation."""

    def __init__(
        self,
        robot,
        *,
        joint_names: Sequence[str],
        joint_states_topic: str,
        base_frame: str,
        base_body: str,
        odom_topic: str | None = None,
        tf_topic: str | None = None,
        odom_frame: str = "odom",
        log_prefix: str = "[Zenoh ROS2]",
    ):
        self.robot = robot
        self.base_frame = base_frame
        self.odom_frame = odom_frame
        self.log_prefix = log_prefix

        self._joint_name_to_index = {name: index for index, name in enumerate(self.robot.data.joint_names)}
        self._published_joint_names = [name for name in joint_names if name in self._joint_name_to_index]
        self._published_joint_ids = [self._joint_name_to_index[name] for name in self._published_joint_names]
        missing_joint_names = [name for name in joint_names if name not in self._joint_name_to_index]
        if missing_joint_names:
            print(f"{self.log_prefix} Missing configured joint_states joints: {missing_joint_names}")

        self._body_names = list(self.robot.data.body_names)
        self.base_body = self._resolve_base_body(base_body)
        self._base_id = self._body_names.index(self.base_body) if self.base_body in self._body_names else None
        self._tf_body_ids = [
            body_id
            for body_id, body_name in enumerate(self._body_names)
            if body_name not in (self.base_body, self.base_frame)
        ]
        self._tf_child_frames = [self._body_names[body_id] for body_id in self._tf_body_ids]

        self._odom_origin_pos_w = None
        self._odom_origin_quat_w = None
        self._odom_covariance = [0.0] * 36
        for index in (0, 7, 14, 21, 28, 35):
            self._odom_covariance[index] = 0.001

        self._warned_missing_base_frame = False

        self.joint_state_writer = create_publisher(joint_states_topic, JOINT_STATE)
        self.odom_writer = create_publisher(odom_topic, ODOMETRY) if odom_topic else None
        self.tf_writer = create_publisher(tf_topic, TF_MESSAGE) if tf_topic else None
        self.publishers = [
            writer for writer in (self.joint_state_writer, self.odom_writer, self.tf_writer) if writer is not None
        ]

    def _resolve_base_body(self, requested_base_body: str) -> str:
        if requested_base_body in self._body_names:
            return requested_base_body

        for candidate in (self.base_frame, "base_link", "world", "arm_base_link"):
            if candidate in self._body_names:
                print(
                    f"{self.log_prefix} Requested base body '{requested_base_body}' was not found; "
                    f"using '{candidate}' for base pose."
                )
                return candidate

        if self._body_names:
            fallback = self._body_names[0]
            print(
                f"{self.log_prefix} Requested base body '{requested_base_body}' was not found; "
                f"using first body '{fallback}' for base pose."
            )
            return fallback
        return requested_base_body

    def reset_odom_origin(self) -> None:
        self._odom_origin_pos_w = None
        self._odom_origin_quat_w = None

    def close(self) -> None:
        close_endpoints(self.publishers)

    def publish_all(self) -> None:
        stamp = now_time_msg()
        self.publish_joint_states(stamp=stamp)
        self.publish_odometry(stamp=stamp)
        self.publish_tf(stamp=stamp)

    def publish_joint_states(self, *, stamp=None) -> None:
        joint_values = self.robot.data.joint_pos[0, self._published_joint_ids]
        joint_velocities = self.robot.data.joint_vel[0, self._published_joint_ids]
        state = torch.stack((joint_values, joint_velocities), dim=1)
        state_cpu = state.detach().cpu().tolist()
        positions = [joint_state[0] for joint_state in state_cpu]
        velocities = [joint_state[1] for joint_state in state_cpu]
        efforts = [0.0] * len(self._published_joint_names)

        try:
            self.joint_state_writer.publish(
                **make_joint_state_kwargs(
                    names=self._published_joint_names,
                    positions=positions,
                    velocities=velocities,
                    efforts=efforts,
                    frame_id=self.base_frame,
                    stamp=stamp,
                )
            )
        except Exception as exc:
            print(f"{self.log_prefix} joint_states publish error: {exc}")

    def publish_odometry(self, *, stamp=None) -> None:
        if self.odom_writer is None:
            return

        try:
            root_pos_w = self.robot.data.root_pos_w[0:1]
            root_quat_w = self.robot.data.root_quat_w[0:1]
            if self._odom_origin_pos_w is None or self._odom_origin_quat_w is None:
                self._odom_origin_pos_w = root_pos_w.detach().clone()
                self._odom_origin_quat_w = root_quat_w.detach().clone()

            odom_pos, odom_quat = math_utils.subtract_frame_transforms(
                self._odom_origin_pos_w,
                self._odom_origin_quat_w,
                root_pos_w,
                root_quat_w,
            )
            odom_state = torch.cat((
                odom_pos[0],
                odom_quat[0],
                self.robot.data.root_lin_vel_b[0],
                self.robot.data.root_ang_vel_b[0],
            ))
            odom_state_cpu = odom_state.detach().cpu().tolist()
            pos = odom_state_cpu[0:3]
            quat_wxyz = odom_state_cpu[3:7]
            linear_velocity = odom_state_cpu[7:10]
            angular_velocity = odom_state_cpu[10:13]

            self.odom_writer.publish(
                **make_odometry_kwargs(
                    frame_id=self.odom_frame,
                    child_frame_id=self.base_frame,
                    position_xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
                    orientation_xyzw=(
                        float(quat_wxyz[1]),
                        float(quat_wxyz[2]),
                        float(quat_wxyz[3]),
                        float(quat_wxyz[0]),
                    ),
                    linear_xyz=(
                        float(linear_velocity[0]),
                        float(linear_velocity[1]),
                        float(linear_velocity[2]),
                    ),
                    angular_xyz=(
                        float(angular_velocity[0]),
                        float(angular_velocity[1]),
                        float(angular_velocity[2]),
                    ),
                    covariance=self._odom_covariance,
                    stamp=stamp,
                )
            )
        except Exception as exc:
            print(f"{self.log_prefix} odom publish error: {exc}")

    def publish_tf(self, *, stamp=None) -> None:
        if self.tf_writer is None:
            return
        if self._base_id is None:
            if not self._warned_missing_base_frame:
                self._warned_missing_base_frame = True
                print(
                    f"{self.log_prefix} Cannot publish TF: base body '{self.base_body}' is not in body names. "
                    f"Available bodies: {self._body_names}"
                )
            return

        try:
            body_pose_w = self.robot.data.body_link_state_w[0, :, :7]
            base_pose_w = body_pose_w[self._base_id]
            child_poses_w = body_pose_w[self._tf_body_ids]
            child_pos_b, child_quat_b = math_utils.subtract_frame_transforms(
                base_pose_w[:3].expand(len(self._tf_body_ids), -1),
                base_pose_w[3:7].expand(len(self._tf_body_ids), -1),
                child_poses_w[:, :3],
                child_poses_w[:, 3:7],
            )
            transforms_b = torch.cat((child_pos_b, child_quat_b), dim=1)
            transforms_cpu = transforms_b.detach().cpu().tolist()

            transforms = []
            for child_frame, transform_b in zip(self._tf_child_frames, transforms_cpu):
                pos = transform_b[:3]
                quat_wxyz = transform_b[3:7]
                transforms.append(
                    transform_stamped_msg(
                        parent_frame=self.base_frame,
                        child_frame=child_frame,
                        translation=(float(pos[0]), float(pos[1]), float(pos[2])),
                        rotation_xyzw=(
                            float(quat_wxyz[1]),
                            float(quat_wxyz[2]),
                            float(quat_wxyz[3]),
                            float(quat_wxyz[0]),
                        ),
                        stamp=stamp,
                    )
                )

            self.tf_writer.publish(**make_tf_message_kwargs(transforms))
        except Exception as exc:
            print(f"{self.log_prefix} tf publish error: {exc}")
