"""Small wrapper layer around zenoh_ros2_sdk.

The runtime scripts only need ROS2 topic compatibility, not a local ROS2
installation. This module keeps the Zenoh SDK details in one place.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from site import addsitedir
from typing import Any, Callable, Iterable, Sequence

import numpy as np


def _ensure_zenoh_sdk_on_path() -> None:
    sdk_path = os.environ.get("ZENOH_SDK_PATH")
    candidates = []
    if sdk_path:
        candidates.append(Path(sdk_path))

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "third_party" / "zenoh_ros2_sdk"
        if candidate.is_dir():
            candidates.append(candidate)
            break

    for candidate in candidates:
        if candidate.is_dir():
            addsitedir(str(candidate))
            return


_ensure_zenoh_sdk_on_path()

from zenoh_ros2_sdk import ROS2Publisher, ROS2Subscriber, get_message_class  # noqa: E402
from zenoh_ros2_sdk.qos import QosDurability, QosHistoryKind, QosProfile, QosReliability  # noqa: E402

JOINT_TRAJECTORY = "trajectory_msgs/msg/JointTrajectory"
JOINT_STATE = "sensor_msgs/msg/JointState"
COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"
ODOMETRY = "nav_msgs/msg/Odometry"
TF_MESSAGE = "tf2_msgs/msg/TFMessage"
TWIST = "geometry_msgs/msg/Twist"
STRING = "std_msgs/msg/String"
EMPTY = "std_msgs/msg/Empty"


def ros_domain_id() -> int:
    raw = os.environ.get("ROS_DOMAIN_ID", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def zenoh_router_ip() -> str:
    return os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1")


def zenoh_router_port() -> int:
    raw = os.environ.get("ZENOH_ROUTER_PORT", "7447")
    try:
        return int(raw)
    except ValueError:
        return 7447


def best_effort_qos(depth: int = 10) -> QosProfile:
    return QosProfile(
        reliability=QosReliability.BEST_EFFORT,
        durability=QosDurability.VOLATILE,
        history_kind=QosHistoryKind.KEEP_LAST,
        history_depth=depth,
    )


def _common_endpoint_kwargs(qos: QosProfile | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "domain_id": ros_domain_id(),
        "router_ip": zenoh_router_ip(),
        "router_port": zenoh_router_port(),
    }
    if qos is not None:
        kwargs["qos"] = qos
    return kwargs


def create_publisher(topic: str, msg_type: str, qos: QosProfile | None = None) -> ROS2Publisher:
    return ROS2Publisher(topic=topic, msg_type=msg_type, **_common_endpoint_kwargs(qos))


def create_subscriber(
    topic: str,
    msg_type: str,
    callback: Callable[[Any], None],
    qos: QosProfile | None = None,
) -> ROS2Subscriber:
    return ROS2Subscriber(topic=topic, msg_type=msg_type, callback=callback, **_common_endpoint_kwargs(qos))


def close_endpoint(endpoint: Any) -> None:
    close = getattr(endpoint, "close", None)
    if close is None:
        close = getattr(endpoint, "Close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        pass


def close_endpoints(endpoints: Iterable[Any]) -> None:
    for endpoint in endpoints:
        close_endpoint(endpoint)


@lru_cache(maxsize=None)
def msg_class(msg_type: str) -> type:
    return get_message_class(msg_type)


def now_time_msg():
    now_ns = time.time_ns()
    return msg_class("builtin_interfaces/msg/Time")(
        sec=now_ns // 1_000_000_000,
        nanosec=now_ns % 1_000_000_000,
    )


def time_msg(sec: int = 0, nanosec: int = 0):
    return msg_class("builtin_interfaces/msg/Time")(sec=int(sec), nanosec=int(nanosec))


def duration_msg(sec: int = 0, nanosec: int = 0):
    return msg_class("builtin_interfaces/msg/Duration")(sec=int(sec), nanosec=int(nanosec))


def header_msg(frame_id: str = "", stamp: Any | None = None):
    return msg_class("std_msgs/msg/Header")(stamp=stamp if stamp is not None else now_time_msg(), frame_id=frame_id)


def vector3_msg(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return msg_class("geometry_msgs/msg/Vector3")(x=float(x), y=float(y), z=float(z))


def quaternion_msg(x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
    return msg_class("geometry_msgs/msg/Quaternion")(x=float(x), y=float(y), z=float(z), w=float(w))


def point_msg(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return msg_class("geometry_msgs/msg/Point")(x=float(x), y=float(y), z=float(z))


def transform_msg(translation: Any, rotation: Any):
    return msg_class("geometry_msgs/msg/Transform")(translation=translation, rotation=rotation)


def transform_stamped_msg(
    parent_frame: str,
    child_frame: str,
    translation: Sequence[float],
    rotation_xyzw: Sequence[float],
    stamp: Any | None = None,
):
    return msg_class("geometry_msgs/msg/TransformStamped")(
        header=header_msg(parent_frame, stamp=stamp if stamp is not None else now_time_msg()),
        child_frame_id=child_frame,
        transform=transform_msg(
            translation=vector3_msg(*translation),
            rotation=quaternion_msg(*rotation_xyzw),
        ),
    )


def make_joint_state_kwargs(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    efforts: Sequence[float],
    frame_id: str = "base_link",
    stamp: Any | None = None,
) -> dict[str, Any]:
    return {
        "header": header_msg(frame_id, stamp=stamp if stamp is not None else now_time_msg()),
        "name": list(names),
        "position": np.asarray(positions, dtype=np.float64),
        "velocity": np.asarray(velocities, dtype=np.float64),
        "effort": np.asarray(efforts, dtype=np.float64),
    }


def make_joint_trajectory_kwargs(
    joint_names: Sequence[str],
    positions: Sequence[float],
    time_from_start_sec: float = 0.0,
    stamp: Any | None = None,
) -> dict[str, Any]:
    point = msg_class("trajectory_msgs/msg/JointTrajectoryPoint")(
        positions=np.asarray(positions, dtype=np.float64),
        velocities=np.zeros(0, dtype=np.float64),
        accelerations=np.zeros(0, dtype=np.float64),
        effort=np.zeros(0, dtype=np.float64),
        time_from_start=duration_msg(
            sec=int(time_from_start_sec),
            nanosec=int((time_from_start_sec % 1.0) * 1_000_000_000),
        ),
    )
    return {
        "header": header_msg("", stamp=stamp if stamp is not None else time_msg()),
        "joint_names": list(joint_names),
        "points": [point],
    }


def make_twist_kwargs(linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0) -> dict[str, Any]:
    return {
        "linear": vector3_msg(linear_x, linear_y, 0.0),
        "angular": vector3_msg(0.0, 0.0, angular_z),
    }


def make_odometry_kwargs(
    frame_id: str,
    child_frame_id: str,
    position_xyz: Sequence[float],
    orientation_xyzw: Sequence[float],
    linear_xyz: Sequence[float],
    angular_xyz: Sequence[float],
    covariance: Sequence[float],
    stamp: Any | None = None,
) -> dict[str, Any]:
    covariance_array = np.asarray(covariance, dtype=np.float64)
    pose = msg_class("geometry_msgs/msg/Pose")(
        position=point_msg(*position_xyz),
        orientation=quaternion_msg(*orientation_xyzw),
    )
    twist = msg_class("geometry_msgs/msg/Twist")(
        linear=vector3_msg(*linear_xyz),
        angular=vector3_msg(*angular_xyz),
    )
    return {
        "header": header_msg(frame_id, stamp=stamp if stamp is not None else now_time_msg()),
        "child_frame_id": child_frame_id,
        "pose": msg_class("geometry_msgs/msg/PoseWithCovariance")(
            pose=pose,
            covariance=covariance_array,
        ),
        "twist": msg_class("geometry_msgs/msg/TwistWithCovariance")(
            twist=twist,
            covariance=covariance_array,
        ),
    }


def make_tf_message_kwargs(transforms: Sequence[Any]) -> dict[str, Any]:
    return {"transforms": list(transforms)}


def make_compressed_image_kwargs(
    data: bytes | bytearray | memoryview | np.ndarray,
    frame_id: str,
    fmt: str = "jpeg",
    stamp: Any | None = None,
) -> dict[str, Any]:
    if isinstance(data, np.ndarray):
        image_data = data.astype(np.uint8, copy=False).reshape(-1)
    else:
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
    return {
        "header": header_msg(frame_id, stamp=stamp if stamp is not None else now_time_msg()),
        "format": fmt,
        "data": image_data,
    }


def make_string_kwargs(data: str) -> dict[str, Any]:
    return {"data": data}
