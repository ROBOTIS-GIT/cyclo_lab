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
# Author: Kiwoong Park

"""Load motion arrays in Isaac Lab 3.0's xyzw and articulation name order."""

from collections.abc import Sequence

import numpy as np


# Isaac Lab 2 / PhysX ordering used by existing K1 motion archives.
_LEGACY_K1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]

_LEGACY_K1_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "torso_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
]


# Legacy K1 25-body archives place head_link after the two hip-roll links.
_LEGACY_K1_BODY_NAMES_WITH_HEAD = [
    *_LEGACY_K1_BODY_NAMES[:6],
    "head_link",
    *_LEGACY_K1_BODY_NAMES[6:],
]


def load_motion_npz(path, joint_names: Sequence[str] | None = None, body_names: Sequence[str] | None = None):
    """Read motion arrays in xyzw order and the requested articulation name order."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    motion_format = _resolve_metadata(data, path)
    source_order = _normalize_quaternions(data)
    source_body_count = len(data["body_names"])
    _reorder_motion(data, joint_names, body_names)

    body_count = str(source_body_count)

    if len(data["body_names"]) != source_body_count:
        body_count += f" -> {len(data['body_names'])}"

    quaternion_order = "wxyz -> xyzw" if source_order == "wxyz" else "xyzw"
    print(
        f"[INFO] [Motion] {path}: {motion_format}, {len(data['joint_names'])} joints / "
        f"{body_count} bodies, {quaternion_order}"
    )
    return data


def _resolve_metadata(data, path) -> str:
    """Validate explicit metadata or supply the known legacy K1 layout."""
    metadata_keys = {"joint_names", "body_names", "quaternion_order", "quat_order"}

    if metadata_keys.intersection(data):
        has_names = {"joint_names", "body_names"}.issubset(data)
        has_order = bool({"quaternion_order", "quat_order"}.intersection(data))

        if not has_names or not has_order:
            raise ValueError(f"{path}: incomplete motion metadata; provide joint_names, body_names and quaternion_order")
        return "metadata"

    joint_shape = data["joint_pos"].shape if "joint_pos" in data else ()
    body_shape = data["body_pos_w"].shape if "body_pos_w" in data else ()
    valid_joints = len(joint_shape) == 2 and joint_shape[1] == 23
    valid_bodies = len(body_shape) == 3 and body_shape[1:] in ((24, 3), (25, 3))

    if not valid_joints or not valid_bodies:
        raise ValueError(f"{path}: unrecognized unnamed motion; expected legacy K1 23 joints and 24/25 bodies")

    if body_shape[1] == 24:
        source_bodies = _LEGACY_K1_BODY_NAMES
    else:
        source_bodies = _LEGACY_K1_BODY_NAMES_WITH_HEAD

    data["joint_names"] = np.asarray(_LEGACY_K1_JOINT_NAMES)
    data["body_names"] = np.asarray(source_bodies)
    data["quaternion_order"] = np.asarray("wxyz")
    return "legacy K1 (assumed PhysX ordering)"


def _normalize_quaternions(data) -> str:
    """Convert quaternion arrays and metadata to xyzw; return the source order."""
    orders = []
    for key in ("quaternion_order", "quat_order"):
        if key in data:
            orders.append(str(np.asarray(data[key]).item()))

    if len(set(orders)) > 1:
        raise ValueError("Conflicting motion quaternion_order and quat_order metadata")

    order = orders[0]

    if order not in ("wxyz", "xyzw"):
        raise ValueError(f"Unsupported motion quaternion order: {order}")

    if "body_quat_w" in data and order == "wxyz":
        data["body_quat_w"] = data["body_quat_w"][..., [1, 2, 3, 0]]

    data["quaternion_order"] = np.asarray("xyzw")

    if "quat_order" in data:
        data["quat_order"] = np.asarray("xyzw")
    return order


def _reorder_motion(data, joint_names, body_names) -> None:
    """Validate array columns and map source names to the articulation order."""
    joint_fields = ("joint_pos", "joint_vel")
    body_fields = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    groups = (
        ("joint_names", joint_names, joint_fields),
        ("body_names", body_names, body_fields),
    )

    for names_key, target_names, fields in groups:
        names = np.asarray(data[names_key])

        if names.ndim != 1:
            raise ValueError(f"Motion {names_key} must be a one-dimensional name array")

        source_names = names.tolist()

        if len(source_names) != len(set(source_names)):
            raise ValueError(f"Motion {names_key} contain duplicate names: {source_names}")

        for field in fields:
            if field not in data:
                continue

            values = data[field]

            if values.ndim < 2 or values.shape[1] != len(source_names):
                raise ValueError(f"{field} count does not match motion {names_key}")

        if target_names is None:
            continue

        target_names = list(target_names)
        source_set = set(source_names)
        target_set = set(target_names)

        if len(target_names) != len(target_set) or not target_set.issubset(source_set):
            raise ValueError(f"Motion {names_key} do not match the articulation: {source_names}")

        # All joints must match. Extra bodies (e.g. a merged head) can be omitted.
        if names_key == "joint_names" and source_set != target_set:
            raise ValueError(f"Motion {names_key} do not match the articulation: {source_names}")

        indices = [source_names.index(name) for name in target_names]
        for field in fields:
            if field in data:
                data[field] = data[field][:, indices]
        data[names_key] = np.asarray(target_names)
