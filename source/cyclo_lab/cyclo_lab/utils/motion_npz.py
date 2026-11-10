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
import warnings

import numpy as np


def load_motion_npz(path, joint_names: Sequence[str] | None = None, body_names: Sequence[str] | None = None):
    """Read named motion data; legacy NPZ files use wxyz and native articulation order."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    orders = [str(np.asarray(data[key]).item()) for key in ("quaternion_order", "quat_order") if key in data]
    if len(set(orders)) > 1:
        raise ValueError("Conflicting motion quaternion_order and quat_order metadata")
    order = orders[0] if orders else "wxyz"
    if order not in ("wxyz", "xyzw"):
        raise ValueError(f"Unsupported motion quaternion order: {order}")
    if "body_quat_w" in data and order == "wxyz":
        data["body_quat_w"] = data["body_quat_w"][..., [1, 2, 3, 0]]
    data["quaternion_order"] = np.asarray("xyzw")
    if "quat_order" in data:
        data["quat_order"] = np.asarray("xyzw")

    for key, target, fields in (
        ("joint_names", joint_names, ("joint_pos", "joint_vel")),
        ("body_names", body_names, ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")),
    ):
        if target is None:
            continue
        if key not in data:
            warnings.warn(f"{path}: no {key}; assuming legacy native articulation order.", stacklevel=2)
            for field in fields:
                if field in data and data[field].shape[1] != len(target):
                    raise ValueError(
                        f"{field} count does not match articulation {key}; regenerate this legacy NPZ from its source CSV"
                    )
            continue
        source = data[key].tolist()
        if len(source) != len(set(source)) or set(source) != set(target):
            raise ValueError(f"Motion {key} do not match the articulation: {source}")
        indices = [source.index(name) for name in target]
        for field in fields:
            if field in data:
                if data[field].shape[1] != len(source):
                    raise ValueError(f"{field} count does not match motion {key}")
                data[field] = data[field][:, indices]
        data[key] = np.asarray(target)
    return data
