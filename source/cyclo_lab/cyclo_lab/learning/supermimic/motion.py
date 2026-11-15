# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Strict, read-only contract for shared named Cyclo NPZ motion input."""

from pathlib import Path
import hashlib
import numpy as np


def real_float32(value, field):
    """Validate the actual learner dtype, without silently dropping complex parts."""
    value = np.asarray(value)
    if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
        raise ValueError(f"{field} must contain finite real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = value.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{field} is not finite in learner float32")
    return result


def validate_motion(path, expected_joint_names=None, expected_body_names=None, fps=50):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = (
            "fps",
            "joint_names",
            "body_names",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(
                f"Motion metadata/fields missing: {sorted(missing)}; unnamed legacy ordering is ambiguous"
            )
        orders = [
            str(np.asarray(data[k]).item())
            for k in ("quaternion_order", "quat_order")
            if k in data
        ]
        if not orders or len(set(orders)) != 1 or orders[0] not in ("xyzw", "wxyz"):
            raise ValueError(
                "Motion needs one consistent explicit quaternion_order: xyzw or wxyz"
            )
        rate = np.asarray(data["fps"])
        if (
            rate.size != 1
            or rate.dtype.kind not in "iuf"
            or not np.isfinite(rate).all()
            or float(rate.reshape(-1)[0]) != fps
        ):
            raise ValueError(
                f"Motion must already be {fps} Hz; input is not resampled or modified"
            )
        names = {}
        for key, expected in [
            ("joint_names", expected_joint_names),
            ("body_names", expected_body_names),
        ]:
            array = np.asarray(data[key])
            if array.ndim != 1 or not all(isinstance(x, str) for x in array.tolist()):
                raise ValueError(f"{key} must be explicit one-dimensional string names")
            value = array.tolist()
            if len(value) != len(set(value)):
                raise ValueError(f"{key} contains duplicates")
            if expected is not None:
                valid = (
                    set(expected) == set(value)
                    if key == "joint_names"
                    else set(expected) <= set(value)
                )
                if not valid:
                    raise ValueError(
                        f"{key} incompatible with the live K1 articulation"
                    )
            names[key] = value
        if len(names["joint_names"]) != 23 or names["body_names"].count("pelvis") != 1:
            raise ValueError("Expected named 23-joint K1 motion with one pelvis")
        if data["joint_pos"].ndim != 2:
            raise ValueError("joint_pos must be a frame-by-joint matrix")
        frames = data["joint_pos"].shape[0]
        if frames < 2:
            raise ValueError("Motion needs at least two frames")
        for key, width in [
            ("joint_pos", None),
            ("joint_vel", None),
            ("body_pos_w", 3),
            ("body_quat_w", 4),
            ("body_lin_vel_w", 3),
            ("body_ang_vel_w", 3),
        ]:
            value = np.asarray(data[key])
            expected = (
                (frames, 23)
                if width is None
                else (frames, len(names["body_names"]), width)
            )
            if value.shape != expected:
                raise ValueError(f"{key}: invalid shape or nonfinite motion data")
            real_float32(value, key)
        norms = np.linalg.norm(
            real_float32(data["body_quat_w"], "body_quat_w"), axis=-1
        )
        if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
            raise ValueError(
                "Motion body quaternions must be unit length; zero/nonunit quaternion rejected"
            )
        return dict(
            path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fps=fps,
            frames=frames,
            quaternion_order=orders[0],
            **names,
        )
