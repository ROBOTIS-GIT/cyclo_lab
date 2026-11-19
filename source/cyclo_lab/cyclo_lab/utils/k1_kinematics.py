# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""URDF kinematics with differentiable joint/root pose and world link twists.

Uses the same URDF as Isaac's K1 asset. Joint order comes from the live
articulation; URDF document order is never assumed to match PhysX order.
"""

import math
import xml.etree.ElementTree as ET
import torch
from cyclo_lab.manager_based.mimic.supermimic.core import quat_apply, quat_mul


def rpy_quat(rpy):
    r, p, y = [float(x) / 2 for x in rpy.split()]
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


class UrdfKinematics:
    def __init__(self, path, joint_names, body_names, device="cpu"):
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        nodes = ET.parse(path).getroot().findall("joint")
        available = {"pelvis"}
        self.edges = []
        while nodes:
            selected = [n for n in nodes if n.find("parent").get("link") in available]
            if not selected:
                raise ValueError("URDF is disconnected from pelvis")
            for n in selected:
                parent, child = (
                    n.find("parent").get("link"),
                    n.find("child").get("link"),
                )
                kind = n.get("type")
                if kind not in ("fixed", "revolute", "continuous"):
                    raise ValueError(f"Unsupported URDF joint type {kind}")
                origin = n.find("origin")
                xyz = (
                    [float(x) for x in origin.get("xyz", "0 0 0").split()]
                    if origin is not None
                    else [0] * 3
                )
                q = (
                    rpy_quat(origin.get("rpy", "0 0 0"))
                    if origin is not None
                    else (1, 0, 0, 0)
                )
                axis = (
                    [float(x) for x in n.find("axis").get("xyz").split()]
                    if kind != "fixed"
                    else [1, 0, 0]
                )
                if kind != "fixed" and n.get("name") not in self.joint_names:
                    raise ValueError(f"Missing articulated joint {n.get('name')}")
                index = (
                    self.joint_names.index(n.get("name")) if kind != "fixed" else None
                )
                constants = [
                    torch.tensor(a, dtype=torch.float32, device=device)
                    for a in (xyz, q, axis)
                ]
                self.edges.append((parent, child, index, *constants))
                available.add(child)
                nodes.remove(n)
        if set(self.body_names) - available:
            raise ValueError("Tracked body names not in URDF")

    def __call__(
        self, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel
    ):
        state = {"pelvis": (root_pos, root_quat, root_lin_vel, root_ang_vel)}
        for parent, child, index, xyz, quat, axis in self.edges:
            p, q, v, w = state[parent]
            offset = quat_apply(q, xyz.expand_as(p))
            p2 = p + offset
            q2 = quat_mul(q, quat.expand_as(q))
            v2 = v + torch.linalg.cross(w, offset)
            if index is not None:
                world_axis = quat_apply(q2, axis.expand_as(p))
                angle = joint_pos[..., index : index + 1] / 2
                rotation = torch.cat([angle.cos(), angle.sin() * axis], -1)
                q2 = quat_mul(q2, rotation)
                w2 = w + world_axis * joint_vel[..., index : index + 1]
            else:
                w2 = w
            state[child] = p2, q2, v2, w2
        return tuple(
            torch.stack([state[n][i] for n in self.body_names], -2) for i in range(4)
        )
