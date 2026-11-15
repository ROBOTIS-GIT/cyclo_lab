# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Native URDF constants for the official SuperMimic analytic Torch IK.

The chain reduction follows HybridRobotics/supermimic analytic_ik._reduce
(abc124c1, Apache-2.0). Only topology extraction is replaced. One-hinge K1
chains use an orthogonal auxiliary axis whose unused joint slot is discarded;
the shared core validates constants and supplies the unchanged tensor solver.
"""

import xml.etree.ElementTree as ET
import torch
from supermimic.utils.math import matrix_from_quat
from supermimic.utils.analytic_ik import AnalyticIK
from .fk import rpy_quat


def build_urdf_analytic_ik(
    path, joint_names, body_names, device="cpu", dtype=torch.float32
):
    if len(set(body_names)) != len(body_names) or body_names[0] != "pelvis":
        raise ValueError("IK requires unique tracked bodies rooted at pelvis")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("IK requires unique articulation joint names")
    joints = ET.parse(path).getroot().findall("joint")
    edge = {n.find("child").get("link"): n for n in joints}
    row_of = {name: row for row, name in enumerate(body_names)}
    col_of = {name: col for col, name in enumerate(joint_names)}
    constants = []
    for body in body_names[1:]:
        chain, ancestor = [], body
        while True:
            node = edge[ancestor]
            chain.append(node)
            ancestor = node.find("parent").get("link")
            if ancestor in row_of:
                break
        y = torch.eye(3, dtype=torch.float64)
        axes, moving = [], []
        for node in reversed(chain):
            origin = node.find("origin")
            rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            y = y @ matrix_from_quat(torch.tensor(rpy_quat(rpy), dtype=torch.float64))
            if node.get("type") != "fixed":
                if node.get("type") != "revolute":
                    raise ValueError("Native IK supports bounded revolute joints")
                axes.append(
                    y
                    @ torch.tensor(
                        [float(v) for v in node.find("axis").get("xyz").split()],
                        dtype=torch.float64,
                    )
                )
                moving.append(node)
        k = len(moving)
        if k not in (1, 2, 3):
            raise ValueError(f"Unsupported tracked IK chain to {body}: {k} hinges")
        if k == 1:
            # Pick the least-aligned Cartesian axis, then Gram-Schmidt.
            seed = torch.eye(3, dtype=torch.float64)[axes[0].abs().argmin()]
            auxiliary = seed - (seed @ axes[0]) * axes[0]
            axes.append(auxiliary / auxiliary.norm())
        x = torch.stack([axes[0], axes[1], torch.linalg.cross(axes[0], axes[1])])
        if not torch.allclose(
            x @ x.T, torch.eye(3, dtype=torch.float64), atol=1e-8, rtol=0
        ):
            raise ValueError(f"Non-orthogonal IK chain to {body}")
        sign, qpos0, lo, hi = torch.zeros(4, 3, dtype=torch.float64)
        sign[:k] = 1
        if k == 3:
            alignment = float(axes[2] @ x[2])
            if abs(abs(alignment) - 1) > 1e-8:
                raise ValueError(f"Unsupported third IK axis to {body}")
            sign[2] = round(alignment)
        for i, node in enumerate(moving):
            lo[i], hi[i] = (
                float(node.find("limit").get("lower")),
                float(node.find("limit").get("upper")),
            )
        cols = [col_of[node.get("name")] for node in moving] + [-1] * (3 - k)
        constants.append((row_of[ancestor], x.T, (x @ y).T, sign, qpos0, lo, hi, cols))
    parent, a, b, sign, qpos0, lo, hi, cols = zip(*constants, strict=True)
    parent_tensor = torch.tensor(parent, device=device)
    child_tensor = torch.arange(1, len(body_names), device=device)
    A, B, signs, neutral, lower, upper = (
        torch.stack(t).to(device=device, dtype=dtype)
        for t in (a, b, sign, qpos0, lo, hi)
    )
    slot_col = torch.tensor(cols).flatten()
    if sorted(slot_col[slot_col >= 0].tolist()) != list(range(len(joint_names))):
        raise ValueError(
            "Tracked IK chains must cover each articulation joint exactly once"
        )
    perm = slot_col.argsort()[slot_col.lt(0).sum() :].to(device)
    return AnalyticIK.from_constants(
        parent=parent_tensor,
        child=child_tensor,
        A=A,
        B=B,
        sign=signs,
        qpos0=neutral,
        lo=lower,
        hi=upper,
        perm=perm,
    )
