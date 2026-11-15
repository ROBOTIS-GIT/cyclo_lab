# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Differentiable wxyz math for original SuperMimic equations, no simulator imports."""

import torch


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat(
        [
            aw * bw - (av * bv).sum(-1, keepdim=True),
            aw * bv + bw * av + torch.linalg.cross(av, bv),
        ],
        -1,
    )


def quat_inv(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1) / q.square().sum(
        -1, keepdim=True
    ).clamp_min(1e-12)


def quat_apply(q, v):
    qv, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = torch.linalg.cross(qv, v)
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(qv, uv))


def quat_apply_inverse(q, v):
    return quat_apply(quat_inv(q), v)


def matrix_from_quat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).unflatten(-1, (3, 3))


def subtract_frame_transforms(p1, q1, p2, q2):
    return quat_apply_inverse(q1, p2 - p1), quat_mul(quat_inv(q1), q2)


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) / 2
    zeros = torch.zeros_like(yaw)
    return torch.stack([yaw.cos(), zeros, zeros, yaw.sin()], -1)
