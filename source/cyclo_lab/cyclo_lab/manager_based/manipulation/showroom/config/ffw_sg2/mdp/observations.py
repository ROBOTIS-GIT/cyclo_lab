"""Observation helpers for the canonical SG2 showroom environment."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv


def joint_pos_name(env: ManagerBasedEnv, joint_names: tuple[str, ...], asset_name: str = "robot") -> torch.Tensor:
    asset: Articulation = env.scene[asset_name]
    joint_ids = [asset.joint_names.index(name) for name in joint_names]
    return asset.data.joint_pos[:, joint_ids]


def joint_pos_target_name(
    env: ManagerBasedEnv,
    joint_names: tuple[str, ...],
    asset_name: str = "robot",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_name]
    joint_ids = [asset.joint_names.index(name) for name in joint_names]
    return asset.data.joint_pos_target[:, joint_ids]


def base_twist(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    asset: Articulation = env.scene[asset_name]
    return torch.cat([asset.data.root_lin_vel_b[:, 0:2], asset.data.root_ang_vel_b[:, 2:3]], dim=-1)
