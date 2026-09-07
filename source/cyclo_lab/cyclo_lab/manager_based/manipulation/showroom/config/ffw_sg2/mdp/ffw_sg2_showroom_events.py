"""Reset events for the continuous SG2 showroom environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def create_joint_position_mapping(joint_names: list[str], desired_values: dict[str, float]) -> torch.Tensor:
    """Create a joint-position tensor ordered by the articulation joint names."""
    return torch.tensor([desired_values.get(joint_name, 0.0) for joint_name in joint_names], dtype=torch.float32)


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_positions: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set SG2 joint state and position target on reset."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = create_joint_position_mapping(asset.joint_names, joint_positions).to(device=env.device)
    if joint_pos.dim() == 1:
        joint_pos = joint_pos.unsqueeze(0).repeat(len(env_ids), 1)
    joint_vel = torch.zeros_like(joint_pos)

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_root_pose_in_radius(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_translation_radius: float,
    max_yaw: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize each robot root uniformly inside a circle around its default pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pose = asset.data.default_root_state[env_ids, :7].clone()

    # sqrt produces a uniform spatial distribution over the disk area.
    samples = torch.rand((len(env_ids), 3), device=asset.device)
    radius = max_translation_radius * torch.sqrt(samples[:, 0])
    azimuth = 2.0 * math.pi * samples[:, 1]
    root_pose[:, 0] += radius * torch.cos(azimuth) + env.scene.env_origins[env_ids, 0]
    root_pose[:, 1] += radius * torch.sin(azimuth) + env.scene.env_origins[env_ids, 1]
    root_pose[:, 2] += env.scene.env_origins[env_ids, 2]

    yaw = (2.0 * samples[:, 2] - 1.0) * max_yaw
    yaw_delta = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    root_pose[:, 3:7] = math_utils.quat_mul(root_pose[:, 3:7], yaw_delta)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)


def randomize_wall_background(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    texture_paths: tuple[str, ...],
    zoom_range: tuple[float, float],
):
    """Randomize the connected wall image and crop while keeping both walls fully covered."""
    if not texture_paths:
        raise ValueError("At least one showroom wall texture is required.")
    if not 1.0 <= zoom_range[0] <= zoom_range[1]:
        raise ValueError(f"Invalid showroom wall background zoom range: {zoom_range}")

    from pxr import Gf, Sdf

    samples = torch.rand((len(env_ids), 2), device=env.device)
    texture_indices = torch.floor(samples[:, 0] * len(texture_paths)).to(dtype=torch.long)
    zooms = zoom_range[0] + samples[:, 1] * (zoom_range[1] - zoom_range[0])

    for sample_index, env_id in enumerate(env_ids.detach().cpu().tolist()):
        shell_path = f"{env.scene.env_prim_paths[env_id]}/RobotisShowroom/ShowroomShell"
        texture_prim = env.scene.stage.GetPrimAtPath(f"{shell_path}/Looks/LabBackground/Texture")
        uv_transform = env.scene.stage.GetPrimAtPath(f"{shell_path}/Looks/LabBackground/UVTransform")
        back_panel = env.scene.stage.GetPrimAtPath(f"{shell_path}/WallBackground/BackPanel")
        left_panel = env.scene.stage.GetPrimAtPath(f"{shell_path}/WallBackground/LeftPanel")
        if (
            not texture_prim.IsValid()
            or not uv_transform.IsValid()
            or not back_panel.IsValid()
            or not left_panel.IsValid()
        ):
            raise RuntimeError(f"Showroom wall background prims are missing below {shell_path}")

        texture_index = int(texture_indices[sample_index])
        zoom = float(zooms[sample_index])
        uv_scale = 1.0 / zoom
        uv_offset = 0.5 * (1.0 - uv_scale)
        texture_prim.GetAttribute("inputs:file").Set(Sdf.AssetPath(texture_paths[texture_index]))
        uv_transform.GetAttribute("inputs:scale").Set(Gf.Vec2f(uv_scale, uv_scale))
        uv_transform.GetAttribute("inputs:translation").Set(Gf.Vec2f(uv_offset, uv_offset))
