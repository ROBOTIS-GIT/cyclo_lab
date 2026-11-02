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
# Author: Sunghyun Park

"""Fingertip tactile sensing for the HX5-D20 hand, from PhysX contact data."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.sensors import SensorBase
from isaaclab.sim.utils.queries import (
    path_expr_to_glob,
    resolve_matching_prims_from_source,
)
from isaaclab.utils.math import matrix_from_quat
from isaaclab_physx.physics import PhysxManager
from pxr import Usd, UsdGeom, UsdPhysics

from .fingertip_pads import (
    CELL_CENTERS,
    CELLS_PER_PAD,
    FINGER_NAMES,
    PadGrid,
    cell_shares,
    pad_poses_in_body,
    tip_body_names,
)
from .hand_tactile_data import HandTactileData
from .heatmap import TactileHeatmap
from .pad_overlay import PadOverlay

if TYPE_CHECKING:
    from .hand_tactile_cfg import HandTactileCfg

logger = logging.getLogger(__name__)

MAX_READING = 255.0
"""Largest reading of a cell, as on the real sensor."""


class HandTactile(SensorBase):
    """Fingertip tactile sensor of an HX5-D20 hand.

    Each fingertip pad is divided into the 3x3 cells of the real sensor. At every update, the force with which each
    contact point on a pad presses into it is shared among the cells, most of it in the cell under the point (see
    :attr:`HandTactileCfg.spread_sigma`), and summed per cell. The forces come from simulated contacts and are not
    calibrated against the real sensor. Each cell's force is then turned into a reading from 0 to 255, the range of the
    real sensor's readings, through the rough curve set by :attr:`HandTactileCfg.reading_scale` and
    :attr:`HandTactileCfg.reading_exponent`.

    PhysX reports the contacts of the fingertips with every body they can touch: each rigid body and each static
    collider in the scene, found when the simulation starts. A body added while it runs is seen from the next
    start.

    The fingertip bodies need the PhysX contact reporter: enable
    :attr:`isaaclab.sim.spawners.RigidObjectSpawnerCfg.activate_contact_sensors` in the robot's spawn
    configuration. The sensor reads PhysX directly, so it runs on the PhysX backend only.
    """

    cfg: HandTactileCfg
    """The configuration parameters."""

    def __init__(self, cfg: HandTactileCfg):
        """Initializes the tactile sensor.

        Args:
            cfg: The configuration parameters.
        """
        self._heatmap: TactileHeatmap | None = None
        self._overlay: PadOverlay | None = None
        self._warned_full_buffer = False
        super().__init__(cfg)
        # PhysX gives no contact data while contact processing is disabled.
        get_settings_manager().set_bool("/physics/disableContactProcessing", False)
        self._data = HandTactileData()

    def __del__(self):
        if self._heatmap is not None and not sys.is_finalizing():
            self._heatmap.destroy()
        super().__del__()

    def __str__(self) -> str:
        return (
            f"Hand tactile sensor @ '{self.cfg.prim_path}': \n"
            f"\thand              : {self.cfg.hand}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self.num_instances}"
        )

    @property
    def data(self) -> HandTactileData:
        self._update_outdated_buffers()
        return self._data

    @property
    def finger_names(self) -> tuple[str, ...]:
        """Fingers in the order of :attr:`HandTactileData.cell_forces`."""
        return FINGER_NAMES

    @property
    def heatmap_title(self) -> str:
        """Title of the heatmap window that ``debug_vis`` opens."""
        return f"{self.cfg.hand.capitalize()} hand tactile"

    def cell_pose_w(self, finger: int, cell: int, env_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Where a cell is in the world: the centre of the cell on the pad surface, and the pad's outward normal.

        Args:
            finger: Index of the finger, in the order of :attr:`finger_names`.
            cell: Index of the cell, from 0 to 8, numbered as in :attr:`HandTactileData.cell_forces`.
            env_index: Environment of the hand. Defaults to 0.

        Returns:
            The position and the unit normal, each of shape (3,).
        """
        rotations, origins = self._pad_poses_w()
        rotation = rotations[env_index, finger]
        y, z = CELL_CENTERS[cell]
        point = origins[env_index, finger] + rotation @ np.array([self._grids[finger].surface_at(cell), y, z])
        return point, rotation[:, 0]

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        super().reset(env_ids, env_mask)
        reset = wp.to_torch(self._resolve_indices_and_mask(env_ids, env_mask))
        self._data.cell_forces.torch[reset] = 0.0
        self._data.cell_readings.torch[reset] = 0.0
        if self._heatmap is not None and reset[self.cfg.heatmap.env_index]:
            self._heatmap.reset()

    def _initialize_impl(self):
        super()._initialize_impl()
        tips = _fingertips(self.cfg.prim_path, tip_body_names(self.cfg.hand))

        # Every environment holds the same hand, so the pad grids are built once, from the source environment.
        rotations, offsets = pad_poses_in_body(self.cfg.hand)
        self._grids = [
            PadGrid(_pad_triangles(tip, rotation, offset), self.cfg.depth_tolerance)
            for (tip, _), rotation, offset in zip(tips, rotations, offsets)
        ]
        self._pad_rotations = torch.tensor(rotations, dtype=torch.float32, device=self._device)
        self._pad_offsets = torch.tensor(offsets, dtype=torch.float32, device=self._device)

        # One pattern per fingertip, so both views are finger-major: entry = finger * num_envs + env.
        tip_globs = [glob for _, glob in tips]
        physics_sim_view = PhysxManager.get_physics_sim_view()
        self._tip_physx_view = physics_sim_view.create_rigid_body_view(tip_globs)
        if self._tip_physx_view.count != len(tips) * self._num_envs:
            raise RuntimeError(
                f"Expected {len(tips)} fingertips in each of {self._num_envs} environments under"
                f" '{self.cfg.prim_path}', found {self._tip_physx_view.count} in all."
            )

        # PhysX only gives contact points against listed bodies, so each fingertip lists everything it can touch.
        partners = _contact_partners(self.stage, *_environment_root(tips[0][0].GetPath().pathString, tip_globs[0]))
        self._contact_physx_view = physics_sim_view.create_rigid_contact_view(
            tip_globs,
            filter_patterns=[[partner for partner in partners if partner != tip] for tip in tip_globs],
            max_contact_data_count=self.cfg.max_contacts_per_pad * len(tips) * self._num_envs,
        )

        self._data.create_buffers(self._num_envs, self._device)
        self._timestamps = wp.to_torch(self._timestamp)
        self._timestamps_last_update = wp.to_torch(self._timestamp_last_update)

        # The pad drawing is added now, before the simulation runs, and debug_vis only shows or hides it: on CPU
        # physics, adding prims to the stage while the simulation runs sent rigid bodies back to their USD poses.
        if self._overlay is None and self.cfg.heatmap.pad_overlay and get_settings_manager().get("/isaaclab/has_gui"):
            self._overlay = PadOverlay(self.stage, f"/Visuals/HandTactile/{self.cfg.hand}", self._grids)
            self._overlay.set_visible(self._is_visualizing)

    def _update_buffers_impl(self, env_mask: wp.array):
        envs, fingers, positions, normals, contact_forces = self._fingertip_contacts()
        pad_rotations, pad_origins = self._pad_poses_w()
        rotations, origins = pad_rotations[envs, fingers], pad_origins[envs, fingers]
        # Contact points in their pad's frame, whose +X is the pad normal.
        positions = np.einsum("mi,mij->mj", positions - origins, rotations)
        alignments = np.abs(np.einsum("mi,mi->m", normals, rotations[:, :, 0]))
        # The pad responds to the part of the contact force along its normal. It reads no pull, and PhysX can report a
        # slightly negative force, so negative forces, and any that are not a number, count as 0.
        pressing_forces = np.fmax(contact_forces, 0.0) * alignments

        cells = np.full(len(envs), -1)
        for finger, grid in enumerate(self._grids):
            on_finger = fingers == finger
            cells[on_finger] = grid.cell_of(positions[on_finger])
        pressing = (cells >= 0) & (alignments >= self.cfg.min_normal_alignment)

        forces = np.zeros((self._num_envs, len(FINGER_NAMES), CELLS_PER_PAD), dtype=np.float32)
        if self.cfg.spread_sigma > 0.0:
            shares = cell_shares(positions[pressing], self.cfg.spread_sigma)
            np.add.at(forces, (envs[pressing], fingers[pressing]), pressing_forces[pressing, None] * shares)
        else:
            np.add.at(forces, (envs[pressing], fingers[pressing], cells[pressing]), pressing_forces[pressing])
        readings = np.minimum(self.cfg.reading_scale * forces**self.cfg.reading_exponent, MAX_READING, dtype=np.float32)
        updated = wp.to_torch(env_mask)
        self._data.cell_forces.torch[updated] = torch.from_numpy(forces).to(self._device)[updated]
        self._data.cell_readings.torch[updated] = torch.from_numpy(readings).to(self._device)[updated]

        shown_env = self.cfg.heatmap.env_index
        if self._heatmap is not None and updated[shown_env]:
            dt = float(self._timestamps[shown_env] - self._timestamps_last_update[shown_env])
            self._heatmap.update(readings[shown_env], dt)

    def _pad_poses_w(self) -> tuple[np.ndarray, np.ndarray]:
        """World rotation and origin of every pad, of shapes (N, 5, 3, 3) and (N, 5, 3)."""
        # Each pose is (x, y, z, qx, qy, qz, qw), the quaternion order Isaac Lab uses.
        tip_poses = wp.to_torch(self._tip_physx_view.get_transforms())
        tip_poses = tip_poses.view(len(FINGER_NAMES), self._num_envs, 7).transpose(0, 1)
        tip_rotations = matrix_from_quat(tip_poses[..., 3:])
        rotations = tip_rotations @ self._pad_rotations
        origins = tip_poses[..., :3] + (tip_rotations @ self._pad_offsets.unsqueeze(-1)).squeeze(-1)
        return rotations.cpu().numpy(), origins.cpu().numpy()

    def _fingertip_contacts(self) -> tuple[np.ndarray, ...]:
        """Contact points on the fingertips in the latest physics step.

        Returns:
            The environment index, finger index, position, normal and normal force of each contact point.
            Positions and normals are in the world frame.
        """
        contact_data = self._contact_physx_view.get_contact_data(dt=self._sim_physics_dt)
        forces, points, normals, _, counts, starts = (wp.to_torch(values) for values in contact_data)
        # One entry per (fingertip, partner) pair. The contact points of a pair are stored together from its start.
        counts, starts = counts.view(-1).long(), starts.view(-1).long()
        pairs = torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), counts)
        first_of_pair = (counts.cumsum(0) - counts).repeat_interleave(counts)
        index = starts[pairs] + torch.arange(pairs.numel(), device=counts.device) - first_of_pair
        # The buffer holds max_contact_data_count points. Points that PhysX had no room for have indices past its end.
        kept = index < points.shape[0]
        if not self._warned_full_buffer and not bool(kept.all()):
            self._warned_full_buffer = True
            logger.warning(
                f"The {self.cfg.hand} fingertips touched at {pairs.numel()} contact points, more than the"
                f" {points.shape[0]} the contact buffer holds, so some contacts are left out. Raise"
                " HandTactileCfg.max_contacts_per_pad."
            )
        pairs, index = pairs[kept], index[kept]
        tips = pairs // self._contact_physx_view.filter_count
        envs, fingers = tips % self._num_envs, tips // self._num_envs
        contacts = envs, fingers, points[index], normals[index], forces.view(-1)[index]
        return tuple(values.cpu().numpy() for values in contacts)

    """
    Debug visualization.
    """

    def set_heatmap_readings(self, show: bool):
        """Shows or hides each cell's reading, from 0 to 255, and the color scale in the heatmap window."""
        if self._heatmap is None:
            self.cfg.heatmap.show_readings = show
        else:
            self._heatmap.set_show_readings(show)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and self._heatmap is None:
            # The right hand's window takes the top right corner and the left hand's sits beside it, as seen from
            # behind the robot.
            slot = 0 if self.cfg.hand == "right" else 1
            self._heatmap = TactileHeatmap(self.cfg.heatmap, title=self.heatmap_title, slot=slot)
        if self._heatmap is not None:
            self._heatmap.set_visible(debug_vis)
        if self._overlay is not None:
            self._overlay.set_visible(debug_vis)

    def _debug_vis_callback(self, event):
        if self._heatmap is None:
            return
        self._heatmap.draw()
        if self._overlay is not None and self.is_initialized:
            rotations, origins = self._pad_poses_w()
            shown = self.cfg.heatmap.env_index
            self._overlay.draw(rotations[shown], origins[shown], self._heatmap.levels())

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._tip_physx_view = None
        self._contact_physx_view = None


def _fingertips(robot_path_expr: str, body_names: Sequence[str]) -> list[tuple[Usd.Prim, str]]:
    """The fingertip rigid bodies under the robot, in the order of ``body_names``.

    Returns:
        For each fingertip, its prim in the source environment and a glob of its path in every environment.
    """
    matches = resolve_matching_prims_from_source(
        robot_path_expr, lambda prim: prim.GetName() in body_names and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    tips = []
    for name in body_names:
        found = [(prim, path_expr) for prim, path_expr in matches if prim.GetName() == name]
        if len(found) != 1:
            raise RuntimeError(f"Expected one rigid body '{name}' under '{robot_path_expr}', found {len(found)}.")
        prim, path_expr = found[0]
        if "PhysxContactReportAPI" not in prim.GetAppliedSchemas():
            raise RuntimeError(
                f"The fingertip '{prim.GetPath()}' has no contact reporter."
                "\nHINT: Enable 'activate_contact_sensors' in the robot's spawn configuration."
            )
        tips.append((prim, path_expr_to_glob(path_expr)))
    return tips


def _environment_root(source_path: str, glob: str) -> tuple[str | None, str | None]:
    """The environment root in a source-environment path, and its glob over all environments.

    Both are None when the path is not inside a per-environment namespace.
    """
    for depth, (source_part, glob_part) in enumerate(zip(source_path.split("/"), glob.split("/"))):
        if source_part != glob_part:
            return "/".join(source_path.split("/")[: depth + 1]), "/".join(glob.split("/")[: depth + 1])
    return None, None


def _contact_partners(stage: Usd.Stage, env_root: str | None, env_glob: str | None) -> list[str]:
    """Path patterns of every body a fingertip can touch.

    Bodies of the source environment become globs over all environments. Bodies outside the environments, such
    as a ground plane, are shared by all of them and keep their own path.
    """
    env_parent = env_root.rsplit("/", 1)[0] + "/" if env_root is not None else None
    partners = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not _is_contact_body(prim):
            continue
        path = prim.GetPath().pathString
        if env_root is not None and path.startswith(env_root + "/"):
            partners.append(env_glob + path[len(env_root) :])
        elif env_parent is None or not path.startswith(env_parent):
            partners.append(path)
    return partners


def _is_contact_body(prim: Usd.Prim) -> bool:
    """Whether PhysX simulates the prim as a body: a rigid body, or a collider outside any rigid body."""
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return True
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        return False
    ancestor = prim.GetParent()
    while ancestor:
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            return False
        ancestor = ancestor.GetParent()
    return True


def _pad_triangles(body: Usd.Prim, rotation: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Triangles of the body's collision meshes in the pad frame. Shape is (T, 3, 3)."""
    world_to_body = np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse())
    triangles = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        mesh = UsdGeom.Mesh(prim)
        mesh_to_world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        # USD matrices act on row vectors: [p, 1] @ M.
        points = (np.c_[points, np.ones(len(points))] @ mesh_to_world @ world_to_body)[:, :3]
        triangles.append(points[_triangle_indices(mesh)])
    if not triangles:
        raise RuntimeError(f"The fingertip '{body.GetPath()}' has no collision mesh.")
    return (np.concatenate(triangles) - offset) @ rotation


def _triangle_indices(mesh: UsdGeom.Mesh) -> np.ndarray:
    """Vertex indices of the mesh faces, fan-triangulated. Shape is (T, 3)."""
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    if (counts == 3).all():
        return indices.reshape(-1, 3)
    triangles = []
    for start, count in zip(np.cumsum(counts) - counts, counts):
        face = indices[start : start + count]
        triangles.extend((face[0], face[k], face[k + 1]) for k in range(1, count - 1))
    return np.array(triangles)
