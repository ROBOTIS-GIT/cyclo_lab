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

"""Where the tactile pads sit on the HX5-D20 fingertips, and which cell a contact falls in.

Each pad has its own frame: +X points out of the pad, the 3x3 grid lies in the Y-Z plane, and +Z runs
along the finger towards its tip.
"""

from __future__ import annotations

import numpy as np

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
"""Fingers in the order the sensor reports them."""

CELLS_PER_PAD = 9

# The pitch and the grid centre were estimated from a photo of the real sensor and the fingertip mesh.
GRID_PITCH = 0.005
"""Side of one cell, in m."""

GRID_CENTER_Z = 0.02725
"""Distance of the grid centre from the fingertip body origin along the pad, in m."""

GRID_HALF_SIZE = 1.5 * GRID_PITCH

CELL_CENTERS = np.array(
    [((column - 1) * GRID_PITCH, GRID_CENTER_Z + (1 - row) * GRID_PITCH) for row in range(3) for column in range(3)]
)
"""(y, z) of each cell centre in the pad frame. Cell 0 is at the fingertip end, on the -Y side."""

# On the four fingers the pad faces the fingertip body's +X, so the pad frame is the body frame. The thumb
# pad faces the body's +Z and sits 15.5 mm along it; the left thumb body is mirrored.
_THUMB_PAD_AXES = {
    "right": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "left": ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
}
_THUMB_PAD_OFFSET = {"right": (0.0, 0.0155, 0.0), "left": (0.0, -0.0155, 0.0)}


def cell_shares(points: np.ndarray, sigma: float) -> np.ndarray:
    """Share of a contact's force that each cell takes, by the distance from the contact to the cell's centre.

    Args:
        points: Contact points in the pad frame. Shape is (M, 3).
        sigma: Standard deviation of the Gaussian, in m.

    Returns:
        The shares, which add up to 1 for each point. Shape is (M, 9).
    """
    squared_distances = ((points[:, None, 1:] - CELL_CENTERS) ** 2).sum(axis=-1)
    # Measured from the nearest centre, so that a narrow curve still leaves that cell its share.
    squared_distances -= squared_distances.min(axis=1, keepdims=True)
    shares = np.exp(-squared_distances / (2.0 * sigma**2))
    return shares / shares.sum(axis=1, keepdims=True)


def tip_body_names(hand: str) -> tuple[str, ...]:
    """Names of the five fingertip rigid bodies, thumb first."""
    side = _side_letter(hand)
    return tuple(f"finger_{side}_link{number}" for number in (4, 8, 12, 16, 20))


def pad_poses_in_body(hand: str) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and offset of each pad in its fingertip body frame.

    Returns:
        Rotations of shape (5, 3, 3), whose columns are the pad axes, and offsets of shape (5, 3).
    """
    _side_letter(hand)
    rotations = np.repeat(np.eye(3)[None], len(FINGER_NAMES), axis=0)
    offsets = np.zeros((len(FINGER_NAMES), 3))
    rotations[0] = np.array(_THUMB_PAD_AXES[hand]).T
    offsets[0] = _THUMB_PAD_OFFSET[hand]
    return rotations, offsets


class PadGrid:
    """The 3x3 grid on one fingertip pad.

    The grid is laid over the front of the fingertip's collision mesh. A contact belongs to a cell when it
    lies over that cell and no deeper than ``depth_tolerance`` below the pad surface. Contacts may lie above
    the surface: PhysX collides with the convex hull it cooks from the mesh, which on the HX5-D20 fingertips
    lies up to about 1.2 mm outside the mesh.
    """

    def __init__(self, triangles: np.ndarray, depth_tolerance: float, resolution: float = 1e-4):
        """Samples the pad surface under the grid.

        Args:
            triangles: Collision mesh triangles in the pad frame. Shape is (T, 3, 3).
            depth_tolerance: Largest depth below the pad surface at which a contact still counts, in m.
            resolution: Spacing of the surface samples, in m.

        Raises:
            ValueError: If part of the grid is not over the fingertip mesh.
        """
        self.depth_tolerance = depth_tolerance
        self._samples_y = np.arange(-GRID_HALF_SIZE, GRID_HALF_SIZE + resolution / 2, resolution)
        self._samples_z = GRID_CENTER_Z + self._samples_y
        self._surface_x = _front_surface(triangles, self._samples_y, self._samples_z)
        if np.isnan(self._surface_x).any():
            raise ValueError("The tactile grid extends past the fingertip surface.")

    def cell_of(self, points: np.ndarray) -> np.ndarray:
        """Cell index of each pad-frame point, or -1 for points off the grid or too deep below the pad surface.

        Args:
            points: Points in the pad frame. Shape is (M, 3).

        Returns:
            Cell indices from 0 to 8, row by row from the fingertip end. Shape is (M,).
        """
        finite = np.isfinite(points).all(axis=1)
        x, y, z = np.where(finite[:, None], points, 0.0).T
        on_grid = finite & (np.abs(y) <= GRID_HALF_SIZE) & (np.abs(z - GRID_CENTER_Z) <= GRID_HALF_SIZE)

        surface_x = self._surface_x[_nearest(self._samples_y, y), _nearest(self._samples_z, z)]
        on_pad = x >= surface_x - self.depth_tolerance

        row = np.clip(((GRID_CENTER_Z + GRID_HALF_SIZE - z) // GRID_PITCH).astype(int), 0, 2)
        column = np.clip(((y + GRID_HALF_SIZE) // GRID_PITCH).astype(int), 0, 2)
        return np.where(on_grid & on_pad, 3 * row + column, -1)

    def surface_x(self, y: float, z: float) -> float:
        """X of the pad surface over the point (y, z) of the grid, in the pad frame, in m."""
        return float(self._surface_x[_nearest(self._samples_y, y), _nearest(self._samples_z, z)])

    def surface_at(self, cell: int) -> float:
        """X of the pad surface over the centre of a cell, in the pad frame, in m."""
        return self.surface_x(*CELL_CENTERS[cell])


def _front_surface(triangles: np.ndarray, samples_y: np.ndarray, samples_z: np.ndarray) -> np.ndarray:
    """X of the front-most mesh surface over each (y, z) sample, or NaN where no triangle covers it."""
    surface = np.full((len(samples_y), len(samples_z)), -np.inf)
    lower, upper = triangles[:, :, 1:].min(axis=1), triangles[:, :, 1:].max(axis=1)
    overlaps = (upper[:, 0] >= samples_y[0]) & (lower[:, 0] <= samples_y[-1])
    overlaps &= (upper[:, 1] >= samples_z[0]) & (lower[:, 1] <= samples_z[-1])

    for a, b, c in triangles[overlaps]:
        # Barycentric coordinates of the samples in the triangle as seen along X.
        denominator = (b[2] - c[2]) * (a[1] - c[1]) + (c[1] - b[1]) * (a[2] - c[2])
        if abs(denominator) < 1e-16:
            continue  # seen edge-on from the front
        in_y = _samples_between(samples_y, min(a[1], b[1], c[1]), max(a[1], b[1], c[1]))
        in_z = _samples_between(samples_z, min(a[2], b[2], c[2]), max(a[2], b[2], c[2]))
        y, z = np.meshgrid(samples_y[in_y] - c[1], samples_z[in_z] - c[2], indexing="ij")
        u = ((b[2] - c[2]) * y + (c[1] - b[1]) * z) / denominator
        v = ((c[2] - a[2]) * y + (a[1] - c[1]) * z) / denominator
        w = 1.0 - u - v
        inside = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
        x = u * a[0] + v * b[0] + w * c[0]
        np.maximum(surface[in_y, in_z], np.where(inside, x, -np.inf), out=surface[in_y, in_z])

    surface[np.isneginf(surface)] = np.nan
    return surface


def _nearest(samples: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of the evenly spaced sample nearest to each value."""
    spacing = samples[1] - samples[0]
    return np.clip(np.rint((values - samples[0]) / spacing).astype(int), 0, len(samples) - 1)


def _samples_between(samples: np.ndarray, low: float, high: float, margin: float = 1e-9) -> slice:
    """The slice of sorted ``samples`` that lie within [low, high]."""
    start = np.searchsorted(samples, low - margin, side="left")
    stop = np.searchsorted(samples, high + margin, side="right")
    return slice(start, stop)


def _side_letter(hand: str) -> str:
    if hand not in ("left", "right"):
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}.")
    return hand[0]
