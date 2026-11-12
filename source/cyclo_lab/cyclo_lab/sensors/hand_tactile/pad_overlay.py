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

"""The 3x3 grid and cell numbers of each fingertip pad, drawn on the hand in the viewport, with the pressed cells lit.

Everything here is visual only: plain USD curves and meshes without physics. Each pad's drawing is authored once,
in the pad frame and following the curved pad surface, and is moved onto the pad at every frame.
"""

from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from .fingertip_pads import (
    CELL_CENTERS,
    CELLS_PER_PAD,
    FINGER_NAMES,
    GRID_CENTER_Z,
    GRID_HALF_SIZE,
    GRID_PITCH,
    PadGrid,
)
from .heatmap import cell_rgb

LINE_LIFT = 0.00035
"""Height of the grid lines and numbers above the pad surface, which keeps them from flickering into it, in m."""

GLOW_LIFT = 0.0002
"""Height of a lit cell above the pad surface, under the lines, in m."""

GLOW_INSET = 0.0003
"""Gap between a lit cell and the grid lines around it, in m."""

GLOW_FACETS = 4
"""Facets along each side of a lit cell, so that it follows the curved pad."""

GLOW_GAIN = 2.0
"""How much brighter than its heatmap color a lit cell glows. Higher values wash the hues out to white."""

BOUNDARY_COLOR = (1.0, 0.72, 0.08)
DIVIDER_COLOR = (0.12, 1.0, 0.30)
NUMBER_COLOR = (0.92, 0.98, 1.0)

# Seven-segment strokes write the cell numbers without fonts or textures. Coordinates are in (y, z), +z up.
_SEGMENTS = {
    "a": ((-1, 1), (1, 1)),
    "b": ((1, 1), (1, 0)),
    "c": ((1, 0), (1, -1)),
    "d": ((-1, -1), (1, -1)),
    "e": ((-1, 0), (-1, -1)),
    "f": ((-1, 1), (-1, 0)),
    "g": ((-1, 0), (1, 0)),
}
_DIGITS = ("bc", "abged", "abgcd", "fgbc", "afgcd", "afgecd", "abc", "abcdefg", "abfgcd")
_DIGIT_HALF_WIDTH, _DIGIT_HALF_HEIGHT = 0.0008, 0.0014


class PadOverlay:
    """The grids, numbers and lit cells of the five pads of one hand."""

    def __init__(self, stage: Usd.Stage, prim_path: str, grids: list[PadGrid]):
        self._visibility = UsdGeom.Xform.Define(stage, prim_path).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
        self._pads = [_PadDrawing(stage, f"{prim_path}/{name}", grid) for name, grid in zip(FINGER_NAMES, grids)]
        self._levels = np.full((len(FINGER_NAMES), CELLS_PER_PAD), -1)

    def set_visible(self, visible: bool):
        self._visibility.Set(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)

    def draw(self, rotations: np.ndarray, origins: np.ndarray, levels: np.ndarray):
        """Moves each drawing onto its pad and lights the cells whose level changed.

        Args:
            rotations: World rotation of each pad, whose columns are the pad axes. Shape is (5, 3, 3).
            origins: World origin of each pad. Shape is (5, 3).
            levels: Heatmap level of each cell, from 0 to 255. Shape is (5, 9).
        """
        for finger, pad in enumerate(self._pads):
            pad.place(rotations[finger], origins[finger])
            for cell in np.flatnonzero(levels[finger] != self._levels[finger]):
                level = levels[finger, cell]
                pad.light(cell, cell_rgb(level, finger) if level > 0 else None)
        self._levels = levels.copy()


class _PadDrawing:
    """One pad's grid lines, cell numbers and lit cells, authored in the pad frame."""

    def __init__(self, stage: Usd.Stage, path: str, grid: PadGrid):
        self._transform = UsdGeom.Xform.Define(stage, path).AddTransformOp()

        half, bottom, top = GRID_HALF_SIZE, GRID_CENTER_Z - GRID_HALF_SIZE, GRID_CENTER_Z + GRID_HALF_SIZE
        boundary = [((-half, bottom), (-half, top)), ((half, bottom), (half, top))]
        boundary += [((-half, bottom), (half, bottom)), ((-half, top), (half, top))]
        dividers = []
        for offset in (-0.5 * GRID_PITCH, 0.5 * GRID_PITCH):
            dividers += [
                ((offset, bottom), (offset, top)),
                ((-half, GRID_CENTER_Z + offset), (half, GRID_CENTER_Z + offset)),
            ]
        numbers = [
            tuple((y + u * _DIGIT_HALF_WIDTH, z + v * _DIGIT_HALF_HEIGHT) for u, v in _SEGMENTS[segment])
            for cell, (y, z) in enumerate(CELL_CENTERS)
            for segment in _DIGITS[cell]
        ]
        _surface_lines(stage, f"{path}/Boundary", grid, boundary, 0.00028, BOUNDARY_COLOR)
        _surface_lines(stage, f"{path}/Dividers", grid, dividers, 0.00016, DIVIDER_COLOR)
        _surface_lines(stage, f"{path}/Numbers", grid, numbers, 0.00020, NUMBER_COLOR)
        self._cells = [
            _cell_patch(stage, f"{path}/Cell{cell + 1}", grid, y, z) for cell, (y, z) in enumerate(CELL_CENTERS)
        ]

    def place(self, rotation: np.ndarray, origin: np.ndarray):
        # USD matrices act on row vectors, so the rotation goes in transposed.
        matrix = np.eye(4)
        matrix[:3, :3], matrix[3, :3] = rotation.T, origin
        self._transform.Set(Gf.Matrix4d(matrix.tolist()))

    def light(self, cell: int, rgb: tuple[int, int, int] | None):
        """Lights a cell in a 0-255 RGB color, or hides it for None."""
        visibility, emissive = self._cells[cell]
        if rgb is None:
            visibility.Set(UsdGeom.Tokens.invisible)
            return
        rgb = np.array(rgb) / 255.0
        glow = rgb * GLOW_GAIN
        # Scale back so that no channel passes 1, which would wash the hue out.
        glow /= max(1.0, glow.max())
        emissive.Set(Gf.Vec3f(*glow.tolist()))
        visibility.Set(UsdGeom.Tokens.inherited)


def _surface_lines(stage: Usd.Stage, path: str, grid: PadGrid, segments: list, width: float, color: tuple):
    """Straight segments of the pad's Y-Z plane, laid on the pad surface every 0.5 mm so that they follow its curve."""
    points, counts = [], []
    for (y0, z0), (y1, z1) in segments:
        steps = max(1, int(np.ceil(max(abs(y1 - y0), abs(z1 - z0)) / 0.0005)))
        for i in range(steps + 1):
            y, z = y0 + (y1 - y0) * i / steps, z0 + (z1 - z0) * i / steps
            points.append(Gf.Vec3f(grid.surface_x(y, z) + LINE_LIFT, y, z))
        counts.append(steps + 1)
    lines = UsdGeom.BasisCurves.Define(stage, path)
    lines.CreateTypeAttr(UsdGeom.Tokens.linear)
    lines.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    lines.CreatePointsAttr(points)
    lines.CreateCurveVertexCountsAttr(counts)
    lines.CreateWidthsAttr([width])
    lines.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    lines.CreateExtentAttr(_extent(points, width))
    _flat_material(stage, lines, color, emissive=color)


def _cell_patch(stage: Usd.Stage, path: str, grid: PadGrid, y: float, z: float):
    """A cell-sized patch on the pad surface, hidden until lit.

    Returns:
        The patch's visibility attribute and its material's emissive color input.
    """
    half, side = GRID_PITCH / 2 - GLOW_INSET, GLOW_FACETS + 1
    points = [
        Gf.Vec3f(grid.surface_x(point_y, point_z) + GLOW_LIFT, point_y, point_z)
        for point_z in np.linspace(z - half, z + half, side)
        for point_y in np.linspace(y - half, y + half, side)
    ]
    indices = []
    for row in range(GLOW_FACETS):
        for column in range(GLOW_FACETS):
            corner = row * side + column
            indices += [corner, corner + 1, corner + side + 1, corner + side]
    patch = UsdGeom.Mesh.Define(stage, path)
    patch.CreatePointsAttr(points)
    patch.CreateFaceVertexCountsAttr([4] * GLOW_FACETS**2)
    patch.CreateFaceVertexIndicesAttr(indices)
    patch.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    patch.CreateDoubleSidedAttr(True)
    patch.CreateExtentAttr(_extent(points))
    emissive = _flat_material(stage, patch, (0.0, 0.0, 0.0), emissive=(0.0, 0.0, 0.0))
    return patch.CreateVisibilityAttr(UsdGeom.Tokens.invisible), emissive


def _flat_material(stage: Usd.Stage, geometry: UsdGeom.Gprim, color: tuple, emissive: tuple) -> UsdShade.Input:
    """Binds a material that shows ``color`` in any light, and returns its emissive color input."""
    material = UsdShade.Material.Define(stage, f"{geometry.GetPath()}Material")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    emissive_input = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
    emissive_input.Set(Gf.Vec3f(*emissive))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)
    geometry.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return emissive_input


def _extent(points: list, pad: float = 0.0) -> list:
    """The bounding box of the points, grown by ``pad``. Gf takes Python floats only, hence ``tolist``."""
    corners = np.array(points, dtype=float)
    return [Gf.Vec3f(*(corners.min(axis=0) - pad).tolist()), Gf.Vec3f(*(corners.max(axis=0) + pad).tolist())]
