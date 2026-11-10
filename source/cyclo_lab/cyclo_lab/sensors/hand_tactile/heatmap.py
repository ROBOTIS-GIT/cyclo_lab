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

"""Debug heatmap of a hand's tactile cells, drawn like the real hand's tactile viewer.

The sensor's readings are shown the way the real viewer shows the real sensor's: smoothed over time, then colored
by ``sqrt(1.5 * reading) * 8`` up to ``color_max``. Only the picture is processed here; the sensor data is never
changed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .fingertip_pads import CELLS_PER_PAD, FINGER_NAMES

if TYPE_CHECKING:
    from .hand_tactile_cfg import TactileHeatmapCfg

logger = logging.getLogger(__name__)

# One hue per finger, thumb to little, as in the real hand's viewer.
FINGER_COLORS = ((255, 59, 48), (255, 149, 0), (255, 214, 10), (52, 199, 89), (0, 122, 255))
UNLIT_LEVEL = 11
# The real viewer shows a reading as sqrt(VIEWER_GAIN * reading) * 8, on a color scale that ends at color_max.
VIEWER_GAIN = 1.5

# omni.ui colors are 0xAABBGGRR.
_BACKGROUND = 0xFFD9D9D9
_GRID_LINES = 0xFF1F1F1F
_TEXT = 0xFF000000
_READING_ON_DARK = 0xFFE6E6E6


class TactileHeatmap:
    """A window with the five 3x3 grids of one hand.

    The window opens along the top right of the app window, clear of the viewport, in the place given by ``slot``:
    0 is the right edge, and each next slot is one window further left.
    """

    def __init__(self, cfg: TactileHeatmapCfg, title: str, slot: int = 0):
        self.cfg = cfg
        self._title = title
        self._slot = slot
        self._smoothed = np.zeros((len(FINGER_NAMES), CELLS_PER_PAD))
        self._painted = np.full(self._smoothed.shape, -1)
        self._written = np.full(self._smoothed.shape, -1)
        self._window = None
        self._cells = []
        self._readings = []
        self._scale = []

    def update(self, readings: np.ndarray, dt: float):
        """Moves the shown readings towards the sensor's.

        Args:
            readings: The sensor's reading of each cell, from 0 to 255. Shape is (5, 9).
            dt: Time since the last update, in s.
        """
        # A value that is not a number would stay in the smoothed readings for good and break the colors.
        readings = np.clip(np.nan_to_num(readings), 0.0, 255.0)
        time_constant = self.cfg.smoothing_time_constant
        alpha = -np.expm1(-dt / time_constant) if time_constant > 0.0 else 1.0
        self._smoothed += alpha * (readings - self._smoothed)

    def reset(self):
        self._smoothed[:] = 0.0

    def readings(self) -> np.ndarray:
        """Shown reading of each cell, from 0 to 255. Shape is (5, 9)."""
        return self._smoothed.copy()

    @property
    def full_color_reading(self) -> float:
        """The reading from which a cell shows in full color."""
        # cell_color takes the square root of the level, so this gives the viewer's scale, full at color_max.
        return self.cfg.color_max**2 / (64.0 * VIEWER_GAIN)

    def levels(self) -> np.ndarray:
        """Heatmap level of each cell, from 0 to 255. Shape is (5, 9)."""
        levels = np.rint(np.minimum(255.0, 255.0 * self._smoothed / self.full_color_reading)).astype(int)
        return np.where(self._smoothed < self.cfg.deadband_counts, 0, levels)

    def set_visible(self, visible: bool):
        if visible and self._window is None:
            self._build_window()
        if self._window is not None:
            self._window.visible = visible

    def set_show_readings(self, show: bool):
        """Shows or hides each cell's reading, from 0 to 255, and the color scale under the grids."""
        self.cfg.show_readings = show
        for label in [*self._scale, *(label for labels in self._readings for label in labels)]:
            label.visible = show
        self._written[:] = -1
        if self._window is not None:
            self._window.frame.invalidate_raster()

    def draw(self):
        """Repaint the cells whose level changed, and rewrite the readings that changed."""
        if self._window is None or not self._window.visible:
            return
        levels = self.levels()
        changed = np.argwhere(levels != self._painted)
        for finger, cell in changed:
            self._cells[finger][cell].set_style({"background_color": cell_color(levels[finger, cell], finger)})
            # Dark numbers on bright cells, light ones on dark cells.
            text = _TEXT if levels[finger, cell] > 128 else _READING_ON_DARK
            self._readings[finger][cell].set_style({"color": text, "font_size": self._reading_font_size})
        rewritten = []
        if self.cfg.show_readings:
            readings = np.rint(self.readings()).astype(int)
            rewritten = np.argwhere(readings != self._written)
            for finger, cell in rewritten:
                self._readings[finger][cell].text = str(readings[finger, cell])
            self._written = readings
        if len(changed) or len(rewritten):
            # A color or text change alone does not redraw the window's cached raster.
            self._window.frame.invalidate_raster()
        self._painted = levels

    def destroy(self):
        if self._window is not None:
            self._window.destroy()
            self._window = None

    def _build_window(self):
        try:
            import omni.ui as ui
        except ModuleNotFoundError:
            # Headless apps have no UI. The levels are still kept, only the window is missing.
            logger.warning(f"No UI in this app, so the '{self._title}' heatmap window is not shown.")
            return

        def px(value: float) -> int:
            return max(1, round(value * self.cfg.window_scale))

        cell_width, cell_height, line = px(27), px(25), px(1.2)
        grid_width, grid_height = 3 * cell_width + 4 * line, 3 * cell_height + 4 * line
        label_height = px(16)
        self._reading_font_size = px(10)
        width = grid_width + px(80)  # room for the window title
        margin = 12
        self._window = ui.Window(
            self._title,
            width=width,
            height=len(FINGER_NAMES) * (grid_height + label_height + px(5)) + 2 * label_height + px(40),
            position_x=max(0, ui.Workspace.get_main_window_width() - (self._slot + 1) * (width + margin)),
            position_y=60,  # below the menu bar and the panel tabs
            # Docked, the window would take the size of its dock area and cut off grids.
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_DOCKING,
        )
        self._cells, self._readings = [], []
        with self._window.frame, ui.ZStack():
            ui.Rectangle(style={"background_color": _BACKGROUND})
            with ui.VStack(spacing=px(4)):
                for name in FINGER_NAMES:
                    with ui.HStack(height=grid_height):
                        ui.Spacer()
                        cells, readings = _build_grid(ui, cell_width, cell_height, line, self._reading_font_size)
                        self._cells.append(cells)
                        self._readings.append(readings)
                        ui.Spacer()
                    ui.Label(
                        name,
                        height=label_height,
                        alignment=ui.Alignment.CENTER,
                        style={"color": _TEXT, "font_size": px(12)},
                    )
                # The scale, and the viewer setting the colors depend on, so that two windows compare on equal terms.
                scale = (f"full color from reading {self.full_color_reading:.0f}", f"color_max {self.cfg.color_max:g}")
                self._scale = [
                    ui.Label(
                        text,
                        height=label_height,
                        alignment=ui.Alignment.CENTER,
                        style={"color": _TEXT, "font_size": px(10)},
                    )
                    for text in scale
                ]
        self._painted[:] = -1
        self.set_show_readings(self.cfg.show_readings)


def cell_rgb(level: int, finger: int) -> tuple[int, int, int]:
    """Color of a heatmap level, from near black to the finger's hue, as 0-255 RGB.

    Brightness grows with the square root of the level, so that light touches stay visible.
    """
    brightness = np.sqrt(level / 255.0)
    red, green, blue = (round(UNLIT_LEVEL + (channel - UNLIT_LEVEL) * brightness) for channel in FINGER_COLORS[finger])
    return red, green, blue


def cell_color(level: int, finger: int) -> int:
    """Color of a heatmap level, as an omni.ui color."""
    red, green, blue = cell_rgb(level, finger)
    return 0xFF000000 | (blue << 16) | (green << 8) | red


def _build_grid(ui, cell_width: int, cell_height: int, line: int, font_size: int) -> tuple[list, list]:
    """A 3x3 grid of cells on dark grid lines. Returns the cells and their reading labels, row by row."""
    cells, readings = [], []
    with ui.ZStack(width=3 * cell_width + 4 * line):
        ui.Rectangle(style={"background_color": _GRID_LINES})
        with ui.VStack():
            ui.Spacer(height=line)
            for _ in range(3):
                with ui.HStack(height=cell_height):
                    ui.Spacer(width=line)
                    for _ in range(3):
                        with ui.ZStack(width=cell_width):
                            cells.append(ui.Rectangle(style={"background_color": cell_color(0, 0)}))
                            style = {"color": _READING_ON_DARK, "font_size": font_size}
                            readings.append(ui.Label("0", alignment=ui.Alignment.CENTER, style=style))
                        ui.Spacer(width=line)
                ui.Spacer(height=line)
    return cells, readings
