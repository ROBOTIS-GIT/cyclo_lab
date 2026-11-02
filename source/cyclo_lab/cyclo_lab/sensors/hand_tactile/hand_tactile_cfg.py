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

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .hand_tactile import HandTactile


@configclass
class TactileHeatmapCfg:
    """How the debug heatmap draws the sensor's readings. These settings change the picture only, never the data."""

    env_index: int = 0
    """Environment whose hand is shown. Defaults to 0."""

    color_max: float = 125.0
    """The real viewer's ``color_max``: a cell shows in full color once ``sqrt(1.5 * reading) * 8`` reaches it.
    Defaults to 125, the real viewer's default."""

    deadband_counts: float = 1.0
    """Readings below this stay dark, as in the real viewer. Defaults to 1.0."""

    smoothing_time_constant: float = 0.035
    """Time constant of the low-pass filter on the shown readings (in s). Defaults to 0.035, close to the real
    viewer's smoothing. 0 turns the smoothing off."""

    window_scale: float = 1.0
    """Size of the heatmap window. Defaults to 1.0, at which the window is about 165 x 570 pixels."""

    show_readings: bool = False
    """Whether each cell of the window shows its reading, from 0 to 255, and the color scale shows under the grids.
    Defaults to False."""

    pad_overlay: bool = True
    """Whether the fingertip pads show their grid and cell numbers in the viewport, with the pressed cells lit in the
    heatmap colors. Only apps with a window draw them. Defaults to True."""


@configclass
class HandTactileCfg(SensorBaseCfg):
    """Configuration for the HX5-D20 fingertip tactile sensor.

    :attr:`prim_path` is the root of the robot that carries the hand, for example ``{ENV_REGEX_NS}/Robot``.
    The fingertip bodies of :attr:`hand` are found under it by name.
    """

    class_type: type[HandTactile] | str = "{DIR}.hand_tactile:HandTactile"

    hand: Literal["left", "right"] = MISSING
    """Which hand to read."""

    depth_tolerance: float = 0.001
    """Largest depth below the pad surface at which a contact still counts (in m). Defaults to 1 mm.

    Deeper contacts lie inside the fingertip, not on its pad, such as contacts on its back.
    """

    min_normal_alignment: float = 0.7
    """Smallest absolute cosine between a contact normal and the pad normal. Defaults to 0.7 (about 45 degrees).

    A safety net for rare cases. The position and depth checks already leave out most touches on the side of the
    finger, but a contact on the pad can still push in at a steep angle. Teleoperated hands touch objects at many
    angles; for them, a depth tolerance of 3 mm and an alignment of 0.3 give a steadier picture.
    """

    max_contacts_per_pad: int = 64
    """Room for contact points per pad and physics step in the PhysX contact buffer. Defaults to 64.

    PhysX also reports contact points that carry no force yet, between bodies closer than their contact offset. An
    open AI Worker hand already reports 20 to 30 per fingertip, from the fingers beside it. Contact points beyond the
    buffer's room are left out, with a warning.
    """

    spread_sigma: float = 0.0015
    """Standard deviation of the Gaussian that spreads a contact's force over the cells (in m). Defaults to 1.5 mm.

    PhysX reports a touch as points between rigid shapes, while a touch near a border on the real hand also lights the
    next cell. Each cell's share is the Gaussian's value at the cell's centre, scaled so that the shares add up to the
    contact's force: a contact at a cell's centre keeps about 98.5 % there, and one 0.5 mm from a border splits about
    75/25. The width was set against one recording of the real hand, not measured on its pad, and it does not change
    with the force. 0 keeps the whole force in the cell under the contact.
    """

    reading_scale: float = 4.02
    """Scale of the curve from a cell's force to its reading. Defaults to 4.02.

    A cell reads ``reading_scale * force ** reading_exponent``, with the force in N, capped at 255. This scale only
    turns simulated forces into simulated readings. It is a rough match, so do not use it the other way, to estimate
    forces from the real sensor's readings.
    """

    reading_exponent: float = 1.4
    """Exponent of the curve from a cell's force to its reading. Defaults to 1.4. See :attr:`reading_scale`."""

    heatmap: TactileHeatmapCfg = TactileHeatmapCfg()
    """How :attr:`debug_vis` draws the readings."""
