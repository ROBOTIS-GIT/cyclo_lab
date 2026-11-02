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

import warp as wp
from isaaclab.utils.warp import ProxyArray

from .fingertip_pads import CELLS_PER_PAD, FINGER_NAMES


class HandTactileData:
    """Data container for the hand tactile sensor."""

    def __init__(self):
        self._cell_forces: ProxyArray | None = None
        self._cell_readings: ProxyArray | None = None

    @property
    def cell_forces(self) -> ProxyArray | None:
        """Force [N] pressing into each tactile cell.

        Shape is (num_instances, 5, 9), dtype = wp.float32. Fingers are ordered thumb, index, middle, ring,
        little. Cells are numbered row by row from the fingertip end, each row starting on the pad's -Y side.

        The forces come from simulated contacts and are not calibrated against the real sensor. A contact's force is
        shared among the cells around it, as :attr:`HandTactileCfg.spread_sigma` describes.
        """
        return self._cell_forces

    @property
    def cell_readings(self) -> ProxyArray | None:
        """Reading of each tactile cell, from 0 to 255, the range of the real sensor's ``pressure_values``.

        Shape is (num_instances, 5, 9), dtype = wp.float32, ordered as :attr:`cell_forces`. Each reading follows
        from the cell's force through the rough curve that :attr:`HandTactileCfg.reading_scale` describes.
        """
        return self._cell_readings

    def create_buffers(self, num_envs: int, device: str):
        """Allocates the data buffers, filled with zeros."""
        shape = (num_envs, len(FINGER_NAMES), CELLS_PER_PAD)
        self._cell_forces = ProxyArray(wp.zeros(shape, dtype=wp.float32, device=device))
        self._cell_readings = ProxyArray(wp.zeros(shape, dtype=wp.float32, device=device))
