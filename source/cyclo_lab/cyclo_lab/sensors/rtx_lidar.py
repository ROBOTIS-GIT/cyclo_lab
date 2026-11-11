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

"""Read existing USD RTX LiDARs through Isaac Lab's sensor lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.sim.utils.extensions import enable_extension
from isaaclab.utils import configclass


@dataclass
class RtxLidarData:
    """Per-environment points in sensor coordinates and their distances, in metres.

    Arrays contain valid returns from the latest render frame. They can be empty
    between sensor ticks; they are not a fixed-length ROS LaserScan.
    """

    point_clouds: list[np.ndarray]
    ranges: list[np.ndarray]


class RtxLidar(SensorBase):
    """Attach a render product and point-cloud annotator to each existing OmniLidar."""

    def __init__(self, cfg: RtxLidarCfg):
        self._render_products = []
        self._annotators = []
        self._azimuth_ranges = []
        self._data = RtxLidarData([], [])
        enable_extension("isaacsim.sensors.rtx")
        super().__init__(cfg)

    def __del__(self):
        self._release_render_products()
        super().__del__()

    @property
    def data(self) -> RtxLidarData:
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = range(self.num_instances)
        for index in env_ids:
            index = int(index)
            self._data.point_clouds[index] = np.empty((0, 3), dtype=np.float32)
            self._data.ranges[index] = np.empty(0, dtype=np.float32)

    def _initialize_impl(self):
        import omni.replicator.core as rep

        super()._initialize_impl()
        self._release_render_products()
        prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if not prims or len(prims) != self.num_instances:
            raise ValueError(f"Expected one RTX LiDAR per environment at {self.cfg.prim_path}.")
        for prim in prims:
            if prim.GetTypeName() != "OmniLidar":
                raise ValueError(f"Expected an OmniLidar at {prim.GetPath()}.")
            if prim.GetAttribute("omni:sensor:Core:outputFrameOfReference").Get() != "SENSOR":
                raise ValueError(f"RTX LiDAR must return sensor-frame points at {prim.GetPath()}.")
            azimuth_range = prim.GetAttribute("cyclo:lidar:azimuthRangeRad").Get()
            self._azimuth_ranges.append(azimuth_range)
            # RTX rays come from the USD profile, independently of image resolution.
            product = rep.create.render_product(str(prim.GetPath()), resolution=(1, 1))
            self._render_products.append(product)
            annotator = rep.AnnotatorRegistry.get_annotator(
                "IsaacExtractRTXSensorPointCloudNoAccumulator", device="cpu"
            )
            annotator.attach([product.path])
            self._annotators.append(annotator)
        self._data = RtxLidarData(
            [np.empty((0, 3), dtype=np.float32) for _ in prims],
            [np.empty(0, dtype=np.float32) for _ in prims],
        )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        for index in env_ids:
            index = int(index)
            raw = self._annotators[index].get_data()
            values = raw.get("data", ()) if isinstance(raw, dict) else raw
            if values is None:
                values = ()
            points = np.asarray(values, dtype=np.float32).reshape(-1, 3)
            distances = np.linalg.norm(points, axis=1)
            valid = np.isfinite(points).all(axis=1) & (distances > 0.0)
            azimuth_range = self._azimuth_ranges[index]
            if azimuth_range is not None:
                # Native Sim 6.0.1 returns no points with a clipped rotary USD profile.
                # Keep the URDF sector here; its limits are authored once in the USD.
                azimuth = np.arctan2(points[:, 1], points[:, 0])
                valid &= (azimuth >= azimuth_range[0]) & (azimuth <= azimuth_range[1])
            # Copy render-owned buffers so callers can retain this sample.
            self._data.point_clouds[index] = points[valid]
            self._data.ranges[index] = distances[valid]

    def _invalidate_initialize_callback(self, event):
        self._release_render_products()
        super()._invalidate_initialize_callback(event)

    def _release_render_products(self):
        for annotator in self._annotators:
            annotator.detach()
        for product in self._render_products:
            product.destroy()
        self._annotators.clear()
        self._render_products.clear()
        self._azimuth_ranges.clear()


@configclass
class RtxLidarCfg(SensorBaseCfg):
    """Read an existing OmniLidar; its mounting pose and scan profile stay in USD."""

    class_type: type = RtxLidar
