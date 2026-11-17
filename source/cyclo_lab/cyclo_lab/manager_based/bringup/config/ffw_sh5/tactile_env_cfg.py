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

"""FFW-SH5 bringup environment with the HX5-D20 fingertip tactile sensors on both hands."""

from __future__ import annotations

from cyclo_lab.sensors import HandTactileCfg, TactileHeatmapCfg
from isaaclab.utils import configclass

from .env_cfg import FFWSH5BringupEnvCfg


def _make_tactile_cfg(hand: str) -> HandTactileCfg:
    # Teleoperated hands touch objects at many angles, so the pad checks are looser than the sensor's defaults. The
    # heatmap shows light grasps brighter than the real viewer's default (color_max 60 instead of 125).
    return HandTactileCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        hand=hand,
        debug_vis=True,
        depth_tolerance=0.003,
        min_normal_alignment=0.3,
        heatmap=TactileHeatmapCfg(color_max=60.0),
    )


@configclass
class FFWSH5TactileBringupEnvCfg(FFWSH5BringupEnvCfg):
    """FFW-SH5 on a ground plane, with fingertip tactile sensors and a heatmap window per hand."""

    def __post_init__(self) -> None:
        super().__post_init__()
        scene = self.scene
        # The tactile sensors read the fingertips' contacts, which PhysX reports only once contact sensors are on.
        scene.robot = scene.robot.replace(spawn=scene.robot.spawn.replace(activate_contact_sensors=True))
        scene.tactile_left = _make_tactile_cfg("left")
        scene.tactile_right = _make_tactile_cfg("right")
