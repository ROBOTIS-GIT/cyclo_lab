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
# Author: Kiwoong Park

"""K1 Rev.2 SuperMimic motion library 001."""

from pathlib import Path

from isaaclab.utils import configclass
from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR
from cyclo_lab.assets.robots.K1_rev2 import K1_REV2_URDF_PATH
from ..k1_rev1.base_env_cfg import SuperMimicLibraryEnvCfg

MOTION_PATHS = (
    Path(CYCLO_LAB_ASSETS_DATA_DIR) / "motions/K1_rev2/dance1/dance1.npz",
    Path(CYCLO_LAB_ASSETS_DATA_DIR) / "motions/K1_rev2/dance2/dance2.npz",
)


@configclass
class K1Rev2EnvCfg(SuperMimicLibraryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.asset_path = K1_REV2_URDF_PATH
        self.set_motion_paths(MOTION_PATHS)
