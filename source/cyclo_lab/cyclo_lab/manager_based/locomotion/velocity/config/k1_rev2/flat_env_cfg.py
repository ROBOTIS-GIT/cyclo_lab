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

from isaaclab.utils import configclass

from cyclo_lab.assets.robots.K1_rev2 import K1_REV2_URDF_PATH
from ..k1_rev1.flat_env_cfg import K1Rev1FlatEnvCfg


@configclass
class K1Rev2FlatEnvCfg(K1Rev1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.asset_path = K1_REV2_URDF_PATH
