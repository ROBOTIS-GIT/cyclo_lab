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


"""K1 Rev.1 mimic environment configuration."""

from isaaclab.utils import configclass

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR
from cyclo_lab.simulation_tasks.manager_based.mimic.config.k1_rev1.base_env_cfg import K1Rev1MimicEnvCfg

TRAJECTORY_FILE = f"{CYCLO_LAB_ASSETS_DATA_DIR}/motions/K1_rev1/dance2/dance2.npz"


@configclass
class K1Rev1EnvCfg(K1Rev1MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.commands.reference_trajectory.trajectory_file = TRAJECTORY_FILE


@configclass
class K1Rev1PlayEnvCfg(K1Rev1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.commands.reference_trajectory.start_from_zero = True
        self.events.push_robot = None
        self.observations.policy.enable_corruption = False
