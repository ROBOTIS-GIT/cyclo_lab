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

"""Numbered K1 tracking tasks registered with Isaac Lab."""

import gymnasium as gym

gym.register(
    id="Cyclo-BlinkMimic-K1-Rev2-001",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_001:K1Rev2EnvCfg",
        "learner_cfg_entry_point": "cyclo_lab.manager_based.mimic.blinkmimic.training.runner:BlinkCfg",
    },
)
