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
# Author: Woojae Lee, Woojin Wie

"""K1 Rev.1 mimic task registration."""

import gymnasium as gym

import cyclo_lab.simulation_tasks.manager_based.mimic.config.k1_rev1.agents as agents


gym.register(
    id="Cyclo-Mimic-K1-Rev1-Dance1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dance1_env_cfg:K1Rev1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1Rev1MimicPPORunnerCfg",
    },
)

gym.register(
    id="Cyclo-Mimic-K1-Rev1-Dance1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dance1_env_cfg:K1Rev1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1Rev1MimicPPORunnerCfg",
    },
)

gym.register(
    id="Cyclo-Mimic-K1-Rev1-Dance2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dance2_env_cfg:K1Rev1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1Rev1MimicPPORunnerCfg",
    },
)

gym.register(
    id="Cyclo-Mimic-K1-Rev1-Dance2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dance2_env_cfg:K1Rev1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1Rev1MimicPPORunnerCfg",
    },
)
