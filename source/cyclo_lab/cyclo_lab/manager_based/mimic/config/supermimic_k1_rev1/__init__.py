# SPDX-License-Identifier: Apache-2.0
"""Distinct learner registrations, plus compatible names for the existing PPO tasks."""

import gymnasium as gym

for motion in ("Dance1", "Dance2"):
    module = motion.lower()
    gym.register(
        id=f"Cyclo-BeyondMimic-K1-Rev1-{motion}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"cyclo_lab.manager_based.mimic.config.k1_rev1.{module}_env_cfg:K1Rev1EnvCfg",
            "rsl_rl_cfg_entry_point": "cyclo_lab.manager_based.mimic.config.k1_rev1.agents.rsl_rl_ppo_cfg:K1Rev1MimicPPORunnerCfg",
        },
    )
    gym.register(
        id=f"Cyclo-SuperMimic-K1-Rev1-{motion}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:SuperMimic{motion}EnvCfg",
            "supermimic_cfg_entry_point": "cyclo_lab.learning.supermimic.runner:RunnerCfg",
        },
    )
