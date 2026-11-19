"""Numbered K1 tracking tasks registered with Isaac Lab."""

import gymnasium as gym

for number in ("001",):
    gym.register(
        id=f"Cyclo-BlinkMimic-K1-Rev1-{number}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.task_{number}:K1Rev1EnvCfg",
            "learner_cfg_entry_point": "cyclo_lab.manager_based.mimic.blinkmimic.core.runner:BlinkCfg",
        },
    )
