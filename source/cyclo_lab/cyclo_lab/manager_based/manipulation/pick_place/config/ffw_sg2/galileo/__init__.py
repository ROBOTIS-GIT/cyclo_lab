"""Galileo pick-and-place task registration for FFW-SG2."""

import gymnasium as gym

from .env_cfg import TASK_ID

gym.register(
    id=TASK_ID,
    entry_point="isaaclab_arena.environments.isaaclab_arena_manager_based_env:IsaacLabArenaManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:make_ffw_sg2_galileo_pick_place_env_cfg",
    },
    disable_env_checker=True,
)
