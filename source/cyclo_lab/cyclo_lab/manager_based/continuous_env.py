"""Gym-compatible wrapper for continuous Isaac Lab manager environments."""

from __future__ import annotations

import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg


class ContinuousManagerBasedEnv(ManagerBasedEnv, gym.Env):
    """Expose a non-episodic ``ManagerBasedEnv`` through Gym registration."""

    def __init__(
        self,
        cfg: ManagerBasedEnvCfg,
        render_mode: str | None = None,
        **_registry_kwargs,
    ) -> None:
        super().__init__(cfg=cfg)
        self.render_mode = render_mode
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
