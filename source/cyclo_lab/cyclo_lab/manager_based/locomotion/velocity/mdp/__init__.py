# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import isaaclab_tasks.core.velocity.mdp as _velocity_mdp

from .rewards import *  # noqa: F401, F403


def __getattr__(name):
    """Resolve upstream MDP terms through Isaac Lab 3.0 lazy exports."""
    return getattr(_velocity_mdp, name)
