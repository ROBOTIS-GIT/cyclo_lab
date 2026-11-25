# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing manager-based task implementations."""

# The standalone tensor modules can also be imported without an Isaac installation.
try:
    from isaaclab_tasks.utils import import_packages
except ModuleNotFoundError as error:
    if error.name != "isaaclab_tasks":
        raise
else:
    # Register Gym tasks without importing the optional learner implementations.
    _BLACKLIST_PKGS = ["utils", ".mdp", ".mimic.blinkmimic.core"]
    import_packages(__name__, _BLACKLIST_PKGS)
