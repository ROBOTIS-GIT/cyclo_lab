# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing manager-based task implementations."""

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##

_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
