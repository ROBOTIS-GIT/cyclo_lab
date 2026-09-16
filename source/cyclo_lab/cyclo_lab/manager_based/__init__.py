# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing manager-based task implementations."""

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# Only K1 locomotion and mimic tasks are supported on this branch. Skipping the
# manipulation tree also prevents loading OMY/FFW and their teleop dependencies.
_BLACKLIST_PKGS = ["utils", "manipulation"]
# Import the supported K1 configs in this package.
import_packages(__name__, _BLACKLIST_PKGS)
