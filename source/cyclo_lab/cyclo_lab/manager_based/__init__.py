# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing manager-based task implementations."""

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##

import os
if os.environ.get("CYCLO_SERVICE_PROFILE") == "omy":
    # Service profile loads only the task certified for this runtime image.
    from .manipulation.reach.config import omy  # noqa: F401
else:
    _BLACKLIST_PKGS = ["utils"]
    import_packages(__name__, _BLACKLIST_PKGS)
