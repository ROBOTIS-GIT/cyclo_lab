# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Python module serving as a project/extension template."""

# Register Gym environments.
from .manager_based import *

# Register UI extensions.
import os
if os.environ.get("CYCLO_SERVICE_PROFILE") != "omy":
    from .ui_extension_example import *
