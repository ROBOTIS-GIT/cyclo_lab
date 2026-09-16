# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Python module serving as a project/extension template."""

# Register K1 simulation environments. Real-world/AI Worker tasks are outside
# the scope of this Newton branch and intentionally are not auto-registered.
from .manager_based import *

# Register UI extensions only when Isaac Sim / Kit is available.  Newton can run
# without Kit, so importing the project must not require ``omni.ui``.
try:
    from .ui_extension_example import *
except ImportError:
    pass
