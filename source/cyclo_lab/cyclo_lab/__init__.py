# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Python module serving as a project/extension template."""

_OPTIONAL_ISAAC_MODULES = {"carb", "isaaclab", "isaaclab_tasks", "isaacsim", "omni"}


def _is_optional_isaac_module(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return any(module_name == name or module_name.startswith(f"{name}.") for name in _OPTIONAL_ISAAC_MODULES)


# Register Gym environments.
try:
    from .manager_based import *
except ModuleNotFoundError as exc:
    if not _is_optional_isaac_module(exc.name):
        raise

# Register UI extensions.
try:
    from .ui_extension_example import *
except ModuleNotFoundError as exc:
    if not _is_optional_isaac_module(exc.name):
        raise
