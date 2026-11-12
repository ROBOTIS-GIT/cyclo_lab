"""Task-local compatibility for Arena on Cyclo's Isaac Lab 2.3 pin."""

from __future__ import annotations

import importlib
from enum import Enum
from typing import Any


class IsaacLabArenaCompatibilityError(RuntimeError):
    """Report an upstream feature unavailable on the pinned Isaac Lab runtime."""


class _RetargeterRequirement(Enum):
    HAND_TRACKING = "hand_tracking"
    HEAD_TRACKING = "head_tracking"
    MOTION_CONTROLLER = "motion_controller"


class _XrAnchorRotationMode(Enum):
    FIXED = "fixed"
    FOLLOW_PRIM = "follow_prim"
    FOLLOW_PRIM_SMOOTHED = "follow_prim_smoothed"
    CUSTOM = "custom_rotation"


_MISSING_G1_RETARGETERS = (
    "G1LowerBodyStandingMotionControllerRetargeterCfg",
    "G1TriHandUpperBodyMotionControllerGripperRetargeterCfg",
)


def _unsupported_retargeter(name: str, base: type) -> type:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IsaacLabArenaCompatibilityError(
            f"{name} requires newer Isaac Lab OpenXR APIs. The FFW-SG2 Galileo "
            "task does not use this G1 teleoperation feature."
        )

    return type(name, (base,), {"__init__": __init__, "__module__": __name__})


def install_isaaclab_arena_compat() -> tuple[str, ...]:
    """Install only APIs needed by Arena's eager asset registry imports."""
    retargeters = importlib.import_module("isaaclab.devices.openxr.retargeters")
    xr_cfg = importlib.import_module("isaaclab.devices.openxr.xr_cfg")
    retargeter_base = importlib.import_module("isaaclab.devices.retargeter_base")
    installed: list[str] = []

    if not hasattr(retargeter_base.RetargeterBase, "Requirement"):
        retargeter_base.RetargeterBase.Requirement = _RetargeterRequirement
        installed.append("RetargeterBase.Requirement")

    for name in _MISSING_G1_RETARGETERS:
        if not hasattr(retargeters, name):
            setattr(retargeters, name, _unsupported_retargeter(name, retargeter_base.RetargeterCfg))
            installed.append(name)

    if not hasattr(xr_cfg, "XrAnchorRotationMode"):
        xr_cfg.XrAnchorRotationMode = _XrAnchorRotationMode
        installed.append("XrAnchorRotationMode")

    return tuple(installed)
