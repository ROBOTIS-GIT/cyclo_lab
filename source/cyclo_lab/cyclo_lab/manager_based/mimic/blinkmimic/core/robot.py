"""Motion reader types used by BlinkMimic's simulator-independent motion library."""

from typing import Callable, Protocol

import numpy as np

MotionArrays = tuple[np.ndarray, np.ndarray, np.ndarray, float]


class RobotSpec(Protocol):
    motion_glob: str
    load_motion: Callable[[str], MotionArrays]
