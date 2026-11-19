"""K1 Rev.1 BlinkMimic motion library 001."""

from pathlib import Path

from isaaclab.utils import configclass
from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR
from .base_env_cfg import BlinkMimicLibraryEnvCfg

MOTION_ROOT = Path(CYCLO_LAB_ASSETS_DATA_DIR) / "motions" / "K1_rev1"
MOTION_PATHS = (MOTION_ROOT / "dance1/dance1.npz", MOTION_ROOT / "dance2/dance2.npz")


@configclass
class K1Rev1EnvCfg(BlinkMimicLibraryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_motion_paths(MOTION_PATHS)
