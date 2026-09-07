"""Robotis showroom asset configuration."""

from __future__ import annotations

from pathlib import Path

from .physics import spawn_environment_with_friction_once

_SHOWROOM_DATA_DIR = Path(__file__).resolve().parents[3] / "data/environments/robotis_showroom"
ROBOTIS_SHOWROOM_USD_PATH = str(_SHOWROOM_DATA_DIR / "robotis_showroom.usda")
ROBOTIS_SHOWROOM_BACKGROUND_TEXTURE_PATHS = tuple(
    str(_SHOWROOM_DATA_DIR / f"textures/lab_background_{index:02d}.jpg") for index in range(1, 4)
)

ROBOTIS_SHOWROOM_ENVIRONMENT_POS = (0.0, 0.0, 0.0)
ROBOTIS_SHOWROOM_ENVIRONMENT_ROT = (1.0, 0.0, 0.0, 0.0)

_SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT = None


def _make_showroom_floor_visual_only(prim_path: str) -> None:
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = get_current_stage()
    showroom_prim = stage.GetPrimAtPath(prim_path)
    if not showroom_prim.IsValid():
        return

    for prim in Usd.PrimRange(showroom_prim):
        if not str(prim.GetPath()).endswith("/ShowroomShell/Floor"):
            continue
        UsdGeom.Imageable(prim).MakeVisible()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)


def _spawn_robotis_showroom_environment_once(
    prim_path,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    prim = spawn_environment_with_friction_once(
        prim_path,
        cfg,
        translation,
        orientation,
        **kwargs,
    )
    _make_showroom_floor_visual_only(prim_path)
    return prim


def spawn_robotis_showroom_environment(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Clone and spawn the showroom while keeping its floor visual-only."""
    global _SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT
    if _SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT is None:
        from isaaclab.sim.utils import clone

        _SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT = clone(_spawn_robotis_showroom_environment_once)
    return _SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT(prim_path, cfg, translation, orientation, **kwargs)


def make_robotis_showroom_environment_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotisShowroom",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_robotis_showroom_environment,
            usd_path=ROBOTIS_SHOWROOM_USD_PATH,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=ROBOTIS_SHOWROOM_ENVIRONMENT_POS,
            rot=ROBOTIS_SHOWROOM_ENVIRONMENT_ROT,
        ),
    )
