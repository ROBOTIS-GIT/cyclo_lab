import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from cyclo_lab.assets.object import CYCLO_LAB_OBJECT_ASSETS_DATA_DIR

OMY_TABLE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{CYCLO_LAB_OBJECT_ASSETS_DATA_DIR}/object/robotis_omy_table.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.0, 0.0, 0.0],
        rot=[0.0, 0.0, 0.0, 0.0],
    ),
)
