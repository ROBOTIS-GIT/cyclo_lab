import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


BACKGROUND_CUBE_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.1, 3.0, 2.0),  # thin, wide, tall cube as background wall
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # Static object, no physics
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,  # No collision
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.5, 0.5, 0.5),  # Initial gray color, will be randomized
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.2, 0.0, 1.0),  # Behind the table at x=1.2
    ),
)
