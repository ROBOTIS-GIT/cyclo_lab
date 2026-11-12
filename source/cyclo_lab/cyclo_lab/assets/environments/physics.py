"""Shared physics-aware USD spawning for environment assets."""

from __future__ import annotations

_SPAWN_ENVIRONMENT_WITH_FRICTION = None


def environment_physics_material_cfg():
    """Return the common high-friction material used by mobile robot scenes."""
    import isaaclab.sim as sim_utils

    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=2.0,
        dynamic_friction=1.8,
        restitution=0.0,
    )


def spawn_environment_with_friction_once(
    prim_path,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn one USD environment and bind the common physics material."""
    from isaaclab.sim.spawners.from_files import from_files
    from isaaclab.sim.utils import bind_physics_material

    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    material_path = f"{prim_path}/environmentPhysicsMaterial"
    physics_material = environment_physics_material_cfg()
    physics_material.func(material_path, physics_material)
    bind_physics_material(prim_path, material_path)
    return prim


def spawn_environment_with_friction(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Clone and spawn a high-friction USD environment."""
    global _SPAWN_ENVIRONMENT_WITH_FRICTION
    if _SPAWN_ENVIRONMENT_WITH_FRICTION is None:
        from isaaclab.sim.utils import clone

        _SPAWN_ENVIRONMENT_WITH_FRICTION = clone(spawn_environment_with_friction_once)
    return _SPAWN_ENVIRONMENT_WITH_FRICTION(prim_path, cfg, translation, orientation, **kwargs)
