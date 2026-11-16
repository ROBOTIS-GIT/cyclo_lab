"""Build the upstream Galileo pick-and-place task with the FFW-SG2 embodiment."""

from __future__ import annotations

TASK_ID = "Cyclo-Arena-Galileo-Pick-Place-FFW-SG2-v0"
DEFAULT_OBJECT = "power_drill"


def make_ffw_sg2_galileo_pick_place_env_cfg():
    """Compose and return the final manager-based Arena environment config."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments.galileo_pick_and_place_environment import (
        GalileoPickAndPlaceEnvironment,
        GalileoPickAndPlaceEnvironmentCfg,
    )

    from . import embodiment as _embodiment  # noqa: F401

    arena_env = GalileoPickAndPlaceEnvironment().build(
        GalileoPickAndPlaceEnvironmentCfg(
            object=DEFAULT_OBJECT,
            embodiment="ffw_sg2_mobile_abs_joint_pos",
            enable_cameras=True,
            teleop_device=None,
        )
    )
    arena_env.name = TASK_ID
    background_pose = arena_env.scene.assets["galileo"].get_initial_pose()
    arena_env.embodiment.set_initial_pose(
        Pose(
            position_xyz=(0.0, 0.0, background_pose.position_xyz[2]),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )

    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=1, env_spacing=30.0, solve_relations=True, placement_seed=42, mimic=False),
    )
    env_cfg, env_kwargs = builder.build_env_cfg()
    # Preserve the FFW bridge's 50 Hz control rate across Arena's default-rate change.
    env_cfg.sim.dt = 1 / 200
    env_cfg.decimation = 4
    # One rendered camera frame per control step is sufficient for the ROS2 bridge.
    env_cfg.sim.render_interval = env_cfg.decimation
    # Keep Arena's runtime recorders and environment class when callers use gym.make(task, cfg=cfg).
    _, env_cfg, _ = builder.build_registered(env_cfg, env_kwargs)
    return env_cfg
