"""Build the upstream Galileo pick-and-place task with the FFW-SG2 embodiment."""

from __future__ import annotations

from argparse import Namespace

from .compat import install_isaaclab_arena_compat

TASK_ID = "Cyclo-Arena-Galileo-Pick-Place-FFW-SG2-v0"
DEFAULT_OBJECT = "power_drill"


def _arena_args() -> Namespace:
    """Return deterministic defaults for the registered Cyclo task."""
    return Namespace(
        object=DEFAULT_OBJECT,
        embodiment="ffw_sg2_mobile_abs_joint_pos",
        enable_cameras=True,
        teleop_device=None,
        num_envs=1,
        env_spacing=30.0,
        solve_relations=True,
        placement_seed=42,
        mimic=False,
    )


def make_ffw_sg2_galileo_pick_place_env_cfg():
    """Compose and return the final manager-based Arena environment config."""
    installed = install_isaaclab_arena_compat()
    if installed:
        print(f"[INFO] Installed IsaacLab-Arena compatibility: {', '.join(installed)}")

    # Importing the embodiment after compatibility installation registers it in Arena.
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments.galileo_pick_and_place_environment import GalileoPickAndPlaceEnvironment

    from . import embodiment as _embodiment  # noqa: F401

    args = _arena_args()
    arena_env = GalileoPickAndPlaceEnvironment().get_env(args)
    arena_env.name = TASK_ID
    background_pose = arena_env.scene.assets["galileo"].get_initial_pose()
    arena_env.embodiment.set_initial_pose(
        Pose(
            position_xyz=(0.0, 0.0, background_pose.position_xyz[2]),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    builder = ArenaEnvBuilder(arena_env, args)
    builder.orchestrate()
    env_cfg = builder.compose_manager_cfg()
    env_cfg = builder.modify_env_cfg(env_cfg)
    # One rendered camera frame per control step is sufficient for the ROS2 bridge.
    env_cfg.sim.render_interval = env_cfg.decimation
    return env_cfg
