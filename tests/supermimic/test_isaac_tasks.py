"""Registration/config regression checks, requiring the pinned Isaac environment."""

import pytest

pytest.importorskip("isaaclab")
import gymnasium as gym
import cyclo_lab  # noqa: F401,E402
from cyclo_lab.manager_based.mimic.beyondmimic.config.k1_rev1.dance1_env_cfg import K1Rev1EnvCfg
from cyclo_lab.manager_based.mimic.supermimic.config.k1_rev1.base_env_cfg import (
    SuperMimicLibraryEnvCfg,
)


@pytest.mark.parametrize("motion", ["Dance1", "Dance2"])
def test_alias_is_existing_ppo_and_supermimic_cannot_select_ppo_by_accident(motion):
    old = gym.spec(f"Cyclo-Mimic-K1-Rev1-{motion}")
    alias = gym.spec(f"Cyclo-BeyondMimic-K1-Rev1-{motion}")
    new = gym.spec("Cyclo-SuperMimic-GenCoord-K1-Rev1-001")
    assert old.entry_point == alias.entry_point
    assert old.kwargs == alias.kwargs
    assert "learner_cfg_entry_point" in new.kwargs
    assert "rsl_rl_cfg_entry_point" not in new.kwargs


def test_new_task_does_not_mutate_existing_beyondmimic_config():
    old = K1Rev1EnvCfg()
    new = SuperMimicLibraryEnvCfg()
    after = K1Rev1EnvCfg()
    assert old.commands.reference_trajectory.anchor_body_name == "torso_link"
    assert new.commands.reference_trajectory.anchor_body_name == "pelvis"
    assert after.commands.reference_trajectory.anchor_body_name == "torso_link"
    assert (
        old.events.physics_material is not None
        and after.events.physics_material is not None
    )
    assert new.events.physics_material.func != old.events.physics_material.func
    assert old.rewards.joint_limit.weight == after.rewards.joint_limit.weight == -10.0
    assert new.rewards.joint_limit.weight == -2.0
    assert (
        old.export_sim2real_cfg
        and after.export_sim2real_cfg
        and not new.export_sim2real_cfg
    )


@pytest.mark.parametrize("task", [
    "Cyclo-BlinkMimic-K1-Rev1-001",
    "Cyclo-SuperMimic-GenCoord-K1-Rev1-001",
    "Cyclo-SuperMimic-MaxCoord-K1-Rev1-001",
])
def test_motion_library_tasks_do_not_inherit_beyondmimic_task_defaults(task):
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config(task, "learner_cfg_entry_point", overrides=[])
    for cls in (type(cfg), type(cfg.rewards), type(cfg.events)):
        assert all(".beyondmimic." not in base.__module__ for base in cls.__mro__)


@pytest.mark.parametrize("rev1_task", [
    "Cyclo-Velocity-Flat-K1-Rev1-v0",
    "Cyclo-BeyondMimic-K1-Rev1-Dance1",
    "Cyclo-BeyondMimic-K1-Rev1-Dance2",
    "Cyclo-Mimic-K1-Rev1-Dance1",
    "Cyclo-Mimic-K1-Rev1-Dance2",
    "Cyclo-BlinkMimic-K1-Rev1-001",
    "Cyclo-SuperMimic-GenCoord-K1-Rev1-001",
    "Cyclo-SuperMimic-MaxCoord-K1-Rev1-001",
])
def test_rev2_selects_own_robot_and_motion_files(rev1_task):
    from importlib import import_module
    from pathlib import Path

    for revision in (2, 1):
        task = rev1_task.replace("Rev1", f"Rev{revision}")
        entry = gym.spec(task).kwargs["env_cfg_entry_point"]
        module, name = entry.split(":")
        cfg = getattr(import_module(module), name)()
        urdf = Path(cfg.scene.robot.spawn.asset_path)
        assert urdf.is_file() and urdf.parent.name == f"k1_rev{revision}"
        if "Velocity" in task:
            continue
        command = cfg.commands.reference_trajectory
        for file in (command.trajectory_file, *getattr(command, "additional_motions", ())):
            path = Path(file)
            assert path.is_file() and path.parent.parent.name == f"K1_rev{revision}"
        if "BlinkMimic" in task:
            assert cfg.foot_contact_z == 0.074


def test_library_expands_mixed_motion_paths_once_in_order(tmp_path):
    cfg = SuperMimicLibraryEnvCfg()
    folder = tmp_path / "library"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    files = [tmp_path / "first.npz", folder / "a.npz", nested / "b.npz", tmp_path / "last.npz"]
    for path in reversed(files):
        path.touch()
    cfg.set_motion_paths([files[0], folder, tmp_path / "**/*.npz"])
    command = cfg.commands.reference_trajectory
    assert [command.trajectory_file, *command.additional_motions] == list(map(str, files))
    with pytest.raises(ValueError, match="No motion files matched"):
        cfg.set_motion_paths([tmp_path / "missing*.npz"])
