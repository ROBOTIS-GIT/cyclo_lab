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
