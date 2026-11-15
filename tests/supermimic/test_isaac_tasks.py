"""Registration/config regression checks, requiring the pinned Isaac environment."""

import pytest

pytest.importorskip("isaaclab")
import gymnasium as gym
import cyclo_lab  # noqa: F401,E402
from cyclo_lab.manager_based.mimic.config.k1_rev1.dance1_env_cfg import K1Rev1EnvCfg
from cyclo_lab.manager_based.mimic.config.supermimic_k1_rev1.env_cfg import (
    SuperMimicDance1EnvCfg,
)


def test_alias_is_existing_ppo_and_supermimic_cannot_select_ppo_by_accident():
    old = gym.spec("Cyclo-Mimic-K1-Rev1-Dance1")
    alias = gym.spec("Cyclo-BeyondMimic-K1-Rev1-Dance1")
    new = gym.spec("Cyclo-SuperMimic-K1-Rev1-Dance1")
    assert old.entry_point == alias.entry_point
    assert old.kwargs == alias.kwargs
    assert "supermimic_cfg_entry_point" in new.kwargs
    assert "rsl_rl_cfg_entry_point" not in new.kwargs


def test_new_task_does_not_mutate_existing_beyondmimic_config():
    old = K1Rev1EnvCfg()
    new = SuperMimicDance1EnvCfg()
    after = K1Rev1EnvCfg()
    assert old.commands.reference_trajectory.anchor_body_name == "torso_link"
    assert new.commands.reference_trajectory.anchor_body_name == "pelvis"
    assert after.commands.reference_trajectory.anchor_body_name == "torso_link"
    assert (
        old.events.physics_material is not None
        and after.events.physics_material is not None
    )
    assert new.events.physics_material is None
    assert old.rewards.joint_limit.weight == after.rewards.joint_limit.weight == -10.0
    assert new.rewards.joint_limit.weight == -2.0
    assert (
        old.export_sim2real_cfg
        and after.export_sim2real_cfg
        and not new.export_sim2real_cfg
    )
