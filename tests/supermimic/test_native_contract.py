"""Actual resolved Isaac configs: guard unsupported modes and checkpoint drift."""

import copy
import pytest
import torch

pytest.importorskip("isaaclab")
from isaaclab_tasks.utils import resolve_task_config
import cyclo_lab  # noqa: F401
from cyclo_lab.learning.supermimic.checkpoint import (
    validate_native_cfg,
    environment_contract,
    check_contract,
    read_checkpoint,
)
from cyclo_lab.learning.supermimic.motion_physics import validate_motion_physics

TASK = "Cyclo-SuperMimic-K1-Rev1-Dance1"


@pytest.fixture(scope="module")
def resolved():
    cfg, _ = resolve_task_config(
        TASK, "supermimic_cfg_entry_point", overrides=["physics=isaacsim_physx"]
    )
    motion, _ = validate_motion_physics(
        cfg.commands.reference_trajectory.trajectory_file
    )
    return cfg, motion


def test_supported_physx_and_fixed_context(resolved):
    cfg, _ = resolved
    validate_native_cfg(cfg)
    assert cfg.sim.dt == 0.005 and cfg.decimation == 4
    newton, _ = resolve_task_config(TASK, "supermimic_cfg_entry_point", overrides=[])
    validate_native_cfg(newton)
    assert type(newton.sim.physics.solver_cfg).__name__ == "MJWarpSolverCfg"
    newton.sim.physics.num_substeps = 2
    with pytest.raises(ValueError, match="num_substeps"):
        validate_native_cfg(newton)
    from cyclo_lab.manager_based.mimic.config.k1_rev1.dance1_env_cfg import K1Rev1EnvCfg

    for name in (
        "physics_material",
        "joint_home_offset_noise",
        "torso_com_offset_noise",
        "push_robot",
    ):
        changed = copy.deepcopy(cfg)
        setattr(changed.events, name, getattr(K1Rev1EnvCfg().events, name))
        with pytest.raises(ValueError, match="fixed dynamics"):
            validate_native_cfg(changed)


def test_checkpoint_contract_rejects_equal_dt_different_decimation_and_model(resolved):
    cfg, motion = resolved
    original = environment_contract(cfg, TASK, motion)
    changed = copy.deepcopy(cfg)
    changed.sim.dt /= 2
    changed.decimation *= 2
    validate_native_cfg(changed)  # supported timing, incompatible learned dynamics
    assert changed.sim.dt * changed.decimation == cfg.sim.dt * cfg.decimation
    with pytest.raises(ValueError, match="decimation"):
        check_contract(original, environment_contract(changed, TASK, motion))
    changed = copy.deepcopy(cfg)
    first = next(iter(changed.scene.robot.actuators.values()))
    first.armature = 0.123
    with pytest.raises(ValueError, match="robot"):
        check_contract(original, environment_contract(changed, TASK, motion))
    changed = copy.deepcopy(cfg)
    changed.commands.reference_trajectory.start_from_zero = True
    check_contract(original, environment_contract(changed, TASK, motion))
    for field in (
        "task",
        "motion_sha256",
        "observation",
        "rewards",
        "terminations",
        "actions",
        "physics",
    ):
        other = copy.deepcopy(original)
        other[field] = "incompatible"
        with pytest.raises(ValueError, match=field):
            check_contract(original, other)


def test_exact_rate_and_reward_override_guard(resolved):
    cfg, _ = resolved
    changed = copy.deepcopy(cfg)
    changed.sim.dt = 0.0050001
    with pytest.raises(ValueError, match="exactly 50"):
        validate_native_cfg(changed)
    changed = copy.deepcopy(cfg)
    changed.rewards.reference_body_position.weight = 2.0
    with pytest.raises(ValueError, match="objective"):
        validate_native_cfg(changed)


def test_checkpoint_reader_rejects_old_or_unrestricted_objects(tmp_path):
    import pickle

    path = tmp_path / "old.pt"
    torch.save({"format": "cyclo-supermimic-isaac-v1"}, path)
    with pytest.raises(ValueError, match="v3"):
        read_checkpoint(path)
    from pathlib import Path

    torch.save({"unsupported": Path("unsupported")}, path)
    with pytest.raises(pickle.UnpicklingError):
        read_checkpoint(path)


def test_v2_physx_gencoord_contract_upgrade_is_strict(resolved):
    cfg, motion = resolved
    current = environment_contract(cfg, TASK, motion)
    legacy = copy.deepcopy(current)
    legacy["version"] = 1
    del legacy["backend_class"]
    del legacy["observation"]["tracked_bodies"]
    check_contract(legacy, current)
    for overrides in (
        [],
        ["physics=isaacsim_physx", "env.sim.dt=0.0025", "env.decimation=8"],
    ):
        changed, _ = resolve_task_config(
            TASK, "supermimic_cfg_entry_point", overrides=overrides
        )
        with pytest.raises(ValueError, match="Incompatible"):
            check_contract(legacy, environment_contract(changed, TASK, motion))
    with pytest.raises(ValueError, match="observation"):
        check_contract(legacy, environment_contract(cfg, TASK, motion, "maxcoord"))


def test_newton_kit_bootstrap_keeps_effective_physics():
    from isaaclab.app.sim_launcher import validate_launch

    cfg, _ = resolve_task_config(TASK, "supermimic_cfg_entry_point", overrides=[])
    original = environment_contract(
        cfg, TASK, {"sha256": "fixture", "physical_profiles": []}
    )
    scan, sources = validate_launch(cfg, {"headless": True, "require_kit": True})
    assert sources and "explicit Kit requirement" in " ".join(sources)
    assert type(scan.resolved_physics_cfg.solver_cfg).__name__ == "MJWarpSolverCfg"
    check_contract(
        original,
        environment_contract(cfg, TASK, {"sha256": "fixture", "physical_profiles": []}),
    )
