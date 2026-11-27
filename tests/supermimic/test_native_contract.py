"""Check only compatibility conditions needed by the native learners."""

import copy
import shutil
import pytest

pytest.importorskip("isaaclab")
from isaaclab_tasks.utils import resolve_task_config
import cyclo_lab  # noqa: F401
from cyclo_lab.manager_based.mimic.supermimic.config.k1_rev1.base_env_cfg import validate_imagined_rewards
from cyclo_lab.manager_based.mimic._shared.tracking_contract import validate_native_cfg


@pytest.fixture
def cfg():
    return resolve_task_config(
        "Cyclo-SuperMimic-GenCoord-K1-Rev1-001", "learner_cfg_entry_point", overrides=[]
    )[0]


@pytest.mark.parametrize("family", ["BlinkMimic", "SuperMimic-GenCoord"])
@pytest.mark.parametrize("revision", [1, 2])
def test_relocated_robot_asset(tmp_path, family, revision):
    task = f"Cyclo-{family}-K1-Rev{revision}-001"
    cfg, _ = resolve_task_config(task, "learner_cfg_entry_point", overrides=[])
    path = tmp_path / "custom.urdf"
    shutil.copyfile(cfg.scene.robot.spawn.asset_path, path)
    cfg.scene.robot.spawn.asset_path = str(path)
    validate_native_cfg(cfg)


def test_only_learner_constraints_restrict_task_configuration(cfg):
    cfg.events.diagnostic = copy.deepcopy(cfg.events.physics_material)
    validate_native_cfg(cfg)
    validate_imagined_rewards(cfg)
    cfg.rewards.reference_body_position.weight = 2
    with pytest.raises(ValueError, match="objective"):
        validate_imagined_rewards(cfg)
    validate_native_cfg(cfg)
    cfg.sim.dt *= 2
    with pytest.raises(ValueError, match="50 Hz"):
        validate_native_cfg(cfg)


def test_heading_aligned_velocity_terms(cfg):
    from types import SimpleNamespace
    import torch

    terms = (cfg.rewards.reference_body_linear_velocity, cfg.rewards.reference_body_angular_velocity)
    h = 0.5**0.5
    command = SimpleNamespace(
        anchor_quat_w=torch.tensor([[0., 0., 0., 1.]]),
        robot_anchor_quat_w=torch.tensor([[0., 0., h, h]]),
        body_lin_vel_w=torch.tensor([[[1., 0., 2.]]]),
        robot_body_lin_vel_w=torch.tensor([[[0., 1., 2.]]]),
        body_ang_vel_w=torch.tensor([[[0., 1., 3.]]]),
        robot_body_ang_vel_w=torch.tensor([[[-1., 0., 3.]]]),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    for term in terms:
        torch.testing.assert_close(term.func(env, **term.params), torch.ones(1))
    command.robot_anchor_quat_w = command.anchor_quat_w
    for term in terms:
        expected = torch.exp(torch.tensor([-2.0 / term.params["std"]**2]))
        torch.testing.assert_close(term.func(env, **term.params), expected)


@pytest.mark.parametrize("world_only,changed", [(False, False), (False, True), (True, True)])
@pytest.mark.filterwarnings("error::UserWarning")
def test_restore_ignores_recorded_physics(cfg, tmp_path, world_only, changed):
    import torch

    from cyclo_lab.manager_based.mimic._shared.checkpoint import decode, load, restore_runner, save
    from cyclo_lab.manager_based.mimic.supermimic.training.runner import IsaacSuperMimicRunner, RunnerCfg
    from test_runner_contract import UnitEnv

    settings = RunnerCfg(actor_hidden_dims=(16, 16), dynamics_hidden_dims=(16, 16), critic_hidden_dims=(16, 16), eager=True)
    source = IsaacSuperMimicRunner(UnitEnv(), settings)
    target = IsaacSuperMimicRunner(UnitEnv(), settings)
    source.iteration = 5
    save(source, tmp_path, [])
    saved = torch.load(tmp_path / "last.pt", weights_only=True)
    assert "dynamics_contract" not in saved and "physics_backend" not in saved
    assert saved["format"] == "cyclo.supermimic.checkpoint.v1"
    if changed:
        saved["dynamics_contract"] = {"collision_model_sha256": "legacy-checkpoint"}
        saved["physics_backend"] = "MJWarpSolverCfg"
        saved["contract"] = {"step_dt": 0.04, "action": [(source.dynamics.action_scale * 2).tolist()]}
        saved["world_model"]["dynamics"]["action_scale"] *= 2
    action_scale = target.dynamics.action_scale.clone()
    restore_runner(target, decode(saved, "supermimic"), training=True, world_only=world_only, iterations=10)
    assert target.iteration == (0 if world_only else 5)
    torch.testing.assert_close(target.dynamics.action_scale, action_scale)
    for actual, expected in zip(target.dynamics.parameters(), source.dynamics.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert "joint_names" not in saved
    saved["joint_names"] = ["old_robot_joint"]
    load(target, decode(saved, "supermimic"), training=True, world_only=world_only)
