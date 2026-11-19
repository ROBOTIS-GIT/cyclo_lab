"""Check only compatibility conditions needed by the native learners."""

import copy
import pytest

pytest.importorskip("isaaclab")
from isaaclab_tasks.utils import resolve_task_config
import cyclo_lab  # noqa: F401
from cyclo_lab.manager_based.mimic.supermimic.config.k1_rev1.base_env_cfg import validate_imagined_rewards
from cyclo_lab.manager_based.mimic.tracking_contract import (
    check_dynamics, collision_model, dynamics_contract, validate_native_cfg,
)


@pytest.fixture
def cfg():
    return resolve_task_config(
        "Cyclo-SuperMimic-GenCoord-K1-Rev1-001", "learner_cfg_entry_point", overrides=[]
    )[0]


def test_full_resume_detects_actuation_and_physics_changes(cfg):
    original = {"dynamics_contract": dynamics_contract(cfg)}
    cfg.scene.terrain.visual_material = None
    cfg.scene.robot.spawn.visible = False
    check_dynamics(original, dynamics_contract(cfg))
    next(iter(cfg.scene.robot.actuators.values())).armature = 0.123
    with pytest.raises(ValueError, match="robot"):
        check_dynamics(original, dynamics_contract(cfg))
    cfg.sim.dt /= 2
    cfg.decimation *= 2
    validate_native_cfg(cfg)
    with pytest.raises(ValueError, match="decimation"):
        check_dynamics(original, dynamics_contract(cfg))


@pytest.mark.parametrize(("old_module", "function"), [
    ("cyclo_lab.learning.tracking_physics", "initialize_dynamics"),
    ("cyclo_lab.manager_based.mimic.mdp.tracking_physics", "initialize_dynamics"),
    ("cyclo_lab.manager_based.mimic.mdp.events", "apply_home_joint_offset_noise"),
])
def test_full_resume_recognizes_relocated_event_functions(cfg, old_module, function):
    if function == "apply_home_joint_offset_noise":
        from cyclo_lab.manager_based.mimic.beyondmimic.mdp.events import apply_home_joint_offset_noise

        cfg.events.physics_material.func = apply_home_joint_offset_noise
    current = dynamics_contract(cfg)
    saved = copy.deepcopy(current)
    event = saved["events"]["physics_material"]
    event["func"] = f"{old_module}:{function}"
    check_dynamics({"dynamics_contract": saved}, current)

    event["func"] = f"{old_module}:different_dynamics"
    with pytest.raises(ValueError, match="events"):
        check_dynamics({"dynamics_contract": saved}, current)


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


def test_visual_meshes_do_not_affect_collision_model(tmp_path):
    path = tmp_path / "test.urdf"
    collision = tmp_path / "collision.stl"
    collision.write_bytes(b"collision")
    xml = '<robot><link name="a"><visual><geometry><mesh filename="v.stl"/></geometry></visual><collision><geometry><mesh filename="collision.stl"/></geometry></collision></link></robot>'
    path.write_text(xml)
    before = collision_model(path)
    path.write_text(xml.replace("v.stl", "updated.stl"))
    assert collision_model(path) == before
    collision.write_bytes(b"new collision")
    assert collision_model(path) != before
    visual_mesh = tmp_path / "updated.stl"
    visual_mesh.write_bytes(b"geometry used for collisions")
    both = collision_model(path, collision_from_visuals=True)
    visual_mesh.write_bytes(b"changed geometry")
    assert collision_model(path, collision_from_visuals=True) != both
