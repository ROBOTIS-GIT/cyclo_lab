"""Training should follow the destination task while playback follows the trained run."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[2] / "scripts/reinforcement_learning/motion_tracker.py"
spec = importlib.util.spec_from_file_location("motion_tracker", path)
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)


@pytest.mark.parametrize(
    ("action", "task", "defaults", "override", "world_only", "expected"),
    [
        ("train", "001", "ab", None, False, "a"),
        ("train", "002", "ab", None, False, "ab"),
        ("train", "001", "abc", None, False, "abc"),
        ("train", "001", "ab", "ac", False, "ac"),
        ("train", "001", "ab", None, True, "ab"),
        ("play", "002", "abc", None, False, "a"),
        ("export", "002", "abc", None, False, "a"),
    ],
)
def test_motion_selection(tmp_path, action, task, defaults, override, world_only, expected):
    files = {c: str(tmp_path / c) for c in "abc"}
    for file in files.values():
        Path(file).touch()

    reference = SimpleNamespace(
        trajectory_file=files[defaults[0]], additional_motions=[files[c] for c in defaults[1:]]
    )
    cfg = SimpleNamespace(scene=SimpleNamespace(), commands=SimpleNamespace(reference_trajectory=reference))
    args = SimpleNamespace(
        num_envs=2,
        motion_file=[files[c] for c in override] if override else None,
        world_model_checkpoint=world_only,
    )
    bundle = dict(task="001", task_motions=[files["a"], files["b"]], motions=[files["a"]])
    selected, _ = tracker.select_motions(cfg, task, action, args, bundle)
    assert selected == [files[c] for c in expected]
    assert [reference.trajectory_file, *reference.additional_motions] == selected


@pytest.mark.parametrize("blink", [True, False], ids=["blinkmimic", "supermimic"])
def test_world_model_initialization_preserves_new_actor_width(blink):
    from cyclo_lab.manager_based.mimic.blinkmimic.core.runner import BlinkCfg
    from cyclo_lab.manager_based.mimic.supermimic.core.runner import RunnerCfg

    settings = BlinkCfg() if blink else RunnerCfg()
    family = "blinkmimic" if blink else "supermimic"
    experiment = importlib.import_module(f"cyclo_lab.manager_based.mimic.{family}.core.experiment")
    saved = {"world_hidden": (32, 32), "structured_dynamics": False, "representation": "gencoord"}
    args = SimpleNamespace(
        velocity_fraction=None,
        batch_size=4,
        policy_horizon=2,
        horizon=2,
        hidden=[16, 24],
        world_model_checkpoint="last.pt",
        eager=True,
        iterations=1,
        num_envs=2,
        collection_steps=4,
    )
    settings = tracker.resolve_learner(settings, args, {"cfg": saved}, experiment)
    assert settings.hidden == (16, 24)
    assert settings.world_hidden == (32, 32)
    if blink:
        assert settings.structured_dynamics is False


@pytest.mark.parametrize(
    ("requested", "overrides", "checkpoint", "expected"),
    [
        (None, [], {"physics_backend": "PhysxCfg"}, ["physics=isaacsim_physx"]),
        (None, [], {"physics_backend": "MJWarpSolverCfg"}, ["physics=newton_mjwarp"]),
        (None, ["physics=newton_mjwarp"], {"physics_backend": "PhysxCfg"}, ["physics=newton_mjwarp"]),
        ("newton_mjwarp", [], {"physics_backend": "PhysxCfg"}, ["physics=newton_mjwarp"]),
        (None, [], None, []),
    ],
)
def test_physics_selection(requested, overrides, checkpoint, expected):
    assert tracker.physics_overrides(requested, overrides, checkpoint) == expected
