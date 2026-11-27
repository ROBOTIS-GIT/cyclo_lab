"""Cross-task playback should use the destination motion library."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


from cyclo_lab.manager_based.mimic._shared import motion_tracker as tracker


@pytest.mark.parametrize("action", ["train", "play"])
@pytest.mark.parametrize("multiple", [False, True])
def test_motion_file_preserves_trailing_overrides(action, multiple):
    motions = ["source/cyclo_lab/data/motions/K1_rev2/dance2/dance2.npz"]
    if multiple:
        motions.append("source/cyclo_lab/data/motions/K1_rev2/dance1/dance1.npz")
    overrides = ["physics=newton_mjwarp", "agent.wm.batch_size=32"]
    args, remaining = tracker.parse_cli(action, ["--motion-file", *motions, *overrides])
    assert args.motion_file == [Path(motion) for motion in motions]
    assert remaining == overrides


def test_motion_file_with_overrides_and_equals_option():
    args, overrides = tracker.parse_cli("train", [
        "physics=isaacsim_physx", "--motion-file", "dance2.npz",
        "--num-envs=4", "physics=newton_mjwarp",
    ])
    assert args.motion_file == [Path("dance2.npz")]
    assert args.num_envs == 4
    assert overrides == ["physics=isaacsim_physx", "physics=newton_mjwarp"]


@pytest.mark.parametrize(
    ("action", "task", "defaults", "override", "world_only", "expected"),
    [
        ("train", "001", "ab", None, False, "ab"),
        ("train", "002", "ab", None, False, "ab"),
        ("train", "001", "abc", None, False, "abc"),
        ("train", "001", "ab", "ac", False, "ac"),
        ("train", "001", "ab", None, True, "ab"),
        ("play", "001", "abc", None, False, "abc"),
        ("play", "002", "abc", None, False, "abc"),
        ("export", "002", "abc", None, False, "abc"),
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
    selected, _ = tracker.select_motions(cfg, action, args)
    assert selected == [files[c] for c in expected]
    assert [reference.trajectory_file, *reference.additional_motions] == selected


@pytest.mark.parametrize("blink", [True, False], ids=["blinkmimic", "supermimic"])
def test_world_model_initialization_preserves_new_actor_width(blink):
    from dataclasses import asdict
    from cyclo_lab.manager_based.mimic.blinkmimic.training.runner import BlinkCfg
    from cyclo_lab.manager_based.mimic.supermimic.training.runner import RunnerCfg

    settings = (BlinkCfg(enc_hidden=(16, 24), dec_hidden=(12, 8)) if blink else
                RunnerCfg(actor_hidden_dims=(16, 24)))
    source = (BlinkCfg(wm_hidden=(32, 32), wm_structured=False) if blink else
              RunnerCfg(dynamics_hidden_dims=(32, 32)))
    family = "blinkmimic" if blink else "supermimic"
    experiment = importlib.import_module(f"cyclo_lab.manager_based.mimic.{family}.experiment")
    args, _ = tracker.parse_cli("train", ["--world-model-checkpoint", "last.pt"])
    saved = asdict(source)
    settings = tracker.resolve_learner(settings, args, {"cfg": saved}, experiment)
    if blink:
        assert settings.enc_hidden == (16, 24) and settings.dec_hidden == (12, 8)
        assert settings.wm_hidden == (32, 32) and settings.wm_structured is True
    else:
        assert settings.actor_hidden_dims == (16, 24) and settings.dynamics_hidden_dims == (32, 32)
        saved["representation"] = "maxcoord"
        with pytest.raises(ValueError, match="representation differs"):
            tracker.resolve_learner(settings, args, {"cfg": saved}, experiment)


@pytest.mark.parametrize("family,task", [
    ("blinkmimic", "Cyclo-BlinkMimic-K1-Rev2-001"),
    ("supermimic", "Cyclo-SuperMimic-GenCoord-K1-Rev2-001"),
])
def test_resume_preserves_explicit_agent_overrides(family, task):
    from dataclasses import asdict
    from isaaclab_tasks.utils import resolve_task_config

    _, original = resolve_task_config(task, "learner_cfg_entry_point", overrides=[])
    saved = asdict(original)
    if family == "supermimic":
        saved["policy"]["horizon"] = 12
        saved["wm"]["num_batches"] = 1
        overrides = ["agent.buffer_size=64", "agent.replay_ratio=4", "agent.eager=true",
                     "agent.wm.batch_size=7", "agent.policy.batch_size=5",
                     f"agent.wm.num_batches={original.wm.num_batches}"]
    else:
        saved["horizon"] = 12
        saved["wm_batches"] = 1
        overrides = ["agent.buffer_steps=64", "agent.utd=0.2", "agent.wm_structured=false",
                     "agent.wm_batch=7", "agent.pi_batch=5", f"agent.wm_batches={original.wm_batches}"]
    args, overrides = tracker.parse_cli("train", ["--iterations", "100", "env.scene.num_envs=2", *overrides])
    env, current = resolve_task_config(task, "learner_cfg_entry_point", overrides=overrides)
    experiment = importlib.import_module(f"cyclo_lab.manager_based.mimic.{family}.experiment")
    resumed = tracker.resolve_learner(current, args, {"cfg": saved}, experiment, overrides=overrides)
    assert env.scene.num_envs == 2
    if family == "supermimic":
        assert resumed.buffer_size == 64 and resumed.replay_ratio == 4
        assert resumed.eager is True
        assert resumed.wm.batch_size == 7 and resumed.policy.batch_size == 5
        assert resumed.wm.num_batches == original.wm.num_batches
        assert resumed.policy.horizon == 12
    else:
        assert resumed.buffer_steps == 64 and resumed.utd == 0.2
        assert resumed.wm_structured is False
        assert resumed.wm_batch == 7 and resumed.pi_batch == 5
        assert resumed.wm_batches == original.wm_batches
        assert resumed.horizon == 12


@pytest.mark.parametrize("current", [True, False])
@pytest.mark.parametrize("world_only", [True, False])
def test_blinkmimic_training_uses_current_wm_structured(current, world_only):
    from dataclasses import asdict
    from cyclo_lab.manager_based.mimic.blinkmimic import experiment
    from cyclo_lab.manager_based.mimic.blinkmimic.training.runner import BlinkCfg

    assert BlinkCfg().wm_structured is True
    saved = {"cfg": asdict(BlinkCfg(wm_structured=not current))}
    args, _ = tracker.parse_cli("train", ["--world-model-checkpoint", "last.pt"] if world_only else [])
    resolved = tracker.resolve_learner(BlinkCfg(wm_structured=current), args, saved, experiment)
    assert resolved.wm_structured is current
    if not world_only:
        playback = tracker.resolve_learner(
            BlinkCfg(wm_structured=current), args, saved, experiment, training=False
        )
        assert playback.wm_structured is not current


def test_agent_override_cannot_silently_change_resume_encoder_mix():
    from dataclasses import asdict
    from cyclo_lab.manager_based.mimic.supermimic import experiment
    from cyclo_lab.manager_based.mimic.supermimic.training.runner import RunnerCfg

    saved = asdict(RunnerCfg())
    args, _ = tracker.parse_cli("train", [])
    with pytest.raises(ValueError, match="Full resume preserves the encoder mix"):
        tracker.resolve_learner(
            RunnerCfg(slot_mix={"motion": 0.5, "velocity": 0.5}), args, {"cfg": saved}, experiment,
            overrides=["agent.slot_mix.motion=0.5", "agent.slot_mix.velocity=0.5"],
        )


def test_checkpoint_search_skips_empty_newer_run(tmp_path):
    from cyclo_lab.manager_based.mimic._shared.checkpoint import find_checkpoint

    old = tmp_path / "task/2026-01-01"
    old.mkdir(parents=True)
    (old / "last.pt").touch()
    (tmp_path / "task/2026-01-02").mkdir()
    assert find_checkpoint(tmp_path, "task", None) == old / "last.pt"
