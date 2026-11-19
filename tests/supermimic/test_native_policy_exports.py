"""Compare flattened deployment inputs with each native actor on the same history."""

import numpy as np
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from cyclo_lab.manager_based.mimic.blinkmimic.core.runner import BlinkCfg, IsaacBlinkMimicRunner
from cyclo_lab.manager_based.mimic.supermimic.core.runner import IsaacSuperMimicRunner, RunnerCfg
from cyclo_lab.manager_based.mimic.export import export_actors
from cyclo_lab.manager_based.mimic.checkpoint import load, save
from test_runner_contract import UnitEnv


@pytest.mark.parametrize("family", ["blinkmimic", "supermimic"])
def test_checkpoint_and_export_match_trained_actor(tmp_path, family):
    env = UnitEnv()
    torch.manual_seed(12)
    if family == "blinkmimic":
        cfg = BlinkCfg(hidden=(16, 16), world_hidden=(16, 16), horizon=2, policy_horizon=2, batch_size=4)
        runner = IsaacBlinkMimicRunner(env, cfg)
        for _ in range(12):
            runner.step(True)
        inputs = [("motion", runner.mdp.obs_dim, runner.motion.goal_dim(cfg.goal_frames))]
    else:
        cfg = RunnerCfg(
            actor_obs="proprio",
            history_length=10,
            velocity_fraction=0.5,
            hidden=(16, 16),
            horizon=2,
            policy_horizon=2,
            batch_size=4,
            eager=True,
        )
        runner = IsaacSuperMimicRunner(env, cfg)
        runner.collect(12)
        inputs = [(s.name, runner.actor.obs_dim, s.goal_dim) for s in runner.specs]

    metrics = runner.update()
    assert all(np.isfinite(list(metrics.values())))
    runner.iteration = 1000
    save(runner, tmp_path, {}, [])
    last = torch.load(tmp_path / "last.pt", weights_only=True)
    historical = torch.load(tmp_path / "model_1000.pt", weights_only=True)
    assert last["optimizers"] and "optimizers" not in historical
    load(runner, last, training=True)
    export_actors(runner, tmp_path)
    runner.actor.eval()

    for encoder, width, goal_width in inputs:
        history = torch.randn(1, cfg.history_length, width)
        goal = torch.randn(1, goal_width)
        if family == "blinkmimic":
            packed = history.flatten(1)
            action = runner.average.module(history, goal)
        else:
            runner.actor.set_slot_mix({encoder: 1})
            chunks = history.split(runner.rep.proprio_term_widths(23), dim=2)
            packed = torch.cat([chunk.flatten(1) for chunk in chunks], dim=1)
            action = runner.actor(TensorDict({"history": history, f"goal_{encoder}": goal}, batch_size=[1]))
        path = tmp_path / ("velocity" if encoder == "velocity" else "") / "policy.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        result = session.run(None, {session.get_inputs()[0].name: torch.cat((packed, goal), -1).numpy()})[0]
        np.testing.assert_allclose(result, action.detach().numpy(), rtol=1e-5, atol=1e-6)


def test_new_motion_library_and_checkpoint_selection(tmp_path):
    from cyclo_lab.manager_based.mimic.checkpoint import find_checkpoint
    from cyclo_lab.manager_based.mimic.motion_library_command import MotionLibraryCommand
    from types import SimpleNamespace

    ends = torch.tensor([3, 8])
    source = SimpleNamespace(cfg=SimpleNamespace(clip_timeout=False), frame_ids=torch.tensor([1, 2, 6, 7]), motion_ends=ends)
    reset_ids = MotionLibraryCommand._advance_reference(source)
    assert reset_ids.tolist() == [1, 3]
    source.cfg = SimpleNamespace(start_from_zero=True)
    MotionLibraryCommand._sample_start_frames(source, reset_ids)
    assert source.frame_ids.tolist() == [2, 3, 7, 0]

    run = tmp_path / "task" / "run1"
    run.mkdir(parents=True)
    checkpoint = run / "last.pt"
    checkpoint.touch()
    assert find_checkpoint(tmp_path, "task", None) == checkpoint
    assert find_checkpoint(tmp_path, "task", run) == checkpoint


def test_numbered_history_retention_and_world_model_only(tmp_path):
    runner = IsaacBlinkMimicRunner(UnitEnv(), BlinkCfg(hidden=(8, 8), world_hidden=(8, 8)))
    for step in range(1000, 17000, 1000):
        runner.iteration = step
        save(runner, tmp_path, {}, [])
    retained = {int(p.stem.split("_")[-1]) for p in tmp_path.glob("model_*.pt")}
    assert retained == {10000, 12000, 13000, 14000, 15000, 16000}
    other = IsaacBlinkMimicRunner(UnitEnv(), runner.cfg)
    actor = {name: value.clone() for name, value in other.actor.state_dict().items()}
    load(other, torch.load(tmp_path / "model_16000.pt", weights_only=True), training=True, world_only=True)
    assert other.iteration == 0
    for name, value in other.dynamics.state_dict().items():
        torch.testing.assert_close(value, runner.dynamics.state_dict()[name], rtol=0, atol=0)
    for name, value in other.actor.state_dict().items():
        torch.testing.assert_close(value, actor[name], rtol=0, atol=0)
