"""Compare flattened deployment inputs with each native actor on the same history."""

import numpy as np
from dataclasses import asdict
from importlib import import_module
import onnxruntime as ort
import pytest
import torch
import yaml
from tensordict import TensorDict

from cyclo_lab.manager_based.mimic.blinkmimic.training.runner import BlinkCfg, IsaacBlinkMimicRunner
from cyclo_lab.manager_based.mimic.supermimic.training.runner import WorldModelLearnerCfg, ActorCriticPolicyLearnerCfg
from cyclo_lab.manager_based.mimic.supermimic.training.runner import IsaacSuperMimicRunner, RunnerCfg
from cyclo_lab.manager_based.mimic._shared.export import export_actors, export_sim2real_cfg
from cyclo_lab.manager_based.mimic._shared.checkpoint import load, save
from test_runner_contract import UnitEnv


@pytest.mark.parametrize("family,step", [("blinkmimic", 1000), ("supermimic", 0), ("supermimic", 1000)])
def test_checkpoint_and_export_match_trained_actor(tmp_path, family, step):
    env = UnitEnv()
    torch.manual_seed(12)
    if family == "blinkmimic":
        cfg = BlinkCfg(
            enc_hidden=(16, 12), dec_hidden=(24, 8), wm_hidden=(32, 16),
            wm_window=2, horizon=2, wm_batch=7, pi_batch=4,
        )
        runner = IsaacBlinkMimicRunner(env, cfg)
        for _ in range(12):
            runner.step(True)
        inputs = [("motion", runner.mdp.obs_dim, runner.motion.goal_dim(cfg.goal_frames))]
    else:
        cfg = RunnerCfg(
            actor_obs="proprio", history_length=10, slot_mix={"motion": 0.5, "velocity": 0.5},
            actor_hidden_dims=(16, 12), dynamics_hidden_dims=(32, 16), critic_hidden_dims=(24, 8),
            eager=True, wm=WorldModelLearnerCfg(num_batches=1, batch_size=7, horizon=2),
            policy=ActorCriticPolicyLearnerCfg(batch_size=4, horizon=2),
        )
        runner = IsaacSuperMimicRunner(env, cfg)
        runner.collect(12)
        inputs = [(s.name, runner.actor.obs_dim, s.goal_dim) for s in runner.specs]

    metrics = runner.update()
    assert all(np.isfinite(list(metrics.values())))
    completed = step + 1 if family == "supermimic" else step
    runner.iteration = completed
    save(runner, tmp_path, [])
    last = torch.load(tmp_path / "last.pt", weights_only=True)
    historical = torch.load(tmp_path / f"model_{step}.pt", weights_only=True)
    assert last["iteration"] == historical["iteration"] == step
    assert last["optimizers"] and "optimizers" not in historical
    experiment = import_module(f"cyclo_lab.manager_based.mimic.{family}.experiment")
    restored_cfg = experiment.restore_config(last["cfg"])
    assert asdict(restored_cfg) == asdict(cfg)
    runner = experiment.Runner(UnitEnv(), restored_cfg)
    load(runner, last, training=True)
    assert runner.iteration == completed
    legacy = dict(last, format="cyclo_lab.mimic.v1", iteration=completed)
    load(runner, legacy, training=True)
    assert runner.iteration == completed
    export_actors(runner, tmp_path)
    runner.actor.eval()

    for encoder, width, goal_width in inputs:
        history = torch.randn(1, cfg.hist if family == "blinkmimic" else cfg.history_length, width)
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


@pytest.mark.parametrize("representation", ["gencoord", "maxcoord"])
def test_rep_state_yaml_matches_onnx_input(tmp_path, representation):
    env = UnitEnv()
    data = env.scene["robot"].data
    data.default_joint_stiffness = data.default_joint_damping = data.default_joint_pos
    cfg = RunnerCfg(representation=representation, actor_obs='rep_state', history_length=1, actor_hidden_dims=(16, 16), dynamics_hidden_dims=(16, 16), critic_hidden_dims=(16, 16))
    runner = IsaacSuperMimicRunner(env, cfg)
    export_sim2real_cfg(runner, tmp_path)
    export_actors(runner, tmp_path)

    observation = yaml.safe_load((tmp_path / "params/sim2real.yaml").read_text())["observations"]["supermimic"]
    terms = observation["history"]["terms"]
    assert list(terms) == ["rep_state"]
    assert terms["rep_state"] == {"scale": [1.0] * runner.rep.rep_state_dim, "clip": None, "history_length": 1}
    goal = observation["motion_goal"]
    goal_width = goal["frames"] * sum(len(term["scale"]) for group in goal["groups"] for term in group["terms"].values())
    session = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, runner.rep.rep_state_dim + goal_width]


def test_new_motion_library_and_checkpoint_selection(tmp_path):
    from cyclo_lab.manager_based.mimic._shared.checkpoint import find_checkpoint
    from cyclo_lab.manager_based.mimic._shared.motion_library_command import MotionLibraryCommand
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
    runner = IsaacBlinkMimicRunner(UnitEnv(), BlinkCfg(enc_hidden=(8, 8), dec_hidden=(8, 8), wm_hidden=(8, 8)))
    for step in range(1000, 17000, 1000):
        runner.iteration = step
        save(runner, tmp_path, [])
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


def test_blinkmimic_play_keeps_reference_skip_and_resume_uses_current_transform():
    cfg = BlinkCfg(enc_hidden=(8, 8), dec_hidden=(8, 8), wm_hidden=(8, 8))
    source = IsaacBlinkMimicRunner(UnitEnv(), cfg)
    source.actor.ref_q0.fill_(0.25)
    source.actor.ref_alpha.fill_(0.5)
    source.average.module.load_state_dict(source.actor.state_dict())
    saved = source.actor_checkpoint()
    target = IsaacBlinkMimicRunner(UnitEnv(), cfg)
    current_q0 = target.actor.ref_q0.clone()
    current_alpha = target.actor.ref_alpha.clone()
    target.restore_actor(saved, training=False)
    torch.testing.assert_close(target.actor.ref_q0, source.actor.ref_q0)
    torch.testing.assert_close(target.average.module.ref_alpha, source.actor.ref_alpha)
    target = IsaacBlinkMimicRunner(UnitEnv(), cfg)
    target.restore_actor(saved, training=True)
    torch.testing.assert_close(target.actor.ref_q0, current_q0)
    torch.testing.assert_close(target.average.module.ref_alpha, current_alpha)
    torch.testing.assert_close(saved["actor"]["ref_q0"], torch.full((23,), 0.25))
