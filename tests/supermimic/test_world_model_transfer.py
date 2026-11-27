"""Native SuperMimic world-model transfer without policy or manifest metadata."""

from types import SimpleNamespace

import pytest
import torch

from cyclo_lab.manager_based.mimic._shared.checkpoint import load_checkpoint, restore_runner, save
from cyclo_lab.manager_based.mimic._shared.motion_tracker import resolve_learner
from cyclo_lab.manager_based.mimic.supermimic import experiment
from cyclo_lab.manager_based.mimic.supermimic.training.runner import IsaacSuperMimicRunner, RunnerCfg
from test_runner_contract import UnitEnv


def context_env():
    env = UnitEnv()
    env.encoder_bias = torch.full((1, 23), 0.003)
    env.tracking_friction = torch.tensor([[1.0]])
    env.tracking_com = torch.tensor([[0.01, -0.02, 0.03]])
    command = env.command_manager.get_term("reference_trajectory")
    command.motion_ends = torch.tensor([command.reference.num_frames])
    env.command_manager.get_term = lambda _: command
    return env


def transfer_args(path):
    return SimpleNamespace(
        world_model_checkpoint=path, checkpoint=None, resume=False,
        run_pattern=None, checkpoint_pattern=None, iterations=10,
        collection_steps=4, eager=True,
    )


@pytest.mark.parametrize("representation", ["gencoord", "maxcoord"])
def test_native_transfer_and_lab_checkpoint_roundtrip(tmp_path, representation):
    source = IsaacSuperMimicRunner(
        context_env(), RunnerCfg(representation=representation, actor_hidden_dims=(32, 24), dynamics_hidden_dims=(32, 24), critic_hidden_dims=(32, 24), eager=True)
    )
    source.bridge.sim_context.mean.add_(0.002)
    source.bridge.sim_context.scale.mul_(1.5)
    # This is exactly the original WorldModel.bundle() format.
    native = {"dynamics": source.dynamics.state_dict(), "sim_context": source.bridge.sim_context.state_dict()}
    path = tmp_path / "world_model.pt"
    torch.save(native, path)
    args = transfer_args(tmp_path)
    loaded_path, bundle = load_checkpoint("train", args, "supermimic", "task", tmp_path)
    assert loaded_path == path
    cfg = resolve_learner(RunnerCfg(representation=representation, actor_hidden_dims=(16, 16)), args, bundle, experiment)
    assert cfg.actor_hidden_dims == (16, 16) and cfg.dynamics_hidden_dims == (32, 24)
    target = IsaacSuperMimicRunner(context_env(), cfg)
    actor = {key: value.clone() for key, value in target.actor.state_dict().items()}
    critic = {key: value.clone() for key, value in target.critic.state_dict().items()}
    target.dynamics.action_scale.fill_(0.2)
    target.dynamics.action_offset.fill_(0.05)
    restore_runner(target, bundle, training=True, world_only=True, iterations=10)
    assert target.iteration == 0
    for module, expected in ((target.actor, actor), (target.critic, critic)):
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[key])
    for opts in target.checkpoint_optimizers().values():
        assert all(not opt.state for opt in opts)
    torch.testing.assert_close(target.bridge.context(), source.bridge.context())
    torch.testing.assert_close(
        target.bridge.sim_context.denormalize(target.bridge.context(), slice(4, None)),
        target.bridge.env.encoder_bias,
    )
    # Equal physical joint commands must produce equal predictions even with
    # a different destination action scale and offset.
    state = source.bridge.state()
    rep, aux = source.rep.represent(state), source.rep.extract_aux_state(state)
    action = torch.zeros(1, 23)
    mapped = (source.dynamics.proc(action) - target.dynamics.action_offset) / target.dynamics.action_scale
    expected = source.dynamics(rep, action, aux, source.bridge.context())
    actual = target.dynamics(rep, mapped, aux, target.bridge.context())
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    target.metrics.total_steps = 1234
    target.metrics.wall_time = 56.0
    save(target, tmp_path, [])
    resumed = IsaacSuperMimicRunner(context_env(), cfg)
    saved = torch.load(tmp_path / "last.pt", weights_only=True)
    restore_runner(resumed, saved, training=True, world_only=False, iterations=10)
    torch.testing.assert_close(resumed.bridge.context(), target.bridge.context())
    assert resumed.metrics.total_steps == 1234
    assert resumed.metrics.wall_time == 56.0
    # Older checkpoints still resume without requiring logging metadata.
    saved.pop("training_stats")
    restore_runner(resumed, saved, training=True, world_only=False, iterations=10)
    assert resumed.metrics.total_steps == 0
    assert resumed.metrics.wall_time == 0.0


def test_native_file_cannot_be_used_for_full_resume(tmp_path):
    path = tmp_path / "world_model.pt"
    torch.save({"dynamics": {}, "sim_context": {}}, path)
    args = transfer_args(None)
    args.checkpoint = path
    with pytest.raises(ValueError, match="use --world-model-checkpoint"):
        load_checkpoint("train", args, "supermimic", "task", tmp_path)


@pytest.mark.parametrize("invalid", ["representation", "context", "scale"])
def test_native_transfer_rejects_incompatible_inputs(invalid):
    source = IsaacSuperMimicRunner(context_env(), RunnerCfg(actor_hidden_dims=(16, 16), dynamics_hidden_dims=(16, 16), critic_hidden_dims=(16, 16), eager=True))
    bundle = {"dynamics": source.dynamics.state_dict(), "sim_context": source.bridge.sim_context.state_dict()}
    representation = "maxcoord" if invalid == "representation" else "gencoord"
    target = IsaacSuperMimicRunner(context_env(), RunnerCfg(representation=representation, actor_hidden_dims=(16, 16), dynamics_hidden_dims=(16, 16), critic_hidden_dims=(16, 16), eager=True))
    if invalid == "context":
        bundle["sim_context"] = {k: v[:-1] for k, v in bundle["sim_context"].items()}
    if invalid == "scale":
        bundle["sim_context"]["scale"][0] = 0
    with pytest.raises(ValueError, match="incompatible|positive scales"):
        restore_runner(target, bundle, training=True, world_only=True, iterations=10)
