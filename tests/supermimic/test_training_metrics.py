"""Check metric definitions with unequal episode lengths and multiple WM batches."""

from types import SimpleNamespace as NS

import pytest
import torch

from cyclo_lab.utils.episode_metrics import EpisodeMetrics
from cyclo_lab.manager_based.mimic._shared.motion_tracker import parse_cli
from cyclo_lab.manager_based.mimic.blinkmimic.training.runner import IsaacBlinkMimicRunner
from cyclo_lab.manager_based.mimic._shared.training_loop import TrainingLogWindow


@pytest.mark.parametrize("family,expected", [("blinkmimic", 3500), ("supermimic", 1_000_000)])
def test_iteration_defaults_and_explicit_override(family, expected):
    args, _ = parse_cli("train", [], family=family)
    assert args.iterations == expected
    args, _ = parse_cli("train", ["--iterations", "7"], family=family)
    assert args.iterations == 7


def test_blink_episode_normalization_and_termination_fraction():
    metrics = EpisodeMetrics(2, "cpu", reward_scale=0.1)
    metrics.record(torch.tensor([0.1, 0.2]), torch.tensor([True, False]), terminated=torch.tensor([True, False]))
    metrics.record(torch.tensor([0.3, 0.4]), torch.tensor([True, True]), terminated=torch.tensor([False, True]))
    logs = metrics.scalars()
    assert logs["BlinkMimic/normalized_episode_return"] == pytest.approx(2.0)
    assert logs["BlinkMimic/termination_fraction"] == pytest.approx(2 / 3)
    assert logs["Train/mean_reward"] == pytest.approx(1 / 3)
    assert metrics.total_steps == 4
    # Reading a new interval must not reuse the previous completed episodes.
    logs = metrics.scalars()
    assert "BlinkMimic/normalized_episode_return" not in logs
    assert logs["BlinkMimic/termination_fraction"] == 0


def test_super_episode_extras_are_snapshotted_and_drained():
    metrics = EpisodeMetrics(1, "cpu")
    value = torch.tensor([2.0, 4.0])
    metrics.record(torch.tensor([1.0]), torch.tensor([False]), extras={"log": {"Reward/test": value}})
    value.fill_(100)
    metrics.record(torch.tensor([1.0]), torch.tensor([True]), extras={"episode": {"Reward/test": 6.0, "score": 5.0}})
    logs = metrics.scalars()
    assert logs["Reward/test"] == 4.0
    assert logs["Episode/score"] == 5.0
    assert logs["Train/mean_reward"] == 2.0
    assert "Reward/test" not in metrics.scalars()


def test_blink_keeps_wm_error_and_native_nonfinite_count():
    runner = IsaacBlinkMimicRunner.__new__(IsaacBlinkMimicRunner)
    runner.cfg = NS(lr=4e-4, wm_batches=2, wm_window=4)
    runner.iteration = 7
    runner.decay_start = None
    runner.sampler = None
    runner.buffer = object()
    runner.optimizers = {name: NS(param_groups=[{}]) for name in ("actor", "critic", "dynamics")}
    results = iter([(torch.tensor(2.0), torch.tensor(4.0)), (torch.tensor(6.0), torch.tensor(8.0))])
    runner.wm_learner = NS(update=lambda *a, **kw: next(results))
    runner.policy_learner = NS(
        update=lambda _: tuple(torch.tensor(v) for v in (-1.0, 0.2, 0.5, 0.1, 0.8)),
        nonfinite=torch.tensor(3),
    )
    logs = runner.update()
    assert logs["Loss/world_model"] == 4.0
    assert logs["BlinkMimic/world_model_error_4_steps"] == 6.0
    assert "BlinkMimic/world_model_error_8_steps" not in logs
    assert logs["BlinkMimic/nonfinite_updates"] == 3
    assert logs["Policy/learning_rate_multiplier"] == 4.0
    assert runner.iteration == 8


def test_log_window_averages_samples_but_not_counters_or_reciprocal_rates():
    window = TrainingLogWindow()
    window.add({"Loss/world_model": 2.0, "Loss/learning_rate": 0.01,
                "Perf/total_steps": 10, "BlinkMimic/nonfinite_updates": 2}, seconds=1.0, steps=10)
    window.add({"Loss/world_model": 6.0, "Loss/learning_rate": 0.005,
                "Perf/total_steps": 40, "BlinkMimic/nonfinite_updates": 3,
                "Train/mean_episode_length": 9.0}, seconds=3.0, steps=30)
    logs = window.scalars()
    assert logs["Loss/world_model"] == 4.0
    assert logs["Loss/learning_rate"] == 0.005
    assert logs["Perf/total_steps"] == 40
    assert logs["BlinkMimic/nonfinite_updates"] == 3
    assert logs["Train/mean_episode_length"] == 9.0
    assert logs["Perf/iteration_time"] == 2.0
    assert logs["Perf/iterations_per_second"] == 0.5
    assert logs["Perf/total_fps"] == 10.0


def test_blink_episode_interval_survives_intermediate_iterations():
    metrics = EpisodeMetrics(2, "cpu", reward_scale=0.1)
    metrics.record(torch.tensor([0.1, 0.2]), torch.tensor([True, False]), terminated=torch.tensor([True, False]))
    assert metrics.scalars(reset=False)["BlinkMimic/termination_fraction"] == 1
    metrics.record(torch.tensor([0.3, 0.4]), torch.tensor([True, True]), terminated=torch.tensor([False, True]))
    logs = metrics.scalars(reset=False)
    assert logs["BlinkMimic/termination_fraction"] == pytest.approx(2 / 3)
    assert logs["BlinkMimic/normalized_episode_return"] == pytest.approx(2.0)
    metrics.reset_interval()
    assert "BlinkMimic/normalized_episode_return" not in metrics.scalars()


def test_native_logging_interval_defaults():
    from cyclo_lab.manager_based.mimic.blinkmimic.training.runner import BlinkCfg
    from cyclo_lab.manager_based.mimic.supermimic.training.runner import RunnerCfg

    assert BlinkCfg().log_every == 50
    assert RunnerCfg().log_interval == 10


@pytest.mark.parametrize("family,start,total,expected,headings,saved_steps", [
    ("supermimic", 0, 25, [0, 10, 20], ["0/2", "1/2", "2/2"], [1, 25]),
    ("supermimic", 11, 25, [20], ["2/2"], [25]),
    ("blinkmimic", 0, 101, [50, 100, 101], ["50/101", "100/101", "101/101"], [100, 101]),
    ("blinkmimic", 1, 25, [25], ["25/25"], [25]),
])
def test_native_log_steps_and_completed_checkpoint_iteration(tmp_path, monkeypatch, capsys,
                                                            family, start, total, expected, headings, saved_steps):
    from contextlib import nullcontext
    from importlib import import_module
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from isaaclab_rl.entrypoints import common
    from cyclo_lab.manager_based.mimic._shared import checkpoint, export, training_loop
    from cyclo_lab.utils import wandb_models

    monkeypatch.setattr(common, "dump_train_configs", lambda *a: None)
    monkeypatch.setattr(export, "export_sim2real_cfg", lambda *a: None)
    monkeypatch.setattr(import_module("cyclo_lab.utils.export_git_state"), "export_git_state", lambda *a: None)
    monkeypatch.setattr(wandb_models, "model_upload_run", lambda *a: nullcontext(None))
    saved = []
    monkeypatch.setattr(checkpoint, "save", lambda runner, *a, **kw: saved.append(runner.iteration))
    runner = NS(family=family, iteration=start, cfg=NS(log_every=50, log_interval=10),
                env=NS(unwrapped=NS(cfg=None)), metrics=EpisodeMetrics(1, "cpu"))

    def train(_):
        runner.iteration += 1
        runner.metrics.record(torch.tensor([1.0]), torch.tensor([False]))
        return {"Loss/world_model": float(runner.iteration)}

    runner.train_iteration = train
    args = NS(output=tmp_path, iterations=total, collection_steps=None)
    training_loop.run_training_loop(runner, args, tmp_path, "task", [], "test", [], None)
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [value.step for value in events.Scalars("Loss/world_model")] == expected
    headers = [line for line in capsys.readouterr().out.splitlines() if "Learning iteration" in line]
    assert len(headers) == len(headings)
    assert all(f"Learning iteration {value} " in line for value, line in zip(headings, headers))
    assert saved == saved_steps
