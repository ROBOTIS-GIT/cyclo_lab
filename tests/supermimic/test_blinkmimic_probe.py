"""Training probes must evaluate the same ordered motion library as training."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cyclo_lab.manager_based.mimic._shared import motion_tracker
from cyclo_lab.manager_based.mimic.blinkmimic.evaluation import run_training_probe_if_due


@pytest.mark.parametrize("names", [("custom walk",), ("z_walk", "a_dance", "z_walk")])
def test_training_probe_preserves_selected_motions(tmp_path, monkeypatch, names):
    motions = [str(tmp_path / f"{name}.npz") for name in names]
    for motion in motions:
        Path(motion).touch()
    runner = SimpleNamespace(
        iteration=1000,
        cfg=SimpleNamespace(eprobe_every=125, eprobe_start=1000),
        bridge=SimpleNamespace(env=SimpleNamespace(num_envs=64)),
        sampler=Mock(),
    )
    save = Mock()

    def run_probe(command, **kwargs):
        save.assert_called_once_with()
        args, overrides = motion_tracker.parse_cli("play", command[3:])
        assert overrides == []
        reference = SimpleNamespace(
            trajectory_file=tmp_path / "default_dance1.npz",
            additional_motions=(tmp_path / "default_dance2.npz",),
        )
        cfg = SimpleNamespace(commands=SimpleNamespace(reference_trajectory=reference))
        selected, _ = motion_tracker.select_motions(cfg, "play", args)
        assert selected == motions
        assert [reference.trajectory_file, *reference.additional_motions] == motions
        args.probe_report.write_text(json.dumps([{"failure_frame": 12}, {"failure_frame": -1}]))

    monkeypatch.setattr("subprocess.run", run_probe)
    run_training_probe_if_due(
        runner, "Cyclo-BlinkMimic-K1-Rev2-001", tmp_path, "PhysxCfg", save, motions
    )
    runner.sampler.record_probe.assert_called_once_with([12])
