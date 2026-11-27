"""Check the articulation selected by each motion tool before starting a simulator."""

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "scripts/tools/motion"


@pytest.mark.parametrize("script,scene", [
    ("csv_to_npz", "ReplayMotionsSceneCfg"),
    ("soma_retargeter_csv_converter", "ReplayMotionsSceneCfg"),
    ("replay_csv", "ReplaySceneCfg"),
    ("replay_npz", "ReplaySceneCfg"),
])
@pytest.mark.parametrize("options,expected", [
    ([], "k1_rev2"),
    (["--robot", "k1_rev1"], "k1_rev1"),
    (["--robot", "k1_rev2"], "k1_rev2"),
])
def test_motion_tool_selects_robot(monkeypatch, script, scene, options, expected):
    from isaaclab.app import AppLauncher

    path = TOOLS / f"{script}.py"
    monkeypatch.setattr(sys, "argv", [str(path), "-f", "motion", *options])
    with patch("isaaclab.app.AppLauncher") as launcher:
        launcher.add_app_launcher_args = AppLauncher.add_app_launcher_args
        module = runpy.run_path(str(path))
    scene_cfg = module[scene](num_envs=1, env_spacing=1.0)
    asset = Path(scene_cfg.robot.spawn.asset_path)
    assert asset.parts[-2:] == (expected, "k1.urdf")
