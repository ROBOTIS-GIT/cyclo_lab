import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

view = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/codex-scene-view.py"))


@pytest.mark.parametrize("multiple", [False, True])
def test_explicit_and_default_settings_preserve_physics_and_camera(multiple):
    explicit = NS(enable_markers=True, enable_live_plots=True, eye=(3, 4, 5))
    default = NS(enable_markers=True, enable_live_plots=True, eye=(1, 2, 3))
    sensor = NS(debug_vis=True, update_period=0.01)
    command = NS(debug_vis=True, resampling_time_range=(2, 4))
    rewards = NS(weight=3)
    cfg = NS(sim=NS(default_visualizer_cfg=default, visualizer_cfgs=[explicit] if multiple else explicit,
                    dt=0.01), scene=NS(contact_forces=sensor), commands=NS(velocity=command), rewards=rewards)
    view["configure_scene_view"](cfg)
    assert not explicit.enable_markers and not default.enable_markers
    assert not explicit.enable_live_plots and not default.enable_live_plots
    assert explicit.eye == (3, 4, 5) and default.eye == (1, 2, 3)
    assert not sensor.debug_vis and not command.debug_vis
    assert sensor.update_period == 0.01 and command.resampling_time_range == (2, 4)
    assert cfg.sim.dt == 0.01 and cfg.rewards is rewards


def test_missing_default_is_created_without_isaac_import():
    cfg = NS(sim=NS(default_visualizer_cfg=None, visualizer_cfgs=[]))
    view["configure_scene_view"](cfg, visualizer_factory=NS)
    assert cfg.sim.default_visualizer_cfg.enable_markers is False
    assert cfg.sim.default_visualizer_cfg.enable_live_plots is False
