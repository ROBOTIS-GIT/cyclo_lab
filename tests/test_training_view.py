"""CPU checks for service framing without importing or launching Isaac Sim."""

import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "training_view", Path(__file__).resolve().parents[1] / "scripts/codex-training-view.py"
)
view = importlib.util.module_from_spec(spec)
spec.loader.exec_module(view)


def grid():
    return [(x, y, 0) for x in (-3.75, -1.25, 1.25, 3.75) for y in (-3.75, -1.25, 1.25, 3.75)]


@pytest.mark.parametrize("origins", [grid(), [(12, -8, 2)], [(-20, 3, 1), (4, 11, -2)]])
def test_every_robot_extent_fits_camera_cone(origins):
    eye, target = view.frame_origins(origins, 65)
    forward = [b - a for a, b in zip(eye, target)]
    distance = math.sqrt(sum(value * value for value in forward))
    forward = [value / distance for value in forward]
    for origin in origins:
        for offset in itertools.product((-1, 1), (-1, 1), (-1, 2.5)):
            ray = [o + d - e for o, d, e in zip(origin, offset, eye)]
            depth = sum(a * b for a, b in zip(ray, forward))
            sideways = math.sqrt(max(0, sum(value * value for value in ray) - depth * depth))
            assert depth > 0
            assert sideways / depth < math.tan(math.radians(65 / 2))


@pytest.mark.parametrize("profile,viz,expected", [
    ("omy", ["viser"], True), ("sapiens", "viser", True),
    ("omy", ["kit"], False), (None, ["viser"], False), ("other", ["viser"], False),
])
def test_only_service_viser_training_is_changed(profile, viz, expected):
    assert view.service_view_enabled(profile, viz) is expected


@pytest.mark.parametrize("origins", [[], [(0, float("nan"), 0)], [(1, 2)]])
def test_invalid_origins_fail_clearly(origins):
    with pytest.raises(ValueError):
        view.frame_origins(origins, 65)


def test_initial_and_reconnecting_clients_keep_full_grid_orientation():
    lights = []
    gui_calls = []
    server = SimpleNamespace(gui=SimpleNamespace(reset=lambda: gui_calls.append("reset"),
        configure_theme=lambda **kwargs: gui_calls.append(kwargs)), initial_camera=SimpleNamespace(), scene=SimpleNamespace(
        add_light_directional=lambda name, **kwargs: lights.append((name, kwargs))))

    class Viewer:
        _server = server

        def set_visible_worlds(self, worlds):
            self.worlds = worlds

        def set_camera(self, eye, pitch, yaw):
            # Pinned Newton caches eye/orientation for every future connection.
            self.eye, self.pitch, self.yaw = eye, pitch, yaw
            server.initial_camera.position = eye

        def connect(self):
            pitch, yaw = map(math.radians, (self.pitch, self.yaw))
            return self.eye, (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))

    viewer = Viewer()
    visualizer = type("ViserVisualizer", (), {})()
    visualizer._viewer = viewer
    visualizer._focal_length_to_vertical_fov_degrees = lambda: 65
    origins = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=grid)))
    pivots = []
    env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins), sim=SimpleNamespace(
        visualizers=[visualizer], set_camera_view=lambda *args: pivots.append(args)))
    eye, target = view.configure_training_view(env)
    assert viewer.worlds == list(range(16))
    assert server.initial_camera.position == eye
    assert server.initial_camera.look_at == target
    assert server.initial_camera.fov == math.radians(65)
    assert pivots == [(eye, target)]
    for _ in range(2):
        position, direction = viewer.connect()
        toward = [b - a for a, b in zip(position, target)]
        length = math.sqrt(sum(value * value for value in toward))
        assert direction == pytest.approx([value / length for value in toward])
    assert len(lights) == 2
    assert gui_calls == ["reset", {"show_logo": False, "show_share_button": False}]
    assert all(not kwargs["cast_shadow"] for _, kwargs in lights)
