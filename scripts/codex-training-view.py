"""Service-only training presentation; importable without Isaac Sim or a GPU."""

import math


def service_view_enabled(profile, visualizers):
    """Leave standalone training and non-Viser runs unchanged."""
    if isinstance(visualizers, str):
        visualizers = visualizers.split(",")
    return profile in {"omy", "sapiens"} and "viser" in (visualizers or ())


def frame_origins(origins, vertical_fov_degrees):
    """Fit actual world origins plus arm/humanoid extents in a landscape view.

    The bounding sphere fits both frustum axes for aspect ratios >= 1. No
    assumptions about grid ordering, spacing, or a particular environment count.
    Margins include the OMY table and the standing/jumping Sapiens robot.
    """
    points = [tuple(float(value) for value in point) for point in origins]
    if not points or any(len(point) != 3 or not all(map(math.isfinite, point)) for point in points):
        raise ValueError("Training view requires finite XYZ environment origins")
    if not math.isfinite(vertical_fov_degrees) or not 0 < vertical_fov_degrees < 180:
        raise ValueError("Camera field of view must be between 0 and 180 degrees")
    low = tuple(min(point[axis] for point in points) - margin for axis, margin in enumerate((1.0, 1.0, 1.0)))
    high = tuple(max(point[axis] for point in points) + margin for axis, margin in enumerate((1.0, 1.0, 2.5)))
    target = tuple((a + b) / 2 for a, b in zip(low, high))
    radius = math.sqrt(sum(((b - a) / 2) ** 2 for a, b in zip(low, high)))
    distance = 1.1 * radius / math.sin(math.radians(vertical_fov_degrees) / 2)
    offset = (-1.0, -1.0, 0.9)
    length = math.sqrt(sum(value * value for value in offset))
    eye = tuple(center + distance * value / length for center, value in zip(target, offset))
    return eye, target


def configure_training_view(env):
    """Frame all training worlds using the pinned Newton/Viser camera lifecycle."""
    origins = env.scene.env_origins.detach().cpu().tolist()
    for visualizer in env.sim.visualizers:
        if type(visualizer).__name__ != "ViserVisualizer":
            continue
        fov = visualizer._focal_length_to_vertical_fov_degrees()
        eye, target = frame_origins(origins, fov)
        direction = tuple(b - a for a, b in zip(eye, target))
        pitch = math.degrees(math.atan2(direction[2], math.hypot(*direction[:2])))
        yaw = math.degrees(math.atan2(direction[1], direction[0]))
        # Compatibility boundary with the pinned Newton backend; no upstream edits.
        viewer = visualizer._viewer
        viewer.set_visible_worlds(list(range(len(origins))))
        viewer.set_camera(eye, pitch, yaw)
        server = viewer._server
        server.initial_camera.look_at = target
        server.initial_camera.fov = math.radians(fov)
        server.scene.add_light_directional("/cyclo/key", intensity=2.5,
                                           position=(3.0, 2.0, 4.0), cast_shadow=False)
        server.scene.add_light_directional("/cyclo/fill", intensity=1.2,
                                           position=(-3.0, -1.0, 2.0), cast_shadow=False)
        env.sim.set_camera_view(eye, target)
        return eye, target
    raise RuntimeError("Service training requires the pinned Viser visualizer")
