#!/usr/bin/env python3

"""Bring up a registered Cyclo manager environment with an optional Zenoh ROS2 bridge."""

from __future__ import annotations

import argparse
import signal
import threading
import time
from dataclasses import fields, is_dataclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", help="Registered Cyclo Gym task ID.")
parser.add_argument(
    "--bridge",
    default="none",
    choices=("none", "ffw_sg2", "ffw_sh5"),
    help="Optional ROS2-compatible topic bridge.",
)
parser.add_argument(
    "--camera_view",
    default="none",
    choices=("none", "operator"),
    help="Optional local operator camera view.",
)
parser.add_argument("--num_steps", type=int, default=0, help="Stop after this many steps; 0 runs indefinitely.")
parser.add_argument("--max_runtime", type=float, default=0.0, help="Stop after this many seconds; 0 runs indefinitely.")
parser.add_argument("--report_interval", type=int, default=120, help="Steps between control-rate reports; 0 disables.")
parser.add_argument("--disable_head", action="store_true", help="FFW-SH5 only: disable the head command topic.")
parser.add_argument("--disable_lift", action="store_true", help="FFW-SH5 only: disable the lift command topic.")
parser.add_argument("--disable_cmd_vel", action="store_true", help="FFW-SH5 only: disable mobile-base commands.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.task is None:
    parser.error("the following argument is required: --task")
if args_cli.camera_view == "operator":
    args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import cyclo_lab  # noqa: F401 - registers Cyclo Gym tasks after AppLauncher
import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg


class RateLimiter:
    """Maintain the environment's configured control period."""

    def __init__(self, hz: float) -> None:
        if hz <= 0.0:
            raise ValueError(f"Control rate must be positive, got {hz} Hz.")
        self.period = 1.0 / hz
        self.next_deadline = time.perf_counter() + self.period

    def sleep(self) -> None:
        remaining = self.next_deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        now = time.perf_counter()
        self.next_deadline += self.period
        if now - self.next_deadline > self.period:
            self.next_deadline = now + self.period


def _camera_sensor_names(scene_cfg) -> tuple[str, ...]:
    if not is_dataclass(scene_cfg):
        return ()
    return tuple(
        field.name for field in fields(scene_cfg) if isinstance(getattr(scene_cfg, field.name, None), CameraCfg)
    )


def _set_camera_sensors_enabled(env_cfg, enabled: bool) -> tuple[str, ...]:
    set_camera_set = getattr(env_cfg, "set_camera_set", None)
    if set_camera_set is not None:
        set_camera_set("all" if enabled else "none")
    camera_names = _camera_sensor_names(env_cfg.scene)
    if enabled:
        return camera_names

    if set_camera_set is None:
        for camera_name in camera_names:
            setattr(env_cfg.scene, camera_name, None)
    if hasattr(env_cfg.observations, "camera_obs"):
        env_cfg.observations.camera_obs = None
    return ()


def _control_hz(env_cfg: ManagerBasedEnvCfg) -> float:
    explicit_hz = getattr(env_cfg, "control_hz", None)
    if explicit_hz is not None:
        return float(explicit_hz)
    control_period = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    if control_period <= 0.0:
        raise ValueError(f"Invalid manager task control period: {control_period} seconds.")
    return 1.0 / control_period


def _camera_publish_hz(env_cfg: ManagerBasedEnvCfg, control_hz: float) -> float:
    explicit_hz = getattr(env_cfg, "camera_hz", None)
    if explicit_hz is not None:
        return float(explicit_hz)
    render_period = float(env_cfg.sim.dt) * int(env_cfg.sim.render_interval)
    render_hz = 1.0 / render_period if render_period > 0.0 else control_hz
    return min(15.0, control_hz, render_hz)


def _configure_environment():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if not isinstance(env_cfg, ManagerBasedEnvCfg):
        raise TypeError(f"Task {args_cli.task} is not a manager-based environment.")

    camera_names = _set_camera_sensors_enabled(env_cfg, args_cli.enable_cameras)
    if args_cli.enable_cameras and not camera_names:
        raise ValueError(f"Task {args_cli.task} does not expose configurable camera sensors.")

    if args_cli.camera_view == "operator":
        enable_operator_cameras = getattr(env_cfg, "enable_operator_preview_cameras", None)
        if enable_operator_cameras is not None:
            enable_operator_cameras()
        if not getattr(env_cfg, "operator_camera_rows", None):
            raise ValueError(f"Task {args_cli.task} does not support --camera_view operator.")
    return env_cfg


def _make_bridge(env, camera_hz: float):
    if args_cli.bridge == "none":
        return None
    if args_cli.bridge == "ffw_sg2":
        from cyclo_lab.runtime.bridges.sg2 import FFWSG2TopicBridge

        return FFWSG2TopicBridge(
            env,
            camera_publish_hz=camera_hz,
            publish_odometry_tf=True,
            subscribe_reset=True,
        )
    if args_cli.bridge == "ffw_sh5":
        from cyclo_lab.runtime.bridges.sh5 import FFWSH5TopicBridge

        return FFWSH5TopicBridge(
            env,
            disable_head=args_cli.disable_head,
            disable_lift=args_cli.disable_lift,
            disable_cmd_vel=args_cli.disable_cmd_vel,
            subscribe_reset=True,
            camera_publish_hz=camera_hz,
        )
    raise ValueError(f"Unsupported bridge: {args_cli.bridge}")


def _make_operator_view(env_cfg, env):
    if args_cli.camera_view != "operator":
        return None
    from cyclo_lab.runtime.viewers import CameraDashboard

    rows = getattr(env_cfg, "operator_camera_rows")
    camera_names = {camera_name for row in rows for camera_name, _label in row}
    missing_cameras = sorted(camera_names.difference(env.scene.sensors))
    if missing_cameras:
        raise ValueError(f"Operator dashboard sensors are unavailable: {missing_cameras}")

    dashboard = CameraDashboard(
        env,
        rows=rows,
        panel_rotations=dict(getattr(env_cfg, "operator_camera_rotations", ())),
        window_title=getattr(env_cfg, "operator_camera_title", "Camera Dashboard"),
        window_size=getattr(env_cfg, "operator_camera_window_size", 1800),
        window_position=(20, 20),
    )
    print(f"[Camera Preview] operator dashboard: {dashboard.width}x{dashboard.height}")
    return dashboard


def main() -> None:
    env = None
    bridge = None
    operator_view = None
    keyboard_listener = None
    shutdown_requested = threading.Event()
    keyboard_start_requested = threading.Event()
    keyboard_reset_requested = threading.Event()
    previous_signal_handlers = {}

    def _request_shutdown(_signum, _frame) -> None:
        shutdown_requested.set()

    def _on_key_press(key) -> None:
        key_char = getattr(key, "char", None)
        if key_char is None:
            return
        key_char = key_char.lower()
        if key_char == "b":
            keyboard_start_requested.set()
        elif key_char == "r":
            keyboard_reset_requested.set()

    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[shutdown_signal] = signal.signal(shutdown_signal, _request_shutdown)

    try:
        env_cfg = _configure_environment()
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        control_hz = _control_hz(env_cfg)
        camera_hz = _camera_publish_hz(env_cfg, control_hz) if args_cli.enable_cameras else 0.0
        bridge = _make_bridge(env, camera_hz)
        zero_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
            dtype=torch.float32,
        )
        operator_view = _make_operator_view(env_cfg, env)
        try:
            from pynput.keyboard import Listener

            keyboard_listener = Listener(on_press=_on_key_press)
            keyboard_listener.start()
        except ImportError as exc:
            print(f"[Control] Keyboard listener unavailable: {exc}")

        requires_activation = bool(getattr(bridge, "requires_activation", True)) if bridge else False
        control_enabled = bridge is not None and not requires_activation
        if bridge is None:
            print("[Control] Zero-action mode; press R to reset the environment.")
        elif requires_activation:
            print("[Control] Press B to enable robot actions; press R to reset and stop actions.")
        else:
            print("[Control] Robot actions are active; press R to reset the environment.")

        rate_limiter = RateLimiter(control_hz)
        report_interval = max(0, int(args_cli.report_interval))
        report_start = time.perf_counter()
        start_time = report_start
        report_steps = 0
        total_steps = 0

        with torch.inference_mode():
            while simulation_app.is_running() and not shutdown_requested.is_set():
                if args_cli.max_runtime > 0.0 and time.perf_counter() - start_time >= args_cli.max_runtime:
                    break

                start_requested = keyboard_start_requested.is_set()
                if start_requested:
                    keyboard_start_requested.clear()
                keyboard_reset = keyboard_reset_requested.is_set()
                if keyboard_reset:
                    keyboard_reset_requested.clear()
                topic_reset = bridge.consume_reset_request() if bridge is not None else False
                if keyboard_reset or topic_reset:
                    control_enabled = bridge is not None and not requires_activation
                    keyboard_start_requested.clear()
                    reset_source = "R key" if keyboard_reset else "/simulation/reset"
                    print(f"[Control] Reset requested by {reset_source}.")
                    env.reset()
                    if bridge is not None:
                        bridge.reset()
                elif bridge is not None and requires_activation and start_requested and not control_enabled:
                    control_enabled = True
                    print("[Control] Robot actions enabled.")

                if bridge is None:
                    action = zero_action
                elif control_enabled:
                    action = bridge.get_action()
                else:
                    action = bridge.get_hold_action()
                env.step(action)
                if bridge is not None:
                    bridge.publish_observations()
                if operator_view is not None:
                    operator_view.update()

                total_steps += 1
                report_steps += 1
                if report_interval and report_steps >= report_interval:
                    now = time.perf_counter()
                    elapsed = max(now - report_start, 1e-9)
                    print(f"[RATE] {report_steps / elapsed:.2f} Hz")
                    report_start = now
                    report_steps = 0
                if args_cli.num_steps > 0 and total_steps >= args_cli.num_steps:
                    break
                rate_limiter.sleep()
    except KeyboardInterrupt:
        print("\n[INFO] Bringup interrupted.")
    finally:
        if keyboard_listener is not None:
            keyboard_listener.stop()
            keyboard_listener.join(timeout=1.0)
        if operator_view is not None:
            operator_view.close()
        if bridge is not None:
            bridge.close()
        if env is not None:
            env.close()
        for shutdown_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(shutdown_signal, previous_handler)


if __name__ == "__main__":
    main()
    simulation_app.close()
