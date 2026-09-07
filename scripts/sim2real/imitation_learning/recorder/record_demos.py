# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a cyclo_lab teleoperation with cyclo_lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import contextlib
from collections import defaultdict

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="cyclo_lab teleoperation for cyclo_lab environments.")
parser.add_argument(
    "--robot_type",
    type=str,
    default="keyboard",
    choices=["OMY", "FFW_SG2"],
    help="Type of robot to use for teleoperation.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")

# recorder_parameter
parser.add_argument(
    "--step_hz",
    type=float,
    default=None,
    help="Environment stepping rate in Hz. Uses the task recording rate, or 60 Hz when unspecified.",
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument("--flush_steps", type=int, default=30, help="Streaming HDF5 flush interval in environment steps.")
parser.add_argument(
    "--publish_camera_topics",
    action="store_true",
    help="Publish compressed camera topics while recording. HDF5 camera recording does not require this.",
)
extra_state_topic_group = parser.add_mutually_exclusive_group()
extra_state_topic_group.add_argument(
    "--publish_extra_state_topics",
    dest="publish_extra_state_topics",
    action="store_true",
    help="Publish odom/tf in addition to /joint_states while recording. Enabled by default.",
)
extra_state_topic_group.add_argument(
    "--no_publish_extra_state_topics",
    dest="publish_extra_state_topics",
    action="store_false",
    help="Disable odom/tf publishing while recording. /joint_states is always published.",
)
parser.set_defaults(publish_extra_state_topics=True)
parser.add_argument("--profile", action="store_true", help="Print timing statistics for the recording loop.")
parser.add_argument("--profile_interval", type=int, default=120, help="Loop iterations between profile reports.")
parser.add_argument(
    "--profile_cuda_sync",
    action="store_true",
    help="Synchronize CUDA around profiled sections for more accurate GPU timing. This adds overhead.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time

import cyclo_lab  # noqa: F401 - registers Cyclo Gym tasks
import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, RecorderTerm, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from recorder_manager.recorder_manager import StreamingRecorderManager


class RateLimiter:
    """Keep a fixed loop schedule while allowing short overruns to catch up."""

    def __init__(self, hz: float):
        if hz <= 0.0:
            raise ValueError(f"Rate must be positive, got {hz} Hz.")
        self.hz = hz
        self.sleep_duration = 1.0 / hz
        self.next_wakeup_time = time.perf_counter() + self.sleep_duration

    def sleep(self):
        """Sleep until the next deadline without discarding a one-cycle overrun."""
        sleep_time = self.next_wakeup_time - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)

        now = time.perf_counter()
        self.next_wakeup_time += self.sleep_duration
        if now - self.next_wakeup_time > self.sleep_duration:
            self.next_wakeup_time = now + self.sleep_duration


class LoopProfiler:
    """Small timing profiler for the demonstration recorder."""

    def __init__(self, enabled: bool, interval: int, cuda_sync: bool = False):
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.cuda_sync = bool(cuda_sync)
        self.loop_count = 0
        self._window_start = time.perf_counter()
        self._stats = defaultdict(lambda: {"count": 0, "total": 0.0, "max": 0.0})
        self._installed_hooks = []

    def _sync(self):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            stat = self._stats[name]
            stat["count"] += 1
            stat["total"] += elapsed
            stat["max"] = max(stat["max"], elapsed)

    def tick(self):
        if not self.enabled:
            return
        self.loop_count += 1
        if self.loop_count % self.interval != 0:
            return

        now = time.perf_counter()
        window_elapsed = max(now - self._window_start, 1e-9)
        measured_hz = self.interval / window_elapsed
        loop_total = self._stats.get("loop_total", {}).get("total", 0.0)

        print(f"\n[PROFILE] last {self.interval} loops: wall_hz={measured_hz:.2f}")
        rows = sorted(
            self._stats.items(),
            key=lambda item: item[1]["total"],
            reverse=True,
        )
        for name, stat in rows:
            if stat["count"] == 0:
                continue
            percent = (stat["total"] / loop_total * 100.0) if loop_total > 0.0 and name != "loop_total" else 0.0
            print(
                f"[PROFILE] {name:36s} "
                f"mean={stat['total'] / stat['count'] * 1000.0:8.2f}ms "
                f"max={stat['max'] * 1000.0:8.2f}ms "
                f"total={stat['total']:7.3f}s "
                f"n={stat['count']:5d} "
                f"{percent:5.1f}%"
            )
        self._stats.clear()
        self._window_start = now

    def register_hook(self, obj, method_name: str, original) -> None:
        """Track an instance-method hook so object lifetimes are restored before shutdown."""
        self._installed_hooks.append((obj, method_name, original))

    def remove_hooks(self) -> None:
        """Restore methods wrapped for profiling and release bound-method references."""
        for obj, method_name, original in reversed(self._installed_hooks):
            try:
                setattr(obj, method_name, original)
            except Exception as exc:
                print(f"[PROFILE] Failed to remove hook for {method_name}: {exc}")
        self._installed_hooks.clear()


def install_profile_hook(obj, method_name: str, label: str, profiler: LoopProfiler):
    """Wrap an instance method with a profiler timer."""
    if not profiler.enabled or not hasattr(obj, method_name):
        return
    original = getattr(obj, method_name)
    if getattr(original, "_cyclo_profile_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        with profiler.time(label):
            return original(*args, **kwargs)

    wrapped._cyclo_profile_wrapped = True
    try:
        setattr(obj, method_name, wrapped)
        profiler.register_hook(obj, method_name, original)
    except Exception as exc:
        print(f"[PROFILE] Failed to install hook for {label}: {exc}")


def install_env_profile_hooks(env, profiler: LoopProfiler):
    """Install coarse Isaac Lab manager timing hooks."""
    install_profile_hook(env.observation_manager, "compute", "isaaclab_observation_compute", profiler)
    install_profile_hook(env.termination_manager, "compute", "isaaclab_termination_compute", profiler)
    install_profile_hook(env.reward_manager, "compute", "isaaclab_reward_compute", profiler)
    install_profile_hook(env.command_manager, "compute", "isaaclab_command_compute", profiler)
    install_profile_hook(env.action_manager, "process_action", "isaaclab_action_process", profiler)
    install_profile_hook(env.action_manager, "apply_action", "isaaclab_action_apply", profiler)
    install_profile_hook(env.scene, "write_data_to_sim", "isaaclab_scene_write", profiler)
    install_profile_hook(env.scene, "update", "isaaclab_scene_update", profiler)
    install_profile_hook(env.sim, "step", "isaaclab_sim_step", profiler)
    install_profile_hook(env.sim, "render", "isaaclab_sim_render", profiler)


def install_recorder_term_profile_hooks(recorder_manager, profiler: LoopProfiler):
    """Install timing hooks for each active recorder term and phase."""
    for term_name, term in getattr(recorder_manager, "_terms", {}).items():
        for phase in ("pre_step", "post_step", "post_physics_decimation_step"):
            method_name = f"record_{phase}"
            if getattr(type(term), method_name) is getattr(RecorderTerm, method_name):
                continue
            install_profile_hook(
                term,
                method_name,
                f"recorder_term_{term_name}_{phase}",
                profiler,
            )


def release_camera_sensors_before_close(env) -> None:
    """Detach camera annotators before Isaac Sim starts clearing simulation callbacks."""
    for sensor in getattr(env.scene, "sensors", {}).values():
        registry = getattr(sensor, "_rep_registry", None)
        if registry is None:
            continue
        try:
            sensor.__del__()
        except Exception as exc:
            print(f"[WARN] Failed to detach camera sensor during shutdown: {exc}")
        finally:
            registry.clear()


def main():  # noqa: C901
    """Running cyclo_lab teleoperation with cyclo_lab manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.init_action_cfg("record")
    env_cfg.seed = args_cli.seed
    task_name = args_cli.task

    # modify configuration
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    if not hasattr(env_cfg.terminations, "success"):
        setattr(env_cfg.terminations, "success", None)
    env_cfg.terminations.success = TerminationTermCfg(
        func=lambda env: torch.zeros(1, dtype=torch.bool, device=env.device)
    )
    # Do not save while stepping; only save explicitly on success (key 'N')
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    del env.recorder_manager
    # Ensure dataset file handler is created, but keep stepping in no-save mode
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
    env.recorder_manager.flush_steps = args_cli.flush_steps
    env.recorder_manager.compression = "lzf"
    env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env.recorder_manager.recording_enabled = False
    profiler = LoopProfiler(
        enabled=args_cli.profile,
        interval=args_cli.profile_interval,
        cuda_sync=args_cli.profile_cuda_sync,
    )
    env.recorder_manager.profiler = profiler
    install_env_profile_hooks(env, profiler)
    install_recorder_term_profile_hooks(env.recorder_manager, profiler)

    # create controller
    if args_cli.robot_type == "OMY":
        from cyclo_lab.runtime.sdk.omy import OMYSdk

        teleop_interface = OMYSdk(
            env,
            mode="record",
            camera_publish_hz=None if args_cli.publish_camera_topics else 0.0,
        )
    elif args_cli.robot_type == "FFW_SG2":
        from cyclo_lab.runtime.sdk.ffw_sg2 import FFWSG2Sdk

        teleop_interface = FFWSG2Sdk(
            env,
            mode="record",
            camera_publish_hz=None if args_cli.publish_camera_topics else 0.0,
            publish_odometry_tf=args_cli.publish_extra_state_topics,
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.robot_type}'. Supported: 'OMY', 'FFW_SG2'.")

    start_record_state = False

    def clear_episode_cache(context: str):
        try:
            env.recorder_manager._clear_episode_cache()
        except Exception as e:
            print(f"Warning: Failed to clear {context} episode cache: {e}")
            print(f"Exception details: {type(e).__name__}: {str(e)}")

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        if not start_record_state:
            print("[Control] Save ignored because recording has not started.")
            return
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)

    target_step_hz = args_cli.step_hz
    if target_step_hz is None:
        target_step_hz = getattr(env_cfg, "recording_control_hz", 60.0)
    rate_limiter = RateLimiter(target_step_hz)

    # reset environment
    env.reset()
    teleop_interface.reset()

    current_recorded_demo_count = 0

    should_start_recording_instance = False

    def start_recording_instance():
        nonlocal start_record_state
        if start_record_state:
            return
        clear_episode_cache("pre-recording")
        env.recorder_manager.recording_enabled = True
        env.recorder_manager.record_post_reset(torch.arange(env.num_envs, device=env.device))
        start_record_state = True
        print("Start Recording!!!")

    def request_start_recording_instance():
        nonlocal should_start_recording_instance
        should_start_recording_instance = True

    teleop_interface.add_callback("B", request_start_recording_instance)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with profiler.time("loop_total"):
            with torch.inference_mode():
                with profiler.time("publish_observations"):
                    teleop_interface.publish_observations()
                with profiler.time("get_action"):
                    actions = teleop_interface.get_action()
                if should_start_recording_instance:
                    should_start_recording_instance = False
                    with profiler.time("start_recording"):
                        start_recording_instance()
                if should_reset_task_success:
                    with profiler.time("save_success_export"):
                        print("Task Success!!!")
                        should_reset_task_success = False
                        env.termination_manager.set_term_cfg(
                            "success",
                            TerminationTermCfg(
                                func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                            ),
                        )
                        env.termination_manager.compute()
                        # Mark current buffered episode(s) as successful and export before resetting
                        try:
                            for env_id, ep in getattr(env.recorder_manager, "_episodes", {}).items():
                                if ep is not None and not ep.is_empty():
                                    ep.success = True
                        except Exception as e:
                            print(f"Warning: Failed to mark episodes as successful: {e}")
                            print(f"Exception details: {type(e).__name__}: {str(e)}")

                        env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                        env.recorder_manager.export_episodes(from_step=False)
                        env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
                        # Update and report successful demo count immediately after export
                        if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                            current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                            print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                if should_reset_recording_instance:
                    with profiler.time("reset_recording"):
                        # Clear any buffered episode so failed episodes (key 'R') aren't saved
                        env.recorder_manager.recording_enabled = False
                        clear_episode_cache("recording")

                        env.reset()
                        should_reset_recording_instance = False
                        if start_record_state:
                            print("Stop Recording!!!")
                        start_record_state = False
                        env.termination_manager.set_term_cfg(
                            "success",
                            TerminationTermCfg(
                                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                            ),
                        )
                        # print out the current demo count if it has changed
                        print(
                            f"Resetting recording instance. Current recorded demo count: {current_recorded_demo_count}"
                        )
                        if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                            current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                            print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                        if (
                            args_cli.num_demos > 0
                            and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos
                        ):
                            print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                            break

                elif actions is None:
                    with profiler.time("env_render_no_action"):
                        env.render()
                # apply actions
                else:
                    if isinstance(actions, dict):
                        # Handle dictionary actions (like reset)
                        if "reset" in actions:
                            # This is a reset action, don't step the environment
                            with profiler.time("env_render_reset_action"):
                                env.render()
                            continue
                    else:
                        # Handle tensor actions
                        if actions.ndim == 1:
                            actions = actions.unsqueeze(0)
                        with profiler.time("env_step"):
                            env.step(actions)
                if rate_limiter:
                    with profiler.time("rate_sleep"):
                        rate_limiter.sleep()
        profiler.tick()

    # close the simulator
    teleop_interface.shutdown()
    profiler.remove_hooks()
    release_camera_sensors_before_close(env)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
