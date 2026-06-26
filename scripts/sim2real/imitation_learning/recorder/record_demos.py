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

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="cyclo_lab teleoperation for cyclo_lab environments.")
parser.add_argument("--robot_type", type=str, default="keyboard", choices=['OMY', 'FFW_SG2'], help="Type of robot to use for teleoperation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")

# recorder_parameter
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")

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
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg, DatasetExportMode

import cyclo_lab
import sys
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recorder_manager.recorder_manager import StreamingRecorderManager

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main():
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
    env_cfg.terminations.success = TerminationTermCfg(func=lambda env: torch.zeros(1, dtype=torch.bool, device=env.device))
    # Do not save while stepping; only save explicitly on success (key 'N')
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    del env.recorder_manager
    # Ensure dataset file handler is created, but keep stepping in no-save mode
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
    env.recorder_manager.flush_steps = 100
    env.recorder_manager.compression = 'lzf'
    env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    # create controller
    if args_cli.robot_type == "OMY":
        from dds_sdk.omy_sdk import OMYSdk
        teleop_interface = OMYSdk(env, mode='record')
    elif args_cli.robot_type == "FFW_SG2":
        from dds_sdk.ffw_sg2_sdk import FFWSG2Sdk
        teleop_interface = FFWSG2Sdk(env, mode='record')
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.robot_type}'. Supported: 'OMY', 'FFW_SG2'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    env.reset()
    teleop_interface.reset()

    current_recorded_demo_count = 0

    start_record_state = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            teleop_interface.publish_observations()
            actions = teleop_interface.get_action()
            if should_reset_task_success:
                print("Task Success!!!")
                should_reset_task_success = False
                env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)))
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
                # Clear any buffered episode so failed episodes (key 'R') aren't saved
                try:
                    env.recorder_manager._clear_episode_cache()
                except Exception as e:
                    print(f"Warning: Failed to clear episode cache: {e}")
                    print(f"Exception details: {type(e).__name__}: {str(e)}")

                env.reset()
                should_reset_recording_instance = False
                if start_record_state:
                    print("Stop Recording!!!")
                start_record_state = False
                env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)))
                # print out the current demo count if it has changed
                print(f"Resetting recording instance. Current recorded demo count: {current_recorded_demo_count}")
                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break

            elif actions is None:
                env.render()
            # apply actions
            else:
                if isinstance(actions, dict):
                    # Handle dictionary actions (like reset)
                    if "reset" in actions:
                        # This is a reset action, don't step the environment
                        env.render()
                        continue
                else:
                    # Handle tensor actions
                    if actions.ndim == 1:
                        actions = actions.unsqueeze(0)
                    if not start_record_state:
                        print("Start Recording!!!")
                        start_record_state = True
                    env.step(actions)
            if rate_limiter:
                rate_limiter.sleep(env)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
