# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL training backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from datetime import datetime

import torch.distributed as dist

from isaaclab.app import add_launcher_args, launch_simulation, report_activity
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection
from isaaclab.utils.version import standalone_importers_available

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.wandb import announce_new_run, resolve_wandb_entity
from isaaclab_rl.entrypoints.common import (
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    scoped_torch_backend_flags,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    validate_distributed_device,
    wrap_sensor_capture,
    write_run_manifest,
)
import cli_args
import cyclo_lab  # noqa: F401
from cyclo_lab.utils.export_git_state import export_git_state
from cyclo_lab.utils.export_sim2real_cfg import export_sim2real_cfg


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    add_common_train_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    parser.add_argument(
        "--resume_optimizer_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore the policy optimizer; disable to keep a fresh optimizer and configured learning rate.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without a Kit window.")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera rendering.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    args_cli.require_kit = not standalone_importers_available()
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def run(argv: list[str]) -> None:
    """Train an RSL-RL agent while restoring the caller's Torch backend settings."""
    args_cli = _parse_args(argv)
    with scoped_torch_backend_flags(
        cuda_matmul_allow_tf32=True,
        cudnn_allow_tf32=True,
        cudnn_deterministic=False,
        cudnn_benchmark=False,
    ):
        _run(args_cli)


def _run(args_cli: argparse.Namespace) -> None:
    """Execute RSL-RL training with parsed arguments."""
    installed_version = check_rsl_rl_version()

    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="train")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
            apply_env_overrides(args_cli, env_cfg)
            validate_distributed_device(args_cli)
            if args_cli.max_iterations is not None:
                agent_cfg.max_iterations = args_cli.max_iterations
            rank = int(os.getenv("RANK", "0")) if args_cli.distributed else None
            if rank is not None:
                agent_cfg.device = env_cfg.sim.device
                agent_cfg.seed += rank
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            run_name = args_cli.run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"Exact experiment name requested from command line: {run_name}")
            if agent_cfg.run_name:
                run_name += f"_{agent_cfg.run_name}"
            log_dir = os.path.join(log_root_path, run_name)
            # rank 0 writes its settings and videos where a single-GPU run does; other ranks use rank_<rank>/
            rank_dir = log_dir if rank in (None, 0) else os.path.join(log_dir, f"rank_{rank}")
            resume_path = cli_args.resolve_checkpoint(args_cli, agent_cfg, log_root_path)
            # All ranks share the run folder, so its manifest is written once.
            if rank in (None, 0):
                write_run_manifest(log_dir, library="rsl_rl", task=args_cli.task, metadata={"agent": args_cli.agent})

            env_cfg.log_dir = rank_dir
            apply_video_recording(env_cfg, rank_dir, args_cli)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            cleanup.callback(lambda: close_env(env))
            env = wrap_sensor_capture(env, rank_dir, args_cli)

            screen.stage("Preparing agent")
            start_time = time.time()
            report_activity("Wrapping environment")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            report_activity(None)
            report_activity("Building policy")
            runner = create_rsl_rl_runner(env, agent_cfg, log_dir=log_dir)
            report_activity(None)

            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)

            runner.add_git_repo_to_log(__file__)
            if resume_path is not None:
                print(f"[INFO]: Loading model checkpoint from: {resume_path}")
                # map to this process's device; the checkpoint's tensors otherwise land on the GPU that saved them
                runner.load(
                    resume_path,
                    load_cfg=cli_args.checkpoint_load_cfg(
                        args_cli.resume_optimizer_state, runner_class=agent_cfg.class_name
                    ),
                    map_location=agent_cfg.device,
                )
            dump_train_configs(rank_dir, env_cfg, agent_cfg)
            export_git_state(rank_dir, __file__)
            export_sim2real_cfg(env.unwrapped, rank_dir)

            if agent_cfg.logger == "wandb":
                announce_new_run(agent_cfg.wandb_project, resolve_wandb_entity())

            screen.close()
            with contextlib.suppress(KeyboardInterrupt):
                runner.learn(
                    num_learning_iterations=agent_cfg.max_iterations,
                    init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
                )
                print(f"Training time: {round(time.time() - start_time, 2)} seconds")
                # the RL library creates the process group but never destroys it, which torch warns about at exit
                if dist.is_initialized():
                    dist.destroy_process_group()


if __name__ == "__main__":
    import sys

    run(sys.argv[1:])
