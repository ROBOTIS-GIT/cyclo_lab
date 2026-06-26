# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
from pathlib import Path
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def validate_task_exists(task_name: str | None):
    """Fail before launching Isaac Sim if the requested task is not known."""
    if task_name is None:
        raise SystemExit("[ERROR] --task must be specified before launching Isaac Sim.")

    import gymnasium as gym

    task_id = task_name.split(":")[-1]
    known_task_ids = set(gym.registry.keys()) | _registered_task_ids_from_sources()
    if task_name in known_task_ids or task_id in known_task_ids:
        return

    close_matches = difflib.get_close_matches(task_id, sorted(known_task_ids), n=5)
    suggestion = ""
    if close_matches:
        suggestion = "\nClosest known tasks:\n" + "\n".join(f"  - {match}" for match in close_matches)
    raise SystemExit(f"[ERROR] Unknown task before launching Isaac Sim: {task_name}{suggestion}")


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if rslrl_cfg.experiment_name == "":
        rslrl_cfg.experiment_name = _default_experiment_name(task_name)
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    if agent_cfg.experiment_name == "" and hasattr(args_cli, "task"):
        agent_cfg.experiment_name = _default_experiment_name(args_cli.task)

    return agent_cfg


def _default_experiment_name(task_name: str) -> str:
    """Generate the default experiment name from a task name."""
    experiment_name = task_name.lower().replace("-", "_")
    experiment_name = _strip_play_suffix(experiment_name)
    experiment_name = _strip_gym_version_suffix(experiment_name)
    return experiment_name


def _strip_play_suffix(experiment_name: str) -> str:
    """Use the training task name for play task logs."""
    play_suffix_index = experiment_name.rfind("_play")
    if play_suffix_index == -1:
        return experiment_name

    suffix = experiment_name[play_suffix_index + len("_play") :]
    if suffix == "" or suffix.startswith("_v"):
        return experiment_name[:play_suffix_index] + suffix
    return experiment_name


def _strip_gym_version_suffix(experiment_name: str) -> str:
    """Drop trailing Gym version suffixes such as _v0."""
    version_suffix_index = experiment_name.rfind("_v")
    if version_suffix_index != -1 and experiment_name[version_suffix_index + 2 :].isdigit():
        return experiment_name[:version_suffix_index]
    return experiment_name


def _registered_task_ids_from_sources() -> set[str]:
    """Collect known task IDs without importing task packages or launching Isaac Sim."""
    roots = []
    for package_name in ("isaaclab_tasks", "cyclo_lab"):
        spec = importlib.util.find_spec(package_name)
        if spec is not None and spec.submodule_search_locations is not None:
            roots.extend(Path(path) for path in spec.submodule_search_locations)

    repo_root = Path(__file__).resolve().parents[3]
    roots.extend(
        [
            repo_root / "source" / "cyclo_lab" / "cyclo_lab",
            repo_root.parent / "IsaacLab" / "source" / "isaaclab_tasks" / "isaaclab_tasks",
        ]
    )

    task_ids = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("__init__.py"):
            task_ids.update(_extract_registered_task_ids(path))
    return task_ids


def _extract_registered_task_ids(path: Path) -> set[str]:
    """Extract literal Gym task IDs from a Python source file."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()

    task_ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "register":
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            task_ids.add(node.args[0].value)
        for keyword in node.keywords:
            if (
                keyword.arg == "id"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                task_ids.add(keyword.value.value)
    return task_ids
