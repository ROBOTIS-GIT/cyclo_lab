# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Cyclo defaults around Isaac Lab's RSL-RL command-line helpers."""

import os
import re

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import add_rsl_rl_args, checkpoint_load_cfg, register_external_tasks
from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import update_rsl_rl_cfg as _update_rsl_rl_cfg
from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    normalize_task_name,
    resolve_checkpoint_selector,
    resolve_published_checkpoint,
)
from isaaclab_rl.utils.wandb import is_wandb_checkpoint, resolve_wandb_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path


def update_rsl_rl_cfg(agent_cfg, args_cli):
    agent_cfg = _update_rsl_rl_cfg(agent_cfg, args_cli)
    if not agent_cfg.experiment_name:
        agent_cfg.experiment_name = re.sub(r"_v\d+$", "", normalize_task_name(args_cli.task).lower().replace("-", "_"))
    return agent_cfg


def resolve_checkpoint(args_cli, agent_cfg, log_root_path, *, play=False, env_cfg=None):
    """Resolve explicit paths or official selectors; only play defaults to the latest file."""
    checkpoint = args_cli.checkpoint
    if checkpoint and is_wandb_checkpoint(checkpoint):
        return resolve_wandb_checkpoint(checkpoint)
    if checkpoint == "pretrained":
        if not play:
            raise ValueError("--checkpoint pretrained is only supported for play.")
        return resolve_published_checkpoint("rsl_rl", args_cli.task, env_cfg)
    if checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path, checkpoint, library="rsl_rl", task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model_.*\.pt", metadata={"agent": args_cli.agent},
        )
    if checkpoint and os.path.isdir(checkpoint):
        return get_checkpoint_path(os.path.dirname(os.path.abspath(checkpoint)), os.path.basename(os.path.normpath(checkpoint)), r"model_.*\.pt")
    if checkpoint:
        return retrieve_file_path(checkpoint)
    if play:
        return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if agent_cfg.algorithm.class_name == "Distillation":
        raise ValueError("Distillation training requires --checkpoint.")
    return None
