# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train Cyclo tasks with Isaac Lab's official sb3 entrypoint."""

import sys

import cyclo_lab  # noqa: F401
from cyclo_lab.utils.rl_cli import prepare_launcher_args
from isaaclab_rl.entrypoints import run_train_cli


if __name__ == "__main__":
    raise SystemExit(run_train_cli(["--rl_library", "sb3", *prepare_launcher_args(sys.argv[1:])]))
