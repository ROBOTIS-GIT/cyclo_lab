# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch a registered motion tracking task to its learner."""

import argparse
from pathlib import Path
import subprocess
import sys


def select_algorithm(algorithm, task):
    prefixes = {
        "blinkmimic": ("Cyclo-BlinkMimic-",),
        "supermimic": ("Cyclo-SuperMimic-",),
        "beyondmimic": ("Cyclo-Mimic-", "Cyclo-BeyondMimic-"),
    }
    family = next((family for family, names in prefixes.items() if task and task.startswith(names)), None)
    if algorithm and family and algorithm != family:
        raise ValueError(f"--algorithm {algorithm} conflicts with task {task}")
    return algorithm or family or "beyondmimic"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("action", choices=("train", "play", "export"))
    parser.add_argument("--algorithm", "--method", choices=("blinkmimic", "supermimic", "beyondmimic"))
    parser.add_argument("--task", help="Registered Cyclo task; may also follow the action as a positional ID")
    if not argv or all(arg in ("--help", "-h", "train", "play", "export") for arg in argv):
        parser.print_help()
        return 0
    args, remaining = parser.parse_known_args(argv)
    task = args.task
    if remaining and remaining[0].startswith("Cyclo-"):
        if task:
            parser.error("Pass the task ID only once")
        task = remaining.pop(0)
    try:
        algorithm = select_algorithm(args.algorithm, task)
    except ValueError as error:
        parser.error(str(error))

    if algorithm in ("blinkmimic", "supermimic"):
        if not task:
            parser.error(f"Choose a registered {algorithm} task")
        from motion_tracker import main as track

        return track(args.action, task, remaining, family=algorithm)
    if args.action == "export":
        parser.error("Export requires a registered motion-library task")

    motion = argparse.ArgumentParser(add_help=False)
    motion.add_argument("--motion-file", type=Path)
    options, remaining = motion.parse_known_args(remaining)
    if options.motion_file:
        import json

        remaining += [
            "env.commands.reference_trajectory.trajectory_file="
            + json.dumps(str(options.motion_file.resolve()))
        ]
    task = task or "Cyclo-BeyondMimic-K1-Rev1-Dance1"
    script = Path(__file__).resolve().parent / "rsl_rl" / f"{args.action}.py"
    remaining = ["--task", task, *remaining]
    return subprocess.call([sys.executable, str(script), *remaining])


if __name__ == "__main__":
    raise SystemExit(main())
