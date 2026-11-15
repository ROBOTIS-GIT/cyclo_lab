# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Select an Isaac K1 tracker; preserve direct PPO and historical --method usage."""

import argparse
from pathlib import Path
import subprocess
import sys


def select_algorithm(algorithm, task):
    family = None
    if task:
        if task.startswith("Cyclo-SuperMimic-"):
            family = "supermimic"
        elif task.startswith(("Cyclo-BeyondMimic-", "Cyclo-Mimic-")):
            family = "beyondmimic"
    if algorithm and family and algorithm != family:
        raise ValueError(f"--algorithm {algorithm} conflicts with task {task}")
    return algorithm or family or "beyondmimic"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--algorithm", choices=("beyondmimic", "supermimic"))
    group.add_argument(
        "--method",
        dest="algorithm",
        choices=("beyondmimic", "supermimic"),
        help="Historical alias for --algorithm",
    )
    parser.add_argument("action", choices=("train", "play"))
    parser.add_argument("--task")
    parser.add_argument("--motion-file", type=Path)
    if not argv or (
        "--help" in argv and all(x in ("--help", "train", "play") for x in argv)
    ):
        parser.print_help()
        return 0
    args, remaining = parser.parse_known_args(argv)
    try:
        algorithm = select_algorithm(args.algorithm, args.task)
    except ValueError as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parent
    if args.task:
        remaining = ["--task", args.task, *remaining]
    if args.motion_file is not None:
        sys.path.insert(0, str(root.parents[1] / "source/cyclo_lab"))
        from cyclo_lab.learning.supermimic.motion import validate_motion

        validate_motion(args.motion_file)
        if algorithm == "beyondmimic":
            import json

            remaining += [
                "env.commands.reference_trajectory.trajectory_file="
                + json.dumps(str(args.motion_file.resolve()))
            ]
        else:
            remaining += ["--motion-file", str(args.motion_file.resolve())]
    if algorithm == "beyondmimic":
        script = root / "rsl_rl" / f"{args.action}.py"
        if not args.task:
            remaining = ["--task", "Cyclo-BeyondMimic-K1-Rev1-Dance1", *remaining]
    else:
        script = root / "supermimic" / "train.py"
        if args.action == "play":
            remaining = ["--play", *remaining]
    return subprocess.call([sys.executable, str(script), *remaining])


if __name__ == "__main__":
    raise SystemExit(main())
