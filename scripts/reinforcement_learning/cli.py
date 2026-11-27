# Copyright 2026 ROBOTIS CO., LTD.
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
# Author: Kiwoong Park

"""Dispatch a registered motion tracking task to its learner."""

import argparse
from pathlib import Path
import subprocess
import sys


def select_algorithm(algorithm, task):
    prefixes = {
        "velocity": ("Cyclo-Velocity-",),
        "blinkmimic": ("Cyclo-BlinkMimic-",),
        "supermimic": ("Cyclo-SuperMimic-",),
        "beyondmimic": ("Cyclo-BeyondMimic-",),
    }
    family = next((family for family, names in prefixes.items() if task and task.startswith(names)), None)
    if algorithm and family and algorithm != family:
        raise ValueError(f"--algorithm {algorithm} conflicts with task {task}")
    return algorithm or family or "beyondmimic"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("action", choices=("train", "play"))
    parser.add_argument("--algorithm", "--method", choices=("blinkmimic", "supermimic", "beyondmimic"))
    parser.add_argument("--task", help="Registered Cyclo task; may also follow the action as a positional ID")
    if not argv or all(arg in ("--help", "-h", "train", "play") for arg in argv):
        parser.print_help()
        return 0
    args, remaining = parser.parse_known_args(argv)
    task = args.task
    if remaining and remaining[0].startswith("Cyclo-"):
        if task:
            parser.error("Pass the task ID only once")
        task = remaining.pop(0)
    if not task:
        parser.error("Specify a task ID as TASK or --task TASK")
    try:
        algorithm = select_algorithm(args.algorithm, task)
    except ValueError as error:
        parser.error(str(error))

    if algorithm in ("blinkmimic", "supermimic"):
        from cyclo_lab.manager_based.mimic._shared.motion_tracker import main as track

        if args.action == "train":
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from scripts.tools.permission_hint import training_permission_hint

            with training_permission_hint():
                return track(args.action, task, remaining, family=algorithm)
        return track(args.action, task, remaining, family=algorithm)

    motion = argparse.ArgumentParser(add_help=False)
    motion.add_argument("--motion-file", type=Path)
    options, remaining = motion.parse_known_args(remaining)
    if options.motion_file:
        import json

        remaining += [
            "env.commands.reference_trajectory.trajectory_file="
            + json.dumps(str(options.motion_file.resolve()))
        ]
    script = Path(__file__).resolve().parent / "rsl_rl" / f"{args.action}.py"
    remaining = ["--task", task, *remaining]
    return subprocess.call([sys.executable, str(script), *remaining])


if __name__ == "__main__":
    raise SystemExit(main())
