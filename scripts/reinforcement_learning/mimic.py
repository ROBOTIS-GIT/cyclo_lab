# SPDX-License-Identifier: Apache-2.0
"""Select an Isaac Lab K1 tracker without changing the existing PPO entrypoints."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--method", required=True, choices=("beyondmimic", "supermimic")
    )
    parser.add_argument("action", choices=("train", "play"))
    parser.add_argument("--motion-file", type=Path)
    if len(sys.argv) == 1 or ("--help" in sys.argv and "--method" not in sys.argv):
        parser.print_help()
        return 0
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parent
    if args.motion_file is not None:
        sys.path.insert(0, str(root.parents[1] / "source/cyclo_lab"))
        from cyclo_lab.learning.supermimic.motion import validate_motion

        validate_motion(args.motion_file)
        if args.method == "beyondmimic":
            import json

            remaining += [
                "env.commands.reference_trajectory.trajectory_file="
                + json.dumps(str(args.motion_file.resolve()))
            ]
        else:
            remaining += ["--motion-file", str(args.motion_file.resolve())]
    if args.method == "beyondmimic":
        script = root / "rsl_rl" / f"{args.action}.py"
        if not any(x == "--task" or x.startswith("--task=") for x in remaining):
            remaining = ["--task", "Cyclo-BeyondMimic-K1-Rev1-Dance1", *remaining]
    else:
        script = root / "supermimic" / "train.py"
        if args.action == "play":
            remaining = ["--play", *remaining]
    return subprocess.call([sys.executable, str(script), *remaining])


if __name__ == "__main__":
    raise SystemExit(main())
