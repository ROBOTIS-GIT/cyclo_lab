"""Keep existing launcher flags when dispatching to the official RL entrypoints."""

import argparse
import os


def prepare_launcher_args(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--headless", action="store_true")
    # Lab 3.0 enables rendering from the task's cameras and recording configuration.
    parser.add_argument("--enable_cameras", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if args.headless:
        os.environ["HEADLESS"] = "1"
    return remaining
