# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""Train a tracker, inferring the algorithm from a recognized task family."""

import sys
from cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
