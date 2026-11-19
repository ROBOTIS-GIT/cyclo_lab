# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Train a tracker, inferring the algorithm from a recognized task family."""

import sys
from mimic import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
