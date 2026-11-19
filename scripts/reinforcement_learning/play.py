# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Play a tracker locally; native Lab uses --checkpoint, not MJLab bundles."""

import sys
from mimic import main

if __name__ == "__main__":
    raise SystemExit(main(["play", *sys.argv[1:]]))
