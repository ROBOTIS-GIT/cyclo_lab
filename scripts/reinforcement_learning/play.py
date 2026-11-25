# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""Play a tracker locally; native Lab uses --checkpoint, not MJLab bundles."""

import sys
from cli import main

if __name__ == "__main__":
    raise SystemExit(main(["play", *sys.argv[1:]]))
