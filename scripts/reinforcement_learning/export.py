# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Export the trained native tracker to ONNX."""

import sys
from mimic import main

if __name__ == "__main__":
    raise SystemExit(main(["export", *sys.argv[1:]]))
