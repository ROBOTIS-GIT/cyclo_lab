# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""Isaac Lab bridge for the shared SuperMimic core submodule (no mjlab runtime)."""

import sys

from .. import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "third_party/supermimic/src"))

from supermimic.utils.math import quat_apply, quat_mul  # noqa: E402,F401
