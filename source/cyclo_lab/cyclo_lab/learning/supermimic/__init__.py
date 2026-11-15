# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab bridge for the vendored SuperMimic algorithm (no mjlab runtime)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "third_party/supermimic/src"))
