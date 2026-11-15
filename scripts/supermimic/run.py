# SPDX-License-Identifier: Apache-2.0
"""Launch the official SuperMimic world-model training or policy playback on K1."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party/supermimic/src'))
from k1 import TASK, register  # noqa: E402

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'play'):
        raise SystemExit('Usage: run.py {train|play} <upstream arguments; --help supported>')
    mode = sys.argv.pop(1)
    register()
    sys.argv.insert(1, TASK)
    importlib.import_module(f'supermimic.scripts.{mode}').main()
