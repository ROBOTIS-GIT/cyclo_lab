# SPDX-License-Identifier: Apache-2.0
"""Fetch the exact official mjlab_research dependency, without changing existing checkouts."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MJLAB_REV = 'dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd'
DEST = ROOT / 'third_party/mjlab_research'

if __name__ == '__main__':
    if DEST.exists():
        current = subprocess.check_output(['git', '-C', str(DEST), 'rev-parse', 'HEAD'], text=True).strip()
        if current != MJLAB_REV:
            raise SystemExit(f'{DEST} already exists at {current}; expected {MJLAB_REV}. Left untouched.')
    else:
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/HybridRobotics/mjlab_research.git', str(DEST)], check=True)
        subprocess.run(['git', '-C', str(DEST), 'checkout', '--detach', MJLAB_REV], check=True)
    print('Dependency ready. Create an isolated Python 3.13 environment with:')
    print(f'uv sync --project {ROOT / "third_party/supermimic"} --python 3.13')
