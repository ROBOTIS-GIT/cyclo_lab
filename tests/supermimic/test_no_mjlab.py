"""Import contract: the native core must not load a second simulator stack."""

import subprocess
import sys
from pathlib import Path


def test_core_import_with_mjlab_blocked():
    source = Path(__file__).resolve().parents[2] / "source/cyclo_lab"
    code = f"""import sys
sys.path.insert(0, {str(source)!r})
class BlockMjlab:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mjlab' or fullname.startswith('mjlab.'):
            raise AssertionError('Unexpected mjlab runtime dependency: '+fullname)
sys.meta_path.insert(0, BlockMjlab())
from cyclo_lab.manager_based.mimic.supermimic.core.runner import IsaacSuperMimicRunner
assert not any(n == 'mjlab' or n.startswith('mjlab.') for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
