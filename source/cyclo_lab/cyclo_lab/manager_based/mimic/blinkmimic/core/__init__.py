"""Use BlinkMimic's torch modules with Isaac motion and kinematics adapters."""

from ... import REPO_ROOT

_core = REPO_ROOT / "third_party/blinkmimic/src/blinkmimic"
if not (_core / "networks.py").is_file():
    raise ImportError("Initialize the third_party/blinkmimic submodule")
__path__.append(str(_core))
