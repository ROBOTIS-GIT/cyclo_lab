import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/cyclo_lab"))
import cyclo_lab.learning.supermimic  # noqa: F401,E402
