"""Common selectors preserve old commands and reject conflicting task families."""

import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[2] / "scripts/reinforcement_learning/mimic.py"
spec = importlib.util.spec_from_file_location("mimic_selector", path)
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


@pytest.mark.parametrize(
    "prefix, algorithm",
    [
        ("Cyclo-Mimic-", "beyondmimic"),
        ("Cyclo-BeyondMimic-", "beyondmimic"),
        ("Cyclo-SuperMimic-", "supermimic"),
    ],
)
def test_task_inference(prefix, algorithm):
    assert selector.select_algorithm(None, prefix + "K1-Rev1-Dance1") == algorithm
    other = "supermimic" if algorithm == "beyondmimic" else "beyondmimic"
    with pytest.raises(ValueError, match="conflicts"):
        selector.select_algorithm(other, prefix + "K1-Rev1-Dance1")


@pytest.mark.parametrize("action", ["train", "play", "export"])
def test_both_learners_use_the_same_entrypoint(monkeypatch, action):
    import sys
    from types import SimpleNamespace

    calls = []
    def record(action, task, argv, *, family):
        calls.append((action, task, argv, family))

    monkeypatch.setitem(sys.modules, "motion_tracker", SimpleNamespace(main=record))
    for family, task in (
        ("supermimic", "Cyclo-SuperMimic-GenCoord-K1-Rev1-001"),
        ("blinkmimic", "Cyclo-BlinkMimic-K1-Rev1-001"),
    ):
        selector.main([action, task, "physics=isaacsim_physx"])
        assert calls[-1] == (action, task, ["physics=isaacsim_physx"], family)
