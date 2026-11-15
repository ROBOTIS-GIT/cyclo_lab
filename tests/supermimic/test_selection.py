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


def test_legacy_method_and_forwarding(monkeypatch):
    calls = []
    monkeypatch.setattr(
        selector.subprocess, "call", lambda args: calls.append(args) or 0
    )
    assert (
        selector.main(
            ["--method", "supermimic", "play", "--checkpoint", "local.pt", "--headless"]
        )
        == 0
    )
    assert "--play" in calls[0] and "local.pt" in calls[0]
    selector.main(
        ["train", "--task", "Cyclo-SuperMimic-K1-Rev1-Dance1", "physics=isaacsim_physx"]
    )
    assert calls[1][1].endswith("supermimic/train.py")
    assert calls[1][-1] == "physics=isaacsim_physx"
    assert selector.select_algorithm(None, None) == "beyondmimic"
