"""Check the container stack independently of Cyclo task API migration."""

import argparse
import importlib
import importlib.metadata as metadata
from pathlib import Path
import sys
import tomllib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gpu", action="store_true", help="Also verify CUDA and a headless Sim launch.")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
lab = root / "third_party" / "IsaacLab"
with (lab / "pyproject.toml").open("rb") as stream:
    versions = tomllib.load(stream)["tool"]["isaaclab"]["versions"]
assert sys.version_info[:2] == (3, 12), sys.version
print(f"Python: {sys.version.split()[0]}")
print(f"Isaac Lab checkout version: {(lab / 'VERSION').read_text().strip()}")
for package in ("torch", "torchvision"):
    module = importlib.import_module(package)
    actual = module.__version__
    assert actual.split("+")[0] == versions[package], (package, actual, versions[package])
    print(f"{package}: {actual}")
for package in ("isaaclab", "isaaclab-physx", "isaaclab-rl", "isaaclab-mimic", "isaaclab-teleop", "cyclo_lab", "robotis_dds_python", "cyclonedds"):
    print(f"{package}: {metadata.version(package)}")
importlib.import_module("cyclonedds")
for distribution, module_name in (("rerun-sdk", "rerun"), ("viser", "viser")):
    importlib.import_module(module_name)
    print(f"{distribution}: {metadata.version(distribution)}")
import torch
assert torch.version.cuda is not None, "A CUDA-enabled PyTorch build is required."
print(f"PyTorch CUDA build: {torch.version.cuda}")
if args.gpu:
    assert torch.cuda.is_available(), "CUDA is not available in this container."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    try:
        from isaacsim.core.version import get_version
        actual_sim = ".".join(str(part) for part in get_version()[2:5])
        assert actual_sim == "6.1.0", actual_sim
        print(f"Isaac Sim: {actual_sim}")
        launcher.app.update()
        print("Headless Sim launch: OK")
        from isaaclab.sim import SimulationCfg, SimulationContext
        sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))
        sim.reset()
        for _ in range(10):
            sim.step()
        print("Empty PhysX scene (10 steps): OK")
    finally:
        launcher.app.close()
print("Environment check: OK")
