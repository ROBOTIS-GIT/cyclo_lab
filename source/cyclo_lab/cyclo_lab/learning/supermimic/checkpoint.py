# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Restricted local checkpoints and declarative native-environment compatibility."""

import dataclasses
import hashlib
import math
from pathlib import Path
import torch
from . import ROOT
from .motion_physics import URDF, PROFILE

FORMAT = "cyclo-supermimic-isaac-v2"
OBSERVATION = {
    "representation": "gencoord",
    "full_state_width": 59,
    "actor": "rep_state",
    "history": 1,
    "encoders": ["motion"],
    "context": [0.0],
    "root_twist": "world_link_origin",
    "quaternion": "wxyz",
    "action_clip": None,
}


def _plain(value):
    """Stable primitives; never executable objects or address-dependent reprs."""
    if value is None or isinstance(value, (str, bool, int)):
        return (
            value.replace(str(ROOT), "<checkout>") if isinstance(value, str) else value
        )
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, Path):
        return _plain(str(value))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    raise TypeError(f"Unsupported declarative contract value: {type(value).__name__}")


def validate_native_cfg(cfg):
    if not type(cfg.sim.physics).__module__.startswith("isaaclab_physx."):
        raise ValueError(
            "Native SuperMimic supports physics=isaacsim_physx only; Newton is not validated"
        )
    if cfg.decimation < 1 or not math.isclose(
        cfg.sim.dt * cfg.decimation, 0.02, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(
            "Native SuperMimic requires exactly 50 Hz policy timing; no resampling"
        )
    active = [
        k
        for k, v in vars(cfg.events).items()
        if not k.startswith("_") and v is not None
    ]
    if active:
        raise ValueError(
            f"Zero-context SuperMimic requires fixed dynamics; active events: {active}"
        )
    if cfg.commands.reference_trajectory.anchor_body_name != "pelvis":
        raise ValueError("GenCoord motion tracking requires the pelvis anchor")
    if Path(cfg.scene.robot.spawn.asset_path).resolve() != URDF.resolve():
        raise ValueError("Only the pinned K1 URDF is supported")
    from .bridge import BODIES

    if list(cfg.commands.reference_trajectory.body_names) != list(BODIES):
        raise ValueError(
            "Tracked body order differs from the supported K1 reward contract"
        )
    if cfg.actions.joint_pos.joint_names != [".*"]:
        raise ValueError(
            "Native action order must cover all K1 joints in articulation order"
        )
    if cfg.actions.joint_pos.clip is not None:
        raise ValueError(
            "Action clipping would differ between real and imagined rollouts"
        )
    # The official imagined objective is fixed. Only anchor thresholds are
    # parameterized; safety/time-limit terms may not be replaced by overrides.
    from cyclo_lab.manager_based.mimic.config.supermimic_k1_rev1.env_cfg import (
        SuperMimicDance1EnvCfg,
    )

    baseline = SuperMimicDance1EnvCfg()
    if _plain(cfg.rewards) != _plain(baseline.rewards):
        raise ValueError("Reward override differs from the official imagined objective")
    for name in ("anchor_height", "anchor_ori"):
        threshold = getattr(cfg.terminations, name).params["threshold"]
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(
                "Anchor termination thresholds must be positive and finite"
            )
        getattr(baseline.terminations, name).params["threshold"] = threshold
    if _plain(cfg.terminations) != _plain(baseline.terminations):
        raise ValueError(
            "Termination override differs from the supported native contract"
        )
    if {
        k
        for k, v in vars(cfg.actions).items()
        if not k.startswith("_") and v is not None
    } != {"joint_pos"}:
        raise ValueError("Native SuperMimic supports one joint-position action term")


def environment_contract(cfg, task, motion):
    """All learning-relevant config; display/device/env-count may differ on reload.

    Playback only changes initial frame selection. No play preset replaces timing,
    physical parameters, episode limits or sensor/learning settings.
    """
    command = _plain(cfg.commands.reference_trajectory)
    for key in ("trajectory_file", "debug_vis", "start_from_zero"):
        command.pop(key, None)
    robot = _plain(cfg.scene.robot)
    robot.pop("prim_path", None)
    # The file hash includes inertial/joint/collision definitions; hash referenced
    # mesh bytes too, since a same-path mesh edit can change collision geometry.
    import xml.etree.ElementTree as ET

    meshes = {}
    description = URDF.parents[2]
    for node in ET.parse(URDF).getroot().findall(".//mesh"):
        filename = node.get("filename")
        if filename.startswith("package://ai_sapiens_description/"):
            mesh = description / filename.removeprefix(
                "package://ai_sapiens_description/"
            )
            meshes[filename] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        else:
            raise ValueError(f"Unsupported model mesh location: {filename}")
    return {
        "version": 1,
        "task": task,
        "motion_sha256": motion["sha256"],
        "motion_profiles": motion["physical_profiles"],
        "model_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
        "motion_profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        "meshes": meshes,
        "observation": OBSERVATION,
        "sim_dt": cfg.sim.dt,
        "decimation": cfg.decimation,
        "physics": _plain(cfg.sim.physics),
        "gravity": _plain(cfg.sim.gravity),
        "physics_material": _plain(cfg.sim.physics_material),
        "use_newton_actuators": cfg.sim.use_newton_actuators,
        "robot": robot,
        "terrain": _plain(cfg.scene.terrain),
        "actions": _plain(cfg.actions),
        "command": command,
        "events": _plain(cfg.events),
        "rewards": _plain(cfg.rewards),
        "terminations": _plain(cfg.terminations),
        "episode_length_s": cfg.episode_length_s,
    }


def check_contract(saved, current):
    mismatches = [
        key
        for key in sorted(saved.keys() | current.keys())
        if saved.get(key) != current.get(key)
    ]
    if mismatches:
        raise ValueError(
            f"Incompatible native checkpoint contract: {mismatches}. Restore the saved configuration; new-motion warm starts are not supported."
        )


def read_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError(
            "Expected native Isaac SuperMimic v2 checkpoint; v1/PPO/MJLab files lack this contract"
        )
    required = {
        "contract",
        "cfg",
        "iteration",
        "collected_steps",
        "motion_file",
        "dynamics",
        "actor",
        "critic",
        "wm_optimizers",
        "actor_optimizers",
        "critic_optimizers",
        "joint_names",
        "action_scale",
        "action_offset",
    }
    if required - checkpoint.keys():
        raise ValueError(
            f"Incomplete checkpoint: {sorted(required - checkpoint.keys())}"
        )
    return checkpoint


def assert_equal_tree(expected, actual, name="checkpoint"):
    if isinstance(expected, torch.Tensor):
        if not torch.equal(expected.cpu(), actual.cpu()):
            raise AssertionError(f"Tensor mismatch: {name}")
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"Keys differ: {name}")
        for key in expected:
            assert_equal_tree(expected[key], actual[key], f"{name}.{key}")
    elif isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise AssertionError(f"Length differs: {name}")
        for i, (a, b) in enumerate(zip(expected, actual, strict=True)):
            assert_equal_tree(a, b, f"{name}[{i}]")
    elif expected != actual:
        raise AssertionError(f"Value differs: {name}")
