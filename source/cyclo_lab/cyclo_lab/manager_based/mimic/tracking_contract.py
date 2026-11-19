# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Isaac task settings used by the motion learners and full training resume."""

import dataclasses
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from . import REPO_ROOT
from cyclo_lab.manager_based.mimic.motion_bridge import BODIES, URDF


def _plain(value):
    """Stable primitives; never executable objects or address-dependent reprs."""
    if value is None or isinstance(value, (str, bool, int)):
        return (
            value.replace(str(REPO_ROOT), "<checkout>") if isinstance(value, str) else value
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


def active_terms(group):
    return {k: _plain(v) for k, v in vars(group).items() if not k.startswith("_") and v is not None}


def validate_native_cfg(cfg):
    module = type(cfg.sim.physics).__module__
    if module.startswith("isaaclab_newton."):
        from isaaclab_newton.physics import MJWarpSolverCfg

        if not isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg):
            raise ValueError("Native motion tracking supports Newton MJWarp only")
        if cfg.sim.physics.num_substeps != 1:
            raise ValueError(
                "Newton num_substeps must be 1; additional solver substeps are not supported"
            )
    elif not module.startswith("isaaclab_physx."):
        raise ValueError("Native motion tracking requires Newton MJWarp or Isaac Sim PhysX")
    if cfg.decimation < 1 or not math.isclose(
        cfg.sim.dt * cfg.decimation, 0.02, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(
            "Native motion tracking requires exactly 50 Hz policy timing; no resampling"
        )
    if cfg.commands.reference_trajectory.anchor_body_name != "pelvis":
        raise ValueError("Motion tracking requires the pelvis anchor")
    if Path(cfg.scene.robot.spawn.asset_path).resolve() != URDF.resolve():
        raise ValueError("Only the pinned K1 URDF is supported")

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
    from .beyondmimic.mdp import terminations
    from cyclo_lab.manager_based.mimic.motion_library_command import at_motion_end

    command = cfg.commands.reference_trajectory
    terms = cfg.terminations
    if (
        not command.clip_timeout or terms.motion_end is None
        or terms.motion_end.func != at_motion_end or not terms.motion_end.time_out
    ):
        raise ValueError("Motion boundary must truncate the episode")
    for term, func in (
        (terms.anchor_height, terminations.reference_anchor_height_deviation),
        (terms.anchor_ori, terminations.reference_anchor_gravity_deviation),
    ):
        threshold = term.params["threshold"]
        if (
            term.func != func or term.params.get("command_name") != "reference_trajectory"
            or term.time_out or not math.isfinite(threshold) or threshold <= 0
        ):
            raise ValueError("Expected a positive finite anchor termination threshold")
    if active_terms(cfg.actions).keys() != {"joint_pos"}:
        raise ValueError("Native motion tracking supports one joint-position action term")


def collision_model(urdf, *, collision_from_visuals=False):
    model = ET.parse(urdf).getroot()
    for link in model.findall("link"):
        for visual in link.findall("visual"):
            if collision_from_visuals:
                material = visual.find("material")
                if material is not None:
                    visual.remove(material)
            else:
                link.remove(visual)
    for material in model.findall("material"):
        model.remove(material)
    meshes = {}
    for mesh in model.findall(".//geometry/mesh"):
        filename = mesh.attrib["filename"]
        if filename in meshes:
            continue
        prefix = "package://ai_sapiens_description/"
        path = urdf.parents[2] / filename[len(prefix):] if filename.startswith(prefix) else urdf.parent / filename
        with path.open("rb") as f:
            meshes[filename] = hashlib.file_digest(f, "sha256").hexdigest()
    xml = ET.canonicalize(ET.tostring(model), strip_text=True)
    return hashlib.sha256(xml.encode()).hexdigest(), meshes


def physical_settings(contract):
    robot = dict(contract["robot"])
    robot.pop("prim_path", None)
    spawn = dict(robot["spawn"])
    for key in ("visible", "visual_material", "visual_material_path", "visual_material_bindings", "semantic_tags"):
        spawn.pop(key, None)
    robot["spawn"] = spawn
    terrain = dict(contract["terrain"])
    terrain.pop("visual_material", None)
    return {**contract, "robot": robot, "terrain": terrain}


def dynamics_contract(cfg):
    model_hash, meshes = collision_model(URDF, collision_from_visuals=cfg.scene.robot.spawn.collision_from_visuals)
    return physical_settings(dict(
        collision_model_sha256=model_hash,
        meshes=meshes,
        backend_class=f"{type(cfg.sim.physics).__module__}:{type(cfg.sim.physics).__qualname__}",
        sim_dt=cfg.sim.dt,
        decimation=cfg.decimation,
        physics=_plain(cfg.sim.physics),
        gravity=_plain(cfg.sim.gravity),
        physics_material=_plain(cfg.sim.physics_material),
        use_newton_actuators=cfg.sim.use_newton_actuators,
        robot=_plain(cfg.scene.robot),
        terrain=_plain(cfg.scene.terrain),
        actions=_plain(cfg.actions),
        events=_plain(cfg.events),
    ))


def check_dynamics(bundle, current):
    saved = bundle.get("dynamics_contract")
    if saved is None:
        raise ValueError("Full resume requires a saved physics configuration")
    if "model_sha256" in saved:
        # Earlier numbered runs recorded the whole URDF and all visual meshes.
        if saved["model_sha256"] != hashlib.sha256(URDF.read_bytes()).hexdigest():
            raise ValueError("Cannot verify the physics of the previous URDF for full resume")
        saved = dict(saved)
        del saved["model_sha256"]
        saved["collision_model_sha256"] = current["collision_model_sha256"]
        saved["meshes"] = {key: saved["meshes"].get(key) for key in current["meshes"]}
    # A moved callback has the same physics; verify unknown callbacks as usual.
    events = dict(saved["events"])
    moved_modules = {
        "cyclo_lab.learning.tracking_physics": "cyclo_lab.manager_based.mimic.tracking_physics",
        "cyclo_lab.manager_based.mimic.mdp.tracking_physics": "cyclo_lab.manager_based.mimic.tracking_physics",
        "cyclo_lab.manager_based.mimic.mdp.events": "cyclo_lab.manager_based.mimic.beyondmimic.mdp.events",
    }
    for name, term in events.items():
        if isinstance(term, dict) and isinstance(term.get("func"), str):
            module, _, function = term["func"].partition(":")
            if module in moved_modules:
                events[name] = dict(term, func=f"{moved_modules[module]}:{function}")
    saved = physical_settings({**saved, "events": events})
    changed = [key for key in sorted(saved.keys() | current.keys()) if saved.get(key) != current.get(key)]
    if changed:
        raise ValueError(f"Checkpoint physics differs for full resume: {', '.join(changed)}")
