# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Kiwoong Park

"""Isaac task settings required by the motion learners."""

import dataclasses
import math
from pathlib import Path
import torch
from .. import REPO_ROOT
from cyclo_lab.manager_based.mimic._shared.motion_bridge import BODIES


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
    from ..beyondmimic.mdp import terminations
    from cyclo_lab.manager_based.mimic._shared.motion_library_command import at_motion_end

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
