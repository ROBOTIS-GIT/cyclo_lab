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

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, TypeAlias

import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict


class _FlowList(list):
    pass


class _Sim2RealYamlDumper(yaml.SafeDumper):
    pass


def _represent_flow_list(dumper: yaml.SafeDumper, data: _FlowList) -> yaml.SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_Sim2RealYamlDumper.add_representer(_FlowList, _represent_flow_list)


_FLOW_LIST_KEYS = {"scale", "offset", "clip", "lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"}
_SIM2REAL_YAML_FILENAME = "sim2real.yaml"
_ACTION_CFG_EXPORT_OMIT_KEYS = {
    "asset_name",
    "class_type",
    "debug_vis",
    "joint_ids",
    "joint_names",
    "preserve_order",
    "use_default_offset",
}
_OBSERVATION_CFG_EXPORT_OMIT_KEYS = {"flatten_history_dim", "func", "modifiers", "noise"}

_CfgDict: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class _ExportContext:
    asset: Articulation
    asset_joint_names: list[str]
    policy_joint_names: list[str]
    policy_joint_ids: list[int]


def _format_value(value: Any) -> Any:
    """Keep exported YAML compact and stable for human review."""
    if isinstance(value, float):
        return float(f"{value:.3g}")
    if isinstance(value, tuple):
        return [_format_value(item) for item in value]
    if isinstance(value, list):
        return [_format_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item) for key, item in value.items()}
    return value


def _format_yaml(value: Any, key: str | None = None) -> Any:
    value = _format_value(value)
    if isinstance(value, list):
        items = [_format_yaml(item) for item in value]
        if key in _FLOW_LIST_KEYS:
            return _FlowList(items)
        return items
    if isinstance(value, dict):
        return {item_key: _format_yaml(item, item_key) for item_key, item in value.items()}
    return value


def _tensor_to_list(value) -> list[Any]:
    return value.detach().cpu().numpy().tolist()


def _is_sim2real_cfg_export_enabled(env: ManagerBasedRLEnv) -> bool:
    return bool(getattr(env.cfg, "export_sim2real_cfg", False))


# IsaacLab managers expose some resolved runtime metadata only through private fields after env construction.
def _resolved_action_terms(env: ManagerBasedRLEnv):
    return env.action_manager.active_terms, list(env.action_manager._terms.values())


def _resolved_policy_observation_terms(env: ManagerBasedRLEnv):
    return (
        env.observation_manager.active_terms["policy"],
        env.observation_manager._group_obs_term_cfgs["policy"],
    )


def _action_joint_names(asset_joint_names: list[str], action_term) -> list[str]:
    if action_term._joint_ids == slice(None):
        return list(asset_joint_names)
    return [asset_joint_names[int(joint_id)] for joint_id in action_term._joint_ids]


def _reorder_by_joint_name(
    values: list[Any],
    source_joint_names: list[str],
    target_joint_names: list[str],
    value_name: str,
) -> list[Any]:
    if len(values) != len(source_joint_names):
        raise ValueError(f"{value_name} has {len(values)} values for {len(source_joint_names)} joints.")
    value_by_joint_name = dict(zip(source_joint_names, values))
    missing_joint_names = [joint_name for joint_name in target_joint_names if joint_name not in value_by_joint_name]
    if missing_joint_names:
        raise ValueError(f"{value_name} is missing joints: {missing_joint_names}")
    return [value_by_joint_name[joint_name] for joint_name in target_joint_names]


def _collect_export_context(env: ManagerBasedRLEnv) -> _ExportContext:
    asset: Articulation = env.scene["robot"]
    asset_joint_names = asset.data.joint_names

    _, action_terms = _resolved_action_terms(env)
    if not action_terms:
        raise ValueError("sim2real export requires at least one action term.")
    policy_joint_names = _action_joint_names(asset_joint_names, action_terms[0])
    if len(action_terms) != 1:
        raise ValueError("sim2real export currently supports a single joint action term.")

    policy_joint_ids = [asset_joint_names.index(joint_name) for joint_name in policy_joint_names]
    return _ExportContext(
        asset=asset,
        asset_joint_names=asset_joint_names,
        policy_joint_names=policy_joint_names,
        policy_joint_ids=policy_joint_ids,
    )


def _export_joint_properties(context: _ExportContext) -> dict[str, _CfgDict]:
    asset = context.asset
    joint_names = context.policy_joint_names
    joint_ids = context.policy_joint_ids

    joint_properties = {}
    stiffness = _tensor_to_list(asset.data.default_joint_stiffness[0, joint_ids])
    damping = _tensor_to_list(asset.data.default_joint_damping[0, joint_ids])
    default_joint_pos = _tensor_to_list(asset.data.default_joint_pos[0, joint_ids])
    for joint_name, default_position, joint_stiffness, joint_damping in zip(
        joint_names, default_joint_pos, stiffness, damping
    ):
        joint_properties[joint_name] = {
            "default_position": default_position,
            "stiffness": joint_stiffness,
            "damping": joint_damping,
            "position_limit": None,
        }
    return joint_properties


def _export_commands(env: ManagerBasedRLEnv) -> _CfgDict:
    commands = {}
    if hasattr(env.cfg.commands, "base_velocity"):
        commands["base_velocity"] = {}
        command_cfg = env.cfg.commands.base_velocity
        if hasattr(command_cfg, "limit_ranges"):
            ranges = command_cfg.limit_ranges.to_dict()
        else:
            ranges = command_cfg.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        commands["base_velocity"]["ranges"] = ranges
    return commands


def _validate_action_joint_names(
    action_name: str,
    action_joint_names: list[str],
    policy_joint_names: list[str],
) -> None:
    if set(action_joint_names) != set(policy_joint_names):
        raise ValueError(
            f"Action term '{action_name}' joints must match policy_joints for sim2real export. "
            f"action_joints={action_joint_names}, policy_joints={policy_joint_names}"
        )


def _action_scale(term_cfg, action_term) -> list[Any]:
    if isinstance(term_cfg.scale, float):
        return [term_cfg.scale for _ in range(action_term.action_dim)]
    return _tensor_to_list(action_term._scale[0])


def _action_offset(term_cfg, action_term) -> list[Any]:
    if term_cfg.use_default_offset:
        return _tensor_to_list(action_term._offset[0])
    return [0.0 for _ in range(action_term.action_dim)]


def _strip_action_cfg_for_export(term_cfg: _CfgDict) -> _CfgDict:
    for key in _ACTION_CFG_EXPORT_OMIT_KEYS:
        term_cfg.pop(key, None)
    return term_cfg


def _export_action_term(action_name: str, action_term, context: _ExportContext) -> _CfgDict:
    action_joint_names = _action_joint_names(context.asset_joint_names, action_term)
    policy_joint_names = context.policy_joint_names
    _validate_action_joint_names(action_name, action_joint_names, policy_joint_names)

    term_cfg = action_term.cfg.copy()
    scale = _action_scale(term_cfg, action_term)
    term_cfg.scale = _reorder_by_joint_name(scale, action_joint_names, policy_joint_names, "action scale")

    if term_cfg.clip is not None:
        clip = _tensor_to_list(action_term._clip[0])
        term_cfg.clip = _reorder_by_joint_name(clip, action_joint_names, policy_joint_names, "action clip")

    if hasattr(term_cfg, "use_default_offset"):
        offset = _action_offset(term_cfg, action_term)
        term_cfg.offset = _reorder_by_joint_name(offset, action_joint_names, policy_joint_names, "action offset")

    return _strip_action_cfg_for_export(term_cfg.to_dict())


def _export_actions(env: ManagerBasedRLEnv, context: _ExportContext) -> _CfgDict:
    action_names, action_terms = _resolved_action_terms(env)
    return {
        action_name: _export_action_term(action_name, action_term, context)
        for action_name, action_term in zip(action_names, action_terms)
    }


def _observation_scale(obs_cfg, obs_dim: int) -> list[Any]:
    if obs_cfg.scale is None:
        return [1.0 for _ in range(obs_dim)]
    scale = obs_cfg.scale.detach().cpu().numpy().tolist()
    if isinstance(scale, float):
        return [scale for _ in range(obs_dim)]
    return scale


def _strip_observation_cfg_for_export(term_cfg: _CfgDict) -> _CfgDict:
    for key in _OBSERVATION_CFG_EXPORT_OMIT_KEYS:
        term_cfg.pop(key, None)
    return term_cfg


def _export_policy_observation_term(env: ManagerBasedRLEnv, obs_cfg) -> _CfgDict:
    obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
    term_cfg = obs_cfg.copy()
    term_cfg.scale = _observation_scale(term_cfg, obs_dims[1])
    if term_cfg.clip is not None:
        term_cfg.clip = list(term_cfg.clip)
    if term_cfg.history_length == 0:
        term_cfg.history_length = 1
    return _strip_observation_cfg_for_export(term_cfg.to_dict())


def _export_policy_observations(env: ManagerBasedRLEnv) -> _CfgDict:
    obs_names, obs_cfgs = _resolved_policy_observation_terms(env)
    return {
        obs_name: _export_policy_observation_term(env, obs_cfg)
        for obs_name, obs_cfg in zip(obs_names, obs_cfgs)
    }


def _build_sim2real_cfg(env: ManagerBasedRLEnv, context: _ExportContext) -> _CfgDict:
    return {
        "policy_joints": context.policy_joint_names,
        "step_dt": env.cfg.sim.dt * env.cfg.decimation,
        "joint_properties": _export_joint_properties(context),
        "commands": _export_commands(env),
        "actions": _export_actions(env, context),
        "observations": _export_policy_observations(env),
    }


def _write_yaml(cfg: _CfgDict, log_dir: str) -> None:
    filename = os.path.join(log_dir, "params", _SIM2REAL_YAML_FILENAME)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cfg = class_to_dict(cfg) if not isinstance(cfg, dict) else cfg
    with open(filename, "w") as file:
        yaml.dump(_format_yaml(cfg), file, Dumper=_Sim2RealYamlDumper, default_flow_style=False, sort_keys=False)


def export_sim2real_cfg(env: ManagerBasedRLEnv, log_dir: str) -> None:
    if not _is_sim2real_cfg_export_enabled(env):
        print("[INFO]: Skipping sim2real.yaml export because env.cfg.export_sim2real_cfg is not true.")
        return

    context = _collect_export_context(env)
    _write_yaml(_build_sim2real_cfg(env, context), log_dir)
