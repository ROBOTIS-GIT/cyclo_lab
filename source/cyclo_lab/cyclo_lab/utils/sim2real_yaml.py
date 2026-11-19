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

"""Write deployment configuration with the shared sim2real.yaml formatting."""

from pathlib import Path
from typing import Any

import yaml


class _FlowList(list):
    pass


class _Sim2RealYamlDumper(yaml.SafeDumper):
    pass


def _represent_flow_list(dumper: yaml.SafeDumper, data: _FlowList) -> yaml.SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_Sim2RealYamlDumper.add_representer(_FlowList, _represent_flow_list)


_FLOW_LIST_KEYS = {"scale", "offset", "clip", "lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"}


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


def write_sim2real_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(_format_yaml(config), stream, Dumper=_Sim2RealYamlDumper, default_flow_style=False, sort_keys=False)
