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

"""Preserve environment-wrapper action clipping in deployed ONNX policies."""

import math
from pathlib import Path


def clip_onnx_actions(path: str | Path, clip_actions: float | None) -> None:
    """Append clipping to the exported actions, preserving the public ONNX interface.

    Both Isaac Lab and RSL-RL exporters name the policy output ``actions``.
    Internal consumers retain the raw tensor; only the public action output is
    clipped, leaving recurrent state outputs and observation normalization intact.
    """
    if clip_actions is None:
        return
    if not math.isfinite(clip_actions) or clip_actions < 0:
        raise ValueError("clip_actions must be finite and non-negative, or None")

    import onnx
    from onnx import helper

    model = onnx.load(path)
    graph = model.graph
    action = next((output for output in graph.output if output.name == "actions"), None)
    if action is None:
        raise ValueError("Expected an ONNX policy output named 'actions'")
    opset = next((item.version for item in model.opset_import if item.domain == ""), 0)
    if opset < 11:
        raise ValueError("Action clipping requires ONNX opset 11 or newer")
    if any(value.name == "actions" for value in (*graph.input, *graph.initializer)):
        raise ValueError("Expected 'actions' to be produced by a policy graph node")

    used = {name for node in graph.node for name in (*node.input, *node.output)}
    used.update(value.name for value in (*graph.input, *graph.output, *graph.initializer, *graph.value_info))

    def unique_name(stem):
        name = stem
        while name in used:
            name += "_"
        used.add(name)
        return name

    raw = unique_name("cyclo_raw_actions")
    lower = unique_name("cyclo_action_clip_min")
    upper = unique_name("cyclo_action_clip_max")
    # Rename the original tensor, including internal references, so only the
    # final exported output is clipped (not e.g. a recurrent hidden state).
    for node in graph.node:
        for names in (node.input, node.output):
            for index, name in enumerate(names):
                if name == "actions":
                    names[index] = raw
    for value in graph.value_info:
        if value.name == "actions":
            value.name = raw
    dtype = action.type.tensor_type.elem_type
    graph.initializer.extend([
        helper.make_tensor(lower, dtype, [], [-clip_actions]),
        helper.make_tensor(upper, dtype, [], [clip_actions]),
    ])
    graph.node.append(helper.make_node("Clip", [raw, lower, upper], ["actions"]))
    onnx.checker.check_model(model)
    onnx.save(model, path)
