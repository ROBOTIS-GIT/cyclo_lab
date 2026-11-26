# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""Export the actor's actual observation contract and live Isaac actuator settings."""

from copy import deepcopy
from pathlib import Path

import torch

from cyclo_lab.utils.sim2real_yaml import write_sim2real_yaml
from .. import REPO_ROOT


def encoder_directory(root, encoder):
    return root if encoder == "motion" else root / encoder


def export_sim2real_cfg(runner, run_dir):
    robot = runner.bridge.robot
    data = robot.data
    names = list(robot.joint_names)
    scale, offset = runner.bridge.action_transform()
    props = {
        name: dict(default_position=q, stiffness=kp, damping=kd, position_limit=None)
        for name, q, kp, kd in zip(
            names,
            data.default_joint_pos.torch[0].tolist(),
            data.default_joint_stiffness.torch[0].tolist(),
            data.default_joint_damping.torch[0].tolist(),
            strict=True,
        )
    }
    for encoder in runner.active_encoders:
        directory = encoder_directory(Path(run_dir) / "params", encoder)
        config = dict(
            policy_joints=names,
            step_dt=runner.bridge.step_dt,
            joint_properties=props,
            commands=runner.actor_export.commands(encoder),
            actions={"joint_pos": dict(clip=None, scale=scale.tolist(), offset=offset.tolist())},
            observations=runner.actor_export.actor_contract(runner, encoder),
        )
        write_sim2real_yaml(directory / "sim2real.yaml", config)


def export_actors(runner, output, *, encoders=None):
    for encoder in runner.active_encoders if encoders is None else encoders:
        directory = encoder_directory(Path(output), encoder)
        directory.mkdir(parents=True, exist_ok=True)
        wrapper, name = runner.actor_export.actor_module(runner, encoder)
        wrapper = deepcopy(wrapper).cpu().eval()
        torch.onnx.export(
            wrapper,
            (torch.zeros(1, wrapper.input_size),),
            str(directory / "policy.onnx"),
            input_names=[name],
            output_names=["actions"],
            opset_version=18,
            dynamo=False,
            dynamic_axes={name: {0: "batch"}, "actions": {0: "batch"}}
            if runner.actor_export.DYNAMIC_BATCH
            else None,
        )
        write_export_notices(directory, runner.family, runner.actor_export)


def write_export_notices(directory, family, actor_export):
    from shutil import copyfile

    vendor = REPO_ROOT / "third_party" / family
    licenses = directory / "licenses"
    licenses.mkdir(exist_ok=True)
    copyfile(REPO_ROOT / "LICENSE", licenses / "Cyclo-Lab-LICENSE.txt")
    copyfile(vendor / "LICENSE", licenses / f"{family}-LICENSE.txt")
    source, name = actor_export.dependency_notice(vendor)
    copyfile(source, licenses / name)
    copyfile(REPO_ROOT / "THIRD_PARTY_LICENSES.md", directory / "NOTICE.txt")
