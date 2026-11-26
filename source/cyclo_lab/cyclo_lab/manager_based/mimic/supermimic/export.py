# Copyright 2026 The SuperMimic Authors.
# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Kiwoong Park, The SuperMimic Authors
# SingleEncoderActor adapts supermimic.runners.exporter._SingleEncoderOnnx.
"""Portable single-encoder ONNX adapter from the upstream exporter."""

import torch
from torch import nn
from . import upstream  # noqa: F401  # Expose the pinned SuperMimic source before upstream imports.
from supermimic.models import MultiEncoderAE
from supermimic.representations import term_major_to_temporal


DYNAMIC_BATCH = False


def velocity_ranges():
    from supermimic.tasks.task_spec import VelocityTaskCfg

    cfg = VelocityTaskCfg()
    return dict(lin_vel_x=cfg.lin_vel_x_range, lin_vel_y=cfg.lin_vel_y_range, ang_vel_z=cfg.ang_vel_z_range)


def commands(encoder):
    if encoder == "velocity":
        return {"base_velocity": {"ranges": {name: list(bounds) for name, bounds in velocity_ranges().items()}}}
    return {}


def dependency_notice(vendor):
    from importlib.metadata import distribution

    rsl = distribution("rsl-rl-lib")
    license_path = next(p for p in rsl.files if p.name == "LICENSE" and ".dist-info" in str(p))
    return rsl.locate_file(license_path), "RSL-RL-BSD-3-Clause.txt"


class SingleEncoderActor(nn.Module):
    """Single-task slice of the multi-encoder actor: ``cat([history, goal]) → action``.

    Bakes the chosen encoder + shared decoder + the matching normalizers into
    one ONNX-compatible module. The deploy-time latent (``tanh(enc_out)``) is
    applied here rather than read from the actor's forward
    to keep the graph closed and free of training-only branches.

    The obs part of the input is the TERM-MAJOR ``history_length``-frame stack —
    mjlab's native layout, ``[term0_t0..tH-1, term1_t0..tH-1, ...]`` — converted
    internally to the frame-major window the decoder consumes; the encoder
    consumes only the goal.
    """

    def __init__(self, actor: MultiEncoderAE, task_name: str, term_widths: list[int]) -> None:
        super().__init__()
        self.obs_normalizer = actor.obs_normalizer
        self.goal_normalizer = actor.goal_normalizers[task_name]
        self.encoder = actor.encoders[task_name]
        self.decoder = actor.decoder
        self.history_length = actor.history_length
        self.hist_size = actor.history_length * actor.obs_dim
        self.term_widths = term_widths
        self.goal_dim = next(ts for ts in actor.task_specs if ts.name == task_name).goal_dim
        self.input_size = self.hist_size + self.goal_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history = term_major_to_temporal(x[..., : self.hist_size], self.term_widths, self.history_length)
        goal = x[..., self.hist_size :]
        history_n = self.obs_normalizer(history)
        z = torch.tanh(self.encoder(self.goal_normalizer(goal)))
        return self.decoder(torch.cat([z, history_n.flatten(start_dim=1)], dim=-1))


def actor_contract(runner, encoder):
    n = runner.bridge.num_actions
    cfg = runner.cfg
    from supermimic.representations import NUM_GOAL_FRAMES, GOAL_FRAME_STRIDE

    def group(fields, history=0):
        return dict(
            layout="term_major" if history else "frame_major",
            terms={
                name: dict(scale=[1.0] * width, clip=None, **({"history_length": history} if history else {}))
                for name, width in fields
            },
        )

    if cfg.actor_obs == "proprio":
        fields = zip(
            ("projected_gravity", "base_ang_vel", "joint_pos_rel", "joint_vel_rel", "last_action"),
            runner.rep.proprio_term_widths(n),
            strict=True,
        )
    elif cfg.actor_obs == "rep_state":
        if cfg.history_length != 1:
            raise ValueError("rep_state export requires history_length=1")
        fields = [("rep_state", runner.rep.rep_state_dim)]
    else:
        raise ValueError(f"Unknown actor observation mode: {cfg.actor_obs}")
    contract = dict(
        schema="cyclo.supermimic.external_goal.v1",
        encoder=encoder,
        representation=cfg.representation,
        actor_obs=cfg.actor_obs,
        history=group(fields, cfg.history_length),
    )
    if encoder == "motion":
        contract["motion_goal"] = dict(
            frames=NUM_GOAL_FRAMES,
            stride=GOAL_FRAME_STRIDE,
            start_offset=1,
            groups=[
                group([("future_joint_pos", n), ("future_joint_vel", n)]),
                group([("future_root_orientation", 6)]),
            ],
        )
    else:
        contract["velocity_goal"] = group([("velocity_command", 3)])
    return {"supermimic": contract}


def actor_module(runner, encoder):
    widths = (
        runner.rep.proprio_term_widths(runner.bridge.num_actions)
        if runner.cfg.actor_obs == "proprio" else [runner.rep.rep_state_dim]
    )
    return SingleEncoderActor(runner.actor, encoder, widths), "obs"
