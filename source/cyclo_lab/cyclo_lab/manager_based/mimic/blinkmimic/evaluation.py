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

"""Evaluate the deployed EMA policy on deterministic reference segments in Isaac."""

from datetime import datetime
import json

import torch

from .. import REPO_ROOT


@torch.no_grad()
def evaluate(runner, *, tiles=False):
    command = runner.bridge.env.command_manager.get_term("reference_trajectory")
    ends = command.motion_ends.tolist()
    segments = []
    lead = int(torch.randint(200, 451, ()).item()) if tiles else 0
    begin = 0
    for index, end in enumerate(ends):
        stride = 300 if tiles else end - begin
        for start in range(begin, end - 1, stride):
            segments.append((index, max(begin, start - lead), min(end - 1, start + stride)))
        begin = end

    envs = runner.bridge.env.num_envs
    records = []
    for i in range(0, len(segments), envs):
        chunk = segments[i : i + envs]
        starts = [s for _, s, _ in chunk]
        command.playback_frames = torch.tensor(
            starts + [starts[-1]] * (envs - len(chunk)), device=runner.device
        )
        try:
            runner.env.reset()
        finally:
            command.playback_frames = None
        runner.history = None
        runner.steps.zero_()
        runner.previous.zero_()
        stops = torch.tensor([stop for _, _, stop in chunk], device=runner.device)
        active = torch.ones_like(stops, dtype=torch.bool)
        failures = torch.full_like(stops, -1)
        for _ in range(max(stop - start for _, start, stop in chunk)):
            frames = runner.bridge.frame_ids()[: len(chunk)]
            terminated, done = runner.step(False)
            failures = torch.where(active & terminated[: len(chunk)], frames, failures)
            active &= ~done[: len(chunk)] & (frames + 1 < stops)
            if not active.any():
                break
        for (clip, start, end), failure in zip(chunk, failures.tolist(), strict=True):
            records.append(dict(clip=clip, start=start, end=end, failure_frame=failure))
    return records


def write_evaluation_report(runner, checkpoint, probe_report=None):
    report = probe_report or (
        checkpoint.parent / "evaluation" / f"{datetime.now():%Y%m%dT%H%M%S%f}" / "clips.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(evaluate(runner, tiles=bool(probe_report)), indent=2))
    print(f"[INFO] Wrote evaluation to {report}", flush=True)


def run_training_probe_if_due(runner, task, output, backend, save_checkpoint, motions):
    step = runner.iteration
    interval = runner.cfg.eprobe_every
    if runner.sampler is None or interval <= 0 or step < runner.cfg.eprobe_start or step % interval:
        return

    save_checkpoint()
    directory = output / "evaluation" / f"{step}-{datetime.now():%Y%m%dT%H%M%S%f}"
    physics = "newton_mjwarp" if backend == "MJWarpSolverCfg" else "isaacsim_physx"
    run_training_probe(runner, task, output / "last.pt", physics, directory, motions)


def run_training_probe(runner, task, checkpoint, backend, directory, motions):
    """A separate Isaac process owns evaluation physics. Training replay is untouched."""
    import subprocess
    import sys

    directory.mkdir(parents=True, exist_ok=False)
    report = directory / "segments.json"
    script = REPO_ROOT / "scripts/reinforcement_learning/play.py"
    with (directory / "console.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(script),
                task,
                "--checkpoint",
                str(checkpoint.resolve()),
                "--headless",
                "--physics",
                backend,
                "--num-envs",
                str(min(32, runner.bridge.env.num_envs)),
                "--probe-report",
                str(report.resolve()),
                # Probe failure frames index the training library in this order.
                "--motion-file",
                *map(str, motions),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    failures = [row["failure_frame"] for row in json.loads(report.read_text()) if row["failure_frame"] >= 0]
    runner.sampler.record_probe(failures)
