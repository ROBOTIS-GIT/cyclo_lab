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

"""World-model prediction and velocity controls for the SuperMimic playback viewer."""

from copy import copy

import torch

from . import upstream  # noqa: F401  # Expose the pinned SuperMimic source before upstream imports.
from supermimic.tasks.task_spec import build_actor_obs_td


class PlaybackPrediction:
    def __init__(self, runner):
        self.runner = runner
        self.fk = copy(runner.bridge.fk)
        self.fk.body_names = list(runner.bridge.robot.body_names)
        self.reset()

    def reset(self):
        self.steps = 0

    @torch.no_grad()
    def advance(self):
        r = self.runner
        rep = r.rep
        if self.steps == 0 or (r.reset_rows is not None and r.reset_rows[0]):
            state = r.bridge.state()[:1]
            self.aux = rep.extract_aux_state(state)
            self.state = rep.represent(state)
            self.history = None
            self.action = torch.zeros_like(r.bridge.env.action_manager.action[:1])
            self.context = r.bridge.context()[:1]
            self.steps = 0
        goals = {
            spec.name: spec.extract_training_goal(r.bridge.motion, r.reference_frames()[:1], self.aux)
            for spec in r.specs
        }
        if r.play_velocity is not None:
            goals["velocity"] = r.play_velocity.reshape(1, -1)
        sensor = rep.compute_actor_sensor(
            self.state, self.context, self.aux, rep.joint_state(self.state), self.action
        )
        if self.history is None:
            self.history = sensor[:, None].expand(-1, r.cfg.history_length, -1)
        else:
            self.history = torch.cat((self.history[:, 1:], sensor[:, None]), dim=1)
        self.action = r.actor(build_actor_obs_td(goals, self.history))
        self.aux, self.state = r.dynamics(self.state, self.action, self.aux, self.context)
        self.steps = (self.steps + 1) % r.cfg.policy.horizon
        state = rep.reconstruct_full_state(self.aux, self.state)
        if r.cfg.representation == "maxcoord":
            count = len(r.bridge.fk.body_names)
            return (
                r.bridge.fk.body_names,
                state[0, :3 * count].reshape(-1, 3),
                state[0, 3 * count:7 * count].reshape(-1, 4)[:, [1, 2, 3, 0]],
            )
        zeros = torch.zeros_like(state[:, :3])
        positions, orientations, _, _ = self.fk(
            state[:, :3], state[:, 3:7], state[:, 7:7 + r.bridge.num_actions],
            zeros, zeros, torch.zeros_like(state[:, 7:7 + r.bridge.num_actions]),
        )
        return self.fk.body_names, positions[0], orientations[0][:, [1, 2, 3, 0]]


def create_play_viewer(runner, motions):
    from cyclo_lab.manager_based.mimic._shared.play_viewer import TrackingPlayViewer
    from .export import velocity_ranges

    controls = None
    if runner.play_velocity is not None:
        ranges = velocity_ranges()
        controls = [
            ("Forward (m/s)", ranges["lin_vel_x"]),
            ("Lateral (m/s)", ranges["lin_vel_y"]),
            ("Yaw (rad/s)", ranges["ang_vel_z"]),
        ]
    return TrackingPlayViewer.create(
        runner, motions, predictor=PlaybackPrediction(runner), velocity_ranges=controls,
    )
