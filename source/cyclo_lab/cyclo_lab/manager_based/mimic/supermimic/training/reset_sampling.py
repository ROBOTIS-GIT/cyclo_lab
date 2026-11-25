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

"""Adapt the SuperMimic return sampler to the Isaac motion library."""

import torch


class ReturnResetSampler:
    def __init__(self, env, command, max_reward):
        from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg

        frames = command.reference.num_frames
        self.count = int(frames // (0.2 / env.step_dt)) + 1
        self.frames = frames
        self.device = env.device
        self.clip_ends = command.motion_ends
        self.learner = ReturnSampler(
            ReturnSamplerCfg(), self.count, frames, env.num_envs, max_reward, env.device
        )
        self.probs = torch.full_like(self.learner.value, 1 / self.count)

    def sample(self, env_ids):
        bins = torch.multinomial(self.probs, len(env_ids), replacement=True)
        total = self.frames - 2
        lo = bins * total // self.count
        span = ((bins + 1) * total // self.count - lo).clamp_min(1)
        frame = lo + (torch.rand(len(env_ids), device=self.device) * span).long()
        return frame - torch.isin(frame, self.clip_ends - 1).long()

    def record(self, *transition):
        probs = self.learner.record(*transition)
        if probs is not None:
            self.probs = probs

    def statistics(self):
        return dict(
            value=self.learner.value, count=self.learner.count, effort=self.learner.effort, probs=self.probs
        )

    def bin_centers(self):
        return ((torch.arange(self.count, device=self.device) + 0.5) * self.frames / self.count).long()

    def bins_at(self, frames):
        return frames * self.count // self.frames
