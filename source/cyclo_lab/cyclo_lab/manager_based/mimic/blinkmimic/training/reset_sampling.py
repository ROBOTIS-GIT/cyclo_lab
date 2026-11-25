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

"""Sample clip-local BlinkMimic resets in Isaac Lab."""

import torch

from ..core.sampling_bins import episode_return, probe_distribution, record_probe_failures, reset_probabilities, update_bin_statistics


class BlinkResetSampler:
    """One-second clip-local bins, with low-return and recent-failure resets."""

    def __init__(self, env, command, gamma):
        self.gamma = gamma
        self.device = env.device
        ends = command.motion_ends
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))
        self.starts = torch.cat(
            [torch.arange(s, e - 1, 50, device=self.device) for s, e in zip(starts.tolist(), ends.tolist())]
        )
        self.ends = ends[torch.bucketize(self.starts, ends, right=True)] - 1
        self.widths = (self.ends - self.starts).clamp(max=50)
        idx = torch.arange(len(self.starts), device=self.device)
        self.prev = torch.where(torch.isin(self.starts, starts), idx, idx - 1)
        following = (idx + 1).clamp_max(len(idx) - 1)
        self.next_bin = torch.where(self.ends[following] == self.ends, following, idx)
        self.returns = torch.zeros_like(self.starts, dtype=torch.float)
        self.failures = torch.zeros_like(self.returns)
        self.effort = torch.full_like(self.returns, 1 / len(idx))
        self.probe_failures = torch.zeros_like(self.returns)
        self.start_bin = self.starts.new_zeros(env.num_envs)
        self.discounted = self.returns.new_zeros(env.num_envs)
        self.total = torch.zeros_like(self.discounted)

    def statistics(self):
        return dict(
            returns=self.returns,
            failures=self.failures,
            effort=self.effort,
            probe_failures=self.probe_failures,
        )

    def bin_centers(self):
        return self.starts + self.widths // 2

    def bins_at(self, frames):
        return torch.bucketize(frames.contiguous(), self.starts, right=True) - 1

    def sample(self, env_ids):
        probs = reset_probabilities(
            self.returns, self.effort, self.failures, self.prev, 0.1 / (1 - self.gamma), 0.3,
        )
        if self.probe_failures.any():
            probs = 0.85 * probs + 0.15 * probe_distribution(self.probe_failures, self.next_bin, 10)
        ids = torch.multinomial(probs, len(env_ids), replacement=True)
        return self.starts[ids] + (torch.rand(len(ids), device=self.device) * self.widths[ids]).long()

    def record_probe(self, failures):
        bins = self.bins_at(torch.as_tensor(failures, dtype=torch.long, device=self.device))
        record_probe_failures(self.probe_failures, bins, 0.7)

    def record(self, frames, reward, terminated, done, steps):
        self.start_bin = torch.where(steps == 0, self.bins_at(frames), self.start_bin)
        self.total += reward
        self.discounted += self.gamma**steps * reward
        if done.any():
            length = steps[done] + 1
            returns = episode_return(self.discounted[done], self.total[done], length, terminated[done], self.gamma)
            update_bin_statistics(self.returns, self.start_bin[done], returns, 0.8)
            self.failures.mul_(0.999)
            update_bin_statistics(self.failures, self.bins_at(frames[done]), terminated[done].float(), 0.9)
            self.total[done] = self.discounted[done] = 0
