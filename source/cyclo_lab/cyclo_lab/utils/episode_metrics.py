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

"""Rolling real-simulator episode metrics shared by the two world-model trainers."""

from collections import deque
from statistics import mean

import torch


class EpisodeMetrics:
    def __init__(self, num_envs, device, *, reward_scale=None):
        self.returns = torch.zeros(num_envs, device=device)
        self.lengths = torch.zeros_like(self.returns)
        self.completed_returns = deque(maxlen=100)
        self.completed_lengths = deque(maxlen=100)
        self.reward_scale = reward_scale
        self.normalized_returns = []
        self.terminated = torch.zeros((), device=device)
        self.done = torch.zeros((), device=device)
        self.extras = []
        self.total_steps = 0
        self.wall_time = 0.0

    def record(self, reward, done, *, terminated=None, extras=None):
        self.total_steps += reward.numel()
        self.returns += reward
        self.lengths += 1
        if terminated is not None:
            self.terminated += terminated.sum()
            self.done += done.sum()
        if extras:
            episode = extras.get("episode", extras.get("log", {}))
            self.extras.append({key: torch.as_tensor(value).detach().clone() for key, value in episode.items()})
        if done.any():
            self.completed_returns.extend(self.returns[done].tolist())
            self.completed_lengths.extend(self.lengths[done].tolist())
            if self.reward_scale is not None:
                # Native BlinkMimic averages normalized returns over the completed
                # episodes of each step, then averages those step means at log time.
                normalized = self.returns[done] / (self.lengths[done] * self.reward_scale)
                self.normalized_returns.append(normalized.mean().item())
            self.returns.masked_fill_(done, 0)
            self.lengths.masked_fill_(done, 0)

    def scalars(self, *, reset=True):
        logs = {
            "Train/mean_reward": mean(self.completed_returns),
            "Train/mean_episode_length": mean(self.completed_lengths),
        } if self.completed_returns else {}
        if self.reward_scale is not None:
            if self.normalized_returns:
                logs["BlinkMimic/normalized_episode_return"] = mean(self.normalized_returns)
            logs["BlinkMimic/termination_fraction"] = (self.terminated / self.done.clamp(min=1)).item()
        for key in dict.fromkeys(key for entry in self.extras for key in entry):
            values = torch.cat([entry[key].reshape(-1) for entry in self.extras if key in entry])
            logs[key if "/" in key else "Episode/" + key] = values.float().mean().item()
        if reset:
            self.reset_interval()
        return logs

    def reset_interval(self):
        self.normalized_returns.clear()
        self.terminated.zero_()
        self.done.zero_()
        self.extras.clear()
