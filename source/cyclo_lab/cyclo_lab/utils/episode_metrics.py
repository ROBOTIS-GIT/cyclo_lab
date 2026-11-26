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
    def __init__(self, num_envs, device):
        self.returns = torch.zeros(num_envs, device=device)
        self.lengths = torch.zeros_like(self.returns)
        self.completed_returns = deque(maxlen=100)
        self.completed_lengths = deque(maxlen=100)

    def record(self, reward, done):
        self.returns += reward
        self.lengths += 1
        if done.any():
            self.completed_returns.extend(self.returns[done].tolist())
            self.completed_lengths.extend(self.lengths[done].tolist())
            self.returns.masked_fill_(done, 0)
            self.lengths.masked_fill_(done, 0)

    def scalars(self):
        if not self.completed_returns:
            return {}
        return {
            "Train/mean_reward": mean(self.completed_returns),
            "Train/mean_episode_length": mean(self.completed_lengths),
        }
