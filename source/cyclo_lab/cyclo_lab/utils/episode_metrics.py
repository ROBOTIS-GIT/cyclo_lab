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
