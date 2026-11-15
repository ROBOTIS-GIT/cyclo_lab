"""Tests for MotionTrackingEnv (single-actor PPO-runner WM env)."""

from __future__ import annotations

import torch

from supermimic.envs.motion_tracking_env import MotionTrackingEnv
from tests.conftest import (
  ACTION_DIM,
  MOTION_GOAL_DIM,
  make_mock_buffer,
  make_mock_command_env,
  make_mock_dynamics,
  make_mock_wm_representation,
)

BATCH = 8


def _make_env(history_length: int = 1) -> MotionTrackingEnv:
  rep = make_mock_wm_representation(BATCH)
  rep.build_actor_obs.side_effect = lambda hist, goal: torch.cat(
    [hist.flatten(start_dim=1), goal], dim=-1
  )

  return MotionTrackingEnv(
    history_length=history_length,
    dynamics=make_mock_dynamics(),
    representation=rep,
    env=make_mock_command_env(),
    batch_size=BATCH,
    replay_buffer=make_mock_buffer(),
    termination_softness=0.05,
    device="cpu",
  )


class TestMotionTrackingEnv:
  def test_reset_obs_has_actor_and_critic(self):
    env = _make_env()
    obs = env.reset()
    assert "actor" in obs.keys()
    assert "critic" in obs.keys()
    assert obs["actor"].shape[0] == BATCH
    assert obs["critic"].shape[0] == BATCH

  def test_step_advances_motion_times(self):
    env = _make_env()
    env.reset()
    times_before = env.motion_times.clone()
    env.step(torch.zeros(BATCH, ACTION_DIM))
    assert (env.motion_times == times_before + 1).all()

  def test_step_returns_correct_shapes(self):
    env = _make_env()
    env.reset()
    obs, rewards, dones, _ = env.step(torch.zeros(BATCH, ACTION_DIM))
    assert obs["actor"].shape[0] == BATCH
    assert rewards.shape == (BATCH,)
    assert dones.shape == (BATCH,)

  def test_num_actions_is_joint_dim(self):
    env = _make_env()
    assert env.num_actions == ACTION_DIM

  def test_history_window_widens_actor_obs(self):
    """history_length > 1 stacks H sensor frames into the actor obs (base ring)."""
    obs1 = _make_env(history_length=1).reset()
    obs4 = _make_env(history_length=4).reset()
    sensor_dim = obs1["actor"].shape[-1] - MOTION_GOAL_DIM
    assert obs4["actor"].shape[-1] == 4 * sensor_dim + MOTION_GOAL_DIM
