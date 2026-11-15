"""Tests for the world model env's soft termination."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from supermimic.envs.motion_tracking_env import MotionTrackingEnv
from tests.conftest import (
  ANCHOR_BODY_INDEX,
  ANCHOR_ORI_THRESHOLD,
  ANCHOR_POS_THRESHOLD,
  identity_aux,
  make_mock_buffer,
  make_mock_command_env,
  make_mock_dynamics,
  make_mock_wm_representation,
)

BATCH = 4
SOFTNESS = 0.02
# Non-zero so a rule scoring against the world frame instead of the reference
# would still have to get the reference right.
REF_Z = 0.9
REF_TILT = 0.4


def _make_env(softness: float = SOFTNESS) -> MotionTrackingEnv:
  return MotionTrackingEnv(
    dynamics=make_mock_dynamics(),
    representation=make_mock_wm_representation(BATCH),
    env=make_mock_command_env(),
    batch_size=BATCH,
    replay_buffer=make_mock_buffer(),
    history_length=1,
    termination_softness=softness,
    device="cpu",
  )


def _reference(env: MotionTrackingEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """The anchor rolled by REF_TILT at REF_Z, every other body elsewhere so a
  rule reading the wrong body index cannot pass."""
  num_bodies = env.representation.num_bodies
  pos = (torch.arange(num_bodies, dtype=torch.float).view(-1, 1) * 0.5).repeat(1, 3)
  quat = torch.zeros(num_bodies, 4)
  quat[:, 0] = 1.0
  pos[ANCHOR_BODY_INDEX] = torch.tensor([0.0, 0.0, REF_Z])
  quat[ANCHOR_BODY_INDEX] = torch.tensor(
    [math.cos(REF_TILT / 2), math.sin(REF_TILT / 2), 0.0, 0.0]
  )
  return pos.expand(BATCH, num_bodies, 3), quat.expand(BATCH, num_bodies, 4)


def _aux(
  z_offset: float = 0.0, x_offset: float = 0.0, tilt_error: float = 0.0
) -> torch.Tensor:
  """Root pose offset from the reference anchor. ``tilt_error`` is the
  projected-gravity gap the orientation rule scores, not an angle."""
  tilt = math.acos(math.cos(REF_TILT) - tilt_error)
  aux = identity_aux(BATCH)
  aux[:, 0] = x_offset
  aux[:, 2] = REF_Z + z_offset
  aux[:, 3] = math.cos(tilt / 2)
  aux[:, 4] = math.sin(tilt / 2)
  return aux


def _term_prob(**offset: float) -> torch.Tensor:
  env = _make_env()
  return env._compute_term_prob(_aux(**offset), *_reference(env))


class TestSoftTermination:
  def test_on_reference_pose_does_not_terminate(self):
    assert (_term_prob() < 1e-6).all()

  def test_horizontal_error_does_not_terminate(self):
    assert (_term_prob(x_offset=5.0) < 1e-6).all()

  def test_height_error_at_threshold_is_even_odds(self):
    torch.testing.assert_close(
      _term_prob(z_offset=ANCHOR_POS_THRESHOLD), torch.full((BATCH,), 0.5)
    )

  def test_height_error_past_threshold_terminates(self):
    assert (_term_prob(z_offset=2 * ANCHOR_POS_THRESHOLD) > 1 - 1e-6).all()

  def test_tilt_error_at_threshold_is_even_odds(self):
    torch.testing.assert_close(
      _term_prob(tilt_error=ANCHOR_ORI_THRESHOLD), torch.full((BATCH,), 0.5)
    )

  def test_tilt_error_past_threshold_terminates(self):
    assert (_term_prob(tilt_error=2 * ANCHOR_ORI_THRESHOLD) > 1 - 1e-6).all()

  def test_the_larger_error_wins(self):
    term_prob = _term_prob(
      z_offset=0.9 * ANCHOR_POS_THRESHOLD, tilt_error=0.5 * ANCHOR_ORI_THRESHOLD
    )
    torch.testing.assert_close(
      term_prob, torch.full((BATCH,), 1.0 / (1.0 + math.exp(0.1 / SOFTNESS)))
    )

  def test_each_env_is_scored_against_its_own_pose(self):
    aux = _aux()
    aux[1::2, 2] += 2 * ANCHOR_POS_THRESHOLD
    env = _make_env()
    term_prob = env._compute_term_prob(aux, *_reference(env))
    assert (term_prob[0::2] < 1e-6).all()
    assert (term_prob[1::2] > 1 - 1e-6).all()

  def test_step_scores_the_post_step_pose(self):
    env = _make_env()
    env.reset()
    env._ref_body_state = (*_reference(env), None, None)
    env.aux_state = _aux()
    env.dynamics = MagicMock(
      return_value=(_aux(z_offset=2 * ANCHOR_POS_THRESHOLD), env.rep_state)
    )
    _, _, term_prob, _ = env.step(torch.zeros(BATCH, env.num_actions))
    assert (term_prob > 1 - 1e-6).all()

  def test_termination_keeps_the_gradient_to_the_predicted_pose(self):
    env = _make_env()
    env.reset()
    env._ref_body_state = (*_reference(env), None, None)
    aux = _aux(z_offset=ANCHOR_POS_THRESHOLD).requires_grad_(True)
    env.dynamics = MagicMock(return_value=(aux, env.rep_state))
    _, _, term_prob, _ = env.step(torch.zeros(BATCH, env.num_actions))
    term_prob.sum().backward()
    assert (aux.grad[:, 2].abs() > 0).all()

  def test_non_positive_softness_is_rejected(self):
    with pytest.raises(ValueError, match="termination_softness"):
      _make_env(softness=0.0)
