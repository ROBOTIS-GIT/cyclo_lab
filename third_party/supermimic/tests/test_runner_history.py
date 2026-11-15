"""Real-sim obs-history ring (BaseRunner): noised frames, shared-on-pad."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from supermimic.runners.supermimic_runner import BaseRunner


def _runner(history_length: int) -> SimpleNamespace:
  rep = MagicMock()
  rep.add_proprio_noise.side_effect = lambda x: x + torch.rand_like(x)
  return SimpleNamespace(
    cfg=SimpleNamespace(history_length=history_length),
    _representation=rep,
    _sim_history=None,
    device="cpu",
    env=MagicMock(),
  )


def test_seed_repeats_one_noised_frame():
  runner = _runner(history_length=4)
  obs0 = torch.arange(8 * 5).float().reshape(8, 5)
  runner.env.get_observations.return_value = {"actor": obs0}
  BaseRunner._seed_sim_history(runner)
  ring = runner._sim_history
  assert ring.shape == (8, 4, 5)
  # All seed slots share one noised realization (shared pad), and it is noised.
  assert torch.equal(ring[:, 0], ring[:, 3])
  assert not torch.allclose(ring[:, 0], obs0)


def test_advance_appends_noised_and_shares_on_done():
  runner = _runner(history_length=4)
  runner._sim_history = torch.zeros(8, 4, 5)
  next_actor = torch.arange(8 * 5).float().reshape(8, 5) + 1.0
  dones = torch.zeros(8, dtype=torch.bool)
  dones[0] = True
  BaseRunner._advance_sim_history(runner, next_actor, dones)
  ring = runner._sim_history
  # Done rows collapse to one shared realization; live rows shift and append
  # the freshly-noised newest frame.
  assert torch.equal(ring[0, 0], ring[0, 3])
  assert torch.equal(ring[1, :3], torch.zeros(3, 5))
  assert not torch.allclose(ring[1, 3], next_actor[1])


def test_history_length_1_keeps_no_ring():
  runner = _runner(history_length=1)
  BaseRunner._seed_sim_history(runner)
  BaseRunner._advance_sim_history(
    runner, torch.randn(8, 5), torch.zeros(8, dtype=torch.bool)
  )
  assert runner._sim_history is None
  # At H==1 the collection window is just the current sensor, unsqueezed.
  sensor = torch.randn(8, 5)
  window = BaseRunner._collection_history(runner, sensor)
  assert torch.equal(window, sensor.unsqueeze(1))
