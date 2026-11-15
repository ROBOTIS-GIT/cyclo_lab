import torch
import torch.nn as nn

from supermimic.learners.learner_utils import _optimize_step, build_optimizers
from supermimic.models import build_mlp


def _ids(params) -> list[int]:
  return [id(p) for p in params]


def test_muon_steps_the_hidden_matrices_and_adam_the_rest():
  torch.manual_seed(0)
  mlp = build_mlp(5, (8, 8, 8), 3)
  linears = [m for m in mlp if isinstance(m, nn.Linear)]
  hidden = [linears[1].weight, linears[2].weight]
  adam, muon = build_optimizers([mlp], 1e-3)
  assert isinstance(muon, torch.optim.Muon)
  assert _ids(muon.param_groups[0]["params"]) == _ids(hidden)
  rest = [p for p in mlp.parameters() if id(p) not in _ids(hidden)]
  assert _ids(adam.param_groups[0]["params"]) == _ids(rest)
  before = [p.detach().clone() for p in mlp.parameters()]
  _optimize_step([adam, muon], mlp(torch.randn(4, 5)).square().mean())
  for param, old in zip(mlp.parameters(), before, strict=True):
    assert not torch.equal(param, old)
