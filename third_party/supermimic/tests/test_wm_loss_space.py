"""Contract tests for the world-model state loss, over both representations.

``loss_blocks`` is the single declaration of what the loss scores, so these pin
what each named block is built from and that its error reaches the dynamics.
"""

import dataclasses

import pytest
import torch

from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics
from supermimic.representations import create_representation

NUM_JOINTS = 20
NUM_BODIES = 16
NUM_ACTIONS = 4


def _randn(*shape, seed: int = 0) -> torch.Tensor:
  return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _random_full_states(rep, batch: int, horizon: int) -> torch.Tensor:
  """Full states whose root quaternion is a unit quaternion, as the sim emits."""
  states = _randn(batch, horizon, rep.full_state_dim, seed=3)
  quat_at = 3 if rep.full_state_dim == 13 + 2 * NUM_JOINTS else NUM_BODIES * 3
  quats = _random_unit_quats(batch * horizon, seed=4).reshape(batch, horizon, 4)
  states[..., quat_at : quat_at + 4] = quats
  return states


def _make_representation(name: str):
  full_state_dim = 13 + 2 * NUM_JOINTS if name == "gencoord" else NUM_BODIES * 13
  return create_representation(
    name,
    full_state_dim,
    NUM_ACTIONS,
    dt=0.02,
    decimation=4,
    default_joint_pos=torch.zeros(NUM_JOINTS),
    actor_obs_mode="rep_state",
  )


def _make_learner(name: str, **cfg_kwargs):
  rep = _make_representation(name)
  dynamics = Dynamics(
    rep, NUM_ACTIONS, 1, torch.zeros(NUM_ACTIONS), torch.ones(NUM_ACTIONS),
    hidden_dims=(16, 16),
  )  # fmt: skip
  cfg = WorldModelLearnerCfg(**cfg_kwargs)
  learner = WorldModelLearner(dynamics, rep, cfg, device="cpu")
  return learner, rep, cfg


def _random_unit_quats(n: int, seed: int) -> torch.Tensor:
  generator = torch.Generator().manual_seed(seed)
  quats = torch.randn(n, 4, generator=generator)
  return quats / quats.norm(dim=-1, keepdim=True)


def _random_aux_states(n: int, seed: int = 0) -> torch.Tensor:
  generator = torch.Generator().manual_seed(seed)
  return torch.cat(
    [torch.randn(n, 3, generator=generator), _random_unit_quats(n, seed)], dim=-1
  )


def _changed_blocks(new: dict[str, torch.Tensor], base: dict[str, torch.Tensor]):
  return {name for name, block in new.items() if not torch.equal(block, base[name])}


def _rep_state_sources(name: str) -> tuple[dict[str, slice], dict[str, slice]]:
  """rep_state slices inside and outside the loss space."""
  if name == "gencoord":
    # [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
    n = NUM_JOINTS
    scored = {"joint_pos": slice(10, 10 + n), "joint_vel": slice(10 + n, 10 + 2 * n)}
    excluded = {"height_gravity_twist": slice(0, 10)}
  else:
    # [local_pos(B*3), rot6d(B*6), vel(B*3), avel(B*3), root_height(1), up_dir(3)]
    b = NUM_BODIES
    scored = {
      "body_pos": slice(3, b * 3),
      "body_rot": slice(b * 3 + 6, b * 9),
      "body_vel": slice(b * 9 + 3, b * 12),
      "body_avel": slice(b * 12 + 3, b * 15),
    }
    excluded = {
      "root_pos": slice(0, 3),
      "root_rot": slice(b * 3, b * 3 + 6),
      "root_vel": slice(b * 9, b * 9 + 3),
      "root_avel": slice(b * 12, b * 12 + 3),
      "root_height": slice(b * 15, b * 15 + 1),
      "up_dir": slice(b * 15 + 1, b * 15 + 4),
    }
  return scored, excluded


# Written out rather than read back from the class, so a rewired entry fails.
EXPECTED_WEIGHT_ATTRS = {
  "gencoord": {
    "root_pos": "nominal_error_root_pose_pos",
    "root_ori": "nominal_error_root_pose_ori",
    "joint_pos": "nominal_error_joint_pos",
    "joint_vel": "nominal_error_joint_vel",
  },
  "maxcoord": {
    "root_pos": "nominal_error_root_pose_pos",
    "root_ori": "nominal_error_root_pose_ori",
    "body_pos": "nominal_error_body_pos",
    "body_rot": "nominal_error_body_rot",
    "body_vel": "nominal_error_body_vel",
    "body_avel": "nominal_error_body_avel",
  },
}


@pytest.mark.parametrize("name", ["gencoord", "maxcoord"])
class TestLossBlocks:
  def test_block_names_match_weight_config(self, name):
    rep = _make_representation(name)
    state = _randn(4, rep.rep_state_dim)
    blocks = rep.loss_blocks(state, _random_aux_states(4))
    assert set(blocks) == set(rep.LOSS_WEIGHT_CONFIG)
    assert rep.LOSS_WEIGHT_CONFIG == EXPECTED_WEIGHT_ATTRS[name]
    fields = {f.name for f in dataclasses.fields(WorldModelLearnerCfg)}
    for block_name, attr in rep.LOSS_WEIGHT_CONFIG.items():
      assert attr in fields, f"{block_name} -> {attr} is not a cfg field"

  def test_named_block_traces_to_its_source(self, name):
    """Perturbing one physical quantity moves exactly its named block."""
    rep = _make_representation(name)
    state = _randn(4, rep.rep_state_dim)
    aux = _random_aux_states(4)
    base = rep.loss_blocks(state, aux)

    moved = aux.clone()
    moved[:, :3] += 0.1
    assert _changed_blocks(rep.loss_blocks(state, moved), base) == {"root_pos"}
    assert torch.equal(base["root_pos"], aux[:, :3])

    turned = aux.clone()
    turned[:, 3:7] = _random_unit_quats(4, seed=99)
    assert _changed_blocks(rep.loss_blocks(state, turned), base) == {"root_ori"}

    scored, _ = _rep_state_sources(name)
    for block_name, source in scored.items():
      nudged = state.clone()
      nudged[:, source] += 0.1
      assert _changed_blocks(rep.loss_blocks(nudged, aux), base) == {block_name}
      assert torch.equal(base[block_name], state[:, source])

  def test_excluded_channels_do_not_leak(self, name):
    rep = _make_representation(name)
    state = _randn(4, rep.rep_state_dim)
    aux = _random_aux_states(4)
    base = rep.loss_blocks(state, aux)
    _, excluded = _rep_state_sources(name)
    for block_name, source in excluded.items():
      nudged = state.clone()
      nudged[:, source] += 0.1
      assert _changed_blocks(rep.loss_blocks(nudged, aux), base) == set(), block_name

  def test_state_loss_matches_closed_form(self, name):
    learner, rep, cfg = _make_learner(name)
    rep_states = torch.zeros(1, 1, rep.rep_state_dim)
    gt_aux = torch.zeros(1, 1, 7)
    gt_aux[..., 3] = 1.0
    theta = torch.tensor(0.3)
    pred_aux = gt_aux.clone()
    pred_aux[..., 0] = 0.02  # 2cm of x error
    pred_aux[..., 3] = torch.cos(theta / 2)
    pred_aux[..., 6] = torch.sin(theta / 2)  # yaw by theta

    loss_state, _ = learner._compute_losses(rep_states, pred_aux, rep_states, gt_aux)
    # 0.5*||R(theta) - I||_F^2 = 2(1 - cos theta)
    width = sum(b.shape[-1] for b in rep.loss_blocks(rep_states, gt_aux).values())
    expected = (
      1.0 / cfg.nominal_error_root_pose_pos**2 * 0.02**2
      + 1.0 / cfg.nominal_error_root_pose_ori**2 * 2.0 * (1.0 - torch.cos(theta))
    ) / width
    assert torch.isclose(loss_state, expected, rtol=1e-5)

  def test_every_block_uses_its_own_nominal(self, name):
    """One block wrong at a time: the loss must scale by that block's nominal."""
    learner, rep, cfg = _make_learner(name)
    aux = torch.zeros(1, 1, 7)
    aux[..., 3] = 1.0
    base_state = torch.zeros(1, 1, rep.rep_state_dim)
    blocks = rep.loss_blocks(base_state, aux)
    width = sum(b.shape[-1] for b in blocks.values())
    scored, _ = _rep_state_sources(name)
    for block_name, source in scored.items():
      pred_state = base_state.clone()
      pred_state[..., source] = 0.03
      loss_state, _ = learner._compute_losses(pred_state, aux, base_state, aux)
      nominal = getattr(cfg, EXPECTED_WEIGHT_ATTRS[name][block_name])
      expected = 0.03**2 / nominal**2 * blocks[block_name].shape[-1] / width
      assert torch.isclose(loss_state, torch.tensor(expected), rtol=1e-5), block_name

  def test_root_pose_error_backprops_through_the_rollout(self, name):
    """Scoring only the last step must reach the first, or BPTT through the
    aux chain is severed."""
    learner, rep, cfg = _make_learner(name, horizon=4)
    batch, horizon = 2, cfg.horizon
    full_states = _random_full_states(rep, batch, horizon)
    init_aux = rep.extract_aux_state(full_states[:, 0, :]).requires_grad_(True)
    _, pred_aux_states = learner._rollout(
      rep.represent(full_states)[:, 0, :],
      _randn(batch, horizon, NUM_ACTIONS, seed=1),
      init_aux,
      torch.zeros(batch, 1),
    )
    last = rep.loss_blocks(
      torch.zeros(batch, rep.rep_state_dim), pred_aux_states[:, -1, :]
    )
    torch.cat(list(last.values()), dim=-1).pow(2).mean().backward()
    assert init_aux.grad is not None and float(init_aux.grad.abs().sum()) > 0.0
