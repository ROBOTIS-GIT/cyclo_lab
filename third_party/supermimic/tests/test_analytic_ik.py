"""Analytic IK on the G1: forward-kinematics consistency, the branch and turn
picked by joint limits, gimbal safety for BPTT, and the encoder bias in the
synthesized sensor."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from supermimic.representations import MaxCoordRepresentation
from supermimic.sim_context import (
  CoMOffsetSpec,
  EncoderBiasSpec,
  FootFrictionSpec,
  SimContext,
)
from supermimic.tasks.env_cfgs import maxcoord_env_cfg
from supermimic.utils.analytic_ik import AnalyticIK
from supermimic.utils.helpers import rot6d_to_matrix
from tests.conftest import identity_aux, make_mock_sim_context

WRIST_PITCH = ("left_wrist_pitch_joint", "right_wrist_pitch_joint")
WRIST_OUTER = (
  "left_wrist_roll_joint",
  "left_wrist_yaw_joint",
  "right_wrist_roll_joint",
  "right_wrist_yaw_joint",
)


class G1:
  """The G1's hinge tables in ``qj`` column order, and MuJoCo FK over the
  bodies the tracking task tracks."""

  def __init__(self) -> None:
    self.model = mujoco.MjModel.from_xml_path(str(G1_XML))
    hinge = mujoco.mjtJoint.mjJNT_HINGE
    joints = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == hinge]
    self.names = [self.model.joint(j).name for j in joints]
    self.q_adr = [int(self.model.jnt_qposadr[j]) for j in joints]
    self.dof_adr = [int(self.model.jnt_dofadr[j]) for j in joints]
    self.lo, self.hi = self.model.jnt_range[joints].T
    self.cfg = maxcoord_env_cfg()
    factor = self.cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor
    mid, half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
    soft = np.stack([mid - factor * half, mid + factor * half], -1)
    self.soft_limits = torch.tensor(soft, dtype=torch.float32)
    body_names = self.cfg.commands["motion"].body_names
    self.body_ids = [self.model.body(n).id for n in body_names]
    self.num_bodies, self.num_joints = len(self.body_ids), len(joints)

  def ik(self, dtype: torch.dtype = torch.float64) -> AnalyticIK:
    return AnalyticIK(
      self.model,
      torch.arange(self.model.nbody),
      torch.tensor(self.q_adr),
      torch.tensor(self.body_ids),
      "cpu",
      dtype,
    )

  def cols(self, *names: str) -> list[int]:
    return [self.names.index(n) for n in names]

  def sample(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = self.lo + (self.hi - self.lo) * rng.random((n, self.num_joints))
    return q, rng.normal(size=(n, self.num_joints)) * 3.0

  def fk(self, q: np.ndarray, qd: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracked-body rotations ``(n, B, 3, 3)`` and angular velocities ``(n, B, 3)``."""
    data = mujoco.MjData(self.model)
    rots = np.zeros((len(q), self.num_bodies, 3, 3))
    avels = np.zeros((len(q), self.num_bodies, 3))
    for i in range(len(q)):
      data.qpos[self.q_adr], data.qvel[self.dof_adr] = q[i], qd[i]
      mujoco.mj_forward(self.model, data)
      for row, body in enumerate(self.body_ids):
        mujoco.mju_quat2Mat(rots[i, row].reshape(9), data.xquat[body])
      avels[i] = data.cvel[self.body_ids, :3]
    return torch.from_numpy(rots), torch.from_numpy(avels)


@pytest.fixture(scope="module")
def g1() -> G1:
  return G1()


def _rot6d(rots: torch.Tensor) -> torch.Tensor:
  return rots[..., :, :2].flatten(-2)


def _rep_state(rots: torch.Tensor, avels: torch.Tensor) -> torch.Tensor:
  """A maxcoord rep state carrying what the solve reads: the rotation and
  angular-velocity blocks of ``[pos, rot6d, vel, avel, height, up]``."""
  n, b = rots.shape[:2]
  zeros = torch.zeros(n, b * 3, dtype=torch.float64)
  blocks = [zeros, _rot6d(rots).flatten(1), zeros, avels.flatten(1), zeros[:, :4]]
  return torch.cat(blocks, dim=-1).float()


def _bound(
  g1: G1, sim_context: SimContext, default_joint_pos: torch.Tensor
) -> MaxCoordRepresentation:
  """A proprio representation bound through the env lookups ``bind`` makes."""
  indexing = SimpleNamespace(
    body_ids=torch.arange(g1.model.nbody), joint_q_adr=torch.tensor(g1.q_adr)
  )
  motion = SimpleNamespace(body_indexes=torch.tensor(g1.body_ids))
  data = SimpleNamespace(soft_joint_pos_limits=g1.soft_limits[None])
  env = SimpleNamespace(
    scene={"robot": SimpleNamespace(indexing=indexing, data=data)},
    sim=SimpleNamespace(mj_model=g1.model),
    command_manager=SimpleNamespace(get_term=lambda _: motion),
  )
  rep = MaxCoordRepresentation(g1.num_bodies, default_joint_pos)
  rep.actor_obs_mode = "proprio"
  rep.bind(env, sim_context)
  return rep


def test_matches_forward_kinematics(g1: G1) -> None:
  q, qd = g1.sample(64, seed=0)
  rots, avels = g1.fk(q, qd)
  # Within cos^2(pitch) < damp of the wrist gimbal the outer rates saturate,
  # and float32 loses the pitch itself to asin' ~ 1/cos.
  pitch = np.abs(q[:, g1.cols(*WRIST_PITCH)])
  clear = torch.from_numpy((pitch < math.pi / 2 - 0.1).all(axis=1))
  for dtype, tol in ((torch.float64, 1e-9), (torch.float32, 2e-6)):
    qp, qv = g1.ik(dtype)(rots.to(dtype), avels.to(dtype))
    assert (qp.double() - torch.from_numpy(q))[clear].abs().max() < tol
    assert (qv.double() - torch.from_numpy(qd))[clear].abs().max() < 10 * tol


def test_limits_pick_the_branch_and_the_turn(g1: G1) -> None:
  """Shoulder pitch swings past -pi and wrist pitch past pi/2: the encoder
  reports each angle inside its joint's range, so must the solve."""
  q, qd = g1.sample(1, seed=1)
  q[0, g1.cols("left_shoulder_pitch_joint", "right_shoulder_pitch_joint")] = -3.15
  for side, flip in (("left", 1.0), ("right", -1.0)):
    wrist = g1.cols(*(f"{side}_wrist_{j}_joint" for j in ("roll", "pitch", "yaw")))
    q[0, wrist] = flip * np.array([0.3, 1.6, -0.4])
  qp, _ = g1.ik()(*g1.fk(q, qd))
  assert torch.allclose(qp[0], torch.from_numpy(q[0]), atol=1e-9)


def _backward_is_bounded(ik: AnalyticIK, rots: torch.Tensor, avels: torch.Tensor):
  """The training path (rot6d -> matrix -> solve) is finite, its gradients bounded."""
  rot6d = _rot6d(rots).clone().requires_grad_()
  avels = avels.clone().requires_grad_()
  qp, qv = ik(rot6d_to_matrix(rot6d), avels)
  assert torch.isfinite(qp).all() and torch.isfinite(qv).all()
  (qp.sum() + qv.sum()).backward()
  assert rot6d.grad.abs().max() < 1e4 and avels.grad.abs().max() < 1e4


def test_gimbal_backward_is_finite_and_bounded(g1: G1) -> None:
  """Wrist pitch at +-pi/2 is a true gimbal: the remaining solve inputs vanish
  and the naive backward hits asin'(+-1), 1/cos^2 and atan2(0, 0). FK poses
  approach it; a world-model prediction can round, in fp32, to the exact
  singular bit pattern."""
  q, qd = g1.sample(4, seed=2)
  half = math.pi / 2
  q[:, g1.cols(*WRIST_PITCH)] = np.array(
    [[half], [-half], [half - 1e-4], [half - 1e-7]]
  )
  rots, avels = g1.fk(q, qd)
  for dtype in (torch.float32, torch.float64):
    _backward_is_bounded(g1.ik(dtype), rots.to(dtype), avels.to(dtype))

  ik = g1.ik(torch.float32)
  r_y_half = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
  rots = torch.eye(3).repeat(1, g1.num_bodies, 1, 1)
  for c in ik.three.nonzero()[:, 0]:
    rots[0, ik.child[c]] = ik.A[c] @ r_y_half @ ik.B[c].T
  _backward_is_bounded(ik, rots, torch.randn(1, g1.num_bodies, 3))


def test_outer_wrist_rates_saturate_at_the_gimbal(g1: G1) -> None:
  """Rotations and angular velocities are independent world-model outputs, so
  nothing makes a predicted pair consistent. Near the gimbal the part of the
  rate the rotation cannot explain is divided by cos^2(pitch): the outer
  pair saturates under damping, while the pitch rate stays exact."""
  q, qd = g1.sample(16, seed=3)
  q[:, g1.cols(*WRIST_OUTER)] = 0.5  # keeps the twin branch out of range
  q[:, g1.cols(*WRIST_PITCH)] = math.pi / 2 - 1e-3
  rots, avels = g1.fk(q, qd)
  gen = torch.Generator().manual_seed(0)
  free = torch.randn(avels.shape, generator=gen, dtype=torch.float64) * 3.0
  scale = free.norm(dim=-1).max()
  outer, pitch = g1.cols(*WRIST_OUTER), g1.cols(*WRIST_PITCH)
  _, undamped = g1.ik()(rots, free, damp=1e-6)
  _, damped = g1.ik()(rots, free)
  assert undamped[:, outer].abs().max() > 100 * scale
  assert damped[:, outer].abs().max() < 11 * scale
  _, consistent = g1.ik()(rots, avels)
  assert torch.allclose(consistent[:, pitch], torch.from_numpy(qd[:, pitch]), atol=1e-9)


def test_sensor_carries_the_encoder_bias(g1: G1) -> None:
  """The env's context lays out foot friction and CoM offset ahead of the
  encoder bias; the sensor's joint_pos block reads relative to the default
  pose, and the bias reaches that block and nothing else."""
  n, j = 8, g1.num_joints
  specs = [FootFrictionSpec(), CoMOffsetSpec(), EncoderBiasSpec()]
  specs[2].dim = j
  ctx = SimContext(specs, [(-1.0, 1.0)] * 4 + [(-0.01, 0.01)] * j)
  bias = torch.empty(n, j).uniform_(-0.01, 0.01)
  encoded = (torch.cat([torch.ones(n, 4), bias], dim=-1) - ctx.mean) / ctx.scale

  q, qd = g1.sample(n, seed=4)
  default = torch.linspace(-0.3, 0.3, j)
  rep_state, aux = _rep_state(*g1.fk(q, qd)), identity_aux(n)
  biased_rep = _bound(g1, ctx, default)
  plain_rep = _bound(g1, make_mock_sim_context(), default)
  joint_state = plain_rep.joint_state(rep_state)
  biased = biased_rep.compute_wm_obs(rep_state, aux, joint_state, encoded)
  plain = plain_rep.compute_wm_obs(rep_state, aux, joint_state, encoded[:, :1])
  joint_pos = slice(6, 6 + j)  # after projected_gravity(3) and base_ang_vel(3)
  assert torch.allclose(
    plain[:, joint_pos], torch.from_numpy(q).float() - default, atol=1e-5
  )
  delta = biased - plain
  assert torch.allclose(delta[:, joint_pos], bias, atol=1e-6)
  delta[:, joint_pos] = 0.0
  assert torch.equal(delta, torch.zeros_like(delta))


def test_joint_state_is_what_the_sensor_reports(g1: G1) -> None:
  """The joint state is the solve over the rep state's rotation and
  angular-velocity blocks; the sensor carries it relative to the default pose."""
  n, j = 8, g1.num_joints
  q, qd = g1.sample(n, seed=4)
  q[:, g1.cols(*WRIST_PITCH)] *= 0.5  # clear of the gimbal, where rates are damped
  default = torch.linspace(-0.3, 0.3, j)
  rep = _bound(g1, make_mock_sim_context(), default)
  rep_state = _rep_state(*g1.fk(q, qd))
  joint_state = rep.joint_state(rep_state)
  assert torch.allclose(joint_state[:, :j], torch.from_numpy(q).float(), atol=1e-5)
  assert torch.allclose(joint_state[:, j:], torch.from_numpy(qd).float(), atol=1e-4)
  sim_ctx = torch.zeros(n, 1)
  obs = rep.compute_wm_obs(rep_state, identity_aux(n), joint_state, sim_ctx)
  assert torch.equal(obs[:, 6 : 6 + j], joint_state[:, :j] - default)
  assert torch.equal(obs[:, 6 + j :], joint_state[:, j:])
