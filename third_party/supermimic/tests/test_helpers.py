"""Tests for supermimic.utils.helpers — rotation and utility functions."""

import torch
from mjlab.utils.lab_api.math import quat_apply

from supermimic.utils.helpers import (
  quat_integrate,
  quat_to_rot6d,
  rodrigues_rotate,
)

IDENTITY_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _random_unit_quats(n: int) -> torch.Tensor:
  q = torch.randn(n, 4)
  return q / q.norm(dim=-1, keepdim=True)


class TestQuatToRot6d:
  def test_identity_encoding(self):
    """Identity quaternion should produce identity rot6d (first two columns of I)."""
    r6d = quat_to_rot6d(IDENTITY_QUAT)
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert torch.allclose(r6d, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# rodrigues_rotate
# ---------------------------------------------------------------------------


class TestRodriguesRotate:
  def test_zero_rotation(self):
    """Zero omega should leave vectors unchanged."""
    v = torch.randn(8, 3)
    omega = torch.zeros(8, 3)
    result = rodrigues_rotate(v, omega)
    assert torch.allclose(result, v, atol=1e-4)

  def test_preserves_length(self):
    """Rotation should preserve vector length."""
    v = torch.randn(16, 3)
    omega = torch.randn(16, 3)
    result = rodrigues_rotate(v, omega)
    assert torch.allclose(result.norm(dim=-1), v.norm(dim=-1), atol=1e-4)

  def test_90_degree_rotation(self):
    """90° around Z should map [1,0,0] to [0,1,0]."""
    import math

    v = torch.tensor([1.0, 0.0, 0.0])
    omega = torch.tensor([0.0, 0.0, math.pi / 2])
    result = rodrigues_rotate(v, omega)
    expected = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(result, expected, atol=1e-4)

  def test_matches_quat_apply(self):
    """Rodrigues rotation should match quaternion rotation."""
    q = _random_unit_quats(16)
    v = torch.randn(16, 3)
    # Use quat_apply as ground truth
    expected = quat_apply(q, v)
    # Convert quat to axis-angle for rodrigues
    angle = torch.acos(torch.clamp(q[:, 0:1], -1, 1)) * 2
    axis = q[:, 1:] / (torch.sin(angle / 2) + 1e-8)
    omega = axis * angle
    result = rodrigues_rotate(v, omega)
    assert torch.allclose(result, expected, atol=1e-3)

  def test_batch_shape(self):
    v = torch.randn(4, 16, 3)
    omega = torch.randn(4, 16, 3)
    result = rodrigues_rotate(v, omega)
    assert result.shape == (4, 16, 3)

  def test_gradient_at_zero_omega(self):
    v = torch.randn(3)
    omega = torch.zeros(3, requires_grad=True)
    result = rodrigues_rotate(v, omega)
    result.sum().backward()
    assert not omega.grad.isnan().any()
    assert not omega.grad.isinf().any()


# ---------------------------------------------------------------------------
# quat_integrate
# ---------------------------------------------------------------------------


class TestQuatIntegrate:
  def test_zero_angular_velocity(self):
    q = IDENTITY_QUAT.clone()
    w = torch.zeros(3)
    q_next = quat_integrate(q, w, dt=0.01)
    assert torch.allclose(q_next, IDENTITY_QUAT, atol=1e-5)

  def test_result_is_unit_quaternion(self):
    q = _random_unit_quats(16)
    w = torch.randn(16, 3)
    q_next = quat_integrate(q, w, dt=0.01)
    assert torch.allclose(q_next.norm(dim=-1), torch.ones(16), atol=1e-5)

  def test_gradient_at_zero_angular_velocity(self):
    q = IDENTITY_QUAT.clone()
    w = torch.zeros(3, requires_grad=True)
    q_next = quat_integrate(q, w, dt=0.01)
    q_next.sum().backward()
    assert not w.grad.isnan().any()
    assert not w.grad.isinf().any()

  def test_small_rotation_matches_first_order(self):
    """For small dt, quat_integrate should approximate first-order update."""
    q = _random_unit_quats(8)
    w = torch.randn(8, 3) * 0.1  # small angular velocity
    dt = 0.001
    q_next = quat_integrate(q, w, dt)
    # First-order: q_next ≈ q + 0.5 * dt * [0, w] * q
    # Just check it's close to q (small step)
    cos_angle = (q * q_next).sum(dim=-1).abs()
    assert torch.allclose(cos_angle, torch.ones(8), atol=1e-3)


def test_term_major_to_temporal_two_terms():
  """mjlab term-major flat → frame-major window: each term's H frames unpacked.

  Pins the deploy-boundary layout directly — a round-trip could hide a
  compensating bug in both directions.
  """
  from supermimic.representations import term_major_to_temporal

  H, w_a, w_b = 3, 2, 4
  flat = torch.tensor(
    [
      # term_a: t0, t1, t2     | term_b: t0,          t1,          t2
      [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
      + [100.0, 101.0, 102.0, 103.0]
      + [110.0, 111.0, 112.0, 113.0]
      + [120.0, 121.0, 122.0, 123.0]
    ]
  )
  expected = torch.tensor(
    [
      [
        [10.0, 11.0, 100.0, 101.0, 102.0, 103.0],
        [12.0, 13.0, 110.0, 111.0, 112.0, 113.0],
        [14.0, 15.0, 120.0, 121.0, 122.0, 123.0],
      ]
    ]
  )
  window = term_major_to_temporal(flat, [w_a, w_b], H)
  assert window.shape == (1, H, w_a + w_b)
  assert torch.equal(window, expected)

  # H=1: the flat is already one frame, gaining a length-1 time axis.
  one = torch.randn(4, w_a + w_b)
  assert torch.equal(term_major_to_temporal(one, [w_a, w_b], 1), one.unsqueeze(1))
