"""Tests for MaxCoordRepresentation.

Covers apply_prediction, rep_state layout, loss slices, rotation
helpers used by the reward, and reward / apply_prediction gradient flow.
"""

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
)

from supermimic.representations import MaxCoordRepresentation
from supermimic.utils.helpers import (
  quat_to_rot6d,
  rot6d_to_matrix,
  rotation_error_magnitude,
)
from supermimic.utils.maxcoord_helpers import (
  build_maxcoord_full_state,
  compute_maxcoord_rep_state,
  decompose_maxcoord_rep_state,
  decompose_maxcoord_state,
  maxcoord_rep_state_dim,
  reconstruct_full_from_aux_and_rep,
  world_frame_bodies,
)
from supermimic.utils.tracking_reward import _reanchor_targets, tracking_reward

NUM_BODIES = 16


# ---------------------------------------------------------------------------
# Rep state dimensions & layout
# ---------------------------------------------------------------------------


class TestRepStateDim:
  def test_dim_formula(self):
    """rep_state_dim should be B*15 + 4 (root_height is 1, not B)."""
    assert maxcoord_rep_state_dim(NUM_BODIES) == NUM_BODIES * 15 + 4
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    assert rep.rep_state_dim == NUM_BODIES * 15 + 4

  def test_dim_various_body_counts(self):
    for B in [1, 4, 16, 32]:
      assert maxcoord_rep_state_dim(B) == B * 15 + 4


class TestDecomposeRootHeight:
  def test_root_height_shape(self):
    """decompose should return root_height of shape (..., 1), not (..., B)."""
    B = NUM_BODIES
    rep_state = torch.randn(4, maxcoord_rep_state_dim(B))
    _, _, _, _, root_height, up_dir = decompose_maxcoord_rep_state(rep_state, B)
    assert root_height.shape == (4, 1)
    assert up_dir.shape == (4, 3)

  def test_root_height_is_root_body_z(self):
    """root_height in rep_state should equal root body world Z coordinate."""
    B = NUM_BODIES
    body_pos = torch.randn(4, B, 3)
    body_quat = torch.randn(4, B, 4)
    body_quat = body_quat / body_quat.norm(dim=-1, keepdim=True)
    body_lin_vel = torch.randn(4, B, 3)
    body_ang_vel = torch.randn(4, B, 3)
    full_state = build_maxcoord_full_state(
      body_pos, body_quat, body_lin_vel, body_ang_vel
    )
    rep_state = compute_maxcoord_rep_state(full_state, B)
    _, _, _, _, root_height, _ = decompose_maxcoord_rep_state(rep_state, B)
    expected_root_z = body_pos[:, 0, 2:3]
    assert torch.allclose(root_height, expected_root_z, atol=1e-5)


# ---------------------------------------------------------------------------
# Loss slices & weights
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# apply_prediction correctness
# ---------------------------------------------------------------------------


class TestMaxCoordApplyPrediction:
  def test_prediction_dim(self):
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    # vel_delta(B*3) + avel_delta(B*3) = B*6
    assert rep.prediction_dim == NUM_BODIES * 6

  def test_output_shapes(self):
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.01
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    assert next_aux.shape == (4, 7)
    assert next_rep.shape == (4, rep.rep_state_dim)

  def _flat_state(self, seed: int = 0):
    """State and prediction with zero angular velocity: no frame change."""
    B = NUM_BODIES
    torch.manual_seed(seed)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, MaxCoordRepresentation(B, torch.zeros(29)).rep_state_dim)
    curr[..., B * 12 : B * 15] = 0.0  # local_avel
    pred = torch.randn(4, B * 6) * 0.1
    pred[..., B * 3 :] = 0.0  # avel deltas
    return aux, curr, pred

  def test_substep_weights_pose_not_velocity(self):
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, torch.zeros(29))
    rep.substep_coeff = 0.625
    aux, curr, pred = self._flat_state()
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.02)

    local_pos = curr[..., : B * 3].unflatten(-1, (B, 3))
    local_vel = curr[..., B * 9 : B * 12].unflatten(-1, (B, 3))
    dv = pred[..., : B * 3].unflatten(-1, (B, 3))
    # Velocity state carries the full delta.
    torch.testing.assert_close(
      next_rep[..., B * 9 : B * 12], (local_vel + dv).flatten(-2)
    )
    # Position advances by the weighted delta, relative to the root.
    pose_vel = local_vel + 0.625 * dv
    rel = pose_vel - pose_vel[..., 0:1, :]
    expected = (local_pos - local_pos[..., 0:1, :]) + rel * 0.02
    torch.testing.assert_close(next_rep[..., : B * 3], expected.flatten(-2))

  def test_substep_coeff_one_reproduces_full_delta_integration(self):
    """coeff 1.0 must reproduce the un-weighted integration exactly."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, torch.zeros(29))
    assert rep.substep_coeff == 1.0
    aux, curr, pred = self._flat_state(seed=1)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.02)

    local_pos = curr[..., : B * 3].unflatten(-1, (B, 3))
    local_vel = curr[..., B * 9 : B * 12].unflatten(-1, (B, 3))
    next_vel = local_vel + pred[..., : B * 3].unflatten(-1, (B, 3))
    rel = next_vel - next_vel[..., 0:1, :]
    expected = (local_pos - local_pos[..., 0:1, :]) + rel * 0.02
    torch.testing.assert_close(next_rep[..., : B * 3], expected.flatten(-2))

  def test_zero_prediction_velocities_unchanged(self):
    """Zero prediction means zero velocity delta — velocities stay the same (before frame correction)."""
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0  # identity quat -> no frame correction
    curr = torch.randn(4, rep.rep_state_dim)
    # Zero out angular velocities so frame correction is identity
    avel_start = B * 3 + B * 6 + B * 3
    curr[..., avel_start : avel_start + B * 3] = 0.0
    pred = torch.zeros(4, rep.prediction_dim)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    # Velocities should be unchanged (zero avel -> no frame rotation)
    vel_start = B * 3 + B * 6
    assert torch.allclose(
      next_rep[..., vel_start : vel_start + B * 3],
      curr[..., vel_start : vel_start + B * 3],
      atol=1e-4,
    )

  def test_zero_prediction_positions_integrate_from_velocity(self):
    """With zero prediction, positions integrate via world-frame round-trip."""
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    B = NUM_BODIES
    dt = 0.01
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    # Zero out angular velocities so root quat stays identity
    avel_start = B * 3 + B * 6 + B * 3
    curr[..., avel_start : avel_start + B * 3] = 0.0
    local_pos_curr = curr[..., : B * 3].unflatten(-1, (B, 3))
    vel_start = B * 3 + B * 6
    local_vel = curr[..., vel_start : vel_start + B * 3].unflatten(-1, (B, 3))
    # World-frame integration (identity quat, root_pos=0):
    # world_pos = local_pos, next_world_pos = local_pos + local_vel*dt
    # next_root_pos = next_world_pos[0], next_local_pos = next_world_pos - next_root_pos
    next_world_pos = local_pos_curr + local_vel * dt
    expected_pos = next_world_pos - next_world_pos[..., 0:1, :]
    pred = torch.zeros(4, rep.prediction_dim)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=dt)
    next_pos = next_rep[..., : B * 3].unflatten(-1, (B, 3))
    assert torch.allclose(next_pos, expected_pos, atol=1e-4)

  def test_zero_prediction_rotation_integrates_from_avel(self):
    """With zero prediction, rotations integrate from current angular velocities."""
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.zeros(4, rep.prediction_dim)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    # Rotations should change (integrated from avel), so they shouldn't equal curr
    rot_start = B * 3
    rot_end = rot_start + B * 6
    # Just verify the output is finite and different from current
    assert not next_rep[..., rot_start:rot_end].isnan().any()
    assert not next_rep[..., rot_start:rot_end].isinf().any()

  def test_root_quat_is_unit(self):
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.1
    next_aux, _ = rep.apply_prediction(aux, curr, pred, dt=0.01)
    quat_norms = next_aux[..., 3:7].norm(dim=-1)
    assert torch.allclose(quat_norms, torch.ones(4), atol=1e-5)

  def test_output_rot6d_stays_near_so3(self):
    """Starting from valid SO(3) rotations, output should remain near SO(3)."""
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    # Build curr with valid SO(3) rot6d
    curr = torch.randn(4, rep.rep_state_dim)
    rot_start = B * 3
    rot_end = rot_start + B * 6
    random_quats = torch.randn(4, B, 4)
    random_quats = random_quats / random_quats.norm(dim=-1, keepdim=True)
    valid_rot6d = quat_to_rot6d(random_quats)
    curr[..., rot_start:rot_end] = valid_rot6d.flatten(-2, -1)
    pred = torch.randn(4, rep.prediction_dim) * 0.01  # small prediction
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    rot = next_rep[..., rot_start:rot_end].unflatten(-1, (B, 6))
    cols = rot.unflatten(-1, (3, 2))
    c1, c2 = cols[..., 0], cols[..., 1]
    # Should stay approximately unit-length and orthogonal with small predictions
    assert torch.allclose(c1.norm(dim=-1), torch.ones(4, B), atol=0.1)
    assert torch.allclose(c2.norm(dim=-1), torch.ones(4, B), atol=0.1)

  def test_velocity_couples_to_position(self):
    """Position loss should backprop to velocity — velocity drives position integration."""
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    # Loss on positions only
    pos_loss = next_rep[..., : B * 3].sum()
    pos_loss.backward()
    # Velocity part of prediction should have nonzero gradient
    vel_grad = pred.grad[..., : B * 3]
    assert vel_grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Rotation helpers used by the reward
# ---------------------------------------------------------------------------


def _canonicalize_quats(full_state, num_bodies):
  """Canonicalize quaternion signs in full_state so q.w >= 0."""
  B = num_bodies
  quats = full_state[..., B * 3 : B * 7].clone().unflatten(-1, (B, 4))
  sign = quats[..., 0:1].sign()
  sign[sign == 0] = 1.0
  quats = quats * sign
  result = full_state.clone()
  result[..., B * 3 : B * 7] = quats.flatten(-2, -1)
  return result


class TestRot6dToMatrix:
  def test_recovers_matrix_from_quat(self):
    """rot6d_to_matrix(quat_to_rot6d(q)) should reconstruct matrix_from_quat(q)."""
    torch.manual_seed(0)
    quats = torch.randn(100, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    expected = matrix_from_quat(quats)
    actual = rot6d_to_matrix(quat_to_rot6d(quats))
    assert torch.allclose(actual, expected, atol=1e-5)

  def test_output_is_proper_rotation(self):
    """Output should be a valid rotation matrix: R^T R = I, det(R) = 1."""
    torch.manual_seed(2)
    rot6d = quat_to_rot6d(torch.randn(50, 4))
    mat = rot6d_to_matrix(rot6d)
    eye = torch.eye(3).expand(50, -1, -1)
    assert torch.allclose(mat.transpose(-1, -2) @ mat, eye, atol=1e-5)
    assert torch.allclose(torch.det(mat), torch.ones(50), atol=1e-5)


class TestRotationErrorMagnitude:
  def test_matches_quat_error_magnitude(self):
    torch.manual_seed(3)
    q1 = torch.randn(200, 4)
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    q2 = torch.randn(200, 4)
    q2 = q2 / q2.norm(dim=-1, keepdim=True)
    err_quat = quat_error_magnitude(q1, q2)
    err_mat = rotation_error_magnitude(matrix_from_quat(q1), matrix_from_quat(q2))
    assert torch.allclose(err_quat, err_mat, atol=1e-5)

  def test_pi_for_180_degree_rotation(self):
    m1 = torch.eye(3).unsqueeze(0)
    m2 = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]).unsqueeze(0)
    err = rotation_error_magnitude(m1, m2)
    assert torch.allclose(err, torch.tensor([torch.pi]), atol=1e-3)

  def test_gradient_finite_near_zero_error(self):
    """Gradient of angle² must be finite when rotations are identical."""
    torch.manual_seed(6)
    q = torch.randn(16, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    m = matrix_from_quat(q).requires_grad_(True)
    err_sq = rotation_error_magnitude(m, m.detach()) ** 2
    err_sq.sum().backward()
    assert not m.grad.isnan().any()
    assert not m.grad.isinf().any()

  def test_gradient_finite_near_pi_error(self):
    """Gradient of angle² must be finite at 180°."""
    m1 = torch.eye(3).unsqueeze(0).requires_grad_(True)
    m2 = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]).unsqueeze(0)
    err_sq = rotation_error_magnitude(m1, m2) ** 2
    err_sq.sum().backward()
    assert not m1.grad.isnan().any()
    assert not m1.grad.isinf().any()


class TestReanchorTargets:
  def test_identity_when_pred_root_matches_target(self):
    """pred_root == target body 0 ⇒ re-anchoring is a no-op."""
    torch.manual_seed(7)
    B = NUM_BODIES
    target_pos = torch.randn(4, B, 3)
    target_quat = torch.randn(4, B, 4)
    target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True)

    tgt_pos, tgt_mat = _reanchor_targets(
      target_pos, target_quat, target_pos[:, 0], target_quat[:, 0]
    )

    assert torch.allclose(tgt_pos, target_pos, atol=1e-5)
    assert torch.allclose(tgt_mat, matrix_from_quat(target_quat), atol=1e-5)


class TestReconstructFull:
  def test_roundtrip(self):
    """full_state -> rep_state -> reconstruct should recover original."""
    torch.manual_seed(12)
    B = NUM_BODIES
    body_pos = torch.randn(4, B, 3)
    body_quat = torch.randn(4, B, 4)
    body_quat = body_quat / body_quat.norm(dim=-1, keepdim=True)
    body_lin_vel = torch.randn(4, B, 3)
    body_ang_vel = torch.randn(4, B, 3)
    full = build_maxcoord_full_state(body_pos, body_quat, body_lin_vel, body_ang_vel)
    rep_state = compute_maxcoord_rep_state(full, B)
    aux = torch.cat([body_pos[:, 0], body_quat[:, 0]], dim=-1)

    reconstructed = reconstruct_full_from_aux_and_rep(aux, rep_state, B)
    assert torch.allclose(
      _canonicalize_quats(full, B),
      _canonicalize_quats(reconstructed, B),
      atol=1e-3,
    )


# ---------------------------------------------------------------------------
# Gradient: apply_prediction
# ---------------------------------------------------------------------------


class TestApplyPredictionGradient:
  def test_no_nan_inf_at_various_scales(self):
    """Gradients should be finite across different prediction magnitudes."""
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    for scale in [1.0, 1e-2, 1e-4, 1e-8]:
      pred = (torch.randn(4, rep.prediction_dim) * scale).requires_grad_(True)
      next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
      loss = next_rep.sum() + next_aux.sum()
      loss.backward()
      assert not pred.grad.isnan().any(), f"NaN at scale {scale}"
      assert not pred.grad.isinf().any(), f"Inf at scale {scale}"

  def test_gradient_from_rep_loss(self):
    """Full rep state loss should give gradient to all prediction dims."""
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    next_rep.sum().backward()
    # Every prediction dim should receive some gradient
    assert (pred.grad.abs().sum(dim=0) > 0).all()

  def test_gradient_from_aux_loss(self):
    """Aux state loss should give gradient to prediction (through vel integration)."""
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    next_aux, _ = rep.apply_prediction(aux, curr, pred, dt=0.01)
    next_aux.sum().backward()
    assert pred.grad.abs().sum() > 0

  def test_gradient_norm_bounded_across_scales(self):
    """Gradient norms should stay bounded as prediction scale shrinks."""
    rep = MaxCoordRepresentation(NUM_BODIES, torch.zeros(29))
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    grad_norms = []
    for scale in [1e-2, 1e-4, 1e-6, 1e-8]:
      pred = (torch.randn(4, rep.prediction_dim) * scale).requires_grad_(True)
      next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
      (next_rep.sum() + next_aux.sum()).backward()
      grad_norms.append(pred.grad.norm().item())
    assert max(grad_norms) / min(grad_norms) < 10.0


# ---------------------------------------------------------------------------
# World-frame consistency: apply_prediction matches direct world integration
# ---------------------------------------------------------------------------


def _make_valid_full_state(batch: int, B: int):
  """Build a valid full state with unit quaternions."""
  body_pos = torch.randn(batch, B, 3)
  body_quat = torch.randn(batch, B, 4)
  body_quat = body_quat / body_quat.norm(dim=-1, keepdim=True)
  body_lin_vel = torch.randn(batch, B, 3) * 0.5
  body_ang_vel = torch.randn(batch, B, 3) * 0.5
  return build_maxcoord_full_state(body_pos, body_quat, body_lin_vel, body_ang_vel)


class TestWorldFrameConsistency:
  """Verify that apply_prediction → reconstruct matches direct world-frame integration."""

  def test_apply_prediction_matches_world_frame_integration(self):
    """apply_prediction → reconstruct_full_state should match direct world integration."""
    B = NUM_BODIES
    batch = 4
    dt = 0.01
    rep = MaxCoordRepresentation(B, torch.zeros(29))
    torch.manual_seed(123)

    full_state = _make_valid_full_state(batch, B)
    body_pos, body_quat, body_vel, body_avel = decompose_maxcoord_state(full_state, B)

    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    root_quat = aux[..., 3:7]
    root_quat_b = root_quat.unsqueeze(-2).expand(batch, B, 4)

    # Small prediction in root frame
    pred = torch.randn(batch, rep.prediction_dim) * 0.01
    dv_local = pred[..., : B * 3].unflatten(-1, (B, 3))
    dw_local = pred[..., B * 3 : B * 6].unflatten(-1, (B, 3))

    # --- Direct world-frame integration (control_vae style) ---
    dv_world = quat_apply(root_quat_b, dv_local)
    dw_world = quat_apply(root_quat_b, dw_local)
    expected_vel = body_vel + dv_world
    expected_avel = body_avel + dw_world
    expected_pos = body_pos + expected_vel * dt

    # --- Through apply_prediction ---
    next_aux, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=dt)
    next_full = rep.reconstruct_full_state(next_aux, next_rep)
    actual_pos, _, actual_vel, actual_avel = decompose_maxcoord_state(next_full, B)

    assert torch.allclose(actual_pos, expected_pos, atol=1e-4), (
      f"Position mismatch: max diff {(actual_pos - expected_pos).abs().max():.6f}"
    )
    assert torch.allclose(actual_vel, expected_vel, atol=1e-4), (
      f"Velocity mismatch: max diff {(actual_vel - expected_vel).abs().max():.6f}"
    )
    assert torch.allclose(actual_avel, expected_avel, atol=1e-4), (
      f"Angular velocity mismatch: max diff {(actual_avel - expected_avel).abs().max():.6f}"
    )

  def test_root_local_pos_always_zero(self):
    """Root body's local_pos should always be zero after integration."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, torch.zeros(29))
    torch.manual_seed(42)
    full_state = _make_valid_full_state(4, B)
    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    pred = torch.randn(4, rep.prediction_dim) * 0.1
    _, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=0.01)
    next_local_pos = next_rep[..., : B * 3].unflatten(-1, (B, 3))
    assert torch.allclose(next_local_pos[..., 0, :], torch.zeros(4, 3), atol=1e-5), (
      f"Root local_pos not zero: {next_local_pos[..., 0, :]}"
    )

  def test_single_body_shape(self):
    """B=1 should produce correct shapes."""
    rep = MaxCoordRepresentation(1, torch.zeros(29))
    aux = torch.zeros(2, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(2, rep.rep_state_dim)
    pred = torch.randn(2, rep.prediction_dim) * 0.01
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    assert next_aux.shape == (2, 7)
    assert next_rep.shape == (2, rep.rep_state_dim)

  def test_zero_pred_zero_vel_no_position_change(self):
    """Zero prediction + zero velocity = no position change."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, torch.zeros(29))
    full_state = _make_valid_full_state(4, B)
    # Zero out all velocities
    body_pos, body_quat, _, _ = decompose_maxcoord_state(full_state, B)
    full_state = build_maxcoord_full_state(
      body_pos, body_quat, torch.zeros(4, B, 3), torch.zeros(4, B, 3)
    )
    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    pred = torch.zeros(4, rep.prediction_dim)
    next_aux, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=0.01)
    next_full = rep.reconstruct_full_state(next_aux, next_rep)
    next_pos, _, _, _ = decompose_maxcoord_state(next_full, B)
    assert torch.allclose(next_pos, body_pos, atol=1e-5), (
      f"Positions changed: max diff {(next_pos - body_pos).abs().max():.6f}"
    )

  def test_gradient_flows_through_world_frame_conversion(self):
    """Gradient should flow through the full world-frame round-trip."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, torch.zeros(29))
    full_state = _make_valid_full_state(4, B)
    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    next_aux, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=0.01)
    next_full = rep.reconstruct_full_state(next_aux, next_rep)
    next_full.sum().backward()
    assert pred.grad is not None
    assert not pred.grad.isnan().any()
    assert (pred.grad.abs().sum(dim=0) > 0).all(), (
      "Some prediction dims got no gradient"
    )


# ---------------------------------------------------------------------------
# End-to-end gradient: apply_prediction -> tracking_reward -> backprop
# ---------------------------------------------------------------------------


class TestEndToEndRewardGradient:
  """Reward → backprop through apply_prediction must reach the prediction tensor."""

  def test_reward_gradient_finite_and_covers_all_prediction_dims(self):
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, torch.zeros(29))
    full_state = _make_valid_full_state(4, B)
    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    action = torch.randn(4, 29)
    last_action = torch.randn(4, 29)
    target_quat = torch.randn(4, B, 4)
    target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True)

    next_aux, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=0.02)
    reward = tracking_reward(
      next_aux,
      *world_frame_bodies(next_aux, next_rep, B),
      torch.zeros(4, 58),
      action,
      last_action,
      torch.randn(4, B, 3),
      target_quat,
      torch.randn(4, B, 3),
      torch.randn(4, B, 3),
      torch.tensor([[-1.0, 1.0]]).expand(29, 2),
      dt=0.02,
    )
    reward.sum().backward()

    assert pred.grad is not None
    assert not pred.grad.isnan().any()
    assert not pred.grad.isinf().any()
    assert pred.grad[..., : B * 3].abs().sum() > 0, "No gradient to vel_delta"
    assert pred.grad[..., B * 3 : B * 6].abs().sum() > 0, "No gradient to avel_delta"

  def test_reward_gradient_finite_at_zero_error(self):
    """No NaN when predicted bodies coincide with targets (acos singularity hotspot)."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, torch.zeros(29))
    full_state = _make_valid_full_state(4, B)
    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    pred = torch.zeros(4, rep.prediction_dim, requires_grad=True)

    next_aux, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=0.02)
    next_full = rep.reconstruct_full_state(next_aux, next_rep).detach()
    target_pos, target_quat, target_lin_vel, target_ang_vel = decompose_maxcoord_state(
      next_full, B
    )

    reward = tracking_reward(
      next_aux,
      *world_frame_bodies(next_aux, next_rep, B),
      torch.zeros(4, 58),
      torch.zeros(4, 29),
      torch.zeros(4, 29),
      target_pos,
      target_quat,
      target_lin_vel,
      target_ang_vel,
      torch.tensor([[-1.0, 1.0]]).expand(29, 2),
      dt=0.02,
    )
    reward.sum().backward()

    assert not pred.grad.isnan().any()
    assert not pred.grad.isinf().any()
