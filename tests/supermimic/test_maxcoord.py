"""Numerical native K1 topology and unmodified official MaxCoord equations."""

import xml.etree.ElementTree as ET
import pytest
import torch
from cyclo_lab.learning.supermimic.bridge import URDF, BODIES
from cyclo_lab.learning.supermimic.fk import UrdfKinematics
from cyclo_lab.learning.supermimic.ik import build_urdf_analytic_ik
from supermimic.utils.math import matrix_from_quat
from supermimic.representations import create_representation
from supermimic.models import Dynamics


def sample(names, bodies):
    torch.manual_seed(42)
    fk = UrdfKinematics(URDF, names, bodies)
    q = torch.randn(3, 23) * 0.15
    qd = torch.randn(3, 23) * 0.3
    rot = torch.randn(3, 4)
    rot = rot / rot.norm(dim=-1, keepdim=True)
    fields = fk(torch.randn(3, 3), rot, q, torch.randn(3, 3), torch.randn(3, 3), qd)
    return q, qd, fields


@pytest.mark.parametrize("permuted", [False, True])
def test_native_ik_roundtrip_joint_body_permutations_and_gradients(permuted):
    names = [
        n.get("name")
        for n in ET.parse(URDF).getroot().findall("joint")
        if n.get("type") != "fixed"
    ]
    bodies = list(BODIES)
    if permuted:
        names.reverse()
        bodies = bodies[:1] + bodies[:0:-1]
    q, qd, fields = sample(names, bodies)
    ik = build_urdf_analytic_ik(URDF, names, bodies)
    # One-hinge chains really contain only one retained slot.
    assert (ik.sign.count_nonzero(-1) == 1).any()
    rotations = matrix_from_quat(fields[1]).detach().requires_grad_()
    velocities = fields[3].detach().requires_grad_()
    actual_q, actual_qd = ik(rotations, velocities)
    torch.testing.assert_close(actual_q, q, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_qd, qd, atol=2e-6, rtol=2e-5)
    (actual_q.square().sum() + actual_qd.square().sum()).backward()
    for grad in (rotations.grad, velocities.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_maxcoord_layout_integration_sensors_and_action_gradient():
    names = [
        n.get("name")
        for n in ET.parse(URDF).getroot().findall("joint")
        if n.get("type") != "fixed"
    ]
    q, qd, fields = sample(names, BODIES)
    full = torch.cat([x.flatten(-2) for x in fields], -1)
    rep = create_representation(
        "maxcoord", 182, 23, 0.02, 4, torch.zeros(23), "rep_state"
    )
    rep.bind_native(
        torch.tensor([[-3.0, 3.0]] * 23), build_urdf_analytic_ik(URDF, names, BODIES)
    )
    state, aux = rep.represent(full), rep.extract_aux_state(full)
    assert_full_close(rep.reconstruct_full_state(aux, state), full)
    torch.testing.assert_close(
        rep.joint_state(state), torch.cat([q, qd], -1), atol=3e-6, rtol=2e-5
    )
    model = Dynamics(rep, 23, 1, torch.ones(23) * 0.1, torch.zeros(23), (16, 16))
    action = torch.randn(3, 23, requires_grad=True)
    # Exercise the actual model-to-integration-to-IK path used by imagined rollout.
    nxt_aux, nxt = model(state, action, aux, torch.zeros(3, 1))
    sensor = rep.joint_state(nxt)
    sensor.square().mean().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    # No FK projection: an independent off-manifold body displacement survives.
    altered = full.clone()
    altered[:, 3:6] += 0.1
    a, r = rep.extract_aux_state(altered), rep.represent(altered)
    next_a, next_r = rep.apply_prediction(a, r, torch.zeros(3, 84), 0.0)
    assert_full_close(rep.reconstruct_full_state(next_a, next_r), altered)


def assert_full_close(actual, expected):
    # q and -q represent the same orientation.
    torch.testing.assert_close(actual[:, :42], expected[:, :42], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual[:, 98:], expected[:, 98:], atol=2e-6, rtol=2e-5)
    a, b = actual[:, 42:98].reshape(-1, 14, 4), expected[:, 42:98].reshape(-1, 14, 4)
    torch.testing.assert_close(
        matrix_from_quat(a), matrix_from_quat(b), atol=2e-6, rtol=2e-5
    )
