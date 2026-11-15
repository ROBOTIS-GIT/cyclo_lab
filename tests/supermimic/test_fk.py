import xml.etree.ElementTree as ET
import torch
from cyclo_lab.learning.supermimic.bridge import URDF, BODIES, wxyz
from cyclo_lab.learning.supermimic.fk import UrdfKinematics
from supermimic.utils.math import quat_mul, quat_apply


def fk():
    names = [
        n.get("name")
        for n in ET.parse(URDF).getroot().findall("joint")
        if n.get("type") != "fixed"
    ]
    return UrdfKinematics(URDF, names, BODIES), names


def test_twist_is_pose_time_derivative_and_gradients_reach_joints():
    k, names = fk()
    torch.manual_seed(21)
    pos = torch.randn(2, 3, dtype=torch.float64)
    rot = torch.randn(2, 4, dtype=torch.float64)
    rot = rot / rot.norm(dim=-1, keepdim=True)
    q = (torch.randn(2, 23, dtype=torch.float64) * 0.1).requires_grad_()
    v = torch.randn(2, 3, dtype=torch.float64)
    w = torch.randn(2, 3, dtype=torch.float64)
    qv = torch.randn(2, 23, dtype=torch.float64)
    # Constants follow input precision for the finite-difference accuracy check.
    k.edges = [(a, b, c, *[x.double() for x in d]) for a, b, c, *d in k.edges]
    out = k(pos, rot, q, v, w, qv)
    from supermimic.utils.helpers import quat_integrate

    eps = 1e-6
    nxt = k(pos + v * eps, quat_integrate(rot, w, eps), q + qv * eps, v, w, qv)
    torch.testing.assert_close((nxt[0] - out[0]) / eps, out[2], atol=2e-5, rtol=2e-5)
    out[0][:, 10].sum().backward()
    assert q.grad.abs().sum() > 0
    # Reordered articulation joints must represent the same robot.
    other = UrdfKinematics(URDF, names[::-1], BODIES)
    other.edges = [(a, b, c, *[x.double() for x in d]) for a, b, c, *d in other.edges]
    reordered = other(pos, rot, q.flip(-1), v, w, qv.flip(-1))
    torch.testing.assert_close(out[0], reordered[0])


def test_quaternion_boundary_and_rotation_composition():
    q = wxyz(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(q, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    a = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    x = torch.tensor([[1.0, 0.0, 0.0]])
    torch.testing.assert_close(
        quat_apply(quat_mul(a, a), x),
        torch.tensor([[-1.0, 0.0, 0.0]]),
        atol=1e-6,
        rtol=1e-6,
    )
