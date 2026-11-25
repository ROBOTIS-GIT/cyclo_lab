"""Source-reference checks, distinct from live-simulator FK parity."""

import hashlib
import numpy as np
import pytest
import torch
from cyclo_lab.manager_based.mimic import REPO_ROOT as ROOT
from cyclo_lab.utils.k1_motion_validation import validate_motion
from cyclo_lab.utils.k1_motion_physics import (
    validate_motion_physics,
    com_profiles,
    URDF,
)
from cyclo_lab.utils.k1_kinematics import UrdfKinematics
from supermimic.utils.math import quat_apply

SOURCE = ROOT / "source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz"


def arrays():
    with np.load(SOURCE, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


@pytest.mark.parametrize("clip", ["dance1", "dance2"])
def test_original_clips_physically_verify_without_source_mutation(clip):
    path = SOURCE.parents[1] / clip / f"{clip}.npz"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata, root_v = validate_motion_physics(path)
    assert metadata["physical_profiles"] == ["current_urdf_com"]
    assert metadata["producer"] == "unspecified"
    assert metadata["fps"] == 50 and root_v.shape[-1] == 3
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("profile", ["current_urdf_com", "ai_sapiens_b0ef567_com"])
def test_rotating_pelvis_com_to_link_conversion(tmp_path, profile):
    data = arrays()
    names, joints = data["body_names"].tolist(), data["joint_names"].tolist()
    offsets = com_profiles(names)[profile]
    fk = UrdfKinematics(URDF, joints, names)
    frames = 3
    pos = torch.tensor([[1.0, 2.0, 0.8]]).expand(frames, -1)
    quat = torch.tensor([[0.70710678, 0.70710678, 0.0, 0.0]]).expand(frames, -1)
    linear = torch.tensor([[0.2, -0.4, 0.1]]).expand(frames, -1)
    angular = torch.tensor([[1.0, 2.0, 3.0]]).expand(frames, -1)
    joint = torch.zeros(frames, 23)
    p, q, v, w = fk(pos, quat, joint, linear, angular, joint)
    com_v = v + torch.linalg.cross(w, quat_apply(q, offsets.expand_as(w)))
    data.update(
        joint_pos=joint.numpy(),
        joint_vel=joint.numpy(),
        body_pos_w=p.numpy(),
        body_quat_w=q[..., [1, 2, 3, 0]].numpy(),
        body_lin_vel_w=com_v.numpy(),
        body_ang_vel_w=w.numpy(),
        velocity_point=np.asarray("com"),
    )
    path = tmp_path / "rotating.npz"
    np.savez(path, **data)
    metadata, converted = validate_motion_physics(path)
    assert metadata["physical_profiles"] == [profile]
    torch.testing.assert_close(converted, linear, rtol=0, atol=1e-7)
    assert not torch.allclose(converted, com_v[:, names.index("pelvis")])


@pytest.mark.parametrize(
    "case",
    [
        "complex",
        "overflow",
        "frame",
        "backend_conflict",
        "point_conflict",
        "wrong_point",
        "wrong_body_pose",
        "wrong_body_twist",
    ],
)
def test_invalid_numeric_or_physical_contract_fails_without_mutation(tmp_path, case):
    data = arrays()
    if case == "complex":
        data["joint_pos"] = data["joint_pos"].astype(np.complex64) + 1j
    elif case == "overflow":
        data["joint_vel"] = data["joint_vel"].astype(np.float64)
        data["joint_vel"][0, 0] = 1e300
    elif case == "frame":
        data["velocity_frame"] = np.asarray("body")
    elif case == "backend_conflict":
        data.update(backend=np.asarray("physx"), physics_backend=np.asarray("mujoco"))
    elif case == "point_conflict":
        data.update(
            velocity_point=np.asarray("com"),
            velocity_reference=np.asarray("link_origin"),
        )
    elif case == "wrong_point":
        data["velocity_point"] = np.asarray("link_origin")
    elif case == "wrong_body_pose":
        data["body_pos_w"][5, 9, 0] += 0.001
    else:
        data["body_lin_vel_w"][5, 9, 0] += 0.001
    path = tmp_path / "invalid.npz"
    np.savez(path, **data)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        (
            validate_motion
            if case in ("complex", "overflow")
            else validate_motion_physics
        )(path)
    assert path.read_bytes() == before
