"""The same immutable 50 Hz archive crosses named producer/backend orders."""

import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from cyclo_lab.manager_based.mimic import REPO_ROOT as ROOT
from cyclo_lab.utils.k1_motion_validation import validate_motion
from cyclo_lab.utils.k1_motion_physics import URDF, validate_motion_physics
from cyclo_lab.utils.motion_npz import load_motion_npz

SOURCE = ROOT / "source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz"


def arrays():
    with np.load(SOURCE, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
def test_shared_loader_reorders_without_resampling_or_rewriting(tmp_path, order):
    original = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data = arrays()
    # Original archive's breadth-first joint order vs URDF's depth-first order.
    target_joints = [
        n.get("name")
        for n in ET.parse(URDF).getroot().findall("joint")
        if n.get("type") != "fixed"
    ]
    target_bodies = data["body_names"].tolist()
    for name in ["joint_names", "body_names"]:
        data[name] = data[name][::-1]
    for field in [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]:
        data[field] = data[field][:, ::-1].copy()
    if order == "wxyz":
        data["body_quat_w"] = data["body_quat_w"][..., [3, 0, 1, 2]]
    data["quaternion_order"] = np.asarray(order)
    path = tmp_path / "other_producer.npz"
    np.savez(path, **data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = validate_motion(path, target_joints, target_bodies)
    assert (
        metadata["fps"] == 50
        and metadata["frames"] == 300
        and metadata["quaternion_order"] == order
    )
    physical, _ = validate_motion_physics(path)
    assert physical["physical_profiles"] == ["current_urdf_com"]
    baseline = load_motion_npz(SOURCE, target_joints, target_bodies)
    # This is the exact loader called by ReferenceTrajectory for BOTH algorithms.
    reordered = load_motion_npz(path, target_joints, target_bodies)
    for name in [
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]:
        np.testing.assert_array_equal(baseline[name], reordered[name])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == original


@pytest.mark.parametrize("with_head", [False, True], ids=["legacy24", "legacy25"])
def test_unnamed_k1_matches_named_reference(tmp_path, with_head):
    data = arrays()
    joints = data.pop("joint_names").tolist()
    bodies = data.pop("body_names").tolist()
    assert data.pop("quaternion_order").item() == "xyzw"
    data["body_quat_w"] = data["body_quat_w"][..., [3, 0, 1, 2]]
    if with_head:
        # The old 25-body PhysX layout inserted head_link at column 6.
        for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            values = data[key]
            data[key] = np.concatenate((values[:, :6], values[:, :1], values[:, 6:]), axis=1)

    path = tmp_path / "unnamed.npz"
    np.savez(path, **data)
    contract = validate_motion(path, joints, bodies)
    assert contract["joint_names"] == joints
    expected_bodies = bodies[:6] + ["head_link"] + bodies[6:] if with_head else bodies
    assert contract["body_names"] == expected_bodies
    assert contract["quaternion_order"] == "wxyz"

    reference = load_motion_npz(SOURCE, joints, bodies)
    loaded = load_motion_npz(path, joints, bodies)
    for key in data:
        np.testing.assert_array_equal(loaded[key], reference[key])


@pytest.mark.parametrize(
    "case",
    [
        "missing_names",
        "missing_order",
        "conflict",
        "fps",
        "duplicate_joint",
        "duplicate_body",
        "zero_quat",
        "nonunit_quat",
        "nan",
        "shape",
        "unknown_joint",
    ],
)
def test_rejects_ambiguous_or_invalid_input_before_collection(tmp_path, case):
    data = arrays()
    expected = data["joint_names"].tolist()
    if case == "missing_names":
        del data["joint_names"]
    elif case == "missing_order":
        del data["quaternion_order"]
    elif case == "conflict":
        data["quat_order"] = np.asarray("wxyz")
    elif case == "fps":
        data["fps"] = np.asarray([30])
    elif case == "duplicate_joint":
        data["joint_names"][0] = data["joint_names"][1]
    elif case == "duplicate_body":
        data["body_names"][0] = data["body_names"][1]
    elif case == "zero_quat":
        data["body_quat_w"][0, 0] = 0
    elif case == "nonunit_quat":
        data["body_quat_w"][0, 0] *= 2
    elif case == "nan":
        data["body_lin_vel_w"][0, 0, 0] = np.nan
    elif case == "shape":
        data["joint_pos"] = data["joint_pos"][:, :22]
    elif case == "unknown_joint":
        data["joint_names"][0] = "unknown"
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        validate_motion(path, expected)
    assert path.read_bytes() == before
