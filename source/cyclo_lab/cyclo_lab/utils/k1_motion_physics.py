# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""Verify redundant named NPZ bodies against source-defined K1 kinematics/twists.

No producer is inferred. A physical convention must explain every frame/body.
The legacy COM profile is extracted from the cited repository URDF, not fitted
from a motion. Absolute tolerances cover float32 FK roundoff, not model changes.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from cyclo_lab.manager_based.mimic.supermimic.upstream import quat_apply
from cyclo_lab.manager_based.mimic import REPO_ROOT
from cyclo_lab.utils.k1_kinematics import UrdfKinematics, rpy_quat
from cyclo_lab.utils.k1_motion_validation import validate_motion, real_float32

URDF = REPO_ROOT / "third_party/ai_sapiens/ai_sapiens_description/urdf/k1_rev1/k1.urdf"
PROFILE = Path(__file__).with_name("k1_motion_profile.json")
# Fixed, absolute limits in meters, quaternion components, m/s and rad/s.
TOLERANCES = {
    "position": 2e-5,
    "orientation": 2e-5,
    "linear_velocity": 5e-5,
    "angular_velocity": 5e-5,
}


def com_profiles(body_names, urdf=URDF):
    root = ET.parse(urdf).getroot()
    current = {
        n.get("name"): {
            "mass": float(n.find("inertial/mass").get("value")),
            "com": [float(x) for x in n.find("inertial/origin").get("xyz").split()],
        }
        for n in root.findall("link")
    }
    historical = json.loads(PROFILE.read_text())
    joints_hash = hashlib.sha256(
        "".join(ET.tostring(n).decode() for n in root.findall("joint")).encode()
    ).hexdigest()
    profiles = {"current_urdf_com": current}
    if joints_hash == historical["joint_xml_sha256"]:
        profiles["ai_sapiens_b0ef567_com"] = historical["links"]
    results = {}
    for name, links in profiles.items():
        # Merge absent fixed links in their parent frame, as the URDF importer does.
        masses = {n: float(v["mass"]) for n, v in links.items()}
        centers = {
            n: torch.tensor(v["com"], dtype=torch.float32) for n, v in links.items()
        }
        pending = [
            n
            for n in root.findall("joint")
            if n.get("type") == "fixed"
            and n.find("child").get("link") not in body_names
        ]
        while pending:
            children = {n.find("parent").get("link") for n in pending}
            leaves = [n for n in pending if n.find("child").get("link") not in children]
            if not leaves:
                raise ValueError("Invalid fixed-link tree")
            for n in leaves:
                parent, child = (
                    n.find("parent").get("link"),
                    n.find("child").get("link"),
                )
                origin = n.find("origin")
                pos = torch.tensor(
                    [float(x) for x in origin.get("xyz", "0 0 0").split()]
                )
                quat = torch.tensor(rpy_quat(origin.get("rpy", "0 0 0")))
                child_center = pos + quat_apply(quat, centers[child])
                total = masses[parent] + masses[child]
                centers[parent] = (
                    masses[parent] * centers[parent] + masses[child] * child_center
                ) / total
                masses[parent] = total
                pending.remove(n)
        if set(body_names) - centers.keys():
            raise ValueError("Motion has bodies outside the supported K1 model")
        results[name] = torch.stack([centers[n] for n in body_names])
    return results


def _metadata(data, keys, allowed=None):
    values = [str(np.asarray(data[k]).item()) for k in keys if k in data]
    if len(set(values)) > 1 or (
        values and allowed is not None and values[0] not in allowed
    ):
        raise ValueError(f"Conflicting/unsupported motion metadata: {keys}")
    return values[0] if values else "unspecified"


def validate_motion_physics(path, urdf=URDF):
    metadata = validate_motion(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    producer = _metadata(data, ("producer", "motion_producer"))
    backend = _metadata(
        data, ("backend", "physics_backend", "simulator_backend", "producer_backend")
    )
    _metadata(
        data,
        ("coordinate_frame", "world_frame", "velocity_frame", "body_velocity_frame"),
        {"world"},
    )
    _metadata(
        data, ("position_reference", "body_position_point"), {"link", "link_origin"}
    )
    _metadata(data, ("up_axis",), {"z"})
    _metadata(data, ("units",), {"SI"})
    declared = _metadata(
        data,
        ("velocity_reference", "velocity_point", "body_velocity_point"),
        {"com", "link_origin"},
    )
    names, joints = metadata["body_names"], metadata["joint_names"]
    model = ET.parse(urdf).getroot()
    moving = {n.get("name") for n in model.findall("joint") if n.get("type") != "fixed"}
    required_bodies = {
        n.find("child").get("link")
        for n in model.findall("joint")
        if n.get("type") != "fixed"
    } | {"pelvis"}
    if set(joints) != moving or not required_bodies <= set(names):
        raise ValueError(
            "Physical validation requires every named K1 moving body and joint"
        )
    tensors = {
        k: torch.from_numpy(real_float32(data[k], k))
        for k in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
    }
    pos, quat = tensors["body_pos_w"], tensors["body_quat_w"]
    if metadata["quaternion_order"] == "xyzw":
        quat = quat[..., [3, 0, 1, 2]]
    linear, angular = tensors["body_lin_vel_w"], tensors["body_ang_vel_w"]
    root = names.index("pelvis")
    fk = UrdfKinematics(urdf, joints, names)
    candidates = {
        "link_origin": torch.zeros(len(names), 3),
        **com_profiles(names, urdf),
    }
    matches, residuals, velocities = [], {}, []
    for profile, offsets in candidates.items():
        convention = "link_origin" if profile == "link_origin" else "com"
        if declared != "unspecified" and declared != convention:
            continue
        root_velocity = linear[:, root] - torch.linalg.cross(
            angular[:, root],
            quat_apply(quat[:, root], offsets[root].expand_as(angular[:, root])),
        )
        p, q, v, w = fk(
            pos[:, root],
            quat[:, root],
            tensors["joint_pos"],
            root_velocity,
            angular[:, root],
            tensors["joint_vel"],
        )
        v = v + torch.linalg.cross(w, quat_apply(q, offsets.expand_as(w)))
        errors = {
            "position": float((p - pos).abs().max()),
            "orientation": float(
                torch.minimum(
                    (q - quat).abs().amax(-1), (q + quat).abs().amax(-1)
                ).max()
            ),
            "linear_velocity": float((v - linear).abs().max()),
            "angular_velocity": float((w - angular).abs().max()),
        }
        residuals[profile] = errors
        if all(value <= TOLERANCES[key] for key, value in errors.items()):
            matches.append(profile)
            velocities.append(root_velocity)
    if not matches:
        raise ValueError(
            f"Motion bodies do not match a supported K1 physical convention: {residuals}"
        )
    # A stationary clip can match several profiles; accept only identical derived state.
    if any(
        not torch.allclose(velocities[0], v, rtol=0, atol=1e-7) for v in velocities[1:]
    ):
        raise ValueError(
            f"Ambiguous motion velocity profiles with different root-link velocities: {matches}"
        )
    metadata.update(
        producer=producer,
        backend=backend,
        physical_profiles=matches,
        physical_residuals={k: residuals[k] for k in matches},
        reference_velocity="world_link_origin",
        profile_source_sha256=json.loads(PROFILE.read_text())["sha256"],
    )
    return metadata, velocities[0]
