# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Woojae Lee

"""K1 Rev.1-specific configuration for mimic tracking tasks."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.robots.K1_rev1 import K1_REV1_INERTIA_TUNED_ACTION_SCALE, K1_REV1_INERTIA_TUNED_CFG
from cyclo_lab.manager_based.mimic.tracking_env_cfg import (
    RewardsCfg,
    TrackingEnvCfg,
)
import cyclo_lab.manager_based.mimic.mdp as mdp

TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
]

END_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
]

UNDESIRED_CONTACT_BODY_NAMES = [
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_roll_rubber_hand$)(?!right_wrist_roll_rubber_hand$).+$"
]

# Existing K1 mimic NPZ files were recorded with the Isaac Lab 2 / PhysX
# breadth-first articulation ordering and WXYZ quaternions. Keep those files
# unchanged and describe their source layout explicitly for the Newton loader.
LEGACY_PHYSX_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]

LEGACY_PHYSX_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "torso_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
]


@configclass
class K1Rev1MimicRewardsCfg(RewardsCfg):
    """K1 Rev.1 reward terms that depend on robot body names."""

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=UNDESIRED_CONTACT_BODY_NAMES),
            "threshold": 1.0,
        },
    )


@configclass
class K1Rev1MimicEnvCfg(TrackingEnvCfg):
    """K1 Rev.1 mimic tracking environment."""

    rewards: K1Rev1MimicRewardsCfg = K1Rev1MimicRewardsCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = K1_REV1_INERTIA_TUNED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_REV1_INERTIA_TUNED_ACTION_SCALE
        self.commands.reference_trajectory.anchor_body_name = "torso_link"
        self.commands.reference_trajectory.body_names = TRACKED_BODY_NAMES
        self.commands.reference_trajectory.source_joint_names = LEGACY_PHYSX_JOINT_NAMES
        self.commands.reference_trajectory.source_body_names = LEGACY_PHYSX_BODY_NAMES
        self.commands.reference_trajectory.source_quat_order = "wxyz"
        # Runtime articulation CoM randomization can produce NaNs with the
        # current Newton/MJWarp backend. Keep the URDF nominal CoM for K1.
        self.events.torso_com_offset_noise = None
        self.terminations.end_body_pos.params["body_names"] = END_BODY_NAMES
