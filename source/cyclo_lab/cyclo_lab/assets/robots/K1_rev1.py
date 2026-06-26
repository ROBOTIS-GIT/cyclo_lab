# Copyright 2025 ROBOTIS CO., LTD.
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

"""K1 Rev.1 humanoid robot asset configuration."""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_EXT_DIR

AI_SAPIENS_DESCRIPTION_DIR = os.path.abspath(
    os.path.join(CYCLO_LAB_ASSETS_EXT_DIR, "../../third_party/ai_sapiens/ai_sapiens_description")
)
K1_REV1_URDF_PATH = f"{AI_SAPIENS_DESCRIPTION_DIR}/urdf/k1_rev1/k1.urdf"

NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0

ARMATURE_QC060_200_R020_RE = 0.00564892
ARMATURE_QC080_240_R020_RE = 0.01936542
MAX_TORQUE_QC060_200_R020_RE = 47.277
MAX_TORQUE_QC080_240_R020_RE = 96.864
MAX_SPEED_QC060_200_R020_RE = 200.0 * 2.0 * math.pi / 60.0
MAX_SPEED_QC080_240_R020_RE = 110.0 * 2.0 * math.pi / 60.0

STIFFNESS_QC060_200_R020_RE = ARMATURE_QC060_200_R020_RE * NATURAL_FREQ**2
STIFFNESS_QC080_240_R020_RE = ARMATURE_QC080_240_R020_RE * NATURAL_FREQ**2

DAMPING_QC060_200_R020_RE = 2.0 * DAMPING_RATIO * ARMATURE_QC060_200_R020_RE * NATURAL_FREQ
DAMPING_QC080_240_R020_RE = 2.0 * DAMPING_RATIO * ARMATURE_QC080_240_R020_RE * NATURAL_FREQ


K1_REV1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=K1_REV1_URDF_PATH,
        activate_contact_sensors=True,
        fix_base=False,
        replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.18,
            ".*_knee_joint": 0.36,
            ".*_ankle_pitch_joint": -0.18,
            ".*_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.95,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": MAX_TORQUE_QC080_240_R020_RE,
                ".*_hip_roll_joint": MAX_TORQUE_QC080_240_R020_RE,
                ".*_hip_pitch_joint": MAX_TORQUE_QC080_240_R020_RE,
                ".*_knee_joint": MAX_TORQUE_QC080_240_R020_RE,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": MAX_SPEED_QC080_240_R020_RE,
                ".*_hip_roll_joint": MAX_SPEED_QC080_240_R020_RE,
                ".*_hip_pitch_joint": MAX_SPEED_QC080_240_R020_RE,
                ".*_knee_joint": MAX_SPEED_QC080_240_R020_RE,
            },
            stiffness={
                ".*_hip_pitch_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_yaw_joint": 120.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_pitch_joint": 3.0,
                ".*_hip_roll_joint": 3.0,
                ".*_hip_yaw_joint": 3.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_pitch_joint": 0.01,
                ".*_hip_roll_joint": 0.01,
                ".*_hip_yaw_joint": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=MAX_TORQUE_QC060_200_R020_RE,
            velocity_limit_sim=MAX_SPEED_QC060_200_R020_RE,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=MAX_TORQUE_QC080_240_R020_RE,
            velocity_limit_sim=MAX_SPEED_QC080_240_R020_RE,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=120.0,
            damping=3.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": MAX_TORQUE_QC060_200_R020_RE,
                ".*_shoulder_roll_joint": MAX_TORQUE_QC060_200_R020_RE,
                ".*_shoulder_yaw_joint": MAX_TORQUE_QC060_200_R020_RE,
                ".*_elbow_joint": MAX_TORQUE_QC060_200_R020_RE,
                ".*_wrist_roll_joint": MAX_TORQUE_QC060_200_R020_RE,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": MAX_SPEED_QC060_200_R020_RE,
                ".*_shoulder_roll_joint": MAX_SPEED_QC060_200_R020_RE,
                ".*_shoulder_yaw_joint": MAX_SPEED_QC060_200_R020_RE,
                ".*_elbow_joint": MAX_SPEED_QC060_200_R020_RE,
                ".*_wrist_roll_joint": MAX_SPEED_QC060_200_R020_RE,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
                ".*_wrist_roll_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.01,
                ".*_shoulder_roll_joint": 0.01,
                ".*_shoulder_yaw_joint": 0.01,
                ".*_elbow_joint": 0.01,
                ".*_wrist_roll_joint": 0.01,
            },
        ),
    }
)


K1_REV1_INERTIA_TUNED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=K1_REV1_URDF_PATH,
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=200.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.764),
        joint_pos={
            ".*_hip_pitch_joint": -0.3,
            ".*_knee_joint": 0.63,
            ".*_ankle_pitch_joint": -0.33,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": MAX_TORQUE_QC080_240_R020_RE,
                ".*_hip_roll_joint": MAX_TORQUE_QC080_240_R020_RE,
                ".*_hip_pitch_joint": MAX_TORQUE_QC080_240_R020_RE,
                ".*_knee_joint": MAX_TORQUE_QC080_240_R020_RE,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": MAX_SPEED_QC080_240_R020_RE,
                ".*_hip_roll_joint": MAX_SPEED_QC080_240_R020_RE,
                ".*_hip_pitch_joint": MAX_SPEED_QC080_240_R020_RE,
                ".*_knee_joint": MAX_SPEED_QC080_240_R020_RE,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_QC080_240_R020_RE,
                ".*_hip_roll_joint": STIFFNESS_QC080_240_R020_RE,
                ".*_hip_yaw_joint": STIFFNESS_QC080_240_R020_RE,
                ".*_knee_joint": STIFFNESS_QC080_240_R020_RE,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_QC080_240_R020_RE,
                ".*_hip_roll_joint": DAMPING_QC080_240_R020_RE,
                ".*_hip_yaw_joint": DAMPING_QC080_240_R020_RE,
                ".*_knee_joint": DAMPING_QC080_240_R020_RE,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_QC080_240_R020_RE,
                ".*_hip_roll_joint": ARMATURE_QC080_240_R020_RE,
                ".*_hip_yaw_joint": ARMATURE_QC080_240_R020_RE,
                ".*_knee_joint": ARMATURE_QC080_240_R020_RE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=MAX_TORQUE_QC060_200_R020_RE,
            velocity_limit_sim=MAX_SPEED_QC060_200_R020_RE,
            stiffness=2.0 * STIFFNESS_QC060_200_R020_RE,
            damping=2.0 * DAMPING_QC060_200_R020_RE,
            armature=2.0 * ARMATURE_QC060_200_R020_RE,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=MAX_TORQUE_QC080_240_R020_RE,
            velocity_limit_sim=MAX_SPEED_QC080_240_R020_RE,
            stiffness=STIFFNESS_QC080_240_R020_RE,
            damping=DAMPING_QC080_240_R020_RE,
            armature=ARMATURE_QC080_240_R020_RE,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": MAX_TORQUE_QC060_200_R020_RE,
                ".*_shoulder_roll_joint": MAX_TORQUE_QC060_200_R020_RE,
                ".*_shoulder_yaw_joint": MAX_TORQUE_QC060_200_R020_RE,
                ".*_elbow_joint": MAX_TORQUE_QC060_200_R020_RE,
                ".*_wrist_roll_joint": MAX_TORQUE_QC060_200_R020_RE,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": MAX_SPEED_QC060_200_R020_RE,
                ".*_shoulder_roll_joint": MAX_SPEED_QC060_200_R020_RE,
                ".*_shoulder_yaw_joint": MAX_SPEED_QC060_200_R020_RE,
                ".*_elbow_joint": MAX_SPEED_QC060_200_R020_RE,
                ".*_wrist_roll_joint": MAX_SPEED_QC060_200_R020_RE,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_QC060_200_R020_RE,
                ".*_shoulder_roll_joint": STIFFNESS_QC060_200_R020_RE,
                ".*_shoulder_yaw_joint": STIFFNESS_QC060_200_R020_RE,
                ".*_elbow_joint": STIFFNESS_QC060_200_R020_RE,
                ".*_wrist_roll_joint": STIFFNESS_QC060_200_R020_RE,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_QC060_200_R020_RE,
                ".*_shoulder_roll_joint": DAMPING_QC060_200_R020_RE,
                ".*_shoulder_yaw_joint": DAMPING_QC060_200_R020_RE,
                ".*_elbow_joint": DAMPING_QC060_200_R020_RE,
                ".*_wrist_roll_joint": DAMPING_QC060_200_R020_RE,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_QC060_200_R020_RE,
                ".*_shoulder_roll_joint": ARMATURE_QC060_200_R020_RE,
                ".*_shoulder_yaw_joint": ARMATURE_QC060_200_R020_RE,
                ".*_elbow_joint": ARMATURE_QC060_200_R020_RE,
                ".*_wrist_roll_joint": ARMATURE_QC060_200_R020_RE,
            },
        ),
    },
)
"""K1 Rev.1 configuration with inertia-tuned PD gains."""


def _action_scale_from_actuators(cfg: ArticulationCfg) -> dict[str, float]:
    """Compute per-regex joint position action scales from actuator effort/stiffness."""
    scale = {}
    for actuator in cfg.actuators.values():
        names = actuator.joint_names_expr
        efforts = actuator.effort_limit_sim
        stiffness = actuator.stiffness
        if not isinstance(efforts, dict):
            efforts = {name: efforts for name in names}
        if not isinstance(stiffness, dict):
            stiffness = {name: stiffness for name in names}
        for name in names:
            if name in efforts and name in stiffness and stiffness[name]:
                scale[name] = 0.25 * efforts[name] / stiffness[name]
    return scale


K1_REV1_INERTIA_TUNED_ACTION_SCALE = _action_scale_from_actuators(K1_REV1_INERTIA_TUNED_CFG)
"""Joint position action scale map for the inertia-tuned K1 Rev.1 configuration."""
